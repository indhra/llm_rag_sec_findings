"""
LLM Integration for SEC 10-K RAG System
========================================

What we're doing:
    Connecting to open-source/open-access LLMs for answer generation.
    We avoid GPT-4/Claude per the assignment requirements.

Why this approach:
    - Multiple provider support for flexibility (Groq, HuggingFace, local)
    - Groq is recommended: free tier, fast inference, good models (Llama, Mixtral)
    - HuggingFace Inference API: free tier, huge model selection
    - Local Ollama: no internet, full privacy, but needs setup

Provider recommendations:
    1. Groq (recommended): Free, fast, Llama 3, Mixtral
    2. HuggingFace: Free tier, many models
    3. Together AI: $25 free credit, good models
    4. Local Ollama: Offline, private, but slower

Prompt design is critical:
    - Use ONLY retrieved context
    - Cite sources properly
    - Handle out-of-scope questions
    - Refuse speculation/forecasts

Author: Indhra
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

# For API calls
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from huggingface_hub import InferenceClient
    HAS_HF = True
except ImportError:
    HAS_HF = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


ERROR_GUIDE = {
    "Invalid API key": "Check your API key in .env file",
    "Rate limit exceeded": "Wait a minute and retry, or upgrade plan",
    "Model not found": "Check model name spelling",
    "Connection error": "Check internet connection",
    "Context too long": "Reduce number of retrieved chunks",
}


# The system prompt is crucial for RAG
# This instructs the LLM on how to behave
SYSTEM_PROMPT = """You are a financial analyst assistant specialized in analyzing SEC 10-K filings. 
You answer questions ONLY using the provided context from Apple and Tesla 10-K annual reports.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided context chunks
2. If the answer is not clearly stated in the context, respond exactly with: "Not specified in the document."
3. For questions asking about future events, forecasts, predictions, or information that would require data beyond the provided documents, respond exactly with: "This question cannot be answered based on the provided documents."
4. Always cite your sources in this exact format at the end of your answer: ["Document Name", "Section", "p. PageNumber"]
5. Be PRECISE with numbers - quote them exactly as they appear in the context. When asked for totals, add the components and show your calculation.
6. Do not make assumptions or infer information not explicitly stated
7. Keep answers concise but complete
8. When asked to list items (like vehicle models, products, etc.), list ALL items mentioned in the context - do not omit any
9. For yes/no questions about whether something exists or not, look for explicit statements like "None" or specific items listed
10. When asked about percentages, calculate them: (part / total) × 100 and show your work

Examples of out-of-scope questions to refuse:
- Stock price forecasts or predictions
- Information about years not covered in the documents (e.g., 2025 when documents are from 2023/2024)
- Personal opinions or investment recommendations
- Anything requiring external knowledge not in the documents

When you cite sources, list all relevant sources that contributed to your answer."""


def build_prompt(query: str, context_chunks: List[Dict]) -> str:
    """
    Build the full prompt with context and query.
    
    Args:
        query: User's question
        context_chunks: List of dicts with 'text', 'document', 'section', 'page_start' keys
    
    Returns:
        Formatted prompt string
    """
    # Format context chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        citation = f"[{chunk['document']}, {chunk['section']}, p. {chunk['page_start']}]"
        context_parts.append(f"--- Context {i} {citation} ---\n{chunk['text']}")
    
    context_str = "\n\n".join(context_parts)
    
    # Build the user prompt
    user_prompt = f"""CONTEXT FROM SEC 10-K FILINGS:

{context_str}

QUESTION: {query}

Please provide your answer based only on the context above. Include citations in the format ["Document Name", "Section", "p. PageNumber"]."""

    return user_prompt


@dataclass
class LLMResponse:
    """Response from the LLM."""
    answer: str
    raw_response: str
    model: str
    provider: str
    latency_ms: float
    tokens_used: Optional[int] = None


class GroqLLM:
    """
    Groq API integration.
    
    Groq offers free tier with:
    - Llama 3.1 8B, 70B
    - Mixtral 8x7B
    - Very fast inference
    
    Get API key at: https://console.groq.com
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant"  # Fast and free
    ):
        if not HAS_GROQ:
            raise ImportError("Install groq: pip install groq")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY env var or pass api_key.\n"
                "Get free key at: https://console.groq.com"
            )
        
        self.model = model
        self.client = Groq(api_key=self.api_key)
        self.provider = "groq"
        
        print(f"GroqLLM initialized with model: {model}")
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.1  # Low temp for factual accuracy
    ) -> LLMResponse:
        """Generate answer using Groq API."""
        
        user_prompt = build_prompt(query, context_chunks)
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency = (time.time() - start_time) * 1000
            
            answer = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else None
            
            return LLMResponse(
                answer=answer,
                raw_response=answer,
                model=self.model,
                provider=self.provider,
                latency_ms=latency,
                tokens_used=tokens
            )
            
        except Exception as e:
            error_msg = str(e)
            
            # Try to provide helpful fix
            for pattern, fix in ERROR_GUIDE.items():
                if pattern.lower() in error_msg.lower():
                    raise RuntimeError(f"Groq API error: {e}\nFix: {fix}")
            
            raise RuntimeError(f"Groq API error: {e}")


class HuggingFaceLLM:
    """
    HuggingFace Inference API integration.
    
    Free tier available with HF token.
    Supports many open models.
    
    Get token at: https://huggingface.co/settings/tokens
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    ):
        if not HAS_HF:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")
        
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not self.api_key:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN env var or pass api_key.\n"
                "Get token at: https://huggingface.co/settings/tokens"
            )
        
        self.model = model
        self.client = InferenceClient(token=self.api_key)
        self.provider = "huggingface"
        
        print(f"HuggingFaceLLM initialized with model: {model}")
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> LLMResponse:
        """Generate answer using HuggingFace Inference API."""
        
        user_prompt = build_prompt(query, context_chunks)
        
        # Format as chat messages
        full_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_prompt} [/INST]"
        
        start_time = time.time()
        
        try:
            response = self.client.text_generation(
                full_prompt,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False
            )
            
            latency = (time.time() - start_time) * 1000
            
            return LLMResponse(
                answer=response,
                raw_response=response,
                model=self.model,
                provider=self.provider,
                latency_ms=latency
            )
            
        except Exception as e:
            raise RuntimeError(f"HuggingFace API error: {e}")


class OllamaLLM:
    """
    Local Ollama integration.
    
    Ollama runs models locally - no API key needed.
    Install from: https://ollama.ai
    
    Then: ollama pull llama3.1:8b
    """
    
    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434"
    ):
        if not HAS_REQUESTS:
            raise ImportError("Install requests: pip install requests")
        
        self.model = model
        self.host = host
        self.provider = "ollama"
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{host}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"OllamaLLM initialized with model: {model}")
            else:
                raise ConnectionError("Ollama not responding")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {host}\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"And you have the model: ollama pull {model}"
            )
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> LLMResponse:
        """Generate answer using local Ollama."""
        
        user_prompt = build_prompt(query, context_chunks)
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=120  # Ollama can be slow
            )
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "")
                
                return LLMResponse(
                    answer=answer,
                    raw_response=answer,
                    model=self.model,
                    provider=self.provider,
                    latency_ms=latency
                )
            else:
                raise RuntimeError(f"Ollama error: {response.text}")
                
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama timeout - model might be too slow")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")


class DummyLLM:
    """
    Dummy LLM for testing without API calls.
    
    Just returns a template response - useful for testing the pipeline.
    """
    
    def __init__(self):
        self.model = "dummy"
        self.provider = "dummy"
        print("Using DummyLLM - for testing only!")
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        **kwargs
    ) -> LLMResponse:
        """Return a template response based on context."""
        
        if not context_chunks:
            answer = "This question cannot be answered based on the provided documents."
            sources = []
        else:
            # Use first chunk as "answer"
            first_chunk = context_chunks[0]
            answer = f"Based on the documents: {first_chunk['text'][:200]}..."
            sources = [f"{first_chunk['document']}, {first_chunk['section']}, p. {first_chunk['page_start']}"]
        
        return LLMResponse(
            answer=answer,
            raw_response=answer,
            model=self.model,
            provider=self.provider,
            latency_ms=0
        )


def get_llm(
    provider: str = "auto",
    model: Optional[str] = None,
    **kwargs
):
    """
    Get an LLM instance.
    
    Args:
        provider: "groq", "huggingface", "ollama", "dummy", or "auto"
        model: Model name (provider-specific)
        **kwargs: Additional arguments for the provider
    
    Returns:
        LLM instance
    """
    if provider == "auto":
        # Try providers in order of preference
        # 1. Groq (fast, free)
        if os.getenv("GROQ_API_KEY") and HAS_GROQ:
            print("Auto-detected Groq API key")
            return get_llm("groq", model, **kwargs)
        
        # 2. HuggingFace
        if (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")) and HAS_HF:
            print("Auto-detected HuggingFace token")
            return get_llm("huggingface", model, **kwargs)
        
        # 3. Ollama (local)
        if HAS_REQUESTS:
            try:
                requests.get("http://localhost:11434/api/tags", timeout=1)
                print("Auto-detected local Ollama")
                return get_llm("ollama", model, **kwargs)
            except:
                pass
        
        # 4. Fallback to dummy
        print("No LLM provider found, using dummy")
        return DummyLLM()
    
    elif provider == "groq":
        default_model = "llama-3.1-8b-instant"
        return GroqLLM(model=model or default_model, **kwargs)
    
    elif provider == "huggingface":
        default_model = "mistralai/Mistral-7B-Instruct-v0.3"
        return HuggingFaceLLM(model=model or default_model, **kwargs)
    
    elif provider == "ollama":
        default_model = "llama3.1:8b"
        return OllamaLLM(model=model or default_model, **kwargs)
    
    elif provider == "dummy":
        return DummyLLM()
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def parse_answer_and_sources(response: str) -> Tuple[str, List[str]]:
    """
    Parse the LLM response to extract answer and sources.
    
    The LLM should cite sources like: ["Apple 10-K", "Item 8", "p. 28"]
    
    Returns:
        (answer_text, list_of_citations)
    """
    # Try to find citations in brackets
    # Pattern: ["Doc", "Section", "p. N"]
    citation_pattern = r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]'
    
    citations = []
    matches = re.findall(citation_pattern, response)
    for match in matches:
        citations.append(list(match))
    
    # Also try simpler patterns
    simple_pattern = r'\[([^,\]]+),\s*([^,\]]+),\s*([^\]]+)\]'
    simple_matches = re.findall(simple_pattern, response)
    for match in simple_matches:
        citation = [m.strip().strip('"') for m in match]
        if citation not in citations:
            citations.append(citation)
    
    # Clean up answer (remove citation brackets for cleaner text)
    answer = response
    for match in re.finditer(r'\s*\[.*?\](?:\s*\[.*?\])*\s*$', response, re.DOTALL):
        answer = response[:match.start()].strip()
        break
    
    return answer, citations


# Test when run directly
if __name__ == "__main__":
    print("Testing LLM Module")
    print("=" * 50)
    
    # Test with dummy LLM
    llm = DummyLLM()
    
    # Create test context
    context = [
        {
            "text": "Apple's total revenue for fiscal year 2024 was $391,036 million, representing a decrease of 2% compared to the prior year.",
            "document": "Apple 10-K",
            "section": "Item 8",
            "page_start": 28
        }
    ]
    
    query = "What was Apple's total revenue for 2024?"
    
    response = llm.generate(query, context)
    
    print(f"\nQuery: {query}")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Answer: {response.answer}")
    
    # Test parsing
    test_response = 'Apple\'s total revenue was $391,036 million. ["Apple 10-K", "Item 8", "p. 28"]'
    answer, sources = parse_answer_and_sources(test_response)
    
    print(f"\n\nParsing test:")
    print(f"  Raw: {test_response}")
    print(f"  Answer: {answer}")
    print(f"  Sources: {sources}")
    
    print("\n✓ LLM module working!")
