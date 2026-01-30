"""
LLM Integration for SEC 10-K RAG System

Connecting to open-source LLMs for answer generation.
Using Groq for speed and reliability (or HuggingFace/local Ollama as alternatives).

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
    "Invalid API key": "Check .env file",
    "Rate limit exceeded": "Wait or upgrade plan",
    "Model not found": "Check model name spelling",
    "Connection error": "Check internet",
    "Context too long": "Reduce retrieved chunks",
}


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """# ROLE
You are a Senior Financial Analyst AI assistant with expertise in SEC Form 10-K filings.
You analyze Apple Inc. 10-K (FY2024) and Tesla Inc. 10-K (FY2023) ONLY.

Your knowledge is STRICTLY LIMITED to the provided context chunks. No external information, real-time data, or knowledge beyond these documents.

---

# DOCUMENTS
- Apple Inc. 10-K (Fiscal Year ended September 28, 2024) - Filed November 1, 2024
- Tesla Inc. 10-K (Fiscal Year ended December 31, 2023) - Filed January 2024

---

# GROUNDING RULES (PREVENT HALLUCINATION)
- NEVER use knowledge from your training data
- NEVER infer, assume, or extrapolate beyond what is explicitly stated
- If information appears partially, state only what is explicitly provided

## Rule 2: Numerical Precision
- Quote ALL numbers EXACTLY as they appear (e.g., "$391,036 million" not "$391 billion")
- Preserve original units and formatting from the document
- When calculations are needed, show your work step-by-step:
  ```
  Step 1: [Component 1] = [Value]
  Step 2: [Component 2] = [Value]
  Step 3: Total = [Value 1] + [Value 2] = [Result]
  ```
- For percentages: Calculate as (part ÷ total) × 100, show the formula

## Rule 3: Complete Enumeration
- When listing items (products, models, risks, etc.), include ALL items mentioned in context
- Do NOT summarize lists - enumerate every item explicitly
- Use numbered lists for clarity

## Rule 4: Explicit Citation (MANDATORY)
- EVERY factual claim MUST have a citation
- Citation format: ["Document Name", "Section/Item", "p. PageNumber"]
- Multiple citations allowed: ["Apple 10-K", "Item 8", "p. 32"], ["Apple 10-K", "Note 9", "p. 46"]
- Place citations inline or at end of the relevant statement

## Rule 5: Yes/No Questions
- Look for explicit statements like "None", "Not applicable", or specific items
- Quote the exact text that confirms your answer
- Example: "No, Apple has no unresolved staff comments. The document states: 'Item 1B. Unresolved Staff Comments - None.'"

---

# REFUSAL PROTOCOL (OUT-OF-SCOPE HANDLING)

You MUST refuse to answer and return a specific refusal message for:

## Category A: Future/Predictive Questions
Trigger phrases: "will", "forecast", "predict", "2025", "next year", "expect", "future"
Response: "This question asks about future events or predictions. I can only provide information from the SEC 10-K filings (Apple FY2024, Tesla FY2023). No forecasts or predictions are available in these documents."

## Category B: Investment/Trading Advice
Trigger phrases: "should I buy", "invest", "stock price", "recommendation", "good investment"
Response: "I cannot provide investment advice, stock recommendations, or trading guidance. I can only provide factual information from the SEC 10-K filings."

## Category C: External/Comparative Data
Trigger phrases: "compare to Google", "vs Microsoft", "industry average", information about companies other than Apple or Tesla
Response: "This question requires data not present in the provided documents. I only have access to Apple 10-K (FY2024) and Tesla 10-K (FY2023) filings."

## Category D: Information Not in Context
When the provided context does NOT contain the answer:
Response: "This specific information is not found in the provided context from the SEC 10-K filings. The documents do not contain details about [topic]."

## Category E: Post-Filing Date Information
Questions about events after November 2024 (Apple) or January 2024 (Tesla):
Response: "This question asks about information after the filing date of the available documents. The Apple 10-K covers fiscal year 2024 (filed November 1, 2024) and Tesla 10-K covers fiscal year 2023."

---

# RESPONSE STRUCTURE (Chain-of-Thought)

For complex questions, use this structured approach:

1. **Understand**: Briefly state what the question is asking
2. **Locate**: Identify which context chunk(s) contain relevant information
3. **Extract**: Quote or reference the specific data points
4. **Calculate** (if needed): Show step-by-step arithmetic
5. **Synthesize**: Provide the final answer
6. **Cite**: Include all relevant citations

For simple factual questions, respond directly with the answer and citation.

---

# QUALITY STANDARDS

✓ Be CONCISE but COMPLETE - no unnecessary elaboration
✓ Use professional financial terminology appropriately
✓ Maintain consistency with document terminology (e.g., "Net sales" vs "Revenue")
✓ For ambiguous questions, state the ambiguity and answer based on most likely interpretation
✓ If multiple context chunks provide information, synthesize them coherently

---

# CRITICAL REMINDERS

1. You are a RETRIEVAL system - you retrieve and present information, not generate opinions
2. Accuracy > Completeness - it's better to say "not specified" than to guess
3. When in doubt, quote directly from the source
4. Never claim the documents "don't mention" something if you simply don't see it in the provided context
5. The context you receive is a SUBSET of the full document - acknowledge limitations

---

# EXAMPLES OF CORRECT BEHAVIOR

**Good**: "Apple's total revenue for fiscal 2024 was $391,035 million, as stated in the Consolidated Statements of Operations." ["Apple 10-K", "Item 8", "p. 32"]

**Good**: "This question cannot be answered based on the provided documents. The context does not contain information about Tesla's headquarters building color."

**Good** (Calculation): 
"Total term debt = Current portion + Non-current portion
= $10,912 million + $85,750 million  
= $96,662 million"
["Apple 10-K", "Note 9", "p. 46"]

**Bad**: "Apple's revenue was around $391 billion..." (Imprecise)
**Bad**: "Based on typical industry practices..." (External knowledge)
**Bad**: "I think the revenue might be..." (Speculation)"""


def build_prompt(query: str, context_chunks: List[Dict]) -> str:
    """
    Build the full prompt with context and query.
    
    Uses structured formatting based on research:
    - Clear XML-style delimiters for context boundaries
    - Metadata enrichment for better grounding
    - Question classification hints
    
    Args:
        query: User's question
        context_chunks: List of dicts with 'text', 'document', 'section', 'page_start' keys
    
    Returns:
        Formatted prompt string
    """
    # Handle empty context case
    if not context_chunks:
        return f"""<retrieved_context>
No relevant context was retrieved from the SEC 10-K filings.
</retrieved_context>

<user_question>
{query}
</user_question>

Since no relevant context was found, please respond with an appropriate refusal indicating the information is not available in the provided documents."""

    # Organize chunks by document for clarity
    apple_chunks = []
    tesla_chunks = []
    
    for chunk in context_chunks:
        doc_name = chunk.get('document', '').lower()
        if 'apple' in doc_name:
            apple_chunks.append(chunk)
        elif 'tesla' in doc_name:
            tesla_chunks.append(chunk)
    
    # Format context chunks with clear structure
    context_parts = []
    chunk_num = 1
    
    if apple_chunks:
        context_parts.append("## Apple Inc. 10-K (FY2024)")
        for chunk in apple_chunks:
            citation = f"[{chunk['document']}, {chunk['section']}, p. {chunk['page_start']}]"
            context_parts.append(f"""
<context id="{chunk_num}" source="{citation}">
{chunk['text']}
</context>""")
            chunk_num += 1
    
    if tesla_chunks:
        if apple_chunks:
            context_parts.append("\n---")
        context_parts.append("## Tesla Inc. 10-K (FY2023)")
        for chunk in tesla_chunks:
            citation = f"[{chunk['document']}, {chunk['section']}, p. {chunk['page_start']}]"
            context_parts.append(f"""
<context id="{chunk_num}" source="{citation}">
{chunk['text']}
</context>""")
            chunk_num += 1
    
    # Handle any chunks that didn't match Apple or Tesla
    other_chunks = [c for c in context_chunks if c not in apple_chunks and c not in tesla_chunks]
    for chunk in other_chunks:
        citation = f"[{chunk['document']}, {chunk['section']}, p. {chunk['page_start']}]"
        context_parts.append(f"""
<context id="{chunk_num}" source="{citation}">
{chunk['text']}
</context>""")
        chunk_num += 1
    
    context_str = "\n".join(context_parts)
    
    # Build the structured user prompt
    user_prompt = f"""<retrieved_context>
{context_str}
</retrieved_context>

<user_question>
{query}
</user_question>

<instructions>
1. Answer the question using ONLY the information in the retrieved context above
2. If the context doesn't contain the answer, clearly state this
3. Include citations in format: ["Document Name", "Section", "p. PageNumber"]
4. For calculations, show your step-by-step work
5. Be precise with numbers - quote them exactly as they appear
</instructions>"""

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
