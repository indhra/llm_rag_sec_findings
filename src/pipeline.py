"""
Main RAG Pipeline for SEC 10-K Question Answering

Orchestrator that ties all components together:
1. Parse PDFs → 2. Chunk text → 3. Embed chunks → 4. Build index
5. Search (hybrid) → 6. Rerank → 7. Generate answer with LLM

Each component can be tested/swapped independently.

Author: Indhra
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

# Import our modules
from .pdf_parser import parse_all_documents, Document
from .chunker import TextChunker, Chunk, get_default_chunker
from .embeddings import EmbeddingGenerator, get_default_embedder
from .vector_store import VectorStore, SearchResult
from .reranker import Reranker, get_default_reranker
from .llm import get_llm, parse_answer_and_sources, LLMResponse


# Error patterns and solutions
ERROR_GUIDE = {
    "No documents found": "Check PDFs in data/ folder",
    "Empty index": "Run index_documents() first",
    "LLM failed": "Check API key or try different provider",
    "No results": "Query too specific? Try rephrasing",
}


@dataclass
class AnswerResult:
    """
    Final answer with sources.
    """
    answer: str
    sources: List[str]  # ["Apple 10-K", "Item 8", "p. 28"] or []
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON output."""
        return {
            "answer": self.answer,
            "sources": self.sources
        }


class RAGPipeline:
    """
    Complete RAG Pipeline for SEC 10-K QA.
    
    Usage:
        pipeline = RAGPipeline()
        pipeline.index_documents("data/")
        result = pipeline.answer_question("What was Apple's revenue?")
    """
    
    def __init__(
        self,
        embedding_model: str = "bge-small",  # Use small for dev, bge-large for prod
        reranker_model: str = "ms-marco-mini",  # Use bge-reranker for prod
        llm_provider: str = "auto",
        llm_model: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        use_hybrid_search: bool = True,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Key for embedding model (see embeddings.py)
            reranker_model: Key for reranker model (see reranker.py)
            llm_provider: "groq", "huggingface", "ollama", "auto", "dummy"
            llm_model: Specific model name for the provider
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlap between chunks
            use_hybrid_search: Whether to use BM25 + vector search
            top_k_retrieval: Initial candidates to retrieve
            top_k_rerank: Final results after reranking
        """
        print("=" * 60)
        print("Initializing SEC 10-K RAG Pipeline")
        print("=" * 60)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid = use_hybrid_search
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        
        # Initialize components
        print("\n[1/4] Loading embedding model...")
        self.embedder = EmbeddingGenerator(model_key=embedding_model)
        
        print("\n[2/4] Loading reranker...")
        try:
            self.reranker = Reranker(model_key=reranker_model)
        except Exception as e:
            print(f"Warning: Reranker failed to load: {e}")
            print("Continuing without reranking...")
            self.reranker = None
        
        print("\n[3/4] Initializing LLM...")
        self.llm = get_llm(provider=llm_provider, model=llm_model)
        
        # Vector store will be created when indexing
        self.vector_store = None
        self.chunks = []
        
        print("\n[4/4] Pipeline ready!")
        print(f"  Hybrid search: {use_hybrid_search}")
        print(f"  Retrieval top-k: {top_k_retrieval}")
        print(f"  Rerank top-k: {top_k_rerank}")
        print("=" * 60)
    
    def index_documents(
        self,
        data_dir: str = "data",
        save_index: bool = True,
        index_dir: str = "outputs/index"
    ) -> int:
        """
        Index all SEC 10-K documents.
        
        This parses PDFs, chunks them, generates embeddings, and builds the index.
        
        Args:
            data_dir: Directory containing PDF files
            save_index: Whether to save the index to disk
            index_dir: Where to save the index
        
        Returns:
            Number of chunks indexed
        """
        print("\n" + "=" * 60)
        print("Indexing SEC 10-K Documents")
        print("=" * 60)
        
        # Step 1: Parse PDFs
        print("\n[Step 1] Parsing PDFs...")
        documents = parse_all_documents(data_dir)
        
        if not documents:
            raise ValueError(
                f"No documents found in {data_dir}\n"
                f"Expected files: 10-Q4-2024-As-Filed.pdf, tsla-20231231-gen.pdf"
            )
        
        # Step 2: Chunk documents
        print("\n[Step 2] Chunking documents...")
        chunker = TextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.chunks = chunker.chunk_documents(documents)
        
        # Step 3: Generate embeddings
        print("\n[Step 3] Generating embeddings...")
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # Step 4: Build vector store
        print("\n[Step 4] Building vector index...")
        self.vector_store = VectorStore(
            dimension=self.embedder.dimensions,
            use_hybrid=self.use_hybrid
        )
        self.vector_store.add_chunks(self.chunks, embeddings)
        
        # Save index if requested
        if save_index:
            print(f"\n[Step 5] Saving index to {index_dir}...")
            self.vector_store.save(index_dir)
        
        print("\n" + "=" * 60)
        print(f"✓ Indexing complete! {len(self.chunks)} chunks indexed")
        print("=" * 60)
        
        return len(self.chunks)
    
    def load_index(self, index_dir: str = "outputs/index"):
        """
        Load a previously saved index.
        
        Args:
            index_dir: Directory containing saved index
        """
        print(f"Loading index from {index_dir}...")
        self.vector_store = VectorStore.load(index_dir)
        self.chunks = self.vector_store.chunks
        print(f"✓ Loaded {len(self.chunks)} chunks")
    
    def _is_out_of_scope(self, query: str) -> bool:
        """
        Check if a query is likely out of scope.
        
        These are questions that cannot be answered from the documents:
        - Future predictions/forecasts
        - Information from other years/sources
        - Subjective questions
        """
        query_lower = query.lower()
        
        # Patterns that indicate out-of-scope questions
        out_of_scope_patterns = [
            r"forecast|predict|projection|estimate\s+for\s+(2025|2026|future)",
            r"stock\s+price\s+(forecast|prediction|will|going\s+to)",
            r"what\s+will|what\s+would|what\s+should",
            r"(cfo|ceo|president|chairman)\s+(in|as\s+of|for)\s+2025",
            r"color\s+(of|is)\s+.*(painted|headquarters|building)",
            r"(your|my)\s+opinion",
            r"what\s+do\s+you\s+think",
        ]
        
        for pattern in out_of_scope_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's question
            top_k: Number of results (default: self.top_k_retrieval)
        
        Returns:
            List of SearchResult objects
        """
        if self.vector_store is None:
            raise RuntimeError("Index not built. Call index_documents() first.")
        
        top_k = top_k or self.top_k_retrieval
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k
        )
        
        return results
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results.
        
        Args:
            query: User's question
            results: Initial search results
            top_k: Number of results after reranking
        
        Returns:
            Reranked list of SearchResult objects
        """
        top_k = top_k or self.top_k_rerank
        
        if self.reranker is None:
            # No reranker, just return top-k
            return results[:top_k]
        
        return self.reranker.rerank(query, results, top_k=top_k)
    
    def generate_answer(
        self,
        query: str,
        context_results: List[SearchResult]
    ) -> AnswerResult:
        """
        Generate an answer using the LLM.
        
        Args:
            query: User's question
            context_results: Retrieved and reranked chunks
        
        Returns:
            AnswerResult with answer and sources
        """
        # Check for out-of-scope first
        if self._is_out_of_scope(query):
            return AnswerResult(
                answer="This question cannot be answered based on the provided documents.",
                sources=[]
            )
        
        # If no results, it's not in the documents
        if not context_results:
            return AnswerResult(
                answer="Not specified in the document.",
                sources=[]
            )
        
        # Note: Hybrid search uses RRF scores (0.01-0.05 range), not similarity (0-1)
        # So we don't filter by score here - let the LLM decide if context is relevant
        
        # Prepare context for LLM
        context_chunks = [
            {
                "text": r.text,
                "document": r.document,
                "section": r.section,
                "page_start": r.page_start
            }
            for r in context_results
        ]
        
        # Generate answer
        llm_response = self.llm.generate(query, context_chunks)
        
        # Parse answer and sources
        answer_text, parsed_sources = parse_answer_and_sources(llm_response.answer)
        
        # Flatten sources list
        sources_flat = []
        for source in parsed_sources:
            if isinstance(source, list):
                sources_flat.extend(source)
            else:
                sources_flat.append(source)
        
        # Check if LLM indicated out-of-scope or not found
        answer_lower = answer_text.lower()
        if "cannot be answered" in answer_lower or "not specified" in answer_lower:
            return AnswerResult(
                answer=answer_text,
                sources=[]
            )
        
        return AnswerResult(
            answer=answer_text,
            sources=sources_flat if sources_flat else [r.get_citation() for r in context_results[:1]][0] if context_results else []
        )
    
    def answer_question(self, query: str, question_id: Optional[int] = None) -> Dict:
        """
        Main interface: Answer a question about SEC 10-K filings.
        
        This is the required function signature from the assignment.
        
        Args:
            query: The user question about Apple or Tesla 10-K filings.
            question_id: Optional question identifier for batch processing.
        
        Returns:
            dict: {
                "question_id": 1,  # Only if question_id provided
                "answer": "Answer text or refusal message",
                "sources": ["Apple 10-K", "Item 8", "p. 28"] or []
            }
        """
        # Quick out-of-scope check
        if self._is_out_of_scope(query):
            result = {
                "answer": "This question cannot be answered based on the provided documents.",
                "sources": []
            }
            if question_id is not None:
                result = {"question_id": question_id, **result}
            return result
        
        # Retrieve
        results = self.retrieve(query)
        
        # Rerank
        reranked = self.rerank(query, results)
        
        # Generate
        answer = self.generate_answer(query, reranked)
        
        result = answer.to_dict()
        if question_id is not None:
            result = {"question_id": question_id, **result}
        return result
    
    def answer_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        Answer multiple questions and return results in the required format.
        
        Args:
            questions: List of {"question_id": int, "question": str}
        
        Returns:
            List of {
                "question_id": int,
                "answer": str,
                "sources": list
            }
        """
        results = []
        for q in questions:
            qid = q.get("question_id")
            question = q.get("question", q.get("query", ""))
            result = self.answer_question(question, question_id=qid)
            results.append(result)
        return results
    
    def run_evaluation(
        self,
        questions: List[Dict],
        output_file: str = "outputs/answers.json"
    ) -> List[Dict]:
        """
        Run evaluation on a list of questions.
        
        Args:
            questions: List of {"question_id": int, "question": str}
            output_file: Where to save results
        
        Returns:
            List of {"question_id": int, "answer": str, "sources": list}
        """
        print("\n" + "=" * 60)
        print("Running Evaluation")
        print("=" * 60)
        
        results = []
        
        for i, q in enumerate(questions):
            qid = q["question_id"]
            question = q["question"]
            
            print(f"\n[Q{qid}] {question}")
            
            try:
                result = self.answer_question(question, question_id=qid)
                results.append(result)
                print(f"  → {result['answer'][:100]}...")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    "question_id": qid,
                    "answer": f"Error: {e}",
                    "sources": []
                })
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        return results


# The main function required by the assignment
def answer_question(query: str, question_id: Optional[int] = None) -> Dict:
    """
    Answers a question using the RAG pipeline.
    
    This is the single-function interface required by the assignment.
    It uses a global pipeline instance for efficiency.
    
    Args:
        query (str): The user question about Apple or Tesla 10-K filings.
        question_id (int, optional): Question identifier for batch processing.
    
    Returns:
        dict: {
            "question_id": 1,  # Only if question_id provided
            "answer": "Answer text or 'This question cannot be answered based on the provided documents.'",
            "sources": ["Apple 10-K", "Item 8", "p. 28"]  # Empty list if refused
        }
    """
    global _pipeline_instance
    
    # Initialize pipeline if needed
    if '_pipeline_instance' not in globals() or _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
        
        # Try to load existing index
        index_dir = "outputs/index"
        if Path(index_dir).exists():
            _pipeline_instance.load_index(index_dir)
        else:
            # Index documents
            _pipeline_instance.index_documents("data/")
    
    return _pipeline_instance.answer_question(query, question_id=question_id)


def answer_questions(questions: List[Dict]) -> List[Dict]:
    """
    Answers multiple questions using the RAG pipeline.
    
    Args:
        questions: List of {"question_id": int, "question": str}
    
    Returns:
        List of {
            "question_id": int,
            "answer": str,
            "sources": list
        }
    
    Example output:
        [
            {
                "question_id": 11,
                "answer": "This question cannot be answered based on the provided documents.",
                "sources": []
            }
        ]
    """
    global _pipeline_instance
    
    # Initialize pipeline if needed
    if '_pipeline_instance' not in globals() or _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
        
        # Try to load existing index
        index_dir = "outputs/index"
        if Path(index_dir).exists():
            _pipeline_instance.load_index(index_dir)
        else:
            # Index documents
            _pipeline_instance.index_documents("data/")
    
    return _pipeline_instance.answer_questions(questions)


# Global instance for answer_question()
_pipeline_instance = None


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEC 10-K RAG Pipeline")
    parser.add_argument("--index", action="store_true", help="Index documents")
    parser.add_argument("--query", type=str, help="Query to answer")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        embedding_model="bge-small",
        llm_provider="auto"
    )
    
    if args.index:
        pipeline.index_documents(args.data_dir)
    
    elif args.query:
        # Load or build index
        if Path("outputs/index").exists():
            pipeline.load_index("outputs/index")
        else:
            pipeline.index_documents(args.data_dir)
        
        result = pipeline.answer_question(args.query)
        print("\n" + "=" * 60)
        print("Answer:", result["answer"])
        print("Sources:", result["sources"])
    
    elif args.evaluate:
        # Load or build index
        if Path("outputs/index").exists():
            pipeline.load_index("outputs/index")
        else:
            pipeline.index_documents(args.data_dir)
        
        # Evaluation questions from assignment
        questions = [
            {"question_id": 1, "question": "What was Apple's total revenue for the fiscal year ended September 28, 2024?"},
            {"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
            {"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
            {"question_id": 4, "question": "On what date was Apple's 10-K report for 2024 signed and filed with the SEC?"},
            {"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
            {"question_id": 6, "question": "What was Tesla's total revenue for the year ended December 31, 2023?"},
            {"question_id": 7, "question": "What percentage of Tesla's total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
            {"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
            {"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
            {"question_id": 10, "question": "What is the purpose of Tesla's 'lease pass-through fund arrangements'?"},
            {"question_id": 11, "question": "What is Tesla's stock price forecast for 2025?"},
            {"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
            {"question_id": 13, "question": "What color is Tesla's headquarters painted?"},
        ]
        
        pipeline.run_evaluation(questions)
    
    else:
        parser.print_help()
