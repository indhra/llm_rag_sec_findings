"""
Cross-Encoder Reranker for SEC 10-K RAG

Taking initial search results and reranking them with a cross-encoder model
that processes query+document together for better accuracy.

Bi-encoder (fast) finds candidates, cross-encoder (slower) ranks them precisely.

Author: Indhra
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Try to import sentence-transformers for CrossEncoder
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    print("Warning: sentence-transformers not installed for CrossEncoder")


ERROR_GUIDE = {
    "No CrossEncoder": "Install: pip install sentence-transformers",
    "CUDA OOM": "Reduce batch_size or use CPU",
    "Model not found": "Check model name and internet connection",
}


# Available reranker models - tested for SEC documents
RERANKER_MODELS = {
    # Fast for dev
    "ms-marco-mini": {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "Fast, good for dev/testing",
        "quality": "good",
        "speed": "fast"
    },
    # Tiny but effective
    "ms-marco-base": {
        "name": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "description": "Very fast",
        "quality": "good",
        "speed": "very fast"
    },
    # Best quality
    "bge-reranker": {
        "name": "BAAI/bge-reranker-v2-m3",
        "description": "State-of-the-art",
        "quality": "excellent",
        "speed": "medium"
    },
    # Good balance
    "bge-reranker-base": {
        "name": "BAAI/bge-reranker-base",
        "description": "Good balance of quality and speed",
        "quality": "very good",
        "speed": "medium"
    }
}


class Reranker:
    """
    Cross-encoder reranker.
    
    Takes initial search results and reranks them based on query-document relevance.
    
    Usage:
        reranker = Reranker(model_key="ms-marco-mini")
        reranked = reranker.rerank(query, search_results, top_k=5)
    """
    
    def __init__(
        self,
        model_key: str = "ms-marco-mini",
        device: Optional[str] = None,
        batch_size: int = 16
    ):
        """Initialize the reranker."""
        
        Args:
            model_key: Key from RERANKER_MODELS or full model name
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
            batch_size: Number of query-doc pairs to process at once
        """
        if not HAS_CROSS_ENCODER:
            raise ImportError(
                "CrossEncoder requires sentence-transformers. Install with:\n"
                "pip install sentence-transformers"
            )
        
        # Get model name
        if model_key in RERANKER_MODELS:
            model_info = RERANKER_MODELS[model_key]
            self.model_name = model_info["name"]
            print(f"Using reranker: {model_key}")
            print(f"  Model: {self.model_name}")
            print(f"  Quality: {model_info['quality']}, Speed: {model_info['speed']}")
        else:
            self.model_name = model_key
            print(f"Using custom reranker: {model_key}")
        
        self.batch_size = batch_size
        
        # Detect device
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device
        
        print(f"  Device: {self.device}")
        
        # Load model
        self._load_model()
    
    def _detect_device(self) -> str:
        """Auto-detect best device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            self.model = CrossEncoder(
                self.model_name,
                device=self.device
            )
            print(f"✓ Reranker loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load reranker: {e}")
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.
        
        Returns a relevance score (higher = more relevant).
        """
        return float(self.model.predict([(query, document)])[0])
    
    def score_pairs(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Score multiple documents against a query.
        
        Returns list of relevance scores.
        """
        if not documents:
            return []
        
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        return [float(s) for s in scores]
    
    def rerank(
        self,
        query: str,
        search_results: List,  # List of SearchResult objects
        top_k: int = 5
    ) -> List:
        """
        Rerank search results based on cross-encoder scores.
        
        Args:
            query: The user's query
            search_results: List of SearchResult objects from vector store
            top_k: Number of results to return after reranking
        
        Returns:
            Reranked list of SearchResult objects with updated scores
        """
        if not search_results:
            return []
        
        # Extract texts
        texts = [r.text for r in search_results]
        
        # Get cross-encoder scores
        scores = self.score_pairs(query, texts)
        
        # Combine with results
        scored_results = list(zip(search_results, scores))
        
        # Sort by reranker score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores in results and return top_k
        reranked = []
        for result, new_score in scored_results[:top_k]:
            # Create a new result with updated score
            # Store original score for debugging
            result.score = new_score
            reranked.append(result)
        
        return reranked


class DummyReranker:
    """
    Fallback reranker that just returns results as-is.
    
    Use this when cross-encoder is not available or for faster testing.
    """
    
    def __init__(self):
        print("Using dummy reranker (no reranking)")
    
    def rerank(
        self,
        query: str,
        search_results: List,
        top_k: int = 5
    ) -> List:
        """Just return top-k results without reranking."""
        return search_results[:top_k]


def get_default_reranker(for_production: bool = False):
    """
    Get a reranker with sensible defaults.
    
    Args:
        for_production: If True, use best quality model.
                       If False, use fast model for development.
    """
    if not HAS_CROSS_ENCODER:
        print("Warning: CrossEncoder not available, using dummy reranker")
        return DummyReranker()
    
    try:
        if for_production:
            return Reranker(model_key="bge-reranker-base")
        else:
            return Reranker(model_key="ms-marco-mini")
    except Exception as e:
        print(f"Warning: Failed to load reranker: {e}")
        print("Using dummy reranker as fallback")
        return DummyReranker()


# Quick test
if __name__ == "__main__":
    print("Testing Reranker")
    print("=" * 50)
    
    # Test with development model
    reranker = get_default_reranker(for_production=False)
    
    # Test scoring
    query = "What was Apple's total revenue?"
    
    documents = [
        "Apple's total revenue for fiscal year 2024 was $391,036 million.",
        "Tesla reported $96,773 million in total revenue for 2023.",
        "The weather in California was sunny today.",
        "Apple Inc. is headquartered in Cupertino, California.",
    ]
    
    if hasattr(reranker, 'score_pairs'):
        scores = reranker.score_pairs(query, documents)
        
        print(f"\nQuery: {query}")
        print(f"\nScores (higher = more relevant):")
        for doc, score in sorted(zip(documents, scores), key=lambda x: x[1], reverse=True):
            print(f"  {score:.4f}: {doc[:60]}...")
    else:
        print("Using dummy reranker - no scoring available")
    
    print("\n✓ Reranker working!")
