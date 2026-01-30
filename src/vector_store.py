"""
Vector Store with Hybrid Search for SEC 10-K RAG

Combining vector search (FAISS) with keyword search (BM25).
Vector search finds semantic matches, BM25 finds exact terms.
Together they're much better than either alone.

Author: Indhra
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pickle
from pathlib import Path
import json

# FAISS for vector search
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss-cpu not installed. Run: pip install faiss-cpu")

# BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("Warning: rank-bm25 not installed. Run: pip install rank-bm25")


ERROR_GUIDE = {
    "faiss not installed": "Install: pip install faiss-cpu",
    "rank_bm25 not installed": "Install: pip install rank-bm25",
    "Index not built": "Call build_index() first",
    "Dimension mismatch": "Query embedding dimension doesn't match index",
}


@dataclass
class SearchResult:
    """
    A search result with metadata.
    """
    chunk_id: str
    text: str
    score: float
    document: str
    section: str
    page_start: int
    page_end: int
    source_file: str
    
    # For hybrid search, track individual scores
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    
    def get_citation(self) -> List[str]:
        """Citation format: ["Apple 10-K", "Item 8", "p. 28"]"""
        page_str = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        return [self.document, self.section, page_str]
    
    def __repr__(self):
        return f"SearchResult(chunk={self.chunk_id}, score={self.score:.4f}, doc={self.document}, sec={self.section})"


class VectorStore:
    """
    FAISS-based vector store with optional BM25 hybrid search.
    
    Usage:
        store = VectorStore(dimension=768)
        store.add_chunks(chunks, embeddings)
        results = store.hybrid_search(query_embedding, query_text, top_k=5)
    """
    
    def __init__(
        self,
        dimension: int,
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.7  # Weight for vector search (1-alpha for BM25)
    ):
        """
        Initialize the vector store.
        
        Args:
            dimension: Embedding dimension (e.g., 768 for bge-base)
            use_hybrid: Whether to use hybrid search (vector + BM25)
            hybrid_alpha: Weight for vector score in hybrid fusion (0-1)
        """
        if not HAS_FAISS:
            raise ImportError(ERROR_GUIDE["faiss not installed"])
        
        self.dimension = int(dimension)  # FAISS requires native int, not numpy.int64
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha
        
        # FAISS index - using L2 distance (we normalize embeddings, so L2 ≈ cosine)
        # IndexFlatIP (inner product) would also work for normalized vectors
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Metadata storage - FAISS only stores vectors, we need to track the rest
        self.chunks = []  # Store chunk objects
        self.chunk_id_to_idx = {}  # Map chunk_id to index position
        
        # BM25 index for keyword search
        self.bm25 = None
        self.tokenized_corpus = []
        
        print(f"VectorStore initialized")
        print(f"  Dimension: {dimension}")
        print(f"  Hybrid search: {use_hybrid}")
        if use_hybrid:
            print(f"  Hybrid alpha: {hybrid_alpha} (vector weight)")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Just lowercase and split on whitespace/punctuation.
        Could be improved with stemming, but keeping it simple.
        """
        import re
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def add_chunks(
        self,
        chunks: List,
        embeddings: np.ndarray
    ):
        """
        Add chunks with their embeddings to the store.
        
        Args:
            chunks: List of Chunk objects from chunker
            embeddings: Numpy array of shape (num_chunks, dimension)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: embeddings have dim {embeddings.shape[1]}, "
                f"index expects {self.dimension}"
            )
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Add to FAISS index
        # FAISS expects float32
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        
        # Store chunks and build lookup
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk.chunk_id] = start_idx + i
        
        # Build BM25 index if using hybrid search
        if self.use_hybrid:
            if HAS_BM25:
                print("Building BM25 index...")
                
                # Tokenize all chunks
                self.tokenized_corpus = [
                    self._tokenize(chunk.text) for chunk in self.chunks
                ]
                
                # Build BM25 index
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                print(f"  BM25 index built with {len(self.tokenized_corpus)} documents")
            else:
                print("Warning: BM25 not available, using vector search only")
                self.use_hybrid = False
        
        print(f"✓ Vector store now contains {self.index.ntotal} vectors")
    
    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Pure vector similarity search.
        
        Returns list of (chunk_idx, distance) tuples.
        Lower distance = more similar (L2 distance).
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure correct shape
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))
        
        # Convert to list of tuples
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for not found
                # Convert L2 distance to similarity score (higher = better)
                # For normalized vectors, L2 distance = 2 - 2*cosine_sim
                # So cosine_sim = 1 - distance/2
                similarity = 1 - dist / 2
                results.append((idx, similarity))
        
        return results
    
    def bm25_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        BM25 keyword search.
        
        Returns list of (chunk_idx, score) tuples.
        Higher score = more relevant.
        """
        if not self.use_hybrid or self.bm25 is None:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                results.append((idx, scores[idx]))
        
        return results
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
        initial_k: int = 20,
        fusion_method: str = "rrf"  # "rrf" or "weighted"
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector and BM25 results.
        
        Args:
            query_embedding: Vector embedding of the query
            query_text: Original query text (for BM25)
            top_k: Number of final results to return
            initial_k: Number of candidates to fetch from each method
            fusion_method: "rrf" (Reciprocal Rank Fusion) or "weighted"
        
        Returns:
            List of SearchResult objects, sorted by combined score
        """
        if self.index.ntotal == 0:
            return []
        
        # Get candidates from both methods
        vector_results = self.vector_search(query_embedding, initial_k)
        
        if self.use_hybrid:
            bm25_results = self.bm25_search(query_text, initial_k)
        else:
            bm25_results = []
        
        # Combine results
        if fusion_method == "rrf":
            combined = self._rrf_fusion(vector_results, bm25_results, k=60)
        else:
            combined = self._weighted_fusion(vector_results, bm25_results)
        
        # Sort by combined score (descending)
        combined_sorted = sorted(combined.items(), key=lambda x: x[1]["combined_score"], reverse=True)
        
        # Build SearchResult objects
        results = []
        for chunk_idx, scores in combined_sorted[:top_k]:
            chunk = self.chunks[chunk_idx]
            
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=scores["combined_score"],
                document=chunk.document,
                section=chunk.section,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                source_file=chunk.source_file,
                vector_score=scores.get("vector_score"),
                bm25_score=scores.get("bm25_score")
            ))
        
        return results
    
    def _rrf_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int = 60
    ) -> Dict[int, Dict]:
        """
        Reciprocal Rank Fusion (RRF).
        
        RRF score = Σ 1/(k + rank)
        
        This is robust and doesn't need score normalization.
        k=60 is the common default (from the original paper).
        """
        combined = {}
        
        # Add vector results
        for rank, (idx, score) in enumerate(vector_results):
            if idx not in combined:
                combined[idx] = {"vector_score": score, "bm25_score": 0, "combined_score": 0}
            combined[idx]["vector_score"] = score
            combined[idx]["combined_score"] += 1 / (k + rank + 1)
        
        # Add BM25 results
        for rank, (idx, score) in enumerate(bm25_results):
            if idx not in combined:
                combined[idx] = {"vector_score": 0, "bm25_score": score, "combined_score": 0}
            combined[idx]["bm25_score"] = score
            combined[idx]["combined_score"] += 1 / (k + rank + 1)
        
        return combined
    
    def _weighted_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]]
    ) -> Dict[int, Dict]:
        """
        Weighted sum fusion.
        
        combined_score = α * vector_score + (1-α) * bm25_score
        
        Requires score normalization for fair combination.
        """
        combined = {}
        
        # Normalize scores
        vector_scores = {idx: score for idx, score in vector_results}
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # Min-max normalize BM25 scores
        if bm25_scores:
            min_bm25 = min(bm25_scores.values())
            max_bm25 = max(bm25_scores.values())
            range_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1
            bm25_scores = {idx: (score - min_bm25) / range_bm25 for idx, score in bm25_scores.items()}
        
        # Combine
        all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        for idx in all_indices:
            v_score = vector_scores.get(idx, 0)
            b_score = bm25_scores.get(idx, 0)
            
            combined[idx] = {
                "vector_score": v_score,
                "bm25_score": b_score,
                "combined_score": self.hybrid_alpha * v_score + (1 - self.hybrid_alpha) * b_score
            }
        
        return combined
    
    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Main search method - uses hybrid if available, else vector only.
        """
        return self.hybrid_search(query_embedding, query_text, top_k=top_k)
    
    def save(self, directory: str):
        """
        Save the vector store to disk.
        
        Creates:
        - index.faiss: FAISS index
        - chunks.pkl: Chunk metadata
        - config.json: Store configuration
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving vector store to {directory}...")
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save chunks
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "chunk_id_to_idx": self.chunk_id_to_idx,
                "tokenized_corpus": self.tokenized_corpus
            }, f)
        
        # Save config
        config = {
            "dimension": self.dimension,
            "use_hybrid": self.use_hybrid,
            "hybrid_alpha": self.hybrid_alpha,
            "num_chunks": len(self.chunks)
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Vector store saved ({len(self.chunks)} chunks)")
    
    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        Load a vector store from disk.
        """
        path = Path(directory)
        
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found: {directory}")
        
        print(f"Loading vector store from {directory}...")
        
        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create instance
        store = cls(
            dimension=config["dimension"],
            use_hybrid=config["use_hybrid"],
            hybrid_alpha=config["hybrid_alpha"]
        )
        
        # Load FAISS index
        store.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load chunks
        with open(path / "chunks.pkl", "rb") as f:
            data = pickle.load(f)
            store.chunks = data["chunks"]
            store.chunk_id_to_idx = data["chunk_id_to_idx"]
            store.tokenized_corpus = data["tokenized_corpus"]
        
        # Rebuild BM25 if using hybrid
        if store.use_hybrid and store.tokenized_corpus:
            store.bm25 = BM25Okapi(store.tokenized_corpus)
        
        print(f"✓ Loaded {len(store.chunks)} chunks")
        
        return store


# Quick test
if __name__ == "__main__":
    print("Testing Vector Store")
    print("=" * 50)
    
    # Create a simple test
    from embeddings import get_default_embedder
    from chunker import Chunk
    
    # Create fake chunks for testing
    test_chunks = [
        Chunk(
            chunk_id="apple_1",
            text="Apple's total revenue for fiscal year 2024 was $391,036 million.",
            document="Apple 10-K",
            section="Item 8",
            page_start=28,
            page_end=28,
            source_file="10-Q4-2024-As-Filed.pdf",
            token_count=20
        ),
        Chunk(
            chunk_id="tesla_1",
            text="Tesla's total revenue for 2023 was $96,773 million from automotive and energy sales.",
            document="Tesla 10-K",
            section="Item 7",
            page_start=50,
            page_end=50,
            source_file="tsla-20231231-gen.pdf",
            token_count=25
        ),
        Chunk(
            chunk_id="apple_2",
            text="The company had 15,115,823,000 shares of common stock outstanding as of October 18, 2024.",
            document="Apple 10-K",
            section="General",
            page_start=1,
            page_end=1,
            source_file="10-Q4-2024-As-Filed.pdf",
            token_count=22
        ),
    ]
    
    # Generate embeddings
    embedder = get_default_embedder(for_production=False)
    texts = [c.text for c in test_chunks]
    embeddings = embedder.embed_texts(texts)
    
    # Create vector store
    store = VectorStore(dimension=embedder.dimensions, use_hybrid=True)
    store.add_chunks(test_chunks, embeddings)
    
    # Test search
    query = "What was Apple's total revenue for 2024?"
    query_embedding = embedder.embed_query(query)
    
    results = store.search(query_embedding, query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nResults:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.document} - {result.section}")
        print(f"   Score: {result.score:.4f} (vector: {result.vector_score:.4f}, bm25: {result.bm25_score:.4f})")
        print(f"   Text: {result.text[:100]}...")
        print(f"   Citation: {result.get_citation()}")
    
    print("\n✓ Vector store working correctly!")
