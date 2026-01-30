"""
Embedding Generator for SEC 10-K RAG System

Converting text chunks into vectors so we can do semantic search.
Using sentence-transformers library with different models to balance
speed vs. quality depending on what we need.

Author: Indhra
"""

import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import os

# Try to import sentence-transformers
# This is the main library we use for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed")
    print("Run: pip install sentence-transformers")


# Error guide for common issues
ERROR_GUIDE = {
    "No module named 'sentence_transformers'": "Install: pip install sentence-transformers",
    "CUDA out of memory": "Reduce batch_size or use CPU",
    "Connection error": "Check internet for model download",
    "Model not found": "Check model name or try different one",
}


# Available models - tested these, noting the tradeoffs
EMBEDDING_MODELS = {
    # Fast and light
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "Fast, lightweight. Good for dev.",
        "quality": "good",
        "speed": "fast"
    },
    # Best quality
    "bge-large": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "description": "Top quality embeddings.",
        "quality": "excellent",
        "speed": "medium"
    },
    # Sweet spot
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        "dimensions": 768,
        "description": "Good balance of quality and speed.",
        "quality": "very good",
        "speed": "medium"
    },
    # Small but solid
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "description": "Small but good quality.",
        "quality": "good",
        "speed": "fast"
    }
}


@dataclass
class EmbeddingResult:
    """Result of embedding generation - text and its vector."""
    text: str
    embedding: np.ndarray
    chunk_id: Optional[str] = None
    
    def __repr__(self):
        return f"EmbeddingResult(chunk_id={self.chunk_id}, dims={len(self.embedding)})"


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks.
    
    Usage:
        generator = EmbeddingGenerator(model_key="bge-base")
        embeddings = generator.embed_chunks(chunks)
    """
    
    def __init__(
        self,
        model_key: str = "bge-base",
        device: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_key: Key from EMBEDDING_MODELS dict, or full HuggingFace model name
            device: 'cuda', 'mps' (Mac), or 'cpu'. Auto-detected if None.
            batch_size: Number of texts to embed at once
            show_progress: Show progress bar during embedding
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required. Install with:\n"
                "pip install sentence-transformers"
            )
        
        # Get model name from key or use directly
        if model_key in EMBEDDING_MODELS:
            model_info = EMBEDDING_MODELS[model_key]
            self.model_name = model_info["name"]
            self.dimensions = model_info["dimensions"]
            print(f"Using embedding model: {model_key}")
            print(f"  Model: {self.model_name}")
            print(f"  Dimensions: {self.dimensions}")
            print(f"  Quality: {model_info['quality']}, Speed: {model_info['speed']}")
        else:
            # Assume it's a full model name
            self.model_name = model_key
            self.dimensions = None  # Will be set after loading
            print(f"Using custom model: {model_key}")
        
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # Detect device
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device
        
        print(f"  Device: {self.device}")
        
        # Load the model
        self._load_model()
    
    def _detect_device(self) -> str:
        """
        Auto-detect the best available device.
        Order: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
        """
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
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Update dimensions if not set
            if self.dimensions is None:
                self.dimensions = self.model.get_sentence_embedding_dimension()
            
            print(f"✓ Model loaded successfully")
            
        except Exception as e:
            error_msg = str(e)
            
            # Try to find a helpful fix
            fix = None
            for pattern, solution in ERROR_GUIDE.items():
                if pattern.lower() in error_msg.lower():
                    fix = solution
                    break
            
            if fix:
                raise RuntimeError(f"Failed to load model: {e}\nFix: {fix}")
            else:
                raise RuntimeError(f"Failed to load model: {e}")
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            normalize: Whether to L2-normalize the embedding (recommended for cosine sim)
        
        Returns:
            Numpy array of shape (dimensions,)
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return np.array(embedding)
    
    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            Numpy array of shape (num_texts, dimensions)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress
        )
        
        return np.array(embeddings)
    
    def embed_chunks(self, chunks: List) -> List[EmbeddingResult]:
        """
        Generate embeddings for Chunk objects.
        
        Args:
            chunks: List of Chunk objects from chunker
        
        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            return []
        
        print(f"Embedding {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Create results
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append(EmbeddingResult(
                text=chunk.text,
                embedding=embedding,
                chunk_id=chunk.chunk_id
            ))
        
        print(f"✓ Generated {len(results)} embeddings")
        
        return results
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query for retrieval.
        
        For some models (like BGE), queries need special handling.
        BGE recommends prepending "Represent this sentence for searching relevant passages: "
        but the sentence-transformers library handles this automatically for most cases.
        
        Args:
            query: The search query
        
        Returns:
            Query embedding as numpy array
        """
        # For BGE models, we could add a prefix, but sentence-transformers handles it
        return self.embed_text(query, normalize=True)


def get_default_embedder(for_production: bool = False) -> EmbeddingGenerator:
    """
    Get an embedder with sensible defaults.
    
    Args:
        for_production: If True, use best quality model (slower).
                       If False, use fast model for development.
    
    Returns:
        Configured EmbeddingGenerator
    """
    if for_production:
        return EmbeddingGenerator(model_key="bge-large")
    else:
        # Use bge-small for development - good quality but fast
        return EmbeddingGenerator(model_key="bge-small")


# Quick test when running directly
if __name__ == "__main__":
    print("Testing Embedding Generator")
    print("=" * 50)
    
    # Test with development model (faster)
    embedder = get_default_embedder(for_production=False)
    
    # Test single text
    test_text = "Apple's total revenue for fiscal year 2024 was $391,036 million."
    embedding = embedder.embed_text(test_text)
    
    print(f"\nSingle text embedding:")
    print(f"  Text: {test_text}")
    print(f"  Shape: {embedding.shape}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Norm (should be ~1.0): {np.linalg.norm(embedding):.4f}")
    
    # Test multiple texts
    test_texts = [
        "Apple's total revenue was $391,036 million.",
        "Tesla's revenue for 2023 was $96,773 million.",
        "The company reported strong growth in services segment.",
        "What is the weather like today?"  # Unrelated - should have low similarity
    ]
    
    embeddings = embedder.embed_texts(test_texts)
    
    print(f"\nMultiple text embeddings:")
    print(f"  Num texts: {len(test_texts)}")
    print(f"  Shape: {embeddings.shape}")
    
    # Test similarity
    print(f"\nSimilarity matrix (first 3 are about revenue, 4th is unrelated):")
    from numpy.linalg import norm
    
    for i, text_i in enumerate(test_texts):
        similarities = []
        for j, text_j in enumerate(test_texts):
            sim = np.dot(embeddings[i], embeddings[j])  # Cosine sim (embeddings are normalized)
            similarities.append(f"{sim:.3f}")
        print(f"  {i}: {' | '.join(similarities)}")
    
    print("\n✓ Embedding generator working correctly!")
