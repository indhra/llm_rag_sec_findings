# Embeddings and Vector Representations

> **Learning Goal**: Master how text becomes numbers and how similarity search powers modern AI systems.

---

## Table of Contents
1. [What Are Embeddings?](#what-are-embeddings)
2. [Vector Spaces and Semantic Similarity](#vector-spaces)
3. [Similarity Metrics](#similarity-metrics)
4. [Embedding Models](#embedding-models)
5. [Practical Implementation](#practical-implementation)
6. [Interview Essentials](#interview-essentials)

---

## What Are Embeddings?

**Embeddings** = Dense vector representations that capture semantic meaning of text, images, or other data.

### The Core Idea

```mermaid
graph LR
    subgraph "Traditional: One-Hot Encoding"
        T1["'cat'"] --> V1["0, 0, 1, 0, 0, ..., 0 (50k dimensions, Sparse)"]
        T2["'dog'"] --> V2["0, 1, 0, 0, 0, ..., 0 (50k dimensions, Sparse)"]
    end
    
    subgraph "Modern: Dense Embeddings"
        T3["'cat'"] --> V3["0.2, -0.5, 0.8, ... (384 dimensions, Dense)"]
        T4["'dog'"] --> V4["0.3, -0.4, 0.7, ... (384 dimensions, Dense)"]
    end
    
    V1 -.->|No similarity info| V2
    V3 <-->|Close in space!| V4
    
    style V3 fill:#c8e6c9
    style V4 fill:#c8e6c9
```

### Why Embeddings Matter

| Problem | One-Hot | Embeddings |
|---------|---------|------------|
| **Vocabulary Size** | 50k+ dimensions | 384-1024 dimensions |
| **Semantic Similarity** | ❌ No relationship captured | ✅ Similar words → similar vectors |
| **Generalization** | ❌ New words = unknown | ✅ Subword tokens handle new words |
| **Storage** | Sparse, inefficient | Dense, compact |

---

## Vector Spaces and Semantic Similarity {#vector-spaces}

### Geometric Intuition

Embeddings live in a **high-dimensional space** where semantic meaning = geometric proximity.

```mermaid
graph TD
    subgraph "2D Projection of 384D Space"
        C1["'cat'"] 
        C2["'kitten'"]
        C3["'dog'"]
        C4["'puppy'"]
        C5["'car'"]
        C6["'vehicle'"]
        
        C1 ---|close| C2
        C1 ---|medium| C3
        C3 ---|close| C4
        C5 ---|close| C6
        C1 -.-|far| C5
    end
    
    style C1 fill:#e3f2fd
    style C2 fill:#e3f2fd
    style C3 fill:#f8bbd0
    style C4 fill:#f8bbd0
    style C5 fill:#fff9c4
    style C6 fill:#fff9c4
```

### Real Example from Your Project

```python
# src/embeddings.py uses BGE-small-en-v1.5
# Input texts:
text1 = "Apple's total revenue for fiscal year 2024"
text2 = "Apple's net sales in 2024"  
text3 = "Tesla's car production"

# Embeddings (384 dimensions each):
emb1 = [0.23, -0.45, 0.12, ..., 0.67]  # 384 floats
emb2 = [0.25, -0.42, 0.15, ..., 0.69]  # Very similar to emb1!
emb3 = [0.01, -0.23, 0.89, ..., -0.34] # Different from emb1, emb2
```

### Dimensionality Tradeoffs

```mermaid
graph LR
    subgraph "Embedding Dimensions"
        D1["Small: 128-384 dims"] --> P1["✅ Fast search - ✅ Low memory - ⚠️ Less nuance"]
        D2["Medium: 768-1024 dims"] --> P2["⚖️ Balanced - Standard choice"]
        D3["Large: 1536+ dims"] --> P3["✅ Rich semantics - ❌ Slow search - ❌ High memory"]
    end
    
    style D2 fill:#c8e6c9
```

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| **all-MiniLM-L6-v2** | 384 | Fast, lightweight, good baseline |
| **BGE-small** | 384 | Your project - balanced performance |
| **BGE-large** | 1024 | Higher quality, slower |
| **OpenAI text-embedding-3-large** | 3072 | Best quality, expensive |

---

## Similarity Metrics {#similarity-metrics}

### Cosine Similarity (Most Common)

**Measures the angle between vectors**, ignoring magnitude.

$$
\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

**Range**: -1 (opposite) to +1 (identical)

```mermaid
graph TD
    subgraph "Cosine Similarity Intuition"
        V1["Vector A: (1, 1) - 'cat kitten'"] 
        V2["Vector B: (1.5, 1.5) - 'feline animal'"]
        V3["Vector C: (1, -1) - 'car vehicle'"]
        
        V1 -->|Angle: 0° - Similarity: 1.0| V2
        V1 -->|Angle: 90° - Similarity: 0.0| V3
    end
    
    style V1 fill:#c8e6c9
    style V2 fill:#c8e6c9
    style V3 fill:#fff9c4
```

**Why cosine?**
- ✅ Normalized: Focuses on direction, not magnitude
- ✅ Works well for text embeddings
- ✅ Fast computation (just dot product if vectors are normalized)

### Dot Product

$$
\text{dot\_product}(A, B) = \sum_{i=1}^{n} A_i \times B_i
$$

**When vectors are normalized (L2 norm = 1):**
$$
\text{cosine\_similarity} = \text{dot\_product}
$$

**In your project:**
```python
# src/embeddings.py - line ~150
# Normalize embeddings → cosine similarity = dot product
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# FAISS then uses dot product for speed
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
```

### Euclidean Distance (L2)

$$
\text{distance}(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
$$

**Range**: 0 (identical) to ∞ (very different)

**Less common for embeddings** because it's sensitive to magnitude.

### Comparison

| Metric | Formula | Best For | Your Project |
|--------|---------|----------|--------------|
| **Cosine** | Angle between vectors | Text embeddings | ✅ Used (via normalization) |
| **Dot Product** | Sum of products | Fast search on normalized vectors | ✅ Used in FAISS |
| **Euclidean** | Straight-line distance | Low-dim data, image features | ❌ Not used |

---

## Embedding Models {#embedding-models}

### Architecture Types

```mermaid
graph TB
    subgraph "Bi-Encoder (SBERT, BGE)"
        T1[Text 1] --> E1[Encoder]
        T2[Text 2] --> E2[Encoder]
        E1 --> V1[Vector 1]
        E2 --> V2[Vector 2]
        V1 -->|Similarity| V2
        
        P1["✅ Fast: Precompute embeddings - ✅ Scalable to millions - ⚠️ Less accurate for fine-grained"]
    end
    
    subgraph "Cross-Encoder (Reranker)"
        T3["Text 1 + Text 2"] --> E3[Joint Encoder]
        E3 --> S[Similarity Score]
        
        P2["✅ Most accurate - ❌ Slow: Must compare all pairs - Used for reranking top-K"]
    end
    
    style E1 fill:#c8e6c9
    style E2 fill:#c8e6c9
    style E3 fill:#f8bbd0
```

### Popular Models (2025-2026)

| Model | Dimensions | MTEB Score | Speed | Best For |
|-------|-----------|------------|-------|----------|
| **all-MiniLM-L6-v2** | 384 | 58.0 | ⚡⚡⚡ Fast | Development, testing |
| **BGE-small-en-v1.5** | 384 | 62.5 | ⚡⚡ Medium | **Your project** - balanced |
| **BGE-large-en-v1.5** | 1024 | 64.2 | ⚡ Slow | High-quality retrieval |
| **bge-m3** | 1024 | 66.3 | ⚡ Slow | Multilingual |
| **GTE-large** | 1024 | 63.5 | ⚡ Slow | General-purpose |
| **E5-mistral-7b** | 4096 | 68.2 | 🐌 Very slow | Best quality, GPU needed |

**MTEB** = Massive Text Embedding Benchmark (higher = better)

### Fine-Tuning for Your Domain

```mermaid
graph LR
    subgraph "Pretrained Model"
        P["BGE pretrained - on general web text"] --> F[Your financial domain]
    end
    
    subgraph "Fine-Tuning Process"
        F --> D["Create pairs: - Query → Relevant Doc"]
        D --> T["Train with - contrastive loss"]
        T --> M["Fine-tuned model - Better on 10-K filings"]
    end
    
    style P fill:#e3f2fd
    style M fill:#c8e6c9
```

**When to fine-tune:**
- ✅ Highly specialized domain (legal, medical, finance)
- ✅ Have 1000+ query-document pairs
- ✅ Baseline retrieval <70% recall

**Your project:** Uses pretrained BGE-small → works well for financial text without fine-tuning!

---

## Practical Implementation {#practical-implementation}

### From Your Project

```python
# src/embeddings.py - Simplified

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name="bge-small"):
        # Load pretrained model
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",
            device="cpu"
        )
        self.dimension = 384
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Convert texts to embeddings."""
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # L2 normalize → cosine = dot product
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings  # Shape: (len(texts), 384)
```

### Embedding Process

```mermaid
sequenceDiagram
    participant Doc as Document
    participant Chunk as Chunker
    participant Embed as Embedder
    participant Store as Vector Store
    
    Doc->>Chunk: Split into chunks
    Chunk->>Embed: ["chunk 1", "chunk 2", ...]
    Embed->>Embed: Tokenize
    Embed->>Embed: Pass through model
    Embed->>Embed: Normalize vectors
    Embed->>Store: [[0.2, -0.5, ...], [0.1, 0.3, ...]]
    Store->>Store: Build index (FAISS)
```

### Batch Processing

**Why batch?** Models process multiple texts faster than one-by-one.

```python
# Inefficient (one-by-one):
for text in texts:
    emb = model.encode([text])  # Slow!

# Efficient (batched):
embeddings = model.encode(texts, batch_size=32)  # Much faster!
```

**Your project:**
```python
# src/pipeline.py - processes ~491 chunks in batches
chunks = chunker.chunk_all_documents(documents)  # 491 chunks
embeddings = embedder.embed_texts([c.content for c in chunks])
```

---

## Interview Essentials

### Key Concepts

**Q1: What are embeddings?**

> "Dense vector representations that capture semantic meaning in a continuous space. Unlike one-hot encoding, embeddings encode similarity - semantically similar inputs have similar vectors."

**Q2: Why normalize embeddings?**

```python
# Before normalization:
v1 = [3, 4]      # magnitude = 5
v2 = [6, 8]      # magnitude = 10, same direction as v1!

# Cosine similarity = 1.0 (identical direction)
# Dot product = 66 (misleading - different magnitudes)

# After normalization:
v1_norm = [0.6, 0.8]   # magnitude = 1
v2_norm = [0.6, 0.8]   # magnitude = 1

# Now: cosine similarity = dot product = 1.0 ✅
```

**Q3: Bi-encoder vs Cross-encoder?**

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|------------|---------------|
| **Speed** | ⚡ Fast (precompute) | 🐌 Slow (pairwise) |
| **Accuracy** | Good | Better |
| **Scalability** | Millions of docs | Top-K only |
| **Use Case** | Initial retrieval | Reranking |

**Your project uses both!**
- Bi-encoder (BGE) for fast vector search
- Cross-encoder for reranking top results

### Real-World Tradeoffs

```mermaid
graph TD
    subgraph "Embedding Model Selection"
        Q[Your Use Case] --> D1{Latency?}
        D1 -->|<100ms| Small[Small model - 384 dims]
        D1 -->|<500ms| Medium[Medium model - 768 dims]
        D1 -->|Offline| Large[Large model - 1024+ dims]
        
        D2{Accuracy?}
        Small --> D2
        D2 -->|Good enough| S1["MiniLM, BGE-small"]
        D2 -->|Need better| S2["Fine-tune or use larger"]
        
        D3{Domain?}
        Medium --> D3
        D3 -->|General| G1["BGE, E5"]
        D3 -->|Specialized| G2["Fine-tune on domain"]
    end
    
    style S1 fill:#c8e6c9
```

### Common Pitfalls

❌ **Mistake 1**: Using raw dot product without normalization
```python
# Wrong:
similarity = np.dot(emb1, emb2)  # Influenced by magnitude!

# Right:
emb1_norm = emb1 / np.linalg.norm(emb1)
emb2_norm = emb2 / np.linalg.norm(emb2)
similarity = np.dot(emb1_norm, emb2_norm)
```

❌ **Mistake 2**: Embedding queries and documents with different models

❌ **Mistake 3**: Not batching - 100x slower!

❌ **Mistake 4**: Forgetting to normalize before FAISS IndexFlatIP

### Code Examples from Your Project

**1. Embedding generation:**
```python
# src/embeddings.py
embeddings = model.encode(
    texts,
    normalize_embeddings=True,  # Critical for cosine similarity
    batch_size=32
)
```

**2. Similarity search:**
```python
# src/vector_store.py
query_embedding = self.embedder.embed_texts([query])[0]
distances, indices = self.index.search(
    query_embedding.reshape(1, -1),
    k=top_k
)
# distances are cosine similarities (because normalized + IndexFlatIP)
```

---

## Advanced: How BGE Model Works

```mermaid
graph TB
    subgraph "BGE-small-en-v1.5 Architecture"
        I[Input Text] --> T[BERT Tokenizer]
        T --> E["BERT Encoder - 12 layers, 768 hidden"]
        E --> P["Pooling - CLS token or mean"]
        P --> L["Linear Projection - 768 → 384 dims"]
        L --> N[L2 Normalization]
        N --> O["Output Embedding - 384 dims, norm=1"]
    end
    
    style I fill:#e3f2fd
    style O fill:#c8e6c9
```

**Training:** Contrastive learning with millions of (query, positive_doc, negative_doc) triplets.

---

## Further Reading

- 📄 **Paper**: [Sentence-BERT](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)
- 📄 **Paper**: [BGE Embedding](https://arxiv.org/abs/2309.07597) (BAAI, 2023)
- 📊 **Benchmark**: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- 🎥 **Tutorial**: [Understanding Sentence Embeddings](https://www.sbert.net/)

---

## Key Takeaways

✅ **Embeddings** = dense vectors that encode semantic meaning  
✅ **Cosine similarity** measures semantic closeness  
✅ **Bi-encoders** (BGE, SBERT) enable fast, scalable retrieval  
✅ **Normalization** makes cosine similarity = dot product  
✅ **Dimensionality** trades off speed vs accuracy  

**Next**: [RAG Fundamentals →](03-rag-fundamentals.md)
