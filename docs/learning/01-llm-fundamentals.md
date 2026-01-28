# LLM Fundamentals: From Basics to Transformers

> **Learning Goal**: Understand the core architecture and mechanisms that power modern Large Language Models.

---

## Table of Contents
1. [What are LLMs?](#what-are-llms)
2. [Evolution: RNNs → LSTMs → Transformers](#evolution)
3. [Transformer Architecture](#transformer-architecture)
4. [Attention Mechanism](#attention-mechanism)
5. [Tokenization](#tokenization)
6. [Interview Essentials](#interview-essentials)

---

## What are LLMs?

**Large Language Models (LLMs)** are neural networks trained on massive text corpora to predict and generate human-like text.

### Key Characteristics

| Property | Description | Example |
|----------|-------------|---------|
| **Scale** | Billions of parameters | GPT-4: ~1.7T, Llama 3.1: 8B-405B |
| **Pre-training** | Trained on internet-scale text | Web pages, books, code |
| **Transfer Learning** | Can be fine-tuned for specific tasks | ChatGPT, coding assistants |
| **Emergent Abilities** | Capabilities that appear at scale | Chain-of-thought reasoning |

### LLM Timeline

```mermaid
timeline
    title Evolution of Language Models
    2017 : Transformer Architecture - "Attention is All You Need"
    2018 : BERT (Google) - Bidirectional pre-training
    2019 : GPT-2 (OpenAI) - 1.5B parameters
    2020 : GPT-3 (OpenAI) - 175B parameters - Few-shot learning
    2022 : ChatGPT - RLHF alignment
    2023 : GPT-4, Claude 2 - Llama 2 (Open Source)
    2024 : Claude 3, Llama 3 - Gemini 1.5
    2025-2026 : Llama 3.1-3.3 - Claude Sonnet 4 - Extended context windows
```

---

## Evolution: RNNs → LSTMs → Transformers {#evolution}

### The Problem: Sequential Processing

```mermaid
graph LR
    A[Word 1] --> B[Hidden State 1]
    B --> C[Word 2]
    C --> D[Hidden State 2]
    D --> E[Word 3]
    E --> F[Hidden State 3]
    
    style A fill:#e3f2fd
    style C fill:#e3f2fd
    style E fill:#e3f2fd
    
    G[⚠️ Issues] --> H[Sequential → Slow]
    G --> I["Long-term dependencies lost"]
    G --> J[Vanishing gradients]
```

### Why Transformers Won

| Architecture | Pros | Cons | Speed |
|--------------|------|------|-------|
| **RNN** | Simple, sequential | Poor long-term memory | Slow (sequential) |
| **LSTM** | Better memory | Complex, still sequential | Slow |
| **Transformer** | ✅ Parallel processing, ✅ Long-range dependencies, ✅ Attention mechanism | Large memory requirements | **Fast** (parallel) |

---

## Transformer Architecture

The transformer architecture from "Attention is All You Need" (2017) revolutionized NLP.

```mermaid
graph TB
    subgraph "Input Processing"
        A["Input Text: 'The cat sat'"] --> B["Tokenization: The, cat, sat"]
        B --> C["Token Embeddings (384/768/1024 dims)"]
        C --> D["+ Positional Encoding"]
    end
    
    subgraph "Encoder (BERT-style)"
        D --> E1["Multi-Head Self-Attention"]
        E1 --> E2["Add & Norm"]
        E2 --> E3["Feed Forward"]
        E3 --> E4["Add & Norm"]
        E4 --> E5["Context Vectors"]
    end
    
    subgraph "Decoder (GPT-style)"
        D2["Previous Tokens"] --> D1["Masked Self-Attention"]
        D1 --> D2A["Add & Norm"]
        E5 --> D3["Cross-Attention"]
        D2A --> D3
        D3 --> D4["Add & Norm"]
        D4 --> D5["Feed Forward"]
        D5 --> D6["Add & Norm"]
    end
    
    D6 --> F["Linear + Softmax"]
    F --> G["Next Token Probabilities"]
    
    style E1 fill:#fff9c4
    style D1 fill:#fff9c4
    style D3 fill:#f8bbd0
    style G fill:#c8e6c9
```

### Key Components

#### 1. **Embeddings + Positional Encoding**

```python
# Conceptual example (not actual code)
token_embedding = model.embed("cat")  # [0.23, -0.45, 0.67, ...]
position_encoding = sin_cos_encoding(position=1)  # [0.84, 0.54, ...]
final_embedding = token_embedding + position_encoding
```

**Why positional encoding?** Transformers process all tokens in parallel, so we need to inject position information.

#### 2. **Self-Attention** (The Game Changer)

**Goal**: Let each word "attend to" (look at) all other words to understand context.

```mermaid
graph TD
    subgraph "Self-Attention for 'bank'"
        W1["Input: 'The river bank'"] --> Q["Query: What am I looking for?"]
        W1 --> K["Key: What do I represent?"]
        W1 --> V["Value: What information do I carry?"]
        
        Q --> S["Similarity Scores"]
        K --> S
        S --> A["Attention Weights (river: 0.7, the: 0.1, bank: 0.2)"]
        A --> O["Weighted Sum of Values"]
        V --> O
        O --> OUT["Context-aware 'bank' → riverbank, not financial"]
    end
    
    style S fill:#fff9c4
    style A fill:#f8bbd0
    style OUT fill:#c8e6c9
```

**Mathematical Formulation:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- $Q$ = Query matrix (what I'm looking for)
- $K$ = Key matrix (what I represent)
- $V$ = Value matrix (information to pass forward)
- $d_k$ = dimension of keys (scaling factor)

#### 3. **Multi-Head Attention**

Instead of one attention mechanism, use multiple "heads" to capture different relationships:

```mermaid
graph LR
    subgraph "Multi-Head Attention"
        I["Input"] --> H1["Head 1: Syntactic patterns"]
        I --> H2["Head 2: Semantic meaning"]
        I --> H3["Head 3: Long-range deps"]
        I --> H4["Head 4-8: Other patterns"]
        
        H1 --> C["Concatenate"]
        H2 --> C
        H3 --> C
        H4 --> C
        
        C --> L["Linear Layer"]
        L --> O["Output"]
    end
    
    style H1 fill:#e3f2fd
    style H2 fill:#f8bbd0
    style H3 fill:#fff9c4
    style O fill:#c8e6c9
```

**Real Example from Llama 3.1 8B:**
- 32 attention heads
- Each head learns different patterns
- Head 1 might learn subject-verb agreement
- Head 5 might learn entity relationships

---

## Attention Mechanism Deep Dive {#attention-mechanism}

### The Intuition

Think of attention as a **similarity search**:

```mermaid
graph TD
    Q["Query: 'bank' - What context do I need?"] --> S1{"Compare with each word"}
    
    S1 --> K1["Key: 'river' - Score: 0.9 ✅"]
    S1 --> K2["Key: 'the' - Score: 0.1"]
    S1 --> K3["Key: 'bank' - Score: 0.3"]
    
    K1 --> W["Weighted combination"]
    K2 --> W
    K3 --> W
    
    W --> O["Context-aware 'bank' = riverbank"]
    
    style K1 fill:#c8e6c9
    style O fill:#66bb6a
```

### Attention Variations

| Type | Used In | Purpose |
|------|---------|---------|
| **Self-Attention** | BERT, GPT | Word attends to words in same sequence |
| **Cross-Attention** | Encoder-Decoder | Decoder attends to encoder outputs |
| **Masked Self-Attention** | GPT | Prevent looking at future tokens |
| **Multi-Query Attention** | Llama 3.1 | Share keys/values across heads → faster |
| **Grouped-Query Attention** | Modern LLMs | Balance between multi-head and multi-query |

### Why Attention Works

**Problem with RNNs**: Information from early tokens gets "diluted" through sequential processing.

**Solution with Attention**: Direct connections between any token pair!

```mermaid
graph LR
    subgraph "RNN: Sequential Path"
        R1[Token 1] --> R2[Token 2] --> R3[Token 3] --> R4[Token 100]
        R4 -.->|Information loss| R1
    end
    
    subgraph "Transformer: Direct Attention"
        T1[Token 1] <--> T100[Token 100]
        T1 <--> T2[Token 2]
        T100 <--> T50[Token 50]
    end
    
    style T1 fill:#c8e6c9
    style T100 fill:#c8e6c9
```

---

## Tokenization

**Tokenization** = Converting text into numerical units that LLMs can process.

### Tokenization Methods

```mermaid
graph TD
    T["Text: 'tokenization'"] --> M{"Method?"}
    
    M -->|"Character-level"| C1["t, o, k, e, n, i, z, a, t, i, o, n"]
    M -->|"Word-level"| W1["tokenization"]
    M -->|"Subword BPE"| S1["token, ization"]
    
    C1 --> CP["Pros: Small vocab, Cons: Long sequences"]
    W1 --> WP["Pros: Meaningful units, Cons: OOV problems"]
    S1 --> SP["Pros: Balance, Cons: Complex encoding"]
    
    style S1 fill:#c8e6c9
```

### Modern Approach: Byte-Pair Encoding (BPE)

**Used by**: GPT-3/4, Llama, most modern LLMs

**How it works:**

1. Start with characters
2. Merge most frequent pairs
3. Build vocabulary of ~50k tokens

```python
# Example from your project (conceptual)
text = "embedding embeddings"

# BPE tokenization:
tokens = ["emb", "edd", "ing", "emb", "edd", "ings"]
# Note: "ing" and "ings" are separate tokens
```

### Token Vocabulary Sizes

| Model | Vocabulary Size | Average Tokens/Word |
|-------|----------------|---------------------|
| GPT-2 | 50,257 | ~1.3 |
| GPT-3/4 | 50,257 | ~1.3 |
| Llama 2/3 | 32,000 | ~1.4 |
| Claude | ~100,000 | ~0.9 |

### Why Tokenization Matters

**In your project:**
```python
# src/chunker.py - chunk size = 512 tokens
# This is ~400-600 words depending on content
# Financial text → more specialized terms → slightly more tokens
```

**Context Window Impact:**
- Llama 3.1 8B: 128k token context window
- Your chunks: 512 tokens each
- Can fit ~250 chunks in context (but you use top 5-7)

---

## Interview Essentials

### Must-Know Concepts

1. **Why Transformers over RNNs?**
   - ✅ Parallel processing → faster training
   - ✅ Better long-range dependencies via attention
   - ✅ No vanishing gradient issues

2. **What is attention?**
   - Mechanism to weigh importance of different inputs
   - Query, Key, Value framework
   - Enables context-aware representations

3. **Encoder vs Decoder?**
   - **Encoder** (BERT): Bidirectional, good for understanding
   - **Decoder** (GPT): Unidirectional, good for generation
   - **Encoder-Decoder** (T5): Best for translation/summarization

4. **Position Encoding - Why needed?**
   - Transformers process all tokens in parallel
   - Need to inject position information
   - Sin/cos functions provide unique position embeddings

### Real-World Examples

**From Your Project:**

```python
# src/embeddings.py uses BGE-small-en-v1.5
# This is a BERT-style encoder:
# - Input: "Apple's total revenue"
# - Output: 384-dim vector capturing semantic meaning
# - Uses 12 transformer layers
# - 6 attention heads per layer
```

**Common Interview Question:**

> **Q**: "Why do we normalize embeddings in your code?"

```python
# src/embeddings.py line ~150
embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
```

> **A**: "Normalized embeddings enable cosine similarity = dot product. This makes vector search faster (just dot product, no need to divide by magnitudes) and more stable."

### Architecture Comparison

```mermaid
graph LR
    subgraph "Use Cases"
        U1[Text Classification] --> BERT
        U2[Question Answering] --> BERT
        U3[Text Generation] --> GPT
        U4["Chat/Completion"] --> GPT
        U5[Translation] --> T5["Encoder-Decoder"]
        U6[Summarization] --> T5
    end
    
    BERT["BERT (Bidirectional Encoder)"]
    GPT["GPT (Decoder-only)"]
    
    style BERT fill:#e3f2fd
    style GPT fill:#f8bbd0
    style T5 fill:#fff9c4
```

---

## Further Reading

- 📄 **Paper**: [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- 📄 **Paper**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- 📄 **Paper**: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
- 🎥 **Video**: [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

---

## Key Takeaways

✅ **Transformers revolutionized NLP** through parallel processing and attention  
✅ **Attention mechanism** lets models focus on relevant context dynamically  
✅ **Tokenization** converts text to numerical representations  
✅ **Positional encoding** provides sequence order information  
✅ **Multi-head attention** captures different linguistic patterns  

**Next**: [Embeddings and Vector Representations →](02-embeddings-vectors.md)
