"""
SEC 10-K RAG Pipeline - Built by Indhra
========================================

A Retrieval-Augmented Generation system for answering complex financial
and legal questions from Apple 2024 10-K and Tesla 2023 10-K SEC filings.

Why this architecture?
- Hybrid search (vector + BM25) for better recall on financial documents
- Cross-encoder reranking for precision
- Open-source LLM for cost efficiency and reproducibility

Modules:
- pdf_parser: Extract text from SEC 10-K PDFs with section awareness
- chunker: Smart text chunking with metadata preservation
- embeddings: Generate vector representations using BGE/MiniLM
- vector_store: FAISS-based storage with hybrid retrieval
- reranker: Cross-encoder based result reranking
- llm: LLM integration (local Ollama or API-based)
- pipeline: Main RAG pipeline with answer_question() interface

Author: Indhra
Date: January 2026
"""

__version__ = "0.1.0"
__author__ = "Indhra"

from .pipeline import answer_question

__all__ = ["answer_question"]
