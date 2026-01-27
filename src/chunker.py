"""
Text Chunker for SEC 10-K Documents
====================================

What we're doing:
    Breaking down large documents into smaller, semantically meaningful chunks
    while preserving metadata (document, section, page) for accurate citations.

Why this approach:
    - 512-768 tokens is optimal for most embedding models
    - 100-150 token overlap prevents losing context at chunk boundaries
    - Section-aware splitting keeps related content together
    - Metadata preservation enables proper citations like ["Apple 10-K", "Item 8", "p. 28"]

Chunking strategies considered:
    1. Fixed-size (naive): Simple but breaks mid-sentence
    2. Recursive (better): Splits on paragraphs, then sentences
    3. Semantic (best): Groups by meaning - but slower
    
We use recursive with section awareness - good balance of quality and speed.

Author: Indhra
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import tiktoken

# Error patterns I've encountered
ERROR_GUIDE = {
    "tiktoken encoding not found": "Run: pip install tiktoken",
    "Empty document": "PDF parsing returned no text. Check pdf_parser.py",
    "Chunk too small": "Increase chunk_size or decrease overlap",
}


@dataclass
class Chunk:
    """
    A chunk of text with metadata for retrieval.
    
    This is the atomic unit that gets embedded and stored in the vector DB.
    """
    chunk_id: str
    text: str
    document: str      # "Apple 10-K" or "Tesla 10-K"
    section: str       # "Item 7", "Note 9", etc.
    page_start: int    # Starting page number
    page_end: int      # Ending page (same as start for single-page chunks)
    source_file: str   # Original PDF filename
    token_count: int   # Number of tokens (for context window management)
    
    def get_source_citation(self) -> List[str]:
        """
        Returns citation in the format required: ["Apple 10-K", "Item 8", "p. 28"]
        """
        page_str = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        return [self.document, self.section, page_str]
    
    def __repr__(self):
        preview = self.text[:50].replace('\n', ' ') + "..."
        return f"Chunk(id={self.chunk_id}, doc={self.document}, sec={self.section}, pages={self.page_start}-{self.page_end}, tokens={self.token_count})"


class TextChunker:
    """
    Chunks documents with metadata preservation.
    
    Usage:
        chunker = TextChunker(chunk_size=512, overlap=100)
        chunks = chunker.chunk_document(document)
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        encoding_name: str = "cl100k_base"  # GPT-4/Claude tokenizer
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target number of tokens per chunk (512-768 recommended)
            chunk_overlap: Overlap between chunks to preserve context
            encoding_name: Tokenizer to use (cl100k_base is standard)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load tokenizer for accurate token counting
        # We use tiktoken (same as OpenAI) for consistency
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            print(f"Warning: Failed to load tiktoken: {e}")
            print("Falling back to approximate token counting (words / 0.75)")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 0.75 words
            return int(len(text.split()) / 0.75)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Handles common cases in financial documents:
        - Abbreviations (Inc., Ltd., Co.)
        - Numbers with decimals ($1.5 billion)
        - Section references (Item 7. Management)
        """
        # Protect common abbreviations from being split
        protected = text
        abbreviations = ['Inc.', 'Ltd.', 'Co.', 'Corp.', 'No.', 'vs.', 'Mr.', 'Mrs.', 'Dr.', 'etc.']
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split on sentence-ending punctuation followed by space and capital
        # This regex handles: . ! ? followed by space and uppercase letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
        
        # Restore protected abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (double newline separated)."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _create_chunk(
        self,
        text: str,
        chunk_id: str,
        document: str,
        section: str,
        page_start: int,
        page_end: int,
        source_file: str
    ) -> Chunk:
        """Create a Chunk object."""
        return Chunk(
            chunk_id=chunk_id,
            text=text.strip(),
            document=document,
            section=section,
            page_start=page_start,
            page_end=page_end,
            source_file=source_file,
            token_count=self.count_tokens(text)
        )
    
    def _recursive_split(self, text: str, separators: List[str] = None) -> List[str]:
        """
        Recursively split text using a hierarchy of separators.
        
        Order: paragraphs -> sentences -> words
        This preserves as much semantic coherence as possible.
        """
        if separators is None:
            separators = ['\n\n', '\n', '. ', ' ']
        
        if not separators:
            # Base case: just return the text
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        if separator == '. ':
            splits = self._split_into_sentences(text)
        else:
            splits = text.split(separator)
        
        # If we got good splits, return them
        # Otherwise, try the next separator
        if len(splits) > 1:
            return splits
        else:
            return self._recursive_split(text, remaining_separators)
    
    def chunk_text(
        self,
        text: str,
        document: str,
        section: str,
        page_start: int,
        page_end: int,
        source_file: str,
        base_chunk_id: str = "chunk"
    ) -> List[Chunk]:
        """
        Chunk a piece of text with the given metadata.
        
        This is the core chunking logic.
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        chunk_idx = 0
        
        # First, split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk_parts = []
        current_token_count = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If adding this paragraph would exceed chunk size
            if current_token_count + para_tokens > self.chunk_size:
                # If we have accumulated content, save it as a chunk
                if current_chunk_parts:
                    chunk_text = '\n\n'.join(current_chunk_parts)
                    chunk_id = f"{base_chunk_id}_{chunk_idx}"
                    
                    chunks.append(self._create_chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        document=document,
                        section=section,
                        page_start=page_start,
                        page_end=page_end,
                        source_file=source_file
                    ))
                    
                    chunk_idx += 1
                    
                    # Start new chunk with overlap
                    # Take the last part(s) for overlap
                    overlap_parts = []
                    overlap_tokens = 0
                    for part in reversed(current_chunk_parts):
                        part_tokens = self.count_tokens(part)
                        if overlap_tokens + part_tokens <= self.chunk_overlap:
                            overlap_parts.insert(0, part)
                            overlap_tokens += part_tokens
                        else:
                            break
                    
                    current_chunk_parts = overlap_parts
                    current_token_count = overlap_tokens
                
                # If single paragraph is too large, split it further
                if para_tokens > self.chunk_size:
                    # Split into sentences
                    sentences = self._split_into_sentences(para)
                    
                    for sent in sentences:
                        sent_tokens = self.count_tokens(sent)
                        
                        if current_token_count + sent_tokens > self.chunk_size and current_chunk_parts:
                            chunk_text = ' '.join(current_chunk_parts)
                            chunk_id = f"{base_chunk_id}_{chunk_idx}"
                            
                            chunks.append(self._create_chunk(
                                text=chunk_text,
                                chunk_id=chunk_id,
                                document=document,
                                section=section,
                                page_start=page_start,
                                page_end=page_end,
                                source_file=source_file
                            ))
                            
                            chunk_idx += 1
                            current_chunk_parts = []
                            current_token_count = 0
                        
                        current_chunk_parts.append(sent)
                        current_token_count += sent_tokens
                else:
                    current_chunk_parts.append(para)
                    current_token_count += para_tokens
            else:
                current_chunk_parts.append(para)
                current_token_count += para_tokens
        
        # Don't forget the last chunk!
        if current_chunk_parts:
            chunk_text = '\n\n'.join(current_chunk_parts)
            chunk_id = f"{base_chunk_id}_{chunk_idx}"
            
            chunks.append(self._create_chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                document=document,
                section=section,
                page_start=page_start,
                page_end=page_end,
                source_file=source_file
            ))
        
        return chunks
    
    def chunk_document(self, document) -> List[Chunk]:
        """
        Chunk an entire Document object from pdf_parser.
        
        Args:
            document: Document object with pages
        
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        current_section = "General"  # Default section
        
        # Track which section we're in as we go through pages
        for page in document.pages:
            # Update current section if we find new section headers
            if page.sections:
                # Use the most specific section found
                # Priority: Notes > Items > Parts
                for section in reversed(page.sections):
                    if section.startswith("Note"):
                        current_section = section
                        break
                    elif section.startswith("Item"):
                        current_section = section
                    elif current_section == "General":
                        current_section = section
            
            # Chunk this page's text
            if page.text.strip():
                base_id = f"{document.name.replace(' ', '_')}_p{page.page_num}"
                
                page_chunks = self.chunk_text(
                    text=page.text,
                    document=document.name,
                    section=current_section,
                    page_start=page.page_num,
                    page_end=page.page_num,
                    source_file=document.source_file,
                    base_chunk_id=base_id
                )
                
                all_chunks.extend(page_chunks)
        
        return all_chunks
    
    def chunk_documents(self, documents: List) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of Document objects
        
        Returns:
            Combined list of all Chunk objects
        """
        all_chunks = []
        
        for doc in documents:
            print(f"Chunking {doc.name}...")
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            print(f"  → Created {len(doc_chunks)} chunks")
        
        print(f"\n✓ Total chunks: {len(all_chunks)}")
        
        # Print some stats
        token_counts = [c.token_count for c in all_chunks]
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            print(f"  Average tokens per chunk: {avg_tokens:.0f}")
            print(f"  Min/Max tokens: {min(token_counts)}/{max(token_counts)}")
        
        return all_chunks


def get_default_chunker() -> TextChunker:
    """
    Get a chunker with sensible defaults for SEC 10-K documents.
    
    These settings are tuned for:
    - BGE embedding models (work well with ~512 tokens)
    - Cross-encoder rerankers (context window ~512)
    - Accurate citation metadata
    """
    return TextChunker(
        chunk_size=512,     # Sweet spot for embedding models
        chunk_overlap=100   # ~20% overlap prevents context loss
    )


# Quick test when running directly
if __name__ == "__main__":
    # Test with a sample text
    sample_text = """
    Item 7. Management's Discussion and Analysis
    
    Revenue for fiscal year 2024 was $391,036 million, representing a decrease 
    of 2% compared to the prior year. The decrease was driven primarily by 
    lower iPhone sales, partially offset by growth in Services revenue.
    
    Our Services segment continued to grow, reaching $85.2 billion in annual 
    revenue. This represents a 14% increase year-over-year, driven by 
    increased subscriptions across Apple Music, Apple TV+, and iCloud.
    
    We returned approximately $100 billion to shareholders during fiscal 2024 
    through dividends and share repurchases. Our capital return program 
    remains a key priority for returning value to shareholders.
    """
    
    chunker = get_default_chunker()
    
    # Test the chunking
    chunks = chunker.chunk_text(
        text=sample_text,
        document="Apple 10-K",
        section="Item 7",
        page_start=50,
        page_end=51,
        source_file="10-Q4-2024-As-Filed.pdf",
        base_chunk_id="test"
    )
    
    print("Test Chunks:")
    print("-" * 50)
    for chunk in chunks:
        print(f"\nChunk: {chunk.chunk_id}")
        print(f"  Section: {chunk.section}")
        print(f"  Pages: {chunk.page_start}-{chunk.page_end}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Citation: {chunk.get_source_citation()}")
        print(f"  Text preview: {chunk.text[:100]}...")
