"""
PDF Parser for SEC 10-K Documents

Extracting text from Apple and Tesla 10-K filings, keeping page numbers
and section headers for proper citations.

Using PyMuPDF (fitz) - it's fast and handles financial docs well.

Author: Indhra
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


# Error patterns I've hit - quick fixes
ERROR_GUIDE = {
    "FileNotFoundError": "Wrong path? Use absolute.",
    "fitz.EmptyFileError": "Corrupt PDF. Re-download.",
    "RuntimeError: cannot open": "PDF locked. Close viewer.",
    "UnicodeDecodeError": "Encoding issue. Try different mode.",
}


@dataclass
class PageContent:
    """A page from the PDF with text and detected sections."""
    page_num: int
    text: str
    sections: List[str]
    
    def __repr__(self):
        preview = self.text[:50].replace('\n', ' ') + "..."
        return f"PageContent(page={self.page_num}, sections={self.sections}, preview='{preview}')"


@dataclass
class Document:
    """A parsed PDF with its pages."""
    name: str  # "Apple 10-K" or "Tesla 10-K"
    source_file: str
    pages: List[PageContent]
    total_pages: int
    
    def get_full_text(self) -> str:
        """All text concatenated - for debugging."""
        return "\n\n".join(p.text for p in self.pages)
    
    def get_text_with_metadata(self) -> List[Dict]:
        """Returns text + metadata dicts for the chunker."""
        results = []
        for page in self.pages:
            results.append({
                "text": page.text,
                "page_num": page.page_num,
                "sections": page.sections,
                "document": self.name,
                "source_file": self.source_file
            })
        return results


# SEC 10-K section patterns
# These are standardized by SEC, so they should match consistently
SEC_SECTION_PATTERNS = [
    # Part headers
    (r"PART\s+I(?:\s|$)", "Part I"),
    (r"PART\s+II(?:\s|$)", "Part II"),
    (r"PART\s+III(?:\s|$)", "Part III"),
    (r"PART\s+IV(?:\s|$)", "Part IV"),
    
    # Item headers (the main sections we care about)
    (r"Item\s+1\.?\s*[-–—]?\s*Business", "Item 1"),
    (r"Item\s+1A\.?\s*[-–—]?\s*Risk\s+Factors", "Item 1A"),
    (r"Item\s+1B\.?\s*[-–—]?\s*Unresolved\s+Staff\s+Comments", "Item 1B"),
    (r"Item\s+2\.?\s*[-–—]?\s*Properties", "Item 2"),
    (r"Item\s+3\.?\s*[-–—]?\s*Legal\s+Proceedings", "Item 3"),
    (r"Item\s+4\.?\s*[-–—]?\s*Mine\s+Safety", "Item 4"),
    (r"Item\s+5\.?\s*[-–—]?\s*Market", "Item 5"),
    (r"Item\s+6\.?\s*[-–—]?\s*Selected\s+Financial", "Item 6"),
    (r"Item\s+7\.?\s*[-–—]?\s*Management", "Item 7"),
    (r"Item\s+7A\.?\s*[-–—]?\s*Quantitative", "Item 7A"),
    (r"Item\s+8\.?\s*[-–—]?\s*Financial\s+Statements", "Item 8"),
    (r"Item\s+9\.?\s*[-–—]?\s*Changes", "Item 9"),
    (r"Item\s+9A\.?\s*[-–—]?\s*Controls", "Item 9A"),
    (r"Item\s+10\.?\s*[-–—]?\s*Directors", "Item 10"),
    (r"Item\s+11\.?\s*[-–—]?\s*Executive\s+Compensation", "Item 11"),
    (r"Item\s+15\.?\s*[-–—]?\s*Exhibits", "Item 15"),
    
    # Financial statement notes (important for detailed questions)
    (r"Note\s+(\d+)\s*[-–—:]", "Note"),  # Will capture note number
    
    # Signature page (for filing date)
    (r"SIGNATURES?(?:\s|$)", "Signatures"),
]


def detect_sections(text: str) -> List[str]:
    """
    Detect SEC section headers in the text.
    
    Returns a list of section names found (e.g., ["Item 7", "Item 8"]).
    """
    found_sections = []
    
    for pattern, section_name in SEC_SECTION_PATTERNS:
        if section_name == "Note":
            # Special handling for notes - capture the note number
            matches = re.findall(pattern, text, re.IGNORECASE)
            for note_num in matches:
                found_sections.append(f"Note {note_num}")
        else:
            if re.search(pattern, text, re.IGNORECASE):
                found_sections.append(section_name)
    
    return found_sections


def clean_text(text: str) -> str:
    """
    Clean up extracted PDF text.
    
    PDFs are messy. Common issues:
    - Multiple spaces and weird whitespace
    - Page headers/footers repeated
    - Hyphenated words across lines
    - Special characters that don't render
    """
    if not text:
        return ""
    
    # Fix hyphenated words split across lines (common in PDFs)
    # "finan-\ncial" -> "financial"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Normalize whitespace - but keep paragraph structure
    # Multiple spaces -> single space
    text = re.sub(r'[^\S\n]+', ' ', text)
    
    # Multiple newlines -> double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove common PDF artifacts
    # Page numbers often appear alone on a line
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Strip leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def parse_pdf(pdf_path: str, document_name: str) -> Document:
    """
    Parse a PDF file and extract text with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        document_name: Friendly name for the document (e.g., "Apple 10-K")
    
    Returns:
        Document object with all pages and metadata
    
    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If PDF can't be parsed
    """
    path = Path(pdf_path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Fix: {ERROR_GUIDE['FileNotFoundError']}"
        )
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        error_type = type(e).__name__
        fix = ERROR_GUIDE.get(error_type, "Unknown error. Check if file is a valid PDF.")
        raise ValueError(f"Failed to open PDF: {e}\nFix: {fix}")
    
    pages = []
    
    print(f"Parsing '{document_name}' ({len(doc)} pages)...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text - using "text" mode which gives us formatted text
        # Other options: "blocks" for structured, "dict" for full layout info
        raw_text = page.get_text("text")
        
        # Clean it up
        cleaned_text = clean_text(raw_text)
        
        # Detect sections on this page
        sections = detect_sections(cleaned_text)
        
        pages.append(PageContent(
            page_num=page_num + 1,  # 1-indexed for humans
            text=cleaned_text,
            sections=sections
        ))
    
    doc.close()
    
    print(f"✓ Parsed {len(pages)} pages from {document_name}")
    
    return Document(
        name=document_name,
        source_file=str(path.name),
        pages=pages,
        total_pages=len(pages)
    )


def parse_apple_10k(data_dir: str = "data") -> Document:
    """
    Parse Apple's 2024 10-K filing.
    
    Convenience function that handles the filename.
    """
    pdf_path = Path(data_dir) / "10-Q4-2024-As-Filed.pdf"
    return parse_pdf(str(pdf_path), "Apple 10-K")


def parse_tesla_10k(data_dir: str = "data") -> Document:
    """
    Parse Tesla's 2023 10-K filing.
    
    Convenience function that handles the filename.
    """
    pdf_path = Path(data_dir) / "tsla-20231231-gen.pdf"
    return parse_pdf(str(pdf_path), "Tesla 10-K")


def parse_all_documents(data_dir: str = "data") -> List[Document]:
    """
    Parse all SEC 10-K documents in the data directory.
    
    Returns list of Document objects.
    """
    documents = []
    
    # Parse Apple 10-K
    try:
        apple_doc = parse_apple_10k(data_dir)
        documents.append(apple_doc)
    except Exception as e:
        print(f"⚠ Warning: Failed to parse Apple 10-K: {e}")
    
    # Parse Tesla 10-K
    try:
        tesla_doc = parse_tesla_10k(data_dir)
        documents.append(tesla_doc)
    except Exception as e:
        print(f"⚠ Warning: Failed to parse Tesla 10-K: {e}")
    
    return documents


# Quick test when running directly
if __name__ == "__main__":
    import sys
    
    # Try to find the data directory
    # Could be running from project root or from src/
    data_dirs = ["data", "../data", "../../data"]
    data_dir = None
    
    for d in data_dirs:
        if Path(d).exists():
            data_dir = d
            break
    
    if not data_dir:
        print("Error: Could not find data directory")
        print("Make sure PDFs are in the 'data/' folder")
        sys.exit(1)
    
    print(f"Using data directory: {data_dir}")
    print("-" * 50)
    
    # Parse documents
    docs = parse_all_documents(data_dir)
    
    # Show summary
    for doc in docs:
        print(f"\n{doc.name}:")
        print(f"  Total pages: {doc.total_pages}")
        
        # Show first few sections found
        all_sections = set()
        for page in doc.pages:
            all_sections.update(page.sections)
        
        print(f"  Sections found: {sorted(all_sections)[:10]}...")
        
        # Show sample text from first page with content
        for page in doc.pages[:5]:
            if len(page.text) > 100:
                print(f"  Sample (page {page.page_num}): {page.text[:100]}...")
                break
