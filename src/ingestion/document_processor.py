"""
Document processing module for secure RAG system.
Handles document loading, parsing, and semantic chunking.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import re

# Document parsers
from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
import markdown

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    chunk_index: int
    
    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if not self.chunk_id:
            self.chunk_id = self._generate_chunk_id()
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID based on content hash."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.document_id}_chunk_{self.chunk_index}_{content_hash}"


class DocumentProcessor:
    """
    Processes documents into chunks suitable for vector storage.
    Supports multiple file formats with configurable chunking strategy.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size for chunks (in tokens/characters)
            chunk_overlap: Overlap between chunks to maintain context
            min_chunk_size: Minimum chunk size to create
            max_chunk_size: Maximum chunk size allowed
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        logger.info(
            f"Initialized DocumentProcessor: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}"
        )
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load document from file path.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with document content and metadata
            
        Raises:
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Determine file type and parse
        suffix = path.suffix.lower()
        
        if suffix == ".txt":
            content = self._load_txt(path)
        elif suffix == ".pdf":
            content = self._load_pdf(path)
        elif suffix == ".docx":
            content = self._load_docx(path)
        elif suffix == ".md":
            content = self._load_markdown(path)
        elif suffix == ".html":
            content = self._load_html(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Generate document ID
        doc_id = self._generate_document_id(path)
        
        metadata = {
            "source": str(path),
            "filename": path.name,
            "file_type": suffix,
            "file_size": path.stat().st_size,
            "document_id": doc_id
        }
        
        logger.info(f"Loaded document: {path.name} ({len(content)} characters)")
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def _load_txt(self, path: Path) -> str:
        """Load plain text file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _load_pdf(self, path: Path) -> str:
        """Load PDF file."""
        reader = PdfReader(path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    def _load_docx(self, path: Path) -> str:
        """Load DOCX file."""
        doc = DocxDocument(path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(paragraphs)
    
    def _load_markdown(self, path: Path) -> str:
        """Load Markdown file and convert to plain text."""
        with open(path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML, then extract text
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator='\n\n')
    
    def _load_html(self, path: Path) -> str:
        """Load HTML file and extract text."""
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        return soup.get_text(separator='\n\n')
    
    def _generate_document_id(self, path: Path) -> str:
        """Generate unique document ID based on file path."""
        path_str = str(path.absolute())
        doc_hash = hashlib.md5(path_str.encode()).hexdigest()[:12]
        return f"doc_{doc_hash}"
    
    def chunk_document(
        self,
        document: Dict[str, Any],
        metadata_override: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Split document into semantic chunks.
        
        Args:
            document: Document dictionary with 'content' and 'metadata'
            metadata_override: Optional metadata to override/extend
            
        Returns:
            List of DocumentChunk objects
        """
        content = document["content"]
        base_metadata = document["metadata"].copy()
        
        if metadata_override:
            base_metadata.update(metadata_override)
        
        # Split into sentences/paragraphs first
        text_splits = self._split_text(content)
        
        # Create chunks with overlap
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for split in text_splits:
            split_length = len(split)
            
            # If single split is too large, force split it
            if split_length > self.max_chunk_size:
                if current_chunk:
                    # Save current chunk first
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        base_metadata,
                        chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0
                
                # Force split large text
                forced_chunks = self._force_split(split)
                for forced_chunk in forced_chunks:
                    chunks.append(self._create_chunk(
                        forced_chunk,
                        base_metadata,
                        chunk_index
                    ))
                    chunk_index += 1
                continue
            
            # Check if adding this split would exceed chunk size
            if current_length + split_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_text,
                        base_metadata,
                        chunk_index
                    ))
                    chunk_index += 1
                
                # Keep overlap
                overlap_text = chunk_text[-self.chunk_overlap:] if len(chunk_text) > self.chunk_overlap else chunk_text
                current_chunk = [overlap_text, split]
                current_length = len(overlap_text) + split_length
            else:
                current_chunk.append(split)
                current_length += split_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_text,
                    base_metadata,
                    chunk_index
                ))
        
        logger.info(
            f"Created {len(chunks)} chunks from document {base_metadata.get('filename', 'unknown')}"
        )
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into semantic units (sentences/paragraphs).
        
        Args:
            text: Text to split
            
        Returns:
            List of text segments
        """
        # Split by double newlines (paragraphs) first
        paragraphs = text.split('\n\n')
        
        splits = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is short enough, keep it
            if len(para) <= self.chunk_size:
                splits.append(para)
            else:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                splits.extend([s.strip() for s in sentences if s.strip()])
        
        return splits
    
    def _force_split(self, text: str) -> List[str]:
        """
        Force split text that's too large.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Find last space before chunk_size
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_index: int
    ) -> DocumentChunk:
        """Create DocumentChunk object."""
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["chunk_length"] = len(content)
        
        return DocumentChunk(
            content=content.strip(),
            metadata=chunk_metadata,
            chunk_id="",  # Will be auto-generated
            document_id=metadata["document_id"],
            chunk_index=chunk_index
        )
    
    def process_documents(
        self,
        file_paths: List[str],
        batch_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process multiple documents.
        
        Args:
            file_paths: List of document file paths
            batch_metadata: Metadata to apply to all documents
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"Processing document {i}/{len(file_paths)}: {file_path}")
                
                # Load document
                document = self.load_document(file_path)
                
                # Chunk document
                chunks = self.chunk_document(document, batch_metadata)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(file_paths)} documents into {len(all_chunks)} chunks")
        
        return all_chunks


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    processor = DocumentProcessor(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Example: Process a single document
    doc = processor.load_document("data/sample_docs/report1.txt")
    chunks = processor.chunk_document(doc)
    
    print(f"\nCreated {len(chunks)} chunks")
    print(f"\nFirst chunk preview:")
    print(f"ID: {chunks[0].chunk_id}")
    print(f"Content: {chunks[0].content[:200]}...")