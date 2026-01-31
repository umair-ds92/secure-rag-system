"""
Document ingestion script for secure RAG system.
Orchestrates document processing and vector storage.
"""

import logging
import yaml
from pathlib import Path
from typing import List, Optional
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.document_processor import DocumentProcessor, DocumentChunk
from src.retrieval.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIngestor:
    """
    Orchestrates document ingestion pipeline.
    Handles processing, chunking, and storage of documents.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize document ingestor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize document processor
        doc_config = self.config.get("document_processing", {})
        self.processor = DocumentProcessor(
            chunk_size=doc_config.get("chunk_size", 512),
            chunk_overlap=doc_config.get("chunk_overlap", 50),
            min_chunk_size=doc_config.get("min_chunk_size", 100),
            max_chunk_size=doc_config.get("max_chunk_size", 1000)
        )
        
        # Initialize vector store
        vs_config = self.config.get("vector_store", {})
        self.vector_store = VectorStoreManager(
            collection_name=vs_config.get("collection_name", "secure_documents"),
            persist_directory=vs_config.get("persist_directory", "./data/chroma_db"),
            embedding_model=vs_config.get("embedding_model", "all-MiniLM-L6-v2"),
            distance_metric=vs_config.get("distance_metric", "cosine")
        )
        
        logger.info("DocumentIngestor initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            recursive: Whether to search subdirectories
            file_extensions: List of file extensions to process (e.g., ['.txt', '.pdf'])
            metadata: Optional metadata to attach to all documents
            
        Returns:
            Number of chunks ingested
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Get supported extensions from config
        if file_extensions is None:
            file_extensions = self.config.get("document_processing", {}).get(
                "supported_formats",
                [".txt", ".pdf", ".docx", ".md", ".html"]
            )
        
        # Find all matching files
        file_paths = []
        for ext in file_extensions:
            if recursive:
                file_paths.extend(dir_path.rglob(f"*{ext}"))
            else:
                file_paths.extend(dir_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(file_paths)} documents to ingest")
        
        if not file_paths:
            logger.warning(f"No documents found in {directory}")
            return 0
        
        # Process documents
        return self.ingest_files([str(p) for p in file_paths], metadata=metadata)
    
    def ingest_files(
        self,
        file_paths: List[str],
        metadata: Optional[dict] = None
    ) -> int:
        """
        Ingest specific document files.
        
        Args:
            file_paths: List of file paths to ingest
            metadata: Optional metadata to attach to all documents
            
        Returns:
            Number of chunks ingested
        """
        if not file_paths:
            logger.warning("No files provided for ingestion")
            return 0
        
        logger.info(f"Starting ingestion of {len(file_paths)} files")
        
        # Process documents into chunks
        all_chunks = self.processor.process_documents(
            file_paths,
            batch_metadata=metadata
        )
        
        if not all_chunks:
            logger.warning("No chunks created from documents")
            return 0
        
        # Prepare chunks for vector store
        vector_docs = self._prepare_chunks_for_storage(all_chunks)
        
        # Add to vector store
        added_count = self.vector_store.add_documents(vector_docs)
        
        logger.info(f"Ingestion complete: {added_count} chunks added to vector store")
        
        return added_count
    
    def _prepare_chunks_for_storage(
        self,
        chunks: List[DocumentChunk]
    ) -> List[dict]:
        """
        Convert DocumentChunk objects to format expected by vector store.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of dictionaries with id, content, and metadata
        """
        vector_docs = []
        
        for chunk in chunks:
            vector_docs.append({
                "id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata
            })
        
        return vector_docs
    
    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        return {
            "total_documents": self.vector_store.get_collection_size(),
            "collection_name": self.vector_store.collection_name,
            "embedding_model": self.vector_store.embedding_model_name
        }


def main():
    """Main entry point for document ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents into secure RAG system")
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing documents to ingest"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before ingestion"
    )
    
    args = parser.parse_args()
    
    # Initialize ingestor
    ingestor = DocumentIngestor(config_path=args.config)
    
    # Reset if requested
    if args.reset:
        logger.info("Resetting vector store...")
        ingestor.vector_store.reset_collection()
    
    # Ingest documents
    try:
        chunks_added = ingestor.ingest_directory(
            args.directory,
            recursive=args.recursive
        )
        
        # Print statistics
        stats = ingestor.get_stats()
        print("\n" + "="*60)
        print("INGESTION COMPLETE")
        print("="*60)
        print(f"Chunks added: {chunks_added}")
        print(f"Total documents in store: {stats['total_documents']}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Embedding model: {stats['embedding_model']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()