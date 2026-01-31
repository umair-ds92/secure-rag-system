"""
Vector store manager for secure RAG system using ChromaDB.
Handles embedding generation, storage, and retrieval operations.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector storage and retrieval using ChromaDB.
    Provides high-level interface for document indexing and search.
    """
    
    def __init__(
        self,
        collection_name: str = "secure_documents",
        persist_directory: str = "./data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        distance_metric: str = "cosine"
    ):
        """
        Initialize vector store manager.
        
        Args:
            collection_name: Name of ChromaDB collection
            persist_directory: Directory to persist vector database
            embedding_model: Sentence transformer model name
            distance_metric: Distance metric for similarity search
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.distance_metric = distance_metric
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(
            f"Initialized VectorStoreManager: collection='{collection_name}', "
            f"documents={self.get_collection_size()}"
        )
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We handle embeddings manually
            )
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
                embedding_function=None
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to vector store.
        
        Args:
            documents: List of document dictionaries with 'content', 'metadata', 'id'
            batch_size: Number of documents to process in each batch
            
        Returns:
            Number of documents successfully added
        """
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        added_count = 0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Extract data from batch
                ids = [doc["id"] for doc in batch]
                texts = [doc["content"] for doc in batch]
                metadatas = [doc["metadata"] for doc in batch]
                
                # Generate embeddings
                embeddings = self.generate_embeddings(texts)
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                added_count += len(batch)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
                
            except Exception as e:
                logger.error(f"Error adding batch starting at index {i}: {str(e)}")
                continue
        
        logger.info(f"Successfully added {added_count} documents to vector store")
        
        return added_count
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Dictionary with search results
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Build search parameters
        search_params = {
            "query_embeddings": [query_embedding],
            "n_results": k
        }
        
        # Add metadata filter if provided
        if filter_metadata:
            search_params["where"] = filter_metadata
        
        # Execute search
        results = self.collection.query(**search_params)
        
        # Format results
        formatted_results = {
            "query": query,
            "k": k,
            "documents": [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }
        
        # Combine results
        if results["documents"] and results["documents"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results["documents"].append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "rank": i + 1
                })
        
        logger.info(f"Search completed: found {len(formatted_results['documents'])} results")
        
        return formatted_results
    
    def get_collection_size(self) -> int:
        """Get number of documents in collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def reset_collection(self):
        """Reset collection (delete and recreate)."""
        self.delete_collection()
        self.collection = self._get_or_create_collection()
        logger.info(f"Reset collection: {self.collection_name}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dictionary or None if not found
        """
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
        
        return None
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of query strings
            k: Number of results per query
            
        Returns:
            List of search result dictionaries
        """
        results = []
        
        for query in queries:
            result = self.search(query, k=k)
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize vector store
    vector_store = VectorStoreManager(
        collection_name="test_collection",
        persist_directory="./data/test_chroma"
    )
    
    # Example: Add sample documents
    sample_docs = [
        {
            "id": "doc_1",
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "ml_intro.txt", "category": "AI"}
        },
        {
            "id": "doc_2",
            "content": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"source": "dl_basics.txt", "category": "AI"}
        }
    ]
    
    vector_store.add_documents(sample_docs)
    
    # Example: Search
    results = vector_store.search("What is machine learning?", k=2)
    
    print(f"\nSearch results:")
    for doc in results["documents"]:
        print(f"- {doc['content'][:100]}... (distance: {doc['distance']:.3f})")