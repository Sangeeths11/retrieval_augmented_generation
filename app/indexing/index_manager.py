import os
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import BaseNode

from app.core.config import STORAGE_DIR

class IndexManager:
    """Class for managing index creation and storage"""
    
    def __init__(self, storage_dir: str = STORAGE_DIR):
        """
        Initialize the index manager.
        
        Args:
            storage_dir: Directory to store the index
        """
        self.storage_dir = storage_dir
    
    def create_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """
        Create a vector store index from nodes.
        
        Args:
            nodes: List of nodes to index
            
        Returns:
            VectorStoreIndex object
        """
        index = VectorStoreIndex(nodes)
        return index
    
    def save_index(self, index: VectorStoreIndex) -> None:
        """
        Save the index to storage.
        
        Args:
            index: Index to save
        """
        index.storage_context.persist(persist_dir=self.storage_dir)
        print(f"Index saved to {self.storage_dir}")
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Load index from storage if it exists.
        
        Returns:
            VectorStoreIndex object or None if no index exists
        """
        if os.path.exists(self.storage_dir):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                index = load_index_from_storage(storage_context)
                print(f"Loaded index from {self.storage_dir}")
                return index
            except Exception as e:
                print(f"Error loading index: {e}")
                return None
        
        print(f"No index found at {self.storage_dir}")
        return None 