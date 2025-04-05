from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from app.core.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class DocumentChunker:
    """Class for chunking documents into nodes"""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        Process documents into nodes using the node parser.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Node objects
        """
        nodes = self.node_parser.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} nodes from {len(documents)} documents")
        
        return nodes 