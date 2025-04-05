from typing import Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine

class QueryProcessor:
    def __init__(self, index: Optional[VectorStoreIndex] = None):
        """
        Initialize the query processor.
        
        Args:
            index: VectorStoreIndex to query
        """
        self.index = index
        self._query_engine = None
    
    @property
    def query_engine(self) -> Optional[BaseQueryEngine]:
        """Get or create the query engine"""
        if self._query_engine is None and self.index is not None:
            self._query_engine = self.index.as_query_engine()
        return self._query_engine
    
    def set_index(self, index: VectorStoreIndex) -> None:
        """
        Set the index to query.
        
        Args:
            index: VectorStoreIndex to query
        """
        self.index = index
        self._query_engine = None
    
    def query(self, query_text: str) -> Optional[dict]:
        """
        Process a query against the index.
        
        Args:
            query_text: The query text
            
        Returns:
            Query response or None if no index is available
        """
        if self.query_engine is None:
            print("No index available for querying")
            return None
        
        response = self.query_engine.query(query_text)
        
        result = {
            "response": str(response),
            "sources": [
                {
                    "text": source_node.node.get_content(),
                    "metadata": source_node.node.metadata
                }
                for source_node in response.source_nodes
            ]
        }
        
        return result 