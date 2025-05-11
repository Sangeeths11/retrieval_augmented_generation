from typing import Optional, Dict, Any
from pathlib import Path

from app.document_processing import PDFLoader, DocumentChunker, DocumentLayoutAnalyzer
from app.indexing import IndexManager
from app.query_engine import QueryProcessor
from app.core.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, initialize_settings

class RAGService:
    """Service for orchestrating the RAG workflow"""
    
    def __init__(self, 
                 chunk_size: int = DEFAULT_CHUNK_SIZE, 
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """
        Initialize the RAG service.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        initialize_settings()
        
        self.pdf_loader = PDFLoader()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.index_manager = IndexManager()
        self.query_processor = QueryProcessor()
        self.layout_analyzer = DocumentLayoutAnalyzer()
        
        self._load_existing_index()
    
    def _load_existing_index(self) -> bool:
        """
        Load existing index if available.
        
        Returns:
            True if index was loaded, False otherwise
        """
        index = self.index_manager.load_index()
        if index:
            self.query_processor.set_index(index)
            return True
        return False
    
    def build_index(self) -> bool:
        """
        Build index from PDF documents.
        
        Returns:
            True if index was built successfully, False otherwise
        """
        try:
            # First run layout analysis
            print("[INFO] Running layout analysis before indexing...")
            layout_success = self.analyze_layouts()
            if not layout_success:
                print("[WARN] Layout analysis had some issues, but proceeding with indexing.")
            
            # Then load documents (which will now include layout information)
            documents = self.pdf_loader.load_all_pdfs()
            if not documents:
                print("No documents loaded.")
                return False
            
            nodes = self.chunker.chunk_documents(documents)
            
            index = self.index_manager.create_index(nodes)
            self.index_manager.save_index(index)
            
            self.query_processor.set_index(index)
            
            return True
        except Exception as e:
            print(f"Error building index: {e}")
            return False
    
    def query(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Process a query using the RAG system.
        
        Args:
            query_text: The query text
            
        Returns:
            Query response or None if query failed
        """
        if self.query_processor.index is None:
            if not self._load_existing_index():
                print("No index available. Building index first...")
                if not self.build_index():
                    return None
        
        return self.query_processor.query(query_text)
    
    
    def analyze_layouts(self) -> bool:
        """
        Run document layout analysis on all loaded PDF documents.

        Returns:
            True if analysis ran successfully on all files, False if any failed.
        """
        try:
            print("[INFO] Loading all PDFS for layout analysis...")
            documents = self.pdf_loader.load_all_pdfs()
            if not documents:
                print("No documents found for layout analysis.")
                return False

            base_output_dir = self.layout_analyzer.base_output_dir

            for doc in documents:
                if not doc.metadata or "file_path" not in doc.metadata:
                    print("[WARN] Skipping document without file_path in metadata.")
                    continue
                pdf_path = doc.metadata["file_path"]
                pdf_name = Path(pdf_path).stem
                output_path = base_output_dir / pdf_name

                if output_path.exists() and any(output_path.iterdir()):
                    print(f"[INFO] Skipping already analyzed PDF: {pdf_name}")
                    continue

                print(f"[INFO] Running layout analysis on {pdf_path}")
                self.layout_analyzer.analyze_pdf(pdf_path)

            return True
        except Exception as e:
            print(f"[ERROR] Layout analysis failed: {e}")
            return False