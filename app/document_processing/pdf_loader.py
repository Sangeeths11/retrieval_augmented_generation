import os
from pathlib import Path
from typing import List, Optional
from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader

from app.core.config import PDF_DIR
from utils.text_utils import clean_text, extract_metadata

class PDFLoader:
    """Class for loading and processing PDF documents"""
    
    def __init__(self, pdf_dir: str = PDF_DIR):
        """
        Initialize the PDF loader.
        
        Args:
            pdf_dir: Directory containing PDF files
        """
        self.pdf_dir = pdf_dir
        self.reader = PyMuPDFReader()
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files in the directory"""
        pdf_dir_path = Path(self.pdf_dir)
        return list(pdf_dir_path.glob("*.pdf"))
    
    def load_single_pdf(self, pdf_path: Path) -> Optional[Document]:
        """
        Load a single PDF file and convert it to a Document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Document object or None if loading fails
        """
        try:
            # Load the PDF
            docs = self.reader.load(file_path=pdf_path)
            
            # Create a single Document with the text from all pages
            doc_text = "\n\n".join([d.get_content() for d in docs])
            
            # Clean the text
            cleaned_text = clean_text(doc_text)
            
            # Extract metadata
            metadata = extract_metadata(cleaned_text)
            
            # Add file metadata
            metadata.update({
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size": os.path.getsize(pdf_path),
                "file_type": "pdf"
            })
            
            document = Document(text=cleaned_text, metadata=metadata)
            
            return document
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            return None
    
    def load_all_pdfs(self) -> List[Document]:
        """
        Load all PDFs from the directory.
        
        Returns:
            List of Document objects
        """
        all_docs = []
        pdf_files = self.get_pdf_files()
        
        print(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}")
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            document = self.load_single_pdf(pdf_file)
            if document:
                all_docs.append(document)
                print(f"Successfully processed {pdf_file.name}")
        
        return all_docs 