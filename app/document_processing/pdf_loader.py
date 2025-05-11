import os
from pathlib import Path
from typing import List, Optional, Dict
from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader

from app.core.config import PDF_DIR, STORAGE_DIR
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
        self.layout_dir = Path(STORAGE_DIR) / "layout_outputs"
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files in the directory"""
        pdf_dir_path = Path(self.pdf_dir)
        return list(pdf_dir_path.glob("*.pdf"))
    
    def _get_layout_info(self, pdf_name: str) -> Dict[str, List[str]]:
        """
        Get layout analysis information for a PDF.
        
        Args:
            pdf_name: Name of the PDF file (without extension)
            
        Returns:
            Dictionary mapping page numbers to lists of descriptive text
        """
        layout_info = {}
        pdf_layout_dir = self.layout_dir / pdf_name
        
        if not pdf_layout_dir.exists():
            return layout_info
            
        for page_file in pdf_layout_dir.glob("page_*.jpg"):
            try:
                page_num = int(page_file.stem.split("_")[1])
                layout_info[page_num] = []
                
                element_types = {
                    "table": "Table",
                    "figure": "Figure",
                    "title": "Title",
                    "plain text": "Text Block"
                }
                

                for element_type, element_name in element_types.items():
                    element_dir = pdf_layout_dir / element_type
                    if not element_dir.exists():
                        continue
                        
                    elements = sorted(
                        element_dir.glob(f"page{page_num}_det*.jpg"),
                        key=lambda x: int(x.stem.split("_")[-1])
                    )
                    
                    if not elements:
                        continue
                        
                    for i, _ in enumerate(elements):
                        description = f"[{element_name} {i+1 if element_type != 'title' else ''} on page {page_num+1}"
                        
                        if element_type == "table":
                            has_caption = bool(list(pdf_layout_dir.glob(f"table_caption/page{page_num}_det*.jpg")))
                            has_footnotes = bool(list(pdf_layout_dir.glob(f"table_footnote/page{page_num}_det*.jpg")))
                            
                            if has_caption:
                                description += " with caption"
                            if has_footnotes:
                                description += " with footnotes"
                                
                            description += ". This table may contain important data or information related to the document.]"
                            
                        elif element_type == "figure":
                            has_caption = bool(list(pdf_layout_dir.glob(f"figure_caption/page{page_num}_det*.jpg")))
                            if has_caption:
                                description += " with caption"
                            description += ". This visual element may contain important information or illustrations related to the document content.]"
                            
                        elif element_type == "title":
                            description += ". This may be a section or subsection heading.]"
                            
                        elif element_type == "plain text":
                            description += ". This may contain important content.]"
                            
                        layout_info[page_num].append(description)
                
                layout_info[page_num].sort()
                
            except Exception as e:
                print(f"Error processing layout for page {page_file}: {e}")
                continue
        
        return layout_info

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
            
            # Get layout information
            pdf_name = pdf_path.stem
            layout_info = self._get_layout_info(pdf_name)
            
            # Create a single Document with the text from all pages
            doc_pages = []
            for i, doc in enumerate(docs):
                page_text = doc.get_content()
                
                # Add layout information for this page
                if i in layout_info and layout_info[i]:
                    page_text += "\n\n" + "\n".join(layout_info[i])
                
                doc_pages.append(page_text)
            
            doc_text = "\n\n".join(doc_pages)
            
            # Clean the text
            cleaned_text = clean_text(doc_text)
            
            # Extract metadata
            metadata = extract_metadata(cleaned_text)
            
            # Add file metadata
            metadata.update({
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size": os.path.getsize(pdf_path),
                "file_type": "pdf",
                "has_layout_analysis": bool(layout_info)
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