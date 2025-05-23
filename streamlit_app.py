import os
import streamlit as st
from app.core.service import RAGService
from app.core.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, PDF_DIR
from datetime import datetime
import time
import base64
from pathlib import Path
import streamlit.watcher.local_sources_watcher
original_get_module_paths = streamlit.watcher.local_sources_watcher.get_module_paths

def patched_get_module_paths(module):
    try:
        if hasattr(module, '__name__') and module.__name__.startswith('torch'):
            return []
        return original_get_module_paths(module)
    except (AttributeError, TypeError):
        return []

streamlit.watcher.local_sources_watcher.get_module_paths = patched_get_module_paths

st.set_page_config(
    page_title="RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        border-bottom: 2px solid transparent;
        font-weight: 500;
        color: #6b7280;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #4da6ff;
        color: #2563eb;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_service():
    """Initialize and return the RAG service"""
    try:
        rag_service = RAGService(
            chunk_size=DEFAULT_CHUNK_SIZE, 
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        return rag_service
    except Exception as e:
        st.error(f"Error initializing RAG service: {e}")
        return None

def get_file_info(file_path):
    """Get file information"""
    stats = os.stat(file_path)
    size_kb = stats.st_size / 1024
    size_display = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
    modified = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M')
    return {
        "size": size_display,
        "modified": modified
    }

def display_pdf(file_path):
    """Display a PDF file in Streamlit"""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.title("📚 Retrieval Augmented Generation System")
    
    if "selected_pdf" not in st.session_state:
        st.session_state.selected_pdf = None
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query Documents", "📄 Manage Documents", "⚙️ Settings", "📐 Layout Analysis"])
    
    st.sidebar.header("System Info")
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    with st.sidebar.expander("📄 Available Documents", expanded=True):
        if pdf_files:
            st.write(f"**{len(pdf_files)} documents loaded**")
            for pdf in pdf_files:
                col1, col2 = st.columns([4, 1])
                col1.write(f"• {pdf}")
                if col2.button("👁️", key=f"view_{pdf}", help="View PDF"):
                    st.session_state.selected_pdf = pdf
                    st.rerun()
        else:
            st.write("No documents found.")
    
    with st.sidebar.expander("🔍 Index Status", expanded=True):
        rag_service = get_rag_service()
        if hasattr(rag_service, 'query_processor') and rag_service.query_processor.index is not None:
            st.write("✅ Index is loaded and ready")
        else:
            st.write("⚠️ Index not loaded")
        
        if st.button("Rebuild Index"):
            with st.spinner("Building index..."):
                success = rag_service.build_index()
                if success:
                    st.sidebar.success("Index created successfully!")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.sidebar.error("Failed to create index.")
    
    with tab1:
        st.header("Ask Questions About Your Documents")
        
        query = st.text_area("Your question:", height=100, placeholder="What is the state of empirical user evaluation in graph visualizations?")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("Submit", use_container_width=True)
        with col2:
            pass
        
        if submit_button and query:
            rag_service = get_rag_service()
            
            with st.spinner("Searching for answer..."):
                response = rag_service.query(query)
                
                if response:
                    st.write("### Answer")
                    st.markdown(response["response"])
                    
                    st.write("### Sources")
                    for i, source in enumerate(response["sources"], 1):
                        source_name = source['metadata'].get('source', 'Unknown')
                        with st.expander(f"Source {i}: {source_name}"):
                            st.markdown(source['text'])
                            st.divider()
                            st.caption(f"From: {source_name}")
                else:
                    st.error("Failed to get a response. Please make sure your index is built.")
        
        if st.session_state.selected_pdf:
            st.divider()
            st.subheader(f"📄 Viewing: {st.session_state.selected_pdf} ")
            if st.button("Close PDF", key="close_pdf_main"):
                st.session_state.selected_pdf = None
                st.rerun()
            
            pdf_path = os.path.join(PDF_DIR, st.session_state.selected_pdf)
            display_pdf(pdf_path)
    
    with tab2:
        st.header("Manage Documents")
        
        st.subheader("Upload New Documents")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            save_button = st.button("Save Files")
            if save_button:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(PDF_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Saved: {uploaded_file.name}")
                
                st.info("Files uploaded. You might want to rebuild the index to include the new documents.")
                time.sleep(2)
                st.rerun()
        
        st.subheader("Existing Documents")
        
        if pdf_files:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            col1.write("**Filename**")
            col2.write("**Size**")
            col3.write("**Modified**")
            col4.write("**Actions**")
            
            st.divider()
            
            for pdf in pdf_files:
                file_path = os.path.join(PDF_DIR, pdf)
                info = get_file_info(file_path)
                
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                col1.write(pdf)
                col2.write(info["size"])
                col3.write(info["modified"])
                
                if col4.button("Delete", key=f"delete_{pdf}"):
                    try:
                        os.remove(file_path)
                        st.success(f"Deleted {pdf}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {e}")
        else:
            st.info("No documents found. Upload some PDF files to get started.")
    
    with tab3:
        st.header("System Settings")
        
        st.subheader("Chunking Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.number_input("Chunk Size", 
                                       min_value=128, 
                                       max_value=2048, 
                                       value=DEFAULT_CHUNK_SIZE,
                                       help="Size of text chunks in tokens")
        
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", 
                                          min_value=0, 
                                          max_value=512, 
                                          value=DEFAULT_CHUNK_OVERLAP,
                                          help="Overlap between chunks in tokens")
        
        if st.button("Apply Settings and Rebuild Index"):
            with st.spinner("Rebuilding index with new settings..."):
                rag_service = RAGService(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                success = rag_service.build_index()
                if success:
                    st.success("Settings applied and index rebuilt successfully!")
                    st.cache_resource.clear()
                else:
                    st.error("Failed to rebuild index with new settings.")
    
    with tab4:
        st.header("Document Layout Analysis")
        
        rag_service = get_rag_service()
        
        if rag_service is None:
            st.error("Error initializing RAG service. Please check logs.")
        else:
            st.write("This tool analyzes the layout of your PDF documents using YOLO object detection.")
            
            if st.button("Run Layout Analysis"):
                with st.spinner("Analyzing document layouts..."):
                    try:
                        success = rag_service.analyze_layouts()
                        if success:
                            st.success("Layout analysis completed successfully!")
                            
                            # Display the layout outputs
                            base_output_dir = rag_service.layout_analyzer.base_output_dir
                            if base_output_dir.exists():
                                st.subheader("Layout Analysis Results")
                                
                                for pdf in pdf_files:
                                    pdf_name = Path(pdf).stem
                                    output_path = base_output_dir / pdf_name
                                    
                                    if output_path.exists() and any(output_path.iterdir()):
                                        with st.expander(f"Results for {pdf}"):
                                            # Display all images in the output directory
                                            image_files = list(output_path.glob("*.png"))
                                            if image_files:
                                                for img_file in image_files:
                                                    try:
                                                        st.image(str(img_file), caption=img_file.name)
                                                    except Exception as img_err:
                                                        st.warning(f"Could not display image {img_file.name}: {img_err}")
                                            else:
                                                st.info(f"No image files found for {pdf}")
                        else:
                            st.error("Layout analysis failed. Please check the console for details.")
                    except Exception as e:
                        st.error(f"Error running layout analysis: {e}")
            
            st.info("""
            The layout analysis tool:
            1. Processes each PDF document
            2. Detects different components (text, images, tables, etc.)
            3. Creates visual representations of the layout
            4. Saves the results in the layout_outputs directory
            
            Note: Layout analysis is automatically run during index building to include 
            information about images and tables in the search index.
            """)

    st.divider()
    with st.expander("ℹ️ About this System"):
        st.write("""
        This Retrieval Augmented Generation (RAG) system uses LlamaIndex to process PDF documents and answer questions about them. 
        
        The system works by:
        1. Loading and processing PDF documents
        2. Creating vector embeddings of document chunks
        3. Finding relevant text when you ask a question
        4. Generating answers based on the retrieved context
        
        All processing is done locally using Ollama.
        """)

if __name__ == "__main__":
    main() 