import streamlit as st
from app.core.service import RAGService
from app.core.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, PDF_DIR
import os
import shutil
from datetime import datetime
import time

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
    rag_service = RAGService(
        chunk_size=DEFAULT_CHUNK_SIZE, 
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    return rag_service

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

def main():
    st.title("📚 Retrieval Augmented Generation System")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Query Documents", "📄 Manage Documents", "⚙️ Settings"])
    
    st.sidebar.header("System Info")
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    
    with st.sidebar.expander("📄 Available Documents", expanded=True):
        if pdf_files:
            st.write(f"**{len(pdf_files)} documents loaded**")
            for pdf in pdf_files:
                st.write(f"• {pdf}")
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
    
    # SETTINGS TAB
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