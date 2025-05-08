import sys
from app.core.service import RAGService
from app.core.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, OLLAMA_BASE_URL
from utils.ollama_utils import print_ollama_status, check_ollama, check_required_models

def print_header():
    """Print application header"""
    print("\n" + "=" * 80)
    print("Retrieval Augmented Generation (RAG) System".center(80))
    print("=" * 80 + "\n")

def print_response(response):
    """Print formatted response"""
    if response:
        print("\nResponse:")
        print("-" * 80)
        print(response["response"])
        print("-" * 80)
        
        print("\nSources:")
        for i, source in enumerate(response["sources"], 1):
            print(f"{i}. From: {source['metadata'].get('source', 'Unknown')}")
    else:
        print("\nFailed to get a response.")

def interactive_mode(rag_service):
    """Run interactive query mode"""
    print_header()
    
    if rag_service.query_processor.index is None:
        print("No index available. Building index first...")
        success = rag_service.build_index()
        if not success:
            print("\nFailed to create index. Please add PDF documents to the 'pdfs' directory and try again.")
            return
        print("Index created successfully!")
    
    print("Enter your questions (type 'exit' to quit):")
    
    while True:
        query = input("\n> ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        response = rag_service.query(query)
        print_response(response)

def check_environment():
    """Check if the environment is properly set up"""
    required_models = [DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL]
    
    if not check_ollama(OLLAMA_BASE_URL):
        print("❌ Ollama is not running. Please start Ollama.")
        print(f"   Expected at: {OLLAMA_BASE_URL}")
        return False
    
    model_status = check_required_models(required_models, OLLAMA_BASE_URL)
    missing_models = [model for model, available in model_status.items() if not available]
    
    if missing_models:
        print("❌ Required models are missing. Please install them:")
        for model in missing_models:
            print(f"   ollama pull {model}")
        return False
    
    return True

def main():
    """Main application entry point"""
    print_header()
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print_ollama_status([DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL], OLLAMA_BASE_URL)
        return
    
    if not check_environment():
        print("\nEnvironment check failed. Please address the issues above and try again.")
        print("You can run 'python app.py status' to see more detailed status information.")
        return
    
    chunk_size = DEFAULT_CHUNK_SIZE
    chunk_overlap = DEFAULT_CHUNK_OVERLAP
    
    for arg in sys.argv:
        if arg.startswith("--chunk-size="):
            try:
                chunk_size = int(arg.split("=")[1])
            except (ValueError, IndexError):
                pass
        elif arg.startswith("--chunk-overlap="):
            try:
                chunk_overlap = int(arg.split("=")[1])
            except (ValueError, IndexError):
                pass
    
    rag_service = RAGService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    print("Running document layout analysis...")
    rag_service.analyze_layouts()

    if len(sys.argv) > 1 and sys.argv[1] == "index":
        print(f"Building index with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
        success = rag_service.build_index()
        if success:
            print("\nIndex created and saved successfully!")
        else:
            print("\nFailed to create index.")
    else:
        interactive_mode(rag_service)

if __name__ == "__main__":
    main()
