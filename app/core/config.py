import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
PDF_DIR = os.path.join(BASE_DIR, "pdfs")

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

DEFAULT_LLM_MODEL = "gemma3:12b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

def configure_llm():
    llm = Ollama(
        model=DEFAULT_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
        temperature=0.1
    )
    Settings.llm = llm
    return llm

def configure_embeddings():
    embed_model = OllamaEmbedding(
        model_name=DEFAULT_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        dimensions=768
    )
    Settings.embed_model = embed_model
    return embed_model

def initialize_settings():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(STORAGE_DIR, exist_ok=True)
    
    configure_llm()
    configure_embeddings()
    
    return Settings 