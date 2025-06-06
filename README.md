# Retrieval Augmented Generation with LlamaIndex

This project implements a Retrieval Augmented Generation (RAG) system using LlamaIndex to process PDF documents and answer questions about them. It uses Ollama to run models locally.

## Architecture

The application follows a clean, modular architecture:

```
.
├── app/                         # Main application package
│   ├── core/                    # Core configuration and services
│   │   ├── config.py            # Application configuration
│   │   └── service.py           # Main RAG service
│   ├── document_processing/     # Document processing modules
│   │   ├── pdf_loader.py        # PDF loading functionality 
│   │   ├── layout_analysis.py   # PDF layout analyzer
│   │   └── chunker.py           # Document chunking functionality
│   ├── indexing/                # Indexing modules
│   │   └── index_manager.py     # Index creation and management
│   └── query_engine/            # Query processing modules
│       └── query_processor.py   # Query processing
├── utils/                       # Utility functions
│   └── text_utils.py            # Text processing utilities
├── pdfs/                        # Directory for PDF files
├── storage/                     # Directory for index and layout storage
│   └── layout_documents/        # Folder for layout documents per analyzed PDF
├── .env                         # Environment variables
├── app.py                       # Main command-line application
├── streamlit_app.py             # Streamlit web interface
├── run_streamlit.py             # Run Streamlit with pytorch dependencies
└── requirements.txt             # Project dependencies
```

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Install Ollama:
   Follow the instructions at [Ollama.ai](https://ollama.ai/) to install Ollama on your system.

3. Pull the required models using Ollama:
   ```
   ollama pull gemma3:12b
   ollama pull nomic-embed-text:latest
   ```

4. Add your PDF files to the `pdfs` directory. If the directory doesn't exist, create it:
   ```
   mkdir -p pdfs
   ```

5. The `.env` file is already set up to work with the default Ollama configuration:
   ```
   OLLAMA_BASE_URL=http://localhost:11434
   ```

## Usage

### Web Interface (Recommended)

The easiest way to use the system is through the Streamlit web interface:

```
streamlit run streamlit_app.py
```

This will start a local web server and open a browser window with the interface. The web interface provides:

- A user-friendly way to ask questions about your documents
- Document management (upload, delete, view)
- System settings configuration
- Index rebuilding
- Visualization of sources

### Command Line Interface

#### Building the Index

To build the index from your PDF files:

```
python app.py index
```

You can also specify custom chunking parameters:

```
python app.py index --chunk-size=768 --chunk-overlap=100
```

This will:
1. Load all PDFs from the `pdfs` directory
2. Clean and extract metadata from the text
3. Chunk the documents based on your specified parameters
4. Create a vector index using locally run embeddings via Ollama
5. Save the index to the `storage` directory

#### Querying the System

To query the system interactively through the command line:

```
python app.py
```

This will load the existing index (or create a new one if none exists) and start an interactive session where you can ask questions about your documents. The queries will be processed using the local Gemma 3 model.

## Customizing the System

### Changing Models

You can modify the models used by editing the settings in `app/core/config.py`:

```python
# Model settings
DEFAULT_LLM_MODEL = "gemma3:12b"  # Change to any other model available in Ollama
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # Change to any other embedding model
```

Available Ollama models can be found by running `ollama list` or by visiting the [Ollama Library](https://ollama.ai/library).

### Changing Chunk Size

The default implementation uses a `SentenceSplitter` with a chunk size of 512 tokens and an overlap of 50 tokens. You can modify these parameters by:

1. Using the Settings tab in the web interface
2. Using command line parameters: `--chunk-size=768 --chunk-overlap=100`
3. Modifying the defaults in `app/core/config.py`

### Adding Custom Document Types

To add support for more document types:

1. Create a new loader class in the `app/document_processing` directory
2. Implement document loading and metadata extraction
3. Update the `RAGService` class to use your new loader

## Adding More Documents

To add more documents:

1. **Using the web interface**: Navigate to the "Manage Documents" tab and use the file uploader
2. **Manually**: Place additional PDF files in the `pdfs` directory and rebuild the index 