
# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about meeting call transcripts using vector search and Azure OpenAI.

## Features

- **Vector Search**: Uses embeddings to find relevant transcript chunks
- **Azure Integration**: Leverages Azure Blob Storage, Cognitive Search, and OpenAI
- **Semantic Understanding**: Powered by text-embedding-3-large for accurate context retrieval
- **LLM Responses**: Generates answers based on retrieved transcript context

## Setup

1. Clone the repository
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Create a `confidential.env` file with:
    ```
    BLOB_SAS_URL=<your_blob_url>
    AZURE_OPENAI_API_KEY=<your_api_key>
    AZURE_SEARCH_ENDPOINT=<your_search_endpoint>
    AZURE_SEARCH_KEY=<your_search_key>
    ```

## Usage

Run the application:
```bash
python src/rag/RAG_bot.py
```

### Commands

1. **Chunk** - Process transcripts and store embeddings in vector database
2. **Query** - Ask questions about transcript content
3. **Exit** - Close the application

## Architecture

- **Text Splitting**: Chunks transcripts into manageable segments
- **Embeddings**: Converts text to vectors for semantic search
- **Vector Database**: Stores and retrieves similar chunks
- **LLM**: Generates contextual responses using GPT-4 mini
