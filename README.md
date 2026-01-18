# RAG API

A streamlined Retrieval-Augmented Generation (RAG) system for document Q&A, built with FastAPI and designed for deployment on Railway + Qdrant Cloud.

## Features

- **Document Ingestion**: Upload PDF and DOCX files for processing
- **Smart Chunking**: Token-based text chunking with configurable overlap
- **Vector Search**: Cosine similarity search using Qdrant
- **Semantic Reranking**: Voyage AI rerank-2.5 for improved relevance
- **AI-Powered Answers**: Google Gemini 2.5 Pro for answer generation with source citations

## Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **Vector Database**: Qdrant Cloud
- **Embeddings & Reranking**: Voyage AI (voyage-3, rerank-2)
- **LLM**: Google Gemini 2.5 Pro
- **Deployment**: Railway (Docker)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/upload` | Upload and process a document |
| GET | `/documents` | List all documents |
| DELETE | `/documents/{id}` | Delete a document |
| POST | `/query` | Ask a question, get answer with sources |
| GET | `/health` | Health check |

## Project Structure

```
/app
  /api
    /routes
      documents.py    # Document CRUD operations
      query.py        # Q&A endpoint
      health.py       # Health check
  /services
    parser.py         # PDF/DOCX text extraction
    chunker.py        # Text chunking logic
    embeddings.py     # Voyage AI embedding calls
    reranker.py       # Voyage AI reranking
    vectorstore.py    # Qdrant operations
    llm.py            # Gemini Q&A generation
  /models
    schemas.py        # Pydantic models
  config.py           # Settings via environment variables
  main.py             # FastAPI app initialization
/tests
requirements.txt
Dockerfile
.env.example
README.md
```

## Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd rag-api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
- `QDRANT_URL` - Qdrant Cloud endpoint
- `QDRANT_API_KEY` - Qdrant Cloud API key
- `VOYAGE_API_KEY` - Voyage AI API key
- `GOOGLE_API_KEY` - Google Gemini API key

### 3. Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `/docs`.

## Deployment on Railway

### 1. Create Railway Project

1. Go to [Railway](https://railway.app) and create a new project
2. Connect your GitHub repository
3. Railway will auto-detect the Dockerfile

### 2. Set Environment Variables

In Railway dashboard, add these environment variables:
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `VOYAGE_API_KEY`
- `GOOGLE_API_KEY`
- `COLLECTION_NAME` (optional, defaults to "documents")

### 3. Deploy

Railway will automatically build and deploy when you push to your main branch.

## Usage Examples

### Upload a Document

```bash
curl -X POST "https://your-api.railway.app/documents/upload" \
  -F "file=@document.pdf"
```

### Ask a Question

```bash
curl -X POST "https://your-api.railway.app/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

### List Documents

```bash
curl "https://your-api.railway.app/documents"
```

### Delete a Document

```bash
curl -X DELETE "https://your-api.railway.app/documents/{document_id}"
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `COLLECTION_NAME` | documents | Qdrant collection name |
| `CHUNK_SIZE` | 1000 | Tokens per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap tokens between chunks |
| `SEARCH_TOP_K` | 50 | Candidates for reranking |
| `RERANK_TOP_K` | 12 | Final chunks for LLM |

## License

MIT
