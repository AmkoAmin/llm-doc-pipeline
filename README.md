# LLM Document Analysis Pipeline

Microservice that classifies, summarizes and extracts structured JSON data from
unstructured PDF and text documents using GPT-4o and LangChain, plus a RAG
pipeline backed by a FAISS vector store. Exposed via async FastAPI endpoints and
containerized with Docker Compose.

## Stack

Python · LangChain · OpenAI API · FastAPI · FAISS · Docker

## API

| Endpoint | Description |
|---|---|
| `POST /documents/analyze` | Classify + summarize + extract in one call (runs concurrently) |
| `POST /documents/classify` | Classify document into a category (invoice, contract, report, …) |
| `POST /documents/summarize` | Summary in the document's language |
| `POST /documents/extract` | Structured metadata as JSON (title, date, author, entities, keywords) |
| `POST /rag/ingest` | Chunk a document and add it to the FAISS index |
| `POST /rag/query` | Ask a question over the ingested documents (RAG) |
| `GET /health` | Health check |

Document endpoints accept `multipart/form-data` uploads (PDF or plain text).
Interactive docs at `http://localhost:8000/docs`.

## Quick start

```bash
cp .env.example .env   # add your OpenAI API key
docker compose up --build
```

Example:

```bash
curl -F "file=@report.pdf" http://localhost:8000/documents/analyze
curl -F "file=@report.pdf" http://localhost:8000/rag/ingest
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue growth?"}'
```

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## Tests

The test suite runs without an OpenAI key — LLM and embeddings are faked:

```bash
pytest
```

## Configuration

Set via environment variables or `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | – | OpenAI API key (required at runtime) |
| `OPENAI_MODEL` | `gpt-4o` | Chat model for all LLM tasks |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for FAISS |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Text splitting for RAG ingestion |
| `VECTOR_STORE_PATH` | `data/faiss_index` | FAISS persistence path (empty = in-memory only) |
