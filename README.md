# LLM Document Analysis Pipeline

Microservice that classifies, summarizes and extracts structured JSON data from
unstructured PDF and text documents using Claude (Anthropic API) and LangChain,
plus a RAG pipeline backed by a FAISS vector store with Voyage AI embeddings.
Exposed via async FastAPI endpoints and containerized with Docker Compose.

## Stack

Python · LangChain · Anthropic API · Voyage AI · FastAPI · FAISS · Docker

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
cp .env.example .env   # add your Anthropic and Voyage AI API keys
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

The test suite runs without any API keys — LLM and embeddings are faked:

```bash
pytest
```

## Configuration

Set via environment variables or `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Anthropic API key (required at runtime) |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model for all LLM tasks |
| `VOYAGE_API_KEY` | – | Voyage AI API key for embeddings (required at runtime) |
| `EMBEDDING_MODEL` | `voyage-3.5` | Voyage embedding model for FAISS |
| `LLM_MAX_TOKENS` | `8192` | Max output tokens per LLM call |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Text splitting for RAG ingestion |
| `VECTOR_STORE_PATH` | `data/faiss_index` | FAISS persistence path (empty = in-memory only) |
