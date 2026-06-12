"""FastAPI application exposing the document analysis and RAG pipeline."""

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings

from app.config import settings
from app.schemas import (
    AnalyzeResponse,
    ClassifyResponse,
    ExtractResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    SummarizeResponse,
)
from app.services.analysis import AnalysisService
from app.services.documents import EmptyDocumentError, chunk_text, extract_text
from app.services.rag import EmptyIndexError, RagService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No temperature: sampling params are rejected by claude-opus-4-8.
    llm = ChatAnthropic(
        model=settings.anthropic_model,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key or None,
    )
    embeddings = VoyageAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.voyage_api_key or None,
    )
    index_path = Path(settings.vector_store_path) if settings.vector_store_path else None

    app.state.analysis = AnalysisService(llm, max_input_chars=settings.max_llm_input_chars)
    app.state.rag = RagService(embeddings, llm, index_path=index_path)
    yield


app = FastAPI(
    title="LLM Document Analysis Pipeline",
    description=(
        "Microservice that classifies, summarizes and extracts structured JSON "
        "data from unstructured PDF and text documents, with a RAG pipeline "
        "backed by a FAISS vector store."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class SlidingWindowRateLimiter:
    """In-memory per-key sliding window. Sufficient for a single instance."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True


rate_limiter = SlidingWindowRateLimiter()


def client_ip(request: Request) -> str:
    # Behind Cloudflare Tunnel every connection comes from localhost; the real
    # client address is only available via the CF-Connecting-IP header.
    return request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else "unknown"
    )


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    limit = settings.rate_limit_per_minute
    if limit > 0 and request.method == "POST":
        if not rate_limiter.allow(client_ip(request), limit):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded, try again in a minute."},
            )
    return await call_next(request)


# Added after the rate limiter so CORS runs outermost and 429 responses
# still carry CORS headers (otherwise the browser hides them from the page).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_analysis(request: Request) -> AnalysisService:
    return request.app.state.analysis


def get_rag(request: Request) -> RagService:
    return request.app.state.rag


async def read_document(file: UploadFile) -> str:
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.max_upload_bytes // (1024 * 1024)} MB).",
        )
    try:
        return extract_text(file.filename or "upload", data)
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    file: UploadFile,
    analysis: AnalysisService = Depends(get_analysis),
) -> AnalyzeResponse:
    """Classify, summarize and extract structured data in one call."""
    text = await read_document(file)
    classification, summary, extracted = await analysis.analyze(text)
    return AnalyzeResponse(
        filename=file.filename or "upload",
        classification=classification,
        summary=summary,
        extracted_data=extracted,
    )


@app.post("/documents/classify", response_model=ClassifyResponse)
async def classify_document(
    file: UploadFile,
    analysis: AnalysisService = Depends(get_analysis),
) -> ClassifyResponse:
    text = await read_document(file)
    return ClassifyResponse(
        filename=file.filename or "upload",
        classification=await analysis.classify(text),
    )


@app.post("/documents/summarize", response_model=SummarizeResponse)
async def summarize_document(
    file: UploadFile,
    analysis: AnalysisService = Depends(get_analysis),
) -> SummarizeResponse:
    text = await read_document(file)
    return SummarizeResponse(
        filename=file.filename or "upload",
        summary=await analysis.summarize(text),
    )


@app.post("/documents/extract", response_model=ExtractResponse)
async def extract_document(
    file: UploadFile,
    analysis: AnalysisService = Depends(get_analysis),
) -> ExtractResponse:
    text = await read_document(file)
    return ExtractResponse(
        filename=file.filename or "upload",
        extracted_data=await analysis.extract(text),
    )


@app.post("/rag/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    rag: RagService = Depends(get_rag),
) -> IngestResponse:
    """Chunk a document and add it to the FAISS vector store."""
    text = await read_document(file)
    chunks = chunk_text(
        text,
        filename=file.filename or "upload",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return IngestResponse(
        filename=file.filename or "upload",
        chunks_indexed=rag.ingest(chunks),
    )


@app.post("/rag/query", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    rag: RagService = Depends(get_rag),
) -> QueryResponse:
    """Answer a question using retrieval-augmented generation over ingested documents."""
    try:
        answer, docs = await rag.query(body.question, top_k=body.top_k)
    except EmptyIndexError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return QueryResponse(
        question=body.question,
        answer=answer,
        sources=[SourceChunk(content=d.page_content, metadata=d.metadata) for d in docs],
    )
