"""FastAPI application exposing the document analysis and RAG pipeline."""

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings

from app.config import settings, Settings
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


def seed_demo_index(rag_demo: RagService) -> None:
    """Populate the read-only demo index from the documents in the seed directory.

    Skips the work when the index was already built and loaded from disk, so the
    seed documents are embedded at most once.
    """
    if not rag_demo.is_empty:
        return
    seed_dir = Path(settings.seed_path)
    if not seed_dir.is_dir():
        return
    for path in sorted(seed_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            text = extract_text(path.name, path.read_bytes())
        except EmptyDocumentError:
            continue
        chunks = chunk_text(
            text,
            filename=path.name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        rag_demo.ingest(chunks)


SUPPORTED_PROVIDERS = ("anthropic", "openai")


def build_llm(settings: Settings, provider: str | None = None) -> BaseChatModel:
    """Build a chat LLM for the given provider ("anthropic" or "openai").

    The provider is selectable because the rest of the pipeline only depends on
    the BaseChatModel interface. This is independent of the embedding model: the
    chat LLM never touches the vector store, so the FAISS index (built with
    Voyage) stays valid regardless of this choice.
    """
    provider = (provider or settings.llm_provider).lower()
    if provider == "openai":
        # No temperature: kept consistent with the Anthropic path / default.
        return ChatOpenAI(
            model=settings.openai_model,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key or None,
        )
    if provider == "anthropic":
        # No temperature: sampling params are rejected by some reasoning models.
        return ChatAnthropic(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.anthropic_api_key or None,
        )
    raise ValueError(f"Unknown llm_provider: {provider!r}")


def build_llm_registry(settings: Settings) -> dict[str, BaseChatModel]:
    """Build every provider that can be constructed (i.e. whose key is present).

    A provider without credentials is simply left out, so the UI can offer only
    what actually works. The configured default provider must be available.
    """
    registry: dict[str, BaseChatModel] = {}
    for provider in SUPPORTED_PROVIDERS:
        try:
            registry[provider] = build_llm(settings, provider)
        except Exception as exc:  # noqa: BLE001 - missing key -> provider unavailable
            print(f"[llm] provider {provider!r} unavailable: {exc}")
    default = settings.llm_provider.lower()
    if default not in registry:
        raise RuntimeError(f"Default llm_provider {default!r} could not be built")
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    llms = build_llm_registry(settings)
    default_provider = settings.llm_provider.lower()
    llm = llms[default_provider]
    app.state.llms = llms
    app.state.default_provider = default_provider
    embeddings = VoyageAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.voyage_api_key or None,
    )
    index_path = Path(settings.vector_store_path) if settings.vector_store_path else None
    demo_index_path = (
        Path(settings.demo_vector_store_path) if settings.demo_vector_store_path else None
    )

    app.state.analysis = AnalysisService(llm, max_input_chars=settings.max_llm_input_chars)
    app.state.rag = RagService(embeddings, llm, index_path=index_path)

    rag_demo = RagService(embeddings, llm, index_path=demo_index_path)
    try:
        seed_demo_index(rag_demo)
    except Exception as exc:  # noqa: BLE001 - demo seeding must never block startup
        print(f"[demo] seeding skipped: {exc}")
    app.state.rag_demo = rag_demo
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


def get_rag_demo(request: Request) -> RagService:
    return request.app.state.rag_demo


def get_llms(request: Request) -> dict[str, BaseChatModel]:
    return request.app.state.llms


def get_default_provider(request: Request) -> str:
    return request.app.state.default_provider


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


@app.get("/providers")
async def providers(
    llms: dict[str, BaseChatModel] = Depends(get_llms),
    default: str = Depends(get_default_provider),
) -> dict[str, Any]:
    """List chat LLM providers.

    "supported" is every provider the code knows about; "providers" is the
    subset whose credentials are configured (i.e. that can actually answer).
    The UI can show the full choice and mark the rest as not enabled.
    """
    return {
        "providers": sorted(llms.keys()),
        "supported": list(SUPPORTED_PROVIDERS),
        "default": default,
    }


@app.post("/rag/query", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    rag: RagService = Depends(get_rag),
    rag_demo: RagService = Depends(get_rag_demo),
    llms: dict[str, BaseChatModel] = Depends(get_llms),
    default_provider: str = Depends(get_default_provider),
) -> QueryResponse:
    """Answer a question using retrieval-augmented generation over ingested documents."""
    service = rag_demo if body.mode == "demo" else rag
    provider = body.provider or default_provider
    if provider not in llms:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' is not available on this server.",
        )
    try:
        answer, docs = await service.query(
            body.question, top_k=body.top_k, llm=llms[provider]
        )
    except EmptyIndexError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return QueryResponse(
        question=body.question,
        answer=answer,
        sources=[SourceChunk(content=d.page_content, metadata=d.metadata) for d in docs],
    )
