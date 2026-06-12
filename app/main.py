"""FastAPI application exposing the document analysis and RAG pipeline."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
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


def get_analysis(request: Request) -> AnalysisService:
    return request.app.state.analysis


def get_rag(request: Request) -> RagService:
    return request.app.state.rag


async def read_document(file: UploadFile) -> str:
    data = await file.read()
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
