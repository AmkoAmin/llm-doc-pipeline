from typing import Any

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class ExtractionResult(BaseModel):
    title: str | None = None
    document_date: str | None = None
    author: str | None = None
    language: str | None = None
    key_entities: list[str] = []
    keywords: list[str] = []


class AnalyzeResponse(BaseModel):
    filename: str
    classification: ClassificationResult
    summary: str
    extracted_data: ExtractionResult


class ClassifyResponse(BaseModel):
    filename: str
    classification: ClassificationResult


class SummarizeResponse(BaseModel):
    filename: str
    summary: str


class ExtractResponse(BaseModel):
    filename: str
    extracted_data: ExtractionResult


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class SourceChunk(BaseModel):
    content: str
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
