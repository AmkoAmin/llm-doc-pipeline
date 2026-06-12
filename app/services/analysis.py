"""LLM chains for document classification, summarization and structured extraction."""

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.schemas import ClassificationResult, ExtractionResult

CATEGORIES = ["invoice", "contract", "report", "resume", "letter", "scientific_paper", "other"]

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a document classification assistant. Classify the document into "
        "exactly one of these categories: {categories}. "
        "Respond with a JSON object with keys: "
        '"category" (string), "confidence" (float between 0 and 1), '
        '"reasoning" (one short sentence). Respond with JSON only.',
    ),
    ("human", "Document:\n\n{text}"),
])

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise summarization assistant. Summarize the document in at "
        "most 5 sentences, in the same language as the document. Respond with the "
        "summary only.",
    ),
    ("human", "Document:\n\n{text}"),
])

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract structured metadata from documents. Respond with a JSON object "
        "with exactly these keys: "
        '"title" (string or null), "document_date" (ISO date string or null), '
        '"author" (string or null), "language" (ISO 639-1 code or null), '
        '"key_entities" (list of strings: people, organizations, places), '
        '"keywords" (list of up to 8 strings). '
        "Use null for anything not present in the document. Respond with JSON only.",
    ),
    ("human", "Document:\n\n{text}"),
])


class AnalysisService:
    """Classify, summarize and extract structured data from document text."""

    def __init__(self, llm: BaseChatModel, max_input_chars: int = 24_000):
        self.max_input_chars = max_input_chars
        self._classify_chain = CLASSIFY_PROMPT | llm | JsonOutputParser()
        self._summarize_chain = SUMMARIZE_PROMPT | llm | StrOutputParser()
        self._extract_chain = EXTRACT_PROMPT | llm | JsonOutputParser()

    def _truncate(self, text: str) -> str:
        return text[: self.max_input_chars]

    async def classify(self, text: str) -> ClassificationResult:
        raw = await self._classify_chain.ainvoke(
            {"text": self._truncate(text), "categories": ", ".join(CATEGORIES)}
        )
        return ClassificationResult.model_validate(raw)

    async def summarize(self, text: str) -> str:
        summary = await self._summarize_chain.ainvoke({"text": self._truncate(text)})
        return summary.strip()

    async def extract(self, text: str) -> ExtractionResult:
        raw = await self._extract_chain.ainvoke({"text": self._truncate(text)})
        return ExtractionResult.model_validate(raw)

    async def analyze(self, text: str) -> tuple[ClassificationResult, str, ExtractionResult]:
        """Run classification, summarization and extraction concurrently."""
        return await asyncio.gather(
            self.classify(text),
            self.summarize(text),
            self.extract(text),
        )
