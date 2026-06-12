import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import settings
from app.main import app, get_analysis, get_rag
from app.services.analysis import AnalysisService
from app.services.rag import RagService


class RoutingFakeChatModel(BaseChatModel):
    """Fake chat model that picks a canned response based on prompt content.

    Routing by substring (instead of call order) keeps responses correct even
    when chains run concurrently, as in AnalysisService.analyze().
    """

    responses: dict[str, str]
    default_response: str = "fake response"

    @property
    def _llm_type(self) -> str:
        return "routing-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        text = self.default_response
        for key, value in self.responses.items():
            if key in prompt:
                text = value
                break
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


CLASSIFY_JSON = '{"category": "report", "confidence": 0.92, "reasoning": "Reads like a report."}'
SUMMARY_TEXT = "This document is a short report about quarterly revenue."
EXTRACT_JSON = (
    '{"title": "Quarterly Report", "document_date": "2026-03-31", "author": "Jane Doe", '
    '"language": "en", "key_entities": ["Acme Corp", "Jane Doe"], '
    '"keywords": ["revenue", "report"]}'
)
RAG_ANSWER = "Revenue grew by 12 percent."


@pytest.fixture
def fake_llm() -> RoutingFakeChatModel:
    return RoutingFakeChatModel(
        responses={
            "document classification assistant": CLASSIFY_JSON,
            "summarization assistant": SUMMARY_TEXT,
            "extract structured metadata": EXTRACT_JSON,
            "strictly based on the provided context": RAG_ANSWER,
        }
    )


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def analysis_service(fake_llm) -> AnalysisService:
    return AnalysisService(fake_llm)


@pytest.fixture
def rag_service(fake_llm, fake_embeddings) -> RagService:
    return RagService(fake_embeddings, fake_llm, index_path=None)


@pytest.fixture
def client(analysis_service, rag_service, monkeypatch) -> TestClient:
    """Test client with fake LLM/embeddings; lifespan is intentionally not run.

    Rate limiting is disabled by default; tests that exercise it opt back in.
    """
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
    app.dependency_overrides[get_analysis] = lambda: analysis_service
    app.dependency_overrides[get_rag] = lambda: rag_service
    yield TestClient(app)
    app.dependency_overrides.clear()


def make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF containing the given text."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_pos,
    )
    return bytes(out)
