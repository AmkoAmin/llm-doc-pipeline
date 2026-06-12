from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services.rag import EmptyIndexError, RagService
from tests.conftest import RAG_ANSWER

DOCS = [
    Document(page_content="Revenue grew by 12 percent in Q1.", metadata={"source": "report.pdf"}),
    Document(page_content="The contract runs until December 2027.", metadata={"source": "contract.pdf"}),
]


def test_ingest_returns_chunk_count(rag_service):
    assert rag_service.ingest(DOCS) == 2


def test_ingest_empty_list(rag_service):
    assert rag_service.ingest([]) == 0


async def test_query_returns_answer_and_sources(rag_service):
    rag_service.ingest(DOCS)
    answer, sources = await rag_service.query("How much did revenue grow?", top_k=2)
    assert answer == RAG_ANSWER
    assert len(sources) == 2
    assert {s.metadata["source"] for s in sources} == {"report.pdf", "contract.pdf"}


async def test_query_without_ingest_raises(rag_service):
    with pytest.raises(EmptyIndexError):
        await rag_service.query("anything")


def test_index_persistence(fake_llm, fake_embeddings, tmp_path: Path):
    index_path = tmp_path / "faiss_index"
    service = RagService(fake_embeddings, fake_llm, index_path=index_path)
    service.ingest(DOCS)
    assert index_path.exists()

    reloaded = RagService(fake_embeddings, fake_llm, index_path=index_path)
    assert reloaded._store is not None
    assert reloaded._store.index.ntotal == 2
