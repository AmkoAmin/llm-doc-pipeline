import pytest

from app.services.documents import EmptyDocumentError, chunk_text, extract_text
from tests.conftest import make_pdf


def test_extract_text_from_plain_text():
    assert extract_text("notes.txt", b"  Hello world  ") == "Hello world"


def test_extract_text_from_pdf():
    pdf = make_pdf("Hello from a PDF document")
    assert "Hello from a PDF document" in extract_text("doc.pdf", pdf)


def test_extract_text_detects_pdf_without_extension():
    pdf = make_pdf("Magic bytes only")
    assert "Magic bytes only" in extract_text("upload", pdf)


def test_extract_text_empty_raises():
    with pytest.raises(EmptyDocumentError):
        extract_text("empty.txt", b"   ")


def test_chunk_text_splits_and_tags_source():
    text = "word " * 1000
    chunks = chunk_text(text, filename="big.txt", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "big.txt" for chunk in chunks)
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("short text", filename="s.txt")
    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"
