"""Loading and chunking of uploaded PDF and text documents."""

import io

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class EmptyDocumentError(ValueError):
    """Raised when no text could be extracted from an uploaded document."""


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from PDF or text file contents."""
    if filename.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = data.decode("utf-8", errors="replace")

    text = text.strip()
    if not text:
        raise EmptyDocumentError(f"No text could be extracted from {filename!r}")
    return text


def chunk_text(
    text: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split text into overlapping chunks for embedding and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.create_documents([text], metadatas=[{"source": filename}])
