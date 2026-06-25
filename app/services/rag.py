"""RAG pipeline: FAISS vector store ingestion and retrieval-augmented answering."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You answer questions strictly based on the provided context. If the "
        "context does not contain the answer, say that you cannot answer from "
        "the indexed documents. Answer in the language of the question.",
    ),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])


class EmptyIndexError(RuntimeError):
    """Raised when querying before any document has been ingested."""


class RagService:
    """Manages a FAISS vector store and answers questions over its contents."""

    def __init__(
        self,
        embeddings: Embeddings,
        llm: BaseChatModel,
        index_path: Path | None = None,
    ):
        self.embeddings = embeddings
        self.index_path = index_path
        self._llm = llm
        self._answer_chain = RAG_PROMPT | llm | StrOutputParser()
        self._store: FAISS | None = None

        if index_path is not None and index_path.exists():
            self._store = FAISS.load_local(
                str(index_path), embeddings, allow_dangerous_deserialization=True
            )

    @property
    def is_empty(self) -> bool:
        """True if no documents have been ingested yet."""
        return self._store is None

    def ingest(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embeddings)
        else:
            self._store.add_documents(documents)
        if self.index_path is not None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self.index_path))
        return len(documents)

    async def query(
        self,
        question: str,
        top_k: int = 4,
        llm: BaseChatModel | None = None,
    ) -> tuple[str, list[Document]]:
        if self._store is None:
            raise EmptyIndexError("No documents have been ingested yet")

        docs = await self._store.asimilarity_search(question, k=top_k)
        context = "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs
        )
        # Retrieval is independent of the answering LLM, so a per-request llm can
        # override the default without touching the (Voyage-built) vector store.
        chain = self._answer_chain if llm is None else RAG_PROMPT | llm | StrOutputParser()
        answer = await chain.ainvoke({"context": context, "question": question})
        return answer.strip(), docs
