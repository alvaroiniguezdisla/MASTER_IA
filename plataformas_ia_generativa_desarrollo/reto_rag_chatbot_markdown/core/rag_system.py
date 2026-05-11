from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGSystem:
    """Sistema RAG sencillo basado en documentos Markdown y vector store en memoria."""

    def __init__(
        self,
        documents_dir: str | Path = "documents",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        default_k: int = 4,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.embedding_model = embedding_model
        self.default_k = default_k

        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        self.chunks: list[Document] = []

    def load_markdown_documents(self) -> list[Document]:
        """Carga todos los documentos .md de la carpeta configurada."""
        if not self.documents_dir.exists():
            raise FileNotFoundError(
                f"No existe la carpeta de documentos: {self.documents_dir.resolve()}"
            )

        markdown_files = sorted(self.documents_dir.glob("*.md"))
        if not markdown_files:
            raise FileNotFoundError(
                f"No se encontraron documentos .md en {self.documents_dir.resolve()}"
            )

        documents: list[Document] = []
        for file_path in markdown_files:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "path": str(file_path),
                    },
                )
            )

        if not documents:
            raise ValueError("Los documentos Markdown existen, pero están vacíos.")

        return documents

    def ingest_documents(self) -> int:
        """Trocea, vectoriza e inserta los documentos en InMemoryVectorStore."""
        raw_documents = self.load_markdown_documents()
        self.chunks = self.text_splitter.split_documents(raw_documents)

        for index, chunk in enumerate(self.chunks, start=1):
            chunk.metadata["chunk_id"] = index

        self.vector_store.add_documents(self.chunks)
        return len(self.chunks)

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Recupera los fragmentos más similares a la consulta del usuario."""
        if not query.strip():
            return []

        top_k = k or self.default_k
        return self.vector_store.similarity_search(query, k=top_k)

    @staticmethod
    def format_context(documents: Iterable[Document]) -> str:
        """Convierte los chunks recuperados en un contexto legible para el LLM."""
        formatted_chunks: list[str] = []
        for idx, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "documento_desconocido")
            chunk_id = document.metadata.get("chunk_id", "sin_id")
            formatted_chunks.append(
                f"[Fragmento {idx} | fuente={source} | chunk={chunk_id}]\n"
                f"{document.page_content}"
            )
        return "\n\n---\n\n".join(formatted_chunks)
