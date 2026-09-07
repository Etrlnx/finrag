from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from finrag.config import config
from finrag.embeddings import get_embeddings


def build_vector_store(
    documents: list[Document],
    vector_store_dir: Path | str | None = None,
    embeddings: Embeddings | None = None,
) -> FAISS:
    store_dir = Path(vector_store_dir) if vector_store_dir else config.paths.vector_store_dir
    emb = embeddings or get_embeddings()

    print(f"Building FAISS vector store with {len(documents)} documents...")
    vector_store = FAISS.from_documents(documents, emb)

    print(f"Saving to {store_dir}...")
    store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(store_dir))

    return vector_store


def load_vector_store(
    vector_store_dir: Path | str | None = None,
    embeddings: Embeddings | None = None,
) -> FAISS:
    store_dir = Path(vector_store_dir) if vector_store_dir else config.paths.vector_store_dir
    emb = embeddings or get_embeddings()

    return FAISS.load_local(str(store_dir), emb, allow_dangerous_deserialization=True)


def get_retriever(
    vector_store: FAISS,
    k: int | None = None,
    **kwargs,
):
    return vector_store.as_retriever(
        search_kwargs={"k": k or config.retrieval.k, **kwargs}
    )
