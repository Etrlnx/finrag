from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finrag.config import config, ChunkingConfig


def get_text_splitter(cfg: ChunkingConfig | None = None) -> RecursiveCharacterTextSplitter:
    cfg = cfg or config.chunking
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(
    documents: list[Document],
    cfg: ChunkingConfig | None = None,
) -> list[Document]:
    splitter = get_text_splitter(cfg)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks (chunk_size={cfg.chunk_size if cfg else config.chunking.chunk_size}, overlap={cfg.chunk_overlap if cfg else config.chunking.chunk_overlap})")
    return chunks
