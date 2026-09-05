from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    provider: str = "huggingface"
    model_name: str = "BAAI/bge-base-en-v1.5"
    gemini_model: str = "models/gemini-embedding-001"
    device: str = "cpu"
    normalize: bool = True

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "huggingface").lower(),
            model_name=os.getenv("DEFAULT_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            gemini_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        )


@dataclass
class LLMConfig:
    provider: str = "gemini"  # gemini, anthropic, openai, ollama
    model_name: str = "gemini-3.6-flash"
    temperature: float = 0.1
    rpm: int = 10
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
            model_name=os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            rpm=int(os.getenv("LLM_RPM", "10")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        )


@dataclass
class RetrievalConfig:
    k: int = 5
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    rerank_top_k: int = 10
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        return cls(
            k=int(os.getenv("RETRIEVAL_K", "5")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.3")),
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.7")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "10")),
            rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        )


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        )


@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    vector_store_dir: Path = Path("data/vector_stores/phase6_table_aware")
    manifest_path: Path = Path("data/manifest.json")

    @classmethod
    def from_env(cls) -> "PathsConfig":
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            raw_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
            processed_dir=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
            vector_store_dir=Path(os.getenv("VECTOR_STORE_DIR", "data/vector_stores/embeddings_bge_base")),
            manifest_path=Path(os.getenv("MANIFEST_PATH", "data/manifest.json")),
        )


@dataclass
class FinRAGConfig:
    embedding: EmbeddingConfig
    llm: LLMConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    paths: PathsConfig

    @classmethod
    def from_env(cls) -> "FinRAGConfig":
        return cls(
            embedding=EmbeddingConfig.from_env(),
            llm=LLMConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            chunking=ChunkingConfig.from_env(),
            paths=PathsConfig.from_env(),
        )

    def get_embedding_model(self) -> Optional[str]:
        if self.embedding.provider == "gemini":
            return self.embedding.gemini_model
        return self.embedding.model_name


config = FinRAGConfig.from_env()