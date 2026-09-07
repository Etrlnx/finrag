from __future__ import annotations

import os
import warnings
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from finrag.config import config, EmbeddingConfig

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub")


def get_huggingface_embeddings(cfg: Optional[EmbeddingConfig] = None) -> Embeddings:
    cfg = cfg or config.embedding
    return HuggingFaceEmbeddings(
        model_name=cfg.model_name,
        model_kwargs={"device": cfg.device, "local_files_only": False},
        encode_kwargs={"normalize_embeddings": cfg.normalize},
    )


def get_gemini_embeddings(cfg: Optional[EmbeddingConfig] = None) -> Embeddings:
    cfg = cfg or config.embedding
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required for Gemini embeddings")
    return GoogleGenerativeAIEmbeddings(
        model=cfg.gemini_model,
        google_api_key=api_key,
    )


def get_embeddings(cfg: Optional[EmbeddingConfig] = None) -> Embeddings:
    cfg = cfg or config.embedding
    if cfg.provider == "gemini":
        return get_gemini_embeddings(cfg)
    return get_huggingface_embeddings(cfg)
