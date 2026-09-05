from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

from finrag.config import config, FinRAGConfig
from finrag.data import load_all_filings, split_documents
from finrag.embeddings import get_embeddings
from finrag.vectorstore import build_vector_store, load_vector_store, get_retriever
from finrag.generation import (
    build_rag_prompt,
    format_docs,
    get_rate_limited_llm,
)
from finrag.retrieval import build_retrieval_pipeline


def _as_text(value) -> str:
    """Normalise model output to a plain str.

    StrOutputParser returns a TextAccessor for some message shapes, which
    behaves like a string for printing but is not one -- json.dumps and
    isinstance checks downstream both break on it.
    """
    return value if isinstance(value, str) else str(value)


class FinRAGPipeline:
    def __init__(self, cfg: FinRAGConfig | None = None):
        self.cfg = cfg or config
        self.vector_store: Optional[FAISS] = None
        self.chain = None
        self.documents: list[Document] = []
        self.retriever = None

    def load_documents(self, include_tables: bool = True) -> list[Document]:
        """Load all filings with table extraction enabled by default."""
        print("Loading documents...")
        docs = load_all_filings(
            self.cfg.paths.manifest_path,
            include_tables=include_tables,
        )
        self.documents = split_documents(docs, self.cfg.chunking)
        return self.documents

    def build_index(self, documents: list[Document] | None = None) -> FAISS:
        docs = documents or self.documents or self.load_documents()
        self.vector_store = build_vector_store(
            docs,
            self.cfg.paths.vector_store_dir,
            get_embeddings(self.cfg.embedding),
        )
        return self.vector_store

    def load_index(self) -> FAISS:
        self.vector_store = load_vector_store(
            self.cfg.paths.vector_store_dir,
            get_embeddings(self.cfg.embedding),
        )
        return self.vector_store

    def load_documents_for_bm25(self, include_tables: bool = True) -> list[Document]:
        """Load documents for BM25 retriever (needed for hybrid retrieval)."""
        print("Loading documents for BM25...")
        docs = load_all_filings(
            self.cfg.paths.manifest_path,
            include_tables=include_tables,
        )
        self.documents = split_documents(docs, self.cfg.chunking)
        return self.documents

    def build_index(self, documents: list[Document] | None = None) -> FAISS:
        docs = documents or self.documents or self.load_documents()
        self.vector_store = build_vector_store(
            docs,
            self.cfg.paths.vector_store_dir,
            get_embeddings(self.cfg.embedding),
        )
        return self.vector_store

    def load_index(self) -> FAISS:
        self.vector_store = load_vector_store(
            self.cfg.paths.vector_store_dir,
            get_embeddings(self.cfg.embedding),
        )
        return self.vector_store

    def build_chain(
        self,
        use_bm25: bool = True,
        use_reranker: bool = True,
        use_filtering: bool = True,
    ):
        if not self.vector_store:
            self.load_index()

        dense_retriever = get_retriever(self.vector_store, k=20)  # Fetch 20 for reranker

        retrieval = build_retrieval_pipeline(
            documents=self.documents,
            dense_retriever=dense_retriever,
            cfg=self.cfg.retrieval,
            use_bm25=use_bm25,
            use_reranker=use_reranker,
            use_filtering=use_filtering,
            fetch_k=20,  # Fetch 20 candidates for reranker
            final_k=self.cfg.retrieval.k,
        )

        llm = get_rate_limited_llm(self.cfg.llm)

        self.retriever = retrieval
        self.chain = (
            {"context": retrieval | format_docs, "question": RunnablePassthrough()}
            | build_rag_prompt()
            | llm
            | StrOutputParser()
            | _as_text
        )
        return self.chain

    def query(self, question: str) -> str:
        if not self.chain:
            self.build_chain()
        return self.chain.invoke(question)

    def query_with_evidence(self, question: str) -> dict:
        """Answer plus the passages it was grounded in, for citation checking."""
        if not self.chain:
            self.build_chain()

        docs = self.retriever.invoke(question)
        return {
            "question": question,
            "answer": self.chain.invoke(question),
            "evidence": [
                {
                    "content": d.page_content,
                    "ticker": d.metadata.get("ticker"),
                    "form": d.metadata.get("form"),
                    "filing_date": d.metadata.get("filing_date"),
                    "section": d.metadata.get("section"),
                    "is_table": bool(d.metadata.get("is_table")),
                }
                for d in docs
            ],
        }


def build_production_pipeline() -> FinRAGPipeline:
    """Build the production pipeline matching the Phase 9 evaluated configuration:
    - Table-aware corpus (phase6_table_aware index)
    - BGE-base embeddings
    - Hybrid retrieval (Dense 0.7 / BM25 0.3)
    - Metadata filtering enabled
    - Cross-encoder reranker (top-20 → top-5)
    - Hardened grounded prompt
    """
    pipeline = FinRAGPipeline()
    pipeline.load_documents(include_tables=True)
    pipeline.build_index()
    pipeline.build_chain(use_bm25=True, use_reranker=True, use_filtering=True)
    return pipeline


def load_production_pipeline() -> FinRAGPipeline:
    """Load the production pipeline from existing index."""
    pipeline = FinRAGPipeline()
    pipeline.load_documents_for_bm25(include_tables=True)
    pipeline.load_index()
    pipeline.build_chain(use_bm25=True, use_reranker=True, use_filtering=True)
    return pipeline


def load_hybrid_pipeline() -> FinRAGPipeline:
    """Load hybrid (Dense + BM25) pipeline with reranking (legacy helper)."""
    pipeline = FinRAGPipeline()
    pipeline.load_documents(include_tables=True)
    pipeline.load_index()
    pipeline.build_chain(use_bm25=True, use_reranker=True, use_filtering=True)
    return pipeline


def build_baseline_pipeline() -> FinRAGPipeline:
    """Legacy baseline pipeline (dense only, no reranker, no tables)."""
    pipeline = FinRAGPipeline()
    pipeline.load_documents(include_tables=False)
    pipeline.build_index()
    pipeline.build_chain(use_bm25=False, use_reranker=False, use_filtering=False)
    return pipeline


def load_baseline_pipeline() -> FinRAGPipeline:
    """Legacy baseline pipeline (dense only, no reranker, no tables)."""
    pipeline = FinRAGPipeline()
    pipeline.load_index()
    pipeline.build_chain(use_bm25=False, use_reranker=False, use_filtering=False)
    return pipeline


def load_hybrid_pipeline() -> FinRAGPipeline:
    """Load hybrid (Dense + BM25) pipeline with reranking."""
    pipeline = FinRAGPipeline()
    pipeline.load_documents(include_tables=True)
    pipeline.load_index()
    pipeline.build_chain(use_bm25=True, use_reranker=True, use_filtering=True)
    return pipeline