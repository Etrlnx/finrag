"""Retrieval Components: Dense (Vector), Sparse (BM25), Hybrid (RRF), Metadata Filtering, and Cross-Encoder Reranking."""

from __future__ import annotations

import warnings
from typing import Optional, List, Any, Dict
from collections import defaultdict

warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


class EnsembleRetriever(BaseRetriever):
    """Simple Reciprocal Rank Fusion (RRF) ensemble retriever."""
    
    retrievers: List[BaseRetriever]
    weights: List[float]
    k: int = 5
    rrf_k: int = 60
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get results from each retriever
        all_results = []
        for i, retriever in enumerate(self.retrievers):
            try:
                docs = retriever.invoke(query)
                all_results.append((i, docs))
            except Exception:
                all_results.append((i, []))
        
        # RRF scoring
        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}
        
        for retriever_idx, docs in all_results:
            weight = self.weights[retriever_idx] if retriever_idx < len(self.weights) else 1.0
            for rank, doc in enumerate(docs):
                doc_id = f"{doc.metadata.get('ticker', '')}_{doc.metadata.get('section', '')}_{hash(doc.page_content[:100])}"
                scores[doc_id] += weight / (self.rrf_k + rank + 1)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
        
        # Sort by score
        sorted_docs = sorted(
            [(doc_map[doc_id], score) for doc_id, score in scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, _ in sorted_docs[:self.k]]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


class ContextualCompressionRetriever(BaseRetriever):
    """Retriever that compresses documents using a cross-encoder reranker."""
    
    base_retriever: BaseRetriever
    base_compressor: Any  # CrossEncoderReranker
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        return self.base_compressor.compress_documents(docs, query)
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        docs = await self.base_retriever.ainvoke(query)
        return self.base_compressor.compress_documents(docs, query)


class CrossEncoderReranker:
    """Wrapper around sentence_transformers.CrossEncoder for reranking."""
    
    def __init__(self, model: CrossEncoder, top_n: int):
        self.model = model
        self.top_n = top_n
    
    def compress_documents(self, documents: List[Document], query: str, **kwargs: Any) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.top_n]]


from finrag.config import config, RetrievalConfig
from finrag.vectorstore import get_retriever as get_dense_retriever
from finrag.retrieval.metadata_filter import (
    MetadataFilter,
    FilteredRetriever,
    extract_metadata_filter,
    COMPANY_ALIASES,
    SECTION_KEYWORDS,
)


def get_bm25_retriever(documents: List[Document], k: Optional[int] = None) -> BM25Retriever:
    """Create BM25 sparse keyword retriever from documents."""
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k or config.retrieval.k
    return retriever


def get_ensemble_retriever(
    dense_retriever: BaseRetriever,
    bm25_retriever: BaseRetriever,
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
    k: Optional[int] = None,
) -> EnsembleRetriever:
    """Create hybrid ensemble retriever combining keyword BM25 and dense vector search via RRF."""
    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm25_weight, dense_weight],
        k=k or config.retrieval.k,
    )


def get_reranker(cfg: Optional[RetrievalConfig] = None) -> CrossEncoderReranker:
    """Create cross-encoder reranker for ContextualCompressionRetriever."""
    cfg = cfg or config.retrieval
    cross_encoder = CrossEncoder(cfg.rerank_model)
    return CrossEncoderReranker(model=cross_encoder, top_n=cfg.rerank_top_k)


def get_filtered_retriever(
    base_retriever: BaseRetriever,
    metadata_filter: Optional[MetadataFilter] = None,
    auto_extract_filter: bool = True,
    k: Optional[int] = None,
    fetch_k: Optional[int] = None,
) -> FilteredRetriever:
    """Wrap retriever with pre-retrieval metadata filter.
    
    Args:
        base_retriever: The base retriever to wrap
        metadata_filter: Optional explicit metadata filter
        auto_extract_filter: Whether to auto-extract filter from query
        k: Final number of results to return (after filtering)
        fetch_k: Number of candidates to fetch before filtering (over-fetch)
    """
    fetch_k = fetch_k or (k or config.retrieval.k) * 6  # Default 6x over-fetch
    return FilteredRetriever(
        base_retriever=base_retriever,
        metadata_filter=metadata_filter,
        auto_extract_filter=auto_extract_filter,
        k=k or config.retrieval.k,
        fetch_k=fetch_k,
    )


def build_retrieval_pipeline(
    documents: List[Document],
    dense_retriever: BaseRetriever,
    cfg: Optional[RetrievalConfig] = None,
    use_bm25: bool = True,
    use_reranker: bool = False,
    use_filtering: bool = True,
    bm25_weight: Optional[float] = None,
    dense_weight: Optional[float] = None,
    metadata_filter: Optional[MetadataFilter] = None,
    fetch_k: Optional[int] = None,  # Number of candidates to fetch before filtering/reranking
    final_k: Optional[int] = None,  # Final number of results to return
) -> BaseRetriever:
    """Build complete retrieval pipeline with optional Hybrid, Filtering, and Reranking.
    
    Args:
        documents: Documents for BM25 index
        dense_retriever: Dense vector retriever
        cfg: Retrieval configuration
        use_bm25: Whether to use BM25 hybrid retrieval
        use_reranker: Whether to apply cross-encoder reranking
        use_filtering: Whether to apply metadata filtering
        bm25_weight: Weight for BM25 in hybrid (default from config)
        dense_weight: Weight for dense in hybrid (default from config)
        metadata_filter: Explicit metadata filter
        fetch_k: Number of candidates to fetch before filtering/reranking (default: final_k * 6)
        final_k: Final number of results to return (default: config.retrieval.k)
    """
    cfg = cfg or config.retrieval
    final_k = final_k or cfg.k
    fetch_k = fetch_k or final_k * 6  # Default 6x over-fetch

    retriever = dense_retriever

    if use_bm25:
        bm25 = get_bm25_retriever(documents, k=fetch_k)
        w_bm25 = bm25_weight if bm25_weight is not None else cfg.bm25_weight
        w_dense = dense_weight if dense_weight is not None else cfg.dense_weight
        retriever = get_ensemble_retriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25,
            bm25_weight=w_bm25,
            dense_weight=w_dense,
            k=fetch_k,  # Fetch more for filtering/reranking
        )

    if use_filtering:
        retriever = get_filtered_retriever(
            base_retriever=retriever,
            metadata_filter=metadata_filter,
            auto_extract_filter=True,
            k=final_k,
            fetch_k=fetch_k,
        )

    if use_reranker:
        reranker = get_reranker(cfg)
        reranker.top_n = final_k
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever,
        )

    return retriever