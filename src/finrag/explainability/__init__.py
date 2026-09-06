"""Explainability module for FinRAG - claim attribution, evidence tracing, and diagnostics."""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from finrag.explainability.models import (
    EvidenceChunk,
    ClaimTrace,
    ExplainableResult,
    RetrievalScore,
    RetrievalStage,
    VerificationStatus,
)

CITATION_RE = re.compile(r"\[([A-Z]+),\s*(10-K|10-Q),\s*(\d{4}-\d{2}-\d{2}),\s*([^\]]+)\]")


def _normalize_section(section: str) -> str:
    return section.strip().rstrip(".")


def _chunk_id(doc: Document) -> str:
    meta = doc.metadata
    return f"{meta.get('ticker','')}_{meta.get('form','')}_{meta.get('filing_date','')}_{_normalize_section(meta.get('section',''))}_{hash(doc.page_content[:100])%10000:04d}"


def extract_dense_scores(docs: List[Document], query: str, embeddings, vector_store) -> Dict[str, float]:
    """Extract dense vector similarity scores for docs."""
    scores = {}
    try:
        query_emb = embeddings.embed_query(query)
        for doc in docs:
            doc_emb = embeddings.embed_query(doc.page_content[:512])
            import numpy as np
            sim = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)))
            scores[_chunk_id(doc)] = sim
    except Exception:
        pass
    return scores


def extract_bm25_scores(docs: List[Document], query: str, bm25_retriever) -> Dict[str, int]:
    """Extract BM25 ranks for docs."""
    ranks = {}
    try:
        bm25_results = bm25_retriever.invoke(query)
        for rank, doc in enumerate(bm25_results, 1):
            ranks[_chunk_id(doc)] = rank
    except Exception:
        pass
    return ranks


def extract_rerank_scores(docs: List[Document], query: str, reranker: CrossEncoder) -> Dict[str, float]:
    """Extract cross-encoder rerank scores."""
    scores = {}
    if not docs:
        return scores
    pairs = [(query, doc.page_content) for doc in docs]
    preds = reranker.predict(pairs)
    for doc, score in zip(docs, preds):
        scores[_chunk_id(doc)] = float(score)
    return scores


def build_evidence_chunks(
    retrieved_docs: List[Document],
    query: str,
    dense_scores: Dict[str, float],
    bm25_ranks: Dict[str, int],
    rrf_ranks: Dict[str, int],
    rerank_scores: Dict[str, float],
    final_ranks: Dict[str, int],
) -> List[EvidenceChunk]:
    """Build EvidenceChunk objects with multi-stage scores."""
    chunks = []
    for doc in retrieved_docs:
        cid = _chunk_id(doc)
        scores = RetrievalScore(
            dense_score=dense_scores.get(cid),
            bm25_rank=bm25_ranks.get(cid),
            rrf_rank=rrf_ranks.get(cid),
            rerank_score=rerank_scores.get(cid),
            final_rank=final_ranks.get(cid),
        )
        chunks.append(EvidenceChunk(
            chunk_id=cid,
            content=doc.page_content,
            metadata=doc.metadata.copy(),
            scores=scores,
            is_table=bool(doc.metadata.get("is_table", False)),
            source_stage=RetrievalStage.RERANKED,
        ))
    return chunks


def extract_citations_from_answer(answer: str) -> List[Dict[str, str]]:
    """Extract all citations from answer text with their positions."""
    citations = []
    for match in CITATION_RE.finditer(answer):
        citations.append({
            "ticker": match.group(1),
            "form": match.group(2),
            "filing_date": match.group(3),
            "section": match.group(4).strip(),
            "raw": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })
    return citations


def extract_claims_from_answer(answer: str) -> List[Tuple[str, Optional[Dict[str, str]]]]:
    """Parse Answer section into claims, associating each with nearest citation."""
    if "Answer:" not in answer:
        return []
    ans_part = answer.split("Answer:", 1)[1]
    if "Evidence:" in ans_part:
        ans_part = ans_part.split("Evidence:", 1)[0]
    ans_part = ans_part.strip()
    citations = extract_citations_from_answer(answer)
    claims = re.split(r"(?<=[.!?])\s+", ans_part)
    result = []
    for claim in claims:
        claim = claim.strip()
        if not claim:
            continue
        claim_citations = [c for c in citations if c["start"] >= claim.find(c["raw"]) - 100 and c["start"] <= claim.find(c["raw"]) + 100]
        nearest_citation = claim_citations[-1] if claim_citations else (citations[-1] if citations else None)
        result.append((claim, nearest_citation))
    return result


def verify_claim_against_chunk(claim: str, chunk: Document, citation: Dict[str, str]) -> Tuple[VerificationStatus, float, List[str], List[str]]:
    """Verify a claim against a chunk's content."""
    # Remove citation from claim text before verification
    claim_clean = CITATION_RE.sub("", claim).strip()
    claim_lower = claim_clean.lower()
    chunk_lower = chunk.page_content.lower()
    citation_key = f"[{citation['ticker']}, {citation['form']}, {citation['filing_date']}, {citation['section']}]"
    chunk_citation = f"[{chunk.metadata.get('ticker')}, {chunk.metadata.get('form')}, {chunk.metadata.get('filing_date')}, {_normalize_section(chunk.metadata.get('section',''))}]"
    if citation_key != chunk_citation:
        return VerificationStatus.UNGROUNDED, 0.0, [], ["citation mismatch"]
    numbers = re.findall(r"\d[\d,\.]*", claim_lower)
    matched = []
    missing = []
    for num in numbers:
        if num in chunk_lower:
            matched.append(num)
        else:
            missing.append(num)
    words = re.findall(r"\b[a-z]{5,}\b", claim_lower)
    term_matched = sum(1 for t in words if t in chunk_lower)
    term_total = len(words)
    if not numbers:
        if term_total == 0:
            return VerificationStatus.VERIFIED, 1.0, [], []
        ratio = term_matched / term_total
        status = VerificationStatus.VERIFIED if ratio >= 0.7 else VerificationStatus.PARTIAL
        return status, ratio, [t for t in words if t in chunk_lower], [t for t in words if t not in chunk_lower]
    num_matched = len(matched)
    num_total = len(numbers)
    ratio = (num_matched + term_matched) / (num_total + term_total) if (num_total + term_total) > 0 else 1.0
    if ratio >= 0.65 and not missing:
        status = VerificationStatus.VERIFIED
    elif ratio >= 0.4:
        status = VerificationStatus.PARTIAL
    elif num_matched == 0 and term_matched == 0:
        status = VerificationStatus.HALLUCINATED
    else:
        status = VerificationStatus.UNGROUNDED
    return status, ratio, matched, missing


def attribute_claims_to_evidence(
    answer: str,
    evidence_chunks: List[EvidenceChunk],
) -> Tuple[List[ClaimTrace], List[str]]:
    """Map each claim in answer to its supporting evidence chunk."""
    chunk_map = {ec.chunk_id: ec for ec in evidence_chunks}
    claims = extract_claims_from_answer(answer)
    traces = []
    ungrounded = []
    for claim_text, citation in claims:
        if not citation:
            traces.append(ClaimTrace(
                claim_text=claim_text,
                citation=None,
                verification_status=VerificationStatus.UNGROUNDED,
                confidence=0.0,
            ))
            continue
        matched_chunk_id = None
        best_status = VerificationStatus.UNGROUNDED
        best_confidence = 0.0
        best_matched = []
        best_missing = []
        for ec in evidence_chunks:
            cid = ec.chunk_id
            chunk_doc = Document(page_content=ec.content, metadata=ec.metadata)
            status, conf, matched, missing = verify_claim_against_chunk(claim_text, chunk_doc, citation)
            if conf > best_confidence:
                best_confidence = conf
                best_status = status
                best_matched = matched
                best_missing = missing
                matched_chunk_id = cid
        if best_status in (VerificationStatus.UNGROUNDED, VerificationStatus.HALLUCINATED):
            ungrounded.append(citation["raw"])
        traces.append(ClaimTrace(
            claim_text=claim_text,
            citation=citation,
            matched_chunk_id=matched_chunk_id,
            verification_status=best_status,
            confidence=best_confidence,
            matched_tokens=best_matched,
            missing_tokens=best_missing,
        ))
    return traces, ungrounded


def build_explainable_result(
    answer: str,
    is_refusal: bool,
    retrieved_docs: List[Document],
    dense_scores: Dict[str, float],
    bm25_ranks: Dict[str, int],
    rrf_ranks: Dict[str, int],
    rerank_scores: Dict[str, float],
) -> ExplainableResult:
    """Construct full ExplainableResult from pipeline outputs."""
    final_ranks = {_chunk_id(doc): i + 1 for i, doc in enumerate(retrieved_docs)}
    evidence_chunks = build_evidence_chunks(
        retrieved_docs=retrieved_docs,
        query="",
        dense_scores=dense_scores,
        bm25_ranks=bm25_ranks,
        rrf_ranks=rrf_ranks,
        rerank_scores=rerank_scores,
        final_ranks=final_ranks,
    )
    claim_traces, ungrounded = attribute_claims_to_evidence(answer, evidence_chunks)
    retrieval_diagnostics = {
        "total_retrieved": len(retrieved_docs),
        "dense_scored": len(dense_scores),
        "bm25_scored": len(bm25_ranks),
        "rrf_scored": len(rrf_ranks),
        "rerank_scored": len(rerank_scores),
    }
    return ExplainableResult(
        answer=answer,
        is_refusal=is_refusal,
        claim_traces=claim_traces,
        evidence_chunks=evidence_chunks,
        ungrounded_citations=ungrounded,
        retrieval_diagnostics=retrieval_diagnostics,
    )