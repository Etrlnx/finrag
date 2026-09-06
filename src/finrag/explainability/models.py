"""Explainability data models for FinRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class RetrievalStage(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    RRF = "rrf"
    FILTERED = "filtered"
    RERANKED = "reranked"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    UNGROUNDED = "ungrounded"
    HALLUCINATED = "hallucinated"


@dataclass
class RetrievalScore:
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    final_rank: Optional[int] = None


@dataclass
class EvidenceChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    scores: RetrievalScore
    is_table: bool = False
    source_stage: RetrievalStage = RetrievalStage.RERANKED


@dataclass
class ClaimTrace:
    claim_text: str
    citation: Optional[Dict[str, str]] = None
    matched_chunk_id: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.UNGROUNDED
    confidence: float = 0.0
    matched_tokens: List[str] = field(default_factory=list)
    missing_tokens: List[str] = field(default_factory=list)


@dataclass
class ExplainableResult:
    answer: str
    is_refusal: bool
    claim_traces: List[ClaimTrace]
    evidence_chunks: List[EvidenceChunk]
    ungrounded_citations: List[str]
    retrieval_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "is_refusal": self.is_refusal,
            "claim_traces": [
                {
                    "claim_text": ct.claim_text,
                    "citation": ct.citation,
                    "matched_chunk_id": ct.matched_chunk_id,
                    "verification_status": ct.verification_status.value,
                    "confidence": ct.confidence,
                    "matched_tokens": ct.matched_tokens,
                    "missing_tokens": ct.missing_tokens,
                }
                for ct in self.claim_traces
            ],
            "evidence_chunks": [
                {
                    "chunk_id": ec.chunk_id,
                    "content": ec.content,
                    "metadata": ec.metadata,
                    "scores": {
                        "dense_score": ec.scores.dense_score,
                        "bm25_rank": ec.scores.bm25_rank,
                        "rrf_rank": ec.scores.rrf_rank,
                        "rerank_score": ec.scores.rerank_score,
                        "final_rank": ec.scores.final_rank,
                    },
                    "is_table": ec.is_table,
                    "source_stage": ec.source_stage.value,
                }
                for ec in self.evidence_chunks
            ],
            "ungrounded_citations": self.ungrounded_citations,
            "retrieval_diagnostics": self.retrieval_diagnostics,
        }

    def to_string(self, verbose: bool = True) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("EXPLAINABLE RESULT")
        lines.append("=" * 80)
        lines.append(f"\nANSWER: {'[REFUSAL]' if self.is_refusal else ''}")
        lines.append(self.answer)
        lines.append(f"\n{'=' * 80}")
        lines.append("CLAIM TRACES")
        lines.append("=" * 80)
        for i, ct in enumerate(self.claim_traces, 1):
            status_emoji = {
                VerificationStatus.VERIFIED: "✅",
                VerificationStatus.PARTIAL: "⚠️",
                VerificationStatus.UNGROUNDED: "❌",
                VerificationStatus.HALLUCINATED: "🚫",
            }.get(ct.verification_status, "❓")
            lines.append(f"\n{i}. {status_emoji} [{ct.verification_status.value.upper()}] confidence={ct.confidence:.2f}")
            lines.append(f"   Claim: {ct.claim_text}")
            if ct.citation:
                lines.append(f"   Citation: [{ct.citation.get('ticker')}, {ct.citation.get('form')}, {ct.citation.get('filing_date')}, {ct.citation.get('section')}]")
            if ct.matched_chunk_id:
                lines.append(f"   Matched Chunk: {ct.matched_chunk_id}")
            if ct.matched_tokens:
                lines.append(f"   Verified tokens: {', '.join(ct.matched_tokens)}")
            if ct.missing_tokens:
                lines.append(f"   Missing tokens: {', '.join(ct.missing_tokens)}")
        lines.append(f"\n{'=' * 80}")
        lines.append("EVIDENCE CHUNKS (Top 5)")
        lines.append("=" * 80)
        for ec in self.evidence_chunks:
            lines.append(f"\n📄 {ec.chunk_id} | Final Rank: {ec.scores.final_rank}")
            lines.append(f"   Metadata: {ec.metadata.get('ticker')}, {ec.metadata.get('form')}, {ec.metadata.get('filing_date')}, {ec.metadata.get('section')}")
            lines.append(f"   Table: {ec.is_table}")
            scores = ec.scores
            lines.append(f"   Scores: dense={scores.dense_score:.3f}" if scores.dense_score else "   Scores: dense=N/A")
            lines.append(f"            BM25 rank={scores.bm25_rank}" if scores.bm25_rank else "            BM25 rank=N/A")
            lines.append(f"            RRF rank={scores.rrf_rank}" if scores.rrf_rank else "            RRF rank=N/A")
            lines.append(f"            Rerank={scores.rerank_score:.3f}" if scores.rerank_score else "            Rerank=N/A")
            if verbose:
                lines.append(f"   Content: {ec.content[:300]}...")
        if self.ungrounded_citations:
            lines.append(f"\n{'=' * 80}")
            lines.append("⚠️ UNGROUNDED CITATIONS")
            lines.append("=" * 80)
            for uc in self.ungrounded_citations:
                lines.append(f"  - {uc}")
        return "\n".join(lines)