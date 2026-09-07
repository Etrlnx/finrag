"""Query logging for FinRAG - persists queries, retrievals, and claim traces."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from finrag.persistence.models import QueryLog, RetrievedChunk, ClaimTrace, QueryStatus
from finrag.persistence.database import get_session
from finrag.explainability.models import ExplainableResult, VerificationStatus


async def log_query(
    request_id: str,
    question: str,
    answer: str,
    is_refusal: bool,
    latency_ms: float,
    model_provider: str,
    model_name: str,
    explainable: Optional[ExplainableResult] = None,
) -> Optional[QueryLog]:
    """Log a query and its results to the database."""
    try:
        async with get_session() as session:
            status = QueryStatus.REFUSAL if is_refusal else QueryStatus.SUCCESS

            query_log = QueryLog(
                id=uuid.uuid4(),
                request_id=request_id,
                question=question,
                answer=answer,
                is_refusal=is_refusal,
                status=status,
                latency_ms=latency_ms,
                model_provider=model_provider,
                model_name=model_name,
                created_at=datetime.utcnow(),
            )
            session.add(query_log)
            await session.flush()  # Get the ID

            if explainable:
                # Log retrieved chunks
                for i, chunk in enumerate(explainable.evidence_chunks):
                    retrieved = RetrievedChunk(
                        id=uuid.uuid4(),
                        query_log_id=query_log.id,
                        chunk_id=chunk.chunk_id,
                        rank=i + 1,
                        content=chunk.content,
                        ticker=chunk.metadata.get("ticker"),
                        form=chunk.metadata.get("form"),
                        filing_date=chunk.metadata.get("filing_date"),
                        section=chunk.metadata.get("section"),
                        is_table=chunk.is_table,
                        dense_score=chunk.scores.dense_score,
                        bm25_rank=chunk.scores.bm25_rank,
                        rrf_rank=chunk.scores.rrf_rank,
                        rerank_score=chunk.scores.rerank_score,
                        chunk_metadata=chunk.metadata,
                    )
                    session.add(retrieved)

                # Log claim traces
                for claim in explainable.claim_traces:
                    trace = ClaimTrace(
                        id=uuid.uuid4(),
                        query_log_id=query_log.id,
                        claim_text=claim.claim_text,
                        citation_ticker=claim.citation.get("ticker") if claim.citation else None,
                        citation_form=claim.citation.get("form") if claim.citation else None,
                        citation_filing_date=claim.citation.get("filing_date") if claim.citation else None,
                        citation_section=claim.citation.get("section") if claim.citation else None,
                        matched_chunk_id=claim.matched_chunk_id,
                        verification_status=claim.verification_status.value,
                        confidence=claim.confidence,
                        matched_tokens=claim.matched_tokens,
                        missing_tokens=claim.missing_tokens,
                        created_at=datetime.utcnow(),
                    )
                    session.add(trace)

            await session.commit()
            return query_log
    except Exception as e:
        print(f"Failed to log query to database: {e}")
        return None


async def get_query_log(request_id: str) -> Optional[QueryLog]:
    """Retrieve a query log by request ID."""
    try:
        async with get_session() as session:
            result = await session.execute(
                select(QueryLog).where(QueryLog.request_id == request_id)
            )
            return result.scalar_one_or_none()
    except Exception as e:
        print(f"Failed to get query log: {e}")
        return None


async def get_recent_queries(limit: int = 50) -> List[QueryLog]:
    """Get recent query logs."""
    try:
        async with get_session() as session:
            result = await session.execute(
                select(QueryLog)
                .order_by(QueryLog.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())
    except Exception as e:
        print(f"Failed to get recent queries: {e}")
        return []


async def get_query_stats() -> Dict[str, Any]:
    """Get aggregate query statistics."""
    try:
        async with get_session() as session:
            # Total queries
            total_result = await session.execute(select(QueryLog))
            total_queries = len(list(result.scalars().all()))

            # Refusal rate
            refusal_result = await session.execute(
                select(QueryLog).where(QueryLog.is_refusal == True)
            )
            refusal_count = len(list(refusal_result.scalars().all()))

            # Average latency
            latency_result = await session.execute(select(QueryLog.latency_ms))
            latencies = [r[0] for r in latency_result.all()]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            return {
                "total_queries": total_queries,
                "refusal_count": refusal_count,
                "refusal_rate": refusal_count / total_queries if total_queries > 0 else 0,
                "avg_latency_ms": round(avg_latency, 2),
            }
    except Exception as e:
        print(f"Failed to get query stats: {e}")
        return {
            "total_queries": 0,
            "refusal_count": 0,
            "refusal_rate": 0,
            "avg_latency_ms": 0,
        }