"""PostgreSQL models for FinRAG persistence layer."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Integer,
    Float,
    Boolean,
    Index,
    ForeignKey,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
import uuid

Base = declarative_base()


class QueryStatus(str, enum.Enum):
    SUCCESS = "success"
    REFUSAL = "refusal"
    ERROR = "error"


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(16), unique=True, nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    is_refusal = Column(Boolean, default=False)
    status = Column(SQLEnum(QueryStatus), default=QueryStatus.SUCCESS)
    latency_ms = Column(Float, nullable=False)
    model_provider = Column(String(32))
    model_name = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    retrieval_chunks = relationship("RetrievedChunk", back_populates="query_log", cascade="all, delete-orphan")
    claim_traces = relationship("ClaimTrace", back_populates="query_log", cascade="all, delete-orphan")


class RetrievedChunk(Base):
    __tablename__ = "retrieved_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_log_id = Column(UUID(as_uuid=True), ForeignKey("query_logs.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(String(128), nullable=False)
    rank = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    ticker = Column(String(16))
    form = Column(String(16))
    filing_date = Column(String(32))
    section = Column(String(128))
    is_table = Column(Boolean, default=False)
    dense_score = Column(Float)
    bm25_rank = Column(Integer)
    rrf_rank = Column(Integer)
    rerank_score = Column(Float)
    chunk_metadata = Column("metadata", JSON)

    query_log = relationship("QueryLog", back_populates="retrieval_chunks")


class ClaimTrace(Base):
    __tablename__ = "claim_traces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_log_id = Column(UUID(as_uuid=True), ForeignKey("query_logs.id", ondelete="CASCADE"), nullable=False, index=True)
    claim_text = Column(Text, nullable=False)
    citation_ticker = Column(String(16))
    citation_form = Column(String(16))
    citation_filing_date = Column(String(32))
    citation_section = Column(String(128))
    matched_chunk_id = Column(String(128))
    verification_status = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    matched_tokens = Column(JSON)
    missing_tokens = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    query_log = relationship("QueryLog", back_populates="claim_traces")


# Indexes for common query patterns
Index("ix_query_logs_ticker_date", QueryLog.created_at)
Index("ix_retrieved_chunks_ticker", RetrievedChunk.ticker)
Index("ix_claim_traces_status", ClaimTrace.verification_status)