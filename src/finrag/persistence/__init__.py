"""FinRAG Persistence Layer - PostgreSQL integration for query logging."""

from __future__ import annotations

from finrag.persistence.database import (
    init_db,
    close_db,
    get_session,
    create_tables,
)
from finrag.persistence.models import QueryLog, RetrievedChunk, ClaimTrace, QueryStatus
from finrag.persistence.query_logger import (
    log_query,
    get_query_log,
    get_recent_queries,
    get_query_stats,
)

__all__ = [
    "init_db",
    "close_db",
    "get_session",
    "create_tables",
    "QueryLog",
    "RetrievedChunk",
    "ClaimTrace",
    "QueryStatus",
    "log_query",
    "get_query_log",
    "get_recent_queries",
    "get_query_stats",
]