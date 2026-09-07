"""Monitoring and metrics for FinRAG."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Request metrics
REQUEST_COUNT = Counter(
    "finrag_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "finrag_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)

# Query metrics
QUERY_COUNT = Counter(
    "finrag_queries_total",
    "Total number of queries processed",
    ["status"],  # success, refusal, error
)

QUERY_LATENCY = Histogram(
    "finrag_query_latency_seconds",
    "Query processing latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

QUERY_REFUSAL_RATE = Gauge(
    "finrag_query_refusal_rate",
    "Current refusal rate (0-1)",
)

# Retrieval metrics
RETRIEVAL_COUNT = Counter(
    "finrag_retrieval_total",
    "Total number of retrieval operations",
)

RETRIEVAL_LATENCY = Histogram(
    "finrag_retrieval_latency_seconds",
    "Retrieval latency in seconds",
)

RETRIEVAL_DOCS = Histogram(
    "finrag_retrieval_docs",
    "Number of documents retrieved",
    buckets=[1, 2, 3, 5, 10, 20, 50],
)

# Pipeline metrics
PIPELINE_LOAD_TIME = Histogram(
    "finrag_pipeline_load_seconds",
    "Pipeline load time in seconds",
)

ACTIVE_QUERIES = Gauge(
    "finrag_active_queries",
    "Number of currently active queries",
)

# LLM metrics
LLM_CALL_COUNT = Counter(
    "finrag_llm_calls_total",
    "Total number of LLM calls",
    ["provider", "model"],
)

LLM_LATENCY = Histogram(
    "finrag_llm_latency_seconds",
    "LLM call latency in seconds",
    ["provider", "model"],
)

LLM_TOKENS = Histogram(
    "finrag_llm_tokens_total",
    "LLM tokens used",
    ["provider", "model", "type"],  # prompt, completion, total
)


def record_request(method: str, endpoint: str, status: int, latency: float) -> None:
    """Record request metrics."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)


def record_query(status: str, latency: float, is_refusal: bool) -> None:
    """Record query metrics."""
    QUERY_COUNT.labels(status=status).inc()
    QUERY_LATENCY.observe(latency)
    if is_refusal:
        # Update refusal rate (simplified - would need a sliding window for accuracy)
        pass


def record_retrieval(latency: float, num_docs: int) -> None:
    """Record retrieval metrics."""
    RETRIEVAL_COUNT.inc()
    RETRIEVAL_LATENCY.observe(latency)
    RETRIEVAL_DOCS.observe(num_docs)


def record_pipeline_load(latency: float) -> None:
    """Record pipeline load time."""
    PIPELINE_LOAD_TIME.observe(latency)


def record_llm_call(provider: str, model: str, latency: float, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """Record LLM call metrics."""
    LLM_CALL_COUNT.labels(provider=provider, model=model).inc()
    LLM_LATENCY.labels(provider=provider, model=model).observe(latency)
    if prompt_tokens:
        LLM_TOKENS.labels(provider=provider, model=model, type="prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(provider=provider, model=model, type="completion").inc(completion_tokens)
    if prompt_tokens or completion_tokens:
        total = prompt_tokens + completion_tokens
        LLM_TOKENS.labels(provider=provider, model=model, type="total").inc(total)


def set_active_queries(count: int) -> None:
    """Set active queries gauge."""
    ACTIVE_QUERIES.set(count)


def set_refusal_rate(rate: float) -> None:
    """Set refusal rate gauge."""
    QUERY_REFUSAL_RATE.set(rate)


async def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)