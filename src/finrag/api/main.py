"""FastAPI service for FinRAG - Financial Document Intelligence."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from finrag.pipeline import load_production_pipeline
from finrag.explainability.models import ExplainableResult
from finrag.persistence import log_query, init_db
from finrag.monitoring import (
    record_request,
    record_query,
    record_pipeline_load,
    metrics_endpoint,
)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for Loki."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {"name", "msg", "args", "created", "filename", "funcName",
                          "levelname", "levelno", "lineno", "module", "msecs",
                          "message", "msg", "name", "pathname", "process",
                          "processName", "relativeCreated", "thread", "threadName",
                          "exc_info", "exc_text", "stack_info", "asctime"}:
                log_data[key] = value
        return json.dumps(log_data)


# Configure JSON logging
logger = logging.getLogger("finrag-api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    load_start = time.perf_counter()
    logger.info("Loading production pipeline...")
    pipeline = load_production_pipeline()
    load_time = time.perf_counter() - load_start
    record_pipeline_load(load_time)
    logger.info("Initializing database...")
    try:
        await init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.error(f"Database initialization failed (continuing without persistence): {e}")
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="FinRAG API",
    description="Financial Document Intelligence & Evidence-Grounded Retrieval System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    record_request(request.method, request.url.path, response.status_code, latency)
    return response


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Financial question to answer")
    include_explainable: bool = Field(False, description="Include claim traces and evidence chunks")


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    is_refusal: bool
    latency_ms: float
    explainable: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    pipeline_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        pipeline_loaded=pipeline is not None,
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    logger.info("Query received", extra={"request_id": request_id, "question": request.question[:100]})

    try:
        if request.include_explainable:
            result: ExplainableResult = pipeline.query_explainable(request.question)
            answer = result.answer
            is_refusal = result.is_refusal
            explainable = result.to_dict()
        else:
            answer = pipeline.query(request.question)
            is_refusal = "Insufficient evidence to answer this question" in answer
            explainable = None

        latency_ms = (time.perf_counter() - start) * 1000
        latency_s = latency_ms / 1000.0

        # Record query metrics
        record_query("refusal" if is_refusal else "success", latency_s, is_refusal)

        # Log to database (fire and forget)
        import asyncio
        asyncio.create_task(log_query(
            request_id=request_id,
            question=request.question,
            answer=answer,
            is_refusal=is_refusal,
            latency_ms=latency_ms,
            model_provider="ollama",
            model_name="llama3.2",
            explainable=result if request.include_explainable else None,
        ))

        logger.info("Query completed",
                    extra={"request_id": request_id, "latency_ms": latency_ms,
                           "is_refusal": is_refusal, "status": "success"})

        return QueryResponse(
            request_id=request_id,
            question=request.question,
            answer=answer,
            is_refusal=is_refusal,
            latency_ms=round(latency_ms, 2),
            explainable=explainable,
        )
    except Exception as e:
        record_query("error", (time.perf_counter() - start) / 1000.0, False)
        logger.error("Query failed", extra={"request_id": request_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get query statistics."""
    from finrag.persistence import get_query_stats
    return await get_query_stats()


@app.get("/queries")
async def get_recent_queries(limit: int = 50):
    """Get recent query logs."""
    from finrag.persistence import get_recent_queries
    queries = await get_recent_queries(limit)
    return [
        {
            "request_id": q.request_id,
            "question": q.question,
            "is_refusal": q.is_refusal,
            "latency_ms": q.latency_ms,
            "created_at": q.created_at.isoformat(),
        }
        for q in queries
    ]


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return await metrics_endpoint()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)