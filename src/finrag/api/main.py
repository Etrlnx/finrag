"""FastAPI service for FinRAG - Financial Document Intelligence."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from finrag.pipeline import load_production_pipeline
from finrag.explainability.models import ExplainableResult


pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading production pipeline...")
    pipeline = load_production_pipeline()
    print("Pipeline loaded and ready.")
    yield
    print("Shutting down...")


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

        return QueryResponse(
            request_id=request_id,
            question=request.question,
            answer=answer,
            is_refusal=is_refusal,
            latency_ms=round(latency_ms, 2),
            explainable=explainable,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)