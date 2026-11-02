"""FastAPI service for controlled inference and latency studies.

One process, one model, and a lock protect non-reentrant inference workloads.
No production scaling, autoscaling, or SLA claim is implied.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from threading import Lock
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.backends import InferenceBackend, build_backend


class GenerateRequest(BaseModel):
    prompts: Annotated[list[str], Field(min_length=1, max_length=8)]
    max_new_tokens: Annotated[int, Field(ge=1, le=128)] = 32


class GenerationResult(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int


class GenerateResponse(BaseModel):
    results: list[GenerationResult]
    latency_ms: float
    output_tokens_per_second: float
    batch_size: int


def create_app(backend: InferenceBackend | None = None) -> FastAPI:
    state: dict = {"backend": backend, "total_requests": 0, "total_failures": 0,
                   "last_latency_ms": None}
    lock = Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["backend"] is None:
            state["backend"] = build_backend()
        yield

    app = FastAPI(title="LLM Inference Serving Lab", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": type(state["backend"]).__name__}

    @app.get("/metrics")
    def metrics():
        return {"total_requests": state["total_requests"],
                "total_failures": state["total_failures"],
                "last_latency_ms": state["last_latency_ms"]}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        if any(not p.strip() for p in req.prompts):
            raise HTTPException(422, "Prompts must be nonempty strings")
        with lock:
            start = perf_counter()
            state["total_requests"] += 1
            try:
                outputs = state["backend"].generate(req.prompts, req.max_new_tokens)
            except Exception as exc:
                state["total_failures"] += 1
                raise HTTPException(500, "Inference failure; check server logs") from exc
            latency_ms = (perf_counter() - start) * 1000
            state["last_latency_ms"] = round(latency_ms, 3)
            tokens = sum(o.output_tokens for o in outputs)
            return GenerateResponse(results=[GenerationResult(**asdict(o)) for o in outputs],
                                    latency_ms=round(latency_ms, 3),
                                    output_tokens_per_second=round(
                                        tokens / (latency_ms / 1000), 2) if latency_ms else 0,
                                    batch_size=len(req.prompts))
    return app


app = create_app()
