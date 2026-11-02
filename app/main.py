"""FastAPI inference service with bounded, compatible-request dynamic batching."""
from __future__ import annotations

import os
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from threading import Lock
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.backends import InferenceBackend, build_backend
from app.batching import BatchScheduler, QueueFull, QueueTimeout, SchedulerClosed
from app.observability import ServiceMetrics

logger = logging.getLogger('inference_service')


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


def create_app(backend: InferenceBackend | None = None, *,
               batch_window_ms: float | None = None,
               max_queue_requests: int | None = None,
               timeout_seconds: float | None = None) -> FastAPI:
    state = {"backend": backend, "scheduler": None, "total_requests": 0,
             "total_failures": 0, "last_latency_ms": None,
             "batches_executed": 0, "last_batch_prompts": 0}
    metrics_lock = Lock()
    telemetry = ServiceMetrics()
    wait_ms = float(os.getenv("BATCH_WINDOW_MS", "20")) if batch_window_ms is None else batch_window_ms
    max_queue = int(os.getenv("MAX_QUEUE_REQUESTS", "64")) if max_queue_requests is None else max_queue_requests
    timeout = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "30")) if timeout_seconds is None else timeout_seconds
    if timeout <= 0:
        raise ValueError("timeout_seconds must be positive")

    def on_batch(size: int, _elapsed_ms: float) -> None:
        telemetry.observe_batch(size, _elapsed_ms / 1000)
        with metrics_lock:
            state["batches_executed"] += 1
            state["last_batch_prompts"] = size

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["backend"] is None:
            state["backend"] = build_backend()
        scheduler = BatchScheduler(state["backend"], batch_window_ms=wait_ms,
                                   max_queue_requests=max_queue, on_batch=on_batch,
                                   on_queue_wait=telemetry.observe_wait)
        state["scheduler"] = scheduler
        try:
            yield
        finally:
            scheduler.close()
            state["scheduler"] = None

    app = FastAPI(title="LLM Inference Serving Lab", version="0.3.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": type(state["backend"]).__name__}

    @app.get("/metrics")
    def metrics():
        with metrics_lock:
            stats = {key: state[key] for key in
                     ("total_requests", "total_failures", "last_latency_ms",
                      "batches_executed", "last_batch_prompts")}
        scheduler = state["scheduler"]
        return {**stats, **(scheduler.snapshot() if scheduler else {"queue_depth": 0})}

    @app.get("/metrics/prometheus")
    def prometheus_metrics():
        scheduler = state["scheduler"]
        depth = scheduler.snapshot()["queue_depth"] if scheduler else 0
        return Response(telemetry.render(depth), media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        if any(not prompt.strip() for prompt in req.prompts):
            raise HTTPException(422, "Prompts must be nonempty strings")
        started = perf_counter()
        with metrics_lock:
            state["total_requests"] += 1
        try:
            scheduler = state["scheduler"]
            if scheduler is None:
                raise SchedulerClosed()
            outputs = scheduler.submit(req.prompts, req.max_new_tokens, timeout_seconds=timeout)
        except (QueueFull, SchedulerClosed) as exc:
            with metrics_lock:
                state["total_failures"] += 1
            telemetry.requests.labels(status="503").inc()
            telemetry.request_seconds.observe(perf_counter() - started)
            raise HTTPException(503, str(exc)) from exc
        except QueueTimeout as exc:
            with metrics_lock:
                state["total_failures"] += 1
            telemetry.requests.labels(status="504").inc()
            telemetry.request_seconds.observe(perf_counter() - started)
            raise HTTPException(504, str(exc)) from exc
        except Exception as exc:
            logger.error(json.dumps({"event": "inference_failed", "error_type": type(exc).__name__, "batch_size": len(req.prompts)}))
            with metrics_lock:
                state["total_failures"] += 1
            telemetry.requests.labels(status="500").inc()
            telemetry.request_seconds.observe(perf_counter() - started)
            raise HTTPException(500, "Inference failure; check server logs") from exc
        elapsed_ms = (perf_counter() - started) * 1000
        telemetry.requests.labels(status="200").inc()
        telemetry.request_seconds.observe(elapsed_ms / 1000)
        with metrics_lock:
            state["last_latency_ms"] = round(elapsed_ms, 3)
        tokens = sum(o.output_tokens for o in outputs)
        return GenerateResponse(results=[GenerationResult(**asdict(o)) for o in outputs],
                                latency_ms=round(elapsed_ms, 3),
                                output_tokens_per_second=round(
                                    tokens / (elapsed_ms / 1000), 2) if elapsed_ms else 0,
                                batch_size=len(req.prompts))

    return app


app = create_app()
