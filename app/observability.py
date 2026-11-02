"""Per-process Prometheus metrics for the experimental inference service.

No prompt text, request IDs, or model IDs are used as metric labels.
"""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REQUEST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)
INFERENCE_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)
QUEUE_BUCKETS = (0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)


class ServiceMetrics:
    def __init__(self) -> None:
        # Explicit registry avoids cross-app contamination during tests.
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "inference_requests", "Completed inference HTTP requests",
            ["status"], registry=self.registry,
        )
        self.request_seconds = Histogram(
            "inference_request_duration_seconds",
            "Request duration including queue wait, seconds",
            buckets=REQUEST_BUCKETS, registry=self.registry,
        )
        self.batch_seconds = Histogram(
            "inference_batch_duration_seconds",
            "Backend generation duration per dispatched batch, seconds",
            buckets=INFERENCE_BUCKETS, registry=self.registry,
        )
        self.queue_wait_seconds = Histogram(
            "inference_queue_wait_seconds",
            "Time each request spent waiting before GPU dispatch, seconds",
            buckets=QUEUE_BUCKETS, registry=self.registry,
        )
        self.batch_prompts = Histogram(
            "inference_batch_prompts", "Number of prompts sent to backend in a batch",
            buckets=(1, 2, 3, 4, 6, 8), registry=self.registry,
        )
        self.queue_depth = Gauge(
            "inference_queue_depth", "Pending inference requests (not in-flight)",
            registry=self.registry,
        )
        self.batches = Counter(
            "inference_batches", "Number of backend batch calls",
            registry=self.registry,
        )

    def observe_batch(self, prompts: int, elapsed_seconds: float,
                      waits_seconds: list[float]) -> None:
        self.batches.inc()
        self.batch_prompts.observe(prompts)
        self.batch_seconds.observe(elapsed_seconds)
        for wait in waits_seconds:
            self.queue_wait_seconds.observe(wait)

    def render(self, queue_depth: int) -> bytes:
        self.queue_depth.set(queue_depth)
        return generate_latest(self.registry)
