"""Observability tests using only the CPU fake backend."""
import json
import logging
from fastapi.testclient import TestClient
from app.backends import FakeBackend
from app.main import create_app


def test_prometheus_metrics_are_exposed_and_updated():
    with TestClient(create_app(FakeBackend(), batch_window_ms=0)) as client:
        assert client.post("/generate", json={"prompts": ["hello"], "max_new_tokens": 8}).status_code == 200
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        data = response.text
        assert 'inference_requests_total{status="200"} 1.0' in data
        assert 'inference_request_duration_seconds_count 1.0' in data
        assert "inference_batch_duration_seconds_count 1.0" in data
        assert "inference_queue_wait_seconds_count 1.0" in data
        assert "inference_queue_depth 0.0" in data
        assert "inference_batches_total 1.0" in data


def test_failed_inference_is_counted_without_logging_prompts(caplog):
    class BrokenBackend(FakeBackend):
        def generate(self, prompts, max_new_tokens):
            raise RuntimeError("fake backend failure")

    with TestClient(create_app(BrokenBackend(), batch_window_ms=0)) as client:
        with caplog.at_level(logging.ERROR, logger="inference_service"):
            response = client.post("/generate", json={"prompts": ["PRIVATE_PROMPT_MARKER"]})
        assert response.status_code == 500
        metrics = client.get("/metrics/prometheus").text
        assert 'inference_requests_total{status="500"} 1.0' in metrics
        assert "PRIVATE_PROMPT_MARKER" not in caplog.text
        records = [json.loads(record.message) for record in caplog.records
                   if record.name == "inference_service"]
        assert any(record["event"] == "inference_failed" for record in records)


def test_metric_registry_is_isolated_for_each_app():
    with TestClient(create_app(FakeBackend(), batch_window_ms=0)) as first:
        assert first.post("/generate", json={"prompts": ["one"]}).status_code == 200
    with TestClient(create_app(FakeBackend(), batch_window_ms=0)) as second:
        report = second.get("/metrics/prometheus").text
        assert 'inference_requests_total{status="200"}' not in report
