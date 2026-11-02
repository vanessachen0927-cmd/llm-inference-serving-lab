from fastapi.testclient import TestClient
from app.main import create_app
from app.backends import FakeBackend


def test_health():
    with TestClient(create_app(FakeBackend())) as client:
        assert client.get("/health").json()["status"] == "ok"


def test_batch_and_metrics():
    with TestClient(create_app(FakeBackend())) as client:
        response = client.post("/generate", json={"prompts": ["hello", "world"], "max_new_tokens": 4})
        assert response.status_code == 200
        body = response.json()
        assert body["batch_size"] == 2
        assert [r["text"] for r in body["results"]] == ["Echo: hello", "Echo: world"]
        assert body["latency_ms"] >= 0
        metrics = client.get("/metrics").json()
        assert metrics["total_requests"] == 1
        assert metrics["total_failures"] == 0


def test_validation():
    with TestClient(create_app(FakeBackend())) as client:
        for request in [
            {"prompts": [], "max_new_tokens": 5},
            {"prompts": ["hi"], "max_new_tokens": 0},
            {"prompts": ["  "], "max_new_tokens": 5},
            {"prompts": ["hi"] * 9, "max_new_tokens": 5},
        ]:
            assert client.post("/generate", json=request).status_code == 422
