"""Smoke test the CPU/API container via HTTP; no GPU benchmark claims."""
import argparse
import json
import sys
from urllib.request import Request, urlopen

def read(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(req, timeout=10) as response:
        return response.status, json.load(response)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    args = parser.parse_args()
    root = args.url.rstrip("/")
    code, health = read(root + "/health")
    assert code == 200 and health["status"] == "ok" and health["backend"] == "FakeBackend"
    code, result = read(root + "/generate", {"prompts": ["Docker test"], "max_new_tokens": 4})
    assert code == 200 and result["results"][0]["text"] == "Echo: Docker test"
    code, metrics = read(root + "/metrics")
    assert code == 200 and metrics["total_requests"] == 1
    with urlopen(root + "/metrics/prometheus", timeout=10) as response:
        body = response.read().decode()
    assert 'inference_requests_total{status="200"} 1.0' in body
    print("Docker smoke test PASS: health, generation, JSON metrics, Prometheus metrics")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Docker smoke test FAIL:", exc, file=sys.stderr)
        sys.exit(1)
