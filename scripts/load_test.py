"""Bounded concurrent HTTP load probe. Sends requests to a server YOU control.

Use with BACKEND=fake for API/concurrency correctness or BACKEND=transformers
for actual GPU-serving measurements. Never label fake backend results as AI speed.
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from urllib.parse import urlsplit

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    host = urlsplit(args.url).hostname
    if host not in ("localhost", "127.0.0.1", "::1"):
        parser.error("This educational load probe only targets a locally controlled service")
    if not (1 <= args.requests <= 1000 and 1 <= args.concurrency <= 32):
        parser.error("requests must be 1..1000 and concurrency 1..32")
    if not (1 <= args.max_new_tokens <= 128 and args.timeout > 0):
        parser.error("invalid generation settings")
    durations_ms = []
    codes: dict[str, int] = {}
    started = perf_counter()

    def send(index: int) -> tuple[str, float]:
        payload = {"prompts": [f"Explain inference request {index}"], "max_new_tokens": args.max_new_tokens}
        began = perf_counter()
        try:
            with httpx.Client(timeout=args.timeout) as client:
                response = client.post(args.url.rstrip("/") + "/generate", json=payload)
            return str(response.status_code), (perf_counter() - began) * 1000
        except httpx.RequestError:
            return "network_error", (perf_counter() - began) * 1000

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(send, i) for i in range(args.requests)]
        for future in as_completed(futures):
            code, duration = future.result()
            codes[code] = codes.get(code, 0) + 1
            durations_ms.append(duration)

    wall_time = perf_counter() - started
    sorted_ms = sorted(durations_ms)

    def percentile(p: float) -> float:
        if not sorted_ms:
            return 0.0
        offset = (len(sorted_ms) - 1) * p
        lo = int(offset)
        hi = min(lo + 1, len(sorted_ms) - 1)
        return sorted_ms[lo] + (sorted_ms[hi] - sorted_ms[lo]) * (offset - lo)

    print(json.dumps({
        "target": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "status_counts": codes,
        "error_rate": round(1 - codes.get("200", 0) / args.requests, 4),
        "p50_ms": round(statistics.median(durations_ms), 2),
        "p95_ms": round(percentile(0.95), 2),
        "wall_time_s": round(wall_time, 2),
        "http_requests_per_second": round(args.requests / wall_time, 2),
        "note": "HTTP load results; use real model backend for GPU inference claims."
    }, indent=2))


if __name__ == "__main__":
    main()
