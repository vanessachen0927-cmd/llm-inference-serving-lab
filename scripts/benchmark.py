"""Run in-process model-inference experiments without HTTP/network confounds."""
from __future__ import annotations

import argparse
import csv
import statistics
import time

from app.backends import TransformersBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", default="results/inference_benchmark.csv")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_new_tokens < 1:
        parser.error("repeats and max-new-tokens must be positive")
    backend = TransformersBackend(args.model, args.device)
    prompts = ["A GPU is", "In a data center", "Machine learning is", "Inference latency means"]
    rows = []
    for batch_size in (1, 2, 4):
        batch = prompts[:batch_size]
        backend.generate(batch, args.max_new_tokens)  # warmup
        for run in range(args.repeats):
            # Synchronize around timed operations for accurate CUDA wall-clock.
            if backend.device.startswith("cuda"):
                backend.torch.cuda.synchronize()
            start = time.perf_counter()
            generated = backend.generate(batch, args.max_new_tokens)
            if backend.device.startswith("cuda"):
                backend.torch.cuda.synchronize()
            duration = time.perf_counter() - start
            tokens = sum(g.output_tokens for g in generated)
            row = {"model": args.model, "device": backend.device, "batch_size": batch_size,
                   "run": run, "max_new_tokens": args.max_new_tokens,
                   "latency_ms": round(duration * 1000, 3),
                   "output_tokens": tokens, "output_tokens_per_second": round(tokens / duration, 2)}
            print(row, flush=True)
            rows.append(row)
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for size in (1, 2, 4):
        samples = [r["latency_ms"] for r in rows if r["batch_size"] == size]
        throughput = [r["output_tokens_per_second"] for r in rows if r["batch_size"] == size]
        print(f"Batch {size}: median latency {statistics.median(samples):.2f} ms; "
              f"median total output tokens/s {statistics.median(throughput):.2f}")
    print("Saved", args.output)


if __name__ == "__main__":
    main()
