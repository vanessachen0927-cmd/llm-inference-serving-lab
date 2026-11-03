# LLM Inference Serving & Performance Lab

Early-career AI infrastructure portfolio project: **Python, PyTorch, Hugging Face Transformers, FastAPI, controlled batching, request validation, instrumentation, and reproducible GPU inference benchmarks.**

> Status: educational inference-serving prototype. API unit tests are CPU-only with a **fake backend**; real T4 GPU inference and performance observations are documented in results/. `distilgpt2` is a small baseline autoregressive language model, not a production serving engine. This project is educational, not an operational scalable service.

## 1. Mac: run fast CPU-only API tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt
python -m pytest -q
BACKEND=fake uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"prompts":["Hello GPU","Hello AI"],"max_new_tokens":8}'
curl -s http://127.0.0.1:8000/metrics
```

## 2. Google Colab T4: actual Transformer inference

```python
!pip -q install -r requirements-ml.txt
!MODEL_DEVICE=cuda BACKEND=transformers MODEL_ID=distilgpt2 python -m scripts.benchmark --model distilgpt2 --device cuda --repeats 5 --max-new-tokens 32
```

This downloads model weights from Hugging Face (internet needed); runtime availability and downloads are not guaranteed. The benchmark produces `results/inference_benchmark.csv` for batch sizes 1, 2 and 4; archive only *observed* results. Bounded dynamic request batching was merged in PR #2. KV-cache analysis and TTFT instrumentation are **not yet implemented**. Real Diffusers inference on Tesla T4 is documented in PR #5; the NCCL all-reduce experiment is being developed in PR #6.

## API

- `GET /health`: process readiness
- `POST /generate`: deterministic greedy text completion, batch sizes 1–8, capped output-token count
- `GET /metrics`: total requests, failures and last-generation elapsed time (process-local only; not Prometheus or multi-worker monitoring)

## Experimental caveats

- The HTTP endpoint uses a process-local bounded scheduler and a single model worker; compatible requests may batch, but this is **not production-ready multi-worker serving**.
- Reported tokens/sec is the sum of returned output token counts divided by measured wall time, not per-request tokens/sec. Endpoint timing excludes network serialization; benchmark measures in-process generation.
- `generate()` may return EOS before the requested maximum; output token counts include generated token IDs that may decode to an empty EOS token.
- Results depend on model, model cache, device, GPU contention, PyTorch/CUDA versions and warmup. Preserve exact versions and raw runs.

## Development milestones

1. **Starter (this version):** tested API, real Transformers backend, explicit batch-size tests and in-process benchmark.
2. Add real GPU benchmark observations, CPU-vs-GPU and FP32-vs-FP16 measurement, memory usage, failure/error logging.
3. Add time-to-first-token via streaming generation and a queue-based batching scheduler with controlled backpressure.
4. Add Prometheus histograms, a load generator, Dockerfile, CI integration and structured logs.
5. Add an **NCCL** all-reduce demo on a *real multi-GPU* runtime if one becomes available; otherwise document a single-GPU limitation. `torch.distributed` Gloo on one process is **not a substitute for NCCL performance proof**.
6. Add Diffusers image generation baseline only if it materially supports an available role requirement and free GPU memory/time permit.

## Bounded dynamic batching (experimental PR)

The `app/batching.py` scheduler queues concurrent `POST /generate` requests, combines compatible requests (matching `max_new_tokens`) for one model invocation, and maps outputs back to the original callers. It has one model worker, queue capacity and request timeouts. At most 8 prompts are sent in each model call. Each request can itself contain multiple prompts.

Configuration (per server process): `BATCH_WINDOW_MS=20`, `MAX_QUEUE_REQUESTS=64`, `INFERENCE_TIMEOUT_SECONDS=30`. HTTP `503` means queue full/closed and `504` means a request exceeded the wait limit; an already-running GPU kernel cannot be preempted. The `GET /metrics` endpoint additionally reports `batches_executed`, `last_batch_prompts`, `queue_depth` and `max_queue_requests`.

For CPU-only validation: `python -m pytest -q`. The four new focused tests cover compatible concurrent requests, incompatible generation settings, queue full/timeout, and backend failures. **Real GPU dynamic batching correctness was verified in Colab** (two HTTP 200 results combined into one actual Tesla T4 Transformer invocation); its multi-client throughput has not been benchmarked. This version does not implement continuous token-level batching, cross-process queue sharing, cancellation of in-flight GPU computation, streaming TTFT or durable production metrics.

## Observability and HTTP load testing (PR #3)

The development branch adds a **per-process** Prometheus text endpoint at `GET /metrics/prometheus` while retaining the original JSON `GET /metrics` endpoint. It reports request outcomes by HTTP status (200/500/503/504), HTTP request duration, actual backend batch execution duration, queue wait time, total backend batches, number of prompts per backend batch, and pending queue depth. Histograms use seconds and can be queried with Prometheus `histogram_quantile` for approximate p50/p95. These are per-process, volatile metrics—not a persistent production monitoring deployment.

The service logs backend errors as structured JSON containing only event, exception type and prompt count, **not prompt contents**. All measurements are CPU fake-backend tested; a real GPU operational test is still recommended. `prometheus-client` is included in API dependencies.

Run a local service with `BACKEND=fake python -m uvicorn app.main:app --port 8001`, then in another terminal:

```bash
curl -s http://127.0.0.1:8001/metrics/prometheus
python -m scripts.load_test --url http://127.0.0.1:8001 --requests 30 --concurrency 4
python -m pytest -q
```

The load probe only targets loopback hosts, caps request and concurrency counts, and prints completed-request p50/p95 **HTTP** latency, status codes, error rate and requests/sec. **Do not represent fake-backend load results as GPU inference performance.** When testing a real GPU, launch the same service with `BACKEND=transformers MODEL_DEVICE=cuda MODEL_ID=distilgpt2` on a CUDA machine and save the actual console output.

## Docker CPU API deployment (PR #4)

This container is intentionally **API-only with BACKEND=fake**, so it builds and runs without CUDA on Apple Silicon and GitHub Actions. It tests deployment, health checks, the non-root process and HTTP monitoring. It is **not GPU inference** and its performance is not evidence of GPU throughput. The image installs only `requirements-api.txt`, not PyTorch or model weights.

```bash
docker build -t llm-inference-serving-lab:local .
docker run -d --name llm-api -p 127.0.0.1:18000:8000 llm-inference-serving-lab:local
docker inspect -f '{{.State.Health.Status}}' llm-api
python scripts/docker_smoke_test.py --url http://127.0.0.1:18000
docker logs llm-api
docker rm -f llm-api
```

The container exposes `/health`, `/generate`, `/metrics` and `/metrics/prometheus`. The Dockerfile uses Python 3.11 slim, a non-root user, a built-in health check and a single Uvicorn worker so that the in-memory request queue and metrics remain process-local. The Docker CI workflow builds an image, waits until healthy and invokes the HTTP smoke test on an ephemeral container. Port 18000 avoids clashing with local development on 8000/8001. A separate CUDA-capable image/runtime and proper production security measures would be required to serve a real model from Docker.

## Diffusers GPU experiment (PR #5, Tesla T4 validated)

The dedicated script `scripts/diffusers_inference.py` uses the Hugging Face Diffusers `DiffusionPipeline` on **CUDA with float16**, an explicit random seed, and a lightweight text-to-image model (`segmind/tiny-sd`). It saves both a PNG and a JSON record of generation latency, peak PyTorch CUDA allocation, model name, seed, steps and library versions. **Real Tesla T4 execution is validated**; see `results/diffusers_t4_experiment_2026-11-03.md` for measured results and limitations.

In free Colab with **T4 GPU** (cloning the PR branch), use:

```python
!pip -q install -r requirements-diffusers.txt
!python -m scripts.diffusers_inference --steps 15 --width 384 --height 384
from IPython.display import display, Image
display(Image(filename="results/diffusers_sample.png"))
```

Use `python -m pytest -q` on CPU to test argument validation without downloading model weights. The image quality of a tiny baseline model does not establish production image quality. Model download/connectivity, dependency compatibility and T4 availability are runtime-dependent. The default output paths under `results/` are not automatically committed: save evidence only after successful execution. This is an image diffusion experiment; it does not implement optimized diffusers serving, dynamic image batching, or NCCL.

## Distributed collectives — Gloo and NCCL (PR #6, validated limited modes)

`scripts/distributed_allreduce.py` demonstrates distributed process-group setup and `all_reduce(SUM)` result validation. Each rank initializes its tensor with `rank + 1`, and each resulting tensor element must equal `world_size * (world_size + 1) / 2`. It supports these **distinct** validation modes:

1. **Two-process Gloo on CPU:** proves correctness across two independent CPU processes (not NCCL). In a terminal with PyTorch installed, or Google Colab: `python -m torch.distributed.run --standalone --nproc_per_node=2 -m scripts.distributed_allreduce --backend gloo --elements 1024`.
2. **One-process NCCL on one Tesla T4:** smoke-tests CUDA/NCCL initialization and the all-reduce API, but **cannot test inter-GPU communication**: `python -m torch.distributed.run --standalone --nproc_per_node=1 -m scripts.distributed_allreduce --backend nccl --elements 1024`.
3. **Real two-GPU NCCL (only if two devices are allocated):** `python -m torch.distributed.run --standalone --nproc_per_node=2 -m scripts.distributed_allreduce --backend nccl --require-multi-gpu --elements 1024`. This cannot be run with a single T4 and should never be claimed without multi-GPU evidence.

**Observed on November 3, 2026:** Both ranks passed the Gloo two-process correctness test on CPU (each returned 3.0), and one-rank NCCL successfully initialized and passed a CUDA all-reduce on one Tesla T4 (result 1.0). The single-GPU NCCL test was performed before the explicit-device mapping improvement, which has not yet been independently rechecked. See `results/gloo_two_process_2026-11-03.md` and `results/nccl_single_t4_2026-11-03.md` for outputs and limits.

The command will print a JSON result with backend, world size, rank, correctness and a diagnostic elapsed time. Output is intended for learning and correctness, **not** GPU-to-GPU bandwidth benchmarking. On Colab install `requirements-ml.txt` first; the script imports PyTorch only during execution, so basic argument/unit tests remain CPU-only without adding torch to the CI test dependencies. This is a single-node example using `torchrun`, not cluster-scale orchestration or production distributed inference.
