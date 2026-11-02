# LLM Inference Serving & Performance Lab

Early-career AI infrastructure portfolio project: **Python, PyTorch, Hugging Face Transformers, FastAPI, controlled batching, request validation, instrumentation, and reproducible GPU inference benchmarks.**

> Status: starter implementation. API unit tests are CPU-only with a **fake backend**; real T4 GPU inference and performance observations are documented in results/. `distilgpt2` is a small baseline autoregressive language model, not a production serving engine. This project is educational, not an operational scalable service.

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

This downloads model weights from Hugging Face (internet needed); runtime availability and downloads are not guaranteed. The benchmark produces `results/inference_benchmark.csv` for batch sizes 1, 2 and 4; archive only *observed* results. A bounded dynamic request batching prototype is included on this development branch. KV-cache analysis, TTFT instrumentation, Diffusers and NCCL are **not yet implemented**.

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

For CPU-only validation: `python -m pytest -q`. The four new focused tests cover compatible concurrent requests, incompatible generation settings, queue full/timeout, and backend failures. **Real GPU dynamic-batching performance has not been measured yet**; initial fake-backend tests only verify scheduling and API behavior. This version does not implement continuous token-level batching, cross-process queue sharing, cancellation of in-flight GPU computation, streaming TTFT or durable production metrics.
