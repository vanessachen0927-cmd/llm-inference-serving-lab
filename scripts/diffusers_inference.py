"""Run a real text-to-image diffusion model on a CUDA GPU, saving evidence.

This is a single-request model-family demonstration, not a throughput
benchmark or image-quality evaluation. Dependencies are imported lazily so
CPU-only API CI does not download models or require NVIDIA hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


DEFAULT_MODEL = "segmind/tiny-sd"
DEFAULT_PROMPT = "A small friendly robot watering flowers in a sunny garden, detailed digital illustration"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU Diffusers text-to-image inference")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/diffusers_sample.png")
    parser.add_argument("--metadata", default="results/diffusers_run.json")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.strip() or not args.prompt.strip():
        raise ValueError("model and prompt must be nonempty")
    if not (1 <= args.steps <= 100):
        raise ValueError("steps must be between 1 and 100")
    if not (128 <= args.width <= 768 and args.width % 8 == 0):
        raise ValueError("width must be 128..768 and divisible by 8")
    if not (128 <= args.height <= 768 and args.height % 8 == 0):
        raise ValueError("height must be 128..768 and divisible by 8")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if Path(args.output).resolve() == Path(args.metadata).resolve():
        raise ValueError("output and metadata paths must differ")


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    import torch
    from diffusers import DiffusionPipeline
    import diffusers
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU (select Colab T4 GPU).")
    device = "cuda"
    gpu = torch.cuda.get_device_name(0)
    # Load is outside the timed generation block.
    pipeline = DiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to(device)
    pipeline.set_progress_bar_config(disable=False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=7.5,
            height=args.height,
            width=args.width,
            generator=generator,
        )
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    if not result.images:
        raise RuntimeError("Diffusers returned no images")
    image = result.images[0]
    image_path = Path(args.output)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    metadata = {
        "model": args.model,
        "device": device,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "diffusers_version": diffusers.__version__,
        "dtype": "float16",
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": 7.5,
        "resolution": [args.width, args.height],
        "latency_seconds": round(duration, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 2),
        "image_path": str(image_path),
        "note": "One real image diffusion generation; model load excluded; not an LLM or throughput benchmark.",
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = build_parser().parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"Invalid argument: {exc}") from exc
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
