"""Single-node distributed all-reduce correctness example for Gloo and NCCL.

Launch with torchrun. Two CPU workers prove cross-rank all-reduce semantics;
one Tesla T4 worker can smoke-test NCCL API availability, but cannot prove
multi-GPU communication or measure inter-GPU bandwidth.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time


def expected_sum(world_size: int) -> float:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    return float(world_size * (world_size + 1) // 2)


def choose_backend(requested: str, cuda_available: bool) -> str:
    if requested not in ("auto", "gloo", "nccl"):
        raise ValueError("backend must be auto, gloo or nccl")
    return "nccl" if requested == "auto" and cuda_available else (
        "gloo" if requested == "auto" else requested)


def validate_configuration(backend: str, world_size: int, local_rank: int,
                           gpu_count: int, cuda_available: bool) -> None:
    if world_size < 1 or local_rank < 0:
        raise ValueError("invalid rank configuration")
    if backend not in ("gloo", "nccl"):
        raise ValueError("unsupported backend")
    if backend == "nccl" and (not cuda_available or local_rank >= gpu_count):
        raise RuntimeError("NCCL requires a distinct visible CUDA GPU for each local rank")


def run(args: argparse.Namespace) -> dict:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = choose_backend(args.backend, torch.cuda.is_available())
    gpu_count = torch.cuda.device_count()
    validate_configuration(backend, world_size, local_rank, gpu_count,
                           torch.cuda.is_available())
    if args.require_multi_gpu and (backend != "nccl" or world_size < 2):
        raise RuntimeError("--require-multi-gpu needs NCCL and at least 2 ranks")
    if args.require_multi_gpu and gpu_count < world_size:
        raise RuntimeError("Each rank needs its own GPU; 1 T4 is not multi-GPU")
    if world_size > 1 and "RANK" not in os.environ:
        raise RuntimeError("Launch multiple ranks with torchrun")
    if backend == "nccl":
        if not dist.is_nccl_available():
            raise RuntimeError("This PyTorch build has no NCCL support")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if not dist.is_gloo_available():
            raise RuntimeError("This PyTorch build has no Gloo support")
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://",
                            timeout=timedelta(seconds=120))
    try:
        tensor = torch.full((args.elements,), float(rank + 1),
                            dtype=torch.float32, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier()
        started = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        expected = expected_sum(world_size)
        passed = bool(torch.allclose(tensor, torch.full_like(tensor, expected),
                                     atol=1e-5, rtol=0))
        if not passed:
            raise AssertionError(f"rank {rank}: wrong all-reduce result")
        result = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "backend": backend,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(local_rank) if backend == "nccl" else None,
            "elements": args.elements,
            "expected_each_element": expected,
            "observed_first_element": float(tensor[0].item()),
            "correctness": "PASS",
            "single_gpu_limitation": backend == "nccl" and world_size == 1,
            "elapsed_ms_diagnostic_only": round(elapsed_ms, 3),
            "torch_version": torch.__version__,
            "note": ("One-rank NCCL smoke test: no inter-GPU communication"
                     if backend == "nccl" and world_size == 1 else
                     "Distributed all-reduce correctness only; not a bandwidth benchmark"),
        }
        print(json.dumps(result), flush=True)
        if rank == 0 and args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        dist.barrier()
        return result
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--require-multi-gpu", action="store_true")
    parser.add_argument("--output", default="results/distributed_allreduce_run.json")
    args = parser.parse_args()
    if not 1 <= args.elements <= 1_000_000:
        parser.error("elements must be between 1 and 1000000")
    run(args)


if __name__ == "__main__":
    main()
