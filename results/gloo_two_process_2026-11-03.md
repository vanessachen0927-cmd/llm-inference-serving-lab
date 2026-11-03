# Two-process Gloo all-reduce test — November 3, 2026

**Provenance:** Google Colab output supplied by project owner using PyTorch `2.11.0+cu130` and the project’s `scripts/distributed_allreduce.py`.

Command:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -m scripts.distributed_allreduce --backend gloo --elements 1024
```

Both processes completed successfully:

| Rank | Backend | World size | Device | Expected each element | Observed first element | Correctness |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Gloo | 2 | CPU | 3.0 | 3.0 | PASS |
| 1 | Gloo | 2 | CPU | 3.0 | 3.0 | PASS |

Each rank initialized every element to `rank + 1` and ran `all_reduce(SUM)`. Both received the sum 3.0. This verifies two-process collective correctness using **Gloo on CPUs**, not NCCL multi-GPU communication.

Reported times (2.070 ms for rank 1 and 2.071 ms for rank 0) are **diagnostic only**, not meaningful networking bandwidth or GPU communication benchmarks. The `OMP_NUM_THREADS` output is a launcher advisory rather than a test failure.

[Raw observed console records](gloo_two_process_2026-11-03.json) are preserved separately.

## Related NCCL result

The [single-T4 NCCL smoke test](nccl_single_t4_2026-11-03.md) passed with world size 1. That result covers NCCL initialization/API use on CUDA but **does not** prove multi-GPU NCCL. True two-GPU NCCL communication remains untested because only one Tesla T4 was available.

A post-observation change passes an explicit `device_id` when constructing the NCCL process group; the revised one-rank NCCL path has not been rerun after that change.
