# Single-T4 NCCL smoke test — November 3, 2026

**Provenance:** Actual Colab console output provided by project owner. Run used PyTorch `2.11.0+cu130` on one NVIDIA Tesla T4.

- NCCL process group: rank 0, world size 1, local rank 0.
- Tensor: 1,024 float32 elements initialized to `rank + 1`.
- NCCL all-reduce SUM result: first element **1.0**; expected **1.0**.
- Correctness: **PASS**.
- Diagnostic elapsed time: **0.287 ms** (not a GPU-to-GPU bandwidth or latency result).
- **Important:** One rank / one GPU means no cross-GPU data transfer happened. This validates NCCL initialization and a local collective API smoke test only, **not true multi-GPU NCCL communication**.
- Initial runtime showed a warning that NCCL guessed the device mapping; following the observation, the source was updated to pass an explicit `device_id` to `init_process_group`. **That fix has not yet been revalidated on the Colab runtime.**

Source observed record: [nccl_single_t4_2026-11-03.json](nccl_single_t4_2026-11-03.json)

The separate two-process CPU Gloo correctness test and any real two-GPU NCCL results must be independently observed before claiming success.
