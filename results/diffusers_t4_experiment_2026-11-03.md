# Diffusers Tiny-SD real Tesla T4 experiment — November 3, 2026

This experiment was **executed by the project owner in Google Colab** using the `scripts/diffusers_inference.py` script in PR #5. The owner provided a generated 384 × 384 robot-in-garden PNG and the script's JSON output. The image was inspected in the conversation but **is not committed to this GitHub PR**; the script regenerates the image from the model and seed.

## Observed settings and outcome

| Property | Observed |
| --- | --- |
| Model | `segmind/tiny-sd` |
| GPU / runtime | NVIDIA Tesla T4, CUDA |
| Precision | FP16 |
| Random seed | 42 |
| Inference steps | 15 |
| Guidance scale | 7.5 |
| Image resolution | 384 × 384 |
| Generation execution time | **5.474 seconds** |
| Peak PyTorch allocated VRAM | **1386.78 MiB** |
| Peak PyTorch reserved VRAM | **1664.0 MiB** |
| PyTorch | `2.11.0+cu130` |
| Transformers | `5.22.0` |
| Diffusers | `0.42.1` |

The owner successfully displayed an image of a small robot in a flower garden. This confirms **functional text-to-image diffusion generation** using a real GPU, distinct from the earlier Transformers LLM experiments.

Raw observed settings and metrics are preserved in [diffusers_t4_run_2026-11-03.json](diffusers_t4_run_2026-11-03.json).

## Scope and limitations

- **Single** observed text-to-image generation, not a repeated speed benchmark.
- Elapsed time excludes model download and initialization, but includes pipeline execution and CUDA synchronization.
- Reported memory is from PyTorch's CUDA allocator and is not total device/system VRAM usage.
- Tiny-SD is an educational small baseline, not a claim about state-of-the-art image generation quality.
- This test does **not** demonstrate diffusion batching, optimized serving, production deployment, or NCCL.
- The PNG is currently available in the user's Colab output and attached conversation; do not claim that it exists in GitHub unless separately committed.
