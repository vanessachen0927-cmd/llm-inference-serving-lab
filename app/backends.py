"""Backends for reproducible text-generation experiments.

Real transformer models are loaded lazily, so API unit tests do not need a GPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Generated:
    text: str
    input_tokens: int
    output_tokens: int


class InferenceBackend(Protocol):
    def generate(self, prompts: list[str], max_new_tokens: int) -> list[Generated]: ...


class FakeBackend:
    """For deterministic functional tests; NEVER use for GPU performance claims."""
    def generate(self, prompts: list[str], max_new_tokens: int) -> list[Generated]:
        return [Generated(text=f"Echo: {p}", input_tokens=len(p.split()), output_tokens=2)
                for p in prompts]


class TransformersBackend:
    def __init__(self, model_id: str = "distilgpt2", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Left padding is critical for decoder-only batched generation.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[Generated]:
        torch = self.torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                             max_length=256).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                         do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        # Output tensors for decoder-only models include the *padded* input prefix.
        input_width = enc["input_ids"].shape[1]
        results = []
        for i, tokens in enumerate(output):
            generated_ids = tokens[input_width:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            input_count = int(enc["attention_mask"][i].sum().item())
            output_count = int(generated_ids.shape[0])
            results.append(Generated(text=text, input_tokens=input_count,
                                     output_tokens=output_count))
        return results


def build_backend() -> InferenceBackend:
    kind = os.getenv("BACKEND", "fake")
    if kind == "fake":
        return FakeBackend()
    if kind == "transformers":
        return TransformersBackend(os.getenv("MODEL_ID", "distilgpt2"),
                                   os.getenv("MODEL_DEVICE") or None)
    raise ValueError(f"Unsupported BACKEND: {kind}")
