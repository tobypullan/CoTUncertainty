#!/usr/bin/env python3
"""Basic sampled inference throughput benchmark."""

from __future__ import annotations

import argparse
import inspect
import math
import os
import threading
import time
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _patch_filelock_for_hf() -> None:
    """Compat patch for old filelock + newer huggingface_hub usage."""
    try:
        import filelock
    except Exception:
        return

    base_cls = getattr(filelock, "BaseFileLock", None)
    if base_cls is None or getattr(base_cls, "_cot_patched", False):
        return

    orig_init = base_cls.__init__
    sig = inspect.signature(orig_init)
    accepts_mode = "mode" in sig.parameters

    def _init(self, lock_file, timeout=-1, *args, **kwargs):
        if not accepts_mode and "mode" in kwargs:
            kwargs.pop("mode", None)
        orig_init(self, lock_file, timeout=timeout, *args, **kwargs)
        if not hasattr(self, "_thread_lock"):
            self._thread_lock = threading.Lock()

    base_cls.__init__ = _init  # type: ignore[assignment]
    base_cls._cot_patched = True  # type: ignore[attr-defined]


_patch_filelock_for_hf()
os.environ.setdefault("HF_HUB_DISABLE_FILELOCK", "1")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def maybe_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def benchmark_mode(
    mode_name: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized_prompt: Dict[str, torch.Tensor],
    total_samples: int,
    samples_per_call: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    warmup_calls: int,
    seed: int,
    force_full_length: bool,
) -> Dict[str, float]:
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if force_full_length:
        gen_kwargs["min_new_tokens"] = max_new_tokens

    for warmup_idx in range(warmup_calls):
        set_seed(seed + warmup_idx)
        with torch.inference_mode():
            _ = model.generate(
                **tokenized_prompt,
                num_return_sequences=samples_per_call,
                **gen_kwargs,
            )
    maybe_sync(model.device.type)

    total_calls = math.ceil(total_samples / samples_per_call)
    emitted_samples = 0
    start = time.perf_counter()
    for call_idx in range(total_calls):
        current_batch = min(samples_per_call, total_samples - emitted_samples)
        set_seed(seed + 10_000 + call_idx)
        with torch.inference_mode():
            _ = model.generate(
                **tokenized_prompt,
                num_return_sequences=current_batch,
                **gen_kwargs,
            )
        emitted_samples += current_batch
    maybe_sync(model.device.type)
    elapsed = time.perf_counter() - start

    generated_tokens = emitted_samples * max_new_tokens
    return {
        "mode": mode_name,
        "samples_per_call": samples_per_call,
        "total_samples": emitted_samples,
        "generated_tokens": generated_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": generated_tokens / elapsed if elapsed > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic sampled inference benchmark")
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help="Hugging Face model id or local path",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Solve briefly: If 3x + 5 = 20, what is x? "
            "Show one short line, then Final Answer."
        ),
        help="Single prompt used for timing",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Generated tokens per sample",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=20,
        help="Total sampled outputs to time in each mode",
    )
    parser.add_argument(
        "--batched-samples-per-call",
        type=int,
        default=5,
        help="`num_return_sequences` used in batched mode",
    )
    parser.add_argument(
        "--warmup-calls",
        type=int,
        default=1,
        help="Warmup generate calls before timing",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for inference, e.g. cuda or cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code from Hub repos",
    )
    parser.add_argument(
        "--no-force-full-length",
        action="store_true",
        help="Allow early EOS (by default, min_new_tokens=max_new_tokens)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_samples < 1:
        raise ValueError("--total-samples must be >= 1.")
    if args.batched_samples_per_call < 1:
        raise ValueError("--batched-samples-per-call must be >= 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1.")
    if args.warmup_calls < 0:
        raise ValueError("--warmup-calls must be >= 0.")

    force_full_length = not args.no_force_full_length
    dtype = resolve_dtype(args.dtype)

    print(f"loading model={args.model} device={args.device} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    tokenized_prompt = tokenizer(args.prompt, return_tensors="pt")
    tokenized_prompt = {k: v.to(model.device) for k, v in tokenized_prompt.items()}

    serial_stats = benchmark_mode(
        mode_name="serial",
        model=model,
        tokenizer=tokenizer,
        tokenized_prompt=tokenized_prompt,
        total_samples=args.total_samples,
        samples_per_call=1,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        warmup_calls=args.warmup_calls,
        seed=args.seed,
        force_full_length=force_full_length,
    )
    batched_stats = benchmark_mode(
        mode_name="batched",
        model=model,
        tokenizer=tokenizer,
        tokenized_prompt=tokenized_prompt,
        total_samples=args.total_samples,
        samples_per_call=args.batched_samples_per_call,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        warmup_calls=args.warmup_calls,
        seed=args.seed + 1000,
        force_full_length=force_full_length,
    )

    speedup = (
        batched_stats["tokens_per_sec"] / serial_stats["tokens_per_sec"]
        if serial_stats["tokens_per_sec"] > 0
        else float("inf")
    )

    print(
        "serial "
        f"tokens={serial_stats['generated_tokens']} "
        f"elapsed={serial_stats['elapsed_sec']:.2f}s "
        f"tokens_per_sec={serial_stats['tokens_per_sec']:.2f}"
    )
    print(
        "batched "
        f"samples_per_call={int(batched_stats['samples_per_call'])} "
        f"tokens={batched_stats['generated_tokens']} "
        f"elapsed={batched_stats['elapsed_sec']:.2f}s "
        f"tokens_per_sec={batched_stats['tokens_per_sec']:.2f}"
    )
    print(f"speedup_vs_serial={speedup:.2f}x")


if __name__ == "__main__":
    main()
