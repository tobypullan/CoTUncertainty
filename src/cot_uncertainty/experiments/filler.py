#!/usr/bin/env python3
"""
CoT filler experiment for the first N math questions.

Workflow:
1) Generate a baseline Chain-of-Thought (CoT) for the first N questions.
2) Split the CoT into sentences and keep exactly the first sentence.
3) Re-prompt the model with the question + that 1-sentence CoT prefix,
   then pad the provided reasoning with filler tokens to match baseline CoT length.
4) Run answer generation and measure correctness.
5) Plot percent-correct for the 1-sentence setting.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Prefer the user's standard HF cache, but fall back to a local cache if writes fail.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

EXPERIMENT_SLUG = "filler"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", EXPERIMENT_SLUG)
PROJECT_HF_HOME = os.path.join(PROJECT_ROOT, ".hf_cache")
DEFAULT_USER_HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def _pick_writable_dir(preferred: str, fallback: str) -> str:
    for candidate in (preferred, fallback):
        try:
            os.makedirs(candidate, exist_ok=True)
            probe_path = os.path.join(candidate, f".cot_write_probe_{os.getpid()}")
            with open(probe_path, "w", encoding="utf-8"):
                pass
            os.remove(probe_path)
            return candidate
        except OSError:
            continue
    raise OSError(
        f"Could not create a writable cache directory: {preferred} or {fallback}"
    )


hf_home = _pick_writable_dir(
    os.environ.get("HF_HOME", DEFAULT_USER_HF_HOME),
    PROJECT_HF_HOME,
)
os.environ["HF_HOME"] = hf_home
os.environ["HF_DATASETS_CACHE"] = _pick_writable_dir(
    os.environ.get("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets")),
    os.path.join(PROJECT_HF_HOME, "datasets"),
)
os.environ["HF_HUB_CACHE"] = _pick_writable_dir(
    os.environ.get("HF_HUB_CACHE", os.path.join(hf_home, "hub")),
    os.path.join(PROJECT_HF_HOME, "hub"),
)
os.environ["TRANSFORMERS_CACHE"] = _pick_writable_dir(
    os.environ.get("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers")),
    os.path.join(PROJECT_HF_HOME, "transformers"),
)
os.environ.setdefault("HF_HUB_DISABLE_FILELOCK", "1")


def _patch_pillow_resampling() -> None:
    """
    Backward-compat patch for Pillow<9.1 where Image.Resampling doesn't exist.
    Newer transformers versions expect this attribute during Gemma3 imports.
    """
    try:
        from PIL import Image as PILImage
    except Exception:
        return

    if hasattr(PILImage, "Resampling"):
        return

    PILImage.Resampling = SimpleNamespace(  # type: ignore[attr-defined]
        NEAREST=getattr(PILImage, "NEAREST", 0),
        BOX=getattr(PILImage, "BOX", getattr(PILImage, "NEAREST", 0)),
        BILINEAR=getattr(PILImage, "BILINEAR", 2),
        HAMMING=getattr(PILImage, "HAMMING", getattr(PILImage, "BILINEAR", 2)),
        BICUBIC=getattr(PILImage, "BICUBIC", 3),
        LANCZOS=getattr(PILImage, "LANCZOS", getattr(PILImage, "BICUBIC", 3)),
    )


_patch_pillow_resampling()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cot_uncertainty.experiment_dataset import (
    DEFAULT_COT_ANSWER_INSTRUCTION,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_DIRECT_ANSWER_INSTRUCTION,
    answers_match,
    extract_question_text,
    get_gold_answer,
    load_experiment_dataset,
    parse_model_answer,
)


def _patch_filelock_for_hf() -> None:
    """Compat patch for old filelock + newer huggingface_hub usage."""
    try:
        import inspect
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


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(
    tokenizer: AutoTokenizer,
    user_text: str,
    system_text: Optional[str] = None,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (ImportError, RuntimeError, ValueError):
            # Fallback for environments missing chat-template deps (e.g. old jinja2).
            pass
    if system_text:
        return f"{system_text}\n\n{user_text}"
    return user_text


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: Optional[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: Optional[int] = None,
    allowed_token_ids: Optional[List[int]] = None,
) -> str:
    if seed is not None:
        set_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if max_new_tokens is None:
        limit = getattr(model.config, "max_position_embeddings", None)
        if not isinstance(limit, int) or limit <= 0:
            limit = getattr(tokenizer, "model_max_length", None)
        if not isinstance(limit, int) or limit <= 0 or limit > 100000:
            limit = 4096
        max_new_tokens = max(1, limit - inputs["input_ids"].shape[1])

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    if allowed_token_ids:
        def _allowed(_batch_id: int, _input_ids: torch.Tensor) -> List[int]:
            return allowed_token_ids
        gen_kwargs["prefix_allowed_tokens_fn"] = _allowed

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def split_sentences(text: str) -> List[str]:
    # Basic sentence splitter; avoids extra dependencies.
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def extract_cot(text: str) -> str:
    # Extract everything before the final answer marker if present.
    patterns = [
        r"\n\s*Final Answer\s*:\s*.*$",
        r"\n\s*Answer\s*:\s*.*$",
        r"\n\s*Final\s*:\s*.*$",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def token_len(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def select_single_token_filler_id(
    tokenizer: AutoTokenizer,
    filler_char: str,
) -> int:
    if len(filler_char) != 1:
        raise ValueError("--filler-token must be exactly one character.")

    for token_id in range(len(tokenizer)):
        try:
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            continue

        if not decoded or filler_char not in decoded:
            continue
        if all(ch.isspace() or ch == filler_char for ch in decoded):
            # Ensure this token remains one token when repeated after decode.
            repeatable = True
            for probe_n in (1, 2, 4, 8, 16, 32):
                probe_text = tokenizer.decode(
                    [token_id] * probe_n,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if token_len(tokenizer, probe_text) != probe_n:
                    repeatable = False
                    break
            if repeatable:
                return token_id

    raise RuntimeError(
        f"Could not find a repeatable single token that decodes to only "
        f"whitespace and '{filler_char}'. Strict token-length matching is not "
        f"possible with this tokenizer/filler combination."
    )


def resolve_results_path(results_dir: str, requested_path: str) -> str:
    if not requested_path:
        return requested_path
    if os.path.isabs(requested_path):
        abs_requested = os.path.abspath(requested_path)
    else:
        abs_requested = os.path.abspath(os.path.join(PROJECT_ROOT, requested_path))
    abs_results = os.path.abspath(results_dir)
    try:
        common = os.path.commonpath([abs_results, abs_requested])
    except ValueError:
        common = ""
    if common == abs_results:
        return abs_requested
    return os.path.join(results_dir, os.path.basename(requested_path))


def display_path(path: str) -> str:
    if not path:
        return path
    try:
        rel = os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path
    if rel.startswith(".."):
        return path
    return rel


def format_problem(question: str) -> str:
    return question.strip()


@dataclass
class ExperimentResult:
    sentence_count: int
    correct: int
    total: int
    percent_correct: float


# -----------------------------
# Main experiment
# -----------------------------

def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    datasets_cache_dir = os.environ.get("HF_DATASETS_CACHE")
    hub_cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")

    dataset = load_experiment_dataset(
        args.dataset,
        args.dataset_config,
        args.split,
        cache_dir=datasets_cache_dir,
    )
    num_questions = min(args.num_questions, len(dataset))
    if num_questions <= 0:
        raise ValueError("--num-questions must be >= 1.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=hub_cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        cache_dir=hub_cache_dir,
        trust_remote_code=args.trust_remote_code,
    ).to("cuda")
    model.eval()
    filler_token_id = select_single_token_filler_id(tokenizer, args.filler_token)

    system_prompt = (
        "You are a concise math solver."
        if args.system_prompt is None
        else args.system_prompt
    )

    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    plot_path = resolve_results_path(results_dir, args.plot_path)
    output_path = resolve_results_path(results_dir, args.output_path)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    baseline_path = os.path.join(results_dir, "baseline_cot.jsonl")
    resampled_path = os.path.join(results_dir, "resampled_cot.jsonl")

    aggregated_counts: Dict[int, Dict[str, int]] = {}
    question_results: List[Dict[str, Any]] = []
    skipped_questions: List[Dict[str, Any]] = []
    used_questions = 0

    baseline_f = open(baseline_path, "w", encoding="utf-8")
    resampled_f = open(resampled_path, "w", encoding="utf-8")
    try:
        for q_idx in range(num_questions):
            example = dataset[q_idx]
            question = extract_question_text(example)
            gold_answer = get_gold_answer(example)
            formatted_problem = format_problem(question)

            # 1) Baseline CoT
            cot_prompt_text = (
                "Solve this math problem briefly. "
                f"{DEFAULT_COT_ANSWER_INSTRUCTION}\n\n"
                f"{formatted_problem}"
            )
            cot_prompt = build_prompt(tokenizer, cot_prompt_text, system_prompt)
            baseline_seed = args.seed + (q_idx * 10000)
            cot_output = generate_text(
                model,
                tokenizer,
                cot_prompt,
                max_new_tokens=args.cot_max_new_tokens,
                do_sample=True,
                temperature=args.cot_temperature,
                top_p=args.cot_top_p,
                seed=baseline_seed,
            )

            cot_text = extract_cot(cot_output)
            baseline_cot_token_count = token_len(tokenizer, cot_text)
            baseline_pred = parse_model_answer(cot_output)
            baseline_correct = answers_match(baseline_pred, gold_answer)
            baseline_entry = {
                "question_index": q_idx,
                "question": question,
                "gold_answer": gold_answer,
                "baseline_seed": baseline_seed,
                "baseline_cot": cot_text,
                "baseline_cot_token_count": baseline_cot_token_count,
                "baseline_pred": baseline_pred,
                "baseline_correct": baseline_correct,
            }
            if args.save_outputs:
                baseline_entry["baseline_output"] = cot_output
            baseline_f.write(json.dumps(baseline_entry, ensure_ascii=False) + "\n")

            used_questions += 1

            sentences = split_sentences(cot_text)

            if args.sentence_limit is not None:
                sentences = sentences[: args.sentence_limit]

            if not sentences:
                raise ValueError(
                    f"No CoT sentences were extracted for question {q_idx}. "
                    "Try increasing --cot-max-new-tokens."
                )

            per_question_results: List[ExperimentResult] = []
            outputs_by_k: Dict[int, List[Dict[str, Any]]] = {}

            # 2) One-sentence CoT + filler padding
            k = 1
            received_sentences = sentences[0]
            current_tokens = token_len(tokenizer, received_sentences)
            if current_tokens > baseline_cot_token_count:
                raise ValueError(
                    f"Current reasoning token count ({current_tokens}) exceeds "
                    f"baseline CoT token count ({baseline_cot_token_count}) "
                    f"for question {q_idx}, sentence_count={k}."
                )
            padding_token_count = baseline_cot_token_count - current_tokens
            padding_text = tokenizer.decode(
                [filler_token_id] * padding_token_count,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ) if padding_token_count > 0 else ""
            padded_reasoning = received_sentences + padding_text
            resampled_reasoning_token_count = token_len(tokenizer, padded_reasoning)
            if resampled_reasoning_token_count != baseline_cot_token_count:
                raise ValueError(
                    f"Token count mismatch after filler padding for question {q_idx}, "
                    f"sentence_count={k}: got {resampled_reasoning_token_count}, "
                    f"expected {baseline_cot_token_count}."
                )
            answer_prompt_text = (
                f"{formatted_problem}\n\n"
                f"Previous reasoning: {padded_reasoning}\n"
                f"{DEFAULT_DIRECT_ANSWER_INSTRUCTION}"
            )
            answer_prompt = build_prompt(tokenizer, answer_prompt_text, system_prompt)
            resampled_input = answer_prompt_text
            correct = 0
            run_outputs: List[Dict[str, Any]] = []

            for r in range(args.num_repeats):
                seed = args.seed + (q_idx * 10000) + (k * 100) + r
                answer_output = generate_text(
                    model,
                    tokenizer,
                    answer_prompt,
                    max_new_tokens=args.answer_max_new_tokens,
                    do_sample=True,
                    temperature=args.answer_temperature,
                    top_p=args.answer_top_p,
                    seed=seed,
                )
                pred = parse_model_answer(answer_output)
                is_correct = answers_match(pred, gold_answer)
                if is_correct:
                    correct += 1
                run_record = {
                    "sentence_count": k,
                    "seed": seed,
                    "baseline_cot_token_count": baseline_cot_token_count,
                    "truncated_reasoning_token_count": current_tokens,
                    "resampled_reasoning_token_count": resampled_reasoning_token_count,
                    "padding_token_count": padding_token_count,
                    "parsed_answer": pred,
                    "correct": is_correct,
                }
                if args.save_outputs:
                    run_record["truncated_reasoning"] = received_sentences
                    run_record["padding_text"] = padding_text
                    run_record["resampled_input"] = resampled_input
                    run_record["raw_output"] = answer_output
                    run_outputs.append(run_record)
                resampled_f.write(json.dumps(run_record, ensure_ascii=False) + "\n")

            percent_correct = 100.0 * correct / args.num_repeats
            per_question_results.append(
                ExperimentResult(
                    sentence_count=k,
                    correct=correct,
                    total=args.num_repeats,
                    percent_correct=percent_correct,
                )
            )
            outputs_by_k[k] = run_outputs

            if k not in aggregated_counts:
                aggregated_counts[k] = {"correct": 0, "total": 0}
            aggregated_counts[k]["correct"] += correct
            aggregated_counts[k]["total"] += args.num_repeats

            question_entry = {
                "question_index": q_idx,
                "question": question,
                "gold_answer": gold_answer,
                "baseline_cot": cot_text,
                "cot_sentences": sentences,
                "results": [r.__dict__ for r in per_question_results],
            }
            if args.save_outputs:
                question_entry["sample_outputs"] = outputs_by_k
            question_results.append(question_entry)
    finally:
        baseline_f.close()
        resampled_f.close()

    if used_questions == 0:
        raise ValueError("No questions were processed.")

    results: List[ExperimentResult] = []
    for k in sorted(aggregated_counts.keys()):
        correct = aggregated_counts[k]["correct"]
        total = aggregated_counts[k]["total"]
        results.append(
            ExperimentResult(
                sentence_count=k,
                correct=correct,
                total=total,
                percent_correct=100.0 * correct / total if total > 0 else 0.0,
            )
        )

    # 3) Plot
    try:
        import matplotlib.pyplot as plt

        x_vals = [r.sentence_count for r in results]
        y_vals = [r.percent_correct for r in results]

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, marker="o")
        plt.ylim(0, 100)
        plt.xlabel("Number of CoT Sentences")
        plt.ylabel("Percent Correct")
        plt.title("CoT Filler: Accuracy vs. CoT Length")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        if args.separate_figures:
            base, ext = os.path.splitext(plot_path)
            for question_entry in question_results:
                q_idx = question_entry["question_index"]
                per_q = question_entry["results"]
                per_x = [r["sentence_count"] for r in per_q]
                per_y = [r["percent_correct"] for r in per_q]

                plt.figure(figsize=(8, 5))
                plt.plot(per_x, per_y, marker="o")
                plt.ylim(0, 100)
                plt.xlabel("Number of CoT Sentences")
                plt.ylabel("Percent Correct")
                plt.title(f"CoT Filler (Q{q_idx}): Accuracy vs. CoT Length")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{base}_q{q_idx}{ext}")
                plt.close()
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or set --plot-path '' to skip."
        ) from exc

    # 4) Save results
    output = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "filler_token": args.filler_token,
        "num_questions_requested": num_questions,
        "num_questions_used": used_questions,
        "num_questions_skipped": len(skipped_questions),
        "output_path": display_path(output_path),
        "plot_path": display_path(plot_path),
        "baseline_cot_path": display_path(baseline_path),
        "resampled_cot_path": display_path(resampled_path),
        "aggregated_results": [r.__dict__ for r in results],
        "question_results": question_results,
        "skipped_questions": skipped_questions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoT filler experiment")
    parser.add_argument(
        "--model",
        default="google/gemma-3-27b-it",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name (or alias: math500, gpqa_diamond)",
    )
    parser.add_argument(
        "--dataset-config",
        default=DEFAULT_DATASET_CONFIG,
        help="Dataset configuration name (leave unset for default)",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_DATASET_SPLIT,
        help="Dataset split to use",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code from the Hub",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="Number of answer generations for the 1-sentence setting",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=1,
        help="Number of questions from the dataset to run",
    )
    parser.add_argument(
        "--sentence-limit",
        type=int,
        default=None,
        help="Optional cap on number of CoT sentences",
    )
    parser.add_argument(
        "--cot-max-new-tokens",
        type=int,
        default=None,
        help="Optional max tokens for baseline CoT generation (default: no cap)",
    )
    parser.add_argument(
        "--cot-temperature",
        type=float,
        default=0.7,
        help="Temperature for baseline CoT generation",
    )
    parser.add_argument(
        "--cot-top-p",
        type=float,
        default=0.9,
        help="Top-p for baseline CoT generation",
    )
    parser.add_argument(
        "--answer-max-new-tokens",
        type=int,
        default=64,
        help="Max tokens for answer generation",
    )
    parser.add_argument(
        "--answer-temperature",
        type=float,
        default=0.7,
        help="Temperature for answer generation",
    )
    parser.add_argument(
        "--answer-top-p",
        type=float,
        default=0.9,
        help="Top-p for answer generation",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt",
    )
    parser.add_argument(
        "--filler-token",
        default="_",
        help="Single-character filler token used for token-length padding",
    )
    parser.add_argument(
        "--output-path",
        default="results/filler/summary.json",
        help="Where to save experiment metadata/results",
    )
    parser.add_argument(
        "--plot-path",
        default="results/filler/plot.png",
        help="Where to save the plot",
    )
    parser.add_argument(
        "--separate-figures",
        action="store_true",
        help="Save a separate plot for each question",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save raw model outputs for each run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_path == "":
        raise ValueError("plot_path cannot be empty. Provide a file name.")

    result = run_experiment(args)

    print(
        "Done. "
        f"results={result['output_path']} plot={result['plot_path']} "
        f"baseline={result['baseline_cot_path']} resampled={result['resampled_cot_path']}"
    )
    print(
        f"used={result['num_questions_used']}/{result['num_questions_requested']} "
        f"skipped={result['num_questions_skipped']}"
    )
    if result.get("aggregated_results"):
        max_k = max(r["sentence_count"] for r in result["aggregated_results"])
        print(f"max_k={max_k}")


if __name__ == "__main__":
    main()
