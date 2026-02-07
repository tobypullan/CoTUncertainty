#!/usr/bin/env python3
"""
CoT necessity experiment for the first N math questions.

Workflow:
1) Generate a baseline Chain-of-Thought (CoT) for the first N questions.
2) Split the CoT into sentences.
3) Re-prompt the model with the question + first k sentences of CoT,
   then force an immediate answer.
4) Repeat 5 times for each k.
5) Plot percent-correct vs. number of sentences.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

# Prefer the user's standard HF cache, but fall back to a local cache if writes fail.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_HF_HOME = os.path.join(SCRIPT_DIR, ".hf_cache")
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def normalize_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        text = boxed_matches[-1].strip()

    marker_match = re.search(
        r"(?:final\s+answer|answer)\s*(?:is|:|-|=)?\s*[-+]?(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if marker_match:
        return str(int(marker_match.group(1)))

    integers = re.findall(r"[-+]?\d+", text)
    if integers:
        return str(int(integers[-1]))

    return text


def parse_model_answer(text: str) -> Optional[str]:
    if not text:
        return None
    return normalize_answer(text)


def resolve_results_path(results_dir: str, requested_path: str) -> str:
    if not requested_path:
        return requested_path
    abs_results = os.path.abspath(results_dir)
    abs_requested = os.path.abspath(requested_path)
    try:
        common = os.path.commonpath([abs_results, abs_requested])
    except ValueError:
        common = ""
    if common == abs_results:
        return requested_path
    return os.path.join(results_dir, os.path.basename(requested_path))


# -----------------------------
# Dataset parsing
# -----------------------------

def _first_nonempty(example: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in example and example[k] is not None:
            val = example[k]
            if isinstance(val, str) and val.strip() == "":
                continue
            return str(val)
    return None


def extract_question_text(example: Dict[str, Any]) -> str:
    question_keys = ["problem", "question", "prompt", "input", "query"]
    question = _first_nonempty(example, question_keys)
    if not question:
        raise ValueError(
            f"Could not find a question field. Available keys: {list(example.keys())}"
        )
    return question


def get_gold_answer(example: Dict[str, Any]) -> str:
    answer_keys = [
        "answer",
        "final_answer",
        "label",
        "correct_answer",
        "gold",
        "gold_answer",
    ]
    ans = None
    for k in answer_keys:
        if k in example:
            ans = example[k]
            break

    normalized = normalize_answer(ans)
    if normalized is None:
        raise ValueError(
            f"Could not parse answer value from keys {answer_keys}. "
            f"Available keys: {list(example.keys())}"
        )
    return normalized


def format_problem(question: str) -> str:
    return question.strip()


def load_first_split(
    dataset_name: str,
    dataset_config: Optional[str],
    split: Optional[str],
    cache_dir: Optional[str] = None,
):
    if not split:
        raise ValueError("split is required for this experiment.")
    if isinstance(dataset_config, str):
        normalized = dataset_config.strip()
        if normalized.lower() in {"", "none", "null"}:
            dataset_config = None
        else:
            dataset_config = normalized
    return load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)


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

    dataset = load_first_split(
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

    system_prompt = (
        "You are a concise math solver."
        if args.system_prompt is None
        else args.system_prompt
    )

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    plot_path = resolve_results_path(results_dir, args.plot_path)
    baseline_path = os.path.join(results_dir, "baselineCoT.jsonl")
    resampled_path = os.path.join(results_dir, "resampled_CoT.jsonl")

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
                "End with 'Final Answer: <integer>'.\n\n"
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
            baseline_pred = parse_model_answer(cot_output)
            baseline_correct = baseline_pred == gold_answer
            baseline_entry = {
                "question_index": q_idx,
                "question": question,
                "gold_answer": gold_answer,
                "baseline_seed": baseline_seed,
                "baseline_cot": cot_text,
                "baseline_pred": baseline_pred,
                "baseline_correct": baseline_correct,
            }
            if args.save_outputs:
                baseline_entry["baseline_output"] = cot_output
            baseline_f.write(json.dumps(baseline_entry, ensure_ascii=False) + "\n")

            if not baseline_correct:
                skipped_questions.append(
                    {
                        "question_index": q_idx,
                        "gold_answer": gold_answer,
                        "baseline_pred": baseline_pred,
                    }
                )
                continue

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

            # 2) Partial-CoT prompting
            for k in range(1, len(sentences) + 1):
                received_sentences = ". ".join(sentences[:k - 1])
                answer_prompt_text = (
                    f"{formatted_problem}\n\n"
                    f"Previous reasoning: {received_sentences}\n"
                    "Give only the final integer answer."
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
                    is_correct = pred == gold_answer
                    if is_correct:
                        correct += 1
                    run_record = {
                        "sentence_count": k,
                        "seed": seed,
                        "parsed_answer": pred,
                        "correct": is_correct,
                    }
                    if args.save_outputs:
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
        raise ValueError(
            "All questions were skipped because the baseline answer was incorrect."
        )

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
        plt.title("CoT Necessity: Accuracy vs. CoT Length")
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
                plt.title(f"CoT Necessity (Q{q_idx}): Accuracy vs. CoT Length")
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
        "num_questions_requested": num_questions,
        "num_questions_used": used_questions,
        "num_questions_skipped": len(skipped_questions),
        "plot_path": plot_path,
        "baseline_cot_path": baseline_path,
        "resampled_cot_path": resampled_path,
        "aggregated_results": [r.__dict__ for r in results],
        "question_results": question_results,
        "skipped_questions": skipped_questions,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoT necessity experiment")
    parser.add_argument(
        "--model",
        default="google/gemma-3-27b-it",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceH4/aime_2024",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Dataset configuration name (leave unset for default)",
    )
    parser.add_argument(
        "--split",
        default="train",
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
        default=5,
        help="Number of samples per sentence count",
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
        default=16,
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
        "--output-path",
        default="cot_necessity_results.json",
        help="Where to save experiment metadata/results",
    )
    parser.add_argument(
        "--plot-path",
        default="results/cot_necessity_plot.png",
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
        f"results={args.output_path} plot={result['plot_path']} "
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
