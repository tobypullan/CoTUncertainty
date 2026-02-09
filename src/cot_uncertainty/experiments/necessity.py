#!/usr/bin/env python3
"""
CoT necessity experiment for the first N math questions.

Workflow:
1) Generate a baseline Chain-of-Thought (CoT) for the first N questions.
2) Split the CoT into sentences.
3) Re-prompt the model with the question + first k sentences of CoT,
   then force an immediate answer.
4) If CoT has fewer than 3 sentences, sweep "_" reasoning lengths in token
   steps up to the baseline CoT token count.
4) Repeat 5 times for each k.
5) Plot percent-correct vs. number of sentences.
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

EXPERIMENT_SLUG = "necessity"
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
    outputs = generate_text_samples(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        allowed_token_ids=allowed_token_ids,
        num_samples=1,
    )
    return outputs[0]


def _resolve_max_new_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_token_count: int,
    max_new_tokens: Optional[int],
) -> int:
    if max_new_tokens is not None:
        return max_new_tokens

    limit = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(limit, int) or limit <= 0:
        limit = getattr(tokenizer, "model_max_length", None)
    if not isinstance(limit, int) or limit <= 0 or limit > 100000:
        limit = 4096
    return max(1, limit - prompt_token_count)


def generate_text_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: Optional[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: Optional[int] = None,
    allowed_token_ids: Optional[List[int]] = None,
    num_samples: int = 1,
) -> List[str]:
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")

    if seed is not None:
        set_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_token_count = inputs["input_ids"].shape[1]
    resolved_max_new_tokens = _resolve_max_new_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt_token_count=prompt_token_count,
        max_new_tokens=max_new_tokens,
    )

    gen_kwargs = {
        "max_new_tokens": resolved_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if num_samples > 1:
        gen_kwargs["num_return_sequences"] = num_samples
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    if allowed_token_ids:
        def _allowed(_batch_id: int, _input_ids: torch.Tensor) -> List[int]:
            return allowed_token_ids

        gen_kwargs["prefix_allowed_tokens_fn"] = _allowed

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[:, prompt_token_count:]
    return [
        tokenizer.decode(sequence, skip_special_tokens=True)
        for sequence in generated_ids
    ]


def generate_repeated_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: Optional[int],
    temperature: float,
    top_p: float,
    num_samples: int,
    seed_base: int,
    sample_batch_size: int,
) -> List[str]:
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1.")

    effective_batch_size = sample_batch_size
    if effective_batch_size <= 0:
        effective_batch_size = num_samples

    outputs: List[str] = []
    produced = 0
    while produced < num_samples:
        current_batch = min(effective_batch_size, num_samples - produced)
        chunk_seed = seed_base + produced
        outputs.extend(
            generate_text_samples(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                seed=chunk_seed,
                num_samples=current_batch,
            )
        )
        produced += current_batch
    return outputs


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


def _is_mmlu_pro_dataset(dataset_name: Optional[str]) -> bool:
    if not dataset_name:
        return False
    normalized = dataset_name.strip().lower()
    return normalized in {"tiger-lab/mmlu-pro", "mmlu-pro", "mmlu_pro"}


def _extract_mmlu_options(example: Dict[str, Any]) -> List[str]:
    options = example.get("options")
    if not isinstance(options, list):
        return []
    normalized: List[str] = []
    for option in options:
        text = str(option).strip()
        if text:
            normalized.append(text)
    return normalized


def _option_letters(options: List[str]) -> List[str]:
    letters = []
    for idx in range(len(options)):
        if idx < 26:
            letters.append(chr(ord("A") + idx))
        else:
            letters.append(str(idx + 1))
    return letters


def parse_mmlu_categories_arg(value: str) -> List[str]:
    if value is None:
        return []
    parsed = [part.strip().lower() for part in str(value).split(",")]
    return [part for part in parsed if part]


def _is_allowed_mmlu_category(category: Any, allowed_categories: List[str]) -> bool:
    if category is None:
        return False
    normalized = str(category).strip().lower()
    return normalized in set(allowed_categories)


def _get_gold_answer_for_example(
    example: Dict[str, Any],
    is_mmlu_pro: bool,
    options: List[str],
) -> str:
    if not is_mmlu_pro:
        return get_gold_answer(example)

    answer_index = example.get("answer_index")
    label_candidates = _option_letters(options)
    indexed_answer: Optional[str] = None
    if isinstance(answer_index, int):
        if answer_index < 0 or answer_index >= len(label_candidates):
            raise ValueError(
                f"Invalid MMLU-Pro answer_index={answer_index} for "
                f"{len(label_candidates)} options."
            )
        indexed_answer = label_candidates[answer_index]

    parsed_answer: Optional[str] = None
    try:
        parsed_answer = get_gold_answer(example)
    except ValueError:
        parsed_answer = None

    if parsed_answer is None and indexed_answer is None:
        raise ValueError(
            "Could not parse MMLU-Pro gold answer from either `answer` "
            "or `answer_index`."
        )
    if parsed_answer is None:
        return indexed_answer  # type: ignore[return-value]
    if indexed_answer is None:
        return parsed_answer
    if parsed_answer != indexed_answer:
        raise ValueError(
            f"MMLU-Pro answer mismatch: answer={parsed_answer} "
            f"answer_index={answer_index}->{indexed_answer}"
        )
    return parsed_answer


def format_problem_for_dataset(
    question: str,
    example: Dict[str, Any],
    is_mmlu_pro: bool,
) -> str:
    question_text = format_problem(question)
    if not is_mmlu_pro:
        return question_text

    options = _extract_mmlu_options(example)
    if not options:
        raise ValueError(
            "MMLU-Pro example is missing non-empty options; cannot build prompt."
        )
    option_labels = _option_letters(options)
    option_lines = [f"{label}. {text}" for label, text in zip(option_labels, options)]
    return f"{question_text}\n\nOptions:\n" + "\n".join(option_lines)


@dataclass
class ExperimentResult:
    sentence_count: int
    correct: int
    total: int
    percent_correct: float


def _aggregate_counts_to_results(
    counts: Dict[int, Dict[str, int]],
) -> List[ExperimentResult]:
    results: List[ExperimentResult] = []
    for k in sorted(counts.keys()):
        correct = counts[k]["correct"]
        total = counts[k]["total"]
        results.append(
            ExperimentResult(
                sentence_count=k,
                correct=correct,
                total=total,
                percent_correct=100.0 * correct / total if total > 0 else 0.0,
            )
        )
    return results


def _mode_axis_label(mode: str) -> str:
    if mode == "underscore_token_sweep":
        return "Number of '_' Tokens in Previous Reasoning"
    return "Number of CoT Sentences"


def _mode_title_suffix(mode: str) -> str:
    if mode == "underscore_token_sweep":
        return "Filler Token Sweep"
    return "Sentence Prefix"


# -----------------------------
# Main experiment
# -----------------------------

def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if args.token_stride < 1:
        raise ValueError("--token-stride must be >= 1.")
    if args.num_repeats < 1:
        raise ValueError("--num-repeats must be >= 1.")
    repeat_batch_size = args.repeat_batch_size
    if repeat_batch_size <= 0:
        repeat_batch_size = args.num_repeats

    datasets_cache_dir = os.environ.get("HF_DATASETS_CACHE")
    hub_cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")

    dataset = load_experiment_dataset(
        args.dataset,
        args.dataset_config,
        args.split,
        cache_dir=datasets_cache_dir,
    )
    is_mmlu_pro = _is_mmlu_pro_dataset(args.dataset)
    mmlu_categories = parse_mmlu_categories_arg(args.mmlu_categories)
    selected_indices: List[int]
    if is_mmlu_pro:
        if not mmlu_categories:
            raise ValueError(
                "--mmlu-categories must include at least one category for MMLU-Pro."
            )
        selected_indices = [
            idx
            for idx in range(len(dataset))
            if _is_allowed_mmlu_category(dataset[idx].get("category"), mmlu_categories)
        ]
        if not selected_indices:
            raise ValueError(
                "No MMLU-Pro examples found for categories: "
                f"{', '.join(mmlu_categories)}."
            )
    else:
        selected_indices = list(range(len(dataset)))

    num_questions = min(args.num_questions, len(selected_indices))
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

    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    plot_path = resolve_results_path(results_dir, args.plot_path)
    output_path = resolve_results_path(results_dir, args.output_path)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    baseline_path = os.path.join(results_dir, "baseline_cot.jsonl")
    resampled_path = os.path.join(results_dir, "resampled_cot.jsonl")

    mode_names = ("sentence_prefix", "underscore_token_sweep")
    aggregated_counts_by_mode: Dict[str, Dict[int, Dict[str, int]]] = {
        mode: {} for mode in mode_names
    }
    mode_usage: Dict[str, int] = {mode: 0 for mode in mode_names}
    question_results: List[Dict[str, Any]] = []
    skipped_questions: List[Dict[str, Any]] = []
    used_questions = 0

    baseline_f = open(baseline_path, "w", encoding="utf-8")
    resampled_f = open(resampled_path, "w", encoding="utf-8")
    try:
        for run_idx, dataset_idx in enumerate(selected_indices[:num_questions]):
            example = dataset[dataset_idx]
            question = extract_question_text(example)
            mmlu_options = _extract_mmlu_options(example) if is_mmlu_pro else []
            gold_answer = _get_gold_answer_for_example(
                example,
                is_mmlu_pro=is_mmlu_pro,
                options=mmlu_options,
            )
            formatted_problem = format_problem_for_dataset(
                question,
                example,
                is_mmlu_pro=is_mmlu_pro,
            )

            cot_instruction = DEFAULT_COT_ANSWER_INSTRUCTION
            direct_instruction = DEFAULT_DIRECT_ANSWER_INSTRUCTION
            if is_mmlu_pro:
                labels = _option_letters(mmlu_options)
                label_hint = "/".join(labels)
                cot_instruction = (
                    "End with 'Final Answer: <option letter>' where the option letter "
                    f"is one of {label_hint}."
                )
                direct_instruction = (
                    "Give only the final option letter "
                    f"({label_hint})."
                )

            # 1) Baseline CoT
            cot_prompt_text = (
                "Solve this math problem briefly. "
                f"{cot_instruction}\n\n"
                f"{formatted_problem}"
            )
            cot_prompt = build_prompt(tokenizer, cot_prompt_text, system_prompt)
            baseline_seed = args.seed + (run_idx * 10000)
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
            full_sentences = split_sentences(cot_text)
            baseline_cot_token_count = token_len(tokenizer, cot_text)
            baseline_pred = parse_model_answer(cot_output)
            baseline_correct = answers_match(baseline_pred, gold_answer)
            baseline_entry = {
                "question_index": dataset_idx,
                "question": question,
                "gold_answer": gold_answer,
                "baseline_seed": baseline_seed,
                "baseline_cot": cot_text,
                "baseline_pred": baseline_pred,
                "baseline_correct": baseline_correct,
            }
            if is_mmlu_pro:
                baseline_entry.update(
                    {
                        "question_id": example.get("question_id"),
                        "options": mmlu_options,
                        "answer_index": example.get("answer_index"),
                        "category": example.get("category"),
                    }
                )
            if args.save_outputs:
                baseline_entry["baseline_output"] = cot_output
            baseline_f.write(json.dumps(baseline_entry, ensure_ascii=False) + "\n")

            if not baseline_correct:
                skipped_questions.append(
                    {
                        "question_index": dataset_idx,
                        "gold_answer": gold_answer,
                        "baseline_pred": baseline_pred,
                        "reason": "baseline_incorrect",
                    }
                )
                if is_mmlu_pro:
                    skipped_questions[-1]["question_id"] = example.get("question_id")
                    skipped_questions[-1]["answer_index"] = example.get("answer_index")
                continue

            if len(full_sentences) < 3:
                # Fallback: test direct-answer robustness by replacing reasoning
                # with increasing counts of "_" tokens.
                mode_usage["underscore_token_sweep"] += 1
                token_steps = list(range(0, baseline_cot_token_count + 1, args.token_stride))
                if not token_steps:
                    token_steps = [0]
                if token_steps[-1] != baseline_cot_token_count:
                    token_steps.append(baseline_cot_token_count)

                used_questions += 1
                per_question_results = []
                outputs_by_k: Dict[int, List[Dict[str, Any]]] = {}

                for k in token_steps:
                    received_tokens = " ".join(["_"] * k)
                    answer_prompt_text = (
                        f"{formatted_problem}\n\n"
                        f"Previous reasoning: {received_tokens}\n"
                        f"{direct_instruction}"
                    )
                    answer_prompt = build_prompt(tokenizer, answer_prompt_text, system_prompt)
                    resampled_input = answer_prompt_text
                    correct = 0
                    run_outputs: List[Dict[str, Any]] = []
                    seed_base = args.seed + (run_idx * 10000) + (k * 100)
                    answer_outputs = generate_repeated_samples(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=answer_prompt,
                        max_new_tokens=args.answer_max_new_tokens,
                        temperature=args.answer_temperature,
                        top_p=args.answer_top_p,
                        num_samples=args.num_repeats,
                        seed_base=seed_base,
                        sample_batch_size=repeat_batch_size,
                    )

                    for r, answer_output in enumerate(answer_outputs):
                        seed = seed_base + r
                        pred = parse_model_answer(answer_output)
                        is_correct = answers_match(pred, gold_answer)
                        if is_correct:
                            correct += 1
                        run_record = {
                            "question_index": dataset_idx,
                            "sentence_count": k,
                            "token_count": k,
                            "reasoning_mode": "underscore_token_sweep",
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

                    mode_counts = aggregated_counts_by_mode["underscore_token_sweep"]
                    if k not in mode_counts:
                        mode_counts[k] = {"correct": 0, "total": 0}
                    mode_counts[k]["correct"] += correct
                    mode_counts[k]["total"] += args.num_repeats

                question_entry = {
                    "question_index": dataset_idx,
                    "question": question,
                    "gold_answer": gold_answer,
                    "baseline_cot": cot_text,
                    "baseline_cot_token_count": baseline_cot_token_count,
                    "cot_sentences": full_sentences,
                    "reasoning_mode": "underscore_token_sweep",
                    "token_stride": args.token_stride,
                    "results": [r.__dict__ for r in per_question_results],
                }
                if is_mmlu_pro:
                    question_entry.update(
                        {
                            "question_id": example.get("question_id"),
                            "options": mmlu_options,
                            "answer_index": example.get("answer_index"),
                            "category": example.get("category"),
                        }
                    )
                if args.save_outputs:
                    question_entry["sample_outputs"] = outputs_by_k
                question_results.append(question_entry)
                continue

            sentences = full_sentences

            if args.sentence_limit is not None:
                sentences = sentences[: args.sentence_limit]

            if not sentences:
                raise ValueError(
                    f"No CoT sentences were extracted for question {dataset_idx}. "
                    "Try increasing --cot-max-new-tokens."
                )

            used_questions += 1
            mode_usage["sentence_prefix"] += 1

            per_question_results: List[ExperimentResult] = []
            outputs_by_k: Dict[int, List[Dict[str, Any]]] = {}

            # 2) Partial-CoT prompting
            for k in range(1, len(sentences) + 1):
                received_sentences = ". ".join(sentences[:k - 1])
                answer_prompt_text = (
                    f"{formatted_problem}\n\n"
                    f"Previous reasoning: {received_sentences}\n"
                    f"{direct_instruction}"
                )
                answer_prompt = build_prompt(tokenizer, answer_prompt_text, system_prompt)
                resampled_input = answer_prompt_text
                correct = 0
                run_outputs: List[Dict[str, Any]] = []
                seed_base = args.seed + (run_idx * 10000) + (k * 100)
                answer_outputs = generate_repeated_samples(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=answer_prompt,
                    max_new_tokens=args.answer_max_new_tokens,
                    temperature=args.answer_temperature,
                    top_p=args.answer_top_p,
                    num_samples=args.num_repeats,
                    seed_base=seed_base,
                    sample_batch_size=repeat_batch_size,
                )

                for r, answer_output in enumerate(answer_outputs):
                    seed = seed_base + r
                    pred = parse_model_answer(answer_output)
                    is_correct = answers_match(pred, gold_answer)
                    if is_correct:
                        correct += 1
                    run_record = {
                        "question_index": dataset_idx,
                        "sentence_count": k,
                        "reasoning_mode": "sentence_prefix",
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

                mode_counts = aggregated_counts_by_mode["sentence_prefix"]
                if k not in mode_counts:
                    mode_counts[k] = {"correct": 0, "total": 0}
                mode_counts[k]["correct"] += correct
                mode_counts[k]["total"] += args.num_repeats

            question_entry = {
                "question_index": dataset_idx,
                "question": question,
                "gold_answer": gold_answer,
                "baseline_cot": cot_text,
                "baseline_cot_token_count": baseline_cot_token_count,
                "cot_sentences": sentences,
                "reasoning_mode": "sentence_prefix",
                "results": [r.__dict__ for r in per_question_results],
            }
            if is_mmlu_pro:
                question_entry.update(
                    {
                        "question_id": example.get("question_id"),
                        "options": mmlu_options,
                        "answer_index": example.get("answer_index"),
                        "category": example.get("category"),
                    }
                )
            if args.save_outputs:
                question_entry["sample_outputs"] = outputs_by_k
            question_results.append(question_entry)
    finally:
        baseline_f.close()
        resampled_f.close()

    if used_questions == 0:
        raise ValueError(
            "All questions were skipped (baseline answers were incorrect)."
        )

    aggregated_results_by_mode: Dict[str, List[ExperimentResult]] = {
        mode: _aggregate_counts_to_results(aggregated_counts_by_mode[mode])
        for mode in mode_names
    }
    if aggregated_results_by_mode["underscore_token_sweep"]:
        primary_plot_mode = "underscore_token_sweep"
    else:
        primary_plot_mode = "sentence_prefix"
    results = aggregated_results_by_mode[primary_plot_mode]

    # 3) Plot
    try:
        import matplotlib.pyplot as plt

        x_vals = [r.sentence_count for r in results]
        y_vals = [r.percent_correct for r in results]

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, marker="o")
        plt.ylim(0, 100)
        plt.xlabel(_mode_axis_label(primary_plot_mode))
        plt.ylabel("Percent Correct")
        plt.title(
            f"CoT Necessity: Accuracy vs. CoT Length ({_mode_title_suffix(primary_plot_mode)})"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        plot_paths_by_mode: Dict[str, Optional[str]] = {mode: None for mode in mode_names}
        if results:
            plot_paths_by_mode[primary_plot_mode] = plot_path

        non_primary_modes = [
            mode
            for mode in mode_names
            if mode != primary_plot_mode and aggregated_results_by_mode[mode]
        ]
        for mode in non_primary_modes:
            mode_results = aggregated_results_by_mode[mode]
            mode_x = [r.sentence_count for r in mode_results]
            mode_y = [r.percent_correct for r in mode_results]
            base, ext = os.path.splitext(plot_path)
            mode_plot_path = f"{base}_{mode}{ext}"

            plt.figure(figsize=(8, 5))
            plt.plot(mode_x, mode_y, marker="o")
            plt.ylim(0, 100)
            plt.xlabel(_mode_axis_label(mode))
            plt.ylabel("Percent Correct")
            plt.title(
                f"CoT Necessity: Accuracy vs. CoT Length ({_mode_title_suffix(mode)})"
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mode_plot_path)
            plt.close()
            plot_paths_by_mode[mode] = mode_plot_path

        if args.separate_figures:
            base, ext = os.path.splitext(plot_path)
            for question_entry in question_results:
                q_idx = question_entry["question_index"]
                mode = question_entry.get("reasoning_mode", "sentence_prefix")
                per_q = question_entry["results"]
                per_x = [r["sentence_count"] for r in per_q]
                per_y = [r["percent_correct"] for r in per_q]

                plt.figure(figsize=(8, 5))
                plt.plot(per_x, per_y, marker="o")
                plt.ylim(0, 100)
                plt.xlabel(_mode_axis_label(mode))
                plt.ylabel("Percent Correct")
                plt.title(
                    f"CoT Necessity (Q{q_idx}): Accuracy vs. CoT Length "
                    f"({_mode_title_suffix(mode)})"
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{base}_q{q_idx}_{mode}{ext}")
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
        "category_filter": (mmlu_categories if is_mmlu_pro else None),
        "num_questions_available_after_filter": len(selected_indices),
        "num_questions_requested": num_questions,
        "num_questions_used": used_questions,
        "num_questions_skipped": len(skipped_questions),
        "output_path": display_path(output_path),
        "plot_path": display_path(plot_path),
        "plot_paths_by_mode": {
            mode: (display_path(mode_path) if mode_path else None)
            for mode, mode_path in plot_paths_by_mode.items()
        },
        "primary_plot_mode": primary_plot_mode,
        "mode_usage": mode_usage,
        "repeat_batch_size": repeat_batch_size,
        "underscore_token_sweep_triggered": mode_usage["underscore_token_sweep"] > 0,
        "sentence_prefix_triggered": mode_usage["sentence_prefix"] > 0,
        "baseline_cot_path": display_path(baseline_path),
        "resampled_cot_path": display_path(resampled_path),
        "aggregated_results": [r.__dict__ for r in results],
        "aggregated_results_by_mode": {
            mode: [r.__dict__ for r in mode_results]
            for mode, mode_results in aggregated_results_by_mode.items()
        },
        "question_results": question_results,
        "skipped_questions": skipped_questions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
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
        "--mmlu-categories",
        default="math,physics",
        help=(
            "Comma-separated categories to keep for TIGER-Lab/MMLU-Pro "
            "(e.g. 'computer science' or 'math,physics')."
        ),
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
        help="Number of samples per x-axis step (sentence or token step)",
    )
    parser.add_argument(
        "--repeat-batch-size",
        type=int,
        default=0,
        help=(
            "How many repeat samples to generate per `model.generate` call. "
            "Use 0 to auto-batch all repeats together."
        ),
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
        "--token-stride",
        type=int,
        default=5,
        help="Step size for underscore-token sweep when baseline CoT is <3 sentences",
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
        "--output-path",
        default="results/necessity/summary.json",
        help="Where to save experiment metadata/results",
    )
    parser.add_argument(
        "--plot-path",
        default="results/necessity/plot.png",
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
    mode_usage = result.get("mode_usage", {})
    print(
        "mode_usage "
        f"sentence_prefix={mode_usage.get('sentence_prefix', 0)} "
        f"underscore_token_sweep={mode_usage.get('underscore_token_sweep', 0)}"
    )
    print(
        "branch_signal "
        f"underscore_token_sweep_triggered={int(bool(result.get('underscore_token_sweep_triggered')))} "
        f"primary_plot_mode={result.get('primary_plot_mode')}"
    )
    print(f"plot_paths_by_mode={result.get('plot_paths_by_mode')}")
    if result.get("aggregated_results"):
        max_x = max(r["sentence_count"] for r in result["aggregated_results"])
        print(f"max_x={max_x}")


if __name__ == "__main__":
    main()
