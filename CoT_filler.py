#!/usr/bin/env python3
"""
Filler-token deliberation experiment for multiple-choice questions.

Workflow:
1) For each question, vary the number of filler tokens provided as a
   "reasoning" scratchpad (e.g., '_ _ _').
2) Ask the model to answer immediately using only the letter.
3) Repeat multiple samples for each filler count.
4) Plot percent-correct vs. number of filler tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if system_text:
        return f"{system_text}\n\n{user_text}"
    return user_text


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        set_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def parse_model_answer(text: str, num_choices: int) -> Optional[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    allowed = letters[: max(2, num_choices)]
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match and match.group(1) in allowed:
        return match.group(1)
    match = re.search(r"([A-Z])", text.upper())
    if match and match.group(1) in allowed:
        return match.group(1)
    return None


def make_filler_text(token: str, count: int, wrap: int) -> str:
    if count <= 0:
        return ""
    tokens = [token] * count
    if wrap <= 0:
        return " ".join(tokens)
    lines = []
    for i in range(0, count, wrap):
        lines.append(" ".join(tokens[i:i + wrap]))
    return "\n".join(lines)


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


def _normalize_choices(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None

    if isinstance(raw, list):
        if all(isinstance(x, str) for x in raw):
            return [x.strip() for x in raw]
        if all(isinstance(x, dict) for x in raw):
            if all("text" in x for x in raw):
                if all("label" in x for x in raw):
                    def key_fn(d: Dict[str, Any]) -> str:
                        return str(d.get("label", ""))
                    sorted_raw = sorted(raw, key=key_fn)
                    return [str(d["text"]).strip() for d in sorted_raw]
                return [str(d["text"]).strip() for d in raw]
    if isinstance(raw, dict):
        keys = list(raw.keys())
        if all(isinstance(k, str) for k in keys):
            letter_keys = sorted(keys)
            return [str(raw[k]).strip() for k in letter_keys]

    return None


def extract_question_and_choices(example: Dict[str, Any]) -> Tuple[str, List[str]]:
    question_keys = ["question", "problem", "prompt", "input", "query"]
    choice_keys = ["choices", "options", "answers", "answer_choices", "answer_options"]

    question = _first_nonempty(example, question_keys)
    if not question:
        raise ValueError(
            f"Could not find a question field. Available keys: {list(example.keys())}"
        )

    raw_choices = None
    for k in choice_keys:
        if k in example:
            raw_choices = example[k]
            break

    choices = _normalize_choices(raw_choices)
    if not choices:
        raise ValueError(
            "Could not parse choices. "
            f"Available keys: {list(example.keys())}. "
            "If your dataset uses different keys, add them to choice_keys."
        )

    return question, choices


def get_gold_letter(example: Dict[str, Any], choices: List[str]) -> str:
    answer_keys = [
        "answer",
        "label",
        "correct_answer",
        "gold",
        "gold_answer",
        "answer_idx",
    ]
    ans = None
    for k in answer_keys:
        if k in example:
            ans = example[k]
            break

    if ans is None:
        raise ValueError(
            f"Could not find an answer field. Available keys: {list(example.keys())}"
        )

    if isinstance(ans, str):
        cleaned = ans.strip().upper()
        m = re.match(r"^[\(\[]?([A-Z])[\)\].]?$", cleaned)
        if m:
            return m.group(1)
        if cleaned.isdigit():
            idx = int(cleaned)
            if idx < 0 or idx >= len(choices):
                raise ValueError(f"Answer index out of range: {idx}")
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[idx]
        for i, c in enumerate(choices):
            if cleaned == c.strip().upper():
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]

    if isinstance(ans, (int, float)):
        idx = int(ans)
        if idx < 0 or idx >= len(choices):
            raise ValueError(f"Answer index out of range: {idx}")
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[idx]

    raise ValueError(f"Could not parse answer value: {ans}")


def format_mcq(question: str, choices: List[str]) -> str:
    lines = [question.strip(), ""]
    for i, choice in enumerate(choices):
        letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
        lines.append(f"{letter}) {choice}")
    return "\n".join(lines)


def load_first_split(
    dataset_name: str,
    dataset_config: Optional[str],
    split: Optional[str],
):
    if not split:
        raise ValueError("split is required for this experiment.")
    return load_dataset(dataset_name, dataset_config, split=split)


@dataclass
class ExperimentResult:
    filler_count: int
    correct: int
    total: int
    percent_correct: float


# -----------------------------
# Main experiment
# -----------------------------

def parse_filler_counts(args: argparse.Namespace) -> List[int]:
    if args.filler_counts:
        raw = [x.strip() for x in args.filler_counts.split(",") if x.strip()]
        counts = [int(x) for x in raw]
    else:
        counts = list(
            range(
                args.min_filler_tokens,
                args.max_filler_tokens + 1,
                args.filler_step,
            )
        )

    counts = sorted(set(counts))
    if not counts:
        raise ValueError("No filler counts provided.")
    if any(c < 0 for c in counts):
        raise ValueError("Filler counts must be >= 0.")
    return counts


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dataset = load_first_split(
        args.dataset,
        args.dataset_config,
        args.split,
    )
    num_questions = min(args.num_questions, len(dataset))
    if num_questions <= 0:
        raise ValueError("--num-questions must be >= 1.")

    filler_counts = parse_filler_counts(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    ).to("cuda")
    model.eval()

    system_prompt = (
        "You are a careful multiple-choice solver."
        if args.system_prompt is None
        else args.system_prompt
    )

    aggregated_counts: Dict[int, Dict[str, int]] = {}
    question_results: List[Dict[str, Any]] = []

    for q_idx in range(num_questions):
        example = dataset[q_idx]
        question, choices = extract_question_and_choices(example)
        gold = get_gold_letter(example, choices)
        formatted_question = format_mcq(question, choices)

        per_question_results: List[ExperimentResult] = []
        outputs_by_k: Dict[int, List[Dict[str, Any]]] = {}

        for k in filler_counts:
            filler_text = make_filler_text(args.filler_token, k, args.filler_wrap)
            answer_prompt_text = (
                "Use the filler tokens below as your reasoning scratchpad. "
                "They contain no information, and you must not add any other reasoning. "
                "Answer immediately with only the letter (A, B, C, ...).\n\n"
                f"Question:\n{formatted_question}\n\n"
                f"Filler tokens ({k}):\n{filler_text}\n\n"
                "Answer:"
            )
            answer_prompt = build_prompt(tokenizer, answer_prompt_text, system_prompt)

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
                pred = parse_model_answer(answer_output, len(choices))
                is_correct = pred == gold
                if is_correct:
                    correct += 1
                run_outputs.append(
                    {
                        "seed": seed,
                        "raw_output": answer_output,
                        "parsed_answer": pred,
                        "correct": is_correct,
                    }
                )

            percent_correct = 100.0 * correct / args.num_repeats
            per_question_results.append(
                ExperimentResult(
                    filler_count=k,
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
            "choices": choices,
            "gold_letter": gold,
            "results": [r.__dict__ for r in per_question_results],
        }
        if args.save_outputs:
            question_entry["sample_outputs"] = outputs_by_k
        question_results.append(question_entry)

    results: List[ExperimentResult] = []
    for k in sorted(aggregated_counts.keys()):
        correct = aggregated_counts[k]["correct"]
        total = aggregated_counts[k]["total"]
        results.append(
            ExperimentResult(
                filler_count=k,
                correct=correct,
                total=total,
                percent_correct=100.0 * correct / total if total > 0 else 0.0,
            )
        )

    try:
        import matplotlib.pyplot as plt

        x_vals = [r.filler_count for r in results]
        y_vals = [r.percent_correct for r in results]

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, marker="o")
        plt.ylim(0, 100)
        plt.xlabel("Number of Filler Tokens")
        plt.ylabel(f"Percent Correct (n={args.num_repeats})")
        plt.title("Filler Tokens: Accuracy vs. Filler Length")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot_path)
        plt.close()
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or set --plot-path '' to skip."
        ) from exc

    output = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "num_questions": num_questions,
        "filler_token": args.filler_token,
        "filler_counts": [r.filler_count for r in results],
        "aggregated_results": [r.__dict__ for r in results],
        "question_results": question_results,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filler-token deliberation experiment")
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--dataset",
        default="TIGER-Lab/MMLU-Pro",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset configuration name",
    )
    parser.add_argument(
        "--split",
        default="test",
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
        "--num-questions",
        type=int,
        default=1,
        help="Number of questions to evaluate",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=10,
        help="Number of samples per filler count",
    )
    parser.add_argument(
        "--min-filler-tokens",
        type=int,
        default=0,
        help="Minimum number of filler tokens",
    )
    parser.add_argument(
        "--max-filler-tokens",
        type=int,
        default=32,
        help="Maximum number of filler tokens",
    )
    parser.add_argument(
        "--filler-step",
        type=int,
        default=1,
        help="Step size between filler counts",
    )
    parser.add_argument(
        "--filler-counts",
        default=None,
        help="Comma-separated filler counts to override min/max/step",
    )
    parser.add_argument(
        "--filler-token",
        default="_",
        help="Token to repeat as filler",
    )
    parser.add_argument(
        "--filler-wrap",
        type=int,
        default=40,
        help="Wrap filler tokens after this many tokens per line",
    )
    parser.add_argument(
        "--answer-max-new-tokens",
        type=int,
        default=8,
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
        default="cot_filler_results.json",
        help="Where to save experiment metadata/results",
    )
    parser.add_argument(
        "--plot-path",
        default="cot_filler_plot.png",
        help="Where to save the plot",
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

    print("Experiment completed.")
    print(f"Saved results to {args.output_path}")
    print(f"Saved plot to {args.plot_path}")
    print(f"Filler token: {result['filler_token']}")
    print(f"Filler counts: {result['filler_counts']}")


if __name__ == "__main__":
    main()
