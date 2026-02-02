#!/usr/bin/env python3
"""
CoT necessity experiment for the first N ARC-Challenge questions.

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


def parse_model_answer(text: str, num_choices: int) -> Optional[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    allowed = letters[: max(2, num_choices)]
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match and match.group(1) in allowed:
        return match.group(1)
    # Sometimes the model returns like "Answer: C" without spaces.
    match = re.search(r"([A-Z])", text.upper())
    if match and match.group(1) in allowed:
        return match.group(1)
    return None


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

    # ARC format: {"label": ["A", "B", ...], "text": ["...", "...", ...]}
    if isinstance(raw, dict) and "label" in raw and "text" in raw:
        labels = raw.get("label")
        texts = raw.get("text")
        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            pairs = list(zip(labels, texts))
            if all(isinstance(l, str) for l, _ in pairs):
                pairs = sorted(pairs, key=lambda p: p[0])
            return [str(t).strip() for _, t in pairs]

    # Case 1: list of strings
    if isinstance(raw, list):
        if all(isinstance(x, str) for x in raw):
            return [x.strip() for x in raw]
        # list of dicts
        if all(isinstance(x, dict) for x in raw):
            # common pattern: {"label": "A", "text": "..."}
            if all("text" in x for x in raw):
                if all("label" in x for x in raw):
                    def key_fn(d: Dict[str, Any]) -> str:
                        return str(d.get("label", ""))
                    sorted_raw = sorted(raw, key=key_fn)
                    return [str(d["text"]).strip() for d in sorted_raw]
                return [str(d["text"]).strip() for d in raw]
    # Case 2: dict of letter->choice
    if isinstance(raw, dict):
        # If keys are letters, sort by letter
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
    # Common answer keys
    answer_keys = [
        "answerKey",
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

    # If already a letter
    if isinstance(ans, str):
        cleaned = ans.strip().upper()
        # e.g., "A" or "A." or "(A)"
        m = re.match(r"^[\(\[]?([A-Z])[\)\].]?$", cleaned)
        if m:
            return m.group(1)
        # If it's a digit string
        if cleaned.isdigit():
            idx = int(cleaned)
            if idx < 0 or idx >= len(choices):
                raise ValueError(f"Answer index out of range: {idx}")
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[idx]
        # Match to choice text
        for i, c in enumerate(choices):
            if cleaned == c.strip().upper():
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]

    # If integer index
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

    dataset = load_first_split(
        args.dataset,
        args.dataset_config,
        args.split,
    )
    num_questions = min(args.num_questions, len(dataset))
    if num_questions <= 0:
        raise ValueError("--num-questions must be >= 1.")

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
    skipped_questions: List[Dict[str, Any]] = []
    used_questions = 0

    for q_idx in range(num_questions):
        example = dataset[q_idx]
        question, choices = extract_question_and_choices(example)
        gold = get_gold_letter(example, choices)
        formatted_question = format_mcq(question, choices)

        # 1) Baseline CoT
        cot_prompt_text = (
            "Solve the following multiple-choice question. "
            "Think step-by-step and end with 'Final Answer: <letter>'.\n\n"
            f"{formatted_question}\n\n"
            "Let's think step by step."
        )
        cot_prompt = build_prompt(tokenizer, cot_prompt_text, system_prompt)
        cot_output = generate_text(
            model,
            tokenizer,
            cot_prompt,
            max_new_tokens=args.cot_max_new_tokens,
            do_sample=True,
            temperature=args.cot_temperature,
            top_p=args.cot_top_p,
            seed=args.seed + (q_idx * 10000),
        )

        baseline_pred = parse_model_answer(cot_output, len(choices))
        baseline_correct = baseline_pred == gold
        if not baseline_correct:
            skipped_questions.append(
                {
                    "question_index": q_idx,
                    "gold_letter": gold,
                    "baseline_pred": baseline_pred,
                }
            )
            continue

        used_questions += 1

        cot_text = extract_cot(cot_output)
        sentences = split_sentences(cot_text)

        if args.sentence_limit is not None:
            sentences = sentences[: args.sentence_limit]

        if not sentences:
            raise ValueError(
                f"No CoT sentences were extracted for question {q_idx}. "
                "Try increasing cot_max_new_tokens."
            )

        per_question_results: List[ExperimentResult] = []
        outputs_by_k: Dict[int, List[Dict[str, Any]]] = {}

        # 2) Partial-CoT prompting
        for k in range(1, len(sentences) + 1):
            partial_cot = " ".join(sentences[:k])
            answer_prompt_text = (
                "Use the partial reasoning below and answer the question immediately. "
                "Respond with only the letter (A, B, C, ...).\n\n"
                f"Question:\n{formatted_question}\n\n"
                f"Partial reasoning:\n{partial_cot}\n\n"
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
            "choices": choices,
            "gold_letter": gold,
            "baseline_cot": cot_text,
            "cot_sentences": sentences,
            "results": [r.__dict__ for r in per_question_results],
        }
        if args.save_outputs:
            question_entry["sample_outputs"] = outputs_by_k
        question_results.append(question_entry)

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
        plt.savefig(args.plot_path)
        plt.close()

        if args.separate_figures:
            base, ext = os.path.splitext(args.plot_path)
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
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--dataset",
        default="allenai/ai2_arc",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--dataset-config",
        default="ARC-Challenge",
        help="Dataset configuration name",
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
        default=512,
        help="Max tokens for baseline CoT generation",
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
        default="cot_necessity_results.json",
        help="Where to save experiment metadata/results",
    )
    parser.add_argument(
        "--plot-path",
        default="cot_necessity_plot.png",
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

    print("Experiment completed.")
    print(f"Saved results to {args.output_path}")
    print(f"Saved plot to {args.plot_path}")
    print(
        "Questions used: "
        f"{result['num_questions_used']}/{result['num_questions_requested']} "
        f"(skipped {result['num_questions_skipped']})"
    )
    if result.get("aggregated_results"):
        max_k = max(r["sentence_count"] for r in result["aggregated_results"])
        print(f"Max CoT sentences used: {max_k}")


if __name__ == "__main__":
    main()
