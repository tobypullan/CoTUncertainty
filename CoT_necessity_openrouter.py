#!/usr/bin/env python3
"""
CoT necessity experiment for the first N math questions, using OpenRouter API.

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
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests
from datasets import load_dataset


# -----------------------------
# Utilities
# -----------------------------

def build_messages(user_text: str, system_text: Optional[str] = None) -> List[Dict[str, str]]:
    if system_text:
        # Some providers/models (including Gemma 3 via certain routes) reject
        # system/developer roles. Keep prompts portable by inlining instruction.
        user_text = f"Instruction: {system_text}\n\n{user_text}"
    return [{"role": "user", "content": user_text}]


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


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str,
        timeout_seconds: float,
        max_retries: int,
        retry_base_seconds: float,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.site_url = site_url
        self.app_name = app_name
        self.seed_supported = True

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    @staticmethod
    def _extract_text(response_json: Dict[str, Any]) -> str:
        choices = response_json.get("choices")
        if not choices:
            raise RuntimeError("API response missing choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)

        return str(content)

    @staticmethod
    def _extract_error_text(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or "Unknown API error"

        if isinstance(payload, dict):
            if isinstance(payload.get("error"), dict):
                err = payload["error"]
                message = err.get("message")
                metadata = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
                provider_name = metadata.get("provider_name")
                raw = metadata.get("raw")

                parts: List[str] = []
                if isinstance(message, str) and message.strip():
                    parts.append(message.strip())
                if isinstance(provider_name, str) and provider_name.strip():
                    parts.append(f"provider={provider_name.strip()}")
                if isinstance(raw, str) and raw.strip():
                    raw_clean = " ".join(raw.split())
                    parts.append(f"details={raw_clean}")

                if parts:
                    return " | ".join(parts)
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

        return response.text.strip() or "Unknown API error"

    @staticmethod
    def _retryable_status(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    @staticmethod
    def _error_mentions_seed(error_text: str) -> bool:
        return bool(re.search(r"\bseed\b", error_text, flags=re.IGNORECASE))

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        seed: Optional[int],
    ) -> str:
        url = f"{self.api_base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None and self.seed_supported:
            payload["seed"] = seed

        last_error = "Unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = f"Network error: {exc}"
                if attempt < self.max_retries:
                    sleep_s = self.retry_base_seconds * (2**attempt)
                    sleep_s += random.uniform(0.0, 0.1)
                    time.sleep(sleep_s)
                    continue
                break

            if response.status_code == 200:
                try:
                    response_json = response.json()
                except ValueError as exc:
                    raise RuntimeError("OpenRouter returned invalid JSON.") from exc
                return self._extract_text(response_json)

            error_text = self._extract_error_text(response)
            last_error = f"HTTP {response.status_code}: {error_text}"

            if (
                response.status_code == 400
                and "seed" in payload
                and self.seed_supported
                and self._error_mentions_seed(error_text)
            ):
                self.seed_supported = False
                payload.pop("seed", None)
                continue

            if self._retryable_status(response.status_code) and attempt < self.max_retries:
                sleep_s = self.retry_base_seconds * (2**attempt)
                sleep_s += random.uniform(0.0, 0.1)
                time.sleep(sleep_s)
                continue

            break

        raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")


# -----------------------------
# Main experiment
# -----------------------------

def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing API key. Set environment variable {args.api_key_env}."
        )

    client = OpenRouterClient(
        api_key=api_key,
        model=args.model,
        api_base=args.api_base,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        site_url=args.site_url,
        app_name=args.app_name,
    )

    datasets_cache_dir = os.environ.get("HF_DATASETS_CACHE")
    dataset = load_first_split(
        args.dataset,
        args.dataset_config,
        args.split,
        cache_dir=datasets_cache_dir,
    )
    num_questions = min(args.num_questions, len(dataset))
    if num_questions <= 0:
        raise ValueError("--num-questions must be >= 1.")

    system_prompt = (
        "You are a concise math solver."
        if args.system_prompt is None
        else args.system_prompt
    )

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    plot_path = resolve_results_path(results_dir, args.plot_path)
    baseline_path = os.path.join(results_dir, "baselineCoT_openrouter.jsonl")
    resampled_path = os.path.join(results_dir, "resampled_CoT_openrouter.jsonl")

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
            baseline_seed = args.seed + (q_idx * 10000)
            cot_output = client.generate(
                messages=build_messages(cot_prompt_text, system_prompt),
                max_tokens=args.cot_max_new_tokens,
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
                resampled_input = answer_prompt_text
                correct = 0
                run_outputs: List[Dict[str, Any]] = []

                for r in range(args.num_repeats):
                    seed = args.seed + (q_idx * 10000) + (k * 100) + r
                    answer_output = client.generate(
                        messages=build_messages(answer_prompt_text, system_prompt),
                        max_tokens=args.answer_max_new_tokens,
                        temperature=args.answer_temperature,
                        top_p=args.answer_top_p,
                        seed=seed,
                    )
                    pred = parse_model_answer(answer_output)
                    is_correct = pred == gold_answer
                    if is_correct:
                        correct += 1
                    run_record = {
                        "question_index": q_idx,
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

    if used_questions == 0 and args.fail_on_all_skipped:
        raise ValueError(
            "All questions were skipped because the baseline answer was incorrect."
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
        plt.title("CoT Necessity (OpenRouter): Accuracy vs. CoT Length")
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
                plt.title(f"CoT Necessity (OpenRouter Q{q_idx}): Accuracy vs. CoT Length")
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
        "provider": "openrouter",
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
    parser = argparse.ArgumentParser(description="CoT necessity experiment via OpenRouter")
    parser.add_argument(
        "--model",
        default="google/gemma-3-27b-it:free",
        help="OpenRouter model slug",
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
        "--api-base",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable name that contains the OpenRouter API key",
    )
    parser.add_argument(
        "--site-url",
        default=None,
        help="Optional HTTP-Referer header value",
    )
    parser.add_argument(
        "--app-name",
        default="CoT Necessity Experiment",
        help="Optional X-Title header value",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries for transient API/network failures",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=1.0,
        help="Base backoff seconds for retries",
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
        help="Optional max tokens for baseline CoT generation",
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
        default="cot_necessity_openrouter_results.json",
        help="Where to save experiment metadata/results",
    )
    parser.add_argument(
        "--plot-path",
        default="results/cot_necessity_openrouter_plot.png",
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
    parser.add_argument(
        "--fail-on-all-skipped",
        action="store_true",
        help="Raise an error if no questions pass baseline correctness filter",
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
