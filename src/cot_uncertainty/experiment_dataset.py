#!/usr/bin/env python3
"""Shared dataset loading and answer parsing for CoT experiments."""

from __future__ import annotations

import inspect
import re
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: Optional[str]
    split: str
    question_keys: tuple[str, ...] = (
        "problem",
        "question",
        "prompt",
        "input",
        "query",
    )
    answer_keys: tuple[str, ...] = (
        "answer",
        "final_answer",
        "label",
        "correct_answer",
        "gold",
        "gold_answer",
        "solution",
    )
    cot_answer_instruction: str = (
        "End with 'Final Answer: <answer>'. Use exact mathematical notation, "
        "including LaTeX when needed."
    )
    direct_answer_instruction: str = (
        "Give only the final answer. Use exact mathematical notation, "
        "including LaTeX when needed."
    )


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "math500": DatasetSpec(
        name="HuggingFaceH4/MATH-500",
        config=None,
        split="test",
    ),
    "gpqa_diamond": DatasetSpec(
        name="fingertap/GPQA-Diamond",
        config=None,
        split="test",
        cot_answer_instruction="End with 'Final Answer: <option-letter>'.",
        direct_answer_instruction="Give only the final option letter.",
    ),
}

DEFAULT_DATASET_ID = "math500"
DEFAULT_DATASET_SPEC = DATASET_SPECS[DEFAULT_DATASET_ID]
DEFAULT_DATASET_NAME = DEFAULT_DATASET_SPEC.name
DEFAULT_DATASET_CONFIG = DEFAULT_DATASET_SPEC.config
DEFAULT_DATASET_SPLIT = DEFAULT_DATASET_SPEC.split
DEFAULT_COT_ANSWER_INSTRUCTION = DEFAULT_DATASET_SPEC.cot_answer_instruction
DEFAULT_DIRECT_ANSWER_INSTRUCTION = DEFAULT_DATASET_SPEC.direct_answer_instruction


def _resolve_dataset_spec(dataset_name: Optional[str]) -> Optional[DatasetSpec]:
    if not dataset_name:
        return None

    key = dataset_name.strip().lower()
    if not key:
        return None

    if key in DATASET_SPECS:
        return DATASET_SPECS[key]

    for spec in DATASET_SPECS.values():
        if key == spec.name.lower():
            return spec

    return None


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

from datasets import load_dataset


def _normalize_dataset_config(dataset_config: Optional[str]) -> Optional[str]:
    if isinstance(dataset_config, str):
        normalized = dataset_config.strip()
        if normalized.lower() in {"", "none", "null"}:
            return None
        return normalized
    return dataset_config


def load_experiment_dataset(
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    if dataset_name is None:
        spec = DEFAULT_DATASET_SPEC
    else:
        spec = _resolve_dataset_spec(dataset_name)

    if spec is not None:
        dataset_name = spec.name
        if dataset_config is None:
            dataset_config = spec.config
        if split is None:
            split = spec.split
    else:
        dataset_name = dataset_name or DEFAULT_DATASET_NAME
        if dataset_config is None:
            dataset_config = DEFAULT_DATASET_CONFIG
        if split is None:
            split = DEFAULT_DATASET_SPLIT

    dataset_config = _normalize_dataset_config(dataset_config)
    split = split or DEFAULT_DATASET_SPLIT
    if not split:
        raise ValueError("split is required for this experiment.")

    return load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)


def _first_nonempty(example: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in example and example[k] is not None:
            val = example[k]
            if isinstance(val, str) and val.strip() == "":
                continue
            return str(val)
    return None


def extract_question_text(
    example: Dict[str, Any],
    question_keys: Iterable[str] = DEFAULT_DATASET_SPEC.question_keys,
) -> str:
    question = _first_nonempty(example, question_keys)
    if not question:
        raise ValueError(
            f"Could not find a question field. Available keys: {list(example.keys())}"
        )
    return question


def get_gold_answer(
    example: Dict[str, Any],
    answer_keys: Iterable[str] = DEFAULT_DATASET_SPEC.answer_keys,
) -> str:
    ans = None
    for k in answer_keys:
        if k in example:
            ans = example[k]
            break

    normalized = normalize_answer(ans)
    if normalized is None:
        raise ValueError(
            f"Could not parse answer value from keys {list(answer_keys)}. "
            f"Available keys: {list(example.keys())}"
        )
    return normalized


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_last_boxed(text: str) -> Optional[str]:
    token = "\\boxed{"
    start = text.rfind(token)
    while start != -1:
        idx = start + len(token)
        depth = 1
        while idx < len(text):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len(token):idx].strip()
            idx += 1
        start = text.rfind(token, 0, start)
    return None


def _extract_marker_candidate(text: str) -> Optional[str]:
    marker_re = re.compile(
        r"(?:final\s+answer|answer)\s*(?:is|:|=|-)?\s*(.+)",
        flags=re.IGNORECASE,
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = marker_re.search(line)
        if match:
            return match.group(1).strip()
    return None


def _strip_math_wrappers(text: str) -> str:
    wrapped = text.strip()
    while True:
        candidate = wrapped
        if candidate.startswith("$") and candidate.endswith("$") and len(candidate) > 1:
            wrapped = candidate[1:-1].strip()
        elif candidate.startswith("\\(") and candidate.endswith("\\)"):
            wrapped = candidate[2:-2].strip()
        elif candidate.startswith("\\[") and candidate.endswith("\\]"):
            wrapped = candidate[2:-2].strip()
        else:
            break
    return wrapped


def _normalize_math_text(text: str) -> str:
    normalized = text.strip()
    # GSM8K answers are often formatted as "#### <answer>".
    normalized = re.sub(r"^\s*#{2,}\s*", "", normalized)
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace("\\dfrac", "\\frac")
    normalized = normalized.replace("\\tfrac", "\\frac")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace("\\;", "")
    normalized = normalized.replace("\\:", "")
    normalized = _strip_math_wrappers(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"[\s\.,;:]+$", "", normalized)
    return normalized


def normalize_answer(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = _strip_code_fences(str(value))
    if not text:
        return None

    boxed = _extract_last_boxed(text)
    if boxed is not None:
        candidate = boxed
    else:
        marker = _extract_marker_candidate(text)
        if marker is not None:
            candidate = marker
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            candidate = lines[-1] if lines else text

    normalized = _normalize_math_text(candidate)
    return normalized or None


def parse_model_answer(text: str) -> Optional[str]:
    if not text:
        return None
    return normalize_answer(text)


def _canonicalize_for_compare(text: str) -> str:
    canonical = _normalize_math_text(text).lower()
    canonical = canonical.replace(" ", "")
    canonical = canonical.replace("\\left", "")
    canonical = canonical.replace("\\right", "")
    canonical = canonical.replace("\\dfrac", "\\frac")
    canonical = canonical.replace("\\tfrac", "\\frac")
    canonical = canonical.replace("°", "")
    canonical = re.sub(r"\^\{?\\circ\}?", "", canonical)
    canonical = canonical.replace("\\circ", "")
    return canonical


def _parse_numeric_value(text: str) -> Optional[Fraction]:
    numeric = _canonicalize_for_compare(text)
    if re.fullmatch(r"[-+]?\d+", numeric):
        return Fraction(int(numeric), 1)

    if re.fullmatch(r"[-+]?\d+/[-+]?\d+", numeric):
        num_s, den_s = numeric.split("/", 1)
        den = int(den_s)
        if den == 0:
            return None
        return Fraction(int(num_s), den)

    frac_match = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", numeric)
    if frac_match:
        den = int(frac_match.group(2))
        if den == 0:
            return None
        return Fraction(int(frac_match.group(1)), den)

    if re.fullmatch(r"[-+]?\d*\.\d+", numeric):
        try:
            return Fraction(Decimal(numeric))
        except (InvalidOperation, ZeroDivisionError):
            return None

    return None


def answers_match(prediction: Any, gold: Any) -> bool:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)

    if pred is None or target is None:
        return False
    if pred == target:
        return True

    pred_canonical = _canonicalize_for_compare(pred)
    gold_canonical = _canonicalize_for_compare(target)
    if pred_canonical == gold_canonical:
        return True

    pred_numeric = _parse_numeric_value(pred)
    gold_numeric = _parse_numeric_value(target)
    if pred_numeric is not None and gold_numeric is not None:
        return pred_numeric == gold_numeric

    return False
