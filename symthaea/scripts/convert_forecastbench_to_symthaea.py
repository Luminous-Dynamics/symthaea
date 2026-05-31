#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convert ForecastBench-like exports into Symthaea's local JSONL schema.

The official/live ecosystem may export records with slightly different field
names. This converter is deliberately tolerant and keeps unresolved questions
with `resolution: null`, so they appear in coverage but are excluded from Brier
and log scoring until resolved.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


TRUE_VALUES = {"true", "yes", "y", "1", "resolved_true", "will_happen"}
FALSE_VALUES = {"false", "no", "n", "0", "resolved_false", "will_not_happen"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--default-category", default="forecastbench_official")
    parser.add_argument("--default-probability", type=float, default=None)
    parser.add_argument("--default-baseline", type=float, default=0.5)
    args = parser.parse_args()

    rows = list(read_records(args.input))
    converted = [
        convert_record(
            row,
            idx,
            args.default_category,
            args.default_probability,
            args.default_baseline,
        )
        for idx, row in enumerate(rows, start=1)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in converted:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"converted {len(converted)} records -> {args.output}")
    return 0


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise SystemExit("top-level JSON array expected")
        return [require_object(item) for item in data]
    if text.startswith("{"):
        data = json.loads(text)
        if isinstance(data.get("questions"), list):
            return [require_object(item) for item in data["questions"]]
        if isinstance(data.get("data"), list):
            return [require_object(item) for item in data["data"]]
    return [require_object(json.loads(line)) for line in text.splitlines() if line.strip()]


def require_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"record is not an object: {value!r}")
    return value


def convert_record(
    row: dict[str, Any],
    idx: int,
    default_category: str,
    default_probability: float | None,
    default_baseline: float,
) -> dict[str, Any]:
    question = first_string(
        row,
        ["question", "title", "prompt", "body", "description", "question_text"],
    )
    if not question:
        question = f"ForecastBench question {idx}"
    identifier = first_string(row, ["id", "question_id", "qid", "slug"]) or slugify(question, idx)
    category = first_string(row, ["category", "domain", "source", "topic"]) or default_category
    evidence = row.get("evidence") or row.get("background") or row.get("sources") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    return {
        "id": str(identifier),
        "category": str(category),
        "question": str(question),
        "resolution": parse_resolution(row),
        "probability": first_probability(row, ["probability", "prediction", "p"])
        if default_probability is None
        else default_probability,
        "baseline_probability": first_probability(
            row,
            ["baseline_probability", "base_rate", "community_prediction", "market_probability"],
            default_baseline,
        ),
        "evidence": [str(item) for item in evidence],
    }


def first_string(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def first_probability(
    row: dict[str, Any],
    keys: list[str],
    default: float | None = None,
) -> float | None:
    for key in keys:
        value = row.get(key)
        parsed = parse_probability(value)
        if parsed is not None:
            return parsed
    return default


def parse_probability(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        probability = float(value)
    elif isinstance(value, str):
        stripped = value.strip().rstrip("%")
        if not stripped:
            return None
        try:
            probability = float(stripped)
        except ValueError:
            return None
        if value.strip().endswith("%"):
            probability /= 100.0
    else:
        return None
    if probability > 1.0:
        probability /= 100.0
    return min(max(probability, 0.0), 1.0)


def parse_resolution(row: dict[str, Any]) -> bool | None:
    for key in ["resolution", "resolved", "answer", "outcome", "result", "label"]:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in TRUE_VALUES:
                return True
            if normalized in FALSE_VALUES:
                return False
    return None


def slugify(question: str, idx: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
    return f"forecast_{idx}_{slug[:48]}" if slug else f"forecast_{idx}"


if __name__ == "__main__":
    raise SystemExit(main())
