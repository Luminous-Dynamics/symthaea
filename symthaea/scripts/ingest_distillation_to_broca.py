#!/usr/bin/env python3
"""Convert verified coding artifacts into Broca training JSONL.

Supported inputs:
- Distillation baseline records already shaped as TrainingPair-like JSON.
- Coding backend repair lessons emitted by:

    cargo run --example benchmark_coding_backends \
      --features code_generation,geodesic_synthesis \
      -- --lane repair --repair-lessons-jsonl /tmp/repair.jsonl

Repair lessons become explicit failure -> hint -> corrected-code examples so
Broca can learn repair as a first-class cognitive act.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DISTILLATION_INPUT = Path("data/benchmarks/humaneval/distillation_baseline.jsonl")
DEFAULT_OUTPUT = Path("data/training/broca_humaneval_pristine.jsonl")
CURRENT_CHANNEL_COUNT = 24


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DISTILLATION_INPUT,
        help="TrainingPair-like distillation JSONL to ingest",
    )
    parser.add_argument(
        "--repair-lessons",
        type=Path,
        help="Repair lesson JSONL from benchmark_coding_backends",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output Broca TrainingPair JSONL",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output instead of replacing it",
    )
    parser.add_argument(
        "--skip-input",
        action="store_true",
        help="Only ingest --repair-lessons; ignore the distillation input path",
    )
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    if args.skip_input:
        pass
    elif args.input.exists():
        records.extend(read_training_pair_records(args.input))
    elif args.input != DEFAULT_DISTILLATION_INPUT:
        raise SystemExit(f"input not found: {args.input}")

    if args.repair_lessons:
        if not args.repair_lessons.exists():
            raise SystemExit(f"repair lessons not found: {args.repair_lessons}")
        records.extend(read_repair_lesson_records(args.repair_lessons))

    if not records:
        raise SystemExit(
            "no records ingested; provide --input, --repair-lessons, or run the benchmark first"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.output.open(mode) as out:
        for record in records:
            out.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"ingested {len(records)} Broca training pair(s) into {args.output}")
    if args.repair_lessons:
        print(
            "repair lessons are encoded as target_text prompts containing the failed source, "
            "diagnostic, repair hint, and corrected source"
        )
    return 0


def read_training_pair_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for item in read_jsonl(path):
        if "channels" in item and "target_text" in item:
            records.append(
                {
                    "channels": normalize_channels(item["channels"]),
                    "target_text": str(item["target_text"]),
                    "target_ids": item.get("target_ids", []),
                    "valence": float(item.get("valence", 0.0)),
                    "arousal": float(item.get("arousal", 0.0)),
                }
            )
    return records


def read_repair_lesson_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for lesson in read_jsonl(path):
        if not lesson.get("broca_training_record"):
            continue
        fixed = lesson.get("fixed_source_preview")
        if not fixed:
            continue
        records.append(repair_lesson_to_training_pair(lesson, fixed))
    return records


def repair_lesson_to_training_pair(lesson: dict[str, Any], fixed_source: str) -> dict[str, Any]:
    category = str(lesson.get("category") or "other")
    diagnostic = str(lesson.get("diagnostic") or "")
    hint = str(lesson.get("hint") or "")
    bad_source = str(lesson.get("source_preview") or "")
    task_name = str(lesson.get("task_name") or "unknown")
    signature = str(lesson.get("signature") or "")
    final_backend = str(lesson.get("final_backend") or "")
    structural_context = repair_lesson_structural_context(lesson)

    target_text = "\n".join(
        [
            "Repair this Rust generation failure.",
            f"task: {task_name}",
            f"signature: {signature}",
            f"category: {category}",
            f"diagnostic: {diagnostic}",
            f"repair_hint: {hint}",
            f"successful_backend: {final_backend}",
            *structural_context,
            "failed_source:",
            bad_source or "<unavailable>",
            "corrected_source:",
            fixed_source,
        ]
    )

    return {
        "channels": repair_channels(category),
        "target_text": target_text,
        "target_ids": [],
        "valence": -0.15,
        "arousal": 0.65,
    }


def repair_lesson_structural_context(lesson: dict[str, Any]) -> list[str]:
    lines = []
    label = lesson.get("structural_prior_label")
    broken_score = lesson.get("broken_structural_prior_score")
    fixed_score = lesson.get("fixed_structural_prior_score")
    delta = lesson.get("structural_prior_delta")
    similarity = lesson.get("structural_similarity")
    l1_delta = lesson.get("structural_l1_delta")

    if label is not None:
        lines.append(f"structural_prior_label: {label}")
    if broken_score is not None:
        lines.append(f"broken_structural_prior_score: {float(broken_score):.4f}")
    if fixed_score is not None:
        lines.append(f"fixed_structural_prior_score: {float(fixed_score):.4f}")
    if delta is not None:
        lines.append(f"structural_prior_delta: {float(delta):.4f}")
    if similarity is not None:
        lines.append(f"structural_repair_similarity: {float(similarity):.4f}")
    if l1_delta is not None:
        lines.append(f"structural_repair_l1_delta: {int(l1_delta)}")

    return lines


def repair_channels(category: str) -> list[float]:
    channels = [0.0] * CURRENT_CHANNEL_COUNT
    # Stable coarse encoding. Broca's encoder can later replace this with a
    # richer HDC projection, but these dimensions already mark the cognitive act.
    channels[0] = 0.82  # technical/code intent
    channels[1] = 0.35  # exploratory pressure
    channels[2] = 0.75  # verification/repair salience
    channels[3] = 0.70  # structured reasoning
    channels[4] = 0.45  # novelty: failure-dependent but bounded
    channels[5] = 0.20  # low free-form creativity
    channels[20] = 0.90  # epistemic tier proxy: compiler/test grounded
    channels[21] = 0.75  # necessity: concrete repair needed
    channels[22] = category_signal(category)
    channels[23] = 0.85  # expected quality: successful correction
    return channels


def category_signal(category: str) -> float:
    buckets = {
        "parse_failure": 0.15,
        "stub": 0.25,
        "type_mismatch": 0.35,
        "unresolved_identifier": 0.45,
        "ownership": 0.55,
        "test_failure": 0.65,
        "sheaf_failure": 0.75,
        "compile_failure": 0.85,
    }
    return buckets.get(category, 0.95)


def normalize_channels(raw: Any) -> list[float]:
    if not isinstance(raw, list):
        return [0.0] * CURRENT_CHANNEL_COUNT
    channels = [float(value) for value in raw[:CURRENT_CHANNEL_COUNT]]
    channels.extend([0.0] * (CURRENT_CHANNEL_COUNT - len(channels)))
    return channels


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {error}") from error
            if isinstance(item, dict):
                records.append(item)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
