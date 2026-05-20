#!/usr/bin/env python3
"""Compare coding backend repair-lane reports with and without repair memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-memory", required=True, type=Path)
    parser.add_argument("--with-memory", required=True, type=Path)
    parser.add_argument("--min-memory-hits", type=int, default=1)
    parser.add_argument("--min-memory-helped-tasks", type=int, default=0)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--require-no-pass-regression", action="store_true")
    parser.add_argument("--require-no-attempt-regression", action="store_true")
    args = parser.parse_args()

    without = load_json(args.without_memory.read_text())
    with_memory = load_json(args.with_memory.read_text())
    failures: list[str] = []

    memory_hits = int(with_memory.get("repair_memory_hits") or 0)
    if memory_hits < args.min_memory_hits:
        failures.append(
            f"repair_memory_hits={memory_hits}, below minimum {args.min_memory_hits}"
        )

    if int(without.get("repair_memory_hits") or 0) != 0:
        failures.append(
            "without-memory report unexpectedly used repair memory: "
            f"{without.get('repair_memory_hits')}"
        )

    task_delta = compare_tasks(without, with_memory)
    if task_delta["memory_helped_tasks"] < args.min_memory_helped_tasks:
        failures.append(
            "memory_helped_tasks="
            f"{task_delta['memory_helped_tasks']}, below minimum "
            f"{args.min_memory_helped_tasks}"
        )

    if args.require_no_pass_regression:
        compare_min(with_memory, without, "pass_rate", failures)
        compare_min(with_memory, without, "quality_pass_rate", failures)
        compare_min(with_memory, without, "repair_success_rate", failures)

    if args.require_no_attempt_regression:
        compare_max(with_memory, without, "mean_attempts_per_task", failures)

    summary = {
        "attempt_delta": (
            (with_memory.get("mean_attempts_per_task") or 0.0)
            - (without.get("mean_attempts_per_task") or 0.0)
        ),
        "repair_success_delta": (
            (with_memory.get("repair_success_rate") or 0.0)
            - (without.get("repair_success_rate") or 0.0)
        ),
        "memory_hits": memory_hits,
        "memory_success_rate": with_memory.get("repair_memory_success_rate"),
        **task_delta,
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")

    if failures:
        print("repair-memory A/B regression detected:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        "repair-memory A/B OK: "
        f"memory_hits={memory_hits} "
        f"memory_success_rate={with_memory.get('repair_memory_success_rate')} "
        f"pass_without={without.get('pass_rate')} "
        f"pass_with={with_memory.get('pass_rate')} "
        f"attempts_without={without.get('mean_attempts_per_task')} "
        f"attempts_with={with_memory.get('mean_attempts_per_task')} "
        f"attempt_delta={summary['attempt_delta']} "
        f"memory_helped_tasks={summary['memory_helped_tasks']} "
        f"memory_hurt_tasks={summary['memory_hurt_tasks']}"
    )
    return 0


def load_json(text: str) -> dict[str, Any]:
    trimmed = text.strip()
    start = trimmed.find("{")
    if start < 0:
        raise SystemExit("JSON object not found in report")
    return json.loads(trimmed[start:])


def compare_min(
    actual_report: dict[str, Any],
    baseline_report: dict[str, Any],
    key: str,
    failures: list[str],
) -> None:
    actual = actual_report.get(key)
    baseline = baseline_report.get(key)
    if actual is None or baseline is None:
        failures.append(f"{key} missing from one report")
    elif actual < baseline:
        failures.append(f"{key} regressed from {baseline!r} to {actual!r}")


def compare_max(
    actual_report: dict[str, Any],
    baseline_report: dict[str, Any],
    key: str,
    failures: list[str],
) -> None:
    actual = actual_report.get(key)
    baseline = baseline_report.get(key)
    if actual is None or baseline is None:
        failures.append(f"{key} missing from one report")
    elif actual > baseline:
        failures.append(f"{key} regressed from {baseline!r} to {actual!r}")


def compare_tasks(
    without_report: dict[str, Any], with_report: dict[str, Any]
) -> dict[str, Any]:
    without_tasks = {
        task.get("id"): task
        for task in without_report.get("tasks", [])
        if isinstance(task, dict)
    }
    with_tasks = {
        task.get("id"): task
        for task in with_report.get("tasks", [])
        if isinstance(task, dict)
    }
    helped: list[str] = []
    hurt: list[str] = []
    unchanged: list[str] = []

    for task_id, before in sorted(without_tasks.items()):
        if not task_id or task_id not in with_tasks:
            continue
        after = with_tasks[task_id]
        before_attempts = int(before.get("attempt_count") or 0)
        after_attempts = int(after.get("attempt_count") or 0)
        before_accepted = bool(before.get("accepted"))
        after_accepted = bool(after.get("accepted"))
        used_memory = any(
            str(label).startswith("repair_memory_")
            for label in after.get("repair_prior_labels_seen", [])
        )

        if used_memory and after_accepted and (
            not before_accepted or after_attempts < before_attempts
        ):
            helped.append(str(task_id))
        elif (before_accepted and not after_accepted) or after_attempts > before_attempts:
            hurt.append(str(task_id))
        else:
            unchanged.append(str(task_id))

    return {
        "memory_helped_tasks": len(helped),
        "memory_hurt_tasks": len(hurt),
        "memory_unchanged_tasks": len(unchanged),
        "helped_task_ids": helped,
        "hurt_task_ids": hurt,
    }


if __name__ == "__main__":
    raise SystemExit(main())
