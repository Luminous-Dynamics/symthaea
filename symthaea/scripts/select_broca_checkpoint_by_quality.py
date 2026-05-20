#!/usr/bin/env python3
"""Select the strongest Broca checkpoint from canonical quality reports.

The selector prefers code-sheaf function coherence when present, then falls
back to case-level code-sheaf coherence, gated coherence, target overlap, and
finally lower perplexity. It is intentionally report-only: callers decide
whether to copy, publish, or train from the selected checkpoint.

Optional coding-backend and repair-memory reports are treated as publication
signals. They can gate selection entirely and are folded into the displayed
score, so a checkpoint is not promoted from language metrics alone when the
code synthesis loop has regressed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def coding_signal(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {
            "available": False,
            "pass_rate": 0.0,
            "quality_pass_rate": 0.0,
            "broca_selection_score": 0.0,
            "broca_eval_gate_passed": None,
            "repair_success_rate": 0.0,
            "repair_memory_success_rate": 0.0,
            "repair_memory_hits": 0,
        }

    return {
        "available": True,
        "benchmark": report.get("benchmark"),
        "task_count": as_int(report.get("task_count")),
        "pass_rate": as_float(report.get("pass_rate")),
        "quality_pass_rate": as_float(
            report.get("quality_pass_rate"), as_float(report.get("pass_rate"))
        ),
        "broca_selection_score": as_float(report.get("broca_selection_score")),
        "broca_eval_gate_passed": report.get("broca_eval_gate_passed"),
        "repair_success_rate": as_float(report.get("repair_success_rate")),
        "repair_memory_success_rate": as_float(report.get("repair_memory_success_rate")),
        "repair_memory_hits": as_int(report.get("repair_memory_hits")),
    }


def repair_ab_signal(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {
            "available": False,
            "attempt_delta": 0.0,
            "memory_hits": 0,
            "memory_success_rate": 0.0,
            "memory_helped_tasks": 0,
            "memory_hurt_tasks": 0,
        }

    return {
        "available": True,
        "attempt_delta": as_float(report.get("attempt_delta")),
        "repair_success_delta": as_float(report.get("repair_success_delta")),
        "memory_hits": as_int(report.get("memory_hits")),
        "memory_success_rate": as_float(report.get("memory_success_rate")),
        "memory_helped_tasks": as_int(report.get("memory_helped_tasks")),
        "memory_hurt_tasks": as_int(report.get("memory_hurt_tasks")),
        "helped_task_ids": report.get("helped_task_ids") or [],
        "hurt_task_ids": report.get("hurt_task_ids") or [],
    }


def quality_key(
    report: dict[str, Any],
    coding: dict[str, Any],
    repair_ab: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    code_sheaf = report.get("code_sheaf") or {}
    gated_sheaf = code_sheaf.get("gated") or {}
    gated = report.get("gated_generation") or {}
    function_coherence = as_float(gated_sheaf.get("function_coherence_rate"))
    case_coherence = as_float(gated_sheaf.get("coherence_rate"))
    avg_coherence = as_float(gated.get("avg_coherence"))
    target_overlap = as_float(gated.get("target_token_overlap"))
    perplexity = as_float(gated.get("perplexity"), 1.0e12)
    coding_score = as_float(coding.get("broca_selection_score"))
    coding_quality = as_float(coding.get("quality_pass_rate"))
    repair_success = as_float(coding.get("repair_success_rate"))
    memory_success = max(
        as_float(coding.get("repair_memory_success_rate")),
        as_float(repair_ab.get("memory_success_rate")),
    )
    memory_help = max(0.0, -as_float(repair_ab.get("attempt_delta")))
    return (
        coding_score,
        coding_quality,
        repair_success,
        memory_success,
        memory_help,
        function_coherence,
        case_coherence,
        avg_coherence,
        target_overlap,
        -perplexity,
    )


def checkpoint_path(report: dict[str, Any], fallback: Path) -> str:
    metadata = report.get("metadata") or {}
    return metadata.get("checkpoint_path") or str(fallback)


def resolve_report_path(
    reports: list[Path],
    explicit: Path | None,
    *,
    needle: str,
) -> Path | None:
    if explicit is not None:
        return explicit
    matches = [path for path in reports if needle in path.name.lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def validate_gates(
    coding: dict[str, Any],
    repair_ab: dict[str, Any],
    reports: list[dict[str, Any]],
    *,
    require_code_signal: bool,
    require_coding_gate: bool,
    require_coding_eval_gate: bool,
    min_coding_score: float,
    min_quality_pass_rate: float,
    require_repair_memory_gate: bool,
    min_memory_hits: int,
    min_memory_success_rate: float,
    max_memory_hurt_tasks: int,
) -> list[str]:
    failures: list[str] = []
    has_report_code_signal = any(bool(report.get("code_sheaf")) for report in reports)
    if require_code_signal and not coding.get("available") and not has_report_code_signal:
        failures.append(
            "--require-code-signal was set but no coding report or code-sheaf eval data was provided"
        )

    if require_coding_gate:
        if not coding.get("available"):
            failures.append("--require-coding-gate was set but no --coding-report was provided")
        if as_float(coding.get("broca_selection_score")) < min_coding_score:
            failures.append(
                "coding broca_selection_score="
                f"{coding.get('broca_selection_score')} below {min_coding_score}"
            )
        if as_float(coding.get("quality_pass_rate")) < min_quality_pass_rate:
            failures.append(
                "coding quality_pass_rate="
                f"{coding.get('quality_pass_rate')} below {min_quality_pass_rate}"
            )
        if require_coding_eval_gate and coding.get("broca_eval_gate_passed") is False:
            failures.append("coding broca_eval_gate_passed=false")

    if require_repair_memory_gate:
        if not repair_ab.get("available"):
            failures.append(
                "--require-repair-memory-gate was set but no --repair-memory-ab-summary was provided"
            )
        if as_int(repair_ab.get("memory_hits")) < min_memory_hits:
            failures.append(
                f"repair memory hits={repair_ab.get('memory_hits')} below {min_memory_hits}"
            )
        if as_float(repair_ab.get("memory_success_rate")) < min_memory_success_rate:
            failures.append(
                "repair memory success_rate="
                f"{repair_ab.get('memory_success_rate')} below {min_memory_success_rate}"
            )
        if as_int(repair_ab.get("memory_hurt_tasks")) > max_memory_hurt_tasks:
            failures.append(
                "repair memory hurt_tasks="
                f"{repair_ab.get('memory_hurt_tasks')} above {max_memory_hurt_tasks}"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable selection")
    parser.add_argument("--coding-report", type=Path)
    parser.add_argument("--repair-memory-ab-summary", type=Path)
    parser.add_argument("--require-code-signal", action="store_true")
    parser.add_argument("--require-coding-gate", action="store_true")
    parser.add_argument("--require-coding-eval-gate", action="store_true")
    parser.add_argument("--min-coding-score", type=float, default=0.0)
    parser.add_argument("--min-quality-pass-rate", type=float, default=0.0)
    parser.add_argument("--require-repair-memory-gate", action="store_true")
    parser.add_argument("--min-memory-hits", type=int, default=0)
    parser.add_argument("--min-memory-success-rate", type=float, default=0.0)
    parser.add_argument("--max-memory-hurt-tasks", type=int, default=0)
    parser.add_argument(
        "--require-trained-improvement",
        action="store_true",
        help="Require the trained report to outrank the baseline report before selection can pass",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Baseline quality report used with --require-trained-improvement",
    )
    parser.add_argument(
        "--trained-report",
        type=Path,
        help="Trained quality report used with --require-trained-improvement",
    )
    args = parser.parse_args()

    coding = coding_signal(load_optional_json(args.coding_report))
    repair_ab = repair_ab_signal(load_optional_json(args.repair_memory_ab_summary))
    reports = [json.loads(path.read_text()) for path in args.reports]
    failures = validate_gates(
        coding,
        repair_ab,
        reports,
        require_code_signal=args.require_code_signal,
        require_coding_gate=args.require_coding_gate,
        require_coding_eval_gate=args.require_coding_eval_gate,
        min_coding_score=args.min_coding_score,
        min_quality_pass_rate=args.min_quality_pass_rate,
        require_repair_memory_gate=args.require_repair_memory_gate,
        min_memory_hits=args.min_memory_hits,
        min_memory_success_rate=args.min_memory_success_rate,
        max_memory_hurt_tasks=args.max_memory_hurt_tasks,
    )
    if failures:
        for failure in failures:
            print(f"selection gate failed: {failure}")
        return 1

    ranked: list[
        tuple[tuple[float, float, float, float, float, float, float, float, float, float], Path, dict[str, Any]]
    ] = []
    for path, report in zip(args.reports, reports, strict=True):
        ranked.append((quality_key(report, coding, repair_ab), path, report))

    if args.require_trained_improvement:
        baseline_path = resolve_report_path(
            args.reports, args.baseline_report, needle="baseline"
        )
        trained_path = resolve_report_path(args.reports, args.trained_report, needle="trained")
        if baseline_path is None or trained_path is None:
            print(
                "selection gate failed: --require-trained-improvement needs "
                "--baseline-report and --trained-report, or uniquely named baseline/trained inputs"
            )
            return 1

        keys_by_path = {path.resolve(): key for key, path, _report in ranked}
        baseline_key = keys_by_path.get(baseline_path.resolve())
        trained_key = keys_by_path.get(trained_path.resolve())
        if baseline_key is None or trained_key is None:
            print(
                "selection gate failed: baseline/trained reports must also be present in positional reports"
            )
            return 1
        if trained_key <= baseline_key:
            print(
                "selection gate failed: trained report did not improve on baseline "
                f"(trained={trained_key}, baseline={baseline_key})"
            )
            return 1

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_key, best_report_path, best_report = ranked[0]
    selection = {
        "selected_report": str(best_report_path),
        "selected_checkpoint": checkpoint_path(best_report, best_report_path),
        "score": {
            "coding_broca_selection_score": best_key[0],
            "coding_quality_pass_rate": best_key[1],
            "coding_repair_success_rate": best_key[2],
            "repair_memory_success_rate": best_key[3],
            "repair_memory_attempt_delta_gain": best_key[4],
            "function_coherence_rate": best_key[5],
            "code_sheaf_coherence_rate": best_key[6],
            "avg_coherence": best_key[7],
            "target_token_overlap": best_key[8],
            "negative_perplexity": best_key[9],
        },
        "coding_signal": coding,
        "repair_memory_ab_signal": repair_ab,
        "ranked_reports": [
            {
                "report": str(path),
                "checkpoint": checkpoint_path(report, path),
                "score": {
                    "coding_broca_selection_score": key[0],
                    "coding_quality_pass_rate": key[1],
                    "coding_repair_success_rate": key[2],
                    "repair_memory_success_rate": key[3],
                    "repair_memory_attempt_delta_gain": key[4],
                    "function_coherence_rate": key[5],
                    "code_sheaf_coherence_rate": key[6],
                    "avg_coherence": key[7],
                    "target_token_overlap": key[8],
                    "negative_perplexity": key[9],
                },
            }
            for key, path, report in ranked
        ],
    }

    if args.json:
        print(json.dumps(selection, indent=2))
    else:
        print(f"selected checkpoint: {selection['selected_checkpoint']}")
        print(f"selected report:     {selection['selected_report']}")
        print(
            "score: "
            f"coding={best_key[0]:.3f} "
            f"quality={best_key[1]:.3f} "
            f"repair={best_key[2]:.3f} "
            f"memory={best_key[3]:.3f} "
            f"function_coherence={best_key[5]:.3f} "
            f"sheaf={best_key[6]:.3f} "
            f"coherence={best_key[7]:.3f} "
            f"target_overlap={best_key[8]:.3f} "
            f"perplexity={-best_key[9]:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
