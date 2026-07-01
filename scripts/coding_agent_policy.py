#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Derive an actionable coding-agent routing policy from quality reports."""

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


def load_json(path: Path) -> dict[str, Any]:
    text = path.read_text().strip()
    start = text.find("{")
    if start < 0:
        raise SystemExit(f"JSON object not found in {path}")
    return json.loads(text[start:])


def policy_from_reports(coding: dict[str, Any], forecast: dict[str, Any]) -> dict[str, Any]:
    quality = as_float(coding.get("quality_pass_rate"), as_float(coding.get("pass_rate")))
    pass_rate = as_float(coding.get("pass_rate"))
    broca_score = as_float(coding.get("broca_selection_score"))
    mean_attempts = as_float(coding.get("mean_attempts_per_task"), 99.0)
    forecast_trust = as_float(forecast.get("router_trust_multiplier"), 0.5)
    brier = as_float(forecast.get("brier_score"), 1.0)
    ece = as_float(forecast.get("expected_calibration_error"), 1.0)

    execution_score = 0.50 * quality + 0.25 * pass_rate + 0.25 * broca_score
    efficiency_score = max(0.0, min(1.0, 1.0 / max(mean_attempts, 1.0)))
    epistemic_score = max(0.0, min(1.0, forecast_trust * (1.0 - min(brier + ece, 1.0))))
    composite = 0.55 * execution_score + 0.15 * efficiency_score + 0.30 * epistemic_score

    if quality >= 0.95 and forecast_trust >= 1.10 and composite >= 0.85:
        action = "promote"
    elif quality >= 0.90 and forecast_trust >= 1.0 and composite >= 0.75:
        action = "hold"
    elif quality >= 0.75 and forecast_trust >= 0.75:
        action = "caution"
    else:
        action = "demote"

    return {
        "policy": "coding_agent_backend_routing",
        "action": action,
        "composite_score": round(composite, 6),
        "execution_score": round(execution_score, 6),
        "efficiency_score": round(efficiency_score, 6),
        "epistemic_score": round(epistemic_score, 6),
        "recommended_backend_order": backend_order(action),
        "repair_mode": repair_mode(action, mean_attempts),
        "human_review_required": action in {"caution", "demote"},
        "signals": {
            "coding_quality_pass_rate": quality,
            "coding_pass_rate": pass_rate,
            "broca_selection_score": broca_score,
            "mean_attempts_per_task": mean_attempts,
            "forecast_calibration_signal": forecast.get("calibration_signal"),
            "forecast_router_trust_multiplier": forecast_trust,
            "forecast_brier_score": brier,
            "forecast_expected_calibration_error": ece,
            "forecast_resolved_count": as_int(forecast.get("resolved_count")),
        },
    }


def backend_order(action: str) -> list[str]:
    if action == "promote":
        return ["geodesic", "broca", "native", "analogy", "deepswe", "llm"]
    if action == "hold":
        return ["geodesic", "native", "broca", "analogy", "deepswe", "llm"]
    if action == "caution":
        return ["native", "geodesic", "analogy", "deepswe", "broca", "llm"]
    return ["native", "analogy", "deepswe", "llm", "geodesic", "broca"]


def repair_mode(action: str, mean_attempts: float) -> str:
    if action == "promote" and mean_attempts <= 1.2:
        return "minimal"
    if action in {"promote", "hold"}:
        return "standard"
    if action == "caution":
        return "aggressive_with_extra_verification"
    return "fallback_only"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coding-report", required=True, type=Path)
    parser.add_argument("--forecast-report", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    policy = policy_from_reports(load_json(args.coding_report), load_json(args.forecast_report))
    text = json.dumps(policy, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    if args.json or not args.output:
        print(text, end="")
    else:
        print(f"policy written: {args.output}")
        print(
            "action={action} composite={score:.3f} order={order}".format(
                action=policy["action"],
                score=policy["composite_score"],
                order=",".join(policy["recommended_backend_order"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
