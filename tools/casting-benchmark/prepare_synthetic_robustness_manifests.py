#!/usr/bin/env python3
"""Generate synthetic post-selection shift manifests for one casting sweep run.

This is a deterministic diagnostic fixture generator, not a validated simulator.
It freezes policy high-fidelity selections, uses a fixed low-fidelity reference
predictor for every candidate, and generates independent-from-search analytic
stress functions after the sweep. No physical-independence claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from multifidelity_compositional_bakeoff import (
    BENCHMARK_ID as SWEEP_BENCHMARK_ID,
    CompositionalScenarioWorld,
    _digest_candidate_pool,
)
from multifidelity_scenario_suite import normalize_fidelity_costs

GENERATOR_BENCHMARK_ID = "CASTING-SYNTHETIC-SHIFT-MANIFEST-GENERATOR-001"
SCHEMA_VERSION = "1.2.0"
REFERENCE_POINT = (1.2, 1.2)
OBJECTIVE_AFFINE_OFFSET = 1.0
OBJECTIVE_AFFINE_SCALE = 1.0 / 3.0

SHIFT_FAMILIES = (
    "zero_shift_control",
    "parameter_drift",
    "constraint_boundary_drift",
    "localized_defect",
    "structural_discontinuity",
    "numerical_discretization",
    "measurement_regime_shift",
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _load(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc


def _exact(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) != expected:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _normalize_objectives(values: tuple[float, float] | list[float], label: str) -> list[float]:
    normalized = [
        (float(value) + OBJECTIVE_AFFINE_OFFSET) * OBJECTIVE_AFFINE_SCALE
        for value in values
    ]
    for index, value in enumerate(normalized):
        if not math.isfinite(value) or value < 0.0 or value >= REFERENCE_POINT[index]:
            raise ValueError(
                f"{label}[{index}] normalizes outside [0, {REFERENCE_POINT[index]}): {value}"
            )
    return normalized


def _shifted_observation(
    truth_objectives: tuple[float, float],
    truth_margin: float,
    x: tuple[float, ...],
    family: str,
    seed: int,
) -> tuple[list[float], float]:
    """Independent stress equations, not the source suite's low-fidelity response."""
    f0, f1 = truth_objectives
    x0, xlast = x[0], x[-1]
    d = len(x)
    margin = truth_margin

    if family == "zero_shift_control":
        objectives = (f0, f1)
    elif family == "parameter_drift":
        objectives = (f0 + 0.07 * (x0 - 0.5), f1 + 0.055 * (xlast - 0.5))
        margin += -0.055 + 0.09 * math.sin(math.pi * x0 + 0.5 * math.pi * xlast)
    elif family == "constraint_boundary_drift":
        objectives = (
            f0 + 0.018 * math.sin(5.0 * math.pi * x0),
            f1 - 0.014 * math.cos(4.0 * math.pi * xlast),
        )
        boundary_weight = math.exp(-abs(truth_margin) / 0.055)
        margin += 0.36 * math.sin(5.0 * math.pi * x0 + 3.0 * math.pi * xlast) * boundary_weight
    elif family == "localized_defect":
        center = tuple(0.72 if i % 2 == 0 else 0.28 for i in range(d))
        dist = sum((x[i] - center[i]) ** 2 for i in range(d)) / d
        bump = math.exp(-dist / (2.0 * 0.14 ** 2))
        objectives = (f0 + 0.38 * bump, f1 - 0.19 * bump)
        margin += -0.42 * bump
    elif family == "structural_discontinuity":
        step = 1.0 if x0 >= 0.53 else -0.25
        objectives = (f0 + 0.22 * step, f1 - 0.17 * step)
        margin += 0.24 if x0 >= 0.53 else -0.16
    elif family == "numerical_discretization":
        wave0 = math.sin(16.0 * math.pi * x0) * math.cos(12.0 * math.pi * xlast)
        wave1 = math.cos(14.0 * math.pi * x0) * math.sin(18.0 * math.pi * xlast)
        objectives = (f0 + 0.026 * wave0, f1 + 0.023 * wave1)
        margin += 0.10 * math.sin(13.0 * math.pi * x0 - 9.0 * math.pi * xlast)
    elif family == "measurement_regime_shift":
        phase = (seed % 997) * 0.013
        boundary_weight = 1.0 + 2.0 * math.exp(-abs(truth_margin) / 0.05)
        edge_weight = 1.0 + 0.8 * (abs(x0 - 0.5) + abs(xlast - 0.5))
        amp = 0.027 * boundary_weight * edge_weight
        objectives = (
            f0 + amp * math.sin(13.0 * x0 + 17.0 * xlast + phase),
            f1 + amp * math.cos(19.0 * x0 - 11.0 * xlast + phase),
        )
        margin += 0.15 * math.sin(11.0 * x0 + 7.0 * xlast + phase) / boundary_weight
    else:
        raise ValueError(f"unsupported shift family: {family}")

    if not all(math.isfinite(value) for value in (*objectives, margin)):
        raise ValueError(f"shift {family} produced non-finite values")
    return _normalize_objectives(objectives, f"shift {family} objectives"), float(margin)


def build_manifests(sweep: Any, sweep_raw: bytes, selector: dict[str, Any], generator_bytes: bytes) -> tuple[dict, dict]:
    sweep = _exact(
        sweep,
        {
            "benchmark_id", "synthetic_fixture_only", "objective_families",
            "constraint_families", "dimensions", "severities", "seeds", "policies",
            "design_count", "cost_budget", "fidelity_costs", "source_sha256",
            "run_count", "paired_runs", "warning",
        },
        "sweep",
    )
    if sweep["benchmark_id"] != SWEEP_BENCHMARK_ID:
        raise ValueError(f"sweep benchmark must be {SWEEP_BENCHMARK_ID}")
    if sweep["synthetic_fixture_only"] is not True:
        raise ValueError("generator accepts synthetic compositional sweeps only")
    if not isinstance(sweep["paired_runs"], list) or sweep["run_count"] != len(sweep["paired_runs"]):
        raise ValueError("sweep run_count does not equal paired_runs length")
    for path, digest in sweep["source_sha256"].items():
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"unsafe source-lock path {path!r}")
        _sha(digest, f"sweep.source_sha256[{path!r}]")
    expected_selector = _exact(
        selector,
        {"objective_family", "constraint_family", "dimension", "severity", "seed", "policy"},
        "selector",
    )
    matches = [
        run for run in sweep["paired_runs"]
        if isinstance(run, dict) and all(run.get(key) == value for key, value in expected_selector.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"selector must match exactly one run, found {len(matches)}")
    run = matches[0]
    design_count = sweep["design_count"]
    if isinstance(design_count, bool) or not isinstance(design_count, int) or design_count < 75:
        raise ValueError("design_count must be an integer >= 75")
    costs = normalize_fidelity_costs(sweep["fidelity_costs"])
    if run.get("fidelity_costs") != costs or run.get("cost_budget") != sweep["cost_budget"]:
        raise ValueError("selected run fidelity costs/budget mismatch sweep")
    if run.get("design_count") != design_count:
        raise ValueError("selected run design_count mismatch")
    world = CompositionalScenarioWorld(
        expected_selector["objective_family"],
        expected_selector["constraint_family"],
        expected_selector["dimension"],
        float(expected_selector["severity"]),
        expected_selector["seed"],
        costs,
        design_count,
    )
    pool_digest = _digest_candidate_pool(world)
    run_pool_digest = _sha(run.get("candidate_pool_sha256"), "run.candidate_pool_sha256")
    if pool_digest != run_pool_digest:
        raise ValueError("reconstructed candidate pool digest does not match source sweep")
    sweep_digest = hashlib.sha256(sweep_raw).hexdigest()
    generator_digest = hashlib.sha256(generator_bytes).hexdigest()

    evaluations = run.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("run.evaluations must be an array")
    high_ids: set[int] = set()
    seen = set()
    for index, row in enumerate(evaluations):
        if not isinstance(row, dict):
            raise ValueError(f"run.evaluations[{index}] must be an object")
        cid, fidelity = row.get("candidate_id"), row.get("fidelity")
        if isinstance(cid, bool) or not isinstance(cid, int) or not 0 <= cid < design_count:
            raise ValueError(f"run.evaluations[{index}] candidate_id is invalid")
        if fidelity not in {"low", "medium", "high"}:
            raise ValueError(f"run.evaluations[{index}] fidelity is invalid")
        if (cid, fidelity) in seen:
            raise ValueError(f"duplicate candidate/fidelity record {(cid, fidelity)}")
        seen.add((cid, fidelity))
        if fidelity == "high":
            high_ids.add(cid)
    if not high_ids:
        raise ValueError("selected policy run has no high-fidelity candidate decisions to audit")

    # The full-pool nominal predictions use a fixed low-fidelity response as
    # an explicit audit reference. They are not misrepresented as the policy's
    # internal posterior or its original acquisition-time prediction.
    predictions = []
    truths = []
    for cid in range(design_count):
        nominal = world._response(cid, "low")
        predictions.append({
            "candidate_id": cid,
            "objectives": _normalize_objectives(nominal.objectives, "nominal prediction objectives"),
            "constraint_margin": float(nominal.constraint_margin),
        })
        truth = world.truth_for_audit_only(cid)
        truths.append((truth.objectives, float(truth.constraint_margin), tuple(world.design(cid).x)))

    selector_out = {
        "objective_family": expected_selector["objective_family"],
        "constraint_family": expected_selector["constraint_family"],
        "dimension": expected_selector["dimension"],
        "severity": float(expected_selector["severity"]),
        "seed": expected_selector["seed"],
        "policy": expected_selector["policy"],
    }
    selection = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "CASTING-DECISION-ROBUSTNESS-ADAPTER-001",
        "source_sweep_sha256": sweep_digest,
        "manifest_generator_sha256": generator_digest,
        "selector": selector_out,
        "candidate_pool_sha256": pool_digest,
        "reference_point": list(REFERENCE_POINT),
        "selection_frozen_before_holdout": True,
        "nominal_prediction_source": "low_fidelity_compositional_reference_not_policy_internal_posterior",
        "selected_candidate_ids": sorted(high_ids),
        "candidate_predictions": predictions,
    }

    shifts = []
    for family in SHIFT_FAMILIES:
        observations = []
        for cid, (objectives, margin, x) in enumerate(truths):
            shifted_objectives, shifted_margin = _shifted_observation(
                objectives, margin, x, family, expected_selector["seed"]
            )
            observations.append({
                "candidate_id": cid,
                "objectives": shifted_objectives,
                "constraint_margin": shifted_margin,
            })
        shifts.append({"shift_id": family, "shift_family": family, "evaluations": observations})
    selection_rendered = json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n"
    selection_manifest_sha = hashlib.sha256(selection_rendered.encode("utf-8")).hexdigest()
    holdout = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "CASTING-DECISION-ROBUSTNESS-ADAPTER-001",
        "source_sweep_sha256": sweep_digest,
        "manifest_generator_sha256": generator_digest,
        "candidate_pool_sha256": pool_digest,
        "selection_manifest_sha256": selection_manifest_sha,
        "selector": selector_out,
        "holdout_role": "synthetic_post_selection_diagnostic",
        "evidence_class": "synthetic",
        "shift_suite_id": GENERATOR_BENCHMARK_ID,
        "shifts": shifts,
    }
    return selection, holdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_json", type=Path)
    parser.add_argument("--objective-family", required=True)
    parser.add_argument("--constraint-family", required=True)
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--severity", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--holdout-output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.sweep_json.read_bytes()
        sweep = _load(raw, "sweep_json")
        generator_bytes = Path(__file__).resolve().read_bytes()
        selection, holdout = build_manifests(
            sweep,
            raw,
            {
                "objective_family": args.objective_family,
                "constraint_family": args.constraint_family,
                "dimension": args.dimension,
                "severity": args.severity,
                "seed": args.seed,
                "policy": args.policy,
            },
            generator_bytes,
        )
        args.selection_output.write_text(
            json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        args.holdout_output.write_text(
            json.dumps(holdout, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"synthetic-shift-manifest: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
