#!/usr/bin/env python3
"""Convert a frozen compositional bake-off run + sealed shift manifests to audit input.

The adapter never creates shifted observations and never infers model predictions
from audit truth. Those must be provided by separately frozen artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from decision_robustness_eval import BENCHMARK_ID as AUDIT_BENCHMARK_ID

ADAPTER_BENCHMARK_ID = "CASTING-DECISION-ROBUSTNESS-ADAPTER-001"
ADAPTER_SCHEMA_VERSION = "1.2.0"
EXPECTED_SWEEP_BENCHMARK = "CASTING-MF-COMPOSITION-001"
REFERENCE_TOL = 1e-12
FROZEN_REFERENCE_POINT = (1.2, 1.2)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _selector(value: Any, label: str) -> dict[str, Any]:
    selector = _exact(
        value,
        {"objective_family", "constraint_family", "dimension", "severity", "seed", "policy"},
        label,
    )
    for key in ("objective_family", "constraint_family", "policy"):
        if not isinstance(selector[key], str) or not selector[key].strip():
            raise ValueError(f"{label}.{key} must be a non-empty string")
    for key in ("dimension", "seed"):
        if isinstance(selector[key], bool) or not isinstance(selector[key], int) or selector[key] < 0:
            raise ValueError(f"{label}.{key} must be a non-negative integer")
    if selector["dimension"] < 2:
        raise ValueError(f"{label}.dimension must be at least 2")
    if _number(selector["severity"], f"{label}.severity") < 0:
        raise ValueError(f"{label}.severity must be non-negative")
    selector["severity"] = float(selector["severity"])
    return selector


def _matches(run: dict[str, Any], selector: dict[str, Any]) -> bool:
    return all(run.get(key) == value for key, value in selector.items())


def _objective_pair(value: Any, label: str, reference: tuple[float, float]) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two objective values")
    result = [_number(value[i], f"{label}[{i}]") for i in range(2)]
    if any(v < 0.0 or v >= reference[i] for i, v in enumerate(result)):
        raise ValueError(f"{label} must lie in [0, reference_point)")
    return result


def _candidate_rows(
    rows: Any,
    label: str,
    design_count: int,
    reference: tuple[float, float],
    *,
    prediction: bool,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be an array")
    seen: dict[int, dict[str, Any]] = {}
    keys = {"candidate_id", "objectives", "constraint_margin"}
    for index, raw in enumerate(rows):
        row = _exact(raw, keys, f"{label}[{index}]")
        cid = row["candidate_id"]
        if isinstance(cid, bool) or not isinstance(cid, int) or cid < 0 or cid >= design_count:
            raise ValueError(f"{label}[{index}].candidate_id must be in [0, {design_count})")
        if cid in seen:
            raise ValueError(f"{label} contains duplicate candidate_id {cid}")
        parsed = {
            "candidate_id": cid,
            "objectives": _objective_pair(row["objectives"], f"{label}[{index}].objectives", reference),
            "constraint_margin": _number(row["constraint_margin"], f"{label}[{index}].constraint_margin"),
        }
        seen[cid] = parsed
    expected = set(range(design_count))
    if set(seen) != expected:
        raise ValueError(
            f"{label} must cover every candidate exactly once; "
            f"missing={sorted(expected - set(seen))[:12]}, extra={sorted(set(seen) - expected)[:12]}"
        )
    return [seen[cid] for cid in range(design_count)]


def adapt_documents(
    sweep: Any,
    sweep_raw: bytes,
    selection: Any,
    selection_raw: bytes,
    holdout: Any,
    holdout_raw: bytes,
) -> dict[str, Any]:
    sweep_sha = hashlib.sha256(sweep_raw).hexdigest()
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
    if sweep["benchmark_id"] != EXPECTED_SWEEP_BENCHMARK:
        raise ValueError(f"sweep.benchmark_id must be {EXPECTED_SWEEP_BENCHMARK}")
    if sweep["synthetic_fixture_only"] is not True:
        raise ValueError("sweep must be explicitly labelled synthetic_fixture_only")
    if isinstance(sweep["design_count"], bool) or not isinstance(sweep["design_count"], int) or sweep["design_count"] < 75:
        raise ValueError("sweep.design_count must be an integer >= 75")
    if not isinstance(sweep["paired_runs"], list) or not sweep["paired_runs"]:
        raise ValueError("sweep.paired_runs must be a non-empty array")
    if sweep["run_count"] != len(sweep["paired_runs"]):
        raise ValueError("sweep.run_count does not match paired_runs length")
    if not isinstance(sweep["source_sha256"], dict) or not sweep["source_sha256"]:
        raise ValueError("sweep.source_sha256 must be a non-empty path-to-digest object")
    for path, digest in sweep["source_sha256"].items():
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"invalid source-lock path: {path!r}")
        _digest(digest, f"sweep.source_sha256[{path!r}]")
    sweep_costs = _exact(sweep["fidelity_costs"], {"low", "medium", "high"}, "sweep.fidelity_costs")
    for level, value in sweep_costs.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"sweep.fidelity_costs.{level} must be a positive integer")
    sweep_budget = _number(sweep["cost_budget"], "sweep.cost_budget")
    if sweep_budget <= 0:
        raise ValueError("sweep.cost_budget must be positive")
    design_count = sweep["design_count"]

    selection = _exact(
        selection,
        {
            "schema_version", "benchmark_id", "source_sweep_sha256", "selector",
            "candidate_pool_sha256", "reference_point", "selected_candidate_ids",
            "candidate_predictions", "selection_frozen_before_holdout",
            "manifest_generator_sha256", "nominal_prediction_source",
            "nominal_prediction_evaluations", "nominal_prediction_fidelity",
            "nominal_prediction_cost_units",
        },
        "selection_manifest",
    )
    if selection["schema_version"] != ADAPTER_SCHEMA_VERSION or selection["benchmark_id"] != ADAPTER_BENCHMARK_ID:
        raise ValueError("selection_manifest schema or benchmark id is unsupported")
    if selection["selection_frozen_before_holdout"] is not True:
        raise ValueError("selection_manifest must declare selection_frozen_before_holdout=true")
    generator_sha = _digest(selection["manifest_generator_sha256"], "selection_manifest.manifest_generator_sha256")
    nominal_prediction_source = selection["nominal_prediction_source"]
    if not isinstance(nominal_prediction_source, str) or not nominal_prediction_source.strip():
        raise ValueError("selection_manifest.nominal_prediction_source must be non-empty")
    if selection["nominal_prediction_evaluations"] != design_count:
        raise ValueError("nominal_prediction_evaluations must equal the full candidate pool size")
    if selection["nominal_prediction_fidelity"] != "low":
        raise ValueError("this synthetic audit currently requires the frozen low-fidelity reference")
    expected_prediction_cost = design_count * sweep_costs["low"]
    if selection["nominal_prediction_cost_units"] != expected_prediction_cost:
        raise ValueError("nominal prediction cost accounting is inconsistent with pool size and low-fidelity cost")
    if _digest(selection["source_sweep_sha256"], "selection_manifest.source_sweep_sha256") != sweep_sha:
        raise ValueError("selection_manifest is not bound to these exact sweep bytes")
    selector = _selector(selection["selector"], "selection_manifest.selector")

    matching = [run for run in sweep["paired_runs"] if isinstance(run, dict) and _matches(run, selector)]
    if len(matching) != 1:
        raise ValueError(f"selector must match exactly one sweep run, found {len(matching)}")
    run = matching[0]
    if run.get("design_count") != design_count:
        raise ValueError("selected run design_count disagrees with sweep")
    if _number(run.get("severity"), "selected run severity") != selector["severity"]:
        raise ValueError("selected run severity disagrees with selector")
    if _number(run.get("cost_budget"), "selected run cost_budget") != sweep_budget:
        raise ValueError("selected run budget disagrees with sweep")
    run_costs = _exact(run.get("fidelity_costs"), {"low", "medium", "high"}, "selected run fidelity_costs")
    if run_costs != sweep_costs:
        raise ValueError("selected run fidelity costs disagree with sweep")
    candidate_pool_sha = _digest(run.get("candidate_pool_sha256"), "selected run candidate_pool_sha256")
    if _digest(selection["candidate_pool_sha256"], "selection_manifest.candidate_pool_sha256") != candidate_pool_sha:
        raise ValueError("selection candidate pool digest does not match selected run")
    reference_raw = selection["reference_point"]
    if not isinstance(reference_raw, list) or len(reference_raw) != 2:
        raise ValueError("selection_manifest.reference_point must have exactly two values")
    reference = tuple(_number(value, f"selection_manifest.reference_point[{i}]") for i, value in enumerate(reference_raw))
    if any(value <= 0 for value in reference):
        raise ValueError("reference point values must be positive")
    if any(abs(reference[i] - FROZEN_REFERENCE_POINT[i]) > REFERENCE_TOL for i in range(2)):
        raise ValueError(f"reference_point must equal the frozen benchmark point {FROZEN_REFERENCE_POINT}")

    selected_ids = selection["selected_candidate_ids"]
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError("selected_candidate_ids must be a non-empty array")
    if any(isinstance(cid, bool) or not isinstance(cid, int) or cid < 0 or cid >= design_count for cid in selected_ids):
        raise ValueError("selected_candidate_ids must be valid pool candidate IDs")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected_candidate_ids contains duplicates")
    predictions = _candidate_rows(
        selection["candidate_predictions"], "selection_manifest.candidate_predictions",
        design_count, reference, prediction=True,
    )

    holdout = _exact(
        holdout,
        {
            "schema_version", "benchmark_id", "source_sweep_sha256",
            "candidate_pool_sha256", "selector", "shifts", "holdout_role",
            "manifest_generator_sha256", "evidence_class", "shift_suite_id",
            "selection_manifest_sha256", "shift_evaluation_count", "shift_cost_accounting",
        },
        "holdout_manifest",
    )
    if holdout["schema_version"] != ADAPTER_SCHEMA_VERSION or holdout["benchmark_id"] != ADAPTER_BENCHMARK_ID:
        raise ValueError("holdout_manifest schema or benchmark id is unsupported")
    if holdout["holdout_role"] not in {"independent_post_selection", "synthetic_post_selection_diagnostic"}:
        raise ValueError("holdout_manifest.holdout_role is not a supported frozen role")
    if holdout["evidence_class"] != "synthetic" or holdout["holdout_role"] != "synthetic_post_selection_diagnostic":
        raise ValueError("this adapter accepts synthetic diagnostics only; do not label generated shifts as independent physical validation")
    if not isinstance(holdout["shift_suite_id"], str) or not holdout["shift_suite_id"].strip():
        raise ValueError("holdout_manifest.shift_suite_id must be non-empty")
    if holdout["shift_cost_accounting"] != "synthetic_analytic_not_priced":
        raise ValueError("synthetic shift observations must not be represented as priced simulator calls")
    if isinstance(holdout["shift_evaluation_count"], bool) or not isinstance(holdout["shift_evaluation_count"], int):
        raise ValueError("shift_evaluation_count must be an integer")
    if _digest(holdout["manifest_generator_sha256"], "holdout_manifest.manifest_generator_sha256") != generator_sha:
        raise ValueError("selection and holdout manifests must use the same generator digest")
    if _digest(holdout["selection_manifest_sha256"], "holdout_manifest.selection_manifest_sha256") != hashlib.sha256(selection_raw).hexdigest():
        raise ValueError("holdout manifest is not bound to these exact selection manifest bytes")
    if _digest(holdout["source_sweep_sha256"], "holdout_manifest.source_sweep_sha256") != sweep_sha:
        raise ValueError("holdout manifest is not bound to these exact sweep bytes")
    if _digest(holdout["candidate_pool_sha256"], "holdout_manifest.candidate_pool_sha256") != candidate_pool_sha:
        raise ValueError("holdout candidate pool digest does not match selected run")
    if _selector(holdout["selector"], "holdout_manifest.selector") != selector:
        raise ValueError("holdout selector does not match selection manifest")
    if not isinstance(holdout["shifts"], list) or not holdout["shifts"]:
        raise ValueError("holdout_manifest.shifts must be a non-empty array")
    shifts = []
    seen_shift_ids = set()
    for i, shift in enumerate(holdout["shifts"]):
        shift = _exact(shift, {"shift_id", "shift_family", "evaluations"}, f"holdout_manifest.shifts[{i}]")
        for key in ("shift_id", "shift_family"):
            if not isinstance(shift[key], str) or not shift[key].strip():
                raise ValueError(f"holdout_manifest.shifts[{i}].{key} must be non-empty")
        if shift["shift_id"] in seen_shift_ids:
            raise ValueError(f"duplicate shift_id {shift['shift_id']}")
        seen_shift_ids.add(shift["shift_id"])
        observations = _candidate_rows(
            shift["evaluations"], f"holdout {shift['shift_id']} evaluations",
            design_count, reference, prediction=False,
        )
        shifts.append({
            "shift_id": shift["shift_id"],
            "shift_family": shift["shift_family"],
            "evaluations": [
                {"candidate_id": row["candidate_id"], "objectives": row["objectives"], "constraint_margin": row["constraint_margin"]}
                for row in observations
            ],
        })

    if holdout["shift_evaluation_count"] != design_count * len(shifts):
        raise ValueError("shift_evaluation_count does not match the complete candidate-by-shift matrix")
    evaluations = run.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("selected sweep run has no evaluation ledger")
    seen_eval_keys = set()
    high_candidate_ids = set()
    summed_cost = 0
    for i, observation in enumerate(evaluations):
        if not isinstance(observation, dict):
            raise ValueError(f"selected run evaluations[{i}] must be an object")
        cid, fidelity = observation.get("candidate_id"), observation.get("fidelity")
        if isinstance(cid, bool) or not isinstance(cid, int) or cid < 0 or cid >= design_count:
            raise ValueError(f"selected run evaluations[{i}] has invalid candidate_id")
        if fidelity not in {"low", "medium", "high"}:
            raise ValueError(f"selected run evaluations[{i}] has invalid fidelity")
        key = (cid, fidelity)
        if key in seen_eval_keys:
            raise ValueError(f"selected run has duplicate candidate/fidelity observation {key}")
        seen_eval_keys.add(key)
        if fidelity == "high":
            high_candidate_ids.add(cid)
        raw_objectives = observation.get("objectives")
        if not isinstance(raw_objectives, list) or len(raw_objectives) != 2:
            raise ValueError(f"selected run evaluation {i}.objectives must contain two values")
        for j, value in enumerate(raw_objectives):
            _number(value, f"selected run evaluation {i}.objectives[{j}]")
        _number(observation.get("constraint_margin"), f"selected run evaluation {i}.constraint_margin")
        cost_units = observation.get("cost_units")
        if isinstance(cost_units, bool) or not isinstance(cost_units, int) or cost_units != run_costs[fidelity]:
            raise ValueError(f"selected run evaluation {i}.cost_units disagrees with its fidelity cost")
        summed_cost += cost_units
    if set(selected_ids) != high_candidate_ids:
        raise ValueError("selected_candidate_ids must equal exactly the run's high-fidelity candidate IDs")
    declared_spent = _number(run.get("cost_spent"), "selected run cost_spent")
    if abs(summed_cost - declared_spent) > REFERENCE_TOL:
        raise ValueError("selected run cost_spent disagrees with its evaluation ledger")
    if declared_spent > sweep_budget + REFERENCE_TOL:
        raise ValueError("selected run cost_spent exceeds the declared budget")
    fidelity_costs = run_costs
    return {
        "schema_version": "1.2.0",
        "benchmark_id": AUDIT_BENCHMARK_ID,
        "task_id": f'{selector["objective_family"]}__{selector["constraint_family"]}__d{selector["dimension"]}__s{selector["severity"]}',
        "run_id": f'{selector["policy"]}__seed{selector["seed"]}__budget{run.get("cost_budget")}',
        "source_run": {
            "sweep_sha256": sweep_sha,
            "adapter_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
            "manifest_generator_sha256": generator_sha,
            "candidate_pool_sha256": candidate_pool_sha,
            "nominal_prediction_source": nominal_prediction_source,
            "evidence_class": holdout["evidence_class"],
            "audit_accounting": {
                "nominal_prediction_evaluations": selection["nominal_prediction_evaluations"],
                "nominal_prediction_fidelity": selection["nominal_prediction_fidelity"],
                "nominal_prediction_cost_units": selection["nominal_prediction_cost_units"],
                "shift_evaluation_count": holdout["shift_evaluation_count"],
                "shift_cost_accounting": holdout["shift_cost_accounting"],
            },
            "objective_family": selector["objective_family"],
            "constraint_family": selector["constraint_family"],
            "dimension": selector["dimension"],
            "severity": selector["severity"],
            "seed": selector["seed"],
            "policy": selector["policy"],
            "fidelity_costs": fidelity_costs,
            "cost_budget": run.get("cost_budget"),
            "cost_spent": run.get("cost_spent"),
        },
        "split": {
            "selection_frozen_before_holdout": selection["selection_frozen_before_holdout"],
            "holdout_role": holdout["holdout_role"],
            "selection_freeze_sha256": hashlib.sha256(selection_raw).hexdigest(),
            "holdout_manifest_sha256": hashlib.sha256(holdout_raw).hexdigest(),
        },
        "reference_point": list(reference),
        "candidate_set": [
            {
                "candidate_id": str(row["candidate_id"]),
                "predicted_objectives": row["objectives"],
                "predicted_constraint_margin": row["constraint_margin"],
                "selected": row["candidate_id"] in selected_ids,
            }
            for row in predictions
        ],
        "holdout_shifts": shifts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_json", type=Path, help="unaltered runner output JSON")
    parser.add_argument("selection_manifest", type=Path, help="frozen selected IDs and full-pool nominal predictions")
    parser.add_argument("holdout_manifest", type=Path, help="sealed shifted high-fidelity observations for the full candidate pool")
    parser.add_argument("--output", type=Path, help="write converted audit input (default stdout)")
    args = parser.parse_args(argv)
    try:
        sweep_raw = args.sweep_json.read_bytes()
        selection_raw = args.selection_manifest.read_bytes()
        holdout_raw = args.holdout_manifest.read_bytes()
        converted = adapt_documents(
            _load(sweep_raw, "sweep_json"), sweep_raw,
            _load(selection_raw, "selection_manifest"), selection_raw,
            _load(holdout_raw, "holdout_manifest"), holdout_raw,
        )
        rendered = json.dumps(converted, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, TypeError) as exc:
        print(f"decision-robustness-adapter: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
