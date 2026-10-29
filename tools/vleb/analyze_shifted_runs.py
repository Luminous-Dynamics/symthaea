#!/usr/bin/env python3
"""Analyze matched clean/shifted coding-agent continuations from pinned starting states.

This is a fail-closed, analysis-only layer. It does not execute shifts, authenticate
receipts, or prove that the evaluator actually applied the declared intervention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

HEX = re.compile(r"^[a-f0-9]{64}$")
REVISION = re.compile(r"^[a-f0-9]{40}$")
IMAGE = re.compile(r"^sha256:[a-f0-9]{64}$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
NETWORK_MODES = {"none", "allowlisted", "enabled", "not_applicable"}
OUTCOMES = ("pass", "fail", "unresolved")
CONDITIONS = ("clean", "shifted")
TOP_KEYS = {"schema_version", "protocol_sha256", "cohort_id", "scenarios", "seeds", "bootstrap", "runs"}
SCENARIO_KEYS = {
    "scenario_id", "task_family", "change_family", "intervention_position",
    "task_contract_before_sha256", "task_contract_after_sha256",
    "shift_event_sha256", "initial_state_sha256", "initial_checkpoint_sha256",
    "initial_revision", "initial_image_id",
}
BOOTSTRAP_KEYS = {"replicates", "seed", "confidence_level"}
RUN_KEYS = {
    "scenario_id", "seed", "condition", "outcome", "run_id", "attempt_number",
    "scenario_sha256", "protocol_sha256", "initial_state_sha256",
    "initial_checkpoint_sha256", "initial_revision", "initial_image_id",
    "model_id", "model_version", "scaffold_sha256",
    "initial_execution_profile_sha256", "post_shift_execution_profile_sha256",
    "initial_network_mode", "post_shift_network_mode", "final_state_sha256",
    "receipt_sha256", "audit_report_sha256", "cost_usd", "duration_ms",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def scenario_digest(scenario: dict[str, Any]) -> str:
    return digest(canonical_json_bytes(scenario))


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} rejected")


def strict_json_loads(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{label} keys invalid: missing={missing}, unexpected={extra}")


def require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def require_label(value: Any, label: str) -> None:
    if not isinstance(value, str) or not LABEL.fullmatch(value):
        raise ValueError(f"{label} must match {LABEL.pattern!r}")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_manifest(manifest: Any, expected_protocol_sha256: str) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    exact_keys(manifest, TOP_KEYS, "manifest")
    if manifest["schema_version"] != "vleb-shift-comparison-v1":
        raise ValueError("unsupported shift-comparison manifest schema_version")
    require_digest(manifest["protocol_sha256"], "manifest protocol_sha256")
    require_digest(expected_protocol_sha256, "externally expected protocol_sha256")
    if manifest["protocol_sha256"] != expected_protocol_sha256:
        raise ValueError("manifest protocol differs from evaluator-supplied protocol pin")
    if not isinstance(manifest["cohort_id"], str) or not manifest["cohort_id"].strip():
        raise ValueError("cohort_id must be a non-empty string")

    scenarios = manifest["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) < 2:
        raise ValueError("at least two locked scenarios are required")
    scenario_by_id: dict[str, dict[str, Any]] = {}
    scenario_hash_by_id: dict[str, str] = {}
    for index, scenario in enumerate(scenarios):
        label = f"scenarios[{index}]"
        if not isinstance(scenario, dict):
            raise ValueError(f"{label} must be an object")
        exact_keys(scenario, SCENARIO_KEYS, label)
        scenario_id = scenario["scenario_id"]
        if not isinstance(scenario_id, str) or not IDENTIFIER.fullmatch(scenario_id):
            raise ValueError(f"{label}.scenario_id is invalid")
        if scenario_id in scenario_by_id:
            raise ValueError("scenario IDs must be unique")
        require_label(scenario["task_family"], label + ".task_family")
        require_label(scenario["change_family"], label + ".change_family")
        if type(scenario["intervention_position"]) is not int or not 1 <= scenario["intervention_position"] <= 9:
            raise ValueError(f"{label}.intervention_position must be an integer in [1, 9]")
        for key in (
            "task_contract_before_sha256", "task_contract_after_sha256", "shift_event_sha256",
            "initial_state_sha256", "initial_checkpoint_sha256",
        ):
            require_digest(scenario[key], label + "." + key)
        if not isinstance(scenario["initial_revision"], str) or not REVISION.fullmatch(scenario["initial_revision"]):
            raise ValueError(f"{label}.initial_revision must be a lowercase 40-character commit SHA")
        if not isinstance(scenario["initial_image_id"], str) or not IMAGE.fullmatch(scenario["initial_image_id"]):
            raise ValueError(f"{label}.initial_image_id must be an immutable sha256 image reference")
        scenario_by_id[scenario_id] = scenario
        scenario_hash_by_id[scenario_id] = scenario_digest(scenario)

    seeds = manifest["seeds"]
    if (
        not isinstance(seeds, list) or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("seeds must be a non-empty list of unique non-negative integers")

    bootstrap = manifest["bootstrap"]
    if not isinstance(bootstrap, dict):
        raise ValueError("bootstrap must be an object")
    exact_keys(bootstrap, BOOTSTRAP_KEYS, "bootstrap")
    if type(bootstrap["replicates"]) is not int or not 1000 <= bootstrap["replicates"] <= 50000:
        raise ValueError("bootstrap replicates must be an integer in [1000, 50000]")
    if type(bootstrap["seed"]) is not int or bootstrap["seed"] < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    if not is_number(bootstrap["confidence_level"]) or float(bootstrap["confidence_level"]) != 0.95:
        raise ValueError("protocol v1 requires a preregistered 0.95 confidence level")

    runs = manifest["runs"]
    expected_rows = len(scenarios) * len(seeds) * 2
    if not isinstance(runs, list) or len(runs) != expected_rows:
        observed = len(runs) if isinstance(runs, list) else "non-list"
        raise ValueError(
            f"runs must contain the complete locked scenario × seed × condition cohort: "
            f"expected {expected_rows}, observed {observed}"
        )

    scenario_ids = set(scenario_by_id)
    seed_set = set(seeds)
    observed_cells: set[tuple[str, int, str]] = set()
    run_ids: set[str] = set()
    receipt_hashes: set[str] = set()
    audit_hashes: set[str] = set()
    rows_by_cell: dict[tuple[str, int, str], dict[str, Any]] = {}
    rows_by_scenario: dict[str, list[dict[str, Any]]] = {scenario_id: [] for scenario_id in scenario_by_id}
    common_by_pair: dict[tuple[str, int], tuple[Any, ...]] = {}
    configuration_by_scenario: dict[str, tuple[Any, ...]] = {}
    post_shift_configuration_by_scenario: dict[str, tuple[Any, ...]] = {}

    for index, row in enumerate(runs):
        label = f"runs[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be an object")
        exact_keys(row, RUN_KEYS, label)
        scenario_id, seed, condition = row["scenario_id"], row["seed"], row["condition"]
        if (
            not isinstance(scenario_id, str) or scenario_id not in scenario_ids
            or type(seed) is not int or seed not in seed_set
            or not isinstance(condition, str) or condition not in CONDITIONS
        ):
            raise ValueError(f"{label} is outside the locked scenario/seed/condition inventory")
        cell = (scenario_id, seed, condition)
        if cell in observed_cells:
            raise ValueError(f"duplicate run cell {cell!r}")
        observed_cells.add(cell)
        scenario = scenario_by_id[scenario_id]
        if row["scenario_sha256"] != scenario_hash_by_id[scenario_id]:
            raise ValueError(f"{label}.scenario_sha256 differs from the canonical locked scenario")
        if row["protocol_sha256"] != expected_protocol_sha256:
            raise ValueError(f"{label}.protocol_sha256 differs from the evaluator-pinned protocol")
        if row["outcome"] not in OUTCOMES:
            raise ValueError(f"{label}.outcome must be pass, fail, or unresolved")
        if not isinstance(row["run_id"], str) or not IDENTIFIER.fullmatch(row["run_id"]):
            raise ValueError(f"{label}.run_id is invalid")
        if row["run_id"] in run_ids:
            raise ValueError("run_id must be unique for every scheduled trial")
        run_ids.add(row["run_id"])
        if type(row["attempt_number"]) is not int or row["attempt_number"] != 1:
            raise ValueError(f"{label}.attempt_number must be 1; retries cannot replace the preregistered trial")

        for key in (
            "initial_state_sha256", "initial_checkpoint_sha256", "scaffold_sha256",
            "initial_execution_profile_sha256", "final_state_sha256",
            "receipt_sha256", "audit_report_sha256",
        ):
            require_digest(row[key], label + "." + key)
        if row["initial_state_sha256"] != scenario["initial_state_sha256"]:
            raise ValueError(f"{label} initial state differs from its locked scenario")
        if row["initial_checkpoint_sha256"] != scenario["initial_checkpoint_sha256"]:
            raise ValueError(f"{label} initial checkpoint differs from its locked scenario")
        if row["initial_revision"] != scenario["initial_revision"]:
            raise ValueError(f"{label} initial revision differs from its locked scenario")
        if row["initial_image_id"] != scenario["initial_image_id"]:
            raise ValueError(f"{label} initial image differs from its locked scenario")
        if row["receipt_sha256"] in receipt_hashes:
            raise ValueError("receipt digest replay detected across shift-comparison trials")
        receipt_hashes.add(row["receipt_sha256"])
        if row["audit_report_sha256"] in audit_hashes:
            raise ValueError("audit-report digest replay detected across shift-comparison trials")
        audit_hashes.add(row["audit_report_sha256"])

        for key in ("model_id", "model_version"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"{label}.{key} must be a non-empty string")
        for key in ("scaffold_sha256", "initial_execution_profile_sha256"):
            require_digest(row[key], label + "." + key)
        if not isinstance(row["initial_network_mode"], str) or row["initial_network_mode"] not in NETWORK_MODES:
            raise ValueError(f"{label}.initial_network_mode must be a valid network-mode string")
        if row["condition"] == "clean":
            if row["post_shift_execution_profile_sha256"] is not None or row["post_shift_network_mode"] is not None:
                raise ValueError(f"{label} clean condition must not claim a post-shift profile or network mode")
        else:
            require_digest(row["post_shift_execution_profile_sha256"], label + ".post_shift_execution_profile_sha256")
            if not isinstance(row["post_shift_network_mode"], str) or row["post_shift_network_mode"] not in NETWORK_MODES:
                raise ValueError(f"{label}.post_shift_network_mode must be a valid network-mode string")
        if not is_number(row["cost_usd"]) or row["cost_usd"] < 0:
            raise ValueError(f"{label}.cost_usd must be finite and non-negative")
        if type(row["duration_ms"]) is not int or row["duration_ms"] < 0:
            raise ValueError(f"{label}.duration_ms must be a non-negative integer")

        paired_key = (scenario_id, seed)
        common = (
            row["scenario_sha256"], row["protocol_sha256"], row["initial_state_sha256"],
            row["initial_checkpoint_sha256"], row["initial_revision"], row["initial_image_id"],
            row["model_id"], row["model_version"], row["scaffold_sha256"],
            row["initial_execution_profile_sha256"], row["initial_network_mode"],
        )
        previous = common_by_pair.setdefault(paired_key, common)
        if previous != common:
            raise ValueError(f"clean and shifted continuations do not share the same pinned starting identity for {paired_key!r}")
        scenario_configuration = (
            row["model_id"], row["model_version"], row["scaffold_sha256"],
            row["initial_execution_profile_sha256"], row["initial_network_mode"],
        )
        previous_configuration = configuration_by_scenario.setdefault(scenario_id, scenario_configuration)
        if previous_configuration != scenario_configuration:
            raise ValueError(f"scenario {scenario_id!r} drifts in model/scaffold/initial profile across seeds")
        if condition == "shifted":
            post_shift_configuration = (
                row["post_shift_execution_profile_sha256"], row["post_shift_network_mode"]
            )
            previous_post_shift = post_shift_configuration_by_scenario.setdefault(scenario_id, post_shift_configuration)
            if previous_post_shift != post_shift_configuration:
                raise ValueError(f"scenario {scenario_id!r} drifts in post-shift profile/network mode across seeds")
        rows_by_cell[cell] = row
        rows_by_scenario[scenario_id].append(row)

    expected_cells = {
        (scenario_id, seed, condition)
        for scenario_id in scenario_ids for seed in seed_set for condition in CONDITIONS
    }
    if observed_cells != expected_cells:
        raise ValueError("one or more locked clean/shifted condition cells are missing")

    paired_rows = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        for seed in seeds:
            clean = rows_by_cell[(scenario_id, seed, "clean")]
            shifted = rows_by_cell[(scenario_id, seed, "shifted")]
            paired_rows.append({
                "scenario_id": scenario_id,
                "seed": seed,
                "task_family": scenario["task_family"],
                "change_family": scenario["change_family"],
                "intervention_position": scenario["intervention_position"],
                "clean_outcome": clean["outcome"],
                "shifted_outcome": shifted["outcome"],
                "clean_cost_usd": float(clean["cost_usd"]),
                "shifted_cost_usd": float(shifted["cost_usd"]),
                "clean_duration_ms": clean["duration_ms"],
                "shifted_duration_ms": shifted["duration_ms"],
            })
    paired_rows_by_scenario: dict[str, list[dict[str, Any]]] = {
        scenario_id: [] for scenario_id in scenario_by_id
    }
    for paired_row in paired_rows:
        paired_rows_by_scenario[paired_row["scenario_id"]].append(paired_row)
    return {
        "scenario_by_id": scenario_by_id,
        "scenario_ids": [scenario["scenario_id"] for scenario in scenarios],
        "seeds": seeds,
        "bootstrap": bootstrap,
        "rows_by_scenario": paired_rows_by_scenario,
        "paired_rows": paired_rows,
        "protocol_sha256": manifest["protocol_sha256"],
    }


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty data")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def raw_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty shift cohort")
    count = len(rows)
    clean_pass = sum(row["clean_outcome"] == "pass" for row in rows)
    shifted_pass = sum(row["shifted_outcome"] == "pass" for row in rows)
    both_pass = sum(row["clean_outcome"] == "pass" and row["shifted_outcome"] == "pass" for row in rows)
    clean_only_fail = sum(row["clean_outcome"] == "pass" and row["shifted_outcome"] == "fail" for row in rows)
    clean_only_unresolved = sum(row["clean_outcome"] == "pass" and row["shifted_outcome"] == "unresolved" for row in rows)
    shifted_only_pass = sum(row["clean_outcome"] != "pass" and row["shifted_outcome"] == "pass" for row in rows)
    unresolved_pairs = sum(
        row["clean_outcome"] == "unresolved" or row["shifted_outcome"] == "unresolved" for row in rows
    )
    transition = {
        f"clean_{clean}__shifted_{shifted}": sum(
            row["clean_outcome"] == clean and row["shifted_outcome"] == shifted for row in rows
        )
        for clean in OUTCOMES for shifted in OUTCOMES
    }
    clean_rate = 100.0 * clean_pass / count
    shifted_rate = 100.0 * shifted_pass / count
    clean_cost = sum(row["clean_cost_usd"] for row in rows)
    shifted_cost = sum(row["shifted_cost_usd"] for row in rows)
    clean_durations = sorted(row["clean_duration_ms"] for row in rows)
    shifted_durations = sorted(row["shifted_duration_ms"] for row in rows)

    def nearest_rank(values: list[int], p: float) -> int:
        return values[int(math.ceil(p * len(values))) - 1]

    return {
        "scheduled_pair_count": count,
        "clean_success_count": clean_pass,
        "shifted_success_count": shifted_pass,
        "paired_success_count": both_pass,
        "clean_success_rate_pct": round(clean_rate, 6),
        "shifted_success_rate_pct": round(shifted_rate, 6),
        "shift_degradation_pp": round(clean_rate - shifted_rate, 6),
        "paired_success_rate_pct": round(100.0 * both_pass / count, 6),
        "robustness_given_clean_success_pct": round(100.0 * both_pass / clean_pass, 6) if clean_pass else None,
        "clean_pass_shifted_fail_count": clean_only_fail,
        "clean_pass_shifted_unresolved_count": clean_only_unresolved,
        "clean_nonpass_shifted_pass_count": shifted_only_pass,
        "unresolved_pair_count": unresolved_pairs,
        "clean_unresolved_count": sum(row["clean_outcome"] == "unresolved" for row in rows),
        "shifted_unresolved_count": sum(row["shifted_outcome"] == "unresolved" for row in rows),
        "transition_matrix": transition,
        "clean_total_cost_usd": round(clean_cost, 8),
        "shifted_total_cost_usd": round(shifted_cost, 8),
        "clean_cost_per_scheduled_pair_usd": round(clean_cost / count, 8),
        "shifted_cost_per_scheduled_pair_usd": round(shifted_cost / count, 8),
        "clean_p50_duration_ms": nearest_rank(clean_durations, 0.50),
        "clean_p90_duration_ms": nearest_rank(clean_durations, 0.90),
        "shifted_p50_duration_ms": nearest_rank(shifted_durations, 0.50),
        "shifted_p90_duration_ms": nearest_rank(shifted_durations, 0.90),
        "duration_percentile_method": "nearest_rank",
    }


def confidence_interval(samples: list[float], level: float) -> dict[str, float]:
    ordered = sorted(samples)
    alpha = 1.0 - level
    return {
        "lower": round(percentile(ordered, alpha / 2.0), 6),
        "upper": round(percentile(ordered, 1.0 - alpha / 2.0), 6),
    }


def summarize_group(
    scenario_ids: list[str],
    rows_by_scenario: dict[str, list[dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    rows = [row for scenario_id in scenario_ids for row in rows_by_scenario[scenario_id]]
    summary = raw_metrics(rows)
    rng = random.Random(seed)
    sample_names = ("clean_success_rate_pct", "shifted_success_rate_pct", "shift_degradation_pp", "paired_success_rate_pct")
    samples = {name: [] for name in sample_names}
    for _ in range(replicates):
        drawn_scenarios = [scenario_ids[rng.randrange(len(scenario_ids))] for _ in scenario_ids]
        drawn_rows = [row for scenario_id in drawn_scenarios for row in rows_by_scenario[scenario_id]]
        values = raw_metrics(drawn_rows)
        for name in sample_names:
            samples[name].append(float(values[name]))
    summary["scenario_count"] = len(scenario_ids)
    summary["scenario_cluster_bootstrap_95ci"] = {
        name: confidence_interval(values, confidence_level) for name, values in samples.items()
    }
    summary["interval_method"] = "percentile bootstrap over scenario clusters; all seeds and both conditions travel together"
    summary["intervals_are_simultaneous"] = False
    return summary


def analyze_manifest(
    manifest: Any,
    *,
    input_sha256: str,
    expected_protocol_sha256: str,
    expected_analyzer_sha256: str,
) -> dict[str, Any]:
    require_digest(input_sha256, "input_sha256")
    require_digest(expected_analyzer_sha256, "externally expected analyzer SHA-256")
    observed_analyzer_sha256 = digest(Path(__file__).read_bytes())
    if observed_analyzer_sha256 != expected_analyzer_sha256:
        raise ValueError("analyzer source differs from evaluator-supplied SHA-256")
    validated = validate_manifest(manifest, expected_protocol_sha256)
    scenarios: dict[str, dict[str, Any]] = validated["scenario_by_id"]
    scenario_ids: list[str] = validated["scenario_ids"]
    rows_by_scenario: dict[str, list[dict[str, Any]]] = validated["rows_by_scenario"]
    boot: dict[str, Any] = validated["bootstrap"]
    replicates, bootstrap_seed = boot["replicates"], boot["seed"]

    overall = summarize_group(
        scenario_ids, rows_by_scenario, replicates=replicates, seed=bootstrap_seed,
        confidence_level=float(boot["confidence_level"]),
    )

    def grouped(key: str) -> dict[str, Any]:
        group_values = sorted({scenario[key] for scenario in scenarios.values()})
        output = {}
        for ordinal, value in enumerate(group_values):
            group_ids = [scenario_id for scenario_id in scenario_ids if scenarios[scenario_id][key] == value]
            output[str(value)] = summarize_group(
                group_ids, rows_by_scenario, replicates=replicates,
                seed=bootstrap_seed + ordinal + 1, confidence_level=float(boot["confidence_level"]),
            )
        return output

    by_change_family = grouped("change_family")
    by_task_family = grouped("task_family")
    by_intervention_position = grouped("intervention_position")
    warnings = []
    if len(scenario_ids) < 30:
        warnings.append("fewer than 30 independent scenarios; overall intervals may be unstable")
    if len(validated["seeds"]) < 2:
        warnings.append("single declared seed; seed variability cannot be estimated")
    for dimension, groups in (
        ("change_family", by_change_family),
        ("task_family", by_task_family),
        ("intervention_position", by_intervention_position),
    ):
        if any(result["scenario_count"] < 10 for result in groups.values()):
            warnings.append(f"one or more {dimension} groups have fewer than 10 scenarios; subgroup intervals are exploratory")

    return {
        "schema_version": "vleb-shift-comparison-report-v1",
        "valid": True,
        "cohort_id": manifest["cohort_id"],
        "input_manifest_sha256": input_sha256,
        "protocol_sha256": validated["protocol_sha256"],
        "analyzer_sha256": observed_analyzer_sha256,
        "scenario_count": len(scenario_ids),
        "seed_count_per_scenario": len(validated["seeds"]),
        "expected_paired_run_count": len(scenario_ids) * len(validated["seeds"]),
        "observed_run_count": len(validated["paired_rows"]) * 2,
        "bootstrap": {
            "method": "paired clean/shifted scenario-cluster percentile bootstrap",
            "confidence_level": boot["confidence_level"],
            "replicates": replicates,
            "seed": bootstrap_seed,
            "cluster_unit": "scenario_id; every resample retains all preregistered seeds and both conditions",
        },
        "overall": overall,
        "by_change_family": by_change_family,
        "by_task_family": by_task_family,
        "by_intervention_position": by_intervention_position,
        "warnings": warnings,
        "analysis_only": True,
        "score_eligible": False,
        "limitations": [
            "The analyzer checks identity, pairing, cohort coverage and summary calculations; it does not authenticate receipt/audit bytes or prove that the declared shift was actually applied.",
            "The trusted evaluator must verify that the shifted continuation received the pinned intervention and that pass requires current requirements, preservation of still-valid work and fresh evidence.",
            "Unresolved outcomes remain in the scheduled denominator and count as non-passes.",
            "Bootstrap intervals are descriptive and non-simultaneous across dimensions or groups; subgroup intervals are exploratory when scenario counts are small.",
            "Scenario-family labels and intervention positions must be frozen before candidate outcomes are observed.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-analyzer-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        observed_input_sha256 = digest(raw)
        require_digest(args.expected_input_sha256, "expected input SHA-256")
        if observed_input_sha256 != args.expected_input_sha256:
            raise ValueError("input manifest differs from externally supplied SHA-256")
        manifest = strict_json_loads(raw)
        result = analyze_manifest(
            manifest, input_sha256=observed_input_sha256,
            expected_protocol_sha256=args.expected_protocol_sha256,
            expected_analyzer_sha256=args.expected_analyzer_sha256,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": "vleb-shift-comparison-report-v1",
            "valid": False, "analysis_only": True, "score_eligible": False,
            "issues": [{"code": "shift_comparison_rejected", "message": str(exc)}],
        }
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
