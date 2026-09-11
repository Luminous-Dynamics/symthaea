#!/usr/bin/env python3
"""Compute central and robust Pareto structure from an LL-009F trade-input receipt.

This tool never computes weighted aggregate scores and never selects a transport
architecture. Robust dominance uses declared interval separation only; it does
not infer probabilities from marginal intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

INPUT_RECEIPT_SCHEMA = "ll009f.trade-input-receipt.v1"
NORMALIZED_SCHEMA = "ll009f.normalized-trade-input.v1"
POLICY_SCHEMA = "ll009g.objective-policy.v1"
OUTPUT_SCHEMA = "ll009g.pareto-analysis.v1"

DIRECTIONS = {"minimize", "maximize"}
DOMAINS = {"nonnegative", "fraction", "unbounded"}


class ParetoError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParetoError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ParetoError(f"expected JSON object: {path}")
    return value


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_input_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = load_json(path)
    if receipt.get("schema_version") != INPUT_RECEIPT_SCHEMA:
        raise ParetoError(
            f"input receipt schema_version must be {INPUT_RECEIPT_SCHEMA}"
        )
    if receipt.get("status") != "pass":
        raise ParetoError("LL-009F receipt status is not pass")
    normalized = receipt.get("normalized_trade_input")
    if not isinstance(normalized, dict):
        raise ParetoError("LL-009F receipt missing normalized_trade_input")
    if normalized.get("schema_version") != NORMALIZED_SCHEMA:
        raise ParetoError("unsupported normalized LL-009F schema")
    digest = sha256_bytes(canonical_json_bytes(normalized))
    if receipt.get("normalized_trade_input_sha256") != digest:
        raise ParetoError("LL-009F normalized input hash mismatch")
    candidates = normalized.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ParetoError("normalized input requires at least two candidates")
    if receipt.get("candidate_count") != len(candidates):
        raise ParetoError("LL-009F candidate_count mismatch")
    return receipt, normalized


def validate_policy(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    policy = load_json(path)
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ParetoError(f"objective policy schema_version must be {POLICY_SCHEMA}")
    rules = policy.get("rules")
    if not isinstance(rules, dict):
        raise ParetoError("objective policy missing rules object")
    if rules.get("weighted_scores_forbidden") is not True:
        raise ParetoError("objective policy must explicitly forbid weighted scores")
    if rules.get("robust_dominance") != "interval_separation":
        raise ParetoError("unsupported robust dominance policy")

    objectives = policy.get("objectives")
    if not isinstance(objectives, list) or not objectives:
        raise ParetoError("objective policy requires non-empty objectives")
    normalized: list[dict[str, str]] = []
    names: set[str] = set()
    for obj in objectives:
        if not isinstance(obj, dict):
            raise ParetoError("objective entries must be objects")
        metric = obj.get("metric")
        direction = obj.get("direction")
        domain = obj.get("domain", "unbounded")
        if not isinstance(metric, str) or not metric:
            raise ParetoError("objective metric must be non-empty")
        if metric in names:
            raise ParetoError(f"duplicate objective: {metric}")
        names.add(metric)
        if direction not in DIRECTIONS:
            raise ParetoError(f"{metric}: invalid direction {direction!r}")
        if domain not in DOMAINS:
            raise ParetoError(f"{metric}: invalid domain {domain!r}")
        normalized.append(
            {"metric": metric, "direction": direction, "domain": domain}
        )

    report_only = policy.get("report_only_metrics", [])
    if not isinstance(report_only, list):
        raise ParetoError("report_only_metrics must be a list")
    report_names: set[str] = set()
    for item in report_only:
        if not isinstance(item, dict) or not isinstance(item.get("metric"), str):
            raise ParetoError("report-only metric entries require metric")
        metric = item["metric"]
        if metric in names or metric in report_names:
            raise ParetoError(f"duplicate/objective report-only metric: {metric}")
        report_names.add(metric)

    return policy, normalized


def validate_domain(metric: str, domain: str, value: dict[str, Any]) -> None:
    for field in ("low", "central", "high"):
        number = value.get(field)
        if not finite_number(number):
            raise ParetoError(f"{metric}.{field} must be finite")
        if domain == "nonnegative" and number < 0:
            raise ParetoError(f"{metric}.{field} must be non-negative")
        if domain == "fraction" and not 0.0 <= number <= 1.0:
            raise ParetoError(f"{metric}.{field} must be in [0,1]")
    if not value["low"] <= value["central"] <= value["high"]:
        raise ParetoError(f"{metric}: invalid interval ordering")


def extract_vectors(
    normalized: dict[str, Any],
    objectives: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    candidates = normalized["candidates"]
    objective_names = {obj["metric"] for obj in objectives}
    vectors: dict[str, dict[str, Any]] = {}
    report_only: dict[str, dict[str, Any]] = {}
    units_by_metric: dict[str, str] = {}
    seen: set[str] = set()

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ParetoError("candidate must be object")
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str) or not cid:
            raise ParetoError("candidate missing candidate_id")
        if cid in seen:
            raise ParetoError(f"duplicate candidate_id: {cid}")
        seen.add(cid)
        metrics = candidate.get("metrics")
        if not isinstance(metrics, dict):
            raise ParetoError(f"{cid}: metrics must be object")
        vectors[cid] = {}
        report_only[cid] = {}
        for obj in objectives:
            metric = obj["metric"]
            if metric not in metrics:
                raise ParetoError(f"{cid}: missing objective metric {metric}")
            value = metrics[metric]
            if not isinstance(value, dict):
                raise ParetoError(f"{cid}.{metric}: metric must be object")
            validate_domain(metric, obj["domain"], value)
            unit = value.get("unit")
            if not isinstance(unit, str) or not unit:
                raise ParetoError(f"{cid}.{metric}: missing unit")
            existing = units_by_metric.setdefault(metric, unit)
            if existing != unit:
                raise ParetoError(
                    f"{metric}: inconsistent units across candidates "
                    f"{existing!r} vs {unit!r}"
                )
            vectors[cid][metric] = {
                "low": value["low"],
                "central": value["central"],
                "high": value["high"],
                "unit": unit,
            }

        for metric, value in sorted(metrics.items()):
            if metric not in objective_names:
                report_only[cid][metric] = value

    return vectors, report_only, units_by_metric


def central_dominates(
    a: dict[str, Any],
    b: dict[str, Any],
    objectives: list[dict[str, str]],
) -> bool:
    all_no_worse = True
    any_strict = False
    for obj in objectives:
        metric = obj["metric"]
        av = a[metric]["central"]
        bv = b[metric]["central"]
        if obj["direction"] == "minimize":
            if av > bv:
                all_no_worse = False
                break
            if av < bv:
                any_strict = True
        else:
            if av < bv:
                all_no_worse = False
                break
            if av > bv:
                any_strict = True
    return all_no_worse and any_strict


def robust_dominates(
    a: dict[str, Any],
    b: dict[str, Any],
    objectives: list[dict[str, str]],
) -> bool:
    """Return true only when A's whole interval is no worse than B's whole interval."""
    all_no_worse = True
    any_strict = False
    for obj in objectives:
        metric = obj["metric"]
        if obj["direction"] == "minimize":
            a_worst = a[metric]["high"]
            b_best = b[metric]["low"]
            if a_worst > b_best:
                all_no_worse = False
                break
            if a_worst < b_best:
                any_strict = True
        else:
            a_worst = a[metric]["low"]
            b_best = b[metric]["high"]
            if a_worst < b_best:
                all_no_worse = False
                break
            if a_worst > b_best:
                any_strict = True
    return all_no_worse and any_strict


def dominance_edges(
    vectors: dict[str, dict[str, Any]],
    objectives: list[dict[str, str]],
    predicate: Any,
) -> list[dict[str, str]]:
    ids = sorted(vectors)
    edges: list[dict[str, str]] = []
    for a in ids:
        for b in ids:
            if a == b:
                continue
            if predicate(vectors[a], vectors[b], objectives):
                edges.append({"dominates": a, "dominated": b})
    return edges


def front_from_edges(
    candidate_ids: Iterable[str],
    edges: list[dict[str, str]],
) -> list[str]:
    dominated = {edge["dominated"] for edge in edges}
    return sorted(cid for cid in candidate_ids if cid not in dominated)


def analyze(
    receipt_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    input_receipt, normalized = validate_input_receipt(receipt_path)
    policy, objectives = validate_policy(policy_path)
    vectors, report_only, units = extract_vectors(normalized, objectives)

    central_edges = dominance_edges(vectors, objectives, central_dominates)
    robust_edges = dominance_edges(vectors, objectives, robust_dominates)
    central_edge_pairs = {
        (edge["dominates"], edge["dominated"]) for edge in central_edges
    }
    robust_edge_pairs = {
        (edge["dominates"], edge["dominated"]) for edge in robust_edges
    }
    central_not_robust = [
        {"dominates": a, "dominated": b}
        for a, b in sorted(central_edge_pairs - robust_edge_pairs)
    ]

    central_front = front_from_edges(vectors, central_edges)
    robust_front = front_from_edges(vectors, robust_edges)

    analysis = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "pass",
        "study_id": normalized.get("study_id"),
        "accounting_contract_id": normalized.get("accounting_contract_id"),
        "input_receipt_sha256": sha256_file(receipt_path),
        "normalized_trade_input_sha256": input_receipt[
            "normalized_trade_input_sha256"
        ],
        "objective_policy_sha256": sha256_file(policy_path),
        "objective_policy_name": policy.get("policy_name"),
        "objectives": objectives,
        "units": units,
        "candidate_vectors": vectors,
        "report_only_metrics": report_only,
        "central_dominance_edges": central_edges,
        "central_front": central_front,
        "robust_dominance_edges": robust_edges,
        "robust_front": robust_front,
        "central_not_robust_edges": central_not_robust,
        "front_stability": {
            "on_both_fronts": sorted(set(central_front) & set(robust_front)),
            "central_only_front": sorted(set(central_front) - set(robust_front)),
            "robust_only_front": sorted(set(robust_front) - set(central_front)),
        },
        "uncertainty_semantics": (
            "Robust dominance requires full marginal-interval separation on every "
            "objective. No probability of superiority is inferred because no joint "
            "uncertainty model is declared."
        ),
        "non_claims": [
            "Pareto-front membership is not an engineering selection or launch authorization.",
            "No weighted aggregate score or scalar winner is computed.",
            "The objective policy is explicit and hash-bound; changing it creates a new analysis lineage.",
        ],
    }
    analysis["analysis_sha256"] = sha256_bytes(canonical_json_bytes(analysis))
    return analysis


def synthetic_receipt(path: Path) -> None:
    metrics = [
        "availability_fraction",
        "energy_kwh_per_delivered_kg",
        "expected_lost_cargo_kg_per_year",
        "imported_mass_kg",
        "latency_hours",
        "local_mass_kg",
        "maintenance_import_mass_kg_per_year",
        "peak_power_kw",
        "propellant_reaction_mass_kg_per_year",
        "reliability_success_probability",
        "reusable_hardware_inventory_kg",
        "throughput_capacity_kg_per_year",
    ]

    base = {
        "imported_mass_kg": 100.0,
        "energy_kwh_per_delivered_kg": 10.0,
        "peak_power_kw": 50.0,
        "maintenance_import_mass_kg_per_year": 5.0,
        "propellant_reaction_mass_kg_per_year": 2.0,
        "expected_lost_cargo_kg_per_year": 1.0,
        "reusable_hardware_inventory_kg": 20.0,
        "latency_hours": 2.0,
        "throughput_capacity_kg_per_year": 1000.0,
        "availability_fraction": 0.95,
        "reliability_success_probability": 0.99,
        "local_mass_kg": 200.0,
    }

    def mk(cid: str, scale_cost: float, scale_service: float, width: float) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in metrics:
            c = base[name]
            if name in {
                "throughput_capacity_kg_per_year",
                "availability_fraction",
                "reliability_success_probability",
            }:
                central = c * scale_service
            elif name == "local_mass_kg":
                central = c
            else:
                central = c * scale_cost
            if name in {"availability_fraction", "reliability_success_probability"}:
                low = max(0.0, central - width * 0.01)
                high = min(1.0, central + width * 0.01)
                unit = "dimensionless"
            else:
                low = max(0.0, central * (1.0 - width))
                high = central * (1.0 + width)
                unit = "unit"
            out[name] = {
                "low": low,
                "central": central,
                "high": high,
                "unit": unit,
                "evidence_class": "simulation",
                "source_ref": f"synthetic:{cid}:{name}",
            }
        return {
            "candidate_id": cid,
            "architecture": cid,
            "accounting_contract_id": "accounting-test",
            "components": {},
            "metrics": out,
        }

    a = mk("A", 1.0, 1.0, 0.10)
    b = mk("B", 1.10, 0.90, 0.10)
    d = mk("D", 1.50, 0.70, 0.01)
    c = mk("C", 0.70, 0.65, 0.05)

    normalized = {
        "schema_version": NORMALIZED_SCHEMA,
        "requested_promotion": "pareto_ready_for_real_site_comparison",
        "study_id": "study-test",
        "frame_contract_id": "frame-test",
        "epoch_contract_id": "epoch-test",
        "accounting_contract_id": "accounting-test",
        "accounting_contract": {"synthetic": True},
        "required_components": [],
        "required_metrics": metrics,
        "candidates": [a, b, c, d],
        "ll009e": {"synthetic": True},
        "non_claim": "synthetic",
    }
    digest = sha256_bytes(canonical_json_bytes(normalized))
    receipt = {
        "schema_version": INPUT_RECEIPT_SCHEMA,
        "status": "pass",
        "requested_promotion": normalized["requested_promotion"],
        "study_id": normalized["study_id"],
        "accounting_contract_id": normalized["accounting_contract_id"],
        "candidate_count": 4,
        "normalized_trade_input_sha256": digest,
        "source_manifest_sha256": "0" * 64,
        "ll009e_index_sha256": "1" * 64,
        "ll009e_receipt_sha256": "2" * 64,
        "normalized_trade_input": normalized,
        "non_claim": "synthetic",
    }
    path.write_bytes(canonical_json_bytes(receipt))


def self_test(policy_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ll009g-selftest-") as tmp:
        root = Path(tmp)
        receipt_path = root / "receipt.json"
        synthetic_receipt(receipt_path)
        result = analyze(receipt_path, policy_path)

        assert result["status"] == "pass"
        assert set(result["central_front"]) == {"A", "C"}
        assert "D" not in result["robust_front"]
        assert "B" in result["robust_front"]
        assert {"dominates": "A", "dominated": "B"} in result[
            "central_not_robust_edges"
        ]
        assert {"dominates": "A", "dominated": "D"} in result[
            "robust_dominance_edges"
        ]
        assert "local_mass_kg" in result["report_only_metrics"]["A"]
        serialized = canonical_json_bytes(result)
        assert b'"weighted_score"' not in serialized
        assert b'"winner"' not in serialized

        tampered = load_json(receipt_path)
        tampered["normalized_trade_input"]["candidates"][1]["metrics"][
            "imported_mass_kg"
        ]["unit"] = "different-unit"
        tampered["normalized_trade_input_sha256"] = sha256_bytes(
            canonical_json_bytes(tampered["normalized_trade_input"])
        )
        bad_path = root / "bad-unit.json"
        bad_path.write_bytes(canonical_json_bytes(tampered))
        try:
            analyze(bad_path, policy_path)
        except ParetoError:
            pass
        else:
            raise AssertionError("inconsistent objective units must fail")

        bad_fraction = load_json(receipt_path)
        bad_fraction["normalized_trade_input"]["candidates"][0]["metrics"][
            "availability_fraction"
        ]["high"] = 1.1
        bad_fraction["normalized_trade_input_sha256"] = sha256_bytes(
            canonical_json_bytes(bad_fraction["normalized_trade_input"])
        )
        frac_path = root / "bad-fraction.json"
        frac_path.write_bytes(canonical_json_bytes(bad_fraction))
        try:
            analyze(frac_path, policy_path)
        except ParetoError:
            pass
        else:
            raise AssertionError("fraction outside [0,1] must fail")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/lunar_transport/ll009g_pareto_objectives.json"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test(args.policy)
        print("LL-009G Pareto self-test: PASS")
        return 0
    if args.receipt is None:
        raise ParetoError("--receipt is required unless --self-test is used")
    result = analyze(args.receipt, args.policy)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParetoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
