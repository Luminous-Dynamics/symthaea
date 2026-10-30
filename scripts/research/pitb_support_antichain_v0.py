#!/usr/bin/env python3
"""Fail-closed finite support-antichain analysis for PITB research."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

OUTCOME_SCHEMA = "pitb.support-ablation-outcome.v0"
PREDICTION_SCHEMA = "pitb.support-antichain-prediction.v0"
REPORT_SCHEMA = "pitb.support-antichain-report.v0"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY = (
    "study_id", "component_id", "target_id", "analysis_endpoint",
    "target_statement_sha256", "environment_sha256", "toolchain_sha256",
    "fixed_context_sha256", "budget_sha256", "solver_policy_sha256",
)
RECEIPTS = (
    "state_receipt_sha256", "run_receipt_sha256",
    "checker_receipt_sha256", "semantic_gate_receipt_sha256",
)
OUTCOME_KEYS = set(IDENTITY) | set(RECEIPTS) | {
    "schema_version", "mutable_declarations", "available_declarations",
    "local_context_sha256", "outcome", "kernel_passed", "semantic_gate_passed",
}
PREDICTION_KEYS = {
    "schema_version", "group_id", "model_id", "model_version", "abstained",
    "claimed_necessary_declarations", "proposed_sufficient_route",
    "confidence", "run_receipt_sha256",
}


class ContractError(ValueError):
    pass


def digest(value, where):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ContractError(f"{where} must be lowercase SHA-256 hex")
    return value


def strings(value, where):
    if (not isinstance(value, list) or
            any(not isinstance(x, str) or not x.strip() for x in value) or
            value != sorted(value) or len(value) != len(set(value))):
        raise ContractError(f"{where} must be sorted unique non-empty strings")
    return value


def load_jsonl(path, label):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ContractError(f"{label} line {n}: {exc}") from exc
    if not rows:
        raise ContractError(f"{label} must contain at least one row")
    return rows, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def powerset(values):
    return {
        tuple(c)
        for n in range(len(values) + 1)
        for c in itertools.combinations(values, n)
    }


def validate_outcome(row, n):
    if not isinstance(row, dict) or set(row) != OUTCOME_KEYS:
        raise ContractError(f"outcome row {n} has missing or unsupported keys")
    if row["schema_version"] != OUTCOME_SCHEMA:
        raise ContractError(f"outcome row {n} has unsupported schema")
    for key in IDENTITY:
        if key.endswith("_sha256"):
            digest(row[key], f"outcome row {n}.{key}")
        elif not isinstance(row[key], str) or not row[key].strip():
            raise ContractError(f"outcome row {n}.{key} must be non-empty")
    if row["analysis_endpoint"] != "exact_target_replan":
        raise ContractError("v0 only accepts exact_target_replan")
    for key in RECEIPTS:
        digest(row[key], f"outcome row {n}.{key}")
    universe = strings(row["mutable_declarations"], f"row {n}.mutable_declarations")
    available = strings(row["available_declarations"], f"row {n}.available_declarations")
    if len(universe) > 16:
        raise ContractError("exact exhaustive lattice is capped at 16 declarations")
    if not set(available) <= set(universe):
        raise ContractError(f"row {n} contains a declaration outside its mutable universe")
    digest(row["local_context_sha256"], f"row {n}.local_context_sha256")
    if row["outcome"] not in ("verified_success", "non_solve"):
        raise ContractError(f"row {n} has unresolved or unsupported outcome")
    if type(row["kernel_passed"]) is not bool or type(row["semantic_gate_passed"]) is not bool:
        raise ContractError(f"row {n} proof gates must be booleans")
    passed = row["kernel_passed"] and row["semantic_gate_passed"]
    if (row["outcome"] == "verified_success") != passed:
        raise ContractError(f"row {n} outcome contradicts the kernel/semantic gates")
    return row


def group_key(row):
    return hashlib.sha256(json.dumps(
        {k: row[k] for k in IDENTITY}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def analyze_group(gid, rows):
    first = rows[0]
    universe = first["mutable_declarations"]
    expected = powerset(universe)
    seen = {}
    contexts = set()
    for row in rows:
        if any(row[k] != first[k] for k in IDENTITY):
            raise ContractError(f"group {gid} mixes target, context, budget, environment or policy")
        if row["mutable_declarations"] != universe:
            raise ContractError(f"group {gid} changes its mutable declaration universe")
        subset = tuple(row["available_declarations"])
        if subset not in expected:
            raise ContractError(f"group {gid} has an invalid ablation subset {subset}")
        if subset in seen:
            raise ContractError(f"group {gid} duplicates ablation subset {subset}")
        if row["local_context_sha256"] in contexts:
            raise ContractError(f"group {gid} reuses a local-context digest across distinct subsets")
        contexts.add(row["local_context_sha256"])
        seen[subset] = row
    missing = expected - set(seen)
    extra = set(seen) - expected
    if missing or extra:
        raise ContractError(
            f"group {gid} is not exhaustive: missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}"
        )
    full = tuple(universe)
    if seen[full]["outcome"] != "verified_success":
        raise ContractError(f"group {gid} full-context baseline is not a verified success")
    successful = {s for s, r in seen.items() if r["outcome"] == "verified_success"}
    minimal = sorted(
        (s for s in successful if not any(set(t) < set(s) for t in successful)),
        key=lambda s: (len(s), s),
    )
    if not minimal:
        raise ContractError(f"group {gid} has no inclusion-minimal successful route")
    necessary = sorted(set.intersection(*(set(s) for s in minimal))) if minimal else []
    possible = sorted(set.union(*(set(s) for s in minimal))) if minimal else []
    failed_supersets = [
        s for s in seen if seen[s]["outcome"] == "non_solve"
        and any(set(route) < set(s) for route in successful)
    ]
    return {
        "group_id": gid, "study_id": first["study_id"],
        "component_id": first["component_id"], "target_id": first["target_id"],
        "analysis_endpoint": first["analysis_endpoint"],
        "target_statement_sha256": first["target_statement_sha256"],
        "environment_sha256": first["environment_sha256"],
        "toolchain_sha256": first["toolchain_sha256"],
        "fixed_context_sha256": first["fixed_context_sha256"],
        "budget_sha256": first["budget_sha256"],
        "solver_policy_sha256": first["solver_policy_sha256"],
        "mutable_declaration_universe": universe,
        "subset_lattice_size": len(expected),
        "verified_success_subset_count": len(successful),
        "non_solve_subset_count": len(expected) - len(successful),
        "full_context_verified_success": True,
        "minimal_sufficient_support_antichain": [list(s) for s in minimal],
        "minimal_support_set_count": len(minimal),
        "necessary_across_all_observed_minimal_routes": necessary,
        "possible_across_observed_minimal_routes": possible,
        "declaration_roles": [
            {"declaration": d, "in_all_minimal_support_sets": d in necessary,
             "in_some_minimal_support_sets": d in possible,
             "minimal_support_set_frequency": sum(d in s for s in minimal) / len(minimal)}
            for d in universe
        ],
        "failed_supersets_of_verified_success_count": len(failed_supersets),
        "observed_success_monotonicity": (
            "violations_present" if failed_supersets else "no_violations_on_exhaustive_lattice"
        ),
        "interpretation_boundary": (
            "Finite, policy-, budget-, toolchain-, target- and context-specific minimal sufficient "
            "routes; not logical necessity, proof-term support, or a global theorem dependency certificate."
        ),
    }


def score_predictions(predictions, groups):
    by_id = {g["group_id"]: g for g in groups}
    pred_by_id = {}
    for n, p in enumerate(predictions, 1):
        if not isinstance(p, dict) or set(p) != PREDICTION_KEYS:
            raise ContractError(f"prediction row {n} has missing or unsupported keys")
        if p["schema_version"] != PREDICTION_SCHEMA:
            raise ContractError(f"prediction row {n} has unsupported schema")
        gid = digest(p["group_id"], f"prediction {n}.group_id")
        if gid not in by_id or gid in pred_by_id:
            raise ContractError(f"prediction {n} has unknown or duplicate group_id")
        for key in ("model_id", "model_version"):
            if not isinstance(p[key], str) or not p[key].strip():
                raise ContractError(f"prediction {n}.{key} must be non-empty")
        digest(p["run_receipt_sha256"], f"prediction {n}.run_receipt_sha256")
        if type(p["abstained"]) is not bool:
            raise ContractError(f"prediction {n}.abstained must be boolean")
        necessary = strings(p["claimed_necessary_declarations"], f"prediction {n}.claimed_necessary")
        confidence = p["confidence"]
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1
        ):
            raise ContractError(f"prediction {n}.confidence must be null or lie in [0,1]")
        route = p["proposed_sufficient_route"]
        if p["abstained"]:
            if necessary or route is not None:
                raise ContractError("abstention must not include support claims")
        else:
            route = strings(route, f"prediction {n}.proposed_sufficient_route")
            universe = set(by_id[gid]["mutable_declaration_universe"])
            if not set(necessary) <= universe or not set(route) <= universe:
                raise ContractError("prediction contains declarations outside the lattice")
        pred_by_id[gid] = p
    if set(pred_by_id) != set(by_id):
        missing = sorted(set(by_id) - set(pred_by_id))
        raise ContractError(f"prediction ledger must cover every analyzed group exactly once; missing={missing[:8]}")

    tp = fp = fn = tn = valid = exact = evaluated = 0
    brier_values = []
    for gid, p in pred_by_id.items():
        g = by_id[gid]
        if p["abstained"]:
            continue
        evaluated += 1
        gold = set(g["necessary_across_all_observed_minimal_routes"])
        predicted = set(p["claimed_necessary_declarations"])
        universe = set(g["mutable_declaration_universe"])
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        tn += len(universe - (gold | predicted))
        route = set(p["proposed_sufficient_route"])
        routes = [set(x) for x in g["minimal_sufficient_support_antichain"]]
        is_valid = any(r <= route for r in routes)
        is_exact = any(r == route for r in routes)
        valid += int(is_valid)
        exact += int(is_exact)
        if p["confidence"] is not None:
            brier_values.append((float(p["confidence"]) - float(is_valid)) ** 2)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "prediction_count": len(predictions), "evaluated_prediction_count": evaluated,
        "prediction_coverage": evaluated / len(predictions) if predictions else 0.0,
        "necessary_declaration_confusion": {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn},
        "necessary_declaration_precision": precision, "necessary_declaration_recall": recall,
        "necessary_declaration_f1": f1,
        "route_sufficiency_rate": valid / evaluated if evaluated else None,
        "exact_minimal_route_rate": exact / evaluated if evaluated else None,
        "route_sufficiency_brier_score": sum(brier_values) / len(brier_values) if brier_values else None,
        "route_sufficiency_probability_scored_count": len(brier_values),
        "selective_route_risk": 1 - valid / evaluated if evaluated else None,
        "interpretation_boundary": "Necessity is across a finite exhaustive antichain under the pinned policy; it is not theorem-level logical independence.",
    }


def analyze(rows, predictions=None, max_variables=12):
    grouped = defaultdict(list)
    for i, raw in enumerate(rows, 1):
        row = validate_outcome(raw, i)
        grouped[group_key(row)].append(row)
    reports = [analyze_group(k, v) for k, v in sorted(grouped.items())]
    return {
        "schema_version": REPORT_SCHEMA,
        "outcomes_jsonl_sha256": None,
        "group_count": len(reports), "exhaustive_group_count": len(reports),
        "total_ablation_rows": len(rows),
        "max_mutable_declarations": max((len(g["mutable_declaration_universe"]) for g in reports), default=0),
        "groups": reports,
        "prediction_analysis": score_predictions(predictions, reports) if predictions is not None else None,
        "receipt_verification_boundary": "Digest syntax and row binding only; referenced receipt bytes must be independently opened and authenticated before benchmark use.",
        "overall_interpretation_boundary": "Finite exhaustive support ablation under fixed target, environment, toolchain, fixed context, prover policy and budget; not logical necessity or global mathematical necessity.",
    }


def synthetic_rows(study, target, component, universe, predicate):
    hashes = {k: hashlib.sha256(f"{study}:{target}:{k}".encode()).hexdigest()
              for k in ("target", "env", "tool", "fixed", "budget", "policy")}
    rows = []
    for subset in sorted(powerset(sorted(universe)), key=lambda s: (len(s), s)):
        available = list(subset)
        passed = bool(predicate(set(available)))
        ctx = hashlib.sha256(json.dumps([target, available], separators=(",", ":")).encode()).hexdigest()
        row = {
            "schema_version": OUTCOME_SCHEMA, "study_id": study, "component_id": component,
            "target_id": target, "analysis_endpoint": "exact_target_replan",
            "target_statement_sha256": hashes["target"], "environment_sha256": hashes["env"],
            "toolchain_sha256": hashes["tool"], "fixed_context_sha256": hashes["fixed"],
            "budget_sha256": hashes["budget"], "solver_policy_sha256": hashes["policy"],
            "mutable_declarations": sorted(universe), "available_declarations": available,
            "local_context_sha256": ctx, "outcome": "verified_success" if passed else "non_solve",
            "kernel_passed": passed, "semantic_gate_passed": passed,
            "state_receipt_sha256": hashlib.sha256(f"state:{available}".encode()).hexdigest(),
            "run_receipt_sha256": hashlib.sha256(f"run:{available}".encode()).hexdigest(),
            "checker_receipt_sha256": hashlib.sha256(f"checker:{available}".encode()).hexdigest(),
            "semantic_gate_receipt_sha256": hashlib.sha256(f"semantic:{available}".encode()).hexdigest(),
        }
        rows.append(row)
    return rows


def self_test():
    alt = synthetic_rows("self", "alternative", "c1", ["d", "h", "hEq"],
                         lambda s: "h" in s or "hEq" in s)
    con = synthetic_rows("self", "conjunctive", "c2", ["d", "h1", "h2"],
                         lambda s: {"h1", "h2"} <= s)
    groups = [analyze_group(group_key(alt[0]), alt), analyze_group(group_key(con[0]), con)]
    assert groups[0]["minimal_sufficient_support_antichain"] == [["h"], ["hEq"]]
    assert groups[0]["necessary_across_all_observed_minimal_routes"] == []
    assert groups[0]["possible_across_observed_minimal_routes"] == ["h", "hEq"]
    assert groups[1]["minimal_sufficient_support_antichain"] == [["h1", "h2"]]
    predictions = []
    for target, route, required in (
        ("alternative", ["h"], []),
        ("conjunctive", ["h1", "h2"], ["h1", "h2"]),
    ):
        group = next(g for g in groups if g["target_id"] == target)
        predictions.append({
            "schema_version": PREDICTION_SCHEMA,
            "group_id": group["group_id"],
            "model_id": "synthetic-control",
            "model_version": "self-test",
            "abstained": False,
            "claimed_necessary_declarations": required,
            "proposed_sufficient_route": route,
            "confidence": 0.8,
            "run_receipt_sha256": hashlib.sha256(target.encode()).hexdigest(),
        })
    scored = score_predictions(predictions, groups)
    assert scored["prediction_coverage"] == 1.0
    assert scored["route_sufficiency_rate"] == 1.0
    assert scored["exact_minimal_route_rate"] == 1.0
    assert abs(scored["route_sufficiency_brier_score"] - 0.04) < 1e-12
    incomplete = alt[:-1]
    try:
        analyze_group(group_key(incomplete[0]), incomplete)
    except ContractError:
        pass
    else:
        raise AssertionError("incomplete lattice was accepted")
    altered = [dict(x) for x in alt]
    altered[0]["budget_sha256"] = hashlib.sha256(b"changed").hexdigest()
    try:
        analyze_group(group_key(altered[0]), altered)
    except ContractError:
        pass
    else:
        raise AssertionError("mixed budget was accepted")
    invalid = dict(alt[-1], outcome="verified_success", kernel_passed=False)
    try:
        validate_outcome(invalid, 999)
    except ContractError:
        pass
    else:
        raise AssertionError("unverified success was accepted")
    print("PITB_SUPPORT_ANTICHAIN_SELF_TEST_PASS: alternative routes, conjunctive support, prediction scoring, exhaustive-lattice rejection, budget identity, and kernel/semantic gate enforcement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            if args.outcomes or args.predictions or args.report:
                raise ContractError("--self-test cannot be combined with file arguments")
            self_test()
            return 0
        if not args.outcomes or not args.report:
            raise ContractError("--outcomes and --report are required")
        rows, raw_digest = load_jsonl(args.outcomes, "outcome ledger")
        predictions = None
        pred_digest = None
        if args.predictions:
            predictions, pred_digest = load_jsonl(args.predictions, "prediction ledger")
        report = analyze(rows, predictions)
        report["outcomes_jsonl_sha256"] = raw_digest
        report["predictions_jsonl_sha256"] = pred_digest
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PITB_SUPPORT_ANTICHAIN_ANALYSIS: groups={report['group_count']} rows={len(rows)} predictions={'scored' if predictions is not None else 'not_supplied'}")
        return 0
    except (ContractError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"PITB support-antichain analysis rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
