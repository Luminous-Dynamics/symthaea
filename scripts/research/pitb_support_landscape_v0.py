#!/usr/bin/env python3
"""Repeated-seed, dependency-aware support-landscape analysis for PITB research."""
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

SCHEMA = "pitb.support-landscape-trial.v0"
REPORT_SCHEMA = "pitb.support-landscape-report.v0"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY = (
    "study_id", "component_id", "target_id", "model_id", "model_version",
    "checker_version", "solver_policy_sha256", "budget_sha256",
    "target_statement_sha256", "environment_sha256", "toolchain_sha256",
    "fixed_context_sha256", "replicate_count",
)
HASH_FIELDS = (
    "solver_policy_sha256", "budget_sha256", "target_statement_sha256",
    "environment_sha256", "toolchain_sha256", "fixed_context_sha256",
)
ROW_KEYS = set(IDENTITY) | set(HASH_FIELDS) | {
    "schema_version", "mutable_declarations", "fixed_declarations",
    "declaration_order", "declaration_dependencies", "requested_declarations",
    "context_well_formed", "local_context_sha256", "context_receipt_sha256",
    "replicate_index", "seed_sha256", "run_receipt_sha256",
    "checker_receipt_sha256", "semantic_gate_receipt_sha256",
    "outcome", "solver_started", "kernel_passed", "semantic_gate_passed",
    "model_calls", "tokens", "wall_ms", "checker_ms", "cost_microusd",
}

class ContractError(ValueError):
    pass


def digest(value, label):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value


def sorted_ids(value, label):
    if (not isinstance(value, list)
            or any(not isinstance(x, str) or not x.strip() for x in value)
            or value != sorted(value)
            or len(value) != len(set(value))):
        raise ContractError(f"{label} must be sorted unique non-empty strings")
    return value


def powerset(values):
    return {
        tuple(c)
        for size in range(len(values) + 1)
        for c in itertools.combinations(values, size)
    }


def wilson(successes, trials, z=1.959963984540054):
    if trials <= 0:
        return None
    p = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def validate_dependency_graph(universe, fixed, order, graph, where):
    if set(universe) & set(fixed):
        raise ContractError(f"{where}: mutable and fixed declaration IDs overlap")
    if len(order) != len(universe) or set(order) != set(universe):
        raise ContractError(f"{where}: declaration_order must contain every mutable declaration exactly once")
    if not isinstance(graph, dict) or set(graph) != set(universe):
        raise ContractError(f"{where}: dependency graph must have exactly one entry per mutable declaration")
    positions = {name: index for index, name in enumerate(order)}
    normalized = {}
    for declaration in universe:
        deps = sorted_ids(graph[declaration], f"{where}.dependencies[{declaration}]")
        for dep in deps:
            if dep not in positions and dep not in fixed:
                raise ContractError(f"{where}: {declaration} depends on undeclared prerequisite {dep}")
            if dep in positions and positions[dep] >= positions[declaration]:
                raise ContractError(f"{where}: local dependency {dep} must precede dependent declaration {declaration}")
        normalized[declaration] = deps
    return normalized


def context_is_well_formed(requested, fixed, graph):
    present = set(requested) | set(fixed)
    missing = []
    for declaration in requested:
        for dep in graph[declaration]:
            if dep not in present:
                missing.append((declaration, dep))
    return not missing, sorted(missing)


def validate_row(row, index, max_variables=8):
    if not isinstance(row, dict) or set(row) != ROW_KEYS:
        missing = sorted(ROW_KEYS - set(row)) if isinstance(row, dict) else sorted(ROW_KEYS)
        extra = sorted(set(row) - ROW_KEYS) if isinstance(row, dict) else []
        raise ContractError(f"row {index}: missing={missing}, extra={extra}")
    if row["schema_version"] != SCHEMA:
        raise ContractError(f"row {index}: unsupported schema version")
    for key in IDENTITY:
        if key in HASH_FIELDS:
            digest(row[key], f"row {index}.{key}")
        elif key == "replicate_count":
            if type(row[key]) is not int or row[key] < 2:
                raise ContractError(f"row {index}.replicate_count must be an integer >= 2")
        else:
            nonempty(row[key], f"row {index}.{key}")
    for key in HASH_FIELDS:
        digest(row[key], f"row {index}.{key}")
    universe = sorted_ids(row["mutable_declarations"], f"row {index}.mutable_declarations")
    fixed = sorted_ids(row["fixed_declarations"], f"row {index}.fixed_declarations")
    order = row["declaration_order"]
    if not isinstance(order, list) or any(not isinstance(x, str) for x in order):
        raise ContractError(f"row {index}.declaration_order must be an array of IDs")
    if len(universe) > max_variables:
        raise ContractError(f"row {index}: {len(universe)} mutable declarations exceeds configured cap {max_variables}")
    graph = validate_dependency_graph(universe, fixed, order, row["declaration_dependencies"], f"row {index}")
    requested = sorted_ids(row["requested_declarations"], f"row {index}.requested_declarations")
    if not set(requested) <= set(universe):
        raise ContractError(f"row {index}: requested subset contains declarations outside the mutable universe")
    well_formed, missing = context_is_well_formed(requested, fixed, graph)
    if type(row["context_well_formed"]) is not bool or row["context_well_formed"] != well_formed:
        raise ContractError(f"row {index}: context_well_formed claim disagrees with the dependency graph")
    digest(row["local_context_sha256"], f"row {index}.local_context_sha256")
    for key in ("context_receipt_sha256", "seed_sha256", "run_receipt_sha256",
                "checker_receipt_sha256", "semantic_gate_receipt_sha256"):
        digest(row[key], f"row {index}.{key}")
    r = row["replicate_index"]
    if type(r) is not int or not 0 <= r < row["replicate_count"]:
        raise ContractError(f"row {index}.replicate_index lies outside the frozen replicate schedule")
    if row["outcome"] not in ("verified_success", "non_solve", "timeout", "invalid_context"):
        raise ContractError(f"row {index}: unsupported or unresolved outcome {row['outcome']!r}")
    for key in ("solver_started", "kernel_passed", "semantic_gate_passed"):
        if type(row[key]) is not bool:
            raise ContractError(f"row {index}.{key} must be boolean")
    costs = ("model_calls", "tokens", "wall_ms", "checker_ms", "cost_microusd")
    for key in costs:
        if type(row[key]) is not int or row[key] < 0:
            raise ContractError(f"row {index}.{key} must be a non-negative integer")
    if row["semantic_gate_passed"] and not row["kernel_passed"]:
        raise ContractError(f"row {index}: semantic gate cannot pass when the trusted kernel check failed")
    passed = row["kernel_passed"] and row["semantic_gate_passed"]
    if row["outcome"] == "verified_success":
        if not well_formed or not row["solver_started"] or not passed:
            raise ContractError(f"row {index}: verified success requires a well-formed context, solver run and both proof gates")
    elif row["outcome"] == "invalid_context":
        if well_formed or row["solver_started"] or passed or any(row[k] for k in costs):
            raise ContractError(f"row {index}: invalid context cannot start the solver, pass gates, or consume solver/checker budget")
    else:
        if not well_formed or not row["solver_started"] or passed:
            raise ContractError(f"row {index}: non-success outcomes require a well-formed attempted context and at least one failed proof gate")
    if row["outcome"] == "invalid_context" and missing == []:
        raise ContractError(f"row {index}: invalid context lacks a dependency-closure witness")
    return row, graph


def identity_key(row):
    bound = {key: row[key] for key in IDENTITY}
    # A context-schema change must not silently split a study into smaller groups.
    for key in ("mutable_declarations", "fixed_declarations", "declaration_order", "declaration_dependencies"):
        bound[key] = row[key]
    return hashlib.sha256(json.dumps(
        bound, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def pareto_frontier(contexts):
    # Lower is better for all four engineering dimensions.
    def vector(c):
        return (
            c["declaration_count"],
            c["mean_tokens_per_attempt"],
            c["mean_wall_ms_per_attempt"],
            c["mean_cost_microusd_per_attempt"],
        )
    frontier = []
    for candidate in contexts:
        v = vector(candidate)
        dominated = False
        for other in contexts:
            if other is candidate:
                continue
            ov = vector(other)
            if all(a <= b for a, b in zip(ov, v)) and any(a < b for a, b in zip(ov, v)):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["requested_declarations"])
    return sorted(frontier, key=lambda s: (len(s), s))


def analyze_group(gid, rows, reliability_floor=0.80, min_replicates=20, max_variables=8):
    first = rows[0]
    universe = first["mutable_declarations"]
    if len(universe) > max_variables:
        raise ContractError(f"group {gid}: variable count exceeds analysis cap")
    expected_subsets = powerset(universe)
    expected_replicates = set(range(first["replicate_count"]))
    expected_pairs = {(subset, r) for subset in expected_subsets for r in expected_replicates}
    seen = {}
    context_digests = {}
    seed_by_replicate = {}
    run_receipts = set()
    context_receipts = {}
    for index, raw in enumerate(rows, 1):
        row, graph = validate_row(raw, index, max_variables)
        if identity_key(row) != gid:
            raise ContractError(f"group {gid}: mixed identity fields")
        if row["mutable_declarations"] != universe:
            raise ContractError(f"group {gid}: mutable declaration universe changed")
        for key in ("fixed_declarations", "declaration_order", "declaration_dependencies"):
            if row[key] != first[key]:
                raise ContractError(f"group {gid}: {key} changed across the paired subset lattice")
        if graph != first["declaration_dependencies"]:
            raise ContractError(f"group {gid}: dependency graph changed")
        subset = tuple(row["requested_declarations"])
        if subset not in expected_subsets:
            raise ContractError(f"group {gid}: unknown requested context {subset}")
        pair = (subset, row["replicate_index"])
        if pair in seen:
            raise ContractError(f"group {gid}: duplicate context/replicate pair {pair}")
        seen[pair] = row
        previous = context_digests.setdefault(subset, row["local_context_sha256"])
        if previous != row["local_context_sha256"]:
            raise ContractError(f"group {gid}: local-context digest changes across replicates for {subset}")
        previous_context_receipt = context_receipts.setdefault(subset, row["context_receipt_sha256"])
        if previous_context_receipt != row["context_receipt_sha256"]:
            raise ContractError(f"group {gid}: context receipt changes across replicates for {subset}")
        previous_seed = seed_by_replicate.setdefault(row["replicate_index"], row["seed_sha256"])
        if previous_seed != row["seed_sha256"]:
            raise ContractError(f"group {gid}: replicate seed differs across paired contexts")
        if row["run_receipt_sha256"] in run_receipts:
            raise ContractError(f"group {gid}: duplicate run receipt")
        run_receipts.add(row["run_receipt_sha256"])
    missing = expected_pairs - set(seen)
    extra = set(seen) - expected_pairs
    if missing or extra:
        raise ContractError(f"group {gid}: repeated subset schedule incomplete; missing={len(missing)}, extra={len(extra)}")

    contexts = []
    for subset in sorted(expected_subsets, key=lambda s: (len(s), s)):
        trial_rows = [seen[(subset, r)] for r in sorted(expected_replicates)]
        first_trial = trial_rows[0]
        if any(r["context_well_formed"] != first_trial["context_well_formed"] for r in trial_rows):
            raise ContractError(f"group {gid}: context validity varies across replicates")
        if first_trial["context_well_formed"]:
            successes = sum(r["outcome"] == "verified_success" for r in trial_rows)
            n = len(trial_rows)
            interval = wilson(successes, n)
            mean_tokens = sum(r["tokens"] for r in trial_rows) / n
            mean_wall = sum(r["wall_ms"] for r in trial_rows) / n
            mean_cost = sum(r["cost_microusd"] for r in trial_rows) / n
            mean_calls = sum(r["model_calls"] for r in trial_rows) / n
            qualified = n >= min_replicates and interval[0] >= reliability_floor
            contexts.append({
                "requested_declarations": list(subset),
                "declaration_count": len(subset),
                "replicate_count": n,
                "verified_successes": successes,
                "timeouts": sum(r["outcome"] == "timeout" for r in trial_rows),
                "non_solves": sum(r["outcome"] == "non_solve" for r in trial_rows),
                "empirical_success_probability": successes / n,
                "wilson_95": {"lower": interval[0], "upper": interval[1]},
                "reliability_floor": reliability_floor,
                "reliability_qualified": qualified,
                "mean_model_calls_per_attempt": mean_calls,
                "mean_tokens_per_attempt": mean_tokens,
                "mean_wall_ms_per_attempt": mean_wall,
                "mean_cost_microusd_per_attempt": mean_cost,
                "total_verified_success_cost_microusd": sum(r["cost_microusd"] for r in trial_rows if r["outcome"] == "verified_success"),
                "context_receipt_sha256": first_trial["context_receipt_sha256"],
                "interpretation_boundary": "Reliability is a finite repeated-seed estimate for one frozen solver/config/budget; it is not theorem-level necessity.",
            })
        else:
            contexts.append({
                "requested_declarations": list(subset),
                "declaration_count": len(subset),
                "replicate_count": len(trial_rows),
                "context_well_formed": False,
                "outcome": "invalid_context",
                "context_receipt_sha256": first_trial["context_receipt_sha256"],
                "interpretation_boundary": "Excluded from support sufficiency; invalid context is not evidence that a mathematical premise is unnecessary.",
            })
    valid = [c for c in contexts if c.get("context_well_formed", True)]
    qualified = [c for c in valid if c["reliability_qualified"]]
    reliable_antichain = [
        c for c in qualified
        if not any(set(other["requested_declarations"]) < set(c["requested_declarations"]) for other in qualified)
    ]
    full = next(c for c in contexts if c["requested_declarations"] == universe)
    full_context_qualified = full.get("reliability_qualified", False)
    monotonicity_violations = 0
    for a in valid:
        sa = set(a["requested_declarations"])
        for b in valid:
            sb = set(b["requested_declarations"])
            if sa < sb and a["empirical_success_probability"] > b["empirical_success_probability"]:
                monotonicity_violations += 1
    pareto = pareto_frontier(qualified)
    attempted_trials = sum(c["replicate_count"] for c in valid)
    success_trials = sum(c["verified_successes"] for c in valid)
    total_cost = sum(r["cost_microusd"] for r in rows)
    overall_cost_per_success = total_cost / success_trials if success_trials else None
    return {
        "group_id": gid,
        "study_id": first["study_id"],
        "component_id": first["component_id"],
        "target_id": first["target_id"],
        "model_id": first["model_id"],
        "model_version": first["model_version"],
        "checker_version": first["checker_version"],
        "mutable_declarations": universe,
        "fixed_declarations": first["fixed_declarations"],
        "declaration_order": first["declaration_order"],
        "declaration_dependencies": first["declaration_dependencies"],
        "replicate_count_per_context": first["replicate_count"],
        "tested_context_count": len(expected_subsets),
        "well_formed_context_count": len(valid),
        "invalid_context_count": len(contexts) - len(valid),
        "tested_attempts": attempted_trials,
        "verified_successes": success_trials,
        "reliability_floor": reliability_floor,
        "minimum_replicates": min_replicates,
        "full_context_reliability_qualified": full_context_qualified,
        "reliable_minimal_support_antichain": [c["requested_declarations"] for c in reliable_antichain] if full_context_qualified else [],
        "reliable_minimal_support_antichain_details": reliable_antichain if full_context_qualified else [],
        "pareto_efficient_reliable_contexts": pareto if full_context_qualified else [],
        "observed_success_monotonicity_violation_pairs": monotonicity_violations,
        "total_cost_microusd": total_cost,
        "verified_cost_microusd_per_success": overall_cost_per_success,
        "contexts": contexts,
        "qualification_status": "qualified_reliability_landscape" if full_context_qualified else "full_context_below_reliability_floor",
        "pass_at_k_reporting": "Direct repeated-seed success estimates and Wilson intervals only; no pass@k extrapolation beyond observed replicate count.",
        "interpretation_boundary": "A finite dependency-aware reliability landscape for one target, prover/checker version, policy, fixed context and budget. Not a proof of logical independence or global necessity.",
    }


def analyze(rows, reliability_floor=0.80, min_replicates=20, max_variables=8):
    grouped = defaultdict(list)
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, dict):
            raise ContractError("each outcome row must be a JSON object")
        validate_row(raw, index, max_variables)
        grouped[identity_key(raw)].append(raw)
    reports = [
        analyze_group(gid, group, reliability_floor, min_replicates, max_variables)
        for gid, group in sorted(grouped.items())
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "group_count": len(reports),
        "groups": reports,
        "reliability_definition": "A context qualifies only when its two-sided 95% Wilson lower confidence bound is at least the frozen reliability_floor and replicate count is at least min_replicates.",
        "context_validity_definition": "A requested context is well formed only if every selected mutable declaration's mutable prerequisites are also selected and every fixed prerequisite belongs to the pinned fixed context.",
        "evidence_boundary": "Receipt digest syntax, row identity and schedule completeness are checked. Referenced receipt bytes must still be independently opened and authenticated.",
    }


def synthetic_rows(study, target, component, universe, dependencies, predicate,
                   replicate_count=50, successes_by_subset=None):
    fixed = ["fixed-base"]
    ordered = sorted(universe)
    hashes = {key: hashlib.sha256(f"{study}:{target}:{key}".encode()).hexdigest()
              for key in HASH_FIELDS}
    graph = {name: sorted(dependencies.get(name, [])) for name in ordered}
    rows = []
    for subset in sorted(powerset(ordered), key=lambda s: (len(s), s)):
        requested = list(subset)
        well, _ = context_is_well_formed(requested, fixed, graph)
        context_digest = hashlib.sha256(json.dumps(
            [target, requested, fixed, graph], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        context_receipt = hashlib.sha256(f"context:{target}:{requested}".encode()).hexdigest()
        for rep in range(replicate_count):
            success_count = successes_by_subset.get(subset, replicate_count) if successes_by_subset else replicate_count
            success = well and rep < success_count and bool(predicate(set(subset)))
            outcome = "invalid_context" if not well else ("verified_success" if success else "non_solve")
            rows.append({
                "schema_version": SCHEMA,
                "study_id": study, "component_id": component, "target_id": target,
                "model_id": "synthetic-agent", "model_version": "v0",
                "checker_version": "synthetic-lean-checker",
                **hashes,
                "replicate_count": replicate_count,
                "mutable_declarations": ordered,
                "fixed_declarations": fixed,
                "declaration_order": ordered,
                "declaration_dependencies": graph,
                "requested_declarations": requested,
                "context_well_formed": well,
                "local_context_sha256": context_digest,
                "context_receipt_sha256": context_receipt,
                "replicate_index": rep,
                "seed_sha256": hashlib.sha256(f"paired-seed-{rep}".encode()).hexdigest(),
                "run_receipt_sha256": hashlib.sha256(f"run:{target}:{subset}:{rep}".encode()).hexdigest(),
                "checker_receipt_sha256": hashlib.sha256(f"checker:{target}:{subset}:{rep}".encode()).hexdigest(),
                "semantic_gate_receipt_sha256": hashlib.sha256(f"semantic:{target}:{subset}:{rep}".encode()).hexdigest(),
                "outcome": outcome,
                "solver_started": well,
                "kernel_passed": success,
                "semantic_gate_passed": success,
                "model_calls": 1 if well else 0,
                "tokens": 120 if well else 0,
                "wall_ms": 40 if well else 0,
                "checker_ms": 5 if success else (1 if well else 0),
                "cost_microusd": 2 if well else 0,
            })
    return rows


def self_test():
    alternative = synthetic_rows(
        "self-test", "alternative-routes", "component-a",
        ["d", "h", "hEq"], {}, lambda s: "h" in s or "hEq" in s
    )
    conjunction = synthetic_rows(
        "self-test", "conjunctive-route", "component-b",
        ["h1", "h2", "noise"], {}, lambda s: {"h1", "h2"} <= s
    )
    dependency = synthetic_rows(
        "self-test", "dependent-context", "component-c",
        ["base", "dependent"], {"dependent": ["base"]}, lambda s: "base" in s
    )
    groups = {}
    for rows in (alternative, conjunction, dependency):
        grouped = defaultdict(list)
        for row in rows:
            grouped[identity_key(row)].append(row)
        gid, group = next(iter(grouped.items()))
        groups[group[0]["target_id"]] = analyze_group(gid, group)
    assert ["h"] in groups["alternative-routes"]["reliable_minimal_support_antichain"]
    assert ["hEq"] in groups["alternative-routes"]["reliable_minimal_support_antichain"]
    assert groups["conjunctive-route"]["reliable_minimal_support_antichain"] == [["h1", "h2"]]
    dep_group = groups["dependent-context"]
    assert dep_group["invalid_context_count"] == 2
    assert ["base"] in dep_group["reliable_minimal_support_antichain"]

    stochastic = synthetic_rows(
        "self-test", "stochastic-route", "component-d",
        ["h", "hEq"], {}, lambda s: bool(s),
        successes_by_subset={(): 0, ("h",): 50, ("hEq",): 48, ("h", "hEq"): 50}
    )
    grouped = defaultdict(list)
    for row in stochastic:
        grouped[identity_key(row)].append(row)
    stochastic_report = analyze_group(next(iter(grouped)), next(iter(grouped.values())))
    assert stochastic_report["reliable_minimal_support_antichain"] == [["h"], ["hEq"]]

    # A 45/50 route must not clear the 0.80 Wilson lower-bound gate.
    weak = synthetic_rows(
        "self-test", "below-reliability-floor", "component-e",
        ["h"], {}, lambda s: bool(s),
        successes_by_subset={(): 0, ("h",): 45}
    )
    grouped = defaultdict(list)
    for row in weak:
        grouped[identity_key(row)].append(row)
    weak_report = analyze_group(next(iter(grouped)), next(iter(grouped.values())))
    assert weak_report["reliable_minimal_support_antichain"] == []
    assert weak_report["qualification_status"] == "full_context_below_reliability_floor"

    # Missing replicate, changed paired seed, and false context-closure claims fail closed.
    incomplete = alternative[:-1]
    try:
        grouped = defaultdict(list)
        for row in incomplete:
            grouped[identity_key(row)].append(row)
        analyze_group(next(iter(grouped)), next(iter(grouped.values())))
    except ContractError:
        pass
    else:
        raise AssertionError("incomplete replicate lattice was accepted")
    altered_seed = [dict(row) for row in alternative]
    altered_seed[-1]["seed_sha256"] = hashlib.sha256(b"wrong-seed").hexdigest()
    try:
        grouped = defaultdict(list)
        for row in altered_seed:
            grouped[identity_key(row)].append(row)
        analyze_group(next(iter(grouped)), next(iter(grouped.values())))
    except ContractError:
        pass
    else:
        raise AssertionError("unpaired seed schedule was accepted")
    false_closure = [dict(row) for row in dependency]
    row = next(r for r in false_closure if r["requested_declarations"] == ["dependent"])
    row["context_well_formed"] = True
    try:
        validate_row(row, 999)
    except ContractError:
        pass
    else:
        raise AssertionError("false context-well-formed claim was accepted")

    changed_fixed = [dict(row) for row in alternative]
    changed_fixed[-1]["fixed_declarations"] = ["different-fixed-base"]
    try:
        grouped = defaultdict(list)
        for row in changed_fixed:
            grouped[identity_key(row)].append(row)
        for gid, group in grouped.items():
            analyze_group(gid, group)
    except ContractError:
        pass
    else:
        raise AssertionError("changed fixed context was accepted")

    wrong_semantics = dict(alternative[0], kernel_passed=False, semantic_gate_passed=True,
                           outcome="non_solve")
    try:
        validate_row(wrong_semantics, 1000)
    except ContractError:
        pass
    else:
        raise AssertionError("semantic pass without kernel pass was accepted")

    print("PITB_SUPPORT_LANDSCAPE_SELF_TEST_PASS: alternative routes, conjunction, dependency closure, repeated-seed reliability, Wilson threshold, incomplete schedule, paired-seed and invalid-context controls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reliability-floor", type=float, default=0.80)
    parser.add_argument("--min-replicates", type=int, default=20)
    parser.add_argument("--max-variables", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if not 0.5 < args.reliability_floor < 1.0:
            raise ContractError("--reliability-floor must lie strictly between 0.5 and 1")
        if args.min_replicates < 2:
            raise ContractError("--min-replicates must be at least 2")
        if not 1 <= args.max_variables <= 10:
            raise ContractError("--max-variables must lie in [1,10]")
        if args.self_test:
            if args.outcomes or args.report:
                raise ContractError("--self-test cannot be combined with input/output paths")
            self_test()
            return 0
        if not args.outcomes or not args.report:
            raise ContractError("--outcomes and --report are required")
        rows = []
        for number, line in enumerate(args.outcomes.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ContractError(f"input line {number}: {exc}") from exc
        if not rows:
            raise ContractError("outcome ledger is empty")
        for i, row in enumerate(rows, 1):
            replicate_count = row.get("replicate_count", 0) if isinstance(row, dict) else 0
            if replicate_count < args.min_replicates:
                raise ContractError(f"row {i}: replicate_count is below the configured minimum")
        report = analyze(rows, args.reliability_floor, args.min_replicates, args.max_variables)
        report["outcomes_jsonl_sha256"] = hashlib.sha256(args.outcomes.read_bytes()).hexdigest()
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PITB_SUPPORT_LANDSCAPE_ANALYSIS: groups={report['group_count']} reliability_floor={args.reliability_floor:.3f} report={args.report}")
        return 0
    except (ContractError, OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"PITB support landscape rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
