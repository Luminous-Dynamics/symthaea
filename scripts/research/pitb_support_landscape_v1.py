#!/usr/bin/env python3
"""Repeated-seed, dependency-aware support-landscape analysis for PITB."""
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

SCHEMA = "pitb.support-landscape-trial.v1"
REPORT_SCHEMA = "pitb.support-landscape-report.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY = (
    "study_id", "component_id", "target_id", "model_id", "model_version",
    "checker_version", "solver_policy_sha256", "budget_sha256",
    "target_statement_sha256", "environment_sha256", "toolchain_sha256",
    "fixed_context_sha256", "replicate_count",
)
HASH_FIELDS = {
    "solver_policy_sha256", "budget_sha256", "target_statement_sha256",
    "environment_sha256", "toolchain_sha256", "fixed_context_sha256",
}
BUDGET_FIELDS = {
    "max_model_calls", "max_tokens", "max_wall_ms", "max_checker_ms",
    "max_cost_microusd",
}
COST_FIELDS = {
    "model_calls": "max_model_calls",
    "tokens": "max_tokens",
    "wall_ms": "max_wall_ms",
    "checker_ms": "max_checker_ms",
    "cost_microusd": "max_cost_microusd",
}
ROW_KEYS = set(IDENTITY) | HASH_FIELDS | {
    "schema_version", "budget", "mutable_declarations", "fixed_declarations",
    "declaration_order", "declaration_dependencies", "requested_declarations",
    "context_well_formed", "local_context_sha256", "context_receipt_sha256",
    "replicate_index", "seed_sha256", "run_receipt_sha256",
    "checker_receipt_sha256", "semantic_gate_receipt_sha256",
    "outcome", "solver_started", "kernel_passed", "semantic_gate_passed",
    *COST_FIELDS,
}


class ContractError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def require_digest(value, label):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def require_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value


def require_ids(value, label):
    if (not isinstance(value, list)
            or any(not isinstance(x, str) or not x.strip() for x in value)
            or value != sorted(value)
            or len(value) != len(set(value))):
        raise ContractError(f"{label} must be sorted unique non-empty strings")
    return value


def powerset(values):
    return {
        tuple(values[i] for i in range(len(values)) if mask & (1 << i))
        for mask in range(1 << len(values))
    }


def wilson(successes, trials, z=1.959963984540054):
    p = successes / trials
    z2 = z * z
    den = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / den
    radius = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def check_budget(row, index):
    budget = row["budget"]
    if not isinstance(budget, dict) or set(budget) != BUDGET_FIELDS:
        raise ContractError(f"row {index}.budget must contain exactly {sorted(BUDGET_FIELDS)}")
    for key, value in budget.items():
        if type(value) is not int or value < 0:
            raise ContractError(f"row {index}.budget.{key} must be a non-negative integer")
    if sha256(canonical(budget)) != row["budget_sha256"]:
        raise ContractError(f"row {index}: budget digest does not match declared caps")
    for observed, cap in COST_FIELDS.items():
        value = row[observed]
        if type(value) is not int or value < 0:
            raise ContractError(f"row {index}.{observed} must be a non-negative integer")
        if value > budget[cap]:
            raise ContractError(f"row {index}.{observed} exceeds frozen cap {cap}")


def validate_graph(universe, fixed, order, graph, label):
    if set(universe) & set(fixed):
        raise ContractError(f"{label}: mutable and fixed declaration IDs overlap")
    if len(order) != len(universe) or set(order) != set(universe):
        raise ContractError(f"{label}: declaration_order must contain every mutable declaration once")
    if not isinstance(graph, dict) or set(graph) != set(universe):
        raise ContractError(f"{label}: dependency graph must have one entry per mutable declaration")
    positions = {name: i for i, name in enumerate(order)}
    result = {}
    for name in universe:
        deps = require_ids(graph[name], f"{label}.declaration_dependencies[{name}]")
        for dep in deps:
            if dep not in positions and dep not in fixed:
                raise ContractError(f"{label}: {name} depends on unknown prerequisite {dep}")
            if dep in positions and positions[dep] >= positions[name]:
                raise ContractError(f"{label}: {dep} must precede dependent declaration {name}")
        result[name] = deps
    return result


def context_valid(subset, fixed, graph):
    present = set(subset) | set(fixed)
    missing = sorted((name, dep) for name in subset for dep in graph[name] if dep not in present)
    return not missing, missing


def validate_row(row, index, max_variables):
    if not isinstance(row, dict) or set(row) != ROW_KEYS:
        missing = sorted(ROW_KEYS - set(row)) if isinstance(row, dict) else sorted(ROW_KEYS)
        extra = sorted(set(row) - ROW_KEYS) if isinstance(row, dict) else []
        raise ContractError(f"row {index}: missing={missing}, extra={extra}")
    if row["schema_version"] != SCHEMA:
        raise ContractError(f"row {index}: unsupported schema version")
    for field in IDENTITY:
        if field in HASH_FIELDS:
            require_digest(row[field], f"row {index}.{field}")
        elif field == "replicate_count":
            if type(row[field]) is not int or row[field] < 2:
                raise ContractError(f"row {index}.replicate_count must be >= 2")
        else:
            require_string(row[field], f"row {index}.{field}")
    for field in HASH_FIELDS:
        require_digest(row[field], f"row {index}.{field}")
    if sha256(canonical(row["budget"])) != row["budget_sha256"]:
        raise ContractError(f"row {index}: budget digest mismatch")
    universe = require_ids(row["mutable_declarations"], f"row {index}.mutable_declarations")
    fixed = require_ids(row["fixed_declarations"], f"row {index}.fixed_declarations")
    if len(universe) > max_variables:
        raise ContractError(f"row {index}: variable count exceeds cap {max_variables}")
    order = row["declaration_order"]
    if not isinstance(order, list):
        raise ContractError(f"row {index}.declaration_order must be an array")
    graph = validate_graph(universe, fixed, order, row["declaration_dependencies"], f"row {index}")
    subset = require_ids(row["requested_declarations"], f"row {index}.requested_declarations")
    if not set(subset) <= set(universe):
        raise ContractError(f"row {index}: requested context is outside mutable universe")
    well_formed, missing = context_valid(subset, fixed, graph)
    if type(row["context_well_formed"]) is not bool or row["context_well_formed"] != well_formed:
        raise ContractError(f"row {index}: context validity disagrees with dependency closure")
    for field in ("local_context_sha256", "context_receipt_sha256", "seed_sha256",
                  "run_receipt_sha256", "checker_receipt_sha256", "semantic_gate_receipt_sha256"):
        require_digest(row[field], f"row {index}.{field}")
    rep = row["replicate_index"]
    if type(rep) is not int or not 0 <= rep < row["replicate_count"]:
        raise ContractError(f"row {index}.replicate_index outside frozen schedule")
    for field in ("solver_started", "kernel_passed", "semantic_gate_passed"):
        if type(row[field]) is not bool:
            raise ContractError(f"row {index}.{field} must be boolean")
    if row["semantic_gate_passed"] and not row["kernel_passed"]:
        raise ContractError(f"row {index}: semantic pass cannot occur without kernel pass")
    check_budget(row, index)
    outcome = row["outcome"]
    if outcome not in ("verified_success", "non_solve", "timeout", "invalid_context"):
        raise ContractError(f"row {index}: unsupported or unresolved outcome {outcome!r}")
    passed = row["kernel_passed"] and row["semantic_gate_passed"]
    if outcome == "invalid_context":
        if well_formed or row["solver_started"] or passed or any(row[k] for k in COST_FIELDS):
            raise ContractError(f"row {index}: invalid context cannot run solver or consume solver/checker budget")
        if not missing:
            raise ContractError(f"row {index}: invalid context has no dependency-closure witness")
    elif outcome == "verified_success":
        if not well_formed or not row["solver_started"] or not passed:
            raise ContractError(f"row {index}: success requires well-formed context, solver run, kernel and semantic pass")
    else:
        if not well_formed or not row["solver_started"] or passed:
            raise ContractError(f"row {index}: non-success requires well-formed attempted context and failed proof gate")
    return row, graph


def group_key(row):
    # Bind the complete context-design manifest, not only target/model metadata.
    bound = {k: row[k] for k in IDENTITY}
    for k in ("mutable_declarations", "fixed_declarations", "declaration_order", "declaration_dependencies"):
        bound[k] = row[k]
    return sha256(canonical(bound))


def pareto_frontier(contexts):
    dims = ("declaration_count", "mean_tokens", "mean_wall_ms", "mean_cost_microusd")
    result = []
    for candidate in contexts:
        cv = tuple(candidate[k] for k in dims)
        dominated = False
        for other in contexts:
            if other is candidate:
                continue
            ov = tuple(other[k] for k in dims)
            if all(x <= y for x, y in zip(ov, cv)) and any(x < y for x, y in zip(ov, cv)):
                dominated = True
                break
        if not dominated:
            result.append(candidate["requested_declarations"])
    return sorted(result, key=lambda x: (len(x), x))


def analyze_group(gid, rows, reliability_floor, min_replicates, max_variables):
    first = rows[0]
    universe = first["mutable_declarations"]
    expected_subsets = powerset(universe)
    expected_reps = set(range(first["replicate_count"]))
    expected_pairs = {(s, r) for s in expected_subsets for r in expected_reps}
    seen = {}
    ctx_sha = {}
    ctx_receipt = {}
    seed_by_rep = {}
    run_receipts = set()
    for index, raw in enumerate(rows, 1):
        row, graph = validate_row(raw, index, max_variables)
        if group_key(row) != gid:
            raise ContractError(f"group {gid}: mixed group identity")
        for field in ("mutable_declarations", "fixed_declarations", "declaration_order", "declaration_dependencies"):
            if row[field] != first[field]:
                raise ContractError(f"group {gid}: {field} changed across rows")
        subset = tuple(row["requested_declarations"])
        pair = (subset, row["replicate_index"])
        if pair in seen:
            raise ContractError(f"group {gid}: duplicate subset/replicate pair")
        seen[pair] = row
        for mapping, value, label in (
            (ctx_sha, row["local_context_sha256"], "local context"),
            (ctx_receipt, row["context_receipt_sha256"], "context receipt"),
        ):
            previous = mapping.setdefault(subset, value)
            if previous != value:
                raise ContractError(f"group {gid}: {label} changed across replicates")
        previous_seed = seed_by_rep.setdefault(row["replicate_index"], row["seed_sha256"])
        if previous_seed != row["seed_sha256"]:
            raise ContractError(f"group {gid}: paired replicate seed differs across contexts")
        if row["run_receipt_sha256"] in run_receipts:
            raise ContractError(f"group {gid}: duplicate run receipt")
        run_receipts.add(row["run_receipt_sha256"])
    if set(seen) != expected_pairs:
        raise ContractError(f"group {gid}: incomplete repeated context schedule; missing={len(expected_pairs-set(seen))}")

    contexts = []
    for subset in sorted(expected_subsets, key=lambda x: (len(x), x)):
        trials = [seen[(subset, rep)] for rep in sorted(expected_reps)]
        well_formed = trials[0]["context_well_formed"]
        if any(row["context_well_formed"] != well_formed for row in trials):
            raise ContractError(f"group {gid}: context validity varies across replicates")
        if not well_formed:
            contexts.append({
                "requested_declarations": list(subset),
                "declaration_count": len(subset),
                "context_well_formed": False,
                "replicate_count": len(trials),
                "status": "invalid_context",
            })
            continue
        successes = sum(row["outcome"] == "verified_success" for row in trials)
        lower, upper = wilson(successes, len(trials))
        item = {
            "requested_declarations": list(subset),
            "declaration_count": len(subset),
            "context_well_formed": True,
            "replicate_count": len(trials),
            "verified_successes": successes,
            "timeouts": sum(row["outcome"] == "timeout" for row in trials),
            "non_solves": sum(row["outcome"] == "non_solve" for row in trials),
            "success_probability": successes / len(trials),
            "wilson_95_lower": lower,
            "wilson_95_upper": upper,
            "reliability_qualified": len(trials) >= min_replicates and lower >= reliability_floor,
            "mean_model_calls": sum(row["model_calls"] for row in trials) / len(trials),
            "mean_tokens": sum(row["tokens"] for row in trials) / len(trials),
            "mean_wall_ms": sum(row["wall_ms"] for row in trials) / len(trials),
            "mean_cost_microusd": sum(row["cost_microusd"] for row in trials) / len(trials),
        }
        contexts.append(item)
    valid = [c for c in contexts if c["context_well_formed"]]
    reliable = [c for c in valid if c["reliability_qualified"]]
    antichain = [
        c for c in reliable
        if not any(set(other["requested_declarations"]) < set(c["requested_declarations"]) for other in reliable)
    ]
    full = next(c for c in contexts if c["requested_declarations"] == universe)
    full_ok = full.get("reliability_qualified", False)
    violations = sum(
        1 for a in valid for b in valid
        if set(a["requested_declarations"]) < set(b["requested_declarations"])
        and a["success_probability"] > b["success_probability"]
    )
    total_cost = sum(row["cost_microusd"] for row in rows)
    total_successes = sum(c.get("verified_successes", 0) for c in valid)
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
        "replicate_count": first["replicate_count"],
        "tested_context_count": len(expected_subsets),
        "well_formed_context_count": len(valid),
        "invalid_context_count": len(contexts) - len(valid),
        "attempted_trials": sum(c["replicate_count"] for c in valid),
        "verified_successes": total_successes,
        "reliability_floor": reliability_floor,
        "minimum_replicates": min_replicates,
        "full_context_reliability_qualified": full_ok,
        "reliable_minimal_support_antichain": [c["requested_declarations"] for c in antichain] if full_ok else [],
        "reliable_minimal_support_details": antichain if full_ok else [],
        "pareto_efficient_reliable_contexts": pareto_frontier(reliable) if full_ok else [],
        "observed_nonmonotone_subset_pairs": violations,
        "total_cost_microusd": total_cost,
        "cost_microusd_per_verified_success": total_cost / total_successes if total_successes else None,
        "contexts": contexts,
        "qualification_status": "qualified_reliability_landscape" if full_ok else "full_context_below_reliability_floor",
        "pass_at_k_reporting": "Direct repeated-seed estimates and Wilson intervals only; no pass@k extrapolation beyond observed replicates.",
        "interpretation_boundary": "Finite dependency-aware reliability landscape for a fixed target, model/checker, policy, environment, context and budget; not theorem-level logical necessity.",
    }


def analyze(rows, reliability_floor=0.80, min_replicates=20, max_variables=8):
    groups = defaultdict(list)
    for index, row in enumerate(rows, 1):
        validate_row(row, index, max_variables)
        groups[group_key(row)].append(row)
    return {
        "schema_version": REPORT_SCHEMA,
        "group_count": len(groups),
        "groups": [
            analyze_group(key, value, reliability_floor, min_replicates, max_variables)
            for key, value in sorted(groups.items())
        ],
        "context_validity_definition": "A context is valid only when each selected declaration's mutable prerequisites are also selected and every fixed prerequisite is in the pinned fixed context.",
        "reliability_definition": "A context qualifies only when the two-sided 95% Wilson lower bound is at least reliability_floor and trial count is at least min_replicates.",
        "evidence_boundary": "Digests and row/schedule bindings are checked; referenced receipt bytes still require independent authentication.",
    }


def synthetic_rows(study, target, component, universe, dependencies, predicate,
                   budget, replicate_count=50, success_counts=None):
    fixed = ["fixed-base"]
    order = sorted(universe)
    graph = {name: sorted(dependencies.get(name, [])) for name in order}
    budget = dict(budget)
    hashes = {
        "solver_policy_sha256": sha256(f"{study}:{target}:policy".encode()),
        "budget_sha256": sha256(canonical(budget)),
        "target_statement_sha256": sha256(f"{study}:{target}:statement".encode()),
        "environment_sha256": sha256(f"{study}:{target}:environment".encode()),
        "toolchain_sha256": sha256(f"{study}:{target}:toolchain".encode()),
        "fixed_context_sha256": sha256(f"{study}:{target}:fixed".encode()),
    }
    rows = []
    for subset in sorted(powerset(order), key=lambda x: (len(x), x)):
        requested = list(subset)
        valid, _ = context_valid(requested, fixed, graph)
        ctx_digest = sha256(canonical([target, requested, fixed, graph]))
        ctx_receipt = sha256(f"context:{target}:{requested}".encode())
        for rep in range(replicate_count):
            passes = success_counts.get(subset, replicate_count) if success_counts else replicate_count
            success = valid and rep < passes and bool(predicate(set(subset)))
            outcome = "invalid_context" if not valid else ("verified_success" if success else "non_solve")
            rows.append({
                "schema_version": SCHEMA,
                "study_id": study, "component_id": component, "target_id": target,
                "model_id": "synthetic-agent", "model_version": "self-test-v1",
                "checker_version": "synthetic-checker",
                **hashes,
                "budget": budget,
                "replicate_count": replicate_count,
                "mutable_declarations": order,
                "fixed_declarations": fixed,
                "declaration_order": order,
                "declaration_dependencies": graph,
                "requested_declarations": requested,
                "context_well_formed": valid,
                "local_context_sha256": ctx_digest,
                "context_receipt_sha256": ctx_receipt,
                "replicate_index": rep,
                "seed_sha256": sha256(f"paired-seed-{rep}".encode()),
                "run_receipt_sha256": sha256(f"run:{target}:{subset}:{rep}".encode()),
                "checker_receipt_sha256": sha256(f"checker:{target}:{subset}:{rep}".encode()),
                "semantic_gate_receipt_sha256": sha256(f"semantic:{target}:{subset}:{rep}".encode()),
                "outcome": outcome,
                "solver_started": valid,
                "kernel_passed": success,
                "semantic_gate_passed": success,
                "model_calls": 1 if valid else 0,
                "tokens": 120 if valid else 0,
                "wall_ms": 40 if valid else 0,
                "checker_ms": 5 if success else (1 if valid else 0),
                "cost_microusd": 2 if valid else 0,
            })
    return rows


def self_test():
    budget = {"max_model_calls": 2, "max_tokens": 500, "max_wall_ms": 100,
              "max_checker_ms": 10, "max_cost_microusd": 10}
    alt = synthetic_rows("self", "alternative", "c1", ["d", "h", "hEq"], {},
                         lambda s: "h" in s or "hEq" in s, budget)
    conj = synthetic_rows("self", "conjunctive", "c2", ["h1", "h2", "noise"], {},
                          lambda s: {"h1", "h2"} <= s, budget)
    dep = synthetic_rows("self", "dependency", "c3", ["base", "dependent"],
                         {"dependent": ["base"]}, lambda s: "base" in s, budget)
    groups = []
    for rows in (alt, conj, dep):
        groups.extend(analyze(rows)["groups"])
    by_target = {g["target_id"]: g for g in groups}
    assert by_target["alternative"]["reliable_minimal_support_antichain"] == [["h"], ["hEq"]]
    assert by_target["conjunctive"]["reliable_minimal_support_antichain"] == [["h1", "h2"]]
    assert by_target["dependency"]["invalid_context_count"] == 1
    assert by_target["dependency"]["reliable_minimal_support_antichain"] == [["base"]]

    stochastic = synthetic_rows("self", "stochastic", "c4", ["h", "hEq"], {},
                                lambda s: bool(s), budget,
                                success_counts={(): 0, ("h",): 50, ("hEq",): 48, ("h", "hEq"): 50})
    sr = analyze(stochastic)["groups"][0]
    assert sr["reliable_minimal_support_antichain"] == [["h"], ["hEq"]]

    weak = synthetic_rows("self", "weak", "c5", ["h"], {}, lambda s: bool(s), budget,
                          success_counts={(): 0, ("h",): 45})
    wr = analyze(weak)["groups"][0]
    assert wr["qualification_status"] == "full_context_below_reliability_floor"
    assert wr["reliable_minimal_support_antichain"] == []

    def rejected(rows, label):
        try:
            analyze(rows)
        except ContractError:
            return
        raise AssertionError(f"negative control accepted: {label}")

    rejected(alt[:-1], "incomplete replicate lattice")
    wrong_seed = [dict(r) for r in alt]
    wrong_seed[-1]["seed_sha256"] = sha256(b"wrong")
    rejected(wrong_seed, "unpaired replicate seeds")
    wrong_fixed = [dict(r) for r in alt]
    wrong_fixed[-1]["fixed_declarations"] = ["other-fixed"]
    rejected(wrong_fixed, "changed fixed context")
    over_budget = [dict(r) for r in alt]
    over_budget[0]["tokens"] = 501
    rejected(over_budget, "over-budget run")
    wrong_gate = [dict(r) for r in alt]
    wrong_gate[0]["kernel_passed"] = False
    wrong_gate[0]["semantic_gate_passed"] = True
    wrong_gate[0]["outcome"] = "non_solve"
    rejected(wrong_gate, "semantic success without kernel success")
    false_context = [dict(r) for r in dep]
    target = next(r for r in false_context if r["requested_declarations"] == ["dependent"])
    target["context_well_formed"] = True
    try:
        validate_row(target, 999, 8)
    except ContractError:
        pass
    else:
        raise AssertionError("false dependency-closure claim accepted")
    print("PITB_SUPPORT_LANDSCAPE_SELF_TEST_PASS: alternatives, conjunction, dependency closure, stochastic reliability, Wilson gate, budget, paired-seed and completeness controls")


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
        if not 0.5 < args.reliability_floor < 1:
            raise ContractError("--reliability-floor must be in (0.5,1)")
        if args.min_replicates < 2:
            raise ContractError("--min-replicates must be >= 2")
        if not 1 <= args.max_variables <= 10:
            raise ContractError("--max-variables must be in [1,10]")
        if args.self_test:
            if args.outcomes or args.report:
                raise ContractError("--self-test cannot be combined with file paths")
            self_test()
            return 0
        if not args.outcomes or not args.report:
            raise ContractError("--outcomes and --report are required")
        rows = []
        for line_number, line in enumerate(args.outcomes.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ContractError(f"line {line_number}: {exc}") from exc
        for index, row in enumerate(rows, 1):
            if row.get("replicate_count", 0) < args.min_replicates:
                raise ContractError(f"row {index}: below minimum replicate count")
        if not rows:
            raise ContractError("empty outcome ledger")
        report = analyze(rows, args.reliability_floor, args.min_replicates, args.max_variables)
        report["outcomes_jsonl_sha256"] = sha256(args.outcomes.read_bytes())
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PITB_SUPPORT_LANDSCAPE_ANALYSIS: groups={report['group_count']} output={args.report}")
        return 0
    except (ContractError, OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"PITB support landscape rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
