#!/usr/bin/env python3
"""Exhaustive small-CNF state/oracle corpus for the D4v2 dynamic-BCE reference model.

Imports the independent model in verify_d4v2_dynamic_bce_trace.py. It is not a
D4 binary differential, full solver reimplementation, or formal theorem.
"""
from __future__ import annotations
import json
import random
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

from verify_d4v2_dynamic_bce_trace import D4BCEStateModel, evaluate

SEED = 0xE7030
SEEDED_FORMULAS = 250


def exact_weights(selected: list[int], salt: int) -> dict[str, dict[str, str]]:
    weights = {}
    for variable in selected:
        false_weight = Fraction(variable + 1 + (salt % 5), 2 * variable + 5 + (salt % 7))
        true_weight = Fraction(variable + 3 + (salt % 3), 3 * variable + 7 + (salt % 11))
        weights[str(variable)] = {
            "false": f"{false_weight.numerator}/{false_weight.denominator}",
            "true": f"{true_weight.numerator}/{true_weight.denominator}",
        }
    return weights


def all_non_taut_clauses(nvars: int) -> list[list[int]]:
    clauses = []
    for choices in product((0, -1, 1), repeat=nvars):
        clause = [sign * variable for variable, sign in enumerate(choices, 1) if sign]
        if clause:
            clauses.append(clause)
    return clauses


def clause_records(clauses: list[list[int]], prefix: str) -> list[dict]:
    return [{"id": f"{prefix}{idx}", "lits": clause.copy()} for idx, clause in enumerate(clauses)]


def check_case(clauses: list[dict], nvars: int, selected: list[int], salt: int) -> dict:
    weights = exact_weights(selected, salt)
    model = D4BCEStateModel(clauses, nvars, selected)
    clause_ids = [clause["id"] for clause in clauses]
    stats = {
        "decision_nodes": 0, "leaf_branches": 0, "backtracks": 0,
        "initial_deletions": sum(event[0] == "initial_blocked" for event in model.events),
        "watch_relocations": 0, "dynamic_deletions": 0,
        "root_weighted_equalities": 0, "leaf_weighted_equalities": 0,
    }

    root_source = evaluate(clauses, nvars, selected, weights)
    root_processed_ids = [cid for cid in clause_ids if model.active[cid]]
    root_processed = evaluate(clauses, nvars, selected, weights, active_ids=root_processed_ids)
    if root_source["projected_support"] != root_processed["projected_support"]:
        raise AssertionError("root projected support changed")
    if root_source["pwmc"] != root_processed["pwmc"]:
        raise AssertionError("root exact PWMC changed")
    stats["root_weighted_equalities"] += 1

    root_support = {tuple(row) for row in root_source["projected_support"]}
    leaf_source_union: set[tuple] = set()
    leaf_processed_union: set[tuple] = set()
    leaf_source_models = 0
    leaf_processed_models = 0

    def visit(depth: int, assignment: dict[int, bool]) -> None:
        nonlocal leaf_source_models, leaf_processed_models
        stats["decision_nodes"] += 1
        if depth == len(selected):
            stats["leaf_branches"] += 1
            source = evaluate(clauses, nvars, selected, weights, assignment=assignment, active_ids=clause_ids)
            active_ids = [cid for cid in clause_ids if model.active[cid]]
            processed = evaluate(clauses, nvars, selected, weights, assignment=assignment, active_ids=active_ids)
            if source["projected_support"] != processed["projected_support"]:
                raise AssertionError(f"leaf support changed: selected={selected}, assignment={assignment}")
            if source["pwmc"] != processed["pwmc"]:
                raise AssertionError(f"leaf exact PWMC changed: selected={selected}, assignment={assignment}")
            stats["leaf_weighted_equalities"] += 1
            leaf_source_models += source["full_models"]
            leaf_processed_models += processed["full_models"]
            leaf_source_union.update(tuple(row) for row in source["projected_support"])
            leaf_processed_union.update(tuple(row) for row in processed["projected_support"])
            return

        variable = selected[depth]
        for value in (False, True):
            before = model.snapshot()
            true_literal = variable if value else -variable
            after = model.pre_update([true_literal], f"v{variable}={int(value)}")
            for event in after["events"]:
                if event[0] == "watch_relocated":
                    stats["watch_relocations"] += 1
                elif event[0] == "blocked_by_exhausted_opposing_set":
                    stats["dynamic_deletions"] += 1
            next_assignment = dict(assignment)
            next_assignment[variable] = value
            visit(depth + 1, next_assignment)
            restored = model.post_update()
            stats["backtracks"] += 1
            if restored["active"] != before["active"]:
                raise AssertionError("postUpdate failed to restore clause active state")
            if restored["assignment"] != before["assignment"]:
                raise AssertionError("postUpdate failed to restore branch assignment")
            # Watchers intentionally persist across postUpdate. Validate the current sentinel invariant.
            model._assert_watch_invariant()

    visit(0, {})
    if leaf_source_union != root_support:
        raise AssertionError("source leaf support union does not equal root support")
    if leaf_processed_union != root_support:
        raise AssertionError("processed leaf support union does not equal root support")
    if leaf_source_models != root_source["full_models"]:
        raise AssertionError("source leaf model accounting does not equal root")
    return {
        "root_source_full_models": root_source["full_models"],
        "root_processed_full_models": root_processed["full_models"],
        "root_projected_support_size": root_source["projected_support_size"],
        "root_pwmc": root_source["pwmc"],
        "leaf_source_full_models": leaf_source_models,
        "leaf_processed_full_models": leaf_processed_models,
        "leaf_support_union_matches_root": True,
        "leaf_pwmc_matches_root": True,
        **stats,
    }


def main() -> None:
    exhaustive = {
        "cases": 0, "decision_nodes": 0, "leaf_branches": 0, "backtracks": 0,
        "initial_deletions": 0, "watch_relocations": 0, "dynamic_deletions": 0,
        "root_weighted_equalities": 0, "leaf_weighted_equalities": 0,
        "full_model_count_changed_cases": 0,
    }
    nvars = 2
    base_clauses = all_non_taut_clauses(nvars)
    if len(base_clauses) != 8:
        raise AssertionError("two-variable non-tautological clause universe changed")
    for formula_mask in range(1 << len(base_clauses)):
        chosen = [clause for idx, clause in enumerate(base_clauses) if formula_mask & (1 << idx)]
        clauses = clause_records(chosen, "e")
        for partition_mask in (1, 2, 3):
            selected = [v for v in range(1, nvars + 1) if partition_mask & (1 << (v - 1))]
            result = check_case(clauses, nvars, selected, salt=formula_mask * 3 + partition_mask)
            exhaustive["cases"] += 1
            for key in (
                "decision_nodes", "leaf_branches", "backtracks", "initial_deletions",
                "watch_relocations", "dynamic_deletions", "root_weighted_equalities",
                "leaf_weighted_equalities",
            ):
                exhaustive[key] += result[key]
            if result["root_source_full_models"] != result["root_processed_full_models"]:
                exhaustive["full_model_count_changed_cases"] += 1

    seeded = {
        "seed": SEED, "formula_count": SEEDED_FORMULAS, "cases": 0,
        "decision_nodes": 0, "leaf_branches": 0, "backtracks": 0,
        "initial_deletions": 0, "watch_relocations": 0, "dynamic_deletions": 0,
        "root_weighted_equalities": 0, "leaf_weighted_equalities": 0,
        "full_model_count_changed_cases": 0,
    }
    rng = random.Random(SEED)
    nvars = 3
    universe = all_non_taut_clauses(nvars)
    for formula_idx in range(SEEDED_FORMULAS):
        chosen = rng.sample(universe, rng.randrange(0, len(universe) + 1))
        clauses = clause_records(chosen, f"s{formula_idx}_")
        for partition_mask in range(1, 1 << nvars):
            selected = [v for v in range(1, nvars + 1) if partition_mask & (1 << (v - 1))]
            result = check_case(clauses, nvars, selected, salt=formula_idx * 17 + partition_mask)
            seeded["cases"] += 1
            for key in (
                "decision_nodes", "leaf_branches", "backtracks", "initial_deletions",
                "watch_relocations", "dynamic_deletions", "root_weighted_equalities",
                "leaf_weighted_equalities",
            ):
                seeded[key] += result[key]
            if result["root_source_full_models"] != result["root_processed_full_models"]:
                seeded["full_model_count_changed_cases"] += 1

    output = {
        "format": "symthaea.d4v2.dynamic-bce-exhaustive-model-results",
        "version": 1,
        "claim_ceiling": "local_independent_source_derived_state_model_plus_exhaustive_exact_projected_oracle_not_D4_binary_differential_or_formal_proof",
        "state_model": "D4BCEStateModel imported from sibling verifier; watches persist across postUpdate; missing watches are allowed only for initially inactive targets after constructor cleanup",
        "weights": "per-case exact positive rational literal weights; PWMC compared using fractions.Fraction",
        "exhaustive_2var": exhaustive,
        "seeded_3var": seeded,
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform},
        "execution": "uploaded_corpus_runner_requires_exact_byte_verified_execution",
    }
    result_path = Path(__file__).with_name("d4v2_dynamic_bce_exhaustive_model_results.v1.json")
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS exhaustive two-variable corpus:", exhaustive["cases"])
    print("PASS seeded three-variable corpus:", seeded["cases"])
    print("PASS exact projected support and PWMC comparisons:",
          exhaustive["root_weighted_equalities"] + exhaustive["leaf_weighted_equalities"] +
          seeded["root_weighted_equalities"] + seeded["leaf_weighted_equalities"])
    print("PASS backtracks:", exhaustive["backtracks"] + seeded["backtracks"])
    print("PASS watch relocations:", exhaustive["watch_relocations"] + seeded["watch_relocations"])


if __name__ == "__main__":
    main()
