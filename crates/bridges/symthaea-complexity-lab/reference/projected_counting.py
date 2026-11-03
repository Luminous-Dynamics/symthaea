#!/usr/bin/env python3
"""Finite projected model-counting and projected-weighted-counting oracle."""
from __future__ import annotations
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import random


def rational(value):
    if (not isinstance(value, list) or len(value) != 2 or
            any(not isinstance(x, int) or isinstance(x, bool) for x in value)):
        raise ValueError("rational shape")
    n, d = value
    if d <= 0:
        raise ValueError("denominator must be positive")
    q = Fraction(n, d)
    if [q.numerator, q.denominator] != value:
        raise ValueError("noncanonical rational")
    return q


def encode(q):
    q = Fraction(q)
    return [q.numerator, q.denominator]


def validate(instance):
    if not isinstance(instance, dict) or set(instance) != {
        "domains", "projected", "factors", "visible_weights"
    }:
        raise ValueError("instance shape")
    domains, projected, factors, weights = (
        instance["domains"], instance["projected"],
        instance["factors"], instance["visible_weights"]
    )
    if not isinstance(domains, dict) or not isinstance(projected, list):
        raise ValueError("domain/projection type")
    if (projected != sorted(set(projected)) or
            any(v not in domains for v in projected)):
        raise ValueError("projection must be sorted, unique and in-domain")
    for variable, cardinality in domains.items():
        if (not isinstance(variable, str) or not variable or
                not isinstance(cardinality, int) or isinstance(cardinality, bool) or
                cardinality != 2):
            raise ValueError("Boolean variable expected")
    if not isinstance(factors, list) or not isinstance(weights, dict):
        raise ValueError("factor/weight type")
    if set(weights) - set(projected):
        raise ValueError("only projected variables may receive visible weights")
    seen_ids = set()
    for factor in factors:
        if not isinstance(factor, dict) or set(factor) != {"id", "scope", "values"}:
            raise ValueError("factor shape")
        fid, scope, values = factor["id"], factor["scope"], factor["values"]
        if not isinstance(fid, str) or not fid or fid in seen_ids:
            raise ValueError("factor ID")
        seen_ids.add(fid)
        if (not isinstance(scope, list) or
                any(not isinstance(v, str) for v in scope) or
                scope != sorted(set(scope)) or
                any(v not in domains for v in scope)):
            raise ValueError("factor scope")
        if (not isinstance(values, list) or len(values) != 2 ** len(scope) or
                any(not isinstance(x, bool) for x in values)):
            raise ValueError("Boolean factor table")
    for variable, pair in weights.items():
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("visible unary weight table")
        for item in pair:
            rational(item)
    return domains, projected, factors, weights


def factor_value(factor, assignment):
    index = 0
    for variable in factor["scope"]:
        index = index * 2 + assignment[variable]
    return factor["values"][index]


def visible_weight(assignment, projected, weights):
    result = Fraction(1)
    for variable in projected:
        if variable in weights:
            result *= rational(weights[variable][assignment[variable]])
    return result


def direct_projection(instance):
    domains, projected, factors, weights = validate(instance)
    hidden = sorted(set(domains) - set(projected))
    total = Fraction(0)
    supported_count = 0
    for visible_values in product((0, 1), repeat=len(projected)):
        visible = dict(zip(projected, visible_values))
        has_witness = False
        for hidden_values in product((0, 1), repeat=len(hidden)):
            assignment = visible | dict(zip(hidden, hidden_values))
            if all(factor_value(f, assignment) for f in factors):
                has_witness = True
                break
        if has_witness:
            supported_count += 1
            total += visible_weight(visible, projected, weights)
    return total, supported_count


def join_boolean(factors):
    scope = sorted({v for factor in factors for v in factor["scope"]})
    table = []
    for values in product((0, 1), repeat=len(scope)):
        assignment = dict(zip(scope, values))
        table.append(all(factor_value(f, assignment) for f in factors))
    return {"id": "joined", "scope": scope, "values": table}


def existentially_eliminate(factor, variable):
    scope = [v for v in factor["scope"] if v != variable]
    table = []
    for values in product((0, 1), repeat=len(scope)):
        assignment = dict(zip(scope, values))
        supported = False
        for hidden_value in (0, 1):
            assignment[variable] = hidden_value
            supported = supported or factor_value(factor, assignment)
        table.append(supported)
    return {"id": f"exists_{variable}", "scope": scope, "values": table}


def mixed_elimination(instance, hidden_order):
    domains, projected, factors, weights = validate(instance)
    hidden = set(domains) - set(projected)
    if len(hidden_order) != len(hidden) or set(hidden_order) != hidden:
        raise ValueError("order must permute only existential variables")
    work = list(factors)
    # Phase one: all hidden variables are existentially eliminated in Boolean algebra.
    for variable in hidden_order:
        bucket = [f for f in work if variable in f["scope"]]
        if not bucket:
            continue
        work = [f for f in work if variable not in f["scope"]]
        work.append(existentially_eliminate(join_boolean(bucket), variable))
    # Phase two: no hidden variables remain; combine support and sum distinct projections.
    predicate = (
        join_boolean(work) if work
        else {"id": "true", "scope": [], "values": [True]}
    )
    total = Fraction(0)
    supported_count = 0
    for visible_values in product((0, 1), repeat=len(projected)):
        assignment = dict(zip(projected, visible_values))
        if factor_value(predicate, assignment):
            supported_count += 1
            total += visible_weight(assignment, projected, weights)
    return total, supported_count


def generate_instance(rng):
    variable_count = rng.randrange(1, 7)
    variables = [f"v{i}" for i in range(variable_count)]
    projected = sorted(rng.sample(
        variables, rng.randrange(0, variable_count + 1)
    ))
    factors = []
    for index in range(rng.randrange(1, 9)):
        width = rng.randrange(0, min(4, variable_count) + 1)
        scope = sorted(rng.sample(variables, width)) if width else []
        factors.append({
            "id": f"f{index}",
            "scope": scope,
            "values": [bool(rng.randrange(2)) for _ in range(2 ** len(scope))],
        })
    weights = {
        variable: [
            encode(Fraction(rng.randrange(-3, 4), rng.randrange(1, 4)))
            for _ in range(2)
        ]
        for variable in projected if rng.randrange(2)
    }
    return {
        "domains": {v: 2 for v in variables},
        "projected": projected,
        "factors": factors,
        "visible_weights": weights,
    }


def main():
    rng = random.Random(0xE7021)
    cases = 160
    checks = 0
    order_counts = {"forward": 0, "reverse": 0, "random": 0}
    for index in range(cases):
        instance = generate_instance(rng)
        domains, projected, _, _ = validate(instance)
        hidden = sorted(set(domains) - set(projected))
        expected = direct_projection(instance)
        orders = {
            "forward": hidden,
            "reverse": list(reversed(hidden)),
            "random": rng.sample(hidden, len(hidden)),
        }
        for name, order in orders.items():
            actual = mixed_elimination(instance, order)
            assert actual == expected, (index, name, actual, expected)
            checks += 1
            order_counts[name] += 1

    # One visible assignment has two hidden witnesses: #SAT counts two,
    # while PMC counts that visible assignment exactly once.
    witness_fixture = {
        "domains": {"x": 2, "y": 2},
        "projected": ["x"],
        "factors": [
            {"id": "x_true", "scope": ["x"], "values": [False, True]},
            {"id": "free_y", "scope": ["y"], "values": [True, True]},
        ],
        "visible_weights": {},
    }
    projected_result = direct_projection(witness_fixture)
    ordinary_count = sum(
        1 for x, y in product((0, 1), repeat=2) if x == 1
    )
    assert projected_result == (Fraction(1), 1)
    assert ordinary_count == 2

    results = {
        "suite": "ETH-021 projected-count mixed-elimination reference",
        "seed": "0xE7021",
        "random_instances": cases,
        "mixed_order_checks": checks,
        "orders": order_counts,
        "visible_weight_domain": "exact Q rational, including negative values",
        "hidden_quantifier": "existential Boolean OR",
        "projected_accumulator": "exact rational sum over distinct supported visible assignments",
        "multiple_witness_control": {
            "ordinary_satisfying_assignments": ordinary_count,
            "projected_assignments": projected_result[1],
            "projected_weighted_value": str(projected_result[0]),
        },
        "all_passed": True,
        "claim_ceiling": (
            "Finite Boolean-factor PMC/PWMC-style semantics and exact rational "
            "visible weights. Not the complete MC2026 parser, general projected "
            "weight semantics, Rust/Lean proof or performance claim."
        ),
    }
    Path(__file__).with_name("projected_counting_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
