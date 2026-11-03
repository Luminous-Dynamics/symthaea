#!/usr/bin/env python3
"""Independent exact assignment-set oracle for typed circuit IR conformance."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import importlib.util
import json
import random

SOURCE = Path(__file__).with_name("typed_circuit_ir.py")
spec = importlib.util.spec_from_file_location("typed_circuit_ir", SOURCE)
ir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir)


def qadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def random_q(rng, positive=False):
    numerator = rng.randint(1 if positive else -4, 4)
    denominator = rng.randint(1, 5)
    return F(numerator, denominator)


def random_qi(rng):
    return random_q(rng), random_q(rng)


def build_case(rng, mode):
    n = rng.randint(1, 5)
    variables = [f"x{i}" for i in range(n)]
    all_assignments = list(product((False, True), repeat=n))
    rng.shuffle(all_assignments)
    selected = all_assignments[:rng.randint(1, len(all_assignments))]
    visible = sorted(rng.sample(variables, rng.randint(1, n))) if mode in {"pmc", "pwmc"} else []
    domain = {"#sat": "bool", "pmc": "bool", "pwmc": "bool", "wmc": "q", "amc-complex": "qi"}[mode]
    nodes, terms, weights = [], [], []
    for row, assignment in enumerate(selected):
        children = []
        for column, (variable, value) in enumerate(zip(variables, assignment)):
            node_id = f"i{row}_{column}"
            nodes.append({"id": node_id, "kind": "indicator", "variable": variable, "value": value})
            children.append(node_id)
        product_id = f"p{row}"
        nodes.append({"id": product_id, "kind": "product", "children": children})
        coefficient = True if mode in {"#sat", "pmc", "pwmc"} else (
            random_q(rng) if mode == "wmc" else random_qi(rng)
        )
        wire = coefficient
        if isinstance(coefficient, F):
            wire = [coefficient.numerator, coefficient.denominator]
        elif isinstance(coefficient, tuple):
            wire = {
                "re": [coefficient[0].numerator, coefficient[0].denominator],
                "im": [coefficient[1].numerator, coefficient[1].denominator],
            }
        terms.append({"child": product_id, "weight": wire})
        weights.append(coefficient)
    nodes.append({"id": "root", "kind": "sum", "deterministic": True, "terms": terms})
    circuit = {"schema": ir.SCHEMA, "contract": mode, "variables": variables, "visible": visible,
               "domain": domain, "nodes": nodes, "root": "root"}
    visible_weights = {}
    if mode == "pwmc":
        for variable in visible:
            false_weight = random_q(rng, positive=True)
            true_weight = random_q(rng, positive=True)
            circuit.setdefault("visible_weights", {})[variable] = [
                [false_weight.numerator, false_weight.denominator],
                [true_weight.numerator, true_weight.denominator],
            ]
            visible_weights[variable] = (false_weight, true_weight)
    return circuit, selected, weights, visible_weights


def oracle(mode, selected, weights, visible, visible_weights, variables):
    positions = [i for i, v in enumerate(variables) if v in visible]
    if mode == "#sat":
        return str(len(selected))
    if mode == "pmc":
        return str(len({tuple(row[i] for i in positions) for row in selected}))
    if mode == "pwmc":
        projections = {tuple(row[i] for i in positions) for row in selected}
        total = F(0)
        for projection in projections:
            contribution = F(1)
            for variable, state in zip(visible, projection):
                contribution *= visible_weights[variable][1 if state else 0]
            total += contribution
        return str(total)
    if mode == "wmc":
        return str(sum(weights, F(0)))
    total = (F(0), F(0))
    for value in weights:
        total = qadd(total, value)
    if total[1] == 0:
        return str(total[0])
    return f"{total[0]}{'+' if total[1] >= 0 else ''}{total[1]}i"


def run(seed=0xE7026, cases_per_mode=100):
    rng = random.Random(seed)
    modes = ["#sat", "wmc", "pmc", "pwmc", "amc-complex"]
    passed = {mode: 0 for mode in modes}
    failures = []
    for mode in modes:
        for case_index in range(cases_per_mode):
            circuit, selected, weights, visible_weights = build_case(rng, mode)
            expected = oracle(mode, selected, weights, circuit["visible"], visible_weights, circuit["variables"])
            actual = ir.evaluate(circuit)["exact_output"]
            if actual == expected:
                passed[mode] += 1
            else:
                failures.append({"mode": mode, "case": case_index, "expected": expected, "actual": actual})

    negative = {"overlapping_product": 0, "false_determinism": 0, "domain_mismatch": 0}
    bad = {"schema": ir.SCHEMA, "contract": "#sat", "variables": ["x"], "visible": [], "domain": "bool",
           "nodes": [{"id": "a", "kind": "indicator", "variable": "x", "value": True},
                     {"id": "b", "kind": "indicator", "variable": "x", "value": True},
                     {"id": "s", "kind": "sum", "deterministic": True,
                      "terms": [{"child": "a", "weight": True}, {"child": "b", "weight": True}]}], "root": "s"}
    try:
        ir.evaluate(bad)
    except ir.Invalid:
        negative["false_determinism"] += 1
    else:
        failures.append({"negative": "false_determinism", "error": "accepted"})

    bad = {"schema": ir.SCHEMA, "contract": "#sat", "variables": ["x"], "visible": [], "domain": "bool",
           "nodes": [{"id": "a", "kind": "indicator", "variable": "x", "value": True},
                     {"id": "p", "kind": "product", "children": ["a", "a"]}], "root": "p"}
    try:
        ir.evaluate(bad)
    except ir.Invalid:
        negative["overlapping_product"] += 1
    else:
        failures.append({"negative": "overlapping_product", "error": "accepted"})

    bad = {"schema": ir.SCHEMA, "contract": "amc-complex", "variables": ["x"], "visible": [], "domain": "bool",
           "nodes": [{"id": "a", "kind": "indicator", "variable": "x", "value": True}], "root": "a"}
    try:
        ir.evaluate(bad)
    except ir.Invalid:
        negative["domain_mismatch"] += 1
    else:
        failures.append({"negative": "domain_mismatch", "error": "accepted"})

    total = cases_per_mode * len(modes)
    rejected = sum(negative.values())
    return {
        "suite": "ETH-026 randomized typed-circuit oracle conformance",
        "python": "3.13.5",
        "seed": hex(seed),
        "cases_per_mode": cases_per_mode,
        "positive_cases": total,
        "positive_passed": sum(passed.values()),
        "passes_by_mode": passed,
        "negative_controls": negative,
        "negative_rejected": rejected,
        "failure_count": len(failures),
        "failures": failures,
        "all_passed": not failures and sum(passed.values()) == total and rejected == 3,
        "claim_ceiling": "Random small deterministic/decomposable circuits checked against an independent assignment-set oracle. Not a proof-format adapter, formal proof, solver differential, large-scale benchmark or SOTA result."
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
