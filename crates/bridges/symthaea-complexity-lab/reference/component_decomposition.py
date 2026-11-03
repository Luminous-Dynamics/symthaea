#!/usr/bin/env python3
"""Finite producer/oracle for exact disconnected factor-graph decomposition."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import copy
import hashlib
import json
import random
import subprocess
import sys

FORMAT = "symthaea.factor-components/v1"
ALGEBRAS = {"bool", "nat", "maxcount", "pareto", "ipoly", "qpoly"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_instance(instance):
    if not isinstance(instance, dict) or set(instance) != {"domains", "factors"}:
        raise ValueError("instance shape")
    domains, factors = instance["domains"], instance["factors"]
    if not isinstance(domains, dict) or not isinstance(factors, list):
        raise ValueError("domains/factors types")
    for variable, cardinality in domains.items():
        if (not isinstance(variable, str) or not variable or
                not isinstance(cardinality, int) or isinstance(cardinality, bool) or
                not 1 <= cardinality <= 8):
            raise ValueError("invalid domain")
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
        table_size = 1
        for variable in scope:
            table_size *= domains[variable]
        if (not isinstance(values, list) or len(values) != table_size or
                any(not isinstance(x, int) or isinstance(x, bool) or x < 0
                    for x in values)):
            raise ValueError("factor table")
    return domains, factors


def produce_components(instance):
    domains, factors = validate_instance(instance)
    # Producer algorithm: union-find over primal variable adjacency.
    parent = {v: v for v in domains}

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for factor in factors:
        scope = factor["scope"]
        for variable in scope[1:]:
            union(scope[0], variable)

    groups = {}
    for variable in sorted(domains):
        groups.setdefault(find(variable), []).append(variable)

    components = []
    for variables in groups.values():
        variable_set = set(variables)
        factor_ids = sorted(
            f["id"] for f in factors
            if f["scope"] and set(f["scope"]) <= variable_set
        )
        components.append({"variables": variables, "factors": factor_ids})

    # Zero-scope factors are independent scalar components.
    for factor in factors:
        if not factor["scope"]:
            components.append({"variables": [], "factors": [factor["id"]]})

    return sorted(components, key=lambda c: (c["variables"], c["factors"]))


def build_certificate(instance, algebra):
    if algebra not in ALGEBRAS:
        raise ValueError("unqualified algebra")
    return {
        "format": FORMAT,
        "algebra": algebra,
        "instance": instance,
        "instance_sha256": digest(instance),
        "components": produce_components(instance),
    }


def algebra_ops(kind):
    if kind == "bool":
        return False, True, lambda a, b: a or b, lambda a, b: a and b
    if kind == "nat":
        return 0, 1, lambda a, b: a + b, lambda a, b: a * b
    if kind == "maxcount":
        def add(a, b):
            if a is None:
                return b
            if b is None:
                return a
            if a[0] > b[0]:
                return a
            if b[0] > a[0]:
                return b
            return (a[0], a[1] + b[1])

        def mul(a, b):
            if a is None or b is None:
                return None
            return (a[0] + b[0], a[1] * b[1])

        return None, (0, 1), add, mul
    raise ValueError("semantic oracle supports bool/nat/maxcount")


def factor_value(factor, assignment, domains):
    index = 0
    for variable in factor["scope"]:
        index = index * domains[variable] + assignment[variable]
    return factor["values"][index]


def semiring_weight(kind, weight):
    if kind == "bool":
        return weight != 0
    if kind == "nat":
        return weight
    if kind == "maxcount":
        return None if weight == 0 else (weight, 1)
    raise ValueError(kind)


def direct_partition(instance, kind):
    domains, factors = validate_instance(instance)
    zero, one, add, mul = algebra_ops(kind)
    variables = sorted(domains)
    total = zero
    for values in product(*(range(domains[v]) for v in variables)):
        assignment = dict(zip(variables, values))
        term = one
        for factor in factors:
            term = mul(term, semiring_weight(
                kind, factor_value(factor, assignment, domains)))
        total = add(total, term)
    return total


def decomposed_partition(instance, components, kind):
    domains, factors = validate_instance(instance)
    factor_by_id = {f["id"]: f for f in factors}
    zero, one, add, mul = algebra_ops(kind)
    combined = one
    for component in components:
        variables = component["variables"]
        local_factors = [factor_by_id[fid] for fid in component["factors"]]
        subtotal = zero
        for values in product(*(range(domains[v]) for v in variables)):
            assignment = dict(zip(variables, values))
            term = one
            for factor in local_factors:
                term = mul(term, semiring_weight(
                    kind, factor_value(factor, assignment, domains)))
            subtotal = add(subtotal, term)
        combined = mul(combined, subtotal)
    return combined


def generate_instance(rng):
    variable_count = rng.randrange(0, 8)
    domains = {f"v{i}": rng.randrange(2, 4) for i in range(variable_count)}
    factors = []
    for index in range(rng.randrange(0, 11)):
        width = rng.randrange(0, min(4, variable_count) + 1) if variable_count else 0
        scope = sorted(rng.sample(list(domains), width)) if width else []
        size = 1
        for variable in scope:
            size *= domains[variable]
        factors.append({
            "id": f"f{index}",
            "scope": scope,
            "values": [rng.randrange(0, 4) for _ in range(size)],
        })
    return {"domains": domains, "factors": factors}


def main():
    rng = random.Random(0xE7019)
    structural_count = 1000
    semantic_count = 240
    max_components = 0

    # Structural corpus is deliberately separate from semantic oracles.
    for _ in range(structural_count):
        instance = generate_instance(rng)
        components = produce_components(instance)
        build_certificate(instance, "nat")
        max_components = max(max_components, len(components))

    certificates = []
    semantic_rng = random.Random(0xE7019 ^ 0x55AA)
    semantic_checks = 0
    for _ in range(semantic_count):
        instance = generate_instance(semantic_rng)
        components = produce_components(instance)
        for kind in ("bool", "nat", "maxcount"):
            direct = direct_partition(instance, kind)
            factored = decomposed_partition(instance, components, kind)
            assert direct == factored, (kind, direct, factored)
            certificates.append(build_certificate(instance, kind))
            semantic_checks += 1

    special = [
        {"domains": {}, "factors": []},
        {"domains": {"x": 3, "y": 2}, "factors": []},
        {
            "domains": {"x": 2, "y": 2, "z": 2},
            "factors": [
                {"id": "xy", "scope": ["x", "y"], "values": [1, 0, 2, 1]},
                {"id": "z", "scope": ["z"], "values": [3, 1]},
                {"id": "scalar", "scope": [], "values": [2]},
            ],
        },
    ]
    for instance in special:
        components = produce_components(instance)
        for kind in ("bool", "nat", "maxcount"):
            assert direct_partition(instance, kind) == decomposed_partition(
                instance, components, kind)
            certificates.append(build_certificate(instance, kind))

    base = build_certificate(special[-1], "nat")
    negatives = []

    bad = copy.deepcopy(base)
    bad["instance_sha256"] = "0" * 64
    negatives.append(bad)

    bad = copy.deepcopy(base)
    bad["components"][0]["variables"].append("bogus")
    negatives.append(bad)

    bad = copy.deepcopy(base)
    bad["components"].pop()
    negatives.append(bad)

    bad = copy.deepcopy(base)
    bad["algebra"] = "unknown"
    negatives.append(bad)

    bad = copy.deepcopy(base)
    bad["components"][0]["factors"].append(bad["components"][0]["factors"][0])
    negatives.append(bad)

    bad = copy.deepcopy(base)
    bad["components"][0], bad["components"][1] = (
        bad["components"][1], bad["components"][0])
    negatives.append(bad)

    structural_certs = [
        build_certificate(generate_instance(rng), "nat")
        for _ in range(structural_count)
    ]
    payload = certificates + structural_certs + negatives
    checker = Path(__file__).with_name("component_decomposition_checker.py")
    process = subprocess.run(
        [sys.executable, str(checker)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    checker_result = json.loads(process.stdout)
    positive_total = len(payload) - len(negatives)
    expected = {
        "accepted": [True] * positive_total + [False] * len(negatives),
        "accepted_count": positive_total,
        "rejected_count": len(negatives),
    }
    assert checker_result == expected, checker_result

    results = {
        "suite": "ETH-019 canonical factor-component witness reference",
        "seed": "0xE7019",
        "structural_random_instances": structural_count,
        "semantic_random_instances": semantic_count,
        "semantic_algebra_checks": semantic_checks,
        "semantic_algebras": ["bool", "nat", "maxcount"],
        "special_semantic_controls": len(special) * 3,
        "separate_process_checker": {
            "accepted": checker_result["accepted_count"],
            "positive_total": positive_total,
            "rejected": checker_result["rejected_count"],
            "negative_total": len(negatives),
        },
        "negative_controls": [
            "stale instance digest",
            "unknown variable",
            "missing component coverage",
            "unknown algebra",
            "duplicate factor assignment",
            "noncanonical component order",
        ],
        "max_components_seen": max_components,
        "all_passed": True,
        "claim_ceiling": (
            "Small finite structural witness and Boolean/natural/MaxCount "
            "exhaustive semantic tests. Not generalized/fractional hypertree "
            "width, tensor planner, Rust, Lean proof, or performance claim."
        ),
    }
    Path(__file__).with_name("component_decomposition_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
