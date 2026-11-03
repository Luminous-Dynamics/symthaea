#!/usr/bin/env python3
"""Finite exact Q(i) VE producer and deterministic replay-certificate oracle."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from pathlib import Path
import copy
import hashlib
import json
import random
import subprocess
import sys

FORMAT = "symthaea.qi-elimination-trace/v1"
ZERO = (Fraction(0), Fraction(0))
ONE = (Fraction(1), Fraction(0))


def qadd(left, right):
    return left[0] + right[0], left[1] + right[1]


def qmul(left, right):
    return (left[0] * right[0] - left[1] * right[1],
            left[0] * right[1] + left[1] * right[0])


def qencode(value):
    return [value[0].numerator, value[0].denominator,
            value[1].numerator, value[1].denominator]


def qdecode(value):
    if (not isinstance(value, list) or len(value) != 4 or
            any(type(part) is not int for part in value)):
        raise ValueError("Gaussian rational must contain four integer fields")
    components = []
    for offset in (0, 2):
        numerator, denominator = value[offset:offset + 2]
        if denominator <= 0:
            raise ValueError("rational denominator must be positive")
        rational = Fraction(numerator, denominator)
        if [rational.numerator, rational.denominator] != [numerator, denominator]:
            raise ValueError("rational encoding is not reduced/canonical")
        components.append(rational)
    return tuple(components)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_instance(instance):
    if not isinstance(instance, dict) or set(instance) != {"domains", "factors"}:
        raise ValueError("instance shape")
    domains, factors = instance["domains"], instance["factors"]
    if not isinstance(domains, dict) or not isinstance(factors, list):
        raise ValueError("instance types")
    for variable, cardinality in domains.items():
        if (not isinstance(variable, str) or not variable or
                type(cardinality) is not int or not 1 <= cardinality <= 4):
            raise ValueError("domain")
    seen = set()
    for factor in factors:
        if not isinstance(factor, dict) or set(factor) != {"id", "scope", "values"}:
            raise ValueError("factor shape")
        fid, scope, values = factor["id"], factor["scope"], factor["values"]
        if (not isinstance(fid, str) or not fid or fid in seen or
                fid.startswith(("free:", "join:", "sum:"))):
            raise ValueError("factor ID is duplicate or collides with a generated ID")
        seen.add(fid)
        if (not isinstance(scope, list) or
                any(not isinstance(v, str) for v in scope) or
                scope != sorted(set(scope)) or any(v not in domains for v in scope)):
            raise ValueError("factor scope")
        expected = 1
        for variable in scope:
            expected *= domains[variable]
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError("factor table size")
        for value in values:
            qdecode(value)
    return domains, factors


def factor_index(scope, assignment, domains):
    index = 0
    for variable in scope:
        index = index * domains[variable] + assignment[variable]
    return index


def factor_value(factor, assignment, domains):
    return qdecode(factor["values"][factor_index(factor["scope"], assignment, domains)])


def prepare_factors(instance):
    domains, factors = validate_instance(instance)
    work = copy.deepcopy(factors)
    mentioned = {variable for factor in factors for variable in factor["scope"]}
    for variable in sorted(set(domains) - mentioned):
        work.append({
            "id": f"free:{variable}",
            "scope": [variable],
            "values": [qencode(ONE) for _ in range(domains[variable])],
        })
    return domains, work


def producer_join(bucket, domains, factor_id):
    scope = sorted({variable for factor in bucket for variable in factor["scope"]})
    values = []
    for coordinates in product(*(range(domains[v]) for v in scope)):
        assignment = dict(zip(scope, coordinates))
        term = ONE
        for factor in bucket:
            term = qmul(term, factor_value(factor, assignment, domains))
        values.append(qencode(term))
    return {"id": factor_id, "scope": scope, "values": values}


def producer_sum_out(factor, variable, domains, factor_id):
    output_scope = [v for v in factor["scope"] if v != variable]
    values = []
    for coordinates in product(*(range(domains[v]) for v in output_scope)):
        assignment = dict(zip(output_scope, coordinates))
        subtotal = ZERO
        for state in range(domains[variable]):
            assignment[variable] = state
            subtotal = qadd(subtotal, factor_value(factor, assignment, domains))
        values.append(qencode(subtotal))
    return {"id": factor_id, "scope": output_scope, "values": values}


def produce_trace(instance, order):
    domains, work = prepare_factors(instance)
    if len(order) != len(domains) or set(order) != set(domains):
        raise ValueError("elimination order must be a permutation of all variables")
    initial = copy.deepcopy(work)
    steps = []
    for index, variable in enumerate(order):
        bucket = sorted(
            [factor for factor in work if variable in factor["scope"]],
            key=lambda factor: factor["id"],
        )
        if not bucket:
            raise ValueError("every variable must have an explicit bucket")
        joined = producer_join(bucket, domains, f"join:{index}:{variable}")
        generated = producer_sum_out(
            joined, variable, domains, f"sum:{index}:{variable}"
        )
        steps.append({
            "variable": variable,
            "bucket_ids": [factor["id"] for factor in bucket],
            "joined_scope": joined["scope"],
            "output_id": generated["id"],
            "output_scope": generated["scope"],
            "output_sha256": digest(generated),
        })
        work = [factor for factor in work if variable not in factor["scope"]]
        work.append(generated)

    if any(factor["scope"] for factor in work):
        raise AssertionError("non-scalar factor remains after all variables")
    final_factors = sorted(work, key=lambda factor: factor["id"])
    final_scalars = [
        {"id": factor["id"], "sha256": digest(factor)}
        for factor in final_factors
    ]
    result = ONE
    for factor in final_factors:
        if len(factor["values"]) != 1:
            raise AssertionError("scalar factor must have one table entry")
        result = qmul(result, qdecode(factor["values"][0]))
    encoded_result = qencode(result)
    return {
        "format": FORMAT,
        "instance": copy.deepcopy(instance),
        "instance_sha256": digest(instance),
        "initial_factors": [
            {"id": factor["id"], "sha256": digest(factor)} for factor in initial
        ],
        "order": list(order),
        "steps": steps,
        "final_scalars": final_scalars,
        "result": encoded_result,
        "result_sha256": digest(encoded_result),
    }


def direct_oracle(instance):
    domains, factors = validate_instance(instance)
    variables = sorted(domains)
    total = ZERO
    for coordinates in product(*(range(domains[v]) for v in variables)):
        assignment = dict(zip(variables, coordinates))
        term = ONE
        for factor in factors:
            term = qmul(term, factor_value(factor, assignment, domains))
        total = qadd(total, term)
    return total


def random_complex(rng):
    return (
        Fraction(rng.randrange(-2, 3), rng.randrange(1, 4)),
        Fraction(rng.randrange(-2, 3), rng.randrange(1, 4)),
    )


def generate_instance(rng):
    count = rng.randint(1, 5)
    domains = {f"x{i}": rng.choice((2, 3)) for i in range(count)}
    factors = []
    for index in range(rng.randint(0, 7)):
        width = rng.randint(0, min(3, count))
        scope = sorted(rng.sample(list(domains), width)) if width else []
        size = 1
        for variable in scope:
            size *= domains[variable]
        factors.append({
            "id": f"f{index}",
            "scope": scope,
            "values": [qencode(random_complex(rng)) for _ in range(size)],
        })
    return {"domains": domains, "factors": factors}


def main():
    rng = random.Random(0xE7022)
    network_count = 120
    orders_per_network = 4
    certificates = []
    direct_checks = 0
    mutation_base = None

    for index in range(network_count):
        instance = generate_instance(rng)
        variables = sorted(instance["domains"])
        orders = [
            variables,
            list(reversed(variables)),
            rng.sample(variables, len(variables)),
            sorted(variables, key=lambda v: sum(
                v in factor["scope"] for factor in instance["factors"]
            )),
        ]
        oracle = direct_oracle(instance)
        for order in orders:
            certificate = produce_trace(instance, order)
            assert qdecode(certificate["result"]) == oracle, (index, order)
            certificates.append(certificate)
            direct_checks += 1
        if mutation_base is None:
            mutation_base = produce_trace(instance, variables)

    assert mutation_base is not None
    mutations = []
    mutators = [
        lambda cert: cert.update(instance_sha256="0" * 64),
        lambda cert: cert["steps"][0].update(output_sha256="0" * 64),
        lambda cert: cert["steps"][0].update(bucket_ids=["forged"]),
        lambda cert: cert["result"].__setitem__(0, cert["result"][0] + 1),
        lambda cert: cert["order"].__setitem__(0, "unbound"),
        lambda cert: cert["final_scalars"].clear(),
        lambda cert: cert["steps"][0].update(joined_scope=["forged"]),
    ]
    for mutate in mutators:
        invalid = copy.deepcopy(mutation_base)
        mutate(invalid)
        mutations.append(invalid)

    payload = certificates + [mutation_base] + mutations
    checker = Path(__file__).with_name("qi_elimination_trace_checker.py")
    checked = subprocess.run(
        [sys.executable, str(checker)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    checker_result = json.loads(checked.stdout)
    positive_count = len(certificates) + 1
    assert checker_result == {
        "accepted": [True] * positive_count + [False] * len(mutations),
        "accepted_count": positive_count,
        "rejected_count": len(mutations),
    }, checker_result

    result = {
        "suite": "ETH-022 deterministic Q(i) elimination replay certificates",
        "seed": "0xE7022",
        "random_networks": network_count,
        "orders_per_network": orders_per_network,
        "direct_vs_variable_elimination_checks": direct_checks,
        "separate_process_checker": {
            "accepted_positive": checker_result["accepted_count"],
            "positive_total": positive_count,
            "rejected_negative": checker_result["rejected_count"],
            "negative_total": len(mutations),
        },
        "negative_controls": [
            "stale input digest",
            "forged intermediate output digest",
            "forged bucket membership",
            "tampered exact result",
            "unbound variable in elimination order",
            "missing final scalar binding",
            "forged joined scope",
        ],
        "certificate_contract": (
            "Input and output factors are content-addressed. Intermediate tables are "
            "not stored in the trace; the checker recomputes exact joins and sums."
        ),
        "all_passed": True,
        "claim_ceiling": (
            "Finite exact Q(i) direct-enumeration comparison and deterministic replay "
            "certificate checks. Not Rust, Lean, production parser, formal proof, "
            "bounded-memory streaming, or performance advantage."
        ),
    }
    Path(__file__).with_name("qi_elimination_trace_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
