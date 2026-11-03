#!/usr/bin/env python3
"""Independent Q(i) VE replay checker. Imports no producer code."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import hashlib
import json
import sys

FORMAT = "symthaea.qi-elimination-trace/v1"
ZERO = (Fraction(0), Fraction(0))
ONE = (Fraction(1), Fraction(0))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def decode_rational_pair(pair):
    if (not isinstance(pair, list) or len(pair) != 2 or
            any(type(part) is not int for part in pair)):
        raise ValueError("rational pair")
    numerator, denominator = pair
    if denominator <= 0:
        raise ValueError("denominator")
    value = Fraction(numerator, denominator)
    if [value.numerator, value.denominator] != pair:
        raise ValueError("noncanonical rational")
    return value


def decode_gaussian(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Gaussian-rational encoding")
    return decode_rational_pair(value[:2]), decode_rational_pair(value[2:])


def add_complex(left, right):
    return left[0] + right[0], left[1] + right[1]


def multiply_complex(left, right):
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def encode_gaussian(value):
    return [
        value[0].numerator, value[0].denominator,
        value[1].numerator, value[1].denominator,
    ]


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
            decode_gaussian(value)
    return domains, factors


def read_factor_value(factor, assignment, domains):
    # The checker independently reconstructs the flat row index.
    offset = 0
    for variable in factor["scope"]:
        offset = offset * domains[variable] + assignment[variable]
    return decode_gaussian(factor["values"][offset])


def check_certificate(certificate):
    try:
        required = {
            "format", "instance", "instance_sha256", "initial_factors",
            "order", "steps", "final_scalars", "result", "result_sha256",
        }
        if not isinstance(certificate, dict) or set(certificate) != required:
            return False
        if certificate["format"] != FORMAT:
            return False

        instance = certificate["instance"]
        domains, original_factors = validate_instance(instance)
        if digest(instance) != certificate["instance_sha256"]:
            return False

        # Reconstruct implicit unit factors for variables absent from all scopes.
        work = [dict(factor) for factor in original_factors]
        mentioned = {variable for factor in original_factors for variable in factor["scope"]}
        for variable in sorted(set(domains) - mentioned):
            work.append({
                "id": f"free:{variable}",
                "scope": [variable],
                "values": [encode_gaussian(ONE) for _ in range(domains[variable])],
            })
        expected_initial = [
            {"id": factor["id"], "sha256": digest(factor)} for factor in work
        ]
        if certificate["initial_factors"] != expected_initial:
            return False

        order = certificate["order"]
        if (not isinstance(order, list) or len(order) != len(domains) or
                set(order) != set(domains) or len(set(order)) != len(order)):
            return False
        steps = certificate["steps"]
        if not isinstance(steps, list) or len(steps) != len(order):
            return False

        for index, (variable, step) in enumerate(zip(order, steps)):
            expected_step_keys = {
                "variable", "bucket_ids", "joined_scope",
                "output_id", "output_scope", "output_sha256",
            }
            if not isinstance(step, dict) or set(step) != expected_step_keys:
                return False
            if step["variable"] != variable:
                return False

            bucket = sorted(
                (factor for factor in work if variable in factor["scope"]),
                key=lambda factor: factor["id"],
            )
            bucket_ids = [factor["id"] for factor in bucket]
            if not bucket or step["bucket_ids"] != bucket_ids:
                return False

            joined_scope = sorted({
                item for factor in bucket for item in factor["scope"]
            })
            output_scope = [item for item in joined_scope if item != variable]
            output_id = f"sum:{index}:{variable}"
            if (step["joined_scope"] != joined_scope or
                    step["output_scope"] != output_scope or
                    step["output_id"] != output_id):
                return False

            # Independent fused bucket replay:
            # each output entry is sum_x product_{f in bucket} f.
            output_values = []
            for coordinates in product(*(range(domains[v]) for v in output_scope)):
                assignment = dict(zip(output_scope, coordinates))
                subtotal = ZERO
                for state in range(domains[variable]):
                    assignment[variable] = state
                    term = ONE
                    for factor in bucket:
                        term = multiply_complex(
                            term, read_factor_value(factor, assignment, domains)
                        )
                    subtotal = add_complex(subtotal, term)
                output_values.append(encode_gaussian(subtotal))

            generated = {
                "id": output_id,
                "scope": output_scope,
                "values": output_values,
            }
            if digest(generated) != step["output_sha256"]:
                return False

            work = [factor for factor in work if variable not in factor["scope"]]
            work.append(generated)

        if any(factor["scope"] for factor in work):
            return False
        final_factors = sorted(work, key=lambda factor: factor["id"])
        expected_scalars = [
            {"id": factor["id"], "sha256": digest(factor)}
            for factor in final_factors
        ]
        if certificate["final_scalars"] != expected_scalars:
            return False

        result = ONE
        for factor in final_factors:
            if len(factor["values"]) != 1:
                return False
            result = multiply_complex(
                result, decode_gaussian(factor["values"][0])
            )
        encoded_result = encode_gaussian(result)
        if certificate["result"] != encoded_result:
            return False
        if digest(encoded_result) != certificate["result_sha256"]:
            return False
        return True
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return False


if __name__ == "__main__":
    certificates = json.load(sys.stdin)
    accepted = [check_certificate(item) for item in certificates]
    print(json.dumps({
        "accepted": accepted,
        "accepted_count": sum(accepted),
        "rejected_count": len(accepted) - sum(accepted),
    }))
