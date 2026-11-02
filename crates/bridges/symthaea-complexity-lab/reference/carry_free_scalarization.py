#!/usr/bin/env python3
"""Carry-free scalar encoding for exact optimum and multiplicity.

For P(W)=sum_j c_j W^j with 0 <= c_j < B, P(B)'s base-B digits
are exactly the profile coefficients. Its highest nonzero digit gives the
maximum score and the coefficient gives the optimum multiplicity.

Research reference only: exact finite tests, not a subexponential solver.
"""
from __future__ import annotations
import json
from pathlib import Path


def highest_digit(value: int, base: int, exponent_bound: int):
    if value <= 0:
        return None
    lo, hi = 0, exponent_bound
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pow(base, mid) <= value:
            lo = mid
        else:
            hi = mid - 1
    return lo


def encode(profile: dict[int, int], base: int, exponent_bound: int):
    if base < 2 or any(c < 0 for c in profile.values()) or any(c >= base for c in profile.values()):
        raise ValueError("requires base >= 2 and every coefficient in [0, base)")
    value = sum(c * pow(base, e) for e, c in profile.items())
    top = highest_digit(value, base, exponent_bound)
    if top is None:
        return None, 0, value
    count = value // pow(base, top)
    if count >= base:
        raise AssertionError("carry detected")
    return top, count, value


def lex_code(vector, radices):
    return sum(value * __import__("math").prod(radices[i + 1:])
               for i, value in enumerate(vector))


def main():
    data = json.loads((Path(__file__).resolve().parents[1] / "weighted_profile_results.json").read_text())
    cases = {x["name"]: x for x in data["cases"]}
    results = []
    for name, nvars, exponent_bound in [
        ("triangle_max_cut", 3, 3),
        ("max_2sat_clause_profile", 3, 4),
        ("path4_independence_polynomial", 4, 4),
        ("even_parity_code_weight_enumerator", 3, 3),
    ]:
        profile = {int(k): v for k, v in cases[name]["direct_profile"].items()}
        base = (1 << nvars) + 1
        opt, count, scalar = encode(profile, base, exponent_bound)
        want_opt = max(profile)
        want_count = profile[want_opt]
        results.append({
            "name": name, "base": base,
            "decoded_optimum": opt, "decoded_multiplicity": count,
            "expected_optimum": want_opt, "expected_multiplicity": want_count,
            "scalar_bits": scalar.bit_length(),
            "passed": opt == want_opt and count == want_count,
        })

    knapsack = {tuple(map(int, k.split(","))): v
                for k, v in cases["two_resource_conflict_knapsack"]["direct_profile"].items()}
    radices = (3, 4)
    encoded = {lex_code(point, radices): count for point, count in knapsack.items()}
    base = 9
    code, count, scalar = encode(encoded, base, 11)
    vector = (code // radices[1], code % radices[1])
    expected = max(knapsack)
    results.append({
        "name": "two_resource_knapsack_lexicographic_optimum",
        "radices": radices, "base": base,
        "decoded_vector": vector, "decoded_multiplicity": count,
        "expected_vector": expected, "expected_multiplicity": knapsack[expected],
        "scalar_bits": scalar.bit_length(),
        "passed": vector == expected and count == knapsack[expected],
    })

    signed_rejected = small_base_rejected = False
    try:
        encode({0: 1, 1: -1}, 9, 1)
    except ValueError:
        signed_rejected = True
    try:
        encode({0: 1, 1: 4, 2: 3}, 4, 2)
    except ValueError:
        small_base_rejected = True

    result = {
        "suite": "ETH-014 carry-free scalarization exact micro-oracle",
        "passed": sum(x["passed"] for x in results) + int(signed_rejected) + int(small_base_rejected),
        "total": len(results) + 2,
        "cases": results,
        "negative_controls": {
            "signed_coefficients_rejected": signed_rejected,
            "insufficient_base_rejected": small_base_rejected,
        },
        "claim_boundary": "Finite exact encoding checks only; requires nonnegative coefficients and proven coefficient/base bounds; no runtime or asymptotic claim.",
    }
    out = Path(__file__).resolve().parents[1] / "carry_free_scalarization_results.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"carry-free scalarization: {result['passed']}/{result['total']} passed")
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
