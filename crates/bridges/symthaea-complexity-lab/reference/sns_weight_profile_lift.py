#!/usr/bin/env python3
"""Independent exact K4 SNS identity over Z[X,Y]/(X^3,Y^3).

Research oracle only: standard library, arbitrary-precision integers, exact
sparse polynomial convolution. This is not an asymptotic solver or Lean proof.
"""
from itertools import product
import json
import random
from pathlib import Path

BOUNDS = (3, 3)
ZERO = {}
ONE = {(0, 0): 1}

def add(a, b):
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, 0) + c
    return {e: c for e, c in out.items() if c}

def scale(a, k):
    return {e: c * k for e, c in a.items() if c * k}

def mul(a, b):
    out = {}
    for (ax, ay), ac in a.items():
        for (bx, by), bc in b.items():
            e = (ax + bx, ay + by)
            if e[0] < BOUNDS[0] and e[1] < BOUNDS[1]:
                out[e] = out.get(e, 0) + ac * bc
    return {e: c for e, c in out.items() if c}

def exact_half(a):
    if any(c % 2 for c in a.values()):
        return None
    return {e: c // 2 for e, c in a.items() if c // 2}

def sigma(p, q):
    P = (p & 1, (p >> 1) & 1, (p >> 2) & 1)
    Q = (q & 1, (q >> 1) & 1, (q >> 2) & 1)
    e = sum(P[i] * Q[j] for i in range(3) for j in range(i, 3))
    e += Q[0] * P[1] * P[2] + P[0] * Q[1] * P[2] + P[0] * P[1] * Q[2]
    return -1 if e & 1 else 1

def flat3(a, i, j, k):
    return a[(i * 8 + j) * 8 + k]

def check_network(seed, wrong_top=False, wrong_bottom=False):
    rng = random.Random(seed)
    choices = [ZERO, ONE, {(1, 0): 1}, {(0, 1): 1},
               {(1, 0): 1, (0, 1): -1}, {(1, 1): 1}]
    Wa, Wa2, Wb, Wb2 = [
        [choices[rng.randrange(len(choices))] for _ in range(512)]
        for _ in range(4)
    ]

    P = [ZERO] * 4096
    Pb = [ZERO] * 4096
    for a, b, x, y in product(range(8), repeat=4):
        idx = ((a * 8 + b) * 8 + x) * 8 + y
        ta, tb = ZERO, ZERO
        for s in range(8):
            ta = add(ta, mul(flat3(Wa, s, a, x), flat3(Wa2, s, b, y)))
            tb = add(tb, mul(flat3(Wb, s, a, x), flat3(Wb2, s, b, y)))
        P[idx], Pb[idx] = ta, tb

    def get(t, a, b, x, y):
        return t[((a * 8 + b) * 8 + x) * 8 + y]

    original = ZERO
    for a, b, x, y in product(range(8), repeat=4):
        original = add(original, mul(get(P, a, b, x, y), get(Pb, a, b, y, x)))

    top, bottom = [ZERO] * 512, [ZERO] * 512
    for a, b, x, y in product(range(8), repeat=4):
        k = x ^ y
        sign = sigma(y, x) if wrong_top else sigma(x, y)
        idx = (a * 8 + b) * 8 + k
        top[idx] = add(top[idx], scale(get(P, a, b, x, y), sign))
    for a, b, u, v in product(range(8), repeat=4):
        k = u ^ v
        sign = sigma(u, v) if wrong_bottom else sigma(v, u)
        idx = (a * 8 + b) * 8 + k
        bottom[idx] = add(bottom[idx], scale(get(Pb, a, b, u, v), sign))

    row = ZERO
    for i in range(512):
        row = add(row, mul(top[i], bottom[i]))

    column = ZERO
    for a, b, x, y in product(range(8), repeat=4):
        inner = ZERO
        for u, v in product(range(8), repeat=2):
            if x ^ y ^ u ^ v == 0:
                inner = add(inner, scale(get(Pb, a, b, u, v),
                                          sigma(x, u) * sigma(v, y)))
        column = add(column, mul(get(P, a, b, x, y), inner))

    lhs = scale(original, 2)
    rhs = add(row, column)
    recovered = exact_half(rhs)
    return {
        "accepted": lhs == rhs,
        "half_recovers_original": recovered == original,
        "lhs": {f"{x},{y}": c for (x, y), c in sorted(lhs.items())},
        "rhs": {f"{x},{y}": c for (x, y), c in sorted(rhs.items())},
    }

def main():
    valid = [check_network(seed) for seed in range(6)]
    wrong_top, wrong_bottom = check_network(0, True, False), check_network(0, False, True)
    assert all(r["accepted"] and r["half_recovers_original"] for r in valid)
    assert not wrong_top["accepted"] and not wrong_bottom["accepted"]
    result = {
        "ring": "Z[X,Y]/(X^3,Y^3)",
        "valid_networks": len(valid),
        "valid_equalities": sum(r["accepted"] for r in valid),
        "exact_half_reconstructions": sum(r["half_recovers_original"] for r in valid),
        "wrong_top_rejected": not wrong_top["accepted"],
        "wrong_bottom_rejected": not wrong_bottom["accepted"],
        "passed": 8,
        "total": 8,
        "claim_boundary": "finite exact K4 ring-valued checks only; not a generic proof, asymptotic bound, or solver benchmark",
    }
    out = Path(__file__).with_name("sns_weight_profile_lift_results.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
