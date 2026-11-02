"""Exact structural checker for arbitrary-intersection compound donation bounds.

Every credited edge is forward-only. The checker verifies, for every finite
subset S, that internal flows cancel and external inflows are covered by the
source reserves of indices below max(S) that are outside S.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Sequence

Rational = Fraction | int | str


def q(value: Rational) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("exact rational input required")
    return value if isinstance(value, Fraction) else Fraction(value)


def compound_scores(
    gamma: Sequence[Rational],
    evidence: Sequence[Rational],
    outgoing: Sequence[Rational],
    incoming: Sequence[Rational],
) -> tuple[Fraction, ...]:
    if not (len(gamma) == len(evidence) == len(outgoing) == len(incoming)):
        raise ValueError("all candidate vectors must have equal length")
    g, x, o, inc = map(lambda row: [q(v) for v in row],
                       (gamma, evidence, outgoing, incoming))
    if any(v < 0 for vector in (g, x, o, inc) for v in vector):
        raise ValueError("all quantities must be nonnegative")
    return tuple(g[i] * x[i] + inc[i] - o[i] for i in range(len(g)))


def intersection_flow_sides(
    n: int,
    intersection: Sequence[int],
    outgoing: Sequence[Rational],
    edges: Sequence[tuple[int, int, Rational]],
) -> tuple[Fraction, Fraction]:
    """Return LHS=sum_{i in S}(I_i-O_i) and RHS=sum_{j<=max S,j notin S}O_j."""
    if not intersection:
        raise ValueError("intersection must be nonempty")
    s = set(intersection)
    if len(s) != len(intersection):
        raise ValueError("intersection IDs must be unique")
    if any(not 0 <= i < n for i in s):
        raise IndexError("intersection ID out of range")
    out = [q(value) for value in outgoing]
    if len(out) != n or any(value < 0 for value in out):
        raise ValueError("invalid outgoing vector")

    incoming = [Fraction(0) for _ in range(n)]
    credited_outgoing = [Fraction(0) for _ in range(n)]
    for source, destination, amount_raw in edges:
        amount = q(amount_raw)
        if not (0 <= source < destination < n):
            raise ValueError("every transfer must be strictly forward")
        if amount < 0:
            raise ValueError("transfer amount must be nonnegative")
        incoming[destination] += amount
        credited_outgoing[source] += amount
    if any(credited_outgoing[i] > out[i] for i in range(n)):
        raise ValueError("credited transfers exceed source reserve")

    prefix_balance = Fraction(0)
    for i in range(n):
        prefix_balance += incoming[i] - out[i]
        if prefix_balance > 0:
            raise ValueError("forward prefix budget violated")

    k = max(s)
    lhs = sum((incoming[i] - out[i] for i in s), Fraction(0))
    rhs = sum((out[j] for j in range(k + 1) if j not in s), Fraction(0))
    return lhs, rhs


def verify_every_intersection(
    n: int,
    outgoing: Sequence[Rational],
    edges: Sequence[tuple[int, int, Rational]],
) -> int:
    """Check every nonempty finite intersection in an n-candidate instance."""
    checks = 0
    ids = range(n)
    for size in range(1, n + 1):
        for subset in combinations(ids, size):
            lhs, rhs = intersection_flow_sides(n, subset, outgoing, edges)
            if lhs > rhs:
                raise AssertionError(
                    f"arbitrary-intersection flow bound failed for S={subset}: "
                    f"{lhs} > {rhs}"
                )
            checks += 1
    return checks
