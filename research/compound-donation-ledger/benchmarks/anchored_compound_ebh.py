"""Projective slack-anchored closure over weighted compound e-values.

Given time-monotone compound scores Z_i and a fixed leftover mass delta>0,
the local intersection e-process is F_S = delta + sum_{i in S} Z_i.
The certified-set condition simplifies to a positive-part deficit budget.
All inputs are exact rationals; floats are rejected.
"""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from typing import Sequence

Rational = Fraction | int | str


def as_fraction(value: Rational) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("exact rational input required")
    return value if isinstance(value, Fraction) else Fraction(value)



def projective_weight_prefix(
    count: int, global_mass: Rational
) -> tuple[Fraction, ...]:
    """First count weights of the globally precommitted telescoping sequence.

    gamma_i = global_mass / [i(i+1)] for i >= 1. Its infinite sum is
    global_mass, independent of the currently materialized horizon.
    """
    mass = as_fraction(global_mass)
    if count < 0:
        raise ValueError("count must be nonnegative")
    if not 0 < mass < 1:
        raise ValueError("global candidate-weight mass must lie in (0,1)")
    return tuple(mass / (i * (i + 1)) for i in range(1, count + 1))


def anchor_slack(global_mass: Rational) -> Fraction:
    """Unused part of a fixed global candidate-weight budget."""
    mass = as_fraction(global_mass)
    if not 0 < mass < 1:
        raise ValueError("global candidate-weight mass must lie in (0,1)")
    return Fraction(1) - mass


def weighted_compound_ebh(
    scores: Sequence[Rational], alpha: Rational
) -> tuple[int, ...]:
    z = [as_fraction(v) for v in scores]
    a = as_fraction(alpha)
    if not 0 < a <= 1:
        raise ValueError("alpha must be in (0,1]")
    if any(v < 0 for v in z):
        raise ValueError("compound scores must be nonnegative")
    order = sorted(range(len(z)), key=lambda i: (-z[i], i))
    k_star = 0
    for k, candidate in enumerate(order, start=1):
        if z[candidate] >= Fraction(1, 1) / (a * k):
            k_star = k
    return tuple(order[:k_star])


def closure_deficit(
    scores: Sequence[Rational],
    alpha: Rational,
    rejection_set: Sequence[int],
) -> Fraction:
    z = [as_fraction(v) for v in scores]
    a = as_fraction(alpha)
    selected = tuple(sorted(set(rejection_set)))
    if len(selected) != len(rejection_set):
        raise ValueError("rejection set must have unique IDs")
    if any(not 0 <= i < len(z) for i in selected):
        raise IndexError("candidate index out of range")
    if not selected:
        return Fraction(0)
    r = len(selected)
    threshold = Fraction(1, r)
    return sum(
        (max(Fraction(0), threshold - a * z[i]) for i in selected),
        Fraction(0),
    )


def is_closure_certified(
    scores: Sequence[Rational],
    alpha: Rational,
    delta: Rational,
    rejection_set: Sequence[int],
) -> bool:
    z = [as_fraction(v) for v in scores]
    a = as_fraction(alpha)
    d = as_fraction(delta)
    if d < 0 or not 0 < a <= 1:
        raise ValueError("invalid alpha/delta")
    selected = set(rejection_set)
    if not selected:
        return True
    if len(selected) != len(rejection_set):
        raise ValueError("rejection set must have unique IDs")
    if any(not 0 <= i < len(z) for i in selected):
        raise IndexError("candidate index out of range")
    r = len(selected)
    # Exhaustive all-intersections definition: this is the slow oracle.
    for mask in range(1, 1 << len(z)):
        intersection = [
            i for i in range(len(z)) if (mask >> i) & 1
        ]
        lhs = Fraction(
            sum(i in selected for i in intersection), r
        )
        rhs = a * (
            sum((z[i] for i in intersection), Fraction(0)) + d
        )
        if lhs > rhs:
            return False
    return True


def anchored_rejections(
    scores: Sequence[Rational],
    alpha: Rational,
    delta: Rational,
) -> tuple[int, ...]:
    """Largest-cardinality certified superset of weighted compound e-BH.

    O(n log n): one descending sort, prefix sums, one binary search per
    cardinality. Deterministic score ties are resolved by ascending candidate ID.
    """
    z = [as_fraction(v) for v in scores]
    a, d = as_fraction(alpha), as_fraction(delta)
    if not 0 < a <= 1:
        raise ValueError("alpha must be in (0,1]")
    if d < 0 or d > 1:
        raise ValueError("delta must be in [0,1]")
    if any(v < 0 for v in z):
        raise ValueError("compound scores must be nonnegative")

    n = len(z)
    if n == 0:
        return ()

    baseline = weighted_compound_ebh(z, a)
    baseline_size = len(baseline)
    order = sorted(range(n), key=lambda i: (-z[i], i))
    ordered_scores = [z[i] for i in order]

    prefix = [Fraction(0)]
    for value in ordered_scores:
        prefix.append(prefix[-1] + value)

    # Negating descending scores yields an ascending sequence for bisect.
    negated = [-value for value in ordered_scores]

    best_r = baseline_size
    if baseline_size:
        base_deficit = closure_deficit(z, a, baseline)
        if base_deficit > a * d:
            raise AssertionError("baseline must be certified by construction")

    for r in range(max(1, baseline_size), n + 1):
        # Z_(k) >= 1/(alpha*r) iff alpha*Z_(k) >= 1/r.
        threshold_z = Fraction(1, 1) / (a * r)
        q = min(r, bisect_right(negated, -threshold_z))
        deficit = Fraction(r - q, r) - a * (prefix[r] - prefix[q])
        if deficit <= a * d:
            best_r = r

    return tuple(order[:best_r])


def brute_largest_anchored_superset(
    scores: Sequence[Rational],
    alpha: Rational,
    delta: Rational,
) -> tuple[int, ...]:
    """Exponential independent oracle for small candidate universes."""
    z = [as_fraction(v) for v in scores]
    a, d = as_fraction(alpha), as_fraction(delta)
    baseline = set(weighted_compound_ebh(z, a))
    best: tuple[int, ...] = ()
    for mask in range(1 << len(z)):
        candidate = tuple(i for i in range(len(z)) if (mask >> i) & 1)
        selected = set(candidate)
        if not baseline.issubset(selected):
            continue
        if len(candidate) < len(best):
            continue
        if is_closure_certified(z, a, d, candidate):
            if len(candidate) > len(best):
                best = candidate
            elif len(candidate) == len(best):
                # Match the canonical top-r tie-break.
                order = sorted(range(len(z)), key=lambda i: (-z[i], i))
                canonical = tuple(order[:len(best)])
                if candidate == canonical:
                    best = candidate
    return best


def local_intersection_process(
    scores: Sequence[Rational],
    intersection: Sequence[int],
    delta: Rational,
) -> Fraction:
    z = [as_fraction(v) for v in scores]
    d = as_fraction(delta)
    if d < 0:
        raise ValueError("delta must be nonnegative")
    if not intersection:
        raise ValueError("intersection must be nonempty")
    if len(set(intersection)) != len(intersection):
        raise ValueError("intersection IDs must be unique")
    if any(not 0 <= i < len(z) for i in intersection):
        raise IndexError("intersection ID out of range")
    return d + sum((z[i] for i in intersection), Fraction(0))


if __name__ == "__main__":
    # Small nontrivial strict-addition witness.
    alpha, delta = Fraction(1, 10), Fraction(1, 5)
    z = [Fraction(199, 20), Fraction(0)]
    baseline = weighted_compound_ebh(z, alpha)
    anchored = anchored_rejections(z, alpha, delta)
    assert baseline == ()
    assert anchored == (0,)
    assert is_closure_certified(z, alpha, delta, anchored)
    print({"baseline": baseline, "anchored": anchored})
