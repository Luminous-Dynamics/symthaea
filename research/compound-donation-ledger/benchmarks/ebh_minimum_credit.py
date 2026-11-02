#!/usr/bin/env python3
"""Exact minimum additional e-BH credit and source-bank allocator.

All score, alpha and credit arithmetic is exact Fraction arithmetic.
The batch algorithm computes all candidate deficits in O(n log n).
This is an allocation primitive, not an e-process or FDR validity proof.
"""
from __future__ import annotations
from fractions import Fraction
from typing import Sequence


Rational = Fraction | int | str


def as_fraction(value: Rational) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("use Fraction, int or rational string; float is forbidden")
    return value if isinstance(value, Fraction) else Fraction(value)


def ebh_rejections(scores: Sequence[Rational], alpha: Rational) -> tuple[int, ...]:
    z = [as_fraction(v) for v in scores]
    a = as_fraction(alpha)
    if not 0 < a <= 1:
        raise ValueError("alpha must be in (0,1]")
    if not z or any(v < 0 for v in z):
        raise ValueError("scores must be a nonempty nonnegative vector")
    order = sorted(range(len(z)), key=lambda i: (-z[i], i))
    k_star = 0
    for rank, candidate in enumerate(order, start=1):
        if z[candidate] >= Fraction(1, 1) / (a * rank):
            k_star = rank
    return tuple(order[:k_star])


def feasible_ranks(scores: Sequence[Rational], alpha: Rational) -> tuple[int, ...]:
    """Return k for which exactly k-1 scores meet the size-k threshold."""
    z = [as_fraction(v) for v in scores]
    a = as_fraction(alpha)
    if not 0 < a <= 1 or not z or any(v < 0 for v in z):
        raise ValueError("invalid alpha or score vector")
    ordered = sorted(z, reverse=True)
    count = 0
    feasible: list[int] = []
    for k in range(1, len(z) + 1):
        threshold = Fraction(1, 1) / (a * k)
        while count < len(z) and ordered[count] >= threshold:
            count += 1
        if count == k - 1:
            feasible.append(k)
    return tuple(feasible)


def minimum_credit_to_enter_ebh(
    scores: Sequence[Rational],
    candidate: int,
    alpha: Rational,
    *,
    _rejections: tuple[int, ...] | None = None,
    _feasible: tuple[int, ...] | None = None,
) -> Fraction:
    """Exact minimum additive score credit that makes candidate e-BH reject."""
    z = [as_fraction(v) for v in scores]
    a = as_fraction(alpha)
    if not 0 <= candidate < len(z):
        raise IndexError("candidate index out of range")
    if not 0 < a <= 1 or any(v < 0 for v in z):
        raise ValueError("invalid alpha or score vector")
    rejected = ebh_rejections(z, a) if _rejections is None else _rejections
    if candidate in rejected:
        return Fraction(0)
    feasible = feasible_ranks(z, a) if _feasible is None else _feasible
    # Feasibility for an unrejected candidate requires z_j < 1/(alpha*k).
    # The thresholds decrease in k, so binary search finds the largest usable k.
    lo, hi = 0, len(feasible)
    while lo < hi:
        mid = (lo + hi) // 2
        k = feasible[mid]
        if z[candidate] < Fraction(1, 1) / (a * k):
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        raise AssertionError("no feasible e-BH funding rank for an unrejected candidate")
    k = feasible[lo - 1]
    return Fraction(1, 1) / (a * k) - z[candidate]


def all_minimum_credits(
    scores: Sequence[Rational], alpha: Rational
) -> tuple[Fraction, ...]:
    """O(n log n) batch calculation for all candidate funding deficits."""
    z = [as_fraction(v) for v in scores]
    a = as_fraction(alpha)
    rejected = ebh_rejections(z, a)
    feasible = feasible_ranks(z, a)
    rejected_set = set(rejected)
    result = []
    for candidate in range(len(z)):
        result.append(minimum_credit_to_enter_ebh(
            z, candidate, a, _rejections=rejected, _feasible=feasible
        ) if candidate not in rejected_set else Fraction(0))
    return tuple(result)


def brute_minimum_credit(
    scores: Sequence[Rational], candidate: int, alpha: Rational
) -> Fraction:
    """Independent breakpoint search oracle for tests."""
    z = [as_fraction(v) for v in scores]
    a = as_fraction(alpha)
    breakpoints = {Fraction(0)}
    for k in range(1, len(z) + 1):
        threshold = Fraction(1, 1) / (a * k)
        breakpoints.add(max(Fraction(0), threshold - z[candidate]))
    for credit in sorted(breakpoints):
        updated = list(z)
        updated[candidate] += credit
        if candidate in ebh_rejections(updated, a):
            return credit
    raise AssertionError("finite e-BH breakpoint set must contain a solution")


def allocate_from_source_bank(
    unassigned_by_source: Sequence[Rational],
    destination: int,
    amount: Rational,
) -> tuple[tuple[tuple[int, Fraction], ...], tuple[Fraction, ...]] | None:
    """Assign exactly amount using sources strictly before destination.

    Returns immutable (source, amount) assignments and the remaining bank.
    If eligible sources cannot cover the amount, returns None without mutation.
    """
    bank = [as_fraction(v) for v in unassigned_by_source]
    needed = as_fraction(amount)
    if not 0 <= destination < len(bank) or needed < 0:
        raise ValueError("invalid destination or assignment amount")
    if any(v < 0 for v in bank):
        raise ValueError("source bank cannot be negative")
    available = sum(bank[:destination], Fraction(0))
    if available < needed:
        return None
    remaining = needed
    allocations: list[tuple[int, Fraction]] = []
    # Deterministic oldest-source-first consumption; destinations never change.
    for source in range(destination):
        take = min(bank[source], remaining)
        if take:
            allocations.append((source, take))
            bank[source] -= take
            remaining -= take
        if remaining == 0:
            break
    assert remaining == 0
    assert all(source < destination for source, _ in allocations)
    assert sum((v for _, v in allocations), Fraction(0)) == needed
    return tuple(allocations), tuple(bank)


def choose_fundable_candidate(
    scores: Sequence[Rational],
    unassigned_by_source: Sequence[Rational],
    alpha: Rational,
) -> tuple[int, Fraction, tuple[tuple[int, Fraction], ...], tuple[Fraction, ...]] | None:
    """Choose the smallest exactly fundable deficit, then lowest candidate ID."""
    z = [as_fraction(v) for v in scores]
    bank = [as_fraction(v) for v in unassigned_by_source]
    if len(z) != len(bank):
        raise ValueError("score and source-bank vectors must have the same length")
    deficits = all_minimum_credits(z, alpha)
    candidates = []
    for destination, deficit in enumerate(deficits):
        if deficit <= 0:
            continue
        if sum(bank[:destination], Fraction(0)) >= deficit:
            candidates.append((deficit, destination))
    if not candidates:
        return None
    deficit, destination = min(candidates)
    allocation = allocate_from_source_bank(bank, destination, deficit)
    assert allocation is not None
    assignments, remaining_bank = allocation
    return destination, deficit, assignments, remaining_bank


if __name__ == "__main__":
    raise SystemExit("Import this module from its exact-rational tests.")
