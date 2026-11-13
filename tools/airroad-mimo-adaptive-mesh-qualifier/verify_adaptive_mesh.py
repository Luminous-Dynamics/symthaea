#!/usr/bin/env python3
"""Exact-rational Bernstein-guided adaptive frequency mesh oracle."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-adaptive-mesh-v0.1.json"
Poly = list[Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def p_add(a: Poly, b: Poly) -> Poly:
    out = [Fraction(0)] * max(len(a), len(b))
    for i, value in enumerate(a):
        out[i] += value
    for i, value in enumerate(b):
        out[i] += value
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def p_mul(a: Poly, b: Poly) -> Poly:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            out[i + j] += left * right
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def p_eval(poly: Poly, x: Fraction) -> Fraction:
    out = Fraction(0)
    for coefficient in reversed(poly):
        out = out * x + coefficient
    return out


def choose(n: int, k: int) -> int:
    value = 1
    for i in range(1, k + 1):
        value = value * (n - i + 1) // i
    return value


def affine_substitute(poly: Poly, a: Fraction, b: Fraction) -> Poly:
    # omega = a + (b-a)t, t∈[0,1]
    h = b - a
    result = [Fraction(0)]
    for degree, coefficient in enumerate(poly):
        term = [Fraction(1)]
        for _ in range(degree):
            term = p_mul(term, [a, h])
        result = p_add(result, [coefficient * value for value in term])
    return result


def bernstein_coefficients(poly_t: Poly, degree: int) -> list[Fraction]:
    # t^k = sum_{i=k}^n C(i,k)/C(n,k) * B_i^n(t)
    return [
        sum(
            (poly_t[k] * Fraction(choose(i, k), choose(degree, k))
             for k in range(min(i, len(poly_t) - 1) + 1)),
            Fraction(0),
        )
        for i in range(degree + 1)
    ]


def curve(omega: Fraction) -> Fraction:
    return 16 * omega * (1 - omega) * (omega - Fraction(1, 2)) ** 2


def interpolation_residual_poly(
    original: Poly, a: Fraction, b: Fraction
) -> Poly:
    ya, yb = p_eval(original, a), p_eval(original, b)
    slope = (yb - ya) / (b - a)
    chord = [ya - slope * a, slope]
    return p_add(original, [-value for value in chord])


def interval_certificate(
    original: Poly, a: Fraction, b: Fraction
) -> tuple[list[Fraction], Fraction]:
    local = affine_substitute(interpolation_residual_poly(original, a, b), a, b)
    coefficients = bernstein_coefficients(local, 4)
    radius = max((abs(value) for value in coefficients), default=Fraction(0))
    return coefficients, radius


def partition_valid(intervals: list[tuple[Fraction, Fraction]],
                    lo: Fraction, hi: Fraction) -> bool:
    return (
        bool(intervals)
        and intervals[0][0] == lo
        and intervals[-1][1] == hi
        and all(a < b for a, b in intervals)
        and all(intervals[i][1] == intervals[i + 1][0]
                for i in range(len(intervals) - 1))
    )


def select_interval(
    intervals: list[tuple[Fraction, Fraction]], bounds: list[Fraction]
) -> int:
    # Largest certified radius; canonical tie-break is lowest lower endpoint.
    return min(
        (i for i in range(len(intervals))),
        key=lambda i: (-bounds[i], intervals[i][0], intervals[i][1]),
    )



def snapshot_matches(
    snapshot: dict,
    intervals: list[tuple[Fraction, Fraction]],
    coefficients: list[list[Fraction]],
    bounds: list[Fraction],
) -> bool:
    rows = snapshot["partition"]
    if len(rows) != len(intervals):
        return False
    for row, (a, b), beta, radius in zip(rows, intervals, coefficients, bounds):
        if (q(row["lower"]), q(row["upper"])) != (a, b):
            return False
        if [Fraction(value) for value in row["bernstein"]] != beta:
            return False
        if q(row["bound"]) != radius:
            return False
    return q(snapshot["global_radius"]) == max(bounds)


def fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_adaptive_bernstein_mesh.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_proof_guided_interval_refinement"
    assert data["model_identity"] == "sha256:synthetic-quartic-mimo-interpolation-curve-v0.1"
    assert data["topology_identity"] == "sha256:two-port-generator-cover-reference-v0.1"
    assert data["operating_point_identity"] == "sha256:synthetic-fixed-operating-point-v0.1"
    assert data["calibration_identity"] == "sha256:synthetic-deterministic-exact-observation-profile-v0.1"
    assert data["frequency_unit"] == "normalized_frequency"
    assert data["matrix_norm"] == "frobenius_norm"

    original = [Fraction(0), Fraction(4), Fraction(-20), Fraction(32), Fraction(-16)]
    assert [str(value) for value in original] == data["polynomial_coefficients_ascending"]
    p = data["matrix_factor"]
    p_values = [[q(x) for x in row] for row in p]
    p_frob_sq = sum((x * x for row in p_values for x in row), Fraction(0))
    assert p_frob_sq == 1

    lo, hi = Fraction(0), Fraction(1)
    knots = data["initial_knots"]
    knot_values = [q(item["omega"]) for item in knots]
    knot_ids = [item["knot_id"] for item in knots]
    assert knot_values == [lo, Fraction(1, 2), hi]
    assert len(knot_ids) == len(set(knot_ids))
    assert all(q(item["value"]) == curve(q(item["omega"])) == 0 for item in knots)

    partition = [
        (q(item["lower"]), q(item["upper"]))
        for item in data["initial_partition"]
    ]
    assert partition_valid(partition, lo, hi)

    # Exact initial Bernstein certificates.
    def bounds_for(parts: list[tuple[Fraction, Fraction]]) -> tuple[list[list[Fraction]], list[Fraction]]:
        coeffs, radii = [], []
        for a, b in parts:
            beta, radius = interval_certificate(original, a, b)
            coeffs.append(beta)
            radii.append(radius)
        return coeffs, radii

    initial_partition_snapshot = list(partition)
    initial_coeffs, initial_bounds = bounds_for(partition)
    assert initial_coeffs == [
        [Fraction(0), Fraction(1, 2), Fraction(1, 6), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1, 6), Fraction(1, 2), Fraction(0)],
    ]
    assert initial_bounds == [Fraction(1, 2), Fraction(1, 2)]
    assert max(initial_bounds) == q(data["expected"]["global_radius_after_zero_splits"])
    trajectory = data["expected_trajectory"]
    assert len(trajectory) == 3
    assert trajectory[0]["step"] == 0 and trajectory[0]["selected_interval"] is None
    assert snapshot_matches(trajectory[0], partition, initial_coeffs, initial_bounds)

    # First greedy split: tie is resolved by smallest lower endpoint.
    first_index = select_interval(partition, initial_bounds)
    assert first_index == 0
    a, b = partition.pop(first_index)
    midpoint = (a + b) / 2
    assert midpoint == Fraction(1, 4)
    assert curve(midpoint) == q(data["expected"]["new_knot_values"]["quarter"]) == Fraction(3, 16)
    partition.extend([(a, midpoint), (midpoint, b)])
    partition.sort()

    coeffs_1, bounds_1 = bounds_for(partition)
    assert partition == [
        (Fraction(0), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1)),
    ]
    assert coeffs_1 == [
        [Fraction(0), Fraction(13, 64), Fraction(19, 96), Fraction(7, 64), Fraction(0)],
        [Fraction(0), Fraction(-1, 64), Fraction(-5, 96), Fraction(-3, 64), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1, 6), Fraction(1, 2), Fraction(0)],
    ]
    assert bounds_1 == [Fraction(13, 64), Fraction(5, 96), Fraction(1, 2)]
    assert max(bounds_1) == q(data["expected"]["global_radius_after_one_split"]) == Fraction(1, 2)
    assert trajectory[1]["step"] == 1 and trajectory[1]["selected_interval"] == "I0"
    assert q(trajectory[1]["inserted_knot"]["omega"]) == Fraction(1, 4)
    assert q(trajectory[1]["inserted_knot"]["value"]) == Fraction(3, 16)
    assert snapshot_matches(trajectory[1], partition, coeffs_1, bounds_1)

    # Second greedy split: the still-largest right interval must be selected.
    second_index = select_interval(partition, bounds_1)
    assert second_index == 2
    a, b = partition.pop(second_index)
    midpoint = (a + b) / 2
    assert midpoint == Fraction(3, 4)
    assert curve(midpoint) == q(data["expected"]["new_knot_values"]["three_quarters"]) == Fraction(3, 16)
    partition.extend([(a, midpoint), (midpoint, b)])
    partition.sort()

    coeffs_2, bounds_2 = bounds_for(partition)
    assert partition_valid(partition, lo, hi)
    assert partition == [
        (Fraction(0), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(3, 4)),
        (Fraction(3, 4), Fraction(1)),
    ]
    assert coeffs_2 == [
        [Fraction(0), Fraction(13, 64), Fraction(19, 96), Fraction(7, 64), Fraction(0)],
        [Fraction(0), Fraction(-1, 64), Fraction(-5, 96), Fraction(-3, 64), Fraction(0)],
        [Fraction(0), Fraction(-3, 64), Fraction(-5, 96), Fraction(-1, 64), Fraction(0)],
        [Fraction(0), Fraction(7, 64), Fraction(19, 96), Fraction(13, 64), Fraction(0)],
    ]
    assert bounds_2 == [
        Fraction(13, 64), Fraction(5, 96),
        Fraction(5, 96), Fraction(13, 64)
    ]
    global_radius = max(bounds_2)
    assert global_radius == q(data["expected"]["global_radius_after_two_splits"])
    assert global_radius == Fraction(13, 64)
    assert global_radius == q(data["policy"]["target_global_radius"])
    assert trajectory[2]["step"] == 2 and trajectory[2]["selected_interval"] == "I1"
    assert q(trajectory[2]["inserted_knot"]["omega"]) == Fraction(3, 4)
    assert q(trajectory[2]["inserted_knot"]["value"]) == Fraction(3, 16)
    assert snapshot_matches(trajectory[2], partition, coeffs_2, bounds_2)
    assert data["expected"]["target_met_after_two_splits"] is True

    # Bernstein convex-hull property: basis weights are nonnegative and sum
    # to one, so each exact coefficient maximum is a universal interval bound.
    for coefficients, radius in zip(coeffs_2, bounds_2):
        assert all(abs(value) <= radius for value in coefficients)
        assert radius >= 0

    # The original global supremum is 1/4; all-zero knot samples do not imply
    # zero remainder. A finite-grid maximum is a lower witness only.
    assert max(abs(curve(x)) for x in knot_values) == 0
    five_grid = [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]
    assert max(abs(curve(x)) for x in five_grid) == Fraction(3, 16)
    assert q(data["expected"]["exact_original_curve_supremum"]) == Fraction(1, 4)
    assert Fraction(3, 16) < Fraction(1, 4)

    # Policy and evidence controls.
    assert data["candidate_measurement_profile"]["approved_under_synthetic_protocol"] is True
    assert q(data["candidate_measurement_profile"]["hard_error_radius"]) == 0
    assert data["candidate_measurement_profile"]["shared_calibration_identity"] == data["calibration_identity"]
    assert data["policy"]["tie_break"] == "largest_bound_then_lowest_lower_endpoint_then_interval_id"
    assert data["policy"]["split_rule"] == "exact_midpoint"
    assert data["policy"]["split_budget"] == 2

    # Reverse-tie policy picks the right interval instead; that is a different
    # policy identity and must not be presented as the same replay.
    reverse_tie_choice = max(
        (i for i in range(len(initial_bounds))),
        key=lambda i: (initial_bounds[i], initial_partition_snapshot[i][0])
    )
    assert reverse_tie_choice == 1
    assert reverse_tie_choice != first_index

    # Missing interval coverage, gaps, duplicate IDs, and stale/unapproved
    # evidence are evaluated from explicit counterfactual records.
    assert not partition_valid([(Fraction(0), Fraction(1, 2))], lo, hi)
    assert not partition_valid([
        (q(item["lower"]), q(item["upper"]))
        for item in data["counterfactuals"]["overstated_gap_partition"]
    ], lo, hi)
    assert len(set(knot_ids + ["omega-half"])) != len(knot_ids + ["omega-half"])

    counter = data["counterfactuals"]
    assert counter["unapproved_midpoint"]["approved"] is False
    assert q(counter["unapproved_midpoint"]["omega"]) == Fraction(1, 4)
    assert counter["stale_model_identity"] != data["model_identity"]
    assert counter["stale_calibration_identity"] != data["calibration_identity"]
    assert counter["wrong_frequency_unit"] != data["frequency_unit"]
    assert counter["statistical_uncertainty_semantics"] != "deterministic_exact_midpoint_observation"

    # A radius smaller than the certified global bound cannot be a PASS.
    too_small_target = q(counter["too_small_target"])
    assert too_small_target == Fraction(1, 5)
    assert too_small_target < global_radius
    exhausted_is_pass = len(partition) >= len(knots) + data["policy"]["split_budget"] and global_radius <= too_small_target
    assert not exhausted_is_pass

    # Missing/stale evidence, unapproved proposal and statistical noise do not
    # enter the deterministic synthetic hard profile.
    assert data["counterfactuals"]["stale_model_identity"] != data["model_identity"]
    assert data["counterfactuals"]["wrong_frequency_unit"] != data["frequency_unit"]

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["sampled-knot-maximum-used-as-universal-upper"] == "reject"
    assert controls["reverse-tie-break"] == "different-policy-identity"
    assert controls["skip-largest-certified-interval"] == "reject-policy"
    assert controls["missing-partition-endpoint"] == "not-established"
    assert controls["partition-gap-or-overlap"] == "reject"
    assert controls["duplicate-knot-id"] == "reject"
    assert controls["stale-model-or-calibration-identity"] == "not-evaluable"
    assert controls["unapproved-midpoint"] == "reject-proposal"
    assert controls["target-below-certified-radius-at-budget"] == "budget-exhausted-not-pass"
    assert controls["statistical-noise-labelled-deterministic-zero"] == "reject-claim-class"
    assert controls["missing-frequency-unit"] == "not-evaluable"

    print(f"PASS initial Bernstein mesh radius {max(initial_bounds)}")
    print(f"PASS first split selects [0,1/2]; remaining radius {max(bounds_1)}")
    print(f"PASS second split selects [1/2,1]; final radius {global_radius}")
    print("PASS deterministic tie-break, complete coverage and exact midpoint evidence")
    print("PASS sampled maxima are not used as universal upper bounds")
    print("PASS exact-rational adaptive Bernstein mesh reference")


if __name__ == "__main__":
    main()
