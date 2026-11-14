#!/usr/bin/env python3
"""Exact-rational oracle for model-discriminating MIMO frequency selection."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-active-excitation-v0.1.json"
ComplexQ = tuple[Fraction, Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def cdiv(a: ComplexQ, b: ComplexQ) -> ComplexQ:
    den = b[0] * b[0] + b[1] * b[1]
    if den == 0:
        raise ZeroDivisionError
    return (a[0] * b[0] + a[1] * b[1]) / den, (a[1] * b[0] - a[0] * b[1]) / den


def cmul(a: ComplexQ, b: ComplexQ) -> ComplexQ:
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def cpow(base: ComplexQ, n: int) -> ComplexQ:
    result = (Fraction(1), Fraction(0))
    for _ in range(n):
        result = cmul(result, base)
    return result


def delta_at_omega(omega: Fraction) -> ComplexQ:
    numerator = Fraction(1000)
    for knot in (Fraction(1), Fraction(2), Fraction(4)):
        numerator *= knot * knot - omega * omega
    return cdiv((numerator, Fraction(0)), cpow((Fraction(5), omega), 7))


def candidate_score(candidate: dict, model: dict) -> tuple[Fraction, bool] | None:
    omega = q(candidate["angular_frequency_rad_s"])
    domain = model["valid_angular_frequency_domain_rad_s"]
    if not (q(domain["lower"]) <= omega <= q(domain["upper"])):
        return None
    if candidate.get("approved_under_protocol") is not True:
        return None
    if "cost" not in candidate:
        return None
    cost = q(candidate["cost"])
    if cost <= 0:
        return None
    radius = q(candidate["uncertainty_disk_radius"])
    if radius < 0:
        return None

    delta = delta_at_omega(omega)
    # Difference exists only in Z[observed bus-a, driven bus-b].
    coverage = (
        candidate["driven_port_id"] == "bus-b"
        and "bus-a" in candidate["observed_port_ids"]
    )
    distance_squared = (
        delta[0] * delta[0] + delta[1] * delta[1] if coverage else Fraction(0)
    )
    # Both models receive the same outer-disk radius in this fixture.
    guaranteed = distance_squared > (2 * radius) ** 2
    return distance_squared / cost, guaranteed


def select(candidates: list[dict], model: dict) -> str | None:
    ids = [candidate.get("candidate_id", "") for candidate in candidates]
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("missing or empty candidate ID")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate IDs")
    ranked: list[tuple[Fraction, str]] = []
    for candidate in candidates:
        result = candidate_score(candidate, model)
        if result is None:
            continue
        score, guaranteed = result
        if score > 0 and guaranteed:
            ranked.append((score, candidate["candidate_id"]))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_active_excitation_reference.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_model_discrimination_reference"

    model = data["model_set"]
    candidates = data["candidate_experiments"]
    assert [q(item["angular_frequency_rad_s"]) for item in candidates] == [
        Fraction(1), Fraction(2), Fraction(3), Fraction(4)
    ]

    scores = {}
    for candidate in candidates:
        result = candidate_score(candidate, model)
        assert result is not None
        scores[candidate["candidate_id"]] = result
        print(f"PASS {candidate['candidate_id']}: score={result[0]}, guaranteed={result[1]}")

    selected = select(candidates, model)
    assert selected == data["expected"]["selected_candidate_id"]
    assert selected == "omega-3-full-mimo"
    assert scores["omega-1-full-mimo"][0] == 0
    assert scores["omega-2-full-mimo"][0] == 0
    assert scores["omega-4-full-mimo"][0] == 0
    assert scores["omega-3-full-mimo"][0] > 0
    expected_separation = q(data["expected"]["off_grid_model_separation_squared"])
    assert scores["omega-3-full-mimo"][0] == expected_separation
    assert expected_separation == Fraction(612500000, 410338673)

    # Without the only off-grid discriminator, no candidate should be forced.
    no_three = [
        item for item in candidates
        if item["candidate_id"] in data["counterfactuals"]["no_omega_3_candidate"]
    ]
    assert select(no_three, model) is None
    assert data["expected"]["no_guaranteed_discriminator_without_omega_3"] is True

    diagonal = data["counterfactuals"]["diagonal_only"]
    wrong_drive = data["counterfactuals"]["wrong_driven_port"]
    wide = data["counterfactuals"]["uncertainty_overlap"]
    out_of_domain = data["counterfactuals"]["out_of_domain"]
    unapproved = data["counterfactuals"]["not_approved"]
    missing_cost = dict(data["counterfactuals"]["wrong_driven_port"])
    missing_cost.pop("cost")
    zero_cost = dict(data["counterfactuals"]["wrong_driven_port"])
    zero_cost["cost"] = {"numerator": "0", "denominator": "1"}

    assert candidate_score(diagonal, model) == (Fraction(0), False)
    assert candidate_score(wrong_drive, model) == (Fraction(0), False)
    assert candidate_score(wide, model)[0] > 0
    assert candidate_score(wide, model)[1] is False
    assert candidate_score(out_of_domain, model) is None
    assert candidate_score(unapproved, model) is None
    assert candidate_score(missing_cost, model) is None
    assert candidate_score(zero_cost, model) is None

    # Duplicate candidate identity is rejected, not made input-order dependent.
    try:
        select([candidates[0], dict(candidates[0])], model)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate candidate IDs were accepted")

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["only-measured-knots"] == "no_guaranteed_discriminating_proposal"
    assert controls["diagonal-only-comparison"] == "misses-crossport-difference"
    assert controls["wrong-driven-port"] == "no-discrimination"
    assert controls["missing-crossport-observation"] == "no-discrimination"
    assert controls["uncertainty-sets-overlap"] == "not-guaranteed"
    assert controls["outside-model-frequency-domain"] == "reject"
    assert controls["not-approved-experiment"] == "reject"
    assert controls["missing-or-nonpositive-cost"] == "reject"
    assert controls["duplicate-candidate-id"] == "reject"

    assert q(data["expected"]["diagonal_only_score_at_omega_3"]) == 0
    assert q(data["expected"]["wrong_driven_port_score_at_omega_3"]) == 0
    assert q(data["expected"]["missing_crossport_score_at_omega_3"]) == 0

    print("PASS exact minimax model-discrimination selection")
    print("PASS abstention without discriminating frequency")
    print("PASS cross-port coverage, uncertainty overlap and domain controls")
    print("PASS exact-rational active-excitation reference")


if __name__ == "__main__":
    main()
