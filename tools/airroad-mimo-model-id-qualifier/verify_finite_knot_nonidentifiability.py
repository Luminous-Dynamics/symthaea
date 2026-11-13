#!/usr/bin/env python3
"""Exact-rational counterexample: finite MIMO knots do not identify Z(s)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-finite-knot-nonidentifiability-v0.1.json"
ComplexQ = tuple[Fraction, Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def read_complex(item: dict) -> ComplexQ:
    return q(item["real"]), q(item["imag"])


def cmul(a: ComplexQ, b: ComplexQ) -> ComplexQ:
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def cdiv(a: ComplexQ, b: ComplexQ) -> ComplexQ:
    den = b[0] * b[0] + b[1] * b[1]
    if den == 0:
        raise ZeroDivisionError("complex denominator is zero")
    return (a[0] * b[0] + a[1] * b[1]) / den, (a[1] * b[0] - a[0] * b[1]) / den


def cpow(base: ComplexQ, power: int) -> ComplexQ:
    result = (Fraction(1), Fraction(0))
    for _ in range(power):
        result = cmul(result, base)
    return result


def delta_at_omega(omega: Fraction) -> ComplexQ:
    numerator = Fraction(1000)
    for knot in (Fraction(1), Fraction(2), Fraction(4)):
        numerator *= knot * knot - omega * omega
    denominator = cpow((Fraction(5), omega), 7)
    return cdiv((numerator, Fraction(0)), denominator)


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_finite_knot_nonidentifiability.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_analytical_reference"

    knots = [q(item) for item in data["measured_angular_frequencies_rad_s"]]
    assert knots == [Fraction(1), Fraction(2), Fraction(4)]
    off_grid = q(data["off_grid_validation_frequency_rad_s"])
    assert off_grid == 3

    model_a = data["model_a"]
    model_b = data["model_b"]
    assert model_a["proper"] and model_a["real_rational"]
    assert model_b["proper"] and model_b["real_rational"]
    assert model_b["numerator_degree"] < model_b["denominator_degree"]
    assert model_b["pole_set"] == [
        {"real": {"numerator": "-5", "denominator": "1"}, "multiplicity": 7}
    ]
    assert all(q(pole["real"]) < 0 for pole in model_b["pole_set"])
    assert model_b["delta_expression"] == (
        "1000*(s^2+1)*(s^2+4)*(s^2+16)/(s+5)^7"
    )

    # Evaluate Delta(jw) exactly from the rational factorization.
    for knot in knots:
        delta = delta_at_omega(knot)
        assert delta == (Fraction(0), Fraction(0))
        print(f"PASS training knot omega={knot}: Delta=0 exactly")

    off_grid_delta = delta_at_omega(off_grid)
    frozen_delta = read_complex(data["expected"]["off_grid_delta_at_3"])
    assert off_grid_delta == frozen_delta
    assert off_grid_delta != (Fraction(0), Fraction(0))
    assert data["expected"]["off_grid_delta_nonzero"] is True

    # Real-rational conjugate symmetry is exact.
    negative_frequency_delta = delta_at_omega(-off_grid)
    assert negative_frequency_delta == (off_grid_delta[0], -off_grid_delta[1])

    # Both complete matrices are identical at training knots because their
    # sole differing cross-port term is exactly zero there.
    base = model_b["base_Z_ohm"]
    for row in range(2):
        for column in range(2):
            assert read_complex(model_a["Z_ohm"][row][column]) == read_complex(base[row][column])
    assert model_b["delta_matrix_position"] == {
        "row": 0,
        "column": 1,
        "observed_port_id": "bus-a",
        "driven_port_id": "bus-b",
    }

    assert data["expected"]["training_responses_equal_at_all_knots"] is True
    assert data["expected"]["continuous_domain_claim_from_knots_only"] == "NOT_ESTABLISHED"

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["knot-only-training-fit-promoted-to-continuous"] == "reject_promotion"
    assert controls["diagonal-only-matrix-comparison"] == "misses-hidden-crossport"
    assert controls["off-grid-validation-observation"] == "distinguishes-models"
    assert controls["model_b-claimed-improper"] == "reject"
    assert controls["model_b-rhp-pole-declaration"] == "reject"
    assert controls["frequency-unit-mismatch"] == "not_evaluable"
    assert controls["model-b-mislabels-passivity"] == "reject_claim_class"

    print(f"PASS off-grid omega={off_grid}: Delta={off_grid_delta[0]} + j*{off_grid_delta[1]}")
    print("PASS properness, real-rational structure, stable pole set and conjugate symmetry")
    print("PASS exact finite-knot non-identifiability reference")
    print("PASS all negative-control declarations")


if __name__ == "__main__":
    main()
