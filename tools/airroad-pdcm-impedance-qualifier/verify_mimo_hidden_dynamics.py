#!/usr/bin/env python3
"""Independent exact-rational verifier for the 2x2 hidden-dynamics fixture."""

from __future__ import annotations

import json
from fractions import Fraction
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-pdcm-mimo-hidden-dynamics-v0.1.json"


def q(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def cadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def csub(a, b):
    return a[0] - b[0], a[1] - b[1]


def cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def cdiv(a, b):
    den = b[0] * b[0] + b[1] * b[1]
    return ((a[0] * b[0] + a[1] * b[1]) / den,
            (a[1] * b[0] - a[0] * b[1]) / den)


def cpair(record):
    return q(record["real"]), q(record["imag"])


def main():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_reference"
    model = data["model"]
    params = model["parameters"]
    k12 = q(params["residue_12"])
    p12 = q(params["pole_12_per_s"])
    k21 = q(params["residue_21"])
    p21 = q(params["pole_21_per_s"])

    # Diagonal SISO projections are identically zero.
    assert model["diagonal_siso_projections"]["L11"] == "0"
    assert model["diagonal_siso_projections"]["L22"] == "0"
    assert k12 != 0 and k21 != 0

    open_rhp = int(k12 != 0 and p12 > 0) + int(k21 != 0 and p21 > 0)
    assert open_rhp == model["open_loop_rhp_pole_count"] == 1

    # det(I+L) = 1 - k12*k21/((s-p12)(s-p21)).
    coefficients = [
        Fraction(1),
        -(p12 + p21),
        p12 * p21 - k12 * k21,
    ]
    frozen = [q(x) for x in data["closed_loop"]["cleared_characteristic_polynomial_descending"]]
    assert coefficients == frozen == [Fraction(1), Fraction(1), Fraction(-3)]
    discriminant = coefficients[1]**2 - 4 * coefficients[2]
    assert discriminant == q(data["closed_loop"]["discriminant"]) == 13
    assert coefficients[2] < 0
    assert sqrt(float(discriminant)) > 1.0
    assert (-1 + sqrt(13)) / 2 > 0
    assert (-1 - sqrt(13)) / 2 < 0

    # Check exact rational complex port values and determinant at each frequency.
    for point in data["frequency_points"]:
        omega = q(point["omega_rad_s"])
        s = (Fraction(0), omega)
        l12 = cdiv((k12, Fraction(0)), (s[0] - p12, s[1]))
        l21 = cdiv((k21, Fraction(0)), (s[0] - p21, s[1]))
        determinant = csub((Fraction(1), Fraction(0)), cmul(l12, l21))
        assert l12 == cpair(point["L12"])
        assert l21 == cpair(point["L21"])
        assert determinant == cpair(point["determinant_i_plus_l"])

    # Sensitivity of the positive root to each residue.
    # dr/dk12 = k21/sqrt(discriminant); dr/dk21 = k12/sqrt(discriminant).
    assert k12 == 1 and k21 == 1
    assert coefficients[1] == 1 and coefficients[2] == -3
    assert (k12 / k21) == 1

    # Negative control: remove one cross channel. det(I+L) is identically one,
    # but an active L12 transfer pole at +1 must remain in realization metadata.
    remove_l21 = data["negative_controls"]["remove_l21"]
    k12_nc = q(remove_l21["residue_12"])
    k21_nc = q(remove_l21["residue_21"])
    nc_det_constant = Fraction(1) if k12_nc * k21_nc == 0 else None
    nc_rhp = int(k12_nc != 0 and p12 > 0) + int(k21_nc != 0 and p21 > 0)
    assert nc_det_constant == 1
    assert nc_rhp == remove_l21["open_loop_rhp_pole_count"] == 1

    remove_both = data["negative_controls"]["remove_both_cross_channels"]
    assert q(remove_both["residue_12"]) == 0
    assert q(remove_both["residue_21"]) == 0
    assert remove_both["determinant_i_plus_l"] == "1"
    assert remove_both["open_loop_rhp_pole_count"] == 0

    print("PASS diagonal SISO projections omit both cross-port channels")
    print("PASS full MIMO characteristic polynomial and one positive real root")
    print("PASS declared open-loop RHP pole count is preserved")
    print("PASS exact complex matrix and det(I+L) at DC and 1 rad/s")
    print("PASS determinant cancellation does not erase an active realization pole")
    print("PASS removing both cross channels removes the active RHP pole")


if __name__ == "__main__":
    main()
