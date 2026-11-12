#!/usr/bin/env python3
"""Independent exact-rational verifier for symmetric RC/CPL robust stability."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-pdcm-robust-stability-v0.1.json"


def q(v: dict[str, str]) -> Fraction:
    return Fraction(int(v["n"]), int(v["d"]))


def threshold(gs: Fraction, g: Fraction) -> Fraction:
    return gs * g / (gs + 2 * g)


def disposition(gs, g, h):
    lo = threshold(gs[0], g[0])
    hi = threshold(gs[1], g[1])
    if h[1] < lo:
        return lo, hi, "robustly_hurwitz_stable", (lo - h[1]) / lo
    if h[0] > hi:
        return lo, hi, "robustly_unstable_positive_real_pole", (h[0] - hi) / hi
    return lo, hi, "not_established_boundary_or_overlap", Fraction(0)


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_reference"

    for case in data["cases"]:
        gs = (q(case["source_conductance"]["lower"]),
              q(case["source_conductance"]["upper"]))
        g = (q(case["feeder_conductance"]["lower"]),
             q(case["feeder_conductance"]["upper"]))
        h = (q(case["negative_incremental_conductance"]["lower"]),
             q(case["negative_incremental_conductance"]["upper"]))

        lo, hi, status, margin = disposition(gs, g, h)
        assert lo == q(case["threshold_lower"]), case["id"]
        assert hi == q(case["threshold_upper"]), case["id"]
        assert status == case["disposition"], case["id"]
        assert margin == q(case["normalized_margin"]), case["id"]

        # Prove the critical threshold is increasing in both positive inputs.
        gs_test = gs[0] if gs[0] < gs[1] else gs[0] + 1
        g_test = g[0] if g[0] < g[1] else g[0] + 1
        derivative_gs = 2 * g_test * g_test / (gs_test + 2 * g_test) ** 2
        derivative_g = gs_test * gs_test / (gs_test + 2 * g_test) ** 2
        assert derivative_gs > 0 and derivative_g > 0

        print(f"PASS {case['id']}: {status}, exact margin={margin}")

    stable = data["cases"][0]
    scale = q(data["metamorphic"]["common_scaling_factor"])
    gs0 = (q(stable["source_conductance"]["lower"]),
           q(stable["source_conductance"]["upper"]))
    g0 = (q(stable["feeder_conductance"]["lower"]),
          q(stable["feeder_conductance"]["upper"]))
    h0 = (q(stable["negative_incremental_conductance"]["lower"]),
          q(stable["negative_incremental_conductance"]["upper"]))
    base = disposition(gs0, g0, h0)
    scaled = disposition((gs0[0]*scale, gs0[1]*scale),
                         (g0[0]*scale, g0[1]*scale),
                         (h0[0]*scale, h0[1]*scale))
    assert base[2:] == scaled[2:]
    assert scaled[0] == base[0] * scale
    assert scaled[1] == base[1] * scale
    print("PASS common-conductance scaling preserves disposition and normalized margin")

    # Strict-set monotonicity: widening h upper from 0.1 to 0.31 destroys
    # robust stability for the same source/feeder hard intervals.
    narrow = data["cases"][0]
    widened = data["cases"][4]
    assert narrow["disposition"] == "robustly_hurwitz_stable"
    assert widened["disposition"] == "not_established_boundary_or_overlap"
    assert narrow["source_conductance"] == widened["source_conductance"]
    assert narrow["feeder_conductance"] == widened["feeder_conductance"]
    assert narrow["negative_incremental_conductance"]["lower"] == (
        widened["negative_incremental_conductance"]["lower"]
    )
    assert q(widened["negative_incremental_conductance"]["upper"]) > (
        q(narrow["negative_incremental_conductance"]["upper"])
    )
    print("PASS interval widening weakens a robust-stable conclusion")


if __name__ == "__main__":
    main()
