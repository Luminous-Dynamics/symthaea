#!/usr/bin/env python3
"""Independent standard-library T0 check for circular two-body Earth-orbit fixtures.

Run with Python 3; emits deterministic JSON and exits nonzero on any failed check.
This is not a supported-ring, rotor, structural, debris, or feasibility model.
"""
import json
import math
import sys

SCHEMA = "luminous-dynamics.orbital-ring-reference-check/v1"
MU = 3.986_004_418e14  # m^3/s^2
EARTH_RADIUS_KM = 6_378.137
ALTITUDES_KM = (200.0, 300.0, 400.0, 500.0)
FIXTURES = {
    200.0: {"radius_km": 6578.137, "circumference_km": 41331.65, "speed_km_s": 7.78426, "period_min": 88.494, "specific_kinetic_energy_mj_kg": 30.297},
    300.0: {"radius_km": 6678.137, "circumference_km": 41959.97, "speed_km_s": 7.72576, "period_min": 90.520, "specific_kinetic_energy_mj_kg": 29.844},
    400.0: {"radius_km": 6778.137, "circumference_km": 42588.29, "speed_km_s": 7.66856, "period_min": 92.560, "specific_kinetic_energy_mj_kg": 29.403},
    500.0: {"radius_km": 6878.137, "circumference_km": 43216.61, "speed_km_s": 7.61261, "period_min": 94.616, "specific_kinetic_energy_mj_kg": 28.976},
}
DIGITS = {"radius_km": 3, "circumference_km": 2, "speed_km_s": 5, "period_min": 3, "specific_kinetic_energy_mj_kg": 3}
RUST_ANCHORS_400 = {
    "circumference_km": (42588.2908, 0.001),
    "speed_km_s": (7.6685582, 1e-6),
    "period_min": (92.5604045, 1e-5),
    "specific_kinetic_energy_mj_kg": (29.4033922, 1e-5),
}


def calculate(h_km):
    r_km = EARTH_RADIUS_KM + h_km
    r_m = r_km * 1000.0
    c_m = 2.0 * math.pi * r_m
    v_m_s = math.sqrt(MU / r_m)
    t_s = 2.0 * math.pi * math.sqrt(r_m**3 / MU)
    e_j_kg = 0.5 * v_m_s**2
    return {"altitude_km": h_km, "radius_km": r_km,
            "circumference_km": c_m / 1000.0, "speed_km_s": v_m_s / 1000.0,
            "period_min": t_s / 60.0, "specific_kinetic_energy_mj_kg": e_j_kg / 1e6}


def main():
    rows, checks = [calculate(h) for h in ALTITUDES_KM], []

    def check(cid, observed, expected=None, tolerance=None, note=None):
        passed = (observed if expected is None else abs(observed - expected) <= tolerance)
        item = {"id": cid, "status": "passed" if passed else "failed"}
        if expected is not None:
            item.update(observed=observed, expected=expected, absolute_tolerance=tolerance)
        if note:
            item["note"] = note
        checks.append(item)

    for row in rows:
        h = row["altitude_km"]
        for field, expected in FIXTURES[h].items():
            tol = 0.5 * 10.0 ** -DIGITS[field] + 1e-12
            check(f"documented_fixture.{int(h)}km.{field}", row[field], expected, tol,
                  "Half-unit in last displayed decimal place.")
        r, c = row["radius_km"] * 1000, row["circumference_km"] * 1000
        v, t = row["speed_km_s"] * 1000, row["period_min"] * 60
        e = row["specific_kinetic_energy_mj_kg"] * 1e6
        for name, observed, expected in (
            ("kepler_speed_squared_radius", v*v*r, MU),
            ("period_equals_circumference_over_speed", t, c/v),
            ("specific_kinetic_energy", e, 0.5*v*v),
        ):
            check(f"invariant.{int(h)}km.{name}", observed, expected,
                  max(abs(expected)*5e-12, 1e-9), "Numerical consistency only.")

    row400 = next(r for r in rows if r["altitude_km"] == 400.0)
    for field, (expected, tol) in RUST_ANCHORS_400.items():
        check(f"rust_unit_test_anchor.400km.{field}", row400[field], expected, tol,
              "Compares numeric anchor only; does not compile or execute Rust.")
    for name, ok in (
        ("circumference_increases", all(a["circumference_km"] < b["circumference_km"] for a,b in zip(rows, rows[1:]))),
        ("period_increases", all(a["period_min"] < b["period_min"] for a,b in zip(rows, rows[1:]))),
        ("speed_decreases", all(a["speed_km_s"] > b["speed_km_s"] for a,b in zip(rows, rows[1:]))),
    ):
        check(f"trend.{name}", ok)

    failed = [c["id"] for c in checks if c["status"] != "passed"]
    report = {
        "schema": SCHEMA,
        "model": "earth_point_mass_circular_orbit",
        "evidence_state": "passed_at_specified_fidelity" if not failed else "failed",
        "scope": {
            "fidelity": "T0 analytical arithmetic / fixture cross-check",
            "claim": "Closed-form circular-orbit values match rounded fixtures, exact Rust test anchors, and algebraic invariants within declared tolerances.",
            "not_claimed": ["Rust compilation/tests, rustfmt, Clippy, or Cargo resolution", "comparison to GMAT/Orekit/Basilisk or flight data", "stationary-ring/launch-loop equilibrium", "rotor/support, structural, debris, power, cost, or buildability validation"],
        },
        "assumptions": {"earth_gravitational_parameter_m3_s2": MU, "earth_equatorial_reference_radius_km": EARTH_RADIUS_KM,
                        "gravity_model": "Earth point mass", "orbit": "circular two-body geocentric",
                        "omitted": ["drag", "J2", "third bodies", "Earth rotation", "ring-support physics"]},
        "formulae": {"radius": "r = R_E + h", "circumference": "C = 2*pi*r", "speed": "v = sqrt(mu/r)",
                     "period": "T = 2*pi*sqrt(r^3/mu)", "specific_kinetic_energy": "e_k = 0.5*v^2"},
        "rows": rows, "check_count": len(checks), "checks": checks, "failed_check_ids": failed,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
