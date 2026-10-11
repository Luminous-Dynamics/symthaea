#!/usr/bin/env python3
"""Reproducible arithmetic audit of power figures quoted in Birch (1982).

This checks arithmetic and stated approximations only. It is not a validated
orbital-ring power model or an independent reconstruction of all source inputs.
"""
import json
import math

SCHEMA = "luminous-dynamics.orbital-ring-power-source-audit/v1"
MU_0_H_M = 4.0 * math.pi * 1e-7
ALUMINUM_CONDUCTIVITY_S_M = 3.54e7
SHEET_THICKNESS_M = 0.05
EXAMPLE_SKYHOOK_LIFT_FORCE_N = 2.0e8
PRECESSING_FORCE_N = 5.0e11
CRYO_PAIRS = [
    {"source_location": "Birch 1982, p. 483, §3.4", "cold_side_heat_load_w": 100e6, "cryostat_electrical_power_w": 10e9},
    {"source_location": "Birch 1982, p. 485, Table 8", "cold_side_heat_load_w": 5e8, "cryostat_electrical_power_w": 5e10},
]
ORS_MASS_KG = 5.6e8
INERTIAL_SPEED_M_S = 8.0e3
TABLE8_ORBITAL_SC_MASS_PER_RING_KG = 5.3e7
TABLE8_ORBITAL_SC_LINE_MASS_KG_M = 1.2
TABLE8_GEO_SC_MASS_PER_RING_KG = 8.8e7
TABLE8_GEO_SC_LINE_MASS_KG_M = 2.0
TABLE8_CRYO_HEAT_LOSS_PER_LENGTH_W_M = 10.0  # Table 8 populates this only in the geosynchronous/support column.
TABLE8_ORBITAL_RING_HAS_ACTIVE_CRYOSTAT = False
EARTH_EQUATORIAL_RADIUS_M = 6_378_137.0
TABLE8_REFERENCE_ALTITUDE_M = 600_000.0


def main():
    equivalent_drag_speed_m_s = 2.0 / (MU_0_H_M * ALUMINUM_CONDUCTIVITY_S_M * SHEET_THICKNESS_M)
    example_eddy_power_w = EXAMPLE_SKYHOOK_LIFT_FORCE_N * equivalent_drag_speed_m_s
    eddy_power_w = PRECESSING_FORCE_N * equivalent_drag_speed_m_s
    stored_energy_j = 0.5 * ORS_MASS_KG * INERTIAL_SPEED_M_S**2
    table8_orbital_length_m = TABLE8_ORBITAL_SC_MASS_PER_RING_KG / TABLE8_ORBITAL_SC_LINE_MASS_KG_M
    table8_geo_length_m = TABLE8_GEO_SC_MASS_PER_RING_KG / TABLE8_GEO_SC_LINE_MASS_KG_M
    # The 10 W/m cryogenic row is only populated in the geosynchronous/support column.
    # Do not average it with the orbiting-ring length (whose row is blank).
    table8_length_based_heat_load_w = TABLE8_CRYO_HEAT_LOSS_PER_LENGTH_W_M * table8_geo_length_m
    table8_implied_radius_m = table8_geo_length_m / (2.0 * math.pi)
    table8_implied_altitude_m = table8_implied_radius_m - EARTH_EQUATORIAL_RADIUS_M
    reference_600km_circumference_m = 2.0 * math.pi * (EARTH_EQUATORIAL_RADIUS_M + TABLE8_REFERENCE_ALTITUDE_M)
    length_difference_from_600km_fraction = abs(table8_geo_length_m - reference_600km_circumference_m) / reference_600km_circumference_m
    checks = []

    def record(check_id, passed, observed=None, expected=None, tolerance=None, note=None):
        item = {"id": check_id, "status": "passed" if passed else "failed"}
        if observed is not None:
            item["observed"] = observed
        if expected is not None:
            item["expected"] = expected
        if tolerance is not None:
            item["tolerance"] = tolerance
        if note:
            item["note"] = note
        checks.append(item)

    record("eddy_drag.equivalent_speed_positive_finite", math.isfinite(equivalent_drag_speed_m_s) and equivalent_drag_speed_m_s > 0,
           equivalent_drag_speed_m_s, note="Computed using the paper's Eq. (73) coefficient with its stated aluminium conductivity and illustrative 5 cm sheet.")
    record("eddy_drag.equivalent_speed_reference", abs(equivalent_drag_speed_m_s - 0.90) <= 0.02,
           equivalent_drag_speed_m_s, 0.90, 0.02)
    record("eddy_drag.example_power_finite_positive", math.isfinite(example_eddy_power_w) and example_eddy_power_w > 0, example_eddy_power_w)
    record("eddy_drag.example_power_reproduced", abs(example_eddy_power_w - 0.2e9) <= 0.03e9,
           example_eddy_power_w, 0.2e9, 0.03e9,
           "Uses the paper's rounded example lift force of 2e8 N; compares to the stated ~0.2 GW example.")
    record("eddy_drag.power_finite_positive", math.isfinite(eddy_power_w) and eddy_power_w > 0, eddy_power_w)
    record("eddy_drag.power_scale_reproduced", abs(eddy_power_w - 5e11) <= 0.12 * 5e11,
           eddy_power_w, 5e11, 0.12 * 5e11,
           "Within 12% of the paper's approximate 500 GW statement; inputs include a rounded force estimate.")
    for i, pair in enumerate(CRYO_PAIRS):
        ratio = pair["cryostat_electrical_power_w"] / pair["cold_side_heat_load_w"]
        record(f"cryo_pair_{i+1}.ratio_is_100", abs(ratio - 100.0) < 1e-12,
               ratio, 100.0, 1e-12,
               pair["source_location"] + "; source numbers are transcribed as published, not assumed to share identical normalization.")
    heat_scale = CRYO_PAIRS[1]["cold_side_heat_load_w"] / CRYO_PAIRS[0]["cold_side_heat_load_w"]
    power_scale = CRYO_PAIRS[1]["cryostat_electrical_power_w"] / CRYO_PAIRS[0]["cryostat_electrical_power_w"]
    record("cryo_reported_cold_load_scale_factor_5", abs(heat_scale - 5.0) < 1e-12, heat_scale, 5.0, 1e-12)
    record("cryo_reported_wall_plug_scale_factor_5", abs(power_scale - 5.0) < 1e-12, power_scale, 5.0, 1e-12)
    record("table8.orbital_length_positive", math.isfinite(table8_orbital_length_m) and table8_orbital_length_m > 0, table8_orbital_length_m)
    record("table8.geosynchronous_length_positive", math.isfinite(table8_geo_length_m) and table8_geo_length_m > 0, table8_geo_length_m)
    length_relative_difference = abs(table8_orbital_length_m - table8_geo_length_m) / (0.5 * (table8_orbital_length_m + table8_geo_length_m))
    record("table8.inferred_ring_lengths_agree_within_1pct", length_relative_difference <= 0.01,
           length_relative_difference, 0.0, 0.01)
    record("table8.cryo_row_assigned_to_support_column",
           TABLE8_ORBITAL_RING_HAS_ACTIVE_CRYOSTAT is False,
           TABLE8_ORBITAL_RING_HAS_ACTIVE_CRYOSTAT,
           False,
           note="The scanned Table 8 leaves the orbital-ring cryogenic row blank and places 10 W/m in the geosynchronous/support column.")
    record("table8.10w_per_m_reconstructs_500mw_total", abs(table8_length_based_heat_load_w - CRYO_PAIRS[1]["cold_side_heat_load_w"]) <= 0.15 * CRYO_PAIRS[1]["cold_side_heat_load_w"],
           table8_length_based_heat_load_w, CRYO_PAIRS[1]["cold_side_heat_load_w"], 0.15 * CRYO_PAIRS[1]["cold_side_heat_load_w"],
           "Uses the length implied by the geosynchronous/support-column total per-ring superconductor mass divided by its mass-per-length row.")
    record("table8.implied_radius_positive_finite", math.isfinite(table8_implied_radius_m) and table8_implied_radius_m > 0, table8_implied_radius_m)
    record("table8.implied_altitude_positive_finite", math.isfinite(table8_implied_altitude_m) and table8_implied_altitude_m > 0, table8_implied_altitude_m)
    record("table8.support_length_within_1pct_of_600km_circular_path", length_difference_from_600km_fraction <= 0.01,
           length_difference_from_600km_fraction, 0.0, 0.01,
           "Comparison benchmark only: 600 km is discussed as a representative ORS height in the paper; this does not prove the Table 8 design uses that exact altitude.")
    record("stored_energy_finite_positive", math.isfinite(stored_energy_j) and stored_energy_j > 0, stored_energy_j)
    record("stored_energy_reference_17_92_PJ", abs(stored_energy_j - 17.92e15) < 1.0,
           stored_energy_j, 17.92e15, 1.0,
           "Conditional arithmetic only: assign the cited 8 km/s speed magnitude to the entire cited 5.6e8 kg mass.")

    failed = [c["id"] for c in checks if c["status"] != "passed"]
    report = {
        "schema": SCHEMA,
        "evidence_state": "passed_at_specified_fidelity" if not failed else "failed",
        "scope": "Source-number arithmetic audit only; not full source reconstruction or engineering feasibility analysis.",
        "source": {"author": "Paul Birch", "title": "Orbital Ring Systems and Jacob's Ladders — I", "year": 1982,
                   "pages": {"electromagnetic_drag": 482, "cryogenic_paragraph": 483, "table_8": 485},
                   "url": "https://nss.org/wp-content/uploads/Orbital-Rings.pdf"},
        "eddy_drag_example": {
            "formula_from_paper": "P_E = F_L * 2/(mu_0 * sigma * delta)",
            "mu_0_h_m": MU_0_H_M, "aluminum_conductivity_s_m": ALUMINUM_CONDUCTIVITY_S_M,
            "sheet_thickness_m": SHEET_THICKNESS_M, "equivalent_drag_speed_m_s": equivalent_drag_speed_m_s,
            "example_skyhook_lift_force_n": EXAMPLE_SKYHOOK_LIFT_FORCE_N,
            "example_calculated_power_gw": example_eddy_power_w / 1e9, "example_source_approximation_gw": 0.2,
            "precessing_force_n": PRECESSING_FORCE_N, "calculated_power_w": eddy_power_w,
            "calculated_power_gw": eddy_power_w / 1e9, "source_approximation_gw": 500.0,
            "interpretation": "Electromagnetic eddy-current drag estimate for the stated conducting-sheet model; distinct from superconducting cryostat wall-plug power."
        },
        "cryogenic_pairs": [{**p, "electrical_to_cold_load_ratio": p["cryostat_electrical_power_w"] / p["cold_side_heat_load_w"]} for p in CRYO_PAIRS],
        "source_reconstruction_warning": "The p. 483 paragraph gives about 100 MW heat loss / 10 GW cryostat power; Table 8 gives 5e8 W / 5e10 W. The paired ratios agree at 100:1. The 10 W/m row is populated under the geosynchronous/support column; dividing its 8.8e7 kg mass by 2.0 kg/m implies a 44.0 Mm support length and 440 MW heat loss, close to the table total. The orbital rings are described as not needing active cryogenic equipment because they sit within the fixed ring vacuum flask. This does not explain why the earlier paragraph's separate superconducting-surface estimate is five times lower or establish the exact physical geometry/configuration.",
        "table8_length_reconstruction": {
            "orbital_superconductor_mass_per_ring_kg": TABLE8_ORBITAL_SC_MASS_PER_RING_KG,
            "orbital_superconductor_mass_per_length_kg_m": TABLE8_ORBITAL_SC_LINE_MASS_KG_M,
            "orbital_ring_length_implied_m": table8_orbital_length_m,
            "geosynchronous_superconductor_mass_per_ring_kg": TABLE8_GEO_SC_MASS_PER_RING_KG,
            "geosynchronous_superconductor_mass_per_length_kg_m": TABLE8_GEO_SC_LINE_MASS_KG_M,
            "geosynchronous_ring_length_implied_m": table8_geo_length_m,
            "cryogenic_heat_loss_column": "geosynchronous_support",
            "orbital_ring_active_cryostat": TABLE8_ORBITAL_RING_HAS_ACTIVE_CRYOSTAT,
            "cryogenic_heat_loss_per_length_w_m": TABLE8_CRYO_HEAT_LOSS_PER_LENGTH_W_M,
            "length_based_heat_load_w": table8_length_based_heat_load_w,
            "table_reported_total_heat_load_w": CRYO_PAIRS[1]["cold_side_heat_load_w"],
            "equivalent_circular_radius_m": table8_implied_radius_m,
            "support_column_equivalent_circular_altitude_m": table8_implied_altitude_m,
            "earth_equatorial_radius_m_assumed_for_altitude": EARTH_EQUATORIAL_RADIUS_M,
            "600km_reference_circular_path_length_m": reference_600km_circumference_m,
            "relative_support_length_difference_from_600km_reference": length_difference_from_600km_fraction,
            "caveat": "Column-aware arithmetic: the 10 W/m cryogenic row is populated only under the geosynchronous/support column; applying it to that column implies about 440 MW. Does not independently establish the exact physical geometry or explain the p. 483 100 MW estimate."
        },
        "conditional_stored_energy": {"mass_kg": ORS_MASS_KG, "speed_m_s": INERTIAL_SPEED_M_S,
                                      "specific_kinetic_energy_j_kg": 0.5 * INERTIAL_SPEED_M_S**2,
                                      "total_energy_j": stored_energy_j, "total_energy_pj": stored_energy_j / 1e15,
                                      "caveat": "Not a failure-release or damage prediction."},
        "checks_passed": len(checks) - len(failed), "checks_failed": len(failed), "checks": checks,
        "failed_check_ids": failed,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
