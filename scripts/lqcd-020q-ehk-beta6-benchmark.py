#!/usr/bin/env python3
"""Freeze the Edwards-Heller-Klassen beta=6.0 Wilson-action scale benchmark.

This subject does not simulate a lattice ensemble. It records one exact external
benchmark extraction from hep-lat/9711003 and derives only algebraic quantities
that follow from the quoted central values.
"""
import hashlib
import json
import math

SOURCE_ID = "arxiv:hep-lat/9711003"
BENCHMARK_ID = "ehk_wilson_beta6_static_potential_v1"

def main():
    result = {
        "benchmark_id": BENCHMARK_ID,
        "source_id": SOURCE_ID,
        "beta": 6.0,
        "action_id": "wilson_pure_gauge_v1",
        "volume": [16, 16, 16, 32],
        "configuration_count": 4000,
        "update_description": "cabibbo_marinari_su2_heatbath_plus_microcanonical_overrelaxation_typical_3_to_1_per_sweep",
        "smearing": {
            "kind": "spatial_ape",
            "epsilon_times_iterations_approx": 4.0,
            "reported_ground_state_overlap_floor": 0.85,
        },
        "measured_base_vectors": [[1,0,0],[1,1,0],[1,1,1],[2,1,0]],
        "fit_family": [
            "free_v0_sigma_e_l",
            "fixed_e_pi_over_12_free_l",
            "fixed_e_pi_over_12_l0",
        ],
        "quoted": {
            "a_sqrt_sigma": 0.2189,
            "a_sqrt_sigma_error": 0.0009,
            "r0_over_a": 5.369,
            "r0_over_a_error": 0.009,
            "r4_over_a": 8.831,
            "r4_over_a_error": 0.021,
            "r6_over_a": 10.89,
            "r6_over_a_error": 0.03,
        },
    }
    x = result["quoted"]["a_sqrt_sigma"]
    dx = result["quoted"]["a_sqrt_sigma_error"]
    r0 = result["quoted"]["r0_over_a"]
    dr0 = result["quoted"]["r0_over_a_error"]
    result["derived"] = {
        "sigma_a2": x * x,
        "sigma_a2_linearized_error": 2.0 * x * dx,
        "r0_times_sqrt_sigma": r0 * x,
        "r0_times_sqrt_sigma_uncorrelated_error_only": math.sqrt((x * dr0)**2 + (r0 * dx)**2),
    }

    assert result["volume"] == [16,16,16,32]
    assert result["configuration_count"] == 4000
    assert result["quoted"]["a_sqrt_sigma"] == 0.2189
    assert result["quoted"]["r0_over_a"] == 5.369
    assert abs(result["derived"]["sigma_a2"] - 0.04791721) < 1e-15
    text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("ok")
    print("result_sha256=" + digest)
    print(text)

if __name__ == "__main__":
    main()
