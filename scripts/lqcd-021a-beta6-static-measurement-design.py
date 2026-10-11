#!/usr/bin/env python3
"""Freeze the beta=6.0 static-potential measurement design.

This subject freezes measurement/operator intent before any target-volume pilot
or production ensemble. It deliberately does NOT freeze burn-in, measurement
stride, seed commitments, retained configuration count, or final V_eff plateau
windows. Those require a separate pre-production target-volume pilot whose
configurations are excluded from the final benchmark sample.
"""
import hashlib
import json
import math

DESIGN_ID = "ehk_beta6_static_measurement_design_v1"
BENCHMARK_RESULT_SHA256 = "cc7a16813bd175b56cc9242425ec4c1f235a9f54483efe9dc173b87fbd3b28b2"

def cubic_orbit_size(v):
    a, b, c = v
    values = set()
    import itertools
    for p in set(itertools.permutations((a, b, c))):
        for signs in itertools.product((-1, 1), repeat=3):
            values.add(tuple(signs[i] * p[i] for i in range(3)))
    return len(values)

def vector_program():
    vectors = []
    for n in range(1, 8):
        vectors.append((n, 0, 0))
    for n in range(1, 8):
        vectors.append((n, n, 0))
    for n in range(1, 8):
        vectors.append((n, n, n))
    for n in range(1, 4):
        vectors.append((2 * n, n, 0))
    return vectors

def main():
    vectors = vector_program()
    if len(vectors) != 24 or len(set(vectors)) != 24:
        raise AssertionError("unexpected vector program")

    measurement_vectors = []
    for vector in vectors:
        max_component = max(abs(x) for x in vector)
        manhattan = sum(abs(x) for x in vector)
        if max_component >= 8:
            raise AssertionError(("half-box ambiguity", vector))
        orbit_size = cubic_orbit_size(vector)
        if orbit_size not in (6, 8, 12, 24):
            raise AssertionError(("unexpected cubic orbit", vector, orbit_size))
        measurement_vectors.append({
            "representative": list(vector),
            "euclidean_radius": math.sqrt(sum(x*x for x in vector)),
            "manhattan_steps_per_orientation": manhattan,
            "cubic_orbit_size": orbit_size,
            "wilson_temporal_extents": list(range(1, 9)),
        })

    epsilon_proxy = 1.0 / (4.0 + 0.7)
    epsilon_times_iterations = epsilon_proxy * 19

    design = {
        "design_id": DESIGN_ID,
        "scientific_scope": "measurement_design_only_not_execution_authority",
        "external_benchmark": {
            "benchmark_id": "ehk_wilson_beta6_static_potential_v1",
            "source_id": "arxiv:hep-lat/9711003",
            "benchmark_extraction_result_sha256": BENCHMARK_RESULT_SHA256,
            "benchmark_pr": 2488,
            "quoted_targets": {
                "a_sqrt_sigma": [0.2189, 0.0009],
                "r0_over_a": [5.369, 0.009],
                "r4_over_a": [8.831, 0.021],
                "r6_over_a": [10.89, 0.03],
            },
        },
        "gauge_ensemble_geometry": {
            "action_id": "wilson_pure_gauge_v1",
            "beta": 6.0,
            "dims": [16, 16, 16, 32],
            "periodic_boundaries": True,
        },
        "transition_kernel": {
            "sampler_id": "cm_heatbath_or_v1:force=staple:or_sweeps=3:max_attempts=256",
            "transition_revision": "56c115b46920fc8b34daa8897051cff59c61a0bf",
            "stable_id_contract_revision": "ce6cb274e563289ae0001a49babba5f2025cb9ae",
            "overrelaxation_sweeps_per_heatbath": 3,
            "max_heatbath_attempts": 256,
            "force_backend": "staple",
        },
        "spatial_operator": {
            "convention_id": "spatial_ape_ehk_polar_v1",
            "implementation_revision": "1c7c039d4e9e90b394108254dc9220211eec9371",
            "alpha": 0.7,
            "iterations": 19,
            "epsilon_proxy": epsilon_proxy,
            "epsilon_times_iterations_proxy": epsilon_times_iterations,
            "projection_tolerance": 2.0e-15,
            "projection_max_iterations": 40,
        },
        "wilson_operator": {
            "convention_id": "generalized_bresenham_cubic_ape_spatial_unsmeared_temporal_wilson_v1",
            "implementation_revision": "98a1d4d7874ccf9860d85651309f34eaf5e42506",
            "geometry_oracle_pr": 2520,
            "geometry_oracle_subject_sha256": "c270fe43785b1c45cfd7e03acb4ac583782eb5e83193bf1fa977a6358242fdb9",
            "geometry_oracle_result_sha256": "884142f2645067e312380be738b618e85bc8bb727cbf6a3b0ea675858852c285",
            "max_orientations": 24,
            "max_steps_per_orientation": 21,
            "temporal_extent_max": 8,
            "vectors": measurement_vectors,
        },
        "analysis_primitives": {
            "effective_potential_revision": "781b0818a9c5daf879ca1458cf032bc6f89f281e",
            "lattice_coulomb_revision": "24f81937b9a7e5221051a79b86cf7b22495d6f16",
            "fit_family_revision": "ff4a5b2614c2cee78f51d80ae082d5a57957431d",
            "sommer_scale_revision": "ecd482f7ca250c938a5cff232b1b4081d2715b3a",
            "fit_family_ids": [
                "free_v0_sigma_e_l_v1",
                "fixed_e_pi_over_12_free_l_v1",
                "fixed_e_pi_over_12_l0_v1",
            ],
            "sommer_c_targets": [1.65, 4.0, 6.0],
        },
        "finalization_requirements": {
            "target_volume_pilot_required": True,
            "pilot_configurations_excluded_from_final_sample": True,
            "production_burn_in_status": "unfrozen_requires_target_volume_pilot",
            "production_measurement_stride_status": "unfrozen_requires_target_volume_pilot",
            "production_seed_commitments_status": "unfrozen_until_final_campaign_freeze",
            "production_configuration_count_status": "unfrozen_until_target_volume_pilot",
            "veff_plateau_windows_status": "unfrozen_requires_preproduction_overlap_pilot",
            "fit_r_ranges_status": "unfrozen_requires_preproduction_analysis_plan",
            "no_production_data_may_choose_these_values": True,
        },
    }

    # EHK-like smearing intensity without post-data optimization.
    if abs(epsilon_times_iterations - 4.0) > 0.1:
        raise AssertionError(("smearing proxy drift", epsilon_times_iterations))
    # The vector program extends beyond the quoted r6/a while staying strictly
    # below the half box in every spatial component.
    if max(item["euclidean_radius"] for item in measurement_vectors) <= 10.89:
        raise AssertionError("vector program does not cover r6")
    if max(max(abs(x) for x in item["representative"]) for item in measurement_vectors) != 7:
        raise AssertionError("unexpected max component")
    if max(item["manhattan_steps_per_orientation"] for item in measurement_vectors) != 21:
        raise AssertionError("unexpected maximum Bresenham work")
    if max(item["cubic_orbit_size"] for item in measurement_vectors) != 24:
        raise AssertionError("unexpected maximum orbit size")
    if any(item["wilson_temporal_extents"] != list(range(1, 9)) for item in measurement_vectors):
        raise AssertionError("incomplete T coverage")
    if not design["finalization_requirements"]["pilot_configurations_excluded_from_final_sample"]:
        raise AssertionError("pilot/final sample boundary missing")

    text = json.dumps(design, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("ok")
    print("result_sha256=" + digest)
    print(text)

if __name__ == "__main__":
    main()
