#!/usr/bin/env python3
"""Freeze the beta=6 target-volume pre-production pilot protocol.

This subject freezes data-flow/authority boundaries, not cycle counts. A separate
pilot execution manifest must freeze machine-feasible budgets after throughput
measurement and before the first scientific pilot cycle.
"""
import hashlib
import json

PROTOCOL_ID = "beta6_target_volume_preproduction_pilot_protocol_v1"

def main():
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "measurement_design": {
            "pr": 2528,
            "result_sha256": "df92cdfbbf882a8f3c37eb911476eb961bbc7790d6d6adae60bccc1f8c999c72",
            "subject_sha256": "fb8ea28b82af95707164d436cdf4bb23531f63ebc1dbbe859d85a914faf9b387",
        },
        "throughput_lane": {
            "integration_pr": 2531,
            "integration_revision": "d7569beecf8bb5c775939d28dba9b8e09ccb2e7b",
            "scope": "performance_and_feasibility_only",
            "allowed_outputs": [
                "phase_wall_time",
                "resource_budget",
                "pilot_execution_budget_feasible_or_infeasible",
            ],
            "forbidden_outputs": [
                "burn_in_cycles",
                "measurement_stride",
                "veff_plateau_windows",
                "fit_r_ranges",
                "physics_thresholds",
                "benchmark_estimates",
            ],
        },
        "equilibration_lane": {
            "target_geometry": {"beta": 6.0, "dims": [16,16,16,32]},
            "pilot_start_classes": [
                "cold_identity_z3_zero",
                "cold_identity_z3_positive",
                "cold_identity_z3_negative",
                "deterministic_stress_v1",
            ],
            "independent_transition_stream_required_per_chain": True,
            "required_observables": [
                "plaquette",
                "polyakov_abs",
                "polyakov_center_aligned_real",
                "z3_sector_mobility",
                "flowed_q",
                "flowed_q_squared",
            ],
            "diagnostic_revisions": {
                "chain_statistics": "a3d631e56ccf052cbd1a9bf33d481e711378cf35",
                "center_symmetry": "5e275c1d2e5e03ee27f592db17fc1a7e84692489",
                "topology_diagnostics": "735b69f24b7cf6e9f7e56c9caace13e697bc1390",
                "rk3_flow": "93017e5207f7b0d52f3387fb1211f62fe10e804a",
            },
            "policy_requirements": {
                "qualification_policy_must_be_frozen_before_first_pilot_cycle": True,
                "pilot_execution_budget_must_be_frozen_before_first_pilot_cycle": True,
                "autocorrelation_window_must_close_for_required_slow_modes": True,
                "rank_and_folded_rhat_required": True,
                "topology_evidence_required": True,
                "center_mobility_evidence_required": True,
            },
            "allowed_outputs": [
                "production_burn_in_candidate",
                "production_measurement_stride_candidate",
                "block_adequacy_inputs",
                "equilibrium_promotion_evidence",
            ],
            "forbidden_outputs": [
                "final_string_tension",
                "final_sommer_scales",
                "external_benchmark_pass_fail",
            ],
        },
        "operator_overlap_lane": {
            "uses_measurement_design_result_sha256": "df92cdfbbf882a8f3c37eb911476eb961bbc7790d6d6adae60bccc1f8c999c72",
            "requires_equilibration_pilot_only_data": True,
            "wilson_t_coverage": [1,2,3,4,5,6,7,8],
            "allowed_outputs": [
                "primary_veff_plateau_windows",
                "diagnostic_veff_windows",
                "static_potential_fit_r_ranges",
                "operator_overlap_evidence",
            ],
            "freeze_outputs_before_final_production": True,
            "production_data_may_not_change_outputs": True,
        },
        "final_campaign_freeze": {
            "benchmark_retained_configuration_target": 4000,
            "retained_count_rule": "freeze_4000_if_pilot_feasible_else_declare_benchmark_reproduction_infeasible",
            "pilot_configurations_must_be_excluded_from_final_sample": True,
            "final_seed_commitments_frozen_after_pilot_before_production": True,
            "final_burn_in_and_stride_must_equal_pilot_bound_candidates": True,
            "final_plateau_and_r_ranges_must_equal_pilot_bound_choices": True,
            "production_data_may_not_relax_or_reselect_any_policy": True,
        },
        "chronology": [
            "freeze_measurement_design",
            "qualify_exact_code_subjects",
            "run_throughput_lane",
            "freeze_pilot_execution_budget_and_diagnostic_policy",
            "run_equilibration_and_operator_overlap_pilot",
            "freeze_final_campaign_schedule_seeds_windows_ranges",
            "authorize_final_production",
            "run_final_4000_configuration_reproduction_sample",
            "analyze_only_under_frozen_choices",
        ],
        "scientific_scope": "preproduction_pilot_protocol_only_not_execution_or_physics_authority",
    }

    allowed = set(protocol["throughput_lane"]["allowed_outputs"])
    forbidden = set(protocol["throughput_lane"]["forbidden_outputs"])
    if allowed & forbidden:
        raise AssertionError("throughput lane authority overlap")
    if "burn_in_cycles" not in forbidden or "measurement_stride" not in forbidden:
        raise AssertionError("throughput may influence physics schedule")
    if len(set(protocol["equilibration_lane"]["pilot_start_classes"])) != 4:
        raise AssertionError("pilot starts not unique")
    if not protocol["equilibration_lane"]["independent_transition_stream_required_per_chain"]:
        raise AssertionError("independent streams not required")
    required = set(protocol["equilibration_lane"]["required_observables"])
    for observable in ("plaquette","z3_sector_mobility","flowed_q","flowed_q_squared"):
        if observable not in required:
            raise AssertionError(("missing required slow-mode evidence", observable))
    if protocol["final_campaign_freeze"]["benchmark_retained_configuration_target"] != 4000:
        raise AssertionError("benchmark target count drift")
    if not protocol["final_campaign_freeze"]["pilot_configurations_must_be_excluded_from_final_sample"]:
        raise AssertionError("pilot/final firewall missing")
    if not protocol["operator_overlap_lane"]["production_data_may_not_change_outputs"]:
        raise AssertionError("post-hoc plateau/r-range selection allowed")
    if protocol["chronology"].index("run_throughput_lane") > protocol["chronology"].index("run_equilibration_and_operator_overlap_pilot"):
        raise AssertionError("throughput must precede scientific pilot")
    if protocol["chronology"].index("freeze_final_campaign_schedule_seeds_windows_ranges") > protocol["chronology"].index("authorize_final_production"):
        raise AssertionError("final choices must precede authorization")

    text = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("ok")
    print("result_sha256=" + digest)
    print(text)

if __name__ == "__main__":
    main()
