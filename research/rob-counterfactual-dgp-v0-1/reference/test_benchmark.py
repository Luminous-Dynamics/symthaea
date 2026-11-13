from __future__ import annotations
import unittest
from dgp import (
    DGPConfig, SCENARIOS, generate_scenario, evaluate_scenario,
    validate_artifact_separation, run_all, FORBIDDEN_OBSERVED_KEYS,
)


class KnownTruthDGPTests(unittest.TestCase):
    def test_deterministic_generation_and_report(self):
        cfg = DGPConfig(seed=17, blocks=200)
        a = generate_scenario("conditional_endpoint_loss", cfg)
        b = generate_scenario("conditional_endpoint_loss", cfg)
        self.assertEqual(a, b)
        self.assertEqual(evaluate_scenario("conditional_endpoint_loss", *a[:2], a[2], cfg),
                         evaluate_scenario("conditional_endpoint_loss", *b[:2], b[2], cfg))

    def test_all_scenarios_have_exactly_one_episode_per_policy_per_block(self):
        for name in SCENARIOS:
            observed, truth, _ = generate_scenario(name, DGPConfig(blocks=100))
            validate_artifact_separation(observed, truth)
            self.assertEqual(len(observed), 200)

    def test_observed_artifact_does_not_leak_potential_outcomes(self):
        observed, truth, _ = generate_scenario("complete_followup", DGPConfig(blocks=20))
        self.assertTrue(all(not (FORBIDDEN_OBSERVED_KEYS & set(row)) for row in observed))
        leaked = [dict(observed[0], reference_event_time=1.0)]
        with self.assertRaisesRegex(ValueError, "leaks oracle-only"):
            validate_artifact_separation(leaked, [truth[0]])

    def test_assignment_and_receipt_precede_time_zero(self):
        observed, truth, _ = generate_scenario("complete_followup", DGPConfig(blocks=20))
        for row in observed:
            self.assertLess(row["assignment_time"], row["time_zero"])
            self.assertTrue(row["assignment_precedes_time_zero"])
            self.assertTrue(row["randomization_receipt_valid"])
        corrupt = [dict(x) for x in observed]
        corrupt[0]["assignment_time"] = 0.0
        with self.assertRaisesRegex(ValueError, "assignment must precede"):
            validate_artifact_separation(corrupt, truth)

    def test_candidate_arm_observation_contains_only_assigned_policy_outcome(self):
        observed, truth, _ = generate_scenario("complete_followup", DGPConfig(blocks=30))
        truth_by_id = {x["episode_id"]: x for x in truth}
        for row in observed:
            expected = truth_by_id[row["episode_id"]]["assigned_policy_outcome"]
            self.assertEqual(row["policy_outcome"], expected)
            if row["assigned_policy"] == "anticipatory_safe" and row["intervention_applied"]:
                self.assertIsNotNone(truth_by_id[row["episode_id"]]["reference_event_time"])
                self.assertNotIn("reference_event_time", row)

    def test_unknown_endpoint_is_not_relabelled_event_free(self):
        observed, truth, _ = generate_scenario("conditional_endpoint_loss", DGPConfig(blocks=100))
        missing = [x for x in observed if not x["followup_complete_by_horizon"]]
        self.assertTrue(missing)
        self.assertTrue(all(x["policy_outcome"] is None for x in missing))
        self.assertTrue(all(x["followup_status"] in {
            "administrative_endpoint_not_obtained", "telemetry_loss_before_horizon"
        } for x in missing))

    def test_mcar_complete_case_and_known_probability_diagnostic_are_close_to_truth(self):
        cfg = DGPConfig(seed=20261113, blocks=4000)
        observed, truth, g = generate_scenario("mcar_endpoint_loss", cfg)
        report = evaluate_scenario("mcar_endpoint_loss", observed, truth, g, cfg)
        lane = report["reference_policy_forecast_lane"]
        self.assertTrue(lane["oracle_ipw_diagnostic"]["estimable"])
        self.assertLess(lane["complete_case_absolute_error"], 0.035)
        self.assertLess(lane["oracle_ipw_absolute_error"], 0.035)

    def test_conditional_endpoint_loss_biases_complete_cases_and_known_g_corrects_diagnostic(self):
        cfg = DGPConfig(seed=20261113, blocks=10_000)
        observed, truth, g = generate_scenario("conditional_endpoint_loss", cfg)
        report = evaluate_scenario("conditional_endpoint_loss", observed, truth, g, cfg)
        lane = report["reference_policy_forecast_lane"]
        self.assertTrue(lane["oracle_ipw_diagnostic"]["estimable"])
        self.assertGreater(lane["complete_case_absolute_error"], 0.025)
        self.assertLess(lane["oracle_ipw_absolute_error"], lane["complete_case_absolute_error"])
        self.assertLess(lane["oracle_ipw_absolute_error"], 0.025)

    def test_positivity_failure_refuses_oracle_ipw_extrapolation(self):
        cfg = DGPConfig(seed=20261113, blocks=4000)
        observed, truth, g = generate_scenario("positivity_failure", cfg)
        report = evaluate_scenario("positivity_failure", observed, truth, g, cfg)
        ipw = report["reference_policy_forecast_lane"]["oracle_ipw_diagnostic"]
        self.assertFalse(ipw["estimable"])
        self.assertEqual(ipw["minimum_observation_probability"], 0.0)
        self.assertIn("positivity_failure", ipw["reason"])

    def test_complete_followup_randomized_itt_matches_finite_sample_oracle(self):
        cfg = DGPConfig(seed=20261113, blocks=3000)
        observed, truth, g = generate_scenario("complete_followup", cfg)
        report = evaluate_scenario("complete_followup", observed, truth, g, cfg)
        value = report["policy_value_lane"]
        itt = value["randomized_fixed_horizon_itt"]
        self.assertTrue(itt["estimable"])
        self.assertAlmostEqual(itt["candidate_minus_control_risk_difference"],
                               value["oracle_candidate_minus_reference_risk_difference"], places=12)
        self.assertFalse(value["individual_counterfactuals_identified_from_observed_episode"])

    def test_naive_as_treated_can_reverse_the_itt_direction(self):
        cfg = DGPConfig(seed=20261113, blocks=8000)
        observed, truth, g = generate_scenario("complete_followup", cfg)
        report = evaluate_scenario("complete_followup", observed, truth, g, cfg)
        value = report["policy_value_lane"]
        true_itt = value["oracle_candidate_minus_reference_risk_difference"]
        naive = value["naive_as_treated"]["naive_received_minus_not_received_risk_difference"]
        self.assertLess(true_itt, 0.0)
        self.assertGreater(naive, 0.0)
        self.assertFalse(value["naive_as_treated"]["is_randomized_itt"])

    def test_incomplete_followup_blocks_unadjusted_itt_but_preserves_denominator(self):
        cfg = DGPConfig(seed=23, blocks=200)
        observed, truth, g = generate_scenario("mcar_endpoint_loss", cfg)
        report = evaluate_scenario("mcar_endpoint_loss", observed, truth, g, cfg)
        itt = report["policy_value_lane"]["randomized_fixed_horizon_itt"]
        self.assertFalse(itt["estimable"])
        self.assertEqual(itt["assigned_denominator_preserved"], len(observed))
        self.assertGreater(itt["missing_outcomes"], 0)

    def test_competing_terminal_events_are_counted_separately(self):
        cfg = DGPConfig(seed=71, blocks=1000)
        observed, truth, g = generate_scenario("complete_followup", cfg)
        report = evaluate_scenario("complete_followup", observed, truth, g, cfg)
        evidence = report["evidence_accounting"]
        self.assertGreater(evidence["oracle_reference_competing_n"], 0)
        self.assertGreater(evidence["oracle_candidate_competing_n"], 0)
        self.assertLessEqual(evidence["oracle_reference_competing_n"], evidence["reference_arm_assigned_n"])

    def test_run_all_emits_separated_scenario_reports(self):
        report, data = run_all(DGPConfig(seed=3, blocks=100))
        self.assertEqual(set(report["scenarios"]), set(SCENARIOS))
        self.assertEqual(set(data), set(SCENARIOS))
        self.assertTrue(report["synthetic_only"])

    def test_config_rejects_odd_or_small_block_count(self):
        with self.assertRaises(ValueError):
            DGPConfig(blocks=3).validate()
        with self.assertRaises(ValueError):
            DGPConfig(blocks=0).validate()


if __name__ == "__main__":
    unittest.main()
