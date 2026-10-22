import unittest

from selective_causal_discovery import (
    ACQUISITION_POLICIES,
    DECISION_RULES,
    calibrate_surprise_threshold,
    evaluate,
    enumerate_terminals,
    fixtures,
    outcome_distribution,
    predictive_outcomes,
    run_benchmark,
    update_posterior,
)


class SelectiveCausalDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.models, self.worlds, self.actions = fixtures()

    def test_true_graph_support_is_explicit(self):
        self.assertTrue(any(m.graph_key() == self.worlds["in_class"].graph_key() for m in self.models))
        self.assertFalse(any(m.graph_key() == self.worlds["out_of_class"].graph_key() for m in self.models))

    def test_true_and_predictive_distributions_normalize(self):
        for world in self.worlds.values():
            for action in self.actions:
                true_dist = outcome_distribution(world, action, 0.2)
                self.assertAlmostEqual(sum(true_dist.values()), 1.0, places=12)
                pred = predictive_outcomes(self.models, (0.5, 0.5), action, 0.1)
                self.assertAlmostEqual(sum(pred.values()), 1.0, places=12)
                for outcome in pred:
                    post = update_posterior(self.models, (0.5, 0.5), action, outcome, 0.1)
                    self.assertAlmostEqual(sum(post), 1.0, places=12)

    def test_all_adaptive_path_masses_normalize(self):
        for policy in ACQUISITION_POLICIES:
            terminals = enumerate_terminals(self.models, self.worlds["out_of_class"], self.actions,
                                            policy, 0.1, 0.25, "surprise_gate_3_bits")
            self.assertAlmostEqual(sum(row.probability for row in terminals), 1.0, places=10)

    def test_false_claims_are_unavoidable_when_forced_to_commit_out_of_class(self):
        result = evaluate(self.models, self.worlds["out_of_class"], self.actions,
                          "greedy_eig", 0.1, 0.1, "always_commit")
        self.assertAlmostEqual(result["coverage"], 1.0, places=12)
        self.assertAlmostEqual(result["selective_error_rate"], 1.0, places=12)

    def test_stricter_surprise_threshold_does_not_increase_coverage(self):
        loose = evaluate(self.models, self.worlds["out_of_class"], self.actions,
                         "greedy_eig", 0.1, 0.1, "surprise_gate_4_bits")
        strict = evaluate(self.models, self.worlds["out_of_class"], self.actions,
                          "greedy_eig", 0.1, 0.1, "surprise_gate_2_bits")
        self.assertLessEqual(strict["coverage"], loose["coverage"] + 1e-12)

    def test_calibrated_threshold_controls_in_class_false_rejection(self):
        for policy in ACQUISITION_POLICIES:
            threshold, fpr = calibrate_surprise_threshold(
                self.models, self.worlds["in_class"], self.actions, policy, 0.1, alpha=0.05)
            result = evaluate(self.models, self.worlds["in_class"], self.actions, policy,
                              0.1, 0.1, "calibrated_surprise_alpha_0.05",
                              surprise_threshold_override=threshold,
                              calibration_false_rejection_rate=fpr)
            self.assertLessEqual(result["early_surprise_abstention_rate"], 0.05 + 1e-12)
            self.assertAlmostEqual(result["early_surprise_abstention_rate"], fpr, places=10)

    def test_calibrated_out_of_class_result_is_not_hidden(self):
        threshold, fpr = calibrate_surprise_threshold(
            self.models, self.worlds["in_class"], self.actions, "assumed_model_oracle", 0.1)
        result = evaluate(self.models, self.worlds["out_of_class"], self.actions,
                          "assumed_model_oracle", 0.1, 0.1,
                          "calibrated_surprise_alpha_0.05",
                          surprise_threshold_override=threshold,
                          calibration_false_rejection_rate=fpr)
        self.assertAlmostEqual(result["selective_error_rate"], 1.0, places=12)
        self.assertGreaterEqual(result["early_surprise_abstention_rate"], 0.0)

    def test_report_shape_and_reproducibility(self):
        one = run_benchmark("test")
        two = run_benchmark("test")
        self.assertEqual(one, two)
        self.assertEqual(one["source_git_blob"], "test")
        self.assertEqual(len(one["rows"]), 2 * 2 * len(ACQUISITION_POLICIES) * len(DECISION_RULES))


if __name__ == "__main__":
    unittest.main()
