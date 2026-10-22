"""Regression tests for the exact active probe-design benchmark."""
from __future__ import annotations
import unittest
from validation import active_probe_design as ap


class ActiveProbeDesignTests(unittest.TestCase):
    def test_all_two_input_boolean_functions_have_unique_truth_tables(self) -> None:
        tables = {tuple((identifier >> state) & 1 for state in range(4)) for identifier in range(16)}
        self.assertEqual(len(tables), 16)

    def test_information_gain_is_bounded_by_prior_entropy(self) -> None:
        case = ap.make_case(3, 3)
        posterior = tuple(range(len(case.hypotheses)))
        prior_entropy = ap.entropy(hypothesis.prior_weight for hypothesis in case.hypotheses)
        for probe in case.probes:
            gain = ap.expected_information_gain(case, posterior, probe)
            self.assertGreaterEqual(gain, -1e-12)
            self.assertLessEqual(gain, prior_entropy + 1e-12)

    def test_uninformative_probes_are_not_spent_by_exact_oracle(self) -> None:
        case = ap.Case(
            1,
            tuple(ap.Hypothesis(index, 1, (0, 0, 0, 0)) for index in range(4)),
            tuple(ap.Probe(index, index, 1, 0) for index in range(4)),
            2,
        )
        self.assertTrue(all(
            ap.expected_information_gain(case, (0, 1, 2, 3), probe) == 0.0
            for probe in case.probes
        ))
        self.assertEqual(ap.evaluate_policy(case, "exact_oracle").expected_cost, 0.0)

    def test_exact_oracle_dominates_heuristics_on_expected_entropy(self) -> None:
        for seed in range(25):
            case = ap.make_case(seed, 3)
            oracle = ap.evaluate_policy(case, "exact_oracle")
            for strategy in ("fixed_order", "cost_first", "greedy_eig", "greedy_efficiency"):
                result = ap.evaluate_policy(case, strategy)
                self.assertGreaterEqual(
                    result.expected_posterior_entropy_bits + 1e-10,
                    oracle.expected_posterior_entropy_bits,
                    (seed, strategy),
                )

    def test_expected_cost_respects_budget(self) -> None:
        for seed in range(20):
            case = ap.make_case(seed, 3)
            for strategy in ("fixed_order", "cost_first", "greedy_eig", "greedy_efficiency", "exact_oracle"):
                result = ap.evaluate_policy(case, strategy)
                self.assertLessEqual(result.expected_cost, case.budget + 1e-12)

    def test_exact_identification_probability_is_bounded(self) -> None:
        for seed in range(20):
            case = ap.make_case(seed, 2)
            for strategy in ("fixed_order", "cost_first", "greedy_eig", "greedy_efficiency", "exact_oracle"):
                result = ap.evaluate_policy(case, strategy)
                self.assertGreaterEqual(result.probability_exact_identification, 0.0)
                self.assertLessEqual(result.probability_exact_identification, 1.0)

    def test_seeded_benchmark_is_repeatable(self) -> None:
        self.assertEqual(ap.run_benchmark(5, 3, "test-source"), ap.run_benchmark(5, 3, "test-source"))

    def test_report_contains_accuracy_cost_and_damage_metrics(self) -> None:
        report = ap.run_benchmark(2, 3, "test-source")
        self.assertEqual(report["schema"], ap.SCHEMA)
        self.assertIn("mean_posterior_entropy_bits", report["summary"]["exact_oracle"])
        self.assertIn("mean_intervention_cost", report["summary"]["exact_oracle"])
        self.assertIn("mean_intervention_damage", report["summary"]["exact_oracle"])


if __name__ == "__main__":
    unittest.main()
