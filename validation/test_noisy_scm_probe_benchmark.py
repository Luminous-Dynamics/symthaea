import unittest
from dataclasses import replace

from validation.noisy_scm_probe_benchmark import (
    STRATEGIES,
    evaluate_policy,
    make_case,
    outcome_likelihood,
    posterior_outcomes,
    run_benchmark,
)


class NoisySCMProbeTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case(23, budget=2, node_count=3, hypothesis_count=6, noise=0.1)
        total = sum(self.case.prior_weights)
        self.prior = tuple(w / total for w in self.case.prior_weights)

    def test_seeded_generator_is_deterministic(self):
        self.assertEqual(self.case, make_case(23, budget=2, node_count=3, hypothesis_count=6, noise=0.1))

    def test_intervention_contract_is_hard_do(self):
        self.assertTrue(all(a.kind == "hard_do" for a in self.case.interventions))
        for action in self.case.interventions:
            branches = list(posterior_outcomes(self.case, self.prior, action))
            self.assertTrue(branches)
            self.assertTrue(all(outcome[action.target] == action.value for _, _, outcome in branches))

    def test_outcome_probabilities_normalize(self):
        for action in self.case.interventions:
            branches = list(posterior_outcomes(self.case, self.prior, action))
            self.assertAlmostEqual(sum(p for p, _, _ in branches), 1.0, places=10)
            for p, posterior, _ in branches:
                self.assertGreater(p, 0.0)
                self.assertAlmostEqual(sum(posterior), 1.0, places=10)

    def test_declared_parent_is_functionally_essential(self):
        for model in self.case.hypotheses:
            for node, parents in enumerate(model.parents):
                for pos in range(len(parents)):
                    table = model.truth_tables[node]
                    found = False
                    for assignment in range(1 << len(parents)):
                        if (assignment >> pos) & 1:
                            continue
                        paired = assignment | (1 << pos)
                        found |= ((table >> assignment) & 1) != ((table >> paired) & 1)
                    self.assertTrue(found)

    def test_hard_do_likelihood_ignores_target_mechanism(self):
        action = next(a for a in self.case.interventions if a.target == 1 and a.value == 0)
        model = self.case.hypotheses[0]
        outcome = [0] * self.case.node_count
        outcome[action.target] = action.value
        outcome = tuple(outcome)
        tables = list(model.truth_tables)
        target_parent_count = len(model.parents[action.target])
        tables[action.target] ^= (1 << (1 << target_parent_count)) - 1
        changed_target_mechanism = replace(model, truth_tables=tuple(tables))
        self.assertAlmostEqual(
            outcome_likelihood(model, action, outcome, self.case.noise),
            outcome_likelihood(changed_target_mechanism, action, outcome, self.case.noise),
            places=12,
        )

    def test_noisy_likelihood_is_positive_for_compatible_outcome(self):
        action = self.case.interventions[0]
        model = self.case.hypotheses[0]
        outcome = [0] * self.case.node_count
        outcome[action.target] = action.value
        self.assertGreater(outcome_likelihood(model, action, tuple(outcome), self.case.noise), 0.0)

    def test_exact_oracle_has_minimum_graph_entropy(self):
        results = {name: evaluate_policy(self.case, name) for name in STRATEGIES}
        oracle = results["exact_oracle"]
        for name, result in results.items():
            self.assertLessEqual(oracle.expected_graph_entropy_bits, result.expected_graph_entropy_bits + 1e-9, name)

    def test_budget_costs_are_bounded(self):
        for strategy in STRATEGIES:
            value = evaluate_policy(self.case, strategy)
            self.assertLessEqual(value.expected_cost, self.case.budget + 1e-10)
            self.assertGreaterEqual(value.expected_cost, 0.0)
            self.assertGreaterEqual(value.expected_disturbance, 0.0)

    def test_report_reproducible_and_source_bound(self):
        first = run_benchmark(2, 2, "receipt-test", 3, 6, 0.1)
        second = run_benchmark(2, 2, "receipt-test", 3, 6, 0.1)
        self.assertEqual(first, second)
        self.assertEqual(first["source_sha"], "receipt-test")
        self.assertEqual(first["case_count"], 2)

    def test_noise_parameter_rejected_at_boundary(self):
        with self.assertRaises(ValueError):
            make_case(1, noise=0.0)
        with self.assertRaises(ValueError):
            make_case(1, noise=0.5)


if __name__ == "__main__":
    unittest.main()
