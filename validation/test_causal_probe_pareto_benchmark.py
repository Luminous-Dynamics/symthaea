import math
import unittest
from causal_probe_pareto_benchmark import (
    fixtures, js_divergence, distribution_mixture, action_components, enumerate_policy_paths,
    score_policy, run, self_test,
)
from intervention_reliability_audit import outcome_distribution

class CausalProbeParetoTests(unittest.TestCase):
    def setUp(self):
        self.models, self.refs, self.ood_test, self.actions = fixtures()

    def test_all_action_distributions_normalize(self):
        for q in (0.0, 0.5, 0.75, 1.0):
            for noise in (0.0, 0.1, 0.25):
                for model in (*self.models, *self.refs, self.ood_test):
                    for action in self.actions:
                        self.assertAlmostEqual(sum(outcome_distribution(model, action, noise, q).values()), 1.0, places=12)

    def test_js_divergence_symmetry_and_bounds(self):
        p = {(0,): 1.0}; q = {(1,): 1.0}
        self.assertEqual(js_divergence(p, q), 1.0)
        self.assertAlmostEqual(js_divergence(p, p), 0.0, places=12)
        self.assertAlmostEqual(js_divergence(p, q), js_divergence(q, p), places=12)

    def test_information_components_are_bounded(self):
        eig, jsd = action_components(self.models, (0.5, 0.5), self.refs, self.actions[0], 0.1, 1.0)
        self.assertGreaterEqual(eig, 0.0)
        self.assertTrue(0.0 <= jsd <= 1.0)

    def test_reference_bank_weights_cover_every_model(self):
        expanded = (*self.refs, *self.models)
        action = self.actions[1]
        narrow = distribution_mixture(self.refs, [1.0] * len(self.refs), action, 0.1, 1.0)
        broad = distribution_mixture(expanded, [1.0] * len(expanded), action, 0.1, 1.0)
        self.assertAlmostEqual(sum(narrow.values()), 1.0, places=12)
        self.assertAlmostEqual(sum(broad.values()), 1.0, places=12)
        self.assertTrue(any(abs(narrow.get(y, 0.0) - broad.get(y, 0.0)) > 1e-8
                            for y in set(narrow) | set(broad)))
        with self.assertRaises(ValueError):
            distribution_mixture(expanded, [1.0] * 2, action, 0.1, 1.0)

    def test_policy_path_mass_normalizes(self):
        terms = enumerate_policy_paths(self.models, self.ood_test, self.refs, self.actions,
                                       "balanced_0_5", 0.5, 0.1, 0.75, 0.75)
        self.assertAlmostEqual(sum(t["p"] for t in terms), 1.0, places=10)

    def test_actuator_misspecification_is_flagged(self):
        for q in (0.5, 0.75):
            bad = score_policy(self.models, self.refs, self.ood_test, self.actions,
                               "eig_only", 0.0, 0.1, q, 1.0)
            good = score_policy(self.models, self.refs, self.ood_test, self.actions,
                                "eig_only", 0.0, 0.1, q, q)
            self.assertGreater(bad["in_class_zero_support_mass"], 0.0)
            self.assertAlmostEqual(good["in_class_zero_support_mass"], 0.0, places=12)

    def test_ood_auc_stays_in_range(self):
        for row in run("test")["rows"]:
            self.assertTrue(0.0 <= row["ood_cumulative_surprise_auc"] <= 1.0)

    def test_run_grid_count_and_determinism(self):
        first, second = run("test"), run("test")
        self.assertEqual(first, second)
        self.assertEqual(len(first["rows"]), 30)

    def test_embedded_self_test(self):
        self_test()

if __name__ == "__main__": unittest.main()
