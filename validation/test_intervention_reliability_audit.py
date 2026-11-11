import math
import unittest
from intervention_reliability_audit import (
    fixtures, outcome_distribution, enumerate_paths, evaluate_cell, run,
    weighted_auc, update,
)

class InterventionReliabilityAuditTests(unittest.TestCase):
    def setUp(self):
        self.models, self.worlds, self.actions = fixtures()

    def test_exact_distributions_normalize_across_reliability_grid(self):
        for q in (0.0, 0.5, 0.75, 1.0):
            for noise in (0.0, 0.1, 0.25):
                for world in self.worlds.values():
                    for action in self.actions:
                        p = outcome_distribution(world, action, noise, q)
                        self.assertAlmostEqual(sum(p.values()), 1.0, places=12)

    def test_perfect_compliance_clamps_target(self):
        action = self.actions[0]
        p = outcome_distribution(self.worlds["chain"], action, 0.1, 1.0)
        self.assertTrue(all(y[action.target] == action.value for y in p))

    def test_zero_compliance_allows_target_to_follow_mechanism(self):
        action = self.actions[0]
        p = outcome_distribution(self.worlds["chain"], action, 0.1, 0.0)
        self.assertGreater(sum(prob for y, prob in p.items() if y[action.target] != action.value), 0.0)

    def test_invalid_noise_and_compliance_rejected(self):
        with self.assertRaises(ValueError): outcome_distribution(self.worlds["chain"], self.actions[0], 0.7, 1.0)
        with self.assertRaises(ValueError): outcome_distribution(self.worlds["chain"], self.actions[0], 0.1, 1.1)

    def test_posterior_normalizes(self):
        post = update(self.models, (0.5, 0.5), self.actions[0], (0,0,0), 0.1, 0.75)
        self.assertAlmostEqual(sum(post),1.0,places=12)

    def test_zero_likelihood_path_is_explicit_and_fail_closed(self):
        terms=enumerate_paths(self.models,self.worlds["chain"],self.actions,"greedy_eig",0.1,0.5,1.0)
        self.assertGreater(sum(t.probability for t in terms if t.zero_likelihood),0.0)
        self.assertTrue(all(math.isinf(t.surprise_bits) for t in terms if t.zero_likelihood))

    def test_auc_known_extremes(self):
        self.assertEqual(weighted_auc([(1.0, 1.0)],[(1.0,0.0)]),1.0)
        self.assertEqual(weighted_auc([(1.0, 0.0)],[(1.0,1.0)]),0.0)
        self.assertEqual(weighted_auc([(1.0, 1.0)],[(1.0,1.0)]),0.5)

    def test_cell_metrics_are_bounded(self):
        row=evaluate_cell(self.models,self.worlds,self.actions,"greedy_eig",0.1,0.5,1.0)
        self.assertTrue(0.0 <= row["ood_surprise_auc"] <= 1.0)
        self.assertTrue(0.0 <= row["in_class_zero_likelihood_path_mass"] <= 1.0)

    def test_report_grid_is_complete_and_deterministic(self):
        one=run("test"); two=run("test")
        self.assertEqual(one,two)
        self.assertEqual(len(one["rows"]),18)
        self.assertEqual({r["actual_compliance"] for r in one["rows"]},{0.5,0.75,1.0})
        self.assertEqual({r["mechanism_noise"] for r in one["rows"]},{0.1,0.25})

    def test_embedded_self_test(self):
        from intervention_reliability_audit import self_test
        self_test()

if __name__ == "__main__": unittest.main()
