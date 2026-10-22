import math
import unittest
from joint_graph_actuator_inference import (
    Q_GRID, fixtures, mixture_predictive, update_joint, action_components,
    enumerate_paths, run, self_test,
)

class JointGraphActuatorInferenceTests(unittest.TestCase):
    def setUp(self):
        self.models, self.references, self.ood, self.actions = fixtures()
        self.q_values = Q_GRID
        self.posterior = tuple([1.0 / (len(self.models) * len(self.q_values))] *
                               (len(self.models) * len(self.q_values)))

    def test_joint_predictive_and_posterior_normalize(self):
        action = self.actions[0]
        dist = mixture_predictive(self.models, self.q_values, self.posterior, action, 0.1)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=12)
        post = update_joint(self.models, self.q_values, self.posterior, action, (0,0,0), 0.1)
        self.assertAlmostEqual(sum(post), 1.0, places=12)

    def test_reference_divergence_is_bounded(self):
        components = action_components(self.models, self.q_values, self.posterior,
                                       self.references, self.actions[0], 0.1)
        self.assertGreaterEqual(components["eig_graph"], 0.0)
        self.assertGreaterEqual(components["eig_q"], 0.0)
        self.assertGreaterEqual(components["eig_joint"], 0.0)
        self.assertTrue(0.0 <= components["jsd_reference"] <= 1.0)

    def test_joint_posterior_has_no_zero_support_on_grid(self):
        for world in self.models:
            terms = enumerate_paths(self.models, self.q_values, world, self.references,
                                    self.actions, "balanced_falsification", 0.1, 0.75)
            self.assertTrue(all(not t["impossible"] for t in terms))
            self.assertAlmostEqual(sum(t["probability"] for t in terms), 1.0, places=10)

    def test_naive_perfect_compliance_baseline_fails_under_partial_actuation(self):
        result = run("test")
        naive = [row for row in result["rows"] if row["policy"] == "naive_perfect_compliance"
                 and row["actual_compliance"] == 0.5]
        self.assertTrue(naive)
        self.assertTrue(all(row["in_class_zero_support_mass"] > 0.0 for row in naive))

    def test_complete_deterministic_grid(self):
        one, two = run("test"), run("test")
        self.assertEqual(one, two)
        self.assertEqual(len(one["rows"]), 3 * 2 * 2 * 5)

    def test_embedded_self_test(self):
        self_test()

if __name__ == "__main__": unittest.main()
