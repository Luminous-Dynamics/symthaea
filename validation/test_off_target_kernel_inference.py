import unittest
from off_target_kernel_inference import (
    Q_GRID, R_GRID, fixtures, outcome_distribution_kernel, mixture_predictive,
    update_posterior, action_information, enumerate_paths, run, self_test,
)

class OffTargetKernelInferenceTests(unittest.TestCase):
    def setUp(self):
        self.models, self.refs, self.ood, self.actions = fixtures()

    def test_kernel_distributions_normalize(self):
        for q in (0.0, 0.5, 0.75, 1.0):
            for r in (0.0, 0.25, 1.0):
                for noise in (0.0, 0.1, 0.25):
                    for world in (*self.models, *self.refs, *self.ood.values()):
                        for action in self.actions:
                            p = outcome_distribution_kernel(world, action, noise, q, r)
                            self.assertAlmostEqual(sum(p.values()), 1.0, places=12)

    def test_hard_target_and_crosstalk_are_distinct(self):
        action = self.actions[0]
        clean = outcome_distribution_kernel(self.models[0], action, 0.1, 1.0, 0.0)
        spill = outcome_distribution_kernel(self.models[0], action, 0.1, 1.0, 0.25)
        self.assertTrue(all(y[action.target] == action.value for y in clean))
        self.assertNotEqual(clean, spill)

    def test_joint_predictive_and_posterior_normalize(self):
        count = len(self.models) * len(Q_GRID) * len(R_GRID)
        posterior = tuple([1.0 / count] * count)
        action = self.actions[0]
        p = mixture_predictive(self.models, Q_GRID, R_GRID, posterior, action, 0.1)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=12)
        post = update_posterior(self.models, Q_GRID, R_GRID, posterior, action, (0,0,0), 0.1)
        self.assertAlmostEqual(sum(post), 1.0, places=12)

    def test_action_information_nonnegative_and_jsd_bounded(self):
        count = len(self.models) * len(Q_GRID) * len(R_GRID)
        posterior = tuple([1.0 / count] * count)
        c = action_information(self.models, Q_GRID, R_GRID, posterior, self.actions[0], 0.1, self.refs)
        for key in ("eig_graph", "eig_q", "eig_r", "eig_joint"):
            self.assertGreaterEqual(c[key], 0.0)
        self.assertTrue(0.0 <= c["jsd_reference"] <= 1.0)

    def test_joint_kernel_in_class_paths_keep_support(self):
        for q, r in ((0.75, 0.0), (0.75, 0.25), (0.5, 0.25)):
            for world in self.models:
                terms = enumerate_paths(self.models, Q_GRID, R_GRID, world, self.refs,
                                        self.actions, "balanced", 0.1, q, r)
                self.assertTrue(all(not t["impossible"] for t in terms))
                self.assertAlmostEqual(sum(t["p"] for t in terms), 1.0, places=10)

    def test_naive_hard_do_has_zero_support_under_partial_off_target_actions(self):
        res = run("test")
        rows = [r for r in res["rows"] if r["policy"] == "naive_perfect_hard_do"
                and r["actual_q"] < 1.0]
        self.assertTrue(rows)
        self.assertTrue(any(r["in_class_zero_support_mass"] > 0.0 for r in rows))

    def test_complete_deterministic_grid(self):
        a, b = run("test"), run("test")
        self.assertEqual(a, b)
        self.assertEqual(len(a["rows"]), 4 * 2 * 6)

    def test_embedded_self_test(self):
        self_test()

if __name__ == "__main__": unittest.main()
