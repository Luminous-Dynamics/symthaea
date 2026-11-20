import unittest
import random
from adaptive_probe_policy_benchmark import (
    POLICIES, choose_action, expected_information, posterior_state, run_benchmark, self_test,
)
from action_subset_kernel_benchmark import true_kernels

class AdaptiveProbePolicyTests(unittest.TestCase):
    def test_information_gain_finite_and_nonnegative(self):
        for objective in ("graph", "kernel", "joint"):
            for action in range(3):
                value = expected_information([], action, "exact_target_subset", objective)
                self.assertGreaterEqual(value, 0.0)

    def test_posterior_normalizes(self):
        state = posterior_state([], "exact_target_subset")
        self.assertAlmostEqual(sum(state["graph_posterior"]), 1.0, places=12)
        for action in range(3):
            for row in state["kernel_conditional"][action]:
                self.assertAlmostEqual(sum(row), 1.0, places=12)

    def test_every_policy_respects_equal_probe_budget(self):
        out = run_benchmark(replicates=1, budget=3, modes=("exact_target_subset",))
        self.assertEqual(len(out["rows"]), 4 * 2 * len(POLICIES))
        self.assertTrue(all(sum(r["action_counts"]) == 3 for r in out["rows"]))

    def test_fixed_policy_distributes_budget_round_robin(self):
        out = run_benchmark(replicates=1, budget=4, modes=("exact_target_subset",))
        rows = [r for r in out["rows"] if r["policy"] == "round_robin"]
        self.assertTrue(all(r["action_counts"] == [2, 1, 1] for r in rows))

    def test_truth_regimes_are_distinct(self):
        inside, _ = true_kernels("in_support")
        outside, _ = true_kernels("out_of_support")
        self.assertNotEqual(inside, outside)

    def test_coverage_policy_visits_each_target_before_repeating(self):
        data = []
        rng = random.Random(3)
        chosen = []
        for step in range(3):
            action = choose_action("coverage_then_joint_information", data, step,
                                  "exact_target_subset", rng, budget=4)
            chosen.append(action)
            data.append({"action": action, "subset": 0, "y": 0, "flag": 0})
        self.assertEqual(len(set(chosen)), 3)

    def test_embedded_self_test(self):
        self.assertTrue(self_test())

if __name__ == "__main__":
    unittest.main()
