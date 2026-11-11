import unittest
from off_target_replicate_design import BUDGET, POLICIES, fixtures, enumerate_paths, run, self_test
from off_target_kernel_inference import Q_GRID, R_GRID

class OffTargetRepeatDesignTests(unittest.TestCase):
    def setUp(self):
        self.models, self.refs, self.ood, self.actions = fixtures()

    def test_both_designs_respect_same_action_budget(self):
        for repeat in (False, True):
            terms = enumerate_paths(self.models, Q_GRID, R_GRID, self.models[0], self.refs,
                                    self.actions, "balanced", 0.1, 0.75, 0.25,
                                    repeat, budget=BUDGET)
            self.assertTrue(all(t["used"] <= BUDGET for t in terms))

    def test_same_budget_designs_produce_different_profiles(self):
        data = run("test")
        distinct = [r for r in data["rows"] if r["design"] == "distinct_only"
                    and r["regime"] == "matched_clean" and r["policy"] == "joint_graph_eig"]
        repeated = [r for r in data["rows"] if r["design"] == "repeatable"
                    and r["regime"] == "matched_clean" and r["policy"] == "joint_graph_eig"]
        self.assertEqual(len(distinct), len(repeated))
        self.assertGreater(repeated[0]["graph_information_bits"], distinct[0]["graph_information_bits"])
        self.assertLess(repeated[0]["ood_surprise_auc"], distinct[0]["ood_surprise_auc"])

    def test_complete_grid_and_determinism(self):
        a, b = run("test"), run("test")
        self.assertEqual(a, b)
        self.assertEqual(len(a["rows"]), 4 * 2 * len(POLICIES) * 2)
        self.assertEqual({r["design"] for r in a["rows"]}, {"repeatable", "distinct_only"})

    def test_embedded_self_test(self):
        self_test()

if __name__ == "__main__": unittest.main()
