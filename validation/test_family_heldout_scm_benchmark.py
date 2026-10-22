import unittest
from family_heldout_scm_benchmark import (
    MECHANISMS, TOPOLOGIES, SCM, build_suite, graph_scores, make_scm, split_label, validate_scm,
)

class FamilyHeldoutTests(unittest.TestCase):
    def test_suite_is_deterministic(self):
        self.assertEqual(build_suite(5, 2), build_suite(5, 2))

    def test_split_separates_topology_and_mechanism_families(self):
        self.assertEqual(split_label("chain", "and"), "train_id")
        self.assertEqual(split_label("collider", "and"), "topology_ood")
        self.assertEqual(split_label("chain", "xor"), "mechanism_ood")
        self.assertEqual(split_label("diamond", "or"), "combined_ood")

    def test_all_generated_parents_are_essential_and_graph_is_acyclic(self):
        for topology in TOPOLOGIES:
            for mechanism in MECHANISMS:
                validate_scm(make_scm(6, topology, mechanism, 13))

    def test_split_grid_covers_all_four_regimes(self):
        suite = build_suite(5, 2)
        self.assertEqual(set(suite["split_counts"]), {"train_id", "topology_ood", "mechanism_ood", "combined_ood"})
        self.assertEqual(sum(suite["split_counts"].values()), 6 * 5 * 2)

    def test_validator_rejects_cycles_in_supplied_parent_lists(self):
        cyclic = SCM("cycle", 0, 4, "chain", "and", ((1,), (0,), (), ()), (0b10, 0b10, 0, 1))
        with self.assertRaises(AssertionError):
            validate_scm(cyclic)

    def test_exact_graph_has_perfect_scores(self):
        graph = [(0, 1), (1, 2)]
        score = graph_scores(graph, graph, 3)
        self.assertEqual(score["shd"], 0)
        self.assertEqual(score["directed_f1"], 1.0)
        self.assertEqual(score["skeleton_f1"], 1.0)

    def test_reversal_separates_skeleton_from_direction(self):
        score = graph_scores([(0, 1), (1, 2)], [(1, 0), (1, 2)], 3)
        self.assertEqual(score["skeleton_f1"], 1.0)
        self.assertLess(score["directed_f1"], 1.0)
        self.assertEqual(score["shd"], 1)

    def test_missing_edges_increase_shd(self):
        score = graph_scores([(0, 1), (1, 2)], [], 3)
        self.assertEqual(score["shd"], 2)
        self.assertEqual(score["directed_recall"], 0.0)

    def test_invalid_and_duplicate_predictions_fail_closed(self):
        with self.assertRaises(ValueError): graph_scores([(0, 1)], [(0, 0)], 3)
        with self.assertRaises(ValueError): graph_scores([(0, 1)], [(0, 1), (0, 1)], 3)

if __name__ == "__main__": unittest.main()
