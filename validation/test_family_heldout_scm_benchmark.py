import unittest
from family_heldout_scm_benchmark import (
    MECHANISMS, TOPOLOGIES, SCM, build_suite, graph_scores, layered_graph_scores, make_scm, sample_scm, split_label, validate_scm,
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

    def test_sampling_is_deterministic_and_digest_bound(self):
        scm = make_scm(5, "chain", "and", 3)
        a = sample_scm(scm, 20, 99, noise_probability=0.1)
        b = sample_scm(scm, 20, 99, noise_probability=0.1)
        self.assertEqual(a, b)
        self.assertEqual(len(a["rows"]), 20)
        self.assertEqual(len(a["samples_sha256"]), 64)

    def test_hard_do_overrides_target_and_propagates_through_chain(self):
        scm = make_scm(5, "chain", "and", 7)
        rows = sample_scm(scm, 12, 1, noise_probability=0.0, intervention=(0, 0))["rows"]
        self.assertTrue(all(row[0] == 0 for row in rows))
        self.assertTrue(all(all(value == 0 for value in row) for row in rows))
        treated = sample_scm(scm, 12, 1, noise_probability=0.0, intervention=(0, 1))["rows"]
        self.assertTrue(all(row[0] == 1 for row in treated))
        self.assertTrue(all(all(value == 1 for value in row) for row in treated))

    def test_sampler_rejects_invalid_contract(self):
        scm = make_scm(4, "chain", "and", 0)
        with self.assertRaises(ValueError): sample_scm(scm, 0, 1)
        with self.assertRaises(ValueError): sample_scm(scm, 2, 1, noise_probability=0.5)
        with self.assertRaises(ValueError): sample_scm(scm, 2, 1, intervention=(4, 1))

    def test_layered_scores_separate_skeleton_direction_and_intervention_budget(self):
        score = layered_graph_scores(
            true_dag_edges=[(0, 1), (1, 2)],
            true_cpdag_directed_edges=[(0, 1)],
            predicted_directed_edges=[(0, 1)],
            predicted_undirected_edges=[(1, 2)],
            node_count=3, minimum_intervention_count=2, interventions_used=2,
        )
        self.assertEqual(score["skeleton_f1"], 1.0)
        self.assertEqual(score["compelled_f1"], 1.0)
        self.assertLess(score["directed_f1"], 1.0)
        self.assertEqual(score["unresolved_true_adjacencies"], 1)
        self.assertEqual(score["dag_shd"], 1)
        self.assertEqual(score["intervention_count_to_minimum_ratio"], 1.0)

    def test_layered_scoring_rejects_conflicting_or_noncanonical_edges(self):
        with self.assertRaises(ValueError):
            layered_graph_scores([(0, 1)], [], [(0, 1)], [(0, 1)], 2, 1, 0)
        with self.assertRaises(ValueError):
            layered_graph_scores([(0, 1)], [], [], [(1, 0)], 2, 1, 0)
        with self.assertRaises(ValueError):
            layered_graph_scores([(0, 1)], [], [(0, 1), (0, 1)], [], 2, 1, 0)
        with self.assertRaises(ValueError):
            layered_graph_scores([(0, 1)], [], [(0, 1), (1, 0)], [], 2, 1, 0)

    def test_invalid_and_duplicate_predictions_fail_closed(self):
        with self.assertRaises(ValueError): graph_scores([(0, 1)], [(0, 0)], 3)
        with self.assertRaises(ValueError): graph_scores([(0, 1)], [(0, 1), (0, 1)], 3)

if __name__ == "__main__": unittest.main()
