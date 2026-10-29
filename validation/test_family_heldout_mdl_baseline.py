import unittest
from family_heldout_scm_benchmark import make_scm, sample_scm
from family_heldout_mdl_baseline import (
    enumerate_dags, graph_scores, sample_arms, select_dag_mdl, run_benchmark,
)

class FamilyHeldoutMdlBaselineTests(unittest.TestCase):
    def test_exact_number_of_four_node_dags(self):
        self.assertEqual(len(enumerate_dags(4)), 543)

    def test_exhaustive_search_rejects_unsupported_sizes(self):
        with self.assertRaises(ValueError): enumerate_dags(5)

    def test_equal_total_sample_budget(self):
        scm = make_scm(4, "chain", "and", 3)
        obs, mixed = sample_arms(scm, observation_count=24, intervention_observation_count=16,
                                 per_intervention_count=1, noise=0.1)
        self.assertEqual(sum(len(batch["rows"]) for batch in obs), 24)
        self.assertEqual(sum(len(batch["rows"]) for batch in mixed), 24)

    def test_deterministic_hard_do_datasets(self):
        scm = make_scm(4, "chain", "and", 2)
        a = sample_scm(scm, 10, 32, 0.0, (0, 1))
        b = sample_scm(scm, 10, 32, 0.0, (0, 1))
        self.assertEqual(a, b)
        self.assertTrue(all(row[0] == 1 for row in a["rows"]))

    def test_mdl_returns_a_valid_dag_and_deterministic_result(self):
        scm = make_scm(4, "chain", "and", 1)
        obs, mixed = sample_arms(scm, 32, 16, 2, 0.1)
        first = select_dag_mdl(obs, 4, 0.1)
        second = select_dag_mdl(obs, 4, 0.1)
        self.assertEqual(first, second)
        score = graph_scores(scm.edges(), first["edges"], 4)
        self.assertIn("directed_f1", score)
        self.assertEqual(first["candidate_dag_count"], 543)

    def test_full_benchmark_groups_holdout_regimes(self):
        report = run_benchmark(cases_per_cell=1, observation_count=24,
                               intervention_observation_count=16, per_intervention_count=1)
        self.assertEqual(report["protocol"]["total_worlds"], 30)
        self.assertEqual(set(report["split_counts"]), {"train_id", "topology_ood", "mechanism_ood", "combined_ood"})
        self.assertEqual(sum(report["split_counts"].values()), 30)
        self.assertTrue(all("cluster_bootstrap_95_ci" in value for split in report["summary_by_split"].values()
                            for value in split["paired_deltas"].values()))

if __name__ == "__main__": unittest.main()
