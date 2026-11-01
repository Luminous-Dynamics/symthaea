import unittest

from benchmark_projective_closure_index import _trial, run


class ProjectiveClosureIndexBenchmarkTests(unittest.TestCase):
    def test_trial_checks_index_against_scan(self):
        result = _trial(n=64, k=5, seed=974601)
        self.assertTrue(result["exact_scores_equal"])
        self.assertGreater(result["build_ms"], 0.0)
        self.assertGreater(result["indexed_update_and_query_ms"], 0.0)
        self.assertGreater(result["scan_update_and_query_ms"], 0.0)

    def test_small_benchmark_manifest(self):
        result = run(
            sizes=(32, 64),
            update_fractions=(0.05, 0.5),
            repeats=1,
            seed=974602,
        )
        self.assertEqual(result["benchmark"], "projective_closure_index_sparse_update")
        self.assertTrue(all(row["all_exact_scores_equal"] for row in result["results"]))
        self.assertTrue(result["build_cost_reported_separately"])


if __name__ == "__main__":
    unittest.main()
