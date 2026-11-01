import unittest

from benchmark_projective_closure_receipt import _trial, run


class ProjectiveClosureReceiptBenchmarkTests(unittest.TestCase):
    def test_trial_authenticates_exact_score_against_scan(self):
        result = _trial(64, 975901)
        self.assertTrue(result["receipt_score_exactly_matches_scan"])
        self.assertGreater(result["proof_bytes"], 0)
        self.assertGreater(result["prefix_proof_nodes"], 0)
        self.assertGreater(result["verification_ms"], 0.0)

    def test_small_benchmark_is_reproducible_in_shape(self):
        result = run(sizes=(16, 32), repeats=1, seed=975902)
        self.assertEqual(result["benchmark"], "merkle_projective_closure_receipt")
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all(row["all_receipts_match_scan"] for row in result["results"]))


if __name__ == "__main__":
    unittest.main()
