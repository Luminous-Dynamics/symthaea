import unittest

from benchmark_donation_e_lond_reference import _trial, run


class DonationELONDBenchmarkTests(unittest.TestCase):
    def test_trial_matches_scan_and_verifier(self):
        result = _trial(128, 981301)
        self.assertTrue(result["final_threshold_exact_match"])
        self.assertTrue(result["final_verifier_passed"])
        self.assertGreater(result["indexed_online_run_ms"], 0.0)
        self.assertGreater(result["final_scan_query_ms"], 0.0)
        self.assertGreater(result["final_verifier_ms"], 0.0)

    def test_small_benchmark_shape(self):
        result = run(sizes=(32, 64), repeats=1, seed=981302)
        self.assertEqual(result["benchmark"], "donation_e_lond_avl_vs_scan")
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all(row["all_final_thresholds_match"] for row in result["results"]))
        self.assertTrue(all(row["all_verifiers_pass"] for row in result["results"]))


if __name__ == "__main__":
    unittest.main()
