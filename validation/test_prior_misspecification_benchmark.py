import unittest
from prior_misspecification_benchmark import (
    HORIZONS, POLICY_SET, PRIOR_NAMES, run_benchmark, self_test,
    terminal_brier_risk, terminal_cross_entropy,
)
from prior_shift_oracle_benchmark import build_prior
from objective_matched_oracle_benchmark import HYPOTHESES, OBJECTIVES

class PriorMisspecificationTests(unittest.TestCase):
    def test_prior_families_are_positive_and_normalized(self):
        for name in PRIOR_NAMES:
            p = build_prior(name)
            self.assertEqual(len(p), len(HYPOTHESES))
            self.assertTrue(all(v > 0 for v in p))
            self.assertAlmostEqual(sum(p), 1.0, places=12)

    def test_true_and_learner_sets_can_differ(self):
        out = run_benchmark(1, true_prior_names=("uniform",),
                            learner_prior_names=("high_spill_biased",),
                            policies=("greedy_joint", "oracle_robust_cross_entropy"))
        self.assertEqual(out["protocol"]["pairings"], 1)
        self.assertEqual(out["protocol"]["misspecified_pairings"], 1)
        self.assertTrue(all(not r["well_specified"] for r in out["rows"]))

    def test_proper_score_matches_identical_distributions(self):
        p = tuple(build_prior("uniform"))
        self.assertAlmostEqual(terminal_cross_entropy(p, p, "joint"), 4.0, places=12)
        self.assertAlmostEqual(terminal_brier_risk(p, p, "joint"), 1.0 - sum(x*x for x in p), places=12)

    def test_robust_oracle_has_zero_excess_cross_entropy(self):
        out = run_benchmark(2, true_prior_names=("uniform", "chain_biased"),
                            learner_prior_names=("uniform", "high_spill_biased"),
                            policies=("round_robin", "greedy_joint", "oracle_robust_cross_entropy"))
        oracle = [r for r in out["rows"] if r["policy"] == "oracle_robust_cross_entropy"]
        self.assertTrue(oracle)
        for row in oracle:
            self.assertAlmostEqual(row["excess_cross_entropy_bits_vs_robust_oracle"], 0.0, places=9)

    def test_policy_regret_is_nonnegative_up_to_numerical_tolerance(self):
        out = run_benchmark(1, true_prior_names=("uniform", "chain_biased"),
                            learner_prior_names=("uniform", "high_spill_biased"),
                            policies=("round_robin", "random_uniform", "greedy_graph", "greedy_joint", "oracle_robust_cross_entropy"))
        self.assertTrue(all(r["excess_cross_entropy_bits_vs_robust_oracle"] >= 0 for r in out["rows"]))

    def test_horizon_validation_fails_closed(self):
        with self.assertRaises(ValueError):
            run_benchmark(4, true_prior_names=("uniform",), learner_prior_names=("uniform",))

    def test_embedded_self_test(self):
        self.assertTrue(self_test())

if __name__ == "__main__":
    unittest.main()
