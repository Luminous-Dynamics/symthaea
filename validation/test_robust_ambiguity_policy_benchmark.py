import math
import unittest

from robust_ambiguity_policy_benchmark import (
    eval_policy, robust_action, run_benchmark, self_test,
)
from objective_matched_oracle_benchmark import ACTIONS, HYPOTHESES
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior


class RobustAmbiguityPolicyTests(unittest.TestCase):
    def test_all_declared_priors_are_positive_and_normalized(self):
        for name in PRIOR_NAMES:
            p = build_prior(name)
            self.assertEqual(len(p), len(HYPOTHESES))
            self.assertTrue(all(x > 0 for x in p))
            self.assertAlmostEqual(sum(p), 1.0, places=12)

    def test_robust_policy_returns_valid_action(self):
        scenarios = tuple((n, tuple(build_prior(n))) for n in PRIOR_NAMES)
        self.assertIn(robust_action(scenarios, tuple(build_prior("uniform")), "joint"), ACTIONS)

    def test_action_deterministic_for_fixed_ambiguity_set_and_posterior(self):
        scenarios = tuple((n, tuple(build_prior(n))) for n in ("uniform", "high_spill_biased", "chain_biased"))
        p = tuple(build_prior("uniform"))
        self.assertEqual(
            robust_action(scenarios, p, "joint"),
            robust_action(scenarios, p, "joint"),
        )

    def test_small_exact_policy_grid(self):
        result = run_benchmark(
            1,
            ("uniform", "high_spill_biased"),
            ("uniform", "chain_biased"),
            ("uniform", "high_spill_biased", "chain_biased"),
            ("robust_minimax_one_step", "greedy_joint", "round_robin", "random_uniform"),
            ("joint",),
        )
        self.assertEqual(len(result["rows"]), 16)
        self.assertTrue(all(math.isfinite(r["expected_terminal_cross_entropy_bits"]) for r in result["rows"]))

    def test_horizon_validation_fails_closed(self):
        with self.assertRaises(ValueError):
            run_benchmark(4)

    def test_policy_validation_fails_closed(self):
        with self.assertRaises(ValueError):
            eval_policy("magic_policy", "joint", "uniform", "uniform")

    def test_embedded_self_test(self):
        self.assertTrue(self_test())


if __name__ == "__main__":
    unittest.main()
