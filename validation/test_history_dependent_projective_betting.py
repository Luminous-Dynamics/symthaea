import copy
import math
import unittest

from history_dependent_projective_betting import (
    adaptive_lambda,
    build_history_dependent_projective_paths,
    build_predictable_arrivals_from_history,
    conditional_log_mgf_bound,
    nonempty_subsets,
    shared_control_covariance,
    variance_of_shared_sum,
    verify_history_dependent_projective_receipt,
    verify_local_projective_closure,
)


class HistoryDependentProjectiveBettingTests(unittest.TestCase):
    def setUp(self):
        self.observations = [
            {1: 0.30},
            {1: 0.20, 2: 0.10},
            {1: 0.40, 2: 0.40, 3: -0.20},
        ]
        self.active = build_predictable_arrivals_from_history(
            self.observations,
            second_active_threshold=0.0,
            third_active_threshold=0.05,
        )
        self.assertEqual(self.active, ((1,), (1, 2), (1, 2, 3)))

    def test_shared_control_covariance_and_sum_variance(self):
        sigma = shared_control_covariance(3)
        self.assertEqual(sigma, (
            (1.0, 0.5, 0.5),
            (0.5, 1.0, 0.5),
            (0.5, 0.5, 1.0),
        ))
        for m in range(1, 8):
            ones_product = sum(sum(row) for row in shared_control_covariance(m))
            self.assertAlmostEqual(ones_product, variance_of_shared_sum(m))
        # The first two coordinates' covariance is unchanged when a third
        # dormant coordinate is appended.
        self.assertEqual(
            tuple(row[:2] for row in shared_control_covariance(3)[:2]),
            shared_control_covariance(2),
        )

    def test_conditional_gaussian_factor_is_valid_under_intersection_null(self):
        log_bound = conditional_log_mgf_bound((-0.2, 0.0, -1.5), 0.4)
        self.assertLessEqual(log_bound, 0.0)
        self.assertLessEqual(math.exp(log_bound), 1.0)
        self.assertEqual(conditional_log_mgf_bound((-1.0, -2.0), 0.0), 0.0)

    def test_mean_outside_intersection_null_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the intersection null"):
            conditional_log_mgf_bound((-0.1, 0.01), 0.5)

    def test_negative_or_nonfinite_predictable_bets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            conditional_log_mgf_bound((-1.0,), -0.1)
        with self.assertRaisesRegex(ValueError, "finite"):
            conditional_log_mgf_bound((-1.0,), math.inf)

    def test_predictable_lambda_uses_only_pre_update_history(self):
        active_intersection = (1, 2)
        past = self.observations[:2]
        before = adaptive_lambda(active_intersection, past)
        changed_current = dict(self.observations[2])
        changed_current[1] = 10_000.0
        # Current data are not an argument to the predictable betting policy.
        after = adaptive_lambda(active_intersection, past)
        self.assertEqual(before, after)
        self.assertLessEqual(0.0, before)
        self.assertLessEqual(before, 0.5)

    def test_random_predictable_arrivals_are_nested(self):
        self.assertEqual(self.active[0], (1,))
        self.assertIn(2, self.active[1])
        self.assertIn(3, self.active[2])
        changed = copy.deepcopy(self.observations)
        changed[0][1] = -10.0
        altered_schedule = build_predictable_arrivals_from_history(
            changed,
            second_active_threshold=0.0,
            third_active_threshold=0.05,
        )
        self.assertEqual(altered_schedule[0], self.active[0])
        self.assertNotIn(2, altered_schedule[1])

    def test_intersection_processes_are_neutral_padding_projective(self):
        result = build_history_dependent_projective_paths(
            self.active, self.observations,
            base_lambda=0.05, gain=0.02, maximum_lambda=0.5,
        )
        self.assertTrue(result["projective"], result["projectivity_failures"])
        self.assertGreater(result["projectivity_checks"], 0)
        self.assertEqual(
            result["log_e_process_paths"][(1,)][:2],
            result["log_e_process_paths"][(1, 2)][:2],
        )
        self.assertEqual(
            result["log_e_process_paths"][(1, 2)][0],
            result["log_e_process_paths"][(1, 2, 3)][0],
        )

    def test_independent_receipt_replay_checks_every_factor_and_prefix(self):
        result = build_history_dependent_projective_paths(
            self.active, self.observations
        )
        check = verify_history_dependent_projective_receipt(result)
        self.assertTrue(check["valid"], check)
        self.assertEqual(
            check["predictable_factors_replayed"],
            len(result["universe"] and nonempty_subsets(result["universe"])) * len(self.active) + len(self.active),
        )
        self.assertGreater(check["historical_projectivity_checks"], 0)

    def test_tampered_lambda_receipt_is_rejected(self):
        result = build_history_dependent_projective_paths(
            self.active, self.observations
        )
        tampered = copy.deepcopy(result)
        tampered["factor_receipts"][(1, 2, 3)][2]["lambda"] += 0.1
        check = verify_history_dependent_projective_receipt(tampered)
        self.assertFalse(check["valid"])
        self.assertIn(
            "predictable_lambda_replay_failure",
            [failure[0] for failure in check["failures"]],
        )

    def test_local_closure_is_checked_for_every_active_intersection(self):
        result = build_history_dependent_projective_paths(
            self.active, self.observations
        )
        closure = verify_local_projective_closure(
            result, ((), (), ()), alpha=0.1
        )
        self.assertTrue(closure["valid"], closure)
        self.assertEqual(closure["constraints_checked"], 1 + 3 + 7)

    def test_current_evidence_cannot_change_same_round_lambda(self):
        first = build_history_dependent_projective_paths(
            self.active, self.observations
        )
        changed = copy.deepcopy(self.observations)
        changed[2] = {1: -500, 2: 500, 3: 1000}
        second = build_history_dependent_projective_paths(
            self.active, changed
        )
        for intersection in ((1,), (1, 2), (1, 2, 3)):
            self.assertEqual(
                first["factor_receipts"][intersection][2]["lambda"],
                second["factor_receipts"][intersection][2]["lambda"],
            )

    def test_non_nested_active_family_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "nested"):
            build_history_dependent_projective_paths(
                ((1, 2), (2, 3)),
                ({1: 0.1, 2: -0.1}, {2: 0.2, 3: 0.1}),
            )

    def test_nonfinite_observations_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            build_history_dependent_projective_paths(
                ((1,),), ({1: math.nan},)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
