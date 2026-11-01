import math
import unittest

from projective_vector_betting import (
    CoordinateHistory,
    CovariancePreconditionedBet,
    ProjectiveEquicorrelationKernel,
    assert_symmetric_psd,
    coordinate_local_bets,
    equicorrelation_covariance,
    equicorrelation_envelope,
    projective_equicorrelation_envelope,
    projective_vector_log_factor,
    fixed_log_mixture,
    gaussian_log_factor_expectation,
    matrix_dominates,
    vector_log_factor,
)


class ProjectiveVectorBettingTests(unittest.TestCase):
    def test_global_envelope_is_projectively_consistent_across_dimensions(self):
        # Every d-dimensional envelope is a principal extension of every
        # smaller one, unlike the tighter fixed-dimension envelope.
        for rho_min, rho_max in ((0.2, 0.8), (0.1, 0.6), (0.0, 1.0)):
            matrices = {
                d: projective_equicorrelation_envelope(d, rho_min, rho_max)
                for d in (2, 3, 5, 8)
            }
            for small_d, large_d in ((2, 3), (2, 8), (3, 8), (5, 8)):
                self.assertEqual(
                    [row[:small_d] for row in matrices[large_d][:small_d]],
                    matrices[small_d],
                )
            for d, envelope in matrices.items():
                for rho in (rho_min, (rho_min + rho_max) / 2, rho_max):
                    self.assertTrue(
                        matrix_dominates(
                            envelope, equicorrelation_covariance(d, rho)
                        )
                    )

    def test_low_rank_kernel_matches_dense_quadratic_and_matvec(self):
        kernel = ProjectiveEquicorrelationKernel(7, 0.2, 0.8)
        dense = kernel.dense()
        values = [0.1, 0.2, 0.0, 0.7, 1.1, 0.0, 0.3]
        active = [True, True, False, True, True, False, True]
        masked = [value if on else 0.0 for value, on in zip(values, active)]
        expected = sum(
            masked[i] * dense[i][j] * masked[j]
            for i in range(7) for j in range(7)
        )
        self.assertAlmostEqual(kernel.quadratic(values, active), expected)
        expected_vec = [
            sum(dense[i][j] * masked[j] for j in range(7)) if active[i] else 0.0
            for i in range(7)
        ]
        self.assertEqual(kernel.matvec(values, active), expected_vec)

    def test_low_rank_factor_matches_dense_factor(self):
        kernel = ProjectiveEquicorrelationKernel(4, 0.2, 0.8)
        scores = [0.4, -0.1, 1.2, 0.7]
        bets = [0.3, 0.0, 0.5, 0.2]
        active = [True, True, True, False]
        dense_factor = vector_log_factor(scores, bets, active, kernel.dense())
        low_rank_factor = projective_vector_log_factor(scores, bets, active, kernel)
        self.assertAlmostEqual(low_rank_factor, dense_factor)

    def test_low_rank_and_dense_bet_states_remain_equivalent(self):
        kernel = ProjectiveEquicorrelationKernel(4, 0.2, 0.8)
        dense_state = CovariancePreconditionedBet(kernel.dense(), step_size=0.07, cap=1.2)
        low_rank_state = CovariancePreconditionedBet(kernel, step_size=0.07, cap=1.2)
        rounds = [
            ([True, False, False, False], [0.8, 99.0, 99.0, 99.0]),
            ([True, True, False, False], [0.4, -0.2, 99.0, 99.0]),
            ([True, True, True, False], [0.7, 0.1, 1.1, 99.0]),
            ([True, True, True, True], [-0.1, 0.3, 0.5, 0.9]),
            ([True, True, True, True], [0.2, -0.4, 0.3, 0.1]),
        ]
        for active, scores in rounds:
            dense_bets = dense_state.bets(active)
            low_rank_bets = low_rank_state.bets(active)
            for x, y in zip(dense_bets, low_rank_bets):
                self.assertAlmostEqual(x, y)
            dense_state.observe(scores)
            low_rank_state.observe(scores)
            for x, y in zip(dense_state.snapshot()[0], low_rank_state.snapshot()[0]):
                self.assertAlmostEqual(x, y)

    def test_low_rank_kernel_principal_extension_is_identical(self):
        small = ProjectiveEquicorrelationKernel(3, 0.2, 0.8)
        large = ProjectiveEquicorrelationKernel(9, 0.2, 0.8)
        self.assertEqual(
            [row[:3] for row in large.dense()[:3]],
            small.dense(),
        )

    def test_projective_envelope_rejects_invalid_interval(self):
        for interval in ((-0.1, 0.8), (0.8, 0.2), (0.2, 1.1)):
            with self.assertRaises(ValueError):
                projective_equicorrelation_envelope(4, *interval)

    def test_equicorrelation_envelope_dominates_grid(self):
        d = 5
        envelope = equicorrelation_envelope(d, 0.2, 0.8, 0.5)
        self.assertAlmostEqual(envelope[0][0], 2.2)
        self.assertTrue(assert_symmetric_psd(envelope) is None)
        for rho_step in range(21):
            rho = 0.2 + 0.03 * rho_step
            self.assertTrue(matrix_dominates(envelope, equicorrelation_covariance(d, rho)))

    def test_gaussian_log_expectation_nonpositive_under_null(self):
        covariance = equicorrelation_covariance(3, 0.5)
        envelope = equicorrelation_envelope(3, 0.2, 0.8, 0.5)
        mean = [-0.1, 0.0, -0.2]
        bets = [0.3, 0.6, 0.2]
        active = [True, True, True]
        self.assertLessEqual(
            gaussian_log_factor_expectation(mean, covariance, bets, active, envelope),
            1e-10,
        )

    def test_bets_use_only_coordinate_local_past(self):
        h1 = [[0.1, 0.2, 0.4], [-1.0, 0.0, 0.1]]
        h2 = h1 + [[100.0, 100.0, 100.0]]
        self.assertEqual(coordinate_local_bets(h1), coordinate_local_bets(h2)[:2])

    def test_history_reads_before_current_observation(self):
        state = CoordinateHistory(2)
        self.assertEqual(state.bets(), [0.0, 0.0])
        state.observe([2.0, -1.0], [True, True])
        after = state.bets()
        self.assertEqual(after, coordinate_local_bets([[2.0], [-1.0]]))
        self.assertEqual(state.snapshot(), ((2.0,), (-1.0,)))

    def test_projective_active_subset_and_dormant_padding(self):
        # V is declared globally for the 3-coordinate universe. The old
        # two-coordinate process uses its unchanged 2x2 principal submatrix.
        envelope = equicorrelation_envelope(3, 0.2, 0.8, 0.5)
        z = [0.7, -0.2, 999.0]
        lam = [0.4, 0.3, 100.0]
        first_two_envelope = [row[:2] for row in envelope[:2]]
        old_path = vector_log_factor(
            z[:2], lam[:2], [True, True], first_two_envelope
        )
        extended_path = vector_log_factor(
            z, lam, [True, True, False], envelope
        )
        self.assertAlmostEqual(old_path, extended_path)
        self.assertEqual(
            vector_log_factor(z, lam, [False, False, False], envelope), 0.0
        )

    def test_rejects_covariance_proxy_that_does_not_dominate(self):
        covariance = equicorrelation_covariance(2, 0.5)
        underpowered_proxy = [[1.0, 0.0], [0.0, 1.0]]
        with self.assertRaises(ValueError):
            gaussian_log_factor_expectation(
                [0.0, 0.0], covariance, [0.4, 0.3],
                [True, True], underpowered_proxy
            )

    def test_fixed_mixture_is_not_pointwise_max(self):
        value = fixed_log_mixture([math.log(8.0), math.log(2.0)], [0.5, 0.5])
        self.assertAlmostEqual(math.exp(value), 5.0)
        self.assertLess(value, math.log(8.0))

    def test_accepts_rank_deficient_psd_covariance_at_multiple_scales(self):
        # Outer products are PSD but can be rank-deficient and badly scaled.
        for scale in (1e-9, 1.0, 1e9):
            matrix = [
                [scale, 2.0 * scale, 0.0],
                [2.0 * scale, 4.0 * scale, 0.0],
                [0.0, 0.0, 0.0],
            ]
            self.assertIsNone(assert_symmetric_psd(matrix))

    def test_preconditioned_bet_is_preupdate_and_bounded(self):
        envelope = equicorrelation_envelope(2, 0.2, 0.8, 0.5)
        state = CovariancePreconditionedBet(envelope, step_size=0.05, cap=0.7)
        prior = state.bets([True, True])
        self.assertEqual(prior, [0.0, 0.0])
        with self.assertRaises(RuntimeError):
            state.bets([True, True])
        state.observe([50.0, 50.0])
        next_bets = state.bets([True, True])
        self.assertTrue(all(0.0 <= value <= 0.7 for value in next_bets))
        self.assertTrue(any(value > 0.0 for value in next_bets))
        state.observe([-50.0, -50.0])
        bounded, _ = state.snapshot()
        self.assertTrue(all(0.0 <= value <= 0.7 for value in bounded))

    def test_preconditioned_bet_rejects_same_round_and_retraction(self):
        envelope = equicorrelation_envelope(2, 0.2, 0.8, 0.5)
        state = CovariancePreconditionedBet(envelope)
        with self.assertRaises(RuntimeError):
            state.observe([1.0, 1.0])
        state.bets([True, True])
        state.observe([1.0, 1.0])
        with self.assertRaises(ValueError):
            state.bets([True, False])

    def test_projective_covariance_envelope_is_dimension_consistent(self):
        for d in range(2, 21):
            envelope = projective_equicorrelation_envelope(d, 0.2, 0.8)
            for rho_step in range(21):
                rho = 0.2 + 0.03 * rho_step
                self.assertTrue(
                    matrix_dominates(envelope, equicorrelation_covariance(d, rho))
                )
            if d > 2:
                smaller = projective_equicorrelation_envelope(d - 1, 0.2, 0.8)
                self.assertEqual(
                    [row[: d - 1] for row in envelope[: d - 1]],
                    smaller,
                )

    def test_preconditioned_bet_ignores_dormant_extension(self):
        envelope3 = projective_equicorrelation_envelope(3, 0.2, 0.8)
        envelope2 = projective_equicorrelation_envelope(2, 0.2, 0.8)
        old = CovariancePreconditionedBet(envelope2, step_size=0.05)
        extended = CovariancePreconditionedBet(envelope3, step_size=0.05)
        for score in ([1.0, -0.5], [0.8, 0.2], [-0.1, 1.2], [1.1, 0.5]):
            old_bets = old.bets([True, True])
            extended_bets = extended.bets([True, True, False])
            self.assertEqual(old_bets, extended_bets[:2])
            old.observe(list(score))
            extended.observe([score[0], score[1], 999.0])
            self.assertEqual(
                old.snapshot()[0],
                extended.snapshot()[0][:2],
            )

    def test_preconditioned_bet_new_coordinate_starts_dormant(self):
        envelope = projective_equicorrelation_envelope(2, 0.2, 0.8)
        state = CovariancePreconditionedBet(envelope, step_size=0.05)
        state.bets([True, False])
        state.observe([4.0, 123.0])
        before_activation = state.bets([True, True])
        self.assertGreater(before_activation[0], 0.0)
        self.assertEqual(before_activation[1], 0.0)
        state.observe([2.0, 4.0])
        after_activation = state.bets([True, True])
        self.assertGreater(after_activation[1], 0.0)

    def test_rejects_scaled_indefinite_covariance(self):
        for scale in (1e-9, 1.0, 1e9):
            matrix = [
                [scale, 2.0 * scale],
                [2.0 * scale, scale],
            ]
            with self.assertRaises(ValueError):
                assert_symmetric_psd(matrix)

    def test_rejects_invalid_bets_and_bad_covariance(self):
        with self.assertRaises(ValueError):
            vector_log_factor([1.0], [-0.1], [True], [[1.0]])
        with self.assertRaises(ValueError):
            vector_log_factor([1.0], [0.1], [True], [[-1.0]])
        with self.assertRaises(ValueError):
            vector_log_factor([1.0, 2.0], [0.1], [True, True], [[1.0]])

    def test_same_round_score_cannot_enter_prior_bet_state(self):
        state = CoordinateHistory(1)
        prior_bet = state.bets()
        current = [50.0]
        self.assertEqual(prior_bet, [0.0])
        state.observe(current, [True])
        self.assertEqual(prior_bet, [0.0])
        self.assertGreater(state.bets()[0], 0.0)


if __name__ == "__main__":
    unittest.main()
