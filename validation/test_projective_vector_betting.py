import math
import unittest

from projective_vector_betting import (
    CoordinateHistory,
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
    def test_equicorrelation_envelope_dominates_grid(self):
        d = 5
        envelope = equicorrelation_envelope(d, 0.2, 0.8, 0.5)
        self.assertAlmostEqual(envelope[0][0], 2.2)
        self.assertTrue(assert_symmetric_psd(envelope) is None)
        for rho_step in range(21):
            rho = 0.2 + 0.03 * rho_step
            self.assertTrue(matrix_dominates(envelope, equicorrelation_covariance(d, rho)))

    def test_global_envelope_is_projectively_consistent_across_dimensions(self):
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
                        matrix_dominates(envelope, equicorrelation_covariance(d, rho))
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
        for actual, expected_value in zip(kernel.matvec(values, active), expected_vec):
            self.assertAlmostEqual(actual, expected_value)

    def test_low_rank_operator_randomized_against_dense_reference(self):
        rng = __import__("random").Random(69851)
        for dimension in range(1, 26):
            kernel = ProjectiveEquicorrelationKernel(dimension, 0.2, 0.8)
            dense = kernel.dense()
            values = [rng.uniform(0.0, 1.5) for _ in range(dimension)]
            active = [rng.random() < 0.7 for _ in range(dimension)]
            masked = [values[i] if active[i] else 0.0 for i in range(dimension)]
            dense_q = sum(
                masked[i] * dense[i][j] * masked[j]
                for i in range(dimension) for j in range(dimension)
            )
            self.assertAlmostEqual(kernel.quadratic(values, active), dense_q, places=9)
            dense_v = [
                sum(dense[i][j] * masked[j] for j in range(dimension))
                if active[i] else 0.0
                for i in range(dimension)
            ]
            for actual, expected in zip(kernel.matvec(values, active), dense_v):
                self.assertAlmostEqual(actual, expected, places=9)

    def test_low_rank_factor_matches_dense_factor_and_prefix(self):
        small = ProjectiveEquicorrelationKernel(3, 0.2, 0.8)
        large = ProjectiveEquicorrelationKernel(8, 0.2, 0.8)
        scores = [0.4, -0.1, 1.2]
        bets = [0.3, 0.0, 0.5]
        active = [True, True, True]
        dense_factor = vector_log_factor(scores, bets, active, small.dense())
        low_rank_factor = projective_vector_log_factor(scores, bets, active, small)
        self.assertAlmostEqual(low_rank_factor, dense_factor)
        extended_scores = scores + [999.0] * 5
        extended_bets = bets + [100.0] * 5
        extended_active = [True, True, True] + [False] * 5
        extended_factor = projective_vector_log_factor(
            extended_scores, extended_bets, extended_active, large
        )
        self.assertAlmostEqual(extended_factor, low_rank_factor)

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
