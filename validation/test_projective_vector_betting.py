import math
import unittest

from projective_vector_betting import (
    CoordinateHistory,
    assert_symmetric_psd,
    coordinate_local_bets,
    equicorrelation_covariance,
    equicorrelation_envelope,
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
