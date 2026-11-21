import math
import unittest

from eig import (
    DesignValidationError,
    evaluate_design_scenarios,
    expected_information_gain_bits,
    theory_entropy_bits,
    weighted_lower_quantile,
)


class ExpectedInformationGainTests(unittest.TestCase):
    def test_identical_predictions_have_zero_information_gain(self):
        prior = {"IIT": 0.5, "GNWT": 0.5}
        likelihoods = {
            "IIT": {"yes": 0.7, "no": 0.2, "missing": 0.1},
            "GNWT": {"yes": 0.7, "no": 0.2, "missing": 0.1},
        }
        self.assertAlmostEqual(expected_information_gain_bits(prior, likelihoods), 0.0, places=14)

    def test_deterministic_opposing_predictions_reach_prior_entropy(self):
        prior = {"theory_A": 0.75, "theory_B": 0.25}
        likelihoods = {
            "theory_A": {"outcome_A": 1.0, "outcome_B": 0.0},
            "theory_B": {"outcome_A": 0.0, "outcome_B": 1.0},
        }
        eig = expected_information_gain_bits(prior, likelihoods)
        self.assertAlmostEqual(eig, theory_entropy_bits(prior), places=14)
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        self.assertAlmostEqual(eig, expected, places=14)

    def test_partial_discrimination_is_positive_but_below_entropy(self):
        prior = {"IIT": 0.5, "GNWT": 0.5}
        likelihoods = {
            "IIT": {"decode": 0.9, "not_decode": 0.1, "unusable": 0.0},
            "GNWT": {"decode": 0.2, "not_decode": 0.8, "unusable": 0.0},
        }
        eig = expected_information_gain_bits(prior, likelihoods)
        self.assertGreater(eig, 0.0)
        self.assertLess(eig, 1.0)
        self.assertAlmostEqual(eig, 0.3973126, places=6)

    def test_missingness_is_an_explicit_outcome_and_changes_eig(self):
        prior = {"A": 0.5, "B": 0.5}
        observed = {
            "A": {"positive": 0.9, "negative": 0.1, "missing": 0.0},
            "B": {"positive": 0.1, "negative": 0.9, "missing": 0.0},
        }
        with_missing = {
            "A": {"positive": 0.45, "negative": 0.05, "missing": 0.5},
            "B": {"positive": 0.05, "negative": 0.45, "missing": 0.5},
        }
        self.assertGreater(expected_information_gain_bits(prior, observed),
                           expected_information_gain_bits(prior, with_missing))

    def test_outcome_order_is_irrelevant(self):
        prior = {"A": 0.4, "B": 0.6}
        first = {
            "A": {"x": 0.8, "y": 0.2},
            "B": {"x": 0.3, "y": 0.7},
        }
        second = {
            "A": {"y": 0.2, "x": 0.8},
            "B": {"y": 0.7, "x": 0.3},
        }
        self.assertAlmostEqual(
            expected_information_gain_bits(prior, first),
            expected_information_gain_bits(prior, second),
            places=14,
        )

    def test_nuisance_scenario_summary_and_cost(self):
        prior = {"A": 0.5, "B": 0.5}
        discriminating = {
            "A": {"yes": 0.9, "no": 0.1},
            "B": {"yes": 0.1, "no": 0.9},
        }
        uninformative = {
            "A": {"yes": 0.5, "no": 0.5},
            "B": {"yes": 0.5, "no": 0.5},
        }
        result = evaluate_design_scenarios(
            prior,
            [(0.75, discriminating), (0.25, uninformative)],
            cost=2.0,
            lower_quantile=0.25,
        )
        self.assertEqual(result["scenario_count"], 2)
        self.assertGreater(result["expected_eig_bits"], 0.0)
        self.assertEqual(result["lower_quantile_eig_bits"], 0.0)
        self.assertAlmostEqual(result["expected_eig_bits_per_cost"], result["expected_eig_bits"] / 2.0)

    def test_weighted_lower_quantile(self):
        self.assertEqual(weighted_lower_quantile([(0.0, 0.1), (1.0, 0.8), (2.0, 0.1)], 0.1), 0.0)
        self.assertEqual(weighted_lower_quantile([(0.0, 0.1), (1.0, 0.8), (2.0, 0.1)], 0.5), 1.0)

    def test_invalid_mass_missing_theory_and_cost_fail_closed(self):
        with self.assertRaises(DesignValidationError):
            expected_information_gain_bits({"A": 0.7, "B": 0.4}, {"A": {"x": 1.0}, "B": {"x": 1.0}})
        with self.assertRaises(DesignValidationError):
            expected_information_gain_bits({"A": 0.5, "B": 0.5}, {"A": {"x": 1.0}})
        with self.assertRaises(DesignValidationError):
            evaluate_design_scenarios({"A": 1.0}, [(1.0, {"A": {"x": 1.0}})], cost=0.0)


if __name__ == "__main__":
    unittest.main()
