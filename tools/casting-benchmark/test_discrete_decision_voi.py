#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

from discrete_decision_voi import (
    ObservationAction,
    evaluate_action,
    rank_actions,
    select_best_action,
)


PRIOR = (0.5, 0.5)
LOSSES = (
    (0.0, 10.0),  # choose decision 0: wrong in world 1
    (10.0, 0.0),  # choose decision 1: wrong in world 0
)


def action(action_id, cost, likelihoods):
    return ObservationAction(action_id, cost, likelihoods)


class DiscreteDecisionVOITests(unittest.TestCase):
    def test_known_answer_noisy_observation(self):
        # Prior risk is 5. A 90%-accurate observation reduces posterior risk to 1.
        score = evaluate_action(
            PRIOR,
            LOSSES,
            action("noisy", 2.0, ((0.9, 0.1), (0.1, 0.9))),
        )
        self.assertEqual(score.current_decision_index, 0)  # stable tie break
        self.assertAlmostEqual(score.current_bayes_risk, 5.0)
        self.assertAlmostEqual(score.expected_posterior_bayes_risk, 1.0)
        self.assertAlmostEqual(score.expected_value_of_information, 4.0)
        self.assertAlmostEqual(score.voi_per_cost, 2.0)
        self.assertEqual(score.posterior_decision_indices, (0, 1))

    def test_perfect_information_has_larger_absolute_voi(self):
        score = evaluate_action(
            PRIOR,
            LOSSES,
            action("perfect", 5.0, ((1.0, 0.0), (0.0, 1.0))),
        )
        self.assertAlmostEqual(score.expected_posterior_bayes_risk, 0.0)
        self.assertAlmostEqual(score.expected_value_of_information, 5.0)
        self.assertAlmostEqual(score.voi_per_cost, 1.0)

    def test_cost_normalization_prefers_noisy_query(self):
        actions = [
            action("perfect", 5.0, ((1.0, 0.0), (0.0, 1.0))),
            action("noisy", 2.0, ((0.9, 0.1), (0.1, 0.9))),
        ]
        result = select_best_action(
            PRIOR, LOSSES, actions, remaining_cost_units=5.0
        )
        self.assertEqual(result.action_id, "noisy")
        self.assertAlmostEqual(result.voi_per_cost, 2.0)

    def test_zero_information_action_is_not_ranked(self):
        result = rank_actions(
            PRIOR,
            LOSSES,
            [action("uninformative", 1.0, ((0.5, 0.5), (0.5, 0.5)))],
            remaining_cost_units=1.0,
        )
        self.assertEqual(result, [])

    def test_unaffordable_action_is_not_selected(self):
        result = select_best_action(
            PRIOR,
            LOSSES,
            [action("perfect", 5.0, ((1.0, 0.0), (0.0, 1.0)))],
            remaining_cost_units=4.0,
        )
        self.assertIsNone(result)

    def test_threshold_can_prefer_stop(self):
        result = select_best_action(
            PRIOR,
            LOSSES,
            [action("noisy", 2.0, ((0.9, 0.1), (0.1, 0.9)))],
            remaining_cost_units=5.0,
            minimum_voi_per_cost=2.1,
        )
        self.assertIsNone(result)

    def test_asymmetric_prior_changes_best_current_decision(self):
        score = evaluate_action(
            (0.9, 0.1),
            LOSSES,
            action("probe", 1.0, ((0.8, 0.2), (0.2, 0.8))),
        )
        self.assertEqual(score.current_decision_index, 0)
        self.assertAlmostEqual(score.current_bayes_risk, 1.0)

    def test_probabilities_and_cost_must_be_valid(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            evaluate_action(PRIOR, LOSSES, action("bad", 1.0, ((0.2, 0.2), (0.5, 0.5))))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            evaluate_action(PRIOR, LOSSES, action("free", 0.0, ((1, 0), (0, 1))))
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_action(PRIOR, LOSSES, action("nan", math.nan, ((1, 0), (0, 1))))

    def test_rejects_malformed_loss_tables(self):
        with self.assertRaisesRegex(ValueError, "one loss per world"):
            evaluate_action(PRIOR, ((0.0,),), action("probe", 1, ((1, 0), (0, 1))))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            evaluate_action(PRIOR, ((0, -1),), action("probe", 1, ((1, 0), (0, 1))))

    def test_rejects_duplicate_action_ids(self):
        same = action("duplicate", 1, ((1, 0), (0, 1)))
        with self.assertRaisesRegex(ValueError, "duplicate action_id"):
            rank_actions(PRIOR, LOSSES, [same, same], remaining_cost_units=10)

    def test_row_order_tie_break_is_deterministic(self):
        a = action("z-action", 2, ((0.9, 0.1), (0.1, 0.9)))
        b = action("a-action", 2, ((0.9, 0.1), (0.1, 0.9)))
        ranked = rank_actions(PRIOR, LOSSES, [a, b], remaining_cost_units=5)
        self.assertEqual([entry.action_id for entry in ranked], ["a-action", "z-action"])

    def test_posterior_risk_never_exceeds_current_risk(self):
        cases = [
            ((0.5, 0.5), ((1.0, 0.0), (0.0, 1.0))),
            ((0.8, 0.2), ((0.7, 0.3), (0.4, 0.6))),
            ((0.2, 0.8), ((0.55, 0.45), (0.45, 0.55))),
        ]
        for index, (prior, likelihoods) in enumerate(cases):
            with self.subTest(index=index):
                score = evaluate_action(
                    prior, LOSSES, action("action-" + str(index), 2, likelihoods)
                )
                self.assertLessEqual(
                    score.expected_posterior_bayes_risk,
                    score.current_bayes_risk + 1e-9,
                )


if __name__ == "__main__":
    unittest.main()
