import unittest

from validation.noisy_scm_misspecification_audit import (
    POLICIES,
    evaluate,
    fixture,
    outcome_distribution,
    predictive_outcomes,
    run_audit,
    update_posterior,
)


class MisspecificationAuditTests(unittest.TestCase):
    def test_true_graph_is_outside_candidate_support(self):
        models, world, _ = fixture()
        self.assertFalse(any(m.graph_key() == world.graph_key() for m in models))

    def test_true_and_predictive_outcomes_normalize(self):
        models, world, actions = fixture()
        for action in actions:
            self.assertAlmostEqual(sum(outcome_distribution(world, action, 0.2).values()), 1.0, places=12)
            predictive = predictive_outcomes(models, (0.5, 0.5), action, 0.1)
            self.assertAlmostEqual(sum(predictive.values()), 1.0, places=12)
            for outcome in predictive:
                after = update_posterior(models, (0.5, 0.5), action, outcome, 0.1)
                self.assertAlmostEqual(sum(after), 1.0, places=12)

    def test_policies_are_deterministic_for_seed(self):
        models, world, actions = fixture()
        for policy in POLICIES:
            first = evaluate(policy, models, world, actions, 0.1, 0.2, 77)
            second = evaluate(policy, models, world, actions, 0.1, 0.2, 77)
            self.assertEqual(first, second)

    def test_audit_explicitly_records_scope_limitations(self):
        report = run_audit("test-sha")
        self.assertEqual(report["source_sha"], "test-sha")
        self.assertFalse(report["true_graph_in_candidate_support"])
        self.assertEqual(len(report["results"]), 3 * 2 * len(POLICIES))
        self.assertTrue(all(not row["true_graph_in_candidate_support"] for row in report["results"]))


if __name__ == "__main__":
    unittest.main()
