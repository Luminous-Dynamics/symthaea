import json
import pathlib
import unittest
from intervention_censoring import (
    T0Disposition, from_dict, classify_untreated_event_time,
    censor_method_admissible, intention_to_treat_summary, cohort_identifiability,
)

ROOT = pathlib.Path(__file__).resolve().parent
RAW = json.loads((ROOT / "fixtures.json").read_text())
EPISODES = [from_dict(x) for x in RAW["episodes"]]
TRIALS = [from_dict(x) for x in RAW["randomized_policy_trials"]]
BY_ID = {e.episode_id: e for e in EPISODES}


class InterventionCensoringTests(unittest.TestCase):
    def test_preventive_intervention_is_not_ordinary_right_censor(self):
        self.assertEqual(classify_untreated_event_time(BY_ID["intervention_prevents_boundary"]), T0Disposition.INTERVENTION_COUNTERFACTUAL_UNOBSERVED)
        self.assertFalse(censor_method_admissible(BY_ID["intervention_prevents_boundary"])[0])

    def test_administrative_censor_requires_separate_model_and_positivity_receipt(self):
        good = BY_ID["administrative_censor_supported"]
        bad = BY_ID["administrative_censor_no_model"]
        positivity = BY_ID["positive_t0_censor_model_but_positivity_failure"]
        self.assertEqual(classify_untreated_event_time(good), T0Disposition.ADMIN_RIGHT_CENSOR)
        self.assertTrue(censor_method_admissible(good)[0])
        self.assertFalse(censor_method_admissible(bad)[0])
        self.assertFalse(censor_method_admissible(positivity)[0])

    def test_post_intervention_event_is_policy_event_not_t0(self):
        self.assertEqual(classify_untreated_event_time(BY_ID["boundary_after_intervention"]), T0Disposition.POST_INTERVENTION_EVENT_ONLY)

    def test_competing_event_and_telemetry_loss_are_distinct(self):
        self.assertEqual(classify_untreated_event_time(BY_ID["competing_motor_fault"]), T0Disposition.COMPETING_TERMINAL_EVENT)
        self.assertEqual(classify_untreated_event_time(BY_ID["telemetry_loss"]), T0Disposition.OBSERVATION_LOSS)

    def test_unknown_clock_or_delivery_fails_closed(self):
        self.assertEqual(classify_untreated_event_time(BY_ID["unknown_clock"]), T0Disposition.TIME_EVIDENCE_INVALID)
        self.assertEqual(classify_untreated_event_time(BY_ID["unknown_actuation_delivery"]), T0Disposition.DELIVERY_UNKNOWN)

    def test_same_tick_event_intervention_is_ambiguous(self):
        self.assertEqual(classify_untreated_event_time(BY_ID["event_action_same_tick"]), T0Disposition.ORDER_AMBIGUOUS)

    def test_event_before_later_intervention_remains_observed(self):
        self.assertEqual(classify_untreated_event_time(BY_ID["event_before_later_intervention"]), T0Disposition.OBSERVED_BOUNDARY_EVENT)

    def test_itt_summary_uses_assignment_and_fixed_horizon_outcomes(self):
        result = intention_to_treat_summary(TRIALS, "reactive_safe", "predictive_safe")
        self.assertTrue(result["estimable"])
        self.assertEqual(result["control"]["adverse_n"], 2)
        self.assertEqual(result["candidate"]["adverse_n"], 1)
        self.assertAlmostEqual(result["candidate_minus_control_risk_difference"], -0.25)
        self.assertFalse(result["individual_counterfactuals_identified"])

    def test_observational_as_treated_comparison_is_not_itt(self):
        observational = [from_dict({**x, "assignment_design": "observational"}) for x in RAW["randomized_policy_trials"]]
        result = intention_to_treat_summary(observational, "reactive_safe", "predictive_safe")
        self.assertFalse(result["estimable"])
        self.assertEqual(result["reason"], "assignment_not_randomized")

    def test_assignment_must_precede_time_zero_and_have_receipt(self):
        late = [from_dict({**x, "assignment_precedes_time_zero": False}) for x in RAW["randomized_policy_trials"]]
        result = intention_to_treat_summary(late, "reactive_safe", "predictive_safe")
        self.assertFalse(result["estimable"])
        self.assertEqual(result["reason"], "assignment_not_pre_episode")
        missing_receipt = [from_dict({**x, "randomization_receipt_valid": False}) for x in RAW["randomized_policy_trials"]]
        result = intention_to_treat_summary(missing_receipt, "reactive_safe", "predictive_safe")
        self.assertFalse(result["estimable"])
        self.assertEqual(result["reason"], "randomization_receipt_invalid_or_missing")

    def test_blocked_randomization_requires_both_arms_in_each_block(self):
        incomplete = [from_dict({**x, "assignment_block_id": "single-block" if x["assigned_policy"] == "reactive_safe" else "other-block"}) for x in RAW["randomized_policy_trials"]]
        result = intention_to_treat_summary(incomplete, "reactive_safe", "predictive_safe")
        self.assertFalse(result["estimable"])
        self.assertEqual(result["reason"], "blocked_randomization_missing_arm_within_block")

    def test_incomplete_followup_blocks_naive_fixed_horizon_itt(self):
        incomplete = list(TRIALS)
        incomplete[0] = from_dict({**RAW["randomized_policy_trials"][0], "policy_followup_complete": False, "policy_endpoint": None})
        result = intention_to_treat_summary(incomplete, "reactive_safe", "predictive_safe")
        self.assertFalse(result["estimable"])
        self.assertIn("incomplete_fixed_horizon", result["reason"])

    def test_all_treated_cohort_does_not_identify_untreated_event_time(self):
        group = [BY_ID["intervention_prevents_boundary"], BY_ID["boundary_after_intervention"]]
        result = cohort_identifiability(group)
        self.assertEqual(result["t0_identifiability"], "not_identifiable_from_this_cohort_alone")

    def test_json_boolean_is_not_accepted_as_integer(self):
        raw = RAW["episodes"][0].copy()
        raw["followup_end"] = True
        with self.assertRaises(ValueError):
            from_dict(raw)


if __name__ == "__main__":
    unittest.main()
