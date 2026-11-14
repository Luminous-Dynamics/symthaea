import json
import pathlib
import unittest
from causal_reflex_benchmark import Arm, from_dict, simulate, compare

ROOT = pathlib.Path(__file__).resolve().parent
SCENARIOS = [from_dict(x) for x in json.loads((ROOT / "scenarios.json").read_text())["scenarios"]]
BY_ID = {s.scenario_id: s for s in SCENARIOS}


class CausalReflexTests(unittest.TestCase):
    def test_identical_exogenous_scenario_exposes_anticipatory_advantage(self):
        s = BY_ID["fast_slip_early_precursor"]
        self.assertFalse(simulate(s, Arm.REACTIVE).safe)
        self.assertTrue(simulate(s, Arm.PREDICTIVE).safe)

    def test_adversarial_sham_can_beat_true_signal_and_must_remain_visible(self):
        s = BY_ID["sham_signal_early_but_true_signal_late"]
        self.assertFalse(simulate(s, Arm.PREDICTIVE).safe)
        self.assertTrue(simulate(s, Arm.SHAM).safe)

    def test_false_positive_is_charged_as_unnecessary_intervention(self):
        r = simulate(BY_ID["false_precursor_no_hazard"], Arm.PREDICTIVE)
        self.assertTrue(r.safe)
        self.assertTrue(r.false_intervention)
        self.assertEqual(r.intervention_count, 1)

    def test_unqualified_or_mismatched_evidence_blocks_prediction(self):
        for key in ("unqualified_sensor_signal", "wrong_task_context", "compute_cost_exceeds_budget"):
            with self.subTest(key=key):
                self.assertIsNone(simulate(BY_ID[key], Arm.PREDICTIVE).trigger_tick)

    def test_report_keeps_oracle_separate_and_reports_paired_effects(self):
        report = compare(SCENARIOS)
        self.assertTrue(report["oracle_is_evaluator_only"])
        self.assertEqual(report["independent_unit"], "scenario_episode")
        self.assertIn("net_paired_safe_lift", report["predictive_vs_reactive"])

    def test_scenario_rejects_boolean_as_integer(self):
        raw = json.loads((ROOT / "scenarios.json").read_text())["scenarios"][0].copy()
        raw["horizon_ticks"] = True
        with self.assertRaises(ValueError):
            from_dict(raw)


if __name__ == "__main__":
    unittest.main()
