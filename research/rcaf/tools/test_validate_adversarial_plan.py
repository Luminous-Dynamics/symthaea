"""Tests for the proposed RCAF adversarial calibration-plan contract."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from validate_adversarial_plan import PlanValidationError, validate_plan

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "adversarial-calibration-plan.schema.json").read_text(encoding="utf-8"))
FIXTURE = json.loads((ROOT / "adversarial-calibration-plan.example.json").read_text(encoding="utf-8"))


class AdversarialPlanTests(unittest.TestCase):
    def test_proposed_fixture_passes(self):
        validate_plan(FIXTURE, SCHEMA)

    def test_development_and_confirmatory_seeds_must_differ(self):
        plan = deepcopy(FIXTURE)
        plan["seed_policy"]["confirmatory_seed"] = plan["seed_policy"]["development_seed"]
        with self.assertRaisesRegex(PlanValidationError, "seeds must differ"):
            validate_plan(plan, SCHEMA)

    def test_critical_null_replicates_cannot_be_lower_than_qualification(self):
        plan = deepcopy(FIXTURE)
        plan["seed_policy"]["critical_null_replicates"] = 999
        with self.assertRaisesRegex(PlanValidationError, "critical_null_replicates"):
            validate_plan(plan, SCHEMA)

    def test_every_null_scenario_must_be_no_effect(self):
        plan = deepcopy(FIXTURE)
        plan["latent_scenarios"][0]["mechanism"] = "localized_module"
        with self.assertRaisesRegex(PlanValidationError, "must not inject"):
            validate_plan(plan, SCHEMA)

    def test_at_least_one_signal_level_injection_is_required(self):
        plan = deepcopy(FIXTURE)
        plan["latent_scenarios"] = [
            item for item in plan["latent_scenarios"]
            if item["injection_stage"] != "parcel_time_series"
        ]
        with self.assertRaisesRegex(PlanValidationError, "signal-level injection"):
            validate_plan(plan, SCHEMA)

    def test_matrix_level_control_is_required(self):
        plan = deepcopy(FIXTURE)
        plan["latent_scenarios"] = [
            item for item in plan["latent_scenarios"]
            if item["injection_stage"] != "connectivity_matrix"
        ]
        with self.assertRaisesRegex(PlanValidationError, "matrix-level control"):
            validate_plan(plan, SCHEMA)

    def test_positive_effect_dose_is_required(self):
        plan = deepcopy(FIXTURE)
        plan["latent_scenarios"][1]["dose_points"] = [0.0]
        with self.assertRaisesRegex(PlanValidationError, "positive dose"):
            validate_plan(plan, SCHEMA)

    def test_nuisance_null_must_preserve_latent_target(self):
        plan = deepcopy(FIXTURE)
        plan["nuisance_scenarios"][0]["preserves_latent_target"] = False
        with self.assertRaises(PlanValidationError):
            validate_plan(plan, SCHEMA)

    def test_scenario_identifiers_must_be_unique(self):
        plan = deepcopy(FIXTURE)
        plan["latent_scenarios"][1]["scenario_id"] = plan["latent_scenarios"][0]["scenario_id"]
        with self.assertRaisesRegex(PlanValidationError, "must be unique"):
            validate_plan(plan, SCHEMA)

    def test_directional_detection_and_wrong_direction_outputs_are_mandatory(self):
        for required in ("directional_power", "wrong_direction_rejection_rate"):
            plan = deepcopy(FIXTURE)
            plan["required_outputs"].remove(required)
            with self.subTest(required=required):
                with self.assertRaisesRegex(PlanValidationError, required):
                    validate_plan(plan, SCHEMA)

    def test_common_mode_nuisance_requires_zero_and_positive_doses(self):
        plan = deepcopy(FIXTURE)
        plan["nuisance_scenarios"][0]["dose_points"] = [0.0]
        with self.assertRaisesRegex(PlanValidationError, "zero and positive nuisance doses"):
            validate_plan(plan, SCHEMA)

    def test_nuisance_only_rejection_output_is_mandatory(self):
        plan = deepcopy(FIXTURE)
        plan["required_outputs"].remove("nuisance_only_rejection_rate")
        with self.assertRaisesRegex(PlanValidationError, "nuisance_only_rejection_rate"):
            validate_plan(plan, SCHEMA)

    def test_mandatory_outputs_are_present(self):
        plan = deepcopy(FIXTURE)
        plan["required_outputs"].remove("nuisance_false_positives")
        with self.assertRaisesRegex(PlanValidationError, "nuisance_false_positives"):
            validate_plan(plan, SCHEMA)


if __name__ == "__main__":
    unittest.main()
