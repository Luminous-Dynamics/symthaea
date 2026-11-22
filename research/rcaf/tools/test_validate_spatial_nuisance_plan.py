"""Tests for the proposed RCAF spatial-nuisance and identifiability plan."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from validate_spatial_nuisance_plan import (
    SpatialPlanValidationError,
    load_json,
    validate_plan,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (ROOT / "spatial-nuisance-plan.schema.json").read_text(encoding="utf-8")
)
FIXTURE = json.loads(
    (ROOT / "spatial-nuisance-plan.example.json").read_text(encoding="utf-8")
)


class SpatialNuisancePlanTests(unittest.TestCase):
    def test_proposed_fixture_passes(self):
        validate_plan(FIXTURE, SCHEMA)

    def test_development_and_confirmatory_seeds_must_differ(self):
        plan = deepcopy(FIXTURE)
        plan["seed_policy"]["confirmatory_seed"] = plan["seed_policy"]["development_seed"]
        with self.assertRaisesRegex(SpatialPlanValidationError, "seeds must differ"):
            validate_plan(plan, SCHEMA)

    def test_all_four_counterfactual_cells_are_required(self):
        plan = deepcopy(FIXTURE)
        plan["counterfactual_cells"].remove("nuisance_only")
        with self.assertRaises(SpatialPlanValidationError):
            validate_plan(plan, SCHEMA)

    def test_target_rois_must_exist_in_geometry(self):
        plan = deepcopy(FIXTURE)
        plan["target"]["roi_labels"].append("roi-not-in-graph")
        with self.assertRaisesRegex(SpatialPlanValidationError, "subset"):
            validate_plan(plan, SCHEMA)

    def test_synthetic_geometry_must_not_claim_anatomical_scope(self):
        plan = deepcopy(FIXTURE)
        plan["geometry"]["interpretation_scope"] = "atlas_geometry"
        with self.assertRaises(SpatialPlanValidationError):
            validate_plan(plan, SCHEMA)

    def test_synthetic_geometry_claim_limit_must_say_not_anatomical(self):
        plan = deepcopy(FIXTURE)
        plan["claim_limits"] = [
            item for item in plan["claim_limits"] if "not anatomical" not in item
        ]
        with self.assertRaisesRegex(SpatialPlanValidationError, "not anatomical"):
            validate_plan(plan, SCHEMA)

    def test_nuisance_dose_grid_must_start_at_zero_and_include_positive_dose(self):
        plan = deepcopy(FIXTURE)
        plan["dose_grid"]["nuisance_variance_fractions"] = [0.025, 0.05, 0.1]
        with self.assertRaisesRegex(SpatialPlanValidationError, "exact zero-dose"):
            validate_plan(plan, SCHEMA)

    def test_nuisance_dose_grid_must_be_strictly_increasing(self):
        plan = deepcopy(FIXTURE)
        plan["dose_grid"]["nuisance_variance_fractions"] = [0.0, 0.1, 0.05]
        with self.assertRaisesRegex(SpatialPlanValidationError, "strictly increasing"):
            validate_plan(plan, SCHEMA)

    def test_spatial_alignment_grid_must_include_orthogonal_and_aligned_cases(self):
        plan = deepcopy(FIXTURE)
        plan["dose_grid"]["spatial_alignment"] = [0.0, 0.25, 0.5, 0.75]
        with self.assertRaisesRegex(SpatialPlanValidationError, "alignment grid"):
            validate_plan(plan, SCHEMA)

    def test_temporal_coherence_grid_must_include_independent_and_perfect_cases(self):
        plan = deepcopy(FIXTURE)
        plan["dose_grid"]["temporal_coherence"] = [0.0, 0.25, 0.5, 0.75]
        with self.assertRaisesRegex(SpatialPlanValidationError, "coherence grid"):
            validate_plan(plan, SCHEMA)

    def test_response_fraction_grid_must_include_zero_and_full_response(self):
        plan = deepcopy(FIXTURE)
        plan["dose_grid"]["participant_response_fractions"] = [0.25, 0.5, 0.75]
        with self.assertRaisesRegex(SpatialPlanValidationError, "response grid"):
            validate_plan(plan, SCHEMA)

    def test_both_spatial_families_are_required(self):
        plan = deepcopy(FIXTURE)
        plan["nuisance_families"] = [
            family
            for family in plan["nuisance_families"]
            if family["kind"] != "heat_kernel_localized"
        ]
        with self.assertRaises(SpatialPlanValidationError):
            validate_plan(plan, SCHEMA)

    def test_nuisance_family_identifiers_must_be_unique(self):
        plan = deepcopy(FIXTURE)
        plan["nuisance_families"][1]["family_id"] = plan["nuisance_families"][0]["family_id"]
        with self.assertRaisesRegex(SpatialPlanValidationError, "identifiers must be unique"):
            validate_plan(plan, SCHEMA)

    def test_identifiability_boundary_must_be_present_in_declared_grid(self):
        plan = deepcopy(FIXTURE)
        plan["identifiability_boundary"]["spatial_alignment"] = 0.75
        with self.assertRaisesRegex(SpatialPlanValidationError, "observational-equivalence boundary"):
            validate_plan(plan, SCHEMA)

    def test_identifiability_boundary_requires_abstention(self):
        plan = deepcopy(FIXTURE)
        plan["identifiability_boundary"]["expected_disposition"] = "positive_detection"
        with self.assertRaises(SpatialPlanValidationError):
            validate_plan(plan, SCHEMA)

    def test_critical_null_replicate_floor_is_enforced(self):
        plan = deepcopy(FIXTURE)
        plan["seed_policy"]["critical_null_replicates"] = 9999
        with self.assertRaises(SpatialPlanValidationError):
            validate_plan(plan, SCHEMA)

    def test_matched_innovation_contract_cannot_be_disabled(self):
        plan = deepcopy(FIXTURE)
        plan["seed_policy"]["matched_nuisance_innovations"] = False
        with self.assertRaises(SpatialPlanValidationError):
            validate_plan(plan, SCHEMA)

    def test_effect_drift_and_abstention_outputs_are_mandatory(self):
        for required in ("effect_drift", "correct_abstention", "false_abstention"):
            plan = deepcopy(FIXTURE)
            plan["required_outputs"].remove(required)
            with self.subTest(required=required):
                with self.assertRaises(SpatialPlanValidationError):
                    validate_plan(plan, SCHEMA)

    def test_non_standard_nan_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"dose": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(SpatialPlanValidationError, "non-standard JSON"):
                load_json(path)


if __name__ == "__main__":
    unittest.main()
