import copy
import json
import unittest
from pathlib import Path

from validate_receipt import QualificationError, validate_receipt


ROOT = Path(__file__).parent
SCHEMA = json.loads((ROOT / "measurement-qualification.schema.json").read_text())
FIXTURE = json.loads((ROOT / "fixtures" / "synthetic-qualification.json").read_text())


class MeasurementQualificationTests(unittest.TestCase):
    def test_synthetic_not_assessed_receipt_passes(self):
        validate_receipt(FIXTURE, SCHEMA)

    def test_signal_analysis_requires_verified_manifest(self):
        receipt = copy.deepcopy(FIXTURE)
        receipt["evidence_status"] = "signal_analysis"
        receipt["data_receipt"]["access_state"] = "metadata_only"
        with self.assertRaisesRegex(QualificationError, "downloaded_verified"):
            validate_receipt(receipt, SCHEMA)

    def test_adequate_repeatability_requires_interval_and_participants(self):
        receipt = copy.deepcopy(FIXTURE)
        receipt["repeatability"]["status"] = "adequate"
        receipt["repeatability"]["within_state_estimate"] = 0.8
        receipt["repeatability"]["participant_count"] = 24
        with self.assertRaisesRegex(QualificationError, "confidence interval"):
            validate_receipt(receipt, SCHEMA)

    def test_reversed_repeatability_interval_fails(self):
        receipt = copy.deepcopy(FIXTURE)
        receipt["repeatability"]["confidence_interval"] = {
            "lower": 0.8, "upper": 0.7, "confidence_level": 0.95
        }
        with self.assertRaisesRegex(QualificationError, "reversed"):
            validate_receipt(receipt, SCHEMA)

    def test_independent_cohort_cannot_be_a_new_split_of_same_release(self):
        receipt = copy.deepcopy(FIXTURE)
        receipt["transportability"]["independent_cohort"] = {
            "status": "adequate",
            "sample_size": 50,
            "source_independence": "same_release_heldout_site",
            "details": "Deliberately invalid claim."
        }
        with self.assertRaisesRegex(QualificationError, "genuinely independent"):
            validate_receipt(receipt, SCHEMA)

    def test_adequate_site_holdout_cannot_be_random_participant_split(self):
        receipt = copy.deepcopy(FIXTURE)
        receipt["transportability"]["site_holdout"] = {
            "status": "adequate",
            "sample_size": 40,
            "source_independence": "same_participants_new_split",
            "details": "Deliberately invalid claim."
        }
        with self.assertRaisesRegex(QualificationError, "random participant split"):
            validate_receipt(receipt, SCHEMA)

    def test_adequate_construct_validity_requires_positive_and_negative_controls(self):
        receipt = copy.deepcopy(FIXTURE)
        receipt["construct_validity"]["status"] = "adequate"
        receipt["construct_validity"]["positive_controls"] = []
        with self.assertRaisesRegex(QualificationError, "positive controls"):
            validate_receipt(receipt, SCHEMA)

    def test_no_composite_score_is_mandatory(self):
        receipt = copy.deepcopy(FIXTURE)
        receipt["no_composite_score"] = False
        with self.assertRaisesRegex(QualificationError, "schema violation"):
            validate_receipt(receipt, SCHEMA)


if __name__ == "__main__":
    unittest.main()
