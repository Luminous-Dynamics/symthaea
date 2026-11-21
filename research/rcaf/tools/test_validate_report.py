"""Unit tests for the isolated RCAF JSON report validator."""

from copy import deepcopy
import unittest

from validate_report import ReportValidationError, exact_sign_test, validate_semantics


def fixture():
    gaps = [0.05] * 5
    return {
        "schema_version": "rcaf.report/1.0.0",
        "run_id": "synthetic-rcaf-fixture-test",
        "evidence_status": "synthetic_fixture",
        "dataset": {
            "provider": "CETB synthetic fixture",
            "dataset_id": "synthetic-paired-example",
            "version": None,
            "doi": None,
            "access_state": "not_downloaded",
            "file_manifest_sha256": None,
            "participant_flow": {
                "acquired_or_recruited": 5,
                "released": 5,
                "qc_eligible": 5,
                "analysis_included": 5,
                "excluded": 0,
            },
        },
        "analysis": {
            "protocol_id": "RCAF-SYNTHETIC-TEST",
            "metric": "weighted_jaccard",
            "estimator": "synthetic known-answer",
            "preprocessing_revision": "none",
            "roi_order_sha256": "a" * 64,
            "segment_volumes": 300,
            "repeat_placements_per_participant": 5,
            "minimum_common_edges": 1,
            "neutral_sign_threshold": 0.05,
        },
        "participants": [
            {
                "participant_key": f"synthetic-{i+1:02d}",
                "site": "synthetic",
                "phase": "unknown",
                "within_similarity": 0.55,
                "between_similarity": 0.50,
                "paired_gap": gaps[i],
                "within_repeat_count": 5,
                "between_repeat_count": 5,
                "mean_framewise_displacement_mm": None,
                "censor_fraction": None,
                "network_similarity_index": None,
            }
            for i in range(5)
        ],
        "summary": {
            "participant_count": 5,
            "median_within_similarity": 0.55,
            "median_between_similarity": 0.50,
            "median_paired_gap": 0.05,
            "positive_gap_fraction": 1.0,
            "sign_test": {
                "positive_count": 5,
                "negative_count": 0,
                "tie_count": 0,
                "non_tied_count": 5,
                "p_two_sided_exact": 0.0625,
            },
            "participant_bootstrap_ci": {
                "lower": 0.05,
                "upper": 0.05,
                "confidence_level": 0.95,
                "draws": 2000,
                "seed": 42,
            },
            "edge_sign_transitions": {
                "total_edges": 3,
                "polarity_reversal_count": 1,
                "sign_state_change_count": 2,
                "neutral_threshold": 0.05,
            },
        },
        "claim_limits": {
            "phenomenal_experience_inference": "not_established_by_rcaf",
            "causal_interpretation": "not_established_by_observational_similarity",
            "clinical_use": "not_validated_for_clinical_use",
            "notes": ["Synthetic fixture only."],
        },
    }


class RcafValidatorTests(unittest.TestCase):
    def test_valid_report_passes(self):
        validate_semantics(fixture())

    def test_exact_sign_test_known_answers(self):
        self.assertAlmostEqual(exact_sign_test([1.0, 1.0, 1.0, 1.0, 1.0])[4], 0.0625)
        self.assertEqual(exact_sign_test([1.0, -1.0, 0.0])[4], 1.0)
        expected = 2.0 / (2**23)
        self.assertAlmostEqual(exact_sign_test([1.0] * 23)[4], expected, places=15)

    def test_gap_arithmetic_is_recomputed(self):
        report = fixture()
        report["participants"][0]["paired_gap"] = 0.2
        with self.assertRaisesRegex(ReportValidationError, "paired_gap"):
            validate_semantics(report)

    def test_duplicate_participant_is_rejected(self):
        report = fixture()
        report["participants"][1]["participant_key"] = report["participants"][0]["participant_key"]
        with self.assertRaisesRegex(ReportValidationError, "duplicated"):
            validate_semantics(report)

    def test_summary_count_must_match_rows(self):
        report = fixture()
        report["summary"]["participant_count"] = 6
        with self.assertRaisesRegex(ReportValidationError, "participant_count"):
            validate_semantics(report)

    def test_sign_test_is_recomputed(self):
        report = fixture()
        report["summary"]["sign_test"]["p_two_sided_exact"] = 0.5
        with self.assertRaisesRegex(ReportValidationError, "exact binomial"):
            validate_semantics(report)

    def test_nonmonotonic_flow_is_rejected(self):
        report = fixture()
        report["dataset"]["participant_flow"]["qc_eligible"] = 6
        with self.assertRaisesRegex(ReportValidationError, "participant flow"):
            validate_semantics(report)

    def test_verified_download_requires_manifest_digest(self):
        report = fixture()
        report["evidence_status"] = "signal_analysis"
        report["dataset"]["access_state"] = "downloaded_verified"
        report["dataset"]["doi"] = "10.1234/example"
        with self.assertRaisesRegex(ReportValidationError, "SHA-256"):
            validate_semantics(report)

    def test_ci_order_is_enforced(self):
        report = fixture()
        report["summary"]["participant_bootstrap_ci"]["lower"] = 0.2
        report["summary"]["participant_bootstrap_ci"]["upper"] = 0.1
        with self.assertRaisesRegex(ReportValidationError, "interval"):
            validate_semantics(report)


if __name__ == "__main__":
    unittest.main()
