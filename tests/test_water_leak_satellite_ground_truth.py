"""Offline tests for the ground-truth schema and fail-closed label validator."""

from __future__ import annotations

import unittest

from tools.water_leak_satellite.ground_truth import (
    label_class,
    validate_corpus,
    validate_record,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def record(record_id="evt-1", split="train", group="zone-a-period-a"):
    return {
        "schema_version": "water-leak-ground-truth/v1",
        "record_id": record_id,
        "zone_id": "zone-a",
        "asset_segment_id": "pipe-seg-1",
        "observed_at": "2026-08-01T10:00:00Z",
        "outcome": "confirmed_leak",
        "location": {
            "longitude": 28.0473,
            "latitude": -26.2041,
            "horizontal_uncertainty_m": 8.0,
            "coordinate_reference_system": "EPSG:4326",
        },
        "verification": {
            "method": "excavation_confirmed",
            "independent": True,
            "evidence_sha256": HASH_A,
        },
        "rights": {
            "owner": "utility-data-owner",
            "allowed_uses": ["internal_research"],
            "permission_reference_sha256": HASH_B,
        },
        "evaluation_split": split,
        "split_group_id": group,
    }


def corpus(records):
    return {"schema_version": "water-leak-ground-truth-corpus/v1", "records": records}


class GroundTruthTests(unittest.TestCase):
    def test_independently_confirmed_leak_is_positive(self):
        item = record()
        self.assertEqual(validate_record(item), [])
        self.assertEqual(label_class(item), "positive")

    def test_acoustic_only_cannot_be_promoted_to_confirmed_leak(self):
        item = record()
        item["verification"]["method"] = "acoustic_localization_only"
        errors = validate_record(item)
        self.assertTrue(any("requires a physical/repair confirmation" in error for error in errors))
        self.assertEqual(label_class(item), "excluded")

    def test_no_leak_requires_an_actual_documented_inspection(self):
        item = record()
        item["outcome"] = "inspected_no_leak"
        item["verification"]["method"] = "utility_record_only"
        errors = validate_record(item)
        self.assertTrue(any("requires documented_no_leak_inspection" in error for error in errors))
        self.assertEqual(label_class(item), "excluded")

    def test_unknown_is_not_silently_treated_as_negative(self):
        item = record()
        item["outcome"] = "unknown"
        self.assertEqual(validate_record(item), [])
        self.assertEqual(label_class(item), "excluded")

    def test_timezone_is_required(self):
        item = record()
        item["observed_at"] = "2026-08-01T10:00:00"
        self.assertTrue(any("ISO-8601 timestamp with timezone" in error for error in validate_record(item)))

    def test_coordinates_and_uncertainty_are_bounded(self):
        item = record()
        item["location"]["longitude"] = 280.0
        item["location"]["horizontal_uncertainty_m"] = 0.0
        errors = validate_record(item)
        self.assertTrue(any("longitude" in error for error in errors))
        self.assertTrue(any("horizontal_uncertainty_m" in error for error in errors))

    def test_commercial_training_requires_explicit_permission(self):
        item = record()
        errors, counts = validate_corpus(corpus([item]), purpose="commercial_model_training")
        self.assertTrue(any("does not explicitly permit commercial_model_training" in error for error in errors))
        self.assertEqual(counts["purpose_eligible"], 0)

    def test_duplicate_record_ids_fail(self):
        a = record(record_id="duplicate", group="g1")
        b = record(record_id="duplicate", group="g2")
        errors, _ = validate_corpus(corpus([a, b]))
        self.assertTrue(any("duplicate record_id" in error for error in errors))

    def test_zone_cannot_cross_splits_when_group_id_is_changed(self):
        a = record(record_id="evt-a", split="train", group="block-1")
        b = record(record_id="evt-b", split="held_out_test", group="block-2")
        b["asset_segment_id"] = "pipe-seg-2"
        b["observed_at"] = "2026-09-01T10:00:00Z"
        errors, _ = validate_corpus(corpus([a, b]))
        self.assertTrue(any("zone crosses evaluation splits" in error for error in errors))

    def test_heldout_test_must_be_later_than_training(self):
        a = record(record_id="evt-a", split="train", group="block-train")
        b = record(record_id="evt-b", split="held_out_test", group="block-test")
        b["zone_id"] = "zone-b"
        b["asset_segment_id"] = "pipe-seg-2"
        b["observed_at"] = "2026-07-01T10:00:00Z"
        errors, _ = validate_corpus(corpus([a, b]))
        self.assertTrue(any("strictly later" in error for error in errors))

    def test_malformed_enum_values_do_not_crash_validator(self):
        bad_outcome = record()
        bad_outcome["outcome"] = {"unexpected": "mapping"}
        bad_method = record()
        bad_method["verification"]["method"] = ["not", "a", "method"]
        bad_uses = record()
        bad_uses["rights"]["allowed_uses"] = [[]]
        huge_coordinate = record()
        huge_coordinate["location"]["longitude"] = 10 ** 400
        for item in (bad_outcome, bad_method, bad_uses, huge_coordinate):
            with self.subTest(record_id=item["record_id"]):
                self.assertTrue(validate_record(item))
        errors, _ = validate_corpus(corpus([bad_outcome]))
        self.assertTrue(errors)

    def test_space_time_group_cannot_cross_splits(self):
        a = record(record_id="evt-a", split="train", group="same-spatiotemporal-block")
        b = record(record_id="evt-b", split="held_out_test", group="same-spatiotemporal-block")
        errors, _ = validate_corpus(corpus([a, b]))
        self.assertTrue(any("group crosses splits" in error for error in errors))

    def test_explicitly_permitted_corpus_passes_contract_only(self):
        a = record(record_id="evt-a", split="train", group="block-train")
        b = record(record_id="evt-b", split="held_out_test", group="block-test")
        b["zone_id"] = "zone-b"
        b["asset_segment_id"] = "pipe-seg-2"
        b["observed_at"] = "2026-09-01T10:00:00Z"
        b["location"]["longitude"] = 28.15
        b["location"]["latitude"] = -26.25
        a["rights"]["allowed_uses"] = ["internal_research", "commercial_model_training"]
        b["rights"]["allowed_uses"] = ["internal_research", "commercial_model_training"]
        errors, counts = validate_corpus(corpus([a, b]), purpose="commercial_model_training")
        self.assertEqual(errors, [])
        self.assertEqual(counts["records"], 2)
        self.assertEqual(counts["positive"], 2)
        self.assertEqual(counts["purpose_eligible"], 2)

    def test_invalid_record_shape_fails_closed(self):
        item = record()
        item["personal_home_address"] = "should-not-be-present"
        self.assertTrue(any("unexpected keys" in error for error in validate_record(item)))


if __name__ == "__main__":
    unittest.main()
