"""Offline tests for the privacy-preserving energy telemetry contract."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools.energy_telemetry.privacy_contract import (
    MAX_RELEASE_DELTA,
    MAX_RELEASE_EPSILON,
    load_and_validate,
    validate_envelope,
)


def valid_envelope():
    return {
        "schema_version": "privacy-preserving-energy-telemetry/v1",
        "batch_id": "0123456789abcdef0123456789abcdef",
        "created_at_utc": "2026-10-12T10:15:00Z",
        "window": {
            "start_utc": "2026-10-12T10:00:00Z",
            "end_utc": "2026-10-12T10:15:00Z",
            "duration_seconds": 900,
        },
        "aggregation": {
            "scope": "distribution_area",
            "region_token": "district_opaque_01",
            "cohort_size": 64,
            "minimum_cohort_size": 20,
            "energy_wh_total": 125000,
            "method": "trusted_edge_aggregation",
            "coverage_fraction": 0.94,
            "measurement_quality": "validated",
        },
        "recipient_class": "grid_operations",
        "dp_release": None,
    }


def add_dp_release(document):
    document["recipient_class"] = "approved_research"
    document["aggregation"]["method"] = "secure_aggregation_with_dp"
    document["dp_release"] = {
        "mechanism": "discrete_laplace",
        "epsilon": 0.5,
        "delta": 0.0,
        "budget_policy_id": "research-weekly-v1",
        "budget_receipt_sha256": "a" * 64,
    }
    return document


class PrivacyContractTests(unittest.TestCase):
    def test_valid_grid_operations_aggregate_passes(self):
        self.assertEqual(validate_envelope(valid_envelope()), [])

    def test_valid_research_aggregate_requires_dp(self):
        self.assertEqual(validate_envelope(add_dp_release(valid_envelope())), [])

    def test_unknown_top_level_personal_data_field_is_rejected(self):
        document = valid_envelope()
        document["meter_id"] = "meter-123"
        self.assertTrue(any("unknown keys" in e or "forbidden" in e for e in validate_envelope(document)))

    def test_customer_address_is_rejected(self):
        document = valid_envelope()
        document["address"] = "1 Example Road"
        self.assertTrue(any("unknown keys" in e or "forbidden" in e for e in validate_envelope(document)))

    def test_nested_meter_identifier_is_rejected(self):
        document = valid_envelope()
        document["aggregation"]["meter_id"] = "meter-123"
        self.assertTrue(any("aggregation" in e and "unknown keys" in e for e in validate_envelope(document)))

    def test_missing_required_key_fails(self):
        document = valid_envelope()
        del document["window"]["duration_seconds"]
        self.assertTrue(any("duration_seconds" in e for e in validate_envelope(document)))

    def test_below_default_cohort_floor_is_rejected(self):
        document = valid_envelope()
        document["aggregation"]["cohort_size"] = 19
        self.assertTrue(any("cohort_size" in e for e in validate_envelope(document)))

    def test_cohort_below_declared_policy_minimum_is_rejected(self):
        document = valid_envelope()
        document["aggregation"]["cohort_size"] = 24
        document["aggregation"]["minimum_cohort_size"] = 25
        self.assertTrue(any("below the declared minimum" in e for e in validate_envelope(document)))

    def test_boolean_is_not_accepted_as_integer_cohort_count(self):
        document = valid_envelope()
        document["aggregation"]["cohort_size"] = True
        self.assertTrue(any("cohort_size" in e for e in validate_envelope(document)))

    def test_negative_energy_is_rejected(self):
        document = valid_envelope()
        document["aggregation"]["energy_wh_total"] = -1
        self.assertTrue(any("energy_wh_total" in e for e in validate_envelope(document)))

    def test_boolean_energy_is_rejected(self):
        document = valid_envelope()
        document["aggregation"]["energy_wh_total"] = True
        self.assertTrue(any("energy_wh_total" in e for e in validate_envelope(document)))

    def test_non_finite_coverage_is_rejected(self):
        document = valid_envelope()
        document["aggregation"]["coverage_fraction"] = float("nan")
        self.assertTrue(any("coverage_fraction" in e for e in validate_envelope(document)))

    def test_extremely_large_coverage_integer_fails_closed_without_overflow(self):
        document = valid_envelope()
        document["aggregation"]["coverage_fraction"] = 10 ** 10000
        self.assertTrue(any("coverage_fraction" in e for e in validate_envelope(document)))

    def test_schema_and_validator_required_fields_stay_aligned(self):
        schema_path = Path(__file__).resolve().parents[1] / "docs/research/energy_telemetry_envelope.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        for section in ("window", "aggregation"):
            definition = schema["properties"][section]
            self.assertEqual(set(definition["required"]), set(definition["properties"]))
        dp = schema["$defs"]["dpRelease"]
        self.assertEqual(set(dp["required"]), set(dp["properties"]))

    def test_coverage_outside_range_is_rejected(self):
        document = valid_envelope()
        document["aggregation"]["coverage_fraction"] = 1.01
        self.assertTrue(any("coverage_fraction" in e for e in validate_envelope(document)))

    def test_window_duration_must_match_timestamps(self):
        document = valid_envelope()
        document["window"]["duration_seconds"] = 600
        self.assertTrue(any("does not match timestamps" in e for e in validate_envelope(document)))

    def test_reversed_window_is_rejected(self):
        document = valid_envelope()
        document["window"]["start_utc"] = "2026-10-12T10:20:00Z"
        document["window"]["end_utc"] = "2026-10-12T10:00:00Z"
        self.assertTrue(any("later than" in e for e in validate_envelope(document)))

    def test_timezone_naive_timestamp_is_rejected(self):
        document = valid_envelope()
        document["created_at_utc"] = "2026-10-12T10:15:00"
        self.assertTrue(any("ending in Z" in e for e in validate_envelope(document)))

    def test_too_short_window_is_rejected(self):
        document = valid_envelope()
        document["window"].update({
            "start_utc": "2026-10-12T10:00:00Z",
            "end_utc": "2026-10-12T10:01:00Z",
            "duration_seconds": 60,
        })
        document["created_at_utc"] = "2026-10-12T10:02:00Z"
        self.assertTrue(any("duration_seconds" in e for e in validate_envelope(document)))

    def test_public_release_without_dp_is_rejected(self):
        document = valid_envelope()
        document["recipient_class"] = "public_reporting"
        self.assertTrue(any("requires secure aggregation with DP" in e for e in validate_envelope(document)))

    def test_dp_method_without_dp_receipt_is_rejected(self):
        document = valid_envelope()
        document["aggregation"]["method"] = "secure_aggregation_with_dp"
        self.assertTrue(any("required when aggregation.method uses DP" in e for e in validate_envelope(document)))

    def test_dp_epsilon_above_cap_is_rejected(self):
        document = add_dp_release(valid_envelope())
        document["dp_release"]["epsilon"] = MAX_RELEASE_EPSILON + 0.01
        self.assertTrue(any("epsilon" in e for e in validate_envelope(document)))

    def test_dp_delta_above_cap_is_rejected(self):
        document = add_dp_release(valid_envelope())
        document["dp_release"]["delta"] = MAX_RELEASE_DELTA * 2
        self.assertTrue(any("delta" in e for e in validate_envelope(document)))

    def test_dp_digest_must_be_sha256_shaped(self):
        document = add_dp_release(valid_envelope())
        document["dp_release"]["budget_receipt_sha256"] = "not-a-hash"
        self.assertTrue(any("budget_receipt_sha256" in e for e in validate_envelope(document)))

    def test_dp_cannot_be_attached_to_non_dp_method(self):
        document = valid_envelope()
        document["dp_release"] = {
            "mechanism": "discrete_laplace",
            "epsilon": 0.5,
            "delta": 0.0,
            "budget_policy_id": "grid-v1",
            "budget_receipt_sha256": "a" * 64,
        }
        self.assertTrue(any("must be null" in e for e in validate_envelope(document)))

    def test_batch_id_must_be_random_id_shaped_not_device_id(self):
        document = valid_envelope()
        document["batch_id"] = "meter-123"
        self.assertTrue(any("batch_id" in e for e in validate_envelope(document)))

    def test_duplicate_json_keys_are_rejected(self):
        document = json.dumps(valid_envelope(), separators=(",", ":"))
        duplicate = document[:-1] + ',"schema_version":"wrong"}'
        _, errors = load_and_validate(duplicate)
        self.assertTrue(any("duplicate JSON object key" in e for e in errors))

    def test_non_standard_json_nan_is_rejected(self):
        raw = '{"schema_version":"privacy-preserving-energy-telemetry/v1","bad":NaN}'
        _, errors = load_and_validate(raw)
        self.assertTrue(any("non-standard JSON numeric constant" in e for e in errors))

    def test_unknown_nested_fields_are_rejected_including_gps(self):
        document = valid_envelope()
        document["aggregation"]["gps"] = [-26.2, 28.0]
        self.assertTrue(any("forbidden or unknown keys" in e for e in validate_envelope(document)))

    def test_list_scope_value_fails_closed_without_type_error(self):
        document = valid_envelope()
        document["aggregation"]["scope"] = ["microgrid"]
        self.assertTrue(any("scope" in e for e in validate_envelope(document)))

    def test_list_method_value_fails_closed_without_type_error(self):
        document = valid_envelope()
        document["aggregation"]["method"] = ["secure_aggregation"]
        self.assertTrue(any("method" in e for e in validate_envelope(document)))

    def test_list_recipient_value_fails_closed_without_type_error(self):
        document = valid_envelope()
        document["recipient_class"] = ["public_reporting"]
        self.assertTrue(any("recipient_class" in e for e in validate_envelope(document)))

    def test_list_dp_mechanism_fails_closed_without_type_error(self):
        document = add_dp_release(valid_envelope())
        document["dp_release"]["mechanism"] = ["gaussian"]
        self.assertTrue(any("mechanism" in e for e in validate_envelope(document)))


if __name__ == "__main__":
    unittest.main()
