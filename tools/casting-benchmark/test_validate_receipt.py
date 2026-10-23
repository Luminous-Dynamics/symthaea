"""Adversarial tests for the strict benchmark receipt validator."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from validate_receipt import validate_receipt

FIXTURE = Path(__file__).parent / "fixtures" / "success_receipt.fixture.json"


class ReceiptValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.receipt = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_fixture_success_has_valid_structure_but_is_not_publishable(self) -> None:
        self.assertEqual(validate_receipt(self.receipt), [])
        errors = validate_receipt(self.receipt, for_publication=True)
        self.assertTrue(any("fixture receipts cannot be published" in error for error in errors))

    def test_skipped_or_queued_status_never_passes_as_terminal_receipt(self) -> None:
        for status in ("skipped", "queued", "pending"):
            with self.subTest(status=status):
                invalid = copy.deepcopy(self.receipt)
                invalid["status"] = status
                self.assertTrue(validate_receipt(invalid))

    def test_unknown_keys_fail_closed(self) -> None:
        invalid = copy.deepcopy(self.receipt)
        invalid["unreviewed_success_flag"] = True
        self.assertTrue(any("unknown keys" in error for error in validate_receipt(invalid)))

    def test_budget_overrun_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.receipt)
        invalid["accounting"]["objective_evaluations"] = 65
        invalid["accounting"]["feasible_evaluations"] = 61
        self.assertTrue(any("exceeds frozen budget" in error for error in validate_receipt(invalid)))

    def test_evaluation_buckets_must_partition_all_calls(self) -> None:
        invalid = copy.deepcopy(self.receipt)
        invalid["accounting"]["failed_evaluations"] = 0
        self.assertTrue(any("must sum exactly" in error for error in validate_receipt(invalid)))

    def test_success_with_error_termination_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.receipt)
        invalid["termination_reason"] = "error"
        self.assertTrue(any("success requires predeclared terminal condition" in error for error in validate_receipt(invalid)))

    def test_hashes_and_git_revisions_are_immutable_format(self) -> None:
        invalid = copy.deepcopy(self.receipt)
        invalid["protocol_sha256"] = "not-a-hash"
        invalid["source_revision"] = "main"
        errors = validate_receipt(invalid)
        self.assertTrue(any("$.protocol_sha256" in error for error in errors))
        self.assertTrue(any("$.source_revision" in error for error in errors))

    def test_nan_and_infinity_are_rejected(self) -> None:
        invalid = copy.deepcopy(self.receipt)
        invalid["metrics"]["normalized_hypervolume_auc"] = float("nan")
        self.assertTrue(any("normalized_hypervolume_auc" in error for error in validate_receipt(invalid)))
        invalid = copy.deepcopy(self.receipt)
        invalid["metrics"]["final_hypervolume"] = float("inf")
        self.assertTrue(any("final_hypervolume" in error for error in validate_receipt(invalid)))

    def test_malformed_enum_types_return_errors_not_exceptions(self) -> None:
        for field in ("status", "termination_reason"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.receipt)
                invalid[field] = []
                self.assertTrue(validate_receipt(invalid))

    def test_success_requires_metrics_and_a_success_termination(self) -> None:
        invalid = copy.deepcopy(self.receipt)
        invalid.pop("metrics")
        self.assertTrue(any("$.metrics" in error for error in validate_receipt(invalid)))
        invalid = copy.deepcopy(self.receipt)
        invalid["termination_reason"] = "missing_or_skipped"
        self.assertTrue(validate_receipt(invalid))


if __name__ == "__main__":
    unittest.main()
