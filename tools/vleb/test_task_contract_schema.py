#!/usr/bin/env python3
"""Task-contract schema admission tests for public fixtures vs scored tasks."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "tools/vleb/task-contract.schema.json"
TASK_PATH = ROOT / "tools/vleb/examples/ledger-reconciliation/task.json"


class TaskContractSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema, format_checker=FormatChecker())

    def errors(self, task: dict[str, Any]) -> list[str]:
        return [
            ".".join(map(str, error.absolute_path)) + ": " + error.message
            for error in self.validator.iter_errors(task)
        ]

    def valid_private_task(self) -> dict[str, Any]:
        task = copy.deepcopy(self.task)
        task["classification"].update({
            "exposure": "private_confirmatory",
            "reviewer_count": 2,
        })
        task["oracle_contract"].update({
            "oracle_review": {
                "independent_reviewer_count": 2,
                "mutation_equivalence_reviewed": True,
                "notes": "synthetic positive control for schema admission",
            },
            "reference_evidence_sha256": {
                "baseline_log": "a" * 64,
                "reference_log": "b" * 64,
                "reference_patch": "c" * 64,
                "test_bundle": "d" * 64,
            },
            "reference_evidence_artifacts": {
                "baseline_log": "evidence/baseline.log",
                "reference_log": "evidence/reference.log",
                "reference_patch": "evidence/reference.patch",
                "test_bundle": "evidence/tests.zip",
            },
        })
        return task

    def test_public_development_fixture_can_remain_incomplete(self) -> None:
        self.assertEqual(self.errors(copy.deepcopy(self.task)), [])

    def test_private_task_without_admission_evidence_is_rejected(self) -> None:
        task = copy.deepcopy(self.task)
        task["classification"]["exposure"] = "private_confirmatory"
        task["classification"]["reviewer_count"] = 0
        errors = self.errors(task)
        self.assertTrue(any("reviewer_count" in error for error in errors), errors)
        self.assertTrue(any("oracle_review" in error for error in errors), errors)
        self.assertTrue(any("reference_evidence_sha256" in error for error in errors), errors)
        self.assertTrue(any("reference_evidence_artifacts" in error for error in errors), errors)

    def test_private_task_requires_independent_oracle_review(self) -> None:
        task = self.valid_private_task()
        task["oracle_contract"]["oracle_review"]["independent_reviewer_count"] = 0
        task["oracle_contract"]["oracle_review"]["mutation_equivalence_reviewed"] = False
        errors = self.errors(task)
        self.assertTrue(any("independent_reviewer_count" in error for error in errors), errors)
        self.assertTrue(any("mutation_equivalence_reviewed" in error for error in errors), errors)

    def test_private_task_requires_each_reference_hash_and_artifact_path(self) -> None:
        task = self.valid_private_task()
        del task["oracle_contract"]["reference_evidence_sha256"]["baseline_log"]
        del task["oracle_contract"]["reference_evidence_artifacts"]["reference_log"]
        errors = self.errors(task)
        self.assertTrue(any("baseline_log" in error for error in errors), errors)
        self.assertTrue(any("reference_log" in error for error in errors), errors)

    def test_reviewed_private_confirmatory_contract_passes_schema(self) -> None:
        task = self.valid_private_task()
        self.assertEqual(self.errors(task), [])

    def test_private_holdout_requires_same_controls_as_confirmatory(self) -> None:
        task = self.valid_private_task()
        task["classification"]["exposure"] = "private_holdout"
        self.assertEqual(self.errors(task), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
