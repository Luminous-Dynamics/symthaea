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


    def traceability_for_task(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        behavior = task["behavior_contract"]
        effects = behavior["observable_effects"]
        required = set(task["oracle_contract"]["required_test_ids"])
        by_invariant = [
            ["ledger.invalid_amount_fails_closed", "ledger.amount_token_canonical", "ledger.overflow_fails_closed", "ledger.maximum_credit_debit_valid"],
            ["ledger.exact_duplicate_idempotency", "ledger.duplicate_multiplicity_idempotent"],
            ["ledger.conflicting_duplicate_rejected", "ledger.conflict_has_no_partial_effect"],
            ["ledger.normal_exact_output", "ledger.transaction_order_invariant"],
            ["ledger.account_totals", "ledger.global_totals", "ledger.generated_differential_reducer"],
            ["ledger.overflow_fails_closed", "ledger.maximum_credit_debit_valid", "ledger.generated_differential_reducer"],
        ]
        by_effect = {
            "stdout_bytes": ["ledger.normal_exact_output"],
            "exit_code": ["ledger.invalid_amount_fails_closed", "ledger.conflicting_duplicate_rejected"],
            "control_totals": ["ledger.account_totals", "ledger.global_totals"],
            "numeric_semantics": ["ledger.amount_token_canonical", "ledger.overflow_fails_closed"],
            "idempotency": ["ledger.exact_duplicate_idempotency"],
        }
        trace: list[dict[str, Any]] = []

        def add(requirement_id: str, source_kind: str, source_text: str, test_ids: list[str], observable_effects: list[str]) -> None:
            selected = [test_id for test_id in test_ids if test_id in required]
            trace.append({
                "requirement_id": requirement_id,
                "source_kind": source_kind,
                "source_text": source_text,
                "required_test_ids": selected,
                "observable_effects": observable_effects,
            })

        add("TRACE.STATEMENT", "task_statement", behavior["task_statement"],
            ["ledger.normal_exact_output", "ledger.unknown_side_fails_closed", "ledger.final_line_ending_invariant", "ledger.malformed_csv_rejected"],
            effects)
        for index, invariant in enumerate(behavior["invariants"]):
            test_ids = by_invariant[index] if index < len(by_invariant) else []
            invariant_effects = (
                ["numeric_semantics", "exit_code"] if index == 0 else
                ["idempotency", "control_totals"] if index == 1 else
                ["exit_code"] if index == 2 else
                ["stdout_bytes"] if index == 3 else
                ["control_totals"] if index == 4 else
                ["numeric_semantics", "control_totals"]
            )
            add(f"TRACE.INVARIANT.{index + 1:03d}", "invariant", invariant, test_ids, invariant_effects)
        for index, change in enumerate(behavior["allowed_behavior_changes"]):
            add(f"TRACE.ALLOWED.{index + 1:03d}", "allowed_behavior_change", change, [], effects)
        for effect in effects:
            add(f"TRACE.EFFECT.{effect.upper()}", "observable_effect", effect, by_effect.get(effect, []), [effect])
        return trace

    def valid_private_task(self) -> dict[str, Any]:
        task = copy.deepcopy(self.task)
        task["classification"].update({
            "exposure": "private_confirmatory",
            "reviewer_count": 2,
        })
        task["behavior_contract"]["requirement_traceability"] = self.traceability_for_task(task)
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

    def test_private_task_requires_requirement_traceability(self) -> None:
        task = self.valid_private_task()
        del task["behavior_contract"]["requirement_traceability"]
        errors = self.errors(task)
        self.assertTrue(any("requirement_traceability" in error for error in errors), errors)

    def test_requirement_traceability_entry_requires_concrete_test_ids(self) -> None:
        task = self.valid_private_task()
        del task["behavior_contract"]["requirement_traceability"][0]["required_test_ids"]
        errors = self.errors(task)
        self.assertTrue(any("required_test_ids" in error for error in errors), errors)

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
