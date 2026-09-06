#!/usr/bin/env python3
"""Adversarial tests for the Spore golden parity migration contract."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_spore_golden_parity.py"
spec = importlib.util.spec_from_file_location("spore_golden_parity", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class GoldenParityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parity = json.loads(mod.PARITY_PATH.read_text())
        cls.migration = json.loads(mod.MIGRATION_PATH.read_text())
        cls.markdown = mod.MARKDOWN_PATH.read_text()
        cls.receipt_schema = json.loads(mod.RECEIPT_SCHEMA_PATH.read_text())

    def errors(self, parity=None, migration=None, markdown=None):
        return mod.validate(
            copy.deepcopy(self.parity if parity is None else parity),
            copy.deepcopy(self.migration if migration is None else migration),
            self.markdown if markdown is None else markdown,
        )

    def assert_rejected(self, mutate, expected):
        parity = copy.deepcopy(self.parity)
        migration = copy.deepcopy(self.migration)
        markdown = self.markdown
        result = mutate(parity, migration, markdown)
        if result is not None:
            markdown = result
        joined = "\n".join(mod.validate(parity, migration, markdown))
        self.assertIn(expected, joined, joined)

    def test_current_contract_is_valid(self):
        self.assertEqual([], self.errors())
        self.assertEqual([], mod.validate_receipt_schema(copy.deepcopy(self.receipt_schema)))

    def test_duplicate_obligation_id_is_rejected(self):
        def mutate(parity, migration, markdown):
            parity["obligations"][1]["id"] = parity["obligations"][0]["id"]
        self.assert_rejected(mutate, "duplicate obligation id")

    def test_orphaned_source_artifact_is_rejected(self):
        def mutate(parity, migration, markdown):
            parity["obligations"][0]["source_artifact_id"] = "missing-fixture"
        self.assert_rejected(mutate, "is not parity-approved")

    def test_non_test_source_cannot_become_parity_evidence(self):
        def mutate(parity, migration, markdown):
            target = next(a for a in migration["artifacts"] if a["id"] == "fail-open-vm")
            target["source_role"] = "product-source"
        self.assert_rejected(mutate, "parity source must remain a test-fixture")

    def test_source_owner_drift_is_rejected(self):
        def mutate(parity, migration, markdown):
            target = next(a for a in migration["artifacts"] if a["id"] == "fail-open-vm")
            target["target_owner"] = "symthaea"
        self.assert_rejected(mutate, "parity source target owner must remain spore")

    def test_source_path_retarget_is_rejected(self):
        def mutate(parity, migration, markdown):
            target = next(a for a in migration["artifacts"] if a["id"] == "fail-open-vm")
            target["source_path"] = "tests/something-else.nix"
        self.assert_rejected(mutate, "source artifact path drifted")

    def test_source_blob_retarget_is_rejected(self):
        def mutate(parity, migration, markdown):
            target = next(a for a in migration["artifacts"] if a["id"] == "fail-open-vm")
            target["source_blob_sha1"] = "0" * 40
        self.assert_rejected(mutate, "source artifact blob drifted")

    def test_transformed_candidate_boundary_cannot_be_strengthened_silently(self):
        def mutate(parity, migration, markdown):
            lineage = next(
                l for l in migration["lineages"]
                if l["id"] == "nixos-config-runtime-expendability-v1.3.2"
            )
            lineage["qualification_boundary"] = "exact-committed-source"
        self.assert_rejected(mutate, "source lineage must remain classified transformed-candidate")

    def test_qualification_inheritance_is_rejected(self):
        def mutate(parity, migration, markdown):
            migration["qualification_transfer_policy"] = "inherit"
        self.assert_rejected(mutate, "parent migration manifest no longer forbids qualification inheritance")

    def test_execution_class_must_match_source_fixture(self):
        def mutate(parity, migration, markdown):
            item = next(o for o in parity["obligations"] if o["id"] == "GP-AUTH-001")
            item["execution_class"] = "vm"
        self.assert_rejected(mutate, "must be 'unit' for source 'systemd-authority-tests'")

    def test_markdown_manifest_drift_is_rejected(self):
        def mutate(parity, migration, markdown):
            return markdown.replace("### GP-FW-005", "### GP-FW-999", 1)
        self.assert_rejected(mutate, "markdown/manifest obligation sets differ")

    def test_early_pass_is_impossible_before_destination_exists(self):
        def mutate(parity, migration, markdown):
            item = parity["obligations"][0]
            item["status"] = "EXECUTED_PASS"
            item["destination_receipt"] = "receipts/fake.json"
        self.assert_rejected(mutate, "no obligation may advance beyond DEFINED")

    def test_executed_state_requires_destination_receipt(self):
        def mutate(parity, migration, markdown):
            migration["destination_repository_status"] = "created"
            item = parity["obligations"][0]
            item["status"] = "EXECUTED_PASS"
        self.assert_rejected(mutate, "executed status requires destination_receipt")

    def test_superseded_state_requires_justification(self):
        def mutate(parity, migration, markdown):
            migration["destination_repository_status"] = "created"
            item = parity["obligations"][0]
            item["status"] = "SUPERSEDED_WITH_JUSTIFICATION"
        self.assert_rejected(mutate, "superseded status requires status_justification")

    def test_parity_receipt_cannot_claim_product_qualification(self):
        schema = copy.deepcopy(self.receipt_schema)
        schema["properties"]["destination_qualification"]["const"] = "QUALIFIED"
        joined = "\n".join(mod.validate_receipt_schema(schema))
        self.assertIn("destination_qualification must be const 'NOT_ESTABLISHED'", joined)

    def test_parity_receipt_cannot_upgrade_its_evidence_tier(self):
        schema = copy.deepcopy(self.receipt_schema)
        schema["properties"]["evidence_tier"]["const"] = "qualification"
        joined = "\n".join(mod.validate_receipt_schema(schema))
        self.assertIn("evidence_tier must be const 'parity-only'", joined)


if __name__ == "__main__":
    unittest.main()
