#!/usr/bin/env python3
"""Adversarial tests for exact Spore product source ownership."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_spore_product_source_lineage.py"
spec = importlib.util.spec_from_file_location("spore_product_source", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class ProductSourceLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(mod.SOURCE_PATH.read_text())
        cls.migration = json.loads(mod.MIGRATION_PATH.read_text())
        cls.parity = json.loads(mod.PARITY_PATH.read_text())

    def errors(self, source=None, migration=None, parity=None):
        return mod.validate(
            copy.deepcopy(self.source if source is None else source),
            copy.deepcopy(self.migration if migration is None else migration),
            copy.deepcopy(self.parity if parity is None else parity),
        )

    def assert_rejected(self, mutate, expected):
        source = copy.deepcopy(self.source)
        migration = copy.deepcopy(self.migration)
        parity = copy.deepcopy(self.parity)
        mutate(source, migration, parity)
        joined = "\n".join(mod.validate(source, migration, parity))
        self.assertIn(expected, joined, joined)

    def test_current_contract_is_valid(self):
        self.assertEqual([], self.errors())

    def test_current_symthaea_repo_cannot_replace_actual_host_pin(self):
        def mutate(source, migration, parity):
            source["host_pin"]["input_repository"] = "Luminous-Dynamics/symthaea"
        self.assert_rejected(mutate, "host pin differs from exact reviewed flake.lock identity")

    def test_actual_source_commit_cannot_drift(self):
        def mutate(source, migration, parity):
            source["recovery_source"]["commit"] = "0" * 40
        self.assert_rejected(mutate, "recovery source differs from exact host-consumed qualified lineage")

    def test_artifact_friendly_id_cannot_retarget_path(self):
        def mutate(source, migration, parity):
            item = next(a for a in source["artifacts"] if a["id"] == "boot-state-lib")
            item["path"] = "crates/core/other/src/lib.rs"
        self.assert_rejected(mutate, "boot-state-lib: path drifted")

    def test_artifact_friendly_id_cannot_retarget_blob(self):
        def mutate(source, migration, parity):
            item = next(a for a in source["artifacts"] if a["id"] == "boot-state-lib")
            item["blob_sha1"] = "0" * 40
        self.assert_rejected(mutate, "boot-state-lib: source blob drifted")

    def test_mixed_package_cannot_be_moved_wholesale_to_spore(self):
        def mutate(source, migration, parity):
            item = next(a for a in source["artifacts"] if a["id"] == "spore-boot-tools-package")
            item["target_owner"] = "spore"
        self.assert_rejected(mutate, "spore-boot-tools-package: target owner drifted")

    def test_mixed_boot_ecology_cannot_be_moved_wholesale_to_spore(self):
        def mutate(source, migration, parity):
            item = next(a for a in source["artifacts"] if a["id"] == "boot-ecology-lib")
            item["target_owner"] = "spore"
        self.assert_rejected(mutate, "boot-ecology-lib: target owner drifted")

    def test_destination_paths_are_impossible_before_repo_exists(self):
        def mutate(source, migration, parity):
            source["artifacts"][0]["destination_path"] = "nix/spore.nix"
        self.assert_rejected(mutate, "destination_path impossible before destination repo exists")

    def test_authority_finding_cannot_be_closed_in_place(self):
        def mutate(source, migration, parity):
            item = next(f for f in source["findings"] if f["id"] == "SRC-003")
            item["status"] = "RESOLVED"
        self.assert_rejected(mutate, "SRC-003: cannot close finding without a new source-audit version")

    def test_authority_finding_remains_bound_to_issue_51(self):
        def mutate(source, migration, parity):
            item = next(f for f in source["findings"] if f["id"] == "SRC-003")
            item["tracking_issue"] = "none"
        self.assert_rejected(mutate, "SRC-003 must remain bound to source authority issue #51")

    def test_presentation_veto_finding_remains_bound_to_issue_56(self):
        def mutate(source, migration, parity):
            item = next(f for f in source["findings"] if f["id"] == "SRC-004")
            item["tracking_issue"] = "none"
        self.assert_rejected(
            mutate,
            "SRC-004 must remain bound to presentation-veto authority issue #56",
        )

    def test_src004_cannot_drop_already_known_good_per_boot_defect(self):
        def mutate(source, migration, parity):
            item = next(f for f in source["findings"] if f["id"] == "SRC-004")
            item["statement"] = (
                "Presentation state remains on recovery preparation and LKG promotion authority paths through morphology lineage and BootEcologyComposer."
            )
        self.assert_rejected(
            mutate,
            "SRC-004 must retain the healthy AlreadyKnownGood per-boot blessing defect",
        )

    def test_repair_002_cannot_leave_presentation_in_preparation_authority(self):
        def mutate(source, migration, parity):
            item = next(
                r for r in source["required_pre_extraction_repairs"]
                if r["id"] == "REPAIR-002"
            )
            item["statement"] = (
                "Remove presentation from qualification and LKG commit authority and use a recovery-native prepared-boot identity independent of morphology history."
            )
        self.assert_rejected(
            mutate,
            "REPAIR-002 must remove presentation authority and preserve exact per-boot qualification truth",
        )

    def test_repair_002_cannot_skip_already_known_good_exact_bless(self):
        def mutate(source, migration, parity):
            item = next(
                r for r in source["required_pre_extraction_repairs"]
                if r["id"] == "REPAIR-002"
            )
            item["statement"] = (
                "Remove morphology/presentation state from factual recovery preparation, qualification and LKG-commit authority and introduce a recovery-native prepared-boot identity independent of morphology history."
            )
        self.assert_rejected(
            mutate,
            "REPAIR-002 must remove presentation authority and preserve exact per-boot qualification truth",
        )

    def test_repairs_cannot_advance_without_new_versioned_lineage(self):
        def mutate(source, migration, parity):
            source["required_pre_extraction_repairs"][0]["status"] = "DONE"
        self.assert_rejected(mutate, "cannot advance before a versioned repaired-source lineage exists")

    def test_parent_migration_cannot_allow_qualification_inheritance(self):
        def mutate(source, migration, parity):
            migration["qualification_transfer_policy"] = "inherit"
        self.assert_rejected(mutate, "parent migration contract no longer forbids qualification inheritance")

    def test_parity_destination_must_match_source_audit(self):
        def mutate(source, migration, parity):
            parity["destination_repository"] = "Luminous-Dynamics/not-spore"
        self.assert_rejected(mutate, "source audit destination differs from parity contract")


if __name__ == "__main__":
    unittest.main()
