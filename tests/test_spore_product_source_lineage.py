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

    def test_unknown_top_level_qualification_field_is_rejected(self):
        def mutate(source, migration, parity):
            source["qualification_status"] = "QUALIFIED"
        self.assert_rejected(mutate, "top-level source audit keys drifted")

    def test_unknown_artifact_authority_field_is_rejected(self):
        def mutate(source, migration, parity):
            source["artifacts"][0]["authority"] = "recovery-root"
        self.assert_rejected(mutate, "artifact[0] keys drifted")

    def test_unknown_finding_qualification_field_is_rejected(self):
        def mutate(source, migration, parity):
            source["findings"][2]["qualification"] = "established"
        self.assert_rejected(mutate, "finding[2] keys drifted")

    def test_unknown_repair_completion_field_is_rejected(self):
        def mutate(source, migration, parity):
            source["required_pre_extraction_repairs"][0]["completed"] = True
        self.assert_rejected(mutate, "repair[0] keys drifted")

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
                "Qualification lacks exact local subject binding; the Linux recovery CLI accepts boot_id and booted-generation identity as caller-provided values rather than locally observed machine facts; executable TransitionPlan/RecoveryOp values are deserializable data; presentation remains on recovery authority paths."
            )
        self.assert_rejected(
            mutate,
            "SRC-004 must retain exact per-boot, local-observation, and executable-capability authority defects",
        )

    def test_src004_cannot_drop_local_machine_identity_defect(self):
        def mutate(source, migration, parity):
            item = next(f for f in source["findings"] if f["id"] == "SRC-004")
            item["statement"] = (
                "Presentation state remains on recovery authority paths; qualification lacks exact subject binding; executable TransitionPlan/RecoveryOp values are deserializable data; and AlreadyKnownGood leaves the exact current boot with last_boot_blessed false after a healthy boot."
            )
        self.assert_rejected(
            mutate,
            "SRC-004 must retain exact per-boot, local-observation, and executable-capability authority defects",
        )

    def test_src004_cannot_drop_serialized_plan_capability_defect(self):
        def mutate(source, migration, parity):
            item = next(f for f in source["findings"] if f["id"] == "SRC-004")
            item["statement"] = (
                "Presentation state remains on recovery authority paths; the Linux recovery CLI accepts identity as caller-provided values rather than locally observed machine facts; and AlreadyKnownGood leaves the exact current boot with last_boot_blessed false after a healthy boot."
            )
        self.assert_rejected(
            mutate,
            "SRC-004 must retain exact per-boot, local-observation, and executable-capability authority defects",
        )

    def test_lifecycle_finding_remains_bound_to_issue_61(self):
        def mutate(source, migration, parity):
            item = next(f for f in source["findings"] if f["id"] == "SRC-005")
            item["tracking_issue"] = "none"
        self.assert_rejected(
            mutate,
            "SRC-005 must remain bound to exact lifecycle evidence issue #61",
        )

    def test_src005_cannot_drop_exact_lifecycle_subject_defect(self):
        def mutate(source, migration, parity):
            item = next(f for f in source["findings"] if f["id"] == "SRC-005")
            item["statement"] = (
                "Clean shutdown state remains available as lifecycle metadata for the previous generation."
            )
        self.assert_rejected(
            mutate,
            "SRC-005 must retain the generation-only lifecycle evidence defect",
        )

    def test_repair_002_cannot_leave_presentation_in_preparation_authority(self):
        def mutate(source, migration, parity):
            item = next(
                r for r in source["required_pre_extraction_repairs"]
                if r["id"] == "REPAIR-002"
            )
            item["statement"] = (
                "Use a recovery-native prepared-boot identity from locally observed kernel/boot filesystem facts rather than caller-selected authority values; make executable qualification plans non-deserializable capabilities distinct from serialized reports; and exact-subject bless every healthy prepared boot including AlreadyKnownGood."
            )
        self.assert_rejected(
            mutate,
            "REPAIR-002 must preserve local exact-subject provenance, execution capability separation, and per-boot qualification truth",
        )

    def test_repair_002_cannot_skip_already_known_good_exact_bless(self):
        def mutate(source, migration, parity):
            item = next(
                r for r in source["required_pre_extraction_repairs"]
                if r["id"] == "REPAIR-002"
            )
            item["statement"] = (
                "Remove morphology/presentation state from factual recovery preparation, qualification and LKG-commit authority; introduce a recovery-native prepared-boot identity whose Linux boot_id and booted generation derive from locally observed kernel/boot filesystem facts rather than caller-selected authority values; make executable qualification plans non-deserializable capabilities distinct from serialized reports; and keep morphology history outside recovery authority."
            )
        self.assert_rejected(
            mutate,
            "REPAIR-002 must preserve local exact-subject provenance, execution capability separation, and per-boot qualification truth",
        )

    def test_repair_002_cannot_trust_caller_selected_subject_identity(self):
        def mutate(source, migration, parity):
            item = next(
                r for r in source["required_pre_extraction_repairs"]
                if r["id"] == "REPAIR-002"
            )
            item["statement"] = (
                "Remove morphology/presentation state from factual recovery preparation, qualification and LKG-commit authority; introduce a recovery-native prepared-boot identity supplied by the host caller; make executable qualification plans non-deserializable capabilities distinct from serialized reports; exact-subject bless every healthy prepared boot including AlreadyKnownGood; and keep morphology history outside recovery authority."
            )
        self.assert_rejected(
            mutate,
            "REPAIR-002 must preserve local exact-subject provenance, execution capability separation, and per-boot qualification truth",
        )

    def test_repair_002_cannot_make_serialized_report_executable(self):
        def mutate(source, migration, parity):
            item = next(
                r for r in source["required_pre_extraction_repairs"]
                if r["id"] == "REPAIR-002"
            )
            item["statement"] = (
                "Remove morphology/presentation state from factual recovery preparation, qualification and LKG-commit authority; introduce a recovery-native prepared-boot identity whose Linux boot_id and booted generation derive from locally observed kernel/boot filesystem facts rather than caller-selected authority values; serialize executable plans as reports that can later be deserialized and replayed; exact-subject bless every healthy prepared boot including AlreadyKnownGood; and keep morphology history outside recovery authority."
            )
        self.assert_rejected(
            mutate,
            "REPAIR-002 must preserve local exact-subject provenance, execution capability separation, and per-boot qualification truth",
        )

    def test_repair_005_cannot_allow_caller_selected_lifecycle_subject(self):
        def mutate(source, migration, parity):
            item = next(
                r for r in source["required_pre_extraction_repairs"]
                if r["id"] == "REPAIR-005"
            )
            item["statement"] = (
                "Bind shutdown, reboot, suspend, and hibernate to a prepared boot subject supplied by the lifecycle caller and preserve legacy lifecycle markers."
            )
        self.assert_rejected(
            mutate,
            "REPAIR-005 must preserve exact lifecycle subject authority and legacy provenance",
        )

    def test_repair_005_cannot_fabricate_legacy_exact_subject(self):
        def mutate(source, migration, parity):
            item = next(
                r for r in source["required_pre_extraction_repairs"]
                if r["id"] == "REPAIR-005"
            )
            item["statement"] = (
                "Bind shutdown, reboot, suspend, and hibernate lifecycle evidence to the exact locally committed prepared boot subject; callers cannot mint or select boot_id or counter authority, and legacy lifecycle markers are upgraded to exact-subject provenance during migration."
            )
        self.assert_rejected(
            mutate,
            "REPAIR-005 must preserve exact lifecycle subject authority and legacy provenance",
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
