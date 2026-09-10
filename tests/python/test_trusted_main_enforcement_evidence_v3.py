#!/usr/bin/env python3
"""Regression tests for independently selected enforcement evidence V3."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import trusted_main_enforcement_evidence_v2 as v2  # noqa: E402
import trusted_main_enforcement_evidence_v3 as v3  # noqa: E402
import trusted_main_ruleset as p0  # noqa: E402
import test_trusted_main_enforcement_evidence_v2 as fx  # noqa: E402

ACTOR_ID = 215346314


def observation_ids(direct: dict, force: dict, deletion: dict) -> dict[str, str]:
    base = v2.derive_enforcement_evidence(
        policy=fx.policy(),
        structural_verification=fx.structural(),
        effective_rules_verification=fx.effective(),
        root_subject=fx.root_subject(),
        direct_update_rule_suite=direct,
        force_push_rule_suite=force,
        deletion_rule_suite=deletion,
    )
    if base["disposition"] != "EnforcementBehaviorallyCorroborated":
        raise AssertionError("fixture must produce V2 positive observations")
    return {
        operation: v3.rule_suite_observation_id(observation)
        for operation, observation in base["operation_observations"].items()
    }


def expected_kwargs() -> dict:
    s = fx.structural()
    e = fx.effective()
    root = fx.root_subject()
    ids = observation_ids(fx.direct_suite(), fx.force_suite(), fx.deletion_suite())
    return {
        "expected_structural_verification_id": s["verification_id"],
        "expected_effective_rules_verification_id": e["verification_id"],
        "expected_root_subject_id": v2._root_subject_id(v2._root_subject(root, p0.normalize_policy(fx.policy()))),
        "expected_direct_update_rule_suite_id": 101,
        "expected_force_push_rule_suite_id": 102,
        "expected_deletion_rule_suite_id": 103,
        "expected_direct_update_observation_id": ids["ordinary_direct_update"],
        "expected_force_push_observation_id": ids["force_push"],
        "expected_deletion_observation_id": ids["deletion"],
        "expected_actor_id": ACTOR_ID,
    }


def derive(**overrides):
    values = {
        "policy": fx.policy(),
        "structural_verification": fx.structural(),
        "effective_rules_verification": fx.effective(),
        "root_subject": fx.root_subject(),
        "direct_update_rule_suite": fx.direct_suite(),
        "force_push_rule_suite": fx.force_suite(),
        "deletion_rule_suite": fx.deletion_suite(),
        **expected_kwargs(),
    }
    values.update(overrides)
    return v3.derive_enforcement_evidence_v3(**values)


class EnforcementEvidenceV3Tests(unittest.TestCase):
    def test_independently_selected_exact_attempts_are_positive(self):
        result = derive()
        self.assertEqual(result["schema"], v3.SCHEMA)
        self.assertEqual(result["disposition"], "EnforcementBehaviorallyCorroborated")
        self.assertEqual(result["selected_actor_id"], ACTOR_ID)
        self.assertEqual(
            result["selected_rule_suite_ids"],
            {"deletion": 103, "force_push": 102, "ordinary_direct_update": 101},
        )
        self.assertEqual(
            result["evidence_selection_basis"],
            "trusted-phase-independent-expected-ids-and-observation-content",
        )
        self.assertEqual(result["receipt_attestation"], "none")
        self.assertEqual(result["scientific_authority"], "none")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertRegex(result["trusted_enforcement_selection_id"], r"^sha256:[0-9a-f]{64}$")
        for identity in result["selected_rule_suite_observation_ids"].values():
            self.assertRegex(identity, r"^sha256:[0-9a-f]{64}$")

    def test_foreign_structural_schema_rejects_before_v2_positive_path(self):
        s = fx.structural()
        s["schema"] = "foreign.structural.v9"
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "structural_verification.schema"):
            derive(structural_verification=s)

    def test_foreign_effective_schema_rejects_before_v2_positive_path(self):
        e = fx.effective()
        e["schema"] = "foreign.effective.v9"
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "effective_rules_verification.schema"):
            derive(effective_rules_verification=e)

    def test_foreign_root_schema_rejects(self):
        root = fx.root_subject()
        root["schema"] = "foreign.root.v9"
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "root_subject.schema"):
            derive(root_subject=root)

    def test_structural_selector_is_independent(self):
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "structural verification bytes"):
            derive(expected_structural_verification_id="sha256:" + "0" * 64)

    def test_effective_selector_is_independent(self):
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "effective-rule verification bytes"):
            derive(expected_effective_rules_verification_id="sha256:" + "0" * 64)

    def test_root_selector_is_independent(self):
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "root subject bytes"):
            derive(expected_root_subject_id="sha256:" + "0" * 64)

    def test_each_suite_selector_is_independent(self):
        for field, operation in (
            ("expected_direct_update_rule_suite_id", "ordinary_direct_update"),
            ("expected_force_push_rule_suite_id", "force_push"),
            ("expected_deletion_rule_suite_id", "deletion"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, operation):
                    derive(**{field: 999})

    def test_expected_suite_ids_must_be_distinct(self):
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "must be distinct"):
            derive(expected_force_push_rule_suite_id=101)

    def test_selected_actor_must_match_all_provider_records(self):
        wrong = fx.force_suite(actor_id=999)
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "force_push.*actor ID"):
            derive(force_push_rule_suite=wrong)

    def test_actor_selector_is_independent(self):
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "ordinary_direct_update.*actor ID"):
            derive(expected_actor_id=999)

    def test_v2_rejection_cannot_be_upgraded_by_valid_selectors(self):
        bad = fx.direct_suite(result="pass")
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "V3 requires V2"):
            derive(direct_update_rule_suite=bad)

    def test_alternate_numeric_suite_id_cannot_replace_selected_attempt(self):
        alternate = fx.direct_suite()
        alternate["id"] = 777
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "ordinary_direct_update"):
            derive(direct_update_rule_suite=alternate)

    def test_same_suite_and_actor_ids_with_changed_provider_time_rejects_by_content(self):
        forged = fx.direct_suite(pushed_at="2026-09-09T20:45:01Z")
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "normalized rule-suite observation"):
            derive(direct_update_rule_suite=forged)

    def test_same_suite_and_actor_ids_with_changed_attempted_sha_rejects_by_content(self):
        forged = fx.direct_suite(after_sha="8" * 40)
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "normalized rule-suite observation"):
            derive(direct_update_rule_suite=forged)

    def test_same_suite_and_actor_ids_with_changed_actor_name_rejects_by_content(self):
        forged = fx.direct_suite(actor_name="Impostor")
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "normalized rule-suite observation"):
            derive(direct_update_rule_suite=forged)

    def test_observation_selector_itself_is_independent(self):
        with self.assertRaisesRegex(v3.EnforcementEvidenceV3Error, "normalized rule-suite observation"):
            derive(expected_direct_update_observation_id="sha256:" + "0" * 64)

    def test_evidence_identity_is_deterministic(self):
        self.assertEqual(derive()["evidence_id"], derive()["evidence_id"])

    def test_selection_identity_changes_with_independently_selected_actor_and_content(self):
        first = derive()
        direct = fx.direct_suite(actor_id=314)
        force = fx.force_suite(actor_id=314)
        deletion = fx.deletion_suite(actor_id=314)
        ids = observation_ids(direct, force, deletion)
        second = derive(
            direct_update_rule_suite=direct,
            force_push_rule_suite=force,
            deletion_rule_suite=deletion,
            expected_actor_id=314,
            expected_direct_update_observation_id=ids["ordinary_direct_update"],
            expected_force_push_observation_id=ids["force_push"],
            expected_deletion_observation_id=ids["deletion"],
        )
        self.assertNotEqual(
            first["trusted_enforcement_selection_id"],
            second["trusted_enforcement_selection_id"],
        )
        self.assertNotEqual(first["evidence_id"], second["evidence_id"])


if __name__ == "__main__":
    unittest.main()
