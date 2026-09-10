#!/usr/bin/env python3
"""Regression tests for exact V2 -> V3 trusted-main enforcement lineage."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import trusted_main_enforcement_evidence_v2 as v2  # noqa: E402
import trusted_main_enforcement_evidence_v3 as v3  # noqa: E402
import trusted_main_enforcement_lineage as lineage  # noqa: E402
import test_trusted_main_enforcement_evidence_v2 as fx  # noqa: E402
import test_trusted_main_enforcement_evidence_v3 as v3fx  # noqa: E402


def evidence_pair() -> tuple[dict, dict]:
    base = v2.derive_enforcement_evidence(
        policy=fx.policy(),
        structural_verification=fx.structural(),
        effective_rules_verification=fx.effective(),
        root_subject=fx.root_subject(),
        direct_update_rule_suite=fx.direct_suite(),
        force_push_rule_suite=fx.force_suite(),
        deletion_rule_suite=fx.deletion_suite(),
    )
    strict = v3.derive_enforcement_evidence_v3(
        policy=fx.policy(),
        structural_verification=fx.structural(),
        effective_rules_verification=fx.effective(),
        root_subject=fx.root_subject(),
        direct_update_rule_suite=fx.direct_suite(),
        force_push_rule_suite=fx.force_suite(),
        deletion_rule_suite=fx.deletion_suite(),
        **v3fx.expected_kwargs(),
    )
    return base, strict


def rehash_v2(value: dict) -> None:
    payload = dict(value)
    payload.pop("evidence_id", None)
    value["evidence_id"] = v2._content_id(v2.DOMAIN, payload)


def rederive_v3(value: dict) -> None:
    selection_payload = {
        "schema": v3.SELECTION_SCHEMA,
        "structural_verification_id": value["structural_verification_id"],
        "effective_rules_verification_id": value["effective_rules_verification_id"],
        "root_subject_id": value["root_subject_id"],
        "rule_suite_ids": value["selected_rule_suite_ids"],
        "rule_suite_observation_ids": value["selected_rule_suite_observation_ids"],
        "actor_id": value["selected_actor_id"],
    }
    value["trusted_enforcement_selection_id"] = v3._content_id(
        v3.SELECTION_DOMAIN, selection_payload
    )
    payload = dict(value)
    payload.pop("evidence_id", None)
    value["evidence_id"] = v3._content_id(v3.DOMAIN, payload)


def derive(
    *,
    v2_evidence: dict | None = None,
    v3_evidence: dict | None = None,
    expected_v2_evidence_id: str | None = None,
    expected_v3_evidence_id: str | None = None,
) -> dict:
    default_v2, default_v3 = evidence_pair()
    v2_evidence = default_v2 if v2_evidence is None else v2_evidence
    v3_evidence = default_v3 if v3_evidence is None else v3_evidence
    return lineage.derive_enforcement_lineage(
        policy=fx.policy(),
        enforcement_evidence_v2=v2_evidence,
        enforcement_evidence_v3=v3_evidence,
        expected_v2_evidence_id=(
            v2_evidence["evidence_id"]
            if expected_v2_evidence_id is None
            else expected_v2_evidence_id
        ),
        expected_v3_evidence_id=(
            v3_evidence["evidence_id"]
            if expected_v3_evidence_id is None
            else expected_v3_evidence_id
        ),
    )


class EnforcementLineageTests(unittest.TestCase):
    def test_exact_positive_v2_v3_lineage_binds_narrowly(self):
        result = derive()
        self.assertEqual(result["schema"], lineage.SCHEMA)
        self.assertEqual(result["disposition"], "EnforcementV3BoundToExactPositiveV2")
        self.assertEqual(
            result["lineage_basis"],
            "independently-selected-v2-v3-content-and-exact-normalized-observations",
        )
        self.assertEqual(result["evidence_authority"], "server-rule-evaluation-readback-only")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertEqual(result["receipt_attestation"], "none")
        self.assertEqual(result["scientific_authority"], "none")
        self.assertRegex(result["lineage_id"], r"^sha256:[0-9a-f]{64}$")

    def test_lineage_identity_is_deterministic(self):
        self.assertEqual(derive()["lineage_id"], derive()["lineage_id"])

    def test_v2_selector_is_independent(self):
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "V2 evidence bytes"):
            derive(expected_v2_evidence_id="sha256:" + "0" * 64)

    def test_v3_selector_is_independent(self):
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "V3 evidence bytes"):
            derive(expected_v3_evidence_id="sha256:" + "0" * 64)

    def test_v3_cannot_repoint_to_alternate_v2_even_when_rehashed(self):
        base, strict = evidence_pair()
        strict = copy.deepcopy(strict)
        strict["v2_evidence_id"] = "sha256:" + "1" * 64
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "does not bind supplied exact V2"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_rehashed_v2_authority_inflation_rejects(self):
        base, strict = evidence_pair()
        base = copy.deepcopy(base)
        base["chronology_authority"] = "externally-anchored"
        rehash_v2(base)
        strict = copy.deepcopy(strict)
        strict["v2_evidence_id"] = base["evidence_id"]
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "canonical authority ceiling"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_rehashed_v3_authority_inflation_rejects(self):
        base, strict = evidence_pair()
        strict = copy.deepcopy(strict)
        strict["current_admission"] = "admitted"
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "revalidation failed"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_rehashed_v2_observation_drift_cannot_preserve_lineage(self):
        base, strict = evidence_pair()
        base = copy.deepcopy(base)
        base["operation_observations"]["force_push"]["actor_name"] = "Different Actor"
        rehash_v2(base)
        strict = copy.deepcopy(strict)
        strict["v2_evidence_id"] = base["evidence_id"]
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "force_push.*exact V2 observation"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_v3_selected_observation_cannot_drift_even_with_rederived_ids(self):
        base, strict = evidence_pair()
        strict = copy.deepcopy(strict)
        strict["selected_rule_suite_observation_ids"]["force_push"] = "sha256:" + "2" * 64
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "force_push.*exact V2 observation"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_v3_selected_suite_cannot_drift_even_with_rederived_ids(self):
        base, strict = evidence_pair()
        strict = copy.deepcopy(strict)
        strict["selected_rule_suite_ids"]["force_push"] = 777
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "force_push.*exact V2 suite"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_v3_selected_actor_cannot_drift_even_with_rederived_ids(self):
        base, strict = evidence_pair()
        strict = copy.deepcopy(strict)
        strict["selected_actor_id"] = 777
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "selected actor"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_v2_duplicate_suite_ids_reject_even_when_rehashed(self):
        base, strict = evidence_pair()
        base = copy.deepcopy(base)
        base["operation_observations"]["force_push"]["rule_suite_id"] = (
            base["operation_observations"]["ordinary_direct_update"]["rule_suite_id"]
        )
        rehash_v2(base)
        strict = copy.deepcopy(strict)
        strict["v2_evidence_id"] = base["evidence_id"]
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "rule-suite IDs must be distinct"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_positive_v2_cannot_hide_missing_operation_after_rehash(self):
        base, strict = evidence_pair()
        base = copy.deepcopy(base)
        base["missing_operations"] = ["deletion"]
        rehash_v2(base)
        strict = copy.deepcopy(strict)
        strict["v2_evidence_id"] = base["evidence_id"]
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "cannot contain missing operations"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_positive_v2_cannot_hide_violation_after_rehash(self):
        base, strict = evidence_pair()
        base = copy.deepcopy(base)
        base["violations"] = ["SyntheticViolation"]
        rehash_v2(base)
        strict = copy.deepcopy(strict)
        strict["v2_evidence_id"] = base["evidence_id"]
        rederive_v3(strict)
        with self.assertRaisesRegex(lineage.EnforcementLineageError, "cannot contain missing operations or violations"):
            derive(v2_evidence=base, v3_evidence=strict)

    def test_changed_exact_v2_changes_lineage_identity(self):
        first = derive()
        base, strict = evidence_pair()
        base = copy.deepcopy(base)
        strict = copy.deepcopy(strict)
        base["operation_observations"]["ordinary_direct_update"]["actor_name"] = "Actor Two"
        rehash_v2(base)
        new_observation_id = v3.rule_suite_observation_id(
            base["operation_observations"]["ordinary_direct_update"]
        )
        strict["v2_evidence_id"] = base["evidence_id"]
        strict["selected_rule_suite_observation_ids"]["ordinary_direct_update"] = new_observation_id
        rederive_v3(strict)
        second = derive(v2_evidence=base, v3_evidence=strict)
        self.assertNotEqual(first["v2_evidence_id"], second["v2_evidence_id"])
        self.assertNotEqual(first["lineage_id"], second["lineage_id"])


if __name__ == "__main__":
    unittest.main()
