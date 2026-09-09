#!/usr/bin/env python3
"""Regression tests for independently selected trusted-main root receipt V2."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import trusted_main_root_receipt as v1  # noqa: E402
import trusted_main_root_receipt_v2 as v2  # noqa: E402

REPO = "Luminous-Dynamics/symthaea"
REPO_ID = 1136141775
REF = "refs/heads/main"
ROOT_SHA = "1" * 40
ROOT_TREE = "2" * 40
RULESET_ID = 4242
HARNESS_ID = "sha256:" + "a" * 64


def policy() -> dict:
    return {
        "schema": v1.POLICY_SCHEMA,
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "ruleset_name": "trusted-main-p0",
        "required_enforcement": "active",
        "allowed_bypass_actors": [],
        "required_rules": ["deletion", "non_fast_forward", "pull_request"],
        "pull_request_policy": {
            "required_approving_review_count_min": 0,
            "required_review_thread_resolution": True,
            "require_code_owner_review": False,
            "require_last_push_approval": False,
            "dismiss_stale_reviews_on_push": False,
            "allowed_merge_methods": ["merge", "rebase", "squash"],
        },
        "p0_required_status_checks": [],
        "non_claims": [
            "does not activate a trusted self-hosted runner",
            "does not prove a negative direct-push test has been performed",
            "does not prove bypass identities are globally impossible",
            "does not prove organization or enterprise rules are absent",
            "does not prove the ruleset is applied on GitHub",
            "does not qualify any source code or scientific result",
            "does not replace later P1 qualification admission policy",
        ],
    }


def _with_id(payload: dict, domain: bytes, field: str) -> dict:
    out = copy.deepcopy(payload)
    out[field] = v1._content_id(domain, payload)
    return out


def structural(p: dict) -> dict:
    payload = {
        "schema": v1.STRUCTURAL_SCHEMA,
        "policy_id": v1.policy_id(p),
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "ruleset_id": RULESET_ID,
        "ruleset_source_type": "Repository",
        "ruleset_source": REPO,
        "disposition": "P0StructurallySatisfied",
        "violations": [],
        "enforcement_claim": "structural-readback-only",
        "negative_push_test": "not-evaluated",
        "organization_rules": "not-enumerated",
        "scientific_authority": "none",
    }
    return _with_id(payload, v1.STRUCTURAL_DOMAIN, "verification_id")


def effective(p: dict) -> dict:
    payload = {
        "schema": v1.EFFECTIVE_SCHEMA,
        "policy_id": v1.policy_id(p),
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "ruleset_id": RULESET_ID,
        "observed_p0_rule_types": ["deletion", "non_fast_forward", "pull_request"],
        "disposition": "P0EffectiveRulesSatisfied",
        "violations": [],
        "enforcement_claim": "active-rule-projection-readback-only",
        "negative_push_test": "not-evaluated",
        "scientific_authority": "none",
    }
    return _with_id(payload, v1.EFFECTIVE_DOMAIN, "verification_id")


def root_subject() -> dict:
    return {
        "schema": v1.ROOT_SUBJECT_SCHEMA,
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "root_sha": ROOT_SHA,
        "root_tree": ROOT_TREE,
        "observation_basis": "github-commit-readback",
    }


def enforcement(p: dict, s: dict, e: dict) -> dict:
    payload = {
        "schema": v1.ENFORCEMENT_SCHEMA,
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "ruleset_id": RULESET_ID,
        "policy_id": v1.policy_id(p),
        "structural_verification_id": s["verification_id"],
        "effective_rules_verification_id": e["verification_id"],
        "root_sha": ROOT_SHA,
        "method": "administrative-verification",
        "disposition": "EnforcementSatisfied",
        "ordinary_direct_push": "blocked",
        "force_push": "blocked",
        "deletion": "blocked",
        "bypass_assurance": "policy-matched",
        "actor_or_scope": "repository administration readback",
        "rule_suite_ids": [],
        "recorded_at": "2026-09-09T20:30:00Z",
        "scientific_authority": "none",
    }
    return _with_id(payload, v1.ENFORCEMENT_DOMAIN, "evidence_id")


def fixture() -> tuple[dict, dict, dict, dict, dict]:
    p = policy()
    s = structural(p)
    e = effective(p)
    subject = root_subject()
    enf = enforcement(p, s, e)
    return p, s, e, subject, enf


def compose(**overrides):
    p, s, e, subject, enf = fixture()
    kwargs = {
        "policy": p,
        "structural_verification": s,
        "effective_rules_verification": e,
        "root_subject": subject,
        "enforcement_evidence": enf,
        "root_harness_identity": HARNESS_ID,
        "expected_protection_policy_id": v1.policy_id(p),
        "expected_protection_verification_id": s["verification_id"],
        "expected_effective_rules_verification_id": e["verification_id"],
        "expected_root_subject_id": v1.root_subject_id(subject),
        "expected_enforcement_evidence_id": enf["evidence_id"],
    }
    kwargs.update(overrides)
    return v2.compose_root_receipt_v2(**kwargs)


class RootReceiptV2Tests(unittest.TestCase):
    def test_independently_selected_bundle_admits_historical_root(self):
        result = compose()
        self.assertEqual(result["schema"], v2.ROOT_RECEIPT_SCHEMA)
        self.assertEqual(result["admission_disposition"], "AdmittedHistoricalRoot")
        self.assertEqual(result["evidence_selection_basis"], "trusted-phase-independent-expected-ids")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertEqual(result["receipt_attestation"], "none")
        self.assertEqual(result["scientific_authority"], "none")
        self.assertEqual(result["bootstrap_authority"], "none")
        self.assertEqual(result["self_hosted_runner_activation"], "not-authorized")
        self.assertRegex(result["trusted_evidence_selection_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(result["root_receipt_id"], r"^sha256:[0-9a-f]{64}$")

    def test_policy_id_must_be_selected_independently(self):
        with self.assertRaisesRegex(v2.RootReceiptV2Error, "protection_policy_id"):
            compose(expected_protection_policy_id="sha256:" + "0" * 64)

    def test_structural_id_must_be_selected_independently(self):
        with self.assertRaisesRegex(v2.RootReceiptV2Error, "protection_verification_id"):
            compose(expected_protection_verification_id="sha256:" + "0" * 64)

    def test_effective_id_must_be_selected_independently(self):
        with self.assertRaisesRegex(v2.RootReceiptV2Error, "effective_rules_verification_id"):
            compose(expected_effective_rules_verification_id="sha256:" + "0" * 64)

    def test_root_subject_id_must_be_selected_independently(self):
        with self.assertRaisesRegex(v2.RootReceiptV2Error, "root_subject_id"):
            compose(expected_root_subject_id="sha256:" + "0" * 64)

    def test_enforcement_id_must_be_selected_independently(self):
        with self.assertRaisesRegex(v2.RootReceiptV2Error, "enforcement_evidence_id"):
            compose(expected_enforcement_evidence_id="sha256:" + "0" * 64)

    def test_self_consistent_alternate_structural_bundle_cannot_replace_trusted_selection(self):
        p, s, e, subject, enf = fixture()
        payload = dict(s)
        payload.pop("verification_id")
        payload["organization_rules"] = "enumerated-separately"
        alternate_s = _with_id(payload, v1.STRUCTURAL_DOMAIN, "verification_id")
        alternate_enf = enforcement(p, alternate_s, e)
        with self.assertRaisesRegex(v2.RootReceiptV2Error, "protection_verification_id"):
            v2.compose_root_receipt_v2(
                policy=p,
                structural_verification=alternate_s,
                effective_rules_verification=e,
                root_subject=subject,
                enforcement_evidence=alternate_enf,
                root_harness_identity=HARNESS_ID,
                expected_protection_policy_id=v1.policy_id(p),
                expected_protection_verification_id=s["verification_id"],
                expected_effective_rules_verification_id=e["verification_id"],
                expected_root_subject_id=v1.root_subject_id(subject),
                expected_enforcement_evidence_id=enf["evidence_id"],
            )

    def test_self_consistent_alternate_enforcement_cannot_replace_trusted_selection(self):
        p, s, e, subject, enf = fixture()
        payload = dict(enf)
        payload.pop("evidence_id")
        payload["actor_or_scope"] = "alternate administrative source"
        alternate = _with_id(payload, v1.ENFORCEMENT_DOMAIN, "evidence_id")
        with self.assertRaisesRegex(v2.RootReceiptV2Error, "enforcement_evidence_id"):
            v2.compose_root_receipt_v2(
                policy=p,
                structural_verification=s,
                effective_rules_verification=e,
                root_subject=subject,
                enforcement_evidence=alternate,
                root_harness_identity=HARNESS_ID,
                expected_protection_policy_id=v1.policy_id(p),
                expected_protection_verification_id=s["verification_id"],
                expected_effective_rules_verification_id=e["verification_id"],
                expected_root_subject_id=v1.root_subject_id(subject),
                expected_enforcement_evidence_id=enf["evidence_id"],
            )

    def test_v1_non_admitted_bundle_cannot_be_upgraded_by_expected_ids(self):
        p, s, e, subject, enf = fixture()
        payload = dict(enf)
        payload.pop("evidence_id")
        payload["disposition"] = "EnforcementInconclusive"
        payload["ordinary_direct_push"] = "not-evaluated"
        payload["force_push"] = "not-evaluated"
        payload["deletion"] = "not-evaluated"
        payload["bypass_assurance"] = "not-evaluated"
        inconclusive = _with_id(payload, v1.ENFORCEMENT_DOMAIN, "evidence_id")
        with self.assertRaisesRegex(v2.RootReceiptV2Error, "V1 AdmittedHistoricalRoot"):
            v2.compose_root_receipt_v2(
                policy=p,
                structural_verification=s,
                effective_rules_verification=e,
                root_subject=subject,
                enforcement_evidence=inconclusive,
                root_harness_identity=HARNESS_ID,
                expected_protection_policy_id=v1.policy_id(p),
                expected_protection_verification_id=s["verification_id"],
                expected_effective_rules_verification_id=e["verification_id"],
                expected_root_subject_id=v1.root_subject_id(subject),
                expected_enforcement_evidence_id=inconclusive["evidence_id"],
            )

    def test_selection_id_changes_when_any_selected_evidence_changes(self):
        base = compose()
        p, s, e, subject, enf = fixture()
        changed_subject = dict(subject)
        changed_subject["root_tree"] = "9" * 40
        changed_enf_payload = dict(enf)
        changed_enf_payload.pop("evidence_id")
        changed_enf_payload["root_sha"] = changed_subject["root_sha"]
        changed_enf = _with_id(changed_enf_payload, v1.ENFORCEMENT_DOMAIN, "evidence_id")
        changed = v2.compose_root_receipt_v2(
            policy=p,
            structural_verification=s,
            effective_rules_verification=e,
            root_subject=changed_subject,
            enforcement_evidence=changed_enf,
            root_harness_identity=HARNESS_ID,
            expected_protection_policy_id=v1.policy_id(p),
            expected_protection_verification_id=s["verification_id"],
            expected_effective_rules_verification_id=e["verification_id"],
            expected_root_subject_id=v1.root_subject_id(changed_subject),
            expected_enforcement_evidence_id=changed_enf["evidence_id"],
        )
        self.assertNotEqual(base["trusted_evidence_selection_id"], changed["trusted_evidence_selection_id"])
        self.assertNotEqual(base["root_receipt_id"], changed["root_receipt_id"])

    def test_v1_receipt_id_is_retained_as_provenance_not_authority(self):
        p, s, e, subject, enf = fixture()
        expected_v1 = v1.compose_root_receipt(
            policy=p,
            structural_verification=s,
            effective_rules_verification=e,
            root_subject=subject,
            enforcement_evidence=enf,
            root_harness_identity=HARNESS_ID,
        )["root_receipt_id"]
        result = compose()
        self.assertEqual(result["v1_root_receipt_id"], expected_v1)
        self.assertNotEqual(result["root_receipt_id"], expected_v1)

    def test_invalid_expected_id_shape_fails_before_selection(self):
        with self.assertRaisesRegex(v2.RootReceiptV2Error, "sha256"):
            compose(expected_enforcement_evidence_id="enforcement-v1")

    def test_receipt_identity_is_deterministic(self):
        self.assertEqual(compose()["root_receipt_id"], compose()["root_receipt_id"])


if __name__ == "__main__":
    unittest.main()
