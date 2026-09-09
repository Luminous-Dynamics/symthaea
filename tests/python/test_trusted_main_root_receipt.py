#!/usr/bin/env python3
"""Regression tests for immutable trusted-main root receipt composition."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import trusted_main_root_receipt as root  # noqa: E402

REPO = "Luminous-Dynamics/symthaea"
REPO_ID = 1136141775
REF = "refs/heads/main"
ROOT_SHA = "1" * 40
ROOT_TREE = "2" * 40
RULESET_ID = 4242
HARNESS_ID = "sha256:" + "a" * 64


def policy(*, approvals: int = 0, bypass=None) -> dict:
    return {
        "schema": root.POLICY_SCHEMA,
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "ruleset_name": "trusted-main-p0",
        "required_enforcement": "active",
        "allowed_bypass_actors": [] if bypass is None else bypass,
        "required_rules": ["deletion", "non_fast_forward", "pull_request"],
        "pull_request_policy": {
            "required_approving_review_count_min": approvals,
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
    out[field] = root._content_id(domain, payload)
    return out


def structural(p=None, *, disposition="P0StructurallySatisfied", violations=None) -> dict:
    p = p or policy()
    payload = {
        "schema": root.STRUCTURAL_SCHEMA,
        "policy_id": root.policy_id(p),
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "ruleset_id": RULESET_ID,
        "ruleset_source_type": "Repository",
        "ruleset_source": REPO,
        "disposition": disposition,
        "violations": [] if violations is None else violations,
        "enforcement_claim": "structural-readback-only",
        "negative_push_test": "not-evaluated",
        "organization_rules": "not-enumerated",
        "scientific_authority": "none",
    }
    return _with_id(payload, root.STRUCTURAL_DOMAIN, "verification_id")


def effective(p=None, *, disposition="P0EffectiveRulesSatisfied", violations=None) -> dict:
    p = p or policy()
    payload = {
        "schema": root.EFFECTIVE_SCHEMA,
        "policy_id": root.policy_id(p),
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "ruleset_id": RULESET_ID,
        "observed_p0_rule_types": ["deletion", "non_fast_forward", "pull_request"],
        "disposition": disposition,
        "violations": [] if violations is None else violations,
        "enforcement_claim": "active-rule-projection-readback-only",
        "negative_push_test": "not-evaluated",
        "scientific_authority": "none",
    }
    return _with_id(payload, root.EFFECTIVE_DOMAIN, "verification_id")


def root_subject(*, sha=ROOT_SHA, tree=ROOT_TREE) -> dict:
    return {
        "schema": root.ROOT_SUBJECT_SCHEMA,
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "root_sha": sha,
        "root_tree": tree,
        "observation_basis": "github-commit-readback",
    }


def enforcement(p=None, s=None, e=None, *, disposition="EnforcementSatisfied", sha=ROOT_SHA) -> dict:
    p = p or policy()
    s = s or structural(p)
    e = e or effective(p)
    satisfied = disposition == "EnforcementSatisfied"
    payload = {
        "schema": root.ENFORCEMENT_SCHEMA,
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": REF,
        "ruleset_id": RULESET_ID,
        "policy_id": root.policy_id(p),
        "structural_verification_id": s["verification_id"],
        "effective_rules_verification_id": e["verification_id"],
        "root_sha": sha,
        "method": "administrative-verification",
        "disposition": disposition,
        "ordinary_direct_push": "blocked" if satisfied else "not-evaluated",
        "force_push": "blocked" if satisfied else "not-evaluated",
        "deletion": "blocked" if satisfied else "not-evaluated",
        "bypass_assurance": "policy-matched" if satisfied else "not-evaluated",
        "actor_or_scope": "repository administration readback",
        "rule_suite_ids": [],
        "recorded_at": "2026-09-09T19:30:00Z",
        "scientific_authority": "none",
    }
    return _with_id(payload, root.ENFORCEMENT_DOMAIN, "evidence_id")


def compose(*, p=None, s=None, e=None, subject=None, enf="default", harness=HARNESS_ID):
    p = p or policy()
    s = s or structural(p)
    e = e or effective(p)
    subject = subject or root_subject()
    evidence = enforcement(p, s, e) if enf == "default" else enf
    return root.compose_root_receipt(
        policy=p,
        structural_verification=s,
        effective_rules_verification=e,
        root_subject=subject,
        enforcement_evidence=evidence,
        root_harness_identity=harness,
    )


class RootReceiptTests(unittest.TestCase):
    def test_complete_evidence_admits_historical_root_only(self):
        result = compose()
        self.assertEqual(result["admission_disposition"], "AdmittedHistoricalRoot")
        self.assertEqual(result["non_admission_reasons"], [])
        self.assertEqual(result["historical_scope"], "exact-root-only")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertEqual(result["receipt_attestation"], "none")
        self.assertEqual(result["scientific_authority"], "none")
        self.assertEqual(result["self_hosted_runner_activation"], "not-authorized")
        self.assertEqual(result["bootstrap_authority"], "none")
        self.assertRegex(result["root_receipt_id"], r"^sha256:[0-9a-f]{64}$")

    def test_missing_enforcement_cannot_admit(self):
        result = compose(enf=None)
        self.assertEqual(result["admission_disposition"], "PendingEnforcementEvidence")
        self.assertEqual(result["enforcement_evidence_id"], None)
        self.assertEqual(result["non_admission_reasons"], ["EnforcementEvidenceMissing"])

    def test_inconclusive_enforcement_cannot_admit(self):
        p = policy()
        s = structural(p)
        e = effective(p)
        result = compose(p=p, s=s, e=e, enf=enforcement(p, s, e, disposition="EnforcementInconclusive"))
        self.assertEqual(result["admission_disposition"], "EnforcementInconclusive")
        self.assertIn("EnforcementEvidenceInconclusive", result["non_admission_reasons"])

    def test_structural_failure_cannot_admit(self):
        p = policy()
        s = structural(p, disposition="P0Rejected", violations=["BranchNotReportedProtected"])
        e = effective(p)
        result = compose(p=p, s=s, e=e, enf=enforcement(p, s, e))
        self.assertEqual(result["admission_disposition"], "ProtectionEvidenceNotSatisfied")
        self.assertIn("StructuralProtectionNotSatisfied", result["non_admission_reasons"])

    def test_effective_rule_failure_cannot_admit(self):
        p = policy()
        s = structural(p)
        e = effective(p, disposition="P0EffectiveRulesRejected", violations=["EffectiveRuleCount:deletion"])
        result = compose(p=p, s=s, e=e, enf=enforcement(p, s, e))
        self.assertEqual(result["admission_disposition"], "ProtectionEvidenceNotSatisfied")
        self.assertIn("EffectiveRulesNotSatisfied", result["non_admission_reasons"])

    def test_structural_content_id_forgery_rejects(self):
        s = structural()
        s["verification_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(root.RootReceiptError, "content ID mismatch"):
            compose(s=s)

    def test_effective_content_id_forgery_rejects(self):
        e = effective()
        e["verification_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(root.RootReceiptError, "content ID mismatch"):
            compose(e=e)

    def test_enforcement_content_id_forgery_rejects(self):
        evidence = enforcement()
        evidence["evidence_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(root.RootReceiptError, "content ID mismatch"):
            compose(enf=evidence)

    def test_policy_id_mismatch_rejects(self):
        s = structural()
        payload = dict(s)
        payload.pop("verification_id")
        payload["policy_id"] = "sha256:" + "0" * 64
        s = _with_id(payload, root.STRUCTURAL_DOMAIN, "verification_id")
        with self.assertRaisesRegex(root.RootReceiptError, "policy ID mismatch"):
            compose(s=s)

    def test_ruleset_id_mismatch_rejects(self):
        e = effective()
        payload = dict(e)
        payload.pop("verification_id")
        payload["ruleset_id"] = RULESET_ID + 1
        e = _with_id(payload, root.EFFECTIVE_DOMAIN, "verification_id")
        with self.assertRaisesRegex(root.RootReceiptError, "ruleset ID mismatch"):
            compose(e=e)

    def test_enforcement_root_sha_mismatch_rejects(self):
        with self.assertRaisesRegex(root.RootReceiptError, "root SHA mismatch"):
            compose(enf=enforcement(sha="9" * 40))

    def test_root_change_changes_receipt_identity(self):
        first = compose()
        second = compose(subject=root_subject(sha="8" * 40), enf=enforcement(sha="8" * 40))
        self.assertNotEqual(first["root_subject_id"], second["root_subject_id"])
        self.assertNotEqual(first["root_receipt_id"], second["root_receipt_id"])

    def test_harness_identity_changes_receipt_identity(self):
        first = compose()
        second = compose(harness="sha256:" + "b" * 64)
        self.assertNotEqual(first["root_receipt_id"], second["root_receipt_id"])

    def test_prebootstrap_harness_is_explicit_and_non_authorizing(self):
        result = compose(harness="none-pre-bootstrap")
        self.assertEqual(result["root_harness_identity"], "none-pre-bootstrap")
        self.assertEqual(result["bootstrap_authority"], "none")

    def test_invalid_harness_identity_rejects(self):
        with self.assertRaisesRegex(root.RootReceiptError, "sha256"):
            compose(harness="recipe-v1")

    def test_zero_approval_policy_states_no_required_approval(self):
        result = compose()
        self.assertEqual(result["review_assurance"], "pr-mediated-no-required-approval")

    def test_required_approval_policy_changes_assurance_and_identity(self):
        p = policy(approvals=1)
        s = structural(p)
        e = effective(p)
        result = compose(p=p, s=s, e=e, enf=enforcement(p, s, e))
        self.assertEqual(result["review_assurance"], "pr-mediated-required-approval")
        self.assertNotEqual(result["protection_policy_id"], compose()["protection_policy_id"])

    def test_bypass_policy_is_separate_content_identity(self):
        base = compose()
        bypass = [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "pull_request"}]
        p = policy(bypass=bypass)
        s = structural(p)
        e = effective(p)
        changed = compose(p=p, s=s, e=e, enf=enforcement(p, s, e))
        self.assertNotEqual(base["bypass_policy_id"], changed["bypass_policy_id"])
        self.assertNotEqual(base["root_receipt_id"], changed["root_receipt_id"])

    def test_enforcement_satisfied_requires_all_protections_blocked(self):
        evidence = enforcement()
        payload = dict(evidence)
        payload.pop("evidence_id")
        payload["force_push"] = "not-evaluated"
        payload["evidence_id"] = root._content_id(root.ENFORCEMENT_DOMAIN, payload)
        with self.assertRaisesRegex(root.RootReceiptError, "force_push"):
            compose(enf=payload)

    def test_enforcement_must_bind_structural_verification(self):
        evidence = enforcement()
        payload = dict(evidence)
        payload.pop("evidence_id")
        payload["structural_verification_id"] = "sha256:" + "c" * 64
        payload["evidence_id"] = root._content_id(root.ENFORCEMENT_DOMAIN, payload)
        with self.assertRaisesRegex(root.RootReceiptError, "structural verification ID mismatch"):
            compose(enf=payload)

    def test_unknown_policy_field_rejects(self):
        p = policy()
        p["future_authority"] = True
        with self.assertRaisesRegex(root.RootReceiptError, "unknown fields"):
            compose(p=p)

    def test_root_subject_requires_provider_readback_basis(self):
        subject = root_subject()
        subject["observation_basis"] = "operator-typed"
        with self.assertRaisesRegex(root.RootReceiptError, "github-commit-readback"):
            compose(subject=subject)

    def test_receipt_identity_is_deterministic(self):
        self.assertEqual(compose()["root_receipt_id"], compose()["root_receipt_id"])


if __name__ == "__main__":
    unittest.main()
