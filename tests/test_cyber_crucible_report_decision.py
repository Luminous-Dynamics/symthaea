"""Tests for exact report-to-decision binding."""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.verify_cyber_crucible_decision_receipt import (
    POLICY_VERSION,
    SCHEMA_VERSION,
    ReplayLedger,
    canonical_json_digest,
    canonical_signed_bytes,
)
from scripts.package_cyber_crucible_verifier import build_bundle
from scripts.verify_cyber_crucible_report_decision import (
    ReportDecisionBindingError,
    validate_input_receipt_binding,
    validate_report_binding,
    verify_report_decision,
)
from scripts.verify_cyber_crucible_verifier_provenance import (
    GITHUB_OIDC_ISSUER,
    IN_TOTO_STATEMENT_V1,
    POLICY_VERSION as PROVENANCE_POLICY_VERSION,
    SLSA_PROVENANCE_V1,
    json_digest,
)


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class ReportBindingTests(unittest.TestCase):
    def setUp(self):
        self.provenance_temp = tempfile.TemporaryDirectory(prefix="report decision provenance ")
        self.addCleanup(self.provenance_temp.cleanup)
        self.provenance_root = Path(self.provenance_temp.name)
        self.verifier_artifact = self.provenance_root / "cyber-crucible-verifier-v1.tar.gz"
        self.provenance_source_commit = "a" * 40
        repository_root = Path(__file__).resolve().parents[1]
        build_bundle(repository_root, self.verifier_artifact, self.provenance_source_commit)
        self.verifier_artifact_sha256 = hashlib.sha256(self.verifier_artifact.read_bytes()).hexdigest()
        self.fake_gh = self.provenance_root / "operator-pinned-gh"
        self.fake_gh.write_bytes(b"test-only fake gh binary")
        self.fake_gh.chmod(0o700)
        self.fake_gh_sha256 = hashlib.sha256(self.fake_gh.read_bytes()).hexdigest()

        self.report = {
            "schema_version": "cyber-crucible-diagnostic-loop.v1",
            "lane": "public_calibration",
            "non_authorizing": True,
            "run": {
                "run_id": "attempt-0001",
                "subject_id": "symthaea-test-build",
                "scenario": {
                    "scenario_id": "scenario-12",
                    "revision": 3,
                    "scenario_digest": "1" * 64,
                },
            },
        }
        self.input_receipt = {
            "payload": {
                "campaign_id": "campaign-fixture-1",
                "subject_id": "symthaea-test-build",
                "attempt_id": "attempt-0001",
                "scenario_id": "scenario-12",
                "scenario_revision": 3,
                "scenario_digest": "1" * 64,
            },
            "signature": "fixture-signature",
        }
        self.context = {
            "campaign_id": "campaign-fixture-1",
            "subject_id": "symthaea-test-build",
            "attempt_id": "attempt-0001",
            "scenario_id": "scenario-12",
            "scenario_revision": 3,
            "scenario_digest": "1" * 64,
            "input_receipt_digest": canonical_json_digest(self.input_receipt),
            "report_digest": canonical_json_digest(self.report),
            "evaluator_policy_digest": "3" * 64,
            "verifier_build_digest": self.verifier_artifact_sha256,
            "decision_nonce": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
        }
        source_ref = "refs/tags/verifier-fixture-v1"
        source_commit = self.provenance_source_commit
        workflow_path = ".github/workflows/cyber-crucible-verifier-release.yml"
        self.provenance_policy = {
            "schema_version": PROVENANCE_POLICY_VERSION,
            "policy_id": "report-consumer-provenance-fixture-v1",
            "repository": "Luminous-Dynamics/symthaea",
            "source_ref": source_ref,
            "source_commit": source_commit,
            "workflow_path": workflow_path,
            "signer_repository": "Luminous-Dynamics/symthaea",
            "signer_workflow_path": ".github/workflows/cyber-crucible-trusted-build.yml",
            "signer_workflow_commit": "b" * 40,
            "oidc_issuer": GITHUB_OIDC_ISSUER,
            "builder_id": "https://github.com/actions/runner",
            "build_type": "https://actions.github.io/buildtypes/workflow/v1",
            "artifact_name": "dist/cyber-crucible-verifier-v1.tar.gz",
            "artifact_sha256": self.verifier_artifact_sha256,
            "gh_cli_path": str(self.fake_gh),
            "gh_cli_sha256": self.fake_gh_sha256,
            "expected_external_parameters": {
                "workflow": {
                    "repository": "https://github.com/Luminous-Dynamics/symthaea",
                    "ref": source_ref,
                    "path": workflow_path,
                }
            },
            "deny_self_hosted_runners": True,
        }
        self.context["verifier_provenance_policy_digest"] = json_digest(self.provenance_policy)

    def _provenance_runner(self, policy=None):
        policy = self.provenance_policy if policy is None else policy
        repo = policy["repository"]
        ref = policy["source_ref"]
        statement = {
            "_type": IN_TOTO_STATEMENT_V1,
            "subject": [{
                "name": policy["artifact_name"],
                "digest": {"sha256": policy["artifact_sha256"]},
            }],
            "predicateType": SLSA_PROVENANCE_V1,
            "predicate": {
                "buildDefinition": {
                    "buildType": policy["build_type"],
                    "externalParameters": policy["expected_external_parameters"],
                    "resolvedDependencies": [{
                        "uri": f"git+https://github.com/{repo}@{ref}",
                        "digest": {"gitCommit": policy["source_commit"]},
                    }],
                },
                "runDetails": {
                    "builder": {"id": policy["builder_id"]},
                    "metadata": {"invocationId": "https://github.com/Luminous-Dynamics/symthaea/actions/runs/999/attempts/1"},
                },
            },
        }
        output = [{
            "attestation": {"fixture": True},
            "verificationResult": {
                "signature": {"certificate": {"test": "pinned signer identity"}},
                "verifiedTimestamps": [{"kind": "rekor", "timestamp": "2030-01-01T00:00:00Z"}],
                "statement": statement,
            },
        }]
        return Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(output), stderr=""))

    def test_exact_upstream_receipt_is_bound(self):
        digest = validate_input_receipt_binding(self.input_receipt, self.context)
        self.assertEqual(digest, self.context["input_receipt_digest"])

    def test_modified_upstream_receipt_is_rejected(self):
        changed = deepcopy(self.input_receipt)
        changed["payload"]["subject_id"] = "other-subject"
        with self.assertRaises(ReportDecisionBindingError):
            validate_input_receipt_binding(changed, self.context)

    def test_upstream_receipt_scenario_mismatch_is_rejected(self):
        changed = deepcopy(self.input_receipt)
        changed["payload"]["scenario_revision"] = 4
        context = deepcopy(self.context)
        context["input_receipt_digest"] = canonical_json_digest(changed)
        with self.assertRaises(ReportDecisionBindingError):
            validate_input_receipt_binding(changed, context)

    def test_exact_report_digest_and_run_context_bind(self):
        digest = validate_report_binding(self.report, self.context)
        self.assertEqual(digest, self.context["report_digest"])

    def test_report_tampering_is_rejected(self):
        changed = deepcopy(self.report)
        changed["run"]["subject_id"] = "substituted-subject"
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(changed, self.context)

    def test_wrong_attempt_id_is_rejected(self):
        context = deepcopy(self.context)
        context["attempt_id"] = "different-attempt"
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(self.report, context)

    def test_wrong_scenario_revision_is_rejected(self):
        context = deepcopy(self.context)
        context["scenario_revision"] = 4
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(self.report, context)

    def test_wrong_report_digest_is_rejected(self):
        context = deepcopy(self.context)
        context["report_digest"] = "f" * 64
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(self.report, context)

    def test_authorizing_report_is_rejected(self):
        changed = deepcopy(self.report)
        changed["non_authorizing"] = False
        context = deepcopy(self.context)
        context["report_digest"] = canonical_json_digest(changed)
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(changed, context)

    def test_end_to_end_public_report_decision_binding(self):
        report_path = Path(__file__).resolve().parents[1] / "validation" / "cyber_crucible_diagnostic_loop_public_v1.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        run = report["run"]
        scenario = run["scenario"]
        campaign_id = "public-calibration-campaign-v1"
        attempt_id = run["run_id"]

        input_receipt = {
            "payload": {
                "schema_version": "cyber-crucible-signed-receipt.v1",
                "receipt_id": "b" * 32,
                "role": "security_verdict",
                "campaign_id": campaign_id,
                "subject_id": run["subject_id"],
                "attempt_id": attempt_id,
                "scenario_id": scenario["scenario_id"],
                "scenario_revision": scenario["revision"],
                "scenario_digest": scenario["scenario_digest"],
            },
            "signature": b64url(bytes([17]) * 64),
        }

        private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        policy = {
            "schema_version": POLICY_VERSION,
            "policy_id": "public-e2e-decision-policy-v1",
            "keys": [{
                "key_id": "e2e-verifier-key",
                "verifier_id": "e2e-independent-verifier",
                "public_key": b64url(public_key),
                "not_before": "2029-01-01T00:00:00Z",
                "not_after": "2031-01-01T00:00:00Z",
                "revoked_at": None,
            }],
        }
        context = {
            "campaign_id": campaign_id,
            "subject_id": run["subject_id"],
            "attempt_id": attempt_id,
            "scenario_id": scenario["scenario_id"],
            "scenario_revision": scenario["revision"],
            "scenario_digest": scenario["scenario_digest"],
            "input_receipt_digest": canonical_json_digest(input_receipt),
            "report_digest": canonical_json_digest(report),
            "evaluator_policy_digest": "3" * 64,
            "verifier_build_digest": self.verifier_artifact_sha256,
            "verifier_provenance_policy_digest": json_digest(self.provenance_policy),
            "decision_nonce": b64url(bytes(range(32, 64))),
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "Ed25519",
            "decision_id": "c" * 32,
            "decision_nonce": context["decision_nonce"],
            "signer_key_id": "e2e-verifier-key",
            "verifier_id": "e2e-independent-verifier",
            "decision": "accept",
            "reason_code": "authenticated_under_policy",
            **context,
            "context_digest": canonical_json_digest(context),
            "decision_trust_policy_digest": canonical_json_digest(policy),
            "issued_at": "2030-01-01T00:00:00Z",
            "expires_at": "2030-01-01T00:05:00Z",
            "non_authorizing": True,
        }
        decision_receipt = {
            "payload": payload,
            "signature": b64url(private_key.sign(canonical_signed_bytes(payload))),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = verify_report_decision(
                report,
                input_receipt,
                decision_receipt,
                policy,
                context,
                verifier_artifact_path=str(self.verifier_artifact),
                provenance_policy=self.provenance_policy,
                provenance_runner=self._provenance_runner(),
                replay_ledger=ReplayLedger(Path(temp_dir) / "e2e-replay.sqlite3"),
                now=datetime(2030, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
            )
        self.assertTrue(result["report_contract_valid"])
        self.assertTrue(result["verifier_provenance_verified"])
        self.assertEqual(result["verifier_artifact_sha256"], context["verifier_build_digest"])

        bad_context = dict(context)
        bad_context["verifier_build_digest"] = "f" * 64
        with self.assertRaises(ReportDecisionBindingError):
            verify_report_decision(
                report,
                input_receipt,
                decision_receipt,
                policy,
                bad_context,
                verifier_artifact_path=str(self.verifier_artifact),
                provenance_policy=self.provenance_policy,
                provenance_runner=self._provenance_runner(),
                replay_ledger=ReplayLedger(Path(temp_dir) / "mismatch-replay.sqlite3"),
                now=datetime(2030, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
            )

        bad_policy_context = dict(context)
        bad_policy_context["verifier_provenance_policy_digest"] = "f" * 64
        with self.assertRaises(ReportDecisionBindingError):
            verify_report_decision(
                report,
                input_receipt,
                decision_receipt,
                policy,
                bad_policy_context,
                verifier_artifact_path=str(self.verifier_artifact),
                provenance_policy=self.provenance_policy,
                provenance_runner=self._provenance_runner(),
                replay_ledger=ReplayLedger(Path(temp_dir) / "policy-mismatch-replay.sqlite3"),
                now=datetime(2030, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
            )
        self.assertTrue(result["decision_receipt_verified"])
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["report_digest"], context["report_digest"])
        self.assertTrue(result["non_authorizing"])


if __name__ == "__main__":
    unittest.main()
