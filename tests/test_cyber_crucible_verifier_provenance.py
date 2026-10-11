"""Adversarial tests for verifier-build provenance policy enforcement."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jsonschema import Draft202012Validator, ValidationError
from scripts.package_cyber_crucible_verifier import build_bundle

from scripts.verify_cyber_crucible_verifier_provenance import (
    GITHUB_OIDC_ISSUER,
    IN_TOTO_STATEMENT_V1,
    POLICY_VERSION,
    SLSA_PROVENANCE_V1,
    ProvenanceVerificationError,
    load_json,
    verify_verifier_provenance,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VerifierProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cyber provenance ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "cyber-crucible-verifier-v1.tar.gz"
        self.repository = "Luminous-Dynamics/symthaea"
        self.source_ref = "refs/tags/cyber-crucible-verifier-v1.0.0"
        self.source_commit = "a" * 40
        repository_root = Path(__file__).resolve().parents[1]
        build_bundle(repository_root, self.artifact, self.source_commit)
        self.gh = self.root / "operator-pinned-gh"
        self.gh.write_bytes(b"fixture-only fake gh CLI")
        self.gh.chmod(0o700)
        self.workflow_path = ".github/workflows/cyber-crucible-verifier-release.yml"
        self.policy = {
            "schema_version": POLICY_VERSION,
            "policy_id": "verifier-provenance-fixture-v1",
            "repository": self.repository,
            "source_ref": self.source_ref,
            "source_commit": self.source_commit,
            "workflow_path": self.workflow_path,
            "signer_repository": self.repository,
            "signer_workflow_path": ".github/workflows/cyber-crucible-trusted-build.yml",
            "signer_workflow_commit": "b" * 40,
            "oidc_issuer": GITHUB_OIDC_ISSUER,
            "builder_id": "https://github.com/actions/runner",
            "build_type": "https://actions.github.io/buildtypes/workflow/v1",
            "artifact_name": "dist/cyber-crucible-verifier-v1.tar.gz",
            "artifact_sha256": sha256_file(self.artifact),
            "gh_cli_path": str(self.gh),
            "gh_cli_sha256": sha256_file(self.gh),
            "expected_external_parameters": {
                "workflow": {
                    "repository": f"https://github.com/{self.repository}",
                    "ref": self.source_ref,
                    "path": self.workflow_path,
                }
            },
            "deny_self_hosted_runners": True,
        }
        self.runner = Mock(return_value=SimpleNamespace(
            returncode=0,
            stdout=json.dumps([self.verified_entry()]),
            stderr="",
        ))

    def verified_entry(self, *, statement=None, certificate=None, timestamps=None):
        if statement is None:
            statement = {
                "_type": IN_TOTO_STATEMENT_V1,
                "subject": [{
                    "name": self.policy["artifact_name"],
                    "digest": {"sha256": self.policy["artifact_sha256"]},
                }],
                "predicateType": SLSA_PROVENANCE_V1,
                "predicate": {
                    "buildDefinition": {
                        "buildType": self.policy["build_type"],
                        "externalParameters": self.policy["expected_external_parameters"],
                        "resolvedDependencies": [{
                            "uri": f"git+https://github.com/{self.repository}@{self.source_ref}",
                            "digest": {"gitCommit": self.source_commit},
                        }],
                    },
                    "runDetails": {
                        "builder": {"id": self.policy["builder_id"]},
                        "metadata": {
                            "invocationId": "https://github.com/Luminous-Dynamics/symthaea/actions/runs/123/attempts/1"
                        },
                    },
                },
            }
        return {
            "attestation": {"fixture": True},
            "verificationResult": {
                "signature": {"certificate": certificate if certificate is not None else {"subjectAlternativeName": "pinned-workflow-identity"}},
                "verifiedTimestamps": timestamps if timestamps is not None else [{"kind": "rekor", "verified": True}],
                "statement": statement,
            },
        }

    def verify(self, *, policy=None, output=None):
        if output is not None:
            self.runner.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(output), stderr="")
        return verify_verifier_provenance(
            self.artifact,
            self.policy if policy is None else policy,
            runner=self.runner,
        )

    def test_policy_fixture_matches_published_schema(self):
        schema_path = Path(__file__).resolve().parents[1] / "validation" / "cyber_crucible_verifier_provenance_policy_v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(self.policy)

    def test_policy_schema_requires_separate_signer_workflow(self):
        schema_path = Path(__file__).resolve().parents[1] / "validation" / "cyber_crucible_verifier_provenance_policy_v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        policy = dict(self.policy)
        del policy["signer_workflow_path"]
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(policy)

    def test_policy_schema_rejects_unknown_fields(self):
        schema_path = Path(__file__).resolve().parents[1] / "validation" / "cyber_crucible_verifier_provenance_policy_v1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        policy = dict(self.policy)
        policy["trust_everything"] = True
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(policy)

    def test_valid_attestation_returns_non_authorizing_provenance_result(self):
        result = self.verify()
        self.assertTrue(result["provenance_verified"])
        self.assertTrue(result["attestation_signature_verified"])
        self.assertTrue(result["verified_transparency_timestamp_present"])
        self.assertTrue(result["non_authorizing"])
        self.assertEqual(result["artifact_sha256"], self.policy["artifact_sha256"])
        self.assertEqual(result["source_commit"], self.source_commit)
        self.assertEqual(result["workflow_path"], self.workflow_path)
        self.assertEqual(result["signer_workflow_path"], ".github/workflows/cyber-crucible-trusted-build.yml")

    def test_cli_receives_exact_identity_and_source_constraints_without_shell(self):
        self.verify()
        args, kwargs = self.runner.call_args
        command = args[0]
        self.assertIsInstance(command, list)
        self.assertFalse(kwargs["shell"])
        self.assertIn("--hostname", command)
        self.assertIn("github.com", command)
        self.assertIn("--signer-workflow", command)
        self.assertIn(f"{self.repository}/.github/workflows/cyber-crucible-trusted-build.yml", command)
        self.assertIn("--signer-repo", command)
        self.assertIn(self.repository, command)
        self.assertIn("--signer-digest", command)
        self.assertIn(self.policy["signer_workflow_commit"], command)
        self.assertIn("--source-ref", command)
        self.assertIn(self.source_ref, command)
        self.assertIn("--source-digest", command)
        self.assertIn(self.source_commit, command)
        self.assertIn("--cert-oidc-issuer", command)
        self.assertIn(GITHUB_OIDC_ISSUER, command)
        self.assertIn("--deny-self-hosted-runners", command)
        snapshot_path = Path(command[3])
        self.assertNotEqual(str(self.artifact), str(snapshot_path))
        self.assertTrue(snapshot_path.is_file())
        self.assertEqual(sha256_file(snapshot_path), self.policy["artifact_sha256"])

    def test_verification_environment_pins_host_and_drops_untrusted_gh_config(self):
        with patch.dict(os.environ, {
            "GH_HOST": "attacker.invalid",
            "GH_CONFIG_DIR": str(self.root / "untrusted-gh-config"),
            "GH_TOKEN": "operator-provided-test-token",
        }, clear=False):
            self.verify()
        env = self.runner.call_args.kwargs["env"]
        self.assertNotIn("GH_HOST", env)
        self.assertNotIn("GH_CONFIG_DIR", env)
        self.assertEqual(env["GH_TOKEN"], "operator-provided-test-token")
        self.assertEqual(env["GH_PROMPT_DISABLED"], "1")

    def test_artifact_digest_mismatch_fails_before_cli(self):
        self.artifact.write_bytes(b"substituted binary")
        with self.assertRaises(ProvenanceVerificationError):
            self.verify()
        self.runner.assert_not_called()

    def test_changed_cli_digest_fails_before_cli(self):
        self.gh.write_bytes(b"replaced gh executable")
        with self.assertRaises(ProvenanceVerificationError):
            self.verify()
        self.runner.assert_not_called()

    def test_nonzero_gh_verification_fails_closed(self):
        self.runner.return_value = SimpleNamespace(returncode=1, stdout="[]", stderr="signature invalid")
        with self.assertRaises(ProvenanceVerificationError):
            self.verify()

    def test_invalid_cli_json_fails_closed(self):
        self.runner.return_value = SimpleNamespace(returncode=0, stdout="{", stderr="")
        with self.assertRaises(ProvenanceVerificationError):
            self.verify()

    def test_duplicate_cli_json_members_are_rejected(self):
        self.runner.return_value = SimpleNamespace(returncode=0, stdout='[{"a":1,"a":2}]', stderr="")
        with self.assertRaises(ProvenanceVerificationError):
            self.verify()

    def test_multiple_verified_attestations_are_rejected_as_ambiguous(self):
        entry = self.verified_entry()
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry, entry])

    def test_empty_timestamp_object_is_rejected(self):
        entry = self.verified_entry(timestamps=[{}])
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_missing_verified_certificate_is_rejected(self):
        entry = self.verified_entry(certificate={})
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_missing_transparency_timestamp_is_rejected(self):
        entry = self.verified_entry(timestamps=[])
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_wrong_statement_type_is_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["_type"] = "https://in-toto.io/Statement/v0.1"
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_wrong_predicate_type_is_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["predicateType"] = "https://example.invalid/predicate"
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_wrong_artifact_subject_name_is_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["subject"][0]["name"] = "dist/other"
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_wrong_artifact_subject_digest_is_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["subject"][0]["digest"]["sha256"] = "f" * 64
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_multiple_subjects_are_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["subject"].append({
            "name": "extra.bin", "digest": {"sha256": "1" * 64}
        })
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_wrong_build_type_is_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["predicate"]["buildDefinition"]["buildType"] = "https://evil.invalid/build"
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_external_parameter_injection_is_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["predicate"]["buildDefinition"]["externalParameters"]["injected"] = True
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_wrong_workflow_repository_ref_or_path_is_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["predicate"]["buildDefinition"]["externalParameters"]["workflow"]["path"] = ".github/workflows/attacker.yml"
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_wrong_builder_identity_is_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["predicate"]["runDetails"]["builder"]["id"] = "https://evil.invalid/builder"
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_wrong_resolved_source_commit_is_rejected(self):
        entry = self.verified_entry()
        entry["verificationResult"]["statement"]["predicate"]["buildDefinition"]["resolvedDependencies"][0]["digest"]["gitCommit"] = "c" * 40
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_ambiguous_resolved_source_dependency_is_rejected(self):
        entry = self.verified_entry()
        dependency = entry["verificationResult"]["statement"]["predicate"]["buildDefinition"]["resolvedDependencies"][0]
        entry["verificationResult"]["statement"]["predicate"]["buildDefinition"]["resolvedDependencies"].append(dependency)
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_missing_invocation_id_is_rejected(self):
        entry = self.verified_entry()
        del entry["verificationResult"]["statement"]["predicate"]["runDetails"]["metadata"]["invocationId"]
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(output=[entry])

    def test_policy_unknown_fields_are_rejected(self):
        policy = dict(self.policy)
        policy["allow_any_builder"] = True
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(policy=policy)

    def test_policy_rejects_conflicting_external_inputs_and_release_parameters(self):
        policy = dict(self.policy)
        policy["expected_external_parameters"] = dict(self.policy["expected_external_parameters"])
        policy["expected_external_parameters"]["inputs"] = {"mode": "release"}
        policy["expected_external_parameters"]["release"] = {"tag": "cyber-crucible-verifier-v1.2.3"}
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(policy=policy)

    def test_self_hosted_runner_policy_cannot_be_disabled(self):
        policy = dict(self.policy)
        policy["deny_self_hosted_runners"] = False
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(policy=policy)

    def test_pull_request_ref_is_not_accepted_as_release_source(self):
        policy = dict(self.policy)
        policy["source_ref"] = "refs/pull/42/merge"
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(policy=policy)

    def test_empty_branch_ref_is_rejected(self):
        policy = dict(self.policy)
        policy["source_ref"] = "refs/heads/"
        with self.assertRaises(ProvenanceVerificationError):
            self.verify(policy=policy)

    def test_symlink_artifact_is_rejected(self):
        link = self.root / "artifact-link"
        link.symlink_to(self.artifact)
        with self.assertRaises(ProvenanceVerificationError):
            verify_verifier_provenance(link, self.policy, runner=self.runner)

    def test_policy_duplicate_json_member_is_rejected(self):
        path = self.root / "bad-policy.json"
        path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
        with self.assertRaises(ProvenanceVerificationError):
            load_json(path)


if __name__ == "__main__":
    unittest.main()
