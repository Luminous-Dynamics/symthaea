import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).with_name("verify_security_audit_trust_anchor.py")
spec = importlib.util.spec_from_file_location("audit_anchor", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class TrustAnchorPolicyTests(unittest.TestCase):
    def test_policy_pins_are_canonical(self):
        self.assertEqual(set(module.POLICY), {
            "Luminous-Dynamics/mycelix", "Luminous-Dynamics/symthaea",
            "Luminous-Dynamics/luminous-platform"})
        for repo, policy in module.POLICY.items():
            with self.subTest(repo=repo):
                self.assertRegex(policy["workflow_blob"], r"^[0-9a-f]{40}$")
                self.assertGreater(policy["workflow_id"], 0)
                if policy.get("engine_sha") is not None:
                    self.assertEqual(policy["engine_sha"], module.ENGINE_SHA)
        self.assertRegex(module.ENGINE_SHA, r"^[0-9a-f]{40}$")
        self.assertRegex(module.ENGINE_BLOB, r"^[0-9a-f]{40}$")

    def test_only_latest_exact_head_run_attempt_is_eligible(self):
        first = {"id": 7, "workflow_id": 42, "event": "pull_request", "head_sha": "a" * 40,
                 "run_number": 15, "run_attempt": 1}
        second = {**first, "id": 8, "run_number": 16}
        rerun = {**second, "run_attempt": 2}
        self.assertFalse(module.is_latest(first, [first, second], {"workflow_id": 42}, "a" * 40))
        self.assertFalse(module.is_latest(second, [second, rerun], {"workflow_id": 42}, "a" * 40))
        self.assertTrue(module.is_latest(rerun, [second, rerun], {"workflow_id": 42}, "a" * 40))

    def test_wrong_head_or_workflow_is_never_latest(self):
        candidate = {"id": 7, "workflow_id": 99, "event": "pull_request", "head_sha": "b" * 40,
                     "run_number": 15, "run_attempt": 1}
        self.assertFalse(module.is_latest(candidate, [candidate], {"workflow_id": 42}, "b" * 40))
        self.assertFalse(module.is_latest(candidate, [candidate], {"workflow_id": 99}, "c" * 40))
        self.assertFalse(module.is_latest(candidate, [], {"workflow_id": 99}, "b" * 40))

    def fixture_run(self):
        return {"id": 10, "workflow_id": 42, "name": "Security Audit",
                "path": ".github/workflows/security-audit.yml@refs/pull/12/merge",
                "event": "pull_request", "head_branch": "security/fix",
                "head_sha": "e" * 40, "run_attempt": 1,
                "head_repository": {"full_name": "owner/repo"},
                "repository": {"full_name": "owner/repo"}}

    def test_valid_run_binding_and_expected_attempt(self):
        run = self.fixture_run()
        policy = {"workflow_id": 42, "workflow_name": "Security Audit",
                  "workflow_path": ".github/workflows/security-audit.yml"}
        self.assertEqual(module.check_binding(run, "owner/repo", policy, "security/fix", "e" * 40, 1), "e" * 40)
        with self.assertRaises(module.VerificationError):
            module.check_binding(run, "owner/repo", policy, "security/fix", "e" * 40, 2)

    def test_binding_rejects_wrong_path_fork_sha_and_event(self):
        policy = {"workflow_id": 42, "workflow_name": "Security Audit",
                  "workflow_path": ".github/workflows/security-audit.yml"}
        base = self.fixture_run()
        bad_runs = [
            {**base, "path": ".github/workflows/other.yml@refs/pull/12/merge"},
            {**base, "head_repository": {"full_name": "fork/repo"}},
            {**base, "head_sha": "f" * 40},
            {**base, "event": "push"},
            {**base, "workflow_id": 43},
            {**base, "head_branch": "main"},
        ]
        for bad in bad_runs:
            with self.subTest(bad=bad):
                with self.assertRaises(module.VerificationError):
                    module.check_binding(bad, "owner/repo", policy, "security/fix", "e" * 40, 1)

    def test_noncanonical_sha_is_rejected(self):
        for value in ("ABC", "0" * 39, "g" * 40, None, 123):
            with self.subTest(value=value):
                with self.assertRaises(module.VerificationError):
                    module.sha(value, "fixture")

    def test_blob_mismatch_fails_closed(self):
        with patch.object(module, "api", return_value={"type": "file", "sha": "0" * 40}):
            with self.assertRaises(module.VerificationError):
                module.check_blob("owner/repo", ".github/workflows/security-audit.yml",
                                  "e" * 40, "1" * 40, "token")

    def test_blob_type_missing_and_mismatch_fail_closed(self):
        for response in (None, {"type": "dir", "sha": "1" * 40}, {"type": "file", "sha": "2" * 40}):
            with self.subTest(response=response), patch.object(module, "api", return_value=response):
                with self.assertRaises(module.VerificationError):
                    module.check_blob("owner/repo", "path.yml", "e" * 40, "1" * 40, "token")

    def make_verdict(self, repo="Luminous-Dynamics/mycelix", run_id=123, attempt=2):
        policy = module.POLICY[repo]
        is_platform = repo == "Luminous-Dynamics/luminous-platform"
        subject = "a" * 40
        engine_sha = "b" * 40 if is_platform else module.ENGINE_SHA
        return ({
            "schema": "luminous.security-audit.verdict.v1",
            "repository": repo, "subject_sha": subject,
            "audit_engine_sha": engine_sha,
            "run_id": str(run_id), "run_attempt": str(attempt),
            "workflow_job_result": "success",
            "rustsec_job_result": "success" if policy["audit_rust"] else "skipped",
            "npm_job_result": "success" if policy["audit_node"] else "skipped",
            "audit_rust": policy["audit_rust"], "audit_node": policy["audit_node"],
            "required_lanes": {
                "workflow_security": "PASS",
                "rustsec": "PASS" if policy["audit_rust"] else "SKIPPED_NOT_REQUESTED",
                "npm": "PASS" if policy["audit_node"] else "SKIPPED_NOT_REQUESTED",
            },
            "status": "PASS", "failure_reasons": [],
            "non_blocking_findings_present": False, "non_blocking_finding_sources": [],
        }, {"id": run_id, "run_attempt": attempt, "head_sha": subject})

    def test_verdict_accepts_only_exact_subject_and_required_lanes(self):
        payload, run = self.make_verdict()
        module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run)
        for change in (
            lambda x: x.update({"subject_sha": "c" * 40}),
            lambda x: x.update({"run_attempt": "1"}),
            lambda x: x.update({"status": "INCOMPLETE"}),
            lambda x: x["required_lanes"].update({"npm": "SKIPPED_NOT_REQUESTED"}),
            lambda x: x.update({"failure_reasons": ["missing coverage"]}),
            lambda x: x.update({"non_blocking_findings_present": True}),
        ):
            broken = json.loads(json.dumps(payload))
            change(broken)
            with self.subTest(broken=broken):
                with self.assertRaises(module.VerificationError):
                    module.validate_verdict(broken, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run)

    def test_verdict_engine_commit_is_policy_bound(self):
        payload, run = self.make_verdict()
        payload["audit_engine_sha"] = "c" * 40
        with self.assertRaises(module.VerificationError):
            module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run)

    def test_pass_with_findings_is_distinct_and_consistent(self):
        payload, run = self.make_verdict()
        payload.update({"status": "PASS_WITH_FINDINGS", "non_blocking_findings_present": True,
                        "non_blocking_finding_sources": ["npm_below_threshold"]})
        module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run)
        payload["non_blocking_finding_sources"] = []
        with self.assertRaises(module.VerificationError):
            module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run)

    def test_artifact_digest_and_run_metadata_are_checked(self):
        payload, run = self.make_verdict()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("verdict.json", json.dumps(payload))
        archive_bytes = buffer.getvalue()
        digest = "sha256:" + hashlib.sha256(archive_bytes).hexdigest()
        artifact = {"id": 77, "name": f"security-audit-mycelix-{run['head_sha']}-verdict",
                    "expired": False, "digest": digest,
                    "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        with patch.object(module, "api", return_value={"artifacts": [artifact]}), \
             patch.object(module, "download_artifact_zip", return_value=archive_bytes):
            module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token")
        altered = dict(artifact, digest="sha256:" + "0" * 64)
        with patch.object(module, "api", return_value={"artifacts": [altered]}), \
             patch.object(module, "download_artifact_zip", return_value=archive_bytes):
            with self.assertRaises(module.VerificationError):
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token")

    def test_artifact_without_digest_or_with_expiry_is_rejected(self):
        payload, run = self.make_verdict()
        artifact = {"id": 77, "name": f"security-audit-mycelix-{run['head_sha']}-verdict",
                    "expired": True, "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        with patch.object(module, "api", return_value={"artifacts": [artifact]}):
            with self.assertRaises(module.VerificationError):
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token")

    def test_artifact_zip_path_traversal_is_rejected(self):
        payload, run = self.make_verdict()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("../verdict.json", json.dumps(payload))
        archive_bytes = buffer.getvalue()
        artifact = {"id": 77, "name": f"security-audit-mycelix-{run['head_sha']}-verdict",
                    "expired": False, "digest": "sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
                    "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        with patch.object(module, "api", return_value={"artifacts": [artifact]}), \
             patch.object(module, "download_artifact_zip", return_value=archive_bytes):
            with self.assertRaises(module.VerificationError):
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token")

    def test_api_rejects_bad_json(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b"not-json"
        with patch.object(module.urllib.request, "urlopen", return_value=Response()):
            with self.assertRaises(module.VerificationError):
                module.api("GET", "/repos/example/repo", "token")


if __name__ == "__main__":
    unittest.main()
