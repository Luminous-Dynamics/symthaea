import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from datetime import datetime, timedelta, timezone
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

    def test_run_pr_association_must_match_exact_open_pr(self):
        repo = "owner/repo"
        subject = "a" * 40
        pr = {"number": 12}
        associated = {"number": 12,
                      "head": {"sha": subject, "ref": "security/fix", "repo": {"id": 123}},
                      "base": {"ref": "main", "repo": {"id": 123}}}
        run = {"pull_requests": [associated],
               "head_repository": {"id": 123, "full_name": repo},
               "repository": {"id": 123, "full_name": repo}}
        module.check_run_pr_binding(run, repo, pr, subject, "security/fix", "main")
        bad_runs = [
            {"pull_requests": [], "head_repository": run["head_repository"], "repository": run["repository"]},
            {"pull_requests": [dict(associated, number=13)], "head_repository": run["head_repository"], "repository": run["repository"]},
            {"pull_requests": [dict(associated, head={**associated["head"], "sha": "b" * 40})], "head_repository": run["head_repository"], "repository": run["repository"]},
            {"pull_requests": [dict(associated, head={**associated["head"], "ref": "other"})], "head_repository": run["head_repository"], "repository": run["repository"]},
            {"pull_requests": [dict(associated, base={**associated["base"], "ref": "release"})], "head_repository": run["head_repository"], "repository": run["repository"]},
            {"pull_requests": [dict(associated, head={**associated["head"], "repo": {"id": 999}})], "head_repository": run["head_repository"], "repository": run["repository"]},
            {"pull_requests": [dict(associated, base={**associated["base"], "repo": {"id": 456}})], "head_repository": run["head_repository"], "repository": run["repository"]},
            {"pull_requests": [associated, associated], "head_repository": run["head_repository"], "repository": run["repository"]},
            {**run, "head_repository": {"id": 123, "full_name": "fork/repo"}},
            {**run, "repository": {"id": 123, "full_name": "other/repo"}},
        ]
        for bad in bad_runs:
            with self.subTest(bad=bad), self.assertRaises(module.VerificationError):
                module.check_run_pr_binding(bad, repo, pr, subject, "security/fix", "main")
    def test_verifier_self_tests_require_explicit_success(self):
        self.assertTrue(module.verifier_tests_passed("success"))
        for outcome in ("", "failure", "cancelled", "skipped", "unexpected"):
            with self.subTest(outcome=outcome):
                self.assertFalse(module.verifier_tests_passed(outcome))
        self.assertFalse(module.verifier_tests_passed(None))

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
                "path": ".github/workflows/security-audit.yml",
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
            {**base, "path": ".github/workflows/security-audit.yml@refs/pull/12/merge"},
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
        subject = "a" * 40
        engine_sha = "b" * 40 if repo == "Luminous-Dynamics/luminous-platform" else module.ENGINE_SHA
        workflow_url = f"https://github.com/{repo}/actions/runs/{run_id}"
        payload = {
            "schema": "luminous.security-audit.verdict.v1",
            "repository": repo, "subject_sha": subject,
            "audit_engine_sha": engine_sha,
            "workflow_ref": f"{repo}/{policy['workflow_path']}@refs/pull/12/merge",
            "workflow_sha": "c" * 40,
            "workflow_run_url": workflow_url,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "aggregate_artifact_retention_days": 30,
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
            "evidence_files": [{"path": "workflows/test.txt", "sha256": hashlib.sha256(b"evidence").hexdigest()}],
        }
        run = {"id": run_id, "run_attempt": attempt, "head_sha": subject, "html_url": workflow_url,
               "path": policy["workflow_path"]}
        return payload, run

    def test_verdict_accepts_only_exact_subject_and_required_lanes(self):
        payload, run = self.make_verdict()
        module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, expected_pr_number=12)
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
                    module.validate_verdict(broken, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, expected_pr_number=12)

    def test_verdict_workflow_path_must_match_authoritative_run_path(self):
        payload, run = self.make_verdict()
        module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, expected_pr_number=12)
        run["path"] = ".github/workflows/other.yml"
        with self.assertRaises(module.VerificationError):
            module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, expected_pr_number=12)

    def test_verdict_engine_commit_is_policy_bound(self):
        payload, run = self.make_verdict()
        payload["audit_engine_sha"] = "c" * 40
        with self.assertRaises(module.VerificationError):
            module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, expected_pr_number=12)

    def test_verdict_freshness_rejects_stale_and_future_timestamps(self):
        repo = "Luminous-Dynamics/mycelix"
        policy = module.POLICY[repo]
        payload, run = self.make_verdict(repo)
        payload["generated_at_utc"] = (datetime.now(timezone.utc) - timedelta(days=8)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        with self.assertRaises(module.VerificationError):
            module.validate_verdict(payload, repo, policy, run, expected_pr_number=12)
        payload, run = self.make_verdict(repo)
        payload["generated_at_utc"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        with self.assertRaises(module.VerificationError):
            module.validate_verdict(payload, repo, policy, run, expected_pr_number=12)

    def test_verdict_ref_must_match_exact_pull_request_number(self):
        repo = "Luminous-Dynamics/mycelix"
        policy = module.POLICY[repo]
        payload, run = self.make_verdict(repo)
        module.validate_verdict(payload, repo, policy, run, expected_pr_number=12)
        payload["workflow_ref"] = f"{repo}/{policy['workflow_path']}@refs/pull/13/merge"
        with self.assertRaises(module.VerificationError):
            module.validate_verdict(payload, repo, policy, run, expected_pr_number=12)

    def test_verdict_rejects_wrong_workflow_ref_url_and_unknown_fields(self):
        repo = "Luminous-Dynamics/mycelix"
        policy = module.POLICY[repo]
        for mutate in (
            lambda x: x.update({"workflow_ref": "attacker/repo/.github/workflows/security-audit.yml@refs/pull/12/merge"}),
            lambda x: x.update({"workflow_run_url": "https://example.invalid/fake-run"}),
            lambda x: x.update({"unexpected": "ignored by consumer"}),
            lambda x: x.update({"workflow_sha": "unknown"}),
        ):
            payload, run = self.make_verdict(repo)
            mutate(payload)
            with self.subTest(payload=payload):
                with self.assertRaises(module.VerificationError):
                    module.validate_verdict(payload, repo, policy, run, expected_pr_number=12)

    def test_pass_with_findings_is_distinct_and_consistent(self):
        payload, run = self.make_verdict()
        payload.update({"status": "PASS_WITH_FINDINGS", "non_blocking_findings_present": True,
                        "non_blocking_finding_sources": ["npm_below_threshold"]})
        module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, expected_pr_number=12)
        payload["non_blocking_finding_sources"] = []
        with self.assertRaises(module.VerificationError):
            module.validate_verdict(payload, "Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, expected_pr_number=12)

    def test_artifact_digest_and_run_metadata_are_checked(self):
        payload, run = self.make_verdict()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("verdict.json", json.dumps(payload))
            archive.writestr("workflows/test.txt", b"evidence")
        archive_bytes = buffer.getvalue()
        digest = "sha256:" + hashlib.sha256(archive_bytes).hexdigest()
        artifact = {"id": 77, "name": f"security-audit-mycelix-{run['head_sha']}-verdict",
                    "expired": False, "digest": digest,
                    "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        with patch.object(module, "api", return_value={"artifacts": [artifact]}), \
             patch.object(module, "download_artifact_zip", return_value=archive_bytes), \
             patch.object(module, "check_blob") as check_blob:
            module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token", expected_pr_number=12)
        check_blob.assert_called_once_with(
            "Luminous-Dynamics/mycelix",
            module.POLICY["Luminous-Dynamics/mycelix"]["workflow_path"],
            "c" * 40,
            module.POLICY["Luminous-Dynamics/mycelix"]["workflow_blob"],
            "token",
        )
        altered = dict(artifact, digest="sha256:" + "0" * 64)
        with patch.object(module, "api", return_value={"artifacts": [altered]}), \
             patch.object(module, "download_artifact_zip", return_value=archive_bytes):
            with self.assertRaises(module.VerificationError):
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token", expected_pr_number=12)

    def test_artifact_evidence_file_digest_mismatch_is_rejected(self):
        payload, run = self.make_verdict()
        payload["evidence_files"][0]["sha256"] = "0" * 64
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("verdict.json", json.dumps(payload))
            archive.writestr("workflows/test.txt", b"evidence")
        archive_bytes = buffer.getvalue()
        artifact = {"id": 77, "name": f"security-audit-mycelix-{run['head_sha']}-verdict",
                    "expired": False, "digest": "sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
                    "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        with patch.object(module, "api", return_value={"artifacts": [artifact]}), \
             patch.object(module, "download_artifact_zip", return_value=archive_bytes):
            with self.assertRaises(module.VerificationError):
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token", expected_pr_number=12)

    def test_artifact_missing_manifest_file_is_rejected(self):
        payload, run = self.make_verdict()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("verdict.json", json.dumps(payload))
        archive_bytes = buffer.getvalue()
        artifact = {"id": 77, "name": f"security-audit-mycelix-{run['head_sha']}-verdict",
                    "expired": False, "digest": "sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
                    "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        with patch.object(module, "api", return_value={"artifacts": [artifact]}), \
             patch.object(module, "download_artifact_zip", return_value=archive_bytes):
            with self.assertRaises(module.VerificationError):
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token", expected_pr_number=12)

    def test_artifact_without_digest_or_with_expiry_is_rejected(self):
        payload, run = self.make_verdict()
        artifact = {"id": 77, "name": f"security-audit-mycelix-{run['head_sha']}-verdict",
                    "expired": True, "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        with patch.object(module, "api", return_value={"artifacts": [artifact]}):
            with self.assertRaises(module.VerificationError):
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token", expected_pr_number=12)

    def test_duplicate_json_keys_are_rejected_even_when_artifact_digest_matches(self):
        payload, run = self.make_verdict()
        raw = json.dumps(payload).replace('"status": "PASS"', '"status": "FAIL", "status": "PASS"', 1)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("verdict.json", raw)
            archive.writestr("workflows/test.txt", b"evidence")
        archive_bytes = buffer.getvalue()
        artifact = {"id": 77, "name": f"security-audit-mycelix-{run['head_sha']}-verdict",
                    "expired": False, "digest": "sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
                    "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        with patch.object(module, "api", return_value={"artifacts": [artifact]}), \
             patch.object(module, "download_artifact_zip", return_value=archive_bytes):
            with self.assertRaises(module.VerificationError):
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token", expected_pr_number=12)

    def test_nested_verdict_path_is_rejected(self):
        payload, run = self.make_verdict()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("nested/verdict.json", json.dumps(payload))
            archive.writestr("workflows/test.txt", b"evidence")
        archive_bytes = buffer.getvalue()
        artifact = {"id": 77, "name": f"security-audit-mycelix-{run['head_sha']}-verdict",
                    "expired": False, "digest": "sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
                    "workflow_run": {"id": run["id"], "head_sha": run["head_sha"]}}
        with patch.object(module, "api", return_value={"artifacts": [artifact]}), patch.object(module, "download_artifact_zip", return_value=archive_bytes):
            with self.assertRaises(module.VerificationError):
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token", expected_pr_number=12)

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
                module.verify_verdict_artifact("Luminous-Dynamics/mycelix", module.POLICY["Luminous-Dynamics/mycelix"], run, "token", expected_pr_number=12)

    def test_pending_status_only_for_uncompleted_runs(self):
        self.assertTrue(module.should_publish_pending_status("workflow_run", "requested", {"status": "queued"}))
        self.assertTrue(module.should_publish_pending_status("workflow_run", "in_progress", {"status": "in_progress"}))
        self.assertFalse(module.should_publish_pending_status("workflow_run", "in_progress", {"status": "completed"}))
        self.assertFalse(module.should_publish_pending_status("workflow_run", "completed", {"status": "completed"}))
        self.assertFalse(module.should_publish_pending_status("workflow_run", "in_progress", {"status": "unexpected"}))
        self.assertFalse(module.should_publish_pending_status("workflow_run", "in_progress", {}))
        self.assertFalse(module.should_publish_pending_status("workflow_run", "in_progress", {"status": "in_progress", "conclusion": "success"}))
        self.assertFalse(module.should_publish_pending_status("workflow_dispatch", "requested", {"status": "queued"}))

    def event_meta(self, repo):
        return {
            "workflow_id": module.POLICY[repo]["workflow_id"],
            "event_name": "pull_request",
            "repository_id": "12345",
            "head_repository_id": "12345",
            "current_repository_id": "12345",
            "pull_request_base_ref": "main",
            "default_branch": "main",
        }

    def test_best_effort_failure_status_clears_same_head_success(self):
        repo = "Luminous-Dynamics/mycelix"
        with patch.object(module, "post_status") as post:
            result = module.best_effort_failure_status(repo, "a" * 40, "token",
                                                       "API unavailable", "https://github.com/run/1",
                                                       **self.event_meta(repo))
        self.assertTrue(result)
        post.assert_called_once()
        self.assertEqual(post.call_args.args[3], "failure")

    def test_best_effort_failure_status_rejects_untrusted_or_invalid_identity(self):
        repo = "Luminous-Dynamics/mycelix"
        cases = [
            ("unknown/repo", "a" * 40, "token", self.event_meta(repo)),
            (repo, "not-a-sha", "token", self.event_meta(repo)),
            (repo, "a" * 40, "", self.event_meta(repo)),
            (repo, "a" * 40, "token", {**self.event_meta(repo), "workflow_id": 1}),
            (repo, "a" * 40, "token", {**self.event_meta(repo), "event_name": "push"}),
            (repo, "a" * 40, "token", {**self.event_meta(repo), "head_repository_id": "54321"}),
            (repo, "a" * 40, "token", {**self.event_meta(repo), "current_repository_id": "54321"}),
            (repo, "a" * 40, "token", {**self.event_meta(repo), "pull_request_base_ref": "release"}),
        ]
        for target_repo, subject, token, event_meta in cases:
            with self.subTest(repo=target_repo, subject=subject, event_meta=event_meta), patch.object(module, "post_status") as post:
                self.assertFalse(module.best_effort_failure_status(target_repo, subject, token,
                                                                    "failure", **event_meta))
                post.assert_not_called()

    def test_best_effort_failure_status_survives_unexpected_status_api_failure(self):
        repo = "Luminous-Dynamics/mycelix"
        with patch.object(module, "post_status", side_effect=RuntimeError("unexpected transport failure")):
            self.assertFalse(module.best_effort_failure_status(repo, "a" * 40, "token",
                                                               "API unavailable", **self.event_meta(repo)))

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
