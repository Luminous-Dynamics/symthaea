#!/usr/bin/env python3
"""Base-branch trust anchor for the Luminous security-audit workflow.

This script reads PR state via GitHub's API. It never checks out or executes PR
source or consumes producer artifacts. Its policy must only be changed through
an independently reviewed change on the trusted default branch.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any

API = "https://api.github.com"
API_VERSION = "2022-11-28"
STATUS_CONTEXT = "Security Audit / Independent Verifier"
VERDICT_MAX_AGE = timedelta(days=7)
VERDICT_FUTURE_SKEW = timedelta(minutes=5)


def reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of accepting the last duplicate key."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result
ENGINE_REPO = "Luminous-Dynamics/luminous-platform"
ENGINE_SHA = "15f8135368e6162ca9b07861713a567f95fe765a"
ENGINE_PATH = ".github/workflows/security-audit.yml"
ENGINE_BLOB = "6d8f9981822a65835550895b4964337182cf3fb5"
POLICY: dict[str, dict[str, Any]] = {
    "Luminous-Dynamics/mycelix": {
        "workflow_id": 379572428,
        "workflow_name": "Security Audit",
        "workflow_path": ".github/workflows/security-audit.yml",
        "workflow_blob": "d7660286694f4f0369120506c6089e544cdc54cb",
        "engine_sha": ENGINE_SHA,
        "audit_rust": True,
        "audit_node": True,
    },
    "Luminous-Dynamics/symthaea": {
        "workflow_id": 379572736,
        "workflow_name": "Security Audit",
        "workflow_path": ".github/workflows/security-audit.yml",
        "workflow_blob": "f24de1c8d249c365498b78c8ad97557e666d97b0",
        "engine_sha": ENGINE_SHA,
        "audit_rust": True,
        "audit_node": False,
    },
    "Luminous-Dynamics/luminous-platform": {
        "workflow_id": 379571857,
        "workflow_name": "Security Audit — Platform Self-Check",
        "workflow_path": ".github/workflows/security-audit-self.yml",
        "workflow_blob": "8f621751d2c6ac255250f47d2ea4b71b99a9c6c4",
        "engine_sha": None,
        "audit_rust": False,
        "audit_node": False,
    },
}


class VerificationError(RuntimeError):
    pass


def sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise VerificationError(f"{label} is not a canonical lowercase Git SHA")
    return value


def api(method: str, path: str, token: str, body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None) -> Any:
    url = API + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    payload = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=payload, method=method, headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "luminous-security-audit-verifier/1",
        **({"Content-Type": "application/json"} if payload is not None else {}),
    })
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode("utf-8", errors="replace")
        raise VerificationError(f"GitHub API {method} {path}: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise VerificationError(f"GitHub API unavailable for {path}: {exc}") from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"GitHub API returned invalid JSON for {path}") from exc


def find_pr(repo: str, subject: str, branch: str, default_branch: str, token: str) -> dict[str, Any] | None:
    owner, name = repo.split("/", 1)
    rows = api("GET", f"/repos/{owner}/{name}/commits/{subject}/pulls", token,
               query={"per_page": 100})
    if not isinstance(rows, list):
        raise VerificationError("commit-to-PR response has an unexpected shape")
    matches = []
    for pr in rows:
        head, base = pr.get("head") or {}, pr.get("base") or {}
        head_repo, base_repo = head.get("repo") or {}, base.get("repo") or {}
        if (pr.get("state") == "open" and head.get("sha") == subject
                and head.get("ref") == branch
                and str(head_repo.get("full_name", "")).lower() == repo.lower()
                and base.get("ref") == default_branch
                and str(base_repo.get("full_name", "")).lower() == repo.lower()):
            matches.append(pr)
    if len(matches) > 1:
        raise VerificationError("ambiguous open same-repository PRs match the exact subject")
    return matches[0] if matches else None


def is_latest(candidate: dict[str, Any], runs: list[dict[str, Any]], policy: dict[str, Any], subject: str) -> bool:
    eligible = [r for r in runs if r.get("workflow_id") == policy["workflow_id"]
                and r.get("event") == "pull_request" and r.get("head_sha") == subject
                and isinstance(r.get("id"), int)]
    if not eligible:
        return False
    latest = max(eligible, key=lambda r: (int(r.get("run_number", 0)),
                                          int(r.get("run_attempt", 0)),
                                          str(r.get("created_at", "")), int(r["id"])))
    return (candidate.get("id") == latest.get("id")
            and candidate.get("run_attempt", 0) == latest.get("run_attempt", 0))


def check_binding(run: dict[str, Any], repo: str, policy: dict[str, Any],
                  branch: str, expected_sha: str, expected_attempt: int | None) -> str:
    if run.get("workflow_id") != policy["workflow_id"] or run.get("name") != policy["workflow_name"]:
        raise VerificationError("workflow ID/name does not match the base-owned policy")
    if str(run.get("path", "")).split("@", 1)[0] != policy["workflow_path"]:
        raise VerificationError("workflow path does not match the base-owned policy")
    if run.get("event") != "pull_request":
        raise VerificationError("only pull_request runs can qualify a merge")
    if run.get("head_branch") != branch:
        raise VerificationError("run branch does not match the exact open PR head")
    subject = sha(run.get("head_sha"), "run.head_sha")
    if subject != expected_sha:
        raise VerificationError("run head SHA differs from the event's exact SHA")
    if expected_attempt is not None and run.get("run_attempt") != expected_attempt:
        raise VerificationError("run attempt differs from the event payload")
    if str((run.get("head_repository") or {}).get("full_name", "")).lower() != repo.lower():
        raise VerificationError("fork-originated runs are not eligible")
    if str((run.get("repository") or {}).get("full_name", "")).lower() != repo.lower():
        raise VerificationError("run repository differs from policy repository")
    return subject


def check_blob(repo: str, path: str, ref: str, expected_blob: str, token: str) -> None:
    owner, name = repo.split("/", 1)
    quoted = urllib.parse.quote(path, safe="/")
    item = api("GET", f"/repos/{owner}/{name}/contents/{quoted}", token, query={"ref": ref})
    if not isinstance(item, dict) or item.get("type") != "file" or item.get("sha") != expected_blob:
        raise VerificationError(f"source blob is missing or differs from reviewed pin: {repo}:{path}")


def post_status(repo: str, subject: str, token: str, state: str, description: str, target: str | None) -> None:
    owner, name = repo.split("/", 1)
    payload: dict[str, Any] = {"state": state, "context": STATUS_CONTEXT, "description": description[:140]}
    if isinstance(target, str) and target.startswith("https://"):
        payload["target_url"] = target
    api("POST", f"/repos/{owner}/{name}/statuses/{subject}", token, body=payload)


def download_artifact_zip(repo: str, artifact_id: int, token: str) -> bytes:
    """Download an artifact archive without forwarding the API token to its signed URL."""
    owner, name = repo.split("/", 1)
    url = f"{API}/repos/{owner}/{name}/actions/artifacts/{artifact_id}/zip"
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "luminous-security-audit-verifier/1",
    })

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect())
    try:
        opener.open(request, timeout=25)
        raise VerificationError("artifact API did not return its expected download redirect")
    except urllib.error.HTTPError as exc:
        if exc.code != 302:
            detail = exc.read(1200).decode("utf-8", errors="replace")
            raise VerificationError(f"artifact download API returned HTTP {exc.code}: {detail}") from exc
        signed_url = exc.headers.get("Location")
        if not signed_url or not signed_url.startswith("https://"):
            raise VerificationError("artifact download API omitted a valid HTTPS signed URL") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise VerificationError(f"artifact download API unavailable: {exc}") from exc
    try:
        # Intentionally no Authorization header on the signed storage URL.
        with urllib.request.urlopen(urllib.request.Request(signed_url, headers={
            "User-Agent": "luminous-security-audit-verifier/1"}), timeout=60) as response:
            content = response.read(100 * 1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise VerificationError(f"artifact archive download failed: {exc}") from exc
    if len(content) > 100 * 1024 * 1024:
        raise VerificationError("artifact archive exceeds the 100 MiB verifier limit")
    return content


def validate_verdict(verdict: Any, repo: str, policy: dict[str, Any], run: dict[str, Any],\n                    expected_pr_number: int | None = None) -> None:
    if not isinstance(verdict, dict):
        raise VerificationError("verdict artifact is not a JSON object")
    expected_keys = {
        "schema", "repository", "subject_sha", "audit_engine_sha", "workflow_ref", "workflow_sha",
        "workflow_run_url", "generated_at_utc", "aggregate_artifact_retention_days", "run_id",
        "run_attempt", "workflow_job_result", "rustsec_job_result", "npm_job_result", "audit_rust",
        "audit_node", "required_lanes", "non_blocking_findings_present", "non_blocking_finding_sources",
        "status", "failure_reasons", "evidence_files",
    }
    if set(verdict) != expected_keys:
        missing = sorted(expected_keys - set(verdict))
        extra = sorted(set(verdict) - expected_keys)
        raise VerificationError(f"verdict fields differ from the v1 contract (missing={missing}, extra={extra})")

    subject = sha(run.get("head_sha"), "run.head_sha")
    attempt = run.get("run_attempt")
    required = verdict.get("required_lanes")
    if verdict.get("schema") != "luminous.security-audit.verdict.v1":
        raise VerificationError("verdict schema is unsupported")
    if verdict.get("repository") != repo or verdict.get("subject_sha") != subject:
        raise VerificationError("verdict repository/subject does not match the authoritative run")
    if verdict.get("run_id") != str(run.get("id")) or verdict.get("run_attempt") != str(attempt):
        raise VerificationError("verdict run ID/attempt does not match the authoritative run")
    if verdict.get("workflow_job_result") != "success":
        raise VerificationError("verdict does not record a successful workflow-security lane")

    workflow_ref = verdict.get("workflow_ref")
    expected_ref_prefix = f"{repo}/{policy['workflow_path']}@"
    if not isinstance(workflow_ref, str) or not workflow_ref.startswith(expected_ref_prefix):
        raise VerificationError("verdict workflow_ref does not name the policy-expected caller workflow")
    ref_suffix = workflow_ref[len(expected_ref_prefix):]
    ref_match = re.fullmatch(r"refs/pull/([1-9][0-9]*)/merge", ref_suffix)
    if ref_match is None:
        raise VerificationError("verdict workflow_ref is not the expected pull-request merge ref")
    if expected_pr_number is not None and int(ref_match.group(1)) != expected_pr_number:
        raise VerificationError("verdict workflow_ref PR number differs from the independently matched open PR")
    sha(verdict.get("workflow_sha"), "verdict.workflow_sha")
    if verdict.get("workflow_run_url") != run.get("html_url") or not isinstance(run.get("html_url"), str):
        raise VerificationError("verdict run URL does not match the authoritative GitHub run")

    generated = verdict.get("generated_at_utc")
    if not isinstance(generated, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", generated
    ):
        raise VerificationError("verdict timestamp is not canonical UTC")
    try:
        generated_at = datetime.fromisoformat(generated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VerificationError("verdict timestamp is not a valid UTC date-time") from exc
    now = datetime.now(timezone.utc)
    if generated_at > now + VERDICT_FUTURE_SKEW:
        raise VerificationError("verdict timestamp is too far in the future")
    if now - generated_at > VERDICT_MAX_AGE:
        raise VerificationError("verdict evidence is older than the seven-day freshness limit")
    if verdict.get("aggregate_artifact_retention_days") != 30:
        raise VerificationError("verdict retention contract differs from the v1 policy")

    if verdict.get("audit_rust") is not policy["audit_rust"] or verdict.get("audit_node") is not policy["audit_node"]:
        raise VerificationError("verdict requested-lane flags differ from the base-owned coverage policy")
    if not isinstance(required, dict) or required.get("workflow_security") != "PASS":
        raise VerificationError("workflow-security lane is not PASS")
    expected_rust = "PASS" if policy["audit_rust"] else "SKIPPED_NOT_REQUESTED"
    expected_node = "PASS" if policy["audit_node"] else "SKIPPED_NOT_REQUESTED"
    if required.get("rustsec") != expected_rust or required.get("npm") != expected_node:
        raise VerificationError("one or more required dependency-audit lanes are absent or not PASS")
    if policy["audit_rust"] and verdict.get("rustsec_job_result") != "success":
        raise VerificationError("RustSec job result is not success although the lane is required")
    if not policy["audit_rust"] and verdict.get("rustsec_job_result") != "skipped":
        raise VerificationError("RustSec job must be skipped when Rust auditing was not requested")
    if policy["audit_node"] and verdict.get("npm_job_result") != "success":
        raise VerificationError("npm job result is not success although the lane is required")
    if not policy["audit_node"] and verdict.get("npm_job_result") != "skipped":
        raise VerificationError("npm job must be skipped when npm auditing was not requested")
    if policy.get("engine_sha") and verdict.get("audit_engine_sha") != policy["engine_sha"]:
        raise VerificationError("verdict engine SHA differs from the trusted immutable engine pin")
    if not policy.get("engine_sha"):
        sha(verdict.get("audit_engine_sha"), "verdict.audit_engine_sha")
    if verdict.get("status") not in {"PASS", "PASS_WITH_FINDINGS"}:
        raise VerificationError(f"producer verdict is {verdict.get('status')!r}, not a passing verdict")
    reasons = verdict.get("failure_reasons")
    sources = verdict.get("non_blocking_finding_sources")
    nonblocking = verdict.get("non_blocking_findings_present")
    if not isinstance(reasons, list) or reasons:
        raise VerificationError("passing verdict has non-empty or malformed failure reasons")
    if not isinstance(sources, list) or not all(x in {"rustsec_warnings", "npm_below_threshold"} for x in sources):
        raise VerificationError("verdict has malformed non-blocking finding sources")
    if nonblocking is not (len(sources) > 0):
        raise VerificationError("verdict non-blocking finding flag and source list disagree")
    if verdict["status"] == "PASS" and nonblocking:
        raise VerificationError("PASS verdict incorrectly suppresses non-blocking findings")
    if verdict["status"] == "PASS_WITH_FINDINGS" and not nonblocking:
        raise VerificationError("PASS_WITH_FINDINGS verdict has no recorded finding source")
    if policy["audit_rust"] and "rustsec_warnings" in sources and required["rustsec"] != "PASS":
        raise VerificationError("RustSec warning source is inconsistent with the RustSec lane")
    if policy["audit_node"] and "npm_below_threshold" in sources and required["npm"] != "PASS":
        raise VerificationError("npm finding source is inconsistent with the npm lane")


def validate_evidence_manifest(zf: zipfile.ZipFile, entries: list[zipfile.ZipInfo], verdict: dict[str, Any]) -> None:
    records = verdict.get("evidence_files")
    if not isinstance(records, list) or not records:
        raise VerificationError("verdict evidence manifest is missing or empty")
    expected: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise VerificationError("evidence manifest entry is not an object")
        path, digest = record.get("path"), record.get("sha256")
        if (not isinstance(path, str) or not path or path.startswith("/")
                or "\\" in path or any(part in {"", ".", ".."} for part in path.split("/"))
                or path == "verdict.json"):
            raise VerificationError("evidence manifest contains an unsafe or non-canonical path")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise VerificationError(f"evidence manifest has an invalid digest for {path}")
        if path in expected:
            raise VerificationError(f"evidence manifest duplicates path {path}")
        expected[path] = digest

    actual: dict[str, zipfile.ZipInfo] = {}
    for entry in entries:
        if entry.is_dir():
            continue
        path = entry.filename.replace("\\", "/")
        if path.startswith("audit-evidence/"):
            path = path[len("audit-evidence/"):]
        if path == "verdict.json":
            continue
        if path in actual:
            raise VerificationError(f"artifact ZIP duplicates evidence path {path}")
        actual[path] = entry
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise VerificationError(f"evidence manifest coverage mismatch (missing={missing}, extra={extra})")
    for path, expected_digest in expected.items():
        actual_digest = hashlib.sha256(zf.read(actual[path])).hexdigest()
        if actual_digest != expected_digest:
            raise VerificationError(f"evidence file digest mismatch for {path}")


def verify_verdict_artifact(repo: str, policy: dict[str, Any], run: dict[str, Any], token: str,\n                            expected_pr_number: int | None = None) -> None:
    owner, name = repo.split("/", 1)
    run_id = run["id"]
    subject = sha(run.get("head_sha"), "run.head_sha")
    expected_name = f"security-audit-{name}-{subject}-verdict"
    payload = api("GET", f"/repos/{owner}/{name}/actions/runs/{run_id}/artifacts", token,
                  query={"per_page": 100, "page": 1})
    if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), list):
        raise VerificationError("artifact listing response has an unexpected shape")
    matches = [item for item in payload["artifacts"] if item.get("name") == expected_name]
    if len(matches) != 1:
        raise VerificationError(f"expected exactly one aggregate verdict artifact, found {len(matches)}")
    artifact = matches[0]
    artifact_run = artifact.get("workflow_run")
    if not isinstance(artifact_run, dict):
        raise VerificationError("verdict artifact is missing authoritative workflow-run metadata")
    if artifact.get("expired") is not False or artifact_run.get("id") != run_id:
        raise VerificationError("verdict artifact is expired or belongs to a different workflow run")
    if artifact_run.get("head_sha") != subject:
        raise VerificationError("artifact metadata head SHA differs from the exact run subject")
    digest = artifact.get("digest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise VerificationError("artifact metadata has no canonical SHA-256 digest")
    artifact_id = artifact.get("id")
    if not isinstance(artifact_id, int) or artifact_id <= 0:
        raise VerificationError("artifact ID is invalid")
    archive = download_artifact_zip(repo, artifact_id, token)
    if "sha256:" + hashlib.sha256(archive).hexdigest() != digest:
        raise VerificationError("downloaded verdict artifact digest does not match GitHub artifact metadata")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            entries = zf.infolist()
            if len(entries) == 0 or len(entries) > 4096:
                raise VerificationError("verdict artifact contains an invalid number of entries")
            total_size = 0
            verdict_names = []
            for entry in entries:
                path = entry.filename.replace("\\", "/")
                parts = path.split("/")
                if path.startswith("/") or ".." in parts or any(ord(ch) < 32 for ch in path):
                    raise VerificationError("artifact ZIP contains an unsafe path")
                total_size += entry.file_size
                if total_size > 50 * 1024 * 1024:
                    raise VerificationError("artifact ZIP uncompressed size exceeds 50 MiB")
                if not entry.is_dir() and parts[-1] == "verdict.json":
                    if entry.filename != "verdict.json":
                        raise VerificationError("verdict.json must be stored at the artifact ZIP root")
                    verdict_names.append(entry.filename)
            if len(verdict_names) != 1:
                raise VerificationError("artifact does not contain exactly one verdict.json")
            info = zf.getinfo(verdict_names[0])
            if info.file_size > 1024 * 1024:
                raise VerificationError("verdict.json exceeds the 1 MiB parsing limit")
            verdict = json.loads(zf.read(info), object_pairs_hook=reject_duplicate_object_keys)
            if not isinstance(verdict, dict):
                raise VerificationError("verdict artifact is not a JSON object")
            validate_evidence_manifest(zf, entries, verdict)
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"verdict artifact ZIP/JSON is invalid: {type(exc).__name__}: {exc}") from exc
    validate_verdict(verdict, repo, policy, run, expected_pr_number)


def should_publish_pending_status(mode: str, activity: str, run: dict[str, Any]) -> bool:
    """Prevent a delayed start event from overwriting the final status."""
    return (mode == "workflow_run"
            and activity in {"requested", "in_progress"}
            and run.get("status") != "completed")


def process(repo: str, policy: dict[str, Any], run_id: int, token: str, mode: str,
            activity: str, expected_sha: str | None, expected_attempt: int | None,
            default_branch: str) -> int:
    owner, name = repo.split("/", 1)
    run = api("GET", f"/repos/{owner}/{name}/actions/runs/{run_id}", token)
    if not isinstance(run, dict) or run.get("id") != run_id:
        raise VerificationError("run lookup returned missing or mismatched run ID")
    subject = sha(run.get("head_sha"), "run.head_sha")
    if expected_sha is not None and subject != expected_sha:
        raise VerificationError("run SHA differs from triggering event")
    branch = str(run.get("head_branch", ""))
    pr = find_pr(repo, subject, branch, default_branch, token)
    if pr is None:
        print("No open same-repository PR targets the default branch; no merge status applies.")
        return 0
    subject = check_binding(run, repo, policy, branch, subject, expected_attempt)
    runs_payload = api("GET", f"/repos/{owner}/{name}/actions/workflows/{policy['workflow_id']}/runs", token,
                       query={"head_sha": subject, "event": "pull_request", "per_page": 100, "page": 1})
    if not isinstance(runs_payload, dict) or not isinstance(runs_payload.get("workflow_runs"), list):
        raise VerificationError("latest-run API response has an unexpected shape")
    if not is_latest(run, runs_payload["workflow_runs"], policy, subject):
        print("Run/attempt superseded by a newer exact-head attempt; status left unchanged.")
        return 0

    target = run.get("html_url")
    # A failed invariant suite must actively replace an old green status.
    self_test_outcome = os.environ.get("VERIFIER_TEST_OUTCOME", "success")
    if self_test_outcome != "success":
        post_status(repo, subject, token, "failure",
                    f"Independent verifier self-tests did not pass: {self_test_outcome}", target)
        print("FAIL: independent verifier self-tests did not pass", file=sys.stderr)
        return 1
    if should_publish_pending_status(mode, activity, run):
        post_status(repo, subject, token, "pending", "Exact-head security audit is running; no pass is implied.", target)
        print(f"PENDING: {repo}@{subject} run {run_id}.")
        return 0

    failure = None
    try:
        if run.get("status") != "completed":
            raise VerificationError("workflow run is not completed")
        if run.get("conclusion") != "success":
            raise VerificationError(f"producer workflow conclusion is {run.get('conclusion')!r}, not success")
        check_blob(repo, policy["workflow_path"], subject, policy["workflow_blob"], token)
        if policy.get("engine_sha"):
            check_blob(ENGINE_REPO, ENGINE_PATH, ENGINE_SHA, ENGINE_BLOB, token)
            if policy["engine_sha"] != ENGINE_SHA:
                raise VerificationError("caller engine commit differs from trusted policy pin")
        else:
            check_blob(repo, ENGINE_PATH, subject, ENGINE_BLOB, token)
        pr_number = pr.get("number")
        if not isinstance(pr_number, int) or pr_number <= 0:
            raise VerificationError("matched open PR has an invalid number")
        verify_verdict_artifact(repo, policy, run, token, expected_pr_number=pr_number)
    except VerificationError as exc:
        failure = str(exc)

    if failure is None:
        post_status(repo, subject, token, "success",
                    "Exact run/attempt, PR, caller workflow blob, engine and verdict artifact verified.", target)
        print(f"PASS: independently verified {repo}@{subject}, run {run_id}, attempt {run.get('run_attempt')}.")
        return 0
    post_status(repo, subject, token, "failure", f"Independent security audit verification failed: {failure}", target)
    print(f"FAIL: {failure}", file=sys.stderr)
    return 1


def best_effort_failure_status(repo: str, subject: str, token: str, reason: str,
                                target: str | None = None, *, workflow_id: int | None = None,
                                event_name: str | None = None, repository_id: str | None = None,
                                head_repository_id: str | None = None, current_repository_id: str | None = None,
                                pull_request_base_ref: str | None = None, default_branch: str = "main") -> bool:
    """Clear stale success only for an event matching the expected same-repo audit source."""
    if repo not in POLICY or not token or not re.fullmatch(r"[0-9a-f]{40}", subject):
        return False
    if workflow_id != POLICY[repo]["workflow_id"] or event_name != "pull_request":
        return False
    if pull_request_base_ref != default_branch:
        return False
    if (not repository_id or not head_repository_id or not current_repository_id
            or repository_id != head_repository_id or repository_id != current_repository_id):
        return False
    try:
        post_status(repo, subject, token, "failure",
                    f"Independent security audit verifier could not qualify this run: {reason}", target)
        return True
    except Exception as exc:
        print(f"Unable to publish fail-closed commit status: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    repo = os.environ.get("REPOSITORY", "").strip()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    mode = os.environ.get("TRUST_ANCHOR_MODE", "").strip()
    default_branch = os.environ.get("DEFAULT_BRANCH", "main").strip() or "main"
    try:
        if repo not in POLICY:
            raise VerificationError(f"no base-owned policy for repository {repo!r}")
        if not token:
            raise VerificationError("GITHUB_TOKEN is required")
        if mode not in {"workflow_run", "workflow_dispatch"}:
            raise VerificationError(f"unsupported verifier mode {mode!r}")
        if mode == "workflow_dispatch":
            raw_id = os.environ.get("MANUAL_WORKFLOW_RUN_ID", "").strip()
            if not raw_id.isdigit() or int(raw_id) <= 0:
                raise VerificationError("manual replay requires a positive workflow_run_id")
            run_id, expected_sha, attempt = int(raw_id), None, None
            activity = "manual_replay"
        else:
            raw_id = os.environ.get("TRIGGER_RUN_ID", "").strip()
            if not raw_id.isdigit() or int(raw_id) <= 0:
                raise VerificationError("workflow_run event has no positive run ID")
            run_id = int(raw_id)
            expected_sha = sha(os.environ.get("TRIGGER_RUN_HEAD_SHA", "").strip(), "event.head_sha")
            raw_attempt = os.environ.get("TRIGGER_RUN_ATTEMPT", "").strip()
            if not raw_attempt.isdigit() or int(raw_attempt) <= 0:
                raise VerificationError("workflow_run event has no positive run attempt")
            attempt = int(raw_attempt)
            activity = os.environ.get("TRIGGER_ACTIVITY_TYPE", "").strip()
        return process(repo, POLICY[repo], run_id, token, mode, activity,
                       expected_sha, attempt, default_branch)
    except Exception as exc:
        # Normalize malformed API payloads and unexpected verifier defects as
        # failures too. Never let an unhandled ordinary exception preserve a
        # stale green status when the trusted event identity is available.
        message = str(exc) if isinstance(exc, VerificationError) else f"unexpected {type(exc).__name__}: {exc}"
        if mode == "workflow_run":
            subject = os.environ.get("TRIGGER_RUN_HEAD_SHA", "").strip()
            raw_id = os.environ.get("TRIGGER_RUN_ID", "").strip()
            target = (f"https://github.com/{repo}/actions/runs/{raw_id}"
                      if raw_id.isdigit() and repo in POLICY else None)
            raw_workflow_id = os.environ.get("TRIGGER_RUN_WORKFLOW_ID", "").strip()
            workflow_id = int(raw_workflow_id) if raw_workflow_id.isdigit() else None
            best_effort_failure_status(
                repo, subject, token, message, target,
                workflow_id=workflow_id,
                event_name=os.environ.get("TRIGGER_RUN_EVENT", "").strip(),
                repository_id=os.environ.get("TRIGGER_RUN_REPOSITORY_ID", "").strip(),
                head_repository_id=os.environ.get("TRIGGER_RUN_HEAD_REPOSITORY_ID", "").strip(),
                current_repository_id=os.environ.get("CURRENT_REPOSITORY_ID", "").strip(),
                pull_request_base_ref=os.environ.get("TRIGGER_RUN_PR_BASE_REF", "").strip(),
                default_branch=default_branch,
            )
        if isinstance(exc, VerificationError):
            raise
        raise VerificationError(message) from exc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"INCOMPLETE: {exc}", file=sys.stderr)
        raise SystemExit(2)
