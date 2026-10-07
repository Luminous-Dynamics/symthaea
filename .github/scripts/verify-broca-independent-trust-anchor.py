#!/usr/bin/env python3
"""Independent, base-owned Broca qualification verifier.

This script is executed only from the default branch by workflow_run. PR source
is fetched as API data and is never checked out, imported, built, or executed.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_VERSION = "2026-03-10"
STATUS_CONTEXT = "Broca / Independent Trust Anchor"
RECEIPT_PATH = Path("BROCA_INDEPENDENT_TRUST_ANCHOR_RECEIPT_V1.json")
POLICY_PATH = Path("docs/broca/independent_trust_policy_v1.json")

ALLOWED_PATH_PREFIXES = (
    ".github/workflows/broca-feature-matrix.yml",
    "crates/domains/symthaea-broca/",
    "docs/broca/",
    "src/voice/live_voice.rs",
)

EXPECTED_ACTION_REFS = [
    "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
    "dtolnay/rust-toolchain@7e38f4b43b4db5c8dd498af069a4f6196df1d067",
    "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830",
    "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
    "dtolnay/rust-toolchain@7e38f4b43b4db5c8dd498af069a4f6196df1d067",
    "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830",
    "actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6",
    "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
]

EXPECTED_BROCA_JOBS = [
    "symthaea-broca (no-default-features)",
    "symthaea-broca (default)",
    "symthaea-broca (mamba-cpu)",
    "symthaea-broca (test-helpers)",
    "symthaea-broca (canonical-eval)",
    "symthaea-broca (canonical-quality-gate)",
    "symthaea-broca (root-live-voice-ssm-language)",
    "UniMorph frozen snapshot audit",
]

EXPECTED_MATRIX_STEPS = [
    "Verify runner-provided Rust bootstrap",
    "Verify qualification action pins",
    "Install Rust toolchain",
    "Cache cargo registry and target",
    "Verify locked dependency graph",
    "Capture native toolchain context",
    "Install system dependencies",
    "Capture native package context",
    "Run feature check",
]

EXPECTED_FREEZE_STEPS = [
    "Verify runner-provided Rust bootstrap",
    "Verify qualification action pins",
    "Install Rust toolchain",
    "Cache cargo registry and target",
    "Verify locked dependency graph",
    "Capture native toolchain context",
    "Install system dependencies",
    "Capture native package context",
    "Audit frozen UniMorph snapshot",
    "Attest structured freeze receipt",
    "Preserve attestation bundle",
    "Upload frozen UniMorph audit evidence",
]

REQUIRED_WORKFLOWS = {
    "Broca Feature Matrix": ".github/workflows/broca-feature-matrix.yml",
    "Workflow Syntax": ".github/workflows/workflow-syntax.yml",
    "PR Governance": ".github/workflows/pr-governance.yml",
}


class VerificationError(Exception):
    pass


class WaitingError(VerificationError):
    pass


class StaleError(VerificationError):
    pass


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise VerificationError(f"missing required environment variable: {name}")
    return value


TOKEN = env_required("GITHUB_TOKEN")
REPOSITORY = env_required("REPOSITORY")
REPOSITORY_ID = int(env_required("REPOSITORY_ID"))
TRUST_ANCHOR_SHA = env_required("TRUST_ANCHOR_SHA")
TRUST_ANCHOR_MODE = env_required("TRUST_ANCHOR_MODE")
TRIGGER_RUN_ID = int(os.environ.get("TRIGGER_RUN_ID", "0") or "0")
TRIGGER_RUN_NAME = os.environ.get("TRIGGER_RUN_NAME", "").strip()
TRIGGER_RUN_EVENT = os.environ.get("TRIGGER_RUN_EVENT", "").strip()
TRIGGER_ACTIVITY_TYPE = os.environ.get("TRIGGER_ACTIVITY_TYPE", "").strip()
TRIGGER_RUN_REPOSITORY_ID = int(os.environ.get("TRIGGER_RUN_REPOSITORY_ID", "0") or "0")
TRIGGER_RUN_HEAD_SHA = os.environ.get("TRIGGER_RUN_HEAD_SHA", "").strip()
TRIGGER_RUN_HEAD_BRANCH = os.environ.get("TRIGGER_RUN_HEAD_BRANCH", "").strip()
TRIGGER_RUN_CONCLUSION = os.environ.get("TRIGGER_RUN_CONCLUSION", "").strip()
TRIGGER_RUN_ATTEMPT = int(os.environ.get("TRIGGER_RUN_ATTEMPT", "0") or "0")


def api_request(
    method: str,
    path: str,
    *,
    query: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    url = f"https://api.github.com/repos/{REPOSITORY}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {TOKEN}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "symthaea-broca-independent-trust-anchor/1",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise VerificationError(
            f"GitHub API {method} {path} failed: HTTP {error.code}: {detail}"
        ) from error
    if not raw:
        return None
    return json.loads(raw)


def get_file(path: str, ref: str) -> tuple[bytes, str]:
    encoded = urllib.parse.quote(path, safe="")
    data = api_request("GET", f"/contents/{encoded}", query={"ref": ref})
    if isinstance(data, list):
        raise VerificationError(f"expected file but API returned directory: {path}")
    if data.get("encoding") != "base64":
        raise VerificationError(
            f"unexpected content encoding for {path}: {data.get('encoding')}"
        )
    return base64.b64decode(data["content"], validate=True), str(data["sha"])


def require_fragments(text: str, fragments: list[str], label: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise VerificationError(
            f"{label} missing required fragments: {', '.join(repr(x) for x in missing)}"
        )


def list_head_runs(head_sha: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for page in range(1, 11):
        page_runs = api_request(
            "GET",
            "/actions/runs",
            query={
                "head_sha": head_sha,
                "event": "pull_request",
                "per_page": "100",
                "page": str(page),
            },
        ).get("workflow_runs", [])
        if not isinstance(page_runs, list):
            raise VerificationError(
                f"unexpected workflow-run response on page {page}"
            )
        runs.extend(page_runs)
        if len(page_runs) < 100:
            return runs
    raise VerificationError(
        "workflow-run enumeration exceeded the independent verifier's 1000-run safety bound"
    )


def list_pr_files(pr_number: int) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for page in range(1, 31):
        page_files = api_request(
            "GET",
            f"/pulls/{pr_number}/files",
            query={"per_page": "100", "page": str(page)},
        )
        if not isinstance(page_files, list):
            raise VerificationError(f"unexpected pull-request file-list response on page {page}")
        files.extend(page_files)
        if len(page_files) < 100:
            return files
    raise VerificationError("pull-request file list exceeded the independent verifier's 3000-file safety bound")


def list_commit_pull_requests(commit_sha: str) -> list[dict[str, Any]]:
    pull_requests: list[dict[str, Any]] = []
    for page in range(1, 11):
        page_pull_requests = api_request(
            "GET",
            f"/commits/{commit_sha}/pulls",
            query={"per_page": "100", "page": str(page)},
        )
        if not isinstance(page_pull_requests, list):
            raise VerificationError(
                f"unexpected commit pull-request response on page {page}"
            )
        pull_requests.extend(page_pull_requests)
        if len(page_pull_requests) < 100:
            return pull_requests
    raise VerificationError(
        "commit pull-request enumeration exceeded the independent verifier's 1000-PR safety bound"
    )


def latest_required_runs(head_sha: str) -> dict[str, dict[str, Any] | None]:
    runs = list_head_runs(head_sha)
    result: dict[str, dict[str, Any] | None] = {}
    for name in REQUIRED_WORKFLOWS:
        expected_path = REQUIRED_WORKFLOWS[name]
        candidates = [
            run
            for run in runs
            if run.get("name") == name
            and run.get("path") == expected_path
            and run.get("head_sha") == head_sha
        ]
        candidates.sort(
            key=lambda run: (
                run.get("updated_at", ""),
                run.get("id", 0),
                run.get("run_attempt", 0),
            )
        )
        result[name] = candidates[-1] if candidates else None
    return result


def verify_trigger_is_current(
    trigger_run: dict[str, Any],
    latest_runs: dict[str, dict[str, Any] | None],
) -> None:
    if trigger_run.get("name") not in REQUIRED_WORKFLOWS:
        raise VerificationError(
            f"unexpected triggering workflow: {trigger_run.get('name')!r}"
        )
    latest = latest_runs.get(trigger_run["name"])
    if latest is None:
        raise WaitingError(
            f"triggering workflow has no current exact-head run: {trigger_run['name']}"
        )
    same_execution = (
        latest.get("id") == trigger_run.get("id")
        and latest.get("run_attempt") == trigger_run.get("run_attempt")
        and latest.get("head_sha") == trigger_run.get("head_sha")
    )
    if same_execution:
        return
    if latest.get("status") != "completed":
        raise WaitingError(
            f"newer exact-head {trigger_run['name']} run {latest.get('id')} is not completed"
        )
    raise StaleError(
        f"triggering run {trigger_run.get('id')} attempt {trigger_run.get('run_attempt')} "
        f"is superseded by exact-head run {latest.get('id')} attempt {latest.get('run_attempt')}"
    )


def list_run_jobs(run_id: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for page in range(1, 11):
        page_jobs = api_request(
            "GET",
            f"/actions/runs/{run_id}/jobs",
            query={"filter": "latest", "per_page": "100", "page": str(page)},
        ).get("jobs", [])
        if not isinstance(page_jobs, list):
            raise VerificationError(
                f"unexpected workflow-job response on page {page}"
            )
        jobs.extend(page_jobs)
        if len(page_jobs) < 100:
            return jobs
    raise VerificationError(
        "workflow-job enumeration exceeded the independent verifier's 1000-job safety bound"
    )


def verify_broca_jobs(run_id: int) -> dict[str, Any]:
    jobs = list_run_jobs(run_id)

    outcomes: dict[str, Any] = {}
    for name in EXPECTED_BROCA_JOBS:
        matches = [job for job in jobs if job.get("name") == name]
        if len(matches) != 1:
            raise VerificationError(
                f"Broca job contract mismatch for {name!r}: expected one job, got {len(matches)}"
            )

        job = matches[0]
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            raise VerificationError(
                f"Broca job {name!r} is not successfully completed: "
                f"status={job.get('status')!r}, conclusion={job.get('conclusion')!r}"
            )

        expected_steps = (
            EXPECTED_FREEZE_STEPS if name == "UniMorph frozen snapshot audit"
            else EXPECTED_MATRIX_STEPS
        )
        step_matches: dict[str, list[dict[str, Any]]] = {}
        for step_name in expected_steps:
            step_matches[step_name] = [
                step for step in (job.get("steps") or [])
                if step.get("name") == step_name
            ]
            if len(step_matches[step_name]) != 1:
                raise VerificationError(
                    f"Broca job {name!r} step contract mismatch for {step_name!r}: "
                    f"expected one step, got {len(step_matches[step_name])}"
                )
            step = step_matches[step_name][0]
            if step.get("status") != "completed" or step.get("conclusion") != "success":
                raise VerificationError(
                    f"Broca job {name!r} critical step {step_name!r} did not complete successfully: "
                    f"status={step.get('status')!r}, conclusion={step.get('conclusion')!r}"
                )

        outcomes[name] = {
            "job_id": job.get("id"),
            "status": job.get("status"),
            "conclusion": job.get("conclusion"),
            "steps": {
                step_name: {
                    "status": step_matches[step_name][0].get("status"),
                    "conclusion": step_matches[step_name][0].get("conclusion"),
                    "number": step_matches[step_name][0].get("number"),
                    "completed_at": step_matches[step_name][0].get("completed_at"),
                }
                for step_name in expected_steps
            },
        }

    if len(jobs) != len(EXPECTED_BROCA_JOBS):
        raise VerificationError(
            f"Broca job-count mismatch: expected {len(EXPECTED_BROCA_JOBS)}, got {len(jobs)}"
        )

    return {
        "expected": EXPECTED_BROCA_JOBS,
        "observed_count": len(jobs),
        "outcomes": outcomes,
    }


def load_policy() -> dict[str, Any]:
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"unable to load independent trust policy: {error}") from error
    if policy.get("schema_version") != "broca-independent-trust-policy-v1":
        raise VerificationError("independent trust policy schema mismatch")
    if policy.get("repository") != REPOSITORY:
        raise VerificationError("independent trust policy repository mismatch")
    return policy


def approved_snapshot(
    policy: dict[str, Any],
    pr: dict[str, Any],
    pr_files: list[dict[str, Any]],
) -> dict[str, Any]:
    approved = policy.get("approved_files")
    if not isinstance(approved, list) or not approved:
        raise VerificationError("independent trust policy has no approved file snapshot")

    expected_keys = {
        (item.get("path"), item.get("status"), item.get("blob_sha"))
        for item in approved
        if isinstance(item, dict)
    }
    if len(expected_keys) != len(approved):
        raise VerificationError(
            "independent trust policy contains malformed duplicate file entries"
        )

    actual_keys = {
        (item.get("filename"), item.get("status"), item.get("sha"))
        for item in pr_files
    }
    if actual_keys != expected_keys:
        raise VerificationError(
            f"approved PR snapshot mismatch: expected {sorted(expected_keys)!r}, got {sorted(actual_keys)!r}"
        )

    if pr.get("number") != policy.get("pull_request"):
        raise VerificationError(
            "PR number does not match independently approved snapshot"
        )
    if pr.get("base", {}).get("ref") != policy.get("base_branch"):
        raise VerificationError(
            "PR base branch does not match independently approved snapshot"
        )
    if pr.get("base", {}).get("sha") != policy.get("base_sha"):
        raise VerificationError(
            "PR base SHA does not match independently approved snapshot"
        )
    if pr.get("head", {}).get("sha") != policy.get("approved_head_sha"):
        raise VerificationError(
            "PR head SHA does not match independently approved snapshot"
        )

    return {
        "schema_version": policy.get("schema_version"),
        "pull_request": policy.get("pull_request"),
        "base_branch": policy.get("base_branch"),
        "base_sha": policy.get("base_sha"),
        "approved_head_sha": policy.get("approved_head_sha"),
        "approved_files": approved,
    }
def post_status(sha: str, state: str, description: str, target_url: str) -> None:
    api_request(
        "POST",
        f"/statuses/{sha}",
        body={
            "state": state,
            "context": STATUS_CONTEXT,
            "description": description[:140],
            "target_url": target_url,
        },
    )


def trusted_file_blob(path: str) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{path}"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(
            f"unable to resolve trusted verifier file blob for {path}: {error}"
        ) from error
    if not re.fullmatch(r"[0-9a-f]{40}", output):
        raise VerificationError(
            f"trusted verifier file blob for {path} is not canonical: {output!r}"
        )
    return output


def trusted_checkout_identity() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(
            f"unable to resolve trusted verifier checkout HEAD: {error}"
        ) from error

    if output != TRUST_ANCHOR_SHA:
        raise VerificationError(
            f"trusted verifier checkout mismatch: workflow SHA {TRUST_ANCHOR_SHA}, checkout HEAD {output}"
        )

    return output


def main() -> int:
    global TRIGGER_RUN_ID, TRIGGER_RUN_NAME, TRIGGER_RUN_EVENT, TRIGGER_ACTIVITY_TYPE
    global TRIGGER_RUN_REPOSITORY_ID, TRIGGER_RUN_HEAD_SHA, TRIGGER_RUN_HEAD_BRANCH
    global TRIGGER_RUN_CONCLUSION, TRIGGER_RUN_ATTEMPT

    verifier_head = trusted_checkout_identity()
    policy_blob = trusted_file_blob(POLICY_PATH.as_posix())
    target_url = ""

    receipt: dict[str, Any] = {
        "schema_version": "broca-independent-trust-anchor-receipt-v1",
        "qualification_result": "NOT_PASS",
        "trust_anchor": {
            "commit_sha": verifier_head,
            "policy_path": POLICY_PATH.as_posix(),
            "policy_blob_sha": policy_blob,
            "workflow_name": "Broca Independent Trust Anchor",
            "status_context": STATUS_CONTEXT,
        },
        "trigger": {
            "repository_id": TRIGGER_RUN_REPOSITORY_ID,
            "activity_type": TRIGGER_ACTIVITY_TYPE,
            "run_id": TRIGGER_RUN_ID,
            "run_attempt": TRIGGER_RUN_ATTEMPT,
            "run_name": TRIGGER_RUN_NAME,
            "event": TRIGGER_RUN_EVENT,
            "head_branch": TRIGGER_RUN_HEAD_BRANCH,
            "head_sha": TRIGGER_RUN_HEAD_SHA,
            "conclusion": TRIGGER_RUN_CONCLUSION,
        },
        "verification": {},
    }

    try:
        if TRUST_ANCHOR_MODE == "workflow_dispatch":
            manual_run_id = int(env_required("MANUAL_WORKFLOW_RUN_ID"))
            manual_run = api_request("GET", f"/actions/runs/{manual_run_id}")
            TRIGGER_RUN_ID = manual_run_id
            TRIGGER_RUN_NAME = str(manual_run.get("name", ""))
            TRIGGER_RUN_EVENT = str(manual_run.get("event", ""))
            TRIGGER_RUN_REPOSITORY_ID = int(manual_run.get("repository", {}).get("id") or 0)
            TRIGGER_RUN_HEAD_SHA = str(manual_run.get("head_sha", ""))
            TRIGGER_RUN_HEAD_BRANCH = str(manual_run.get("head_branch", ""))
            TRIGGER_RUN_CONCLUSION = str(manual_run.get("conclusion") or "")
            TRIGGER_RUN_ATTEMPT = int(manual_run.get("run_attempt") or 0)
            TRIGGER_ACTIVITY_TYPE = "completed"
            if (
                TRIGGER_RUN_REPOSITORY_ID != REPOSITORY_ID
                or TRIGGER_RUN_EVENT != "pull_request"
                or TRIGGER_RUN_NAME not in REQUIRED_WORKFLOWS
                or not TRIGGER_RUN_HEAD_SHA
                or not TRIGGER_RUN_HEAD_BRANCH
                or not TRIGGER_RUN_ATTEMPT
            ):
                raise StaleError(
                    "manual workflow-run target is not a valid same-repository pull-request run"
                )
        else:
            if TRIGGER_RUN_EVENT != "pull_request":
                raise StaleError(
                    f"triggering event is not pull_request: {TRIGGER_RUN_EVENT!r}"
                )
            if TRIGGER_ACTIVITY_TYPE not in {"requested", "in_progress", "completed"}:
                raise StaleError(
                    f"unexpected workflow_run activity type: {TRIGGER_ACTIVITY_TYPE!r}"
                )

            trigger_run = api_request("GET", f"/actions/runs/{TRIGGER_RUN_ID}")
            if (
                trigger_run.get("repository", {}).get("id") != REPOSITORY_ID
                or TRIGGER_RUN_REPOSITORY_ID != REPOSITORY_ID
                or trigger_run.get("id") != TRIGGER_RUN_ID
                or trigger_run.get("name") != TRIGGER_RUN_NAME
                or trigger_run.get("event") != TRIGGER_RUN_EVENT
                or trigger_run.get("head_sha") != TRIGGER_RUN_HEAD_SHA
                or trigger_run.get("head_branch") != TRIGGER_RUN_HEAD_BRANCH
                or trigger_run.get("run_attempt") != TRIGGER_RUN_ATTEMPT
                or (
                    TRIGGER_ACTIVITY_TYPE == "completed"
                    and trigger_run.get("conclusion") != TRIGGER_RUN_CONCLUSION
                )
                or not trigger_run.get("workflow_id")
            ):
                raise StaleError(
                    "workflow_run event payload does not match authoritative GitHub run state"
                )

        target_url = f"https://github.com/{REPOSITORY}/actions/runs/{TRIGGER_RUN_ID}"
        receipt["trigger"] = {
            "repository_id": TRIGGER_RUN_REPOSITORY_ID,
            "activity_type": TRIGGER_ACTIVITY_TYPE,
            "run_id": TRIGGER_RUN_ID,
            "run_attempt": TRIGGER_RUN_ATTEMPT,
            "run_name": TRIGGER_RUN_NAME,
            "event": TRIGGER_RUN_EVENT,
            "head_branch": TRIGGER_RUN_HEAD_BRANCH,
            "head_sha": TRIGGER_RUN_HEAD_SHA,
            "conclusion": TRIGGER_RUN_CONCLUSION,
        }

        if TRIGGER_RUN_EVENT != "pull_request":
            raise StaleError(
                f"triggering event is not pull_request: {TRIGGER_RUN_EVENT!r}"
            )

        if TRIGGER_ACTIVITY_TYPE in {"requested", "in_progress"}:
            receipt["qualification_result"] = "WAITING"
            receipt["verification"]["error"] = (
                f"required workflow entered {TRIGGER_ACTIVITY_TYPE}; "
                "clearing any prior trust status until completion is independently verified"
            )
            post_status(
                TRIGGER_RUN_HEAD_SHA,
                "pending",
                "Independent Broca trust anchor waiting on workflow completion",
                target_url,
            )
            print(
                f"WAITING: required workflow activity is {TRIGGER_ACTIVITY_TYPE}",
                file=sys.stderr,
            )
            return 0

        associated = list_commit_pull_requests(TRIGGER_RUN_HEAD_SHA)
        candidates = [
            pr
            for pr in associated
            if pr.get("state") == "open"
            and pr.get("head", {}).get("sha") == TRIGGER_RUN_HEAD_SHA
            and pr.get("head", {}).get("repo", {}).get("full_name") == REPOSITORY
            and pr.get("head", {}).get("repo", {}).get("id") == REPOSITORY_ID
            and pr.get("base", {}).get("repo", {}).get("full_name") == REPOSITORY
            and pr.get("base", {}).get("repo", {}).get("id") == REPOSITORY_ID
        ]

        if len(candidates) != 1:
            raise StaleError(
                f"expected exactly one open same-repository PR for head {TRIGGER_RUN_HEAD_SHA}, got {len(candidates)}"
            )

        if TRIGGER_RUN_CONCLUSION != "success":
            raise VerificationError(
                f"triggering workflow run did not succeed: {TRIGGER_RUN_CONCLUSION!r}"
            )

        pr_number = int(candidates[0]["number"])
        pr = api_request("GET", f"/pulls/{pr_number}")
        if (
            pr.get("state") != "open"
            or pr.get("draft") is not False
            or pr.get("head", {}).get("sha") != TRIGGER_RUN_HEAD_SHA
            or pr.get("head", {}).get("ref") != TRIGGER_RUN_HEAD_BRANCH
            or pr.get("head", {}).get("repo", {}).get("full_name") != REPOSITORY
            or pr.get("head", {}).get("repo", {}).get("id") != REPOSITORY_ID
            or pr.get("base", {}).get("repo", {}).get("id") != REPOSITORY_ID
        ):
            raise StaleError("PR state/head/repository changed after association lookup")

        receipt["pull_request"] = {
            "number": pr_number,
            "base_branch": pr.get("base", {}).get("ref"),
            "base_sha": pr.get("base", {}).get("sha"),
            "head_branch": pr.get("head", {}).get("ref"),
            "head_sha": pr.get("head", {}).get("sha"),
            "head_repo_full_name": pr.get("head", {}).get("repo", {}).get("full_name"),
            "merge_commit_sha": pr.get("merge_commit_sha"),
        }

        policy = load_policy()
        pr_files = list_pr_files(pr_number)
        changed_files = sorted(
            (item.get("filename"), item.get("status"))
            for item in pr_files
        )

        snapshot = approved_snapshot(policy, pr, pr_files)

        relevant_files = [
            name
            for name, _status in changed_files
            if name
            and any(
                name == prefix
                or (prefix.endswith("/") and name.startswith(prefix))
                for prefix in ALLOWED_PATH_PREFIXES
            )
        ]
        if not relevant_files:
            receipt["qualification_result"] = "NOT_APPLICABLE"
            receipt["verification"]["source_scope"] = {
                "allowed_prefixes": list(ALLOWED_PATH_PREFIXES),
                "changed_files": [
                    {"filename": name, "status": status}
                    for name, status in changed_files
                ],
                "verified": False,
                "relevant": False,
            }
            print("NOT_APPLICABLE: PR does not touch Broca qualification scope")
            return 0

        disallowed = [
            name
            for name, _status in changed_files
            if name
            and not any(
                name == prefix
                or (prefix.endswith("/") and name.startswith(prefix))
                for prefix in ALLOWED_PATH_PREFIXES
            )
        ]
        if disallowed:
            raise VerificationError(
                f"PR changed files outside the Broca trust-anchor allowlist: {disallowed}"
            )

        unsupported_statuses = [
            (name, status)
            for name, status in changed_files
            if status not in {"added", "modified"}
        ]
        if unsupported_statuses:
            raise VerificationError(
                f"PR contains deleted/renamed files outside the allowed model: {unsupported_statuses}"
            )

        receipt["verification"]["source_scope"] = {
            "allowed_prefixes": list(ALLOWED_PATH_PREFIXES),
            "changed_files": [
                {"filename": name, "status": status}
                for name, status in changed_files
            ],
            "verified": True,
        }
        receipt["verification"]["approved_snapshot"] = snapshot

        workflow_bytes, workflow_blob = get_file(
            ".github/workflows/broca-feature-matrix.yml", pr["head"]["sha"]
        )
        workflow = workflow_bytes.decode("utf-8")

        uses = [
            match.group(1)
            for line in workflow.splitlines()
            if (match := re.match(r"^\s*(?:-\s*)?uses:\s+(\S+)", line))
        ]
        if uses != EXPECTED_ACTION_REFS:
            raise VerificationError(
                f"Broca workflow Action refs mismatch: expected {EXPECTED_ACTION_REFS!r}, got {uses!r}"
            )

        if "pull_request_target" in workflow or "workflow_run" in workflow:
            raise VerificationError(
                "Broca Feature Matrix contains an elevated/cascading trust trigger"
            )
        if re.search(r"\bsecrets\.", workflow):
            raise VerificationError(
                "Broca Feature Matrix directly references repository secrets"
            )

        require_fragments(
            workflow,
            [
                "permissions:\n  contents: read",
                "permissions:\n      contents: read\n      id-token: write\n      attestations: write\n      artifact-metadata: write",
                "BROCA_QUALIFICATION_HEAD_SHA: ${{ github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha }}",
                "ref: ${{ env.BROCA_QUALIFICATION_HEAD_SHA }}",
                "toolchain: 1.96.0",
                "runs-on: ubuntu-24.04",
                "Verify runner-provided Rust bootstrap",
                "command -v rustup",
                "BROCA_RUSTUP_VERSION=$(rustup --version)",
                "persist-credentials: false",
                "Verify locked dependency graph",
                "cargo fetch --locked",
                "Capture native toolchain context",
                "BROCA_NATIVE_TOOLCHAIN_CONTEXT<<EOF",
                "cc --version | head -n 1",
                "clang --version | head -n 1",
                "ld --version | head -n 1",
                "ldd --version | head -n 1",
                "Capture native package context",
                "BROCA_NATIVE_PACKAGE_CONTEXT",
                "BROCA_FREEZE_AUDIT_OUTPUT: target/broca-unimorph-freeze/receipt.json",
                "Attest structured freeze receipt",
                "Preserve attestation bundle",
                "receipt.attestation.bundle.json",
            ],
            "Broca workflow",
        )
        workflow_fragments = [
            "${{ env.BROCA_QUALIFICATION_HEAD_SHA }}"
        ]
        if workflow.count(workflow_fragments[0]) != 2:
            raise VerificationError(
                "Broca workflow does not use the exact PR head in both jobs"
            )
        if workflow.count("toolchain: 1.96.0") != 2:
            raise VerificationError(
                "Broca workflow Rust toolchain selection mismatch"
            )
        if workflow.count("persist-credentials: false") != 2:
            raise VerificationError(
                "Broca workflow checkout credential policy mismatch"
            )
        if workflow.count("Verify locked dependency graph") != 2:
            raise VerificationError(
                "Broca workflow dependency admission count mismatch"
            )
        if workflow.count("Capture native toolchain context") != 2:
            raise VerificationError(
                "Broca workflow native toolchain capture count mismatch"
            )
        if workflow.count("Capture native package context") != 2:
            raise VerificationError(
                "Broca workflow native package capture count mismatch"
            )

        audit_bytes, audit_blob = get_file(
            "crates/domains/symthaea-broca/src/bin/broca_unimorph_freeze_audit.rs",
            pr["head"]["sha"],
        )
        audit = audit_bytes.decode("utf-8")
        require_fragments(
            audit,
            [
                "verify_qualification_checkout",
                "git rev-parse HEAD",
                "write_structured_receipt",
                "qualification_workflow_git_blob_sha",
                "qualification_action_refs",
                "EXPECTED_QUALIFICATION_ACTION_REFS: [&str; 8]",
                "frozen UniMorph artifact is not valid UTF-8",
                "EXPECTED_LANGUAGE_TAG",
                "EXPECTED_DIALECT_SCOPE",
                "EXPECTED_RULE_SET_ID",
                "EXPECTED_COMPILER_ID",
                "EXPECTED_COMPILER_VERSION",
                "EXPECTED_NORMALIZATION_POLICY",
            ],
            "freeze auditor",
        )

        build_bytes, build_blob = get_file(
            "crates/domains/symthaea-broca/build.rs", pr["head"]["sha"]
        )
        build = build_bytes.decode("utf-8")
        require_fragments(
            build,
            [
                "symthaea-broca-unimorph-compiler-build-context-revision-v9",
                "BROCA_RUSTUP_VERSION",
                "BROCA_NATIVE_TOOLCHAIN_CONTEXT",
                "rerun-if-env-changed=BROCA_NATIVE_TOOLCHAIN_CONTEXT",
                "CC",
                "CXX",
                "RUSTC_LINKER",
                "PKG_CONFIG_PATH",
                "OPENSSL_DIR",
                "LIBCLANG_PATH",
                "CMAKE_TOOLCHAIN_FILE",
                "VULKAN_SDK",
                "BROCA_NATIVE_PACKAGE_CONTEXT",
                "rerun-if-env-changed=BROCA_NATIVE_PACKAGE_CONTEXT",
                "BROCA_RUSTUP_VERSION",
                "rerun-if-env-changed=BROCA_RUSTUP_VERSION",
                "RUNNER_OS",
                "RUNNER_ARCH",
                "ImageOS",
                "ImageVersion",
            ],
            "Broca build context",
        )

        manifest_bytes, manifest_blob = get_file(
            "docs/broca/unimorph_eng_4_selection_manifest.json", pr["head"]["sha"]
        )
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        expected_namespace = {
            "language_tag": "en",
            "dialect_scope": "en-unspecified",
            "rule_set_id": "unimorph-eng-4",
            "compilation_provenance": "unimorph-eng-4-selection",
            "compiler_id": "symthaea-unimorph-tsv-compiler",
            "compiler_version": "broca-unimorph-tsv-compiler-v1",
            "normalization_policy": "trim-one-line-ending-sort-feature-tokens-sort-output-rules-v1",
        }
        for key, expected in expected_namespace.items():
            if manifest.get(key) != expected:
                raise VerificationError(
                    f"selection manifest namespace mismatch for {key}: expected {expected!r}, got {manifest.get(key)!r}"
                )

        receipt["verification"]["critical_source_files"] = {
            ".github/workflows/broca-feature-matrix.yml": {
                "blob_sha": workflow_blob,
                "byte_length": len(workflow_bytes),
            },
            "crates/domains/symthaea-broca/src/bin/broca_unimorph_freeze_audit.rs": {
                "blob_sha": audit_blob,
                "byte_length": len(audit_bytes),
            },
            "crates/domains/symthaea-broca/build.rs": {
                "blob_sha": build_blob,
                "byte_length": len(build_bytes),
            },
            "docs/broca/unimorph_eng_4_selection_manifest.json": {
                "blob_sha": manifest_blob,
                "byte_length": len(manifest_bytes),
                "namespace": expected_namespace,
            },
        }

        workflow_runs = latest_required_runs(pr["head"]["sha"])
        verify_trigger_is_current(trigger_run, workflow_runs)
        workflow_gate_states: dict[str, Any] = {}
        for name, run in workflow_runs.items():
            if run is None:
                raise WaitingError(
                    f"required workflow has not run for exact head: {name}"
                )
            workflow_gate_states[name] = {
                "id": run.get("id"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "head_sha": run.get("head_sha"),
                "updated_at": run.get("updated_at"),
            }
            if run.get("status") != "completed":
                raise WaitingError(
                    f"required workflow is not completed for exact head: {name}"
                )
            if run.get("head_sha") != pr["head"]["sha"]:
                raise VerificationError(
                    f"required workflow head mismatch for {name}: {run.get('head_sha')} != {pr['head']['sha']}"
                )
            if run.get("conclusion") != "success":
                raise VerificationError(
                    f"required workflow is not successful for exact head: {name} -> {run.get('conclusion')}"
                )

        broca_run = workflow_runs["Broca Feature Matrix"]
        assert broca_run is not None
        receipt["verification"]["workflow_gates"] = workflow_gate_states
        receipt["verification"]["broca_jobs"] = verify_broca_jobs(int(broca_run["id"]))

        # Reconcile the control-plane state immediately before publishing success.
        # A new exact-head run must invalidate the earlier observation rather than
        # leaving a successful status attached to an execution that has been superseded.
        final_workflow_runs = latest_required_runs(pr["head"]["sha"])
        verify_trigger_is_current(trigger_run, final_workflow_runs)
        final_gate_states: dict[str, Any] = {}
        for name, run in final_workflow_runs.items():
            if run is None:
                raise WaitingError(
                    f"required workflow disappeared for exact head during final reconciliation: {name}"
                )
            final_gate_states[name] = {
                "id": run.get("id"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "head_sha": run.get("head_sha"),
                "updated_at": run.get("updated_at"),
            }
            if run.get("status") != "completed":
                raise WaitingError(
                    f"required workflow became incomplete during final reconciliation: {name}"
                )
            if run.get("head_sha") != pr["head"]["sha"]:
                raise VerificationError(
                    f"required workflow head changed during final reconciliation for {name}"
                )
            if run.get("conclusion") != "success":
                raise VerificationError(
                    f"required workflow ceased to be successful during final reconciliation: {name} -> {run.get('conclusion')}"
                )

        receipt["verification"]["final_reconciliation"] = {
            "workflow_gates": final_gate_states,
            "verified": True,
        }
        broca_run = final_workflow_runs["Broca Feature Matrix"]
        assert broca_run is not None
        receipt["verification"]["workflow_gates"] = final_gate_states
        receipt["verification"]["broca_jobs"] = verify_broca_jobs(int(broca_run["id"]))

        workflow_paths = REQUIRED_WORKFLOWS

        for supporting_name in ("Workflow Syntax", "PR Governance"):
            supporting_path = workflow_paths[supporting_name]
            trusted_bytes, trusted_blob = get_file(supporting_path, TRUST_ANCHOR_SHA)
            pr_bytes, pr_blob = get_file(supporting_path, pr["head"]["sha"])
            if trusted_blob != pr_blob or trusted_bytes != pr_bytes:
                raise VerificationError(
                    f"{supporting_name} workflow differs from trusted base-owned definition"
                )
            receipt["verification"].setdefault("supporting_workflows", {})[
                supporting_name
            ] = {
                "path": supporting_path,
                "trusted_blob_sha": trusted_blob,
                "pr_blob_sha": pr_blob,
                "verified": True,
            }
        expected_path = workflow_paths[TRIGGER_RUN_NAME]
        if trigger_run.get("path") != expected_path:
            raise VerificationError(
                f"triggering workflow path mismatch: expected {expected_path!r}, got {trigger_run.get('path')!r}"
            )

        receipt["qualification_result"] = "PASS"
        post_status(
            pr["head"]["sha"],
            "success",
            "Independent Broca trust anchor passed",
            target_url,
        )

    except StaleError as error:
        receipt["qualification_result"] = "STALE"
        receipt["verification"]["error"] = str(error)
        print(f"STALE: {error}", file=sys.stderr)
    except WaitingError as error:
        receipt["qualification_result"] = "WAITING"
        receipt["verification"]["error"] = str(error)
        post_status(
            TRIGGER_RUN_HEAD_SHA,
            "pending",
            "Independent Broca trust anchor waiting on required gates",
            target_url,
        )
        print(f"WAITING: {error}", file=sys.stderr)
    except Exception as error:
        receipt["qualification_result"] = "NOT_PASS"
        receipt["verification"]["error"] = str(error)
        post_status(
            TRIGGER_RUN_HEAD_SHA,
            "failure",
            "Independent Broca trust anchor rejected",
            target_url,
        )
        print(f"NOT_PASS: {error}", file=sys.stderr)
    finally:
        RECEIPT_PATH.write_text(
            json.dumps(
                {
                    **receipt,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
