#!/usr/bin/env python3
"""Offline policy verifier for ci-draft-global-admission-v0.

This verifies the staged source/patch identities and the intended concurrency truth
table without modifying ci.yml or scheduling GitHub Actions. It is deliberately
not a substitute for GitHub workflow-parser qualification.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
PATCH = ROOT / "docs/qualification-admissions/ci-draft-global-admission-v0.patch"
MANIFEST = ROOT / "docs/qualification-admissions/ci-draft-global-admission-v0.json"

EXPECTED_WORKFLOW_BLOB = "a48366076b30eb8e12d22c927a3b8bf333181409"
EXPECTED_PATCH_BLOB = "763b3ef4d2def578e4859c315c2f196e33acc321"
EXPECTED_SOURCE_COMMIT = "4bad8af72ff775e7c869b6df83faba718a339a36"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def model_group(event: str, ref: str, draft: bool = False, run_id: str = "") -> tuple[str, bool]:
    base = "draft-global" if event == "pull_request" and draft else ref
    suffix = run_id if event == "workflow_dispatch" else "auto"
    group = f"CI-{base}-{suffix}"
    cancel = not (event == "pull_request" and draft)
    return group, cancel


def main() -> int:
    require(WORKFLOW.is_file(), f"missing {WORKFLOW}")
    require(PATCH.is_file(), f"missing {PATCH}")
    require(MANIFEST.is_file(), f"missing {MANIFEST}")

    workflow_blob = git("hash-object", str(WORKFLOW.relative_to(ROOT)))
    patch_blob = git("hash-object", str(PATCH.relative_to(ROOT)))
    tracked_workflow_blob = git("rev-parse", f"HEAD:{WORKFLOW.relative_to(ROOT)}")
    require(workflow_blob == EXPECTED_WORKFLOW_BLOB, f"worktree ci.yml drifted: {workflow_blob}")
    require(tracked_workflow_blob == EXPECTED_WORKFLOW_BLOB, f"tracked ci.yml drifted: {tracked_workflow_blob}")
    require(patch_blob == EXPECTED_PATCH_BLOB, f"patch blob drifted: {patch_blob}")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest["schema"] == "symthaea.ci-draft-global-admission-patch.v0", "unexpected manifest schema")
    require(manifest["status"] == "staged-not-applied", "staged artifact claims applied status")
    require(manifest["source_commit"] == EXPECTED_SOURCE_COMMIT, "source commit mismatch")
    require(manifest["source_blob"] == EXPECTED_WORKFLOW_BLOB, "manifest workflow blob mismatch")

    patch = PATCH.read_text()
    require(patch.count("diff --git ") == 1, "patch must contain exactly one file diff")
    require("diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml" in patch, "patch targets unexpected file")
    require("+    types: [opened, synchronize, reopened, ready_for_review]" in patch, "ready_for_review trigger absent")
    require("&& 'draft-global' || github.ref" in patch, "draft-global group expression absent")
    require("+  cancel-in-progress: ${{ github.event_name != 'pull_request' || github.event.pull_request.draft != true }}" in patch, "conditional cancellation absent")
    require("-  cancel-in-progress: true" in patch, "old unconditional cancellation is not replaced")

    cases = [
        ("draft-synchronize", model_group("pull_request", "refs/pull/10/merge", True), ("CI-draft-global-auto", False)),
        ("ready-for-review", model_group("pull_request", "refs/pull/10/merge", False), ("CI-refs/pull/10/merge-auto", True)),
        ("ready-synchronize", model_group("pull_request", "refs/pull/10/merge", False), ("CI-refs/pull/10/merge-auto", True)),
        ("main-push", model_group("push", "refs/heads/main"), ("CI-refs/heads/main-auto", True)),
        ("manual", model_group("workflow_dispatch", "refs/heads/topic", run_id="123"), ("CI-refs/heads/topic-123", True)),
        ("schedule", model_group("schedule", "refs/heads/main"), ("CI-refs/heads/main-auto", True)),
    ]
    for name, actual, expected in cases:
        require(actual == expected, f"{name}: expected {expected!r}, got {actual!r}")

    policy = manifest["policy"]
    require(policy["draft_group_cancel_in_progress"] is False, "manifest draft cancellation mismatch")
    require(policy["ready_for_review_triggers_broad_ci"] is True, "manifest ready transition mismatch")
    require(policy["focused_qualification_workflows_unchanged"] is True, "manifest focused-lane boundary mismatch")

    print("ci_draft_global_admission_v0_policy=PASS")
    print(f"source_workflow_blob={workflow_blob}")
    print(f"patch_blob={patch_blob}")
    print("cases=6")
    print("claim_boundary=offline_policy_model_not_github_parser_qualification")
    return 0


if __name__ == "__main__":
    sys.exit(main())
