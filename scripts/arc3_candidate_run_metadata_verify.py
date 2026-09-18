#!/usr/bin/env python3
"""Verify saved GitHub run/jobs/artifacts metadata for ARC3 candidate evidence.

This script performs no network access. A trusted phase is expected to fetch the
three GitHub REST JSON documents and pass them here as untrusted data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

GIT_HEX = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")

WORKFLOW_PATH = ".github/workflows/arc3-protocol-qualify-slim-v3.yml"
JOB_NAME = "ARC3 protocol full exact-head qualification"
ARTIFACT_PREFIX = "arc3-protocol-slim-v3-candidate-"

REQUIRED_STEPS = [
    "Checkout exact candidate recipe",
    "Validate helper and exact subject binding",
    "Checkout exact product subject",
    "Bind exact product and record preflight identity",
    "Independent protocol oracle precheck",
    "Install Rust 1.96",
    "Record Rust toolchain",
    "Require exact committed Cargo.lock resolution",
    "Check affected formatting",
    "Test protocol crate",
    "Strict Clippy protocol crate",
    "Check psych-bench integration",
    "Test psych-bench compatibility namespace",
    "Independent protocol oracle postcheck",
    "Verify helper and subject immutability",
    "Emit candidate qualification receipt",
    "Upload candidate qualification receipt",
]


class MetadataError(ValueError):
    pass


def fail(message: str) -> None:
    raise MetadataError(message)


def load_json(path: str, label: str):
    raw = Path(path).read_bytes()
    if len(raw) > 2 * 1024 * 1024:
        fail(f"{label} JSON exceeds 2 MiB cap")
    try:
        return json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid {label} JSON: {exc}")


def require_equal(observed, expected, label: str) -> None:
    if observed != expected:
        fail(f"{label} mismatch: expected {expected!r}, observed {observed!r}")


def require_git_hex(value: str, label: str) -> None:
    if not isinstance(value, str) or GIT_HEX.fullmatch(value) is None:
        fail(f"{label} must be lowercase 40- or 64-hex Git identity")


def verify_run(run: dict, args: argparse.Namespace) -> None:
    require_equal(run.get("id"), int(args.expected_run_id), "run.id")
    require_equal(run.get("run_attempt"), int(args.expected_run_attempt), "run.run_attempt")
    require_equal(run.get("event"), args.expected_event, "run.event")
    require_equal(run.get("status"), "completed", "run.status")
    require_equal(run.get("conclusion"), "success", "run.conclusion")
    require_equal(run.get("head_sha"), args.expected_helper_sha, "run.head_sha")
    require_equal(run.get("head_branch"), args.expected_head_branch, "run.head_branch")
    require_equal(run.get("path"), WORKFLOW_PATH, "run.path")
    require_equal(run.get("workflow_id"), int(args.expected_workflow_id), "run.workflow_id")

    repository = run.get("repository") or {}
    require_equal(repository.get("full_name"), args.expected_repository, "run.repository.full_name")

    pulls = run.get("pull_requests")
    if not isinstance(pulls, list):
        fail("run.pull_requests must be a list")
    matching = [pr for pr in pulls if pr.get("number") == int(args.expected_pr_number)]
    if len(matching) != 1:
        fail("run does not bind exactly once to the expected PR number")

    require_git_hex(run.get("head_sha"), "run.head_sha")


def verify_jobs(jobs_doc: dict, args: argparse.Namespace) -> int:
    jobs = jobs_doc.get("jobs")
    if not isinstance(jobs, list):
        fail("jobs document must contain a jobs list")

    matching = [job for job in jobs if job.get("name") == JOB_NAME]
    if len(matching) != 1:
        fail(f"expected exactly one {JOB_NAME!r} job; observed {len(matching)}")
    job = matching[0]

    require_equal(job.get("run_id"), int(args.expected_run_id), "job.run_id")
    require_equal(job.get("status"), "completed", "job.status")
    require_equal(job.get("conclusion"), "success", "job.conclusion")

    labels = job.get("labels")
    if not isinstance(labels, list) or args.expected_runner_label not in labels:
        fail(f"job labels do not contain expected runner label {args.expected_runner_label!r}")

    steps = job.get("steps")
    if not isinstance(steps, list):
        fail("job.steps must be a list")

    names = [step.get("name") for step in steps]
    if len(names) != len(set(names)):
        fail("job contains duplicate step names")

    for name in REQUIRED_STEPS:
        matches = [step for step in steps if step.get("name") == name]
        if len(matches) != 1:
            fail(f"required step {name!r} observed {len(matches)} times")
        step = matches[0]
        require_equal(step.get("status"), "completed", f"step {name}.status")
        require_equal(step.get("conclusion"), "success", f"step {name}.conclusion")

    job_id = job.get("id")
    if not isinstance(job_id, int) or job_id <= 0:
        fail("job.id must be a positive integer")
    return job_id


def verify_artifacts(artifacts_doc: dict, args: argparse.Namespace) -> tuple[int, str, int]:
    artifacts = artifacts_doc.get("artifacts")
    if not isinstance(artifacts, list):
        fail("artifacts document must contain an artifacts list")

    expected_name = ARTIFACT_PREFIX + args.expected_subject_sha
    matching = [artifact for artifact in artifacts if artifact.get("name") == expected_name]
    if len(matching) != 1:
        fail(f"expected exactly one artifact {expected_name!r}; observed {len(matching)}")
    artifact = matching[0]

    require_equal(artifact.get("expired"), False, "artifact.expired")
    digest = artifact.get("digest")
    if not isinstance(digest, str) or SHA256_DIGEST.fullmatch(digest) is None:
        fail("artifact.digest must be sha256:<64 lowercase hex>")

    size = artifact.get("size_in_bytes")
    if not isinstance(size, int) or size <= 0 or size > 1024 * 1024:
        fail("artifact.size_in_bytes must be within (0, 1 MiB]")

    artifact_id = artifact.get("id")
    if not isinstance(artifact_id, int) or artifact_id <= 0:
        fail("artifact.id must be a positive integer")

    workflow_run = artifact.get("workflow_run")
    if workflow_run is not None:
        if not isinstance(workflow_run, dict):
            fail("artifact.workflow_run must be an object when present")
        require_equal(workflow_run.get("id"), int(args.expected_run_id), "artifact.workflow_run.id")
        head_sha = workflow_run.get("head_sha")
        if head_sha is not None:
            require_equal(head_sha, args.expected_helper_sha, "artifact.workflow_run.head_sha")

    return artifact_id, digest, size


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-json", required=True)
    p.add_argument("--jobs-json", required=True)
    p.add_argument("--artifacts-json", required=True)
    p.add_argument("--expected-repository", required=True)
    p.add_argument("--expected-run-id", required=True)
    p.add_argument("--expected-run-attempt", required=True)
    p.add_argument("--expected-workflow-id", required=True)
    p.add_argument("--expected-event", default="pull_request")
    p.add_argument("--expected-head-branch", required=True)
    p.add_argument("--expected-helper-sha", required=True)
    p.add_argument("--expected-pr-number", required=True)
    p.add_argument("--expected-subject-sha", required=True)
    p.add_argument("--expected-runner-label", default="ubuntu-slim")
    return p


def verify(args: argparse.Namespace) -> tuple[int, int, str, int]:
    run = load_json(args.run_json, "run")
    jobs = load_json(args.jobs_json, "jobs")
    artifacts = load_json(args.artifacts_json, "artifacts")

    if not isinstance(run, dict) or not isinstance(jobs, dict) or not isinstance(artifacts, dict):
        fail("run/jobs/artifacts top-level JSON values must be objects")

    verify_run(run, args)
    job_id = verify_jobs(jobs, args)
    artifact_id, artifact_digest, artifact_size = verify_artifacts(artifacts, args)
    return job_id, artifact_id, artifact_digest, artifact_size


def main() -> int:
    try:
        job_id, artifact_id, artifact_digest, artifact_size = verify(parser().parse_args())
    except (OSError, MetadataError, ValueError) as exc:
        print(f"ARC3 candidate run metadata verification FAILED: {exc}", file=sys.stderr)
        return 1

    print("arc3_candidate_run_metadata_verification=PASS")
    print(f"job_id={job_id}")
    print(f"artifact_id={artifact_id}")
    print(f"artifact_digest={artifact_digest}")
    print(f"artifact_size={artifact_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
