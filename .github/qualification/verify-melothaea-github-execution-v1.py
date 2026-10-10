#!/usr/bin/env python3
"""Fail-closed validator for captured GitHub Actions execution evidence.

The semantic receipt verifier proves receipt contents. This validator proves that
an independently captured GitHub Actions API record has the expected identity.
It is intentionally separate from receipt semantics and never treats an
in-progress execution as a completed qualification.
"""

from __future__ import annotations

import argparse
import json
import re
import sys


HEX40 = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_REPOSITORY = "Luminous-Dynamics/symthaea"
DEFAULT_WORKFLOW_ID = 369356198
DEFAULT_WORKFLOW_PATH = ".github/workflows/qual-melothaea-prog-suite-tonal-chain-v4-verifier-tests.yml"
DEFAULT_JOB_NAME = "Fail-closed receipt verifier mutation suite"
DEFAULT_STEP_NAME = "Run adversarial verifier tests"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha40(value: object, field: str) -> str:
    require(isinstance(value, str) and HEX40.fullmatch(value), f"{field} is not lowercase 40-hex")
    return value


def validate(evidence: dict, *, expected_head_sha: str, require_completed: bool) -> None:
    require(evidence.get("repository") == DEFAULT_REPOSITORY, "repository mismatch")
    require(evidence.get("workflow_id") == DEFAULT_WORKFLOW_ID, "workflow id mismatch")
    require(evidence.get("workflow_path") == DEFAULT_WORKFLOW_PATH, "workflow path mismatch")
    require(evidence.get("job_name") == DEFAULT_JOB_NAME, "job name mismatch")
    require(evidence.get("run_id", 0) > 0, "run id must be positive")
    require(evidence.get("run_attempt", 0) > 0, "run attempt must be positive")
    sha40(evidence.get("run_head_sha"), "run_head_sha")
    sha40(evidence.get("job_head_sha"), "job_head_sha")
    require(evidence["run_head_sha"] == expected_head_sha, "run head SHA mismatch")
    require(evidence["job_head_sha"] == expected_head_sha, "job head SHA mismatch")
    require(evidence.get("job_run_id") == evidence.get("run_id"), "job/run identity mismatch")
    require(evidence.get("required_step") == DEFAULT_STEP_NAME, "required step mismatch")
    require(evidence.get("run_status") in {"queued", "in_progress", "completed"}, "invalid run status")
    require(evidence.get("job_status") in {"queued", "in_progress", "completed"}, "invalid job status")

    if require_completed:
        require(evidence.get("run_status") == "completed", "run is not completed")
        require(evidence.get("run_conclusion") == "success", "run is not successful")
        require(evidence.get("job_status") == "completed", "job is not completed")
        require(evidence.get("job_conclusion") == "success", "job is not successful")
        require(evidence.get("required_step_status") == "completed", "required step is not completed")
        require(evidence.get("required_step_conclusion") == "success", "required step is not successful")
    else:
        require(evidence.get("required_step_status") == "completed", "required step has not completed")
        require(evidence.get("required_step_conclusion") == "success", "required step is not successful")

    if "artifact_run_id" in evidence:
        require(evidence["artifact_run_id"] == evidence["run_id"], "artifact/run identity mismatch")
    if "artifact_head_sha" in evidence:
        require(evidence["artifact_head_sha"] == expected_head_sha, "artifact head SHA mismatch")


def self_test() -> int:
    base = {
        "repository": DEFAULT_REPOSITORY,
        "workflow_id": DEFAULT_WORKFLOW_ID,
        "workflow_path": DEFAULT_WORKFLOW_PATH,
        "run_id": 36570163177,
        "run_attempt": 1,
        "run_status": "completed",
        "run_conclusion": "success",
        "run_head_sha": "a" * 40,
        "job_run_id": 36570163177,
        "job_name": DEFAULT_JOB_NAME,
        "job_status": "completed",
        "job_conclusion": "success",
        "job_head_sha": "a" * 40,
        "required_step": DEFAULT_STEP_NAME,
        "required_step_status": "completed",
        "required_step_conclusion": "success",
        "artifact_run_id": 36570163177,
        "artifact_head_sha": "a" * 40,
    }
    validate(base, expected_head_sha="a" * 40, require_completed=True)

    mutations = {
        "wrong run id": {"run_id": 1},
        "wrong attempt": {"run_attempt": 0},
        "wrong workflow": {"workflow_id": 1},
        "wrong path": {"workflow_path": "wrong.yml"},
        "wrong run SHA": {"run_head_sha": "b" * 40},
        "wrong job SHA": {"job_head_sha": "b" * 40},
        "wrong job/run binding": {"job_run_id": 1},
        "wrong job name": {"job_name": "other"},
        "failed run": {"run_conclusion": "failure"},
        "failed job": {"job_conclusion": "failure"},
        "failed step": {"required_step_conclusion": "failure"},
        "wrong artifact run": {"artifact_run_id": 1},
        "wrong artifact SHA": {"artifact_head_sha": "b" * 40},
    }
    for label, mutation in mutations.items():
        candidate = dict(base)
        candidate.update(mutation)
        try:
            validate(candidate, expected_head_sha="a" * 40, require_completed=True)
        except VerificationError:
            continue
        raise VerificationError(f"mutation was accepted: {label}")

    running = dict(base)
    running.update(
        run_status="in_progress",
        run_conclusion=None,
        job_status="in_progress",
        job_conclusion=None,
    )
    validate(running, expected_head_sha="a" * 40, require_completed=False)
    try:
        validate(running, expected_head_sha="a" * 40, require_completed=True)
    except VerificationError:
        pass
    else:
        raise VerificationError("in-progress execution accepted as completed")

    print("PASS: GitHub execution evidence validator self-test")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--evidence", type=argparse.FileType("r"))
    parser.add_argument("--head-sha")
    parser.add_argument("--allow-running", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.evidence or not args.head_sha:
        parser.error("--evidence and --head-sha are required unless --self-test is used")
    evidence = json.load(args.evidence)
    validate(evidence, expected_head_sha=args.head_sha, require_completed=not args.allow_running)
    print("PASS: captured GitHub execution evidence satisfies the expected binding policy")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
