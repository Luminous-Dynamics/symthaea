#!/usr/bin/env python3
"""Classify saved GitHub Actions runner state without mutating remote state.

This tool is intentionally offline and evidence-oriented. It consumes a saved,
normalized fixture containing one workflow-run summary plus job/step
observations and classifies runner-plane state. It does not call GitHub, cancel
jobs, or decide whether a scientific/product subject is qualified.

The strongest run-level statement emitted is
`required_checks_terminal_success`: every explicitly required job in the
fixture was observed as completed/success. Consumers must still bind that
runner evidence to the exact product theorem they are qualifying.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

FIXTURE_SCHEMA = "symthaea-ci-runner-fixture-v1"
OUTPUT_SCHEMA = "symthaea-ci-runner-classification-v1"

COMPLETED_SUCCESS = "CompletedSuccess"
COMPLETED_FAILURE = "CompletedFailure"
COMPLETED_CANCELLED = "CompletedCancelled"
COMPLETED_SKIPPED = "CompletedSkipped"
QUEUED = "Queued"
ACTUALLY_RUNNING = "ActuallyRunning"
SUMMARY_STALE = "SummaryStaleAgainstSteps"
TIMEOUT_INCONSISTENT = "TimeoutInconsistent"
UNKNOWN = "Unknown"

SUCCESS_CONCLUSIONS = {"success"}
FAILURE_CONCLUSIONS = {
    "failure",
    "timed_out",
    "action_required",
    "startup_failure",
}
CANCELLED_CONCLUSIONS = {"cancelled"}
SKIPPED_CONCLUSIONS = {"skipped", "neutral"}
QUEUED_STATUSES = {"queued", "pending", "waiting", "requested"}

ANOMALOUS_CLASSIFICATIONS = {SUMMARY_STALE, TIMEOUT_INCONSISTENT, UNKNOWN}


class FixtureError(ValueError):
    """Raised when a runner-state fixture is structurally invalid."""


@dataclass(frozen=True)
class JobClassification:
    job_id: int | str
    name: str
    required: bool
    runner_class: str
    timeout_minutes: int | None
    started_at: str | None
    summary_status: str
    summary_conclusion: str | None
    classification: str
    reasons: tuple[str, ...]


def parse_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise FixtureError(f"{field_name} must be a non-empty RFC3339 timestamp")

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise FixtureError(f"{field_name} is not valid RFC3339: {value!r}") from exc

    if parsed.tzinfo is None:
        raise FixtureError(f"{field_name} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def normalize_timestamp(value: Any, field_name: str) -> str:
    return parse_timestamp(value, field_name).isoformat()


def require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FixtureError(f"{field_name} must be an object")
    return value


def require_jobs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FixtureError("jobs must be a non-empty array")
    jobs: list[dict[str, Any]] = []
    for index, job in enumerate(value):
        if not isinstance(job, dict):
            raise FixtureError(f"jobs[{index}] must be an object")
        jobs.append(job)
    return jobs


def validate_fixture(fixture: Any) -> tuple[dict[str, Any], list[dict[str, Any]], datetime]:
    root = require_mapping(fixture, "fixture")
    if root.get("schema") != FIXTURE_SCHEMA:
        raise FixtureError(
            f"schema must be {FIXTURE_SCHEMA!r}, got {root.get('schema')!r}"
        )

    observed_at = parse_timestamp(root.get("observed_at"), "observed_at")
    run = require_mapping(root.get("run"), "run")
    jobs = require_jobs(root.get("jobs"))

    for field in (
        "id",
        "workflow_id",
        "workflow_name",
        "run_attempt",
        "status",
        "head_sha",
        "created_at",
        "run_started_at",
        "updated_at",
    ):
        if field not in run:
            raise FixtureError(f"run.{field} is required")

    if not isinstance(run["workflow_name"], str) or not run["workflow_name"]:
        raise FixtureError("run.workflow_name must be a non-empty string")
    if (
        isinstance(run["run_attempt"], bool)
        or not isinstance(run["run_attempt"], int)
        or run["run_attempt"] <= 0
    ):
        raise FixtureError("run.run_attempt must be a positive integer")
    if not isinstance(run["status"], str) or not run["status"]:
        raise FixtureError("run.status must be a non-empty string")
    if not isinstance(run["head_sha"], str) or not run["head_sha"]:
        raise FixtureError("run.head_sha must be a non-empty string")

    for field in ("created_at", "run_started_at", "updated_at"):
        parse_timestamp(run[field], f"run.{field}")

    for index, job in enumerate(jobs):
        for field in ("id", "name", "required", "runner_class", "status"):
            if field not in job:
                raise FixtureError(f"jobs[{index}].{field} is required")
        if not isinstance(job["name"], str) or not job["name"]:
            raise FixtureError(f"jobs[{index}].name must be a non-empty string")
        if not isinstance(job["required"], bool):
            raise FixtureError(f"jobs[{index}].required must be boolean")
        if not isinstance(job["runner_class"], str) or not job["runner_class"]:
            raise FixtureError(
                f"jobs[{index}].runner_class must be a non-empty string"
            )
        if not isinstance(job["status"], str) or not job["status"]:
            raise FixtureError(f"jobs[{index}].status must be a non-empty string")

        timeout = job.get("timeout_minutes")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0
        ):
            raise FixtureError(
                f"jobs[{index}].timeout_minutes must be a positive integer"
            )

        started_at = job.get("started_at")
        if started_at is not None:
            parse_timestamp(started_at, f"jobs[{index}].started_at")

        steps = job.get("steps", [])
        if not isinstance(steps, list):
            raise FixtureError(f"jobs[{index}].steps must be an array when present")
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise FixtureError(
                    f"jobs[{index}].steps[{step_index}] must be an object"
                )
            if "status" not in step or not isinstance(step["status"], str):
                raise FixtureError(
                    f"jobs[{index}].steps[{step_index}].status must be a string"
                )

    return run, jobs, observed_at


def all_observed_steps_terminal(job: dict[str, Any]) -> bool:
    steps = job.get("steps", [])
    return bool(steps) and all(step.get("status") == "completed" for step in steps)


def classify_job(job: dict[str, Any], observed_at: datetime) -> JobClassification:
    status = job["status"]
    conclusion = job.get("conclusion")
    reasons: list[str] = []

    if status == "completed":
        if conclusion in SUCCESS_CONCLUSIONS:
            classification = COMPLETED_SUCCESS
        elif conclusion in FAILURE_CONCLUSIONS:
            classification = COMPLETED_FAILURE
        elif conclusion in CANCELLED_CONCLUSIONS:
            classification = COMPLETED_CANCELLED
        elif conclusion in SKIPPED_CONCLUSIONS:
            classification = COMPLETED_SKIPPED
        else:
            classification = UNKNOWN
            reasons.append(
                f"completed job has unrecognized conclusion {conclusion!r}"
            )
    else:
        terminal_conclusion = conclusion in (
            SUCCESS_CONCLUSIONS
            | FAILURE_CONCLUSIONS
            | CANCELLED_CONCLUSIONS
            | SKIPPED_CONCLUSIONS
        )
        if terminal_conclusion:
            classification = SUMMARY_STALE
            reasons.append("non-terminal job status carries a terminal conclusion")
        elif all_observed_steps_terminal(job):
            classification = SUMMARY_STALE
            reasons.append(
                "job summary is non-terminal while every observed step is completed"
            )
        elif status == "in_progress":
            timeout = job.get("timeout_minutes")
            started_at_raw = job.get("started_at")
            if timeout is not None and started_at_raw is not None:
                started_at = parse_timestamp(started_at_raw, f"job {job['id']} started_at")
                deadline = started_at + timedelta(minutes=timeout)
                if observed_at > deadline:
                    classification = TIMEOUT_INCONSISTENT
                    reasons.append(
                        f"observed after configured {timeout}-minute timeout deadline "
                        f"{deadline.isoformat()}"
                    )
                else:
                    classification = ACTUALLY_RUNNING
            else:
                classification = ACTUALLY_RUNNING
                if timeout is None:
                    reasons.append("no configured timeout supplied in fixture")
                if started_at_raw is None:
                    reasons.append("no started_at supplied in fixture")
        elif status in QUEUED_STATUSES:
            classification = QUEUED
        else:
            classification = UNKNOWN
            reasons.append(f"unrecognized non-terminal status {status!r}")

    started_at = job.get("started_at")
    return JobClassification(
        job_id=job["id"],
        name=job["name"],
        required=job["required"],
        runner_class=job["runner_class"],
        timeout_minutes=job.get("timeout_minutes"),
        started_at=(
            normalize_timestamp(started_at, f"job {job['id']} started_at")
            if started_at is not None
            else None
        ),
        summary_status=status,
        summary_conclusion=conclusion,
        classification=classification,
        reasons=tuple(reasons),
    )


def classify_fixture(fixture: Any) -> dict[str, Any]:
    run, jobs, observed_at = validate_fixture(fixture)
    classified = [classify_job(job, observed_at) for job in jobs]
    required = [job for job in classified if job.required]

    required_checks_terminal_success = bool(required) and all(
        job.classification == COMPLETED_SUCCESS for job in required
    )

    return {
        "schema": OUTPUT_SCHEMA,
        "source_fixture_schema": FIXTURE_SCHEMA,
        "observed_at": observed_at.isoformat(),
        "run": {
            "id": run["id"],
            "workflow_id": run["workflow_id"],
            "workflow_name": run["workflow_name"],
            "run_attempt": run["run_attempt"],
            "status": run["status"],
            "conclusion": run.get("conclusion"),
            "head_sha": run["head_sha"],
            "created_at": normalize_timestamp(run["created_at"], "run.created_at"),
            "run_started_at": normalize_timestamp(
                run["run_started_at"], "run.run_started_at"
            ),
            "updated_at": normalize_timestamp(run["updated_at"], "run.updated_at"),
        },
        "jobs": [asdict(job) for job in classified],
        "required_checks_terminal_success": required_checks_terminal_success,
        "runner_plane_anomaly_count": sum(
            job.classification in ANOMALOUS_CLASSIFICATIONS for job in classified
        ),
        "claim_boundary": (
            "required_checks_terminal_success is runner evidence only; "
            "it is not a scientific/product qualification verdict"
        ),
    }


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FixtureError(f"cannot read fixture {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FixtureError(f"fixture {path} is not valid JSON: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify a saved Symthaea GitHub Actions runner-state fixture."
    )
    parser.add_argument("fixture", type=Path)
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print classification JSON.",
    )
    parser.add_argument(
        "--assert-required-success",
        action="store_true",
        help=(
            "Exit 1 unless every explicitly required job is terminal success. "
            "This still does not assert product qualification."
        ),
    )
    args = parser.parse_args()

    try:
        result = classify_fixture(load_json(args.fixture))
    except FixtureError as exc:
        print(f"runner-state fixture error: {exc}", file=sys.stderr)
        return 2

    json.dump(
        result,
        sys.stdout,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    sys.stdout.write("\n")

    if args.assert_required_success and not result["required_checks_terminal_success"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
