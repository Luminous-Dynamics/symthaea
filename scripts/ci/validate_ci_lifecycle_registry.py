#!/usr/bin/env python3
"""Validate the CI lifecycle registry against the actual monolithic workflow.

No YAML dependency is required. The parser intentionally recognizes only the
simple top-level shape used by ``.github/workflows/ci.yml``:

    jobs:
      job-id:
        ...

If the workflow stops matching this conservative shape, validation fails rather
than guessing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any

JOB_ID = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
JOB_NEEDS = re.compile(r"^    needs:\s*(.*?)\s*$")


@dataclass(frozen=True)
class JobBlock:
    job_id: str
    lines: tuple[str, ...]

    @property
    def has_needs(self) -> bool:
        return any(JOB_NEEDS.match(line) for line in self.lines)


def extract_job_blocks(workflow_text: str) -> dict[str, JobBlock]:
    lines = workflow_text.splitlines()
    try:
        jobs_index = next(i for i, line in enumerate(lines) if line == "jobs:")
    except StopIteration as exc:
        raise ValueError("workflow has no exact top-level 'jobs:' line") from exc

    starts: list[tuple[int, str]] = []
    for index in range(jobs_index + 1, len(lines)):
        match = JOB_ID.match(lines[index])
        if match:
            starts.append((index, match.group(1)))

    if not starts:
        raise ValueError("workflow contains no recognizable top-level job IDs")

    blocks: dict[str, JobBlock] = {}
    for position, (start, job_id) in enumerate(starts):
        if job_id in blocks:
            raise ValueError(f"duplicate top-level job id: {job_id}")
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        blocks[job_id] = JobBlock(job_id=job_id, lines=tuple(lines[start:end]))
    return blocks


def load_registry(path: pathlib.Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("registry must be a JSON object")
    if raw.get("schema") != "symthaea.ci-lifecycle-jobs.v1":
        raise ValueError("unexpected lifecycle registry schema")
    policy = raw.get("policy")
    if not isinstance(policy, dict) or policy.get("unknown_job_default") != "full":
        raise ValueError("registry must fail unknown jobs toward full verification")
    jobs = raw.get("iteration_jobs")
    if not isinstance(jobs, list) or not jobs or not all(isinstance(v, str) for v in jobs):
        raise ValueError("iteration_jobs must be a non-empty string list")
    if len(jobs) != len(set(jobs)):
        raise ValueError("iteration_jobs contains duplicates")
    return raw


def validate(workflow_text: str, registry: dict[str, Any]) -> dict[str, Any]:
    blocks = extract_job_blocks(workflow_text)
    iteration = list(registry["iteration_jobs"])

    missing = sorted(job_id for job_id in iteration if job_id not in blocks)
    if missing:
        raise ValueError(f"iteration jobs absent from ci.yml: {', '.join(missing)}")

    coupled = sorted(job_id for job_id in iteration if blocks[job_id].has_needs)
    if coupled:
        raise ValueError(
            "iteration jobs must be independently runnable and may not have job-level needs: "
            + ", ".join(coupled)
        )

    examples = registry.get("full_only_examples", [])
    if not isinstance(examples, list) or not all(isinstance(v, str) for v in examples):
        raise ValueError("full_only_examples must be a string list")
    missing_examples = sorted(job_id for job_id in examples if job_id not in blocks)
    if missing_examples:
        raise ValueError(
            "documented full-only example jobs absent from ci.yml: "
            + ", ".join(missing_examples)
        )

    full_only = sorted(set(blocks) - set(iteration))
    return {
        "schema": "symthaea.ci-lifecycle-registry-validation.v1",
        "workflow_job_count": len(blocks),
        "iteration_job_count": len(iteration),
        "full_only_default_count": len(full_only),
        "iteration_jobs": sorted(iteration),
        "full_only_jobs": full_only,
        "unknown_job_default": "full",
        "tier1_green_is_merge_qualification": False,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default=".github/workflows/ci.yml")
    parser.add_argument(
        "--registry", default="scripts/ci/ci_lifecycle_jobs_v1.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    workflow_path = pathlib.Path(args.workflow)
    registry_path = pathlib.Path(args.registry)
    try:
        registry = load_registry(registry_path)
        result = validate(
            workflow_path.read_text(encoding="utf-8"),
            registry,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ci lifecycle registry validation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
