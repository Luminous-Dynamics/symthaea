#!/usr/bin/env python3
"""Deterministically apply CI lifecycle tiering to the audited ci.yml generation.

The transformer is intentionally exact-input-bound. It refuses to rewrite a
workflow whose Git blob SHA differs from the reviewed v1 source generation.

It does not call GitHub and does not commit anything. By default it writes the
transformed workflow to stdout; ``--output`` writes a separate file. In-place
replacement is deliberately not supported by v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any

from scripts.ci.validate_ci_lifecycle_registry import extract_job_blocks, load_registry

EXPECTED_CI_BLOB_V1 = "a48366076b30eb8e12d22c927a3b8bf333181409"
FULL_GUARD = "github.event_name != 'pull_request' || github.event.pull_request.draft != true"
MARKER = "# CI lifecycle full-only v1"

PR_TRIGGER_OLD = "  pull_request:\n"
PR_TRIGGER_NEW = """  pull_request:
    types:
      - opened
      - synchronize
      - reopened
      - ready_for_review
      - converted_to_draft
"""

JOB_ID = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
JOB_IF = re.compile(r"^    if:\s*(.*?)\s*$")
JOB_KEY = re.compile(r"^    [A-Za-z0-9_-]+:\s*(?:.*)?$")


@dataclass(frozen=True)
class RewriteResult:
    text: str
    tier1_jobs: tuple[str, ...]
    full_only_jobs: tuple[str, ...]


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _job_ranges(lines: list[str]) -> list[tuple[str, int, int]]:
    try:
        jobs_index = next(i for i, line in enumerate(lines) if line == "jobs:")
    except StopIteration as exc:
        raise ValueError("workflow has no exact top-level jobs: key") from exc

    starts: list[tuple[str, int]] = []
    for index in range(jobs_index + 1, len(lines)):
        match = JOB_ID.match(lines[index])
        if match:
            starts.append((match.group(1), index))
    if not starts:
        raise ValueError("workflow has no recognizable jobs")

    ranges: list[tuple[str, int, int]] = []
    for pos, (job_id, start) in enumerate(starts):
        end = starts[pos + 1][1] if pos + 1 < len(starts) else len(lines)
        ranges.append((job_id, start, end))
    return ranges


def _find_job_if(block: list[str]) -> tuple[int, int, str] | None:
    """Return (start, end, expression) for a job-level if mapping.

    ``end`` is exclusive. Both scalar and literal/folded block forms are
    accepted. Step-level ``if`` keys are indented more deeply and ignored.
    """

    found: list[tuple[int, int, str]] = []
    for index, line in enumerate(block):
        match = JOB_IF.match(line)
        if not match:
            continue
        raw = match.group(1)
        if raw in ("|", ">", "|-", ">-"):
            expr_lines: list[str] = []
            cursor = index + 1
            while cursor < len(block):
                candidate = block[cursor]
                if JOB_KEY.match(candidate):
                    break
                if candidate.strip():
                    if not candidate.startswith("      "):
                        raise ValueError(
                            "job-level if block contains unexpected indentation: "
                            + candidate
                        )
                    expr_lines.append(candidate.strip())
                cursor += 1
            if not expr_lines:
                raise ValueError("job-level block if has no expression")
            found.append((index, cursor, " ".join(expr_lines)))
        else:
            if not raw:
                raise ValueError("empty job-level if expression")
            found.append((index, index + 1, raw))

    if len(found) > 1:
        raise ValueError("job contains multiple job-level if keys")
    return found[0] if found else None


def _guard_full_job(block: list[str]) -> list[str]:
    if any(MARKER in line for line in block):
        raise ValueError("workflow already contains lifecycle marker")

    existing = _find_job_if(block)
    marker_line = f"    {MARKER}"

    if existing is None:
        return [
            block[0],
            marker_line,
            f"    if: ${{{{ {FULL_GUARD} }}}}",
            *block[1:],
        ]

    start, end, old_expr = existing
    # GitHub permits expressions in a job-level if without the ${{ }} wrapper.
    # Use one folded scalar so the old predicate remains visibly conjunctive.
    replacement = [
        marker_line,
        "    if: >-",
        f"      ({FULL_GUARD}) &&",
        f"      ({old_expr})",
    ]
    return [*block[:start], *replacement, *block[end:]]


def _validate_transformed(text: str, iteration_jobs: set[str]) -> None:
    if PR_TRIGGER_NEW not in text:
        raise ValueError("transformed workflow lacks explicit PR lifecycle trigger")
    if PR_TRIGGER_OLD in text:
        raise ValueError("broad pull_request trigger survived transformation")

    blocks = extract_job_blocks(text)
    for job_id, block in blocks.items():
        has_marker = any(MARKER in line for line in block.lines)
        has_guard = any(FULL_GUARD in line for line in block.lines)
        if job_id in iteration_jobs:
            if has_marker or has_guard:
                raise ValueError(f"Tier-1 job {job_id} unexpectedly received full-only guard")
        else:
            if not has_marker or not has_guard:
                raise ValueError(f"full-only job {job_id} lacks lifecycle guard")
            # Exactly one job-level if key after transformation. Step-level ifs
            # have deeper indentation and are intentionally ignored.
            job_if_count = sum(1 for line in block.lines if JOB_IF.match(line))
            if job_if_count != 1:
                raise ValueError(
                    f"full-only job {job_id} has {job_if_count} job-level if keys"
                )


def transform(workflow_text: str, registry: dict[str, Any]) -> RewriteResult:
    if workflow_text.count(PR_TRIGGER_OLD) != 1:
        raise ValueError("expected exactly one broad pull_request trigger")

    # Validate the unmodified registry/workflow relationship first.
    from scripts.ci.validate_ci_lifecycle_registry import validate

    validate(workflow_text, registry)

    iteration_jobs = set(registry["iteration_jobs"])
    text = workflow_text.replace(PR_TRIGGER_OLD, PR_TRIGGER_NEW, 1)
    lines = text.splitlines()
    ranges = _job_ranges(lines)

    # Rewrite from the bottom so earlier line offsets remain stable.
    for job_id, start, end in reversed(ranges):
        if job_id in iteration_jobs:
            continue
        lines[start:end] = _guard_full_job(lines[start:end])

    transformed = "\n".join(lines)
    if text.endswith("\n"):
        transformed += "\n"

    _validate_transformed(transformed, iteration_jobs)
    all_jobs = set(extract_job_blocks(transformed))
    return RewriteResult(
        text=transformed,
        tier1_jobs=tuple(sorted(iteration_jobs)),
        full_only_jobs=tuple(sorted(all_jobs - iteration_jobs)),
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default=".github/workflows/ci.yml")
    parser.add_argument("--registry", default="scripts/ci/ci_lifecycle_jobs_v1.json")
    parser.add_argument("--output")
    parser.add_argument(
        "--expected-blob",
        default=EXPECTED_CI_BLOB_V1,
        help="Reviewed Git blob SHA required for input ci.yml.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print transformation summary to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    workflow_path = pathlib.Path(args.workflow)
    registry_path = pathlib.Path(args.registry)

    try:
        raw_bytes = workflow_path.read_bytes()
        actual_blob = git_blob_sha(raw_bytes)
        if actual_blob != args.expected_blob:
            raise ValueError(
                f"ci.yml Git blob drift: expected {args.expected_blob}, got {actual_blob}"
            )
        registry = load_registry(registry_path)
        result = transform(raw_bytes.decode("utf-8"), registry)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"CI lifecycle transform refused: {exc}", file=sys.stderr)
        return 1

    if args.output:
        output_path = pathlib.Path(args.output)
        if output_path.resolve() == workflow_path.resolve():
            print("CI lifecycle transform refused: v1 does not support in-place output", file=sys.stderr)
            return 1
        output_path.write_text(result.text, encoding="utf-8")
    else:
        sys.stdout.write(result.text)

    if args.summary:
        print(
            json.dumps(
                {
                    "schema": "symthaea.ci-lifecycle-transform.v1",
                    "source_blob": actual_blob,
                    "tier1_jobs": list(result.tier1_jobs),
                    "full_only_job_count": len(result.full_only_jobs),
                    "full_only_jobs": list(result.full_only_jobs),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
