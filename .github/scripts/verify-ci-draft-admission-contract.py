#!/usr/bin/env python3
"""Audit and mechanically plan draft admission for the full CI workflow.

Authority: RunnerPlaneContractOnly. This tool never edits ci.yml in place. It parses
only the top-level jobs mapping and job-level `if:` keys, including block scalars,
then emits an outcome-blind plan for a future exact-source transformer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any

CONTRACT_SCHEMA = "symthaea.ci.draft-admission-contract.v2"
AUTHORITY = "RunnerPlaneContractOnly"
BLOCK_MARKERS = {"|", ">", "|-", ">-", "|+", ">+"}


@dataclass(frozen=True)
class Job:
    key: str
    start: int
    end: int
    if_line: int | None
    if_value: str | None
    if_kind: str
    block_end: int | None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def find_jobs(lines: list[str]) -> list[Job]:
    jobs_line = next((i for i, line in enumerate(lines) if line in {"jobs:\n", "jobs:"}), None)
    require(jobs_line is not None, "top-level jobs mapping not found")

    key_re = re.compile(r"^  ([A-Za-z0-9_.-]+):\s*(?:#.*)?\n?$")
    starts: list[tuple[str, int]] = []
    for i in range(jobs_line + 1, len(lines)):
        line = lines[i]
        if line and not line.startswith((" ", "\t", "#", "\n")):
            break
        match = key_re.match(line)
        if match:
            starts.append((match.group(1), i))
    require(starts, "no top-level jobs found")
    require(len({name for name, _ in starts}) == len(starts), "duplicate top-level job key")

    jobs: list[Job] = []
    for index, (key, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(lines)
        if_lines: list[tuple[int, str]] = []
        for i in range(start + 1, end):
            if lines[i].startswith("    if:"):
                if_lines.append((i, lines[i].split("if:", 1)[1].strip()))
        require(len(if_lines) <= 1, f"{key}: duplicate job-level if keys")

        if not if_lines:
            jobs.append(Job(key, start, end, None, None, "none", None))
            continue

        if_line, value = if_lines[0]
        require(value != "", f"{key}: empty job-level if")
        if value not in BLOCK_MARKERS:
            require("#" not in value, f"{key}: inline-comment scalar if requires manual adjudication")
            jobs.append(Job(key, start, end, if_line, value, "scalar", None))
            continue

        block_end = if_line + 1
        saw_content = False
        while block_end < end:
            line = lines[block_end]
            if line.strip() == "":
                block_end += 1
                continue
            if indent(line) <= 4:
                break
            require(indent(line) >= 6, f"{key}: nonstandard block-if indentation")
            saw_content = True
            block_end += 1
        require(saw_content, f"{key}: empty block-scalar if")
        jobs.append(Job(key, start, end, if_line, value, "block", block_end))
    return jobs


def block_payload(lines: list[str], job: Job) -> list[str]:
    require(job.if_kind == "block", f"{job.key}: not a block if")
    assert job.if_line is not None and job.block_end is not None
    payload = lines[job.if_line + 1 : job.block_end]
    require(any(line.strip() for line in payload), f"{job.key}: empty block payload")
    for line in payload:
        if line.strip():
            require(indent(line) >= 6, f"{job.key}: block payload indentation too shallow")
    return payload


def plan(contract: dict[str, Any], workflow: pathlib.Path) -> dict[str, Any]:
    require(contract["schema"] == CONTRACT_SCHEMA, "contract schema mismatch")
    require(contract["authority"] == AUTHORITY, "contract authority mismatch")
    require(contract["source_workflow"] == workflow.as_posix(), "workflow path mismatch")
    require(not any(contract["claims"].values()), "contract exceeds authority")

    source = workflow.read_text()
    require(contract["admission_predicate"] not in source, "source already contains admission predicate")
    lines = source.splitlines(keepends=True)
    jobs = find_jobs(lines)
    actual_block_jobs = [job.key for job in jobs if job.if_kind == "block"]
    require(
        actual_block_jobs == contract["expected_block_if_jobs"],
        f"block-if census mismatch: {actual_block_jobs!r}",
    )

    entries: list[dict[str, Any]] = []
    for job in jobs:
        entry: dict[str, Any] = {
            "job": job.key,
            "source_if_kind": job.if_kind,
            "source_if": job.if_value,
            "planned_operation": {
                "none": "insert_admission",
                "scalar": "conjoin_admission_with_exact_scalar_if",
                "block": "wrap_admission_around_exact_block_payload",
            }[job.if_kind],
        }
        if job.if_kind == "block":
            payload = "".join(block_payload(lines, job)).encode()
            entry["source_block_marker"] = job.if_value
            entry["source_block_payload_sha256"] = sha256_bytes(payload)
        entries.append(entry)

    return {
        "schema": "symthaea.ci.draft-admission-plan.v2",
        "authority": AUTHORITY,
        "source_workflow": workflow.as_posix(),
        "source_workflow_sha256": sha256(workflow),
        "admission_predicate": contract["admission_predicate"],
        "job_count": len(jobs),
        "jobs_without_if": sum(job.if_kind == "none" for job in jobs),
        "jobs_with_scalar_if": sum(job.if_kind == "scalar" for job in jobs),
        "jobs_with_block_if": sum(job.if_kind == "block" for job in jobs),
        "block_if_jobs": actual_block_jobs,
        "mechanically_transformable": True,
        "jobs": entries,
        "claims": dict(contract["claims"]),
    }


def self_test() -> dict[str, Any]:
    samples = {
        "none": "jobs:\n  a:\n    runs-on: ubuntu-latest\n",
        "scalar": "jobs:\n  a:\n    if: github.event_name == 'pull_request'\n    runs-on: ubuntu-latest\n",
        "block": "jobs:\n  a:\n    if: |\n      github.event_name == 'push' ||\n      github.event_name == 'schedule'\n    runs-on: ubuntu-latest\n",
    }
    none = find_jobs(samples["none"].splitlines(keepends=True))[0]
    scalar = find_jobs(samples["scalar"].splitlines(keepends=True))[0]
    block_lines = samples["block"].splitlines(keepends=True)
    block = find_jobs(block_lines)[0]
    require(none.if_kind == "none", "none parser regression")
    require(
        scalar.if_kind == "scalar" and scalar.if_value == "github.event_name == 'pull_request'",
        "scalar parser regression",
    )
    require(block.if_kind == "block" and block.if_value == "|", "block parser regression")
    require(len(block_payload(block_lines, block)) == 2, "block payload census regression")

    malformed = "jobs:\n  a:\n    if: |\n    runs-on: ubuntu-latest\n"
    try:
        find_jobs(malformed.splitlines(keepends=True))
    except ValueError:
        pass
    else:
        raise ValueError("empty block-if was accepted")

    return {
        "schema": "symthaea.ci.draft-admission-self-test.v2",
        "authority": AUTHORITY,
        "cases": 4,
        "none_if_parse": "PASS",
        "scalar_if_parse": "PASS",
        "block_if_parse": "PASS",
        "empty_block_rejected": True,
        "block_if_is_mechanical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path)
    parser.add_argument("--workflow", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    require(args.contract is not None and args.workflow is not None, "--contract and --workflow required")
    print(json.dumps(plan(load_json(args.contract), args.workflow), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
