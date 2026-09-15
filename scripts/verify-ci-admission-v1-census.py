#!/usr/bin/env python3
"""Fail-closed verifier for the CI Admission V1 job census and dependency cut."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

JOB_KEY = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
NEEDS_SCALAR = re.compile(r"^    needs:\s*([A-Za-z0-9_-]+)\s*$")
NEEDS_INLINE = re.compile(r"^    needs:\s*\[([^\]]*)\]\s*$")
NEEDS_BLOCK_ITEM = re.compile(r"^      -\s*([A-Za-z0-9_-]+)\s*$")
EXPECTED_CLASSES = frozenset(
    {"cheap_static_candidate", "heavy_admitted", "heavy_or_maintenance", "event_special"}
)


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def job_ranges(lines: list[str]) -> list[tuple[str, int, int]]:
    try:
        jobs_line = next(i for i, line in enumerate(lines) if line == "jobs:")
    except StopIteration as exc:
        raise ValueError("workflow has no exact top-level jobs: marker") from exc

    starts: list[tuple[str, int]] = []
    for i in range(jobs_line + 1, len(lines)):
        match = JOB_KEY.match(lines[i])
        if match:
            starts.append((match.group(1), i))
    if not starts:
        raise ValueError("no top-level job keys discovered")
    if len(starts) != len({name for name, _ in starts}):
        raise ValueError("duplicate top-level job key discovered")
    return [
        (name, start, starts[index + 1][1] if index + 1 < len(starts) else len(lines))
        for index, (name, start) in enumerate(starts)
    ]


def parse_needs(name: str, block: list[str]) -> list[str]:
    indexes = [i for i, line in enumerate(block) if line.startswith("    needs:")]
    if len(indexes) > 1:
        raise ValueError(f"multiple top-level needs keys in {name}")
    if not indexes:
        return []

    index = indexes[0]
    line = block[index]
    scalar = NEEDS_SCALAR.match(line)
    if scalar:
        return [scalar.group(1)]

    inline = NEEDS_INLINE.match(line)
    if inline:
        deps = [item.strip() for item in inline.group(1).split(",") if item.strip()]
        if not deps or any(not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in deps):
            raise ValueError(f"unsupported inline needs in {name}: {line}")
        if len(deps) != len(set(deps)):
            raise ValueError(f"duplicate dependency in {name}")
        return deps

    if line == "    needs:":
        deps: list[str] = []
        cursor = index + 1
        while cursor < len(block) and (
            block[cursor].startswith("      ") or block[cursor].strip() == ""
        ):
            if block[cursor].strip():
                match = NEEDS_BLOCK_ITEM.match(block[cursor])
                if not match:
                    raise ValueError(f"unsupported block needs in {name}: {block[cursor]}")
                deps.append(match.group(1))
            cursor += 1
        if not deps:
            raise ValueError(f"empty block needs in {name}")
        if len(deps) != len(set(deps)):
            raise ValueError(f"duplicate dependency in {name}")
        return deps

    raise ValueError(f"unsupported needs syntax in {name}: {line}")


def block_contains(block: list[str], token: str) -> bool:
    return token in "\n".join(block)


def verify_dependency_cut(
    blocks: dict[str, list[str]], class_of: dict[str, str]
) -> int:
    mode_sets = {
        "cheap_static_candidate": frozenset({"heavy", "maintenance", "cheap_only"}),
        "heavy_admitted": frozenset({"heavy"}),
        "heavy_or_maintenance": frozenset({"heavy", "maintenance"}),
    }
    edge_count = 0
    for job, block in blocks.items():
        deps = parse_needs(job, block)
        if job in deps:
            raise ValueError(f"self dependency in {job}")
        for dep in deps:
            edge_count += 1
            if dep not in blocks:
                raise ValueError(f"unknown dependency {job}->{dep}")

        job_class = class_of[job]
        if job_class == "event_special":
            if deps:
                raise ValueError(
                    f"event_special job {job} has dependencies; V1 requires event-special jobs to be independent"
                )
            continue

        job_modes = mode_sets[job_class]
        for dep in deps:
            dep_class = class_of[dep]
            if dep_class == "event_special":
                raise ValueError(f"{job} depends on event_special job {dep}")
            dep_modes = mode_sets[dep_class]
            if not job_modes.issubset(dep_modes):
                raise ValueError(
                    f"dependency cut violation {job}({job_class})->{dep}({dep_class}): "
                    f"job_modes={sorted(job_modes)} dep_modes={sorted(dep_modes)}"
                )
    return edge_count


def verify(workflow: bytes, census: dict) -> None:
    actual_blob = git_blob_sha1(workflow)
    expected_blob = census["workflow_git_blob_sha1"]
    if actual_blob != expected_blob:
        raise ValueError(f"workflow blob drift: expected {expected_blob}, got {actual_blob}")

    text = workflow.decode("utf-8")
    lines = text.splitlines()
    ranges = job_ranges(lines)
    blocks = {name: lines[start:end] for name, start, end in ranges}
    actual_jobs = set(blocks)

    classes = census["classes"]
    if set(classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"unexpected class set: expected={sorted(EXPECTED_CLASSES)} got={sorted(classes)}"
        )
    class_sets = {name: set(values) for name, values in classes.items()}

    all_classified: set[str] = set()
    class_of: dict[str, str] = {}
    for class_name, values in class_sets.items():
        overlap = all_classified & values
        if overlap:
            raise ValueError(f"job classified more than once in {class_name}: {sorted(overlap)}")
        for job in values:
            class_of[job] = class_name
        all_classified |= values

    if len(actual_jobs) != census["expected_total_jobs"]:
        raise ValueError(
            f"job count drift: expected {census['expected_total_jobs']}, got {len(actual_jobs)}"
        )

    missing = actual_jobs - all_classified
    stale = all_classified - actual_jobs
    if missing or stale:
        raise ValueError(
            f"census mismatch: unclassified_actual={sorted(missing)} stale_census={sorted(stale)}"
        )

    for job, token in census.get("event_special_invariants", {}).items():
        if class_of.get(job) != "event_special":
            raise ValueError(f"event-special invariant names non-event-special job {job}")
        if not block_contains(blocks[job], token):
            raise ValueError(f"event-special invariant not established for {job}")

    for job, token in census.get("maintenance_event_invariants", {}).items():
        if class_of.get(job) != "heavy_or_maintenance":
            raise ValueError(f"maintenance invariant names wrong class for {job}")
        if not block_contains(blocks[job], token):
            raise ValueError(f"maintenance event invariant not established for {job}")

    edge_count = verify_dependency_cut(blocks, class_of)

    print(
        "ci_admission_v1_census=PASS "
        f"jobs={len(actual_jobs)} "
        f"cheap={len(class_sets['cheap_static_candidate'])} "
        f"heavy={len(class_sets['heavy_admitted'])} "
        f"heavy_or_maintenance={len(class_sets['heavy_or_maintenance'])} "
        f"event_special={len(class_sets['event_special'])} "
        f"dependency_edges={edge_count}"
    )
    print("claim_boundary=census_and_dependency_cut_only_not_active_gating_or_qualification")


def self_test() -> None:
    sample = b"""name: CI
jobs:
  cheap:
    runs-on: ubuntu-latest
  test:
    runs-on: ubuntu-latest
  psych-bench:
    needs: test
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
  heavy:
    needs: test
    runs-on: ubuntu-latest
  sbom:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
  stress-tests:
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
"""
    sample_census = {
        "workflow_git_blob_sha1": git_blob_sha1(sample),
        "expected_total_jobs": 6,
        "classes": {
            "cheap_static_candidate": ["cheap"],
            "heavy_admitted": ["heavy"],
            "heavy_or_maintenance": ["test", "psych-bench"],
            "event_special": ["sbom", "stress-tests"],
        },
        "event_special_invariants": {
            "sbom": "github.ref == 'refs/heads/main' && github.event_name == 'push'",
            "stress-tests": "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'",
        },
        "maintenance_event_invariants": {
            "psych-bench": "github.event_name == 'schedule'",
        },
    }
    verify(sample, sample_census)

    broken = json.loads(json.dumps(sample_census))
    broken["classes"]["cheap_static_candidate"] = ["cheap", "heavy"]
    broken["classes"]["heavy_admitted"] = []
    try:
        verify(sample, broken)
    except ValueError as exc:
        assert "dependency cut violation" in str(exc), exc
    else:
        raise AssertionError("cheap->maintenance dependency cut must fail closed")

    unknown = sample.replace(b"needs: test", b"needs: missing", 1)
    unknown_census = json.loads(json.dumps(sample_census))
    unknown_census["workflow_git_blob_sha1"] = git_blob_sha1(unknown)
    try:
        verify(unknown, unknown_census)
    except ValueError as exc:
        assert "unknown dependency" in str(exc), exc
    else:
        raise AssertionError("unknown dependency must fail closed")

    print("ci_admission_v1_census_self_test=PASS cases=3")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.workflow is None or args.census is None:
        parser.error("--workflow and --census are required unless --self-test is used")

    workflow = args.workflow.read_bytes()
    census = json.loads(args.census.read_text())
    verify(workflow, census)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
