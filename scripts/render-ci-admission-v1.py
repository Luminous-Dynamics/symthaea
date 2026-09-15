#!/usr/bin/env python3
"""Render CI Admission V1 onto one exact, census-bound ci.yml source.

The renderer is lexical and fail-closed. It accepts only the exact source blob
named by the census, inserts one routing job that executes policy bytes from a
trusted base commit for PR/merge-group events, and gates every classified job.
"""

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

PULL_REQUEST_TRIGGER = "  pull_request:\n"
PULL_REQUEST_REPLACEMENT = (
    "  pull_request:\n"
    "    types: [opened, synchronize, reopened, ready_for_review, "
    "converted_to_draft, labeled, unlabeled]\n"
)

HEAVY_GATE = "needs.admission.outputs.heavy_eligible == 'true'"
MAINTENANCE_GATE = (
    "(needs.admission.outputs.heavy_eligible == 'true' || "
    "needs.admission.outputs.maintenance_eligible == 'true')"
)

ADMISSION_JOB = r'''  admission:
    name: CI Admission
    runs-on: ubuntu-latest
    outputs:
      disposition: ${{ steps.route.outputs.disposition }}
      heavy_eligible: ${{ steps.route.outputs.heavy_eligible }}
      maintenance_eligible: ${{ steps.route.outputs.maintenance_eligible }}
      exact_subject_required: ${{ steps.route.outputs.exact_subject_required }}
      head_sha: ${{ steps.route.outputs.head_sha }}
      policy_sha: ${{ steps.route.outputs.policy_sha }}
      control_plane_modified: ${{ steps.route.outputs.control_plane_modified }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: false

      - name: Materialize trusted admission policy
        env:
          ADMISSION_POLICY_SHA: ${{ github.event.pull_request.base.sha || github.event.merge_group.base_sha || github.sha }}
        run: |
          set -euo pipefail
          test -n "$ADMISSION_POLICY_SHA"
          git cat-file -e "${ADMISSION_POLICY_SHA}^{commit}"
          git show "${ADMISSION_POLICY_SHA}:scripts/ci-admission-v1.py" > "$RUNNER_TEMP/ci-admission-v1.py"
          python3 -m py_compile "$RUNNER_TEMP/ci-admission-v1.py"

      - name: Route qualification admission
        id: route
        env:
          ADMISSION_EVENT_NAME: ${{ github.event_name }}
          ADMISSION_ACTION: ${{ github.event.action }}
          ADMISSION_DRAFT: ${{ github.event.pull_request.draft }}
          ADMISSION_LABELS_JSON: ${{ toJSON(github.event.pull_request.labels.*.name) }}
          ADMISSION_REF: ${{ github.ref }}
          ADMISSION_HEAD_SHA: ${{ github.event.pull_request.head.sha || github.event.merge_group.head_sha || github.sha }}
          ADMISSION_BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.merge_group.base_sha || '' }}
          ADMISSION_POLICY_SHA: ${{ github.event.pull_request.base.sha || github.event.merge_group.base_sha || github.sha }}
        run: |
          set -euo pipefail
          args=(
            --event-name "$ADMISSION_EVENT_NAME"
            --action "${ADMISSION_ACTION:-}"
            --draft "${ADMISSION_DRAFT:-false}"
            --labels-json "${ADMISSION_LABELS_JSON:-[]}"
            --ref "$ADMISSION_REF"
            --head-sha "$ADMISSION_HEAD_SHA"
            --policy-sha "$ADMISSION_POLICY_SHA"
            --github-output "$GITHUB_OUTPUT"
          )
          if [ "$ADMISSION_EVENT_NAME" = "pull_request" ] || [ "$ADMISSION_EVENT_NAME" = "merge_group" ]; then
            args+=(--repository "$GITHUB_WORKSPACE" --base-sha "$ADMISSION_BASE_SHA")
          fi
          python3 "$RUNNER_TEMP/ci-admission-v1.py" "${args[@]}"

'''


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def job_ranges(lines: list[str]) -> list[tuple[str, int, int]]:
    try:
        jobs_line = next(i for i, line in enumerate(lines) if line == "jobs:")
    except StopIteration as exc:
        raise ValueError("missing exact jobs: marker") from exc

    starts: list[tuple[str, int]] = []
    for i in range(jobs_line + 1, len(lines)):
        match = JOB_KEY.match(lines[i])
        if match:
            starts.append((match.group(1), i))
    if len(starts) != len({name for name, _ in starts}):
        raise ValueError("duplicate top-level jobs")
    return [
        (name, start, starts[index + 1][1] if index + 1 < len(starts) else len(lines))
        for index, (name, start) in enumerate(starts)
    ]


def add_admission_need(name: str, block: list[str]) -> list[str]:
    needs_indexes = [i for i, line in enumerate(block) if line.startswith("    needs:")]
    if len(needs_indexes) > 1:
        raise ValueError(f"unsupported multiple needs keys in {name}")

    out = list(block)
    if not needs_indexes:
        insert_at = 2 if len(out) > 1 and out[1].startswith("    name:") else 1
        out.insert(insert_at, "    needs: admission")
        return out

    index = needs_indexes[0]
    line = out[index]
    scalar = NEEDS_SCALAR.match(line)
    if scalar:
        dependency = scalar.group(1)
        if dependency == "admission":
            raise ValueError(f"{name} already depends on admission")
        out[index] = f"    needs: [admission, {dependency}]"
        return out

    inline = NEEDS_INLINE.match(line)
    if inline:
        dependencies = [item.strip() for item in inline.group(1).split(",") if item.strip()]
        if not dependencies or any(not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in dependencies):
            raise ValueError(f"unsupported inline needs in {name}: {line}")
        if "admission" in dependencies:
            raise ValueError(f"{name} already depends on admission")
        out[index] = f"    needs: [admission, {', '.join(dependencies)}]"
        return out

    if line == "    needs:":
        end = index + 1
        dependencies: list[str] = []
        while end < len(out) and (out[end].startswith("      ") or out[end].strip() == ""):
            if out[end].strip():
                match = NEEDS_BLOCK_ITEM.match(out[end])
                if not match:
                    raise ValueError(f"unsupported block needs in {name}: {out[end]}")
                dependencies.append(match.group(1))
            end += 1
        if not dependencies:
            raise ValueError(f"empty block needs in {name}")
        if "admission" in dependencies:
            raise ValueError(f"{name} already depends on admission")
        out.insert(index + 1, "      - admission")
        return out

    raise ValueError(f"unsupported needs syntax in {name}: {line}")


def gate_block(name: str, block: list[str], gate: str) -> list[str]:
    if not block or block[0] != f"  {name}:":
        raise ValueError(f"malformed block for {name}")

    out = add_admission_need(name, block)
    if_indexes = [i for i, line in enumerate(out) if line.startswith("    if:")]
    if len(if_indexes) > 1:
        raise ValueError(f"unsupported multiple if keys in {name}")
    if not if_indexes:
        needs_index = next(i for i, line in enumerate(out) if line.startswith("    needs:"))
        out.insert(needs_index + 1, f"    if: {gate}")
        return out

    index = if_indexes[0]
    line = out[index]
    if line == "    if: |":
        content_end = index + 1
        while content_end < len(out) and (
            out[content_end].startswith("      ") or out[content_end].strip() == ""
        ):
            content_end += 1
        content = out[index + 1 : content_end]
        if not content:
            raise ValueError(f"empty block if in {name}")
        normalized = [
            content_line[6:] if content_line.startswith("      ") else content_line.lstrip()
            for content_line in content
        ]
        if any("${{" in item for item in normalized):
            raise ValueError(f"unsupported explicit expression wrapper in block if for {name}")
        replacement = [line, f"      {gate} &&", "      ("]
        replacement.extend(f"        {item}" for item in normalized)
        replacement.append("      )")
        out[index:content_end] = replacement
        return out

    prefix = "    if: "
    if not line.startswith(prefix):
        raise ValueError(f"unsupported if syntax in {name}: {line}")
    expression = line[len(prefix):]
    if "${{" in expression:
        raise ValueError(f"unsupported explicit expression wrapper in if for {name}")
    out[index] = f"    if: {gate} && ({expression})"
    return out


def render(source: bytes, census: dict) -> bytes:
    expected_blob = census["workflow_git_blob_sha1"]
    actual_blob = git_blob_sha1(source)
    if actual_blob != expected_blob:
        raise ValueError(f"source blob drift: expected {expected_blob}, got {actual_blob}")

    text = source.decode("utf-8")
    if text.count(PULL_REQUEST_TRIGGER) != 1:
        raise ValueError("expected exactly one bare pull_request trigger")
    text = text.replace(PULL_REQUEST_TRIGGER, PULL_REQUEST_REPLACEMENT, 1)

    lines = text.splitlines()
    ranges = job_ranges(lines)
    actual_jobs = {name for name, _, _ in ranges}
    classified = {job for values in census["classes"].values() for job in values}
    if actual_jobs != classified:
        raise ValueError(
            f"census mismatch: actual_only={sorted(actual_jobs-classified)} "
            f"census_only={sorted(classified-actual_jobs)}"
        )
    if len(actual_jobs) != census["expected_total_jobs"]:
        raise ValueError("job count drift")

    heavy = set(census["classes"]["heavy_admitted"])
    maintenance = set(census["classes"]["heavy_or_maintenance"])
    if heavy & maintenance:
        raise ValueError("heavy and maintenance classes overlap")

    rebuilt: list[str] = []
    cursor = 0
    heavy_gated = 0
    maintenance_gated = 0
    for name, start, end in ranges:
        rebuilt.extend(lines[cursor:start])
        block = lines[start:end]
        if name in heavy:
            block = gate_block(name, block, HEAVY_GATE)
            heavy_gated += 1
        elif name in maintenance:
            block = gate_block(name, block, MAINTENANCE_GATE)
            maintenance_gated += 1
        rebuilt.extend(block)
        cursor = end
    rebuilt.extend(lines[cursor:])

    jobs_line = next(i for i, line in enumerate(rebuilt) if line == "jobs:")
    admission_lines = ADMISSION_JOB.rstrip("\n").splitlines()
    rebuilt[jobs_line + 1:jobs_line + 1] = [""] + admission_lines

    rendered = ("\n".join(rebuilt) + "\n").encode("utf-8")
    rendered_text = rendered.decode("utf-8")
    rendered_jobs = {name for name, _, _ in job_ranges(rendered_text.splitlines())}
    if rendered_jobs != actual_jobs | {"admission"}:
        raise ValueError("rendered job census is not source jobs + admission")
    if heavy_gated != len(heavy):
        raise ValueError(f"heavy gated count mismatch: {heavy_gated} != {len(heavy)}")
    if maintenance_gated != len(maintenance):
        raise ValueError(
            f"maintenance gated count mismatch: {maintenance_gated} != {len(maintenance)}"
        )
    if rendered_text.count(HEAVY_GATE) != len(heavy) + len(maintenance):
        raise ValueError("heavy eligibility expression count mismatch")
    if rendered_text.count("maintenance_eligible == 'true'") != len(maintenance):
        raise ValueError("maintenance eligibility expression count mismatch")
    if rendered_text.count("  admission:\n") != 1:
        raise ValueError("admission job insertion count mismatch")
    if "git show \"${ADMISSION_POLICY_SHA}:scripts/ci-admission-v1.py\"" not in rendered_text:
        raise ValueError("trusted base-policy materialization missing")
    return rendered


def self_test() -> None:
    source = b"""name: CI
on:
  pull_request:
  workflow_dispatch:
  schedule:
    - cron: '0 4 * * 0'
jobs:
  cheap:
    name: Cheap
    runs-on: ubuntu-latest
  heavy-a:
    name: Heavy A
    runs-on: ubuntu-latest
  test:
    name: Test
    runs-on: ubuntu-latest
  psych-bench:
    name: Psych
    needs: test
    if: |
      github.event_name == 'schedule' ||
      github.event_name == 'pull_request'
    runs-on: ubuntu-latest
  heavy-b:
    name: Heavy B
    needs: test
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
  sbom:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
  stress-tests:
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
"""
    census = {
        "workflow_git_blob_sha1": git_blob_sha1(source),
        "expected_total_jobs": 7,
        "classes": {
            "cheap_static_candidate": ["cheap"],
            "heavy_admitted": ["heavy-a", "heavy-b"],
            "heavy_or_maintenance": ["test", "psych-bench"],
            "event_special": ["sbom", "stress-tests"],
        },
    }
    rendered = render(source, census).decode("utf-8")
    assert "needs: admission" in rendered
    assert rendered.count("needs: [admission, test]") == 2
    assert rendered.count(HEAVY_GATE) == 4
    assert rendered.count("maintenance_eligible == 'true'") == 2
    assert "converted_to_draft" in rendered
    assert rendered.count("  admission:\n") == 1
    assert "git show \"${ADMISSION_POLICY_SHA}:scripts/ci-admission-v1.py\"" in rendered
    assert "persist-credentials: false" in rendered

    drifted = source + b"# drift\n"
    try:
        render(drifted, census)
    except ValueError as exc:
        assert "source blob drift" in str(exc)
    else:
        raise AssertionError("source drift must fail closed")

    print("ci_admission_v1_renderer_self_test=PASS cases=2")
    print("claim_boundary=renderer_only_not_github_parser_or_live_execution")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.source is None or args.census is None or args.output is None:
        parser.error("--source, --census, and --output are required unless --self-test is used")

    source = args.source.read_bytes()
    census = json.loads(args.census.read_text())
    rendered = render(source, census)
    args.output.write_bytes(rendered)
    print(
        "ci_admission_v1_render=PASS "
        f"source_blob={git_blob_sha1(source)} rendered_sha256={hashlib.sha256(rendered).hexdigest()}"
    )
    print("claim_boundary=rendered_candidate_not_github_parser_or_live_execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
