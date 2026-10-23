#!/usr/bin/env python3
"""Fail closed if PR workflows can accumulate stale runner work.

This is a structural queue-admission ratchet, complementary to
check-workflow-draft-safety.py:

* every runner-capable pull_request workflow must declare top-level concurrency;
* the concurrency group must be stable across automatic updates to one PR/ref;
* stale automatic runs must be superseded with cancel-in-progress: true;
* per-run/per-SHA keys may not defeat automatic supersession;
* workflow_dispatch may use github.run_id only behind an explicit manual-event
  discriminator, preserving deliberate evidence/reproduction runs.

benchmarks.yml is the sole current exception because the existing draft-safety
ratchet separately proves that pull_request events cannot allocate any runner in
that workflow. Adding another exception requires editing this file explicitly.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

WORKFLOW_DIR = Path(".github/workflows")
RUNNERLESS_PR_EXEMPTIONS = {Path(".github/workflows/benchmarks.yml")}

PR_TRIGGER = re.compile(r"^  pull_request:\s*(.*)$")
GROUP_LINE = re.compile(r"^  group:\s*(.+)$")
CANCEL_LINE = re.compile(r"^  cancel-in-progress:\s*(.+)$")
QUEUE_LINE = re.compile(r"^  queue:\s*(.+)$")
PR_SCOPE_TOKENS = (
    "github.ref",
    "github.head_ref",
    "github.event.pull_request.number",
)
AUTOMATIC_UNIQUENESS_TOKENS = (
    "github.sha",
    "github.event.pull_request.head.sha",
    "github.run_number",
    "github.run_attempt",
)


class QueuePolicyError(ValueError):
    pass


def fail(message: str) -> None:
    print(f"workflow-queue-policy: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def indentation(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def top_level_section(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    marker = f"{key}:"
    try:
        start = next(i for i, line in enumerate(lines) if line == marker)
    except StopIteration:
        return []

    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if indentation(line) == 0:
            end = i
            break
    return lines[start + 1 : end]


def has_pull_request_trigger(text: str) -> bool:
    return any(PR_TRIGGER.fullmatch(line) for line in top_level_section(text, "on"))


def exactly_one(lines: list[str], pattern: re.Pattern[str], label: str, path: Path) -> str:
    values = [match.group(1).strip() for line in lines if (match := pattern.fullmatch(line))]
    if len(values) != 1:
        raise QueuePolicyError(
            f"{path}: concurrency must contain exactly one {label}; observed={values!r}"
        )
    if not values[0]:
        raise QueuePolicyError(f"{path}: empty concurrency {label}")
    return values[0]


def validate_queue_policy(path: Path, text: str) -> str:
    section = top_level_section(text, "concurrency")
    if not section:
        raise QueuePolicyError(f"{path}: pull_request workflow lacks top-level concurrency")

    group = exactly_one(section, GROUP_LINE, "group", path)
    cancel = exactly_one(section, CANCEL_LINE, "cancel-in-progress", path)
    queues = [match.group(1).strip() for line in section if (match := QUEUE_LINE.fullmatch(line))]

    if queues:
        raise QueuePolicyError(
            f"{path}: explicit concurrency queue={queues!r} is not permitted for automatic PR validation"
        )
    if cancel != "true":
        raise QueuePolicyError(
            f"{path}: cancel-in-progress must be literal true for stale automatic validation"
        )
    if not any(token in group for token in PR_SCOPE_TOKENS):
        raise QueuePolicyError(
            f"{path}: concurrency group is not scoped to a stable PR/ref identity: {group!r}"
        )

    # Cross-workflow cancellation is dangerous. github.workflow is preferred;
    # a literal namespace equal to the workflow filename stem is also accepted.
    if "github.workflow" not in group and path.stem not in group:
        raise QueuePolicyError(
            f"{path}: concurrency group lacks workflow namespace ({path.stem!r} or github.workflow): {group!r}"
        )

    for token in AUTOMATIC_UNIQUENESS_TOKENS:
        if token in group:
            raise QueuePolicyError(
                f"{path}: concurrency group contains {token}, which makes stale automatic heads non-superseding"
            )

    if "github.run_id" in group:
        manual_guard = "github.event_name == 'workflow_dispatch'" in group or 'github.event_name == "workflow_dispatch"' in group
        if not manual_guard:
            raise QueuePolicyError(
                f"{path}: github.run_id is allowed only behind an explicit workflow_dispatch discriminator"
            )

    return group


def self_test() -> None:
    safe = """name: Safe\non:\n  pull_request:\n    types: [opened, synchronize, ready_for_review]\nconcurrency:\n  group: ${{ github.workflow }}-${{ github.ref }}\n  cancel-in-progress: true\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: true\n"""
    assert has_pull_request_trigger(safe)
    assert validate_queue_policy(Path("safe.yml"), safe)

    literal = safe.replace("${{ github.workflow }}-${{ github.ref }}", "literal-${{ github.event.pull_request.number }}")
    assert validate_queue_policy(Path("literal.yml"), literal)

    cases = {
        "missing": safe.replace("concurrency:\n  group: ${{ github.workflow }}-${{ github.ref }}\n  cancel-in-progress: true\n", ""),
        "no-scope": safe.replace("${{ github.workflow }}-${{ github.ref }}", "${{ github.workflow }}-global"),
        "no-cancel": safe.replace("cancel-in-progress: true", "cancel-in-progress: false"),
        "sha-unique": safe.replace("${{ github.workflow }}-${{ github.ref }}", "${{ github.workflow }}-${{ github.ref }}-${{ github.sha }}"),
        "run-id": safe.replace("${{ github.workflow }}-${{ github.ref }}", "${{ github.workflow }}-${{ github.ref }}-${{ github.run_id }}"),
    }
    for label, text in cases.items():
        try:
            validate_queue_policy(Path(f"{label}.yml"), text)
        except QueuePolicyError:
            pass
        else:
            raise AssertionError(f"unsafe queue policy accepted: {label}")

    manual = safe.replace(
        "${{ github.workflow }}-${{ github.ref }}",
        "${{ github.workflow }}-${{ github.ref }}-${{ github.event_name == 'workflow_dispatch' && github.run_id || 'auto' }}",
    )
    assert validate_queue_policy(Path("manual.yml"), manual)


def main() -> int:
    self_test()
    if not WORKFLOW_DIR.is_dir():
        fail(f"workflow directory not found: {WORKFLOW_DIR}")

    workflows = sorted(
        path for path in WORKFLOW_DIR.iterdir()
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )
    pr_workflows = 0
    checked = 0
    exempted = 0

    for path in workflows:
        text = path.read_text(encoding="utf-8")
        if not has_pull_request_trigger(text):
            continue
        pr_workflows += 1
        if path in RUNNERLESS_PR_EXEMPTIONS:
            exempted += 1
            continue
        try:
            validate_queue_policy(path, text)
        except QueuePolicyError as error:
            fail(str(error))
        checked += 1

    missing_exemptions = [path for path in RUNNERLESS_PR_EXEMPTIONS if not path.is_file()]
    if missing_exemptions:
        fail(f"runnerless PR exemptions disappeared: {missing_exemptions!r}")

    print("workflow_queue_policy=PASS")
    print(f"workflows_total={len(workflows)}")
    print(f"pull_request_workflows={pr_workflows}")
    print(f"queue_policies_checked={checked}")
    print(f"runnerless_pr_exemptions={exempted}")
    print("automatic_stale_run_supersession=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
