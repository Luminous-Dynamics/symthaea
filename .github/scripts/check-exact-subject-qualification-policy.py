#!/usr/bin/env python3
"""Fail closed on weak automatic exact-subject qualification workflows.

Hard enforcement is intentionally narrow: lightweight pull-request workflows
whose filenames follow the repository's reference/custody/oracle conventions,
or whose source explicitly declares the canonical exact-subject checkout step.

Those lanes are latest-head evidence by construction. They must be draft-safe,
revocable when review readiness is withdrawn/closed, cancel superseded PR
attempts, and execute the raw PR head rather than GitHub's synthetic merge ref.

Broader/legacy "*qualification*" workflows are advisory migration candidates
unless they also match the hard exact-subject convention.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

WORKFLOW_DIR = Path(".github/workflows")
HARD_NAME_HINTS = ("reference", "custody", "oracle")
ADVISORY_NAME_PATTERNS = (
    re.compile(r"(?:^|[-_])qual(?:$|[-_])"),
    re.compile(r"qualification"),
    re.compile(r"qualify"),
)

JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
JOB_LEVEL_IF = re.compile(r"^    if:\s*(.*)$")
RUNNER_JOB = re.compile(r"^    (?:runs-on|uses):")
TIMEOUT = re.compile(r"^    timeout-minutes:\s*[1-9][0-9]*\s*$", re.MULTILINE)
BLOCK_MARKERS = {"|", "|-", "|+", ">", ">-", ">+"}
CHECKOUT_USE = re.compile(r"^\s+uses:\s*actions/checkout@([^\s#]+)\s*$")
PINNED_40 = re.compile(r"^[0-9a-f]{40}$")
WRITE_PERMISSION = re.compile(r"^\s+[A-Za-z0-9_-]+:\s*write\s*$", re.MULTILINE)
JOB_PERMISSION = re.compile(r"^    permissions:\s*", re.MULTILINE)

REQUIRED_PR_EVENTS = ("ready_for_review", "converted_to_draft", "closed")
CANONICAL_IF = (
    "github.event_name != 'pull_request' || "
    "(github.event.pull_request.draft == false && github.event.action != 'closed')"
)
CANONICAL_GROUP = (
    "group: ${{ github.workflow }}-${{ github.event_name == 'workflow_dispatch' "
    "&& github.run_id || github.event.pull_request.number || github.ref }}"
)
EXACT_REF = "ref: ${{ github.event.pull_request.head.sha || github.sha }}"
FETCH_DEPTH = "fetch-depth: 0"
NO_PERSIST = "persist-credentials: false"
EXPECTED_ENV = "EXPECTED_HEAD: ${{ github.event.pull_request.head.sha || github.sha }}"
HEAD_ASSERT = 'run: test \"$(git rev-parse HEAD)\" = \"$EXPECTED_HEAD\"'


class PolicyError(ValueError):
    pass


def fail(message: str) -> None:
    print(f"exact-subject-qualification-policy: FAIL: {message}", file=sys.stderr)
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


def pull_request_block(text: str) -> list[str] | None:
    on_lines = top_level_section(text, "on")
    for index, line in enumerate(on_lines):
        match = re.fullmatch(r"  pull_request:\s*(.*)", line)
        if not match:
            continue
        block = [line]
        if match.group(1).strip():
            return block
        for candidate in on_lines[index + 1 :]:
            if not candidate.strip() or candidate.lstrip().startswith("#"):
                block.append(candidate)
                continue
            if indentation(candidate) <= 2:
                break
            block.append(candidate)
        return block
    return None


def parse_jobs(text: str) -> dict[str, str]:
    lines = text.splitlines()
    try:
        jobs_index = next(i for i, line in enumerate(lines) if line == "jobs:")
    except StopIteration as error:
        raise PolicyError("top-level jobs: mapping not found") from error
    starts: list[tuple[int, str]] = []
    for index in range(jobs_index + 1, len(lines)):
        match = JOB_HEADER.fullmatch(lines[index])
        if match:
            starts.append((index, match.group(1)))
    if not starts:
        raise PolicyError("no top-level jobs found")
    jobs: dict[str, str] = {}
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        if name in jobs:
            raise PolicyError(f"duplicate top-level job id {name!r}")
        jobs[name] = "\n".join(lines[start:end]) + "\n"
    return jobs


def job_level_if_expression(block: str, job: str) -> str | None:
    lines = block.splitlines()
    matches: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = JOB_LEVEL_IF.fullmatch(line)
        if match:
            matches.append((index, match.group(1).strip()))
    if len(matches) > 1:
        raise PolicyError(f"job {job!r} has duplicate job-level if keys")
    if not matches:
        return None
    index, value = matches[0]
    if value not in BLOCK_MARKERS:
        if not value:
            raise PolicyError(f"job {job!r} has an empty job-level if expression")
        return " ".join(value.split())
    payload: list[str] = []
    for line in lines[index + 1 :]:
        if not line.strip():
            continue
        if indentation(line) <= 4:
            break
        payload.append(line.strip())
    if not payload:
        raise PolicyError(f"job {job!r} has an empty block-scalar job-level if")
    return " ".join(" ".join(payload).split())


def has_runner_allocation(block: str) -> bool:
    return any(RUNNER_JOB.match(line) for line in block.splitlines())


def hard_exact_subject_lane(path: Path, text: str) -> bool:
    stem = path.stem.lower()
    return (
        any(hint in stem for hint in HARD_NAME_HINTS)
        or "      - name: Checkout exact subject" in text
        or "      - name: Verify exact subject checkout" in text
    )


def advisory_qualification_lane(path: Path) -> bool:
    stem = path.stem.lower()
    return any(pattern.search(stem) for pattern in ADVISORY_NAME_PATTERNS)


def require_lifecycle(path: Path, pr_block: list[str]) -> None:
    active = [
        line.split("#", 1)[0].strip()
        for line in pr_block
        if line.split("#", 1)[0].strip()
    ]
    joined = "\n".join(active)
    missing = [event for event in REQUIRED_PR_EVENTS if event not in joined]
    if missing:
        raise PolicyError(f"{path}: exact-subject lifecycle missing PR events {missing!r}")


def require_read_only_permissions(path: Path, text: str) -> None:
    active = [
        line for line in top_level_section(text, "permissions")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if active != ["  contents: read"]:
        raise PolicyError(
            f"{path}: exact-subject lane permissions must be exactly contents: read"
        )
    if WRITE_PERMISSION.search(text):
        raise PolicyError(f"{path}: exact-subject lane contains write permission")
    if JOB_PERMISSION.search(text):
        raise PolicyError(f"{path}: exact-subject lane must not override permissions per job")


def require_concurrency(path: Path, text: str) -> None:
    active = [
        line.strip() for line in top_level_section(text, "concurrency")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected = [CANONICAL_GROUP, "cancel-in-progress: true"]
    if active != expected:
        raise PolicyError(
            f"{path}: exact-subject concurrency must match canonical latest-head profile"
        )


def step_blocks(job_block: str) -> list[str]:
    lines = job_block.splitlines()
    starts = [
        i for i, line in enumerate(lines)
        if re.match(r"^      - (?:name|uses|run):", line)
    ]
    blocks: list[str] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        blocks.append("\n".join(lines[start:end]) + "\n")
    return blocks


def named_step(job_block: str, name: str) -> str:
    matches = [
        block for block in step_blocks(job_block)
        if block.splitlines()[0].strip() == f"- name: {name}"
    ]
    if len(matches) != 1:
        raise PolicyError(f"expected exactly one step named {name!r}")
    return matches[0]


def require_runner_admission(path: Path, job: str, block: str) -> None:
    expression = job_level_if_expression(block, job)
    if expression != CANONICAL_IF:
        raise PolicyError(
            f"{path}: runner job {job!r} must use the canonical exact-subject admission expression"
        )
    if TIMEOUT.search(block) is None:
        raise PolicyError(f"{path}: runner job {job!r} lacks a positive timeout-minutes")


def require_exact_execution_subject(path: Path, job: str, block: str) -> None:
    checkout_steps = [step for step in step_blocks(block) if "actions/checkout@" in step]
    if len(checkout_steps) != 1:
        raise PolicyError(
            f"{path}: runner job {job!r} must contain exactly one checkout step"
        )
    checkout = named_step(block, "Checkout exact subject")
    if checkout_steps[0] != checkout:
        raise PolicyError(
            f"{path}: runner job {job!r} checkout must be the named exact-subject step"
        )
    uses = [
        match.group(1)
        for line in checkout.splitlines()
        if (match := CHECKOUT_USE.match(line))
    ]
    if len(uses) != 1 or PINNED_40.fullmatch(uses[0]) is None:
        raise PolicyError(
            f"{path}: runner job {job!r} checkout must use one pinned 40-hex commit"
        )
    for needle in (EXACT_REF, FETCH_DEPTH, NO_PERSIST):
        if needle not in checkout:
            raise PolicyError(f"{path}: runner job {job!r} checkout missing {needle!r}")

    verify = named_step(block, "Verify exact subject checkout")
    if EXPECTED_ENV not in verify or HEAD_ASSERT not in verify:
        raise PolicyError(
            f"{path}: runner job {job!r} exact-subject verification is incomplete"
        )


def validate_exact_subject(path: Path, text: str, pr_block: list[str]) -> int:
    require_lifecycle(path, pr_block)
    require_read_only_permissions(path, text)
    require_concurrency(path, text)

    jobs = parse_jobs(text)
    runner_jobs = 0
    for job, block in jobs.items():
        if not has_runner_allocation(block):
            continue
        runner_jobs += 1
        require_runner_admission(path, job, block)
        require_exact_execution_subject(path, job, block)
    if runner_jobs == 0:
        raise PolicyError(f"{path}: exact-subject PR lane has no runner-capable job")
    return runner_jobs


def safe_fixture() -> str:
    return """name: Example independent reference

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review, converted_to_draft, closed]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.event_name == 'workflow_dispatch' && github.run_id || github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  validate:
    name: Validate exact reference
    if: github.event_name != 'pull_request' || (github.event.pull_request.draft == false && github.event.action != 'closed')
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Checkout exact subject
        uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          fetch-depth: 0
          persist-credentials: false
      - name: Verify exact subject checkout
        env:
          EXPECTED_HEAD: ${{ github.event.pull_request.head.sha || github.sha }}
        run: test \"$(git rev-parse HEAD)\" = \"$EXPECTED_HEAD\"
      - run: python3 scripts/reference.py
"""


def expect_rejected(name: str, text: str) -> None:
    pr = pull_request_block(text)
    assert pr is not None
    try:
        validate_exact_subject(Path(name), text, pr)
    except PolicyError:
        return
    raise AssertionError(f"{name}: hostile fixture was accepted")


def self_test() -> None:
    safe = safe_fixture()
    pr = pull_request_block(safe)
    assert pr is not None
    assert validate_exact_subject(Path("example-reference.yml"), safe, pr) == 1

    hostile = {
        "no-converted-reference.yml": safe.replace(", converted_to_draft", ""),
        "no-closed-reference.yml": safe.replace(", closed", ""),
        "no-cancel-reference.yml": safe.replace(
            "cancel-in-progress: true", "cancel-in-progress: false"
        ),
        "merge-ref-reference.yml": safe.replace(
            "          ref: ${{ github.event.pull_request.head.sha || github.sha }}\n", ""
        ),
        "credential-reference.yml": safe.replace(
            "persist-credentials: false", "persist-credentials: true"
        ),
        "no-verify-reference.yml": safe.replace(
            '        run: test \"$(git rev-parse HEAD)\" = \"$EXPECTED_HEAD\"\n', ""
        ),
        "closed-runs-reference.yml": safe.replace(
            " && github.event.action != 'closed'", ""
        ),
        "draft-runs-reference.yml": safe.replace(
            "github.event.pull_request.draft == false",
            "github.event.pull_request.draft == true",
        ),
        "mutable-checkout-reference.yml": safe.replace(
            "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
            "actions/checkout@v4",
        ),
        "second-mutable-checkout-reference.yml": safe.replace(
            "      - name: Verify exact subject checkout\n",
            "      - name: Shadow checkout\n"
            "        uses: actions/checkout@v4\n"
            "      - name: Verify exact subject checkout\n",
        ),
        "job-permissions-reference.yml": safe.replace(
            "    runs-on: ubuntu-latest\n",
            "    permissions:\n      contents: write\n    runs-on: ubuntu-latest\n",
        ),
        "write-reference.yml": safe.replace("contents: read", "contents: write"),
        "or-true-admission-reference.yml": safe.replace(
            "github.event.action != 'closed')",
            "github.event.action != 'closed') || true",
        ),
        "no-timeout-reference.yml": safe.replace("    timeout-minutes: 5\n", ""),
    }
    for name, fixture in hostile.items():
        expect_rejected(name, fixture)

    assert hard_exact_subject_lane(Path("x-reference.yml"), safe)
    assert hard_exact_subject_lane(Path("x-custody.yml"), safe)
    assert hard_exact_subject_lane(Path("plain.yml"), safe)
    assert not hard_exact_subject_lane(Path("coding-agent-quality.yml"), "name: quality\n")
    assert advisory_qualification_lane(Path("foo-qualification.yml"))
    assert not advisory_qualification_lane(Path("coding-agent-quality.yml"))


def main() -> int:
    self_test()
    if not WORKFLOW_DIR.is_dir():
        fail(f"workflow directory not found: {WORKFLOW_DIR}")

    hard_checked = 0
    runner_jobs = 0
    advisory: list[str] = []

    for path in sorted(WORKFLOW_DIR.iterdir()):
        if not path.is_file() or path.suffix not in {".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8")
        pr_block = pull_request_block(text)
        if pr_block is None:
            continue

        if hard_exact_subject_lane(path, text):
            try:
                runner_jobs += validate_exact_subject(path, text, pr_block)
            except PolicyError as error:
                fail(str(error))
            hard_checked += 1
        elif advisory_qualification_lane(path):
            advisory.append(str(path))

    print("exact_subject_qualification_policy=PASS")
    print(f"hard_exact_subject_pr_workflows={hard_checked}")
    print(f"hard_exact_subject_runner_jobs={runner_jobs}")
    print(f"legacy_qualification_advisory_count={len(advisory)}")
    for path in advisory:
        print(f"legacy_qualification_advisory={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
