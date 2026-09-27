#!/usr/bin/env python3
"""Fail closed on weak automatic exact-subject qualification workflows.

Hard enforcement is intentionally narrow: lightweight pull-request workflows
whose filenames follow the repository's reference/custody/oracle conventions,
or whose source explicitly declares an exact-subject checkout step.

Those lanes are latest-head evidence by construction. They must be draft-safe,
revocable when review readiness is withdrawn/closed, cancel superseded PR
attempts, and execute the raw PR head rather than GitHub's synthetic merge ref.

Broader/legacy "*qualification*" workflows are reported as advisory candidates
for migration unless they also match the hard exact-subject convention. Formal
verifier workflows may also have stronger dedicated policy ratchets.
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
EXACT_STEP_HINTS = ("Checkout exact subject", "Verify exact subject checkout")

JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
JOB_LEVEL_IF = re.compile(r"^    if:\s*(.*)$")
RUNNER_JOB = re.compile(r"^    (?:runs-on|uses):")
TIMEOUT = re.compile(r"^    timeout-minutes:\s*[1-9][0-9]*\s*$", re.MULTILINE)
BLOCK_MARKERS = {"|", "|-", "|+", ">", ">-", ">+"}
DRAFT_FALSE = re.compile(r"github\.event\.pull_request\.draft\s*==\s*false")
CLOSED_EXCLUDED = re.compile(r"github\.event\.action\s*!=\s*['\"]closed['\"]")
PINNED_CHECKOUT = re.compile(r"uses:\s*actions/checkout@[0-9a-f]{40}\b")

REQUIRED_PR_EVENTS = ("ready_for_review", "converted_to_draft", "closed")
EXACT_REF = "ref: ${{ github.event.pull_request.head.sha || github.sha }}"
FETCH_DEPTH = "fetch-depth: 0"
NO_PERSIST = "persist-credentials: false"
EXPECTED_ENV = "EXPECTED_HEAD: ${{ github.event.pull_request.head.sha || github.sha }}"
HEAD_ASSERT = 'test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"'


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
        return value

    payload: list[str] = []
    for line in lines[index + 1 :]:
        if not line.strip():
            payload.append("")
            continue
        if indentation(line) <= 4:
            break
        payload.append(line.strip())
    if not payload or not any(payload):
        raise PolicyError(f"job {job!r} has an empty block-scalar job-level if")
    return "\n".join(payload)


def has_runner_allocation(block: str) -> bool:
    return any(RUNNER_JOB.match(line) for line in block.splitlines())


def hard_exact_subject_lane(path: Path, text: str) -> bool:
    stem = path.stem.lower()
    return (
        any(hint in stem for hint in HARD_NAME_HINTS)
        or any(hint in text for hint in EXACT_STEP_HINTS)
    )


def advisory_qualification_lane(path: Path) -> bool:
    stem = path.stem.lower()
    return any(pattern.search(stem) for pattern in ADVISORY_NAME_PATTERNS)


def require_lifecycle(path: Path, pr_block: list[str]) -> None:
    joined = "\n".join(pr_block)
    missing = [event for event in REQUIRED_PR_EVENTS if event not in joined]
    if missing:
        raise PolicyError(
            f"{path}: exact-subject lifecycle missing PR events {missing!r}"
        )


def require_read_only_permissions(path: Path, text: str) -> None:
    section = "\n".join(top_level_section(text, "permissions"))
    if "contents: read" not in section:
        raise PolicyError(f"{path}: exact-subject lane must declare contents: read")
    if re.search(r"\bwrite\b", section):
        raise PolicyError(f"{path}: exact-subject lane requests write permission")


def require_concurrency(path: Path, text: str) -> None:
    section = "\n".join(top_level_section(text, "concurrency"))
    if not section:
        raise PolicyError(f"{path}: exact-subject lane lacks top-level concurrency")
    required = (
        "cancel-in-progress: true",
        "github.workflow",
        "github.event.pull_request.number",
        "github.event_name == 'workflow_dispatch'",
        "github.run_id",
    )
    missing = [needle for needle in required if needle not in section]
    if missing:
        raise PolicyError(
            f"{path}: exact-subject concurrency missing {missing!r}"
        )


def require_runner_admission(path: Path, job: str, block: str) -> None:
    expression = job_level_if_expression(block, job)
    if expression is None or DRAFT_FALSE.search(expression) is None:
        raise PolicyError(
            f"{path}: runner job {job!r} lacks pull_request draft == false"
        )
    if CLOSED_EXCLUDED.search(expression) is None:
        raise PolicyError(
            f"{path}: runner job {job!r} does not explicitly exclude closed"
        )
    if TIMEOUT.search(block) is None:
        raise PolicyError(
            f"{path}: runner job {job!r} lacks a positive timeout-minutes"
        )


def require_exact_execution_subject(path: Path, text: str) -> None:
    if PINNED_CHECKOUT.search(text) is None:
        raise PolicyError(f"{path}: actions/checkout is not pinned to a 40-hex commit")
    required = (EXACT_REF, FETCH_DEPTH, NO_PERSIST, EXPECTED_ENV, HEAD_ASSERT)
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise PolicyError(
            f"{path}: exact execution-subject contract missing {missing!r}"
        )


def validate_exact_subject(path: Path, text: str, pr_block: list[str]) -> int:
    require_lifecycle(path, pr_block)
    require_read_only_permissions(path, text)
    require_concurrency(path, text)
    require_exact_execution_subject(path, text)

    jobs = parse_jobs(text)
    runner_jobs = 0
    for job, block in jobs.items():
        if not has_runner_allocation(block):
            continue
        runner_jobs += 1
        require_runner_admission(path, job, block)
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
            "          ref: ${{ github.event.pull_request.head.sha || github.sha }}\n",
            "",
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
        "write-reference.yml": safe.replace(
            "contents: read", "contents: write"
        ),
        "no-timeout-reference.yml": safe.replace(
            "    timeout-minutes: 5\n", ""
        ),
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
