#!/usr/bin/env python3
"""Fail closed if repository PR workflows can allocate runners while draft.

This is a structural admission ratchet, not a general YAML interpreter. It
covers the repository's current workflow conventions and deliberately fails on
unsupported shapes so changes receive explicit review.

The heavyweight ci.yml theorem remains delegated to check-draft-ci-gates.py.
benchmarks.yml is separately proven PR-runnerless. `workflow-syntax.yml` is the
single deliberate draft-runner exception: the admission gate itself must run on
drafts to reject unsafe workflow definitions before those definitions can earn
runner access. That exception is shape-checked here and is not available to any
other workflow.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

WORKFLOW_DIR = Path(".github/workflows")
CI_WORKFLOW = Path(".github/workflows/ci.yml")
BENCHMARKS_WORKFLOW = Path(".github/workflows/benchmarks.yml")
WORKFLOW_SYNTAX_WORKFLOW = Path(".github/workflows/workflow-syntax.yml")
CI_RATCHET = Path(".github/scripts/check-draft-ci-gates.py")

JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
JOB_LEVEL_IF = re.compile(r"^    if:\s*(.*)$")
BLOCK_MARKERS = {"|", "|-", "|+", ">", ">-", ">+"}
RUNNER_JOB = re.compile(r"^    (?:runs-on|uses):")

DRAFT_FALSE = re.compile(r"github\.event\.pull_request\.draft\s*==\s*false")
EVENT_EQ = {
    event: re.compile(rf"github\.event_name\s*==\s*['\"]{re.escape(event)}['\"]")
    for event in ("push", "workflow_dispatch", "schedule")
}


class SafetyError(ValueError):
    pass


def fail(message: str) -> None:
    print(f"workflow-draft-safety: FAIL: {message}", file=sys.stderr)
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
        inline = match.group(1).strip()
        if inline:
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
        raise SafetyError("top-level jobs: mapping not found") from error

    starts: list[tuple[int, str]] = []
    for index in range(jobs_index + 1, len(lines)):
        match = JOB_HEADER.fullmatch(lines[index])
        if match:
            starts.append((index, match.group(1)))
    if not starts:
        raise SafetyError("no top-level jobs found")

    jobs: dict[str, str] = {}
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        if name in jobs:
            raise SafetyError(f"duplicate top-level job id {name!r}")
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
        raise SafetyError(f"job {job!r} has duplicate job-level if keys")
    if not matches:
        return None

    index, value = matches[0]
    if value not in BLOCK_MARKERS:
        if not value:
            raise SafetyError(f"job {job!r} has an empty job-level if expression")
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
        raise SafetyError(f"job {job!r} has an empty block-scalar job-level if")
    return "\n".join(payload)


def has_runner_allocation(block: str) -> bool:
    return any(RUNNER_JOB.match(line) for line in block.splitlines())


def explicitly_excludes_pull_request(expression: str | None) -> bool:
    if expression is None:
        return False
    if "||" in expression:
        return False
    return any(pattern.search(expression) for pattern in EVENT_EQ.values())


def has_draft_guard(expression: str | None) -> bool:
    return expression is not None and DRAFT_FALSE.search(expression) is not None


def require_ready_event(path: Path, pr_block: list[str]) -> None:
    if not any("ready_for_review" in line for line in pr_block):
        raise SafetyError(
            f"{path}: runner-capable pull_request workflow must include ready_for_review"
        )


def validate_generic(path: Path, text: str, pr_block: list[str]) -> tuple[int, int]:
    jobs = parse_jobs(text)
    runner_jobs = 0
    draft_guarded = 0
    for job, block in jobs.items():
        if not has_runner_allocation(block):
            continue
        runner_jobs += 1
        expression = job_level_if_expression(block, job)
        if explicitly_excludes_pull_request(expression):
            continue
        if not has_draft_guard(expression):
            raise SafetyError(
                f"{path}: runner-capable job {job!r} lacks a job-level "
                "pull_request draft == false guard or explicit non-PR event guard"
            )
        draft_guarded += 1

    if runner_jobs:
        require_ready_event(path, pr_block)
    return runner_jobs, draft_guarded


def validate_workflow_syntax(text: str, pr_block: list[str]) -> tuple[int, int, int]:
    """Admit the one lightweight draft runner that enforces workflow safety."""
    jobs = parse_jobs(text)
    if set(jobs) != {"actionlint"}:
        raise SafetyError(
            "workflow-syntax.yml draft exception requires exactly one job named 'actionlint'"
        )
    block = jobs["actionlint"]
    if "    runs-on: ubuntu-slim" not in block.splitlines():
        raise SafetyError("workflow-syntax.yml draft exception must stay on ubuntu-slim")
    if "    timeout-minutes: 5" not in block.splitlines():
        raise SafetyError("workflow-syntax.yml draft exception must retain timeout-minutes: 5")
    if job_level_if_expression(block, "actionlint") is not None:
        raise SafetyError(
            "workflow-syntax.yml safety gate must not carry a draft-false guard; "
            "it is the single draft admission checker"
        )
    if "permissions:\n  contents: read\n" not in text:
        raise SafetyError("workflow-syntax.yml draft exception must remain read-only")
    require_ready_event(WORKFLOW_SYNTAX_WORKFLOW, pr_block)
    return 1, 0, 1


def require_contains(expression: str | None, needle: str, label: str) -> None:
    if expression is None or needle not in expression:
        raise SafetyError(f"benchmarks.yml {label} lost required expression {needle!r}")


def validate_benchmarks(text: str) -> tuple[int, int]:
    jobs = parse_jobs(text)
    expected = {
        "quick-check",
        "full-benchmark",
        "consciousness-benchmarks",
        "ethics-benchmarks",
        "aggregate-results",
        "osworld",
    }
    if set(jobs) != expected:
        raise SafetyError(
            "benchmarks.yml job census changed without updating the runnerless PR proof: "
            f"missing={sorted(expected - set(jobs))!r} "
            f"unexpected={sorted(set(jobs) - expected)!r}"
        )
    quick = job_level_if_expression(jobs["quick-check"], "quick-check")
    require_contains(quick, "github.event_name == 'push'", "quick-check")
    for job in ("full-benchmark", "consciousness-benchmarks", "ethics-benchmarks", "osworld"):
        expression = job_level_if_expression(jobs[job], job)
        require_contains(expression, "github.event_name == 'workflow_dispatch'", job)

    aggregate = job_level_if_expression(jobs["aggregate-results"], "aggregate-results")
    require_contains(
        aggregate,
        "needs.full-benchmark.result == 'success'",
        "aggregate-results",
    )
    if "    needs: [full-benchmark]" not in jobs["aggregate-results"].splitlines():
        raise SafetyError(
            "benchmarks.yml aggregate-results must keep exact needs: [full-benchmark]"
        )
    return len(jobs), 0


def validate_ci_delegation() -> tuple[int, int]:
    if not CI_RATCHET.is_file():
        raise SafetyError(f"delegated CI ratchet missing: {CI_RATCHET}")
    completed = subprocess.run([sys.executable, str(CI_RATCHET)], check=False, text=True)
    if completed.returncode != 0:
        raise SafetyError("delegated ci.yml draft-safety ratchet failed")
    return 1, 1


def self_test() -> None:
    safe = """on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
jobs:
  test:
    if: github.event_name != 'pull_request' || github.event.pull_request.draft == false
    runs-on: ubuntu-latest
    steps:
      - run: true
"""
    pr = pull_request_block(safe)
    assert pr is not None
    assert validate_generic(Path("safe.yml"), safe, pr) == (1, 1)

    unsafe = safe.replace(
        "    if: github.event_name != 'pull_request' || github.event.pull_request.draft == false\n",
        "",
    )
    try:
        validate_generic(Path("unsafe.yml"), unsafe, pull_request_block(unsafe) or [])
    except SafetyError:
        pass
    else:
        raise AssertionError("unguarded runner job was accepted")

    no_ready = safe.replace(", ready_for_review", "")
    try:
        validate_generic(Path("no-ready.yml"), no_ready, pull_request_block(no_ready) or [])
    except SafetyError:
        pass
    else:
        raise AssertionError("runner workflow without ready_for_review was accepted")

    manual = """on:
  pull_request:
  workflow_dispatch:
jobs:
  manual:
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - run: true
"""
    pr = pull_request_block(manual)
    assert pr is not None
    assert validate_generic(Path("manual.yml"), manual, pr) == (1, 0)

    syntax_gate = """on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
permissions:
  contents: read
jobs:
  actionlint:
    name: actionlint
    runs-on: ubuntu-slim
    timeout-minutes: 5
    steps:
      - run: true
"""
    pr = pull_request_block(syntax_gate)
    assert pr is not None
    assert validate_workflow_syntax(syntax_gate, pr) == (1, 0, 1)

    guarded_syntax = syntax_gate.replace(
        "    runs-on: ubuntu-slim\n",
        "    if: github.event.pull_request.draft == false\n    runs-on: ubuntu-slim\n",
    )
    try:
        validate_workflow_syntax(guarded_syntax, pull_request_block(guarded_syntax) or [])
    except SafetyError:
        pass
    else:
        raise AssertionError("draft-disabled workflow-syntax gate was admitted")


def main() -> int:
    self_test()
    if not WORKFLOW_DIR.is_dir():
        fail(f"workflow directory not found: {WORKFLOW_DIR}")

    workflows = sorted(
        path for path in WORKFLOW_DIR.iterdir()
        if path.is_file() and path.suffix in {".yml", ".yaml"}
    )
    checked = 0
    pr_workflows = 0
    runner_jobs = 0
    draft_guarded = 0
    draft_exceptions = 0

    for path in workflows:
        text = path.read_text(encoding="utf-8")
        pr_block = pull_request_block(text)
        if pr_block is None:
            continue
        pr_workflows += 1
        try:
            if path == CI_WORKFLOW:
                runners, guarded = validate_ci_delegation()
                exceptions = 0
            elif path == BENCHMARKS_WORKFLOW:
                runners, guarded = validate_benchmarks(text)
                exceptions = 0
            elif path == WORKFLOW_SYNTAX_WORKFLOW:
                runners, guarded, exceptions = validate_workflow_syntax(text, pr_block)
            else:
                runners, guarded = validate_generic(path, text, pr_block)
                exceptions = 0
        except SafetyError as error:
            fail(str(error))
        checked += 1
        runner_jobs += runners
        draft_guarded += guarded
        draft_exceptions += exceptions

    if draft_exceptions != 1:
        fail(f"expected exactly one draft-runner admission exception, observed {draft_exceptions}")

    print("workflow_draft_safety=PASS")
    print(f"workflows_total={len(workflows)}")
    print(f"pull_request_workflows={pr_workflows}")
    print(f"pull_request_workflows_checked={checked}")
    print(f"runner_jobs_or_delegated_roots={runner_jobs}")
    print(f"direct_draft_guarded_or_delegated={draft_guarded}")
    print(f"draft_runner_admission_exceptions={draft_exceptions}")
    print("draft_runner_exception=workflow-syntax.yml:actionlint")
    print("benchmarks_pull_request_runnerless_proof=PASS")
    print("ci_draft_safety_delegation=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
