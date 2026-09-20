#!/usr/bin/env python3
"""Fail closed if ci.yml can enqueue heavyweight jobs for a draft PR.

This is intentionally a tiny structural ratchet rather than a general YAML
interpreter. The full CI workflow is large and runner-expensive; the contract
we need to preserve is correspondingly narrow:

* directly runnable pull-request jobs must have the frozen draft guard;
* jobs that rely on `needs: test` may stay unguarded only while `test` itself is
  directly draft-gated and their JOB-LEVEL condition cannot override normal
  prerequisite-success propagation;
* step-level cleanup/upload conditions such as `if: always()` are irrelevant to
  runner admission because they are evaluated only after the job has started;
* the only root jobs without a draft guard are jobs already restricted away
  from pull_request events (SBOM and scheduled/manual stress tests);
* adding/removing/renaming a CI job requires an explicit update here.

GitHub evaluates jobs.<job_id>.if before matrix expansion, so a false root-job
condition prevents matrix legs from requesting runners. A skipped `needs`
prerequisite also skips dependent jobs unless a job-level conditional overrides
that propagation. GitHub's status-check functions suppress the implicit
`success()` guard, so dependent jobs using any such function require an explicit
review instead of being accepted transitively here.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

CI_PATH = Path(".github/workflows/ci.yml")

GENERIC_DRAFT_GUARD = (
    "if: github.event_name != 'pull_request' || "
    "github.event.pull_request.draft == false"
)
GOVERNANCE_DRAFT_GUARD = (
    "if: github.event_name == 'pull_request' && "
    "github.event.pull_request.draft == false"
)

DIRECT_GENERIC = {
    "fmt",
    "cls-field-count",
    "workspace-targets",
    "embodiment-safety-composition",
    "orphan-modules",
    "doc-tests",
    "muse",
    "clippy",
    "test",
    "test-spark-engine",
    "test-nextest",
    "test-hardened-lib",
    "test-cognitive-hardening",
    "test-hardened-daemon",
    "test-hardened-api",
    "test-hardened-nix",
    "test-all-features",
    "test-feature-matrix",
    "wasm-compat",
    "deny",
    "audit",
    "compliance-safety-ethics",
    "compliance-robustness",
    "secrets-scan",
}

DIRECT_SPECIAL = {"governance"}

DEPENDENT_ON_TEST = {
    "test-integration",
    "test-feature-matrix-critical",
    "psych-bench",
    "feature-interactions",
    "test-subcrates",
    "genesis-benchmarks",
    "compliance-consciousness",
}

NON_PULL_REQUEST_ROOTS = {
    "sbom": "if: github.ref == 'refs/heads/main' && github.event_name == 'push'",
    "stress-tests": (
        "if: github.event_name == 'schedule' || "
        "github.event_name == 'workflow_dispatch'"
    ),
}

EXPECTED_JOBS = (
    DIRECT_GENERIC
    | DIRECT_SPECIAL
    | DEPENDENT_ON_TEST
    | set(NON_PULL_REQUEST_ROOTS)
)

JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
JOB_LEVEL_IF = re.compile(r"^    if:\s*(.*)$")
STATUS_CHECK = re.compile(r"\b(?:always|cancelled|failure|success)\s*\(")
BLOCK_MARKERS = {"|", "|-", "|+", ">", ">-", ">+"}


def fail(message: str) -> None:
    print(f"draft-ci-gate-check: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_jobs(text: str) -> dict[str, str]:
    lines = text.splitlines()
    try:
        jobs_index = next(i for i, line in enumerate(lines) if line == "jobs:")
    except StopIteration:
        fail("top-level jobs: mapping not found")

    starts: list[tuple[int, str]] = []
    for index in range(jobs_index + 1, len(lines)):
        match = JOB_HEADER.fullmatch(lines[index])
        if match:
            starts.append((index, match.group(1)))

    if not starts:
        fail("no top-level CI jobs found")

    jobs: dict[str, str] = {}
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        if name in jobs:
            fail(f"duplicate top-level job id {name!r}")
        jobs[name] = "\n".join(lines[start:end]) + "\n"
    return jobs


def require_exact_line(block: str, line: str, job: str) -> None:
    expected = f"    {line}"
    observed = [
        candidate
        for candidate in block.splitlines()
        if candidate.strip().startswith("if:")
    ]
    if expected not in block.splitlines():
        fail(
            f"job {job!r} is missing exact line {expected!r}; "
            f"observed_if_lines={observed!r}"
        )


def job_level_if_expression(block: str, job: str) -> str | None:
    """Return only the job-level `if` expression, never step-level conditions.

    The workflow currently contains one block-scalar job condition (psych-bench),
    so support the standard literal/folded YAML markers while failing closed on
    malformed or duplicate job-level `if` keys.
    """

    lines = block.splitlines()
    matches: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = JOB_LEVEL_IF.fullmatch(line)
        if match:
            matches.append((index, match.group(1).strip()))

    if len(matches) > 1:
        fail(f"job {job!r} has duplicate job-level if keys")
    if not matches:
        return None

    index, value = matches[0]
    if value not in BLOCK_MARKERS:
        if not value:
            fail(f"job {job!r} has an empty job-level if expression")
        return value

    payload: list[str] = []
    for line in lines[index + 1 :]:
        if not line.strip():
            payload.append("")
            continue
        # A job-level YAML key is indented four spaces. The block-scalar payload
        # must be deeper; once indentation returns to job scope, the expression
        # is complete. Step-level `if` keys therefore never enter this payload.
        indent = len(line) - len(line.lstrip(" "))
        if indent <= 4:
            break
        payload.append(line.strip())

    if not payload or not any(payload):
        fail(f"job {job!r} has an empty block-scalar job-level if expression")
    return "\n".join(payload)


def require_transitive_draft_safety(block: str, job: str) -> None:
    if "    needs: test" not in block.splitlines():
        fail(f"dependent job {job!r} no longer has exact `needs: test`")

    expression = job_level_if_expression(block, job)
    if expression is not None and STATUS_CHECK.search(expression):
        fail(
            f"dependent job {job!r} uses a job-level status-check function; "
            "that can suppress GitHub's implicit success() prerequisite guard "
            "and requires explicit draft-admission review"
        )


def self_test_job_scope_parser() -> None:
    safe = """  psych-bench:
    needs: test
    if: |
      github.event_name == 'pull_request'
    steps:
      - name: Upload even after a failing step
        if: steps.example.outcome == 'failure' && always()
"""
    expression = job_level_if_expression(safe, "self-test-safe")
    if expression != "github.event_name == 'pull_request'":
        fail(
            "internal self-test failed: block job-level expression was not "
            f"isolated correctly: {expression!r}"
        )
    if STATUS_CHECK.search(expression):
        fail("internal self-test failed: step-level always() leaked into job scope")

    for function in ("always", "cancelled", "failure", "success"):
        unsafe = f"""  dependent:
    needs: test
    if: {function}() || github.event_name == 'pull_request'
    steps:
      - run: true
"""
        observed = job_level_if_expression(unsafe, f"self-test-{function}")
        if observed is None or not STATUS_CHECK.search(observed):
            fail(
                "internal self-test failed: job-level status function was not "
                f"detected: {function}"
            )


def main() -> int:
    self_test_job_scope_parser()

    text = CI_PATH.read_text(encoding="utf-8")
    jobs = parse_jobs(text)

    observed = set(jobs)
    if observed != EXPECTED_JOBS:
        missing = sorted(EXPECTED_JOBS - observed)
        unexpected = sorted(observed - EXPECTED_JOBS)
        fail(
            "top-level CI job census changed without updating the draft-safety "
            f"ratchet: missing={missing!r} unexpected={unexpected!r}"
        )

    for job in sorted(DIRECT_GENERIC):
        require_exact_line(jobs[job], GENERIC_DRAFT_GUARD, job)

    require_exact_line(
        jobs["governance"],
        GOVERNANCE_DRAFT_GUARD,
        "governance",
    )

    for job in sorted(DEPENDENT_ON_TEST):
        require_transitive_draft_safety(jobs[job], job)

    for job, guard in sorted(NON_PULL_REQUEST_ROOTS.items()):
        require_exact_line(jobs[job], guard, job)

    # The dependency theorem relies on the test root being directly gated.
    require_exact_line(jobs["test"], GENERIC_DRAFT_GUARD, "test")

    print("draft_ci_gate_check=PASS")
    print(f"ci_jobs_total={len(jobs)}")
    print(f"direct_generic_guarded={len(DIRECT_GENERIC)}")
    print(f"direct_special_guarded={len(DIRECT_SPECIAL)}")
    print(f"transitively_guarded_via_test={len(DEPENDENT_ON_TEST)}")
    print(f"non_pull_request_roots={len(NON_PULL_REQUEST_ROOTS)}")
    print("job_scope_status_override_check=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
