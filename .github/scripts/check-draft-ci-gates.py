#!/usr/bin/env python3
"""Fail closed if ci.yml can enqueue heavyweight jobs for a draft PR.

This is intentionally a tiny structural ratchet rather than a general YAML
interpreter.  The full CI workflow is large and runner-expensive; the contract
we need to preserve is correspondingly narrow:

* directly runnable pull-request jobs must have the frozen draft guard;
* jobs that rely on `needs: test` may stay unguarded only while `test` itself is
  directly draft-gated and they do not use `always()` to bypass that skip;
* the only root jobs without a draft guard are jobs already restricted away
  from pull_request events (SBOM and scheduled/manual stress tests);
* adding/removing/renaming a CI job requires an explicit update here.

GitHub evaluates jobs.<job_id>.if before matrix expansion, so a false root-job
condition prevents matrix legs from requesting runners.
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
    observed = [candidate for candidate in block.splitlines() if candidate.strip().startswith("if:")]
    if expected not in block.splitlines():
        fail(
            f"job {job!r} is missing exact line {expected!r}; "
            f"observed_if_lines={observed!r}"
        )


def main() -> int:
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
        block = jobs[job]
        if "    needs: test" not in block.splitlines():
            fail(f"dependent job {job!r} no longer has exact `needs: test`")
        if "always()" in block:
            fail(
                f"dependent job {job!r} contains always(), which can bypass a "
                "skipped draft-safe prerequisite"
            )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
