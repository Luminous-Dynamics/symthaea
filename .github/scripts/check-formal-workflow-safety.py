#!/usr/bin/env python3
"""Fail closed on unsafe formal-evidence GitHub workflow shapes.

This is intentionally a structural ratchet, not a YAML interpreter. It protects
proof/model-checking qualification workflows from status-masking and source-head
ambiguity that can turn a red prover into misleading green evidence.

The generic draft-runner policy remains owned by check-workflow-draft-safety.py.
This checker adds formal-specific invariants: exact-head checkout, pinned checkout
action, non-persistent credentials, no `prover | tee` authority path, no
continue-on-error, and hostile-prover controls for Lean lanes.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

WORKFLOW_DIR = Path(".github/workflows")

FORMAL_FILENAME_MARKERS = (
    "sym-fv-",
    "sym-hdc-crypto-fv-",
    "formal-",
)
FORMAL_CONTENT_MARKERS = (
    "nix develop .#formal",
    "symthaea-proof-audit",
    "AxiomPolicy::",
    "#print axioms",
)
CHECKOUT = re.compile(r"uses:\s*actions/checkout@([^\s]+)")
PINNED_SHA = re.compile(r"^[0-9a-f]{40}$")
PROVER_TEE = re.compile(
    r"(?im)^.*\b(?:lean|lake|verus|tlc|apalache|cargo\s+test)\b[^\n]*\|\s*tee\b"
)
LEAN_COMMAND = re.compile(r"(?im)(?:^|[;&|()]\s*|\s)lean(?:\s|$)")
CONTINUE_ON_ERROR = re.compile(r"(?im)^\s*continue-on-error:\s*true\s*$")
PULL_REQUEST_TARGET = re.compile(r"(?m)^\s*pull_request_target\s*:")

EXACT_REF = 'ref: ${{ github.event.pull_request.head.sha || github.sha }}'
PERSIST_FALSE = "persist-credentials: false"
FETCH_FULL = "fetch-depth: 0"


class SafetyError(ValueError):
    pass


def fail(message: str) -> None:
    print(f"formal-workflow-safety: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def is_formal_workflow(path: Path, text: str) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in FORMAL_FILENAME_MARKERS) or any(
        marker in text for marker in FORMAL_CONTENT_MARKERS
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SafetyError(message)


def validate_formal_workflow(path: Path, text: str) -> None:
    require(
        PULL_REQUEST_TARGET.search(text) is None,
        f"{path}: pull_request_target is forbidden for formal evidence workflows",
    )
    require(
        CONTINUE_ON_ERROR.search(text) is None,
        f"{path}: continue-on-error: true is forbidden",
    )

    checkouts = CHECKOUT.findall(text)
    require(checkouts, f"{path}: formal workflow has no actions/checkout step")
    for ref in checkouts:
        require(
            PINNED_SHA.fullmatch(ref) is not None,
            f"{path}: actions/checkout must be pinned by full commit SHA, got {ref!r}",
        )

    checkout_count = len(checkouts)
    require(
        text.count(EXACT_REF) >= checkout_count,
        f"{path}: every checkout must bind the immutable PR head / github.sha",
    )
    require(
        text.count(PERSIST_FALSE) >= checkout_count,
        f"{path}: every checkout must set persist-credentials: false",
    )
    require(
        text.count(FETCH_FULL) >= checkout_count,
        f"{path}: every formal checkout must retain full history for exact lineage checks",
    )
    require(
        "git rev-parse HEAD" in text and "github.event.pull_request.head.sha" in text,
        f"{path}: formal workflow must explicitly compare checked-out HEAD to requested head",
    )

    match = PROVER_TEE.search(text)
    require(
        match is None,
        f"{path}: prover/test command may not pipe into tee as authority: {match.group(0).strip() if match else ''}",
    )

    # Lean theorem lanes must prove that a deliberately invalid subject is
    # observed as failure and must keep the axiom/error sentinels visible.
    if LEAN_COMMAND.search(text):
        lowered = text.lower()
        require(
            "known-bad" in lowered,
            f"{path}: Lean formal workflow lacks a known-bad rejection control",
        )
        require(
            "sorryax" in lowered,
            f"{path}: Lean formal workflow lacks an explicit sorryAx rejection sentinel",
        )
        require(
            "error:" in lowered,
            f"{path}: Lean formal workflow lacks an explicit Lean error-marker sentinel/control",
        )


def self_test() -> None:
    safe = r'''name: Safe formal
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
jobs:
  prove:
    if: github.event_name != 'pull_request' || github.event.pull_request.draft == false
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
        with:
          fetch-depth: 0
          persist-credentials: false
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
      - run: |
          set -euo pipefail
          test "$(git rev-parse HEAD)" = "${{ github.event.pull_request.head.sha || github.sha }}"
          if lean /tmp/known-bad.lean > /tmp/known-bad-output.txt 2>&1; then exit 1; fi
          grep -q 'error:' /tmp/known-bad-output.txt
          if ! lean proof.lean > /tmp/output.txt 2>&1; then cat /tmp/output.txt; exit 1; fi
          if grep -q 'sorryAx' /tmp/output.txt; then exit 1; fi
'''
    validate_formal_workflow(Path("sym-fv-safe.yml"), safe)

    mutants = {
        "unpinned checkout": safe.replace(
            "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
            "actions/checkout@v6",
        ),
        "missing exact ref": safe.replace(
            "          ref: ${{ github.event.pull_request.head.sha || github.sha }}\n", ""
        ),
        "persistent credentials": safe.replace(
            "persist-credentials: false", "persist-credentials: true"
        ),
        "prover tee": safe.replace(
            "if ! lean proof.lean > /tmp/output.txt 2>&1; then cat /tmp/output.txt; exit 1; fi",
            "lean proof.lean | tee /tmp/output.txt",
        ),
        "continue on error": safe.replace(
            "    runs-on: ubuntu-latest", "    continue-on-error: true\n    runs-on: ubuntu-latest"
        ),
        "missing known-bad": safe.replace("known-bad", "negative-control"),
        "pull request target": safe.replace("  pull_request:", "  pull_request_target:"),
    }
    for label, mutant in mutants.items():
        try:
            validate_formal_workflow(Path(f"mutant-{label}.yml"), mutant)
        except SafetyError:
            pass
        else:
            raise AssertionError(f"unsafe formal workflow mutant admitted: {label}")


def main() -> int:
    self_test()
    if not WORKFLOW_DIR.is_dir():
        fail(f"workflow directory not found: {WORKFLOW_DIR}")

    workflows = sorted(
        p for p in WORKFLOW_DIR.iterdir() if p.is_file() and p.suffix in {".yml", ".yaml"}
    )
    formal: list[Path] = []
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        if not is_formal_workflow(path, text):
            continue
        formal.append(path)
        try:
            validate_formal_workflow(path, text)
        except SafetyError as error:
            fail(str(error))

    print("formal_workflow_safety=PASS")
    print(f"workflows_total={len(workflows)}")
    print(f"formal_workflows_checked={len(formal)}")
    for path in formal:
        print(f"formal_workflow={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
