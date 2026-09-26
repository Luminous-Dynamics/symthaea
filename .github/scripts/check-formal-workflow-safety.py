#!/usr/bin/env python3
"""Static fail-closed contract for Symthaea formal-evidence workflows.

This checker exists because an earlier Lean qualification lane produced a green
GitHub conclusion while the retained Lean output contained real tactic errors and
`sorryAx`: `lean ... | tee ...` ran under a shell without `pipefail`, and an
older checkout also qualified the synthetic PR merge commit instead of the exact
source branch head.

The checker is intentionally narrow. It only governs workflow files whose names
belong to the SYM-FV / SYM-HDC-CRYPTO-FV formal-evidence families. It does not
pretend to validate arbitrary GitHub Actions semantics.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FORMAL_WORKFLOW = re.compile(
    r"(?:^|/)(?:sym-fv|sym-hdc-crypto-fv)[^/]*\.ya?ml$", re.IGNORECASE
)

EXACT_HEAD_EXPR = "${{ github.event.pull_request.head.sha || github.sha }}"
EXACT_REF_MARKER = f"ref: {EXACT_HEAD_EXPR}"

# Commands whose exit status may establish or deny formal-evidence authority.
# They must never be hidden behind `| tee` in a formal workflow. Capture output
# to a file first, preserve the command status, then print/retain the file.
AUTHORITATIVE_COMMAND = re.compile(
    r"(?:\blean\b|\blake\b|\bcargo\s+(?:test|nextest)\b|\bverus\b|"
    r"\bapalache\b|\btlc\b|\bquint\b)",
    re.IGNORECASE,
)
TEE_PIPE = re.compile(r"\|\s*tee\b")


class ContractError(RuntimeError):
    pass


def normalize_shell(text: str) -> str:
    """Collapse shell continuation newlines for conservative line checks."""
    return re.sub(r"\\\r?\n\s*", " ", text)


def authoritative_tee_lines(text: str) -> list[str]:
    normalized = normalize_shell(text)
    bad: list[str] = []
    for raw in normalized.splitlines():
        line = raw.strip()
        if TEE_PIPE.search(line) and AUTHORITATIVE_COMMAND.search(line):
            bad.append(line)
    return bad


def validate_formal_workflow(path: Path, text: str) -> None:
    errors: list[str] = []

    if EXACT_REF_MARKER not in text:
        errors.append(
            "checkout must explicitly pin `ref: " + EXACT_HEAD_EXPR + "`"
        )

    # We require an explicit observed-vs-requested HEAD assertion, not merely a
    # checkout ref. Formatting may vary, so bind this semantically by requiring
    # both the Git observation and exact requested-head expression in the file.
    if "git rev-parse HEAD" not in text or EXACT_HEAD_EXPR not in text:
        errors.append(
            "workflow must assert observed `git rev-parse HEAD` against the requested exact head"
        )

    bad_pipes = authoritative_tee_lines(text)
    if bad_pipes:
        errors.append(
            "authoritative prover/test command piped through tee; capture output before printing: "
            + " || ".join(bad_pipes)
        )

    if "cancel-in-progress: false" not in text:
        errors.append(
            "formal evidence workflows must retain non-cancelling per-subject qualification"
        )

    if "contents: read" not in text:
        errors.append("formal evidence workflow must keep explicit read-only contents permission")

    if errors:
        joined = "\n  - ".join(errors)
        raise ContractError(f"{path}:\n  - {joined}")


def selected_formal_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if FORMAL_WORKFLOW.search(path.as_posix())]


def run_files(paths: list[Path]) -> None:
    selected = selected_formal_paths(paths)
    for path in selected:
        if not path.is_file():
            raise ContractError(f"formal workflow path does not exist: {path}")
        validate_formal_workflow(path, path.read_text(encoding="utf-8"))
    print(f"formal workflow safety: PASS ({len(selected)} formal workflow(s) checked)")


def self_test() -> None:
    good = f"""
name: good
permissions:\n  contents: read
concurrency:\n  cancel-in-progress: false
steps:
  - uses: actions/checkout@pinned
    with:
      ref: {EXACT_HEAD_EXPR}
  - run: |
      test \"$(git rev-parse HEAD)\" = \"{EXACT_HEAD_EXPR}\"
      if ! lean proof.lean > /tmp/lean.txt 2>&1; then
        cat /tmp/lean.txt
        exit 1
      fi
      cat /tmp/lean.txt
      git diff --name-only HEAD^ HEAD | sort | tee /tmp/paths.txt
"""
    validate_formal_workflow(Path("sym-fv-good.yml"), good)

    cases = {
        "lean-pipe": good.replace(
            "if ! lean proof.lean > /tmp/lean.txt 2>&1; then\n        cat /tmp/lean.txt\n        exit 1\n      fi\n      cat /tmp/lean.txt",
            "lean proof.lean | tee /tmp/lean.txt",
        ),
        "cargo-pipe": good.replace(
            "if ! lean proof.lean > /tmp/lean.txt 2>&1; then\n        cat /tmp/lean.txt\n        exit 1\n      fi\n      cat /tmp/lean.txt",
            "cargo test -p bridge --test formal_audit \\\n        | tee /tmp/test.txt",
        ),
        "missing-exact-ref": good.replace(f"ref: {EXACT_HEAD_EXPR}", "ref: ${{ github.sha }}"),
        "missing-head-assert": good.replace("git rev-parse HEAD", "git status --short"),
        "cancelling": good.replace("cancel-in-progress: false", "cancel-in-progress: true"),
    }

    for name, mutant in cases.items():
        try:
            validate_formal_workflow(Path(f"sym-fv-{name}.yml"), mutant)
        except ContractError:
            continue
        raise ContractError(f"self-test false green: {name}")

    # A benign tee used only to retain path selection is intentionally allowed.
    if authoritative_tee_lines("git diff --name-only HEAD^ HEAD | sort | tee /tmp/paths.txt"):
        raise ContractError("self-test false positive: benign path-list tee")

    print(f"formal workflow safety self-test: PASS ({len(cases)} hostile mutants rejected)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            self_test()
        if args.paths:
            run_files(args.paths)
        elif not args.self_test:
            raise ContractError("no workflow paths supplied")
        return 0
    except (ContractError, OSError, UnicodeError) as exc:
        print(f"formal workflow safety: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
