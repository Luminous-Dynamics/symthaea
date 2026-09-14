#!/usr/bin/env python3
"""Fail-closed classifier for Symthaea pull-request CI scope.

V2 admits only non-product research/evidence artifacts:
  * Markdown/TSV evidence under docs/release/evidence/
  * top-level standalone PIE Python oracles named scripts/pie-*-oracle.py
  * top-level standalone CORE Python/Rust oracles named scripts/core-*-oracle.py/.rs

Everything else requires full generic CI.

`--name-status-z` parses `git diff --name-status -z`, including both source and
destination paths for rename/copy records. That prevents a product file renamed
into an admitted evidence path from bypassing full CI. Git type changes are
always full-CI because a regular file <-> symlink transition is not an ordinary
evidence-only content edit.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

EVIDENCE_PREFIX = "docs/release/evidence/"
PIE_ORACLE_RE = re.compile(r"^scripts/pie-[a-z0-9][a-z0-9-]*-oracle\.py$")
CORE_ORACLE_RE = re.compile(r"^scripts/core-[a-z0-9][a-z0-9-]*-oracle\.(?:py|rs)$")
EVIDENCE_SUFFIXES = (".md", ".tsv")

EVIDENCE_ONLY = "evidence_only"
FULL_CI_REQUIRED = "full_ci_required"


@dataclass(frozen=True)
class Change:
    status: str
    paths: tuple[str, ...]


def _safe_repo_path(path: str) -> bool:
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        return False
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in path):
        return False
    parts = path.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def _admitted_path(path: str) -> bool:
    if not _safe_repo_path(path):
        return False
    if path.startswith(EVIDENCE_PREFIX):
        relative = path[len(EVIDENCE_PREFIX) :]
        return bool(relative) and relative.endswith(EVIDENCE_SUFFIXES)
    return PIE_ORACLE_RE.fullmatch(path) is not None or CORE_ORACLE_RE.fullmatch(path) is not None


def _valid_status(status: str) -> tuple[bool, int]:
    """Return whether a Git name-status token is admitted and its path count."""
    if status in {"A", "M", "D"}:
        return True, 1
    if status.startswith(("R", "C")):
        score = status[1:]
        if score.isdigit() and 0 <= int(score) <= 100:
            return True, 2
        return False, 0
    # T/U/X/B and any future/unknown status fail closed to full CI.
    return False, 0


def classify(changes: tuple[Change, ...]) -> str:
    if not changes:
        return FULL_CI_REQUIRED

    for change in changes:
        valid, expected_paths = _valid_status(change.status)
        if not valid or len(change.paths) != expected_paths:
            return FULL_CI_REQUIRED
        if any(not _admitted_path(path) for path in change.paths):
            return FULL_CI_REQUIRED
    return EVIDENCE_ONLY


def parse_name_status_z(data: bytes) -> tuple[Change, ...]:
    if not data:
        return ()

    fields = data.split(b"\0")
    if fields[-1] != b"":
        raise ValueError("name-status stream must end with NUL")
    fields.pop()

    out: list[Change] = []
    i = 0
    while i < len(fields):
        try:
            status = fields[i].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("non-ASCII git status") from exc
        i += 1
        valid, path_count = _valid_status(status)
        if not valid:
            # Preserve the unknown/type-change record for fail-closed classify,
            # but consume one ordinary path when present so malformed framing is
            # still distinguished from a valid unsupported status.
            path_count = 1
        if i + path_count > len(fields):
            raise ValueError("truncated name-status record")
        paths: list[str] = []
        for raw in fields[i : i + path_count]:
            try:
                paths.append(raw.decode("utf-8", "strict"))
            except UnicodeDecodeError as exc:
                raise ValueError("non-UTF-8 path") from exc
        i += path_count
        out.append(Change(status, tuple(paths)))
    return tuple(out)


def self_test() -> None:
    ev_md = "docs/release/evidence/PIE_TEST.md"
    ev_tsv = "docs/release/evidence/CORE_TEST.tsv"
    pie_oracle = "scripts/pie-test-oracle.py"
    core_py = "scripts/core-test-oracle.py"
    core_rs = "scripts/core-test-oracle.rs"

    assert classify((Change("M", (ev_md,)),)) == EVIDENCE_ONLY
    assert classify((Change("A", (ev_tsv,)),)) == EVIDENCE_ONLY
    assert classify((Change("A", (pie_oracle,)), Change("M", (ev_md,)))) == EVIDENCE_ONLY
    assert classify((Change("A", (core_py,)), Change("A", (core_rs,)))) == EVIDENCE_ONLY
    assert classify((Change("D", (pie_oracle,)),)) == EVIDENCE_ONLY
    assert (
        classify(
            (Change("R100", (ev_md, "docs/release/evidence/PIE_RENAMED.md")),)
        )
        == EVIDENCE_ONLY
    )
    assert (
        classify((Change("C87", (core_py, "scripts/core-copy-oracle.py")),))
        == EVIDENCE_ONLY
    )

    required = (
        Change("M", ("crates/domains/foo/src/lib.rs",)),
        Change("M", ("Cargo.lock",)),
        Change("M", (".github/workflows/ci.yml",)),
        Change("M", ("scripts/foo.sh",)),
        Change("M", ("scripts/core-test.rs",)),
        Change("M", ("scripts/pie-test-oracle.rs",)),
        Change("M", ("docs/release/evidence/receipt.json",)),
        Change("M", ("docs/release/evidence/../src/lib.rs",)),
        Change("M", ("docs/release/evidence/bad\nname.md",)),
        Change("R100", ("crates/domains/foo/src/lib.rs", ev_md)),
        Change("R100", (ev_md, "crates/domains/foo/src/lib.rs")),
        Change("T", (ev_md,)),
        Change("Mgarbage", (ev_md,)),
        Change("R", (ev_md, "docs/release/evidence/x.md")),
        Change("R101", (ev_md, "docs/release/evidence/x.md")),
    )
    for change in required:
        assert classify((change,)) == FULL_CI_REQUIRED, change

    assert classify(()) == FULL_CI_REQUIRED
    assert classify((Change("?", (ev_md,)),)) == FULL_CI_REQUIRED
    assert classify((Change("R100", (ev_md,)),)) == FULL_CI_REQUIRED

    payload = (
        b"A\0scripts/pie-test-oracle.py\0"
        b"A\0scripts/core-test-oracle.rs\0"
        b"M\0docs/release/evidence/PIE_TEST.md\0"
        b"M\0docs/release/evidence/CORE_TEST.tsv\0"
        b"R100\0docs/release/evidence/OLD.md\0docs/release/evidence/NEW.md\0"
    )
    assert classify(parse_name_status_z(payload)) == EVIDENCE_ONLY

    smuggle = (
        b"R100\0crates/domains/foo/src/lib.rs\0"
        b"docs/release/evidence/lib.md\0"
    )
    assert classify(parse_name_status_z(smuggle)) == FULL_CI_REQUIRED

    type_change = b"T\0docs/release/evidence/PIE_TEST.md\0"
    assert classify(parse_name_status_z(type_change)) == FULL_CI_REQUIRED

    for malformed in (
        b"A\0foo",
        b"R100\0old\0",
        b"\xff\0x\0",
        b"R100\0old\0new",  # missing terminal NUL
    ):
        try:
            parse_name_status_z(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed stream accepted: {malformed!r}")

    print("ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-status-z", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("ok")
        return 0
    if args.name_status_z:
        try:
            changes = parse_name_status_z(sys.stdin.buffer.read())
        except ValueError as exc:
            print(f"{FULL_CI_REQUIRED}: {exc}", file=sys.stderr)
            print(FULL_CI_REQUIRED)
            return 0
        print(classify(changes))
        return 0
    parser.error("choose --name-status-z or --self-test")


if __name__ == "__main__":
    raise SystemExit(main())
