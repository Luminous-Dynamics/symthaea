#!/usr/bin/env python3
"""Fail-closed classifier for Symthaea pull-request CI scope.

V1 admits only:
  * Markdown evidence under docs/release/evidence/
  * top-level standalone PIE oracle scripts named scripts/pie-*-oracle.py

Everything else requires full generic CI.

`--name-status-z` parses `git diff --name-status -z`, including both source and
destination paths for rename/copy records. That prevents a product file renamed
into an admitted evidence path from bypassing full CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

EVIDENCE_PREFIX = "docs/release/evidence/"
ORACLE_RE = re.compile(r"^scripts/pie-[a-z0-9][a-z0-9-]*-oracle\.py$")

EVIDENCE_ONLY = "evidence_only"
FULL_CI_REQUIRED = "full_ci_required"


@dataclass(frozen=True)
class Change:
    status: str
    paths: tuple[str, ...]


def _safe_repo_path(path: str) -> bool:
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        return False
    parts = path.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def _admitted_path(path: str) -> bool:
    if not _safe_repo_path(path):
        return False
    if path.startswith(EVIDENCE_PREFIX):
        relative = path[len(EVIDENCE_PREFIX) :]
        return bool(relative) and relative.endswith(".md")
    return ORACLE_RE.fullmatch(path) is not None


def classify(changes: tuple[Change, ...]) -> str:
    if not changes:
        return FULL_CI_REQUIRED

    for change in changes:
        if not change.status:
            return FULL_CI_REQUIRED
        kind = change.status[0]
        expected_paths = 2 if kind in {"R", "C"} else 1
        if len(change.paths) != expected_paths:
            return FULL_CI_REQUIRED
        if kind not in {"A", "M", "D", "R", "C", "T"}:
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
        if not status:
            raise ValueError("empty git status")
        kind = status[0]
        path_count = 2 if kind in {"R", "C"} else 1
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
    ev = "docs/release/evidence/PIE_TEST.md"
    oracle = "scripts/pie-test-oracle.py"

    assert classify((Change("M", (ev,)),)) == EVIDENCE_ONLY
    assert classify((Change("A", (oracle,)), Change("M", (ev,)))) == EVIDENCE_ONLY
    assert classify((Change("D", (oracle,)),)) == EVIDENCE_ONLY
    assert (
        classify(
            (Change("R100", (ev, "docs/release/evidence/PIE_RENAMED.md")),)
        )
        == EVIDENCE_ONLY
    )

    required = (
        Change("M", ("crates/domains/foo/src/lib.rs",)),
        Change("M", ("Cargo.lock",)),
        Change("M", (".github/workflows/ci.yml",)),
        Change("M", ("scripts/foo.sh",)),
        Change("M", ("docs/release/evidence/receipt.json",)),
        Change("M", ("docs/release/evidence/../src/lib.rs",)),
        Change("R100", ("crates/domains/foo/src/lib.rs", ev)),
        Change("R100", (ev, "crates/domains/foo/src/lib.rs")),
    )
    for change in required:
        assert classify((change,)) == FULL_CI_REQUIRED, change

    assert classify(()) == FULL_CI_REQUIRED
    assert classify((Change("?", (ev,)),)) == FULL_CI_REQUIRED
    assert classify((Change("R100", (ev,)),)) == FULL_CI_REQUIRED

    payload = (
        b"A\0scripts/pie-test-oracle.py\0"
        b"M\0docs/release/evidence/PIE_TEST.md\0"
        b"R100\0docs/release/evidence/OLD.md\0docs/release/evidence/NEW.md\0"
    )
    assert classify(parse_name_status_z(payload)) == EVIDENCE_ONLY

    smuggle = (
        b"R100\0crates/domains/foo/src/lib.rs\0"
        b"docs/release/evidence/lib.md\0"
    )
    assert classify(parse_name_status_z(smuggle)) == FULL_CI_REQUIRED

    for malformed in (b"A\0foo", b"R100\0old\0", b"\xff\0x\0"):
        try:
            parse_name_status_z(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed stream accepted: {malformed!r}")


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
