#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed qualifier for RSK Cargo.lock repair candidates.

This tool does not generate Cargo.lock. Cargo must generate the candidate on a
pinned executor. This tool only proves that the candidate preserves every
pre-existing non-RSK package record exactly and introduces/retains only the
expected RSK workspace path-package records and dependency edges.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

EXPECTED_RSK: dict[str, set[str]] = {
    "symthaea-replicator-semantics": set(),
    "symthaea-replicator-safety": {
        "symthaea-replicator-semantics",
        "sha2",
        "serde_json",
        "hex",
    },
    "symthaea-replicator-ledger": {"symthaea-replicator-safety"},
}
EXPECTED_VERSION = "0.1.0"


class LockfileDeltaError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse(data: bytes) -> dict[str, Any]:
    try:
        parsed = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise LockfileDeltaError(f"invalid Cargo.lock TOML: {exc}") from exc
    if not isinstance(parsed.get("package"), list):
        raise LockfileDeltaError("Cargo.lock must contain [[package]] records")
    return parsed


def _canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _dependency_name(raw: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise LockfileDeltaError("package dependency entries must be non-empty strings")
    return raw.split(" ", 1)[0]


def _records_by_name(
    packages: list[dict[str, Any]], names: set[str]
) -> dict[str, list[dict[str, Any]]]:
    out = {name: [] for name in names}
    for record in packages:
        name = record.get("name")
        if name in out:
            out[name].append(record)
    return out


def _validate_rsk_record(name: str, record: dict[str, Any]) -> None:
    if record.get("name") != name:
        raise LockfileDeltaError(f"{name}: record name mismatch")
    if record.get("version") != EXPECTED_VERSION:
        raise LockfileDeltaError(
            f"{name}: expected version {EXPECTED_VERSION}, got {record.get('version')!r}"
        )
    if "source" in record or "checksum" in record:
        raise LockfileDeltaError(
            f"{name}: workspace path package must not have source/checksum"
        )
    deps_raw = record.get("dependencies", [])
    if not isinstance(deps_raw, list):
        raise LockfileDeltaError(f"{name}: dependencies must be a list")
    deps = {_dependency_name(dep) for dep in deps_raw}
    if len(deps) != len(deps_raw):
        raise LockfileDeltaError(f"{name}: duplicate dependency names are not allowed")
    expected = EXPECTED_RSK[name]
    if deps != expected:
        raise LockfileDeltaError(
            f"{name}: dependency set mismatch; expected {sorted(expected)}, got {sorted(deps)}"
        )


def qualify_lockfile_delta(before: bytes, after: bytes) -> dict[str, Any]:
    old = _parse(before)
    new = _parse(after)

    if old.get("version") != new.get("version"):
        raise LockfileDeltaError(
            f"lockfile format version changed: {old.get('version')!r} -> {new.get('version')!r}"
        )
    if old.get("version") != 4:
        raise LockfileDeltaError(
            f"expected Cargo.lock version = 4, got {old.get('version')!r}"
        )

    old_top = {key: value for key, value in old.items() if key != "package"}
    new_top = {key: value for key, value in new.items() if key != "package"}
    if old_top != new_top:
        raise LockfileDeltaError("non-package top-level Cargo.lock data changed")

    names = set(EXPECTED_RSK)
    old_packages = old["package"]
    new_packages = new["package"]
    old_rsk = _records_by_name(old_packages, names)
    new_rsk = _records_by_name(new_packages, names)

    for name in sorted(names):
        if len(old_rsk[name]) > 1:
            raise LockfileDeltaError(
                f"{name}: before lock contains duplicate package records"
            )
        if len(new_rsk[name]) != 1:
            raise LockfileDeltaError(
                f"{name}: after lock must contain exactly one package record, got {len(new_rsk[name])}"
            )
        _validate_rsk_record(name, new_rsk[name][0])
        if old_rsk[name]:
            _validate_rsk_record(name, old_rsk[name][0])
            if _canonical_record(old_rsk[name][0]) != _canonical_record(new_rsk[name][0]):
                raise LockfileDeltaError(f"{name}: pre-existing RSK record changed")

    old_non_rsk = Counter(
        _canonical_record(record)
        for record in old_packages
        if record.get("name") not in names
    )
    new_non_rsk = Counter(
        _canonical_record(record)
        for record in new_packages
        if record.get("name") not in names
    )
    if old_non_rsk != new_non_rsk:
        removed = list((old_non_rsk - new_non_rsk).elements())
        added = list((new_non_rsk - old_non_rsk).elements())
        raise LockfileDeltaError(
            "non-RSK package graph drift detected"
            + (f"; removed/changed={len(removed)}" if removed else "")
            + (f"; added/changed={len(added)}" if added else "")
        )

    added_rsk = sorted(name for name in names if not old_rsk[name])
    status = (
        "rsk-path-records-added"
        if added_rsk
        else "already-qualified-no-rsk-delta"
    )
    return {
        "schema": "symthaea.rsk.lockfile-delta-report.v1",
        "status": status,
        "before_sha256": _sha256(before),
        "after_sha256": _sha256(after),
        "lockfile_version": old["version"],
        "added_rsk_packages": added_rsk,
        "rsk_packages_after": {
            name: {
                "version": new_rsk[name][0]["version"],
                "dependencies": sorted(
                    _dependency_name(dep)
                    for dep in new_rsk[name][0].get("dependencies", [])
                ),
            }
            for name in sorted(names)
        },
        "non_rsk_package_records_unchanged": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    try:
        report = qualify_lockfile_delta(
            Path(args.before).read_bytes(), Path(args.after).read_bytes()
        )
    except (OSError, LockfileDeltaError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
