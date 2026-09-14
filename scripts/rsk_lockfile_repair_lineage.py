#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classify and verify RSK Cargo.lock repair lineage.

The committed Cargo.lock remains the authority boundary. Cargo metadata may
justify only narrowly bounded pre-existing dependency-edge additions; it never
permits package identity, source, version, checksum, removal, or new external
package drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import rsk_lockfile_delta as delta


class LockfileLineageError(ValueError):
    def __init__(self, message: str, *, code: str = "lockfile_lineage_rejected") -> None:
        super().__init__(message)
        self.code = code


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def qualify_repair_lineage(
    base_lock: bytes,
    head_lock: bytes,
    cargo_lock: bytes,
    cargo_metadata: bytes | None = None,
) -> dict[str, object]:
    if base_lock == head_lock:
        candidate_report = delta.qualify_lockfile_delta(
            base_lock, cargo_lock, cargo_metadata
        )
        mode = (
            "already-qualified-idempotent"
            if cargo_lock == head_lock
            else "generated-candidate"
        )
        return {
            "schema": "symthaea.rsk.lockfile-repair-lineage.v2",
            "status": "accepted",
            "mode": mode,
            "base_sha256": _sha256(base_lock),
            "head_sha256": _sha256(head_lock),
            "post_cargo_sha256": _sha256(cargo_lock),
            "head_equals_base": True,
            "post_cargo_equals_head": cargo_lock == head_lock,
            "candidate_delta": candidate_report,
        }

    committed_report = delta.qualify_lockfile_delta(
        base_lock, head_lock, cargo_metadata
    )
    if cargo_lock != head_lock:
        raise LockfileLineageError(
            "committed Cargo.lock is not byte-idempotent under pinned Cargo",
            code="committed_candidate_not_idempotent",
        )

    idempotent_report = delta.qualify_lockfile_delta(
        head_lock, cargo_lock, cargo_metadata
    )
    return {
        "schema": "symthaea.rsk.lockfile-repair-lineage.v2",
        "status": "accepted",
        "mode": "committed-candidate-verified",
        "base_sha256": _sha256(base_lock),
        "head_sha256": _sha256(head_lock),
        "post_cargo_sha256": _sha256(cargo_lock),
        "head_equals_base": False,
        "post_cargo_equals_head": True,
        "committed_delta": committed_report,
        "post_cargo_idempotence": idempotent_report,
    }


def _read_optional(path: str | None) -> bytes | None:
    return Path(path).read_bytes() if path is not None else None


def _rejection_report(
    *,
    exc: Exception,
    base_lock: bytes | None,
    head_lock: bytes | None,
    cargo_lock: bytes | None,
) -> dict[str, Any]:
    reason_code = getattr(exc, "code", "io_or_unclassified_failure")
    report: dict[str, Any] = {
        "schema": "symthaea.rsk.lockfile-repair-lineage.v2",
        "status": "rejected",
        "reason_code": reason_code,
        "reason": str(exc),
    }
    if base_lock is not None:
        report["base_sha256"] = _sha256(base_lock)
    if head_lock is not None:
        report["head_sha256"] = _sha256(head_lock)
    if cargo_lock is not None:
        report["post_cargo_sha256"] = _sha256(cargo_lock)
    details = getattr(exc, "details", None)
    if isinstance(details, dict) and details:
        report["details"] = details
    return report


def _write_report(path: str | None, report: dict[str, Any]) -> None:
    if path is not None:
        Path(path).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_lock")
    parser.add_argument("head_lock")
    parser.add_argument("post_cargo_lock")
    parser.add_argument("--cargo-metadata")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    base_lock: bytes | None = None
    head_lock: bytes | None = None
    cargo_lock: bytes | None = None
    try:
        base_lock = Path(args.base_lock).read_bytes()
        head_lock = Path(args.head_lock).read_bytes()
        cargo_lock = Path(args.post_cargo_lock).read_bytes()
        metadata_path = (
            Path(args.cargo_metadata)
            if args.cargo_metadata is not None
            else Path(args.post_cargo_lock).with_name("cargo-metadata.json")
        )
        cargo_metadata = (
            metadata_path.read_bytes()
            if metadata_path.exists()
            else None
        )
        report = qualify_repair_lineage(
            base_lock, head_lock, cargo_lock, cargo_metadata
        )
    except (OSError, delta.LockfileDeltaError, LockfileLineageError) as exc:
        report = _rejection_report(
            exc=exc,
            base_lock=base_lock,
            head_lock=head_lock,
            cargo_lock=cargo_lock,
        )
        try:
            _write_report(args.json_out, report)
        except OSError as write_exc:
            print(f"FAIL: could not write rejection report: {write_exc}", file=sys.stderr)
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    _write_report(args.json_out, report)
    print(json.dumps(report, sort_keys=True, indent=2) + "\n", end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
