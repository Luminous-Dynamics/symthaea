#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classify and verify RSK Cargo.lock repair lineage.

This module distinguishes three byte sequences:

* base_lock: Cargo.lock from the PR base commit;
* head_lock: Cargo.lock committed at the exact candidate head;
* cargo_lock: Cargo.lock after pinned Cargo re-resolves that exact head.

Generation mode is allowed only when base_lock == head_lock.  Cargo may then
produce a non-admissible candidate, which must satisfy the RSK-only delta
qualifier.

Commit-verification mode applies when head_lock differs from base_lock.  The
committed transition must satisfy the RSK-only delta qualifier, and pinned Cargo
must leave the committed bytes exactly unchanged.  Semantic equivalence is not
sufficient after the candidate is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import rsk_lockfile_delta as delta


class LockfileLineageError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def qualify_repair_lineage(
    base_lock: bytes,
    head_lock: bytes,
    cargo_lock: bytes,
) -> dict[str, object]:
    if base_lock == head_lock:
        candidate_report = delta.qualify_lockfile_delta(base_lock, cargo_lock)
        mode = (
            "already-qualified-idempotent"
            if cargo_lock == head_lock
            else "generated-candidate"
        )
        return {
            "schema": "symthaea.rsk.lockfile-repair-lineage.v1",
            "mode": mode,
            "base_sha256": _sha256(base_lock),
            "head_sha256": _sha256(head_lock),
            "post_cargo_sha256": _sha256(cargo_lock),
            "head_equals_base": True,
            "post_cargo_equals_head": cargo_lock == head_lock,
            "candidate_delta": candidate_report,
        }

    committed_report = delta.qualify_lockfile_delta(base_lock, head_lock)
    if cargo_lock != head_lock:
        raise LockfileLineageError(
            "committed Cargo.lock is not byte-idempotent under pinned Cargo"
        )

    idempotent_report = delta.qualify_lockfile_delta(head_lock, cargo_lock)
    return {
        "schema": "symthaea.rsk.lockfile-repair-lineage.v1",
        "mode": "committed-candidate-verified",
        "base_sha256": _sha256(base_lock),
        "head_sha256": _sha256(head_lock),
        "post_cargo_sha256": _sha256(cargo_lock),
        "head_equals_base": False,
        "post_cargo_equals_head": True,
        "committed_delta": committed_report,
        "post_cargo_idempotence": idempotent_report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_lock")
    parser.add_argument("head_lock")
    parser.add_argument("post_cargo_lock")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    try:
        report = qualify_repair_lineage(
            Path(args.base_lock).read_bytes(),
            Path(args.head_lock).read_bytes(),
            Path(args.post_cargo_lock).read_bytes(),
        )
    except (OSError, delta.LockfileDeltaError, LockfileLineageError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
