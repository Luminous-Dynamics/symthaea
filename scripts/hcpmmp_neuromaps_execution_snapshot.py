#!/usr/bin/env python3
"""Private, content-verified scientific-input snapshots for HCP-MMP Lineage B.

This module deliberately performs no Workbench execution and grants no scientific,
provenance, independence, or runtime authority. It only copies the exact scientific
input bytes named by a validated run manifest into a fresh private directory and
re-verifies every copied digest before returning paths that a later execution layer
may choose to consume.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from hcpmmp_neuromaps_common import (
    INPUT_KEYS,
    REQUIRED_INPUT_ROLES,
    ContractError,
    canonical_json_bytes,
    digest_bytes,
    digest_file,
    exact,
    nonempty,
    sha256,
)

SNAPSHOT_PROFILE = "symthaea-hcpmmp-scientific-input-snapshot-v1"


@dataclass(frozen=True)
class ScientificInputSnapshotV1:
    """Authority-free receipt for one private scientific-input snapshot."""

    profile: str
    root: Path
    paths: Mapping[str, Path]
    expected_digests: Mapping[str, str]
    input_set_digest: str


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _input_contract(run: Any) -> dict[str, dict[str, str]]:
    if not isinstance(run, dict) or not isinstance(run.get("inputs"), dict):
        raise ContractError("execution snapshot: run inputs object required")
    inputs = run["inputs"]
    if set(inputs) != REQUIRED_INPUT_ROLES:
        raise ContractError("execution snapshot: exact v1 scientific input roles required")
    validated: dict[str, dict[str, str]] = {}
    for role in sorted(inputs):
        entry = exact(inputs[role], INPUT_KEYS, f"snapshot input {role}")
        path = nonempty(entry["path"], f"snapshot input {role} path")
        digest = sha256(entry["sha256"], f"snapshot input {role} sha256")
        validated[role] = {"path": path, "sha256": digest}
    return validated


def _open_regular_source(source: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(source, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ContractError("execution snapshot: opened source is not a regular file")
        return fd
    except Exception:
        os.close(fd)
        raise


def _copy_verified(source: Path, destination: Path, expected_digest: str) -> None:
    hasher = hashlib.sha256()
    source_fd = _open_regular_source(source)
    destination_fd: int | None = None
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(source_fd, "rb", closefd=False) as in_handle:
            with os.fdopen(destination_fd, "wb", closefd=False) as out_handle:
                for chunk in iter(lambda: in_handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
                    out_handle.write(chunk)
                out_handle.flush()
                os.fsync(out_handle.fileno())
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)

    copied_digest = "sha256:" + hasher.hexdigest()
    if copied_digest != expected_digest:
        raise ContractError("execution snapshot: copied bytes do not match committed digest")
    if digest_file(destination) != expected_digest:
        raise ContractError("execution snapshot: read-back digest mismatch")
    os.chmod(destination, 0o400)


def _cleanup_failed_snapshot(root: Path, parent: Path) -> None:
    try:
        if os.path.lexists(root):
            shutil.rmtree(root)
        if os.path.lexists(root):
            raise ContractError("execution snapshot: failed snapshot cleanup not confirmed")
        _fsync_dir(parent)
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError("execution snapshot: failed snapshot cleanup not confirmed") from exc


def build_scientific_input_snapshot(
    run: Any,
    snapshot_root: Path,
) -> ScientificInputSnapshotV1:
    """Copy all 14 scientific inputs into one fresh private snapshot.

    The source paths come only from the run manifest. Each copied file is verified
    against that role's committed SHA-256 both while copying and by read-back. The
    actual opened source descriptor must be a regular file. The snapshot is a
    cooperative local custody primitive; it is not protection against a malicious
    same-UID process or a compromised kernel/filesystem.
    """

    inputs = _input_contract(run)
    requested = snapshot_root.expanduser()
    parent = requested.parent.resolve(strict=True)
    root = parent / requested.name
    if os.path.lexists(root):
        raise ContractError("execution snapshot: destination already exists")

    os.mkdir(root, 0o700)
    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    try:
        for role in sorted(inputs):
            entry = inputs[role]
            source = Path(entry["path"]).expanduser().resolve(strict=True)
            destination = root / f"{role}.input"
            _copy_verified(source, destination, entry["sha256"])
            paths[role] = destination
            expected[role] = entry["sha256"]
        _fsync_dir(root)
        _fsync_dir(parent)
    except Exception:
        _cleanup_failed_snapshot(root, parent)
        raise

    input_set_digest = digest_bytes(
        canonical_json_bytes({role: expected[role] for role in sorted(expected)})
    )
    return ScientificInputSnapshotV1(
        profile=SNAPSHOT_PROFILE,
        root=root,
        paths=MappingProxyType(dict(paths)),
        expected_digests=MappingProxyType(dict(expected)),
        input_set_digest=input_set_digest,
    )
