#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the canonical MuJoCo 3.8.0 runtime manifest consumed by 003B2.

The extracted-tree digest is path-independent and excludes runtime-manifest.json
itself. Version `symthaea.extracted-tree.sha256.v1` binds the runtime root mode,
sorted relative paths, entry type, permission mode, regular-file size/content
SHA-256, and relative symlink targets. Ownership, timestamps, and xattrs are
intentionally normalized out of this semantic digest. Absolute, broken,
looping, escaping, or special filesystem entries are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

SCHEMA_ID = "symthaea.qualification-mujoco-runtime.v1"
EXTRACTED_TREE_DIGEST_ALGORITHM = "symthaea.extracted-tree.sha256.v1"
VERSION = "3.8.0"
MUJOCO_RS_COMPATIBILITY = "4.0.1+mj-3.8.0"
UPSTREAM_RELEASE_COMMIT = "34d69ad4cb1a21846b8297e2bc5e68a4938276c1"
MANIFEST_NAME = "runtime-manifest.json"

ASSETS: dict[str, dict[str, Any]] = {
    "x86_64-linux": {
        "upstream_asset_id": 404891937,
        "upstream_asset_name": "mujoco-3.8.0-linux-x86_64.tar.gz",
        "upstream_asset_size": 20812715,
        "upstream_asset_sha256": "2be88c6f92a06c3eaffdb47d3a6d3fbf159fbc057e9d272d592fb194e41fefab",
    },
    "aarch64-linux": {
        "upstream_asset_id": 404891905,
        "upstream_asset_name": "mujoco-3.8.0-linux-aarch64.tar.gz",
        "upstream_asset_size": 20674425,
        "upstream_asset_sha256": "adc4a7856d2b8d42ba4e889b57cbceb13a329c869f410cf1ad110b153c4745e4",
    },
}


class ManifestError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _entry_record(root: Path, path: Path) -> bytes:
    relative = path.relative_to(root).as_posix()
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)

    if stat.S_ISLNK(info.st_mode):
        target = path.readlink()
        if target.is_absolute():
            raise ManifestError(f"absolute symlink is not admitted: {relative} -> {target}")
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            raise ManifestError(f"unresolved symlink: {relative}") from exc
        if not _within(resolved, root):
            raise ManifestError(f"symlink escapes runtime tree: {relative} -> {resolved}")
        return f"L\0{relative}\0{mode:o}\0{target.as_posix()}\n".encode("utf-8")

    if stat.S_ISDIR(info.st_mode):
        return f"D\0{relative}\0{mode:o}\n".encode("utf-8")

    if stat.S_ISREG(info.st_mode):
        return (
            f"F\0{relative}\0{mode:o}\0{info.st_size}\0{_sha256_file(path)}\n"
        ).encode("utf-8")

    raise ManifestError(f"special filesystem entry is not admitted: {relative}")


def extracted_tree_sha256(root: Path) -> str:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ManifestError("runtime root must be a directory")

    digest = hashlib.sha256()
    root_mode = stat.S_IMODE(root.lstat().st_mode)
    digest.update(f"D\0.\0{root_mode:o}\n".encode("utf-8"))
    paths = sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix())
    for path in paths:
        if path.parent == root and path.name == MANIFEST_NAME:
            continue
        digest.update(_entry_record(root, path))
    return digest.hexdigest()


def build_manifest(root: Path, platform: str) -> dict[str, Any]:
    if platform not in ASSETS:
        raise ManifestError(f"unsupported MuJoCo qualification platform: {platform}")

    root = root.resolve(strict=True)
    lib = root / "lib" / "libmujoco.so"
    if not lib.exists():
        raise ManifestError("runtime must provide lib/libmujoco.so")
    try:
        resolved_lib = lib.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise ManifestError("libmujoco.so cannot be resolved") from exc
    if not _within(resolved_lib, root):
        raise ManifestError("libmujoco.so escapes runtime root")
    if not resolved_lib.is_file():
        raise ManifestError("resolved libmujoco.so must be a regular file")

    manifest: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "platform": platform,
        "version": VERSION,
        "mujoco_rs_compatibility": MUJOCO_RS_COMPATIBILITY,
        "upstream_release_commit": UPSTREAM_RELEASE_COMMIT,
        **ASSETS[platform],
        "extracted_tree_digest_algorithm": EXTRACTED_TREE_DIGEST_ALGORITHM,
        "extracted_tree_sha256": extracted_tree_sha256(root),
        "libmujoco_sha256": _sha256_file(resolved_lib),
    }
    return manifest


def write_manifest(root: Path, platform: str) -> Path:
    root = root.resolve(strict=True)
    manifest = build_manifest(root, platform)
    output = root / MANIFEST_NAME
    temporary = root / f".{MANIFEST_NAME}.tmp"
    temporary.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--platform", required=True, choices=sorted(ASSETS))
    args = parser.parse_args()

    output = write_manifest(Path(args.root), args.platform)
    print(output)
    print(_sha256_file(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
