#!/usr/bin/env python3
"""PHI-SEM-001A-R2: Git-object-backed execution profile for the Phi census.

R1's exact qualifier demonstrated that filesystem dereferencing of a tracked
symlink can make a lexical inventory fail before it reaches any Phi semantics.
R2 reuses the frozen R1 lexical engine byte-for-byte, but supplies tracked bytes
from Git index objects and makes symlink handling explicit.

Claim ceiling remains measurement-only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "scripts" / "audit_phi_authority_flow.py"
WRAPPER_REL = "scripts/audit_phi_authority_flow_r2.py"
DOC_REL = "docs/research/PHI_SEM_001A_INVENTORY_V2.md"
PROFILE_ID = "phi-sem-001a-lexical-v2"
IO_SEMANTICS = "git-index-object-bytes-v2"
REGULAR_MODES = {"100644", "100755"}
SYMLINK_MODE = "120000"

spec = importlib.util.spec_from_file_location("phi_sem_001a_r1_engine", ENGINE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen R1 engine: {ENGINE_PATH}")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

_index_cache: dict[str, tuple[str, str]] | None = None
_blob_cache: dict[str, bytes] = {}


def index_entries() -> dict[str, tuple[str, str]]:
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    raw = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    entries: dict[str, tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, path_bytes = record.split(b"\t", 1)
        mode, blob, stage = meta.decode("ascii").split()
        if stage != "0":
            raise RuntimeError(f"unmerged index entry for {path_bytes!r}")
        path = path_bytes.decode("utf-8")
        entries[path] = (mode, blob)
    _index_cache = entries
    return entries


def blob_bytes(blob: str) -> bytes:
    cached = _blob_cache.get(blob)
    if cached is not None:
        return cached
    result = subprocess.run(
        ["git", "cat-file", "blob", blob],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    _blob_cache[blob] = result.stdout
    return result.stdout


def tracked_object(path: str) -> tuple[str, str, bytes]:
    try:
        mode, blob = index_entries()[path]
    except KeyError as exc:
        raise RuntimeError(f"tracked path missing from index: {path}") from exc
    if mode not in REGULAR_MODES | {SYMLINK_MODE}:
        raise RuntimeError(f"unsupported tracked object mode {mode} for {path}")
    return mode, blob, blob_bytes(blob)


def read_text_v2(path: str) -> tuple[str, bytes] | None:
    mode, _blob, data = tracked_object(path)

    # A symlink blob contains the link target path, not the target file bytes.
    # Treating those bytes as source text would confuse pointer metadata with the
    # referenced artifact, while dereferencing would make the result checkout-
    # and filesystem-dependent. Preserve its bytes for fingerprints, but expose
    # no lexical text.
    if mode == SYMLINK_MODE:
        return "", data

    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8"), data
    except UnicodeDecodeError:
        return None


_original_build_inventory = engine.build_inventory
_original_profile_sha256 = engine.profile_sha256


def build_inventory_v2(entries: list[tuple[str, str]]) -> list[dict[str, Any]]:
    inventory = _original_build_inventory(entries)
    modes = index_entries()
    for item in inventory:
        path = item["path"]
        item["git_mode"] = modes[path][0]
    return inventory


def subtree_fingerprint_v2(
    entries_by_path: dict[str, str], prefix: str
) -> dict[str, object]:
    members: list[tuple[str, str, str, str]] = []
    modes = index_entries()
    for path in sorted(entries_by_path):
        if not path.startswith(prefix):
            continue
        rel = path[len(prefix):]
        mode, expected_blob = modes[path]
        if expected_blob != entries_by_path[path]:
            raise RuntimeError(f"index/blob disagreement for {path}")
        if mode not in REGULAR_MODES | {SYMLINK_MODE}:
            raise RuntimeError(f"unsupported tracked object mode {mode} under {prefix}: {path}")
        data = blob_bytes(expected_blob)
        members.append((rel, mode, expected_blob, hashlib.sha256(data).hexdigest()))

    digest = hashlib.sha256()
    for rel, mode, blob, sha256 in members:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(mode.encode("ascii"))
        digest.update(b"\0")
        digest.update(blob.encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\0")

    return {
        "prefix": prefix,
        "fingerprint_semantics": "relative-path+git-mode+blob+sha256-v2",
        "tracked_files": len(members),
        "fingerprint_sha256": digest.hexdigest(),
        "members": [
            {
                "relative_path": rel,
                "git_mode": mode,
                "git_blob": blob,
                "sha256": sha,
            }
            for rel, mode, blob, sha in members
        ],
    }


def audit_artifact_v2(entries_by_path: dict[str, str]) -> dict[str, object]:
    rel = engine.AUDIT_SCRIPT_PATH
    mode, blob, data = tracked_object(rel)
    if entries_by_path.get(rel) != blob:
        raise RuntimeError("audit artifact blob disagrees with tracked entry map")
    return {
        "path": rel,
        "git_mode": mode,
        "git_blob": blob,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "byte_source": IO_SEMANTICS,
    }


def profile_sha256_v2() -> str:
    base = _original_profile_sha256()
    payload = json.dumps(
        {
            "base_profile_sha256": base,
            "io_semantics": IO_SEMANTICS,
            "regular_modes": sorted(REGULAR_MODES),
            "symlink_mode": SYMLINK_MODE,
            "symlink_lexical_policy": "no-scan-pointer-bytes-only",
            "subtree_fingerprint": "relative-path+git-mode+blob+sha256-v2",
            "matched_inventory_addition": "git_mode",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Configure the frozen engine as an explicitly distinct v2 profile.
engine.PROFILE_ID = PROFILE_ID
engine.REPORT_VERSION = 2
engine.PROFILE_EXCLUDED_PATHS = set(engine.PROFILE_EXCLUDED_PATHS) | {
    WRAPPER_REL,
    DOC_REL,
}
engine.read_text = read_text_v2
engine.build_inventory = build_inventory_v2
engine.subtree_fingerprint = subtree_fingerprint_v2
engine.audit_artifact = audit_artifact_v2
engine.profile_sha256 = profile_sha256_v2


if __name__ == "__main__":
    raise SystemExit(engine.main())
