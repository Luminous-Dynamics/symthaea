#!/usr/bin/env python3
"""Canonical SPINE-000B-M1 subject-manifest v1 encoder.

Measurement-only. JSON is transport; canonical identity is the domain-separated
binary encoding defined here and in the frozen contract.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import PurePosixPath

DOMAIN = b"symthaea.spine.000b.subject-manifest.v1\0"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _u8(v: int) -> bytes:
    if not isinstance(v, int) or not 0 <= v <= 0xFF:
        raise ValueError("u8 out of range")
    return struct.pack("<B", v)


def _u16(v: int) -> bytes:
    if not isinstance(v, int) or not 0 <= v <= 0xFFFF:
        raise ValueError("u16 out of range")
    return struct.pack("<H", v)


def _u64(v: int) -> bytes:
    if not isinstance(v, int) or not 0 <= v <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("u64 out of range")
    return struct.pack("<Q", v)


def _text(value: str, *, ascii_only: bool = False) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("string must be non-empty")
    raw = value.encode("ascii" if ascii_only else "utf-8")
    return _u16(len(raw)) + raw


def _sha(hex_value: str) -> bytes:
    if not isinstance(hex_value, str) or HEX64.fullmatch(hex_value) is None:
        raise ValueError("sha256 must be lowercase 64-hex")
    return bytes.fromhex(hex_value)


def _git_hex(value: str) -> bytes:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise ValueError("git id must be lowercase 40-hex")
    return _text(value, ascii_only=True)


def canonical_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("path must be non-empty")
    if path.startswith("/") or "\\" in path or "//" in path:
        raise ValueError("path must be canonical repo-relative POSIX")
    parts = path.split("/")
    if any(p in {"", ".", ".."} for p in parts):
        raise ValueError("path has noncanonical segment")
    normalized = str(PurePosixPath(path))
    if normalized != path:
        raise ValueError("path normalization drift")
    return path


def _sorted_unique_strings(values: list[str], label: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label} entries must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")
    return sorted(values)


def validate_manifest(m: dict) -> None:
    required = {
        "repository", "git_head", "git_tree", "clean_worktree_required",
        "runtime_profile_id", "target_triple", "rustc_version", "cargo_version",
        "default_features", "features", "bound_files", "protocol_ids",
        "workload_id", "workload_digest", "seeds", "cycle_start", "cycle_count",
        "stopping_rule_id", "capacities", "qualification_policy_digest",
    }
    if set(m) != required:
        raise ValueError(f"manifest fields drifted missing={sorted(required-set(m))} extra={sorted(set(m)-required)}")
    if m["repository"] != "Luminous-Dynamics/symthaea":
        raise ValueError("repository identity mismatch")
    _git_hex(m["git_head"])
    _git_hex(m["git_tree"])
    if m["clean_worktree_required"] is not True:
        raise ValueError("qualified v1 manifest requires clean_worktree_required=true")
    for key in ("runtime_profile_id", "target_triple", "rustc_version", "cargo_version", "workload_id", "stopping_rule_id"):
        _text(m[key])
    if not isinstance(m["default_features"], bool):
        raise ValueError("default_features must be bool")
    _sorted_unique_strings(m["features"], "features")
    _sorted_unique_strings(m["protocol_ids"], "protocol_ids")
    _sha(m["workload_digest"])
    _sha(m["qualification_policy_digest"])

    files = m["bound_files"]
    if not isinstance(files, list) or not files:
        raise ValueError("bound_files must be non-empty list")
    paths = []
    for entry in files:
        if set(entry) != {"path", "sha256"}:
            raise ValueError("bound file entry shape drifted")
        paths.append(canonical_path(entry["path"]))
        _sha(entry["sha256"])
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate bound file path")

    seeds = m["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be non-empty list")
    names = []
    for entry in seeds:
        if set(entry) != {"name", "value"}:
            raise ValueError("seed entry shape drifted")
        _text(entry["name"])
        _u64(entry["value"])
        names.append(entry["name"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate seed name")

    _u64(m["cycle_start"])
    if not isinstance(m["cycle_count"], int) or m["cycle_count"] <= 0:
        raise ValueError("cycle_count must be positive")
    _u64(m["cycle_count"])

    caps = m["capacities"]
    if set(caps) != {"manager", "application", "guard"}:
        raise ValueError("capacity fields drifted")
    for key in ("manager", "application", "guard"):
        if not isinstance(caps[key], int) or caps[key] <= 0:
            raise ValueError("observer capacities must be positive")
        _u16(caps[key])


def canonical_bytes(m: dict) -> bytes:
    validate_manifest(m)
    out = bytearray(DOMAIN)
    out += _text(m["repository"], ascii_only=True)
    out += _git_hex(m["git_head"])
    out += _git_hex(m["git_tree"])
    out += _u8(1)
    out += _text(m["runtime_profile_id"])
    out += _text(m["target_triple"], ascii_only=True)
    out += _text(m["rustc_version"])
    out += _text(m["cargo_version"])
    out += _u8(1 if m["default_features"] else 0)

    features = _sorted_unique_strings(m["features"], "features")
    out += _u16(len(features))
    for feature in features:
        out += _text(feature, ascii_only=True)

    files = sorted(m["bound_files"], key=lambda e: e["path"])
    out += _u16(len(files))
    for entry in files:
        out += _text(canonical_path(entry["path"]), ascii_only=True)
        out += _sha(entry["sha256"])

    protocols = _sorted_unique_strings(m["protocol_ids"], "protocol_ids")
    out += _u16(len(protocols))
    for protocol in protocols:
        out += _text(protocol, ascii_only=True)

    out += _text(m["workload_id"])
    out += _sha(m["workload_digest"])

    seeds = sorted(m["seeds"], key=lambda e: e["name"])
    out += _u16(len(seeds))
    for entry in seeds:
        out += _text(entry["name"], ascii_only=True)
        out += _u64(entry["value"])

    out += _u64(m["cycle_start"])
    out += _u64(m["cycle_count"])
    out += _text(m["stopping_rule_id"])
    out += _u16(m["capacities"]["manager"])
    out += _u16(m["capacities"]["application"])
    out += _u16(m["capacities"]["guard"])
    out += _sha(m["qualification_policy_digest"])
    return bytes(out)


def digest_hex(m: dict) -> str:
    return hashlib.sha256(canonical_bytes(m)).hexdigest()


def sample_manifest() -> dict:
    return {
        "repository": "Luminous-Dynamics/symthaea",
        "git_head": "11" * 20,
        "git_tree": "22" * 20,
        "clean_worktree_required": True,
        "runtime_profile_id": "spine-runtime-v1",
        "target_triple": "x86_64-unknown-linux-gnu",
        "rustc_version": "rustc 1.96.0 (sample)",
        "cargo_version": "cargo 1.96.0 (sample)",
        "default_features": False,
        "features": ["vision-manifold", "swarm"],
        "bound_files": [
            {"path": "Cargo.lock", "sha256": "33" * 32},
            {"path": "rust-toolchain.toml", "sha256": "44" * 32},
            {"path": "src/cognitive_loop/subsystem_trait.rs", "sha256": "55" * 32},
        ],
        "protocol_ids": ["C2", "G1", "N1", "N2", "P1R", "R2"],
        "workload_id": "synthetic-spine-workload-v1",
        "workload_digest": "66" * 32,
        "seeds": [{"name": "genesis", "value": 42}, {"name": "workload", "value": 7}],
        "cycle_start": 100,
        "cycle_count": 256,
        "stopping_rule_id": "fixed-cycle-count-v1",
        "capacities": {"manager": 64, "application": 32, "guard": 32},
        "qualification_policy_digest": "77" * 32,
    }


def sample_vectors() -> dict:
    base = sample_manifest()
    reordered = json.loads(json.dumps(base))
    reordered["features"] = list(reversed(reordered["features"]))
    reordered["bound_files"] = list(reversed(reordered["bound_files"]))
    reordered["protocol_ids"] = list(reversed(reordered["protocol_ids"]))
    reordered["seeds"] = list(reversed(reordered["seeds"]))
    mutated_capacity = json.loads(json.dumps(base))
    mutated_capacity["capacities"]["application"] = 33
    mutated_hash = json.loads(json.dumps(base))
    mutated_hash["bound_files"][0]["sha256"] = "88" * 32
    cases = {
        "base": base,
        "reordered": reordered,
        "mutated_capacity": mutated_capacity,
        "mutated_hash": mutated_hash,
    }
    return {
        name: {"bytes_hex": canonical_bytes(m).hex(), "sha256": digest_hex(m)}
        for name, m in cases.items()
    }


if __name__ == "__main__":
    print(json.dumps(sample_vectors(), sort_keys=True, indent=2))
