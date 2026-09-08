#!/usr/bin/env python3
"""Pure no-Nix verifier for membership of a path inside a canonical Nix Archive (NAR)."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

SCHEMA = "symthaea-workbench-root-nar-membership-v1"
MAGIC = b"nix-archive-1"
MAX_DEPTH = 1024
MAX_STRUCTURAL_STRING = 16 * 1024 * 1024
CHUNK = 1024 * 1024


class NarError(ValueError):
    pass


@dataclass(frozen=True)
class TargetObservation:
    node_type: str
    executable: bool | None
    content_length: int | None
    content_sha256: str | None
    symlink_target_hex: str | None


def canonical_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise NarError(f"{label}: canonical sha256:<64 lowercase hex> required")
    if any(ch not in "0123456789abcdef" for ch in value[7:]):
        raise NarError(f"{label}: canonical sha256:<64 lowercase hex> required")
    return value


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def canonical_target(value: str) -> tuple[bytes, ...]:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise NarError("target path: non-empty relative POSIX path required")
    p = PurePosixPath(value)
    if value != "/".join(p.parts) or any(part in {"", ".", ".."} for part in p.parts):
        raise NarError("target path: exact canonical spelling required")
    try:
        encoded = tuple(part.encode("utf-8") for part in p.parts)
    except UnicodeEncodeError as exc:
        raise NarError("target path: UTF-8 required") from exc
    if any(b"/" in part or b"\x00" in part for part in encoded):
        raise NarError("target path: invalid component")
    return encoded


class Reader:
    def __init__(self, handle: BinaryIO):
        self.handle = handle

    def read_exact(self, count: int, label: str) -> bytes:
        chunks: list[bytes] = []
        remaining = count
        while remaining:
            chunk = self.handle.read(min(CHUNK, remaining))
            if not chunk:
                raise NarError(f"{label}: unexpected EOF")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read_u64(self, label: str) -> int:
        return struct.unpack("<Q", self.read_exact(8, label))[0]

    def padding(self, length: int, label: str) -> None:
        count = (-length) % 8
        if count and self.read_exact(count, f"{label} padding") != b"\x00" * count:
            raise NarError(f"{label}: non-zero NAR padding")

    def structural(self, label: str) -> bytes:
        length = self.read_u64(f"{label} length")
        if length > MAX_STRUCTURAL_STRING:
            raise NarError(f"{label}: structural string exceeds v1 bound")
        value = self.read_exact(length, label)
        self.padding(length, label)
        return value

    def expect(self, expected: bytes, label: str) -> None:
        actual = self.structural(label)
        if actual != expected:
            raise NarError(f"{label}: expected {expected!r}, got {actual!r}")

    def contents(self, length: int, label: str, *, hash_bytes: bool) -> str | None:
        h = hashlib.sha256() if hash_bytes else None
        remaining = length
        while remaining:
            chunk = self.handle.read(min(CHUNK, remaining))
            if not chunk:
                raise NarError(f"{label}: unexpected EOF")
            if h is not None:
                h.update(chunk)
            remaining -= len(chunk)
        self.padding(length, label)
        return None if h is None else "sha256:" + h.hexdigest()


class Parser:
    def __init__(self, handle: BinaryIO, target: tuple[bytes, ...]):
        self.r = Reader(handle)
        self.target = target
        self.observation: TargetObservation | None = None

    def parse(self) -> TargetObservation:
        self.r.expect(MAGIC, "NAR magic")
        self.node((), 0)
        if self.r.handle.read(1):
            raise NarError("NAR: trailing bytes forbidden")
        if self.observation is None:
            raise NarError("target path: not present in NAR")
        return self.observation

    def node(self, path: tuple[bytes, ...], depth: int) -> None:
        if depth > MAX_DEPTH:
            raise NarError("NAR: maximum directory depth exceeded")
        self.r.expect(b"(", "node open")
        self.r.expect(b"type", "node type key")
        kind = self.r.structural("node type")
        if kind == b"regular":
            self.regular(path)
            self.r.expect(b")", "node close")
        elif kind == b"symlink":
            self.symlink(path)
            self.r.expect(b")", "node close")
        elif kind == b"directory":
            self.directory(path, depth)
        else:
            raise NarError(f"node type: unsupported value {kind!r}")

    def regular(self, path: tuple[bytes, ...]) -> None:
        field = self.r.structural("regular field")
        executable = False
        if field == b"executable":
            self.r.expect(b"", "regular executable marker")
            executable = True
            field = self.r.structural("regular contents key")
        if field != b"contents":
            raise NarError("regular node: contents field required")
        length = self.r.read_u64("regular contents length")
        sha = self.r.contents(length, "regular contents", hash_bytes=path == self.target)
        if path == self.target:
            self.record(TargetObservation("regular", executable, length, sha, None))

    def directory(self, path: tuple[bytes, ...], depth: int) -> None:
        previous: bytes | None = None
        while True:
            field = self.r.structural("directory field")
            if field == b")":
                return
            if field != b"entry":
                raise NarError("directory node: entry or close required")
            self.r.expect(b"(", "entry open")
            self.r.expect(b"name", "entry name key")
            name = self.r.structural("entry name")
            if not name or name in {b".", b".."} or b"/" in name or b"\x00" in name:
                raise NarError("directory entry: canonical name required")
            if previous is not None and name <= previous:
                raise NarError("directory entry: names must be strictly increasing")
            previous = name
            self.r.expect(b"node", "entry node key")
            self.node(path + (name,), depth + 1)
            self.r.expect(b")", "entry close")

    def symlink(self, path: tuple[bytes, ...]) -> None:
        self.r.expect(b"target", "symlink target key")
        target = self.r.structural("symlink target")
        if path == self.target:
            self.record(TargetObservation("symlink", None, None, None, target.hex()))

    def record(self, observation: TargetObservation) -> None:
        if self.observation is not None:
            raise NarError("target path: duplicate observation")
        self.observation = observation


def verify_membership(nar_path: Path, expected_nar_sha256: str, target_path: str) -> dict[str, Any]:
    expected = canonical_sha256(expected_nar_sha256, "expected NAR SHA-256")
    actual = digest_file(nar_path)
    if actual != expected:
        raise NarError("NAR: file SHA-256 does not match verified root NAR hash")
    target = canonical_target(target_path)
    with nar_path.open("rb") as handle:
        observation = Parser(handle, target).parse()
    return {
        "schema": SCHEMA,
        "status": "verified-nar-membership-only",
        "nar_sha256": actual,
        "target_path": target_path,
        "target": {
            "node_type": observation.node_type,
            "executable": observation.executable,
            "content_length": observation.content_length,
            "content_sha256": observation.content_sha256,
            "symlink_target_hex": observation.symlink_target_hex,
        },
        "authority": {
            "nar_bytes_match_verified_root": True,
            "target_membership_verified": True,
            "workbench_execution_qualified": False,
            "transform_executed": False,
            "fmq010_established": False,
            "neural_alignment_established": False,
            "consciousness_evidence": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nar", required=True, type=Path)
    parser.add_argument("--expected-nar-sha256", required=True)
    parser.add_argument("--target", default="bin/wb_command")
    args = parser.parse_args(argv)
    try:
        value = verify_membership(args.nar, args.expected_nar_sha256, args.target)
    except (NarError, OSError, OverflowError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
