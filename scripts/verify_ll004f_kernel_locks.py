#!/usr/bin/env python3
"""Verify LL-004F kernel bytes against checked-in authoritative locks.

This is an offline evidence gate. It does not download kernels and it does not
run SPICE. The snapshot generator separately records SHA-256 over every kernel
actually consumed; this verifier answers the upstream identity question first:
are the locked, authority-published artifacts present under the expected names?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


class LockError(RuntimeError):
    pass


def digest_file(path: Path, algorithm: str) -> str:
    try:
        h = hashlib.new(algorithm.lower())
    except ValueError as exc:
        raise LockError(f"unsupported digest algorithm: {algorithm}") from exc
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LockError(f"expected JSON object: {path}")
    return value


def load_locks(path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    root = load_json(path)
    entries = root.get("locks")
    if not isinstance(entries, list) or not entries:
        raise LockError("lock file must contain a non-empty locks array")
    required_roles = set(root.get("policy", {}).get("required_roles", []))
    locks: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise LockError("lock entries must be objects")
        filename = str(entry.get("filename", "")).strip()
        role = str(entry.get("role", "")).strip()
        algorithm = str(entry.get("algorithm", "")).strip().lower()
        digest = str(entry.get("digest", "")).strip().lower()
        authority = str(entry.get("source_authority", "")).strip()
        source_url = str(entry.get("source_url", "")).strip()
        if not all((filename, role, algorithm, digest, authority, source_url)):
            raise LockError("each lock requires filename, role, algorithm, digest, source_authority, source_url")
        if filename in locks:
            raise LockError(f"duplicate lock for {filename}")
        locks[filename] = entry
    return locks, required_roles


def verify_config(config_path: Path, kernel_dir: Path, locks_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    kernels = config.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise LockError("config kernels must be a non-empty list")
    locks, required_roles = load_locks(locks_path)

    results: list[dict[str, Any]] = []
    roles_seen: set[str] = set()
    required_seen: set[str] = set()

    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise LockError("kernel config entries must be objects")
        filename = str(kernel.get("filename", "")).strip()
        role = str(kernel.get("role", "")).strip()
        if not filename or not role:
            raise LockError("kernel config entries require filename and role")
        roles_seen.add(role)
        path = kernel_dir / filename
        if not path.is_file():
            raise LockError(f"missing kernel: {path}")

        lock = locks.get(filename)
        if role in required_roles and lock is None:
            raise LockError(f"required upstream lock missing for {filename} ({role})")
        if lock is None:
            results.append({
                "filename": filename,
                "role": role,
                "status": "unlocked_nonrequired",
                "bytes": path.stat().st_size,
                "sha256": digest_file(path, "sha256"),
            })
            continue

        if str(lock.get("role", "")).strip() != role:
            raise LockError(f"role mismatch for {filename}: config={role} lock={lock.get('role')}")
        algorithm = str(lock["algorithm"]).lower()
        actual = digest_file(path, algorithm)
        expected = str(lock["digest"]).lower()
        if actual != expected:
            raise LockError(
                f"upstream checksum mismatch for {filename}: {algorithm} expected={expected} actual={actual}"
            )
        required_seen.add(role)
        results.append({
            "filename": filename,
            "role": role,
            "status": "locked_match",
            "algorithm": algorithm,
            "digest": actual,
            "source_authority": lock["source_authority"],
            "source_url": lock["source_url"],
            "bytes": path.stat().st_size,
            "sha256": digest_file(path, "sha256"),
        })

    absent_required = sorted((required_roles & roles_seen) - required_seen)
    if absent_required:
        raise LockError(f"required roles not satisfied by locks: {', '.join(absent_required)}")

    return {
        "schema_version": "ll004f.kernel-verification-receipt.v1",
        "config": config_path.name,
        "lineage_id": config.get("lineage_id"),
        "lock_file": locks_path.name,
        "status": "pass",
        "kernels": results,
    }


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        kernel_dir = root / "kernels"
        kernel_dir.mkdir()
        ephemeris = kernel_dir / "demo.bsp"
        orientation = kernel_dir / "demo.bpc"
        ephemeris.write_bytes(b"ephemeris-bytes")
        orientation.write_bytes(b"orientation-bytes")

        md5 = digest_file(ephemeris, "md5")
        config = root / "config.json"
        config.write_text(json.dumps({
            "lineage_id": "synthetic",
            "kernels": [
                {"filename": "demo.bsp", "role": "planetary_ephemeris"},
                {"filename": "demo.bpc", "role": "orientation"},
            ],
        }), encoding="utf-8")
        locks = root / "locks.json"
        locks.write_text(json.dumps({
            "locks": [{
                "filename": "demo.bsp",
                "role": "planetary_ephemeris",
                "algorithm": "md5",
                "digest": md5,
                "source_authority": "synthetic",
                "source_url": "https://example.invalid/checksums",
            }],
            "policy": {"required_roles": ["planetary_ephemeris"]},
        }), encoding="utf-8")

        receipt = verify_config(config, kernel_dir, locks)
        assert receipt["status"] == "pass"
        assert receipt["kernels"][0]["status"] == "locked_match"
        assert receipt["kernels"][1]["status"] == "unlocked_nonrequired"

        ephemeris.write_bytes(b"tampered")
        try:
            verify_config(config, kernel_dir, locks)
        except LockError:
            pass
        else:
            raise AssertionError("tampered locked kernel must fail")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--locks", type=Path, default=Path("configs/lunar_transport/ll004f_kernel_locks.json"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-004F kernel-lock self-test: PASS")
        return 0
    if args.config is None or args.kernel_dir is None:
        raise LockError("--config and --kernel-dir are required unless --self-test is used")
    receipt = verify_config(args.config, args.kernel_dir, args.locks)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LockError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
