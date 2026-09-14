#!/usr/bin/env python3
"""Fail-closed oracle for one additive local-package Cargo.lock transition.

The oracle does not generate Cargo.lock and does not prove Cargo provenance,
compilation, qualification PASS, registry authenticity, or execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any


class LockTransitionError(ValueError):
    pass


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _packages(lock: dict[str, Any], where: str) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise LockTransitionError(f"{where}.package: expected array")
    if not all(isinstance(package, dict) for package in packages):
        raise LockTransitionError(f"{where}.package: every record must be an object")
    return packages


def verify_additive_local_package(
    base: dict[str, Any],
    candidate: dict[str, Any],
    *,
    expected_name: str,
    expected_version: str,
    expected_dependencies: list[str],
) -> dict[str, Any]:
    if not expected_name or not expected_version:
        raise LockTransitionError("expected package name/version must be non-empty")
    if expected_dependencies != sorted(set(expected_dependencies)):
        raise LockTransitionError("expected dependencies must be sorted unique")

    if {k: v for k, v in base.items() if k != "package"} != {
        k: v for k, v in candidate.items() if k != "package"
    }:
        raise LockTransitionError("non-package Cargo.lock metadata changed")

    base_packages = _packages(base, "base")
    candidate_packages = _packages(candidate, "candidate")

    if any(
        package.get("name") == expected_name
        and package.get("version") == expected_version
        and "source" not in package
        for package in base_packages
    ):
        raise LockTransitionError("base already contains expected local package")

    base_counter = Counter(map(_canon, base_packages))
    candidate_counter = Counter(map(_canon, candidate_packages))

    if base_counter - candidate_counter:
        raise LockTransitionError("candidate removed or changed an existing package record")

    added = candidate_counter - base_counter
    if sum(added.values()) != 1 or len(added) != 1:
        raise LockTransitionError("candidate must add exactly one package record")

    record = json.loads(next(iter(added)))
    if set(record) - {"name", "version", "dependencies"}:
        raise LockTransitionError("new local package must not contain source/checksum or unknown keys")
    if record.get("name") != expected_name:
        raise LockTransitionError("new local package name mismatch")
    if record.get("version") != expected_version:
        raise LockTransitionError("new local package version mismatch")

    dependencies = record.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise LockTransitionError("new local package dependencies must be an array of strings")
    if dependencies != sorted(set(dependencies)):
        raise LockTransitionError("new local package dependencies must be sorted unique")
    if dependencies != expected_dependencies:
        raise LockTransitionError("new local package dependency set mismatch")

    return {
        "disposition": "ExactAdditiveLocalPackageTransition",
        "package": {
            "name": expected_name,
            "version": expected_version,
            "dependencies": expected_dependencies,
        },
        "base_package_records": len(base_packages),
        "candidate_package_records": len(candidate_packages),
        "non_claims": [
            "does not prove Cargo generated the candidate lockfile",
            "does not prove the new package compiles",
            "does not prove registry/source authenticity",
            "does not establish qualification PASS",
        ],
    }


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        lock = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise LockTransitionError(f"{path}: invalid UTF-8 Cargo.lock TOML") from error
    return lock, "sha256:" + hashlib.sha256(raw).hexdigest()


def verify_lock_files(
    base_path: Path,
    candidate_path: Path,
    *,
    expected_name: str,
    expected_version: str,
    expected_dependencies: list[str],
) -> dict[str, Any]:
    base, base_sha256 = _load(base_path)
    candidate, candidate_sha256 = _load(candidate_path)
    result = verify_additive_local_package(
        base,
        candidate,
        expected_name=expected_name,
        expected_version=expected_version,
        expected_dependencies=expected_dependencies,
    )
    return {
        "schema": "symthaea.cargo-lock-additive-local-package-transition.v1",
        "base_lock_sha256": base_sha256,
        "candidate_lock_sha256": candidate_sha256,
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("candidate")
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dependency", action="append", default=[])
    args = parser.parse_args()
    dependencies = sorted(args.dependency)
    if len(dependencies) != len(set(dependencies)):
        raise LockTransitionError("--dependency values must be unique")
    print(
        json.dumps(
            verify_lock_files(
                Path(args.base),
                Path(args.candidate),
                expected_name=args.name,
                expected_version=args.version,
                expected_dependencies=dependencies,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
