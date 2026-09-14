#!/usr/bin/env python3
"""WCARE-47 standalone Cargo.lock admission verifier.

MeasurementOnly. Admitting the lock removes only the dependency-resolution
blocker; it never promotes WCARE-42 execution or downstream authority claims.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

PROTOCOL = "wcare47-standalone-lock-admission-v1"
W46 = "8a3cacb449b923ceee32b6b08e2c811ce532c676"
MANIFEST = "tools/wcare42_builder_attestation_verifier/Cargo.toml"
LOCK = "tools/wcare42_builder_attestation_verifier/Cargo.lock"
SOURCE_BLOBS = {
    MANIFEST: "5410040e5616241dd4ba581af8f297675d083830",
    "tools/wcare42_builder_attestation_verifier/src/main.rs": "1c300a455f054d118e55556aac81b623824629bc",
    "tools/wcare42_builder_attestation_verifier/tests/golden.rs": "666242f74fb302f9be2b62fa4b3050f3ee9ffefd",
}
TOOLCHAIN_BLOB = ("rust-toolchain.toml", "4f0430eac96d545bcfaa0df23ce475faf4aee96a")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def run(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def blob_at(root: Path, revision: str, path: str) -> str | None:
    result = run(root, "git", "rev-parse", f"{revision}:{path}")
    return result.stdout.strip() if result.returncode == 0 else None


def clean(root: Path) -> bool:
    status = run(root, "git", "status", "--porcelain", "--untracked-files=no")
    return status.returncode == 0 and not status.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def infra_failure(text: str) -> bool:
    lowered = text.lower()
    needles = [
        "failed to download",
        "could not resolve host",
        "dns",
        "timed out",
        "timeout",
        "connection reset",
        "connection refused",
        "network failure",
        "proxy error",
        "no space left on device",
        "permission denied",
    ]
    return any(needle in lowered for needle in needles)


def base(head: str) -> dict:
    return {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "classification": "INVALID_PROTOCOL",
        "detail": "uninitialized",
        "head": head,
        "wcare46_head": W46,
        "exact_source_subject_bound": False,
        "rust_toolchain_subject_bound": False,
        "candidate_lock_present": False,
        "lock_sha256": None,
        "lock_git_blob": None,
        "lock_format": None,
        "package_count": None,
        "registry_checksum_policy_satisfied": False,
        "rustc_1_96_0_verified": False,
        "cargo_identity": None,
        "metadata_locked_passed": False,
        "tests_locked_passed": False,
        "source_postflight_unchanged": False,
        "lock_generation_provenance_established": False,
        "lock_admitted": False,
        "wcare42_executable_qualification_established": False,
        "builder_authentication_established": False,
        "preregistration_temporal_precedence_established": False,
        "runtime_authority_granted": False,
    }


def emit(out: dict, code: int) -> int:
    print(json.dumps(out, sort_keys=True, separators=(",", ":")))
    return code


def reject(out: dict, detail: str) -> int:
    out["classification"] = "LOCK_REJECTED"
    out["detail"] = detail
    return emit(out, 1)


def indeterminate(out: dict, detail: str) -> int:
    out["classification"] = "INFRASTRUCTURE_INDETERMINATE"
    out["detail"] = detail
    return emit(out, 3)


def main() -> int:
    root_result = subprocess.run(["git", "rev-parse", "--show-toplevel"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if root_result.returncode != 0:
        return emit(base("0" * 40) | {"detail": "not_in_git_worktree"}, 4)
    root = Path(root_result.stdout.strip())
    head_result = run(root, "git", "rev-parse", "HEAD")
    head = head_result.stdout.strip() if head_result.returncode == 0 else "0" * 40
    out = base(head)

    ancestry = run(root, "git", "merge-base", "--is-ancestor", W46, "HEAD")
    if ancestry.returncode != 0:
        out["detail"] = "wcare46_subject_not_ancestor"
        return emit(out, 4)

    if any(blob_at(root, "HEAD", path) != expected for path, expected in SOURCE_BLOBS.items()):
        out["detail"] = "wcare42_source_subject_drift"
        return emit(out, 4)
    out["exact_source_subject_bound"] = True

    toolchain_path, toolchain_sha = TOOLCHAIN_BLOB
    if blob_at(root, "HEAD", toolchain_path) != toolchain_sha:
        out["detail"] = "rust_toolchain_subject_drift"
        return emit(out, 4)
    out["rust_toolchain_subject_bound"] = True

    lock_path = root / LOCK
    lock_blob = blob_at(root, "HEAD", LOCK)
    if not lock_path.is_file() or lock_blob is None:
        return indeterminate(out, "candidate_lock_missing")
    out["candidate_lock_present"] = True

    if not clean(root):
        out["detail"] = "working_tree_not_clean"
        return emit(out, 4)

    working_lock_blob = run(root, "git", "hash-object", LOCK)
    if working_lock_blob.returncode != 0 or working_lock_blob.stdout.strip() != lock_blob:
        out["detail"] = "working_lock_differs_from_committed_lock"
        return emit(out, 4)

    out["lock_git_blob"] = lock_blob
    out["lock_sha256"] = sha256_file(lock_path)

    try:
        lock_data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return reject(out, f"lock_parse_failed:{type(exc).__name__}")

    out["lock_format"] = lock_data.get("version")
    packages = lock_data.get("package")
    if out["lock_format"] != 4 or not isinstance(packages, list) or not packages:
        return reject(out, "unsupported_lock_format_or_empty_package_census")
    out["package_count"] = len(packages)

    root_seen = False
    for package in packages:
        if not isinstance(package, dict):
            return reject(out, "non_object_lock_package")
        name = package.get("name")
        source = package.get("source")
        checksum = package.get("checksum")
        if name == "wcare42-builder-attestation-verifier" and source is None:
            root_seen = True
            continue
        if not isinstance(source, str) or not source.startswith("registry+"):
            return reject(out, f"non_registry_dependency:{name}")
        if not isinstance(checksum, str) or HEX64.fullmatch(checksum) is None:
            return reject(out, f"missing_or_invalid_registry_checksum:{name}")
    if not root_seen:
        return reject(out, "standalone_root_package_missing_from_lock")
    out["registry_checksum_policy_satisfied"] = True

    try:
        rustc = run(root, "rustc", "--version")
        cargo = run(root, "cargo", "--version")
    except FileNotFoundError:
        return indeterminate(out, "rust_toolchain_command_missing")
    if rustc.returncode != 0 or cargo.returncode != 0:
        return indeterminate(out, "rust_toolchain_identity_unavailable")
    if not rustc.stdout.strip().startswith("rustc 1.96.0"):
        return reject(out, f"unexpected_rustc_identity:{rustc.stdout.strip()}")
    out["rustc_1_96_0_verified"] = True
    out["cargo_identity"] = cargo.stdout.strip()

    metadata = run(root, "cargo", "metadata", "--manifest-path", MANIFEST, "--locked", "--format-version", "1")
    if metadata.returncode != 0:
        text = metadata.stdout + "\n" + metadata.stderr
        if infra_failure(text):
            return indeterminate(out, "cargo_metadata_infrastructure_failure")
        return reject(out, "cargo_metadata_locked_failed")
    out["metadata_locked_passed"] = True

    tests = run(root, "cargo", "test", "--manifest-path", MANIFEST, "--locked")
    if tests.returncode != 0:
        text = tests.stdout + "\n" + tests.stderr
        if infra_failure(text):
            return indeterminate(out, "cargo_test_infrastructure_failure")
        return reject(out, "cargo_test_locked_failed")
    out["tests_locked_passed"] = True

    source_unchanged = all(blob_at(root, "HEAD", path) == expected for path, expected in SOURCE_BLOBS.items())
    source_unchanged = source_unchanged and blob_at(root, "HEAD", toolchain_path) == toolchain_sha and clean(root)
    out["source_postflight_unchanged"] = source_unchanged
    if not source_unchanged:
        return reject(out, "source_or_tree_drift_after_qualification")

    out["classification"] = "LOCK_ADMITTED"
    out["detail"] = "exact_lock_admitted_for_dependency_resolution_only"
    out["lock_admitted"] = True
    return emit(out, 0)


if __name__ == "__main__":
    raise SystemExit(main())