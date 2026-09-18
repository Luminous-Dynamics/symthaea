#!/usr/bin/env python3
"""Safely unwrap and verify an ARC3 candidate receipt artifact ZIP.

The ZIP and contained receipt are untrusted data. This script rejects archive
shape surprises before invoking the strict receipt verifier. It executes no
candidate code and performs no network access or repository mutation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import stat
import sys
import tempfile
import zipfile

import arc3_candidate_receipt_verify as receipt_verify

EXPECTED_MEMBER = "arc3-protocol-slim-v3-candidate-receipt.txt"
MAX_ARTIFACT_BYTES = 1 * 1024 * 1024
MAX_RECEIPT_BYTES = 16 * 1024


class ArtifactError(ValueError):
    pass


def fail(message: str) -> None:
    raise ArtifactError(message)


def normalize_sha256(value: str, label: str) -> str:
    if value.startswith("sha256:"):
        value = value.removeprefix("sha256:")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        fail(f"{label} must be lowercase 64-hex SHA-256")
    return value


def read_archive(path: Path) -> bytes:
    if path.is_symlink():
        fail("artifact path must not be a symlink")
    if not path.is_file():
        fail("artifact path is not a regular file")
    size = path.stat().st_size
    if size <= 0:
        fail("artifact is empty")
    if size > MAX_ARTIFACT_BYTES:
        fail(f"artifact exceeds {MAX_ARTIFACT_BYTES} byte cap")
    return path.read_bytes()


def extract_exact_receipt(archive_path: Path) -> tuple[bytes, str]:
    raw_archive = read_archive(archive_path)
    artifact_sha256 = hashlib.sha256(raw_archive).hexdigest()

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            infos = zf.infolist()
            if len(infos) != 1:
                fail(f"artifact must contain exactly one member; observed {len(infos)}")

            info = infos[0]
            if info.filename != EXPECTED_MEMBER:
                fail(
                    f"unexpected artifact member: expected {EXPECTED_MEMBER!r}, "
                    f"observed {info.filename!r}"
                )
            if info.is_dir():
                fail("artifact member must be a regular file, not a directory")
            if info.flag_bits & 0x1:
                fail("encrypted ZIP members are forbidden")

            unix_type = (info.external_attr >> 16) & 0o170000
            if unix_type == stat.S_IFLNK:
                fail("symlink ZIP members are forbidden")

            if info.file_size <= 0:
                fail("receipt member is empty")
            if info.file_size > MAX_RECEIPT_BYTES:
                fail(f"receipt exceeds {MAX_RECEIPT_BYTES} byte cap")
            if info.compress_size > MAX_ARTIFACT_BYTES:
                fail("compressed receipt member exceeds artifact cap")

            receipt = zf.read(info)
    except zipfile.BadZipFile as exc:
        fail(f"artifact is not a valid ZIP: {exc}")

    if len(receipt) > MAX_RECEIPT_BYTES:
        fail(f"decompressed receipt exceeds {MAX_RECEIPT_BYTES} byte cap")

    return receipt, artifact_sha256


def build_receipt_namespace(args: argparse.Namespace, path: Path, digest: str) -> argparse.Namespace:
    return argparse.Namespace(
        receipt=str(path),
        expected_receipt_sha256=digest,
        expected_repository=args.expected_repository,
        expected_run_id=args.expected_run_id,
        expected_run_attempt=args.expected_run_attempt,
        expected_helper_sha=args.expected_helper_sha,
        expected_helper_tree=args.expected_helper_tree,
        expected_workflow_blob=args.expected_workflow_blob,
        expected_subject_sha=args.expected_subject_sha,
        expected_subject_tree=args.expected_subject_tree,
        expected_subject_binding_sha256=args.expected_subject_binding_sha256,
        expected_cargo_lock_sha256=args.expected_cargo_lock_sha256,
        expected_oracle_sha256=args.expected_oracle_sha256,
        expected_fixture_sha256=args.expected_fixture_sha256,
        expected_vector_sha256=args.expected_vector_sha256,
        expected_runner_class=args.expected_runner_class,
    )


def verify(args: argparse.Namespace) -> tuple[str, str, int]:
    archive_path = Path(args.artifact)
    receipt_bytes, artifact_sha256 = extract_exact_receipt(archive_path)

    expected_artifact_sha256 = normalize_sha256(
        args.expected_artifact_sha256, "expected artifact digest"
    )
    if artifact_sha256 != expected_artifact_sha256:
        fail(
            "artifact SHA-256 mismatch: "
            f"expected {expected_artifact_sha256}, observed {artifact_sha256}"
        )

    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="arc3-receipt-witness-") as td:
        receipt_path = Path(td) / EXPECTED_MEMBER
        receipt_path.write_bytes(receipt_bytes)
        receipt_verify.verify(build_receipt_namespace(args, receipt_path, receipt_sha256))

    return artifact_sha256, receipt_sha256, len(receipt_bytes)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True)
    p.add_argument("--expected-artifact-sha256", required=True)
    p.add_argument("--expected-repository", required=True)
    p.add_argument("--expected-run-id", required=True)
    p.add_argument("--expected-run-attempt", required=True)
    p.add_argument("--expected-helper-sha", required=True)
    p.add_argument("--expected-helper-tree", required=True)
    p.add_argument("--expected-workflow-blob", required=True)
    p.add_argument("--expected-subject-sha", required=True)
    p.add_argument("--expected-subject-tree", required=True)
    p.add_argument("--expected-subject-binding-sha256", required=True)
    p.add_argument("--expected-cargo-lock-sha256", required=True)
    p.add_argument("--expected-oracle-sha256", required=True)
    p.add_argument("--expected-fixture-sha256", required=True)
    p.add_argument("--expected-vector-sha256", required=True)
    p.add_argument("--expected-runner-class", default="ubuntu-slim")
    return p


def main() -> int:
    try:
        artifact_sha256, receipt_sha256, receipt_bytes = verify(parser().parse_args())
    except (OSError, ArtifactError, receipt_verify.ReceiptError) as exc:
        print(f"ARC3 candidate artifact verification FAILED: {exc}", file=sys.stderr)
        return 1

    print("arc3_candidate_artifact_verification=PASS")
    print(f"artifact_sha256={artifact_sha256}")
    print(f"receipt_sha256={receipt_sha256}")
    print(f"receipt_bytes={receipt_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
