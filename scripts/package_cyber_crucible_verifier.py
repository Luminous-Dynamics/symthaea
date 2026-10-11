#!/usr/bin/env python3
"""Build and verify a deterministic, hashable Cyber Crucible verifier bundle.

The bundle binds the exact verifier/validator sources, schemas, pinned dependencies,
public calibration fixture, qualification tests, docs, and workflow inputs in one
deterministic tar.gz. Build provenance for the resulting archive is a separate gate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile
import zlib
from typing import Any

BUNDLE_VERSION = "cyber-crucible-verifier-bundle.v1"
BUNDLE_ROOT = "cyber-crucible-verifier-v1"
MANIFEST_NAME = "BUNDLE-MANIFEST.json"
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_MEMBER_BYTES = 20 * 1024 * 1024
MAX_BUNDLE_BYTES_COMPRESSED = 128 * 1024 * 1024
MAX_BUNDLE_BYTES_UNCOMPRESSED = 128 * 1024 * 1024

PACKAGE_FILES = tuple(sorted((
    ".github/workflows/cyber-crucible-authenticated-receipts.yml",
    ".github/workflows/cyber-crucible-decision-receipt.yml",
    ".github/workflows/cyber-crucible-trusted-build.yml",
    ".github/workflows/cyber-crucible-verifier-release.yml",
    ".github/workflows/cyber-crucible-diagnostic-loop.yml",
    "scripts/package_cyber_crucible_verifier.py",
    "scripts/validate_cyber_crucible_diagnostic_loop.py",
    "scripts/verify_cyber_crucible_signed_receipt.py",
    "scripts/verify_cyber_crucible_decision_receipt.py",
    "scripts/verify_cyber_crucible_report_decision.py",
    "scripts/verify_cyber_crucible_verifier_provenance.py",
    "tests/test_cyber_crucible_signed_receipt.py",
    "tests/test_cyber_crucible_diagnostic_loop.py",
    "tests/test_cyber_crucible_decision_receipt.py",
    "tests/test_cyber_crucible_report_decision.py",
    "tests/test_cyber_crucible_verifier_provenance.py",
    "tests/test_cyber_crucible_verifier_bundle.py",
    "validation/CYBER_CRUCIBLE_DIAGNOSTIC_LOOP_V1.md",
    "validation/CYBER_CRUCIBLE_SIGNED_RECEIPT_V1.md",
    "validation/CYBER_CRUCIBLE_VERIFIER_DECISION_V1.md",
    "validation/cyber_crucible_diagnostic_loop_public_v1.json",
    "validation/cyber_crucible_diagnostic_loop_v1.schema.json",
    "validation/cyber_crucible_signed_receipt_v1.schema.json",
    "validation/cyber_crucible_verifier_decision_v1.schema.json",
    "validation/cyber_crucible_decision_trust_policy_v1.schema.json",
    "validation/cyber_crucible_verifier_provenance_policy_v1.schema.json",
    "validation/cyber_crucible_verifier_requirements_v1.txt",
)))


class BundleError(ValueError):
    """The source tree or bundle violates the deterministic bundle contract."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise BundleError(f"manifest is not canonicalizable JSON: {exc}") from exc


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON member in bundle manifest: {key}")
        result[key] = value
    return result


def _safe_relpath(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise BundleError(f"unsafe bundle member path: {value!r}")
    if "\\" in value or "\x00" in value:
        raise BundleError(f"unsafe bundle member path: {value!r}")
    return path


def _assert_no_symlink_components(path: Path) -> None:
    """Reject symlinks along an absolute path, including parents that exist."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise BundleError(f"cannot inspect path component: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise BundleError(f"path contains a symlink component: {current}")


def _read_source_tree(root: str | Path) -> tuple[Path, list[dict[str, Any]], dict[str, bytes]]:
    root_path = Path(root).absolute()
    if root_path.is_symlink() or not root_path.is_dir():
        raise BundleError("source root must be an existing directory and not a symlink")
    _assert_no_symlink_components(root_path)
    root_path = root_path.resolve(strict=True)
    manifest_files: list[dict[str, Any]] = []
    data_by_path: dict[str, bytes] = {}
    for name in PACKAGE_FILES:
        _safe_relpath(name)
        current = root_path
        for part in PurePosixPath(name).parts:
            current = current / part
            try:
                info = current.lstat()
            except OSError as exc:
                raise BundleError(f"required verifier bundle input is missing/unreadable: {name}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise BundleError(f"symlinked source input is rejected: {name}")
        if not stat.S_ISREG(info.st_mode):
            raise BundleError(f"source input must be a regular file: {name}")
        try:
            data = current.read_bytes()
        except OSError as exc:
            raise BundleError(f"cannot read source input: {name}") from exc
        if len(data) > MAX_MEMBER_BYTES:
            raise BundleError(f"source input exceeds per-file size limit: {name}")
        data_by_path[name] = data
        manifest_files.append({
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        })
    return root_path, manifest_files, data_by_path


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def _canonical_gzip_bytes(tar_bytes: bytes) -> bytes:
    """Compress canonical tar bytes using the exact reproducible release profile."""
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        filename="",
        mtime=0,
        compresslevel=9,
    ) as compressed:
        compressed.write(tar_bytes)
    return output.getvalue()


def _canonical_tar_bytes(entries: dict[str, bytes]) -> bytes:
    """Serialize the exact fixed metadata used by the deterministic bundle writer."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in sorted(entries.items()):
            tar.addfile(_tar_info(name, len(data)), io.BytesIO(data))
    return output.getvalue()


def _sha256_file(path: Path) -> str:
    """Hash a regular file without following a final-component symlink."""
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise BundleError(f"cannot safely open file for hashing: {path}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise BundleError(f"file to hash must be regular: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def build_bundle(root: str | Path, output_path: str | Path, source_commit: str) -> dict[str, Any]:
    """Build an atomic, reproducible tar.gz and verify it before returning."""
    if not isinstance(source_commit, str) or not SOURCE_COMMIT.fullmatch(source_commit):
        raise BundleError("source_commit must be a lowercase 40-character Git SHA")
    root_path, manifest_files, data_by_path = _read_source_tree(root)
    raw_output = Path(output_path)
    if ".." in raw_output.parts:
        raise BundleError("output path must not contain parent-directory traversal")
    output = raw_output.absolute()
    _assert_no_symlink_components(output.parent)
    if output.is_symlink():
        raise BundleError("output path must not be a symlink")
    if output.exists() and not output.is_file():
        raise BundleError("output path must be a regular file or not exist")
    try:
        relative_output = output.relative_to(root_path).as_posix()
    except ValueError:
        relative_output = None
    if relative_output in PACKAGE_FILES:
        raise BundleError("output path must not overwrite a packaged source input")
    total = sum(len(raw) for raw in data_by_path.values())
    if total > MAX_BUNDLE_BYTES_UNCOMPRESSED:
        raise BundleError("source bundle exceeds uncompressed size limit")

    manifest = {
        "schema_version": BUNDLE_VERSION,
        "source_commit": source_commit,
        "files": manifest_files,
    }
    manifest_bytes = _canonical_json(manifest)
    entries = {f"{BUNDLE_ROOT}/{name}": data for name, data in data_by_path.items()}
    entries[f"{BUNDLE_ROOT}/{MANIFEST_NAME}"] = manifest_bytes
    ordered_entries = sorted(entries.items())

    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(output.parent)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as raw_file:
            with gzip.GzipFile(fileobj=raw_file, mode="wb", filename="", mtime=0, compresslevel=9) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                    for name, data in ordered_entries:
                        tar.addfile(_tar_info(name, len(data)), io.BytesIO(data))
            raw_file.flush()
            os.fsync(raw_file.fileno())

        # Verify the temporary archive before replacing any existing output. A
        # failed self-check must never leave an invalid archive at the destination.
        digest = _sha256_file(temporary)
        verification = verify_bundle(temporary, expected_source_commit=source_commit)
        if verification["bundle_sha256"] != digest:
            raise BundleError("bundle digest changed during self-verification")
        os.replace(temporary, output)
        if _sha256_file(output) != digest:
            raise BundleError("bundle digest changed when committing the verified output")
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return {
        "schema_version": BUNDLE_VERSION,
        "bundle_path": str(output),
        "bundle_sha256": digest,
        "source_commit": source_commit,
        "file_count": len(manifest_files),
        "bundle_verified": True,
    }


def _read_bundle_snapshot(bundle: Path) -> tuple[bytes, str]:
    """Read one regular-file snapshot through O_NOFOLLOW, then hash those exact bytes."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(bundle, flags)
    except OSError as exc:
        raise BundleError("bundle must be a readable regular non-symlink file") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise BundleError("bundle must be a regular file")
        if info.st_size < 0 or info.st_size > MAX_BUNDLE_BYTES_COMPRESSED:
            raise BundleError("compressed bundle exceeds configured size limit")
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = -1
            raw = handle.read(MAX_BUNDLE_BYTES_COMPRESSED + 1)
        if len(raw) > MAX_BUNDLE_BYTES_COMPRESSED:
            raise BundleError("compressed bundle exceeds configured size limit")
        if len(raw) != info.st_size:
            raise BundleError("bundle changed size while taking a verification snapshot")
        expected_header = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
        if len(raw) < 18 or raw[:10] != expected_header:
            raise BundleError("gzip header is not the canonical mtime-zero, no-filename form")
        return raw, hashlib.sha256(raw).hexdigest()
    finally:
        if fd >= 0:
            os.close(fd)


def verify_bundle(bundle_path: str | Path, *, expected_source_commit: str | None = None) -> dict[str, Any]:
    """Verify tar structure, fixed file inventory, manifest digests, and reproducible metadata."""
    bundle = Path(bundle_path)
    bundle_bytes, bundle_digest = _read_bundle_snapshot(bundle)

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(bundle_bytes), mode="rb") as compressed:
            tar_bytes = compressed.read(MAX_BUNDLE_BYTES_UNCOMPRESSED + 1)
        if len(tar_bytes) > MAX_BUNDLE_BYTES_UNCOMPRESSED:
            raise BundleError("bundle decompressed tar exceeds configured size limit")
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
            expected_names = sorted(
                [f"{BUNDLE_ROOT}/{MANIFEST_NAME}"] +
                [f"{BUNDLE_ROOT}/{name}" for name in PACKAGE_FILES]
            )
            payloads: dict[str, bytes] = {}
            total_size = 0
            observed_count = 0
            # Iterate incrementally. getmembers() would materialize every attacker-
            # controlled tar header before the member-count and expansion limits run.
            for member in tar:
                if observed_count >= len(expected_names):
                    raise BundleError("bundle contains more members than the fixed inventory")
                if member.name != expected_names[observed_count]:
                    raise BundleError("bundle inventory is missing, duplicated, extra, or out of deterministic order")
                observed_count += 1
                _safe_relpath(member.name)
                if not member.isfile() or member.issym() or member.islnk():
                    raise BundleError(f"non-regular tar member rejected: {member.name}")
                if member.uid != 0 or member.gid != 0 or member.mtime != 0:
                    raise BundleError(f"non-deterministic tar metadata rejected: {member.name}")
                if member.uname not in ("", None) or member.gname not in ("", None):
                    raise BundleError(f"non-deterministic owner metadata rejected: {member.name}")
                if member.mode != 0o644:
                    raise BundleError(f"unexpected tar mode rejected: {member.name}")
                if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                    raise BundleError(f"tar member size limit exceeded: {member.name}")
                total_size += member.size
                if total_size > MAX_BUNDLE_BYTES_UNCOMPRESSED:
                    raise BundleError("bundle expanded size limit exceeded")
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise BundleError(f"cannot read tar member: {member.name}")
                data = extracted.read(MAX_MEMBER_BYTES + 1)
                if len(data) != member.size or len(data) > MAX_MEMBER_BYTES:
                    raise BundleError(f"member size mismatch or limit exceeded: {member.name}")
                payloads[member.name] = data
            if observed_count != len(expected_names):
                raise BundleError("bundle inventory is missing required members")
    except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
        raise BundleError(f"cannot read deterministic verifier bundle: {exc}") from exc

    manifest_path = f"{BUNDLE_ROOT}/{MANIFEST_NAME}"
    try:
        manifest = json.loads(
            payloads[manifest_path].decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(BundleError(f"invalid JSON constant: {value}")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError("bundle manifest must be strict UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "source_commit", "files"}:
        raise BundleError("bundle manifest has missing or unknown fields")
    if payloads[manifest_path] != _canonical_json(manifest):
        raise BundleError("bundle manifest JSON is not in canonical serialization")
    if manifest["schema_version"] != BUNDLE_VERSION:
        raise BundleError("unsupported bundle manifest schema version")
    source_commit = manifest["source_commit"]
    if not isinstance(source_commit, str) or not SOURCE_COMMIT.fullmatch(source_commit):
        raise BundleError("bundle manifest source_commit is malformed")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise BundleError("bundle source_commit does not match expected source commit")
    files = manifest["files"]
    if not isinstance(files, list) or len(files) != len(PACKAGE_FILES):
        raise BundleError("bundle manifest does not enumerate the exact required file inventory")

    expected_file_records = []
    for index, name in enumerate(PACKAGE_FILES):
        record = files[index]
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise BundleError(f"manifest file record {index} has invalid shape")
        if record["path"] != name:
            raise BundleError("manifest file inventory is not exact and sorted")
        if not isinstance(record["sha256"], str) or not SHA256.fullmatch(record["sha256"]):
            raise BundleError(f"manifest digest is malformed for {name}")
        size_bytes = record["size_bytes"]
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise BundleError(f"manifest size_bytes is malformed for {name}")
        raw = payloads[f"{BUNDLE_ROOT}/{name}"]
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record["sha256"] or len(raw) != size_bytes:
            raise BundleError(f"manifest digest or size mismatch for {name}")
        expected_file_records.append(record)
    canonical_entries = {
        f"{BUNDLE_ROOT}/{name}": payloads[f"{BUNDLE_ROOT}/{name}"]
        for name in PACKAGE_FILES
    }
    canonical_entries[f"{BUNDLE_ROOT}/{MANIFEST_NAME}"] = payloads[manifest_path]
    if tar_bytes != _canonical_tar_bytes(canonical_entries):
        raise BundleError("tar bytes are not the canonical deterministic serialization")
    if bundle_bytes != _canonical_gzip_bytes(tar_bytes):
        raise BundleError("gzip bytes are not the canonical deterministic compression")
    return {
        "schema_version": BUNDLE_VERSION,
        "bundle_sha256": bundle_digest,
        "source_commit": source_commit,
        "file_count": len(expected_file_records),
        "bundle_verified": True,
        "non_authorizing": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build deterministic verifier bundle")
    build.add_argument("--root", default=".", help="repository root")
    build.add_argument("--output", required=True, help="output tar.gz path")
    build.add_argument("--source-commit", required=True, help="trusted build source commit SHA")
    verify = subparsers.add_parser("verify", help="verify an existing bundle and its manifest")
    verify.add_argument("--bundle", required=True, help="bundle tar.gz path")
    verify.add_argument("--expected-source-commit", help="optional independently expected source commit")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_bundle(args.root, args.output, args.source_commit)
        else:
            result = verify_bundle(args.bundle, expected_source_commit=args.expected_source_commit)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (BundleError, OSError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
