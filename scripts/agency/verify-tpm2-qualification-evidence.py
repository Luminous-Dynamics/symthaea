#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strict verifier for a successful Agency TPM2 local-qualification capsule.

The verifier never extracts the archive. It accepts only the closed-world V1
PASS shape produced by `qualify-tpm2-local.sh` and treats every tar member as
hostile input.

The release path snapshots the archive through one no-follow regular-file
descriptor, hashes those exact bytes, and parses those same bytes. This prevents
pathname replacement between the externally bound archive hash check and the
semantic evidence interpretation.

Archive + sidecar integrity is not producer authentication. For a release gate,
use --release with independently obtained archive/head/tree commitments.
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
import tomllib
from typing import Dict, Iterable, Tuple

PRODUCER_SCHEMA = "symthaea.agency-tpm2-local-qualification.v1"
ACCEPTANCE_SCHEMA = "symthaea.agency-tpm2-evidence-acceptance.v1"
PINNED_RUST_RELEASE = "1.96.0"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_TAR_STREAM_BYTES = 320 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_FILE_BYTES = 256 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
NIX_STORE_ENTRY = re.compile(r"^[0-9abcdfghijklmnpqrsvwxyz]{32}-[^/]+$")
ZERO64 = "0" * 64

# V1 is intentionally closed-world. Producer evidence additions require an
# explicit verifier/profile revision instead of silently widening acceptance.
ALLOWED_FILES = {
    "HEAD", "TREE", "DETACHED_WORKTREE_STATUS.txt", "RUSTC.txt", "CARGO.txt",
    "NIX.txt", "UNAME.txt", "CARGO_LOCK_BEFORE_SHA256.txt", "FLAKE_LOCK_SHA256.txt",
    "RUST_TOOLCHAIN_TOML_SHA256.txt", "Cargo.lock.before", "FLAKE_METADATA.json",
    "NIXPKGS_LOCKED.json", "Cargo.lock.candidate", "CARGO_LOCK_DIFF.patch",
    "LOCK_RECONCILIATION.txt", "CARGO_LOCK_CANDIDATE_SHA256.txt", "RUSTFMT.stdout",
    "RUSTFMT.stderr", "RUSTFMT_EXIT_CODE.txt", "CARGO_TEST.log", "CARGO_CLIPPY.log",
    "PROBE_BUILD.log", "PROBE_SHA256.txt", "TPM2_VERIFIER_STORE.txt",
    "QUOTE_WRAPPER_PATH.txt", "CHECKQUOTE_WRAPPER_PATH.txt", "QUOTE_WRAPPER_SHA256.txt",
    "CHECKQUOTE_WRAPPER_SHA256.txt", "TPM2_WRAPPER_FILE.txt", "QUOTE_WRAPPER_ELF.txt",
    "CHECKQUOTE_WRAPPER_ELF.txt", "TPM2_VERIFIER_REFERENCES.txt",
    "QUOTE_TCTI_OVERRIDE.stderr", "QUOTE_FORMAT_OVERRIDE.stderr",
    "CHECK_TCTI_OVERRIDE.stderr", "SWTPM.log", "SWTPM_VERSION.txt",
    "TPM2_TOOLS_VERSION.txt", "AK_PUBLIC_SHA256.txt", "HERMETIC_BASELINE.stdout",
    "HERMETIC_BASELINE.stderr", "APPROVED_PCR_PROFILE.txt", "TPM2_VERIFIED.txt",
    "TPM2_MUTATED.stdout", "TPM2_MUTATED.stderr", "RESULT.txt", "LAST_PHASE.txt",
    "EXIT_CODE.txt", "CARGO_LOCK_STALE.txt", "QUALIFICATION_RESULT.json",
    "MANIFEST.sha256",
}
MAY_BE_EMPTY = {
    "DETACHED_WORKTREE_STATUS.txt", "CARGO_LOCK_DIFF.patch", "RUSTFMT.stdout",
    "RUSTFMT.stderr", "HERMETIC_BASELINE.stdout", "HERMETIC_BASELINE.stderr",
    "TPM2_MUTATED.stdout", "SWTPM.log",
}


class EvidenceError(Exception):
    """Closed-world evidence rejection."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def one_line(data: bytes, name: str) -> str:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"{name} is not UTF-8") from exc
    if len(lines) != 1:
        raise EvidenceError(f"{name} must contain exactly one line")
    return lines[0]


def parse_sha_line(data: bytes, name: str) -> Tuple[str, str]:
    match = SHA_LINE.fullmatch(one_line(data, name))
    if not match:
        raise EvidenceError(f"{name} is not a canonical sha256sum line")
    return match.group(1), match.group(2)


def nonzero_hex64(value: str, what: str) -> str:
    if not HEX64.fullmatch(value) or value == ZERO64:
        raise EvidenceError(f"{what} is not a nonzero 32-byte lowercase hex digest")
    return value


def canonical_member_name(raw: str) -> str:
    if raw.startswith("/") or "\\" in raw or "\x00" in raw:
        raise EvidenceError(f"unsafe tar member name: {raw!r}")
    parts = list(PurePosixPath(raw).parts)
    if any(part in ("", "..") for part in parts):
        raise EvidenceError(f"unsafe tar member path: {raw!r}")
    if parts and parts[0] == ".":
        parts = parts[1:]
    if len(parts) != 1:
        raise EvidenceError(f"nested/unexpected tar member path: {raw!r}")
    return parts[0]


def canonical_nix_store_path(value: str, what: str, *, root_only: bool = False) -> str:
    if not value or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise EvidenceError(f"{what} contains invalid control characters")
    path = PurePosixPath(value)
    if str(path) != value:
        raise EvidenceError(f"{what} is not a canonical POSIX path")
    parts = path.parts
    if len(parts) < 4 or parts[:3] != ("/", "nix", "store"):
        raise EvidenceError(f"{what} is not beneath /nix/store")
    if not NIX_STORE_ENTRY.fullmatch(parts[3]):
        raise EvidenceError(f"{what} has a malformed Nix store entry")
    if root_only and len(parts) != 4:
        raise EvidenceError(f"{what} must name one exact Nix store output root")
    return value


def read_archive_snapshot(archive: Path) -> bytes:
    """Read one bounded, no-follow regular-file snapshot of the archive path."""
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise EvidenceError("platform lacks O_NOFOLLOW required by the V1 archive snapshot")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    try:
        fd = os.open(archive, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open qualification archive safely: {archive}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise EvidenceError("qualification archive path is not a regular file")
        if metadata.st_size <= 0 or metadata.st_size > MAX_ARCHIVE_BYTES:
            raise EvidenceError("archive size is empty or exceeds the V1 bound")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(MAX_ARCHIVE_BYTES + 1)
        if len(data) != metadata.st_size or len(data) > MAX_ARCHIVE_BYTES:
            raise EvidenceError("archive changed during snapshot or exceeds the V1 bound")
        return data
    finally:
        os.close(fd)


def check_gzip_header(archive_bytes: bytes) -> None:
    header = archive_bytes[:10]
    if len(header) != 10 or header[:3] != b"\x1f\x8b\x08":
        raise EvidenceError("archive is not a gzip/deflate stream")
    if int.from_bytes(header[4:8], "little") != 0:
        raise EvidenceError("gzip mtime is not normalized to zero")
    if header[3] != 0:
        raise EvidenceError("gzip header contains non-normalized optional metadata")


def bounded_tar_stream(archive_bytes: bytes) -> bytes:
    check_gzip_header(archive_bytes)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(archive_bytes), mode="rb") as stream:
            tar_bytes = stream.read(MAX_TAR_STREAM_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise EvidenceError("gzip stream is malformed") from exc
    if not tar_bytes or len(tar_bytes) > MAX_TAR_STREAM_BYTES:
        raise EvidenceError("decompressed tar stream exceeds the V1 bound")
    return tar_bytes


def load_archive_bytes(archive_bytes: bytes) -> Dict[str, bytes]:
    if not archive_bytes or len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise EvidenceError("archive snapshot is empty or exceeds the V1 bound")
    tar_bytes = bounded_tar_stream(archive_bytes)

    files: Dict[str, bytes] = {}
    raw_names = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        for member in tf.getmembers():
            if member.name in raw_names:
                raise EvidenceError(f"duplicate tar member: {member.name!r}")
            raw_names.add(member.name)
            if member.uid != 0 or member.gid != 0 or int(member.mtime) != 0:
                raise EvidenceError(f"non-normalized tar metadata: {member.name!r}")
            if member.isdir():
                if member.name not in (".", "./"):
                    raise EvidenceError(f"unexpected directory member: {member.name!r}")
                continue
            if not member.isfile():
                raise EvidenceError(f"link/device/FIFO/special tar member rejected: {member.name!r}")
            name = canonical_member_name(member.name)
            if name in files:
                raise EvidenceError(f"duplicate normalized evidence file: {name}")
            if name not in ALLOWED_FILES:
                raise EvidenceError(f"file outside the V1 evidence allowlist: {name}")
            if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise EvidenceError(f"evidence file exceeds V1 size bound: {name}")
            total += member.size
            if total > MAX_TOTAL_FILE_BYTES:
                raise EvidenceError("expanded evidence exceeds V1 total file-size bound")
            reader = tf.extractfile(member)
            if reader is None:
                raise EvidenceError(f"cannot read evidence member: {name}")
            data = reader.read(MAX_MEMBER_BYTES + 1)
            if len(data) != member.size or len(data) > MAX_MEMBER_BYTES:
                raise EvidenceError(f"tar/member size mismatch: {name}")
            files[name] = data

    if set(files) != ALLOWED_FILES:
        raise EvidenceError(
            f"closed-world evidence mismatch; missing={sorted(ALLOWED_FILES - set(files))}, "
            f"extra={sorted(set(files) - ALLOWED_FILES)}"
        )
    for name in ALLOWED_FILES - MAY_BE_EMPTY:
        if not files[name]:
            raise EvidenceError(f"required evidence file is empty: {name}")
    return files


def load_archive(archive: Path) -> Dict[str, bytes]:
    """Convenience helper for tests/non-release callers; still snapshots once."""
    return load_archive_bytes(read_archive_snapshot(archive))


def verify_manifest(files: Dict[str, bytes]) -> str:
    try:
        lines = files["MANIFEST.sha256"].decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvidenceError("MANIFEST.sha256 is not UTF-8") from exc
    entries: Dict[str, str] = {}
    for line in lines:
        match = SHA_LINE.fullmatch(line)
        if not match:
            raise EvidenceError(f"malformed manifest line: {line!r}")
        raw_name = match.group(2)
        if not raw_name.startswith("./"):
            raise EvidenceError(f"non-canonical manifest path: {raw_name!r}")
        name = canonical_member_name(raw_name)
        if name == "MANIFEST.sha256" or name in entries:
            raise EvidenceError(f"duplicate/self-referential manifest entry: {name}")
        entries[name] = match.group(1)

    expected = set(files) - {"MANIFEST.sha256"}
    if set(entries) != expected:
        raise EvidenceError(
            f"manifest coverage mismatch; missing={sorted(expected - set(entries))}, "
            f"extra={sorted(set(entries) - expected)}"
        )
    for name, expected_hash in entries.items():
        if sha256_bytes(files[name]) != expected_hash:
            raise EvidenceError(f"manifest digest mismatch: {name}")
    return sha256_bytes(files["MANIFEST.sha256"])


def verify_lock_evidence(files: Dict[str, bytes]) -> None:
    try:
        before = tomllib.loads(files["Cargo.lock.before"].decode("utf-8"))
        candidate = tomllib.loads(files["Cargo.lock.candidate"].decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EvidenceError("Cargo lock evidence is malformed") from exc

    def package_key(package: dict) -> tuple:
        return (package["name"], package["version"], package.get("source"))

    before_map = {package_key(p): p for p in before.get("package", [])}
    after_map = {package_key(p): p for p in candidate.get("package", [])}
    removed = set(before_map) - set(after_map)
    changed = {key for key in before_map.keys() & after_map if before_map[key] != after_map[key]}
    added = set(after_map) - set(before_map)
    sourced_added = {key for key in added if key[2] is not None}
    if removed or changed or sourced_added:
        raise EvidenceError("Cargo reconciliation includes a forbidden dependency/source change")

    before_hash, before_path = parse_sha_line(
        files["CARGO_LOCK_BEFORE_SHA256.txt"], "CARGO_LOCK_BEFORE_SHA256.txt"
    )
    after_hash, after_path = parse_sha_line(
        files["CARGO_LOCK_CANDIDATE_SHA256.txt"], "CARGO_LOCK_CANDIDATE_SHA256.txt"
    )
    if before_path != "Cargo.lock" or after_path != "Cargo.lock":
        raise EvidenceError("Cargo lock hash evidence names an unexpected path")
    if before_hash != sha256_bytes(files["Cargo.lock.before"]):
        raise EvidenceError("Cargo.lock.before hash mismatch")
    if after_hash != sha256_bytes(files["Cargo.lock.candidate"]):
        raise EvidenceError("Cargo.lock.candidate hash mismatch")
    if files["Cargo.lock.before"] != files["Cargo.lock.candidate"]:
        raise EvidenceError("PASS archive contains a stale checked-in Cargo.lock")
    if files["CARGO_LOCK_DIFF.patch"] != b"":
        raise EvidenceError("PASS archive retains a non-empty Cargo.lock diff")


def verify_flake_evidence(files: Dict[str, bytes]) -> dict:
    try:
        metadata = json.loads(files["FLAKE_METADATA.json"])
        retained = json.loads(files["NIXPKGS_LOCKED.json"])
        locked = metadata["locks"]["nodes"]["nixpkgs"]["locked"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EvidenceError("flake/nixpkgs lineage evidence is malformed") from exc
    if retained != locked:
        raise EvidenceError("NIXPKGS_LOCKED.json disagrees with FLAKE_METADATA.json")
    flake_hash, flake_path = parse_sha_line(
        files["FLAKE_LOCK_SHA256.txt"], "FLAKE_LOCK_SHA256.txt"
    )
    toolchain_hash, toolchain_path = parse_sha_line(
        files["RUST_TOOLCHAIN_TOML_SHA256.txt"], "RUST_TOOLCHAIN_TOML_SHA256.txt"
    )
    if flake_path != "flake.lock" or toolchain_path != "rust-toolchain.toml":
        raise EvidenceError("lock/toolchain digest evidence names an unexpected path")
    nonzero_hex64(flake_hash, "flake.lock SHA-256")
    nonzero_hex64(toolchain_hash, "rust-toolchain.toml SHA-256")
    return {
        "nixpkgs_locked": retained,
        "flake_lock_sha256": flake_hash,
        "rust_toolchain_sha256": toolchain_hash,
    }


def verify_status(files: Dict[str, bytes]) -> Tuple[str, str]:
    head = one_line(files["HEAD"], "HEAD")
    tree = one_line(files["TREE"], "TREE")
    if not HEX40.fullmatch(head) or not HEX40.fullmatch(tree):
        raise EvidenceError("HEAD/TREE is not canonical 40-character lowercase Git hex")
    if files["DETACHED_WORKTREE_STATUS.txt"] != b"":
        raise EvidenceError("qualified detached worktree was not clean")

    try:
        result = json.loads(files["QUALIFICATION_RESULT.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("QUALIFICATION_RESULT.json is malformed") from exc
    expected_keys = {"schema", "result", "last_phase", "exit_code", "cargo_lock_stale"}
    if set(result) != expected_keys or result.get("schema") != PRODUCER_SCHEMA:
        raise EvidenceError("qualification result schema/field set mismatch")

    text_result = one_line(files["RESULT.txt"], "RESULT.txt")
    text_phase = one_line(files["LAST_PHASE.txt"], "LAST_PHASE.txt")
    text_exit = one_line(files["EXIT_CODE.txt"], "EXIT_CODE.txt")
    text_lock = one_line(files["CARGO_LOCK_STALE.txt"], "CARGO_LOCK_STALE.txt")
    if text_exit != "0" or text_lock != "0":
        raise EvidenceError("PASS archive must record exit=0 and cargo_lock_stale=0")
    if text_result != "PASS" or text_phase != "complete":
        raise EvidenceError("archive is not a complete PASS qualification")
    if (
        result["result"] != "PASS"
        or result["last_phase"] != "complete"
        or result["exit_code"] != 0
        or result["cargo_lock_stale"] is not False
    ):
        raise EvidenceError("QUALIFICATION_RESULT.json does not encode exact PASS semantics")

    rustc = files["RUSTC.txt"].decode("utf-8", errors="strict")
    cargo = files["CARGO.txt"].decode("utf-8", errors="strict")
    if f"release: {PINNED_RUST_RELEASE}" not in rustc:
        raise EvidenceError(f"Rust evidence is not pinned release {PINNED_RUST_RELEASE}")
    if not cargo.startswith(f"cargo {PINNED_RUST_RELEASE} "):
        raise EvidenceError(f"Cargo evidence is not pinned release {PINNED_RUST_RELEASE}")
    return head, tree


def verify_tpm_evidence(files: Dict[str, bytes]) -> dict:
    probe_hash, probe_path = parse_sha_line(files["PROBE_SHA256.txt"], "PROBE_SHA256.txt")
    nonzero_hex64(probe_hash, "qualification probe SHA-256")
    if probe_path != "target/debug/tpm2_attestation_probe":
        raise EvidenceError("qualification probe path is not canonical")

    store = canonical_nix_store_path(
        one_line(files["TPM2_VERIFIER_STORE.txt"], "TPM2_VERIFIER_STORE.txt"),
        "TPM verifier store",
        root_only=True,
    )
    quote_path = canonical_nix_store_path(
        one_line(files["QUOTE_WRAPPER_PATH.txt"], "QUOTE_WRAPPER_PATH.txt"),
        "quote wrapper path",
    )
    check_path = canonical_nix_store_path(
        one_line(files["CHECKQUOTE_WRAPPER_PATH.txt"], "CHECKQUOTE_WRAPPER_PATH.txt"),
        "checkquote wrapper path",
    )
    if quote_path != f"{store}/bin/symthaea-tpm2-quote":
        raise EvidenceError("quote wrapper path does not match the recorded verifier store")
    if check_path != f"{store}/bin/symthaea-tpm2-checkquote":
        raise EvidenceError("checkquote wrapper path does not match the recorded verifier store")

    quote_hash, quote_hash_path = parse_sha_line(
        files["QUOTE_WRAPPER_SHA256.txt"], "QUOTE_WRAPPER_SHA256.txt"
    )
    check_hash, check_hash_path = parse_sha_line(
        files["CHECKQUOTE_WRAPPER_SHA256.txt"], "CHECKQUOTE_WRAPPER_SHA256.txt"
    )
    nonzero_hex64(quote_hash, "quote wrapper SHA-256")
    nonzero_hex64(check_hash, "checkquote wrapper SHA-256")
    if quote_hash_path != quote_path or check_hash_path != check_path:
        raise EvidenceError("wrapper digest evidence names a different executable")
    if b"INTERP" in files["QUOTE_WRAPPER_ELF.txt"] or b"INTERP" in files["CHECKQUOTE_WRAPPER_ELF.txt"]:
        raise EvidenceError("TPM verifier launcher has a dynamic ELF interpreter")

    references = files["TPM2_VERIFIER_REFERENCES.txt"].decode(
        "utf-8", errors="strict"
    ).splitlines()
    if not references:
        raise EvidenceError("TPM verifier closure reference set is empty")
    for reference in references:
        canonical_nix_store_path(reference, "TPM verifier closure reference", root_only=True)

    for name in (
        "QUOTE_TCTI_OVERRIDE.stderr",
        "QUOTE_FORMAT_OVERRIDE.stderr",
        "CHECK_TCTI_OVERRIDE.stderr",
    ):
        if b"option override rejected" not in files[name]:
            raise EvidenceError(f"missing fail-before-execution override evidence: {name}")
    baseline = files["HERMETIC_BASELINE.stderr"].lower()
    if any(marker in baseline for marker in (b"ld_preload", b"cannot be preloaded", b"ld.so")):
        raise EvidenceError("hermetic baseline contains a dynamic-loader warning")

    approved = nonzero_hex64(
        one_line(files["APPROVED_PCR_PROFILE.txt"], "APPROVED_PCR_PROFILE.txt"),
        "approved PCR profile",
    )
    lines = files["TPM2_VERIFIED.txt"].decode("utf-8", errors="strict").splitlines()
    if not lines or lines[0] != "platform_attestation=verified":
        raise EvidenceError("fresh TPM success marker is missing")
    fields: Dict[str, str] = {}
    for line in lines[1:]:
        if "=" not in line:
            raise EvidenceError("malformed TPM2_VERIFIED field")
        key, value = line.split("=", 1)
        if key in fields:
            raise EvidenceError(f"duplicate TPM2_VERIFIED field: {key}")
        fields[key] = value
    expected = {"policy_digest", "pcr_profile_digest", "ak_public_digest", "challenge_digest"}
    if set(fields) != expected:
        raise EvidenceError("TPM2_VERIFIED field set is not closed-world V1")
    for key, value in fields.items():
        nonzero_hex64(value, key)
    if fields["pcr_profile_digest"] != approved:
        raise EvidenceError("verified PCR profile disagrees with reviewed profile")
    if b"PCR state is not an approved profile" not in files["TPM2_MUTATED.stderr"]:
        raise EvidenceError("PCR mutation fail-closed evidence is absent")

    return {
        "approved_pcr_profile": approved,
        "policy_digest": fields["policy_digest"],
        "ak_public_digest": fields["ak_public_digest"],
        "challenge_digest": fields["challenge_digest"],
        "probe_sha256": probe_hash,
        "quote_wrapper_sha256": quote_hash,
        "checkquote_wrapper_sha256": check_hash,
        "verifier_store": store,
    }


def archive_expected_hash(args: argparse.Namespace, archive: Path) -> Tuple[str, str]:
    if args.expected_archive_sha256:
        value = args.expected_archive_sha256.lower()
        if not HEX64.fullmatch(value):
            raise EvidenceError("--expected-archive-sha256 must be 64 hexadecimal characters")
        return value, "caller"
    sidecar = Path(args.sidecar) if args.sidecar else Path(str(archive) + ".sha256")
    if not sidecar.is_file():
        raise EvidenceError("archive hash sidecar missing; supply --expected-archive-sha256 or --sidecar")
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise EvidenceError("archive hash sidecar must contain one sha256sum line")
    match = SHA_LINE.fullmatch(lines[0])
    if not match:
        raise EvidenceError("archive hash sidecar is malformed")
    return match.group(1), "sidecar-unanchored"


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="qualification .tar.gz archive")
    parser.add_argument("--expected-archive-sha256", help="independently obtained archive SHA-256")
    parser.add_argument("--sidecar", help="sha256sum sidecar; default ARCHIVE.sha256")
    parser.add_argument("--expected-head", help="independently obtained exact 40-hex Git HEAD")
    parser.add_argument("--expected-tree", help="independently obtained exact 40-hex Git tree")
    parser.add_argument(
        "--release",
        action="store_true",
        help="require independently supplied archive/head/tree bindings",
    )
    parser.add_argument("--json-out", help="optional new path for acceptance JSON")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    archive = Path(args.archive)
    try:
        if args.release and not (
            args.expected_archive_sha256 and args.expected_head and args.expected_tree
        ):
            raise EvidenceError(
                "--release requires --expected-archive-sha256, --expected-head, and --expected-tree"
            )
        expected_hash, hash_source = archive_expected_hash(args, archive)

        # One snapshot is the security boundary: hash and semantic parsing both
        # consume these exact bytes, never two independent pathname opens.
        archive_bytes = read_archive_snapshot(archive)
        actual_hash = sha256_bytes(archive_bytes)
        if actual_hash != expected_hash:
            raise EvidenceError(
                f"archive SHA-256 mismatch: actual={actual_hash}, expected={expected_hash}"
            )

        expected_head = args.expected_head.lower() if args.expected_head else None
        expected_tree = args.expected_tree.lower() if args.expected_tree else None
        if expected_head is not None and not HEX40.fullmatch(expected_head):
            raise EvidenceError("--expected-head must be 40 lowercase hex characters")
        if expected_tree is not None and not HEX40.fullmatch(expected_tree):
            raise EvidenceError("--expected-tree must be 40 lowercase hex characters")

        files = load_archive_bytes(archive_bytes)
        manifest_hash = verify_manifest(files)
        head, tree = verify_status(files)
        if expected_head is not None and head != expected_head:
            raise EvidenceError(f"Git HEAD mismatch: evidence={head}, expected={expected_head}")
        if expected_tree is not None and tree != expected_tree:
            raise EvidenceError(f"Git tree mismatch: evidence={tree}, expected={expected_tree}")
        verify_lock_evidence(files)
        flake = verify_flake_evidence(files)
        tpm = verify_tpm_evidence(files)

        acceptance = {
            "schema": ACCEPTANCE_SCHEMA,
            "accepted": True,
            "qualification_result": "PASS",
            "archive_sha256": actual_hash,
            "archive_hash_source": hash_source,
            "manifest_sha256": manifest_hash,
            "head": head,
            "tree": tree,
            "external_head_bound": expected_head is not None,
            "external_tree_bound": expected_tree is not None,
            "release_bound": bool(args.release),
            **flake,
            **tpm,
        }
        encoded = json.dumps(acceptance, sort_keys=True, indent=2) + "\n"
        sys.stdout.write(encoded)
        if args.json_out:
            out = Path(args.json_out)
            try:
                with out.open("x", encoding="utf-8") as handle:
                    handle.write(encoded)
            except FileExistsError as exc:
                raise EvidenceError(f"refusing to overwrite --json-out path: {out}") from exc
        return 0
    except (
        EvidenceError,
        OSError,
        EOFError,
        UnicodeDecodeError,
        tarfile.TarError,
    ) as exc:
        print(f"evidence_verification=REJECTED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
