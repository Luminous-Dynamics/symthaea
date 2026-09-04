#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify a Symthaea Agency TPM2 local-qualification evidence archive.

This verifier is intentionally independent of the producer script. It never
extracts the archive and treats tar metadata/content as hostile input.

A successful default invocation means the archive is internally consistent and
claims PASS. It does *not* authenticate who produced the archive. For release
use, pass an independently obtained --expected-archive-sha256 and
--expected-head (and preferably --expected-tree).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tomllib
from typing import Dict, Iterable, Tuple

SCHEMA = "symthaea.agency-tpm2-local-qualification.v1"
VERIFIER_SCHEMA = "symthaea.agency-tpm2-evidence-acceptance.v1"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_FILE_BYTES = 256 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")

# V1 is deliberately closed-world. Adding evidence to the producer requires an
# explicit verifier/profile update instead of silently widening the archive.
ALLOWED_FILES = {
    "HEAD",
    "TREE",
    "DETACHED_WORKTREE_STATUS.txt",
    "RUSTC.txt",
    "CARGO.txt",
    "NIX.txt",
    "UNAME.txt",
    "CARGO_LOCK_BEFORE_SHA256.txt",
    "FLAKE_LOCK_SHA256.txt",
    "RUST_TOOLCHAIN_TOML_SHA256.txt",
    "Cargo.lock.before",
    "FLAKE_METADATA.json",
    "NIXPKGS_LOCKED.json",
    "Cargo.lock.candidate",
    "CARGO_LOCK_DIFF.patch",
    "LOCK_RECONCILIATION.txt",
    "CARGO_LOCK_CANDIDATE_SHA256.txt",
    "RUSTFMT.stdout",
    "RUSTFMT.stderr",
    "RUSTFMT_EXIT_CODE.txt",
    "CARGO_TEST.log",
    "CARGO_CLIPPY.log",
    "PROBE_BUILD.log",
    "PROBE_SHA256.txt",
    "TPM2_VERIFIER_STORE.txt",
    "QUOTE_WRAPPER_PATH.txt",
    "CHECKQUOTE_WRAPPER_PATH.txt",
    "QUOTE_WRAPPER_SHA256.txt",
    "CHECKQUOTE_WRAPPER_SHA256.txt",
    "TPM2_WRAPPER_FILE.txt",
    "QUOTE_WRAPPER_ELF.txt",
    "CHECKQUOTE_WRAPPER_ELF.txt",
    "TPM2_VERIFIER_REFERENCES.txt",
    "QUOTE_TCTI_OVERRIDE.stderr",
    "QUOTE_FORMAT_OVERRIDE.stderr",
    "CHECK_TCTI_OVERRIDE.stderr",
    "SWTPM.log",
    "SWTPM_VERSION.txt",
    "TPM2_TOOLS_VERSION.txt",
    "AK_PUBLIC_SHA256.txt",
    "HERMETIC_BASELINE.stdout",
    "HERMETIC_BASELINE.stderr",
    "APPROVED_PCR_PROFILE.txt",
    "TPM2_VERIFIED.txt",
    "TPM2_MUTATED.stdout",
    "TPM2_MUTATED.stderr",
    "RESULT.txt",
    "LAST_PHASE.txt",
    "EXIT_CODE.txt",
    "CARGO_LOCK_STALE.txt",
    "QUALIFICATION_RESULT.json",
    "MANIFEST.sha256",
}
REQUIRED_NONEMPTY = ALLOWED_FILES - {
    "DETACHED_WORKTREE_STATUS.txt",
    "CARGO_LOCK_DIFF.patch",
    "RUSTFMT.stdout",
    "RUSTFMT.stderr",
    "HERMETIC_BASELINE.stdout",
    "HERMETIC_BASELINE.stderr",
    "TPM2_MUTATED.stdout",
    "SWTPM.log",
}


class EvidenceError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def one_line(data: bytes, name: str) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"{name} is not UTF-8") from exc
    lines = text.splitlines()
    if len(lines) != 1:
        raise EvidenceError(f"{name} must contain exactly one line")
    return lines[0]


def parse_sha256sum(data: bytes, name: str) -> Tuple[str, str]:
    line = one_line(data, name)
    match = SHA_LINE.fullmatch(line)
    if not match:
        raise EvidenceError(f"{name} is not a canonical sha256sum line")
    return match.group(1), match.group(2)


def canonical_member_name(raw: str) -> str:
    # GNU tar -C <dir> -cf - . emits '.' plus './name'. Accept directory
    # entries separately, but regular files must normalize to one top-level
    # evidence filename with no traversal or nested path.
    if raw.startswith("/") or "\\" in raw or "\x00" in raw:
        raise EvidenceError(f"unsafe tar member name: {raw!r}")
    path = PurePosixPath(raw)
    if any(part in ("", "..") for part in path.parts):
        raise EvidenceError(f"unsafe tar member path: {raw!r}")
    parts = list(path.parts)
    if parts and parts[0] == ".":
        parts = parts[1:]
    if len(parts) != 1:
        raise EvidenceError(f"nested/unexpected tar member path: {raw!r}")
    return parts[0]


def check_gzip_header(archive: Path) -> None:
    with archive.open("rb") as handle:
        header = handle.read(10)
    if len(header) != 10 or header[0:3] != b"\x1f\x8b\x08":
        raise EvidenceError("archive is not a gzip stream using deflate")
    mtime = int.from_bytes(header[4:8], "little")
    if mtime != 0:
        raise EvidenceError("gzip mtime is not normalized to zero")
    flags = header[3]
    # gzip -n must not retain original filename or comment.
    if flags & 0x18:
        raise EvidenceError("gzip header retains filename/comment metadata")


def load_archive(archive: Path) -> Dict[str, bytes]:
    size = archive.stat().st_size
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise EvidenceError("archive size is empty or exceeds V1 bound")
    check_gzip_header(archive)

    files: Dict[str, bytes] = {}
    seen_raw = set()
    total = 0
    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf.getmembers():
            if member.name in seen_raw:
                raise EvidenceError(f"duplicate tar member: {member.name!r}")
            seen_raw.add(member.name)
            if member.uid != 0 or member.gid != 0 or int(member.mtime) != 0:
                raise EvidenceError(f"non-normalized tar metadata: {member.name!r}")
            if member.isdir():
                # Only the producer's root '.' directory is expected.
                if member.name not in (".", "./"):
                    raise EvidenceError(f"unexpected directory member: {member.name!r}")
                continue
            if not member.isfile():
                raise EvidenceError(f"non-regular tar member rejected: {member.name!r}")
            name = canonical_member_name(member.name)
            if name in files:
                raise EvidenceError(f"duplicate normalized file: {name}")
            if name not in ALLOWED_FILES:
                raise EvidenceError(f"file is outside the V1 evidence allowlist: {name}")
            if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise EvidenceError(f"evidence member exceeds V1 size bound: {name}")
            total += member.size
            if total > MAX_TOTAL_FILE_BYTES:
                raise EvidenceError("expanded evidence exceeds V1 total size bound")
            extracted = tf.extractfile(member)
            if extracted is None:
                raise EvidenceError(f"unable to read evidence member: {name}")
            data = extracted.read(MAX_MEMBER_BYTES + 1)
            if len(data) != member.size or len(data) > MAX_MEMBER_BYTES:
                raise EvidenceError(f"evidence member size mismatch: {name}")
            files[name] = data

    missing = ALLOWED_FILES - files.keys()
    extra = files.keys() - ALLOWED_FILES
    if missing or extra:
        raise EvidenceError(f"closed-world evidence mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    for name in REQUIRED_NONEMPTY:
        if not files[name]:
            raise EvidenceError(f"required evidence file is empty: {name}")
    return files


def verify_manifest(files: Dict[str, bytes]) -> str:
    try:
        text = files["MANIFEST.sha256"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError("MANIFEST.sha256 is not UTF-8") from exc
    entries: Dict[str, str] = {}
    for line in text.splitlines():
        match = SHA_LINE.fullmatch(line)
        if not match:
            raise EvidenceError(f"malformed MANIFEST.sha256 line: {line!r}")
        raw_name = match.group(2)
        if not raw_name.startswith("./"):
            raise EvidenceError(f"manifest path is not producer-canonical: {raw_name!r}")
        name = canonical_member_name(raw_name)
        if name == "MANIFEST.sha256" or name in entries:
            raise EvidenceError(f"duplicate/self-referential manifest entry: {name}")
        entries[name] = match.group(1)

    expected_names = set(files) - {"MANIFEST.sha256"}
    if set(entries) != expected_names:
        raise EvidenceError(
            f"manifest coverage mismatch; missing={sorted(expected_names - set(entries))}, "
            f"extra={sorted(set(entries) - expected_names)}"
        )
    for name, expected in entries.items():
        actual = sha256_bytes(files[name])
        if actual != expected:
            raise EvidenceError(f"manifest digest mismatch: {name}")
    return sha256_bytes(files["MANIFEST.sha256"])


def verify_lock_reconciliation(files: Dict[str, bytes], require_pass: bool) -> None:
    try:
        before = tomllib.loads(files["Cargo.lock.before"].decode("utf-8"))
        candidate = tomllib.loads(files["Cargo.lock.candidate"].decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EvidenceError("retained Cargo lock evidence is malformed") from exc

    def key(pkg: dict) -> tuple:
        return (pkg["name"], pkg["version"], pkg.get("source"))

    b = {key(pkg): pkg for pkg in before.get("package", [])}
    a = {key(pkg): pkg for pkg in candidate.get("package", [])}
    removed = set(b) - set(a)
    changed = {k for k in b.keys() & a if b[k] != a[k]}
    added = set(a) - set(b)
    sourced_added = {k for k in added if k[2] is not None}
    if removed or changed or sourced_added:
        raise EvidenceError("Cargo lock reconciliation contains forbidden sourced/dependency changes")

    before_hash, _ = parse_sha256sum(files["CARGO_LOCK_BEFORE_SHA256.txt"], "CARGO_LOCK_BEFORE_SHA256.txt")
    candidate_hash, _ = parse_sha256sum(files["CARGO_LOCK_CANDIDATE_SHA256.txt"], "CARGO_LOCK_CANDIDATE_SHA256.txt")
    if before_hash != sha256_bytes(files["Cargo.lock.before"]):
        raise EvidenceError("Cargo.lock.before retained hash mismatch")
    if candidate_hash != sha256_bytes(files["Cargo.lock.candidate"]):
        raise EvidenceError("Cargo.lock.candidate retained hash mismatch")
    if require_pass and files["Cargo.lock.before"] != files["Cargo.lock.candidate"]:
        raise EvidenceError("PASS evidence contains a stale checked-in Cargo.lock")


def verify_flake_lineage(files: Dict[str, bytes]) -> None:
    try:
        metadata = json.loads(files["FLAKE_METADATA.json"])
        retained = json.loads(files["NIXPKGS_LOCKED.json"])
        from_metadata = metadata["locks"]["nodes"]["nixpkgs"]["locked"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EvidenceError("flake/nixpkgs lineage evidence is malformed") from exc
    if retained != from_metadata:
        raise EvidenceError("retained NIXPKGS_LOCKED.json disagrees with FLAKE_METADATA.json")


def require_hex_line(files: Dict[str, bytes], name: str, pattern: re.Pattern[str]) -> str:
    value = one_line(files[name], name)
    if not pattern.fullmatch(value):
        raise EvidenceError(f"{name} has invalid digest/identifier syntax")
    return value


def verify_semantics(files: Dict[str, bytes], require_pass: bool, expected_head: str | None, expected_tree: str | None) -> dict:
    head = require_hex_line(files, "HEAD", HEX40)
    tree = require_hex_line(files, "TREE", HEX40)
    if expected_head is not None and head != expected_head:
        raise EvidenceError(f"HEAD mismatch: evidence={head}, expected={expected_head}")
    if expected_tree is not None and tree != expected_tree:
        raise EvidenceError(f"TREE mismatch: evidence={tree}, expected={expected_tree}")
    if files["DETACHED_WORKTREE_STATUS.txt"] != b"":
        raise EvidenceError("qualified detached worktree was not clean")

    try:
        result = json.loads(files["QUALIFICATION_RESULT.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("QUALIFICATION_RESULT.json is malformed") from exc
    expected_keys = {"schema", "result", "last_phase", "exit_code", "cargo_lock_stale"}
    if set(result) != expected_keys or result.get("schema") != SCHEMA:
        raise EvidenceError("qualification result schema/field set mismatch")

    txt_result = one_line(files["RESULT.txt"], "RESULT.txt")
    txt_phase = one_line(files["LAST_PHASE.txt"], "LAST_PHASE.txt")
    txt_exit = one_line(files["EXIT_CODE.txt"], "EXIT_CODE.txt")
    txt_lock = one_line(files["CARGO_LOCK_STALE.txt"], "CARGO_LOCK_STALE.txt")
    try:
        exit_code = int(txt_exit)
        lock_stale_int = int(txt_lock)
    except ValueError as exc:
        raise EvidenceError("exit/lock status files are not canonical integers") from exc
    if lock_stale_int not in (0, 1):
        raise EvidenceError("CARGO_LOCK_STALE.txt must be 0 or 1")
    if (
        result["result"] != txt_result
        or result["last_phase"] != txt_phase
        or result["exit_code"] != exit_code
        or result["cargo_lock_stale"] != bool(lock_stale_int)
    ):
        raise EvidenceError("qualification status files disagree")

    qualification_pass = (
        txt_result == "PASS"
        and txt_phase == "complete"
        and exit_code == 0
        and lock_stale_int == 0
    )
    if require_pass and not qualification_pass:
        raise EvidenceError(
            f"evidence is internally valid but is not a PASS: result={txt_result}, phase={txt_phase}, exit={exit_code}"
        )

    verify_lock_reconciliation(files, require_pass=qualification_pass)
    verify_flake_lineage(files)

    # The retained producer-level SHA lines must describe the retained objects.
    probe_hash, probe_path = parse_sha256sum(files["PROBE_SHA256.txt"], "PROBE_SHA256.txt")
    if not HEX64.fullmatch(probe_hash) or probe_path != "target/debug/tpm2_attestation_probe":
        raise EvidenceError("qualification probe identity line is not canonical")

    store = one_line(files["TPM2_VERIFIER_STORE.txt"], "TPM2_VERIFIER_STORE.txt")
    quote = one_line(files["QUOTE_WRAPPER_PATH.txt"], "QUOTE_WRAPPER_PATH.txt")
    check = one_line(files["CHECKQUOTE_WRAPPER_PATH.txt"], "CHECKQUOTE_WRAPPER_PATH.txt")
    if not store.startswith("/nix/store/"):
        raise EvidenceError("TPM verifier store is not a Nix-store path")
    if quote != f"{store}/bin/symthaea-tpm2-quote":
        raise EvidenceError("quote wrapper path is not exactly under the recorded verifier store")
    if check != f"{store}/bin/symthaea-tpm2-checkquote":
        raise EvidenceError("checkquote wrapper path is not exactly under the recorded verifier store")
    quote_hash, quote_path = parse_sha256sum(files["QUOTE_WRAPPER_SHA256.txt"], "QUOTE_WRAPPER_SHA256.txt")
    check_hash, check_path = parse_sha256sum(files["CHECKQUOTE_WRAPPER_SHA256.txt"], "CHECKQUOTE_WRAPPER_SHA256.txt")
    if quote_path != quote or check_path != check or not HEX64.fullmatch(quote_hash) or not HEX64.fullmatch(check_hash):
        raise EvidenceError("wrapper SHA/path evidence is inconsistent")
    if b"INTERP" in files["QUOTE_WRAPPER_ELF.txt"] or b"INTERP" in files["CHECKQUOTE_WRAPPER_ELF.txt"]:
        raise EvidenceError("reviewed TPM launcher evidence contains an ELF interpreter")

    references = files["TPM2_VERIFIER_REFERENCES.txt"].decode("utf-8", errors="strict").splitlines()
    if not references or any(not ref.startswith("/nix/store/") for ref in references):
        raise EvidenceError("TPM verifier closure contains a non-Nix-store reference")

    for name in ("QUOTE_TCTI_OVERRIDE.stderr", "QUOTE_FORMAT_OVERRIDE.stderr", "CHECK_TCTI_OVERRIDE.stderr"):
        if b"option override rejected" not in files[name]:
            raise EvidenceError(f"missing override-rejection evidence: {name}")
    baseline_stderr = files["HERMETIC_BASELINE.stderr"].lower()
    if any(marker in baseline_stderr for marker in (b"ld_preload", b"cannot be preloaded", b"ld.so")):
        raise EvidenceError("hermetic baseline retained a dynamic-loader warning")

    approved = one_line(files["APPROVED_PCR_PROFILE.txt"], "APPROVED_PCR_PROFILE.txt")
    if not HEX64.fullmatch(approved):
        raise EvidenceError("approved PCR profile is not a 32-byte hex digest")

    verified_lines = files["TPM2_VERIFIED.txt"].decode("utf-8", errors="strict").splitlines()
    if not verified_lines or verified_lines[0] != "platform_attestation=verified":
        raise EvidenceError("fresh TPM attestation success marker is missing")
    fields = {}
    for line in verified_lines[1:]:
        if "=" not in line:
            raise EvidenceError("malformed TPM2_VERIFIED.txt field")
        k, v = line.split("=", 1)
        if k in fields:
            raise EvidenceError(f"duplicate TPM verification field: {k}")
        fields[k] = v
    required_fields = {"policy_digest", "pcr_profile_digest", "ak_public_digest", "challenge_digest"}
    if set(fields) != required_fields or any(not HEX64.fullmatch(v) for v in fields.values()):
        raise EvidenceError("TPM verification digest field set is not canonical")
    if fields["pcr_profile_digest"] != approved:
        raise EvidenceError("verified PCR profile disagrees with reviewed profile")
    if b"PCR state is not an approved profile" not in files["TPM2_MUTATED.stderr"]:
        raise EvidenceError("PCR mutation fail-closed evidence is missing")

    return {
        "head": head,
        "tree": tree,
        "qualification_pass": qualification_pass,
        "result": txt_result,
        "last_phase": txt_phase,
        "exit_code": exit_code,
        "cargo_lock_stale": bool(lock_stale_int),
        "approved_pcr_profile": approved,
        "policy_digest": fields["policy_digest"],
        "challenge_digest": fields["challenge_digest"],
        "quote_wrapper_sha256": quote_hash,
        "checkquote_wrapper_sha256": check_hash,
    }


def expected_archive_hash(args: argparse.Namespace, archive: Path) -> Tuple[str, str]:
    if args.expected_archive_sha256:
        value = args.expected_archive_sha256.lower()
        if not HEX64.fullmatch(value):
            raise EvidenceError("--expected-archive-sha256 must be exactly 64 lowercase hex characters")
        return value, "caller"

    sidecar = Path(args.sidecar) if args.sidecar else Path(str(archive) + ".sha256")
    if not sidecar.is_file():
        raise EvidenceError(
            "no independently supplied archive hash and default sidecar is absent; "
            "pass --expected-archive-sha256 or --sidecar"
        )
    try:
        line = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvidenceError(f"cannot read archive hash sidecar: {exc}") from exc
    if len(line) != 1:
        raise EvidenceError("archive hash sidecar must contain exactly one sha256sum line")
    match = SHA_LINE.fullmatch(line[0])
    if not match:
        raise EvidenceError("archive hash sidecar is malformed")
    return match.group(1), "sidecar-unanchored"


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="qualification .tar.gz archive")
    parser.add_argument("--expected-archive-sha256", help="independently obtained archive SHA-256")
    parser.add_argument("--sidecar", help="sha256sum sidecar path; default is ARCHIVE.sha256")
    parser.add_argument("--expected-head", help="independently obtained exact 40-hex Git HEAD")
    parser.add_argument("--expected-tree", help="independently obtained exact 40-hex Git tree")
    parser.add_argument(
        "--allow-nonpass",
        action="store_true",
        help="validate a failure archive structurally instead of requiring qualification PASS",
    )
    parser.add_argument("--json-out", help="optional path for the acceptance JSON")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(argv)
    archive = Path(args.archive)
    try:
        if not archive.is_file():
            raise EvidenceError(f"archive does not exist: {archive}")
        expected_hash, hash_source = expected_archive_hash(args, archive)
        actual_hash = sha256_file(archive)
        if actual_hash != expected_hash:
            raise EvidenceError(f"archive SHA-256 mismatch: actual={actual_hash}, expected={expected_hash}")

        expected_head = args.expected_head.lower() if args.expected_head else None
        expected_tree = args.expected_tree.lower() if args.expected_tree else None
        if expected_head is not None and not HEX40.fullmatch(expected_head):
            raise EvidenceError("--expected-head must be exactly 40 lowercase hex characters")
        if expected_tree is not None and not HEX40.fullmatch(expected_tree):
            raise EvidenceError("--expected-tree must be exactly 40 lowercase hex characters")

        files = load_archive(archive)
        manifest_sha = verify_manifest(files)
        semantics = verify_semantics(
            files,
            require_pass=not args.allow_nonpass,
            expected_head=expected_head,
            expected_tree=expected_tree,
        )
        acceptance = {
            "schema": VERIFIER_SCHEMA,
            "archive_sha256": actual_hash,
            "archive_hash_source": hash_source,
            "manifest_sha256": manifest_sha,
            "external_head_bound": expected_head is not None,
            "external_tree_bound": expected_tree is not None,
            **semantics,
        }
        encoded = json.dumps(acceptance, sort_keys=True, indent=2) + "\n"
        sys.stdout.write(encoded)
        if args.json_out:
            out = Path(args.json_out)
            if out.exists():
                raise EvidenceError(f"refusing to overwrite --json-out path: {out}")
            out.write_text(encoded, encoding="utf-8")
        return 0
    except (EvidenceError, OSError, tarfile.TarError) as exc:
        print(f"evidence_verification=REJECTED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
