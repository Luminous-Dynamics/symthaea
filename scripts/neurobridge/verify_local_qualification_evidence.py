#!/usr/bin/env python3
"""Hostile-input verifier for NeuroBridge local qualification PASS archives."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import zlib
from pathlib import Path
from typing import Any

PROFILE = "symthaea-neurobridge-local-qualification-v0.1"
VERIFIER_PROFILE = "symthaea-neurobridge-local-evidence-verifier-v0.1"

MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024

PHASES = (
    "source-clean-before",
    "source-identity",
    "tool-identity",
    "qualifier-shell-syntax",
    "qualifier-static-contracts",
    "rust-core-format",
    "rust-core-substrate",
    "rust-psych-tests",
    "rust-core-clippy",
    "rust-psych-format",
    "rust-psych-clippy",
    "python-syntax",
    "map-compiler",
    "map-crosscheck",
    "fsaverage-extractor",
    "lineage-b-core",
    "lineage-b-source-pair",
    "generator-provenance",
    "bundle-custody",
    "input-snapshot",
    "cli-contracts",
    "source-clean-after",
)

BASE_FILES = {
    "STATUS.env",
    "PHASES.tsv",
    "GIT_STATUS_before.txt",
    "GIT_STATUS_after.txt",
    "GIT_HEAD.txt",
    "GIT_TREE.txt",
    "GIT_COMMIT.txt",
    "SOURCE_LOCKS.sha256",
    "QUALIFIER_FILES.sha256",
    "FOCUSED_WORKFLOWS.sha256",
    "TOOLS.txt",
}
EXPECTED_FILES = BASE_FILES | {f"{phase}.log" for phase in PHASES} | {"MANIFEST.sha256"}
MANIFEST_FILES = EXPECTED_FILES - {"MANIFEST.sha256"}

SOURCE_LOCK_PATHS = {"Cargo.lock", "flake.lock", "rust-toolchain.toml"}
QUALIFIER_PATHS = {
    "scripts/neurobridge/qualify-local.sh",
    "nix/neurobridge-qualification-shell.nix",
    "scripts/neurobridge/check_contracts.py",
    "docs/neuroscience/NEUROBRIDGE_LOCAL_QUALIFICATION_V01.md",
}
WORKFLOW_PATHS = {
    ".github/workflows/substrate-evidence-boundary.yml",
    ".github/workflows/neural-benchmark-quarantine.yml",
    ".github/workflows/fsaverage5-glasser-map-compiler.yml",
    ".github/workflows/fsaverage5-glasser-crosscheck.yml",
    ".github/workflows/fsaverage-hcpmmp-semantic-extractor.yml",
    ".github/workflows/hcpmmp-neuromaps-lineage-b.yml",
    ".github/workflows/hcpmmp-lineage-b-generator-provenance.yml",
    ".github/workflows/hcpmmp-lineage-b-bundle-custody.yml",
    ".github/workflows/hcpmmp-lineage-b-input-snapshot.yml",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
MANIFEST_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")
SHA_LINE_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class VerificationError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_sha256(value: str, context: str) -> str:
    require(isinstance(value, str) and value.startswith("sha256:"), f"{context}: expected sha256: prefix")
    digest = value[7:]
    require(SHA256_RE.fullmatch(digest) is not None, f"{context}: invalid SHA-256")
    return value


def _decompress_normalized_gzip(raw: bytes) -> bytes:
    require(len(raw) <= MAX_COMPRESSED_BYTES, "archive: compressed size limit exceeded")
    require(len(raw) >= 18, "archive: truncated gzip")
    require(raw[:3] == b"\x1f\x8b\x08", "archive: not gzip/deflate")
    require(raw[3] == 0, "archive: gzip flags are not normalized")
    require(raw[4:8] == b"\x00\x00\x00\x00", "archive: gzip mtime is not normalized")

    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        expanded = decompressor.decompress(raw, MAX_EXPANDED_BYTES + 1)
        require(len(expanded) <= MAX_EXPANDED_BYTES, "archive: expanded size limit exceeded")
        require(decompressor.eof, "archive: gzip stream incomplete or exceeds expanded size limit")
        require(not decompressor.unused_data, "archive: trailing or concatenated gzip data rejected")
        tail = decompressor.flush()
    except zlib.error as exc:
        raise VerificationError("archive: invalid gzip stream") from exc
    require(len(expanded) + len(tail) <= MAX_EXPANDED_BYTES, "archive: expanded size limit exceeded")
    return expanded + tail


def _normalize_member_name(name: str) -> str:
    if name == ".":
        return ""
    require(name.startswith("./"), f"archive: non-normalized member name {name!r}")
    value = name[2:]
    require(value and "/" not in value, f"archive: nested/traversal member rejected: {name!r}")
    require(value not in {".", ".."}, f"archive: traversal member rejected: {name!r}")
    return value


def _read_archive(path: Path) -> tuple[dict[str, bytes], str]:
    raw = path.read_bytes()
    archive_digest = sha256_bytes(raw)
    expanded = _decompress_normalized_gzip(raw)

    files: dict[str, bytes] = {}
    saw_root = False
    total_size = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
            for member in archive.getmembers():
                normalized = _normalize_member_name(member.name)
                require(member.uid == 0 and member.gid == 0, "archive: tar owner is not normalized")
                require(int(member.mtime) == 0, "archive: tar mtime is not normalized")
                require(not member.pax_headers, "archive: pax headers rejected")
                if normalized == "":
                    require(member.isdir(), "archive: root member must be a directory")
                    require(not saw_root, "archive: duplicate root member")
                    require((member.mode & 0o777) == 0o700, "archive: root directory mode mismatch")
                    saw_root = True
                    continue

                require(member.isreg(), f"archive: special member rejected: {member.name!r}")
                require((member.mode & 0o777) == 0o600, f"archive: retained file mode mismatch: {normalized}")
                require(normalized not in files, f"archive: duplicate normalized member: {normalized}")
                require(normalized in EXPECTED_FILES, f"archive: unexpected member: {normalized}")
                require(member.size <= MAX_MEMBER_BYTES, f"archive: member too large: {normalized}")
                total_size += member.size
                require(total_size <= MAX_EXPANDED_BYTES, "archive: retained member size limit exceeded")
                handle = archive.extractfile(member)
                require(handle is not None, f"archive: cannot read member: {normalized}")
                data = handle.read(MAX_MEMBER_BYTES + 1)
                require(len(data) == member.size and len(data) <= MAX_MEMBER_BYTES, f"archive: member size mismatch: {normalized}")
                files[normalized] = data
    except (tarfile.TarError, OSError) as exc:
        raise VerificationError("archive: invalid tar stream") from exc

    require(saw_root, "archive: normalized root directory missing")
    require(set(files) == EXPECTED_FILES, "archive: exact PASS allowlist mismatch")
    return files, archive_digest


def _decode_text(data: bytes, context: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(f"{context}: expected UTF-8") from exc


def _parse_kv(data: bytes, expected_keys: set[str], context: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in _decode_text(data, context).splitlines():
        require("=" in raw_line, f"{context}: malformed line")
        key, value = raw_line.split("=", 1)
        require(key and key not in result, f"{context}: duplicate/empty key")
        result[key] = value
    require(set(result) == expected_keys, f"{context}: closed-world key mismatch")
    return result


def _parse_sha_lines(data: bytes, expected_paths: set[str], context: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in _decode_text(data, context).splitlines():
        match = SHA_LINE_RE.fullmatch(line)
        require(match is not None, f"{context}: malformed sha256sum line")
        digest, name = match.groups()
        require(name not in result, f"{context}: duplicate path")
        result[name] = "sha256:" + digest
    require(set(result) == expected_paths, f"{context}: exact path set mismatch")
    return result


def _verify_manifest(files: dict[str, bytes]) -> str:
    seen: dict[str, str] = {}
    for line in _decode_text(files["MANIFEST.sha256"], "manifest").splitlines():
        match = MANIFEST_RE.fullmatch(line)
        require(match is not None, "manifest: malformed line")
        digest, name = match.groups()
        require(name not in seen, "manifest: duplicate file")
        seen[name] = digest
    require(set(seen) == MANIFEST_FILES, "manifest: exact retained-file coverage mismatch")
    for name, expected in seen.items():
        require(hashlib.sha256(files[name]).hexdigest() == expected, f"manifest: digest mismatch for {name}")
    return sha256_bytes(files["MANIFEST.sha256"])


def _verify_phases(data: bytes) -> None:
    require(
        _decode_text(data, "phases").splitlines() == [f"{phase}\tPASS" for phase in PHASES],
        "phases: exact ordered PASS sequence mismatch",
    )


def _verify_tools(data: bytes) -> None:
    value = _decode_text(data, "tools")
    for marker in (
        "rustc_path=",
        "rustc 1.96.0",
        "cargo_path=",
        "cargo 1.96.0",
        "rustfmt_path=",
        "clippy_path=",
        "python_path=",
        "Python 3.11.",
        "nix_path=",
        "nix (Nix) ",
    ):
        require(marker in value, f"tools: missing {marker!r}")


def verify_archive(
    archive_path: Path,
    *,
    expected_archive_sha256: str | None = None,
    expected_head: str | None = None,
    expected_tree: str | None = None,
    release: bool = False,
) -> dict[str, Any]:
    if release:
        require(expected_archive_sha256 is not None, "release: expected archive SHA-256 required")
        require(expected_head is not None, "release: expected Git HEAD required")
        require(expected_tree is not None, "release: expected Git tree required")

    files, archive_digest = _read_archive(archive_path)
    if expected_archive_sha256 is not None:
        require_sha256(expected_archive_sha256, "expected archive digest")
        require(archive_digest == expected_archive_sha256, "archive: external SHA-256 mismatch")

    manifest_digest = _verify_manifest(files)
    _verify_phases(files["PHASES.tsv"])
    require(files["GIT_STATUS_before.txt"] == b"", "source: pre-qualification worktree was not clean")
    require(files["GIT_STATUS_after.txt"] == b"", "source: post-qualification worktree was not clean")

    head = _decode_text(files["GIT_HEAD.txt"], "git head").strip()
    tree = _decode_text(files["GIT_TREE.txt"], "git tree").strip()
    require(GIT_RE.fullmatch(head) is not None, "source: invalid Git HEAD")
    require(GIT_RE.fullmatch(tree) is not None, "source: invalid Git tree")
    if expected_head is not None:
        require(GIT_RE.fullmatch(expected_head) is not None, "expected Git HEAD invalid")
        require(head == expected_head, "source: external Git HEAD mismatch")
    if expected_tree is not None:
        require(GIT_RE.fullmatch(expected_tree) is not None, "expected Git tree invalid")
        require(tree == expected_tree, "source: external Git tree mismatch")

    commit_lines = _decode_text(files["GIT_COMMIT.txt"], "git commit").splitlines()
    require(len(commit_lines) >= 4, "source: incomplete Git commit record")
    require(commit_lines[0] == head and commit_lines[1] == tree, "source: Git commit/head/tree disagreement")

    status = _parse_kv(
        files["STATUS.env"],
        {"PROFILE", "EXECUTION_RESULT", "EXECUTION_EXIT_CODE", "LAST_PHASE", "SOURCE_HEAD", "SOURCE_TREE"},
        "status",
    )
    require(status["PROFILE"] == PROFILE, "status: wrong profile")
    require(status["EXECUTION_RESULT"] == "PASS", "status: execution did not PASS")
    require(status["EXECUTION_EXIT_CODE"] == "0", "status: execution exit was not zero")
    require(status["LAST_PHASE"] == "complete", "status: qualification did not complete")
    require(status["SOURCE_HEAD"] == head and status["SOURCE_TREE"] == tree, "status: source identity mismatch")

    source_locks = _parse_sha_lines(files["SOURCE_LOCKS.sha256"], SOURCE_LOCK_PATHS, "source locks")
    qualifier_files = _parse_sha_lines(files["QUALIFIER_FILES.sha256"], QUALIFIER_PATHS, "qualifier files")
    workflows = _parse_sha_lines(files["FOCUSED_WORKFLOWS.sha256"], WORKFLOW_PATHS, "focused workflows")
    _verify_tools(files["TOOLS.txt"])

    return {
        "profile": VERIFIER_PROFILE,
        "status": "ACCEPTED_PASS_ARCHIVE",
        "archive_sha256": archive_digest,
        "manifest_sha256": manifest_digest,
        "source_head": head,
        "source_tree": tree,
        "source_locks": source_locks,
        "qualifier_files": qualifier_files,
        "focused_workflows": workflows,
        "producer_authenticity_established": False,
        "hosted_ci_agreement_established": False,
        "real_neuroscience_evidence_established": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-tree")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify_archive(
            args.archive,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_head=args.expected_head,
            expected_tree=args.expected_tree,
            release=args.release,
        )
    except (VerificationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
