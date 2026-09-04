#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "tpm2_evidence_verifier", HERE / "verify-tpm2-qualification-evidence.py"
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def h(byte: str) -> str:
    return byte * 64


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def base_files() -> dict[str, bytes]:
    files = {name: b"ok\n" for name in verifier.ALLOWED_FILES}
    for name in verifier.MAY_BE_EMPTY:
        files[name] = b""

    files["HEAD"] = ("1" * 40 + "\n").encode()
    files["TREE"] = ("2" * 40 + "\n").encode()
    files["DETACHED_WORKTREE_STATUS.txt"] = b""
    files["RUSTC.txt"] = b"rustc 1.96.0 (fixture)\nrelease: 1.96.0\n"
    files["CARGO.txt"] = b"cargo 1.96.0 (fixture)\n"

    lock = b'version = 4\n\n[[package]]\nname = "fixture"\nversion = "0.1.0"\n'
    files["Cargo.lock.before"] = lock
    files["Cargo.lock.candidate"] = lock
    files["CARGO_LOCK_BEFORE_SHA256.txt"] = f"{sha(lock)}  Cargo.lock\n".encode()
    files["CARGO_LOCK_CANDIDATE_SHA256.txt"] = f"{sha(lock)}  Cargo.lock\n".encode()
    files["CARGO_LOCK_DIFF.patch"] = b""
    files["FLAKE_LOCK_SHA256.txt"] = f"{h('a')}  flake.lock\n".encode()
    files["RUST_TOOLCHAIN_TOML_SHA256.txt"] = f"{h('b')}  rust-toolchain.toml\n".encode()

    locked = {
        "type": "github",
        "owner": "NixOS",
        "repo": "nixpkgs",
        "rev": "3" * 40,
        "narHash": "sha256-fixture",
    }
    metadata = {"locks": {"nodes": {"nixpkgs": {"locked": locked}}}}
    files["FLAKE_METADATA.json"] = (json.dumps(metadata, sort_keys=True) + "\n").encode()
    files["NIXPKGS_LOCKED.json"] = (json.dumps(locked, sort_keys=True) + "\n").encode()

    store = "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-agency-tpm2-verifier"
    quote = store + "/bin/symthaea-tpm2-quote"
    check = store + "/bin/symthaea-tpm2-checkquote"
    files["TPM2_VERIFIER_STORE.txt"] = (store + "\n").encode()
    files["QUOTE_WRAPPER_PATH.txt"] = (quote + "\n").encode()
    files["CHECKQUOTE_WRAPPER_PATH.txt"] = (check + "\n").encode()
    files["QUOTE_WRAPPER_SHA256.txt"] = f"{h('c')}  {quote}\n".encode()
    files["CHECKQUOTE_WRAPPER_SHA256.txt"] = f"{h('d')}  {check}\n".encode()
    files["QUOTE_WRAPPER_ELF.txt"] = b"Elf file type is EXEC\nLOAD 0x0\n"
    files["CHECKQUOTE_WRAPPER_ELF.txt"] = b"Elf file type is EXEC\nLOAD 0x0\n"
    files["TPM2_VERIFIER_REFERENCES.txt"] = (
        b"/nix/store/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb-tpm2-tools\n"
    )
    files["TPM2_WRAPPER_FILE.txt"] = b"statically linked fixture launchers\n"
    files["QUOTE_TCTI_OVERRIDE.stderr"] = b"option override rejected: -T\n"
    files["QUOTE_FORMAT_OVERRIDE.stderr"] = b"option override rejected: -F\n"
    files["CHECK_TCTI_OVERRIDE.stderr"] = b"option override rejected: -T\n"
    files["PROBE_SHA256.txt"] = (
        f"{h('e')}  target/debug/tpm2_attestation_probe\n".encode()
    )

    approved = h("f")
    files["APPROVED_PCR_PROFILE.txt"] = (approved + "\n").encode()
    files["TPM2_VERIFIED.txt"] = (
        "platform_attestation=verified\n"
        f"policy_digest={h('1')}\n"
        f"pcr_profile_digest={approved}\n"
        f"ak_public_digest={h('2')}\n"
        f"challenge_digest={h('3')}\n"
    ).encode()
    files["TPM2_MUTATED.stderr"] = (
        b"fresh TPM quote PCR state is not an approved profile\n"
    )
    files["AK_PUBLIC_SHA256.txt"] = f"{h('4')}  /tmp/akpub.pem\n".encode()

    files["RESULT.txt"] = b"PASS\n"
    files["LAST_PHASE.txt"] = b"complete\n"
    files["EXIT_CODE.txt"] = b"0\n"
    files["CARGO_LOCK_STALE.txt"] = b"0\n"
    files["QUALIFICATION_RESULT.json"] = (
        json.dumps(
            {
                "schema": verifier.PRODUCER_SCHEMA,
                "result": "PASS",
                "last_phase": "complete",
                "exit_code": 0,
                "cargo_lock_stale": False,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()
    files["RUSTFMT_EXIT_CODE.txt"] = b"0\n"

    rebuild_manifest(files)
    return files


def rebuild_manifest(files: dict[str, bytes]) -> None:
    lines = []
    for name in sorted(set(files) - {"MANIFEST.sha256"}):
        lines.append(f"{sha(files[name])}  ./{name}\n")
    files["MANIFEST.sha256"] = "".join(lines).encode()


def write_archive(
    files: dict[str, bytes],
    archive: Path,
    extra_member: tarfile.TarInfo | None = None,
) -> None:
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        root.uid = root.gid = root.mtime = 0
        root.mode = 0o755
        tf.addfile(root)
        for name in sorted(files):
            data = files[name]
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(data)
            info.uid = info.gid = info.mtime = 0
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(data))
        if extra_member is not None:
            extra_member.uid = extra_member.gid = extra_member.mtime = 0
            if extra_member.isfile() and extra_member.size:
                tf.addfile(extra_member, io.BytesIO(b"x" * extra_member.size))
            else:
                tf.addfile(extra_member)
    with archive.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as gz:
            gz.write(raw_tar.getvalue())


class EvidenceVerifierTests(unittest.TestCase):
    def test_complete_pass_fixture_is_accepted(self) -> None:
        files = base_files()
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "evidence.tar.gz"
            write_archive(files, archive)
            snapshot = verifier.read_archive_snapshot(archive)
            self.assertEqual(sha(snapshot), sha(archive.read_bytes()))
            loaded = verifier.load_archive_bytes(snapshot)
            verifier.verify_manifest(loaded)
            head, tree = verifier.verify_status(loaded)
            self.assertEqual(head, "1" * 40)
            self.assertEqual(tree, "2" * 40)
            verifier.verify_lock_evidence(loaded)
            verifier.verify_flake_evidence(loaded)
            tpm = verifier.verify_tpm_evidence(loaded)
            self.assertEqual(tpm["approved_pcr_profile"], h("f"))

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW is Linux/POSIX-specific")
    def test_archive_path_symlink_is_rejected_before_snapshot(self) -> None:
        files = base_files()
        with tempfile.TemporaryDirectory() as td:
            real = Path(td) / "real.tar.gz"
            link = Path(td) / "link.tar.gz"
            write_archive(files, real)
            link.symlink_to(real)
            with self.assertRaises(verifier.EvidenceError):
                verifier.read_archive_snapshot(link)

    def test_bounded_gzip_expansion_is_rejected(self) -> None:
        raw = b"A" * 4096
        encoded = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=encoded, mtime=0) as gz:
            gz.write(raw)
        old_limit = verifier.MAX_TAR_STREAM_BYTES
        verifier.MAX_TAR_STREAM_BYTES = 1024
        try:
            with self.assertRaises(verifier.EvidenceError):
                verifier.bounded_tar_stream(encoded.getvalue())
        finally:
            verifier.MAX_TAR_STREAM_BYTES = old_limit

    def test_manifest_tampering_is_rejected(self) -> None:
        files = base_files()
        files["CARGO_TEST.log"] = b"tampered after manifest\n"
        with self.assertRaises(verifier.EvidenceError):
            verifier.verify_manifest(files)

    def test_path_traversal_member_is_rejected_without_extraction(self) -> None:
        files = base_files()
        extra = tarfile.TarInfo("../escape")
        extra.size = 1
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "traversal.tar.gz"
            write_archive(files, archive, extra)
            with self.assertRaises(verifier.EvidenceError):
                verifier.load_archive(archive)

    def test_symlink_member_is_rejected(self) -> None:
        files = base_files()
        extra = tarfile.TarInfo("./escape-link")
        extra.type = tarfile.SYMTYPE
        extra.linkname = "/etc/passwd"
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "symlink.tar.gz"
            write_archive(files, archive, extra)
            with self.assertRaises(verifier.EvidenceError):
                verifier.load_archive(archive)

    def test_noncanonical_nix_store_path_is_rejected(self) -> None:
        files = base_files()
        files["TPM2_VERIFIER_STORE.txt"] = (
            "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-agency-tpm2-verifier/../evil\n"
        ).encode()
        rebuild_manifest(files)
        with self.assertRaises(verifier.EvidenceError):
            verifier.verify_tpm_evidence(files)

    def test_self_consistent_stale_cargo_candidate_cannot_be_pass(self) -> None:
        files = base_files()
        candidate = (
            files["Cargo.lock.candidate"]
            + b'\n[[package]]\nname = "local-added"\nversion = "0.1.0"\n'
        )
        files["Cargo.lock.candidate"] = candidate
        files["CARGO_LOCK_CANDIDATE_SHA256.txt"] = (
            f"{sha(candidate)}  Cargo.lock\n".encode()
        )
        files["CARGO_LOCK_DIFF.patch"] = b"+ local-added\n"
        rebuild_manifest(files)
        with self.assertRaises(verifier.EvidenceError):
            verifier.verify_lock_evidence(files)

    def test_verified_pcr_must_equal_reviewed_profile(self) -> None:
        files = base_files()
        files["TPM2_VERIFIED.txt"] = files["TPM2_VERIFIED.txt"].replace(
            h("f").encode(), h("e").encode()
        )
        rebuild_manifest(files)
        with self.assertRaises(verifier.EvidenceError):
            verifier.verify_tpm_evidence(files)


if __name__ == "__main__":
    unittest.main()
