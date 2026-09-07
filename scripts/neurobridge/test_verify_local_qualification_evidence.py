#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import verify_local_qualification_evidence as verifier


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_lines(paths: set[str]) -> bytes:
    return "".join(
        f"{hashlib.sha256(path.encode()).hexdigest()}  {path}\n"
        for path in sorted(paths)
    ).encode()


def base_files() -> dict[str, bytes]:
    head = "1" * 40
    tree = "2" * 40
    files: dict[str, bytes] = {
        "STATUS.env": (
            f"PROFILE={verifier.PROFILE}\n"
            "EXECUTION_RESULT=PASS\n"
            "EXECUTION_EXIT_CODE=0\n"
            "LAST_PHASE=complete\n"
            f"SOURCE_HEAD={head}\n"
            f"SOURCE_TREE={tree}\n"
        ).encode(),
        "PHASES.tsv": "".join(f"{phase}\tPASS\n" for phase in verifier.PHASES).encode(),
        "GIT_STATUS_before.txt": b"",
        "GIT_STATUS_after.txt": b"",
        "GIT_HEAD.txt": (head + "\n").encode(),
        "GIT_TREE.txt": (tree + "\n").encode(),
        "GIT_COMMIT.txt": (head + "\n" + tree + "\n2026-09-07T00:00:00Z\nsynthetic fixture\n").encode(),
        "SOURCE_LOCKS.sha256": sha_lines(verifier.SOURCE_LOCK_PATHS),
        "QUALIFIER_FILES.sha256": sha_lines(verifier.QUALIFIER_PATHS),
        "FOCUSED_WORKFLOWS.sha256": sha_lines(verifier.WORKFLOW_PATHS),
        "TOOLS.txt": (
            "rustc_path=/nix/store/rustc\n"
            "rustc 1.96.0 (synthetic)\n"
            "cargo_path=/nix/store/cargo\n"
            "cargo 1.96.0 (synthetic)\n"
            "rustfmt_path=/nix/store/rustfmt\n"
            "rustfmt 1.8.0-stable\n"
            "clippy_path=/nix/store/cargo-clippy\n"
            "clippy 0.1.96\n"
            "python_path=/nix/store/python\n"
            "Python 3.11.9\n"
            "nix_path=/nix/store/nix\n"
            "nix (Nix) 2.30.0\n"
            "Linux synthetic\n"
        ).encode(),
    }
    for phase in verifier.PHASES:
        files[f"{phase}.log"] = f"{phase}: synthetic pass\n".encode()
    rebuild_manifest(files)
    return files


def rebuild_manifest(files: dict[str, bytes]) -> None:
    files["MANIFEST.sha256"] = "".join(
        f"{digest(files[name])}  {name}\n"
        for name in sorted(files)
        if name != "MANIFEST.sha256"
    ).encode()


def archive_bytes(
    files: dict[str, bytes],
    *,
    extra_members: list[tarfile.TarInfo] | None = None,
    extra_payloads: list[bytes] | None = None,
    file_mode: int = 0o600,
    root_mtime: int = 0,
    file_mtime: int = 0,
    uid: int = 0,
    gzip_mtime: int = 0,
) -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.GNU_FORMAT) as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        root.mode = 0o700
        root.uid = uid
        root.gid = 0
        root.mtime = root_mtime
        archive.addfile(root)
        for name in sorted(files):
            data = files[name]
            info = tarfile.TarInfo("./" + name)
            info.size = len(data)
            info.mode = file_mode
            info.uid = uid
            info.gid = 0
            info.mtime = file_mtime
            archive.addfile(info, io.BytesIO(data))
        if extra_members:
            payloads = extra_payloads or [b""] * len(extra_members)
            for index, info in enumerate(extra_members):
                archive.addfile(info, io.BytesIO(payloads[index]) if info.isreg() else None)

    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=gzip_mtime) as handle:
        handle.write(tar_buffer.getvalue())
    return output.getvalue()


class VerifierTests(unittest.TestCase):
    def write_archive(self, root: Path, raw: bytes) -> Path:
        path = root / "evidence.tar.gz"
        path.write_bytes(raw)
        return path

    def verify(self, root: Path, files: dict[str, bytes] | None = None, **kwargs):
        payload = base_files() if files is None else files
        return verifier.verify_archive(self.write_archive(root, archive_bytes(payload, **kwargs)))

    def test_valid_pass_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.verify(Path(temp))
            self.assertEqual(result["status"], "ACCEPTED_PASS_ARCHIVE")
            self.assertFalse(result["producer_authenticity_established"])
            self.assertFalse(result["hosted_ci_agreement_established"])

    def test_manifest_content_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = base_files()
            files["rust-core-format.log"] = b"tampered\n"
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(self.write_archive(root, archive_bytes(files)))

    def test_traversal_member_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            info = tarfile.TarInfo("../evil")
            info.size = 1
            info.mode = 0o600
            info.uid = info.gid = info.mtime = 0
            path = self.write_archive(root, archive_bytes(base_files(), extra_members=[info], extra_payloads=[b"x"]))
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(path)

    def test_symlink_member_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = base_files()
            files.pop("rust-core-format.log")
            info = tarfile.TarInfo("./rust-core-format.log")
            info.type = tarfile.SYMTYPE
            info.linkname = "./STATUS.env"
            info.mode = 0o600
            info.uid = info.gid = info.mtime = 0
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(self.write_archive(root, archive_bytes(files, extra_members=[info])))

    def test_duplicate_member_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = base_files()
            info = tarfile.TarInfo("./STATUS.env")
            info.size = len(files["STATUS.env"])
            info.mode = 0o600
            info.uid = info.gid = info.mtime = 0
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(
                    self.write_archive(
                        root,
                        archive_bytes(files, extra_members=[info], extra_payloads=[files["STATUS.env"]]),
                    )
                )

    def test_self_consistent_status_failure_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = base_files()
            files["STATUS.env"] = files["STATUS.env"].replace(b"EXECUTION_RESULT=PASS", b"EXECUTION_RESULT=FAIL")
            rebuild_manifest(files)
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(self.write_archive(root, archive_bytes(files)))

    def test_reordered_phase_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = base_files()
            lines = files["PHASES.tsv"].splitlines()
            lines[0], lines[1] = lines[1], lines[0]
            files["PHASES.tsv"] = b"\n".join(lines) + b"\n"
            rebuild_manifest(files)
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(self.write_archive(root, archive_bytes(files)))

    def test_dirty_post_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = base_files()
            files["GIT_STATUS_after.txt"] = b" M Cargo.lock\n"
            rebuild_manifest(files)
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(self.write_archive(root, archive_bytes(files)))

    def test_release_requires_external_commitments(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_archive(Path(temp), archive_bytes(base_files()))
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(path, release=True)

    def test_release_external_archive_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_archive(Path(temp), archive_bytes(base_files()))
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(
                    path,
                    expected_archive_sha256="sha256:" + "0" * 64,
                    expected_head="1" * 40,
                    expected_tree="2" * 40,
                    release=True,
                )

    def test_nonzero_gzip_mtime_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_archive(Path(temp), archive_bytes(base_files(), gzip_mtime=1))
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(path)

    def test_non_normalized_tar_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_archive(Path(temp), archive_bytes(base_files(), file_mtime=1))
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(path)

    def test_unexpected_file_rejected_even_when_manifest_rehashed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = base_files()
            files["extra.txt"] = b"unexpected\n"
            rebuild_manifest(files)
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_archive(self.write_archive(root, archive_bytes(files)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
