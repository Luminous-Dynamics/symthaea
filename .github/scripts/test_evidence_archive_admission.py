#!/usr/bin/env python3
"""Adversarial qualification corpus for evidence_archive_admission.py."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evidence_archive_admission import (  # noqa: E402
    ArchiveAdmissionError,
    admit_archive,
    load_profile,
    _validate_posix_path,
)


class EvidenceArchiveAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.profile_path = self.root / "profile.json"
        self.archive_path = self.root / "evidence.tar.gz"
        self.output_path = self.root / "out"
        self.profile = {
            "schema": "butlin-evidence-archive-admission-profile-v1",
            "profile_id": "test-evidence-v1",
            "root": "evidence",
            "canonical_uid": 0,
            "canonical_gid": 0,
            "canonical_mtime": 0,
            "max_archive_bytes": 1024 * 1024,
            "max_member_bytes": 1024,
            "max_total_unpacked_bytes": 2048,
            "members": [
                {"path": "evidence", "kind": "directory", "mode": 0o755},
                {
                    "path": "evidence/a.json",
                    "kind": "file",
                    "mode": 0o644,
                    "allow_empty": False,
                    "max_bytes": 1024,
                },
                {
                    "path": "evidence/b.txt",
                    "kind": "file",
                    "mode": 0o644,
                    "allow_empty": False,
                    "max_bytes": 1024,
                },
            ],
        }
        self._write_profile()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_profile(self) -> None:
        self.profile_path.write_text(json.dumps(self.profile), encoding="utf-8")

    def _member(
        self,
        name: str,
        *,
        data: bytes | None = None,
        type_: bytes | None = None,
        linkname: str = "",
        mode: int | None = None,
        uid: int = 0,
        gid: int = 0,
        mtime: int = 0,
        pax_headers: dict[str, str] | None = None,
    ) -> tuple[tarfile.TarInfo, bytes | None]:
        info = tarfile.TarInfo(name)
        info.uid = uid
        info.gid = gid
        info.mtime = mtime
        info.pax_headers = {} if pax_headers is None else dict(pax_headers)
        if type_ is None:
            type_ = tarfile.DIRTYPE if data is None else tarfile.REGTYPE
        info.type = type_
        info.linkname = linkname
        if info.isdir():
            info.mode = 0o755 if mode is None else mode
            info.size = 0
            return info, None
        info.mode = 0o644 if mode is None else mode
        payload = b"" if data is None else data
        info.size = len(payload) if info.isfile() else 0
        return info, payload if info.isfile() else None

    def _valid_members(self) -> list[tuple[tarfile.TarInfo, bytes | None]]:
        return [
            self._member("evidence"),
            self._member("evidence/a.json", data=b'{"ok":true}\n'),
            self._member("evidence/b.txt", data=b"qualified\n"),
        ]

    def _write_archive(self, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
        with tarfile.open(self.archive_path, mode="w:gz", format=tarfile.PAX_FORMAT) as tf:
            for info, payload in members:
                tf.addfile(info, None if payload is None else BytesIO(payload))

    def _admit(self) -> dict[str, object]:
        profile = load_profile(self.profile_path)
        return admit_archive(profile, self.archive_path, self.output_path)

    def _assert_rejected(self, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
        self._write_archive(members)
        with self.assertRaises(ArchiveAdmissionError):
            self._admit()

    def test_valid_archive_is_admitted_and_extracted_exactly(self) -> None:
        self._write_archive(self._valid_members())
        result = self._admit()
        self.assertEqual(result["profile_id"], "test-evidence-v1")
        self.assertEqual(result["member_count"], 3)
        self.assertEqual((self.output_path / "evidence/a.json").read_bytes(), b'{"ok":true}\n')
        self.assertEqual((self.output_path / "evidence/b.txt").read_bytes(), b"qualified\n")

    def test_missing_required_member_is_rejected(self) -> None:
        self._assert_rejected(self._valid_members()[:-1])

    def test_unexpected_extra_member_is_rejected(self) -> None:
        members = self._valid_members() + [self._member("evidence/extra", data=b"x")]
        self._assert_rejected(members)

    def test_duplicate_member_path_is_rejected(self) -> None:
        members = self._valid_members() + [self._member("evidence/b.txt", data=b"again")]
        self._assert_rejected(members)

    def test_symlink_substitution_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member(
            "evidence/a.json", type_=tarfile.SYMTYPE, linkname="/etc/passwd"
        )
        self._assert_rejected(members)

    def test_hardlink_substitution_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member(
            "evidence/a.json", type_=tarfile.LNKTYPE, linkname="evidence/b.txt"
        )
        self._assert_rejected(members)

    def test_fifo_substitution_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member("evidence/a.json", type_=tarfile.FIFOTYPE)
        self._assert_rejected(members)

    def test_character_device_substitution_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member("evidence/a.json", type_=tarfile.CHRTYPE)
        self._assert_rejected(members)

    def test_noncanonical_file_mode_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member("evidence/a.json", data=b"x", mode=0o755)
        self._assert_rejected(members)

    def test_noncanonical_uid_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member("evidence/a.json", data=b"x", uid=1000)
        self._assert_rejected(members)

    def test_noncanonical_gid_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member("evidence/a.json", data=b"x", gid=1000)
        self._assert_rejected(members)

    def test_noncanonical_mtime_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member("evidence/a.json", data=b"x", mtime=1)
        self._assert_rejected(members)

    def test_extended_pax_metadata_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member(
            "evidence/a.json", data=b"x", pax_headers={"comment": "unexpected"}
        )
        self._assert_rejected(members)

    def test_zero_length_disallowed_file_is_rejected(self) -> None:
        members = self._valid_members()
        members[1] = self._member("evidence/a.json", data=b"")
        self._assert_rejected(members)

    def test_member_size_bound_is_enforced(self) -> None:
        self.profile["members"][1]["max_bytes"] = 4
        self._write_profile()
        members = self._valid_members()
        members[1] = self._member("evidence/a.json", data=b"12345")
        self._assert_rejected(members)

    def test_total_unpacked_size_bound_is_enforced(self) -> None:
        self.profile["max_total_unpacked_bytes"] = 8
        self._write_profile()
        self._assert_rejected(self._valid_members())

    def test_compressed_archive_size_bound_is_enforced(self) -> None:
        self.profile["max_archive_bytes"] = 1
        self._write_profile()
        self._write_archive(self._valid_members())
        with self.assertRaises(ArchiveAdmissionError):
            self._admit()

    def test_nonempty_output_directory_is_rejected(self) -> None:
        self._write_archive(self._valid_members())
        self.output_path.mkdir()
        (self.output_path / "preexisting").write_text("x", encoding="utf-8")
        with self.assertRaises(ArchiveAdmissionError):
            self._admit()

    def test_profile_duplicate_member_is_rejected(self) -> None:
        self.profile["members"].append(dict(self.profile["members"][1]))
        self._write_profile()
        with self.assertRaises(ArchiveAdmissionError):
            load_profile(self.profile_path)

    def test_profile_member_cannot_escape_root(self) -> None:
        self.profile["members"][1]["path"] = "other/a.json"
        self._write_profile()
        with self.assertRaises(ArchiveAdmissionError):
            load_profile(self.profile_path)

    def test_ambiguous_paths_are_rejected_by_path_contract(self) -> None:
        for path in (
            "/evidence/a",
            "../evidence/a",
            "evidence/../a",
            "evidence/./a",
            "evidence//a",
            "evidence\\a",
        ):
            with self.subTest(path=path), self.assertRaises(ArchiveAdmissionError):
                _validate_posix_path(path, "test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
