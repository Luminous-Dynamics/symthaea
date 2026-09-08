#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import struct
import tempfile
import unittest
from pathlib import Path

import workbench_root_nar_membership as nar


def s(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value + b"\x00" * ((-len(value)) % 8)


def regular(contents: bytes, *, executable: bool = False, field: bytes = b"contents") -> bytes:
    body = s(b"(") + s(b"type") + s(b"regular")
    if executable:
        body += s(b"executable") + s(b"")
    body += s(field) + s(contents) + s(b")")
    return body


def symlink(target: bytes) -> bytes:
    return s(b"(") + s(b"type") + s(b"symlink") + s(b"target") + s(target) + s(b")")


def directory(entries: list[tuple[bytes, bytes]]) -> bytes:
    body = s(b"(") + s(b"type") + s(b"directory")
    for name, node in entries:
        body += s(b"entry") + s(b"(") + s(b"name") + s(name) + s(b"node") + node + s(b")")
    return body + s(b")")


def archive(root_node: bytes) -> bytes:
    return s(nar.MAGIC) + root_node


def workbench_archive(contents: bytes = b"#!/bin/sh\necho wb\n", *, executable: bool = True) -> bytes:
    return archive(directory([
        (b"bin", directory([(b"wb_command", regular(contents, executable=executable))])),
        (b"share", directory([])),
    ]))


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def verify_bytes(td: Path, data: bytes, *, expected: str | None = None, target: str = "bin/wb_command"):
    path = td / "root.nar"
    path.write_bytes(data)
    return nar.verify_membership(path, expected or digest(data), target)


class RootNarMembershipContracts(unittest.TestCase):
    def test_valid_executable_regular_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = workbench_archive()
            result = verify_bytes(Path(tmp), data)
            self.assertEqual(result["status"], "verified-nar-membership-only")
            self.assertEqual(result["target"]["node_type"], "regular")
            self.assertTrue(result["target"]["executable"])
            self.assertEqual(result["target"]["content_sha256"], digest(b"#!/bin/sh\necho wb\n"))
            self.assertTrue(result["authority"]["target_membership_verified"])
            self.assertFalse(result["authority"]["workbench_execution_qualified"])

    def test_non_executable_regular_member_is_reported_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_bytes(Path(tmp), workbench_archive(executable=False))
            self.assertEqual(result["target"]["node_type"], "regular")
            self.assertFalse(result["target"]["executable"])

    def test_symlink_member_is_reported_without_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = archive(directory([(b"bin", directory([(b"wb_command", symlink(b".wb_command-wrapped"))]))]))
            result = verify_bytes(Path(tmp), data)
            self.assertEqual(result["target"]["node_type"], "symlink")
            self.assertIsNone(result["target"]["content_sha256"])
            self.assertEqual(result["target"]["symlink_target_hex"], b".wb_command-wrapped".hex())

    def test_hash_mismatch_rejected_before_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = workbench_archive()
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, expected="sha256:" + "0" * 64)

    def test_noncanonical_expected_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = workbench_archive()
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, expected="sha256-AAAA")

    def test_wrong_magic_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = s(b"not-a-nar") + directory([])
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, target="x")

    def test_nonzero_padding_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = bytearray(workbench_archive())
            # Magic is 13 bytes and therefore has three zero padding bytes.
            data[8 + len(nar.MAGIC)] = 1
            raw = bytes(data)
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), raw)

    def test_unsorted_directory_entries_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = archive(directory([(b"z", regular(b"z")), (b"a", regular(b"a"))]))
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, target="z")

    def test_duplicate_directory_entries_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = archive(directory([(b"x", regular(b"1")), (b"x", regular(b"2"))]))
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, target="x")

    def test_dot_directory_entry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = archive(directory([(b".", regular(b"x"))]))
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, target="x")

    def test_slash_directory_entry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = archive(directory([(b"a/b", regular(b"x"))]))
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, target="a/b")

    def test_absolute_target_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), workbench_archive(), target="/bin/wb_command")

    def test_parent_traversal_target_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), workbench_archive(), target="bin/../wb_command")

    def test_missing_target_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), workbench_archive(), target="bin/not-there")

    def test_trailing_bytes_rejected_even_when_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = workbench_archive() + b"junk"
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data)

    def test_unknown_node_type_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = archive(s(b"(") + s(b"type") + s(b"socket") + s(b")"))
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, target="x")

    def test_regular_missing_contents_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = archive(directory([(b"x", regular(b"payload", field=b"payload"))]))
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, target="x")

    def test_truncated_regular_contents_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = workbench_archive()[:-20]
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data)

    def test_oversized_structural_string_rejected_without_large_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Correct magic, then a claimed node-open token beyond the v1 structural bound.
            data = s(nar.MAGIC) + struct.pack("<Q", nar.MAX_STRUCTURAL_STRING + 1)
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, target="x")

    def test_content_change_changes_both_nar_and_file_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            a = workbench_archive(b"A", executable=True)
            b = workbench_archive(b"B", executable=True)
            ra = verify_bytes(td, a)
            rb = verify_bytes(td, b)
            self.assertNotEqual(digest(a), digest(b))
            self.assertNotEqual(ra["target"]["content_sha256"], rb["target"]["content_sha256"])

    def test_execute_bit_changes_nar_but_not_file_contents_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            contents = b"same bytes"
            a = workbench_archive(contents, executable=True)
            b = workbench_archive(contents, executable=False)
            ra = verify_bytes(td, a)
            rb = verify_bytes(td, b)
            self.assertNotEqual(digest(a), digest(b))
            self.assertEqual(ra["target"]["content_sha256"], rb["target"]["content_sha256"])
            self.assertNotEqual(ra["target"]["executable"], rb["target"]["executable"])

    def test_empty_directory_parses_but_missing_target_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = archive(directory([]))
            with self.assertRaises(nar.NarError):
                verify_bytes(Path(tmp), data, target="bin/wb_command")

    def test_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            data = workbench_archive()
            path = td / "root.nar"
            path.write_bytes(data)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = nar.main(["--nar", str(path), "--expected-nar-sha256", digest(data)])
            self.assertEqual(rc, 0)
            self.assertIn('"target_membership_verified":true', out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
