#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import collect_ci_changed_paths as collect


class ChangedPathContracts(unittest.TestCase):
    def test_empty_stream_is_empty_diff(self):
        self.assertEqual(collect.parse_name_only_z(b""), [])

    def test_single_path(self):
        self.assertEqual(collect.parse_name_only_z(b"README.md\0"), ["README.md"])

    def test_paths_are_sorted_deterministically(self):
        self.assertEqual(
            collect.parse_name_only_z(b"z.txt\0a.txt\0m.txt\0"),
            ["a.txt", "m.txt", "z.txt"],
        )

    def test_rename_disabled_shape_preserves_old_and_new_names(self):
        self.assertEqual(
            collect.parse_name_only_z(b"old/path.py\0new/path.py\0"),
            ["new/path.py", "old/path.py"],
        )

    def test_missing_terminal_nul_rejected(self):
        with self.assertRaises(collect.ChangedPathError):
            collect.parse_name_only_z(b"README.md")

    def test_embedded_empty_path_rejected(self):
        with self.assertRaises(collect.ChangedPathError):
            collect.parse_name_only_z(b"a\0\0b\0")

    def test_duplicate_path_rejected(self):
        with self.assertRaises(collect.ChangedPathError):
            collect.parse_name_only_z(b"a\0a\0")

    def test_non_utf8_path_rejected(self):
        with self.assertRaises(collect.ChangedPathError):
            collect.parse_name_only_z(b"\xff\0")

    def test_control_characters_are_preserved_for_registry_rejection(self):
        self.assertEqual(collect.parse_name_only_z(b"a\nb\0"), ["a\nb"])

    def test_output_is_create_only_and_canonical_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "changed.z"
            output = root / "changed.json"
            source.write_bytes(b"b\0a\0")
            self.assertEqual(
                collect.main(["--input", str(source), "--output", str(output)]),
                0,
            )
            self.assertEqual(output.read_bytes(), b'["a","b"]\n')
            self.assertEqual(json.loads(output.read_text()), ["a", "b"])
            self.assertEqual(
                collect.main(["--input", str(source), "--output", str(output)]),
                2,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
