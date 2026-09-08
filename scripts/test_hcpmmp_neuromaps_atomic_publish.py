#!/usr/bin/env python3
from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import derive_hcpmmp1_neuromaps_lineage_b as lineage


class AtomicNoReplacePublicationTests(unittest.TestCase):
    def test_atomic_noreplace_moves_complete_directory(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            staging = parent / "staging"
            final = parent / "bundle"
            staging.mkdir(mode=0o700)
            (staging / "left.semantic.json").write_text("left")
            (staging / "right.semantic.json").write_text("right")
            (staging / "derivation-evidence.json").write_text("evidence")

            lineage._atomic_rename_noreplace(staging, final)

            self.assertFalse(staging.exists())
            self.assertEqual(
                {path.name for path in final.iterdir()},
                {
                    "left.semantic.json",
                    "right.semantic.json",
                    "derivation-evidence.json",
                },
            )

    def test_atomic_noreplace_refuses_existing_empty_destination(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            staging = parent / "staging"
            final = parent / "bundle"
            staging.mkdir(mode=0o700)
            (staging / "payload").write_text("new")
            final.mkdir(mode=0o700)
            final_inode = final.stat().st_ino

            with self.assertRaises(lineage.DerivationError):
                lineage._atomic_rename_noreplace(staging, final)

            self.assertTrue(staging.exists())
            self.assertEqual(final.stat().st_ino, final_inode)
            self.assertEqual(list(final.iterdir()), [])
            self.assertEqual((staging / "payload").read_text(), "new")

    def test_publish_bundle_closes_check_to_publish_race(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            final = parent / "bundle"
            original = lineage._atomic_rename_noreplace
            raced_inode: int | None = None

            def inject_empty_destination(source: Path, destination: Path) -> None:
                nonlocal raced_inode
                destination.mkdir(mode=0o700)
                raced_inode = destination.stat().st_ino
                original(source, destination)

            with mock.patch.object(
                lineage,
                "_atomic_rename_noreplace",
                side_effect=inject_empty_destination,
            ):
                with self.assertRaises(lineage.DerivationError):
                    lineage._publish_bundle(
                        final,
                        {"hemisphere": "left"},
                        {"hemisphere": "right"},
                        {"schema": "test-evidence"},
                    )

            self.assertIsNotNone(raced_inode)
            self.assertTrue(final.is_dir())
            self.assertEqual(final.stat().st_ino, raced_inode)
            self.assertEqual(list(final.iterdir()), [])
            self.assertFalse((parent / ".bundle.publish-lock").exists())
            self.assertEqual(list(parent.glob(".bundle.staging-*")), [])

    def test_publish_bundle_retains_complete_private_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            final = parent / "bundle"
            evidence = lineage._publish_bundle(
                final,
                {"hemisphere": "left"},
                {"hemisphere": "right"},
                {"schema": "test-evidence"},
            )

            self.assertEqual(evidence["schema"], "test-evidence")
            self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o700)
            expected = {
                "left.semantic.json",
                "right.semantic.json",
                "derivation-evidence.json",
            }
            self.assertEqual({path.name for path in final.iterdir()}, expected)
            for name in expected:
                self.assertEqual(stat.S_IMODE((final / name).stat().st_mode), 0o600)
            self.assertFalse((parent / ".bundle.publish-lock").exists())
            self.assertEqual(list(parent.glob(".bundle.staging-*")), [])

    def test_missing_renameat2_fails_closed_without_publish(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            staging = parent / "staging"
            final = parent / "bundle"
            staging.mkdir(mode=0o700)
            (staging / "payload").write_text("new")

            with mock.patch.object(lineage.ctypes, "CDLL", return_value=object()):
                with self.assertRaises(lineage.DerivationError):
                    lineage._atomic_rename_noreplace(staging, final)

            self.assertTrue(staging.exists())
            self.assertFalse(final.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
