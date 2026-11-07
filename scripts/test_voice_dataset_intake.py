#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free tests for local voice dataset intake and split integrity."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from voice_dataset_intake import audit_dataset, write_report_exclusive


class VoiceDatasetIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        self.root.mkdir()
        (self.root / "clips").mkdir()
        self.archive = self.base / "common-voice-test.tar.gz"
        self.archive.write_bytes(b"synthetic archive fixture; never a production corpus")
        self.source_args = {
            "dataset_root": self.root,
            "archive": self.archive,
            "locale": "af",
            "dataset_version": "synthetic-fixture-only",
            "dataset_id": "cmu-fixture-123",
            "source_url": "https://mozilladatacollective.com/datasets/cmu-fixture-123",
            "license_id": "CC0-1.0",
            "usage_restrictions_acknowledged": True,
        }
        self.write_split("train", [{"client_id":"speaker-a","path":"a.mp3","sentence":"Goeie môre."}])
        self.write_split("dev", [{"client_id":"speaker-b","path":"b.mp3","sentence":"Dit is 'n toets."}])
        self.write_split("test", [{"client_id":"speaker-c","path":"c.mp3","sentence":"Ons toets spraak."}])

    def write_split(self, split: str, rows: list[dict[str, str]]) -> None:
        path = self.root / f"{split}.tsv"
        fields = ["client_id", "path", "sentence"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            audio = self.root / "clips" / row["path"]
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"synthetic-audio-fixture")

    def audit(self, **kwargs):
        return audit_dataset(**(self.source_args | kwargs))

    def test_ingestion_smoke_reports_digests_and_counts_not_raw_speaker_tokens(self):
        report = self.audit()
        self.assertEqual(report["mode"], "ingestion-smoke")
        self.assertEqual(report["language_tag"], "af")
        self.assertEqual(report["split_audits"]["train"]["rows"], 1)
        self.assertEqual(report["dataset"]["archive_sha256"],
                         hashlib.sha256(self.archive.read_bytes()).hexdigest())
        serialized = json.dumps(report)
        self.assertNotIn("speaker-a", serialized)
        self.assertNotIn("Goeie môre", serialized)
        self.assertIn("does not demonstrate ASR accuracy", report["interpretation"])

    def test_heldout_evaluation_accepts_disjoint_synthetic_fixture_when_minima_are_met(self):
        report = self.audit(
            mode="heldout-evaluation", minimum_test_clips=1, minimum_test_speakers=1
        )
        self.assertTrue(report["integrity"]["speaker_disjoint"])
        self.assertIn("no model quality is implied", report["interpretation"])

    def test_heldout_evaluation_rejects_speaker_overlap_without_disclosing_tokens(self):
        self.write_split("test", [{"client_id":"speaker-a","path":"c.mp3","sentence":"Ons toets spraak."}])
        with self.assertRaisesRegex(ValueError, "requires speaker-disjoint splits"):
            self.audit(mode="heldout-evaluation", minimum_test_clips=1, minimum_test_speakers=1)

    def test_small_test_partition_fails_configured_evaluation_floor(self):
        with self.assertRaisesRegex(ValueError, "minimum is 2"):
            self.audit(
                mode="heldout-evaluation", minimum_test_clips=2, minimum_test_speakers=1
            )

    def test_heldout_evaluation_rejects_missing_speaker_tokens(self):
        self.write_split("dev", [{"client_id":"","path":"b.mp3","sentence":"Dit is 'n toets."}])
        with self.assertRaisesRegex(ValueError, "requires a non-empty pseudonymous client_id"):
            self.audit(mode="heldout-evaluation", minimum_test_clips=1, minimum_test_speakers=1)

    def test_path_traversal_is_rejected(self):
        self.write_split("train", [{"client_id":"speaker-a","path":"../outside.mp3","sentence":"Goeie môre."}])
        with self.assertRaisesRegex(ValueError, "traversal components"):
            self.audit()

    def test_duplicate_clip_paths_across_splits_are_rejected(self):
        self.write_split("test", [{"client_id":"speaker-c","path":"a.mp3","sentence":"Ons toets spraak."}])
        with self.assertRaisesRegex(ValueError, "audio path appears in both"):
            self.audit()

    def test_missing_audio_file_is_rejected_for_smoke_by_default(self):
        (self.root / "clips" / "c.mp3").unlink()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.audit()

    def test_symlinked_archive_is_rejected(self):
        alias = self.base / "archive-link.tar.gz"
        alias.symlink_to(self.archive)
        with self.assertRaisesRegex(ValueError, "archive must be an existing regular file"):
            self.audit(archive=alias)

    def test_symlinked_clips_root_is_rejected(self):
        external = self.base / "external-clips"
        (self.root / "clips").rename(external)
        (self.root / "clips").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "root must not be a symlink"):
            self.audit()

    def test_non_mozilla_url_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mozilla Data Collective URL"):
            self.audit(source_url="https://example.com/dataset")

    def test_usage_restrictions_must_be_acknowledged(self):
        with self.assertRaisesRegex(ValueError, "acknowledge Common Voice restrictions"):
            self.audit(usage_restrictions_acknowledged=False)

    def test_metadata_only_mode_is_explicit_in_report(self):
        report = self.audit(check_audio=False)
        self.assertEqual(report["split_audits"]["test"]["audio_files_checked"], 0)

    def test_report_writer_refuses_overwrite(self):
        report = self.audit()
        path = self.base / "report.json"
        write_report_exclusive(path, report)
        with self.assertRaisesRegex(ValueError, "must be new"):
            write_report_exclusive(path, report)


if __name__ == "__main__":
    unittest.main()
