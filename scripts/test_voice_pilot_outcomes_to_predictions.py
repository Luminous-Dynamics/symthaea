#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic tests for converting local Symthaea pilot outcomes to private predictions."""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from voice_dataset_intake import audit_dataset
from voice_asr_score import sha256_file
from voice_pilot_outcomes_to_predictions import (
    OUTCOME_KEYS,
    build_predictions,
    write_private_exclusive,
)


class OutcomesToPredictionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.dataset = self.base / "dataset"
        (self.dataset / "clips").mkdir(parents=True)
        self.archive = self.base / "cv-af.tar.gz"
        self.archive.write_bytes(b"synthetic dataset archive")
        self.intake_path = self.base / "intake.json"
        self.model_root = self.base / "model"
        self.model_root.mkdir()
        self.artifact = self.model_root / "model.bin"
        self.artifact.write_bytes(b"synthetic converted weights")
        self.runtime_binary = self.base / "whisper-cli"
        self.runtime_binary.write_bytes(b"synthetic whisper executable")
        self.outcomes_path = self.base / "outcomes.jsonl"
        self.output = self.base / "private-predictions.json"

        for split, row in (
            ("train", {"client_id":"train-person","path":"train.mp3","sentence":"Goeie."}),
            ("dev", {"client_id":"dev-person","path":"dev.mp3","sentence":"Toets."}),
            ("test", {"client_id":"test-person","path":"nested/clip.mp3","sentence":"Môre!"}),
        ):
            with (self.dataset / f"{split}.tsv").open("w",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=["client_id","path","sentence"],delimiter="\t",lineterminator="\n")
                writer.writeheader(); writer.writerow(row)
            audio=self.dataset/"clips"/row["path"]
            audio.parent.mkdir(parents=True,exist_ok=True)
            audio.write_bytes(b"synthetic audio")

        self.intake = audit_dataset(
            dataset_root=self.dataset, archive=self.archive, locale="af",
            dataset_version="cv-27-test-fixture", dataset_id="cmu-fixture-123",
            source_url="https://mozilladatacollective.com/datasets/cmu-fixture-123",
            license_id="CC0-1.0", usage_restrictions_acknowledged=True,
        )
        self.intake_path.write_text(json.dumps(self.intake),encoding="utf-8")
        self.model_revision="a"*40
        self.runtime_commit="b"*40
        self.converter_commit="c"*40
        self.outcome = {
            "id":"nested/clip.mp3", "language":"af", "detected_language":None,
            "language_confidence":0.0, "transcript":"Môre", "word_error_rate":0.0,
            "character_error_rate":0.0, "entity_preservation":1.0,
            "translation_en":None, "translation_chrf":None,
            "response_language_correct":None, "latency_ms":12.5, "error":None,
        }
        self.outcomes_path.write_text(json.dumps(self.outcome,ensure_ascii=False)+"\n",encoding="utf-8")

    def build(self, **kwargs):
        return build_predictions(
            outcomes_path=kwargs.pop("outcomes_path",self.outcomes_path),
            intake_report_path=self.intake_path,
            intake_report=self.intake,
            dataset_root=kwargs.pop("dataset_root",self.dataset),
            archive=kwargs.pop("archive",self.archive),
            model_root=kwargs.pop("model_root",self.model_root),
            runtime_binary=kwargs.pop("runtime_binary",self.runtime_binary),
            locale=kwargs.pop("locale","af"),
            model_id=kwargs.pop("model_id","BuzzASR/afrikaans"),
            model_revision=kwargs.pop("model_revision",self.model_revision),
            model_license=kwargs.pop("model_license","MIT"),
            runtime_commit=kwargs.pop("runtime_commit",self.runtime_commit),
            converter_commit=kwargs.pop("converter_commit",self.converter_commit),
            artifact_path=kwargs.pop("artifact_path","model.bin"),
            quantization=kwargs.pop("quantization","f16"),
            mode=kwargs.pop("mode","smoke"),
            **kwargs,
        )

    def test_valid_pack_pins_every_artifact_and_dataset_input(self):
        doc=self.build()
        self.assertEqual(set(doc),{
            "schema_version","locale","dataset_id","dataset_version","archive_sha256",
            "test_metadata_sha256","model_id","model_revision","model_license",
            "outcomes_sha256","runtime","predictions",
        })
        self.assertEqual(doc["runtime"]["kind"],"whisper.cpp")
        self.assertEqual(doc["runtime"]["artifact_sha256"],sha256_file(self.artifact))
        self.assertEqual(doc["runtime"]["runtime_binary_sha256"],sha256_file(self.runtime_binary))
        self.assertEqual(doc["predictions"],[{"path":"nested/clip.mp3","text":"Môre"}])
        self.assertEqual(len(doc["outcomes_sha256"]),64)

    def test_empty_transcript_is_preserved_not_discarded(self):
        self.outcome["transcript"]=""
        self.outcomes_path.write_text(json.dumps(self.outcome)+"\n",encoding="utf-8")
        self.assertEqual(self.build()["predictions"],[{"path":"nested/clip.mp3","text":""}])

    def test_error_outcome_blocks_pack(self):
        self.outcome["error"]="worker timeout"
        self.outcomes_path.write_text(json.dumps(self.outcome)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"records a worker/inference error"):
            self.build()

    def test_missing_outcome_blocks_pack(self):
        self.outcomes_path.write_text("",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"contains no rows"):
            self.build()

    def test_duplicate_outcome_blocks_pack(self):
        self.outcomes_path.write_text(
            json.dumps(self.outcome)+"\n"+json.dumps(self.outcome)+"\n",encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError,"duplicate outcome"):
            self.build()

    def test_unknown_clip_blocks_pack(self):
        self.outcome["id"]="unknown.mp3"
        self.outcomes_path.write_text(json.dumps(self.outcome)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"unknown test clip"):
            self.build()

    def test_outcome_schema_drift_blocks_pack(self):
        row=copy.deepcopy(self.outcome); row["reference_transcript"]="should not be allowed"
        self.outcomes_path.write_text(json.dumps(row)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"unexpected fields"):
            self.build()

    def test_changed_archive_fails_against_intake(self):
        self.archive.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError,"archive SHA-256 differs"):
            self.build()

    def test_model_and_runtime_revisions_must_be_pinned(self):
        with self.assertRaisesRegex(ValueError,"model_revision must be a 40-hex"):
            self.build(model_revision="main")
        with self.assertRaisesRegex(ValueError,"runtime_commit must be a 40-hex"):
            self.build(runtime_commit="main")

    def test_artifact_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"traversal components"):
            self.build(artifact_path="../other.bin")

    def test_private_writer_is_exclusive_and_does_not_overwrite(self):
        doc=self.build()
        write_private_exclusive(self.output,doc,[self.dataset,self.model_root,self.outcomes_path.parent])
        self.assertEqual(self.output.stat().st_mode & 0o777,0o600)
        with self.assertRaisesRegex(ValueError,"must be new"):
            write_private_exclusive(self.output,doc,[self.dataset,self.model_root,self.outcomes_path.parent])


if __name__=="__main__":
    unittest.main()
