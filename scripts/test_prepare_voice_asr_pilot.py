#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for private voice ASR sample preparation using a synthetic ffmpeg stub."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path

from prepare_voice_asr_pilot import _read_json, prepare
from voice_dataset_intake import audit_dataset


class PrepareVoiceAsrPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.dataset = self.base / "dataset"
        (self.dataset / "clips").mkdir(parents=True)
        self.archive = self.base / "cv-fixture.tar.gz"
        self.archive.write_bytes(b"not a real archive; synthetic fixture")
        self.intake_path = self.base / "intake.json"
        self.output = self.base / "private-prepared"
        self.fake_ffmpeg = self.base / "fake-ffmpeg"
        self.fake_ffmpeg.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, wave\n"
            "from pathlib import Path\n"
            "out = Path(sys.argv[-1])\n"
            "with wave.open(str(out), 'wb') as w:\n"
            "    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)\n"
            "    w.writeframes(b'\\x00\\x00' * 160)\n",
            encoding="utf-8",
        )
        self.fake_ffmpeg.chmod(0o700)
        for split, row in (
            ("train", {"client_id":"train-person","path":"train.mp3","sentence":"Goeie dag."}),
            ("dev", {"client_id":"dev-person","path":"dev.mp3","sentence":"Toetsing."}),
            ("test", {"client_id":"test-person","path":"sub/clip.mp3","sentence":"Môre, Suid-Afrika!"}),
        ):
            path=self.dataset/f"{split}.tsv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=["client_id","path","sentence"],delimiter="\t",lineterminator="\n")
                writer.writeheader(); writer.writerow(row)
            audio=self.dataset/"clips"/row["path"]
            audio.parent.mkdir(parents=True,exist_ok=True)
            audio.write_bytes(b"synthetic encoded audio placeholder")
        report=audit_dataset(
            dataset_root=self.dataset,archive=self.archive,locale="af",
            dataset_version="synthetic-fixture-only",dataset_id="cmu-fixture-123",
            source_url="https://mozilladatacollective.com/datasets/cmu-fixture-123",
            license_id="CC0-1.0",usage_restrictions_acknowledged=True,
        )
        self.intake_path.write_text(json.dumps(report),encoding="utf-8")

    def run_prepare(self, **kwargs):
        return prepare(
            dataset_root=kwargs.pop("dataset_root",self.dataset),
            archive=kwargs.pop("archive",self.archive),
            intake_report_path=kwargs.pop("intake_report_path",self.intake_path),
            output_dir=kwargs.pop("output_dir",self.output),
            locale=kwargs.pop("locale","af"),
            ffmpeg=kwargs.pop("ffmpeg",str(self.fake_ffmpeg)),
            **kwargs,
        )

    def test_prepares_16khz_mono_wav_and_private_sample_manifest(self):
        summary=self.run_prepare()
        self.assertEqual(summary["sample_count"],1)
        output=Path(summary["output_dir"])
        self.assertEqual(output.stat().st_mode & 0o777,0o700)
        samples_path=output/"samples.jsonl"
        self.assertEqual(samples_path.stat().st_mode & 0o777,0o600)
        sample=json.loads(samples_path.read_text(encoding="utf-8").strip())
        self.assertEqual(sample["id"],"sub/clip.mp3")
        self.assertIn("Môre, Suid-Afrika!",sample["reference_transcript"])
        self.assertEqual(Path(sample["audio_path"]).stat().st_mode & 0o777,0o600)
        with wave.open(sample["audio_path"],"rb") as wav:
            self.assertEqual((wav.getnchannels(),wav.getsampwidth(),wav.getframerate()),(1,2,16000))
            self.assertEqual(wav.getnframes(),160)
        manifest=json.loads((output/"prepared_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sample_count"],1)
        self.assertIn("preprocessing evidence only",manifest["interpretation"])

    def test_duplicate_keys_in_intake_receipt_are_rejected(self):
        duplicate = self.base / "duplicate-intake.json"
        duplicate.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            _read_json(duplicate)

    def test_source_archive_drift_is_rejected(self):
        self.archive.write_bytes(b"different archive")
        with self.assertRaisesRegex(ValueError,"archive SHA-256 differs"):
            self.run_prepare()

    def test_modified_test_metadata_is_rejected(self):
        with (self.dataset/"test.tsv").open("a",encoding="utf-8") as stream:
            stream.write("another-person\tnew.mp3\tNog 'n sin.\n")
        with self.assertRaisesRegex(ValueError,"test.tsv SHA-256 differs"):
            self.run_prepare()

    def test_existing_output_directory_is_never_overwritten(self):
        self.output.mkdir()
        (self.output/"operator-file.txt").write_text("keep",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"must be new"):
            self.run_prepare()
        self.assertEqual((self.output/"operator-file.txt").read_text(encoding="utf-8"),"keep")

    def test_empty_or_symlinked_source_audio_is_rejected(self):
        (self.dataset/"clips"/"sub"/"clip.mp3").unlink()
        (self.dataset/"clips"/"sub"/"clip.mp3").symlink_to(self.archive)
        with self.assertRaisesRegex(ValueError,"symlink"):
            self.run_prepare()

    def test_invalid_mono_pcm_contract_fails_after_conversion(self):
        self.fake_ffmpeg.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, wave\n"
            "from pathlib import Path\n"
            "with wave.open(sys.argv[-1], 'wb') as w:\n"
            "    w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100)\n"
            "    w.writeframes(b'\\x00\\x00\\x00\\x00' * 8)\n",
            encoding="utf-8",
        )
        self.fake_ffmpeg.chmod(0o700)
        with self.assertRaisesRegex(ValueError,"must be mono"):
            self.run_prepare()

    def test_speaker_or_transcript_values_are_not_written_into_audio_manifest(self):
        self.run_prepare()
        manifest=json.loads((self.output/"prepared_manifest.json").read_text(encoding="utf-8"))
        payload=json.dumps(manifest,ensure_ascii=False)
        self.assertNotIn("Môre, Suid-Afrika!",payload)
        self.assertNotIn("test-person",payload)


if __name__=="__main__":
    unittest.main()
