#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic tests for provenance-checked voice ASR scoring; no real corpus is used."""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from voice_asr_score import (
    NORMALIZATION_ID,
    _hash_tree,
    _read_json,
    normalize_text,
    score_predictions,
    sha256_file,
    write_report_exclusive,
)
from voice_dataset_intake import audit_dataset


class VoiceAsrScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "dataset"
        (self.root / "clips").mkdir(parents=True)
        self.archive = self.base / "cv-af.tar.gz"
        self.archive.write_bytes(b"synthetic test archive")
        self.model_root = self.base / "model-export"
        self.model_root.mkdir()
        (self.model_root / "config.json").write_text('{"synthetic": true}\n', encoding="utf-8")
        self.source_model_root = self.base / "upstream-model-snapshot"
        self.source_model_root.mkdir()
        (self.source_model_root / "config.json").write_text('{"synthetic-source": true}\n', encoding="utf-8")
        (self.source_model_root / "weights.safetensors").write_bytes(b"synthetic upstream weights")
        (self.model_root / "model.bin").write_bytes(b"synthetic converted model artifact")
        self.runtime_binary = self.base / "whisper-cli"
        self.runtime_binary.write_bytes(b"synthetic whisper runtime executable")
        self.worker_root = self.base / "worker-source"
        self.worker_root.mkdir()
        self.runtime_worker = self.worker_root / "voice_transformers_worker.py"
        self.runtime_worker.write_text("# synthetic test worker\\n", encoding="utf-8")
        self.runtime_metadata_log = self.base / "transformers-worker-stderr.log"
        model_sha, model_count, model_bytes = _hash_tree(self.source_model_root)
        self.runtime_metadata = {
            "python_version": "3.12.1",
            "torch_version": "2.8.0",
            "transformers_version": "4.56.0",
            "numpy_version": "2.1.0",
            "device": "cpu",
            "dtype": "float32",
            "offline": True,
            "model_snapshot_manifest_sha256": model_sha,
            "model_snapshot_file_count": model_count,
            "model_snapshot_bytes": model_bytes,
            "python_executable_sha256": sha256_file(self.runtime_binary),
            "worker_script_sha256": sha256_file(self.runtime_worker),
        }
        self.runtime_metadata_log.write_text(
            "startup noise\\nSYMTHEA_TRANSFORMERS_RUNTIME "
            + json.dumps(self.runtime_metadata, sort_keys=True) + "\\n",
            encoding="utf-8",
        )
        self.runtime = {
            "kind": "whisper.cpp",
            "runtime_commit": "d" * 40,
            "converter_commit": "e" * 40,
            "runtime_binary_sha256": sha256_file(self.runtime_binary),
            "artifact_path": "model.bin",
            "artifact_sha256": sha256_file(self.model_root / "model.bin"),
            "quantization": "f16",
        }
        self.model_id = "BuzzASR/afrikaans"
        self.model_revision = "a" * 40
        self.model_license = "MIT"
        self.locale = "af"
        self.intake_path = self.base / "intake.json"
        self.predictions_path = self.base / "predictions.json"
        self.outcomes_path = self.base / "outcomes.jsonl"
        self.outcomes_path.write_text("synthetic fixture outcomes\\n",encoding="utf-8")
        self.report_path = self.base / "report.json"
        self._write_dataset(
            [{"client_id":"train-speaker","path":"train.mp3","sentence":"Opleiding."}],
            [{"client_id":"dev-speaker","path":"dev.mp3","sentence":"Toetsing."}],
            [
                {"client_id":"speaker-a","path":"clip-001.mp3","sentence":"Goeie môre!"},
                {"client_id":"speaker-b","path":"clip-002.mp3","sentence":"Ek weet dit nie."},
                {"client_id":"speaker-c","path":"clip-003.mp3","sentence":"Die stem is mooi."},
            ],
        )
        self._build_intake("ingestion-smoke")
        self._write_predictions()

    def _write_dataset(self, train, dev, test) -> None:
        for split, rows in (("train", train), ("dev", dev), ("test", test)):
            path = self.root / f"{split}.tsv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["client_id", "path", "sentence"],
                                        delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            for row in rows:
                audio = self.root / "clips" / row["path"]
                audio.write_bytes(b"fake-audio-content")

    def _build_intake(self, mode: str) -> dict:
        self.intake = audit_dataset(
            dataset_root=self.root,
            archive=self.archive,
            locale=self.locale,
            dataset_version="cv-corpus-27.0-test-fixture",
            dataset_id="cmu-test-123",
            source_url="https://mozilladatacollective.com/datasets/cmu-test-123",
            license_id="CC0-1.0",
            usage_restrictions_acknowledged=True,
            mode=mode,
            minimum_test_clips=1 if mode == "heldout-evaluation" else 100,
            minimum_test_speakers=1 if mode == "heldout-evaluation" else 10,
        )
        self.intake_path.write_text(json.dumps(self.intake, ensure_ascii=False), encoding="utf-8")
        return self.intake

    def _prediction_doc(self, rows=None) -> dict:
        test_rows = list(csv.DictReader((self.root / "test.tsv").open(encoding="utf-8"), delimiter="\t"))
        hypotheses = [
            {"path": row["path"], "text": {
                "clip-001.mp3": "Goeie môre.",
                "clip-002.mp3": "Ek weet",
                "clip-003.mp3": "Die stem is mooi!",
            }.get(row["path"], row["sentence"])}
            for row in test_rows
        ] if rows is None else rows
        return {
            "schema_version": 1,
            "locale": self.locale,
            "dataset_id": self.intake["dataset"]["id"],
            "dataset_version": self.intake["dataset"]["version"],
            "archive_sha256": self.intake["dataset"]["archive_sha256"],
            "test_metadata_sha256": self.intake["split_audits"]["test"]["metadata_sha256"],
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_license": self.model_license,
            "source_model_snapshot_sha256": _hash_tree(self.source_model_root)[0],
            "outcomes_sha256": sha256_file(self.outcomes_path),
            "runtime": copy.deepcopy(self.runtime),
            "predictions": hypotheses,
        }

    def _write_outcomes(self, predictions) -> None:
        rows = []
        for index, prediction in enumerate(predictions):
            rows.append({
                "id": prediction["path"],
                "language": self.locale,
                "detected_language": None,
                "language_confidence": 0.0,
                "transcript": prediction["text"],
                "word_error_rate": 0.0,
                "character_error_rate": 0.0,
                "entity_preservation": 1.0,
                "translation_en": None,
                "translation_chrf": None,
                "response_language_correct": None,
                "latency_ms": 12.5 + index,
                "preflight_latency_ms": 1.5 if index == 0 else 0.25,
                "transcription_latency_ms": 10.0 + index,
                "translation_latency_ms": None,
                "error": None,
            })
        self.outcomes_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_predictions(self, rows=None) -> dict:
        self.pred_doc = self._prediction_doc(rows)
        self._write_outcomes(self.pred_doc["predictions"])
        self.pred_doc["outcomes_sha256"] = sha256_file(self.outcomes_path)
        self.predictions_path.write_text(json.dumps(self.pred_doc, ensure_ascii=False), encoding="utf-8")
        return self.pred_doc

    def score(self, **kwargs):
        return score_predictions(
            prediction_doc=kwargs.pop("prediction_doc", self.pred_doc),
            intake_report=kwargs.pop("intake_report", self.intake),
            dataset_root=kwargs.pop("dataset_root", self.root),
            archive=kwargs.pop("archive", self.archive),
            model_root=kwargs.pop("model_root", self.model_root),
            source_model_root=kwargs.pop("source_model_root", self.source_model_root),
            predictions_path=kwargs.pop("predictions_path", self.predictions_path),
            outcomes_path=kwargs.pop("outcomes_path", self.outcomes_path),
            model_id=kwargs.pop("model_id", self.model_id),
            model_revision=kwargs.pop("model_revision", self.model_revision),
            model_license=kwargs.pop("model_license", self.model_license),
            locale=kwargs.pop("locale", self.locale),
            intake_report_path=kwargs.pop("intake_report_path", self.intake_path),
            runtime_binary_path=kwargs.pop("runtime_binary_path", self.runtime_binary),
            runtime_metadata_log_path=kwargs.pop("runtime_metadata_log_path", self.runtime_metadata_log),
            mode=kwargs.pop("mode", "smoke"),
            bootstrap_iterations=kwargs.pop("bootstrap_iterations", 100),
            **kwargs,
        )

    def test_smoke_scores_micro_wer_cer_and_deterministic_intervals(self):
        report = self.score()
        self.assertEqual(report["normalization"], NORMALIZATION_ID)
        self.assertEqual(len(report["scorer"]["script_sha256"]), 64)
        self.assertRegex(report["scorer"]["python_version"], r"^\\d+\\.\\d+")
        self.assertEqual(report["metrics"]["sample_count"], 3)
        self.assertEqual(report["speaker_macro_metrics"]["speaker_count"], 3)
        self.assertGreaterEqual(report["speaker_macro_metrics"]["mean_speaker_wer"], 0.0)
        self.assertAlmostEqual(report["metrics"]["wer"], 2 / 10)
        self.assertLessEqual(report["confidence_intervals_95"]["wer"]["low_95"],
                             report["confidence_intervals_95"]["wer"]["high_95"])
        self.assertEqual(report["model"]["converted_export_file_count"], 2)
        self.assertEqual(report["model"]["source_snapshot_file_count"], 2)
        self.assertEqual(report["runtime"]["kind"], "whisper.cpp")
        self.assertEqual(report["runtime"]["quantization"], "f16")
        self.assertEqual(report["inputs"]["outcomes_sha256"], sha256_file(self.outcomes_path))
        self.assertEqual(report["latency"]["end_to_end_ms"]["sample_count"], 3)
        self.assertAlmostEqual(report["latency"]["transcription_ms"]["mean_ms"], 11.0)
        self.assertEqual(report["latency"]["first_sample_preflight_ms"], 1.5)
        self.assertEqual(report["latency"]["translation_ms"]["sample_count"], 0)
        self.assertIn("may include first-use model startup", report["latency"]["interpretation"])
        self.assertEqual(report["confidence_intervals_95"]["resampling_unit"], "pseudonymous speaker clusters")
        self.assertEqual(report["confidence_intervals_95"]["speaker_cluster_count"], 3)
        self.assertEqual(report["mode"], "smoke")
        self.assertIn("exploratory smoke metric only", report["interpretation"])
        self.assertEqual(report["training_data_overlap_status"], "not_assessed")

    def test_transformers_runtime_scores_against_the_exact_native_model_snapshot(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["runtime"] = {
            "kind": "transformers",
            "python_executable_sha256": sha256_file(self.runtime_binary),
            "worker_script_sha256": sha256_file(self.runtime_worker),
            "model_snapshot_manifest_sha256": _hash_tree(self.source_model_root)[0],
            "startup_metadata_log_sha256": sha256_file(self.runtime_metadata_log),
            "python_version": "3.12.1",
            "torch_version": "2.8.0",
            "transformers_version": "4.56.0",
            "numpy_version": "2.1.0",
            "device": "cpu",
            "dtype": "float32",
            "offline": True,
        }
        report = self.score(
            prediction_doc=doc,
            model_root=self.source_model_root,
            source_model_root=self.source_model_root,
            runtime_worker_path=self.runtime_worker,
        )
        self.assertEqual(report["runtime"]["kind"], "transformers")
        self.assertEqual(report["runtime"]["worker_script_sha256"], sha256_file(self.runtime_worker))
        self.assertEqual(report["runtime"]["startup_metadata_log_sha256"], sha256_file(self.runtime_metadata_log))
        self.assertEqual(report["runtime"]["offline"], True)
        self.assertEqual(report["model"]["runtime_model_snapshot_file_count"], 2)
        self.assertNotIn("converted_export_file_count", report["model"])

    def test_transformers_runtime_rejects_startup_metadata_log_drift(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["runtime"] = {
            "kind": "transformers",
            "python_executable_sha256": sha256_file(self.runtime_binary),
            "worker_script_sha256": sha256_file(self.runtime_worker),
            "model_snapshot_manifest_sha256": _hash_tree(self.source_model_root)[0],
            "startup_metadata_log_sha256": "0" * 64,
            "python_version": "3.12.1",
            "torch_version": "2.8.0",
            "transformers_version": "4.56.0",
            "numpy_version": "2.1.0",
            "device": "cpu",
            "dtype": "float32",
            "offline": True,
        }
        with self.assertRaisesRegex(ValueError, "startup metadata log SHA-256"):
            self.score(
                prediction_doc=doc,
                model_root=self.source_model_root,
                source_model_root=self.source_model_root,
                runtime_worker_path=self.runtime_worker,
            )

    def test_transformers_runtime_rejects_model_snapshot_digest_drift(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["runtime"] = {
            "kind": "transformers",
            "python_executable_sha256": sha256_file(self.runtime_binary),
            "worker_script_sha256": sha256_file(self.runtime_worker),
            "model_snapshot_manifest_sha256": "0" * 64,
            "startup_metadata_log_sha256": sha256_file(self.runtime_metadata_log),
            "python_version": "3.12.1",
            "torch_version": "2.8.0",
            "transformers_version": "4.56.0",
            "numpy_version": "2.1.0",
            "device": "cpu",
            "dtype": "float32",
            "offline": True,
        }
        with self.assertRaisesRegex(ValueError, "model snapshot manifest SHA-256"):
            self.score(
                prediction_doc=doc,
                model_root=self.source_model_root,
                source_model_root=self.source_model_root,
                runtime_worker_path=self.runtime_worker,
            )

    def test_transformers_runtime_requires_matching_worker_digest(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["runtime"] = {
            "kind": "transformers",
            "python_executable_sha256": sha256_file(self.runtime_binary),
            "worker_script_sha256": "0" * 64,
            "model_snapshot_manifest_sha256": _hash_tree(self.source_model_root)[0],
            "startup_metadata_log_sha256": sha256_file(self.runtime_metadata_log),
            "python_version": "3.12.1",
            "torch_version": "2.8.0",
            "transformers_version": "4.56.0",
            "numpy_version": "2.1.0",
            "device": "cpu",
            "dtype": "float32",
            "offline": True,
        }
        with self.assertRaisesRegex(ValueError, "worker script SHA-256"):
            self.score(
                prediction_doc=doc,
                model_root=self.source_model_root,
                source_model_root=self.source_model_root,
                runtime_worker_path=self.runtime_worker,
            )

    def test_aggregate_report_contains_no_transcripts_paths_or_speaker_tokens(self):
        report = self.score()
        encoded = json.dumps(report, ensure_ascii=False)
        for secret in ("Goeie môre", "clip-001.mp3", "speaker-a", "Ek weet dit nie"):
            self.assertNotIn(secret, encoded)
        self.assertFalse(report["privacy"]["raw_transcripts_included"])
        self.assertFalse(report["privacy"]["clip_paths_included"])
        self.assertFalse(report["privacy"]["speaker_tokens_included"])

    def test_json_loader_rejects_duplicate_object_keys(self):
        duplicate = self.base / "duplicate.json"
        duplicate.write_text('{"locale":"af","locale":"en"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            _read_json(duplicate)

    def test_normalizer_preserves_afrikaans_diacritics_and_folds_case(self):
        self.assertEqual(normalize_text("MÔRE!"), "môre")
        self.assertNotEqual(normalize_text("môre"), normalize_text("more"))

    def test_empty_hypothesis_is_scored_as_deletions_not_rejected(self):
        rows = copy.deepcopy(self.pred_doc["predictions"])
        rows[0]["text"] = ""
        self._write_predictions(rows)
        report = self.score()
        self.assertGreater(report["metrics"]["wer"], 0.2)

    def test_prediction_text_must_match_raw_outcome_transcript(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["predictions"][0]["text"] = "tampered transcript"
        with self.assertRaisesRegex(ValueError, "differs from raw pilot outcome transcript"):
            self.score(prediction_doc=doc)

    def test_latency_phase_summary_excludes_private_identifiers(self):
        report = self.score()
        encoded = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("clip-001.mp3", encoded)
        self.assertNotIn("speaker-a", encoded)
        self.assertEqual(report["latency"]["preflight_request_ms"]["sample_count"], 3)
        self.assertEqual(report["latency"]["transcription_ms"]["p50_ms"], 11.0)

    def test_missing_prediction_fails_closed(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["predictions"].pop()
        with self.assertRaisesRegex(ValueError, "missing 1 ASR predictions"):
            self.score(prediction_doc=doc)

    def test_duplicate_prediction_fails_closed(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["predictions"].append(copy.deepcopy(doc["predictions"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            self.score(prediction_doc=doc)

    def test_unknown_clip_path_fails_closed(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["predictions"][0]["path"] = "unknown.mp3"
        with self.assertRaisesRegex(ValueError, "unknown clip"):
            self.score(prediction_doc=doc)

    def test_prediction_dataset_metadata_drift_fails_closed(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["test_metadata_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "test metadata hash mismatch"):
            self.score(prediction_doc=doc)

    def test_model_identity_drift_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "model_revision mismatch"):
            self.score(model_revision="b" * 40)

    def test_cluster_bootstrap_marks_interval_unestimable_for_single_speaker(self):
        for split, row in (
            ("train", {"client_id":"single","path":"train.mp3","sentence":"Opleiding."}),
            ("dev", {"client_id":"single","path":"dev.mp3","sentence":"Toetsing."}),
            ("test", {"client_id":"single","path":"clip-001.mp3","sentence":"Goeie môre."}),
        ):
            path = self.root / f"{split}.tsv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["client_id","path","sentence"],
                                        delimiter="\\t", lineterminator="\\n")
                writer.writeheader(); writer.writerow(row)
            audio = self.root / "clips" / row["path"]
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"synthetic-audio-fixture")
        self._build_intake("ingestion-smoke")
        self._write_predictions([{"path":"clip-001.mp3","text":"Goeie môre."}])
        report = self.score()
        self.assertEqual(report["confidence_intervals_95"]["speaker_cluster_count"], 1)
        self.assertIsNone(report["confidence_intervals_95"]["wer"]["low_95"])
        self.assertEqual(report["confidence_intervals_95"]["wer"]["interval_status"],
                         "not_estimable_fewer_than_2_speakers")

    def test_raw_pilot_outcome_drift_fails_closed(self):
        self.outcomes_path.write_text("different outcomes\\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"raw pilot outcomes SHA-256 differs"):
            self.score()

    def test_runtime_binary_drift_fails_closed(self):
        self.runtime_binary.write_bytes(b"updated runtime binary")
        with self.assertRaisesRegex(ValueError, "runtime binary SHA-256 differs"):
            self.score()

    def test_converted_model_artifact_drift_fails_closed(self):
        (self.model_root / "model.bin").write_bytes(b"updated converted model")
        with self.assertRaisesRegex(ValueError, "runtime model artifact SHA-256 differs"):
            self.score()

    def test_upstream_source_snapshot_drift_fails_closed(self):
        (self.source_model_root / "weights.safetensors").write_bytes(b"changed source weights")
        with self.assertRaisesRegex(ValueError, "source model snapshot SHA-256 differs"):
            self.score()

    def test_source_model_root_symlink_is_rejected(self):
        alias = self.base / "source-model-link"
        alias.symlink_to(self.source_model_root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "source_model_root must be a real directory"):
            self.score(source_model_root=alias)

    def test_runtime_commit_must_be_pinned(self):
        doc = copy.deepcopy(self.pred_doc)
        doc["runtime"]["runtime_commit"] = "unversioned"
        with self.assertRaisesRegex(ValueError, "runtime.runtime_commit must be a 40-hex commit SHA"):
            self.score(prediction_doc=doc)

    def test_archive_drift_fails_closed_against_intake_receipt(self):
        self.archive.write_bytes(b"modified archive")
        with self.assertRaisesRegex(ValueError, "archive SHA-256 differs"):
            self.score()

    def test_modified_test_tsv_fails_closed_against_intake_receipt(self):
        with (self.root / "test.tsv").open("a", encoding="utf-8") as stream:
            stream.write("new-speaker\textra.mp3\tDit is nu anders.\n")
        with self.assertRaisesRegex(ValueError, "test.tsv SHA-256 differs"):
            self.score()

    def test_model_root_symlink_is_rejected(self):
        alias = self.base / "model-link"
        alias.symlink_to(self.model_root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "real exported directory"):
            self.score(model_root=alias)

    def test_model_snapshot_symlink_file_is_rejected(self):
        target = self.base / "external-weights.bin"
        target.write_bytes(b"outside weights")
        (self.model_root / "linked-weights.bin").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinked file"):
            self.score()

    def test_smoke_intake_cannot_be_promoted_to_heldout_evaluation(self):
        with self.assertRaisesRegex(ValueError, "requires intake mode heldout-evaluation"):
            self.score(mode="heldout-evaluation")

    def test_heldout_evaluation_requires_minimum_size_and_speaker_disjointness(self):
        self._build_intake("heldout-evaluation")
        self._write_predictions()
        with self.assertRaisesRegex(ValueError, "at least 100 test clips"):
            self.score(mode="heldout-evaluation")

    def test_heldout_evaluation_accepts_sufficient_disjoint_fixture(self):
        train = [{"client_id":"train-speaker","path":"train.mp3","sentence":"Opleiding."}]
        dev = [{"client_id":"dev-speaker","path":"dev.mp3","sentence":"Toetsing."}]
        test = [
            {"client_id":f"speaker-{i % 10}","path":f"heldout-{i:03}.mp3","sentence":"Dit is 'n held-out sin."}
            for i in range(100)
        ]
        self._write_dataset(train,dev,test)
        self._build_intake("heldout-evaluation")
        self._write_predictions()
        report = self.score(mode="heldout-evaluation",bootstrap_iterations=100)
        self.assertEqual(report["metrics"]["sample_count"],100)
        self.assertIn("model-training corpus overlap is not assessed",report["interpretation"])
        self.assertEqual(report["training_data_overlap_status"], "not_assessed")

    def test_report_writer_rejects_worker_source_tree(self):
        report = self.score()
        with self.assertRaisesRegex(ValueError, "aggregate report must be outside dataset and model directories"):
            write_report_exclusive(
                self.worker_root / "report.json", report,
                [self.root, self.model_root, self.source_model_root, self.worker_root],
            )

    def test_report_writer_refuses_overwrite(self):
        report = self.score()
        write_report_exclusive(self.report_path, report, [self.root, self.model_root])
        with self.assertRaisesRegex(ValueError, "must be new"):
            write_report_exclusive(self.report_path, report, [self.root, self.model_root])


if __name__ == "__main__":
    unittest.main()
