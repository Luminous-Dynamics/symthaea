#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free fail-closed tests for the voice-language registry."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from validate_voice_language_registry import (
    DEFAULT_REGISTRY,
    validate_registry,
    _read_json,
)

REGISTRY = _read_json(DEFAULT_REGISTRY)


class VoiceLanguageRegistryTests(unittest.TestCase):
    def test_repository_registry_is_valid_and_keeps_unmeasured_tasks_unverified(self):
        summary = validate_registry(REGISTRY)
        self.assertEqual(summary["languages"], 4)
        self.assertEqual(summary["capability_records"], 32)
        self.assertEqual(summary["status_implementation_present"], 3)
        self.assertEqual(summary["status_unverified"], 29)
        for tag in ("af", "zu", "xh"):
            language = next(row for row in REGISTRY["languages"] if row["tag"] == tag)
            self.assertTrue(all(record["status"] == "unverified"
                                for record in language["capabilities"].values()))

    def test_duplicate_tags_are_case_insensitive(self):
        registry = copy.deepcopy(REGISTRY)
        registry["languages"].append(copy.deepcopy(registry["languages"][0]))
        registry["languages"][-1]["tag"] = "EN"
        with self.assertRaisesRegex(ValueError, "duplicates another tag"):
            validate_registry(registry)

    def test_malformed_language_tag_is_rejected(self):
        registry = copy.deepcopy(REGISTRY)
        registry["languages"][0]["tag"] = "en--US"
        with self.assertRaisesRegex(ValueError, "BCP 47-shaped"):
            validate_registry(registry)

    def test_unknown_capability_is_rejected(self):
        registry = copy.deepcopy(REGISTRY)
        registry["languages"][0]["capabilities"]["animal_translation"] = {}
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            validate_registry(registry)

    def test_code_present_without_backend_or_source_is_rejected(self):
        registry = copy.deepcopy(REGISTRY)
        record = registry["languages"][0]["capabilities"]["g2p"]
        record["source_refs"] = []
        with self.assertRaisesRegex(ValueError, "requires backend and source_refs"):
            validate_registry(registry)

    def test_source_refs_reject_windows_paths_and_parent_components(self):
        registry = copy.deepcopy(REGISTRY)
        record = registry["languages"][0]["capabilities"]["g2p"]
        record["source_refs"] = [r"C:\private\source.py"]
        with self.assertRaisesRegex(ValueError, "safe repository-relative paths"):
            validate_registry(registry)

        record["source_refs"] = ["../private/source.py"]
        with self.assertRaisesRegex(ValueError, "safe repository-relative paths"):
            validate_registry(registry)

    def test_automated_status_requires_receipt(self):
        registry = copy.deepcopy(REGISTRY)
        registry["languages"][0]["capabilities"]["g2p"]["status"] = "automated_checks_passed"
        with self.assertRaisesRegex(ValueError, "requires a verification receipt"):
            validate_registry(registry)

    def test_speaker_reviewed_status_requires_verification_receipt(self):
        registry = copy.deepcopy(REGISTRY)
        registry["languages"][0]["capabilities"]["g2p"]["status"] = "speaker_reviewed"
        registry["languages"][0]["capabilities"]["g2p"]["verification"] = None
        with self.assertRaisesRegex(ValueError, "requires a verification receipt"):
            validate_registry(registry)

    def test_speaker_reviewed_status_requires_passing_evaluation_and_review(self):
        registry = copy.deepcopy(REGISTRY)
        row = registry["languages"][0]["capabilities"]["g2p"]
        row["status"] = "speaker_reviewed"
        row["verification"] = {
            "commit_sha": "a" * 40,
            "run_url": "https://github.com/Luminous-Dynamics/symthaea/actions/runs/123",
            "checks": ["unit-tests"],
        }
        with self.assertRaisesRegex(ValueError, "requires a passing held-out evaluation"):
            validate_registry(registry)

        row["evaluation"] = {
            "metric": "phone accuracy", "value": 0.99, "threshold": 0.95,
            "comparator": "gte", "sample_count": 10, "held_out": True,
            "split_description": "held-out speakers", "outcome": "pass",
        }
        with self.assertRaisesRegex(ValueError, "requires speaker review evidence"):
            validate_registry(registry)

    def test_evaluation_outcome_cannot_contradict_threshold(self):
        record = {
            "metric": "WER", "value": 0.2, "threshold": 0.1,
            "comparator": "lte", "sample_count": 10, "held_out": True,
            "split_description": "held-out speakers", "outcome": "pass",
        }
        registry = copy.deepcopy(REGISTRY)
        from validate_voice_language_registry import _validate_evaluation
        with self.assertRaisesRegex(ValueError, "contradicts value/threshold"):
            _validate_evaluation(record, "test")

    def test_production_status_requires_pinned_model_and_dataset(self):
        registry = copy.deepcopy(REGISTRY)
        row = registry["languages"][0]["capabilities"]["g2p"]
        row["status"] = "production_qualified"
        row["verification"] = {
            "commit_sha": "a" * 40,
            "run_url": "https://github.com/Luminous-Dynamics/symthaea/actions/runs/123",
            "checks": ["unit-tests"],
        }
        row["evaluation"] = {
            "metric": "phone accuracy", "value": 0.99, "threshold": 0.95,
            "comparator": "gte", "sample_count": 10, "held_out": True,
            "split_description": "held-out speakers", "outcome": "pass",
        }
        row["speaker_review"] = {
            "reviewer_count": 2, "method": "independent listening",
            "scope": "en, tested phone set", "date": "2026-10-12",
        }
        with self.assertRaisesRegex(ValueError, "requires pinned model and dataset"):
            validate_registry(registry)

    def test_complete_production_qualification_fixture_is_structurally_accepted(self):
        # Synthetic metadata only: this checks the validator contract, not a
        # real model, licensed corpus, successful CI run, or language quality.
        registry = copy.deepcopy(REGISTRY)
        row = registry["languages"][0]["capabilities"]["g2p"]
        row.update({
            "status": "production_qualified",
            "model": {
                "name": "synthetic-test-model", "revision": "test-only",
                "sha256": "a" * 64, "license": "test-fixture-license",
            },
            "dataset": {
                "name": "synthetic-test-corpus", "revision": "test-only",
                "sha256": "b" * 64, "license": "test-fixture-license",
                "held_out_split": "test-only held-out speaker split",
            },
            "verification": {
                "commit_sha": "c" * 40,
                "run_url": "https://github.com/Luminous-Dynamics/symthaea/actions/runs/123",
                "checks": ["synthetic-validator-fixture"],
            },
            "evaluation": {
                "metric": "phone accuracy", "value": 0.99, "threshold": 0.95,
                "comparator": "gte", "sample_count": 100, "held_out": True,
                "split_description": "synthetic test fixture only", "outcome": "pass",
            },
            "speaker_review": {
                "reviewer_count": 2, "method": "fixture-only",
                "scope": "synthetic test case", "date": "2026-10-12",
            },
        })
        summary = validate_registry(registry)
        self.assertEqual(summary["status_production_qualified"], 1)

    def test_nonstandard_json_constants_fail_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "invalid.json"
            path.write_text('{"schema_version": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-standard JSON numeric constant"):
                _read_json(path)


if __name__ == "__main__":
    unittest.main()
