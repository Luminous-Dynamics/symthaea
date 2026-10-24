#!/usr/bin/env python3
"""Unit tests for strict JSON parsing in the socioeconomic receipt validator."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from validate_socioeconomic_run_receipt import (
    ARTIFACT_MANIFEST_PATH,
    FIXTURE_PATH,
    REGISTRY_PATH,
    ValidationFailure,
    build_metric_catalog,
    digest_binding_errors,
    load_json,
    semantic_errors,
    source_artifact_errors,
)


class StrictJsonLoaderTests(unittest.TestCase):
    def load_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(text, encoding="utf-8")
            return load_json(path)

    def test_unique_object_keys_are_accepted(self) -> None:
        self.assertEqual(self.load_text('{"schema":"receipt-v1","count":2}'), {
            "schema": "receipt-v1",
            "count": 2,
        })

    def test_duplicate_top_level_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationFailure, "duplicate JSON object key"):
            self.load_text('{"claim_ceiling":"synthetic_only","claim_ceiling":"production"}')

    def test_duplicate_nested_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationFailure, "duplicate JSON object key"):
            self.load_text('{"authority_boundary":{"can_settle":false,"can_settle":true}}')

    def test_non_finite_numeric_constants_are_rejected(self) -> None:
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationFailure, "non-standard numeric constant"):
                    self.load_text('{"metric_value":' + value + '}')

    def test_float_overflow_and_underflow_are_rejected(self) -> None:
        for value in ("1e400", "1e-4000"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationFailure, "(overflows|underflows) float range"):
                    self.load_text('{"metric_value":' + value + '}')

    def test_malformed_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationFailure, "cannot load strict JSON"):
            self.load_text('{"missing_value":')



class MetricRegistrySemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = load_json(FIXTURE_PATH)
        self.registry = load_json(REGISTRY_PATH)
        self.metric_catalog = build_metric_catalog(self.registry.get("metrics"))
        self.registry_digest = "sha256:" + hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
        self.scenario_ids = set(self.registry["scenario_ids"])
        self.manifest = load_json(ARTIFACT_MANIFEST_PATH)
        self.manifest_digest = "sha256:" + hashlib.sha256(ARTIFACT_MANIFEST_PATH.read_bytes()).hexdigest()
        _source_errors, self.source_digests = source_artifact_errors(self.manifest)

    def check_semantics(self) -> list[str]:
        return semantic_errors(
            self.fixture,
            self.registry_digest,
            self.scenario_ids,
            self.metric_catalog,
            self.manifest,
            self.manifest_digest,
            self.source_digests,
        )

    def test_unknown_metric_id_is_rejected(self) -> None:
        self.fixture["metrics"][0]["metric_id"] = "unregistered_metric"
        errors = self.check_semantics()
        self.assertTrue(any("is not declared by the scenario registry" in error for error in errors))

    def test_registered_metric_with_wrong_dimension_is_rejected(self) -> None:
        self.fixture["metrics"][0]["dimension"] = "distribution"
        errors = self.check_semantics()
        self.assertTrue(any("does not match registered dimension" in error for error in errors))


    def test_missing_digest_binding_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["digest_bindings"] = [
            entry for entry in manifest["digest_bindings"]
            if entry["receipt_pointer"] != "/run/world_seed_digest"
        ]
        errors = digest_binding_errors(
            manifest, self.fixture, self.manifest_digest, self.source_digests
        )
        self.assertTrue(any("pointer set mismatch" in error for error in errors))

    def test_receipt_digest_must_match_manifest_binding(self) -> None:
        receipt = copy.deepcopy(self.fixture)
        receipt["run"]["environment_digest"] = "sha256:" + "1" * 64
        errors = digest_binding_errors(
            self.manifest, receipt, self.manifest_digest, self.source_digests
        )
        self.assertTrue(any("claimed_digest differs" in error for error in errors))

    def test_source_artifact_byte_digest_mismatch_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["source_artifacts"][0]["digest"] = "sha256:" + "0" * 64
        errors, _actual = source_artifact_errors(manifest)
        self.assertTrue(any("digest does not match exact source-file bytes" in error for error in errors))

if __name__ == "__main__":
    unittest.main()
