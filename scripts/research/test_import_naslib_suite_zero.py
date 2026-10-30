#!/usr/bin/env python3
"""Unit tests for the one-time NASLib SuiteZero importer."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("import_naslib_suite_zero.py")
SPEC = importlib.util.spec_from_file_location("import_naslib_suite_zero", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load importer module at {SCRIPT}")
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def _record(accuracy: float, params: float, synflow: float = 0.5) -> dict:
    return {
        "val_accuracy": accuracy,
        "params": {"score": params, "time": 0.01},
        "synflow": {"score": synflow, "time": 0.02},
    }


def _raw_artifact() -> bytes:
    return json.dumps(
        {
            "cifar10": {
                str((0, 1, 2, 3, 4, 0)): _record(91.0, 100.0),
                str((4, 3, 2, 1, 0, 4)): _record(89.0, 120.0),
            }
        }
    ).encode("utf-8")


def _build(raw: bytes | None = None, **overrides):
    args = {
        "dataset": "cifar10",
        "benchmark_id": "suitezero-test",
        "evaluation_split_id": "challenge-cifar10",
        "proxy_calibration_split_id": "development-cifar100",
        "proxy_calibration_sha256": "b" * 64,
        "max_unique_queries": 2,
        "top_k": 2,
        "proxy_names": ("params", "synflow"),
    }
    args.update(overrides)
    return IMPORTER.build_canonical_import(raw if raw is not None else _raw_artifact(), **args)


class SuiteZeroImporterTests(unittest.TestCase):
    def test_builds_canonical_import_with_realistic_schema_and_digests(self):
        raw = _raw_artifact()
        canonical, report = _build(raw)
        self.assertEqual(canonical["schema"], IMPORTER.CANONICAL_SCHEMA)
        self.assertEqual(canonical["manifest"]["metric_id"], "val_accuracy")
        self.assertEqual(canonical["manifest"]["metric_direction"], "maximize")
        self.assertEqual(canonical["manifest"]["artifact_sha256"], IMPORTER.hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["architecture_count"], 2)
        self.assertEqual(canonical["architectures"][0]["topology_encoding"], [0, 1, 2, 3, 4, 0])
        self.assertEqual(canonical["architectures"][0]["proxy_times_seconds"]["params"], 0.01)
        self.assertEqual(canonical["target_labels"][0]["value"], 91.0)

    def test_normalizes_infinite_proxy_scores_but_rejects_nan(self):
        source = {
            "cifar10": {
                str((0, 1, 2, 3, 4, 0)): _record(91.0, float("inf")),
                str((4, 3, 2, 1, 0, 4)): _record(89.0, float("-inf")),
            }
        }
        canonical, report = _build(json.dumps(source).encode())
        self.assertEqual(report["normalized_infinite_proxy_scores"], 2)
        self.assertEqual(canonical["architectures"][0]["proxy_scores"]["params"], 1e9)
        self.assertEqual(canonical["architectures"][1]["proxy_scores"]["params"], -1e9)

        source["cifar10"][str((0, 1, 2, 3, 4, 0))]["params"]["score"] = float("nan")
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(json.dumps(source).encode())

    def test_rejects_duplicate_json_keys(self):
        raw = b'{"cifar10":{"(0, 1, 2, 3, 4, 0)":{"val_accuracy":1},"(0, 1, 2, 3, 4, 0)":{"val_accuracy":2}}}'
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(raw)

    def test_rejects_noncanonical_architecture_tuples(self):
        source = {
            "cifar10": {
                str((0, 1, 2, 3, 4)): _record(91.0, 100.0),
                str((4, 3, 2, 1, 0, 4)): _record(89.0, 120.0),
            }
        }
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(json.dumps(source).encode())

        source["cifar10"] = {
            str((0, 1, 2, 3, 9, 0)): _record(91.0, 100.0),
            str((4, 3, 2, 1, 0, 4)): _record(89.0, 120.0),
        }
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(json.dumps(source).encode())

    def test_requires_disjoint_calibration_and_evaluation_splits(self):
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(proxy_calibration_split_id="challenge-cifar10")

    def test_rejects_missing_proxy_and_negative_runtime(self):
        source = {
            "cifar10": {
                str((0, 1, 2, 3, 4, 0)): _record(91.0, 100.0),
                str((4, 3, 2, 1, 0, 4)): _record(89.0, 120.0),
            }
        }
        del source["cifar10"][str((0, 1, 2, 3, 4, 0))]["synflow"]
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(json.dumps(source).encode())

        source["cifar10"][str((0, 1, 2, 3, 4, 0))] = _record(91.0, 100.0)
        source["cifar10"][str((0, 1, 2, 3, 4, 0))]["params"]["time"] = -0.1
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(json.dumps(source).encode())

    def test_rejects_unpinned_source_commit_and_non_finite_target(self):
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(source_commit="zerocost")

        source = {
            "cifar10": {
                str((0, 1, 2, 3, 4, 0)): _record(float("nan"), 100.0),
                str((4, 3, 2, 1, 0, 4)): _record(89.0, 120.0),
            }
        }
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(json.dumps(source).encode())

    def test_rejects_missing_dataset_and_unknown_proxy_selection(self):
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(json.dumps({"cifar100": {}}).encode())
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(proxy_names=("params", "invented_proxy"))

    def test_rejects_query_budget_and_top_k_outside_table(self):
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(max_unique_queries=3)
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            _build(top_k=1)


if __name__ == "__main__":
    unittest.main()
