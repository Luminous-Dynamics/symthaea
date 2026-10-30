#!/usr/bin/env python3
"""Unit tests for the hash-gated NAS-Bench-201 curve importer."""

from __future__ import annotations

import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("import_naslib_nb201_learning_curves.py")
SPEC = importlib.util.spec_from_file_location("import_naslib_nb201_learning_curves", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load curve importer at {SCRIPT}")
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


IDS = [
    "(0, 1, 2, 3, 4, 0)",
    "(4, 3, 2, 1, 0, 4)",
]


def suitezero_fixture() -> dict:
    return {
        "schema": IMPORTER.SUITEZERO_SCHEMA,
        "version": IMPORTER.CANONICAL_VERSION,
        "manifest": {
            "schema": IMPORTER.BENCHMARK_SCHEMA,
            "version": IMPORTER.CANONICAL_VERSION,
            "benchmark_id": "suitezero-nb201-cifar10-v1",
            "metric_id": "val_accuracy",
            "metric_direction": "maximize",
            "split_id": "challenge-cifar10",
            "proxy_calibration_split_id": "development-cifar100",
            "proxy_calibration_sha256": "b" * 64,
            "source_description": "NAS-Bench-Suite-Zero; dataset=cifar10",
            "artifact_sha256": "a" * 64,
            "max_unique_queries": 1,
            "top_k": 2,
        },
        "architectures": [
            {
                "architecture_id": IDS[0],
                "proxy_scores": {"jacov": 1.0},
                "proxy_times_seconds": {"jacov": 0.01},
                "topology_encoding": [0, 1, 2, 3, 4, 0],
            },
            {
                "architecture_id": IDS[1],
                "proxy_scores": {"jacov": 2.0},
                "proxy_times_seconds": {"jacov": 0.02},
                "topology_encoding": [4, 3, 2, 1, 0, 4],
            },
        ],
        "target_labels": [
            {"architecture_id": IDS[0], "value": 90.0},
            {"architecture_id": IDS[1], "value": 91.0},
        ],
    }


def curve_fixture() -> dict:
    return {
        "nb201_data": {
            IMPORTER.suitezero_id_to_naslib_curve_key(IDS[0]): {
                "cifar10-valid": {"eval_acc1es": [float(epoch) for epoch in range(1, 20)] + [90.0]}
            },
            IMPORTER.suitezero_id_to_naslib_curve_key(IDS[1]): {
                "cifar10-valid": {"eval_acc1es": [float(epoch + 1) for epoch in range(1, 20)] + [91.0]}
            },
        }
    }


def build(suitezero=None, curves=None, **overrides):
    args = {
        "suitezero_canonical": suitezero if suitezero is not None else suitezero_fixture(),
        "suitezero_source_sha256": "c" * 64,
        "curve_data": curves if curves is not None else curve_fixture(),
        "curve_source_sha256": "d" * 64,
        "dataset": "cifar10",
        "epochs": (5, 15, 20),
        "max_final_queries": 1,
        "max_total_cost_units": 500,
        "final_accuracy_tolerance": 0.1,
        "fidelity_cost_model": "resumable_prefix",
    }
    args.update(overrides)
    return IMPORTER.build_multifidelity_import(**args)


class NasBench201CurveImporterTests(unittest.TestCase):
    def test_suitezero_tuple_ids_convert_to_exact_naslib_curve_keys(self):
        self.assertEqual(
            IMPORTER.suitezero_id_to_naslib_curve_key("(4, 3, 0, 3, 0, 0)"),
            "|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|",
        )
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            IMPORTER.suitezero_id_to_naslib_curve_key("(4, 3, 0, 3, 0)")
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            IMPORTER.suitezero_id_to_naslib_curve_key("(4, 3, 0, 3, 9, 0)")

    def test_emits_canonical_multifidelity_import_and_expected_rungs(self):
        canonical, report = build()
        self.assertEqual(canonical["schema"], IMPORTER.CANONICAL_SCHEMA)
        self.assertEqual(canonical["manifest"]["fidelity_cost_model"], "resumable_prefix")
        self.assertEqual(canonical["manifest"]["resource_unit"], "epoch_equivalent")
        self.assertEqual(
            [level["cumulative_cost_units"] for level in canonical["manifest"]["fidelity_levels"]],
            [5, 15, 20],
        )
        self.assertEqual(len(canonical["target_labels"]), 6)
        self.assertEqual(canonical["target_labels"][0]["value"], 5.0)
        self.assertEqual(canonical["target_labels"][2]["value"], 90.0)
        self.assertEqual(report["dataset_key"], "cifar10-valid")
        self.assertEqual(report["architecture_count"], 2)

    def test_validates_full_fidelity_endpoint_against_suitezero_accuracy(self):
        suitezero = suitezero_fixture()
        suitezero["target_labels"][0]["value"] = 10.0
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            build(suitezero=suitezero)

    def test_rejects_missing_or_extra_architecture_curve_records(self):
        curves = curve_fixture()
        del curves["nb201_data"][IMPORTER.suitezero_id_to_naslib_curve_key(IDS[0])]
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            build(curves=curves)

        curves = curve_fixture()
        curves["nb201_data"][IMPORTER.suitezero_id_to_naslib_curve_key("(1, 1, 1, 1, 1, 1)")] = {
            "cifar10-valid": {"eval_acc1es": [1.0] * 20}
        }
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            build(curves=curves)

    def test_rejects_missing_dataset_key_or_short_learning_curve(self):
        curves = curve_fixture()
        del curves["nb201_data"][IMPORTER.suitezero_id_to_naslib_curve_key(IDS[0])]["cifar10-valid"]
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            build(curves=curves)

        curves = curve_fixture()
        curves["nb201_data"][IMPORTER.suitezero_id_to_naslib_curve_key(IDS[0])]["cifar10-valid"]["eval_acc1es"] = [1.0] * 14
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            build(curves=curves)

    def test_rejects_nonfinite_learning_curve_values(self):
        curves = curve_fixture()
        curves["nb201_data"][IMPORTER.suitezero_id_to_naslib_curve_key(IDS[0])]["cifar10-valid"]["eval_acc1es"][4] = float("nan")
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            build(curves=curves)

    def test_rejects_invalid_fidelity_schedule_and_budget(self):
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            build(epochs=(15, 5, 20))
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            build(epochs=(5, 15, 20), max_final_queries=3)
        with self.assertRaises(IMPORTER.ImportErrorDetail):
            build(max_total_cost_units=0)

    def test_cifar10_validation_dataset_name_matches_naslib(self):
        canonical, _ = build()
        self.assertIn("dataset=cifar10", canonical["manifest"]["base"]["source_description"])
        self.assertIn("cifar10-valid", canonical["manifest"]["base"]["source_description"])

    def test_curve_pickle_digest_is_checked_before_deserialization(self):
        raw = pickle.dumps(curve_fixture(), protocol=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.pickle"
            path.write_bytes(raw)
            with self.assertRaises(IMPORTER.ImportErrorDetail):
                IMPORTER._load_trusted_pickle(path, "0" * 64)
            data, loaded, actual = IMPORTER._load_trusted_pickle(
                path, IMPORTER.sha256_hex(raw)
            )
            self.assertEqual(data["nb201_data"].keys(), curve_fixture()["nb201_data"].keys())
            self.assertEqual(loaded, raw)
            self.assertEqual(actual, IMPORTER.sha256_hex(raw))


if __name__ == "__main__":
    unittest.main()
