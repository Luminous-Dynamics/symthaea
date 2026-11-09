# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed synthetic tests for the IIT4 landscape comparison utility."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from validation.compare_iit4_landscape_reference import compare


class LandscapeCrossEngineComparisonTests(unittest.TestCase):
    def _write_inputs(self, root: Path, mutate: str | None = None):
        pyphi_points = []
        rust_points = []
        rust_scores = []
        for index in range(13):
            theta = round(0.20 + 0.05 * index, 2)
            model_sha = f"{index + 1:064x}"
            eval_sha = f"{index + 101:064x}"
            request_sha = f"{index + 201:064x}"
            cut_sha = f"{index + 301:064x}"
            cause_state = [0, 1, 1]
            effect_state = [1, 0, 1]
            score = {
                "cut_sha256": cut_sha,
                "signed_phi": -0.25,
                "phi": 0.0,
                "normalized_phi": 0.0,
                "signed_normalized_phi": -0.125,
                "cause_signed_phi": -0.25,
                "cause_phi": 0.0,
                "effect_signed_phi": -0.10,
                "effect_phi": 0.0,
            }
            pyphi_points.append({
                "theta": theta,
                "model_sha256": model_sha,
                "evaluation_sha256": eval_sha,
                "analysis_request_sha256": request_sha,
                "cause_state": cause_state,
                "effect_state": effect_state,
                "mip_tie_cut_sha256s": [cut_sha],
                "mip_cut_sha256": cut_sha,
                "phi": 0.0,
                "signed_phi": -0.25,
                "normalized_phi": 0.0,
                "signed_normalized_phi": -0.125,
                "regime": 0,
                "partition_scores": [score],
            })
            rust_points.append({
                "theta": f"{theta:.8f}",
                "canonical_dense_model_sha256": model_sha,
                "canonical_dense_evaluation_sha256": eval_sha,
                "canonical_dense_analysis_request_sha256": request_sha,
                "selected_cause_state": "011",
                "selected_effect_state": "101",
                "tied_mip_cut_sha256s": cut_sha,
                "mip_tie_count": "1",
                "winning_mip_cut_sha256": cut_sha,
                "phi_s": "0",
                "signed_phi": "-0.25",
                "normalized_phi": "0",
                "signed_normalized_phi": "-0.125",
                "selection_regime": "0",
            })
            rust_scores.append({
                "theta": f"{theta:.8f}",
                "cause_state": "011",
                "effect_state": "101",
                "cut_sha256": cut_sha,
                "cause_signed_phi": "-0.25",
                "effect_signed_phi": "-0.10",
                "signed_phi": "-0.25",
                "phi": "0",
                "normalized_phi": "0",
                "signed_normalized_phi": "-0.125",
            })

        if mutate == "score":
            rust_scores[0]["signed_phi"] = "-0.50"
        elif mutate == "cut":
            rust_scores[0]["cut_sha256"] = "f" * 64
        elif mutate == "digest":
            rust_points[0]["canonical_dense_model_sha256"] = "f" * 64

        pyphi_path = root / "pyphi.json"
        rust_path = root / "rust.csv"
        score_path = root / "scores.csv"
        pyphi_path.write_text(json.dumps({
            "status": "PASS",
            "schema_version": "pyphi-iit4-2023-landscape-reference-v0.1",
            "provenance": {
                "formalism": "IIT_4_0_2023",
                "version": "2.0.0rc4",
                "wheel_sha256": "a" * 64,
            },
            "skipped_points": [],
            "points": pyphi_points,
        }), encoding="utf-8")
        for path, rows in ((rust_path, rust_points), (score_path, rust_scores)):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return pyphi_path, rust_path, score_path

    def test_same_input_per_cut_and_mip_family_parity_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_inputs(Path(directory))
            result = compare(*paths, tolerance=1e-12)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["summary"]["compared_point_count"], 13)
            self.assertEqual(result["summary"]["matched_cut_score_count"], 13)

    def test_directional_or_combined_score_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_inputs(Path(directory), mutate="score")
            result = compare(*paths, tolerance=1e-12)
            self.assertEqual(result["status"], "NOT_QUALIFIED")
            self.assertTrue(result["mismatches"])

    def test_missing_cut_and_digest_mismatch_fail_closed(self):
        for mutation in ("cut", "digest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                paths = self._write_inputs(Path(directory), mutate=mutation)
                result = compare(*paths, tolerance=1e-12)
                self.assertEqual(result["status"], "NOT_QUALIFIED")
                self.assertTrue(result["mismatches"])


if __name__ == "__main__":
    unittest.main()
