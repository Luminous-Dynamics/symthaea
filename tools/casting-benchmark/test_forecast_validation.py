#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from forecast_validation import main, make_outcome_record, score_forecasts


MODEL = "a" * 64
STATE = "b" * 64


def forecast(forecast_id, probabilities, regime="nominal", task_id=None):
    return {
        "forecast_id": forecast_id,
        "task_id": task_id or forecast_id,
        "regime": regime,
        "model_sha256": MODEL,
        "state_sha256": STATE,
        "probabilities": probabilities,
    }


def outcomes_for(forecasts, outcome_indices):
    return [
        make_outcome_record(forecast_item, outcome_index)
        for forecast_item, outcome_index in zip(forecasts, outcome_indices, strict=True)
    ]


class ForecastValidationTests(unittest.TestCase):
    def test_perfect_forecast_has_zero_brier_and_log_loss(self):
        forecasts = [forecast("f1", [1.0, 0.0])]
        result = score_forecasts(forecasts, outcomes_for(forecasts, [0]), bin_count=10)
        self.assertEqual(result["overall"]["multiclass_brier_score_mean"], 0.0)
        self.assertEqual(result["overall"]["log_loss_mean"], 0.0)
        self.assertEqual(result["overall"]["top_label_accuracy"], 1.0)
        self.assertEqual(result["overall"]["top_label_expected_calibration_error"], 0.0)

    def test_surprising_event_has_known_scores_without_probability_clipping(self):
        forecasts = [forecast("f1", [0.9, 0.1])]
        result = score_forecasts(forecasts, outcomes_for(forecasts, [1]), bin_count=10)
        self.assertAlmostEqual(result["overall"]["multiclass_brier_score_mean"], 1.62)
        self.assertAlmostEqual(result["overall"]["log_loss_mean"], -math.log(0.1))
        self.assertEqual(result["overall"]["top_label_accuracy"], 0.0)

    def test_zero_probability_surprise_reports_infinite_log_loss_as_null_not_clipped(self):
        forecasts = [forecast("f1", [1.0, 0.0])]
        result = score_forecasts(forecasts, outcomes_for(forecasts, [1]), bin_count=10)
        self.assertIsNone(result["overall"]["log_loss_mean"])
        self.assertEqual(result["overall"]["log_loss_zero_probability_count"], 1)
        self.assertAlmostEqual(result["overall"]["multiclass_brier_score_mean"], 2.0)

    def test_regime_stratification_exposes_shift_degradation(self):
        forecasts = [
            forecast("nominal-1", [0.9, 0.1], "nominal"),
            forecast("nominal-2", [0.8, 0.2], "nominal"),
            forecast("shift-1", [0.9, 0.1], "boundary_drift"),
            forecast("shift-2", [0.8, 0.2], "boundary_drift"),
        ]
        outcomes = outcomes_for(forecasts, [0, 0, 1, 1])
        result = score_forecasts(forecasts, outcomes, bin_count=5)
        self.assertEqual(result["by_regime"]["nominal"]["top_label_accuracy"], 1.0)
        self.assertEqual(result["by_regime"]["boundary_drift"]["top_label_accuracy"], 0.0)
        self.assertGreater(
            result["by_regime"]["boundary_drift"]["multiclass_brier_score_mean"],
            result["by_regime"]["nominal"]["multiclass_brier_score_mean"],
        )

    def test_binds_outcome_to_exact_forecast_bytes(self):
        forecasts = [forecast("f1", [0.5, 0.5])]
        outcomes = outcomes_for(forecasts, [0])
        forecasts[0]["probabilities"] = [0.6, 0.4]
        with self.assertRaisesRegex(ValueError, "forecast digest mismatch"):
            score_forecasts(forecasts, outcomes)

    def test_rejects_outcome_with_bad_digest(self):
        forecasts = [forecast("f1", [0.5, 0.5])]
        outcomes = outcomes_for(forecasts, [0])
        outcomes[0]["outcome_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not bind"):
            score_forecasts(forecasts, outcomes)

    def test_rejects_missing_or_extra_outcomes(self):
        forecasts = [forecast("f1", [0.5, 0.5]), forecast("f2", [0.5, 0.5])]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            score_forecasts(forecasts, outcomes_for(forecasts[:1], [0]))
        with self.assertRaisesRegex(ValueError, "unknown forecast_id"):
            score_forecasts(forecasts[:1], outcomes_for(forecasts, [0, 1]))

    def test_rejects_duplicate_ids(self):
        forecasts = [forecast("f1", [0.5, 0.5]), forecast("f1", [0.4, 0.6])]
        with self.assertRaisesRegex(ValueError, "duplicate forecast_id"):
            score_forecasts(forecasts, outcomes_for(forecasts, [0, 1]))

    def test_rejects_bad_probabilities_and_outcome_ranges(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            score_forecasts([forecast("f1", [0.2, 0.2])], [], bin_count=10)
        forecasts = [forecast("f1", [0.5, 0.5])]
        outcomes = outcomes_for(forecasts, [0])
        outcomes[0]["outcome_index"] = 2
        with self.assertRaisesRegex(ValueError, "outside the forecast class range"):
            score_forecasts(forecasts, outcomes)

    def test_cli_binds_exact_input_manifest_bytes_and_evaluator(self):
        forecasts = [forecast("f1", [0.7, 0.3])]
        outcomes = outcomes_for(forecasts, [0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast_path = root / "forecasts.json"
            outcome_path = root / "outcomes.json"
            output_path = root / "receipt.json"
            forecast_path.write_text(json.dumps(forecasts, separators=(",", ":")), encoding="utf-8")
            outcome_path.write_text(json.dumps(outcomes, separators=(",", ":")), encoding="utf-8")
            forecast_bytes = forecast_path.read_bytes()
            outcome_bytes = outcome_path.read_bytes()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([
                    str(forecast_path), str(outcome_path),
                    "--output", str(output_path), "--bins", "5",
                ])
            self.assertEqual(code, 0)
            receipt = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["receipt_label"], "PREQUENTIAL_PREDICTIVE_SCORING_NOT_CALIBRATION_CERTIFICATION")
            self.assertEqual(
                receipt["source_artifacts"]["forecast_manifest_sha256"],
                hashlib.sha256(forecast_bytes).hexdigest(),
            )
            self.assertEqual(
                receipt["source_artifacts"]["outcome_manifest_sha256"],
                hashlib.sha256(outcome_bytes).hexdigest(),
            )
            self.assertEqual(len(receipt["evaluator_sha256"]), 64)

    def test_cli_rejects_duplicate_json_keys_and_emits_no_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast_path = root / "forecasts.json"
            outcome_path = root / "outcomes.json"
            output_path = root / "receipt.json"
            forecast_path.write_text('[{"forecast_id":"x","forecast_id":"y"}]', encoding="utf-8")
            outcome_path.write_text("[]", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main([str(forecast_path), str(outcome_path), "--output", str(output_path)])
            self.assertEqual(code, 2)
            self.assertIn("duplicate JSON key", stderr.getvalue())
            self.assertFalse(output_path.exists())

    def test_cli_refuses_to_overwrite_an_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast_path = root / "forecasts.json"
            outcome_path = root / "outcomes.json"
            forecast_path.write_text("[]", encoding="utf-8")
            original = forecast_path.read_bytes()
            outcome_path.write_text("[]", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main([str(forecast_path), str(outcome_path), "--output", str(forecast_path)])
            self.assertEqual(code, 2)
            self.assertEqual(forecast_path.read_bytes(), original)

    def test_reliability_bins_are_fixed_and_descriptive(self):
        forecasts = [
            forecast("f1", [0.9, 0.1]),
            forecast("f2", [0.8, 0.2]),
            forecast("f3", [0.6, 0.4]),
        ]
        result = score_forecasts(forecasts, outcomes_for(forecasts, [0, 1, 0]), bin_count=5)
        self.assertEqual(len(result["overall"]["top_label_reliability_bins"]), 5)
        self.assertEqual(result["overall"]["top_label_reliability_bins"][4]["count"], 2)
        self.assertGreater(result["overall"]["top_label_expected_calibration_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
