import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from check_regressions import analyze_regressions, load_baseline, load_criterion_changes, load_criterion_results, main


def write_estimates(path: Path, mean: float, std_dev: float = 1.0, median: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": {"point_estimate": mean},
        "std_dev": {"point_estimate": std_dev},
        "median": {"point_estimate": mean if median is None else median},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_change_estimate(path: Path, change: float, lower: float, upper: float, confidence: float = 0.95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": {
            "point_estimate": change,
            "confidence_interval": {
                "lower_bound": lower,
                "upper_bound": upper,
                "confidence_level": confidence,
            },
            "standard_error": 0.01,
        },
        "median": {
            "point_estimate": change,
            "confidence_interval": {
                "lower_bound": lower,
                "upper_bound": upper,
                "confidence_level": confidence,
            },
            "standard_error": 0.01,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class CriterionBaselineTests(unittest.TestCase):
    def test_loads_current_and_named_baseline_from_criterion_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_estimates(root / "quick" / "harmonic" / "main" / "estimates.json", 100.0)
            write_estimates(root / "quick" / "harmonic" / "new" / "estimates.json", 115.0)

            current = load_criterion_results(root)
            baseline = load_baseline(root, "main")

            self.assertEqual(set(current), {"quick/harmonic"})
            self.assertEqual(set(baseline), {"quick/harmonic"})
            self.assertEqual(current["quick/harmonic"]["mean"], 115.0)
            self.assertEqual(baseline["quick/harmonic"]["mean"], 100.0)

    def test_detects_critical_regression_in_standard_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_estimates(root / "quick" / "harmonic" / "main" / "estimates.json", 100.0)
            write_estimates(root / "quick" / "harmonic" / "new" / "estimates.json", 115.0)
            write_change_estimate(root / "quick" / "harmonic" / "change" / "estimates.json", 0.15, 0.12, 0.18)

            regressions, has_critical = analyze_regressions(
                root, threshold=5.0, critical_threshold=10.0, baseline_name="main"
            )

            self.assertTrue(has_critical)
            self.assertEqual(len(regressions), 1)
            self.assertAlmostEqual(regressions[0].change_percent, 15.0)
            self.assertTrue(regressions[0].is_critical)
            self.assertEqual(regressions[0].change_ci_lower_percent, 12.0)
            self.assertEqual(regressions[0].change_ci_upper_percent, 18.0)
            self.assertEqual(regressions[0].confidence_level, 0.95)
            self.assertFalse(regressions[0].is_inconclusive)

    def test_required_baseline_fails_closed_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_estimates(root / "quick" / "harmonic" / "new" / "estimates.json", 100.0)
            stderr = io.StringIO()
            with patch.object(sys, "argv", [
                "check_regressions.py", "--results", str(root), "--require-baseline"
            ]), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Required Criterion baseline", stderr.getvalue())


    def test_required_baseline_rejects_partial_benchmark_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_estimates(root / "quick" / "harmonic" / "main" / "estimates.json", 100.0)
            write_estimates(root / "quick" / "removed" / "main" / "estimates.json", 90.0)
            write_estimates(root / "quick" / "harmonic" / "new" / "estimates.json", 100.0)
            write_estimates(root / "quick" / "added" / "new" / "estimates.json", 100.0)
            write_change_estimate(root / "quick" / "harmonic" / "change" / "estimates.json", 0.0, -0.02, 0.02)

            stderr = io.StringIO()
            with patch.object(sys, "argv", [
                "check_regressions.py", "--results", str(root), "--require-baseline"
            ]), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Incomplete Criterion comparison", stderr.getvalue())
            self.assertIn("current benchmarks missing a baseline: quick/added", stderr.getvalue())
            self.assertIn("baseline benchmarks missing current results: quick/removed", stderr.getvalue())

    def test_required_baseline_rejects_missing_change_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_estimates(root / "quick" / "harmonic" / "main" / "estimates.json", 100.0)
            write_estimates(root / "quick" / "harmonic" / "new" / "estimates.json", 100.0)

            stderr = io.StringIO()
            with patch.object(sys, "argv", [
                "check_regressions.py", "--results", str(root), "--require-baseline"
            ]), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("missing Criterion change estimates", stderr.getvalue())

    def test_rejects_non_finite_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            estimate = root / "quick" / "harmonic" / "new" / "estimates.json"
            write_estimates(estimate, float("nan"))

            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_criterion_results(root)

    def test_rejects_non_finite_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stderr = io.StringIO()
            with patch.object(sys, "argv", [
                "check_regressions.py", "--results", str(root), "--threshold", "nan"
            ]), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Thresholds must be finite", stderr.getvalue())

    def test_rejects_incoherent_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stderr = io.StringIO()
            with patch.object(sys, "argv", [
                "check_regressions.py", "--results", str(root),
                "--threshold", "10", "--critical-threshold", "5"
            ]), redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("critical-threshold must be >=", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
