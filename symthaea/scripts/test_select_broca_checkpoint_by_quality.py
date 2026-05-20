#!/usr/bin/env python3
"""Regression tests for Broca checkpoint quality selection."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "scripts" / "select_broca_checkpoint_by_quality.py"


def write_report(path: Path, *, perplexity: float, checkpoint: str) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {"checkpoint_path": checkpoint},
                "gated_generation": {
                    "perplexity": perplexity,
                    "avg_coherence": 0.75,
                    "target_token_overlap": 0.5,
                },
                "code_sheaf": {
                    "gated": {
                        "function_coherence_rate": 0.8,
                        "coherence_rate": 0.8,
                    }
                },
            }
        )
    )


class SelectBrocaCheckpointByQualityTests(unittest.TestCase):
    def run_selector(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SELECTOR), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_require_trained_improvement_passes_when_trained_is_better(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline-quality.json"
            trained = root / "trained-quality.json"
            write_report(baseline, perplexity=50.0, checkpoint="baseline.bin")
            write_report(trained, perplexity=40.0, checkpoint="trained.bin")

            result = self.run_selector(
                "--json",
                "--require-trained-improvement",
                "--baseline-report",
                str(baseline),
                "--trained-report",
                str(trained),
                str(baseline),
                str(trained),
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            selection = json.loads(result.stdout)
            self.assertEqual(selection["selected_checkpoint"], "trained.bin")

    def test_require_trained_improvement_rejects_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline-quality.json"
            trained = root / "trained-quality.json"
            write_report(baseline, perplexity=40.0, checkpoint="baseline.bin")
            write_report(trained, perplexity=50.0, checkpoint="trained.bin")

            result = self.run_selector(
                "--require-trained-improvement",
                "--baseline-report",
                str(baseline),
                "--trained-report",
                str(trained),
                str(baseline),
                str(trained),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("trained report did not improve", result.stdout)


if __name__ == "__main__":
    unittest.main()
