#!/usr/bin/env python3
"""Regression tests for Broca quality summaries."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "scripts" / "summarize_broca_quality.py"


class BrocaQualitySummaryTests(unittest.TestCase):
    def test_prints_collapse_identity(self) -> None:
        report = {
            "metadata": {"backend": "cpu", "eval_lane": "fast"},
            "gated_generation": {
                "perplexity": 42.5,
                "avg_coherence": 0.25,
                "english_word_ratio": 0.5,
                "top_token_collapse_rate": 0.75,
                "top_token_collapse": {
                    "token_id": 89,
                    "token": "t",
                    "count": 3,
                    "total": 4,
                    "rate": 0.75,
                },
            },
            "raw_generation": {
                "top_token_collapse_rate": 0.5,
                "top_token_collapse": {
                    "token_id": 1,
                    "token": "<unk>",
                    "count": 2,
                    "total": 4,
                    "rate": 0.5,
                },
            },
            "delta": {"perplexity": -1.0, "avg_coherence": 0.1, "top_token_collapse_rate": 0.25},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SUMMARY), str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status=report-only backend=cpu lane=fast", result.stdout)
        self.assertIn("token='t' token_id=89 count=3/4", result.stdout)
        self.assertIn("collapse=0.2500", result.stdout)


if __name__ == "__main__":
    unittest.main()
