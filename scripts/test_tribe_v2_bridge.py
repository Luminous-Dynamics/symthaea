#!/usr/bin/env python3
"""Contract tests for scripts/tribe_v2_bridge.py.

These tests exercise the evidence-authority boundary without requiring TRIBE v2
weights. Shadow modules provide both a released-API success fixture and a forced
import failure, so the bridge contract is deterministic in CI.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "scripts" / "tribe_v2_bridge.py"
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "tribe_v2_bridge.py requires numpy")
class TribeV2BridgeContractTests(unittest.TestCase):
    def run_bridge(
        self,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(BRIDGE), *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def shadow_pythonpath(path: Path) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(path) if not existing else f"{path}{os.pathsep}{existing}"
        )
        return env

    def test_mock_is_explicitly_synthetic_and_not_empirical_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mock.json"
            result = self.run_bridge(
                "--mock",
                "--stimulus",
                "fixture.mp4",
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(payload["source"], "SyntheticFixture")
            self.assertEqual(payload["evidence_authority"], "SyntheticFixture")
            self.assertFalse(payload["eligible_for_empirical_benchmarks"])
            self.assertFalse(payload["eligible_for_surrogate_benchmarks"])
            self.assertEqual(payload["model"], "symthaea-tribev2-mock")
            self.assertEqual(payload["coordinate_system"], "symthaea12")
            self.assertIn("synthetic_region_activations", payload)
            self.assertNotIn("region_activations", payload)
            self.assertNotIn("surface_activations", payload)

    def test_mock_is_reproducible_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"

            for output in (first, second):
                result = self.run_bridge(
                    "--mock",
                    "--stimulus",
                    "same-stimulus.mp4",
                    "--output",
                    str(output),
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            first_payload = json.loads(first.read_text(encoding="utf-8"))
            second_payload = json.loads(second.read_text(encoding="utf-8"))
            self.assertEqual(
                first_payload["synthetic_region_activations"],
                second_payload["synthetic_region_activations"],
            )

    def test_released_api_surrogate_preserves_native_surface_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Minimal stand-in for the released `tribev2.TribeModel` API. Four
            # vertices are intentional: the bridge must preserve arbitrary native
            # vertex counts rather than assuming or truncating to 360.
            (tmp_path / "tribev2.py").write_text(
                """
import numpy as np

class TribeModel:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        instance = cls()
        instance.model_id = model_id
        instance.kwargs = kwargs
        return instance

    def get_events_dataframe(self, **kwargs):
        return kwargs

    def predict(self, events):
        return np.array([
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
        ], dtype=np.float32), None
""".lstrip(),
                encoding="utf-8",
            )
            output = tmp_path / "surface.json"
            result = self.run_bridge(
                "--stimulus",
                "real.mp4",
                "--output",
                str(output),
                env=self.shadow_pythonpath(tmp_path),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "FmriPredicted")
            self.assertEqual(payload["evidence_authority"], "ExternalSurrogate")
            self.assertFalse(payload["eligible_for_empirical_benchmarks"])
            self.assertTrue(payload["eligible_for_surrogate_benchmarks"])
            self.assertEqual(payload["model"], "facebook/tribev2")
            self.assertEqual(payload["coordinate_system"], "fsaverage5")
            self.assertEqual(payload["aggregation"], "temporal_mean")
            self.assertEqual(payload["n_timesteps"], 2)
            self.assertEqual(payload["n_vertices"], 4)
            self.assertEqual(payload["surface_activations"], [2.0, 3.0, 4.0, 5.0])
            self.assertNotIn("region_activations", payload)
            self.assertNotIn("synthetic_region_activations", payload)

    def test_real_mode_import_failure_refuses_synthetic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Shadow any installed tribev2 package and force ImportError at import.
            (tmp_path / "tribev2.py").write_text(
                "raise ImportError('forced qualification failure')\n",
                encoding="utf-8",
            )
            output = tmp_path / "real.json"

            result = self.run_bridge(
                "--stimulus",
                "real.mp4",
                "--output",
                str(output),
                env=self.shadow_pythonpath(tmp_path),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing synthetic fallback", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
