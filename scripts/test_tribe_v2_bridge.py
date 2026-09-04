#!/usr/bin/env python3
"""Contract tests for scripts/tribe_v2_bridge.py.

These tests exercise the evidence-authority boundary without requiring TRIBE v2
weights. Real-model import failure is forced with a shadow module so the bridge
must demonstrate fail-closed behavior deterministically.
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

    def test_real_mode_import_failure_refuses_synthetic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Shadow any installed tribev2 package and force ImportError at import.
            (tmp_path / "tribev2.py").write_text(
                "raise ImportError('forced qualification failure')\n",
                encoding="utf-8",
            )
            output = tmp_path / "real.json"
            env = os.environ.copy()
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(tmp_path) if not existing else f"{tmp_path}{os.pathsep}{existing}"
            )

            result = self.run_bridge(
                "--stimulus",
                "real.mp4",
                "--output",
                str(output),
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing synthetic fallback", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
