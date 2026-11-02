# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free contract tests for the PyPhi reference runner's input identity."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from validation import generate_pyphi_reference_fixtures as runner


ROOT = Path(__file__).parent
FIXTURE = ROOT / "fixtures" / "pyphi-iit4-2023-fig1a-reference-v0.1.json"


class ArrayLike:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeFrame:
    def __init__(self, rows, index):
        self.values = ArrayLike(rows)
        self.index = ArrayLike(index)


class FakeTpm:
    def __init__(self, rows, index):
        self._frame = FakeFrame(rows, index)

    def to_pandas(self):
        return self._frame


class FakeSubstrate:
    def __init__(self, rows=None, index=None):
        self.tpm = FakeTpm(
            rows or [
                [0.1, 0.2],
                [0.9, 0.2],
                [0.1, 0.8],
                [0.9, 0.8],
            ],
            index or [(0, 0), (1, 0), (0, 1), (1, 1)],
        )
        self.cm = ArrayLike([[1, 1], [0, 1]])
        self.node_labels = ("A", "B")


class PyPhiReferenceRunnerContractTests(unittest.TestCase):
    def test_reviewed_anchor_set_has_exact_expected_identity(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        reference = fixture["reference"]
        self.assertEqual(reference["exact_version"], runner.PINNED_PYPHI_VERSION)
        self.assertEqual(reference["api_example"], runner.EXPECTED_EXAMPLE)
        self.assertEqual(reference["formalism"], runner.EXPECTED_FORMALISM)
        self.assertEqual(tuple(reference["state"]), runner.EXPECTED_STATE)
        actual = {item["id"]: item["quantity_id"] for item in fixture["quantities"]}
        self.assertEqual(actual, runner.EXPECTED_QUANTITY_IDS)

    def test_model_and_evaluation_digest_match_rust_golden_vector(self):
        digest, evaluation, _manifest, _model, _bytes = runner._model_digest(
            FakeSubstrate(), (1, 0), "IIT_4_0_2023"
        )
        self.assertEqual(
            digest,
            "728eb74b44c4aec7896db360919b6efcb0f06e4b7eb487b2974c486c42a80b62",
        )
        self.assertEqual(
            evaluation,
            "6188b51af953ac133cd3c4af0d628ceba73692ff36c2f37f6e05bdb1531214e2",
        )

    def test_wrong_tpm_row_order_is_rejected_before_digesting(self):
        fake = FakeSubstrate(index=[(0, 0), (0, 1), (1, 0), (1, 1)])
        with self.assertRaisesRegex(RuntimeError, "TPM row order does not match"):
            runner._model_digest(fake, (1, 0), "IIT_4_0_2023")

    def test_truncated_anchor_catalogue_fails_before_importing_pyphi(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        fixture["quantities"] = fixture["quantities"][:-1]
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "short.json"
            output_path = Path(directory) / "should-not-exist.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            with patch.object(runner, "FIXTURE_PATH", fixture_path), patch.object(
                runner.importlib.metadata,
                "version",
                return_value=runner.PINNED_PYPHI_VERSION,
            ):
                with self.assertRaisesRegex(RuntimeError, "exactly the six reviewed"):
                    runner.run(output_path)
            self.assertFalse(output_path.exists())

    def test_fixture_with_unknown_quantity_is_rejected(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        fixture["quantities"][0]["quantity_id"] = "an-unreviewed-scalar"
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "wrong-quantity.json"
            output_path = Path(directory) / "should-not-exist.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            with patch.object(runner, "FIXTURE_PATH", fixture_path), patch.object(
                runner.importlib.metadata,
                "version",
                return_value=runner.PINNED_PYPHI_VERSION,
            ):
                with self.assertRaisesRegex(RuntimeError, "quantity id mismatch"):
                    runner.run(output_path)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
