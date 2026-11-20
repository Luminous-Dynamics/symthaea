"""Tests for canonical Boolean-network manifest and cross-language model identity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from shared_boolean_network import (
    BooleanNetworkCase,
    BooleanRule,
    f64_from_hex,
    f64_to_hex,
    load_model_manifest,
    model_digest,
    prior_digest,
    transition_tpm,
)


class SharedBooleanNetworkTests(unittest.TestCase):
    def test_manifest_is_complete_and_unique(self) -> None:
        cases = load_model_manifest()
        names = [case.name for case in cases]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(cases), 10)
        self.assertIn("recurrent_ring_uniform_prior", names)
        self.assertIn("recurrent_ring_skewed_prior", names)

    def test_binary64_round_trip_is_exact(self) -> None:
        for value in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7):
            self.assertEqual(f64_from_hex(f64_to_hex(value)), value)

    def test_canonical_specification_matches_rust_fixture_encoding(self) -> None:
        cases = {case.name: case for case in load_model_manifest()}
        case = cases["ring_noise_0.10"]
        self.assertEqual(
            case.canonical_model_specification(),
            "schema=boolean-network-v1;rules=copy:1,copy:2,copy:0;"
            "flip=0.100000;flip_bits=3fb999999999999a",
        )

    def test_same_rules_different_prior_preserves_model_digest(self) -> None:
        cases = {case.name: case for case in load_model_manifest()}
        uniform = cases["recurrent_ring_uniform_prior"]
        skewed = cases["recurrent_ring_skewed_prior"]
        self.assertEqual(uniform.canonical_model_specification(), skewed.canonical_model_specification())
        self.assertEqual(model_digest(uniform), model_digest(skewed))
        self.assertNotEqual(prior_digest(uniform), prior_digest(skewed))
        self.assertNotEqual(uniform.canonical_prior_specification(), skewed.canonical_prior_specification())

    def test_tpm_rows_match_expected_noise_extremes(self) -> None:
        cases = {case.name: case for case in load_model_manifest()}
        clean = transition_tpm(cases["recurrent_ring_uniform_prior"])
        noisy = transition_tpm(cases["ring_noise_0.50"])
        self.assertTrue(np.all((clean == 0.0) | (clean == 1.0)))
        np.testing.assert_allclose(noisy, 0.5, atol=0.0, rtol=0.0)

    def test_every_tpm_entry_is_finite_and_bounded(self) -> None:
        for case in load_model_manifest():
            tpm = transition_tpm(case)
            self.assertEqual(tpm.shape, (1 << case.node_count, case.node_count))
            self.assertTrue(np.all(np.isfinite(tpm)))
            self.assertTrue(np.all((tpm >= 0.0) & (tpm <= 1.0)))

    def test_invalid_prior_mass_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "bad.tsv"
            manifest.write_text(
                "# test manifest\n"
                "bad_prior\t2\tcopy:0,copy:1\t0000000000000000\t"
                "sparse:0:3fe0000000000000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sum to 1"):
                load_model_manifest(manifest)

    def test_noise_above_half_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "bad.tsv"
            manifest.write_text(
                "bad_noise\t2\tcopy:0,copy:1\t3fe8000000000000\tuniform\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "flip probability"):
                load_model_manifest(manifest)

    def test_model_digest_is_deterministic(self) -> None:
        case = load_model_manifest()[0]
        self.assertEqual(model_digest(case), model_digest(case))


if __name__ == "__main__":
    unittest.main()
