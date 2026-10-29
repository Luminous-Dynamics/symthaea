#!/usr/bin/env python3
"""Regression tests for the radial rule used by the PySCF matcher.

The Gaussian integral is analytic:
    integral_R3 exp(-alpha * |r|^2) d^3r = (pi / alpha)^(3/2)

This tests the actual custom PySCF radial callback before any molecular SCF
runs; it is independent of solver convergence and uses no tuned references.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_dft_molecular_grid_pyscf import (
    GRID_SETTINGS,
    symthaea_becke_radial,
    symthaea_radial_scale_bohr,
)


class SymthaeaRadialRuleTests(unittest.TestCase):
    def test_grid_settings_have_expected_product(self) -> None:
        expected_points = {
            "coarse": 120,
            "medium": 910,
            "fine": 2500,
            "high_accuracy": 22650,
        }
        for quality, (radial_count, angular_count) in GRID_SETTINGS.items():
            with self.subTest(quality=quality):
                self.assertEqual(radial_count * angular_count, expected_points[quality])

    def test_radial_scale_policy_is_explicit(self) -> None:
        expected_scales = {
            1: 0.661,
            2: 0.567,
            3: 2.268,
            4: 2.268,
            5: 1.323,
            8: 1.323,
            10: 1.323,
            11: 1.5,
            26: 1.5,
        }
        for atomic_number, expected in expected_scales.items():
            with self.subTest(atomic_number=atomic_number):
                self.assertEqual(symthaea_radial_scale_bohr(atomic_number), expected)
        with self.assertRaises(ValueError):
            symthaea_radial_scale_bohr(0)

    def test_radial_nodes_and_weights_are_finite_positive(self) -> None:
        for quality, (radial_count, _angular_count) in GRID_SETTINGS.items():
            for atomic_number in (1, 2, 8):
                radii, radial_weights = symthaea_becke_radial(radial_count, atomic_number)
                with self.subTest(quality=quality, atomic_number=atomic_number):
                    self.assertEqual(len(radii), radial_count)
                    self.assertEqual(len(radial_weights), radial_count)
                    self.assertTrue(np.all(np.isfinite(radii)))
                    self.assertTrue(np.all(np.isfinite(radial_weights)))
                    self.assertTrue(np.all(radii > 0.0))
                    self.assertTrue(np.all(radial_weights > 0.0))

    def test_radial_quadrature_reproduces_analytic_gaussian_integrals(self) -> None:
        # Tolerances are explicit by preset and cover the complete H/He/O scale
        # range used in this molecular comparison, not only the hydrogen scale.
        max_relative_error = {
            "coarse": 2e-3,
            "medium": 2e-6,
            "fine": 5e-8,
            "high_accuracy": 5e-11,
        }
        alphas = (0.25, 0.5, 1.0, 2.0, 4.0)
        for quality, (radial_count, _angular_count) in GRID_SETTINGS.items():
            for atomic_number in (1, 2, 8):
                radii, radial_weights = symthaea_becke_radial(radial_count, atomic_number)
                for alpha in alphas:
                    with self.subTest(
                        quality=quality, atomic_number=atomic_number, alpha=alpha
                    ):
                        observed = float(
                            np.sum(
                                4.0
                                * math.pi
                                * radii**2
                                * radial_weights
                                * np.exp(-alpha * radii**2)
                            )
                        )
                        exact = (math.pi / alpha) ** 1.5
                        relative_error = abs(observed - exact) / exact
                        self.assertLessEqual(
                            relative_error,
                            max_relative_error[quality],
                            f"Z={atomic_number}, {quality}, alpha={alpha}: "
                            f"relative error {relative_error:.6e}",
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
