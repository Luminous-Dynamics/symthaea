// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Newtonian baryonic-only model — the null hypothesis.
//!
//! Predicts V(r) = √(V_bar²(r)) with no dark component and no modification
//! of gravity. Expected to fail badly in the outer regions of most galaxies;
//! that failure is the rotation-curve problem itself, and every other model
//! is judged by how much of it they repair.

use super::{FittedCurve, RotationModel, curve_chi2, v_baryonic_sq};
use crate::sparc::Galaxy;

pub struct Newtonian;

impl RotationModel for Newtonian {
    fn name(&self) -> &'static str {
        "newtonian_baryonic"
    }

    fn n_free_params(&self) -> usize {
        0
    }

    fn fit(&self, galaxy: &Galaxy) -> FittedCurve {
        let v_pred: Vec<f64> = galaxy
            .points
            .iter()
            .map(|p| v_baryonic_sq(p).max(0.0).sqrt())
            .collect();
        let chi2 = curve_chi2(galaxy, &v_pred);
        FittedCurve {
            v_pred,
            params: vec![],
            chi2,
            converged: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_models::test_util::galaxy_from_rows;

    #[test]
    fn predicts_pure_baryonic_velocity() {
        // v_gas=30, v_disk=40 → v² = 900 + 0.5·1600 = 1700
        let g = galaxy_from_rows(&[(2.0, 45.0, 2.0, 30.0, 40.0)]);
        let fit = Newtonian.fit(&g);
        assert!((fit.v_pred[0] - 1700.0_f64.sqrt()).abs() < 1e-12);
        assert!(fit.converged);
        assert!(fit.params.is_empty());
    }

    #[test]
    fn negative_baryonic_sq_clamps_to_zero_velocity() {
        let g = galaxy_from_rows(&[(0.2, 5.0, 2.0, -20.0, 1.0)]);
        let fit = Newtonian.fit(&g);
        assert_eq!(fit.v_pred[0], 0.0);
    }
}
