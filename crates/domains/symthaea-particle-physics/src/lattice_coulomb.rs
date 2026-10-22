// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tree-level Wilson-action lattice Coulomb kernel.
//!
//! The short-distance static potential on a hypercubic lattice is not exactly
//! described by continuum `1/r`. This module evaluates the three-dimensional
//! tree-level lattice Coulomb basis used by coarse-lattice potential fits.
//! It is a numerical basis-function primitive only; it has no authority to
//! select a potential model, fit range, string tension, or physical scale.

pub const TREE_LEVEL_LATTICE_COULOMB_ID: &str =
    "tree_level_wilson_lattice_coulomb_midpoint_richardson_v1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeCoulombEstimate {
    pub convention_id: &'static str,
    pub separation: [i32; 3],
    pub base_resolution: usize,
    pub coarse: f64,
    pub fine: f64,
    pub extrapolated: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeCoulombError {
    ZeroSeparation,
    InvalidResolution(usize),
    ResolutionOverflow(usize),
    NonFiniteResult,
}

fn validate(separation: [i32; 3], resolution: usize) -> Result<(), LatticeCoulombError> {
    if separation == [0, 0, 0] {
        return Err(LatticeCoulombError::ZeroSeparation);
    }
    if resolution < 8 || resolution % 2 != 0 {
        return Err(LatticeCoulombError::InvalidResolution(resolution));
    }
    Ok(())
}

/// Midpoint-grid approximation to
///
/// `4π ∫_BZ d³k/(2π)³ cos(k·R)/(4 Σ_j sin²(k_j/2))`.
///
/// An even midpoint grid never samples the integrable `k=0` singularity
/// directly. The caller should normally use `lattice_coulomb_richardson` rather
/// than promote one raw midpoint resolution.
pub fn lattice_coulomb_midpoint(
    separation: [i32; 3],
    resolution: usize,
) -> Result<f64, LatticeCoulombError> {
    validate(separation, resolution)?;
    let n = resolution as f64;
    let step = 2.0 * std::f64::consts::PI / n;
    let [rx, ry, rz] = separation.map(f64::from);
    let mut total = 0.0;

    for ix in 0..resolution {
        let kx = -std::f64::consts::PI + (ix as f64 + 0.5) * step;
        let sx = (0.5 * kx).sin().powi(2);
        for iy in 0..resolution {
            let ky = -std::f64::consts::PI + (iy as f64 + 0.5) * step;
            let sy = (0.5 * ky).sin().powi(2);
            for iz in 0..resolution {
                let kz = -std::f64::consts::PI + (iz as f64 + 0.5) * step;
                let denominator = sx + sy + (0.5 * kz).sin().powi(2);
                let phase = kx * rx + ky * ry + kz * rz;
                total += phase.cos() / denominator;
            }
        }
    }

    let value = std::f64::consts::PI * total / n.powi(3);
    if !value.is_finite() {
        return Err(LatticeCoulombError::NonFiniteResult);
    }
    Ok(value)
}

/// Two-grid first-order Richardson extrapolation of the midpoint sequence.
pub fn lattice_coulomb_richardson(
    separation: [i32; 3],
    base_resolution: usize,
) -> Result<LatticeCoulombEstimate, LatticeCoulombError> {
    validate(separation, base_resolution)?;
    let fine_resolution = base_resolution
        .checked_mul(2)
        .ok_or(LatticeCoulombError::ResolutionOverflow(base_resolution))?;
    let coarse = lattice_coulomb_midpoint(separation, base_resolution)?;
    let fine = lattice_coulomb_midpoint(separation, fine_resolution)?;
    let extrapolated = 2.0 * fine - coarse;
    if !extrapolated.is_finite() {
        return Err(LatticeCoulombError::NonFiniteResult);
    }
    Ok(LatticeCoulombEstimate {
        convention_id: TREE_LEVEL_LATTICE_COULOMB_ID,
        separation,
        base_resolution,
        coarse,
        fine,
        extrapolated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_neighbor_matches_independent_lqcd_020g_oracle() {
        let estimate = lattice_coulomb_richardson([1, 0, 0], 64).unwrap();
        assert!((estimate.coarse - 1.054_205_059_285_06).abs() < 2.0e-12);
        assert!((estimate.fine - 1.067_862_875_493_544).abs() < 2.0e-12);
        assert!((estimate.extrapolated - 1.081_520_691_702_028).abs() < 4.0e-12);
    }

    #[test]
    fn permutation_and_sign_symmetry_hold() {
        let a = lattice_coulomb_richardson([1, 1, 0], 32).unwrap();
        let b = lattice_coulomb_richardson([0, -1, 1], 32).unwrap();
        assert!((a.extrapolated - b.extrapolated).abs() < 2.0e-12);
    }

    #[test]
    fn lattice_short_distance_is_not_silently_continuum() {
        let estimate = lattice_coulomb_richardson([1, 0, 0], 32).unwrap();
        assert!((estimate.extrapolated - 1.0).abs() > 0.05);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert_eq!(
            lattice_coulomb_richardson([0, 0, 0], 64),
            Err(LatticeCoulombError::ZeroSeparation)
        );
        assert_eq!(
            lattice_coulomb_richardson([1, 0, 0], 7),
            Err(LatticeCoulombError::InvalidResolution(7))
        );
        assert_eq!(
            lattice_coulomb_richardson([1, 0, 0], 9),
            Err(LatticeCoulombError::InvalidResolution(9))
        );
    }
}
