// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Correlated lattice-Cornell fits for a predeclared set of static-potential points.
//!
//! This module does not compute a static potential, choose a separation range,
//! choose a Coulomb basis, or infer a physical lattice spacing. It consumes
//! already-qualified `V(R)` estimates, their full covariance, and explicit
//! tree-level lattice-Coulomb values with provenance.

use std::collections::BTreeSet;

use crate::lattice_covariance::CovarianceError;
use crate::lattice_linear_gls::{CorrelatedLinearFit, correlated_linear_fit};

pub const LATTICE_CORNELL_GLS_ID: &str = "correlated_declared_range_lattice_cornell_gls_v1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeCornellPoint {
    pub separation: [i32; 3],
    pub potential: f64,
    pub lattice_coulomb: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatticeCornellFit {
    pub analysis_id: &'static str,
    pub lattice_coulomb_lineage: String,
    pub separation_vectors: Vec<[i32; 3]>,
    pub v0: f64,
    pub sigma: f64,
    pub coulomb_coefficient: f64,
    pub parameter_covariance: Vec<Vec<f64>>,
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    pub chi_square_per_dof: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeCornellError {
    MissingCoulombLineage,
    TooFewPoints(usize),
    ZeroSeparation { index: usize },
    DuplicateSeparation([i32; 3]),
    NonFinitePoint { index: usize },
    Covariance(CovarianceError),
}

impl From<CovarianceError> for LatticeCornellError {
    fn from(value: CovarianceError) -> Self {
        Self::Covariance(value)
    }
}

fn separation_norm(vector: [i32; 3]) -> f64 {
    let squared = vector
        .iter()
        .map(|component| {
            let value = *component as f64;
            value * value
        })
        .sum::<f64>();
    squared.sqrt()
}

/// Fit the explicitly supplied model
///
/// `V(R) = V0 + sigma * |R| - e * [1/R]_lat`
///
/// over exactly the points supplied by the caller. The input point list is the
/// declared fit range; this function contains no range search, point dropping,
/// or continuum-`1/R` substitution.
pub fn fit_declared_lattice_cornell(
    points: &[LatticeCornellPoint],
    covariance: &[Vec<f64>],
    lattice_coulomb_lineage: &str,
) -> Result<LatticeCornellFit, LatticeCornellError> {
    if lattice_coulomb_lineage.trim().is_empty() {
        return Err(LatticeCornellError::MissingCoulombLineage);
    }
    if points.len() <= 3 {
        return Err(LatticeCornellError::TooFewPoints(points.len()));
    }

    let mut seen = BTreeSet::new();
    let mut values = Vec::with_capacity(points.len());
    let mut design = Vec::with_capacity(points.len());
    let mut separation_vectors = Vec::with_capacity(points.len());

    for (index, point) in points.iter().copied().enumerate() {
        if point.separation == [0, 0, 0] {
            return Err(LatticeCornellError::ZeroSeparation { index });
        }
        if !seen.insert(point.separation) {
            return Err(LatticeCornellError::DuplicateSeparation(point.separation));
        }
        if !point.potential.is_finite() || !point.lattice_coulomb.is_finite() {
            return Err(LatticeCornellError::NonFinitePoint { index });
        }
        let radius = separation_norm(point.separation);
        if !radius.is_finite() || radius <= 0.0 {
            return Err(LatticeCornellError::NonFinitePoint { index });
        }
        values.push(point.potential);
        design.push(vec![1.0, radius, -point.lattice_coulomb]);
        separation_vectors.push(point.separation);
    }

    let fit = correlated_linear_fit(&values, covariance, &design)?;
    Ok(from_linear_fit(
        fit,
        lattice_coulomb_lineage,
        separation_vectors,
    ))
}

fn from_linear_fit(
    fit: CorrelatedLinearFit,
    lattice_coulomb_lineage: &str,
    separation_vectors: Vec<[i32; 3]>,
) -> LatticeCornellFit {
    LatticeCornellFit {
        analysis_id: LATTICE_CORNELL_GLS_ID,
        lattice_coulomb_lineage: lattice_coulomb_lineage.to_owned(),
        separation_vectors,
        v0: fit.parameters[0],
        sigma: fit.parameters[1],
        coulomb_coefficient: fit.parameters[2],
        parameter_covariance: fit.parameter_covariance,
        chi_square: fit.chi_square,
        degrees_of_freedom: fit.degrees_of_freedom,
        chi_square_per_dof: fit.chi_square_per_dof,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEPARATIONS: [[i32; 3]; 6] = [
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [2, 1, 0],
    ];
    const LATTICE_COULOMB: [f64; 6] = [
        1.081_520_691_702_028,
        0.538_967_317_975_499_6,
        0.346_146_798_287_502_7,
        0.693_560_259_534_994_4,
        0.547_625_982_437_823_3,
        0.451_534_104_465_745_8,
    ];
    const OFFSETS: [f64; 6] = [0.0010, -0.0015, 0.0008, -0.0007, 0.0012, -0.0004];

    fn oracle_covariance() -> Vec<Vec<f64>> {
        let variance = 0.004_f64.powi(2);
        (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| variance * 0.45_f64.powi((i as i32 - j as i32).abs()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn oracle_points(lattice_basis: bool) -> Vec<LatticeCornellPoint> {
        SEPARATIONS
            .iter()
            .copied()
            .zip(LATTICE_COULOMB)
            .zip(OFFSETS)
            .map(|((separation, lattice_coulomb), offset)| {
                let radius = separation_norm(separation);
                let potential = 0.7 + 0.18 * radius - 0.25 * lattice_coulomb + offset;
                LatticeCornellPoint {
                    separation,
                    potential,
                    lattice_coulomb: if lattice_basis {
                        lattice_coulomb
                    } else {
                        1.0 / radius
                    },
                }
            })
            .collect()
    }

    #[test]
    fn reproduces_independent_lqcd_020h_lattice_cornell_oracle() {
        let fit = fit_declared_lattice_cornell(
            &oracle_points(true),
            &oracle_covariance(),
            "lqcd-020g:8cfcafca1ba64358ba51d9c5350b3074d025ccde",
        )
        .unwrap();

        assert_eq!(fit.analysis_id, LATTICE_CORNELL_GLS_ID);
        assert_eq!(fit.degrees_of_freedom, 3);
        assert!((fit.v0 - 0.690_924_450_087_134_1).abs() < 2.0e-12);
        assert!((fit.sigma - 0.182_457_146_108_443_6).abs() < 2.0e-12);
        assert!((fit.coulomb_coefficient - 0.242_730_226_285_588_2).abs() < 2.0e-12);
        assert!((fit.chi_square - 0.559_935_102_749_612_4).abs() < 2.0e-10);
        assert!((fit.chi_square_per_dof - 0.186_645_034_249_870_8).abs() < 2.0e-10);
        assert_eq!(fit.separation_vectors.as_slice(), &SEPARATIONS);
    }

    #[test]
    fn continuum_coulomb_substitution_is_exposed_as_bad_fit() {
        let fit = fit_declared_lattice_cornell(
            &oracle_points(false),
            &oracle_covariance(),
            "negative-control:continuum-1-over-r",
        )
        .unwrap();
        assert!((fit.sigma - 0.170_401_034_408_76).abs() < 2.0e-12);
        assert!(fit.chi_square_per_dof > 6.0);
    }

    #[test]
    fn duplicate_and_zero_separations_fail_closed() {
        let covariance = vec![vec![1.0; 4]; 4];
        let mut points = vec![
            LatticeCornellPoint { separation: [1, 0, 0], potential: 1.0, lattice_coulomb: 1.0 },
            LatticeCornellPoint { separation: [2, 0, 0], potential: 1.1, lattice_coulomb: 0.5 },
            LatticeCornellPoint { separation: [3, 0, 0], potential: 1.2, lattice_coulomb: 0.3 },
            LatticeCornellPoint { separation: [0, 0, 0], potential: 1.3, lattice_coulomb: 0.2 },
        ];
        assert_eq!(
            fit_declared_lattice_cornell(&points, &covariance, "fixture"),
            Err(LatticeCornellError::ZeroSeparation { index: 3 })
        );

        points[3].separation = [1, 0, 0];
        assert_eq!(
            fit_declared_lattice_cornell(&points, &covariance, "fixture"),
            Err(LatticeCornellError::DuplicateSeparation([1, 0, 0]))
        );
    }

    #[test]
    fn missing_basis_lineage_fails_closed() {
        let covariance = oracle_covariance();
        assert_eq!(
            fit_declared_lattice_cornell(&oracle_points(true), &covariance, "   "),
            Err(LatticeCornellError::MissingCoulombLineage)
        );
    }
}
