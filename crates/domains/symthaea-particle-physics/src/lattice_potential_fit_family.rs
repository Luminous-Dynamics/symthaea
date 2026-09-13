// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Declared correlated fit family for the Wilson static potential.
//!
//! This module evaluates several caller-independent, predeclared linear models
//! against exactly the same potential points and covariance. It does not rank,
//! select, average, or otherwise choose a preferred model.

use std::collections::BTreeSet;

use crate::lattice_cornell::LatticeCornellPoint;
use crate::lattice_covariance::CovarianceError;
use crate::lattice_linear_gls::{CorrelatedLinearFit, correlated_linear_fit};

pub const LATTICE_POTENTIAL_FIT_FAMILY_ID: &str =
    "declared_lattice_potential_fit_family_gls_v1";
pub const UNIVERSAL_IR_COULOMB_COEFFICIENT: f64 = std::f64::consts::PI / 12.0;

#[derive(Debug, Clone, PartialEq)]
pub struct PotentialFitMember {
    pub model_id: &'static str,
    pub v0: f64,
    pub sigma: f64,
    pub coulomb_coefficient: f64,
    pub lattice_artifact_coefficient: f64,
    pub parameter_covariance: Vec<Vec<f64>>,
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    pub chi_square_per_dof: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatticePotentialFitFamily {
    pub family_id: &'static str,
    pub lattice_coulomb_lineage: String,
    pub separation_vectors: Vec<[i32; 3]>,
    pub free_four_parameter: PotentialFitMember,
    pub fixed_e_three_parameter: PotentialFitMember,
    pub fixed_e_l0_two_parameter: PotentialFitMember,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PotentialFitFamilyError {
    MissingCoulombLineage,
    TooFewPoints(usize),
    ZeroSeparation { index: usize },
    DuplicateSeparation([i32; 3]),
    NonFinitePoint { index: usize },
    Covariance(CovarianceError),
}

impl From<CovarianceError> for PotentialFitFamilyError {
    fn from(value: CovarianceError) -> Self {
        Self::Covariance(value)
    }
}

fn separation_norm(vector: [i32; 3]) -> f64 {
    vector
        .iter()
        .map(|component| {
            let value = *component as f64;
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

fn validate_points(
    points: &[LatticeCornellPoint],
    lattice_coulomb_lineage: &str,
) -> Result<(Vec<f64>, Vec<[i32; 3]>), PotentialFitFamilyError> {
    if lattice_coulomb_lineage.trim().is_empty() {
        return Err(PotentialFitFamilyError::MissingCoulombLineage);
    }
    if points.len() <= 4 {
        return Err(PotentialFitFamilyError::TooFewPoints(points.len()));
    }

    let mut seen = BTreeSet::new();
    let mut radii = Vec::with_capacity(points.len());
    let mut vectors = Vec::with_capacity(points.len());
    for (index, point) in points.iter().copied().enumerate() {
        if point.separation == [0, 0, 0] {
            return Err(PotentialFitFamilyError::ZeroSeparation { index });
        }
        if !seen.insert(point.separation) {
            return Err(PotentialFitFamilyError::DuplicateSeparation(point.separation));
        }
        if !point.potential.is_finite() || !point.lattice_coulomb.is_finite() {
            return Err(PotentialFitFamilyError::NonFinitePoint { index });
        }
        let radius = separation_norm(point.separation);
        if !radius.is_finite() || radius <= 0.0 {
            return Err(PotentialFitFamilyError::NonFinitePoint { index });
        }
        radii.push(radius);
        vectors.push(point.separation);
    }
    Ok((radii, vectors))
}

fn member_from_free_four(fit: CorrelatedLinearFit) -> PotentialFitMember {
    PotentialFitMember {
        model_id: "free_v0_sigma_e_l_v1",
        v0: fit.parameters[0],
        sigma: fit.parameters[1],
        coulomb_coefficient: fit.parameters[2],
        lattice_artifact_coefficient: fit.parameters[3],
        parameter_covariance: fit.parameter_covariance,
        chi_square: fit.chi_square,
        degrees_of_freedom: fit.degrees_of_freedom,
        chi_square_per_dof: fit.chi_square_per_dof,
    }
}

fn member_from_fixed_e_three(fit: CorrelatedLinearFit) -> PotentialFitMember {
    PotentialFitMember {
        model_id: "fixed_e_pi_over_12_free_l_v1",
        v0: fit.parameters[0],
        sigma: fit.parameters[1],
        coulomb_coefficient: UNIVERSAL_IR_COULOMB_COEFFICIENT,
        lattice_artifact_coefficient: fit.parameters[2],
        parameter_covariance: fit.parameter_covariance,
        chi_square: fit.chi_square,
        degrees_of_freedom: fit.degrees_of_freedom,
        chi_square_per_dof: fit.chi_square_per_dof,
    }
}

fn member_from_fixed_e_l0_two(fit: CorrelatedLinearFit) -> PotentialFitMember {
    PotentialFitMember {
        model_id: "fixed_e_pi_over_12_l0_v1",
        v0: fit.parameters[0],
        sigma: fit.parameters[1],
        coulomb_coefficient: UNIVERSAL_IR_COULOMB_COEFFICIENT,
        lattice_artifact_coefficient: 0.0,
        parameter_covariance: fit.parameter_covariance,
        chi_square: fit.chi_square,
        degrees_of_freedom: fit.degrees_of_freedom,
        chi_square_per_dof: fit.chi_square_per_dof,
    }
}

/// Evaluate the predeclared static-potential fit family on exactly one point set.
///
/// Models:
/// 1. `V0 + sigma*r - e*C_lat + l*(C_lat - 1/r)`;
/// 2. the same with `e = pi/12` fixed;
/// 3. `V0 + sigma*r - (pi/12)*C_lat` with `l = 0`.
///
/// No model-selection result is returned. Model consistency is a downstream,
/// preregistered evidence question.
pub fn fit_declared_lattice_potential_family(
    points: &[LatticeCornellPoint],
    covariance: &[Vec<f64>],
    lattice_coulomb_lineage: &str,
) -> Result<LatticePotentialFitFamily, PotentialFitFamilyError> {
    let (radii, separation_vectors) = validate_points(points, lattice_coulomb_lineage)?;
    let values = points.iter().map(|point| point.potential).collect::<Vec<_>>();

    let free_four_design = points
        .iter()
        .zip(&radii)
        .map(|(point, radius)| {
            vec![
                1.0,
                *radius,
                -point.lattice_coulomb,
                point.lattice_coulomb - 1.0 / radius,
            ]
        })
        .collect::<Vec<_>>();
    let free_four = correlated_linear_fit(&values, covariance, &free_four_design)?;

    let fixed_e_values = points
        .iter()
        .zip(&values)
        .map(|(point, value)| value + UNIVERSAL_IR_COULOMB_COEFFICIENT * point.lattice_coulomb)
        .collect::<Vec<_>>();
    let fixed_e_three_design = points
        .iter()
        .zip(&radii)
        .map(|(point, radius)| vec![1.0, *radius, point.lattice_coulomb - 1.0 / radius])
        .collect::<Vec<_>>();
    let fixed_e_three =
        correlated_linear_fit(&fixed_e_values, covariance, &fixed_e_three_design)?;

    let fixed_e_l0_two_design = radii
        .iter()
        .map(|radius| vec![1.0, *radius])
        .collect::<Vec<_>>();
    let fixed_e_l0_two =
        correlated_linear_fit(&fixed_e_values, covariance, &fixed_e_l0_two_design)?;

    Ok(LatticePotentialFitFamily {
        family_id: LATTICE_POTENTIAL_FIT_FAMILY_ID,
        lattice_coulomb_lineage: lattice_coulomb_lineage.to_owned(),
        separation_vectors,
        free_four_parameter: member_from_free_four(free_four),
        fixed_e_three_parameter: member_from_fixed_e_three(fixed_e_three),
        fixed_e_l0_two_parameter: member_from_fixed_e_l0_two(fixed_e_l0_two),
    })
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

    fn covariance() -> Vec<Vec<f64>> {
        let variance = 0.004_f64.powi(2);
        (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| variance * 0.45_f64.powi((i as i32 - j as i32).abs()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn points() -> Vec<LatticeCornellPoint> {
        SEPARATIONS
            .iter()
            .copied()
            .zip(LATTICE_COULOMB)
            .zip(OFFSETS)
            .map(|((separation, lattice_coulomb), offset)| {
                let radius = separation_norm(separation);
                let potential = 0.7
                    + 0.18 * radius
                    - 0.25 * lattice_coulomb
                    + 0.04 * (lattice_coulomb - 1.0 / radius)
                    + offset;
                LatticeCornellPoint {
                    separation,
                    potential,
                    lattice_coulomb,
                }
            })
            .collect()
    }

    #[test]
    fn reproduces_independent_lqcd_020o_fit_family() {
        let family = fit_declared_lattice_potential_family(
            &points(),
            &covariance(),
            "lqcd-020g:8cfcafca1ba64358ba51d9c5350b3074d025ccde",
        )
        .unwrap();

        assert_eq!(family.family_id, LATTICE_POTENTIAL_FIT_FAMILY_ID);
        assert_eq!(family.separation_vectors, SEPARATIONS);

        let free = &family.free_four_parameter;
        assert!((free.v0 - 0.681_872_546_874_426_9).abs() < 2.0e-12);
        assert!((free.sigma - 0.184_767_000_423_900_7).abs() < 2.0e-12);
        assert!((free.coulomb_coefficient - 0.233_826_425_701_892_05).abs() < 2.0e-12);
        assert!((free.lattice_artifact_coefficient + 0.002_738_126_913_273_575_3).abs() < 2.0e-12);
        assert!((free.chi_square_per_dof - 0.135_661_394_403_886_55).abs() < 2.0e-10);

        let fixed_three = &family.fixed_e_three_parameter;
        assert!((fixed_three.sigma - 0.176_935_000_203_852_83).abs() < 2.0e-12);
        assert!((fixed_three.lattice_artifact_coefficient - 0.078_723_929_919_030_77).abs() < 2.0e-12);
        assert!((fixed_three.chi_square_per_dof - 0.666_539_133_548_737_4).abs() < 2.0e-10);

        let fixed_two = &family.fixed_e_l0_two_parameter;
        assert!((fixed_two.sigma - 0.176_077_574_252_747_84).abs() < 2.0e-12);
        assert_eq!(fixed_two.lattice_artifact_coefficient, 0.0);
        assert!((fixed_two.chi_square_per_dof - 1.122_373_588_370_452_2).abs() < 2.0e-10);
    }

    #[test]
    fn family_exposes_models_without_selecting_one() {
        let family = fit_declared_lattice_potential_family(&points(), &covariance(), "fixture").unwrap();
        let sigmas = [
            family.free_four_parameter.sigma,
            family.fixed_e_three_parameter.sigma,
            family.fixed_e_l0_two_parameter.sigma,
        ];
        assert!(sigmas.iter().all(|sigma| (sigma - 0.18).abs() < 0.006));
    }

    #[test]
    fn malformed_point_program_fails_closed() {
        let mut bad = points();
        bad[5].separation = bad[0].separation;
        assert_eq!(
            fit_declared_lattice_potential_family(&bad, &covariance(), "fixture"),
            Err(PotentialFitFamilyError::DuplicateSeparation([1, 0, 0]))
        );
        assert_eq!(
            fit_declared_lattice_potential_family(&points()[..4], &covariance()[..4], "fixture"),
            Err(PotentialFitFamilyError::TooFewPoints(4))
        );
    }
}
