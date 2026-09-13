// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sommer-type scales derived from correlated static-potential fits.
//!
//! This module implements the LQCD-020R convention
//!
//! `r_c = sqrt((c - e) / sigma)`
//!
//! and propagates the fit covariance appropriate to the exact declared model.
//! It does not choose a fit model, fit range, or physical scale convention.

use crate::lattice_potential_fit_family::{
    PotentialFitMember, UNIVERSAL_IR_COULOMB_COEFFICIENT,
};

pub const SOMMER_SCALE_FROM_CORRELATED_FIT_ID: &str =
    "sommer_scale_from_correlated_potential_fit_v1";
pub const FREE_V0_SIGMA_E_L_MODEL_ID: &str = "free_v0_sigma_e_l_v1";
pub const FIXED_E_PI_OVER_12_FREE_L_MODEL_ID: &str = "fixed_e_pi_over_12_free_l_v1";
pub const FIXED_E_PI_OVER_12_L0_MODEL_ID: &str = "fixed_e_pi_over_12_l0_v1";

pub const R0_C: f64 = 1.65;
pub const R4_C: f64 = 4.0;
pub const R6_C: f64 = 6.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SommerScaleEstimate {
    pub c: f64,
    pub value: f64,
    pub standard_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardSommerScales {
    pub r0: SommerScaleEstimate,
    pub r4: SommerScaleEstimate,
    pub r6: SommerScaleEstimate,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SommerScaleError {
    NonFiniteInput,
    InvalidSigma(f64),
    ScaleOutsideModelDomain { c: f64, e: f64 },
    UnknownModelId(String),
    CovarianceDimensionMismatch { expected: usize, actual: usize },
    CovarianceRowDimensionMismatch { row: usize, expected: usize, actual: usize },
    NonFiniteCovariance { row: usize, column: usize },
    NonSymmetricCovariance { row: usize, column: usize },
    FixedCoulombCoefficientMismatch { expected: f64, actual: f64 },
    FixedLatticeArtifactMismatch { expected: f64, actual: f64 },
    InvalidPropagatedVariance(f64),
}

fn validate_member_scalars(member: &PotentialFitMember) -> Result<(), SommerScaleError> {
    let values = [
        member.v0,
        member.sigma,
        member.coulomb_coefficient,
        member.lattice_artifact_coefficient,
        member.chi_square,
        member.chi_square_per_dof,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SommerScaleError::NonFiniteInput);
    }
    if member.sigma <= 0.0 {
        return Err(SommerScaleError::InvalidSigma(member.sigma));
    }
    Ok(())
}

fn validate_covariance(
    covariance: &[Vec<f64>],
    expected: usize,
) -> Result<(), SommerScaleError> {
    if covariance.len() != expected {
        return Err(SommerScaleError::CovarianceDimensionMismatch {
            expected,
            actual: covariance.len(),
        });
    }
    for (row_index, row) in covariance.iter().enumerate() {
        if row.len() != expected {
            return Err(SommerScaleError::CovarianceRowDimensionMismatch {
                row: row_index,
                expected,
                actual: row.len(),
            });
        }
        for (column_index, value) in row.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(SommerScaleError::NonFiniteCovariance {
                    row: row_index,
                    column: column_index,
                });
            }
        }
    }
    for row in 0..expected {
        for column in (row + 1)..expected {
            let a = covariance[row][column];
            let b = covariance[column][row];
            let scale = a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > 1.0e-12 * scale {
                return Err(SommerScaleError::NonSymmetricCovariance { row, column });
            }
        }
    }
    Ok(())
}

fn require_fixed_e(member: &PotentialFitMember) -> Result<(), SommerScaleError> {
    if member.coulomb_coefficient.to_bits() != UNIVERSAL_IR_COULOMB_COEFFICIENT.to_bits() {
        return Err(SommerScaleError::FixedCoulombCoefficientMismatch {
            expected: UNIVERSAL_IR_COULOMB_COEFFICIENT,
            actual: member.coulomb_coefficient,
        });
    }
    Ok(())
}

fn require_fixed_l_zero(member: &PotentialFitMember) -> Result<(), SommerScaleError> {
    if member.lattice_artifact_coefficient.to_bits() != 0.0_f64.to_bits() {
        return Err(SommerScaleError::FixedLatticeArtifactMismatch {
            expected: 0.0,
            actual: member.lattice_artifact_coefficient,
        });
    }
    Ok(())
}

fn central_value(c: f64, sigma: f64, e: f64) -> Result<f64, SommerScaleError> {
    if !c.is_finite() || !sigma.is_finite() || !e.is_finite() {
        return Err(SommerScaleError::NonFiniteInput);
    }
    if sigma <= 0.0 {
        return Err(SommerScaleError::InvalidSigma(sigma));
    }
    if c <= e {
        return Err(SommerScaleError::ScaleOutsideModelDomain { c, e });
    }
    let value = ((c - e) / sigma).sqrt();
    if !value.is_finite() || value <= 0.0 {
        return Err(SommerScaleError::NonFiniteInput);
    }
    Ok(value)
}

fn free_e_estimate(
    member: &PotentialFitMember,
    c: f64,
) -> Result<SommerScaleEstimate, SommerScaleError> {
    validate_covariance(&member.parameter_covariance, 4)?;
    let value = central_value(c, member.sigma, member.coulomb_coefficient)?;
    let d_sigma = -value / (2.0 * member.sigma);
    let d_e = -1.0 / (2.0 * member.sigma * value);
    let var_sigma = member.parameter_covariance[1][1];
    let var_e = member.parameter_covariance[2][2];
    let cov_sigma_e = member.parameter_covariance[1][2];
    let variance = d_sigma * d_sigma * var_sigma
        + d_e * d_e * var_e
        + 2.0 * d_sigma * d_e * cov_sigma_e;
    if !variance.is_finite() || variance < 0.0 {
        return Err(SommerScaleError::InvalidPropagatedVariance(variance));
    }
    Ok(SommerScaleEstimate {
        c,
        value,
        standard_error: variance.sqrt(),
    })
}

fn fixed_e_estimate(
    member: &PotentialFitMember,
    c: f64,
    covariance_dimension: usize,
) -> Result<SommerScaleEstimate, SommerScaleError> {
    require_fixed_e(member)?;
    validate_covariance(&member.parameter_covariance, covariance_dimension)?;
    let value = central_value(c, member.sigma, member.coulomb_coefficient)?;
    let d_sigma = -value / (2.0 * member.sigma);
    let variance = d_sigma * d_sigma * member.parameter_covariance[1][1];
    if !variance.is_finite() || variance < 0.0 {
        return Err(SommerScaleError::InvalidPropagatedVariance(variance));
    }
    Ok(SommerScaleEstimate {
        c,
        value,
        standard_error: variance.sqrt(),
    })
}

/// Derive one Sommer-type scale from one exact predeclared fit member.
///
/// The covariance interpretation is model-ID-specific and fails closed for an
/// unknown model identity rather than guessing parameter ordering.
pub fn sommer_scale_from_fit_member(
    member: &PotentialFitMember,
    c: f64,
) -> Result<SommerScaleEstimate, SommerScaleError> {
    validate_member_scalars(member)?;
    match member.model_id {
        FREE_V0_SIGMA_E_L_MODEL_ID => free_e_estimate(member, c),
        FIXED_E_PI_OVER_12_FREE_L_MODEL_ID => fixed_e_estimate(member, c, 3),
        FIXED_E_PI_OVER_12_L0_MODEL_ID => {
            require_fixed_l_zero(member)?;
            fixed_e_estimate(member, c, 2)
        }
        other => Err(SommerScaleError::UnknownModelId(other.to_owned())),
    }
}

/// Derive the conventional `r0`, `r4`, and `r6` scale triplet.
pub fn standard_sommer_scales_from_fit_member(
    member: &PotentialFitMember,
) -> Result<StandardSommerScales, SommerScaleError> {
    Ok(StandardSommerScales {
        r0: sommer_scale_from_fit_member(member, R0_C)?,
        r4: sommer_scale_from_fit_member(member, R4_C)?,
        r6: sommer_scale_from_fit_member(member, R6_C)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_fixture() -> PotentialFitMember {
        let mut covariance = vec![vec![0.0; 4]; 4];
        covariance[1][1] = 2.5e-5;
        covariance[2][2] = 4.0e-5;
        covariance[1][2] = -1.2e-5;
        covariance[2][1] = -1.2e-5;
        PotentialFitMember {
            model_id: FREE_V0_SIGMA_E_L_MODEL_ID,
            v0: 0.7,
            sigma: 0.18,
            coulomb_coefficient: 0.25,
            lattice_artifact_coefficient: 0.04,
            parameter_covariance: covariance,
            chi_square: 1.0,
            degrees_of_freedom: 10,
            chi_square_per_dof: 0.1,
        }
    }

    fn fixed_fixture(model_id: &'static str, dimension: usize, l: f64) -> PotentialFitMember {
        let mut covariance = vec![vec![0.0; dimension]; dimension];
        covariance[1][1] = 1.6e-5;
        PotentialFitMember {
            model_id,
            v0: 0.7,
            sigma: 0.18,
            coulomb_coefficient: UNIVERSAL_IR_COULOMB_COEFFICIENT,
            lattice_artifact_coefficient: l,
            parameter_covariance: covariance,
            chi_square: 1.0,
            degrees_of_freedom: 10,
            chi_square_per_dof: 0.1,
        }
    }

    #[test]
    fn reproduces_independent_lqcd_020r_free_e_fixture() {
        let scales = standard_sommer_scales_from_fit_member(&free_fixture()).unwrap();
        assert!((scales.r0.value - 2.788_866_755_113_585).abs() < 2.0e-14);
        assert!((scales.r0.standard_error - 0.036_808_155_210_842_55).abs() < 2.0e-14);
        assert!((scales.r4.value - 4.564_354_645_876_384).abs() < 2.0e-14);
        assert!((scales.r4.standard_error - 0.062_035_516_841_517_8).abs() < 2.0e-14);
        assert!((scales.r6.value - 5.651_941_652_604_39).abs() < 2.0e-14);
        assert!((scales.r6.standard_error - 0.077_373_118_209_638_4).abs() < 2.0e-14);
    }

    #[test]
    fn reproduces_independent_lqcd_020r_fixed_e_fixture() {
        for member in [
            fixed_fixture(FIXED_E_PI_OVER_12_FREE_L_MODEL_ID, 3, 0.04),
            fixed_fixture(FIXED_E_PI_OVER_12_L0_MODEL_ID, 2, 0.0),
        ] {
            let scales = standard_sommer_scales_from_fit_member(&member).unwrap();
            assert!((scales.r0.value - 2.777_089_415_798_141).abs() < 2.0e-14);
            assert!((scales.r0.standard_error - 0.030_856_549_064_423_786).abs() < 2.0e-14);
            assert!((scales.r4.value - 4.557_168_109_571_295).abs() < 2.0e-14);
            assert!((scales.r4.standard_error - 0.050_635_201_217_458_83).abs() < 2.0e-14);
            assert!((scales.r6.value - 5.646_139_591_792_318).abs() < 2.0e-14);
            assert!((scales.r6.standard_error - 0.062_734_884_353_247_98).abs() < 2.0e-14);
        }
    }

    #[test]
    fn covariance_cross_term_is_not_optional_for_free_e_model() {
        let with_covariance = sommer_scale_from_fit_member(&free_fixture(), R0_C).unwrap();
        let mut no_covariance = free_fixture();
        no_covariance.parameter_covariance[1][2] = 0.0;
        no_covariance.parameter_covariance[2][1] = 0.0;
        let without_covariance = sommer_scale_from_fit_member(&no_covariance, R0_C).unwrap();
        assert!((with_covariance.standard_error - without_covariance.standard_error).abs() > 1.0e-4);
        assert!((without_covariance.standard_error - 0.039_243_158_323_593_944).abs() < 2.0e-14);
    }

    #[test]
    fn fit_family_model_ids_match_the_scale_dispatch_contract() {
        assert_eq!(FREE_V0_SIGMA_E_L_MODEL_ID, "free_v0_sigma_e_l_v1");
        assert_eq!(FIXED_E_PI_OVER_12_FREE_L_MODEL_ID, "fixed_e_pi_over_12_free_l_v1");
        assert_eq!(FIXED_E_PI_OVER_12_L0_MODEL_ID, "fixed_e_pi_over_12_l0_v1");
    }

    #[test]
    fn forged_public_records_fail_closed() {
        let mut wrong = fixed_fixture(FIXED_E_PI_OVER_12_FREE_L_MODEL_ID, 3, 0.04);
        wrong.coulomb_coefficient = 0.2;
        assert!(matches!(
            sommer_scale_from_fit_member(&wrong, R0_C),
            Err(SommerScaleError::FixedCoulombCoefficientMismatch { .. })
        ));

        let mut wrong_l = fixed_fixture(FIXED_E_PI_OVER_12_L0_MODEL_ID, 2, 0.1);
        assert!(matches!(
            sommer_scale_from_fit_member(&wrong_l, R0_C),
            Err(SommerScaleError::FixedLatticeArtifactMismatch { .. })
        ));
        wrong_l.lattice_artifact_coefficient = 0.0;
        wrong_l.parameter_covariance.push(vec![0.0, 0.0]);
        assert!(matches!(
            sommer_scale_from_fit_member(&wrong_l, R0_C),
            Err(SommerScaleError::CovarianceDimensionMismatch { .. })
        ));
    }
}
