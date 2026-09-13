// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spatial APE smearing for Wilson-loop / spectroscopy operator construction.
//!
//! This module implements the convention independently frozen by LQCD-020A:
//!
//! `M_k(x) = alpha U_k(x) + sum of four spatial staples transverse to k`,
//!
//! for spatial directions `k = 0,1,2`.  The candidate is projected to SU(3)
//! using the unitary polar factor followed by determinant-phase removal.
//! Temporal links are copied unchanged and each iteration is synchronous.
//!
//! **Authority boundary:** smeared links are measurement/operator data.  This
//! API returns a new field and does not mutate the input ensemble configuration.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_dagger, su3_determinant,
    su3_identity, su3_mul, validate_su3,
};
use crate::symmetry_groups::Complex;

pub const SPATIAL_APE_EHK_POLAR_ID: &str = "spatial_ape_ehk_polar_v1";
const SU3_VALIDATION_TOLERANCE: f64 = 1.0e-12;
const SINGULAR_DETERMINANT_NORM: f64 = 1.0e-15;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialApeConfig {
    pub alpha: f64,
    pub iterations: usize,
    pub projection_tolerance: f64,
    pub projection_max_iterations: usize,
}

impl Default for SpatialApeConfig {
    fn default() -> Self {
        Self {
            alpha: 0.7,
            iterations: 1,
            projection_tolerance: 2.0e-15,
            projection_max_iterations: 40,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpatialSmearingError {
    Gauge(LatticeGaugeError),
    InvalidAlpha(f64),
    InvalidIterations(usize),
    InvalidProjectionTolerance(f64),
    InvalidProjectionIterations(usize),
    InvalidSpatialDirection(usize),
    DegenerateSpatialExtent([usize; 4]),
    SingularProjection { determinant_norm: f64 },
    NonFiniteProjection,
    ProjectionDidNotConverge {
        max_iterations: usize,
        last_error: f64,
    },
}

impl From<LatticeGaugeError> for SpatialSmearingError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

impl SpatialApeConfig {
    pub fn validate(&self) -> Result<(), SpatialSmearingError> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(SpatialSmearingError::InvalidAlpha(self.alpha));
        }
        if self.iterations == 0 {
            return Err(SpatialSmearingError::InvalidIterations(0));
        }
        if !self.projection_tolerance.is_finite() || self.projection_tolerance <= 0.0 {
            return Err(SpatialSmearingError::InvalidProjectionTolerance(
                self.projection_tolerance,
            ));
        }
        if self.projection_max_iterations == 0 {
            return Err(SpatialSmearingError::InvalidProjectionIterations(0));
        }
        Ok(())
    }
}

#[inline]
fn c_add(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re + b.re, a.im + b.im)
}

#[inline]
fn c_sub(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re - b.re, a.im - b.im)
}

#[inline]
fn c_mul(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn c_div(a: Complex, b: Complex) -> Result<Complex, SpatialSmearingError> {
    let denominator = b.re * b.re + b.im * b.im;
    if !denominator.is_finite() || denominator <= SINGULAR_DETERMINANT_NORM.powi(2) {
        return Err(SpatialSmearingError::SingularProjection {
            determinant_norm: denominator.sqrt(),
        });
    }
    Ok(Complex::new(
        (a.re * b.re + a.im * b.im) / denominator,
        (a.im * b.re - a.re * b.im) / denominator,
    ))
}

fn zero_matrix() -> Su3Matrix {
    [[Complex::ZERO; 3]; 3]
}

fn matrix_add(a: &Su3Matrix, b: &Su3Matrix) -> Su3Matrix {
    let mut out = zero_matrix();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = c_add(a[i][j], b[i][j]);
        }
    }
    out
}

fn matrix_add_assign(dst: &mut Su3Matrix, src: &Su3Matrix) {
    for i in 0..3 {
        for j in 0..3 {
            dst[i][j] = c_add(dst[i][j], src[i][j]);
        }
    }
}

fn matrix_scale_real(value: f64, matrix: &Su3Matrix) -> Su3Matrix {
    let mut out = zero_matrix();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = Complex::new(value * matrix[i][j].re, value * matrix[i][j].im);
        }
    }
    out
}

fn matrix_scale_complex(value: Complex, matrix: &Su3Matrix) -> Su3Matrix {
    let mut out = zero_matrix();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = c_mul(value, matrix[i][j]);
        }
    }
    out
}

fn complex_abs(value: Complex) -> f64 {
    value.norm_sq().sqrt()
}

fn matrix_frobenius_norm(matrix: &Su3Matrix) -> f64 {
    matrix
        .iter()
        .flatten()
        .map(|value| value.norm_sq())
        .sum::<f64>()
        .sqrt()
}

fn matrix_max_abs_error(a: &Su3Matrix, b: &Su3Matrix) -> f64 {
    let mut error: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            error = error.max(complex_abs(c_sub(a[i][j], b[i][j])));
        }
    }
    error
}

fn matrix_inverse(matrix: &Su3Matrix) -> Result<Su3Matrix, SpatialSmearingError> {
    let determinant = su3_determinant(matrix);
    let determinant_norm = complex_abs(determinant);
    if !determinant_norm.is_finite() || determinant_norm < SINGULAR_DETERMINANT_NORM {
        return Err(SpatialSmearingError::SingularProjection { determinant_norm });
    }

    let mut numerator = zero_matrix();
    numerator[0][0] = c_sub(c_mul(matrix[1][1], matrix[2][2]), c_mul(matrix[1][2], matrix[2][1]));
    numerator[0][1] = c_sub(c_mul(matrix[0][2], matrix[2][1]), c_mul(matrix[0][1], matrix[2][2]));
    numerator[0][2] = c_sub(c_mul(matrix[0][1], matrix[1][2]), c_mul(matrix[0][2], matrix[1][1]));
    numerator[1][0] = c_sub(c_mul(matrix[1][2], matrix[2][0]), c_mul(matrix[1][0], matrix[2][2]));
    numerator[1][1] = c_sub(c_mul(matrix[0][0], matrix[2][2]), c_mul(matrix[0][2], matrix[2][0]));
    numerator[1][2] = c_sub(c_mul(matrix[0][2], matrix[1][0]), c_mul(matrix[0][0], matrix[1][2]));
    numerator[2][0] = c_sub(c_mul(matrix[1][0], matrix[2][1]), c_mul(matrix[1][1], matrix[2][0]));
    numerator[2][1] = c_sub(c_mul(matrix[0][1], matrix[2][0]), c_mul(matrix[0][0], matrix[2][1]));
    numerator[2][2] = c_sub(c_mul(matrix[0][0], matrix[1][1]), c_mul(matrix[0][1], matrix[1][0]));

    let mut out = zero_matrix();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = c_div(numerator[i][j], determinant)?;
        }
    }
    Ok(out)
}

/// Deterministic SU(3) projection used by the frozen LQCD-020A convention.
///
/// Newton iteration computes the unitary polar factor, then the phase of the
/// determinant is divided equally across all colors so `det(U)=1`.
pub fn polar_project_su3(
    matrix: &Su3Matrix,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Su3Matrix, SpatialSmearingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(SpatialSmearingError::InvalidProjectionTolerance(tolerance));
    }
    if max_iterations == 0 {
        return Err(SpatialSmearingError::InvalidProjectionIterations(0));
    }

    let norm = matrix_frobenius_norm(matrix);
    if !norm.is_finite() || norm <= 0.0 {
        return Err(SpatialSmearingError::NonFiniteProjection);
    }
    let mut current = matrix_scale_real(3.0_f64.sqrt() / norm, matrix);
    let mut last_error = f64::INFINITY;
    let mut converged = false;

    for _ in 0..max_iterations {
        let inverse = matrix_inverse(&current)?;
        let inverse_dagger = su3_dagger(&inverse);
        let next = matrix_scale_real(0.5, &matrix_add(&current, &inverse_dagger));
        last_error = matrix_max_abs_error(&current, &next);
        if !last_error.is_finite() {
            return Err(SpatialSmearingError::NonFiniteProjection);
        }
        current = next;
        if last_error < tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SpatialSmearingError::ProjectionDidNotConverge {
            max_iterations,
            last_error,
        });
    }

    let determinant = su3_determinant(&current);
    if !determinant.re.is_finite() || !determinant.im.is_finite() {
        return Err(SpatialSmearingError::NonFiniteProjection);
    }
    let phase = determinant.im.atan2(determinant.re) / 3.0;
    let correction = Complex::new((-phase).cos(), (-phase).sin());
    let projected = matrix_scale_complex(correction, &current);
    validate_su3(&projected, SU3_VALIDATION_TOLERANCE)?;
    Ok(projected)
}

fn site_from_index(mut index: usize, dims: [usize; 4]) -> Site4 {
    let t = index % dims[3];
    index /= dims[3];
    let z = index % dims[2];
    index /= dims[2];
    let y = index % dims[1];
    index /= dims[1];
    [index, y, z, t]
}

fn validate_spatial_extents(field: &WilsonGaugeField) -> Result<(), SpatialSmearingError> {
    let dims = field.dims();
    if dims[..3].iter().any(|&extent| extent < 2) {
        return Err(SpatialSmearingError::DegenerateSpatialExtent(dims));
    }
    Ok(())
}

/// Sum the four APE staples transverse to one spatial link.
pub fn spatial_staple_sum(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<Su3Matrix, SpatialSmearingError> {
    validate_spatial_extents(field)?;
    if mu >= 3 {
        return Err(SpatialSmearingError::InvalidSpatialDirection(mu));
    }

    let mut staple = zero_matrix();
    let x_plus_mu = field.shift(site, mu, 1)?;
    for nu in 0..3 {
        if nu == mu {
            continue;
        }

        let x_plus_nu = field.shift(site, nu, 1)?;
        let forward = su3_mul(
            &su3_mul(field.link(site, nu)?, field.link(x_plus_nu, mu)?),
            &su3_dagger(field.link(x_plus_mu, nu)?),
        );
        matrix_add_assign(&mut staple, &forward);

        let x_minus_nu = field.shift(site, nu, -1)?;
        let x_minus_nu_plus_mu = field.shift(x_minus_nu, mu, 1)?;
        let backward = su3_mul(
            &su3_mul(
                &su3_dagger(field.link(x_minus_nu, nu)?),
                field.link(x_minus_nu, mu)?,
            ),
            field.link(x_minus_nu_plus_mu, nu)?,
        );
        matrix_add_assign(&mut staple, &backward);
    }
    Ok(staple)
}

/// Apply one synchronous spatial APE step and return a new gauge field.
pub fn spatial_ape_step(
    field: &WilsonGaugeField,
    alpha: f64,
    projection_tolerance: f64,
    projection_max_iterations: usize,
) -> Result<WilsonGaugeField, SpatialSmearingError> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(SpatialSmearingError::InvalidAlpha(alpha));
    }
    validate_spatial_extents(field)?;

    let mut out = field.clone();
    let dims = field.dims();
    for index in 0..field.site_count() {
        let site = site_from_index(index, dims);
        for mu in 0..3 {
            let candidate = matrix_add(
                &matrix_scale_real(alpha, field.link(site, mu)?),
                &spatial_staple_sum(field, site, mu)?,
            );
            let projected = polar_project_su3(
                &candidate,
                projection_tolerance,
                projection_max_iterations,
            )?;
            out.set_link(site, mu, projected)?;
        }
    }
    Ok(out)
}

/// Apply the declared number of synchronous APE iterations.
pub fn spatial_ape_smear(
    field: &WilsonGaugeField,
    config: &SpatialApeConfig,
) -> Result<WilsonGaugeField, SpatialSmearingError> {
    config.validate()?;
    validate_spatial_extents(field)?;
    let mut current = field.clone();
    for _ in 0..config.iterations {
        current = spatial_ape_step(
            &current,
            config.alpha,
            config.projection_tolerance,
            config.projection_max_iterations,
        )?;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedded_rotation(pair: (usize, usize), axis: [f64; 3], angle: f64) -> Su3Matrix {
        let norm = axis.iter().map(|value| value * value).sum::<f64>().sqrt();
        let [nx, ny, nz] = axis.map(|value| value / norm);
        let a0 = angle.cos();
        let scale = angle.sin();
        let [a1, a2, a3] = [scale * nx, scale * ny, scale * nz];
        let mut out = su3_identity();
        let (i, j) = pair;
        out[i][i] = Complex::new(a0, a3);
        out[i][j] = Complex::new(a2, a1);
        out[j][i] = Complex::new(-a2, a1);
        out[j][j] = Complex::new(a0, -a3);
        out
    }

    fn fixture() -> WilsonGaugeField {
        let mut field = WilsonGaugeField::identity([3, 3, 3, 2]).unwrap();
        let operations = [
            ([0, 0, 0, 0], 0, (0, 1), [1.0, 2.0, 3.0], 0.31),
            ([1, 0, 1, 0], 1, (0, 2), [2.0, -1.0, 1.0], -0.27),
            ([2, 1, 0, 1], 2, (1, 2), [1.0, 1.0, -2.0], 0.22),
            ([1, 2, 2, 0], 0, (0, 1), [-2.0, 1.0, 1.0], 0.19),
        ];
        for (site, mu, pair, axis, angle) in operations {
            let updated = su3_mul(&embedded_rotation(pair, axis, angle), field.link(site, mu).unwrap());
            field.set_link(site, mu, updated).unwrap();
        }
        field
    }

    fn assert_matrix_close(got: &Su3Matrix, want: &Su3Matrix, tolerance: f64) {
        assert!(matrix_max_abs_error(got, want) < tolerance, "max error {}", matrix_max_abs_error(got, want));
    }

    #[test]
    fn polar_projection_returns_scaled_su3_input() {
        let source = embedded_rotation((0, 2), [2.0, -1.0, 1.0], 0.37);
        let scaled = matrix_scale_real(2.75, &source);
        let projected = polar_project_su3(&scaled, 2.0e-15, 40).unwrap();
        assert_matrix_close(&projected, &source, 8.0e-15);
    }

    #[test]
    fn identity_is_fixed_and_temporal_links_are_unchanged() {
        let identity = WilsonGaugeField::identity([3, 3, 3, 2]).unwrap();
        let smeared = spatial_ape_smear(&identity, &SpatialApeConfig::default()).unwrap();
        let dims = identity.dims();
        for index in 0..identity.site_count() {
            let site = site_from_index(index, dims);
            for mu in 0..4 {
                assert_matrix_close(smeared.link(site, mu).unwrap(), &su3_identity(), 1.0e-14);
            }
        }

        let field = fixture();
        let smeared = spatial_ape_smear(&field, &SpatialApeConfig::default()).unwrap();
        for index in 0..field.site_count() {
            let site = site_from_index(index, field.dims());
            assert_eq!(smeared.link(site, 3).unwrap(), field.link(site, 3).unwrap());
        }
    }

    #[test]
    fn one_step_matches_independent_lqcd_020a_probe_matrices() {
        let smeared = spatial_ape_smear(&fixture(), &SpatialApeConfig::default()).unwrap();
        let expected = [
            (
                [0, 0, 0, 0],
                0,
                [
                    [Complex::new(0.998_954_692_724_952_6, 0.036_650_567_083_08), Complex::new(0.024_433_711_388_719_994, 0.012_216_855_694_359_997), Complex::ZERO],
                    [Complex::new(-0.024_433_711_388_719_994, 0.012_216_855_694_359_997), Complex::new(0.998_954_692_724_952_6, -0.036_650_567_083_08), Complex::ZERO],
                    [Complex::ZERO, Complex::ZERO, Complex::new(1.0, -4.319_411_479_499_165e-19)],
                ],
            ),
            (
                [1, 0, 1, 0],
                1,
                [
                    [Complex::new(0.999_203_292_381_323_7, -0.016_293_048_079_488_053), Complex::ZERO, Complex::new(0.016_293_048_079_488_053, -0.032_586_096_158_976_105)],
                    [Complex::ZERO, Complex::new(1.0, -2.161_765_924_422_137_8e-19), Complex::ZERO],
                    [Complex::new(-0.016_293_048_079_488_053, -0.032_586_096_158_976_105), Complex::ZERO, Complex::new(0.999_203_292_381_323_7, 0.016_293_048_079_488_053)],
                ],
            ),
            (
                [2, 1, 0, 1],
                2,
                [
                    [Complex::ONE, Complex::ZERO, Complex::ZERO],
                    [Complex::ZERO, Complex::new(0.999_468_412_221_735_7, -0.026_619_453_675_369_345), Complex::new(0.013_309_726_837_684_673, 0.013_309_726_837_684_673)],
                    [Complex::ZERO, Complex::new(-0.013_309_726_837_684_673, 0.013_309_726_837_684_673), Complex::new(0.999_468_412_221_735_7, 0.026_619_453_675_369_345)],
                ],
            ),
        ];
        for (site, mu, want) in expected {
            assert_matrix_close(smeared.link(site, mu).unwrap(), &want, 2.0e-12);
        }
    }

    #[test]
    fn smearing_is_gauge_covariant() {
        let field = fixture();
        let mut gauges = Vec::with_capacity(field.site_count());
        for index in 0..field.site_count() {
            let k = index as f64 + 1.0;
            gauges.push(su3_mul(
                &embedded_rotation((0, 1), [1.0, 2.0, 3.0], 0.017 * k),
                &embedded_rotation((0, 2), [2.0, -1.0, 1.0], -0.011 * k),
            ));
        }
        let transformed = field.gauge_transform(&gauges).unwrap();
        let smear_then_transform = spatial_ape_smear(&field, &SpatialApeConfig::default())
            .unwrap()
            .gauge_transform(&gauges)
            .unwrap();
        let transform_then_smear =
            spatial_ape_smear(&transformed, &SpatialApeConfig::default()).unwrap();
        let dims = field.dims();
        for index in 0..field.site_count() {
            let site = site_from_index(index, dims);
            for mu in 0..4 {
                assert_matrix_close(
                    smear_then_transform.link(site, mu).unwrap(),
                    transform_then_smear.link(site, mu).unwrap(),
                    2.0e-12,
                );
            }
        }
    }

    #[test]
    fn degenerate_spatial_extents_and_bad_parameters_fail_closed() {
        let degenerate = WilsonGaugeField::identity([3, 1, 3, 2]).unwrap();
        assert!(matches!(
            spatial_ape_smear(&degenerate, &SpatialApeConfig::default()),
            Err(SpatialSmearingError::DegenerateSpatialExtent([3, 1, 3, 2]))
        ));

        let field = fixture();
        let mut config = SpatialApeConfig::default();
        config.alpha = 0.0;
        assert!(matches!(
            spatial_ape_smear(&field, &config),
            Err(SpatialSmearingError::InvalidAlpha(0.0))
        ));
        config = SpatialApeConfig::default();
        config.projection_max_iterations = 0;
        assert!(matches!(
            spatial_ape_smear(&field, &config),
            Err(SpatialSmearingError::InvalidProjectionIterations(0))
        ));
    }
}
