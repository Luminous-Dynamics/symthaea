// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference clover topology and Wilson-action gradient flow for pure SU(3).
//!
//! This module intentionally mirrors the independent standard-library oracle
//! in LQCD-017D. The flow implementation is deliberately slow: it reconstructs
//! the Wilson-action gradient by central finite differences along all eight
//! Gell-Mann directions and then applies one simultaneous Lie-Euler step.
//! It is a semantic reference for a later direct-staple / higher-order flow,
//! not the performance implementation.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_dagger,
    su3_determinant_error, su3_identity, su3_mul, su3_trace, su3_unitarity_error,
};
use crate::symmetry_groups::{Complex, gell_mann_matrix};
use std::f64::consts::PI;

const MATRIX_EXP_TERMS: usize = 50;

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeTopologyFlowError {
    Gauge(LatticeGaugeError),
    SamePlaneDirection(usize),
    InvalidFlowStep(f64),
    InvalidGradientEpsilon(f64),
    InvalidSignedDirection(i8),
}

impl From<LatticeGaugeError> for LatticeTopologyFlowError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceFlowStepStats {
    pub action_before: f64,
    pub action_after: f64,
    pub clover_q_before: f64,
    pub clover_q_after: f64,
    pub max_unitarity_error: f64,
    pub max_determinant_error: f64,
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

fn matrix_sub(a: &Su3Matrix, b: &Su3Matrix) -> Su3Matrix {
    let mut out = zero_matrix();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = c_sub(a[i][j], b[i][j]);
        }
    }
    out
}

fn matrix_scale(a: &Su3Matrix, scalar: Complex) -> Su3Matrix {
    let mut out = zero_matrix();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = c_mul(scalar, a[i][j]);
        }
    }
    out
}

fn matrix_frobenius_norm(a: &Su3Matrix) -> f64 {
    let mut sum = 0.0;
    for row in a {
        for value in row {
            sum += value.norm_sq();
        }
    }
    sum.sqrt()
}

/// Small dense-matrix exponential used only by the semantic reference path.
/// Scaling/squaring plus a 50-term Taylor expansion mirrors LQCD-017D.
fn matrix_exp(a: &Su3Matrix) -> Su3Matrix {
    let norm = matrix_frobenius_norm(a);
    let squarings = if norm > 0.5 {
        (norm / 0.5).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let denominator = 2.0_f64.powi(squarings as i32);
    let x = matrix_scale(a, Complex::new(1.0 / denominator, 0.0));
    let mut out = su3_identity();
    let mut term = su3_identity();
    for k in 1..=MATRIX_EXP_TERMS {
        term = matrix_scale(
            &su3_mul(&term, &x),
            Complex::new(1.0 / k as f64, 0.0),
        );
        out = matrix_add(&out, &term);
    }
    for _ in 0..squarings {
        out = su3_mul(&out, &out);
    }
    out
}

fn su3_generator_rotation(generator: usize, theta: f64) -> Su3Matrix {
    let lambda = gell_mann_matrix(generator);
    matrix_exp(&matrix_scale(&lambda, Complex::new(0.0, theta)))
}

fn oriented_link(
    field: &WilsonGaugeField,
    site: Site4,
    signed_direction: i8,
) -> Result<(Su3Matrix, Site4), LatticeTopologyFlowError> {
    if signed_direction == 0 || signed_direction.abs() > 4 {
        return Err(LatticeTopologyFlowError::InvalidSignedDirection(
            signed_direction,
        ));
    }
    if signed_direction > 0 {
        let mu = (signed_direction - 1) as usize;
        let next = field.shift(site, mu, 1)?;
        return Ok((*field.link(site, mu)?, next));
    }
    let mu = (-signed_direction - 1) as usize;
    let previous = field.shift(site, mu, -1)?;
    Ok((su3_dagger(field.link(previous, mu)?), previous))
}

fn transporter(
    field: &WilsonGaugeField,
    start: Site4,
    directions: &[i8],
) -> Result<Su3Matrix, LatticeTopologyFlowError> {
    let mut product = su3_identity();
    let mut site = start;
    for &direction in directions {
        let (edge, next) = oriented_link(field, site, direction)?;
        product = su3_mul(&product, &edge);
        site = next;
    }
    Ok(product)
}

pub fn clover_sum(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    nu: usize,
) -> Result<Su3Matrix, LatticeTopologyFlowError> {
    if mu >= 4 {
        return Err(LatticeGaugeError::InvalidDirection(mu).into());
    }
    if nu >= 4 {
        return Err(LatticeGaugeError::InvalidDirection(nu).into());
    }
    if mu == nu {
        return Err(LatticeTopologyFlowError::SamePlaneDirection(mu));
    }
    let pmu = (mu + 1) as i8;
    let pnu = (nu + 1) as i8;
    let loops = [
        [pmu, pnu, -pmu, -pnu],
        [pnu, -pmu, -pnu, pmu],
        [-pmu, -pnu, pmu, pnu],
        [-pnu, pmu, pnu, -pmu],
    ];
    let mut sum = zero_matrix();
    for path in loops {
        sum = matrix_add(&sum, &transporter(field, site, &path)?);
    }
    Ok(sum)
}

/// Hermitian traceless clover field strength
/// `F_munu = (C_munu - C_munu^dagger) / (8 i)` with lattice spacing `a=1`.
pub fn clover_field_strength(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    nu: usize,
) -> Result<Su3Matrix, LatticeTopologyFlowError> {
    let clover = clover_sum(field, site, mu, nu)?;
    let antihermitian = matrix_sub(&clover, &su3_dagger(&clover));
    let mut field_strength = matrix_scale(&antihermitian, Complex::new(0.0, -1.0 / 8.0));
    let trace = su3_trace(&field_strength);
    let singlet = Complex::new(trace.re / 3.0, trace.im / 3.0);
    for color in 0..3 {
        field_strength[color][color] = c_sub(field_strength[color][color], singlet);
    }
    Ok(field_strength)
}

pub fn clover_topological_density(
    field: &WilsonGaugeField,
    site: Site4,
) -> Result<f64, LatticeTopologyFlowError> {
    let f01 = clover_field_strength(field, site, 0, 1)?;
    let f02 = clover_field_strength(field, site, 0, 2)?;
    let f03 = clover_field_strength(field, site, 0, 3)?;
    let f12 = clover_field_strength(field, site, 1, 2)?;
    let f13 = clover_field_strength(field, site, 1, 3)?;
    let f23 = clover_field_strength(field, site, 2, 3)?;

    let contraction = su3_trace(&su3_mul(&f01, &f23)).re
        - su3_trace(&su3_mul(&f02, &f13)).re
        + su3_trace(&su3_mul(&f03, &f12)).re;
    Ok(contraction / (4.0 * PI * PI))
}

pub fn clover_topological_charge(
    field: &WilsonGaugeField,
) -> Result<f64, LatticeTopologyFlowError> {
    let dims = field.dims();
    let mut charge = 0.0;
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    charge += clover_topological_density(field, [x, y, z, t])?;
                }
            }
        }
    }
    Ok(charge)
}

fn finite_difference_gradient(
    field: &WilsonGaugeField,
    epsilon: f64,
) -> Result<Vec<(Site4, usize, [f64; 8])>, LatticeTopologyFlowError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(LatticeTopologyFlowError::InvalidGradientEpsilon(epsilon));
    }
    let dims = field.dims();
    let mut gradient = Vec::with_capacity(field.site_count() * 4);
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let site = [x, y, z, t];
                    for mu in 0..4 {
                        let original = *field.link(site, mu)?;
                        let mut components = [0.0; 8];
                        for (generator, component) in components.iter_mut().enumerate() {
                            let plus_rotation = su3_generator_rotation(generator, epsilon);
                            let minus_rotation = su3_generator_rotation(generator, -epsilon);
                            let mut plus = field.clone();
                            let mut minus = field.clone();
                            plus.set_link(site, mu, su3_mul(&plus_rotation, &original))?;
                            minus.set_link(site, mu, su3_mul(&minus_rotation, &original))?;
                            let plus_action = plus.wilson_action(1.0)?;
                            let minus_action = minus.wilson_action(1.0)?;
                            *component = (plus_action - minus_action) / (2.0 * epsilon);
                        }
                        gradient.push((site, mu, components));
                    }
                }
            }
        }
    }
    Ok(gradient)
}

/// One simultaneous Lie-Euler descent step of the Wilson action.
///
/// This is intentionally expensive and exists as a parity/reference kernel. A
/// production staple-based flow must match it within a separately declared
/// small-step tolerance before replacing it in ensemble workflows.
pub fn finite_difference_wilson_flow_step_reference(
    field: &mut WilsonGaugeField,
    dt: f64,
    epsilon: f64,
) -> Result<ReferenceFlowStepStats, LatticeTopologyFlowError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(LatticeTopologyFlowError::InvalidFlowStep(dt));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(LatticeTopologyFlowError::InvalidGradientEpsilon(epsilon));
    }

    let action_before = field.wilson_action(1.0)?;
    let clover_q_before = clover_topological_charge(field)?;
    let gradient = finite_difference_gradient(field, epsilon)?;
    let original = field.clone();
    let mut updated = original.clone();

    for (site, mu, components) in gradient {
        let mut hermitian = zero_matrix();
        for (generator, coefficient) in components.into_iter().enumerate() {
            let term = matrix_scale(
                &gell_mann_matrix(generator),
                Complex::new(coefficient, 0.0),
            );
            hermitian = matrix_add(&hermitian, &term);
        }
        let exponent = matrix_scale(&hermitian, Complex::new(0.0, -dt));
        let rotation = matrix_exp(&exponent);
        let next = su3_mul(&rotation, original.link(site, mu)?);
        updated.set_link(site, mu, next)?;
    }

    let dims = updated.dims();
    let mut max_unitarity_error: f64 = 0.0;
    let mut max_determinant_error: f64 = 0.0;
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let site = [x, y, z, t];
                    for mu in 0..4 {
                        let link = updated.link(site, mu)?;
                        max_unitarity_error = max_unitarity_error.max(su3_unitarity_error(link));
                        max_determinant_error = max_determinant_error.max(su3_determinant_error(link));
                    }
                }
            }
        }
    }

    let action_after = updated.wilson_action(1.0)?;
    let clover_q_after = clover_topological_charge(&updated)?;
    *field = updated;

    Ok(ReferenceFlowStepStats {
        action_before,
        action_after,
        clover_q_before,
        clover_q_after,
        max_unitarity_error,
        max_determinant_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_OPS: [(Site4, usize, usize, f64); 40] = [
        ([0, 0, 1, 0], 0, 3, -0.5819354339098058),
        ([0, 1, 1, 1], 0, 2, -0.32393958624710995),
        ([0, 1, 1, 1], 0, 6, -0.1250464331865352),
        ([1, 0, 1, 0], 1, 2, 0.19161237542499698),
        ([0, 1, 0, 1], 1, 2, 0.22983089548025992),
        ([1, 0, 1, 1], 1, 3, 0.4032003002590304),
        ([1, 1, 1, 1], 3, 5, 0.5114974521920254),
        ([1, 1, 1, 1], 2, 4, -0.09168927490735845),
        ([1, 1, 1, 0], 1, 6, -0.4726273806149346),
        ([0, 1, 0, 0], 2, 2, 0.3411282616087933),
        ([1, 1, 0, 0], 3, 3, -0.341627594608156),
        ([1, 0, 1, 1], 2, 5, -0.11361497865301007),
        ([1, 0, 1, 0], 1, 4, 0.003219526879812973),
        ([0, 1, 1, 1], 3, 1, 0.2092875716255429),
        ([0, 0, 1, 1], 1, 0, -0.2872940926296875),
        ([1, 0, 1, 1], 1, 3, -0.33032520366686874),
        ([1, 1, 1, 0], 3, 1, 0.5401678740668779),
        ([0, 0, 1, 0], 2, 2, 0.31304313667691264),
        ([1, 1, 0, 0], 0, 1, 0.2375215627437346),
        ([0, 1, 0, 0], 0, 0, 0.3958604638855575),
        ([1, 1, 0, 0], 0, 5, -0.42942950746005015),
        ([1, 1, 1, 1], 2, 6, 0.5455840065263183),
        ([1, 0, 0, 0], 2, 3, 0.017115673418346744),
        ([1, 0, 0, 1], 1, 2, 0.13107989509472173),
        ([1, 0, 1, 1], 1, 7, -0.4551702244963054),
        ([1, 0, 0, 0], 0, 0, -0.5395094927947152),
        ([1, 1, 1, 1], 2, 2, 0.3670444298654235),
        ([0, 0, 0, 0], 2, 6, -0.24861863368491577),
        ([1, 1, 1, 1], 2, 3, 0.335177509798799),
        ([0, 1, 1, 1], 2, 5, 0.5656866202369394),
        ([1, 1, 1, 1], 2, 2, 0.47986932117951764),
        ([1, 0, 0, 0], 3, 4, 0.013402168869181441),
        ([0, 0, 0, 1], 2, 4, 0.10603947222843657),
        ([0, 1, 0, 0], 2, 0, 0.23018975518126505),
        ([0, 0, 0, 0], 1, 0, 0.19239685997822842),
        ([1, 0, 1, 0], 0, 6, -0.08507387388380139),
        ([1, 1, 0, 0], 0, 2, -0.22404873035871248),
        ([1, 0, 1, 0], 3, 0, -0.35908049548949106),
        ([1, 1, 0, 0], 0, 6, 0.07694552778636243),
        ([1, 0, 0, 0], 0, 5, -0.12989873801836926),
    ];

    fn fixture() -> WilsonGaugeField {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        for (site, mu, generator, theta) in FIXTURE_OPS {
            let rotation = su3_generator_rotation(generator, theta);
            let original = *field.link(site, mu).unwrap();
            field.set_link(site, mu, su3_mul(&rotation, &original)).unwrap();
        }
        field
    }

    fn gauge_transform_fixture(field: &WilsonGaugeField) -> WilsonGaugeField {
        let dims = field.dims();
        let mut gauges = Vec::with_capacity(field.site_count());
        let mut k = 0usize;
        for _x in 0..dims[0] {
            for _y in 0..dims[1] {
                for _z in 0..dims[2] {
                    for _t in 0..dims[3] {
                        gauges.push(su3_mul(
                            &su3_generator_rotation(k % 8, 0.04 * (k + 1) as f64),
                            &su3_generator_rotation((k + 3) % 8, -0.02 * (k + 1) as f64),
                        ));
                        k += 1;
                    }
                }
            }
        }
        field.gauge_transform(&gauges).unwrap()
    }

    fn matrix_max_error(a: &Su3Matrix, b: &Su3Matrix) -> f64 {
        let mut error: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let dr = a[i][j].re - b[i][j].re;
                let di = a[i][j].im - b[i][j].im;
                error = error.max((dr * dr + di * di).sqrt());
            }
        }
        error
    }

    fn field_max_error(a: &WilsonGaugeField, b: &WilsonGaugeField) -> f64 {
        let dims = a.dims();
        let mut error: f64 = 0.0;
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                for z in 0..dims[2] {
                    for t in 0..dims[3] {
                        let site = [x, y, z, t];
                        for mu in 0..4 {
                            error = error.max(matrix_max_error(
                                a.link(site, mu).unwrap(),
                                b.link(site, mu).unwrap(),
                            ));
                        }
                    }
                }
            }
        }
        error
    }

    #[test]
    fn identity_has_zero_clover_charge() {
        let field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        assert!(clover_topological_charge(&field).unwrap().abs() < 1.0e-15);
    }

    #[test]
    fn clover_charge_matches_independent_oracle_and_is_gauge_invariant() {
        let field = fixture();
        assert!((field.wilson_action(1.0).unwrap() - 8.222_361_191_068_600_3).abs() < 2.0e-12);
        let q = clover_topological_charge(&field).unwrap();
        assert!((q - (-0.000_417_133_400_269_600_05)).abs() < 2.0e-15);

        let transformed = gauge_transform_fixture(&field);
        assert!((transformed.wilson_action(1.0).unwrap() - field.wilson_action(1.0).unwrap()).abs() < 1.0e-11);
        assert!((clover_topological_charge(&transformed).unwrap() - q).abs() < 2.0e-15);
    }

    #[test]
    fn reference_flow_step_matches_independent_oracle() {
        let mut field = fixture();
        let stats = finite_difference_wilson_flow_step_reference(&mut field, 1.0e-3, 2.0e-6).unwrap();
        assert!((stats.action_before - 8.222_361_191_068_600_3).abs() < 2.0e-12);
        assert!((stats.clover_q_before - (-0.000_417_133_400_269_600_05)).abs() < 2.0e-15);
        assert!((stats.action_after - 8.131_857_099_782_706_7).abs() < 3.0e-10);
        assert!((stats.clover_q_after - (-0.000_411_907_836_303_515_9)).abs() < 3.0e-14);
        assert!(stats.action_after < stats.action_before);
        assert!(stats.max_determinant_error < 2.0e-12);
        assert!(stats.max_unitarity_error < 2.0e-12);
    }

    #[test]
    fn reference_flow_is_gauge_covariant_on_oracle_fixture() {
        let field = fixture();
        let transformed = gauge_transform_fixture(&field);
        let mut flowed = field.clone();
        let mut flowed_transformed = transformed.clone();
        finite_difference_wilson_flow_step_reference(&mut flowed, 1.0e-3, 2.0e-6).unwrap();
        finite_difference_wilson_flow_step_reference(&mut flowed_transformed, 1.0e-3, 2.0e-6).unwrap();
        let transformed_flowed = gauge_transform_fixture(&flowed);
        assert!(field_max_error(&flowed_transformed, &transformed_flowed) < 4.0e-12);
    }

    #[test]
    fn invalid_flow_parameters_fail_closed() {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        assert!(matches!(
            finite_difference_wilson_flow_step_reference(&mut field, 0.0, 2.0e-6),
            Err(LatticeTopologyFlowError::InvalidFlowStep(0.0))
        ));
        assert!(matches!(
            finite_difference_wilson_flow_step_reference(&mut field, 1.0e-3, 0.0),
            Err(LatticeTopologyFlowError::InvalidGradientEpsilon(0.0))
        ));
    }
}
