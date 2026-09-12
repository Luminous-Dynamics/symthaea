// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Optimized six-staple Wilson-action flow with reference parity.
//!
//! This module is an optimization of `lattice_topology_flow`, not a new source
//! of semantics. Non-degenerate lattices use the analytic six-staple Wilson
//! gradient; any length-1 extent falls back to the finite-difference reference
//! so canonical plaquettes are not silently double-counted.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_dagger,
    su3_determinant_error, su3_identity, su3_mul, su3_trace, su3_unitarity_error,
};
use crate::lattice_topology_flow::{
    LatticeTopologyFlowError, ReferenceFlowStepStats, clover_topological_charge,
    finite_difference_wilson_flow_step_reference,
};
use crate::symmetry_groups::{Complex, gell_mann_matrix};

const MATRIX_EXP_TERMS: usize = 50;

#[derive(Debug, Clone, PartialEq)]
pub enum StapleFlowError {
    Gauge(LatticeGaugeError),
    Reference(LatticeTopologyFlowError),
    DegenerateExtent([usize; 4]),
    InvalidFlowStep(f64),
    InvalidReferenceEpsilon(f64),
}

impl From<LatticeGaugeError> for StapleFlowError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

impl From<LatticeTopologyFlowError> for StapleFlowError {
    fn from(value: LatticeTopologyFlowError) -> Self {
        Self::Reference(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StapleFlowStepStats {
    pub flow: ReferenceFlowStepStats,
    pub used_reference_fallback: bool,
}

#[inline]
fn c_add(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re + b.re, a.im + b.im)
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

fn matrix_add_assign(dst: &mut Su3Matrix, src: &Su3Matrix) {
    for i in 0..3 {
        for j in 0..3 {
            dst[i][j] = c_add(dst[i][j], src[i][j]);
        }
    }
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
    a.iter()
        .flatten()
        .map(|value| value.norm_sq())
        .sum::<f64>()
        .sqrt()
}

fn matrix_exp(a: &Su3Matrix) -> Su3Matrix {
    let norm = matrix_frobenius_norm(a);
    let squarings = if norm > 0.5 {
        (norm / 0.5).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let x = matrix_scale(
        a,
        Complex::new(1.0 / 2.0_f64.powi(squarings as i32), 0.0),
    );
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

fn generator_rotation(generator: usize, theta: f64) -> Su3Matrix {
    matrix_exp(&matrix_scale(
        &gell_mann_matrix(generator),
        Complex::new(0.0, theta),
    ))
}

/// Six-staple matrix H oriented so the local Wilson trace is `Re Tr(U_mu(x) H)`.
pub fn link_staple_non_degenerate(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<Su3Matrix, StapleFlowError> {
    if mu >= 4 {
        return Err(LatticeGaugeError::InvalidDirection(mu).into());
    }
    let dims = field.dims();
    if dims.iter().any(|&extent| extent < 2) {
        return Err(StapleFlowError::DegenerateExtent(dims));
    }

    let mut staple = zero_matrix();
    let x_plus_mu = field.shift(site, mu, 1)?;
    for nu in 0..4 {
        if nu == mu {
            continue;
        }

        let x_plus_nu = field.shift(site, nu, 1)?;
        let forward = su3_mul(
            &su3_mul(
                field.link(x_plus_mu, nu)?,
                &su3_dagger(field.link(x_plus_nu, mu)?),
            ),
            &su3_dagger(field.link(site, nu)?),
        );
        matrix_add_assign(&mut staple, &forward);

        let x_minus_nu = field.shift(site, nu, -1)?;
        let x_minus_nu_plus_mu = field.shift(x_minus_nu, mu, 1)?;
        let backward = su3_mul(
            &su3_mul(
                &su3_dagger(field.link(x_minus_nu_plus_mu, nu)?),
                &su3_dagger(field.link(x_minus_nu, mu)?),
            ),
            field.link(x_minus_nu, nu)?,
        );
        matrix_add_assign(&mut staple, &backward);
    }
    Ok(staple)
}

/// Analytic Wilson-action gradient components for the left perturbation
/// `U -> exp(i theta_a lambda_a) U`, with beta fixed to one.
///
/// For `X = U H`, `dS/dtheta_a = Im Tr(lambda_a X) / 3`.
pub fn analytic_link_gradient(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<[f64; 8], StapleFlowError> {
    let staple = link_staple_non_degenerate(field, site, mu)?;
    let x = su3_mul(field.link(site, mu)?, &staple);
    let mut gradient = [0.0; 8];
    for (generator, component) in gradient.iter_mut().enumerate() {
        *component = su3_trace(&su3_mul(&gell_mann_matrix(generator), &x)).im / 3.0;
    }
    Ok(gradient)
}

/// One optimized Lie-Euler Wilson-action flow step.
///
/// `reference_epsilon` is only used when a degenerate extent forces the entire
/// step through the finite-difference semantic reference.
pub fn staple_wilson_flow_step(
    field: &mut WilsonGaugeField,
    dt: f64,
    reference_epsilon: f64,
) -> Result<StapleFlowStepStats, StapleFlowError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(StapleFlowError::InvalidFlowStep(dt));
    }
    if !reference_epsilon.is_finite() || reference_epsilon <= 0.0 {
        return Err(StapleFlowError::InvalidReferenceEpsilon(reference_epsilon));
    }

    if field.dims().iter().any(|&extent| extent < 2) {
        let flow = finite_difference_wilson_flow_step_reference(field, dt, reference_epsilon)?;
        return Ok(StapleFlowStepStats {
            flow,
            used_reference_fallback: true,
        });
    }

    let action_before = field.wilson_action(1.0)?;
    let clover_q_before = clover_topological_charge(field)?;
    let dims = field.dims();
    let original = field.clone();
    let mut gradients = Vec::with_capacity(field.site_count() * 4);
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let site = [x, y, z, t];
                    for mu in 0..4 {
                        gradients.push((site, mu, analytic_link_gradient(&original, site, mu)?));
                    }
                }
            }
        }
    }

    let mut updated = original.clone();
    for (site, mu, components) in gradients {
        let mut hermitian = zero_matrix();
        for (generator, coefficient) in components.into_iter().enumerate() {
            let term = matrix_scale(
                &gell_mann_matrix(generator),
                Complex::new(coefficient, 0.0),
            );
            hermitian = matrix_add(&hermitian, &term);
        }
        let rotation = matrix_exp(&matrix_scale(&hermitian, Complex::new(0.0, -dt)));
        updated.set_link(site, mu, su3_mul(&rotation, original.link(site, mu)?))?;
    }

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

    Ok(StapleFlowStepStats {
        flow: ReferenceFlowStepStats {
            action_before,
            action_after,
            clover_q_before,
            clover_q_after,
            max_unitarity_error,
            max_determinant_error,
        },
        used_reference_fallback: false,
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
            let rotation = generator_rotation(generator, theta);
            let original = *field.link(site, mu).unwrap();
            field.set_link(site, mu, su3_mul(&rotation, &original)).unwrap();
        }
        field
    }

    fn finite_difference_link_gradient(
        field: &WilsonGaugeField,
        site: Site4,
        mu: usize,
        epsilon: f64,
    ) -> [f64; 8] {
        let original = *field.link(site, mu).unwrap();
        let mut gradient = [0.0; 8];
        for (generator, component) in gradient.iter_mut().enumerate() {
            let mut plus = field.clone();
            let mut minus = field.clone();
            plus.set_link(site, mu, su3_mul(&generator_rotation(generator, epsilon), &original)).unwrap();
            minus.set_link(site, mu, su3_mul(&generator_rotation(generator, -epsilon), &original)).unwrap();
            *component = (plus.wilson_action(1.0).unwrap() - minus.wilson_action(1.0).unwrap())
                / (2.0 * epsilon);
        }
        gradient
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
    fn analytic_staple_gradient_matches_finite_difference_oracle_target() {
        let field = fixture();
        let dims = field.dims();
        let mut max_error: f64 = 0.0;
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                for z in 0..dims[2] {
                    for t in 0..dims[3] {
                        let site = [x, y, z, t];
                        for mu in 0..4 {
                            let finite = finite_difference_link_gradient(&field, site, mu, 2.0e-6);
                            let analytic = analytic_link_gradient(&field, site, mu).unwrap();
                            for generator in 0..8 {
                                max_error = max_error.max((finite[generator] - analytic[generator]).abs());
                            }
                        }
                    }
                }
            }
        }
        assert!(max_error < 1.1e-9);
    }

    #[test]
    fn optimized_one_step_matches_reference_full_field() {
        let field = fixture();
        let mut reference = field.clone();
        let mut optimized = field.clone();
        finite_difference_wilson_flow_step_reference(&mut reference, 1.0e-3, 2.0e-6).unwrap();
        let stats = staple_wilson_flow_step(&mut optimized, 1.0e-3, 2.0e-6).unwrap();
        assert!(!stats.used_reference_fallback);
        assert!(field_max_error(&reference, &optimized) < 1.2e-12);
        assert!((stats.flow.action_after - 8.131_857_099_777_850_1).abs() < 3.0e-10);
        assert!((stats.flow.clover_q_after - (-0.000_411_907_836_302_700_52)).abs() < 3.0e-14);
        assert!(stats.flow.action_after < stats.flow.action_before);
    }

    #[test]
    fn degenerate_extent_falls_back_to_reference() {
        let mut field = WilsonGaugeField::identity([2, 2, 1, 2]).unwrap();
        let rotation = generator_rotation(2, 0.31);
        let original = *field.link([0, 0, 0, 0], 0).unwrap();
        field.set_link([0, 0, 0, 0], 0, su3_mul(&rotation, &original)).unwrap();
        let mut reference = field.clone();
        let mut optimized = field.clone();
        finite_difference_wilson_flow_step_reference(&mut reference, 1.0e-3, 2.0e-6).unwrap();
        let stats = staple_wilson_flow_step(&mut optimized, 1.0e-3, 2.0e-6).unwrap();
        assert!(stats.used_reference_fallback);
        assert!(field_max_error(&reference, &optimized) < 1.0e-15);
    }
}
