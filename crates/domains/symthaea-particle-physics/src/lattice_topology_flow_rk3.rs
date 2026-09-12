// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Third-order Lie-group Runge-Kutta integration of Wilson flow.
//!
//! The stage coefficients are independently qualified by LQCD-017G. This
//! module consumes the analytic six-staple Wilson-action gradient from
//! `lattice_topology_flow_staple`; it does not redefine the force semantics.
//! Lie-Euler remains available as the simpler reference/regression path.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, WilsonGaugeField, su3_determinant_error, su3_mul,
    su3_unitarity_error,
};
use crate::lattice_su3_lie::{matrix_add, matrix_exp, matrix_scale, zero_matrix};
use crate::lattice_topology_flow::{ReferenceFlowStepStats, clover_topological_charge};
use crate::lattice_topology_flow_staple::{StapleFlowError, analytic_link_gradient};
use crate::symmetry_groups::{Complex, gell_mann_matrix};

#[derive(Debug, Clone, PartialEq)]
pub enum Rk3FlowError {
    Gauge(LatticeGaugeError),
    Staple(StapleFlowError),
    InvalidFlowStep(f64),
    DegenerateExtent([usize; 4]),
}

impl From<LatticeGaugeError> for Rk3FlowError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

impl From<StapleFlowError> for Rk3FlowError {
    fn from(value: StapleFlowError) -> Self {
        Self::Staple(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rk3FlowStepStats {
    pub flow: ReferenceFlowStepStats,
    /// Three full Wilson-force evaluations per RK3 step.
    pub force_evaluations: usize,
}

type LinkGradient = (Site4, usize, [f64; 8]);

fn collect_gradients(field: &WilsonGaugeField) -> Result<Vec<LinkGradient>, Rk3FlowError> {
    let dims = field.dims();
    if dims.iter().any(|&extent| extent < 2) {
        return Err(Rk3FlowError::DegenerateExtent(dims));
    }
    let mut gradients = Vec::with_capacity(field.site_count() * 4);
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let site = [x, y, z, t];
                    for mu in 0..4 {
                        gradients.push((site, mu, analytic_link_gradient(field, site, mu)?));
                    }
                }
            }
        }
    }
    Ok(gradients)
}

fn same_layout(a: &[LinkGradient], b: &[LinkGradient]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((site_a, mu_a, _), (site_b, mu_b, _))| site_a == site_b && mu_a == mu_b)
}

fn scaled_gradient(gradient: &[LinkGradient], scale: f64) -> Vec<LinkGradient> {
    gradient
        .iter()
        .map(|(site, mu, components)| {
            let mut out = [0.0; 8];
            for index in 0..8 {
                out[index] = scale * components[index];
            }
            (*site, *mu, out)
        })
        .collect()
}

fn combine_two(
    a: &[LinkGradient],
    ca: f64,
    b: &[LinkGradient],
    cb: f64,
) -> Result<Vec<LinkGradient>, Rk3FlowError> {
    if !same_layout(a, b) {
        return Err(Rk3FlowError::Gauge(LatticeGaugeError::InvalidExtent([
            0, 0, 0, 0,
        ])));
    }
    Ok(a.iter()
        .zip(b)
        .map(|((site, mu, av), (_, _, bv))| {
            let mut out = [0.0; 8];
            for index in 0..8 {
                out[index] = ca * av[index] + cb * bv[index];
            }
            (*site, *mu, out)
        })
        .collect())
}

fn combine_three(
    a: &[LinkGradient],
    ca: f64,
    b: &[LinkGradient],
    cb: f64,
    c: &[LinkGradient],
    cc: f64,
) -> Result<Vec<LinkGradient>, Rk3FlowError> {
    if !same_layout(a, b) || !same_layout(a, c) {
        return Err(Rk3FlowError::Gauge(LatticeGaugeError::InvalidExtent([
            0, 0, 0, 0,
        ])));
    }
    Ok(a.iter()
        .zip(b)
        .zip(c)
        .map(|(((site, mu, av), (_, _, bv)), (_, _, cv))| {
            let mut out = [0.0; 8];
            for index in 0..8 {
                out[index] = ca * av[index] + cb * bv[index] + cc * cv[index];
            }
            (*site, *mu, out)
        })
        .collect())
}

/// Apply a Lie-algebra increment represented as coefficients multiplying the
/// Gell-Mann basis inside `exp(i sum_a c_a lambda_a)`.
fn apply_algebra(
    base: &WilsonGaugeField,
    algebra: &[LinkGradient],
) -> Result<WilsonGaugeField, Rk3FlowError> {
    let mut updated = base.clone();
    for (site, mu, components) in algebra {
        let mut hermitian = zero_matrix();
        for (generator, coefficient) in components.iter().copied().enumerate() {
            hermitian = matrix_add(
                &hermitian,
                &matrix_scale(
                    &gell_mann_matrix(generator),
                    Complex::new(coefficient, 0.0),
                ),
            );
        }
        let rotation = matrix_exp(&matrix_scale(&hermitian, Complex::I));
        updated.set_link(*site, *mu, su3_mul(&rotation, base.link(*site, *mu)?))?;
    }
    Ok(updated)
}

fn su3_drift(field: &WilsonGaugeField) -> Result<(f64, f64), Rk3FlowError> {
    let dims = field.dims();
    let mut max_unitarity_error: f64 = 0.0;
    let mut max_determinant_error: f64 = 0.0;
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let site = [x, y, z, t];
                    for mu in 0..4 {
                        let link = field.link(site, mu)?;
                        max_unitarity_error = max_unitarity_error.max(su3_unitarity_error(link));
                        max_determinant_error = max_determinant_error.max(su3_determinant_error(link));
                    }
                }
            }
        }
    }
    Ok((max_unitarity_error, max_determinant_error))
}

/// Advance one third-order Wilson-flow step.
///
/// With `Z_i = -dt * grad S_W(W_i)` the stages are
///
/// - `W1 = exp((1/4) Z0) W0`,
/// - `W2 = exp((8/9) Z1 - (17/36) Z0) W1`,
/// - `V' = exp((3/4) Z2 - (8/9) Z1 + (17/36) Z0) W2`.
///
/// Length-one lattice extents fail closed. Falling back to Lie-Euler would
/// silently change the declared integrator order and is therefore forbidden.
pub fn rk3_wilson_flow_step(
    field: &mut WilsonGaugeField,
    dt: f64,
) -> Result<Rk3FlowStepStats, Rk3FlowError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(Rk3FlowError::InvalidFlowStep(dt));
    }
    let dims = field.dims();
    if dims.iter().any(|&extent| extent < 2) {
        return Err(Rk3FlowError::DegenerateExtent(dims));
    }

    let action_before = field.wilson_action(1.0)?;
    let clover_q_before = clover_topological_charge(field).map_err(StapleFlowError::from)?;
    let w0 = field.clone();

    let g0 = collect_gradients(&w0)?;
    let z0 = scaled_gradient(&g0, -dt);
    let w1 = apply_algebra(&w0, &scaled_gradient(&z0, 1.0 / 4.0))?;

    let g1 = collect_gradients(&w1)?;
    let z1 = scaled_gradient(&g1, -dt);
    let stage2 = combine_two(&z1, 8.0 / 9.0, &z0, -17.0 / 36.0)?;
    let w2 = apply_algebra(&w1, &stage2)?;

    let g2 = collect_gradients(&w2)?;
    let z2 = scaled_gradient(&g2, -dt);
    let stage3 = combine_three(
        &z2,
        3.0 / 4.0,
        &z1,
        -8.0 / 9.0,
        &z0,
        17.0 / 36.0,
    )?;
    let updated = apply_algebra(&w2, &stage3)?;

    let action_after = updated.wilson_action(1.0)?;
    let clover_q_after = clover_topological_charge(&updated).map_err(StapleFlowError::from)?;
    let (max_unitarity_error, max_determinant_error) = su3_drift(&updated)?;
    *field = updated;

    Ok(Rk3FlowStepStats {
        flow: ReferenceFlowStepStats {
            action_before,
            action_after,
            clover_q_before,
            clover_q_after,
            max_unitarity_error,
            max_determinant_error,
        },
        force_evaluations: 3,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_su3_lie::generator_rotation;

    fn oracle_fixture() -> WilsonGaugeField {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        let operations = [
            ([0, 0, 1, 0], 0, 3, -0.5819354339098058),
            ([0, 1, 1, 1], 0, 2, -0.32393958624710995),
            ([1, 0, 1, 0], 1, 2, 0.19161237542499698),
            ([1, 1, 1, 1], 3, 5, 0.5114974521920254),
            ([1, 1, 1, 0], 1, 6, -0.4726273806149346),
            ([1, 0, 1, 1], 1, 3, 0.4032003002590304),
            ([0, 1, 0, 0], 2, 2, 0.3411282616087933),
            ([1, 1, 0, 0], 0, 5, -0.42942950746005015),
        ];
        for (site, mu, generator, theta) in operations {
            let original = *field.link(site, mu).unwrap();
            field
                .set_link(
                    site,
                    mu,
                    su3_mul(&generator_rotation(generator, theta), &original),
                )
                .unwrap();
        }
        field
    }

    fn run_to_fixed_time(dt: f64, total_time: f64) -> (f64, f64) {
        let steps = (total_time / dt).round() as usize;
        assert!(((steps as f64) * dt - total_time).abs() < 1.0e-14);
        let mut field = oracle_fixture();
        let mut previous_action = field.wilson_action(1.0).unwrap();
        for _ in 0..steps {
            let stats = rk3_wilson_flow_step(&mut field, dt).unwrap();
            assert_eq!(stats.force_evaluations, 3);
            assert!(stats.flow.action_after <= previous_action + 1.0e-12);
            previous_action = stats.flow.action_after;
        }
        (
            field.wilson_action(1.0).unwrap(),
            clover_topological_charge(&field).unwrap(),
        )
    }

    #[test]
    fn rk3_fixed_time_values_match_independent_oracle() {
        let cases = [
            (0.004, 2.511_434_591_009_796_2, 1.028_890_449_068_059_9e-5),
            (0.002, 2.511_434_785_428_211_3, 1.028_890_515_856_474_8e-5),
            (0.001, 2.511_434_809_397_332_7, 1.028_890_524_107_447_2e-5),
            (0.0005, 2.511_434_812_372_878_7, 1.028_890_525_132_800_4e-5),
        ];
        for (dt, expected_action, expected_q) in cases {
            let (action, q) = run_to_fixed_time(dt, 0.008);
            assert!((action - expected_action).abs() < 5.0e-12);
            assert!((q - expected_q).abs() < 5.0e-15);
        }
    }

    #[test]
    fn rk3_exhibits_third_order_step_halving_on_oracle_fixture() {
        let dts = [0.004, 0.002, 0.001, 0.0005];
        let results: Vec<(f64, f64)> = dts
            .into_iter()
            .map(|dt| run_to_fixed_time(dt, 0.008))
            .collect();
        let action_diff = [
            (results[0].0 - results[1].0).abs(),
            (results[1].0 - results[2].0).abs(),
            (results[2].0 - results[3].0).abs(),
        ];
        let q_diff = [
            (results[0].1 - results[1].1).abs(),
            (results[1].1 - results[2].1).abs(),
            (results[2].1 - results[3].1).abs(),
        ];
        for ratio in [
            action_diff[0] / action_diff[1],
            action_diff[1] / action_diff[2],
            q_diff[0] / q_diff[1],
            q_diff[1] / q_diff[2],
        ] {
            assert!((7.5..8.5).contains(&ratio));
        }
    }

    #[test]
    fn rk3_rejects_degenerate_extents_instead_of_changing_integrator() {
        let mut field = WilsonGaugeField::identity([2, 2, 1, 2]).unwrap();
        assert!(matches!(
            rk3_wilson_flow_step(&mut field, 1.0e-3),
            Err(Rk3FlowError::DegenerateExtent([2, 2, 1, 2]))
        ));
    }

    #[test]
    fn invalid_dt_fails_closed() {
        let mut field = oracle_fixture();
        assert!(matches!(
            rk3_wilson_flow_step(&mut field, 0.0),
            Err(Rk3FlowError::InvalidFlowStep(0.0))
        ));
    }
}
