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
    su3_determinant_error, su3_mul, su3_trace, su3_unitarity_error,
};
use crate::lattice_su3_lie::{
    matrix_add, matrix_add_assign, matrix_exp, matrix_scale, zero_matrix,
};
use crate::lattice_topology_flow::{
    LatticeTopologyFlowError, ReferenceFlowStepStats, clover_topological_charge,
    finite_difference_wilson_flow_step_reference,
};
use crate::symmetry_groups::{Complex, gell_mann_matrix};

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

/// Six-staple matrix `H` oriented so the local Wilson trace is `Re Tr(U_mu(x) H)`.
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

/// One optimized simultaneous Lie-Euler Wilson-action flow step.
///
/// `reference_epsilon` is numerically consumed only when a degenerate extent
/// forces this step through the finite-difference semantic reference. It stays
/// in the API so fallback behavior is explicit and reproducible.
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
            hermitian = matrix_add(
                &hermitian,
                &matrix_scale(
                    &gell_mann_matrix(generator),
                    Complex::new(coefficient, 0.0),
                ),
            );
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
    use crate::lattice_su3_lie::generator_rotation;

    fn fixture() -> WilsonGaugeField {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        let operations = [
            ([0, 0, 0, 0], 0, 2, 0.31),
            ([1, 0, 1, 0], 2, 4, -0.27),
            ([0, 1, 0, 1], 3, 6, 0.22),
            ([1, 1, 1, 1], 1, 0, 0.19),
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
            plus
                .set_link(
                    site,
                    mu,
                    su3_mul(&generator_rotation(generator, epsilon), &original),
                )
                .unwrap();
            minus
                .set_link(
                    site,
                    mu,
                    su3_mul(&generator_rotation(generator, -epsilon), &original),
                )
                .unwrap();
            *component = (plus.wilson_action(1.0).unwrap()
                - minus.wilson_action(1.0).unwrap())
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
    fn analytic_gradient_matches_reference_on_every_link_and_generator() {
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
        assert!(max_error < 1.5e-9);
    }

    #[test]
    fn optimized_one_step_matches_finite_difference_reference() {
        let field = fixture();
        let mut reference = field.clone();
        let mut optimized = field.clone();
        finite_difference_wilson_flow_step_reference(&mut reference, 1.0e-3, 2.0e-6).unwrap();
        let stats = staple_wilson_flow_step(&mut optimized, 1.0e-3, 2.0e-6).unwrap();
        assert!(!stats.used_reference_fallback);
        assert!(field_max_error(&reference, &optimized) < 1.5e-12);
        assert!(stats.flow.action_after < stats.flow.action_before);
        assert!(stats.flow.max_determinant_error < 2.0e-12);
        assert!(stats.flow.max_unitarity_error < 2.0e-12);
    }

    #[test]
    fn degenerate_extent_falls_back_exactly() {
        let mut field = WilsonGaugeField::identity([2, 2, 1, 2]).unwrap();
        let original = *field.link([0, 0, 0, 0], 0).unwrap();
        field
            .set_link(
                [0, 0, 0, 0],
                0,
                su3_mul(&generator_rotation(2, 0.31), &original),
            )
            .unwrap();
        let mut reference = field.clone();
        let mut optimized = field.clone();
        finite_difference_wilson_flow_step_reference(&mut reference, 1.0e-3, 2.0e-6).unwrap();
        let stats = staple_wilson_flow_step(&mut optimized, 1.0e-3, 2.0e-6).unwrap();
        assert!(stats.used_reference_fallback);
        assert!(field_max_error(&reference, &optimized) < 1.0e-15);
    }
}
