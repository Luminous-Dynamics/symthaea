// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent 3+1 ADM constraint projections for the static amplifier shift ansatz.
//!
//! The reference metric has alpha=1, gamma_ij=delta_ij and a static beta^i(x).
//! With the declared convention
//!
//!     K_ij = (1/2) (partial_i beta_j + partial_j beta_i),
//!
//! the flat-slice constraints are
//!
//!     H = K^2 - K_ij K^ij - 16 pi rho = 0,
//!     M^i = partial_j(K^ij - delta^ij K) - 8 pi j^i = 0.
//!
//! Here rho and j^i are projected independently from the 4D Einstein-implied
//! T_mu_nu already reconstructed by implied_source_diagnostics. These
//! constraints do not establish a realizable matter source; they cross-check
//! whether the 4D curvature calculation and the ADM projections agree for
//! the same metric ansatz.
//!
//! Spatial derivatives of beta are computed by centered finite differences.
//! The method is deliberately independent of the 4D curvature route, but it
//! is not analytic differentiation. Step-size sweeps and a future AD/analytic
//! derivative route remain required for numerical qualification.

use crate::adm_reference::{
    AdmReferenceError, AmplifierAdmMetric, implied_source_diagnostics,
};
use std::error::Error;
use std::fmt;

/// Independent 3+1 constraint diagnostics at one spacetime point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmConstraintDiagnostics {
    pub derivative_step_m: f64,
    pub extrinsic_curvature: [[f64; 3]; 3],
    pub extrinsic_curvature_trace: f64,
    pub extrinsic_curvature_squared: f64,
    pub hamiltonian_geometric_term: f64,
    pub eulerian_energy_density: f64,
    /// R^(3) + K^2 - K_ij K^ij - 16 pi rho; R^(3)=0 for this ansatz.
    pub hamiltonian_residual: f64,
    /// Eulerian momentum density j^i in Cartesian coordinates.
    pub eulerian_momentum_density: [f64; 3],
    /// D_j(K^ij - delta^ij K), with D the flat spatial derivative.
    pub momentum_geometric_term: [f64; 3],
    /// D_j(K^ij - delta^ij K) - 8 pi j^i.
    pub momentum_constraint_residual: [f64; 3],
    pub momentum_residual_norm: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AdmConstraintError {
    InvalidDerivativeStep,
    NonFiniteDerivative,
    Source(AdmReferenceError),
}

impl fmt::Display for AdmConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDerivativeStep => {
                write!(f, "ADM constraint derivative step must be finite and positive")
            }
            Self::NonFiniteDerivative => {
                write!(f, "ADM shift derivative is not finite")
            }
            Self::Source(error) => write!(f, "4D source projection failed: {error}"),
        }
    }
}

impl Error for AdmConstraintError {}

impl From<AdmReferenceError> for AdmConstraintError {
    fn from(value: AdmReferenceError) -> Self {
        Self::Source(value)
    }
}

/// Compute Hamiltonian and momentum constraints from an independent spatial
/// derivative path and compare them with the 4D Einstein-implied source.
///
/// The 4D source is evaluated using implied_source_diagnostics; the
/// geometric constraint terms are assembled separately from first and second
/// spatial derivatives of the ADM shift. The caller should sweep
/// field.derivative_step_m and report the raw residuals and convergence,
/// rather than treating one finite-difference setting as proof of consistency.
pub fn adm_constraint_diagnostics(
    field: &AmplifierAdmMetric,
    coordinates: [f64; 4],
    null_direction_count: usize,
) -> Result<AdmConstraintDiagnostics, AdmConstraintError> {
    let step = field.derivative_step_m;
    if !step.is_finite() || step <= 0.0 {
        return Err(AdmConstraintError::InvalidDerivativeStep);
    }

    let position = [coordinates[1], coordinates[2], coordinates[3]];
    let beta = field.shift_at(position)?;
    let first = first_shift_derivatives(field, position, step)?;
    let second = second_shift_derivatives(field, position, step)?;

    let mut extrinsic_curvature = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            extrinsic_curvature[i][j] = 0.5 * (first[j][i] + first[i][j]);
        }
    }
    let extrinsic_curvature_trace =
        (0..3).map(|axis| extrinsic_curvature[axis][axis]).sum::<f64>();
    let extrinsic_curvature_squared = extrinsic_curvature
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>();
    let hamiltonian_geometric_term =
        extrinsic_curvature_trace * extrinsic_curvature_trace - extrinsic_curvature_squared;

    // For alpha=1 and n^mu=(1,-beta^i), rho=T_mu_nu n^mu n^nu and
    // j_i=-T_i_mu n^mu = -T_i0 + beta^k T_ik.
    let source = implied_source_diagnostics(field, coordinates, null_direction_count)?;
    let stress = source.stress_energy;
    let mut eulerian_momentum_density = [0.0; 3];
    for i in 0..3 {
        eulerian_momentum_density[i] = -stress[i + 1][0]
            + (0..3)
                .map(|k| beta[k] * stress[i + 1][k + 1])
                .sum::<f64>();
    }

    // For Euclidean gamma_ij=delta_ij, D_j is an ordinary partial derivative.
    // With K_ij=1/2(partial_i beta_j + partial_j beta_i),
    // partial_j K_ij - partial_i K = 1/2(Delta beta_i - partial_i div beta).
    let mut momentum_geometric_term = [0.0; 3];
    for i in 0..3 {
        let laplacian_beta_i = (0..3).map(|axis| second[i][axis][axis]).sum::<f64>();
        let gradient_divergence = (0..3).map(|axis| second[axis][i][axis]).sum::<f64>();
        momentum_geometric_term[i] = 0.5 * (laplacian_beta_i - gradient_divergence);
    }

    let hamiltonian_residual =
        hamiltonian_geometric_term - 16.0 * std::f64::consts::PI * source.eulerian_energy_density;
    let mut momentum_constraint_residual = [0.0; 3];
    for i in 0..3 {
        momentum_constraint_residual[i] =
            momentum_geometric_term[i] - 8.0 * std::f64::consts::PI * eulerian_momentum_density[i];
    }
    let momentum_residual_norm = momentum_constraint_residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();

    if !hamiltonian_geometric_term.is_finite()
        || !hamiltonian_residual.is_finite()
        || !momentum_residual_norm.is_finite()
        || eulerian_momentum_density
            .iter()
            .chain(momentum_geometric_term.iter())
            .chain(momentum_constraint_residual.iter())
            .any(|value| !value.is_finite())
    {
        return Err(AdmConstraintError::NonFiniteDerivative);
    }

    Ok(AdmConstraintDiagnostics {
        derivative_step_m: step,
        extrinsic_curvature,
        extrinsic_curvature_trace,
        extrinsic_curvature_squared,
        hamiltonian_geometric_term,
        eulerian_energy_density: source.eulerian_energy_density,
        hamiltonian_residual,
        eulerian_momentum_density,
        momentum_geometric_term,
        momentum_constraint_residual,
        momentum_residual_norm,
    })
}

fn first_shift_derivatives(
    field: &AmplifierAdmMetric,
    position: [f64; 3],
    step: f64,
) -> Result<[[f64; 3]; 3], AdmConstraintError> {
    let mut derivatives = [[0.0; 3]; 3];
    for axis in 0..3 {
        let mut plus = position;
        let mut minus = position;
        plus[axis] += step;
        minus[axis] -= step;
        let beta_plus = field.shift_at(plus)?;
        let beta_minus = field.shift_at(minus)?;
        for component in 0..3 {
            derivatives[axis][component] =
                (beta_plus[component] - beta_minus[component]) / (2.0 * step);
            if !derivatives[axis][component].is_finite() {
                return Err(AdmConstraintError::NonFiniteDerivative);
            }
        }
    }
    Ok(derivatives)
}

fn second_shift_derivatives(
    field: &AmplifierAdmMetric,
    position: [f64; 3],
    step: f64,
) -> Result<[[[f64; 3]; 3]; 3], AdmConstraintError> {
    // result[component][first_axis][second_axis]
    let centre = field.shift_at(position)?;
    let mut second = [[[0.0; 3]; 3]; 3];

    for first_axis in 0..3 {
        for second_axis in first_axis..3 {
            if first_axis == second_axis {
                let mut plus = position;
                let mut minus = position;
                plus[first_axis] += step;
                minus[first_axis] -= step;
                let beta_plus = field.shift_at(plus)?;
                let beta_minus = field.shift_at(minus)?;
                for component in 0..3 {
                    let value = (beta_plus[component] - 2.0 * centre[component]
                        + beta_minus[component])
                        / (step * step);
                    second[component][first_axis][second_axis] = value;
                    second[component][second_axis][first_axis] = value;
                }
            } else {
                let mut pp = position;
                let mut pm = position;
                let mut mp = position;
                let mut mm = position;
                pp[first_axis] += step;
                pp[second_axis] += step;
                pm[first_axis] += step;
                pm[second_axis] -= step;
                mp[first_axis] -= step;
                mp[second_axis] += step;
                mm[first_axis] -= step;
                mm[second_axis] -= step;
                let beta_pp = field.shift_at(pp)?;
                let beta_pm = field.shift_at(pm)?;
                let beta_mp = field.shift_at(mp)?;
                let beta_mm = field.shift_at(mm)?;
                for component in 0..3 {
                    let value = (beta_pp[component] - beta_pm[component] - beta_mp[component]
                        + beta_mm[component])
                        / (4.0 * step * step);
                    second[component][first_axis][second_axis] = value;
                    second[component][second_axis][first_axis] = value;
                }
            }
        }
    }

    if second.iter().flatten().flatten().any(|value| !value.is_finite()) {
        return Err(AdmConstraintError::NonFiniteDerivative);
    }
    Ok(second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GravcraftConfig, MetricCommand};

    fn flat_field() -> AmplifierAdmMetric {
        AmplifierAdmMetric::new(
            [0.0; 3],
            MetricCommand::default(),
            GravcraftConfig::default(),
        )
        .unwrap()
    }

    fn shifted_field(step: f64) -> AmplifierAdmMetric {
        let mut field = AmplifierAdmMetric::new(
            [0.0; 3],
            MetricCommand {
                amplifiers: [
                    (0.05, 10.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),
                ],
            },
            GravcraftConfig::default(),
        )
        .unwrap();
        field.derivative_step_m = step;
        field
    }

    #[test]
    fn zero_shift_satisfies_both_adm_constraints() {
        let diagnostics =
            adm_constraint_diagnostics(&flat_field(), [0.0, 15.0, 0.0, 0.0], 24).unwrap();
        assert!(diagnostics.hamiltonian_residual.abs() < 1.0e-10);
        assert!(diagnostics.momentum_residual_norm < 1.0e-10);
        assert!(diagnostics.extrinsic_curvature_squared.abs() < 1.0e-12);
    }

    #[test]
    fn shifted_metric_reports_finite_independent_constraint_residuals() {
        let diagnostics =
            adm_constraint_diagnostics(&shifted_field(1.0e-2), [0.0, 15.0, 0.5, 0.25], 48)
                .unwrap();
        assert!(diagnostics.hamiltonian_residual.is_finite());
        assert!(diagnostics.momentum_residual_norm.is_finite());
        assert!(diagnostics
            .extrinsic_curvature
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn residuals_are_reported_across_derivative_steps_without_a_single_step_claim() {
        let steps = [2.0e-2, 1.0e-2, 5.0e-3];
        let diagnostics: Vec<_> = steps
            .iter()
            .map(|step| {
                adm_constraint_diagnostics(
                    &shifted_field(*step),
                    [0.0, 15.0, 0.5, 0.25],
                    24,
                )
                .unwrap()
            })
            .collect();
        assert!(diagnostics
            .iter()
            .all(|entry| entry.hamiltonian_residual.is_finite()
                && entry.momentum_residual_norm.is_finite()));
        // Do not require monotone convergence from nested finite differences;
        // the executable report exposes the actual step dependence.
    }

    #[test]
    fn invalid_derivative_step_fails_closed() {
        let mut field = flat_field();
        field.derivative_step_m = 0.0;
        assert_eq!(
            adm_constraint_diagnostics(&field, [0.0, 1.0, 0.0, 0.0], 24),
            Err(AdmConstraintError::InvalidDerivativeStep)
        );
    }
}
