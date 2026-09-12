// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Clover gauge-energy density and ensemble-mean gradient-flow scale algebra.
//!
//! The energy normalization and crossing semantics are independently qualified
//! by LQCD-018A. This module does not generate an ensemble and does not assign a
//! physical lattice spacing. `t0`/`w0`-like extraction accepts only explicitly
//! named ensemble-mean input points and a caller-supplied target.

use crate::lattice_gauge::{Site4, Su3Matrix, WilsonGaugeField, su3_mul, su3_trace};
use crate::lattice_topology_flow::{LatticeTopologyFlowError, clover_field_strength};

pub const CLOVER_FLOW_ENERGY_ID: &str = "symthaea_clover_energy_v1";

const PLANES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnsembleFlowEnergyPoint {
    pub flow_time: f64,
    /// Ensemble mean `<E(t)>` in the declared lattice/flow convention.
    pub ensemble_mean_energy: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlowEnergyError {
    Topology(LatticeTopologyFlowError),
    TooFewPoints { required: usize, actual: usize },
    InvalidTarget(f64),
    NonFinitePoint { index: usize },
    NonPositiveFlowTime { index: usize, value: f64 },
    NegativeMeanEnergy { index: usize, value: f64 },
    NonIncreasingFlowTime {
        previous: f64,
        current: f64,
    },
    CrossingNotFound,
    AmbiguousCrossing { count: usize },
    NonPositiveW0Squared(f64),
}

impl From<LatticeTopologyFlowError> for FlowEnergyError {
    fn from(value: LatticeTopologyFlowError) -> Self {
        Self::Topology(value)
    }
}

/// `E = sum_{mu<nu} Tr(F_munu F_munu)` for six Hermitian traceless
/// fundamental-representation field-strength matrices.
///
/// With `F=F^a lambda^a/2` this equals `(1/4) F^a_munu F^a_munu` under the
/// usual repeated-index continuum convention.
pub fn energy_density_from_field_strengths(fields: &[Su3Matrix; 6]) -> f64 {
    fields
        .iter()
        .map(|field| su3_trace(&su3_mul(field, field)).re)
        .sum()
}

pub fn clover_energy_density_at_site(
    field: &WilsonGaugeField,
    site: Site4,
) -> Result<f64, FlowEnergyError> {
    let fields = [
        clover_field_strength(field, site, 0, 1)?,
        clover_field_strength(field, site, 0, 2)?,
        clover_field_strength(field, site, 0, 3)?,
        clover_field_strength(field, site, 1, 2)?,
        clover_field_strength(field, site, 1, 3)?,
        clover_field_strength(field, site, 2, 3)?,
    ];
    Ok(energy_density_from_field_strengths(&fields))
}

/// Lattice-volume mean of the clover energy density on one gauge field.
///
/// This is a per-configuration observable. It is **not** an ensemble mean and
/// therefore cannot by itself be passed off as a physical `t0`/`w0` result.
pub fn mean_clover_energy_density(field: &WilsonGaugeField) -> Result<f64, FlowEnergyError> {
    let dims = field.dims();
    let mut sum = 0.0;
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    sum += clover_energy_density_at_site(field, [x, y, z, t])?;
                }
            }
        }
    }
    Ok(sum / field.site_count() as f64)
}

fn validate_points(
    points: &[EnsembleFlowEnergyPoint],
    required: usize,
) -> Result<(), FlowEnergyError> {
    if points.len() < required {
        return Err(FlowEnergyError::TooFewPoints {
            required,
            actual: points.len(),
        });
    }
    for (index, point) in points.iter().enumerate() {
        if !point.flow_time.is_finite() || !point.ensemble_mean_energy.is_finite() {
            return Err(FlowEnergyError::NonFinitePoint { index });
        }
        if point.flow_time <= 0.0 {
            return Err(FlowEnergyError::NonPositiveFlowTime {
                index,
                value: point.flow_time,
            });
        }
        if point.ensemble_mean_energy < 0.0 {
            return Err(FlowEnergyError::NegativeMeanEnergy {
                index,
                value: point.ensemble_mean_energy,
            });
        }
        if index > 0 && point.flow_time <= points[index - 1].flow_time {
            return Err(FlowEnergyError::NonIncreasingFlowTime {
                previous: points[index - 1].flow_time,
                current: point.flow_time,
            });
        }
    }
    Ok(())
}

fn unique_linear_crossing(xs: &[f64], ys: &[f64], target: f64) -> Result<f64, FlowEnergyError> {
    if !target.is_finite() || target <= 0.0 {
        return Err(FlowEnergyError::InvalidTarget(target));
    }

    let exact = ys
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value == target).then_some(index))
        .collect::<Vec<_>>();
    if exact.len() == 1 {
        return Ok(xs[exact[0]]);
    }
    if exact.len() > 1 {
        return Err(FlowEnergyError::AmbiguousCrossing { count: exact.len() });
    }

    let mut bracket: Option<(usize, f64)> = None;
    let mut count = 0usize;
    for index in 0..(xs.len() - 1) {
        let left = ys[index] - target;
        let right = ys[index + 1] - target;
        if left * right < 0.0 {
            count += 1;
            let fraction = (target - ys[index]) / (ys[index + 1] - ys[index]);
            bracket = Some((index, fraction));
        }
    }
    match (count, bracket) {
        (0, _) => Err(FlowEnergyError::CrossingNotFound),
        (1, Some((index, fraction))) => Ok(xs[index] + fraction * (xs[index + 1] - xs[index])),
        _ => Err(FlowEnergyError::AmbiguousCrossing { count }),
    }
}

pub fn dimensionless_flow_energy_curve(
    points: &[EnsembleFlowEnergyPoint],
) -> Result<Vec<(f64, f64)>, FlowEnergyError> {
    validate_points(points, 2)?;
    Ok(points
        .iter()
        .map(|point| {
            (
                point.flow_time,
                point.flow_time * point.flow_time * point.ensemble_mean_energy,
            )
        })
        .collect())
}

/// Generic `t0`-like crossing of `t^2 <E(t)>` at a caller-supplied target.
pub fn t0_like_from_ensemble_mean(
    points: &[EnsembleFlowEnergyPoint],
    target: f64,
) -> Result<f64, FlowEnergyError> {
    let curve = dimensionless_flow_energy_curve(points)?;
    let xs = curve.iter().map(|point| point.0).collect::<Vec<_>>();
    let ys = curve.iter().map(|point| point.1).collect::<Vec<_>>();
    unique_linear_crossing(&xs, &ys, target)
}

/// Generic `w0`-like crossing of `t d/dt[t^2<E(t)>]`.
///
/// A centered finite difference is formed at interior samples. The returned
/// value is `sqrt(t_cross)`. The target remains an external scientific input.
pub fn w0_like_from_ensemble_mean(
    points: &[EnsembleFlowEnergyPoint],
    target: f64,
) -> Result<f64, FlowEnergyError> {
    validate_points(points, 4)?;
    if !target.is_finite() || target <= 0.0 {
        return Err(FlowEnergyError::InvalidTarget(target));
    }
    let curve = points
        .iter()
        .map(|point| point.flow_time * point.flow_time * point.ensemble_mean_energy)
        .collect::<Vec<_>>();
    let mut times = Vec::with_capacity(points.len() - 2);
    let mut response = Vec::with_capacity(points.len() - 2);
    for index in 1..(points.len() - 1) {
        let derivative = (curve[index + 1] - curve[index - 1])
            / (points[index + 1].flow_time - points[index - 1].flow_time);
        times.push(points[index].flow_time);
        response.push(points[index].flow_time * derivative);
    }
    let t_cross = unique_linear_crossing(&times, &response, target)?;
    if t_cross <= 0.0 {
        return Err(FlowEnergyError::NonPositiveW0Squared(t_cross));
    }
    Ok(t_cross.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetry_groups::{Complex, gell_mann_matrix};

    const COEFFICIENTS: [[f64; 8]; 6] = [
        [0.12, -0.04, 0.07, 0.00, 0.03, -0.02, 0.01, 0.05],
        [-0.03, 0.08, 0.00, -0.06, 0.02, 0.01, 0.04, -0.02],
        [0.05, 0.00, -0.09, 0.03, -0.01, 0.02, 0.00, 0.04],
        [0.00, -0.02, 0.06, 0.05, 0.01, -0.03, 0.02, 0.00],
        [0.04, 0.03, -0.01, 0.00, -0.05, 0.07, 0.02, 0.01],
        [-0.02, 0.01, 0.03, -0.04, 0.06, 0.00, -0.05, 0.02],
    ];

    fn matrix_from_coefficients(coefficients: [f64; 8]) -> Su3Matrix {
        let mut out = [[Complex::ZERO; 3]; 3];
        for (generator, coefficient) in coefficients.into_iter().enumerate() {
            let lambda = gell_mann_matrix(generator);
            for i in 0..3 {
                for j in 0..3 {
                    out[i][j].re += 0.5 * coefficient * lambda[i][j].re;
                    out[i][j].im += 0.5 * coefficient * lambda[i][j].im;
                }
            }
        }
        out
    }

    #[test]
    fn field_strength_energy_matches_independent_oracle_normalization() {
        let fields = COEFFICIENTS.map(matrix_from_coefficients);
        let energy = energy_density_from_field_strengths(&fields);
        assert!((energy - 0.039_850_000_000_000_003).abs() < 2.0e-16);
    }

    #[test]
    fn identity_gauge_field_has_zero_clover_energy() {
        let field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        assert_eq!(mean_clover_energy_density(&field).unwrap(), 0.0);
    }

    fn synthetic_t0_points() -> Vec<EnsembleFlowEnergyPoint> {
        let times = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60];
        let f = [0.10, 0.18, 0.26, 0.34, 0.42, 0.50];
        times
            .into_iter()
            .zip(f)
            .map(|(flow_time, dimensionless)| EnsembleFlowEnergyPoint {
                flow_time,
                ensemble_mean_energy: dimensionless / (flow_time * flow_time),
            })
            .collect()
    }

    #[test]
    fn t0_like_interpolation_matches_independent_synthetic_oracle() {
        let value = t0_like_from_ensemble_mean(&synthetic_t0_points(), 0.30).unwrap();
        assert!((value - 0.35).abs() < 1.0e-15);
    }

    #[test]
    fn exact_sampled_crossing_is_not_double_counted() {
        let points = [
            EnsembleFlowEnergyPoint { flow_time: 0.1, ensemble_mean_energy: 10.0 },
            EnsembleFlowEnergyPoint { flow_time: 0.2, ensemble_mean_energy: 5.0 },
            EnsembleFlowEnergyPoint { flow_time: 0.3, ensemble_mean_energy: 0.4 / 0.09 },
        ];
        // t^2<E> = [0.1, 0.2, 0.4].
        assert_eq!(t0_like_from_ensemble_mean(&points, 0.2).unwrap(), 0.2);
    }

    #[test]
    fn w0_like_interpolation_matches_independent_synthetic_oracle() {
        let times = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60];
        let points = times
            .into_iter()
            .map(|flow_time| {
                let dimensionless = 0.02 + 0.8 * flow_time;
                EnsembleFlowEnergyPoint {
                    flow_time,
                    ensemble_mean_energy: dimensionless / (flow_time * flow_time),
                }
            })
            .collect::<Vec<_>>();
        let value = w0_like_from_ensemble_mean(&points, 0.32).unwrap();
        assert!((value - 0.4_f64.sqrt()).abs() < 2.0e-15);
    }

    #[test]
    fn ambiguous_multiple_crossing_fails_closed() {
        let values = [0.2, 0.4, 0.2];
        let times = [0.1, 0.2, 0.3];
        let points = times
            .into_iter()
            .zip(values)
            .map(|(flow_time, dimensionless)| EnsembleFlowEnergyPoint {
                flow_time,
                ensemble_mean_energy: dimensionless / (flow_time * flow_time),
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            t0_like_from_ensemble_mean(&points, 0.3),
            Err(FlowEnergyError::AmbiguousCrossing { count: 2 })
        ));
    }
}
