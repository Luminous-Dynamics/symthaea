// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Positive, conservative cell-average pullback for an instantaneous velocity kick.
//!
//! Given the canonical map v+ = v- + delta_v(x), the distribution pullback is
//! f+(x,v) = f-(x,v-delta_v(x)). This module evaluates that pullback on
//! cell-average data using exact overlap integration of a piecewise-constant
//! reconstruction. The remap is positivity-preserving and conservative on an
//! unbounded velocity line. On a finite velocity domain it reports boundary
//! outflow explicitly; it never renormalizes or hides lost mass.
//!
//! This is a first-order remap operator for one intervention. It is not the
//! collisionless Vlasov transport solver and does not supply a full nonlinear
//! distribution tangent/adjoint.

use crate::pulse_map::{PulseMapError, SinusoidalVelocityKick};
use thiserror::Error;

/// Errors from the conservative velocity kick remap.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum VelocityKickRemapError {
    #[error("spatial positions, velocity edges, and distribution dimensions are inconsistent")]
    ShapeMismatch,
    #[error("velocity grid edges must be finite and strictly increasing")]
    InvalidVelocityGrid,
    #[error("spatial position is non-finite")]
    InvalidSpatialPosition,
    #[error("spatial quadrature weights must be finite, positive, and match spatial rows")]
    InvalidSpatialQuadrature,
    #[error("distribution contains a negative or non-finite cell average")]
    InvalidDistribution,
    #[error("remap produced a non-finite cell average or mass ledger")]
    NonFiniteRemap,
    #[error("remap mass balance disagrees with analytically computed boundary outflow")]
    MassBalanceFailure,
    #[error("remap Jacobian configuration is invalid")]
    InvalidJacobianConfiguration,
    #[error("remap Jacobian storage budget exceeded")]
    JacobianStorageBudgetExceeded,
    #[error("velocity remap preimage lies at a cell-topology boundary")]
    TopologyBoundary,
    #[error(transparent)]
    Pulse(#[from] PulseMapError),
}

/// Mass ledger for one spatial cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialMassLedger {
    /// Positive spatial quadrature weight for this row.
    pub spatial_weight: f64,
    /// Velocity-integrated mass density before the kick, for this row.
    pub mass_before: f64,
    /// Velocity-integrated mass density after the kick, for this row.
    pub mass_after: f64,
    /// Positive mass transported outside the declared finite velocity domain.
    pub boundary_outflow: f64,
    /// mass_before - mass_after - boundary_outflow.
    pub residual: f64,
}

/// Result of applying a velocity kick to a cell-average 1D1V distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct VelocityKickRemapReport {
    /// Remapped cell averages, indexed [spatial cell][velocity cell].
    pub distribution: Vec<Vec<f64>>,
    /// Per-spatial-cell mass accounting.
    pub spatial_mass: Vec<SpatialMassLedger>,
    /// Spatial-quadrature-weighted mass before the kick.
    /// The unit-weight wrapper returns a row-summed diagnostic, not a general
    /// physical spatial integral.
    pub total_mass_before: f64,
    /// Spatial-quadrature-weighted mass after the kick.
    pub total_mass_after: f64,
    /// Spatial-quadrature-weighted boundary outflow.
    pub total_boundary_outflow: f64,
    pub total_mass_balance_residual: f64,
    pub relative_boundary_outflow: f64,
}

/// Apply the instantaneous characteristic pullback to a cell-average distribution.
///
/// This convenience entry point uses unit spatial weights. For nonuniform
/// spatial quadrature or a physically integrated 1D1V mass, call
/// remap_distribution_after_velocity_kick_weighted instead.
///
/// Inputs:
/// - spatial_positions: one representative x coordinate per row;
/// - velocity_edges: strictly increasing cell edges, length Nv+1;
/// - distribution: row-major cell averages, shape Nx by Nv.
///
/// The source reconstruction is piecewise constant inside each velocity cell.
/// For each target cell [v_j, v_(j+1)], the preimage interval
/// [v_j-delta_v(x), v_(j+1)-delta_v(x)] is intersected with the source cells.
/// Overlap integrals preserve positivity and finite-domain mass up to the
/// explicitly reported outflow. No periodicity is assumed in velocity.
pub fn remap_distribution_after_velocity_kick(
    kick: &SinusoidalVelocityKick,
    spatial_positions: &[f64],
    velocity_edges: &[f64],
    distribution: &[Vec<f64>],
) -> Result<VelocityKickRemapReport, VelocityKickRemapError> {
    let unit_weights = vec![1.0; spatial_positions.len()];
    remap_distribution_after_velocity_kick_weighted(
        kick,
        spatial_positions,
        &unit_weights,
        velocity_edges,
        distribution,
    )
}

/// Apply the velocity kick and integrate row ledgers with explicit spatial
/// quadrature weights. Weights must be finite and strictly positive.
pub fn remap_distribution_after_velocity_kick_weighted(
    kick: &SinusoidalVelocityKick,
    spatial_positions: &[f64],
    spatial_weights: &[f64],
    velocity_edges: &[f64],
    distribution: &[Vec<f64>],
) -> Result<VelocityKickRemapReport, VelocityKickRemapError> {
    kick.validate()?;
    if spatial_positions.is_empty()
        || velocity_edges.len() < 2
        || distribution.len() != spatial_positions.len()
    {
        return Err(VelocityKickRemapError::ShapeMismatch);
    }
    if spatial_weights.len() != spatial_positions.len()
        || spatial_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(VelocityKickRemapError::InvalidSpatialQuadrature);
    }
    if velocity_edges.iter().any(|edge| !edge.is_finite())
        || velocity_edges.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(VelocityKickRemapError::InvalidVelocityGrid);
    }
    let velocity_cells = velocity_edges.len() - 1;
    if distribution.iter().any(|row| row.len() != velocity_cells) {
        return Err(VelocityKickRemapError::ShapeMismatch);
    }

    let mut remapped = Vec::with_capacity(spatial_positions.len());
    let mut spatial_mass = Vec::with_capacity(spatial_positions.len());
    let mut total_mass_before = 0.0_f64;
    let mut total_mass_after = 0.0_f64;
    let mut total_boundary_outflow = 0.0_f64;
    let mut total_mass_balance_residual = 0.0_f64;

    for (row_index, (&x, &spatial_weight)) in
        spatial_positions.iter().zip(spatial_weights).enumerate()
    {
        if !x.is_finite() {
            return Err(VelocityKickRemapError::InvalidSpatialPosition);
        }
        let source = &distribution[row_index];
        if source
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(VelocityKickRemapError::InvalidDistribution);
        }
        let shift = kick.delta_v(x)?;
        let mut output = vec![0.0; velocity_cells];
        let mut mass_before = 0.0_f64;
        let mut mass_after = 0.0_f64;

        for (cell, &cell_average) in source.iter().enumerate() {
            let width = velocity_edges[cell + 1] - velocity_edges[cell];
            mass_before += cell_average * width;
            if !mass_before.is_finite() {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }
        }

        for (target_cell, output_cell) in output.iter_mut().enumerate() {
            let preimage_left = velocity_edges[target_cell] - shift;
            let preimage_right = velocity_edges[target_cell + 1] - shift;
            if !preimage_left.is_finite() || !preimage_right.is_finite() {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }

            // Locate the source cell containing the left endpoint. If the
            // preimage starts outside the grid, begin at the first cell.
            let partition = velocity_edges.partition_point(|edge| *edge <= preimage_left);
            let source_start = partition.saturating_sub(1);
            let mut integral = 0.0_f64;
            for (source_cell, &source_average) in source.iter().enumerate().skip(source_start) {
                let source_left = velocity_edges[source_cell];
                let source_right = velocity_edges[source_cell + 1];
                if source_left >= preimage_right {
                    break;
                }
                let overlap_left = preimage_left.max(source_left);
                let overlap_right = preimage_right.min(source_right);
                let overlap = (overlap_right - overlap_left).max(0.0);
                integral += source_average * overlap;
                if !integral.is_finite() {
                    return Err(VelocityKickRemapError::NonFiniteRemap);
                }
            }
            let target_width = velocity_edges[target_cell + 1] - velocity_edges[target_cell];
            let average = integral / target_width;
            if !average.is_finite() || average < 0.0 {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }
            *output_cell = average;
            mass_after += average * target_width;
            if !mass_after.is_finite() {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }
        }

        let domain_min = velocity_edges[0];
        let domain_max = velocity_edges[velocity_cells];
        let outflow_interval = if shift > 0.0 {
            Some((domain_max - shift, domain_max))
        } else if shift < 0.0 {
            Some((domain_min, domain_min - shift))
        } else {
            None
        };
        let mut boundary_outflow = 0.0_f64;
        if let Some((out_left, out_right)) = outflow_interval {
            for (source_cell, &source_average) in source.iter().enumerate() {
                let overlap_left = out_left.max(velocity_edges[source_cell]);
                let overlap_right = out_right.min(velocity_edges[source_cell + 1]);
                let overlap = (overlap_right - overlap_left).max(0.0);
                boundary_outflow += source_average * overlap;
                if !boundary_outflow.is_finite() {
                    return Err(VelocityKickRemapError::NonFiniteRemap);
                }
            }
        }

        let residual = mass_before - mass_after - boundary_outflow;
        let mass_scale = mass_before
            .abs()
            .max(mass_after.abs())
            .max(boundary_outflow.abs());
        let tolerance = 256.0 * f64::EPSILON * mass_scale * velocity_cells as f64;
        if !residual.is_finite() || residual.abs() > tolerance {
            return Err(VelocityKickRemapError::MassBalanceFailure);
        }

        total_mass_before += spatial_weight * mass_before;
        total_mass_after += spatial_weight * mass_after;
        total_boundary_outflow += spatial_weight * boundary_outflow;
        total_mass_balance_residual += spatial_weight * residual;
        if !total_mass_before.is_finite()
            || !total_mass_after.is_finite()
            || !total_boundary_outflow.is_finite()
            || !total_mass_balance_residual.is_finite()
        {
            return Err(VelocityKickRemapError::NonFiniteRemap);
        }
        spatial_mass.push(SpatialMassLedger {
            spatial_weight,
            mass_before,
            mass_after,
            boundary_outflow,
            residual,
        });
        remapped.push(output);
    }

    let relative_boundary_outflow = if total_mass_before > 0.0 {
        total_boundary_outflow / total_mass_before
    } else {
        0.0
    };
    if !relative_boundary_outflow.is_finite() {
        return Err(VelocityKickRemapError::NonFiniteRemap);
    }

    Ok(VelocityKickRemapReport {
        distribution: remapped,
        spatial_mass,
        total_mass_before,
        total_mass_after,
        total_boundary_outflow,
        total_mass_balance_residual,
        relative_boundary_outflow,
    })
}

/// Resource and topology controls for the fixed-overlap remap Jacobian.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VelocityKickRemapJacobianConfig {
    /// Minimum distance from shifted target edges to source velocity edges.
    pub topology_tolerance: f64,
    /// Peak scalar storage for all row Jacobians and derivative vectors.
    pub maximum_jacobian_scalars: usize,
}

impl VelocityKickRemapJacobianConfig {
    pub fn new(topology_tolerance: f64, maximum_jacobian_scalars: usize) -> Self {
        Self {
            topology_tolerance,
            maximum_jacobian_scalars,
        }
    }
}

/// Derivatives of one row's boundary outflow with respect to the local shift
/// and the sinusoidal-kick controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryOutflowJacobian {
    pub velocity_shift: f64,
    pub amplitude: f64,
    pub phase_rad: f64,
    pub spatial_position: f64,
}

/// Fixed-topology Jacobians for one spatial row.
#[derive(Clone, Debug, PartialEq)]
pub struct VelocityKickRemapRowJacobian {
    /// d(output cell averages) / d(input cell averages), shape Nv x Nv.
    pub distribution_wrt_input: Vec<Vec<f64>>,
    pub distribution_wrt_velocity_shift: Vec<f64>,
    pub distribution_wrt_amplitude: Vec<f64>,
    pub distribution_wrt_phase_rad: Vec<f64>,
    pub distribution_wrt_spatial_position: Vec<f64>,
    pub boundary_outflow: BoundaryOutflowJacobian,
    pub topology_margin: f64,
}

/// Fixed-topology Jacobians for the distribution pullback and outflow ledger.
#[derive(Clone, Debug, PartialEq)]
pub struct VelocityKickRemapJacobianReport {
    pub rows: Vec<VelocityKickRemapRowJacobian>,
    pub minimum_topology_margin: f64,
    pub total_boundary_outflow_d_amplitude: f64,
    pub total_boundary_outflow_d_phase_rad: f64,
    /// Weighted derivatives for the independent spatial-row shifts.
    pub total_boundary_outflow_d_velocity_shift: Vec<f64>,
    pub total_boundary_outflow_d_spatial_position: Vec<f64>,
    pub jacobian_scalar_count: usize,
}

/// Linearize the piecewise-constant velocity-kick pullback on a fixed overlap
/// topology. The supplied spatial weights are used for integrated outflow
/// derivatives; row-local distribution Jacobians remain unweighted.
///
/// For target cell j and source cell k:
///
/// d fbar_j+ / d fbar_k- = overlap_jk / target_width_j
///
/// d fbar_j+ / d delta_v =
///     (f-(v_j-delta_v) - f-(v_(j+1)-delta_v)) / target_width_j.
///
/// Amplitude, phase and spatial-position derivatives follow by the chain rule.
/// A topology boundary is refused rather than differentiating across a change
/// in the active overlap set.
pub fn linearize_velocity_kick_remap(
    kick: &SinusoidalVelocityKick,
    spatial_positions: &[f64],
    spatial_weights: &[f64],
    velocity_edges: &[f64],
    distribution: &[Vec<f64>],
    config: VelocityKickRemapJacobianConfig,
) -> Result<VelocityKickRemapJacobianReport, VelocityKickRemapError> {
    kick.validate()?;
    if !config.topology_tolerance.is_finite()
        || config.topology_tolerance <= 0.0
        || config.maximum_jacobian_scalars == 0
    {
        return Err(VelocityKickRemapError::InvalidJacobianConfiguration);
    }
    if spatial_positions.is_empty()
        || spatial_positions.len() != spatial_weights.len()
        || distribution.len() != spatial_positions.len()
        || velocity_edges.len() < 2
    {
        return Err(VelocityKickRemapError::ShapeMismatch);
    }
    if spatial_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(VelocityKickRemapError::InvalidSpatialQuadrature);
    }
    if velocity_edges.iter().any(|edge| !edge.is_finite())
        || velocity_edges.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(VelocityKickRemapError::InvalidVelocityGrid);
    }

    let velocity_cells = velocity_edges.len() - 1;
    if distribution.iter().any(|row| row.len() != velocity_cells) {
        return Err(VelocityKickRemapError::ShapeMismatch);
    }
    for row in distribution {
        if row.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(VelocityKickRemapError::InvalidDistribution);
        }
    }

    let per_row_matrix_scalars = velocity_cells
        .checked_mul(velocity_cells)
        .ok_or(VelocityKickRemapError::JacobianStorageBudgetExceeded)?;
    let per_row_vector_scalars = velocity_cells
        .checked_mul(4)
        .and_then(|value| value.checked_add(5))
        .ok_or(VelocityKickRemapError::JacobianStorageBudgetExceeded)?;
    let per_row_scalars = per_row_matrix_scalars
        .checked_add(per_row_vector_scalars)
        .ok_or(VelocityKickRemapError::JacobianStorageBudgetExceeded)?;
    let jacobian_scalar_count = spatial_positions
        .len()
        .checked_mul(per_row_scalars)
        .ok_or(VelocityKickRemapError::JacobianStorageBudgetExceeded)?;
    if jacobian_scalar_count > config.maximum_jacobian_scalars {
        return Err(VelocityKickRemapError::JacobianStorageBudgetExceeded);
    }

    let wave_number = std::f64::consts::TAU * f64::from(kick.mode) / kick.domain_length;
    let domain_min = velocity_edges[0];
    let domain_max = velocity_edges[velocity_cells];
    let mut rows = Vec::with_capacity(spatial_positions.len());
    let mut minimum_topology_margin = f64::INFINITY;
    let mut total_boundary_outflow_d_amplitude = 0.0_f64;
    let mut total_boundary_outflow_d_phase_rad = 0.0_f64;
    let mut total_boundary_outflow_d_velocity_shift =
        Vec::with_capacity(spatial_positions.len());
    let mut total_boundary_outflow_d_spatial_position =
        Vec::with_capacity(spatial_positions.len());

    for (row_index, (&position, &spatial_weight)) in
        spatial_positions.iter().zip(spatial_weights).enumerate()
    {
        if !position.is_finite() {
            return Err(VelocityKickRemapError::InvalidSpatialPosition);
        }
        let source = &distribution[row_index];
        let angle = kick.angle(position)?;
        let shift = kick.delta_v(position)?;
        let shift_wrt_position = kick.amplitude * wave_number * angle.cos();
        let shift_wrt_amplitude = angle.sin();
        let shift_wrt_phase = kick.amplitude * angle.cos();
        if !shift_wrt_position.is_finite()
            || !shift_wrt_amplitude.is_finite()
            || !shift_wrt_phase.is_finite()
        {
            return Err(VelocityKickRemapError::NonFiniteRemap);
        }

        let mut topology_margin = f64::INFINITY;
        for &target_edge in velocity_edges {
            let preimage = target_edge - shift;
            if !preimage.is_finite() {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }
            topology_margin = topology_margin.min(nearest_velocity_edge_distance(
                preimage,
                velocity_edges,
            ));
        }
        if !topology_margin.is_finite()
            || topology_margin <= config.topology_tolerance
        {
            return Err(VelocityKickRemapError::TopologyBoundary);
        }
        minimum_topology_margin = minimum_topology_margin.min(topology_margin);

        let mut distribution_wrt_input =
            vec![vec![0.0; velocity_cells]; velocity_cells];
        let mut distribution_wrt_velocity_shift = vec![0.0; velocity_cells];
        let mut distribution_wrt_amplitude = vec![0.0; velocity_cells];
        let mut distribution_wrt_phase_rad = vec![0.0; velocity_cells];
        let mut distribution_wrt_spatial_position = vec![0.0; velocity_cells];

        for target_cell in 0..velocity_cells {
            let preimage_left = velocity_edges[target_cell] - shift;
            let preimage_right = velocity_edges[target_cell + 1] - shift;
            let target_width =
                velocity_edges[target_cell + 1] - velocity_edges[target_cell];
            if !preimage_left.is_finite()
                || !preimage_right.is_finite()
                || !target_width.is_finite()
                || target_width <= 0.0
            {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }

            for source_cell in 0..velocity_cells {
                let overlap_left = preimage_left.max(velocity_edges[source_cell]);
                let overlap_right = preimage_right.min(velocity_edges[source_cell + 1]);
                let overlap = (overlap_right - overlap_left).max(0.0);
                let derivative = overlap / target_width;
                if !derivative.is_finite() {
                    return Err(VelocityKickRemapError::NonFiniteRemap);
                }
                distribution_wrt_input[target_cell][source_cell] = derivative;
            }

            let left_value =
                piecewise_constant_value_at(source, velocity_edges, preimage_left);
            let right_value =
                piecewise_constant_value_at(source, velocity_edges, preimage_right);
            let shift_derivative = (left_value - right_value) / target_width;
            let amplitude_derivative = shift_derivative * shift_wrt_amplitude;
            let phase_derivative = shift_derivative * shift_wrt_phase;
            let position_derivative = shift_derivative * shift_wrt_position;
            if !shift_derivative.is_finite()
                || !amplitude_derivative.is_finite()
                || !phase_derivative.is_finite()
                || !position_derivative.is_finite()
            {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }
            distribution_wrt_velocity_shift[target_cell] = shift_derivative;
            distribution_wrt_amplitude[target_cell] = amplitude_derivative;
            distribution_wrt_phase_rad[target_cell] = phase_derivative;
            distribution_wrt_spatial_position[target_cell] = position_derivative;
        }

        let outflow_derivative_wrt_shift = if shift > 0.0 {
            piecewise_constant_value_at(source, velocity_edges, domain_max - shift)
        } else if shift < 0.0 {
            -piecewise_constant_value_at(source, velocity_edges, domain_min - shift)
        } else {
            0.0
        };
        let boundary_outflow = BoundaryOutflowJacobian {
            velocity_shift: outflow_derivative_wrt_shift,
            amplitude: outflow_derivative_wrt_shift * shift_wrt_amplitude,
            phase_rad: outflow_derivative_wrt_shift * shift_wrt_phase,
            spatial_position: outflow_derivative_wrt_shift * shift_wrt_position,
        };
        if [
            boundary_outflow.velocity_shift,
            boundary_outflow.amplitude,
            boundary_outflow.phase_rad,
            boundary_outflow.spatial_position,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(VelocityKickRemapError::NonFiniteRemap);
        }

        total_boundary_outflow_d_amplitude +=
            spatial_weight * boundary_outflow.amplitude;
        total_boundary_outflow_d_phase_rad +=
            spatial_weight * boundary_outflow.phase_rad;
        let weighted_shift_derivative =
            spatial_weight * boundary_outflow.velocity_shift;
        let weighted_position_derivative =
            spatial_weight * boundary_outflow.spatial_position;
        if !total_boundary_outflow_d_amplitude.is_finite()
            || !total_boundary_outflow_d_phase_rad.is_finite()
            || !weighted_shift_derivative.is_finite()
            || !weighted_position_derivative.is_finite()
        {
            return Err(VelocityKickRemapError::NonFiniteRemap);
        }
        total_boundary_outflow_d_velocity_shift.push(weighted_shift_derivative);
        total_boundary_outflow_d_spatial_position.push(weighted_position_derivative);
        rows.push(VelocityKickRemapRowJacobian {
            distribution_wrt_input,
            distribution_wrt_velocity_shift,
            distribution_wrt_amplitude,
            distribution_wrt_phase_rad,
            distribution_wrt_spatial_position,
            boundary_outflow,
            topology_margin,
        });
    }

    Ok(VelocityKickRemapJacobianReport {
        rows,
        minimum_topology_margin,
        total_boundary_outflow_d_amplitude,
        total_boundary_outflow_d_phase_rad,
        total_boundary_outflow_d_velocity_shift,
        total_boundary_outflow_d_spatial_position,
        jacobian_scalar_count,
    })
}

fn nearest_velocity_edge_distance(point: f64, edges: &[f64]) -> f64 {
    let partition = edges.partition_point(|edge| *edge < point);
    let left_distance = if partition > 0 {
        (point - edges[partition - 1]).abs()
    } else {
        f64::INFINITY
    };
    let right_distance = if partition < edges.len() {
        (edges[partition] - point).abs()
    } else {
        f64::INFINITY
    };
    left_distance.min(right_distance)
}

fn piecewise_constant_value_at(values: &[f64], edges: &[f64], point: f64) -> f64 {
    if point < edges[0] || point >= edges[values.len()] {
        return 0.0;
    }
    let partition = edges.partition_point(|edge| *edge <= point);
    values[partition.saturating_sub(1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kick(amplitude: f64) -> SinusoidalVelocityKick {
        SinusoidalVelocityKick::new(0.0, 2.0 * std::f64::consts::PI, 1, amplitude, 0.0).unwrap()
    }

    fn uniform_edges(min: f64, max: f64, cells: usize) -> Vec<f64> {
        (0..=cells)
            .map(|index| min + (max - min) * index as f64 / cells as f64)
            .collect()
    }

    #[test]
    fn zero_kick_is_identity_and_preserves_each_cell_average() {
        let edges = uniform_edges(-2.0, 2.0, 8);
        let distribution = vec![vec![0.0, 0.2, 0.5, 1.0, 0.8, 0.4, 0.1, 0.0]];
        let report =
            remap_distribution_after_velocity_kick(&kick(0.0), &[0.7], &edges, &distribution)
                .unwrap();
        assert_eq!(report.distribution, distribution);
        assert_eq!(report.total_boundary_outflow, 0.0);
        assert_eq!(report.total_mass_balance_residual, 0.0);
    }

    #[test]
    fn interior_translation_preserves_mass_and_positivity() {
        let edges = uniform_edges(-4.0, 4.0, 64);
        let distribution = vec![
            (0..64)
                .map(|index| {
                    let center = 0.5 * (edges[index] + edges[index + 1]);
                    if (-1.0..=1.0).contains(&center) {
                        1.0 + 0.2 * center
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
        ];
        let report = remap_distribution_after_velocity_kick(
            &kick(0.1),
            &[std::f64::consts::PI / 2.0],
            &edges,
            &distribution,
        )
        .unwrap();
        assert!(report.distribution[0].iter().all(|value| *value >= 0.0));
        assert!(report.total_boundary_outflow < 1.0e-14);
        assert!((report.total_mass_after - report.total_mass_before).abs() < 1.0e-12);
        assert!(report.total_mass_balance_residual.abs() < 1.0e-12);
    }

    #[test]
    fn finite_domain_boundary_loss_is_reported_not_renormalized() {
        let edges = uniform_edges(0.0, 1.0, 10);
        let distribution = vec![vec![1.0; 10]];
        let report = remap_distribution_after_velocity_kick(
            &kick(0.1),
            &[std::f64::consts::PI / 2.0],
            &edges,
            &distribution,
        )
        .unwrap();
        assert!((report.total_mass_before - 1.0).abs() < 1.0e-12);
        assert!((report.total_boundary_outflow - 0.1).abs() < 1.0e-12);
        assert!((report.total_mass_after - 0.9).abs() < 1.0e-12);
        assert!(
            report
                .distribution
                .iter()
                .flatten()
                .all(|value| *value >= 0.0)
        );
    }

    #[test]
    fn negative_shift_reports_lower_boundary_outflow() {
        let edges = uniform_edges(0.0, 1.0, 10);
        let distribution = vec![vec![1.0; 10]];
        let reverse = SinusoidalVelocityKick::new(
            0.0,
            2.0 * std::f64::consts::PI,
            1,
            0.1,
            std::f64::consts::PI,
        )
        .unwrap();
        let report = remap_distribution_after_velocity_kick(
            &reverse,
            &[std::f64::consts::PI / 2.0],
            &edges,
            &distribution,
        )
        .unwrap();
        assert!((report.total_boundary_outflow - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn multiple_spatial_rows_keep_independent_kick_ledgers() {
        let edges = uniform_edges(-2.0, 2.0, 32);
        let row = vec![1.0; 32];
        let distribution = vec![row.clone(), row];
        let report = remap_distribution_after_velocity_kick(
            &kick(0.1),
            &[std::f64::consts::PI / 2.0, 3.0 * std::f64::consts::PI / 2.0],
            &edges,
            &distribution,
        )
        .unwrap();
        assert_eq!(report.spatial_mass.len(), 2);
        assert!((report.spatial_mass[0].boundary_outflow - 0.1).abs() < 1.0e-12);
        assert!((report.spatial_mass[1].boundary_outflow - 0.1).abs() < 1.0e-12);
        assert!(
            report
                .spatial_mass
                .iter()
                .all(|ledger| ledger.residual.abs() < 1.0e-12)
        );
    }

    #[test]
    fn fixed_topology_remap_jacobian_matches_amplitude_phase_and_outflow_finite_differences() {
        let kick = SinusoidalVelocityKick::new(
            0.0,
            2.0 * std::f64::consts::PI,
            1,
            0.1,
            0.2,
        )
        .unwrap();
        let positions = [0.73];
        let weights = [0.5];
        let edges = uniform_edges(0.0, 0.8, 16);
        let distribution = vec![vec![1.0; 16]];
        let config = VelocityKickRemapJacobianConfig::new(1.0e-5, 10_000);
        let analytic = linearize_velocity_kick_remap(
            &kick,
            &positions,
            &weights,
            &edges,
            &distribution,
            config,
        )
        .unwrap();
        let step = 1.0e-6;

        let amplitude_plus = SinusoidalVelocityKick {
            amplitude: kick.amplitude + step,
            ..kick
        };
        let amplitude_minus = SinusoidalVelocityKick {
            amplitude: kick.amplitude - step,
            ..kick
        };
        let plus = remap_distribution_after_velocity_kick_weighted(
            &amplitude_plus,
            &positions,
            &weights,
            &edges,
            &distribution,
        )
        .unwrap();
        let minus = remap_distribution_after_velocity_kick_weighted(
            &amplitude_minus,
            &positions,
            &weights,
            &edges,
            &distribution,
        )
        .unwrap();
        for cell in 0..16 {
            let finite_difference =
                (plus.distribution[0][cell] - minus.distribution[0][cell]) / (2.0 * step);
            assert!(
                (finite_difference - analytic.rows[0].distribution_wrt_amplitude[cell]).abs()
                    < 1.0e-7
            );
        }
        let boundary_fd =
            (plus.total_boundary_outflow - minus.total_boundary_outflow) / (2.0 * step);
        assert!(
            (boundary_fd - analytic.total_boundary_outflow_d_amplitude).abs() < 1.0e-7
        );

        let phase_plus = SinusoidalVelocityKick {
            phase_rad: kick.phase_rad + step,
            ..kick
        };
        let phase_minus = SinusoidalVelocityKick {
            phase_rad: kick.phase_rad - step,
            ..kick
        };
        let plus = remap_distribution_after_velocity_kick_weighted(
            &phase_plus,
            &positions,
            &weights,
            &edges,
            &distribution,
        )
        .unwrap();
        let minus = remap_distribution_after_velocity_kick_weighted(
            &phase_minus,
            &positions,
            &weights,
            &edges,
            &distribution,
        )
        .unwrap();
        for cell in 0..16 {
            let finite_difference =
                (plus.distribution[0][cell] - minus.distribution[0][cell]) / (2.0 * step);
            assert!(
                (finite_difference - analytic.rows[0].distribution_wrt_phase_rad[cell]).abs()
                    < 1.0e-7
            );
        }
        let boundary_fd =
            (plus.total_boundary_outflow - minus.total_boundary_outflow) / (2.0 * step);
        assert!(
            (boundary_fd - analytic.total_boundary_outflow_d_phase_rad).abs() < 1.0e-7
        );
        assert!(analytic.minimum_topology_margin > config.topology_tolerance);
    }

    #[test]
    fn fixed_topology_remap_jacobian_refuses_topology_boundaries_and_budget_overrun() {
        let zero_kick = kick(0.0);
        let edges = uniform_edges(-1.0, 1.0, 8);
        let distribution = vec![vec![1.0; 8]];
        assert_eq!(
            linearize_velocity_kick_remap(
                &zero_kick,
                &[0.3],
                &[1.0],
                &edges,
                &distribution,
                VelocityKickRemapJacobianConfig::new(1.0e-8, 10_000),
            ),
            Err(VelocityKickRemapError::TopologyBoundary)
        );

        let nonzero_kick = SinusoidalVelocityKick::new(
            0.0,
            2.0 * std::f64::consts::PI,
            1,
            0.1,
            0.2,
        )
        .unwrap();
        assert_eq!(
            linearize_velocity_kick_remap(
                &nonzero_kick,
                &[0.73],
                &[1.0],
                &uniform_edges(0.0, 0.8, 16),
                &[vec![1.0; 16]],
                VelocityKickRemapJacobianConfig::new(1.0e-5, 1),
            ),
            Err(VelocityKickRemapError::JacobianStorageBudgetExceeded)
        );
    }

    #[test]
    fn explicit_spatial_quadrature_weights_produce_integrated_mass_ledger() {
        let edges = uniform_edges(0.0, 1.0, 10);
        let distribution = vec![vec![1.0; 10], vec![2.0; 10]];
        let report = remap_distribution_after_velocity_kick_weighted(
            &kick(0.1),
            &[
                std::f64::consts::PI / 2.0,
                3.0 * std::f64::consts::PI / 2.0,
            ],
            &[0.25, 0.75],
            &edges,
            &distribution,
        )
        .unwrap();

        assert!((report.total_mass_before - 1.75).abs() < 1.0e-12);
        assert!((report.total_boundary_outflow - 0.175).abs() < 1.0e-12);
        assert!((report.total_mass_after - 1.575).abs() < 1.0e-12);
        assert!((report.total_mass_balance_residual).abs() < 1.0e-12);
        assert_eq!(report.spatial_mass[0].spatial_weight, 0.25);
        assert_eq!(report.spatial_mass[1].spatial_weight, 0.75);
    }

    #[test]
    fn invalid_spatial_quadrature_is_rejected() {
        let edges = uniform_edges(0.0, 1.0, 4);
        let distribution = vec![vec![1.0; 4], vec![1.0; 4]];
        assert_eq!(
            remap_distribution_after_velocity_kick_weighted(
                &kick(0.1),
                &[0.1, 0.2],
                &[1.0],
                &edges,
                &distribution,
            ),
            Err(VelocityKickRemapError::InvalidSpatialQuadrature)
        );
        assert_eq!(
            remap_distribution_after_velocity_kick_weighted(
                &kick(0.1),
                &[0.1, 0.2],
                &[1.0, 0.0],
                &edges,
                &distribution,
            ),
            Err(VelocityKickRemapError::InvalidSpatialQuadrature)
        );
    }

    #[test]
    fn mass_balance_tolerance_scales_with_tiny_distributions() {
        let edges = uniform_edges(-1.0, 1.0, 32);
        let tiny = vec![vec![1.0e-200; 32]];
        let report = remap_distribution_after_velocity_kick(
            &kick(0.1),
            &[std::f64::consts::PI / 2.0],
            &edges,
            &tiny,
        )
        .unwrap();
        assert!((report.total_mass_before - report.total_mass_after).abs() < 1.0e-212);
        assert!(report.total_boundary_outflow < 1.0e-212);
        assert!(report.total_mass_balance_residual.abs() < 1.0e-212);
    }

    #[test]
    fn invalid_grids_shapes_and_negative_distribution_are_rejected() {
        assert_eq!(
            remap_distribution_after_velocity_kick(
                &kick(0.1),
                &[0.0],
                &[0.0, 1.0, 0.5],
                &[vec![1.0, 1.0]]
            ),
            Err(VelocityKickRemapError::InvalidVelocityGrid)
        );
        assert_eq!(
            remap_distribution_after_velocity_kick(&kick(0.1), &[0.0], &[0.0, 1.0], &[vec![-1.0]]),
            Err(VelocityKickRemapError::InvalidDistribution)
        );
        assert_eq!(
            remap_distribution_after_velocity_kick(
                &kick(0.1),
                &[0.0, 1.0],
                &[0.0, 1.0],
                &[vec![1.0]]
            ),
            Err(VelocityKickRemapError::ShapeMismatch)
        );
    }
}
