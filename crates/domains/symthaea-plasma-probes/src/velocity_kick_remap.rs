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
    #[error("distribution contains a negative or non-finite cell average")]
    InvalidDistribution,
    #[error("remap produced a non-finite cell average or mass ledger")]
    NonFiniteRemap,
    #[error("remap mass balance disagrees with analytically computed boundary outflow")]
    MassBalanceFailure,
    #[error(transparent)]
    Pulse(#[from] PulseMapError),
}

/// Mass ledger for one spatial cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialMassLedger {
    pub mass_before: f64,
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
    pub total_mass_before: f64,
    pub total_mass_after: f64,
    pub total_boundary_outflow: f64,
    pub total_mass_balance_residual: f64,
    pub relative_boundary_outflow: f64,
}

/// Apply the instantaneous characteristic pullback to a cell-average distribution.
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
    kick.validate()?;
    if spatial_positions.is_empty()
        || velocity_edges.len() < 2
        || distribution.len() != spatial_positions.len()
    {
        return Err(VelocityKickRemapError::ShapeMismatch);
    }
    if velocity_edges.iter().any(|edge| !edge.is_finite())
        || velocity_edges.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(VelocityKickRemapError::InvalidVelocityGrid);
    }
    let velocity_cells = velocity_edges.len() - 1;
    if distribution
        .iter()
        .any(|row| row.len() != velocity_cells)
    {
        return Err(VelocityKickRemapError::ShapeMismatch);
    }

    let mut remapped = Vec::with_capacity(spatial_positions.len());
    let mut spatial_mass = Vec::with_capacity(spatial_positions.len());
    let mut total_mass_before = 0.0_f64;
    let mut total_mass_after = 0.0_f64;
    let mut total_boundary_outflow = 0.0_f64;
    let mut total_mass_balance_residual = 0.0_f64;

    for (row_index, &x) in spatial_positions.iter().enumerate() {
        if !x.is_finite() {
            return Err(VelocityKickRemapError::InvalidSpatialPosition);
        }
        let source = &distribution[row_index];
        if source.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(VelocityKickRemapError::InvalidDistribution);
        }
        let shift = kick.delta_v(x)?;
        let mut output = vec![0.0; velocity_cells];
        let mut mass_before = 0.0_f64;
        let mut mass_after = 0.0_f64;

        for cell in 0..velocity_cells {
            let width = velocity_edges[cell + 1] - velocity_edges[cell];
            mass_before += source[cell] * width;
            if !mass_before.is_finite() {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }
        }

        for target_cell in 0..velocity_cells {
            let preimage_left = velocity_edges[target_cell] - shift;
            let preimage_right = velocity_edges[target_cell + 1] - shift;
            if !preimage_left.is_finite() || !preimage_right.is_finite() {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }

            // Locate the source cell containing the left endpoint. If the
            // preimage starts outside the grid, begin at the first cell.
            let partition = velocity_edges.partition_point(|edge| *edge <= preimage_left);
            let mut source_cell = partition.saturating_sub(1);
            let mut integral = 0.0_f64;
            while source_cell < velocity_cells {
                let source_left = velocity_edges[source_cell];
                let source_right = velocity_edges[source_cell + 1];
                if source_left >= preimage_right {
                    break;
                }
                let overlap_left = preimage_left.max(source_left);
                let overlap_right = preimage_right.min(source_right);
                let overlap = (overlap_right - overlap_left).max(0.0);
                integral += source[source_cell] * overlap;
                if !integral.is_finite() {
                    return Err(VelocityKickRemapError::NonFiniteRemap);
                }
                source_cell += 1;
            }
            let target_width = velocity_edges[target_cell + 1] - velocity_edges[target_cell];
            let average = integral / target_width;
            if !average.is_finite() || average < 0.0 {
                return Err(VelocityKickRemapError::NonFiniteRemap);
            }
            output[target_cell] = average;
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
            for source_cell in 0..velocity_cells {
                let overlap_left = out_left.max(velocity_edges[source_cell]);
                let overlap_right = out_right.min(velocity_edges[source_cell + 1]);
                let overlap = (overlap_right - overlap_left).max(0.0);
                boundary_outflow += source[source_cell] * overlap;
                if !boundary_outflow.is_finite() {
                    return Err(VelocityKickRemapError::NonFiniteRemap);
                }
            }
        }

        let residual = mass_before - mass_after - boundary_outflow;
        let tolerance = 256.0
            * f64::EPSILON
            * mass_before.abs().max(mass_after.abs()).max(1.0)
            * velocity_cells as f64;
        if !residual.is_finite() || residual.abs() > tolerance {
            return Err(VelocityKickRemapError::MassBalanceFailure);
        }

        total_mass_before += mass_before;
        total_mass_after += mass_after;
        total_boundary_outflow += boundary_outflow;
        total_mass_balance_residual += residual;
        if !total_mass_before.is_finite()
            || !total_mass_after.is_finite()
            || !total_boundary_outflow.is_finite()
            || !total_mass_balance_residual.is_finite()
        {
            return Err(VelocityKickRemapError::NonFiniteRemap);
        }
        spatial_mass.push(SpatialMassLedger {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn kick(amplitude: f64) -> SinusoidalVelocityKick {
        SinusoidalVelocityKick::new(0.0, 2.0 * std::f64::consts::PI, 1, amplitude, 0.0)
            .unwrap()
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
        let distribution = vec![(0..64)
            .map(|index| {
                let center = 0.5 * (edges[index] + edges[index + 1]);
                if (-1.0..=1.0).contains(&center) {
                    1.0 + 0.2 * center
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>()];
        let report =
            remap_distribution_after_velocity_kick(&kick(0.1), &[std::f64::consts::PI / 2.0], &edges, &distribution)
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
        let report = remap_distribution_after_velocity_kick(&kick(0.1), &[std::f64::consts::PI / 2.0], &edges, &distribution)
            .unwrap();
        assert!((report.total_mass_before - 1.0).abs() < 1.0e-12);
        assert!((report.total_boundary_outflow - 0.1).abs() < 1.0e-12);
        assert!((report.total_mass_after - 0.9).abs() < 1.0e-12);
        assert!(report.distribution.iter().flatten().all(|value| *value >= 0.0));
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
        assert!(report.spatial_mass.iter().all(|ledger| ledger.residual.abs() < 1.0e-12));
    }

    #[test]
    fn invalid_grids_shapes_and_negative_distribution_are_rejected() {
        assert_eq!(
            remap_distribution_after_velocity_kick(&kick(0.1), &[0.0], &[0.0, 1.0, 0.5], &[vec![1.0, 1.0]]),
            Err(VelocityKickRemapError::InvalidVelocityGrid)
        );
        assert_eq!(
            remap_distribution_after_velocity_kick(&kick(0.1), &[0.0], &[0.0, 1.0], &[vec![-1.0]]),
            Err(VelocityKickRemapError::InvalidDistribution)
        );
        assert_eq!(
            remap_distribution_after_velocity_kick(&kick(0.1), &[0.0, 1.0], &[0.0, 1.0], &[vec![1.0]]),
            Err(VelocityKickRemapError::ShapeMismatch)
        );
    }
}
