// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact fixed-`w` target-planner intervention for HYPERSPACE-001 H3.
//!
//! The analytic evaluator already proves that the canonical R4 shell fixture is
//! blocked when `w` is fixed to zero. This module additionally runs the same
//! bounded seeded RRT class through exact fixed-axis reduction, so H3 is no
//! longer planner-unsupported. Planner failure remains only an observation; the
//! radial separation certificate retains infeasibility authority.

use blake3::Hasher;
use thiserror::Error;

use crate::continuous_reachability::ContinuousValidityOracle;
use crate::fixed_axis_sampling::{
    FixedAxisSamplingError, FixedAxisSamplingResult, seeded_rrt_with_fixed_axes,
};
use crate::hyperspace_benchmark::{
    FiniteWShellOracle, HyperspaceBenchmarkError, HyperspaceDimension,
    canonical_shell_problem, certify_radial_separation,
};
use crate::reachability::UnknownReason;
use crate::sampling_reachability::RrtConfig;

/// Exact receipt for the fixed-`w` R4 planner intervention.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedWInterventionReceipt {
    problem_identity: [u8; 32],
    oracle_identity: [u8; 32],
    certificate_identity: [u8; 32],
    reduction_identity: [u8; 32],
    solver_identity: [u8; 32],
    planner_reason: UnknownReason,
    iterations: u32,
    accepted_nodes: u64,
    state_queries: u64,
    segment_queries: u64,
    fixed_w: f64,
    identity: [u8; 32],
}

impl FixedWInterventionReceipt {
    /// Exact original full-dimensional H3 problem identity.
    pub fn problem_identity(&self) -> [u8; 32] {
        self.problem_identity
    }

    /// Exact original full-dimensional shell-oracle identity.
    pub fn oracle_identity(&self) -> [u8; 32] {
        self.oracle_identity
    }

    /// Independently verified analytic blocking-certificate identity.
    pub fn certificate_identity(&self) -> [u8; 32] {
        self.certificate_identity
    }

    /// Exact dimensional-reduction receipt identity.
    pub fn reduction_identity(&self) -> [u8; 32] {
        self.reduction_identity
    }

    /// Exact underlying seeded-RRT solver/config identity.
    pub fn solver_identity(&self) -> [u8; 32] {
        self.solver_identity
    }

    /// Bounded planner outcome category; this is not the infeasibility proof.
    pub fn planner_reason(&self) -> &UnknownReason {
        &self.planner_reason
    }

    /// Number of reduced-space RRT expansion attempts actually executed.
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// Number of accepted reduced-space tree nodes including the root.
    pub fn accepted_nodes(&self) -> u64 {
        self.accepted_nodes
    }

    /// State-oracle queries issued by the reduced-space RRT.
    pub fn state_queries(&self) -> u64 {
        self.state_queries
    }

    /// Segment-oracle queries issued by the reduced-space RRT.
    pub fn segment_queries(&self) -> u64 {
        self.segment_queries
    }

    /// Exact fourth-coordinate value maintained by every lifted planner state.
    pub fn fixed_w(&self) -> f64 {
        self.fixed_w
    }

    /// Exact intervention receipt identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Fail-closed H3 fixed-axis qualification errors.
#[derive(Debug, Error)]
pub enum FixedWInterventionError {
    /// Shared analytic HYPERSPACE evaluator failed.
    #[error(transparent)]
    Hyperspace(#[from] HyperspaceBenchmarkError),
    /// Fixed-axis reduction or bounded RRT failed structurally.
    #[error(transparent)]
    FixedAxis(#[from] FixedAxisSamplingError),
    /// Target planner returned a path despite an independent analytic blocking
    /// theorem for the exact same full-dimensional problem.
    #[error("false feasible result in analytically blocked fixed-w H3 intervention")]
    FalseFeasible,
    /// The dimensional reduction did not preserve the intended R4 -> R3 fixed-w map.
    #[error("unexpected H3 fixed-axis reduction: {reason}")]
    UnexpectedReduction {
        /// Exact reduction mismatch.
        reason: String,
    },
}

/// Execute HYPERSPACE-001 H3 through the target RRT with `w` exactly fixed at zero.
///
/// Success means the analytic certificate verifies and the bounded target planner
/// does **not** emit a false feasible path. The returned `Unknown` reason remains
/// a planner observation rather than the proof of blockage.
pub fn qualify_fixed_w_h3(
    config: RrtConfig,
) -> Result<FixedWInterventionReceipt, FixedWInterventionError> {
    let oracle = FiniteWShellOracle::new(
        HyperspaceDimension::R4,
        1.0,
        2.0,
        0.25,
        4.0,
        0.0,
    )?;
    let problem = canonical_shell_problem(&oracle, 3.0)?;
    let certificate = certify_radial_separation(&problem, &oracle)?;

    match seeded_rrt_with_fixed_axes(&problem, &oracle, config)? {
        FixedAxisSamplingResult::Feasible { .. } => Err(FixedWInterventionError::FalseFeasible),
        FixedAxisSamplingResult::Unknown {
            reason,
            reduction,
            solver,
            best_partial_path,
            ..
        } => {
            if reduction.full_dimension() != 4
                || reduction.free_axes() != &[0, 1, 2]
                || reduction.fixed_coordinates().len() != 1
                || reduction.fixed_coordinates()[0].axis() != 3
                || reduction.fixed_coordinates()[0].value().to_bits() != 0.0_f64.to_bits()
            {
                return Err(FixedWInterventionError::UnexpectedReduction {
                    reason: format!(
                        "expected free xyz + fixed w=0, got free={:?}, fixed={:?}",
                        reduction.free_axes(),
                        reduction.fixed_coordinates()
                    ),
                });
            }
            if best_partial_path.iter().any(|state| {
                state.len() != 4 || state[3].to_bits() != 0.0_f64.to_bits()
            }) {
                return Err(FixedWInterventionError::UnexpectedReduction {
                    reason: "lifted partial planner path did not preserve exact w=0".to_string(),
                });
            }

            let fixed_w = reduction.fixed_coordinates()[0].value();
            let iterations = solver.iterations();
            let accepted_nodes = solver.accepted_nodes();
            let state_queries = solver.state_queries();
            let segment_queries = solver.segment_queries();
            let mut hasher = Hasher::new();
            hasher.update(b"symthaea-hyperspace-001-fixed-w-h3-v1\0");
            hasher.update(&problem.identity());
            hasher.update(&oracle.profile().identity());
            hasher.update(&certificate.identity());
            hasher.update(&reduction.identity());
            hasher.update(&solver.solver_identity());
            hasher.update(&[unknown_reason_tag(&reason)]);
            hasher.update(&iterations.to_le_bytes());
            hasher.update(&accepted_nodes.to_le_bytes());
            hasher.update(&state_queries.to_le_bytes());
            hasher.update(&segment_queries.to_le_bytes());
            hasher.update(&fixed_w.to_bits().to_le_bytes());
            let identity = *hasher.finalize().as_bytes();

            Ok(FixedWInterventionReceipt {
                problem_identity: problem.identity(),
                oracle_identity: oracle.profile().identity(),
                certificate_identity: certificate.identity(),
                reduction_identity: reduction.identity(),
                solver_identity: solver.solver_identity(),
                planner_reason: reason,
                iterations,
                accepted_nodes,
                state_queries,
                segment_queries,
                fixed_w,
                identity,
            })
        }
    }
}

fn unknown_reason_tag(reason: &UnknownReason) -> u8 {
    match reason {
        UnknownReason::BudgetExceeded => 0,
        UnknownReason::NoPathFound => 1,
        UnknownReason::ValidityOracleIncomplete => 2,
        UnknownReason::NumericalFailure => 3,
        UnknownReason::ModelOutOfDomain => 4,
        UnknownReason::ConstraintProjectionFailure => 5,
        UnknownReason::UnsupportedSpace => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_runs_same_rrt_class_with_w_exactly_fixed_and_never_false_feasible() {
        let config = RrtConfig::new(11, 1_000, 0.4, 0.05, 0.6).unwrap();
        let receipt = qualify_fixed_w_h3(config).unwrap();
        assert_eq!(receipt.fixed_w().to_bits(), 0.0_f64.to_bits());
        assert_eq!(receipt.planner_reason(), &UnknownReason::BudgetExceeded);
        assert_eq!(receipt.iterations(), 1_000);
        assert!(receipt.accepted_nodes() > 0);
        assert!(receipt.state_queries() > 0);
        assert!(receipt.segment_queries() > 0);
    }
}
