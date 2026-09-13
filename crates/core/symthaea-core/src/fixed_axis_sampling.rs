// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact fixed-coordinate reduction for bounded Euclidean sampling planners.
//!
//! A coordinate whose sampling lower and upper bounds are exactly equal is a
//! zero-dimensional factor, not an invalid planning axis. This adapter removes
//! such coordinates from the proposal space, delegates all validity queries to
//! the original full-dimensional oracle after exact lifting, and independently
//! replays every feasible lifted path against the original planning problem.
//!
//! The wrapped RRT still has no infeasibility authority: bounded failure remains
//! `Unknown` even when an external theorem can separately certify blockage.

use blake3::Hasher;
use thiserror::Error;

use crate::continuous_reachability::{
    ContinuousPathReplay, ContinuousPathValidationReceipt, ContinuousReachabilityError,
    ContinuousValidityOracle, ContinuousValidityOracleProfile, EuclideanPlanningProblem,
    OracleVerdict, validate_euclidean_path,
};
use crate::reachability::UnknownReason;
use crate::sampling_reachability::{
    BoundedEuclideanSamplingOracle, RrtConfig, SamplingReachabilityError,
    SamplingReachabilityResult, SamplingSolverReceipt, seeded_rrt,
};
use crate::state_space::{EuclideanSpace, StateSpace};

/// One exact coordinate removed from the sampled proposal space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedCoordinate {
    axis: usize,
    value: f64,
}

impl FixedCoordinate {
    /// Original full-dimensional axis index.
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Exact canonical fixed coordinate value.
    pub fn value(&self) -> f64 {
        self.value
    }
}

/// Exact receipt describing one fixed-axis dimensional reduction.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedAxisReductionReceipt {
    original_problem_identity: [u8; 32],
    original_oracle_identity: [u8; 32],
    reduced_problem_identity: [u8; 32],
    reduced_oracle_identity: [u8; 32],
    full_dimension: usize,
    free_axes: Vec<usize>,
    fixed_coordinates: Vec<FixedCoordinate>,
    identity: [u8; 32],
}

impl FixedAxisReductionReceipt {
    /// Exact original full-dimensional planning problem identity.
    pub fn original_problem_identity(&self) -> [u8; 32] {
        self.original_problem_identity
    }

    /// Exact original full-dimensional validity-oracle identity.
    pub fn original_oracle_identity(&self) -> [u8; 32] {
        self.original_oracle_identity
    }

    /// Exact reduced planning-problem identity consumed by RRT.
    pub fn reduced_problem_identity(&self) -> [u8; 32] {
        self.reduced_problem_identity
    }

    /// Exact lifted reduced-oracle identity consumed by RRT.
    pub fn reduced_oracle_identity(&self) -> [u8; 32] {
        self.reduced_oracle_identity
    }

    /// Original ambient coordinate dimension.
    pub fn full_dimension(&self) -> usize {
        self.full_dimension
    }

    /// Ordered original-axis indices retained in the sampled proposal space.
    pub fn free_axes(&self) -> &[usize] {
        &self.free_axes
    }

    /// Ordered coordinates held exactly fixed during all lifted queries.
    pub fn fixed_coordinates(&self) -> &[FixedCoordinate] {
        &self.fixed_coordinates
    }

    /// Exact identity binding both planning problems and the dimensional map.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Proof-strength result of running seeded RRT through fixed-axis reduction.
#[derive(Clone, Debug, PartialEq)]
pub enum FixedAxisSamplingResult {
    /// A reduced-space candidate lifted and independently replayed as valid in
    /// the original full-dimensional problem.
    Feasible {
        /// Full-dimensional canonical path.
        path: Vec<Vec<f64>>,
        /// Independent full-dimensional replay receipt.
        validation: ContinuousPathValidationReceipt,
        /// Exact reduction receipt.
        reduction: FixedAxisReductionReceipt,
        /// Underlying reduced-space RRT execution receipt.
        solver: SamplingSolverReceipt,
    },
    /// The bounded reduced-space RRT attempt remained unresolved.
    Unknown {
        /// Proof-strength unknown category from the underlying sampler/replay.
        reason: UnknownReason,
        /// Bounded diagnostic detail.
        detail: String,
        /// Best lifted full-dimensional partial path.
        best_partial_path: Vec<Vec<f64>>,
        /// Exact reduction receipt.
        reduction: FixedAxisReductionReceipt,
        /// Underlying reduced-space RRT execution receipt.
        solver: SamplingSolverReceipt,
    },
}

/// Fail-closed fixed-axis reduction errors.
#[derive(Debug, Error)]
pub enum FixedAxisSamplingError {
    /// The proposed dimensional reduction is malformed or inconsistent with the
    /// exact original problem.
    #[error("invalid fixed-axis reduction: {reason}")]
    InvalidReduction { reason: String },
    /// Shared continuous validity/replay contract failed.
    #[error(transparent)]
    Continuous(#[from] ContinuousReachabilityError),
    /// Underlying bounded RRT failed structurally.
    #[error(transparent)]
    Sampling(#[from] SamplingReachabilityError),
    /// A reduced-space feasible candidate contradicted full-dimensional replay.
    #[error("lifted fixed-axis candidate contradicted original replay: {reason}")]
    CandidateReplayContradiction { reason: String },
}

#[derive(Clone, Debug)]
struct ReductionPlan {
    full_dimension: usize,
    free_axes: Vec<usize>,
    fixed_coordinates: Vec<FixedCoordinate>,
    reduced_min: Vec<f64>,
    reduced_max: Vec<f64>,
    identity: [u8; 32],
}

impl ReductionPlan {
    fn from_problem<O: BoundedEuclideanSamplingOracle>(
        problem: &EuclideanPlanningProblem,
        oracle: &O,
    ) -> Result<Self, FixedAxisSamplingError> {
        if problem.oracle_identity() != oracle.profile().identity() {
            return Err(FixedAxisSamplingError::InvalidReduction {
                reason: "oracle identity does not match original problem".to_string(),
            });
        }
        let full_dimension = problem.space().dimension();
        let min = oracle.sampling_min();
        let max = oracle.sampling_max();
        if min.len() != full_dimension || max.len() != full_dimension {
            return Err(FixedAxisSamplingError::InvalidReduction {
                reason: "sampling bounds do not match original problem dimension".to_string(),
            });
        }

        let mut free_axes = Vec::new();
        let mut fixed_coordinates = Vec::new();
        let mut reduced_min = Vec::new();
        let mut reduced_max = Vec::new();
        let mut identity_hasher = Hasher::new();
        identity_hasher.update(b"symthaea-fixed-axis-reduction-plan-v1\0");
        identity_hasher.update(&oracle.profile().identity());
        identity_hasher.update(&(full_dimension as u64).to_le_bytes());

        for axis in 0..full_dimension {
            let lower = canonical_f64(min[axis]);
            let upper = canonical_f64(max[axis]);
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(FixedAxisSamplingError::InvalidReduction {
                    reason: format!(
                        "axis {axis} requires finite lower <= upper, got [{lower}, {upper}]"
                    ),
                });
            }
            identity_hasher.update(&(axis as u64).to_le_bytes());
            if lower == upper {
                if canonical_f64(problem.start()[axis]) != lower
                    || canonical_f64(problem.goal()[axis]) != lower
                {
                    return Err(FixedAxisSamplingError::InvalidReduction {
                        reason: format!(
                            "fixed axis {axis}={lower} disagrees with original start/goal"
                        ),
                    });
                }
                identity_hasher.update(&[0]);
                identity_hasher.update(&lower.to_bits().to_le_bytes());
                fixed_coordinates.push(FixedCoordinate { axis, value: lower });
            } else {
                identity_hasher.update(&[1]);
                identity_hasher.update(&lower.to_bits().to_le_bytes());
                identity_hasher.update(&upper.to_bits().to_le_bytes());
                free_axes.push(axis);
                reduced_min.push(lower);
                reduced_max.push(upper);
            }
        }

        if free_axes.is_empty() {
            return Err(FixedAxisSamplingError::InvalidReduction {
                reason: "V1 fixed-axis reduction requires at least one free coordinate".to_string(),
            });
        }

        Ok(Self {
            full_dimension,
            free_axes,
            fixed_coordinates,
            reduced_min,
            reduced_max,
            identity: *identity_hasher.finalize().as_bytes(),
        })
    }

    fn project(&self, full: &[f64]) -> Result<Vec<f64>, FixedAxisSamplingError> {
        if full.len() != self.full_dimension {
            return Err(FixedAxisSamplingError::InvalidReduction {
                reason: format!(
                    "full state dimension {} does not match reduction dimension {}",
                    full.len(), self.full_dimension
                ),
            });
        }
        Ok(self
            .free_axes
            .iter()
            .map(|axis| canonical_f64(full[*axis]))
            .collect())
    }

    fn lift(&self, reduced: &[f64]) -> Result<Vec<f64>, FixedAxisSamplingError> {
        if reduced.len() != self.free_axes.len() {
            return Err(FixedAxisSamplingError::InvalidReduction {
                reason: format!(
                    "reduced state dimension {} does not match {} free axes",
                    reduced.len(),
                    self.free_axes.len()
                ),
            });
        }
        let mut full = vec![0.0; self.full_dimension];
        for fixed in &self.fixed_coordinates {
            full[fixed.axis] = fixed.value;
        }
        for (reduced_axis, full_axis) in self.free_axes.iter().enumerate() {
            let value = canonical_f64(reduced[reduced_axis]);
            if !value.is_finite() {
                return Err(FixedAxisSamplingError::InvalidReduction {
                    reason: format!("reduced coordinate {reduced_axis} is non-finite"),
                });
            }
            full[*full_axis] = value;
        }
        Ok(full)
    }
}

struct ReducedFixedAxisOracle<'a, O: BoundedEuclideanSamplingOracle> {
    inner: &'a O,
    plan: ReductionPlan,
    profile: ContinuousValidityOracleProfile,
}

impl<'a, O: BoundedEuclideanSamplingOracle> ReducedFixedAxisOracle<'a, O> {
    fn new(inner: &'a O, plan: ReductionPlan) -> Result<Self, FixedAxisSamplingError> {
        let reduced_space = EuclideanSpace::new(plan.free_axes.len());
        let mut parameters = Vec::new();
        parameters.extend_from_slice(&inner.profile().identity());
        parameters.extend_from_slice(&plan.identity);
        let profile = ContinuousValidityOracleProfile::new(
            reduced_space.profile().identity(),
            "fixed-axis-lifted-validity-oracle-v1",
            parameters,
        )?;
        Ok(Self {
            inner,
            plan,
            profile,
        })
    }
}

impl<O: BoundedEuclideanSamplingOracle> ContinuousValidityOracle
    for ReducedFixedAxisOracle<'_, O>
{
    fn profile(&self) -> &ContinuousValidityOracleProfile {
        &self.profile
    }

    fn state_verdict(
        &self,
        state: &[f64],
    ) -> Result<OracleVerdict, ContinuousReachabilityError> {
        let full = self
            .plan
            .lift(state)
            .map_err(|error| ContinuousReachabilityError::OracleEvaluation {
                reason: error.to_string(),
            })?;
        self.inner.state_verdict(&full)
    }

    fn segment_verdict(
        &self,
        from: &[f64],
        to: &[f64],
    ) -> Result<OracleVerdict, ContinuousReachabilityError> {
        let full_from = self
            .plan
            .lift(from)
            .map_err(|error| ContinuousReachabilityError::OracleEvaluation {
                reason: error.to_string(),
            })?;
        let full_to = self
            .plan
            .lift(to)
            .map_err(|error| ContinuousReachabilityError::OracleEvaluation {
                reason: error.to_string(),
            })?;
        self.inner.segment_verdict(&full_from, &full_to)
    }
}

impl<O: BoundedEuclideanSamplingOracle> BoundedEuclideanSamplingOracle
    for ReducedFixedAxisOracle<'_, O>
{
    fn sampling_min(&self) -> &[f64] {
        &self.plan.reduced_min
    }

    fn sampling_max(&self) -> &[f64] {
        &self.plan.reduced_max
    }
}

/// Run the existing seeded RRT after removing exact fixed-coordinate factors.
///
/// The reduced planner never receives a special collision oracle. Every validity
/// query is lifted to the original full-dimensional oracle. A reduced feasible
/// path must then survive a second independent replay in the original problem.
pub fn seeded_rrt_with_fixed_axes<O: BoundedEuclideanSamplingOracle>(
    problem: &EuclideanPlanningProblem,
    oracle: &O,
    config: RrtConfig,
) -> Result<FixedAxisSamplingResult, FixedAxisSamplingError> {
    let plan = ReductionPlan::from_problem(problem, oracle)?;
    let reduced_oracle = ReducedFixedAxisOracle::new(oracle, plan.clone())?;
    let reduced_start = plan.project(problem.start())?;
    let reduced_goal = plan.project(problem.goal())?;
    let reduced_problem = EuclideanPlanningProblem::new(
        plan.free_axes.len(),
        reduced_start,
        reduced_goal,
        &reduced_oracle,
    )?;

    let reduction = build_reduction_receipt(problem, oracle, &reduced_problem, &reduced_oracle);
    match seeded_rrt(&reduced_problem, &reduced_oracle, config)? {
        SamplingReachabilityResult::Feasible {
            path,
            solver,
            ..
        } => {
            let lifted_path = lift_path(&plan, &path)?;
            match validate_euclidean_path(problem, oracle, &lifted_path)? {
                ContinuousPathReplay::Valid(validation) => Ok(FixedAxisSamplingResult::Feasible {
                    path: lifted_path,
                    validation,
                    reduction,
                    solver,
                }),
                ContinuousPathReplay::Unknown { reason, .. } => {
                    Ok(FixedAxisSamplingResult::Unknown {
                        reason: UnknownReason::ValidityOracleIncomplete,
                        detail: format!(
                            "lifted reduced-space candidate became unknown on full replay: {reason}"
                        ),
                        best_partial_path: lifted_path,
                        reduction,
                        solver,
                    })
                }
                ContinuousPathReplay::Invalid { reason, .. } => {
                    Err(FixedAxisSamplingError::CandidateReplayContradiction { reason })
                }
            }
        }
        SamplingReachabilityResult::Unknown {
            reason,
            detail,
            best_partial_path,
            solver,
        } => Ok(FixedAxisSamplingResult::Unknown {
            reason,
            detail,
            best_partial_path: lift_path(&plan, &best_partial_path)?,
            reduction,
            solver,
        }),
    }
}

fn lift_path(
    plan: &ReductionPlan,
    path: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, FixedAxisSamplingError> {
    path.iter().map(|state| plan.lift(state)).collect()
}

fn build_reduction_receipt<O: BoundedEuclideanSamplingOracle>(
    original_problem: &EuclideanPlanningProblem,
    original_oracle: &O,
    reduced_problem: &EuclideanPlanningProblem,
    reduced_oracle: &ReducedFixedAxisOracle<'_, O>,
) -> FixedAxisReductionReceipt {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-fixed-axis-reduction-receipt-v1\0");
    hasher.update(&original_problem.identity());
    hasher.update(&original_oracle.profile().identity());
    hasher.update(&reduced_problem.identity());
    hasher.update(&reduced_oracle.profile().identity());
    hasher.update(&reduced_oracle.plan.identity);
    let identity = *hasher.finalize().as_bytes();
    FixedAxisReductionReceipt {
        original_problem_identity: original_problem.identity(),
        original_oracle_identity: original_oracle.profile().identity(),
        reduced_problem_identity: reduced_problem.identity(),
        reduced_oracle_identity: reduced_oracle.profile().identity(),
        full_dimension: reduced_oracle.plan.full_dimension,
        free_axes: reduced_oracle.plan.free_axes.clone(),
        fixed_coordinates: reduced_oracle.plan.fixed_coordinates.clone(),
        identity,
    }
}

fn canonical_f64(value: f64) -> f64 {
    value + 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous_reachability::ContinuousValidityOracleProfile;

    #[derive(Clone, Debug)]
    struct FixedLineOracle {
        profile: ContinuousValidityOracleProfile,
        min: Vec<f64>,
        max: Vec<f64>,
    }

    impl FixedLineOracle {
        fn new(fixed_y: f64) -> Self {
            let space = EuclideanSpace::new(2);
            let fixed_y = canonical_f64(fixed_y);
            let mut parameters = Vec::new();
            parameters.extend_from_slice(&fixed_y.to_bits().to_le_bytes());
            Self {
                profile: ContinuousValidityOracleProfile::new(
                    space.profile().identity(),
                    "fixed-line-oracle-test-v1",
                    parameters,
                )
                .unwrap(),
                min: vec![-2.0, fixed_y],
                max: vec![2.0, fixed_y],
            }
        }
    }

    impl ContinuousValidityOracle for FixedLineOracle {
        fn profile(&self) -> &ContinuousValidityOracleProfile {
            &self.profile
        }

        fn state_verdict(
            &self,
            state: &[f64],
        ) -> Result<OracleVerdict, ContinuousReachabilityError> {
            if state.len() != 2 || state.iter().any(|value| !value.is_finite()) {
                return Err(ContinuousReachabilityError::OracleEvaluation {
                    reason: "malformed fixed-line state".to_string(),
                });
            }
            if state[0] < -2.0 || state[0] > 2.0 || canonical_f64(state[1]) != self.min[1] {
                return Ok(OracleVerdict::Invalid {
                    reason: "state outside fixed-line domain".to_string(),
                });
            }
            Ok(OracleVerdict::Valid)
        }

        fn segment_verdict(
            &self,
            from: &[f64],
            to: &[f64],
        ) -> Result<OracleVerdict, ContinuousReachabilityError> {
            match (self.state_verdict(from)?, self.state_verdict(to)?) {
                (OracleVerdict::Valid, OracleVerdict::Valid) => Ok(OracleVerdict::Valid),
                (OracleVerdict::Unknown { reason }, _)
                | (_, OracleVerdict::Unknown { reason }) => Ok(OracleVerdict::Unknown { reason }),
                (OracleVerdict::Invalid { reason }, _)
                | (_, OracleVerdict::Invalid { reason }) => Ok(OracleVerdict::Invalid { reason }),
            }
        }
    }

    impl BoundedEuclideanSamplingOracle for FixedLineOracle {
        fn sampling_min(&self) -> &[f64] {
            &self.min
        }

        fn sampling_max(&self) -> &[f64] {
            &self.max
        }
    }

    #[test]
    fn exact_fixed_axis_reduces_dimension_and_replays_full_path() {
        let oracle = FixedLineOracle::new(0.0);
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![-1.0, 0.0],
            vec![1.0, 0.0],
            &oracle,
        )
        .unwrap();
        let result = seeded_rrt_with_fixed_axes(
            &problem,
            &oracle,
            RrtConfig::new(7, 100, 0.25, 0.05, 0.5).unwrap(),
        )
        .unwrap();
        let FixedAxisSamplingResult::Feasible {
            path,
            validation,
            reduction,
            ..
        } = result
        else {
            panic!("fixed-line problem should be feasible");
        };
        assert_eq!(reduction.full_dimension(), 2);
        assert_eq!(reduction.free_axes(), &[0]);
        assert_eq!(reduction.fixed_coordinates().len(), 1);
        assert_eq!(reduction.fixed_coordinates()[0].axis(), 1);
        assert_eq!(reduction.fixed_coordinates()[0].value(), 0.0);
        assert!(path.iter().all(|state| state[1] == 0.0));
        assert_eq!(validation.problem_identity(), problem.identity());
    }

    #[test]
    fn signed_zero_fixed_coordinate_has_one_reduction_semantics() {
        let minus = FixedLineOracle::new(-0.0);
        let plus = FixedLineOracle::new(0.0);
        let minus_problem = EuclideanPlanningProblem::new(
            2,
            vec![-1.0, -0.0],
            vec![1.0, -0.0],
            &minus,
        )
        .unwrap();
        let plus_problem = EuclideanPlanningProblem::new(
            2,
            vec![-1.0, 0.0],
            vec![1.0, 0.0],
            &plus,
        )
        .unwrap();
        assert_eq!(minus.profile().identity(), plus.profile().identity());
        assert_eq!(minus_problem.identity(), plus_problem.identity());
    }
}
