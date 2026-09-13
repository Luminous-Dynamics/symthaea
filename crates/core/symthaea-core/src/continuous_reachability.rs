// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Tri-state continuous validity and independent Euclidean path replay.
//!
//! This module deliberately separates a validity oracle from a planner. `Unknown`
//! is epistemic and must never be treated as free space. A blocked straight line
//! establishes only that the straight-line attempt failed, not that no detour
//! exists.

use std::cmp::Ordering;

use blake3::Hasher;
use thiserror::Error;

use crate::state_space::{EuclideanSpace, MetricSpace, StateSpace};

/// Tri-state answer from a continuous validity oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleVerdict {
    /// The exact queried state/segment is established valid under this oracle.
    Valid,
    /// The exact queried state/segment is established invalid under this oracle.
    Invalid { reason: String },
    /// The oracle cannot establish either validity or invalidity.
    Unknown { reason: String },
}

/// Exact identity-bearing descriptor for one continuous validity oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuousValidityOracleProfile {
    state_space_identity: [u8; 32],
    kind: String,
    parameters: Vec<u8>,
    identity: [u8; 32],
}

impl ContinuousValidityOracleProfile {
    /// Construct an oracle profile bound to one exact state-space profile.
    pub fn new(
        state_space_identity: [u8; 32],
        kind: impl Into<String>,
        parameters: Vec<u8>,
    ) -> Result<Self, ContinuousReachabilityError> {
        let kind = kind.into();
        if kind.trim().is_empty() {
            return Err(ContinuousReachabilityError::InvalidOracleProfile {
                reason: "oracle kind must not be empty".to_string(),
            });
        }
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-continuous-validity-oracle-v1\0");
        hasher.update(&state_space_identity);
        update_len_prefixed(&mut hasher, kind.as_bytes());
        update_len_prefixed(&mut hasher, &parameters);
        let identity = *hasher.finalize().as_bytes();
        Ok(Self {
            state_space_identity,
            kind,
            parameters,
            identity,
        })
    }

    /// Exact state-space identity this oracle interprets.
    pub fn state_space_identity(&self) -> [u8; 32] {
        self.state_space_identity
    }

    /// Stable oracle-kind identifier.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Opaque exact parameters bound into oracle identity.
    pub fn parameters(&self) -> &[u8] {
        &self.parameters
    }

    /// Deterministic exact oracle identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

fn update_len_prefixed(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Narrow continuous state/segment validity boundary.
pub trait ContinuousValidityOracle {
    /// Exact oracle profile.
    fn profile(&self) -> &ContinuousValidityOracleProfile;

    /// Qualify one state.
    fn state_verdict(&self, state: &[f64]) -> Result<OracleVerdict, ContinuousReachabilityError>;

    /// Qualify the exact straight interpolation segment between two states.
    fn segment_verdict(
        &self,
        from: &[f64],
        to: &[f64],
    ) -> Result<OracleVerdict, ContinuousReachabilityError>;
}

/// Exact continuous Euclidean start/goal problem.
#[derive(Clone, Debug)]
pub struct EuclideanPlanningProblem {
    space: EuclideanSpace,
    start: Vec<f64>,
    goal: Vec<f64>,
    oracle_identity: [u8; 32],
    identity: [u8; 32],
}

impl EuclideanPlanningProblem {
    /// Construct a finite-coordinate Euclidean query bound to one exact oracle.
    pub fn new(
        dimension: usize,
        start: Vec<f64>,
        goal: Vec<f64>,
        oracle: &impl ContinuousValidityOracle,
    ) -> Result<Self, ContinuousReachabilityError> {
        let space = EuclideanSpace::new(dimension);
        if oracle.profile().state_space_identity() != space.profile().identity() {
            return Err(ContinuousReachabilityError::OracleSpaceMismatch);
        }
        require_space_state(&space, "start", &start)?;
        require_space_state(&space, "goal", &goal)?;
        let start = canonical_vector(start);
        let goal = canonical_vector(goal);
        let oracle_identity = oracle.profile().identity();
        let identity = hash_problem(space.profile().identity(), oracle_identity, &start, &goal);
        Ok(Self {
            space,
            start,
            goal,
            oracle_identity,
            identity,
        })
    }

    /// Exact Euclidean state space.
    pub fn space(&self) -> &EuclideanSpace {
        &self.space
    }

    /// Exact canonical start state.
    pub fn start(&self) -> &[f64] {
        &self.start
    }

    /// Exact canonical goal state.
    pub fn goal(&self) -> &[f64] {
        &self.goal
    }

    /// Exact oracle identity expected by this problem.
    pub fn oracle_identity(&self) -> [u8; 32] {
        self.oracle_identity
    }

    /// Exact problem identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Independent replay receipt for a continuous waypoint path.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousPathValidationReceipt {
    problem_identity: [u8; 32],
    oracle_identity: [u8; 32],
    validator_identity: [u8; 32],
    path_identity: [u8; 32],
    state_queries: u64,
    segment_queries: u64,
    total_cost: f64,
}

impl ContinuousPathValidationReceipt {
    /// Exact problem identity.
    pub fn problem_identity(&self) -> [u8; 32] {
        self.problem_identity
    }
    /// Exact validity-oracle identity.
    pub fn oracle_identity(&self) -> [u8; 32] {
        self.oracle_identity
    }
    /// Independent validator identity.
    pub fn validator_identity(&self) -> [u8; 32] {
        self.validator_identity
    }
    /// Identity binding exact problem, canonical path and recomputed cost.
    pub fn path_identity(&self) -> [u8; 32] {
        self.path_identity
    }
    /// Number of state queries replayed.
    pub fn state_queries(&self) -> u64 {
        self.state_queries
    }
    /// Number of segment queries replayed.
    pub fn segment_queries(&self) -> u64 {
        self.segment_queries
    }
    /// Exact recomputed Euclidean waypoint length.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }
}

/// Result of independently replaying one continuous waypoint path.
#[derive(Clone, Debug, PartialEq)]
pub enum ContinuousPathReplay {
    /// Every state and every straight interpolation segment was established valid.
    Valid(ContinuousPathValidationReceipt),
    /// At least one state/segment was established invalid.
    Invalid {
        /// Human-readable oracle reason.
        reason: String,
        /// State index if failure was a state query.
        state_index: Option<usize>,
        /// Segment start index if failure was a segment query.
        segment_index: Option<usize>,
    },
    /// At least one required oracle query could not be decided.
    Unknown {
        /// Human-readable oracle reason.
        reason: String,
        /// State index if uncertainty arose on a state query.
        state_index: Option<usize>,
        /// Segment start index if uncertainty arose on a segment query.
        segment_index: Option<usize>,
    },
}

/// Outcome of the deliberately weak straight-line baseline planner.
#[derive(Clone, Debug, PartialEq)]
pub enum StraightLineAttempt {
    /// The two-state straight path was independently replayed as valid.
    Feasible {
        /// Canonical start/goal path.
        path: Vec<Vec<f64>>,
        /// Independent replay receipt.
        validation: ContinuousPathValidationReceipt,
    },
    /// The direct segment is known invalid; a detour may still exist.
    Blocked {
        /// Exact oracle reason for direct-path failure.
        reason: String,
    },
    /// The oracle could not decide a required query.
    Unknown {
        /// Exact oracle uncertainty reason.
        reason: String,
    },
}

/// Fail-closed continuous reachability/oracle errors.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ContinuousReachabilityError {
    /// Oracle profile is malformed.
    #[error("invalid continuous validity oracle profile: {reason}")]
    InvalidOracleProfile { reason: String },
    /// Oracle and query do not bind the same state-space profile.
    #[error("continuous validity oracle state-space identity mismatch")]
    OracleSpaceMismatch,
    /// Euclidean start/goal/path state does not satisfy the exact space profile.
    #[error("invalid {role} state: {reason}")]
    InvalidState {
        role: &'static str,
        reason: String,
    },
    /// A replay path must contain at least one state.
    #[error("continuous path is empty")]
    EmptyPath,
    /// Replay path endpoints disagree with the exact planning problem.
    #[error("continuous path endpoint mismatch: {which}")]
    EndpointMismatch { which: &'static str },
    /// Oracle implementation returned malformed data/arithmetic.
    #[error("continuous validity oracle evaluation failed: {reason}")]
    OracleEvaluation { reason: String },
    /// Path metric computation became non-finite.
    #[error("continuous path cost became non-finite")]
    NonFiniteCost,
    /// Axis-aligned box specification is malformed.
    #[error("invalid axis-aligned box: {reason}")]
    InvalidBox { reason: String },
}

/// Independently replay one Euclidean waypoint path against the exact oracle.
pub fn validate_euclidean_path(
    problem: &EuclideanPlanningProblem,
    oracle: &impl ContinuousValidityOracle,
    path: &[Vec<f64>],
) -> Result<ContinuousPathReplay, ContinuousReachabilityError> {
    require_oracle(problem, oracle)?;
    if path.is_empty() {
        return Err(ContinuousReachabilityError::EmptyPath);
    }
    if canonical_slice(&path[0]) != problem.start {
        return Err(ContinuousReachabilityError::EndpointMismatch { which: "start" });
    }
    if canonical_slice(path.last().expect("non-empty path has end")) != problem.goal {
        return Err(ContinuousReachabilityError::EndpointMismatch { which: "goal" });
    }

    let mut state_queries = 0_u64;
    let mut segment_queries = 0_u64;
    let mut canonical_path = Vec::with_capacity(path.len());
    for (index, state) in path.iter().enumerate() {
        require_space_state(&problem.space, "path", state)?;
        state_queries += 1;
        match oracle.state_verdict(state)? {
            OracleVerdict::Valid => canonical_path.push(canonical_slice(state)),
            OracleVerdict::Invalid { reason } => {
                return Ok(ContinuousPathReplay::Invalid {
                    reason,
                    state_index: Some(index),
                    segment_index: None,
                });
            }
            OracleVerdict::Unknown { reason } => {
                return Ok(ContinuousPathReplay::Unknown {
                    reason,
                    state_index: Some(index),
                    segment_index: None,
                });
            }
        }
    }

    let mut total_cost = 0.0_f64;
    for (index, pair) in canonical_path.windows(2).enumerate() {
        segment_queries += 1;
        match oracle.segment_verdict(&pair[0], &pair[1])? {
            OracleVerdict::Valid => {}
            OracleVerdict::Invalid { reason } => {
                return Ok(ContinuousPathReplay::Invalid {
                    reason,
                    state_index: None,
                    segment_index: Some(index),
                });
            }
            OracleVerdict::Unknown { reason } => {
                return Ok(ContinuousPathReplay::Unknown {
                    reason,
                    state_index: None,
                    segment_index: Some(index),
                });
            }
        }
        total_cost += problem
            .space
            .distance(&pair[0], &pair[1])
            .map_err(|error| ContinuousReachabilityError::OracleEvaluation {
                reason: format!("metric replay failed: {error}"),
            })?;
        if !total_cost.is_finite() {
            return Err(ContinuousReachabilityError::NonFiniteCost);
        }
    }
    total_cost += 0.0;

    let path_identity = hash_path(problem.identity, &canonical_path, total_cost);
    Ok(ContinuousPathReplay::Valid(ContinuousPathValidationReceipt {
        problem_identity: problem.identity,
        oracle_identity: oracle.profile().identity(),
        validator_identity: continuous_path_validator_identity(),
        path_identity,
        state_queries,
        segment_queries,
        total_cost,
    }))
}

/// Attempt only the direct Euclidean start-to-goal segment.
///
/// `Blocked` is not a global infeasibility result: another route may exist.
pub fn straight_line_attempt(
    problem: &EuclideanPlanningProblem,
    oracle: &impl ContinuousValidityOracle,
) -> Result<StraightLineAttempt, ContinuousReachabilityError> {
    let path = vec![problem.start.clone(), problem.goal.clone()];
    match validate_euclidean_path(problem, oracle, &path)? {
        ContinuousPathReplay::Valid(validation) => {
            Ok(StraightLineAttempt::Feasible { path, validation })
        }
        ContinuousPathReplay::Invalid { reason, .. } => Ok(StraightLineAttempt::Blocked { reason }),
        ContinuousPathReplay::Unknown { reason, .. } => Ok(StraightLineAttempt::Unknown { reason }),
    }
}

fn require_oracle(
    problem: &EuclideanPlanningProblem,
    oracle: &impl ContinuousValidityOracle,
) -> Result<(), ContinuousReachabilityError> {
    if oracle.profile().identity() == problem.oracle_identity
        && oracle.profile().state_space_identity() == problem.space.profile().identity()
    {
        Ok(())
    } else {
        Err(ContinuousReachabilityError::OracleSpaceMismatch)
    }
}

fn require_space_state(
    space: &EuclideanSpace,
    role: &'static str,
    state: &[f64],
) -> Result<(), ContinuousReachabilityError> {
    let owned = state.to_vec();
    let validity = space.validate_state(&owned);
    if validity.is_valid() {
        Ok(())
    } else {
        Err(ContinuousReachabilityError::InvalidState {
            role,
            reason: format!("{validity:?}"),
        })
    }
}

fn canonical_vector(mut state: Vec<f64>) -> Vec<f64> {
    for value in &mut state {
        *value += 0.0;
    }
    state
}

fn canonical_slice(state: &[f64]) -> Vec<f64> {
    state.iter().map(|value| *value + 0.0).collect()
}

fn hash_problem(
    state_space_identity: [u8; 32],
    oracle_identity: [u8; 32],
    start: &[f64],
    goal: &[f64],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-euclidean-planning-problem-v1\0");
    hasher.update(&state_space_identity);
    hasher.update(&oracle_identity);
    hash_vector_into(&mut hasher, start);
    hash_vector_into(&mut hasher, goal);
    *hasher.finalize().as_bytes()
}

fn hash_path(problem_identity: [u8; 32], path: &[Vec<f64>], cost: f64) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-continuous-path-witness-v1\0");
    hasher.update(&problem_identity);
    hasher.update(&(path.len() as u64).to_le_bytes());
    for state in path {
        hash_vector_into(&mut hasher, state);
    }
    hasher.update(&cost.to_bits().to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn hash_vector_into(hasher: &mut Hasher, values: &[f64]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        let canonical = *value + 0.0;
        hasher.update(&canonical.to_bits().to_le_bytes());
    }
}

fn continuous_path_validator_identity() -> [u8; 32] {
    *blake3::hash(b"symthaea-independent-continuous-path-validator-v1").as_bytes()
}

/// Closed axis-aligned box in `R^n`.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisAlignedBox {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl AxisAlignedBox {
    /// Construct a finite closed box. Degenerate axes are permitted for obstacles
    /// and unknown regions so a zero-thickness wall remains representable.
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Result<Self, ContinuousReachabilityError> {
        if min.is_empty() || min.len() != max.len() {
            return Err(ContinuousReachabilityError::InvalidBox {
                reason: "box min/max must have equal non-zero dimension".to_string(),
            });
        }
        for index in 0..min.len() {
            if !min[index].is_finite() || !max[index].is_finite() || min[index] > max[index] {
                return Err(ContinuousReachabilityError::InvalidBox {
                    reason: format!(
                        "axis {index} requires finite min <= max, got [{}, {}]",
                        min[index], max[index]
                    ),
                });
            }
        }
        Ok(Self {
            min: canonical_vector(min),
            max: canonical_vector(max),
        })
    }

    /// Box dimension.
    pub fn dimension(&self) -> usize {
        self.min.len()
    }

    /// Inclusive lower bounds.
    pub fn min(&self) -> &[f64] {
        &self.min
    }

    /// Inclusive upper bounds.
    pub fn max(&self) -> &[f64] {
        &self.max
    }

    fn contains(&self, point: &[f64]) -> bool {
        point.len() == self.dimension()
            && point
                .iter()
                .enumerate()
                .all(|(index, value)| *value >= self.min[index] && *value <= self.max[index])
    }

    fn intersects_segment(&self, from: &[f64], to: &[f64]) -> bool {
        if from.len() != self.dimension() || to.len() != self.dimension() {
            return false;
        }
        let mut t_min = 0.0_f64;
        let mut t_max = 1.0_f64;
        for index in 0..self.dimension() {
            let delta = to[index] - from[index];
            if delta == 0.0 {
                if from[index] < self.min[index] || from[index] > self.max[index] {
                    return false;
                }
                continue;
            }
            let mut first = (self.min[index] - from[index]) / delta;
            let mut second = (self.max[index] - from[index]) / delta;
            if first > second {
                std::mem::swap(&mut first, &mut second);
            }
            t_min = t_min.max(first);
            t_max = t_max.min(second);
            if t_min > t_max {
                return false;
            }
        }
        true
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + 16 * self.dimension());
        bytes.extend_from_slice(&(self.dimension() as u64).to_le_bytes());
        for value in &self.min {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        for value in &self.max {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        bytes
    }
}

/// Synthetic analytic Euclidean validity oracle with known obstacles and optional
/// explicitly unknown regions.
///
/// This is geometry-test evidence only. It is not physical-world clearance.
#[derive(Clone, Debug)]
pub struct AnalyticBoxValidityOracle {
    space: EuclideanSpace,
    domain: AxisAlignedBox,
    obstacles: Vec<AxisAlignedBox>,
    unknown_regions: Vec<AxisAlignedBox>,
    profile: ContinuousValidityOracleProfile,
}

impl AnalyticBoxValidityOracle {
    /// Construct a canonical synthetic oracle.
    pub fn new(
        domain: AxisAlignedBox,
        mut obstacles: Vec<AxisAlignedBox>,
        mut unknown_regions: Vec<AxisAlignedBox>,
    ) -> Result<Self, ContinuousReachabilityError> {
        let dimension = domain.dimension();
        if domain
            .min
            .iter()
            .zip(&domain.max)
            .any(|(min, max)| min >= max)
        {
            return Err(ContinuousReachabilityError::InvalidBox {
                reason: "planning domain must have strict positive width on every axis".to_string(),
            });
        }
        for (kind, boxes) in [
            ("obstacle", obstacles.as_slice()),
            ("unknown region", unknown_regions.as_slice()),
        ] {
            for box_ in boxes {
                if box_.dimension() != dimension {
                    return Err(ContinuousReachabilityError::InvalidBox {
                        reason: format!("{kind} dimension does not match planning domain"),
                    });
                }
            }
        }
        canonicalize_boxes(&mut obstacles);
        canonicalize_boxes(&mut unknown_regions);
        let space = EuclideanSpace::new(dimension);
        let mut parameters = domain.canonical_bytes();
        parameters.extend_from_slice(&(obstacles.len() as u64).to_le_bytes());
        for obstacle in &obstacles {
            update_len_prefixed_bytes(&mut parameters, &obstacle.canonical_bytes());
        }
        parameters.extend_from_slice(&(unknown_regions.len() as u64).to_le_bytes());
        for region in &unknown_regions {
            update_len_prefixed_bytes(&mut parameters, &region.canonical_bytes());
        }
        let profile = ContinuousValidityOracleProfile::new(
            space.profile().identity(),
            "analytic-axis-aligned-box-oracle-v1",
            parameters,
        )?;
        Ok(Self {
            space,
            domain,
            obstacles,
            unknown_regions,
            profile,
        })
    }

    /// Exact Euclidean dimension.
    pub fn dimension(&self) -> usize {
        self.space.dimension()
    }

    /// Exact closed planning domain.
    pub fn domain(&self) -> &AxisAlignedBox {
        &self.domain
    }

    /// Canonically ordered known-invalid obstacle boxes.
    pub fn obstacles(&self) -> &[AxisAlignedBox] {
        &self.obstacles
    }

    /// Canonically ordered explicitly undecidable boxes.
    pub fn unknown_regions(&self) -> &[AxisAlignedBox] {
        &self.unknown_regions
    }

    fn structural_state_check(&self, state: &[f64]) -> Result<(), ContinuousReachabilityError> {
        require_space_state(&self.space, "oracle", state)
    }
}

impl ContinuousValidityOracle for AnalyticBoxValidityOracle {
    fn profile(&self) -> &ContinuousValidityOracleProfile {
        &self.profile
    }

    fn state_verdict(&self, state: &[f64]) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.structural_state_check(state)?;
        if !self.domain.contains(state) {
            return Ok(OracleVerdict::Invalid {
                reason: "state lies outside exact synthetic planning domain".to_string(),
            });
        }
        if self.obstacles.iter().any(|obstacle| obstacle.contains(state)) {
            return Ok(OracleVerdict::Invalid {
                reason: "state lies inside a known synthetic obstacle".to_string(),
            });
        }
        if self
            .unknown_regions
            .iter()
            .any(|region| region.contains(state))
        {
            return Ok(OracleVerdict::Unknown {
                reason: "state lies inside explicitly unknown synthetic region".to_string(),
            });
        }
        Ok(OracleVerdict::Valid)
    }

    fn segment_verdict(
        &self,
        from: &[f64],
        to: &[f64],
    ) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.structural_state_check(from)?;
        self.structural_state_check(to)?;
        for (role, state) in [("from", from), ("to", to)] {
            match self.state_verdict(state)? {
                OracleVerdict::Valid => {}
                OracleVerdict::Invalid { reason } => {
                    return Ok(OracleVerdict::Invalid {
                        reason: format!("{role} endpoint invalid: {reason}"),
                    });
                }
                OracleVerdict::Unknown { reason } => {
                    return Ok(OracleVerdict::Unknown {
                        reason: format!("{role} endpoint unknown: {reason}"),
                    });
                }
            }
        }

        // The domain is a convex AABB, so two in-domain endpoints imply the
        // straight segment remains in-domain. Known invalidity is decisive even
        // if another part of the same segment enters an unknown region.
        if self
            .obstacles
            .iter()
            .any(|obstacle| obstacle.intersects_segment(from, to))
        {
            return Ok(OracleVerdict::Invalid {
                reason: "straight segment intersects a known synthetic obstacle".to_string(),
            });
        }
        if self
            .unknown_regions
            .iter()
            .any(|region| region.intersects_segment(from, to))
        {
            return Ok(OracleVerdict::Unknown {
                reason: "straight segment intersects explicitly unknown synthetic region"
                    .to_string(),
            });
        }
        Ok(OracleVerdict::Valid)
    }
}

fn canonicalize_boxes(boxes: &mut Vec<AxisAlignedBox>) {
    boxes.sort_by(compare_boxes);
    boxes.dedup();
}

fn compare_boxes(left: &AxisAlignedBox, right: &AxisAlignedBox) -> Ordering {
    for (a, b) in left.min.iter().zip(&right.min) {
        let ordering = a.total_cmp(b);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    for (a, b) in left.max.iter().zip(&right.max) {
        let ordering = a.total_cmp(b);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

fn update_len_prefixed_bytes(target: &mut Vec<u8>, bytes: &[u8]) {
    target.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    target.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box2(min: [f64; 2], max: [f64; 2]) -> AxisAlignedBox {
        AxisAlignedBox::new(min.to_vec(), max.to_vec()).unwrap()
    }

    fn open_oracle() -> AnalyticBoxValidityOracle {
        AnalyticBoxValidityOracle::new(box2([0.0, 0.0], [10.0, 10.0]), vec![], vec![]).unwrap()
    }

    #[test]
    fn open_world_straight_line_is_independently_validated() {
        let oracle = open_oracle();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![1.0, 1.0],
            vec![9.0, 9.0],
            &oracle,
        )
        .unwrap();
        let attempt = straight_line_attempt(&problem, &oracle).unwrap();
        let StraightLineAttempt::Feasible { path, validation } = attempt else {
            panic!("open convex world should validate straight line");
        };
        assert_eq!(path, vec![vec![1.0, 1.0], vec![9.0, 9.0]]);
        assert_eq!(validation.state_queries(), 2);
        assert_eq!(validation.segment_queries(), 1);
        assert!((validation.total_cost() - 128.0_f64.sqrt()).abs() <= 1e-12);
    }

    #[test]
    fn known_blocked_straight_line_is_not_global_infeasibility() {
        let oracle = AnalyticBoxValidityOracle::new(
            box2([0.0, 0.0], [10.0, 10.0]),
            vec![box2([4.0, 4.0], [6.0, 6.0])],
            vec![],
        )
        .unwrap();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![1.0, 1.0],
            vec![9.0, 9.0],
            &oracle,
        )
        .unwrap();
        assert!(matches!(
            straight_line_attempt(&problem, &oracle).unwrap(),
            StraightLineAttempt::Blocked { .. }
        ));
    }

    #[test]
    fn unknown_segment_is_not_treated_as_free_space() {
        let oracle = AnalyticBoxValidityOracle::new(
            box2([0.0, 0.0], [10.0, 10.0]),
            vec![],
            vec![box2([4.0, 4.0], [6.0, 6.0])],
        )
        .unwrap();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![1.0, 1.0],
            vec![9.0, 9.0],
            &oracle,
        )
        .unwrap();
        assert!(matches!(
            straight_line_attempt(&problem, &oracle).unwrap(),
            StraightLineAttempt::Unknown { .. }
        ));
    }

    #[test]
    fn invalid_middle_segment_is_rejected_by_replay() {
        let oracle = AnalyticBoxValidityOracle::new(
            box2([0.0, 0.0], [10.0, 10.0]),
            vec![box2([4.0, 4.0], [6.0, 6.0])],
            vec![],
        )
        .unwrap();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![1.0, 1.0],
            vec![9.0, 1.0],
            &oracle,
        )
        .unwrap();
        let path = vec![
            vec![1.0, 1.0],
            vec![3.0, 7.0],
            vec![7.0, 3.0],
            vec![9.0, 1.0],
        ];
        assert!(matches!(
            validate_euclidean_path(&problem, &oracle, &path).unwrap(),
            ContinuousPathReplay::Invalid {
                segment_index: Some(1),
                ..
            }
        ));
    }

    #[test]
    fn path_with_unknown_middle_segment_is_unknown() {
        let oracle = AnalyticBoxValidityOracle::new(
            box2([0.0, 0.0], [10.0, 10.0]),
            vec![],
            vec![box2([4.0, 4.0], [6.0, 6.0])],
        )
        .unwrap();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![1.0, 1.0],
            vec![9.0, 1.0],
            &oracle,
        )
        .unwrap();
        let path = vec![
            vec![1.0, 1.0],
            vec![3.0, 7.0],
            vec![7.0, 3.0],
            vec![9.0, 1.0],
        ];
        assert!(matches!(
            validate_euclidean_path(&problem, &oracle, &path).unwrap(),
            ContinuousPathReplay::Unknown {
                segment_index: Some(1),
                ..
            }
        ));
    }

    #[test]
    fn obstacle_order_does_not_change_oracle_identity() {
        let domain = box2([0.0, 0.0], [10.0, 10.0]);
        let a = box2([2.0, 2.0], [3.0, 3.0]);
        let b = box2([7.0, 7.0], [8.0, 8.0]);
        let first = AnalyticBoxValidityOracle::new(
            domain.clone(),
            vec![a.clone(), b.clone()],
            vec![],
        )
        .unwrap();
        let second = AnalyticBoxValidityOracle::new(domain, vec![b, a], vec![]).unwrap();
        assert_eq!(first.profile().identity(), second.profile().identity());
    }

    #[test]
    fn touching_closed_obstacle_counts_as_invalid() {
        let oracle = AnalyticBoxValidityOracle::new(
            box2([0.0, 0.0], [10.0, 10.0]),
            vec![box2([4.0, 4.0], [6.0, 6.0])],
            vec![],
        )
        .unwrap();
        assert!(matches!(
            oracle.segment_verdict(&[1.0, 4.0], &[9.0, 4.0]).unwrap(),
            OracleVerdict::Invalid { .. }
        ));
    }

    #[test]
    fn signed_zero_does_not_split_problem_identity() {
        let oracle = open_oracle();
        let a = EuclideanPlanningProblem::new(
            2,
            vec![0.0, 1.0],
            vec![9.0, 9.0],
            &oracle,
        )
        .unwrap();
        let b = EuclideanPlanningProblem::new(
            2,
            vec![-0.0, 1.0],
            vec![9.0, 9.0],
            &oracle,
        )
        .unwrap();
        assert_eq!(a.identity(), b.identity());
    }
}