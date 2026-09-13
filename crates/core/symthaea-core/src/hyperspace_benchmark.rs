// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Analytic evaluator for the synthetic HYPERSPACE-001 R3/R4 shell benchmark.
//!
//! The benchmark separates oracle truth from planner behavior. A radial-separation
//! certificate may establish that the exact bounded R3 or R4 fixture is blocked,
//! while a positive R4 result must carry an independently replayed full-dimensional
//! path. Planner failure is never used as an impossibility theorem.

use blake3::Hasher;
use thiserror::Error;

use crate::continuous_reachability::{
    ContinuousPathReplay, ContinuousReachabilityError, ContinuousValidityOracle,
    ContinuousValidityOracleProfile, EuclideanPlanningProblem, OracleVerdict,
    validate_euclidean_path,
};
use crate::sampling_reachability::BoundedEuclideanSamplingOracle;
use crate::state_space::{EuclideanSpace, StateSpace};

/// Supported ambient dimensions for the canonical benchmark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyperspaceDimension {
    /// Three spatial coordinates `(x, y, z)`.
    R3,
    /// Four spatial coordinates `(x, y, z, w)`.
    R4,
}

impl HyperspaceDimension {
    /// Coordinate dimension.
    pub fn dimension(self) -> usize {
        match self {
            Self::R3 => 3,
            Self::R4 => 4,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::R3 => 3,
            Self::R4 => 4,
        }
    }
}

/// Exact synthetic shell validity oracle.
///
/// The obstacle is
/// `R_inner <= sqrt(x^2+y^2+z^2) <= R_outer` and, in R4,
/// additionally `|w| <= W_obstacle`. The R4 planning domain independently bounds
/// allowed motion to `|w| <= w_motion_bound`.
#[derive(Clone, Debug)]
pub struct FiniteWShellOracle {
    dimension: HyperspaceDimension,
    inner_radius: f64,
    outer_radius: f64,
    w_obstacle_half_thickness: f64,
    xyz_bound: f64,
    w_motion_bound: f64,
    sampling_min: Vec<f64>,
    sampling_max: Vec<f64>,
    profile: ContinuousValidityOracleProfile,
}

impl FiniteWShellOracle {
    /// Construct an exact canonical shell oracle.
    pub fn new(
        dimension: HyperspaceDimension,
        inner_radius: f64,
        outer_radius: f64,
        w_obstacle_half_thickness: f64,
        xyz_bound: f64,
        w_motion_bound: f64,
    ) -> Result<Self, HyperspaceBenchmarkError> {
        if !inner_radius.is_finite() || inner_radius <= 0.0 {
            return Err(HyperspaceBenchmarkError::InvalidFixture {
                reason: format!("inner radius must be finite and > 0, got {inner_radius}"),
            });
        }
        if !outer_radius.is_finite() || outer_radius <= inner_radius {
            return Err(HyperspaceBenchmarkError::InvalidFixture {
                reason: format!(
                    "outer radius must be finite and > inner radius, got {outer_radius}"
                ),
            });
        }
        if !w_obstacle_half_thickness.is_finite() || w_obstacle_half_thickness < 0.0 {
            return Err(HyperspaceBenchmarkError::InvalidFixture {
                reason: format!(
                    "w obstacle half-thickness must be finite and >= 0, got {w_obstacle_half_thickness}"
                ),
            });
        }
        if !xyz_bound.is_finite() || xyz_bound <= outer_radius {
            return Err(HyperspaceBenchmarkError::InvalidFixture {
                reason: format!("xyz bound must be finite and > outer radius, got {xyz_bound}"),
            });
        }
        if !w_motion_bound.is_finite() || w_motion_bound < 0.0 {
            return Err(HyperspaceBenchmarkError::InvalidFixture {
                reason: format!("w motion bound must be finite and >= 0, got {w_motion_bound}"),
            });
        }
        let radial_square_bound = 12.0 * xyz_bound * xyz_bound;
        if !radial_square_bound.is_finite() {
            return Err(HyperspaceBenchmarkError::InvalidFixture {
                reason: "xyz bound is too large for finite analytic radial arithmetic".to_string(),
            });
        }

        // R3 has no fourth coordinate. Canonicalize fourth-coordinate-only
        // parameters so irrelevant caller representation cannot split identity.
        let effective_w_obstacle = if dimension == HyperspaceDimension::R4 {
            canonical_f64(w_obstacle_half_thickness)
        } else {
            0.0
        };
        let effective_w_motion = if dimension == HyperspaceDimension::R4 {
            canonical_f64(w_motion_bound)
        } else {
            0.0
        };

        let state_space = EuclideanSpace::new(dimension.dimension());
        let mut parameters = Vec::new();
        parameters.push(dimension.tag());
        for value in [
            inner_radius,
            outer_radius,
            effective_w_obstacle,
            xyz_bound,
            effective_w_motion,
        ] {
            parameters.extend_from_slice(&canonical_f64(value).to_bits().to_le_bytes());
        }
        let profile = ContinuousValidityOracleProfile::new(
            state_space.profile().identity(),
            "hyperspace-finite-w-shell-v1",
            parameters,
        )?;

        let mut sampling_min = vec![-xyz_bound; dimension.dimension()];
        let mut sampling_max = vec![xyz_bound; dimension.dimension()];
        if dimension == HyperspaceDimension::R4 {
            sampling_min[3] = -effective_w_motion;
            sampling_max[3] = effective_w_motion;
        }

        Ok(Self {
            dimension,
            inner_radius: canonical_f64(inner_radius),
            outer_radius: canonical_f64(outer_radius),
            w_obstacle_half_thickness: effective_w_obstacle,
            xyz_bound: canonical_f64(xyz_bound),
            w_motion_bound: effective_w_motion,
            sampling_min,
            sampling_max,
            profile,
        })
    }

    /// Ambient benchmark dimension.
    pub fn benchmark_dimension(&self) -> HyperspaceDimension {
        self.dimension
    }

    /// Inner shell radius.
    pub fn inner_radius(&self) -> f64 {
        self.inner_radius
    }

    /// Outer shell radius.
    pub fn outer_radius(&self) -> f64 {
        self.outer_radius
    }

    /// Half-thickness of obstacle support in the fourth coordinate.
    ///
    /// Returns zero in R3 because no fourth-coordinate support exists there.
    pub fn w_obstacle_half_thickness(&self) -> f64 {
        self.w_obstacle_half_thickness
    }

    /// Symmetric xyz domain bound.
    pub fn xyz_bound(&self) -> f64 {
        self.xyz_bound
    }

    /// Symmetric allowed `w` motion bound for R4; zero in R3.
    pub fn w_motion_bound(&self) -> f64 {
        self.w_motion_bound
    }

    fn require_state(&self, state: &[f64]) -> Result<(), ContinuousReachabilityError> {
        if state.len() != self.dimension.dimension() {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: format!(
                    "expected {} coordinates, got {}",
                    self.dimension.dimension(),
                    state.len()
                ),
            });
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "non-finite hyperspace coordinate".to_string(),
            });
        }
        Ok(())
    }

    fn state_inside_domain(&self, state: &[f64]) -> bool {
        state[0..3]
            .iter()
            .all(|coordinate| coordinate.abs() <= self.xyz_bound)
            && (self.dimension == HyperspaceDimension::R3
                || state[3].abs() <= self.w_motion_bound)
    }

    fn state_in_obstacle(&self, state: &[f64]) -> bool {
        let radius = state[0].hypot(state[1]).hypot(state[2]);
        let radial_support = radius >= self.inner_radius && radius <= self.outer_radius;
        let w_support = self.dimension == HyperspaceDimension::R3
            || state[3].abs() <= self.w_obstacle_half_thickness;
        radial_support && w_support
    }

    fn w_support_interval(
        &self,
        from: &[f64],
        to: &[f64],
    ) -> Result<Option<(f64, f64)>, ContinuousReachabilityError> {
        if self.dimension == HyperspaceDimension::R3 {
            return Ok(Some((0.0, 1.0)));
        }
        let w0 = from[3];
        let dw = to[3] - w0;
        if dw == 0.0 {
            return Ok((w0.abs() <= self.w_obstacle_half_thickness).then_some((0.0, 1.0)));
        }
        let a = (-self.w_obstacle_half_thickness - w0) / dw;
        let b = (self.w_obstacle_half_thickness - w0) / dw;
        if !a.is_finite() || !b.is_finite() {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "non-finite w-support interval".to_string(),
            });
        }
        let lower = a.min(b).max(0.0);
        let upper = a.max(b).min(1.0);
        Ok((lower <= upper).then_some((lower, upper)))
    }

    fn radial_square_range(
        &self,
        from: &[f64],
        to: &[f64],
        interval: (f64, f64),
    ) -> Result<(f64, f64), ContinuousReachabilityError> {
        let mut quadratic = 0.0_f64;
        let mut linear = 0.0_f64;
        let mut constant = 0.0_f64;
        for axis in 0..3 {
            let delta = to[axis] - from[axis];
            quadratic += delta * delta;
            linear += 2.0 * from[axis] * delta;
            constant += from[axis] * from[axis];
        }
        if !quadratic.is_finite() || !linear.is_finite() || !constant.is_finite() {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "non-finite radial quadratic".to_string(),
            });
        }
        let evaluate = |t: f64| quadratic * t * t + linear * t + constant;
        let (lower, upper) = interval;
        let q_lower = evaluate(lower);
        let q_upper = evaluate(upper);
        if !q_lower.is_finite() || !q_upper.is_finite() {
            return Err(ContinuousReachabilityError::OracleEvaluation {
                reason: "non-finite radial interval endpoint".to_string(),
            });
        }
        let mut minimum = q_lower.min(q_upper);
        let maximum = q_lower.max(q_upper);
        if quadratic > 0.0 {
            let vertex = -linear / (2.0 * quadratic);
            if vertex >= lower && vertex <= upper {
                let q_vertex = evaluate(vertex);
                if !q_vertex.is_finite() {
                    return Err(ContinuousReachabilityError::OracleEvaluation {
                        reason: "non-finite radial quadratic vertex".to_string(),
                    });
                }
                minimum = minimum.min(q_vertex);
            }
        }
        Ok((minimum.max(0.0), maximum.max(0.0)))
    }
}

impl ContinuousValidityOracle for FiniteWShellOracle {
    fn profile(&self) -> &ContinuousValidityOracleProfile {
        &self.profile
    }

    fn state_verdict(&self, state: &[f64]) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.require_state(state)?;
        if !self.state_inside_domain(state) {
            return Ok(OracleVerdict::Invalid {
                reason: "state lies outside the exact bounded benchmark domain".to_string(),
            });
        }
        if self.state_in_obstacle(state) {
            return Ok(OracleVerdict::Invalid {
                reason: "state lies inside the closed finite-w spherical shell".to_string(),
            });
        }
        Ok(OracleVerdict::Valid)
    }

    fn segment_verdict(
        &self,
        from: &[f64],
        to: &[f64],
    ) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.require_state(from)?;
        self.require_state(to)?;
        if !self.state_inside_domain(from) || !self.state_inside_domain(to) {
            return Ok(OracleVerdict::Invalid {
                reason: "segment endpoint lies outside the exact bounded benchmark domain"
                    .to_string(),
            });
        }
        let Some(interval) = self.w_support_interval(from, to)? else {
            return Ok(OracleVerdict::Valid);
        };
        let (minimum_radius_sq, maximum_radius_sq) =
            self.radial_square_range(from, to, interval)?;
        let inner_sq = self.inner_radius * self.inner_radius;
        let outer_sq = self.outer_radius * self.outer_radius;
        if maximum_radius_sq >= inner_sq && minimum_radius_sq <= outer_sq {
            Ok(OracleVerdict::Invalid {
                reason: "straight segment analytically intersects the closed shell obstacle"
                    .to_string(),
            })
        } else {
            Ok(OracleVerdict::Valid)
        }
    }
}

impl BoundedEuclideanSamplingOracle for FiniteWShellOracle {
    fn sampling_min(&self) -> &[f64] {
        &self.sampling_min
    }

    fn sampling_max(&self) -> &[f64] {
        &self.sampling_max
    }
}

/// Independent analytic certificate that radial separation blocks the exact problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadialSeparationCertificate {
    problem_identity: [u8; 32],
    oracle_identity: [u8; 32],
    certificate_identity: [u8; 32],
}

impl RadialSeparationCertificate {
    /// Exact planning-problem identity.
    pub fn problem_identity(&self) -> [u8; 32] {
        self.problem_identity
    }

    /// Exact shell-oracle identity.
    pub fn oracle_identity(&self) -> [u8; 32] {
        self.oracle_identity
    }

    /// Deterministic analytic-certificate identity.
    pub fn identity(&self) -> [u8; 32] {
        self.certificate_identity
    }
}

/// Receipt proving that a valid R4 path collides after projection to xyz.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionTrapReceipt {
    full_problem_identity: [u8; 32],
    full_path_identity: [u8; 32],
    projected_problem_identity: [u8; 32],
    receipt_identity: [u8; 32],
}

impl ProjectionTrapReceipt {
    /// Exact full-dimensional problem identity.
    pub fn full_problem_identity(&self) -> [u8; 32] {
        self.full_problem_identity
    }

    /// Exact independently validated R4 path identity.
    pub fn full_path_identity(&self) -> [u8; 32] {
        self.full_path_identity
    }

    /// Exact projected R3 problem identity.
    pub fn projected_problem_identity(&self) -> [u8; 32] {
        self.projected_problem_identity
    }

    /// Exact projection-trap receipt identity.
    pub fn identity(&self) -> [u8; 32] {
        self.receipt_identity
    }
}

/// Fail-closed benchmark construction/evaluation errors.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum HyperspaceBenchmarkError {
    /// Fixture parameters are malformed.
    #[error("invalid HYPERSPACE-001 fixture: {reason}")]
    InvalidFixture { reason: String },
    /// Query does not satisfy the canonical radial-separation theorem assumptions.
    #[error("radial separation theorem does not apply: {reason}")]
    SeparationNotApplicable { reason: String },
    /// A requested R4 escape witness cannot fit inside the declared domain.
    #[error("R4 escape witness unavailable: {reason}")]
    EscapeUnavailable { reason: String },
    /// Full-dimensional candidate was not independently valid.
    #[error("R4 path failed independent validation: {reason}")]
    InvalidFullPath { reason: String },
    /// Projection did not demonstrate the intended lower-dimensional collision trap.
    #[error("projection trap not established: {reason}")]
    ProjectionTrapNotEstablished { reason: String },
    /// Shared continuous reachability contract failed.
    #[error(transparent)]
    Continuous(#[from] ContinuousReachabilityError),
}

/// Construct the canonical start/goal planning problem for one exact shell oracle.
///
/// Start is the origin and goal is `(goal_x, 0, 0[, 0])`.
pub fn canonical_shell_problem(
    oracle: &FiniteWShellOracle,
    goal_x: f64,
) -> Result<EuclideanPlanningProblem, HyperspaceBenchmarkError> {
    if !goal_x.is_finite() || goal_x <= oracle.outer_radius || goal_x > oracle.xyz_bound {
        return Err(HyperspaceBenchmarkError::InvalidFixture {
            reason: format!(
                "goal_x must be finite and in (outer_radius, xyz_bound], got {goal_x}"
            ),
        });
    }
    let dimension = oracle.dimension.dimension();
    let start = vec![0.0; dimension];
    let mut goal = vec![0.0; dimension];
    goal[0] = goal_x;
    Ok(EuclideanPlanningProblem::new(
        dimension, start, goal, oracle,
    )?)
}

/// Construct and independently verify the analytic radial-separation certificate.
///
/// For R3, the closed shell spans the whole available state space. For R4 this
/// theorem applies only when the allowed `w` domain is entirely contained inside
/// the obstacle's `w` support. The proof then follows from continuity of the xyz
/// radius between a start strictly inside `R_inner` and goal strictly outside
/// `R_outer`.
pub fn certify_radial_separation(
    problem: &EuclideanPlanningProblem,
    oracle: &FiniteWShellOracle,
) -> Result<RadialSeparationCertificate, HyperspaceBenchmarkError> {
    require_problem_oracle(problem, oracle)?;
    let start_radius = xyz_radius(problem.start());
    let goal_radius = xyz_radius(problem.goal());
    if !(start_radius < oracle.inner_radius) {
        return Err(HyperspaceBenchmarkError::SeparationNotApplicable {
            reason: format!("start radius {start_radius} is not inside inner radius"),
        });
    }
    if !(goal_radius > oracle.outer_radius) {
        return Err(HyperspaceBenchmarkError::SeparationNotApplicable {
            reason: format!("goal radius {goal_radius} is not outside outer radius"),
        });
    }
    if oracle.dimension == HyperspaceDimension::R4
        && oracle.w_motion_bound > oracle.w_obstacle_half_thickness
    {
        return Err(HyperspaceBenchmarkError::SeparationNotApplicable {
            reason: "R4 domain permits motion outside the obstacle's w support".to_string(),
        });
    }
    if oracle.state_verdict(problem.start())? != OracleVerdict::Valid
        || oracle.state_verdict(problem.goal())? != OracleVerdict::Valid
    {
        return Err(HyperspaceBenchmarkError::SeparationNotApplicable {
            reason: "canonical endpoints are not valid free states".to_string(),
        });
    }

    let problem_identity = problem.identity();
    let oracle_identity = oracle.profile.identity();
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-radial-separation-certificate-v1\0");
    hasher.update(&problem_identity);
    hasher.update(&oracle_identity);
    let certificate_identity = *hasher.finalize().as_bytes();
    let certificate = RadialSeparationCertificate {
        problem_identity,
        oracle_identity,
        certificate_identity,
    };
    verify_radial_separation_certificate(problem, oracle, &certificate)?;
    Ok(certificate)
}

/// Independently verify one radial-separation certificate against exact inputs.
pub fn verify_radial_separation_certificate(
    problem: &EuclideanPlanningProblem,
    oracle: &FiniteWShellOracle,
    certificate: &RadialSeparationCertificate,
) -> Result<(), HyperspaceBenchmarkError> {
    require_problem_oracle(problem, oracle)?;
    if certificate.problem_identity != problem.identity()
        || certificate.oracle_identity != oracle.profile.identity()
    {
        return Err(HyperspaceBenchmarkError::SeparationNotApplicable {
            reason: "certificate identity binding does not match exact problem/oracle".to_string(),
        });
    }
    let start_radius = xyz_radius(problem.start());
    let goal_radius = xyz_radius(problem.goal());
    if start_radius >= oracle.inner_radius || goal_radius <= oracle.outer_radius {
        return Err(HyperspaceBenchmarkError::SeparationNotApplicable {
            reason: "endpoint radial separation assumptions do not hold".to_string(),
        });
    }
    if oracle.dimension == HyperspaceDimension::R4
        && oracle.w_motion_bound > oracle.w_obstacle_half_thickness
    {
        return Err(HyperspaceBenchmarkError::SeparationNotApplicable {
            reason: "allowed R4 w-domain exceeds obstacle support".to_string(),
        });
    }
    if oracle.state_verdict(problem.start())? != OracleVerdict::Valid
        || oracle.state_verdict(problem.goal())? != OracleVerdict::Valid
    {
        return Err(HyperspaceBenchmarkError::SeparationNotApplicable {
            reason: "certificate endpoints are not valid free states".to_string(),
        });
    }
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-radial-separation-certificate-v1\0");
    hasher.update(&problem.identity());
    hasher.update(&oracle.profile.identity());
    if *hasher.finalize().as_bytes() != certificate.certificate_identity {
        return Err(HyperspaceBenchmarkError::SeparationNotApplicable {
            reason: "certificate digest mismatch".to_string(),
        });
    }
    Ok(())
}

/// Evaluator-only constructive R4 escape witness.
///
/// This helper is for oracle/evaluator qualification and must not be supplied to
/// a target planner during a discovery run.
pub fn evaluator_reference_escape_path(
    problem: &EuclideanPlanningProblem,
    oracle: &FiniteWShellOracle,
    clearance_margin: f64,
) -> Result<Vec<Vec<f64>>, HyperspaceBenchmarkError> {
    require_problem_oracle(problem, oracle)?;
    if oracle.dimension != HyperspaceDimension::R4 {
        return Err(HyperspaceBenchmarkError::EscapeUnavailable {
            reason: "reference escape requires R4".to_string(),
        });
    }
    if !clearance_margin.is_finite() || clearance_margin <= 0.0 {
        return Err(HyperspaceBenchmarkError::EscapeUnavailable {
            reason: format!("clearance margin must be finite and > 0, got {clearance_margin}"),
        });
    }
    if problem.start()[3] != 0.0 || problem.goal()[3] != 0.0 {
        return Err(HyperspaceBenchmarkError::EscapeUnavailable {
            reason: "canonical reference witness requires start/goal on w=0".to_string(),
        });
    }
    let w_star = oracle.w_obstacle_half_thickness + clearance_margin;
    if w_star > oracle.w_motion_bound {
        return Err(HyperspaceBenchmarkError::EscapeUnavailable {
            reason: format!(
                "required w*={w_star} exceeds allowed w bound {}",
                oracle.w_motion_bound
            ),
        });
    }
    let mut lifted_start = problem.start().to_vec();
    lifted_start[3] = w_star;
    let mut lifted_goal = problem.goal().to_vec();
    lifted_goal[3] = w_star;
    Ok(vec![
        problem.start().to_vec(),
        lifted_start,
        lifted_goal,
        problem.goal().to_vec(),
    ])
}

/// Independently qualify that a full R4 path is valid while its xyz projection
/// collides with the corresponding R3 shell.
pub fn qualify_projection_trap(
    problem: &EuclideanPlanningProblem,
    oracle: &FiniteWShellOracle,
    path: &[Vec<f64>],
) -> Result<ProjectionTrapReceipt, HyperspaceBenchmarkError> {
    require_problem_oracle(problem, oracle)?;
    if oracle.dimension != HyperspaceDimension::R4 {
        return Err(HyperspaceBenchmarkError::ProjectionTrapNotEstablished {
            reason: "projection trap requires an R4 source problem".to_string(),
        });
    }
    let full_validation = match validate_euclidean_path(problem, oracle, path)? {
        ContinuousPathReplay::Valid(validation) => validation,
        ContinuousPathReplay::Invalid { reason, .. } => {
            return Err(HyperspaceBenchmarkError::InvalidFullPath { reason });
        }
        ContinuousPathReplay::Unknown { reason, .. } => {
            return Err(HyperspaceBenchmarkError::InvalidFullPath {
                reason: format!("full-dimensional replay remained unknown: {reason}"),
            });
        }
    };

    let projected_oracle = FiniteWShellOracle::new(
        HyperspaceDimension::R3,
        oracle.inner_radius,
        oracle.outer_radius,
        0.0,
        oracle.xyz_bound,
        0.0,
    )?;
    let projected_problem = EuclideanPlanningProblem::new(
        3,
        project_xyz(problem.start()),
        project_xyz(problem.goal()),
        &projected_oracle,
    )?;
    let projected_path: Vec<Vec<f64>> = path.iter().map(|state| project_xyz(state)).collect();
    match validate_euclidean_path(&projected_problem, &projected_oracle, &projected_path)? {
        ContinuousPathReplay::Invalid { .. } => {}
        ContinuousPathReplay::Valid(_) => {
            return Err(HyperspaceBenchmarkError::ProjectionTrapNotEstablished {
                reason: "xyz projection remained collision-free".to_string(),
            });
        }
        ContinuousPathReplay::Unknown { reason, .. } => {
            return Err(HyperspaceBenchmarkError::ProjectionTrapNotEstablished {
                reason: format!("projected replay was unknown: {reason}"),
            });
        }
    }

    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-hyperspace-projection-trap-v1\0");
    hasher.update(&problem.identity());
    hasher.update(&full_validation.path_identity());
    hasher.update(&projected_problem.identity());
    let receipt_identity = *hasher.finalize().as_bytes();
    Ok(ProjectionTrapReceipt {
        full_problem_identity: problem.identity(),
        full_path_identity: full_validation.path_identity(),
        projected_problem_identity: projected_problem.identity(),
        receipt_identity,
    })
}

fn require_problem_oracle(
    problem: &EuclideanPlanningProblem,
    oracle: &FiniteWShellOracle,
) -> Result<(), HyperspaceBenchmarkError> {
    if problem.oracle_identity() != oracle.profile.identity()
        || problem.space().dimension() != oracle.dimension.dimension()
    {
        return Err(HyperspaceBenchmarkError::InvalidFixture {
            reason: "planning problem is not bound to the exact shell oracle".to_string(),
        });
    }
    Ok(())
}

fn xyz_radius(state: &[f64]) -> f64 {
    state[0].hypot(state[1]).hypot(state[2])
}

fn project_xyz(state: &[f64]) -> Vec<f64> {
    vec![canonical_f64(state[0]), canonical_f64(state[1]), canonical_f64(state[2])]
}

fn canonical_f64(value: f64) -> f64 {
    value + 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle3() -> FiniteWShellOracle {
        FiniteWShellOracle::new(HyperspaceDimension::R3, 2.0, 4.0, 99.0, 8.0, 77.0)
            .unwrap()
    }

    fn oracle4(w_bound: f64) -> FiniteWShellOracle {
        FiniteWShellOracle::new(HyperspaceDimension::R4, 2.0, 4.0, 1.0, 8.0, w_bound)
            .unwrap()
    }

    #[test]
    fn r3_closed_shell_has_independent_radial_separation_certificate() {
        let oracle = oracle3();
        let problem = canonical_shell_problem(&oracle, 6.0).unwrap();
        let certificate = certify_radial_separation(&problem, &oracle).unwrap();
        verify_radial_separation_certificate(&problem, &oracle, &certificate).unwrap();
    }

    #[test]
    fn r4_insufficient_w_bound_remains_radially_blocked() {
        let oracle = oracle4(0.75);
        let problem = canonical_shell_problem(&oracle, 6.0).unwrap();
        assert!(certify_radial_separation(&problem, &oracle).is_ok());
    }

    #[test]
    fn r4_w_disabled_is_an_explicit_blocked_intervention() {
        let oracle = oracle4(0.0);
        let problem = canonical_shell_problem(&oracle, 6.0).unwrap();
        assert!(certify_radial_separation(&problem, &oracle).is_ok());
    }

    #[test]
    fn r4_sufficient_w_bound_invalidates_radial_blocking_theorem() {
        let oracle = oracle4(2.0);
        let problem = canonical_shell_problem(&oracle, 6.0).unwrap();
        assert!(matches!(
            certify_radial_separation(&problem, &oracle),
            Err(HyperspaceBenchmarkError::SeparationNotApplicable { .. })
        ));
    }

    #[test]
    fn evaluator_reference_escape_is_full_4d_valid() {
        let oracle = oracle4(2.0);
        let problem = canonical_shell_problem(&oracle, 6.0).unwrap();
        let path = evaluator_reference_escape_path(&problem, &oracle, 0.5).unwrap();
        assert!(matches!(
            validate_euclidean_path(&problem, &oracle, &path).unwrap(),
            ContinuousPathReplay::Valid(_)
        ));
    }

    #[test]
    fn projection_trap_is_established_independently() {
        let oracle = oracle4(2.0);
        let problem = canonical_shell_problem(&oracle, 6.0).unwrap();
        let path = evaluator_reference_escape_path(&problem, &oracle, 0.5).unwrap();
        let trap = qualify_projection_trap(&problem, &oracle, &path).unwrap();
        assert_eq!(trap.full_problem_identity(), problem.identity());
    }

    #[test]
    fn full_4d_middle_segment_is_valid_where_xyz_projection_crosses_shell() {
        let oracle = oracle4(2.0);
        let from = [0.0, 0.0, 0.0, 1.5];
        let to = [6.0, 0.0, 0.0, 1.5];
        assert_eq!(oracle.segment_verdict(&from, &to).unwrap(), OracleVerdict::Valid);

        let projected = oracle3();
        assert!(matches!(
            projected.segment_verdict(&from[0..3], &to[0..3]).unwrap(),
            OracleVerdict::Invalid { .. }
        ));
    }

    #[test]
    fn segment_touching_closed_shell_boundary_is_invalid() {
        let oracle = oracle4(3.0);
        let from = [0.0, 0.0, 0.0, 0.0];
        let to = [2.0, 0.0, 0.0, 1.0];
        assert!(matches!(
            oracle.segment_verdict(&from, &to).unwrap(),
            OracleVerdict::Invalid { .. }
        ));
    }

    #[test]
    fn r3_identity_ignores_nonexistent_w_parameters() {
        let a = FiniteWShellOracle::new(HyperspaceDimension::R3, 2.0, 4.0, 1.0, 8.0, 2.0)
            .unwrap();
        let b = FiniteWShellOracle::new(HyperspaceDimension::R3, 2.0, 4.0, 9.0, 8.0, 7.0)
            .unwrap();
        assert_eq!(a.profile().identity(), b.profile().identity());
        assert_eq!(a.w_obstacle_half_thickness(), 0.0);
        assert_eq!(a.w_motion_bound(), 0.0);
    }

    #[test]
    fn exact_oracle_identity_changes_with_dimensional_intervention() {
        let a = oracle4(0.75);
        let b = oracle4(2.0);
        let c = oracle3();
        assert_ne!(a.profile().identity(), b.profile().identity());
        assert_ne!(a.profile().identity(), c.profile().identity());
    }
}
