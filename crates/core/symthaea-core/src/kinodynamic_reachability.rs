// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Proof-strength kinodynamic reachability primitives for MANIFOLD-006.
//!
//! This first tranche deliberately starts with the analytically solvable bounded
//! one-dimensional single integrator
//!
//! `x_dot = u,  u in [u_min, u_max]`
//!
//! over an exact plant/model-time horizon. The authority boundary is strict:
//! geometric path existence is not dynamic reachability, numerical ambiguity is
//! not certified infeasibility, and no type in this module grants actuation authority.

use blake3::Hasher;
use thiserror::Error;

use crate::reachability::UnknownReason;
use crate::state_space::{EuclideanSpace, StateSpace};

/// Maximum accepted absolute replay tolerance for the analytic V1 fixture.
pub const MAX_SINGLE_INTEGRATOR_REPLAY_TOLERANCE: f64 = 1.0e-6;
/// Default absolute state replay tolerance for the analytic V1 fixture.
pub const DEFAULT_SINGLE_INTEGRATOR_REPLAY_TOLERANCE: f64 = 1.0e-12;
/// Maximum number of control segments accepted by independent single-integrator replay.
pub const MAX_SINGLE_INTEGRATOR_REPLAY_SEGMENTS: usize = 16_384;

/// Explicit dynamics semantics for a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicsSemantics {
    /// Continuous-time derivative semantics.
    ContinuousTime,
    /// Discrete transition semantics.
    DiscreteTime,
    /// Hybrid continuous/discrete semantics.
    Hybrid,
    /// Semantics are inherited from an external qualified adapter.
    Adapter,
}

/// Explicit time domain used by a reachability model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicsTimeDomain {
    /// Physical/model plant time measured in seconds.
    PlantModelSeconds,
}

/// Exact plant/model-time horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantTimeProfile {
    horizon_seconds: f64,
    identity: [u8; 32],
}

impl PlantTimeProfile {
    /// Construct a strictly positive finite plant-time horizon.
    pub fn new(horizon_seconds: f64) -> Result<Self, KinodynamicError> {
        if !horizon_seconds.is_finite() || horizon_seconds <= 0.0 {
            return Err(KinodynamicError::InvalidTimeProfile {
                reason: format!(
                    "plant/model horizon must be finite and > 0 seconds, got {horizon_seconds}"
                ),
            });
        }
        let horizon_seconds = canonical_zero(horizon_seconds);
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-plant-time-profile-v1\0");
        hasher.update(b"plant-model-seconds\0");
        hasher.update(&horizon_seconds.to_bits().to_le_bytes());
        Ok(Self {
            horizon_seconds,
            identity: *hasher.finalize().as_bytes(),
        })
    }

    /// Exact horizon in plant/model seconds.
    pub fn horizon_seconds(&self) -> f64 {
        self.horizon_seconds
    }

    /// Explicit time domain. Cognitive cadence is intentionally absent.
    pub fn domain(&self) -> DynamicsTimeDomain {
        DynamicsTimeDomain::PlantModelSeconds
    }

    /// Deterministic profile identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Closed admissible scalar-control interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlInterval1D {
    minimum: f64,
    maximum: f64,
    identity: [u8; 32],
}

impl ControlInterval1D {
    /// Construct a finite closed interval `[minimum, maximum]`.
    pub fn new(minimum: f64, maximum: f64) -> Result<Self, KinodynamicError> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err(KinodynamicError::InvalidControlSpace {
                reason: format!(
                    "control interval must be finite with minimum <= maximum, got [{minimum}, {maximum}]"
                ),
            });
        }
        let minimum = canonical_zero(minimum);
        let maximum = canonical_zero(maximum);
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-control-interval-1d-v1\0");
        hasher.update(&minimum.to_bits().to_le_bytes());
        hasher.update(&maximum.to_bits().to_le_bytes());
        Ok(Self {
            minimum,
            maximum,
            identity: *hasher.finalize().as_bytes(),
        })
    }

    /// Inclusive minimum admissible control.
    pub fn minimum(&self) -> f64 {
        self.minimum
    }

    /// Inclusive maximum admissible control.
    pub fn maximum(&self) -> f64 {
        self.maximum
    }

    /// Whether a finite control lies inside the exact closed interval.
    pub fn contains(&self, control: f64) -> bool {
        control.is_finite() && control >= self.minimum && control <= self.maximum
    }

    /// Deterministic control-space identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Explicit replay/numerical policy for a dynamics witness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicsReplayPolicy1D {
    absolute_state_tolerance: f64,
    identity: [u8; 32],
}

impl DynamicsReplayPolicy1D {
    /// Construct a bounded finite absolute replay tolerance.
    pub fn new(absolute_state_tolerance: f64) -> Result<Self, KinodynamicError> {
        if !absolute_state_tolerance.is_finite()
            || absolute_state_tolerance < 0.0
            || absolute_state_tolerance > MAX_SINGLE_INTEGRATOR_REPLAY_TOLERANCE
        {
            return Err(KinodynamicError::InvalidReplayPolicy {
                reason: format!(
                    "state tolerance must be finite and in [0, {MAX_SINGLE_INTEGRATOR_REPLAY_TOLERANCE}], got {absolute_state_tolerance}"
                ),
            });
        }
        let absolute_state_tolerance = canonical_zero(absolute_state_tolerance);
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-dynamics-replay-policy-1d-v1\0");
        hasher.update(&absolute_state_tolerance.to_bits().to_le_bytes());
        Ok(Self {
            absolute_state_tolerance,
            identity: *hasher.finalize().as_bytes(),
        })
    }

    /// Named reference policy used by analytic fixtures.
    pub fn reference_v1() -> Self {
        Self::new(DEFAULT_SINGLE_INTEGRATOR_REPLAY_TOLERANCE)
            .expect("reference replay policy is valid")
    }

    /// Absolute state residual tolerance.
    pub fn absolute_state_tolerance(&self) -> f64 {
        self.absolute_state_tolerance
    }

    /// Deterministic policy identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Exact bounded one-dimensional continuous-time single-integrator model.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundedSingleIntegrator1D {
    controls: ControlInterval1D,
    state_space_identity: [u8; 32],
    identity: [u8; 32],
}

impl BoundedSingleIntegrator1D {
    /// Construct `x_dot = u` with one exact control interval.
    pub fn new(controls: ControlInterval1D) -> Self {
        let state_space = EuclideanSpace::new(1);
        let state_space_identity = state_space.profile().identity();
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-bounded-single-integrator-1d-v1\0");
        hasher.update(&state_space_identity);
        hasher.update(&controls.identity());
        hasher.update(b"continuous-time\0plant-model-seconds\0");
        Self {
            controls,
            state_space_identity,
            identity: *hasher.finalize().as_bytes(),
        }
    }

    /// Exact scalar control space.
    pub fn controls(&self) -> ControlInterval1D {
        self.controls
    }

    /// Exact state-space profile identity (`R^1`).
    pub fn state_space_identity(&self) -> [u8; 32] {
        self.state_space_identity
    }

    /// Continuous-time derivative semantics.
    pub fn semantics(&self) -> DynamicsSemantics {
        DynamicsSemantics::ContinuousTime
    }

    /// Plant/model time domain.
    pub fn time_domain(&self) -> DynamicsTimeDomain {
        DynamicsTimeDomain::PlantModelSeconds
    }

    /// Deterministic model identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Exact point-target reachability query for the bounded single integrator.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleIntegratorQuery1D {
    model_identity: [u8; 32],
    initial_state: f64,
    target_state: f64,
    plant_time: PlantTimeProfile,
    replay_policy: DynamicsReplayPolicy1D,
    identity: [u8; 32],
}

impl SingleIntegratorQuery1D {
    /// Construct one point-target query under exact model/time/replay assumptions.
    pub fn new(
        model: &BoundedSingleIntegrator1D,
        initial_state: f64,
        target_state: f64,
        plant_time: PlantTimeProfile,
        replay_policy: DynamicsReplayPolicy1D,
    ) -> Result<Self, KinodynamicError> {
        if !initial_state.is_finite() || !target_state.is_finite() {
            return Err(KinodynamicError::InvalidQuery {
                reason: "initial and target state must be finite".to_string(),
            });
        }
        let initial_state = canonical_zero(initial_state);
        let target_state = canonical_zero(target_state);
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-single-integrator-query-1d-v1\0");
        hasher.update(&model.identity());
        hasher.update(&initial_state.to_bits().to_le_bytes());
        hasher.update(&target_state.to_bits().to_le_bytes());
        hasher.update(&plant_time.identity());
        hasher.update(&replay_policy.identity());
        Ok(Self {
            model_identity: model.identity(),
            initial_state,
            target_state,
            plant_time,
            replay_policy,
            identity: *hasher.finalize().as_bytes(),
        })
    }

    /// Exact model identity bound into this query.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }

    /// Initial scalar state.
    pub fn initial_state(&self) -> f64 {
        self.initial_state
    }

    /// Desired terminal scalar state at the exact plant horizon.
    pub fn target_state(&self) -> f64 {
        self.target_state
    }

    /// Exact plant/model-time profile.
    pub fn plant_time(&self) -> PlantTimeProfile {
        self.plant_time
    }

    /// Replay/numerical policy used for witness qualification.
    pub fn replay_policy(&self) -> DynamicsReplayPolicy1D {
        self.replay_policy
    }

    /// Deterministic query identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Closed analytic terminal reachable interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReachableInterval1D {
    lower: f64,
    upper: f64,
}

impl ReachableInterval1D {
    /// Inclusive lower terminal state.
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Inclusive upper terminal state.
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Whether a finite target is inside the closed analytic interval.
    pub fn contains(&self, target: f64) -> bool {
        target.is_finite() && target >= self.lower && target <= self.upper
    }
}

/// One piecewise-constant control segment with declared state endpoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedControlSegment1D {
    start_time: f64,
    end_time: f64,
    control: f64,
    start_state: f64,
    end_state: f64,
}

impl TimedControlSegment1D {
    /// Construct a raw segment candidate. Independent replay validates it.
    pub fn new(
        start_time: f64,
        end_time: f64,
        control: f64,
        start_state: f64,
        end_state: f64,
    ) -> Self {
        Self {
            start_time: canonical_zero(start_time),
            end_time: canonical_zero(end_time),
            control: canonical_zero(control),
            start_state: canonical_zero(start_state),
            end_state: canonical_zero(end_state),
        }
    }

    /// Segment start in plant/model seconds.
    pub fn start_time(&self) -> f64 {
        self.start_time
    }
    /// Segment end in plant/model seconds.
    pub fn end_time(&self) -> f64 {
        self.end_time
    }
    /// Piecewise-constant scalar control.
    pub fn control(&self) -> f64 {
        self.control
    }
    /// Declared state at segment start.
    pub fn start_state(&self) -> f64 {
        self.start_state
    }
    /// Declared state at segment end.
    pub fn end_state(&self) -> f64 {
        self.end_state
    }
}

/// Timed/control trajectory witness. Construction alone carries no feasibility authority.
#[derive(Clone, Debug, PartialEq)]
pub struct TimedControlledTrajectory1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    segments: Vec<TimedControlSegment1D>,
    identity: [u8; 32],
}

impl TimedControlledTrajectory1D {
    /// Construct a trajectory candidate bound to one exact model/query.
    pub fn new(
        model: &BoundedSingleIntegrator1D,
        query: &SingleIntegratorQuery1D,
        segments: Vec<TimedControlSegment1D>,
    ) -> Result<Self, KinodynamicError> {
        if query.model_identity() != model.identity() {
            return Err(KinodynamicError::IdentityMismatch {
                reason: "query/model identity mismatch while constructing trajectory".to_string(),
            });
        }
        if segments.is_empty() {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: "trajectory must contain at least one control segment".to_string(),
            });
        }
        let identity = hash_trajectory(model.identity(), query.identity(), &segments);
        Ok(Self {
            model_identity: model.identity(),
            query_identity: query.identity(),
            segments,
            identity,
        })
    }

    /// Exact model identity claimed by the witness.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }
    /// Exact query identity claimed by the witness.
    pub fn query_identity(&self) -> [u8; 32] {
        self.query_identity
    }
    /// Ordered piecewise-constant segments.
    pub fn segments(&self) -> &[TimedControlSegment1D] {
        &self.segments
    }
    /// Exact witness identity binding all timestamps, controls and declared states.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Independent replay receipt for a dynamics-valid timed/control witness.
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicsReplayReceipt1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    trajectory_identity: [u8; 32],
    validator_identity: [u8; 32],
    segment_count: usize,
    maximum_state_residual: f64,
    final_state: f64,
}

impl DynamicsReplayReceipt1D {
    /// Exact dynamics model identity.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }
    /// Exact query identity.
    pub fn query_identity(&self) -> [u8; 32] {
        self.query_identity
    }
    /// Exact trajectory identity.
    pub fn trajectory_identity(&self) -> [u8; 32] {
        self.trajectory_identity
    }
    /// Independent replay-validator identity.
    pub fn validator_identity(&self) -> [u8; 32] {
        self.validator_identity
    }
    /// Number of replayed control segments.
    pub fn segment_count(&self) -> usize {
        self.segment_count
    }
    /// Largest absolute declared-vs-recomputed state residual.
    pub fn maximum_state_residual(&self) -> f64 {
        self.maximum_state_residual
    }
    /// Globally recomputed terminal state.
    pub fn final_state(&self) -> f64 {
        self.final_state
    }
}

/// Side of the analytic reachable interval excluding a certified target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfeasibleSide1D {
    /// Target is strictly below the conservative lower certification boundary.
    Below,
    /// Target is strictly above the conservative upper certification boundary.
    Above,
}

/// Analytic certificate that a target lies outside the bounded single-integrator reachable set.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleIntegratorInfeasibilityCertificate1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    reachable_interval: ReachableInterval1D,
    certification_lower: f64,
    certification_upper: f64,
    target_state: f64,
    side: InfeasibleSide1D,
    identity: [u8; 32],
}

impl SingleIntegratorInfeasibilityCertificate1D {
    /// Raw analytic reachable interval before the numerical ambiguity guard.
    pub fn reachable_interval(&self) -> ReachableInterval1D {
        self.reachable_interval
    }
    /// Conservative lower boundary used for certification.
    pub fn certification_lower(&self) -> f64 {
        self.certification_lower
    }
    /// Conservative upper boundary used for certification.
    pub fn certification_upper(&self) -> f64 {
        self.certification_upper
    }
    /// Target excluded by the certificate.
    pub fn target_state(&self) -> f64 {
        self.target_state
    }
    /// Which side of the interval excludes the target.
    pub fn side(&self) -> InfeasibleSide1D {
        self.side
    }
    /// Exact certificate identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Independent verification receipt for an analytic infeasibility certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KinodynamicCertificateVerificationReceipt {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    certificate_identity: [u8; 32],
    verifier_identity: [u8; 32],
}

impl KinodynamicCertificateVerificationReceipt {
    /// Exact model identity.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }
    /// Exact query identity.
    pub fn query_identity(&self) -> [u8; 32] {
        self.query_identity
    }
    /// Verified certificate identity.
    pub fn certificate_identity(&self) -> [u8; 32] {
        self.certificate_identity
    }
    /// Independent verifier identity.
    pub fn verifier_identity(&self) -> [u8; 32] {
        self.verifier_identity
    }
}

/// Proof-strength result for the complete analytic single-integrator reference solver.
#[derive(Clone, Debug, PartialEq)]
pub enum SingleIntegratorReachabilityResult1D {
    /// Target has a timed/control witness that independently replays under the exact model.
    Feasible {
        /// Timed/control trajectory witness.
        trajectory: TimedControlledTrajectory1D,
        /// Independent dynamics replay receipt.
        replay: DynamicsReplayReceipt1D,
    },
    /// Target is outside the exact analytic reachable interval by more than the declared numerical guard.
    CertifiedInfeasible {
        /// Analytic separating certificate.
        certificate: SingleIntegratorInfeasibilityCertificate1D,
        /// Independent certificate-verification receipt.
        verification: KinodynamicCertificateVerificationReceipt,
    },
    /// Numerical boundary ambiguity prevents either strong claim.
    Unknown {
        /// Shared proof-strength unknown reason.
        reason: UnknownReason,
        /// Bounded diagnostic detail.
        detail: String,
    },
}

/// Fail-closed errors in the generic kinodynamic/reference substrate.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum KinodynamicError {
    /// Plant/model time profile is malformed.
    #[error("invalid plant/model time profile: {reason}")]
    InvalidTimeProfile { reason: String },
    /// Control-space profile is malformed.
    #[error("invalid control space: {reason}")]
    InvalidControlSpace { reason: String },
    /// Replay policy is malformed.
    #[error("invalid dynamics replay policy: {reason}")]
    InvalidReplayPolicy { reason: String },
    /// Reachability query is malformed.
    #[error("invalid kinodynamic query: {reason}")]
    InvalidQuery { reason: String },
    /// Timed/control trajectory is malformed or violates the declared model contract.
    #[error("invalid timed/control trajectory: {reason}")]
    InvalidTrajectory { reason: String },
    /// Exact profile/subject identities do not agree.
    #[error("kinodynamic identity mismatch: {reason}")]
    IdentityMismatch { reason: String },
    /// Arithmetic escaped the finite numerical model.
    #[error("kinodynamic numerical failure: {reason}")]
    Numerical { reason: String },
    /// Candidate infeasibility certificate failed independent verification.
    #[error("invalid kinodynamic infeasibility certificate: {reason}")]
    InvalidCertificate { reason: String },
}

/// Compute the complete analytic terminal reachable interval for `x_dot = u`.
pub fn single_integrator_reachable_interval(
    model: &BoundedSingleIntegrator1D,
    query: &SingleIntegratorQuery1D,
) -> Result<ReachableInterval1D, KinodynamicError> {
    require_model_query_identity(model, query)?;
    let horizon = query.plant_time().horizon_seconds();
    let lower = checked_affine_terminal(
        query.initial_state(),
        model.controls().minimum(),
        horizon,
    )?;
    let upper = checked_affine_terminal(
        query.initial_state(),
        model.controls().maximum(),
        horizon,
    )?;
    if lower > upper {
        return Err(KinodynamicError::Numerical {
            reason: "ordered controls produced an inverted reachable interval".to_string(),
        });
    }
    Ok(ReachableInterval1D { lower, upper })
}

/// Complete analytic reachability solver for the bounded one-dimensional single integrator.
pub fn solve_single_integrator_analytic(
    model: &BoundedSingleIntegrator1D,
    query: &SingleIntegratorQuery1D,
) -> Result<SingleIntegratorReachabilityResult1D, KinodynamicError> {
    require_model_query_identity(model, query)?;
    let interval = single_integrator_reachable_interval(model, query)?;
    let tolerance = query.replay_policy().absolute_state_tolerance();
    let certification_lower = checked_sub(interval.lower(), tolerance)?;
    let certification_upper = checked_add(interval.upper(), tolerance)?;
    let target = query.target_state();

    if target < certification_lower || target > certification_upper {
        let side = if target < certification_lower {
            InfeasibleSide1D::Below
        } else {
            InfeasibleSide1D::Above
        };
        let certificate = build_infeasibility_certificate(
            model,
            query,
            interval,
            certification_lower,
            certification_upper,
            side,
        );
        let verification = verify_single_integrator_infeasibility(model, query, &certificate)?;
        return Ok(SingleIntegratorReachabilityResult1D::CertifiedInfeasible {
            certificate,
            verification,
        });
    }

    if !interval.contains(target) {
        return Ok(SingleIntegratorReachabilityResult1D::Unknown {
            reason: UnknownReason::NumericalFailure,
            detail: format!(
                "target {target} lies inside the declared numerical certification guard but outside raw analytic interval [{}, {}]",
                interval.lower(),
                interval.upper()
            ),
        });
    }

    let horizon = query.plant_time().horizon_seconds();
    let delta = checked_sub(target, query.initial_state())?;
    let control = checked_div(delta, horizon)?;
    if !model.controls().contains(control) {
        return Ok(SingleIntegratorReachabilityResult1D::Unknown {
            reason: UnknownReason::NumericalFailure,
            detail: format!(
                "analytic target lies in raw reachable interval but reconstructed constant control {control} escaped [{}, {}] under finite arithmetic",
                model.controls().minimum(),
                model.controls().maximum()
            ),
        });
    }
    let segment = TimedControlSegment1D::new(
        0.0,
        horizon,
        control,
        query.initial_state(),
        target,
    );
    let trajectory = TimedControlledTrajectory1D::new(model, query, vec![segment])?;
    let replay = replay_single_integrator_trajectory(model, query, &trajectory)?;
    Ok(SingleIntegratorReachabilityResult1D::Feasible { trajectory, replay })
}

/// Independently replay a timed/control witness through the declared single-integrator model.
pub fn replay_single_integrator_trajectory(
    model: &BoundedSingleIntegrator1D,
    query: &SingleIntegratorQuery1D,
    trajectory: &TimedControlledTrajectory1D,
) -> Result<DynamicsReplayReceipt1D, KinodynamicError> {
    require_model_query_identity(model, query)?;
    if trajectory.model_identity() != model.identity()
        || trajectory.query_identity() != query.identity()
    {
        return Err(KinodynamicError::IdentityMismatch {
            reason: "trajectory is not bound to the supplied model/query".to_string(),
        });
    }

    let interval = single_integrator_reachable_interval(model, query)?;
    if !interval.contains(query.target_state()) {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: "query target is outside the raw analytic reachable interval; replay cannot promote it to feasibility".to_string(),
        });
    }

    let segments = trajectory.segments();
    if segments.len() > MAX_SINGLE_INTEGRATOR_REPLAY_SEGMENTS {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: format!(
                "trajectory has {} control segments, exceeding replay bound {}",
                segments.len(),
                MAX_SINGLE_INTEGRATOR_REPLAY_SEGMENTS
            ),
        });
    }
    let first = segments.first().expect("trajectory constructor forbids empty segments");
    if first.start_time().to_bits() != 0.0_f64.to_bits()
        || first.start_state().to_bits() != query.initial_state().to_bits()
    {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: "trajectory must begin exactly at plant time 0 and the exact query initial state"
                .to_string(),
        });
    }

    let tolerance = query.replay_policy().absolute_state_tolerance();
    let mut previous_end_time = 0.0_f64;
    let mut previous_declared_end_state = query.initial_state();
    let mut replay_state = query.initial_state();
    let mut maximum_residual = 0.0_f64;

    for (index, segment) in segments.iter().enumerate() {
        for (name, value) in [
            ("start_time", segment.start_time()),
            ("end_time", segment.end_time()),
            ("control", segment.control()),
            ("start_state", segment.start_state()),
            ("end_state", segment.end_state()),
        ] {
            if !value.is_finite() {
                return Err(KinodynamicError::InvalidTrajectory {
                    reason: format!("segment {index} has non-finite {name}"),
                });
            }
        }
        if segment.start_time().to_bits() != previous_end_time.to_bits() {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!("segment {index} does not begin at the exact preceding plant time"),
            });
        }
        if segment.start_state().to_bits() != previous_declared_end_state.to_bits() {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!("segment {index} does not begin at the exact preceding declared state"),
            });
        }
        let start_residual = checked_abs_diff(segment.start_state(), replay_state)?;
        if start_residual > tolerance {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!(
                    "segment {index} declared start residual {start_residual} exceeds replay tolerance {tolerance} from the globally recomputed state"
                ),
            });
        }
        if segment.end_time() <= segment.start_time() {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!("segment {index} must have strictly increasing plant time"),
            });
        }
        if !model.controls().contains(segment.control()) {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!(
                    "segment {index} control {} violates exact bounds [{}, {}]",
                    segment.control(),
                    model.controls().minimum(),
                    model.controls().maximum()
                ),
            });
        }
        let duration = checked_sub(segment.end_time(), segment.start_time())?;
        let predicted_end = checked_affine_terminal(replay_state, segment.control(), duration)?;
        let end_residual = checked_abs_diff(predicted_end, segment.end_state())?;
        if end_residual > tolerance {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!(
                    "segment {index} dynamics residual {end_residual} exceeds replay tolerance {tolerance} from the globally recomputed state"
                ),
            });
        }
        maximum_residual = maximum_residual.max(start_residual).max(end_residual);
        previous_end_time = segment.end_time();
        previous_declared_end_state = segment.end_state();
        replay_state = predicted_end;
    }

    if previous_end_time.to_bits() != query.plant_time().horizon_seconds().to_bits() {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: "trajectory final timestamp must equal the exact plant/model horizon".to_string(),
        });
    }
    let declared_terminal_residual =
        checked_abs_diff(previous_declared_end_state, query.target_state())?;
    if declared_terminal_residual > tolerance {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: format!(
                "declared trajectory terminal state residual {declared_terminal_residual} exceeds replay tolerance {tolerance}"
            ),
        });
    }
    let replay_terminal_residual = checked_abs_diff(replay_state, query.target_state())?;
    if replay_terminal_residual > tolerance {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: format!(
                "globally replayed terminal state residual {replay_terminal_residual} exceeds replay tolerance {tolerance}"
            ),
        });
    }
    maximum_residual = maximum_residual
        .max(declared_terminal_residual)
        .max(replay_terminal_residual);

    let validator_identity = dynamics_replay_validator_identity(query.replay_policy().identity());
    Ok(DynamicsReplayReceipt1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        trajectory_identity: trajectory.identity(),
        validator_identity,
        segment_count: segments.len(),
        maximum_state_residual: maximum_residual,
        final_state: replay_state,
    })
}

/// Independently verify an analytic single-integrator infeasibility certificate.
pub fn verify_single_integrator_infeasibility(
    model: &BoundedSingleIntegrator1D,
    query: &SingleIntegratorQuery1D,
    certificate: &SingleIntegratorInfeasibilityCertificate1D,
) -> Result<KinodynamicCertificateVerificationReceipt, KinodynamicError> {
    require_model_query_identity(model, query)?;
    if certificate.model_identity != model.identity()
        || certificate.query_identity != query.identity()
        || certificate.target_state.to_bits() != query.target_state().to_bits()
    {
        return Err(KinodynamicError::InvalidCertificate {
            reason: "certificate subject identities/target do not match the supplied query"
                .to_string(),
        });
    }
    let interval = single_integrator_reachable_interval(model, query)?;
    let tolerance = query.replay_policy().absolute_state_tolerance();
    let certification_lower = checked_sub(interval.lower(), tolerance)?;
    let certification_upper = checked_add(interval.upper(), tolerance)?;
    if certificate.reachable_interval != interval
        || certificate.certification_lower.to_bits() != certification_lower.to_bits()
        || certificate.certification_upper.to_bits() != certification_upper.to_bits()
    {
        return Err(KinodynamicError::InvalidCertificate {
            reason: "certificate reachable interval or numerical guard does not replay exactly"
                .to_string(),
        });
    }
    let side_holds = match certificate.side {
        InfeasibleSide1D::Below => query.target_state() < certification_lower,
        InfeasibleSide1D::Above => query.target_state() > certification_upper,
    };
    if !side_holds {
        return Err(KinodynamicError::InvalidCertificate {
            reason: "target is not separated from the reachable interval on the declared side"
                .to_string(),
        });
    }
    let expected_identity = hash_infeasibility_certificate(
        model.identity(),
        query.identity(),
        interval,
        certification_lower,
        certification_upper,
        query.target_state(),
        certificate.side,
    );
    if certificate.identity != expected_identity {
        return Err(KinodynamicError::InvalidCertificate {
            reason: "certificate identity does not match independently recomputed theorem inputs"
                .to_string(),
        });
    }
    Ok(KinodynamicCertificateVerificationReceipt {
        model_identity: model.identity(),
        query_identity: query.identity(),
        certificate_identity: certificate.identity(),
        verifier_identity: kinodynamic_certificate_verifier_identity(),
    })
}

fn require_model_query_identity(
    model: &BoundedSingleIntegrator1D,
    query: &SingleIntegratorQuery1D,
) -> Result<(), KinodynamicError> {
    if query.model_identity() != model.identity() {
        return Err(KinodynamicError::IdentityMismatch {
            reason: "query was constructed for a different dynamics model".to_string(),
        });
    }
    Ok(())
}

fn build_infeasibility_certificate(
    model: &BoundedSingleIntegrator1D,
    query: &SingleIntegratorQuery1D,
    interval: ReachableInterval1D,
    certification_lower: f64,
    certification_upper: f64,
    side: InfeasibleSide1D,
) -> SingleIntegratorInfeasibilityCertificate1D {
    let identity = hash_infeasibility_certificate(
        model.identity(),
        query.identity(),
        interval,
        certification_lower,
        certification_upper,
        query.target_state(),
        side,
    );
    SingleIntegratorInfeasibilityCertificate1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        reachable_interval: interval,
        certification_lower,
        certification_upper,
        target_state: query.target_state(),
        side,
        identity,
    }
}

fn hash_trajectory(
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    segments: &[TimedControlSegment1D],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-timed-controlled-trajectory-1d-v1\0");
    hasher.update(&model_identity);
    hasher.update(&query_identity);
    hasher.update(&(segments.len() as u64).to_le_bytes());
    for segment in segments {
        for value in [
            segment.start_time(),
            segment.end_time(),
            segment.control(),
            segment.start_state(),
            segment.end_state(),
        ] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

fn hash_infeasibility_certificate(
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    interval: ReachableInterval1D,
    certification_lower: f64,
    certification_upper: f64,
    target_state: f64,
    side: InfeasibleSide1D,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-single-integrator-infeasibility-certificate-v1\0");
    hasher.update(&model_identity);
    hasher.update(&query_identity);
    hasher.update(&interval.lower().to_bits().to_le_bytes());
    hasher.update(&interval.upper().to_bits().to_le_bytes());
    hasher.update(&certification_lower.to_bits().to_le_bytes());
    hasher.update(&certification_upper.to_bits().to_le_bytes());
    hasher.update(&target_state.to_bits().to_le_bytes());
    hasher.update(&[match side {
        InfeasibleSide1D::Below => 0,
        InfeasibleSide1D::Above => 1,
    }]);
    *hasher.finalize().as_bytes()
}

fn dynamics_replay_validator_identity(policy_identity: [u8; 32]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-single-integrator-dynamics-replay-validator-v2\0");
    hasher.update(&policy_identity);
    hasher.update(&(MAX_SINGLE_INTEGRATOR_REPLAY_SEGMENTS as u64).to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn kinodynamic_certificate_verifier_identity() -> [u8; 32] {
    *blake3::hash(b"symthaea-single-integrator-certificate-verifier-v1\0").as_bytes()
}

fn checked_affine_terminal(start: f64, control: f64, duration: f64) -> Result<f64, KinodynamicError> {
    let value = control.mul_add(duration, start);
    if !value.is_finite() {
        return Err(KinodynamicError::Numerical {
            reason: "single-integrator affine transition became non-finite".to_string(),
        });
    }
    Ok(canonical_zero(value))
}

fn checked_add(left: f64, right: f64) -> Result<f64, KinodynamicError> {
    let value = left + right;
    if !value.is_finite() {
        return Err(KinodynamicError::Numerical {
            reason: "addition became non-finite".to_string(),
        });
    }
    Ok(canonical_zero(value))
}

fn checked_sub(left: f64, right: f64) -> Result<f64, KinodynamicError> {
    let value = left - right;
    if !value.is_finite() {
        return Err(KinodynamicError::Numerical {
            reason: "subtraction became non-finite".to_string(),
        });
    }
    Ok(canonical_zero(value))
}

fn checked_div(numerator: f64, denominator: f64) -> Result<f64, KinodynamicError> {
    let value = numerator / denominator;
    if !value.is_finite() {
        return Err(KinodynamicError::Numerical {
            reason: "division became non-finite".to_string(),
        });
    }
    Ok(canonical_zero(value))
}

fn checked_abs_diff(left: f64, right: f64) -> Result<f64, KinodynamicError> {
    let value = (left - right).abs();
    if !value.is_finite() {
        return Err(KinodynamicError::Numerical {
            reason: "state residual became non-finite".to_string(),
        });
    }
    Ok(value)
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(
        target: f64,
    ) -> (
        BoundedSingleIntegrator1D,
        SingleIntegratorQuery1D,
    ) {
        let model = BoundedSingleIntegrator1D::new(ControlInterval1D::new(-2.0, 3.0).unwrap());
        let query = SingleIntegratorQuery1D::new(
            &model,
            1.0,
            target,
            PlantTimeProfile::new(2.0).unwrap(),
            DynamicsReplayPolicy1D::reference_v1(),
        )
        .unwrap();
        (model, query)
    }

    #[test]
    fn closed_form_reachable_interval_matches_reference() {
        let (model, query) = fixture(5.0);
        let interval = single_integrator_reachable_interval(&model, &query).unwrap();
        assert_eq!(interval.lower(), -3.0);
        assert_eq!(interval.upper(), 7.0);
    }

    #[test]
    fn reachable_target_gets_independently_replayed_control_witness() {
        let (model, query) = fixture(5.0);
        let result = solve_single_integrator_analytic(&model, &query).unwrap();
        match result {
            SingleIntegratorReachabilityResult1D::Feasible { trajectory, replay } => {
                assert_eq!(trajectory.segments().len(), 1);
                assert_eq!(trajectory.segments()[0].control(), 2.0);
                assert_eq!(replay.trajectory_identity(), trajectory.identity());
                assert_eq!(replay.final_state(), 5.0);
                assert!(
                    replay.maximum_state_residual()
                        <= query.replay_policy().absolute_state_tolerance()
                );
            }
            other => panic!("expected feasible result, got {other:?}"),
        }
    }

    #[test]
    fn exact_upper_boundary_is_reachable() {
        let (model, query) = fixture(7.0);
        let result = solve_single_integrator_analytic(&model, &query).unwrap();
        assert!(matches!(result, SingleIntegratorReachabilityResult1D::Feasible { .. }));
    }

    #[test]
    fn target_well_outside_horizon_gets_verified_analytic_certificate() {
        let (model, query) = fixture(8.0);
        let result = solve_single_integrator_analytic(&model, &query).unwrap();
        match result {
            SingleIntegratorReachabilityResult1D::CertifiedInfeasible {
                certificate,
                verification,
            } => {
                assert_eq!(certificate.side(), InfeasibleSide1D::Above);
                assert_eq!(certificate.reachable_interval().upper(), 7.0);
                assert_eq!(verification.certificate_identity(), certificate.identity());
                verify_single_integrator_infeasibility(&model, &query, &certificate).unwrap();
            }
            other => panic!("expected certified infeasible result, got {other:?}"),
        }
    }

    #[test]
    fn numerical_guard_prevents_boundary_roundoff_from_minting_infeasibility() {
        let (model, raw_query) = fixture(7.0);
        let target = 7.0 + raw_query.replay_policy().absolute_state_tolerance() * 0.5;
        let query = SingleIntegratorQuery1D::new(
            &model,
            1.0,
            target,
            raw_query.plant_time(),
            raw_query.replay_policy(),
        )
        .unwrap();
        let result = solve_single_integrator_analytic(&model, &query).unwrap();
        assert!(matches!(
            result,
            SingleIntegratorReachabilityResult1D::Unknown {
                reason: UnknownReason::NumericalFailure,
                ..
            }
        ));
    }

    #[test]
    fn replay_rejects_control_bound_violation() {
        let (model, query) = fixture(5.0);
        let bad = TimedControlledTrajectory1D::new(
            &model,
            &query,
            vec![TimedControlSegment1D::new(0.0, 2.0, 4.0, 1.0, 5.0)],
        )
        .unwrap();
        assert!(matches!(
            replay_single_integrator_trajectory(&model, &query, &bad),
            Err(KinodynamicError::InvalidTrajectory { .. })
        ));
    }

    #[test]
    fn replay_rejects_wrong_plant_horizon_even_if_endpoint_matches() {
        let (model, query) = fixture(5.0);
        let bad = TimedControlledTrajectory1D::new(
            &model,
            &query,
            vec![TimedControlSegment1D::new(0.0, 1.0, 4.0, 1.0, 5.0)],
        )
        .unwrap();
        assert!(matches!(
            replay_single_integrator_trajectory(&model, &query, &bad),
            Err(KinodynamicError::InvalidTrajectory { .. })
        ));
    }

    #[test]
    fn plant_time_identity_changes_with_physical_horizon_only() {
        let short = PlantTimeProfile::new(1.0).unwrap();
        let long = PlantTimeProfile::new(2.0).unwrap();
        assert_ne!(short.identity(), long.identity());
        assert_eq!(short.domain(), DynamicsTimeDomain::PlantModelSeconds);
    }

    #[test]
    fn signed_zero_does_not_split_control_or_query_identity() {
        let a = ControlInterval1D::new(-0.0, 1.0).unwrap();
        let b = ControlInterval1D::new(0.0, 1.0).unwrap();
        assert_eq!(a.identity(), b.identity());

        let model_a = BoundedSingleIntegrator1D::new(a);
        let model_b = BoundedSingleIntegrator1D::new(b);
        assert_eq!(model_a.identity(), model_b.identity());
        let time = PlantTimeProfile::new(1.0).unwrap();
        let policy = DynamicsReplayPolicy1D::reference_v1();
        let qa = SingleIntegratorQuery1D::new(&model_a, -0.0, 1.0, time, policy).unwrap();
        let qb = SingleIntegratorQuery1D::new(&model_b, 0.0, 1.0, time, policy).unwrap();
        assert_eq!(qa.identity(), qb.identity());
    }

    #[test]
    fn non_finite_query_fails_closed() {
        let model = BoundedSingleIntegrator1D::new(ControlInterval1D::new(-1.0, 1.0).unwrap());
        assert!(matches!(
            SingleIntegratorQuery1D::new(
                &model,
                0.0,
                f64::NAN,
                PlantTimeProfile::new(1.0).unwrap(),
                DynamicsReplayPolicy1D::reference_v1(),
            ),
            Err(KinodynamicError::InvalidQuery { .. })
        ));
    }
}
