// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Analytic bounded double-integrator reachability for MANIFOLD-006B.
//!
//! This module extends the qualified MANIFOLD-006A plant-time/control substrate
//! to the one-dimensional continuous-time double integrator
//!
//! `x_dot = v`, `v_dot = u`, `u in [u_min, u_max]`.
//!
//! The reference truth is closed-form. Terminal velocity fixes total acceleration
//! impulse; terminal position is bounded by the extremal first moments of that
//! impulse. Numerical optimization is not used as infeasibility authority.

use blake3::Hasher;

use crate::kinodynamic_reachability::{
    ControlInterval1D, DynamicsReplayPolicy1D, DynamicsSemantics, DynamicsTimeDomain,
    KinodynamicError, PlantTimeProfile,
};
use crate::reachability::UnknownReason;
use crate::state_space::{EuclideanSpace, StateSpace};

/// Maximum number of piecewise-constant acceleration segments accepted by the
/// reference replay validator. The bound keeps replay work finite and explicit.
pub const MAX_DOUBLE_INTEGRATOR_REPLAY_SEGMENTS: usize = 16_384;

/// Position/velocity state for a one-dimensional double integrator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleIntegratorState1D {
    position: f64,
    velocity: f64,
}

impl DoubleIntegratorState1D {
    /// Construct a finite `(position, velocity)` state.
    pub fn new(position: f64, velocity: f64) -> Result<Self, KinodynamicError> {
        if !position.is_finite() || !velocity.is_finite() {
            return Err(KinodynamicError::InvalidQuery {
                reason: "double-integrator state must contain finite position and velocity"
                    .to_string(),
            });
        }
        Ok(Self {
            position: canonical_zero(position),
            velocity: canonical_zero(velocity),
        })
    }

    /// Position coordinate.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Velocity coordinate.
    pub fn velocity(&self) -> f64 {
        self.velocity
    }
}

/// Exact bounded continuous-time one-dimensional double-integrator model.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundedDoubleIntegrator1D {
    controls: ControlInterval1D,
    state_space_identity: [u8; 32],
    identity: [u8; 32],
}

impl BoundedDoubleIntegrator1D {
    /// Construct `x_dot=v, v_dot=u` with one exact acceleration interval.
    pub fn new(controls: ControlInterval1D) -> Self {
        let state_space = EuclideanSpace::new(2);
        let state_space_identity = state_space.profile().identity();
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-bounded-double-integrator-1d-v1\0");
        hasher.update(&state_space_identity);
        hasher.update(&controls.identity());
        hasher.update(b"continuous-time\0plant-model-seconds\0");
        Self {
            controls,
            state_space_identity,
            identity: *hasher.finalize().as_bytes(),
        }
    }

    /// Exact scalar acceleration space inherited from MANIFOLD-006A.
    pub fn controls(&self) -> ControlInterval1D {
        self.controls
    }

    /// Exact `R^2` state-space profile identity.
    pub fn state_space_identity(&self) -> [u8; 32] {
        self.state_space_identity
    }

    /// Continuous-time derivative semantics.
    pub fn semantics(&self) -> DynamicsSemantics {
        DynamicsSemantics::ContinuousTime
    }

    /// Explicit plant/model time domain.
    pub fn time_domain(&self) -> DynamicsTimeDomain {
        DynamicsTimeDomain::PlantModelSeconds
    }

    /// Deterministic model identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Exact point-target query at one plant/model horizon.
#[derive(Clone, Debug, PartialEq)]
pub struct DoubleIntegratorQuery1D {
    model_identity: [u8; 32],
    initial: DoubleIntegratorState1D,
    target: DoubleIntegratorState1D,
    plant_time: PlantTimeProfile,
    replay_policy: DynamicsReplayPolicy1D,
    identity: [u8; 32],
}

impl DoubleIntegratorQuery1D {
    /// Construct an exact model-bound position/velocity target query.
    pub fn new(
        model: &BoundedDoubleIntegrator1D,
        initial: DoubleIntegratorState1D,
        target: DoubleIntegratorState1D,
        plant_time: PlantTimeProfile,
        replay_policy: DynamicsReplayPolicy1D,
    ) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-double-integrator-query-1d-v1\0");
        hasher.update(&model.identity());
        hash_state(&mut hasher, initial);
        hash_state(&mut hasher, target);
        hasher.update(&plant_time.identity());
        hasher.update(&replay_policy.identity());
        Self {
            model_identity: model.identity(),
            initial,
            target,
            plant_time,
            replay_policy,
            identity: *hasher.finalize().as_bytes(),
        }
    }

    /// Exact model identity.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }

    /// Initial `(position, velocity)` state.
    pub fn initial(&self) -> DoubleIntegratorState1D {
        self.initial
    }

    /// Target `(position, velocity)` state.
    pub fn target(&self) -> DoubleIntegratorState1D {
        self.target
    }

    /// Exact plant/model horizon.
    pub fn plant_time(&self) -> PlantTimeProfile {
        self.plant_time
    }

    /// Explicit finite-arithmetic/replay policy.
    pub fn replay_policy(&self) -> DynamicsReplayPolicy1D {
        self.replay_policy
    }

    /// Deterministic query identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Conditional terminal reachable slice at the requested target velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleIntegratorTerminalSlice1D {
    velocity_lower: f64,
    velocity_upper: f64,
    position_lower: f64,
    position_upper: f64,
    high_acceleration_duration: f64,
}

impl DoubleIntegratorTerminalSlice1D {
    /// Raw lower terminal velocity bound.
    pub fn velocity_lower(&self) -> f64 {
        self.velocity_lower
    }
    /// Raw upper terminal velocity bound.
    pub fn velocity_upper(&self) -> f64 {
        self.velocity_upper
    }
    /// Raw lower terminal position bound conditional on target velocity.
    pub fn position_lower(&self) -> f64 {
        self.position_lower
    }
    /// Raw upper terminal position bound conditional on target velocity.
    pub fn position_upper(&self) -> f64 {
        self.position_upper
    }
    /// Required total duration at `u_max` above the `u_min` baseline.
    pub fn high_acceleration_duration(&self) -> f64 {
        self.high_acceleration_duration
    }
}

/// One piecewise-constant acceleration segment with declared endpoint states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedAccelerationSegment1D {
    start_time: f64,
    end_time: f64,
    acceleration: f64,
    start_state: DoubleIntegratorState1D,
    end_state: DoubleIntegratorState1D,
}

impl TimedAccelerationSegment1D {
    /// Construct a raw segment candidate. Replay owns dynamics validity.
    pub fn new(
        start_time: f64,
        end_time: f64,
        acceleration: f64,
        start_state: DoubleIntegratorState1D,
        end_state: DoubleIntegratorState1D,
    ) -> Self {
        Self {
            start_time: canonical_zero(start_time),
            end_time: canonical_zero(end_time),
            acceleration: canonical_zero(acceleration),
            start_state,
            end_state,
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
    /// Piecewise-constant acceleration.
    pub fn acceleration(&self) -> f64 {
        self.acceleration
    }
    /// Declared start state.
    pub fn start_state(&self) -> DoubleIntegratorState1D {
        self.start_state
    }
    /// Declared end state.
    pub fn end_state(&self) -> DoubleIntegratorState1D {
        self.end_state
    }
}

/// Timed acceleration trajectory candidate. Construction alone carries no authority.
#[derive(Clone, Debug, PartialEq)]
pub struct TimedAccelerationTrajectory1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    segments: Vec<TimedAccelerationSegment1D>,
    identity: [u8; 32],
}

impl TimedAccelerationTrajectory1D {
    /// Construct a model/query-bound candidate from ordered positive-duration segments.
    pub fn new(
        model: &BoundedDoubleIntegrator1D,
        query: &DoubleIntegratorQuery1D,
        segments: Vec<TimedAccelerationSegment1D>,
    ) -> Result<Self, KinodynamicError> {
        require_model_query_identity(model, query)?;
        if segments.is_empty() {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: "double-integrator trajectory must contain at least one segment"
                    .to_string(),
            });
        }
        if segments.len() > MAX_DOUBLE_INTEGRATOR_REPLAY_SEGMENTS {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!(
                    "double-integrator trajectory has {} segments, exceeding replay bound {}",
                    segments.len(),
                    MAX_DOUBLE_INTEGRATOR_REPLAY_SEGMENTS
                ),
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

    /// Exact model identity claimed by the candidate.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }
    /// Exact query identity claimed by the candidate.
    pub fn query_identity(&self) -> [u8; 32] {
        self.query_identity
    }
    /// Ordered acceleration segments.
    pub fn segments(&self) -> &[TimedAccelerationSegment1D] {
        &self.segments
    }
    /// Exact candidate identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Independent dynamics replay receipt for a double-integrator witness.
#[derive(Clone, Debug, PartialEq)]
pub struct DoubleIntegratorReplayReceipt1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    trajectory_identity: [u8; 32],
    validator_identity: [u8; 32],
    segment_count: usize,
    maximum_position_residual: f64,
    maximum_velocity_residual: f64,
    final_state: DoubleIntegratorState1D,
}

impl DoubleIntegratorReplayReceipt1D {
    /// Exact trajectory identity replayed.
    pub fn trajectory_identity(&self) -> [u8; 32] {
        self.trajectory_identity
    }
    /// Independent validator identity.
    pub fn validator_identity(&self) -> [u8; 32] {
        self.validator_identity
    }
    /// Number of replayed segments.
    pub fn segment_count(&self) -> usize {
        self.segment_count
    }
    /// Largest position transition/terminal residual.
    pub fn maximum_position_residual(&self) -> f64 {
        self.maximum_position_residual
    }
    /// Largest velocity transition/terminal residual.
    pub fn maximum_velocity_residual(&self) -> f64 {
        self.maximum_velocity_residual
    }
    /// Replayed terminal state.
    pub fn final_state(&self) -> DoubleIntegratorState1D {
        self.final_state
    }
    /// Exact model identity.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }
    /// Exact query identity.
    pub fn query_identity(&self) -> [u8; 32] {
        self.query_identity
    }
}

/// Analytic separator responsible for a certified unreachable target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoubleIntegratorSeparator1D {
    /// Target velocity is below the guarded impulse interval.
    VelocityBelow,
    /// Target velocity is above the guarded impulse interval.
    VelocityAbove,
    /// Target position is below the guarded first-moment interval.
    PositionBelow,
    /// Target position is above the guarded first-moment interval.
    PositionAbove,
}

/// Independently verifiable analytic double-integrator infeasibility certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct DoubleIntegratorInfeasibilityCertificate1D {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    terminal_slice: DoubleIntegratorTerminalSlice1D,
    separator: DoubleIntegratorSeparator1D,
    identity: [u8; 32],
}

impl DoubleIntegratorInfeasibilityCertificate1D {
    /// Raw terminal bounds used by the certificate.
    pub fn terminal_slice(&self) -> DoubleIntegratorTerminalSlice1D {
        self.terminal_slice
    }
    /// Analytic separator owning the impossibility claim.
    pub fn separator(&self) -> DoubleIntegratorSeparator1D {
        self.separator
    }
    /// Exact certificate identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Independent verification receipt for a double-integrator certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoubleIntegratorCertificateVerificationReceipt {
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    certificate_identity: [u8; 32],
    verifier_identity: [u8; 32],
}

impl DoubleIntegratorCertificateVerificationReceipt {
    /// Verified certificate identity.
    pub fn certificate_identity(&self) -> [u8; 32] {
        self.certificate_identity
    }
    /// Independent verifier identity.
    pub fn verifier_identity(&self) -> [u8; 32] {
        self.verifier_identity
    }
    /// Exact model identity.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }
    /// Exact query identity.
    pub fn query_identity(&self) -> [u8; 32] {
        self.query_identity
    }
}

/// Proof-strength result for the complete analytic double-integrator reference solver.
#[derive(Clone, Debug, PartialEq)]
pub enum DoubleIntegratorReachabilityResult1D {
    /// A constructive acceleration witness independently replayed under the exact model.
    Feasible {
        /// Constructive bang-bang/constant acceleration trajectory.
        trajectory: TimedAccelerationTrajectory1D,
        /// Independent dynamics replay receipt.
        replay: DoubleIntegratorReplayReceipt1D,
    },
    /// An independently verified analytic separator proves the target unreachable.
    CertifiedInfeasible {
        /// Analytic certificate.
        certificate: DoubleIntegratorInfeasibilityCertificate1D,
        /// Independent certificate-verification receipt.
        verification: DoubleIntegratorCertificateVerificationReceipt,
    },
    /// Finite arithmetic is too close to an analytic boundary for a strong claim.
    Unknown {
        /// Shared proof-strength unknown category.
        reason: UnknownReason,
        /// Bounded diagnostic detail.
        detail: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Analysis {
    Feasible {
        slice: DoubleIntegratorTerminalSlice1D,
        switch_start: f64,
    },
    Certifiable {
        slice: DoubleIntegratorTerminalSlice1D,
        separator: DoubleIntegratorSeparator1D,
    },
    Ambiguous,
}

/// Compute the conditional terminal slice at the query's target velocity.
///
/// An error means the exact model/query arithmetic is malformed or non-finite.
pub fn double_integrator_terminal_slice(
    model: &BoundedDoubleIntegrator1D,
    query: &DoubleIntegratorQuery1D,
) -> Result<DoubleIntegratorTerminalSlice1D, KinodynamicError> {
    require_model_query_identity(model, query)?;
    terminal_slice_raw(model, query)
}

/// Complete closed-form reachability solver for the bounded 1-D double integrator.
pub fn solve_double_integrator_analytic(
    model: &BoundedDoubleIntegrator1D,
    query: &DoubleIntegratorQuery1D,
) -> Result<DoubleIntegratorReachabilityResult1D, KinodynamicError> {
    match analyze_target(model, query)? {
        Analysis::Ambiguous => Ok(DoubleIntegratorReachabilityResult1D::Unknown {
            reason: UnknownReason::NumericalFailure,
            detail: "target lies inside an explicit finite-arithmetic guard or bang-bang switch reconstruction is numerically ambiguous".to_string(),
        }),
        Analysis::Certifiable { slice, separator } => {
            let certificate = build_certificate(model, query, slice, separator);
            let verification = verify_double_integrator_infeasibility(model, query, &certificate)?;
            Ok(DoubleIntegratorReachabilityResult1D::CertifiedInfeasible {
                certificate,
                verification,
            })
        }
        Analysis::Feasible { slice, switch_start } => {
            let trajectory = construct_witness(model, query, slice, switch_start)?;
            let replay = replay_double_integrator_trajectory(model, query, &trajectory)?;
            Ok(DoubleIntegratorReachabilityResult1D::Feasible { trajectory, replay })
        }
    }
}

/// Independently replay a timed acceleration witness through exact plant dynamics.
pub fn replay_double_integrator_trajectory(
    model: &BoundedDoubleIntegrator1D,
    query: &DoubleIntegratorQuery1D,
    trajectory: &TimedAccelerationTrajectory1D,
) -> Result<DoubleIntegratorReplayReceipt1D, KinodynamicError> {
    require_model_query_identity(model, query)?;
    if trajectory.model_identity() != model.identity()
        || trajectory.query_identity() != query.identity()
    {
        return Err(KinodynamicError::IdentityMismatch {
            reason: "double-integrator trajectory is bound to a different model/query".to_string(),
        });
    }

    // Match the qualified 006A replay boundary: check only raw analytic target
    // membership, then let the supplied trajectory stand or fall on dynamics replay.
    // Do not require the solver's canonical bang-bang switch reconstruction.
    let raw_slice = terminal_slice_raw(model, query)?;
    let target = query.target();
    let raw_velocity_feasible =
        target.velocity() >= raw_slice.velocity_lower() && target.velocity() <= raw_slice.velocity_upper();
    let raw_position_feasible = raw_velocity_feasible
        && target.position() >= raw_slice.position_lower()
        && target.position() <= raw_slice.position_upper();
    if !raw_position_feasible {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: "query target is outside the raw analytic terminal reachable slice; replay cannot promote it to feasibility"
                .to_string(),
        });
    }

    let segments = trajectory.segments();
    let first = segments.first().expect("trajectory constructor forbids empty segments");
    if first.start_time().to_bits() != 0.0_f64.to_bits()
        || !same_state_bits(first.start_state(), query.initial())
    {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: "trajectory must begin exactly at plant time 0 and the exact initial state"
                .to_string(),
        });
    }

    let tolerance = query.replay_policy().absolute_state_tolerance();
    let mut previous_time = 0.0_f64;
    let mut previous_declared_state = query.initial();
    let mut replayed_state = query.initial();
    let mut max_position_residual = 0.0_f64;
    let mut max_velocity_residual = 0.0_f64;

    for (index, segment) in segments.iter().enumerate() {
        for (name, value) in [
            ("start_time", segment.start_time()),
            ("end_time", segment.end_time()),
            ("acceleration", segment.acceleration()),
            ("start_position", segment.start_state().position()),
            ("start_velocity", segment.start_state().velocity()),
            ("end_position", segment.end_state().position()),
            ("end_velocity", segment.end_state().velocity()),
        ] {
            if !value.is_finite() {
                return Err(KinodynamicError::InvalidTrajectory {
                    reason: format!("segment {index} has non-finite {name}"),
                });
            }
        }
        if segment.start_time().to_bits() != previous_time.to_bits() {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!("segment {index} does not begin at exact preceding plant time"),
            });
        }
        if !same_state_bits(segment.start_state(), previous_declared_state) {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!("segment {index} does not begin at exact preceding declared state"),
            });
        }
        if segment.end_time() <= segment.start_time() {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!("segment {index} must have strictly positive duration"),
            });
        }
        if !model.controls().contains(segment.acceleration()) {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!("segment {index} acceleration violates exact control bounds"),
            });
        }

        let start_position_residual =
            checked_abs_diff(replayed_state.position(), segment.start_state().position())?;
        let start_velocity_residual =
            checked_abs_diff(replayed_state.velocity(), segment.start_state().velocity())?;
        if start_position_residual > tolerance || start_velocity_residual > tolerance {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!(
                    "segment {index} declared start diverges from globally replayed state beyond tolerance {tolerance}: position={start_position_residual}, velocity={start_velocity_residual}"
                ),
            });
        }

        let duration = checked_sub(segment.end_time(), segment.start_time())?;
        let predicted = transition(replayed_state, segment.acceleration(), duration)?;
        let position_residual = checked_abs_diff(predicted.position(), segment.end_state().position())?;
        let velocity_residual = checked_abs_diff(predicted.velocity(), segment.end_state().velocity())?;
        if position_residual > tolerance || velocity_residual > tolerance {
            return Err(KinodynamicError::InvalidTrajectory {
                reason: format!(
                    "segment {index} dynamics residual exceeds replay tolerance {tolerance}: position={position_residual}, velocity={velocity_residual}"
                ),
            });
        }
        max_position_residual = max_position_residual
            .max(start_position_residual)
            .max(position_residual);
        max_velocity_residual = max_velocity_residual
            .max(start_velocity_residual)
            .max(velocity_residual);
        previous_time = segment.end_time();
        previous_declared_state = segment.end_state();
        replayed_state = predicted;
    }

    if previous_time.to_bits() != query.plant_time().horizon_seconds().to_bits() {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: "trajectory final timestamp must equal the exact plant/model horizon"
                .to_string(),
        });
    }
    let terminal_position_residual =
        checked_abs_diff(replayed_state.position(), query.target().position())?;
    let terminal_velocity_residual =
        checked_abs_diff(replayed_state.velocity(), query.target().velocity())?;
    if terminal_position_residual > tolerance || terminal_velocity_residual > tolerance {
        return Err(KinodynamicError::InvalidTrajectory {
            reason: format!(
                "globally replayed terminal state residual exceeds tolerance {tolerance}: position={terminal_position_residual}, velocity={terminal_velocity_residual}"
            ),
        });
    }
    max_position_residual = max_position_residual.max(terminal_position_residual);
    max_velocity_residual = max_velocity_residual.max(terminal_velocity_residual);

    Ok(DoubleIntegratorReplayReceipt1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        trajectory_identity: trajectory.identity(),
        validator_identity: replay_validator_identity(query.replay_policy().identity()),
        segment_count: segments.len(),
        maximum_position_residual: max_position_residual,
        maximum_velocity_residual: max_velocity_residual,
        final_state: replayed_state,
    })
}

/// Independently verify an analytic double-integrator infeasibility certificate.
pub fn verify_double_integrator_infeasibility(
    model: &BoundedDoubleIntegrator1D,
    query: &DoubleIntegratorQuery1D,
    certificate: &DoubleIntegratorInfeasibilityCertificate1D,
) -> Result<DoubleIntegratorCertificateVerificationReceipt, KinodynamicError> {
    require_model_query_identity(model, query)?;
    if certificate.model_identity != model.identity() || certificate.query_identity != query.identity() {
        return Err(KinodynamicError::InvalidCertificate {
            reason: "double-integrator certificate subject identities do not match".to_string(),
        });
    }

    // Recompute theorem inputs directly rather than reusing `analyze_target`, so
    // certificate verification does not inherit the solver classifier's branch logic.
    let slice = terminal_slice_raw(model, query)?;
    if certificate.terminal_slice != slice {
        return Err(KinodynamicError::InvalidCertificate {
            reason: "certificate terminal slice does not match independently recomputed theorem inputs"
                .to_string(),
        });
    }
    let tolerance = query.replay_policy().absolute_state_tolerance();
    let target = query.target();
    let velocity_raw_feasible =
        target.velocity() >= slice.velocity_lower() && target.velocity() <= slice.velocity_upper();

    let separator_holds = match certificate.separator {
        DoubleIntegratorSeparator1D::VelocityBelow => {
            target.velocity() < checked_sub(slice.velocity_lower(), tolerance)?
        }
        DoubleIntegratorSeparator1D::VelocityAbove => {
            target.velocity() > checked_add(slice.velocity_upper(), tolerance)?
        }
        DoubleIntegratorSeparator1D::PositionBelow => {
            velocity_raw_feasible
                && target.position() < checked_sub(slice.position_lower(), tolerance)?
        }
        DoubleIntegratorSeparator1D::PositionAbove => {
            velocity_raw_feasible
                && target.position() > checked_add(slice.position_upper(), tolerance)?
        }
    };
    if !separator_holds {
        return Err(KinodynamicError::InvalidCertificate {
            reason: "declared analytic separator does not independently exclude the target"
                .to_string(),
        });
    }

    let expected = hash_certificate(
        model.identity(),
        query.identity(),
        slice,
        certificate.separator,
    );
    if certificate.identity != expected {
        return Err(KinodynamicError::InvalidCertificate {
            reason: "certificate identity does not match independently recomputed theorem inputs"
                .to_string(),
        });
    }
    Ok(DoubleIntegratorCertificateVerificationReceipt {
        model_identity: model.identity(),
        query_identity: query.identity(),
        certificate_identity: certificate.identity(),
        verifier_identity: certificate_verifier_identity(),
    })
}

fn analyze_target(
    model: &BoundedDoubleIntegrator1D,
    query: &DoubleIntegratorQuery1D,
) -> Result<Analysis, KinodynamicError> {
    require_model_query_identity(model, query)?;
    let tolerance = query.replay_policy().absolute_state_tolerance();
    let slice = terminal_slice_raw(model, query)?;
    let target = query.target();

    if target.velocity() < checked_sub(slice.velocity_lower(), tolerance)? {
        return Ok(Analysis::Certifiable {
            slice,
            separator: DoubleIntegratorSeparator1D::VelocityBelow,
        });
    }
    if target.velocity() > checked_add(slice.velocity_upper(), tolerance)? {
        return Ok(Analysis::Certifiable {
            slice,
            separator: DoubleIntegratorSeparator1D::VelocityAbove,
        });
    }
    if target.velocity() < slice.velocity_lower() || target.velocity() > slice.velocity_upper() {
        return Ok(Analysis::Ambiguous);
    }

    if target.position() < checked_sub(slice.position_lower(), tolerance)? {
        return Ok(Analysis::Certifiable {
            slice,
            separator: DoubleIntegratorSeparator1D::PositionBelow,
        });
    }
    if target.position() > checked_add(slice.position_upper(), tolerance)? {
        return Ok(Analysis::Certifiable {
            slice,
            separator: DoubleIntegratorSeparator1D::PositionAbove,
        });
    }
    if target.position() < slice.position_lower() || target.position() > slice.position_upper() {
        return Ok(Analysis::Ambiguous);
    }

    let controls = model.controls();
    let horizon = query.plant_time().horizon_seconds();
    let delta_u = checked_sub(controls.maximum(), controls.minimum())?;
    let p = slice.high_acceleration_duration();
    let switch_start = if delta_u == 0.0 || p == 0.0 || p.to_bits() == horizon.to_bits() {
        0.0
    } else {
        let initial = query.initial();
        let y = checked_sub(
            checked_sub(query.target().position(), initial.position())?,
            checked_mul(initial.velocity(), horizon)?,
        )?;
        let t_p = checked_mul(horizon, p)?;
        let half_p2 = checked_mul(0.5, checked_mul(p, p)?)?;
        let j0 = checked_mul(
            0.5,
            checked_mul(controls.minimum(), checked_mul(horizon, horizon)?)?,
        )?;
        let normalized = checked_div(checked_sub(y, j0)?, delta_u)?;
        let numerator = checked_sub(checked_sub(t_p, half_p2)?, normalized)?;
        let s = canonical_zero(checked_div(numerator, p)?);
        let latest = canonical_zero(checked_sub(horizon, p)?);
        if s < 0.0 || s > latest {
            return Ok(Analysis::Ambiguous);
        }
        s
    };

    Ok(Analysis::Feasible { slice, switch_start })
}

fn terminal_slice_raw(
    model: &BoundedDoubleIntegrator1D,
    query: &DoubleIntegratorQuery1D,
) -> Result<DoubleIntegratorTerminalSlice1D, KinodynamicError> {
    let controls = model.controls();
    let horizon = query.plant_time().horizon_seconds();
    let initial = query.initial();
    let target = query.target();

    let min_impulse = checked_mul(controls.minimum(), horizon)?;
    let max_impulse = checked_mul(controls.maximum(), horizon)?;
    let velocity_lower = checked_add(initial.velocity(), min_impulse)?;
    let velocity_upper = checked_add(initial.velocity(), max_impulse)?;

    let delta_v = checked_sub(target.velocity(), initial.velocity())?;
    let delta_u = checked_sub(controls.maximum(), controls.minimum())?;
    let horizon2 = checked_mul(horizon, horizon)?;
    let drift = checked_add(initial.position(), checked_mul(initial.velocity(), horizon)?)?;

    if delta_u == 0.0 {
        let p = 0.0;
        let moment = checked_mul(0.5, checked_mul(controls.minimum(), horizon2)?)?;
        let position = checked_add(drift, moment)?;
        return Ok(DoubleIntegratorTerminalSlice1D {
            velocity_lower,
            velocity_upper,
            position_lower: position,
            position_upper: position,
            high_acceleration_duration: p,
        });
    }

    let baseline_impulse = min_impulse;
    let p = canonical_zero(checked_div(checked_sub(delta_v, baseline_impulse)?, delta_u)?);
    if !p.is_finite() {
        return Err(KinodynamicError::Numerical {
            reason: "required high-acceleration duration became non-finite".to_string(),
        });
    }

    // Outside the raw velocity interval, position bounds are diagnostic only. Use a
    // finite boundary duration so certificate identity remains deterministic.
    let bounded_p = if p < 0.0 {
        0.0
    } else if p > horizon {
        horizon
    } else {
        p
    };
    let p2 = checked_mul(bounded_p, bounded_p)?;
    let baseline_moment = checked_mul(0.5, checked_mul(controls.minimum(), horizon2)?)?;
    let minimum_extra = checked_mul(0.5, checked_mul(delta_u, p2)?)?;
    let maximum_kernel = checked_sub(checked_mul(horizon, bounded_p)?, checked_mul(0.5, p2)?)?;
    let maximum_extra = checked_mul(delta_u, maximum_kernel)?;
    let position_lower = checked_add(drift, checked_add(baseline_moment, minimum_extra)?)?;
    let position_upper = checked_add(drift, checked_add(baseline_moment, maximum_extra)?)?;

    Ok(DoubleIntegratorTerminalSlice1D {
        velocity_lower,
        velocity_upper,
        position_lower,
        position_upper,
        high_acceleration_duration: p,
    })
}

fn construct_witness(
    model: &BoundedDoubleIntegrator1D,
    query: &DoubleIntegratorQuery1D,
    slice: DoubleIntegratorTerminalSlice1D,
    switch_start: f64,
) -> Result<TimedAccelerationTrajectory1D, KinodynamicError> {
    let controls = model.controls();
    let horizon = query.plant_time().horizon_seconds();
    let delta_u = checked_sub(controls.maximum(), controls.minimum())?;
    let p = slice.high_acceleration_duration();
    let mut segments = Vec::with_capacity(3);
    let mut time = 0.0;
    let mut state = query.initial();

    if delta_u == 0.0 || p == 0.0 {
        push_segment(&mut segments, &mut state, &mut time, horizon, controls.minimum())?;
    } else if p.to_bits() == horizon.to_bits() {
        push_segment(&mut segments, &mut state, &mut time, horizon, controls.maximum())?;
    } else {
        if switch_start > 0.0 {
            push_segment(
                &mut segments,
                &mut state,
                &mut time,
                switch_start,
                controls.minimum(),
            )?;
        }
        let high_end = checked_add(switch_start, p)?;
        if high_end > horizon {
            return Err(KinodynamicError::Numerical {
                reason: "bang-bang high-acceleration interval escaped plant horizon".to_string(),
            });
        }
        push_segment(
            &mut segments,
            &mut state,
            &mut time,
            high_end,
            controls.maximum(),
        )?;
        if time.to_bits() != horizon.to_bits() {
            if time > horizon {
                return Err(KinodynamicError::Numerical {
                    reason: "bang-bang reconstruction passed exact plant horizon".to_string(),
                });
            }
            push_segment(
                &mut segments,
                &mut state,
                &mut time,
                horizon,
                controls.minimum(),
            )?;
        }
    }

    TimedAccelerationTrajectory1D::new(model, query, segments)
}

fn push_segment(
    segments: &mut Vec<TimedAccelerationSegment1D>,
    state: &mut DoubleIntegratorState1D,
    time: &mut f64,
    end_time: f64,
    acceleration: f64,
) -> Result<(), KinodynamicError> {
    if end_time <= *time {
        return Err(KinodynamicError::Numerical {
            reason: "constructive bang-bang witness produced non-positive segment duration"
                .to_string(),
        });
    }
    let duration = checked_sub(end_time, *time)?;
    let end_state = transition(*state, acceleration, duration)?;
    segments.push(TimedAccelerationSegment1D::new(
        *time,
        end_time,
        acceleration,
        *state,
        end_state,
    ));
    *state = end_state;
    *time = canonical_zero(end_time);
    Ok(())
}

fn transition(
    state: DoubleIntegratorState1D,
    acceleration: f64,
    duration: f64,
) -> Result<DoubleIntegratorState1D, KinodynamicError> {
    if !acceleration.is_finite() || !duration.is_finite() || duration <= 0.0 {
        return Err(KinodynamicError::Numerical {
            reason: "double-integrator transition requires finite acceleration and positive duration"
                .to_string(),
        });
    }
    let velocity_delta = checked_mul(acceleration, duration)?;
    let next_velocity = checked_add(state.velocity(), velocity_delta)?;
    let drift = checked_mul(state.velocity(), duration)?;
    let duration2 = checked_mul(duration, duration)?;
    let accel_position = checked_mul(0.5, checked_mul(acceleration, duration2)?)?;
    let next_position = checked_add(state.position(), checked_add(drift, accel_position)?)?;
    DoubleIntegratorState1D::new(next_position, next_velocity)
}

fn build_certificate(
    model: &BoundedDoubleIntegrator1D,
    query: &DoubleIntegratorQuery1D,
    terminal_slice: DoubleIntegratorTerminalSlice1D,
    separator: DoubleIntegratorSeparator1D,
) -> DoubleIntegratorInfeasibilityCertificate1D {
    let identity = hash_certificate(model.identity(), query.identity(), terminal_slice, separator);
    DoubleIntegratorInfeasibilityCertificate1D {
        model_identity: model.identity(),
        query_identity: query.identity(),
        terminal_slice,
        separator,
        identity,
    }
}

fn require_model_query_identity(
    model: &BoundedDoubleIntegrator1D,
    query: &DoubleIntegratorQuery1D,
) -> Result<(), KinodynamicError> {
    if query.model_identity() != model.identity() {
        return Err(KinodynamicError::IdentityMismatch {
            reason: "double-integrator query was constructed for a different model".to_string(),
        });
    }
    Ok(())
}

fn hash_trajectory(
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    segments: &[TimedAccelerationSegment1D],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-timed-acceleration-trajectory-1d-v1\0");
    hasher.update(&model_identity);
    hasher.update(&query_identity);
    hasher.update(&(segments.len() as u64).to_le_bytes());
    for segment in segments {
        for value in [segment.start_time(), segment.end_time(), segment.acceleration()] {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        hash_state(&mut hasher, segment.start_state());
        hash_state(&mut hasher, segment.end_state());
    }
    *hasher.finalize().as_bytes()
}

fn hash_certificate(
    model_identity: [u8; 32],
    query_identity: [u8; 32],
    slice: DoubleIntegratorTerminalSlice1D,
    separator: DoubleIntegratorSeparator1D,
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-double-integrator-infeasibility-certificate-v1\0");
    hasher.update(&model_identity);
    hasher.update(&query_identity);
    for value in [
        slice.velocity_lower(),
        slice.velocity_upper(),
        slice.position_lower(),
        slice.position_upper(),
        slice.high_acceleration_duration(),
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&[match separator {
        DoubleIntegratorSeparator1D::VelocityBelow => 0,
        DoubleIntegratorSeparator1D::VelocityAbove => 1,
        DoubleIntegratorSeparator1D::PositionBelow => 2,
        DoubleIntegratorSeparator1D::PositionAbove => 3,
    }]);
    *hasher.finalize().as_bytes()
}

fn replay_validator_identity(policy_identity: [u8; 32]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-double-integrator-replay-validator-v1\0");
    hasher.update(&policy_identity);
    hasher.update(&(MAX_DOUBLE_INTEGRATOR_REPLAY_SEGMENTS as u64).to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn certificate_verifier_identity() -> [u8; 32] {
    *blake3::hash(b"symthaea-double-integrator-certificate-verifier-v1\0").as_bytes()
}

fn hash_state(hasher: &mut Hasher, state: DoubleIntegratorState1D) {
    hasher.update(&state.position().to_bits().to_le_bytes());
    hasher.update(&state.velocity().to_bits().to_le_bytes());
}

fn same_state_bits(left: DoubleIntegratorState1D, right: DoubleIntegratorState1D) -> bool {
    left.position().to_bits() == right.position().to_bits()
        && left.velocity().to_bits() == right.velocity().to_bits()
}

fn checked_add(left: f64, right: f64) -> Result<f64, KinodynamicError> {
    let value = left + right;
    finite_arithmetic("addition", value)
}

fn checked_sub(left: f64, right: f64) -> Result<f64, KinodynamicError> {
    let value = left - right;
    finite_arithmetic("subtraction", value)
}

fn checked_mul(left: f64, right: f64) -> Result<f64, KinodynamicError> {
    let value = left * right;
    finite_arithmetic("multiplication", value)
}

fn checked_div(left: f64, right: f64) -> Result<f64, KinodynamicError> {
    if right == 0.0 {
        return Err(KinodynamicError::Numerical {
            reason: "division by zero in double-integrator analytic theorem".to_string(),
        });
    }
    let value = left / right;
    finite_arithmetic("division", value)
}

fn checked_abs_diff(left: f64, right: f64) -> Result<f64, KinodynamicError> {
    let value = (left - right).abs();
    if !value.is_finite() {
        return Err(KinodynamicError::Numerical {
            reason: "double-integrator residual became non-finite".to_string(),
        });
    }
    Ok(value)
}

fn finite_arithmetic(operation: &str, value: f64) -> Result<f64, KinodynamicError> {
    if !value.is_finite() {
        return Err(KinodynamicError::Numerical {
            reason: format!("double-integrator {operation} became non-finite"),
        });
    }
    Ok(canonical_zero(value))
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}
