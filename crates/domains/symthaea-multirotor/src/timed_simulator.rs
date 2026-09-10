// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Timing-evidence wrapper for multirotor physics backends.
//!
//! `PhysicsSimulator::step(cmd, dt)` historically accepts a caller-provided `dt`
//! but does not expose whether the backend actually advanced model time by exactly
//! that duration. `TimedPhysicsSimulator` preserves the existing simulator behavior
//! while recording the requested duration and the observed model-time transition as
//! separate evidence.

use crate::simulator::PhysicsSimulator;
use crate::types::{FlightState, QuadrotorCommand};
use symthaea_core::embodiment_evidence::{ClockDomainId, TimestampV1};
use symthaea_core::embodiment_timing::{
    PlantStepTimingV1, PlantTimingSourceV1, PlantTimingValidationError,
};

const SIM_CLOCK_DOMAIN: &str = "symthaea.multirotor.sim.model-time";

/// Failure to capture a timing receipt around a simulator step.
#[derive(Debug, Clone, PartialEq)]
pub enum SimulatorTimingCaptureError {
    /// Caller-provided step duration was NaN or infinite.
    NonFiniteRequestedStep,
    /// Caller-provided step duration was zero or negative.
    NonPositiveRequestedStep,
    /// Positive requested duration could not be represented as nanoseconds.
    UnrepresentableRequestedStep,
    /// Pre-step simulator timestamp could not be represented as non-negative nanoseconds.
    InvalidPreStepTimestamp,
    /// Post-step simulator timestamp could not be represented as non-negative nanoseconds.
    InvalidPostStepTimestamp,
    /// The captured receipt failed core timing validation.
    Receipt(PlantTimingValidationError),
}

impl From<PlantTimingValidationError> for SimulatorTimingCaptureError {
    fn from(value: PlantTimingValidationError) -> Self {
        Self::Receipt(value)
    }
}

impl std::fmt::Display for SimulatorTimingCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteRequestedStep => write!(f, "requested simulator step is non-finite"),
            Self::NonPositiveRequestedStep => {
                write!(f, "requested simulator step must be positive")
            }
            Self::UnrepresentableRequestedStep => {
                write!(f, "requested simulator step cannot be represented as nanoseconds")
            }
            Self::InvalidPreStepTimestamp => write!(f, "invalid pre-step simulator timestamp"),
            Self::InvalidPostStepTimestamp => write!(f, "invalid post-step simulator timestamp"),
            Self::Receipt(error) => write!(f, "invalid simulator timing receipt: {error}"),
        }
    }
}

impl std::error::Error for SimulatorTimingCaptureError {}

/// Existing multirotor physics backend plus explicit model-time receipts.
pub struct TimedPhysicsSimulator<S> {
    inner: S,
    backend_profile_id: String,
    last_timing: Option<Result<PlantStepTimingV1, SimulatorTimingCaptureError>>,
}

impl<S> TimedPhysicsSimulator<S>
where
    S: PhysicsSimulator,
{
    /// Wrap an existing simulator backend.
    ///
    /// `backend_profile_id` must identify the concrete stepping/model profile used
    /// for qualification. Empty, padded, and control-character identifiers fail.
    pub fn new(
        inner: S,
        backend_profile_id: impl Into<String>,
    ) -> Result<Self, SimulatorTimingCaptureError> {
        let backend_profile_id = backend_profile_id.into();
        validate_identifier(&backend_profile_id)?;
        Ok(Self {
            inner,
            backend_profile_id,
            last_timing: None,
        })
    }

    /// Borrow the wrapped backend.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Mutably borrow the wrapped backend.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    /// Consume the wrapper and return the backend.
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Last timing capture attempt, if at least one step was executed.
    ///
    /// `Some(Err(..))` is intentionally preserved so malformed timing cannot look
    /// equivalent to "no step has occurred yet".
    pub fn last_step_timing(
        &self,
    ) -> Option<&Result<PlantStepTimingV1, SimulatorTimingCaptureError>> {
        self.last_timing.as_ref()
    }

    fn capture_timing(
        &self,
        pre_seconds: f64,
        post_seconds: f64,
        requested_seconds: f64,
    ) -> Result<PlantStepTimingV1, SimulatorTimingCaptureError> {
        if !requested_seconds.is_finite() {
            return Err(SimulatorTimingCaptureError::NonFiniteRequestedStep);
        }
        if requested_seconds <= 0.0 {
            return Err(SimulatorTimingCaptureError::NonPositiveRequestedStep);
        }

        let pre_ns = seconds_to_ns(pre_seconds)
            .ok_or(SimulatorTimingCaptureError::InvalidPreStepTimestamp)?;
        let post_ns = seconds_to_ns(post_seconds)
            .ok_or(SimulatorTimingCaptureError::InvalidPostStepTimestamp)?;
        let requested_ns = seconds_to_ns(requested_seconds)
            .ok_or(SimulatorTimingCaptureError::UnrepresentableRequestedStep)?;

        let clock = ClockDomainId::new(SIM_CLOCK_DOMAIN)
            .map_err(PlantTimingValidationError::from)?;
        let mut receipt = PlantStepTimingV1::new(
            TimestampV1::new(clock.clone(), pre_ns),
            TimestampV1::new(clock, post_ns),
            PlantTimingSourceV1::SimulatorModelClock,
            self.backend_profile_id.clone(),
        )?;

        let effective_ns = receipt.observed_transition_ns()?;
        receipt.requested_step_ns = Some(requested_ns);
        receipt.backend_effective_step_ns = Some(effective_ns);
        receipt.validate()?;
        Ok(receipt)
    }
}

impl<S> PhysicsSimulator for TimedPhysicsSimulator<S>
where
    S: PhysicsSimulator,
{
    fn step(&mut self, cmd: &QuadrotorCommand, dt: f64) {
        let pre_seconds = self.inner.state().timestamp;
        self.inner.step(cmd, dt);
        let post_seconds = self.inner.state().timestamp;
        self.last_timing = Some(self.capture_timing(pre_seconds, post_seconds, dt));
    }

    fn state(&self) -> &FlightState {
        self.inner.state()
    }

    fn reset(&mut self, altitude: f64) {
        self.inner.reset(altitude);
        self.last_timing = None;
    }

    fn reset_with_perturbation(&mut self, altitude: f64, perturbation: f64, seed: u64) {
        self.inner
            .reset_with_perturbation(altitude, perturbation, seed);
        self.last_timing = None;
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        self.inner.apply_external_force(force);
    }
}

fn validate_identifier(value: &str) -> Result<(), SimulatorTimingCaptureError> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() != value.len() || value.chars().any(char::is_control) {
        return Err(SimulatorTimingCaptureError::Receipt(
            PlantTimingValidationError::InvalidIdentifier("backend_profile_id"),
        ));
    }
    Ok(())
}

fn seconds_to_ns(seconds: f64) -> Option<u64> {
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    let ns = seconds * 1_000_000_000.0;
    if !ns.is_finite() || ns > u64::MAX as f64 {
        return None;
    }
    Some(ns.round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::SimplePhysicsSimulator;

    #[test]
    fn simple_backend_emits_matching_requested_and_effective_time() {
        let mut sim = TimedPhysicsSimulator::new(
            SimplePhysicsSimulator::new(),
            "multirotor.simple-physics.v1",
        )
        .unwrap();

        assert!(sim.last_step_timing().is_none());
        sim.step(&QuadrotorCommand::hover(), 0.002);

        let receipt = sim.last_step_timing().unwrap().as_ref().unwrap();
        assert_eq!(receipt.requested_step_ns, Some(2_000_000));
        assert_eq!(receipt.backend_effective_step_ns, Some(2_000_000));
        assert_eq!(receipt.observed_transition_ns().unwrap(), 2_000_000);
        assert_eq!(receipt.backend_step_error_ns(), Some(0));
        assert_eq!(
            receipt.pre_state_at.clock_domain.as_str(),
            SIM_CLOCK_DOMAIN
        );
    }

    struct QuantizedSimulator {
        state: FlightState,
    }

    impl Default for QuantizedSimulator {
        fn default() -> Self {
            Self {
                state: FlightState::hover(0.1),
            }
        }
    }

    impl PhysicsSimulator for QuantizedSimulator {
        fn step(&mut self, _cmd: &QuadrotorCommand, dt: f64) {
            let quantum = 0.001;
            let substeps = (dt / quantum).ceil().max(1.0);
            self.state.timestamp += substeps * quantum;
        }

        fn state(&self) -> &FlightState {
            &self.state
        }

        fn reset(&mut self, altitude: f64) {
            self.state = FlightState::hover(altitude);
        }

        fn reset_with_perturbation(&mut self, altitude: f64, _perturbation: f64, _seed: u64) {
            self.state = FlightState::hover(altitude);
        }

        fn apply_external_force(&mut self, _force: [f64; 3]) {}
    }

    #[test]
    fn quantized_backend_keeps_requested_and_effective_duration_distinct() {
        let mut sim = TimedPhysicsSimulator::new(
            QuantizedSimulator::default(),
            "test.quantized-simulator.v1",
        )
        .unwrap();

        sim.step(&QuadrotorCommand::zero(), 0.0015);
        let receipt = sim.last_step_timing().unwrap().as_ref().unwrap();

        assert_eq!(receipt.requested_step_ns, Some(1_500_000));
        assert_eq!(receipt.backend_effective_step_ns, Some(2_000_000));
        assert_eq!(receipt.backend_step_error_ns(), Some(500_000));
    }

    #[test]
    fn reset_clears_prior_timing_receipt() {
        let mut sim = TimedPhysicsSimulator::new(
            SimplePhysicsSimulator::new(),
            "multirotor.simple-physics.v1",
        )
        .unwrap();
        sim.step(&QuadrotorCommand::hover(), 0.002);
        assert!(sim.last_step_timing().is_some());

        sim.reset(0.5);
        assert!(sim.last_step_timing().is_none());
    }

    #[test]
    fn invalid_requested_step_is_recorded_as_error_not_absence() {
        let mut sim = TimedPhysicsSimulator::new(
            SimplePhysicsSimulator::new(),
            "multirotor.simple-physics.v1",
        )
        .unwrap();

        sim.step(&QuadrotorCommand::hover(), f64::NAN);
        assert_eq!(
            sim.last_step_timing().unwrap(),
            &Err(SimulatorTimingCaptureError::NonFiniteRequestedStep)
        );
    }

    #[test]
    fn invalid_backend_profile_is_rejected_before_stepping() {
        assert!(matches!(
            TimedPhysicsSimulator::new(SimplePhysicsSimulator::new(), "  "),
            Err(SimulatorTimingCaptureError::Receipt(
                PlantTimingValidationError::InvalidIdentifier("backend_profile_id")
            ))
        ));
    }
}
