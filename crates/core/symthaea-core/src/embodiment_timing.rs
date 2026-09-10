// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit timing receipts for Embodiment Contract v2 migration.
//!
//! Cognitive integration time, scheduler cadence, transport delay, and physical
//! plant time are different quantities. This module provides a small behavior-
//! neutral receipt for one observed plant transition without assuming that a
//! requested step duration is the duration the plant actually experienced.

use serde::{Deserialize, Serialize};

use crate::embodiment_evidence::{EvidenceValidationError, TimestampV1};

/// Schema version for [`PlantStepTimingV1`].
pub const PLANT_STEP_TIMING_SCHEMA_V1: u16 = 1;

/// Source of the physical timing assertion attached to a transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlantTimingSourceV1 {
    /// Deterministic/in-process simulator model clock.
    SimulatorModelClock,
    /// Monotonic clock supplied by a physical device/controller.
    DeviceMonotonicClock,
    /// Host monotonic clock used to observe a physical transition.
    HostMonotonicClock,
    /// Remote clock whose relationship to the local system has separate sync evidence.
    SynchronizedRemoteClock,
}

/// Timing evidence for one observed physical state transition.
///
/// The pre/post timestamps establish the observed transition interval in one clock
/// domain. `requested_step_ns` records what the caller asked for. A simulator or
/// backend may additionally report `backend_effective_step_ns` when its actual
/// integration duration differs from the request (for example because of substep
/// quantization). These values are intentionally not forced to be equal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlantStepTimingV1 {
    /// Schema version. Must equal [`PLANT_STEP_TIMING_SCHEMA_V1`].
    pub schema_version: u16,
    /// Timestamp of the predictor/plant-visible pre-state.
    pub pre_state_at: TimestampV1,
    /// Timestamp of the corresponding post-state.
    pub post_state_at: TimestampV1,
    /// Duration requested by the caller/scheduler, when one was explicitly requested.
    pub requested_step_ns: Option<u64>,
    /// Effective integration duration reported by the backend, when known.
    pub backend_effective_step_ns: Option<u64>,
    /// Source class for the physical clock/timing assertion.
    pub timing_source: PlantTimingSourceV1,
    /// Backend/profile identity defining the stepping semantics.
    pub backend_profile_id: String,
    /// Optional scheduler-cycle identity; this is not a physical timestamp.
    pub scheduler_cycle_id: Option<u64>,
    /// Optional cognitive-cycle identity; this is not a physical timestamp.
    pub cognitive_cycle_id: Option<u64>,
    /// Optional provenance reference for a richer evidence object.
    pub provenance_id: Option<String>,
}

impl PlantStepTimingV1 {
    /// Construct a timing receipt and validate all local invariants.
    pub fn new(
        pre_state_at: TimestampV1,
        post_state_at: TimestampV1,
        timing_source: PlantTimingSourceV1,
        backend_profile_id: impl Into<String>,
    ) -> Result<Self, PlantTimingValidationError> {
        let value = Self {
            schema_version: PLANT_STEP_TIMING_SCHEMA_V1,
            pre_state_at,
            post_state_at,
            requested_step_ns: None,
            backend_effective_step_ns: None,
            timing_source,
            backend_profile_id: backend_profile_id.into(),
            scheduler_cycle_id: None,
            cognitive_cycle_id: None,
            provenance_id: None,
        };
        value.validate()?;
        Ok(value)
    }

    /// Observed elapsed time between pre-state and post-state timestamps.
    pub fn observed_transition_ns(&self) -> Result<u64, PlantTimingValidationError> {
        let elapsed = self.post_state_at.elapsed_since(&self.pre_state_at)?;
        if elapsed == 0 {
            return Err(PlantTimingValidationError::ZeroObservedTransition);
        }
        Ok(elapsed)
    }

    /// Difference between requested and backend-reported effective duration.
    ///
    /// Returns `None` unless both durations were reported. The signed value is
    /// `effective - requested`, allowing simulator substep rounding to be audited.
    pub fn backend_step_error_ns(&self) -> Option<i128> {
        let requested = self.requested_step_ns? as i128;
        let effective = self.backend_effective_step_ns? as i128;
        Some(effective - requested)
    }

    /// Validate schema, clock, duration, and identifier invariants.
    pub fn validate(&self) -> Result<(), PlantTimingValidationError> {
        if self.schema_version != PLANT_STEP_TIMING_SCHEMA_V1 {
            return Err(PlantTimingValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }

        self.pre_state_at.validate()?;
        self.post_state_at.validate()?;
        self.observed_transition_ns()?;
        validate_identifier(&self.backend_profile_id, "backend_profile_id")?;
        if let Some(provenance_id) = &self.provenance_id {
            validate_identifier(provenance_id, "provenance_id")?;
        }
        if self.requested_step_ns == Some(0) {
            return Err(PlantTimingValidationError::ZeroRequestedDuration);
        }
        if self.backend_effective_step_ns == Some(0) {
            return Err(PlantTimingValidationError::ZeroBackendEffectiveDuration);
        }

        Ok(())
    }
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), PlantTimingValidationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.chars().any(char::is_control)
    {
        return Err(PlantTimingValidationError::InvalidIdentifier(field));
    }
    Ok(())
}

/// Validation failure for [`PlantStepTimingV1`].
#[derive(Debug, Clone, PartialEq)]
pub enum PlantTimingValidationError {
    /// Receipt uses an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Unsupported schema version encountered.
        found: u16,
    },
    /// Timestamp validation or same-clock arithmetic failed.
    Timestamp(EvidenceValidationError),
    /// Pre/post timestamps were identical for an asserted transition.
    ZeroObservedTransition,
    /// An explicitly requested step duration was zero.
    ZeroRequestedDuration,
    /// A backend-reported effective integration duration was zero.
    ZeroBackendEffectiveDuration,
    /// A required or optional identifier was empty, padded, or contained controls.
    InvalidIdentifier(&'static str),
}

impl From<EvidenceValidationError> for PlantTimingValidationError {
    fn from(value: EvidenceValidationError) -> Self {
        Self::Timestamp(value)
    }
}

impl std::fmt::Display for PlantTimingValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported plant-step timing schema version {found}")
            }
            Self::Timestamp(error) => write!(f, "invalid plant-step timestamp: {error}"),
            Self::ZeroObservedTransition => write!(f, "plant transition duration must be non-zero"),
            Self::ZeroRequestedDuration => write!(f, "requested plant-step duration must be non-zero"),
            Self::ZeroBackendEffectiveDuration => {
                write!(f, "backend effective plant-step duration must be non-zero")
            }
            Self::InvalidIdentifier(field) => write!(f, "invalid {field} identifier"),
        }
    }
}

impl std::error::Error for PlantTimingValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_evidence::{ClockDomainId, EvidenceValidationError};

    fn ts(domain: &str, ns: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new(domain).unwrap(), ns)
    }

    #[test]
    fn same_clock_transition_reports_observed_duration() {
        let timing = PlantStepTimingV1::new(
            ts("sim.model", 1_000),
            ts("sim.model", 3_500),
            PlantTimingSourceV1::SimulatorModelClock,
            "simple-quadrotor.v1",
        )
        .unwrap();

        assert_eq!(timing.observed_transition_ns().unwrap(), 2_500);
    }

    #[test]
    fn requested_and_effective_duration_are_not_conflated() {
        let mut timing = PlantStepTimingV1::new(
            ts("sim.model", 0),
            ts("sim.model", 2_100_000),
            PlantTimingSourceV1::SimulatorModelClock,
            "mujoco.profile.v1",
        )
        .unwrap();
        timing.requested_step_ns = Some(2_000_000);
        timing.backend_effective_step_ns = Some(2_100_000);
        timing.validate().unwrap();

        assert_eq!(timing.backend_step_error_ns(), Some(100_000));
        assert_eq!(timing.observed_transition_ns().unwrap(), 2_100_000);
    }

    #[test]
    fn cross_clock_transition_is_rejected() {
        let error = PlantStepTimingV1::new(
            ts("host.monotonic", 1),
            ts("device.boot", 2),
            PlantTimingSourceV1::DeviceMonotonicClock,
            "hardware.v1",
        )
        .unwrap_err();

        assert_eq!(
            error,
            PlantTimingValidationError::Timestamp(EvidenceValidationError::ClockDomainMismatch)
        );
    }

    #[test]
    fn backward_and_zero_time_are_rejected() {
        let backwards = PlantStepTimingV1::new(
            ts("sim.model", 10),
            ts("sim.model", 9),
            PlantTimingSourceV1::SimulatorModelClock,
            "sim.v1",
        )
        .unwrap_err();
        assert_eq!(
            backwards,
            PlantTimingValidationError::Timestamp(EvidenceValidationError::NonMonotonicTimestamp)
        );

        let zero = PlantStepTimingV1::new(
            ts("sim.model", 10),
            ts("sim.model", 10),
            PlantTimingSourceV1::SimulatorModelClock,
            "sim.v1",
        )
        .unwrap_err();
        assert_eq!(zero, PlantTimingValidationError::ZeroObservedTransition);
    }

    #[test]
    fn zero_requested_or_backend_duration_is_rejected() {
        let mut timing = PlantStepTimingV1::new(
            ts("sim.model", 1),
            ts("sim.model", 2),
            PlantTimingSourceV1::SimulatorModelClock,
            "sim.v1",
        )
        .unwrap();

        timing.requested_step_ns = Some(0);
        assert_eq!(
            timing.validate(),
            Err(PlantTimingValidationError::ZeroRequestedDuration)
        );

        timing.requested_step_ns = None;
        timing.backend_effective_step_ns = Some(0);
        assert_eq!(
            timing.validate(),
            Err(PlantTimingValidationError::ZeroBackendEffectiveDuration)
        );
    }

    #[test]
    fn cognitive_and_scheduler_ids_do_not_define_physical_time() {
        let mut timing = PlantStepTimingV1::new(
            ts("sim.model", 100),
            ts("sim.model", 200),
            PlantTimingSourceV1::SimulatorModelClock,
            "sim.v1",
        )
        .unwrap();
        timing.scheduler_cycle_id = Some(99);
        timing.cognitive_cycle_id = Some(7);

        assert_eq!(timing.observed_transition_ns().unwrap(), 100);
        timing.scheduler_cycle_id = Some(1_000_000);
        timing.cognitive_cycle_id = Some(42_000);
        assert_eq!(timing.observed_transition_ns().unwrap(), 100);
    }
}
