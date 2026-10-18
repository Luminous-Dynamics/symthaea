//! Deterministic synthetic perturbations for orbital-servicing simulation.
//!
//! These are scenario inputs, not a validated debris-impact, communications,
//! radiation, power, or thermal environment model.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrbitalPerturbation {
    /// Apply an instantaneous exogenous velocity increment after the nominated
    /// simulator step. This is a delta-v proxy for a collision disturbance,
    /// not a finite-element/contact model of debris impact. Units: m/s in the
    /// spacecraft inertial frame.
    DebrisImpact {
        delta_v_m_s: [f64; 3],
        at_step: usize,
    },
    /// Suppress the communications link for `duration` simulator steps,
    /// beginning at `at_step` (inclusive). This affects link availability,
    /// not geometric line-of-sight.
    CommBlackout {
        duration: usize,
        at_step: usize,
    },
    /// Reserved for a later explicit power/thermal model. Until that model
    /// exists this event is rejected; it must not be represented by altering
    /// geometric eclipse/solar-exposure state.
    SolarFlare {
        power_reduction: f64,
        at_step: usize,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerturbationSchedule {
    /// Stable caller-provided order is the execution order for events sharing
    /// a step. Keep this order in the experiment manifest.
    pub perturbations: Vec<OrbitalPerturbation>,
}

impl PerturbationSchedule {
    /// Validate the complete schedule before changing simulator state.
    ///
    /// Unsupported event types fail closed even when scheduled in the future,
    /// so a campaign cannot silently omit a declared event.
    pub fn validate(&self) -> Result<(), PerturbationError> {
        for perturbation in &self.perturbations {
            match perturbation {
                OrbitalPerturbation::DebrisImpact { delta_v_m_s, at_step } => {
                    if delta_v_m_s.iter().any(|value| !value.is_finite()) {
                        return Err(PerturbationError::NonFiniteImpactDeltaV {
                            at_step: *at_step,
                        });
                    }
                }
                OrbitalPerturbation::CommBlackout { duration, at_step } => {
                    if *duration == 0 {
                        return Err(PerturbationError::ZeroDurationBlackout {
                            at_step: *at_step,
                        });
                    }
                    if at_step.checked_add(*duration).is_none() {
                        return Err(PerturbationError::BlackoutEndOverflow {
                            at_step: *at_step,
                            duration: *duration,
                        });
                    }
                }
                OrbitalPerturbation::SolarFlare { at_step, .. } => {
                    return Err(PerturbationError::UnsupportedSolarFlare {
                        at_step: *at_step,
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PerturbationReceipt {
    /// The full external delta-v proxy was applied; no spacecraft propellant
    /// budget is charged for an exogenous disturbance.
    DebrisImpactApplied {
        at_step: usize,
        requested_delta_v_m_s: [f64; 3],
        applied_delta_v_m_s: [f64; 3],
    },
    CommBlackoutStarted {
        at_step: usize,
        duration_steps: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerturbationStepReport {
    /// Zero-based simulator step that was just propagated.
    pub step_index: usize,
    /// Start/application receipts for events first triggered on this step.
    pub receipts: Vec<PerturbationReceipt>,
    /// True if a scheduled communications blackout is active for this step.
    /// This is distinct from the geometric line-of-sight result.
    pub communication_suppressed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerturbationError {
    InvalidTimeStep,
    UnexpectedStepIndex { expected: usize, actual: usize },
    ChangedScheduleRequiresReset,
    MissedScheduledEvent {
        event_index: usize,
        scheduled_step: usize,
        actual_step: usize,
    },
    NonFiniteImpactDeltaV { at_step: usize },
    ZeroDurationBlackout { at_step: usize },
    BlackoutEndOverflow { at_step: usize, duration: usize },
    UnsupportedSolarFlare { at_step: usize },
}

impl std::fmt::Display for PerturbationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTimeStep => write!(f, "time step must be finite and positive"),
            Self::UnexpectedStepIndex { expected, actual } => write!(
                f,
                "perturbation step index mismatch: expected {expected}, received {actual}"
            ),
            Self::ChangedScheduleRequiresReset => write!(
                f,
                "perturbation schedule changed during a campaign; reset the simulator before starting a new schedule"
            ),
            Self::MissedScheduledEvent { event_index, scheduled_step, actual_step } => write!(
                f,
                "scheduled event {event_index} at step {scheduled_step} was missed before campaign start at step {actual_step}"
            ),
            Self::NonFiniteImpactDeltaV { at_step } => {
                write!(f, "non-finite impact delta-v at step {at_step}")
            }
            Self::ZeroDurationBlackout { at_step } => {
                write!(f, "communications blackout at step {at_step} has zero duration")
            }
            Self::BlackoutEndOverflow { at_step, duration } => write!(
                f,
                "communications blackout end overflows usize (step {at_step}, duration {duration})"
            ),
            Self::UnsupportedSolarFlare { at_step } => write!(
                f,
                "solar-flare perturbation at step {at_step} is unsupported until a power/thermal model exists"
            ),
        }
    }
}

impl std::error::Error for PerturbationError {}
