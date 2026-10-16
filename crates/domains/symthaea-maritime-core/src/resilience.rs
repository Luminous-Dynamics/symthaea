// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, mission-neutral progressive-failure recording.
//!
//! This module intentionally does not collapse safety into a single score. It records a small
//! set of independent utility axes and checks monotonic invariants for experiments where faults
//! are only accumulated. Domain-specific simulators decide how a perturbation changes each axis.

use crate::OperatingEnvelope;
use serde::{Deserialize, Serialize};

/// Coarse availability for one mission-neutral utility axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UtilityAvailability {
    Unavailable,
    Degraded,
    Available,
}

impl UtilityAvailability {
    const fn diagnostic_units(self) -> u8 {
        match self {
            Self::Unavailable => 0,
            Self::Degraded => 1,
            Self::Available => 2,
        }
    }
}

/// One observation in a resilience campaign.
///
/// The axes deliberately stay separate. `diagnostic_retained_units` is suitable for plots and
/// regression comparison only; it is not an admission or physical-safety gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualUtilitySnapshot {
    pub safe_navigation: UtilityAvailability,
    pub observation: UtilityAvailability,
    pub communications: UtilityAvailability,
    pub recoverability: UtilityAvailability,
    pub operator_support: UtilityAvailability,
    pub energy_remaining_fraction: f32,
    pub full_fleet_admission: bool,
    pub envelope: OperatingEnvelope,
}

impl ResidualUtilitySnapshot {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.energy_remaining_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.energy_remaining_fraction)
        {
            return Err("energy_remaining_fraction must be finite and within [0, 1]");
        }
        Ok(())
    }

    /// Diagnostic only: five independent availability axes, each 0/1/2.
    pub fn diagnostic_retained_units(&self) -> u8 {
        self.safe_navigation.diagnostic_units()
            + self.observation.diagnostic_units()
            + self.communications.diagnostic_units()
            + self.recoverability.diagnostic_units()
            + self.operator_support.diagnostic_units()
    }

    /// True when any independent utility axis becomes more available.
    ///
    /// This intentionally does not use the aggregate diagnostic units: an improvement
    /// on one axis must not be hidden by an equal degradation on another axis during an
    /// accumulating-failure experiment.
    fn any_axis_improved_over(&self, previous: &Self) -> bool {
        self.safe_navigation > previous.safe_navigation
            || self.observation > previous.observation
            || self.communications > previous.communications
            || self.recoverability > previous.recoverability
            || self.operator_support > previous.operator_support
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureObservation {
    pub label: String,
    pub utility: ResidualUtilitySnapshot,
}

/// Ordered observations from a campaign where failures only accumulate.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProgressiveFailureTrace {
    pub observations: Vec<FailureObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressiveFailureViolation {
    EmptyTrace,
    InvalidEnergy { step: usize },
    EnvelopeExpanded { step: usize },
    UtilityImproved { step: usize },
    EnergyIncreased { step: usize },
    FleetReadmitted { step: usize },
}

impl ProgressiveFailureTrace {
    /// Validate invariants for an *accumulating-failure* experiment.
    ///
    /// A separate recovery experiment should start a new trace rather than weakening these
    /// checks. This makes accidental authority/capability expansion during fault accumulation
    /// visible instead of averaging it away.
    pub fn validate_accumulating_failures(&self) -> Result<(), ProgressiveFailureViolation> {
        let Some(first) = self.observations.first() else {
            return Err(ProgressiveFailureViolation::EmptyTrace);
        };
        if first.utility.validate().is_err() {
            return Err(ProgressiveFailureViolation::InvalidEnergy { step: 0 });
        }

        for (index, pair) in self.observations.windows(2).enumerate() {
            let previous = &pair[0].utility;
            let current = &pair[1].utility;
            let step = index + 1;

            if current.validate().is_err() {
                return Err(ProgressiveFailureViolation::InvalidEnergy { step });
            }
            if current.envelope < previous.envelope {
                return Err(ProgressiveFailureViolation::EnvelopeExpanded { step });
            }
            if current.any_axis_improved_over(previous) {
                return Err(ProgressiveFailureViolation::UtilityImproved { step });
            }
            if current.energy_remaining_fraction > previous.energy_remaining_fraction {
                return Err(ProgressiveFailureViolation::EnergyIncreased { step });
            }
            if !previous.full_fleet_admission && current.full_fleet_admission {
                return Err(ProgressiveFailureViolation::FleetReadmitted { step });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        envelope: OperatingEnvelope,
        nav: UtilityAvailability,
        observation: UtilityAvailability,
        comms: UtilityAvailability,
        recovery: UtilityAvailability,
        operator: UtilityAvailability,
        energy: f32,
        admitted: bool,
    ) -> ResidualUtilitySnapshot {
        ResidualUtilitySnapshot {
            safe_navigation: nav,
            observation,
            communications: comms,
            recoverability: recovery,
            operator_support: operator,
            energy_remaining_fraction: energy,
            full_fleet_admission: admitted,
            envelope,
        }
    }

    #[test]
    fn progressive_failure_campaign_degrades_without_authority_expansion() {
        use UtilityAvailability::{Available, Degraded, Unavailable};

        let trace = ProgressiveFailureTrace {
            observations: vec![
                FailureObservation {
                    label: "baseline".into(),
                    utility: snapshot(
                        OperatingEnvelope::Normal,
                        Available,
                        Available,
                        Available,
                        Available,
                        Available,
                        0.90,
                        true,
                    ),
                },
                FailureObservation {
                    label: "fleet-link partition".into(),
                    utility: snapshot(
                        OperatingEnvelope::ReducedCapability,
                        Available,
                        Available,
                        Degraded,
                        Available,
                        Available,
                        0.88,
                        false,
                    ),
                },
                FailureObservation {
                    label: "trusted-time loss".into(),
                    utility: snapshot(
                        OperatingEnvelope::SafeTransit,
                        Available,
                        Available,
                        Degraded,
                        Available,
                        Available,
                        0.86,
                        false,
                    ),
                },
                FailureObservation {
                    label: "external-positioning loss / dead reckoning".into(),
                    utility: snapshot(
                        OperatingEnvelope::SafeTransit,
                        Degraded,
                        Available,
                        Degraded,
                        Available,
                        Available,
                        0.83,
                        false,
                    ),
                },
                FailureObservation {
                    label: "stale generation".into(),
                    utility: snapshot(
                        OperatingEnvelope::SafeTransit,
                        Degraded,
                        Available,
                        Degraded,
                        Available,
                        Available,
                        0.81,
                        false,
                    ),
                },
                FailureObservation {
                    label: "missing health evidence".into(),
                    utility: snapshot(
                        OperatingEnvelope::SafeTransit,
                        Degraded,
                        Degraded,
                        Degraded,
                        Available,
                        Available,
                        0.78,
                        false,
                    ),
                },
                FailureObservation {
                    label: "remote-operator loss".into(),
                    utility: snapshot(
                        OperatingEnvelope::SafeTransit,
                        Degraded,
                        Degraded,
                        Degraded,
                        Available,
                        Unavailable,
                        0.75,
                        false,
                    ),
                },
                FailureObservation {
                    label: "critical local component fault".into(),
                    utility: snapshot(
                        OperatingEnvelope::RecoverOrSurface,
                        Degraded,
                        Unavailable,
                        Degraded,
                        Degraded,
                        Unavailable,
                        0.70,
                        false,
                    ),
                },
            ],
        };

        assert_eq!(trace.validate_accumulating_failures(), Ok(()));
    }

    #[test]
    fn progressive_failure_trace_rejects_silent_capability_recovery() {
        use UtilityAvailability::{Available, Degraded};
        let trace = ProgressiveFailureTrace {
            observations: vec![
                FailureObservation {
                    label: "partition".into(),
                    utility: snapshot(
                        OperatingEnvelope::ReducedCapability,
                        Available,
                        Available,
                        Degraded,
                        Available,
                        Available,
                        0.8,
                        false,
                    ),
                },
                FailureObservation {
                    label: "more failures".into(),
                    utility: snapshot(
                        OperatingEnvelope::Normal,
                        Available,
                        Available,
                        Available,
                        Available,
                        Available,
                        0.9,
                        true,
                    ),
                },
            ],
        };
        assert!(matches!(
            trace.validate_accumulating_failures(),
            Err(ProgressiveFailureViolation::EnvelopeExpanded { step: 1 })
        ));
    }

    #[test]
    fn aggregate_tradeoff_cannot_hide_one_axis_improving() {
        use UtilityAvailability::{Available, Degraded};

        // Aggregate diagnostic units are equal (8 -> 8), but navigation improves
        // while observation degrades. Accumulating failures must reject that hidden
        // capability expansion rather than treating the axes as fungible.
        let trace = ProgressiveFailureTrace {
            observations: vec![
                FailureObservation {
                    label: "before".into(),
                    utility: snapshot(
                        OperatingEnvelope::ReducedCapability,
                        Degraded,
                        Available,
                        Available,
                        Available,
                        Available,
                        0.8,
                        false,
                    ),
                },
                FailureObservation {
                    label: "after".into(),
                    utility: snapshot(
                        OperatingEnvelope::ReducedCapability,
                        Available,
                        Degraded,
                        Available,
                        Available,
                        Available,
                        0.79,
                        false,
                    ),
                },
            ],
        };

        assert_eq!(
            trace.observations[0].utility.diagnostic_retained_units(),
            trace.observations[1].utility.diagnostic_retained_units()
        );
        assert!(matches!(
            trace.validate_accumulating_failures(),
            Err(ProgressiveFailureViolation::UtilityImproved { step: 1 })
        ));
    }
}
