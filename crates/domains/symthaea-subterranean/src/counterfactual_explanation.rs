// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Counterfactual "why not this instead?" evaluation.
//!
//! `operator_challenge.rs`'s `ChallengeKind::WhyNotAlternative` needs a
//! real answer, not a stored opinion that could quietly drift from what
//! the vehicle actually does. [`explain_counterfactual`] answers it
//! honestly: it re-runs the proposed alternative command through the same
//! [`crate::invariant_monitor::RuntimeInvariantMonitor`] enforcement the
//! real command path uses, under the same safety context, and reports
//! whether -- and why -- that alternative would have been rejected.

use crate::actuator_isolation::ActuatorIsolationReport;
use crate::capability_profile::CapabilityDisposition;
use crate::embodiment::MotorSafetyLevel;
use crate::invariant_monitor::{InvariantContext, RuntimeInvariant, RuntimeInvariantMonitor};
use crate::safety::SubterraneanHazard;
use crate::types::{SubterraneanCommand, SubterraneanState};
use serde::{Deserialize, Serialize};

/// The single actuator a counterfactual question proposes changing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CounterfactualActuator {
    CutterHead,
    AugerFeed,
    LeftTrack,
    RightTrack,
    BallastTrim,
    ThermalPump,
}

impl CounterfactualActuator {
    pub const fn label(self) -> &'static str {
        match self {
            Self::CutterHead => "cutter_head",
            Self::AugerFeed => "auger_feed",
            Self::LeftTrack => "left_track",
            Self::RightTrack => "right_track",
            Self::BallastTrim => "ballast_trim",
            Self::ThermalPump => "thermal_pump",
        }
    }

    fn apply(self, command: &mut SubterraneanCommand, value: f32) {
        match self {
            Self::CutterHead => command.set_cutter_head(value),
            Self::AugerFeed => command.set_auger_feed(value),
            Self::LeftTrack => command.set_left_track(value),
            Self::RightTrack => command.set_right_track(value),
            Self::BallastTrim => command.set_ballast_trim(value),
            Self::ThermalPump => command.set_thermal_pump(value),
        }
    }
}

/// An operator's "why didn't you do this instead?" question about one
/// recorded decision.
///
/// Deliberately integer (not `f32`/`f64`): `SubterraneanCommand` is
/// float-valued and therefore not `Eq`, but `operator_challenge.rs`'s
/// `ChallengeEnvelope` (which embeds `Option<CounterfactualQuestion>`)
/// derives `Eq` for its replay-resistant ledger key, so this type must
/// too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualQuestion {
    pub actuator: CounterfactualActuator,
    /// Proposed authority as integer percent of full range, `-100..=100`.
    pub proposed_percent: i8,
    /// The recorded decision step this question is asked about.
    pub decision_step: u64,
}

impl CounterfactualQuestion {
    /// The proposed value as the `f32` `SubterraneanCommand` expects,
    /// clamped to the valid `-1.0..=1.0` range regardless of the raw
    /// integer's bounds.
    pub fn proposed_value(self) -> f32 {
        (self.proposed_percent as f32 / 100.0).clamp(-1.0, 1.0)
    }
}

/// The system's answer to a [`CounterfactualQuestion`]: what actually
/// happens when the proposed alternative is run through real enforcement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualAnswer {
    pub question: CounterfactualQuestion,
    pub would_have_been_rejected: bool,
    pub violated_invariants: Vec<RuntimeInvariant>,
}

/// Re-evaluate a proposed alternative command against the same safety
/// context (hazard, tunnel conflict, capability disposition, actuator
/// isolation) that governed the actual recorded decision.
///
/// This is the whole point of the module: the answer comes from actually
/// running the alternative through the real enforcement pipeline, not
/// from an independently reasoned judgment that could silently diverge
/// from what the vehicle would really do.
#[allow(clippy::too_many_arguments)]
pub fn explain_counterfactual(
    question: CounterfactualQuestion,
    state: &SubterraneanState,
    safety_level: MotorSafetyLevel,
    primary_hazard: SubterraneanHazard,
    tunnel_conflict: bool,
    return_feasible: bool,
    capability_disposition: CapabilityDisposition,
    actuator_isolation: ActuatorIsolationReport,
) -> CounterfactualAnswer {
    let mut command = SubterraneanCommand::zero();
    question
        .actuator
        .apply(&mut command, question.proposed_value());
    let (_, assessment) = RuntimeInvariantMonitor::default().enforce(
        command,
        InvariantContext {
            state,
            safety_level,
            primary_hazard,
            tunnel_conflict,
            return_feasible,
            capability_disposition,
            actuator_isolation,
        },
    );
    CounterfactualAnswer {
        question,
        would_have_been_rejected: !assessment.passed(),
        violated_invariants: assessment.violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal_context() -> (
        SubterraneanState,
        MotorSafetyLevel,
        SubterraneanHazard,
        bool,
        bool,
        CapabilityDisposition,
        ActuatorIsolationReport,
    ) {
        (
            SubterraneanState::home(),
            MotorSafetyLevel::Green,
            SubterraneanHazard::None,
            false,
            true,
            CapabilityDisposition::FullMission,
            ActuatorIsolationReport::nominal(),
        )
    }

    #[test]
    fn safe_alternative_under_nominal_conditions_is_accepted() {
        let (state, safety_level, hazard, conflict, feasible, disposition, isolation) =
            nominal_context();
        let question = CounterfactualQuestion {
            actuator: CounterfactualActuator::CutterHead,
            proposed_percent: 50,
            decision_step: 10,
        };
        let answer = explain_counterfactual(
            question,
            &state,
            safety_level,
            hazard,
            conflict,
            feasible,
            disposition,
            isolation,
        );
        assert!(!answer.would_have_been_rejected);
        assert!(answer.violated_invariants.is_empty());
    }

    #[test]
    fn productive_alternative_under_red_tier_would_have_been_rejected() {
        let (state, _, hazard, conflict, feasible, disposition, isolation) = nominal_context();
        let question = CounterfactualQuestion {
            actuator: CounterfactualActuator::CutterHead,
            proposed_percent: 80,
            decision_step: 10,
        };
        let answer = explain_counterfactual(
            question,
            &state,
            MotorSafetyLevel::Red,
            hazard,
            conflict,
            feasible,
            disposition,
            isolation,
        );
        assert!(answer.would_have_been_rejected);
        assert!(
            answer
                .violated_invariants
                .contains(&RuntimeInvariant::RedTierRemovesProductiveWork)
        );
    }

    #[test]
    fn motion_alternative_during_tunnel_conflict_would_have_been_rejected() {
        let (state, safety_level, hazard, _, feasible, disposition, isolation) = nominal_context();
        let question = CounterfactualQuestion {
            actuator: CounterfactualActuator::LeftTrack,
            proposed_percent: 60,
            decision_step: 10,
        };
        let answer = explain_counterfactual(
            question,
            &state,
            safety_level,
            hazard,
            true,
            feasible,
            disposition,
            isolation,
        );
        assert!(answer.would_have_been_rejected);
        assert!(
            answer
                .violated_invariants
                .contains(&RuntimeInvariant::TunnelConflictStopsMotion)
        );
    }

    #[test]
    fn proposed_value_clamps_to_valid_range() {
        let question = CounterfactualQuestion {
            actuator: CounterfactualActuator::CutterHead,
            proposed_percent: 100,
            decision_step: 0,
        };
        assert_eq!(question.proposed_value(), 1.0);
        let negative = CounterfactualQuestion {
            proposed_percent: -100,
            ..question
        };
        assert_eq!(negative.proposed_value(), -1.0);
    }
}
