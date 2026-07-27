// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composite temporal and causal assurance authority.
//!
//! The supervisor cannot create actuator authority. It constrains nominal
//! commands when clocks, observation age, event order, plan freshness, or
//! command-response attribution no longer support immediate control.

use crate::causal_attribution::{
    AttributionDisposition, CausalAttributionLedger, CommandCause, ResponseObservation,
};
use crate::delayed_observation::{
    DelayedObservationSupervisor, ObservationAgeDisposition, ObservationBatchAssessment,
    TimedObservation,
};
use crate::embodiment::MotorSafetyLevel;
use crate::mission::SubterraneanMissionIntent;
use crate::plan_freshness::{
    PlanBasis, PlanFreshnessAssessment, PlanFreshnessSupervisor, RuntimeRevisions,
};
use crate::temporal_clock::{
    ClockAssessment, ClockDisposition, ClockSample, TemporalClockSupervisor,
};
use crate::temporal_event::{CausalEvent, CausalEventLedger, EventAppendError, EventOrdering};
use crate::types::SubterraneanCommand;
use serde::{Deserialize, Serialize};

pub const TEMPORAL_ASSURANCE_SCHEMA_VERSION: u16 = 1;
pub const TEMPORAL_RUNTIME_FRAME_SCHEMA_VERSION: u16 = 1;
pub const MAX_TEMPORAL_REASONS: usize = 16;
pub const TEMPORAL_REVIEW_CLEAN_DWELL_STEPS: u32 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TemporalAuthority {
    Nominal,
    ProbeOnly,
    ReturnOnly,
    HoldForReview,
}

impl TemporalAuthority {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::ProbeOnly => "probe_only",
            Self::ReturnOnly => "return_only",
            Self::HoldForReview => "hold_for_review",
        }
    }

    /// A broken clock or unattributable command/response history means the
    /// platform cannot trust its own recent history, which is a strictly
    /// worse position than merely losing comms (`DegradedMode::AutonomousReturn`,
    /// which floors at Yellow) -- so `ReturnOnly` floors at Orange and
    /// `HoldForReview` at Red, rather than mirroring that lighter mapping.
    pub const fn safety_floor(self) -> Option<MotorSafetyLevel> {
        match self {
            Self::Nominal => None,
            Self::ProbeOnly => Some(MotorSafetyLevel::Yellow),
            Self::ReturnOnly => Some(MotorSafetyLevel::Orange),
            Self::HoldForReview => Some(MotorSafetyLevel::Red),
        }
    }

    pub const fn mission_override(self) -> Option<SubterraneanMissionIntent> {
        match self {
            Self::Nominal | Self::ProbeOnly => None,
            Self::ReturnOnly => Some(SubterraneanMissionIntent::ReturnHome),
            Self::HoldForReview => Some(SubterraneanMissionIntent::HoldPosition),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemporalRuntimeFrame {
    pub schema_version: u16,
    pub clock_samples: Vec<ClockSample>,
    pub observations: Vec<TimedObservation>,
    pub events: Vec<CausalEvent>,
    pub plan: Option<PlanBasis>,
    pub causes: Vec<CommandCause>,
    pub responses: Vec<ResponseObservation>,
}

impl TemporalRuntimeFrame {
    pub fn validate(&self) -> bool {
        self.schema_version == TEMPORAL_RUNTIME_FRAME_SCHEMA_VERSION
            && self.clock_samples.len() <= 16
            && self.observations.len() <= 32
            && self.events.len() <= 32
            && self.causes.len() <= 32
            && self.responses.len() <= 32
            && self
                .observations
                .iter()
                .copied()
                .all(TimedObservation::validate)
            && self.events.iter().all(CausalEvent::validate)
            && self.causes.iter().copied().all(CommandCause::validate)
            && self
                .responses
                .iter()
                .copied()
                .all(ResponseObservation::validate)
            && self.plan.is_none_or(PlanBasis::validate)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalAssuranceAssessment {
    pub authority: TemporalAuthority,
    pub control_time_ns: u64,
    pub worst_clock: ClockAssessment,
    pub observation_timing: ObservationBatchAssessment,
    pub plan: PlanFreshnessAssessment,
    pub late_events: usize,
    pub concurrent_events: usize,
    pub rejected_events: usize,
    pub causal_contradictions: usize,
    pub causal_ambiguities: usize,
    pub hold_latched: bool,
    pub return_feasible: bool,
    pub reasons: Vec<String>,
}

impl TemporalAssuranceAssessment {
    pub fn nominal() -> Self {
        Self {
            authority: TemporalAuthority::Nominal,
            control_time_ns: 0,
            worst_clock: ClockAssessment::nominal(),
            observation_timing: ObservationBatchAssessment::nominal(),
            plan: PlanFreshnessAssessment::nominal(),
            late_events: 0,
            concurrent_events: 0,
            rejected_events: 0,
            causal_contradictions: 0,
            causal_ambiguities: 0,
            hold_latched: false,
            return_feasible: true,
            reasons: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAssuranceSupervisor {
    schema_version: u16,
    control_time_ns: u64,
    clock: TemporalClockSupervisor,
    observations: DelayedObservationSupervisor,
    events: CausalEventLedger,
    plans: PlanFreshnessSupervisor,
    attribution: CausalAttributionLedger,
    hold_latched: bool,
    clean_dwell_steps: u32,
    total_assessments: u64,
    last: TemporalAssuranceAssessment,
}

impl Default for TemporalAssuranceSupervisor {
    fn default() -> Self {
        Self {
            schema_version: TEMPORAL_ASSURANCE_SCHEMA_VERSION,
            control_time_ns: 0,
            clock: TemporalClockSupervisor::default(),
            observations: DelayedObservationSupervisor::default(),
            events: CausalEventLedger::default(),
            plans: PlanFreshnessSupervisor::default(),
            attribution: CausalAttributionLedger::default(),
            hold_latched: false,
            clean_dwell_steps: 0,
            total_assessments: 0,
            last: TemporalAssuranceAssessment::nominal(),
        }
    }
}

impl TemporalAssuranceSupervisor {
    pub fn validate(&self) -> bool {
        self.schema_version == TEMPORAL_ASSURANCE_SCHEMA_VERSION
            && self.clock.validate()
            && self.observations.validate()
            && self.events.validate()
            && self.plans.validate()
            && self.attribution.validate()
            && self.clean_dwell_steps <= TEMPORAL_REVIEW_CLEAN_DWELL_STEPS
            && self.last.reasons.len() <= MAX_TEMPORAL_REASONS
    }

    pub fn assess(
        &mut self,
        dt_seconds: f32,
        current_step: u64,
        revisions: RuntimeRevisions,
        frame: &TemporalRuntimeFrame,
        return_feasible: bool,
        at_safe_service_location: bool,
    ) -> TemporalAssuranceAssessment {
        self.total_assessments = self.total_assessments.saturating_add(1);
        let mut reasons = Vec::new();
        let valid_dt = dt_seconds.is_finite() && dt_seconds > 0.0 && dt_seconds <= 1.0;
        if valid_dt {
            let increment = (f64::from(dt_seconds) * 1_000_000_000.0).round();
            self.control_time_ns = self
                .control_time_ns
                .saturating_add(increment.max(1.0) as u64);
        }

        let mut authority = if valid_dt {
            TemporalAuthority::Nominal
        } else {
            reasons.push("invalid_control_interval".to_string());
            TemporalAuthority::HoldForReview
        };
        if !frame.validate() {
            promote(
                &mut authority,
                TemporalAuthority::HoldForReview,
                &mut reasons,
                "malformed_temporal_frame",
            );
        }

        let mut worst_clock = ClockAssessment::nominal();
        for sample in frame.clock_samples.iter().copied() {
            let assessment = self.clock.observe(self.control_time_ns, sample);
            if clock_rank(assessment.disposition) > clock_rank(worst_clock.disposition) {
                worst_clock = assessment;
            }
            match assessment.disposition {
                ClockDisposition::Accepted => {}
                ClockDisposition::Degraded => promote(
                    &mut authority,
                    TemporalAuthority::ProbeOnly,
                    &mut reasons,
                    assessment.issue.label(),
                ),
                ClockDisposition::Rejected => promote(
                    &mut authority,
                    TemporalAuthority::HoldForReview,
                    &mut reasons,
                    assessment.issue.label(),
                ),
            }
        }
        if frame.clock_samples.is_empty() {
            promote(
                &mut authority,
                TemporalAuthority::HoldForReview,
                &mut reasons,
                "missing_clock_sample",
            );
        }

        let observation_timing = self
            .observations
            .assess_batch(self.control_time_ns, &frame.observations);
        if !observation_timing.immediate_control_complete {
            promote(
                &mut authority,
                if return_feasible {
                    TemporalAuthority::ReturnOnly
                } else {
                    TemporalAuthority::HoldForReview
                },
                &mut reasons,
                "immediate_control_observation_not_fresh",
            );
        } else if observation_timing.worst >= ObservationAgeDisposition::Degraded {
            promote(
                &mut authority,
                TemporalAuthority::ProbeOnly,
                &mut reasons,
                "observation_age_degraded",
            );
        }

        if let Some(plan) = frame.plan {
            if !self.plans.install(plan) {
                promote(
                    &mut authority,
                    TemporalAuthority::HoldForReview,
                    &mut reasons,
                    "malformed_plan_basis",
                );
            }
        }
        let plan = self.plans.assess(current_step, revisions);
        if !plan.current {
            promote(
                &mut authority,
                if return_feasible {
                    TemporalAuthority::ReturnOnly
                } else {
                    TemporalAuthority::HoldForReview
                },
                &mut reasons,
                format!("plan:{}", plan.reason.label()),
            );
        } else if !plan.work_authorized {
            promote(
                &mut authority,
                TemporalAuthority::ProbeOnly,
                &mut reasons,
                "plan_does_not_authorize_work",
            );
        }

        let mut late_events = 0usize;
        let mut concurrent_events = 0usize;
        let mut rejected_events = 0usize;
        let mut causal_contradictions = 0usize;
        for event in frame.events.iter().cloned() {
            match self.events.append(event) {
                Ok(assessment) => match assessment.ordering {
                    EventOrdering::Ordered => {}
                    EventOrdering::Concurrent => concurrent_events += 1,
                    EventOrdering::Late => late_events += 1,
                },
                Err(EventAppendError::DependencyContradiction) => {
                    rejected_events += 1;
                    causal_contradictions += 1;
                }
                Err(_) => rejected_events += 1,
            }
        }
        if causal_contradictions > 0 {
            promote(
                &mut authority,
                TemporalAuthority::HoldForReview,
                &mut reasons,
                "causal_dependency_contradiction",
            );
        } else if rejected_events > 0 {
            promote(
                &mut authority,
                TemporalAuthority::ReturnOnly,
                &mut reasons,
                "rejected_causal_event",
            );
        } else if late_events > 0 || concurrent_events > 0 {
            promote(
                &mut authority,
                TemporalAuthority::ProbeOnly,
                &mut reasons,
                "ambiguous_event_order",
            );
        }

        for cause in frame.causes.iter().copied() {
            if !self.attribution.register_cause(cause) {
                promote(
                    &mut authority,
                    TemporalAuthority::ReturnOnly,
                    &mut reasons,
                    "rejected_command_cause",
                );
            }
        }
        let mut causal_ambiguities = 0usize;
        for response in frame.responses.iter().copied() {
            match self.attribution.attribute(response).disposition {
                AttributionDisposition::Supported | AttributionDisposition::Unattributed => {}
                AttributionDisposition::Ambiguous => causal_ambiguities += 1,
                AttributionDisposition::Contradicted => causal_contradictions += 1,
            }
        }
        if causal_contradictions > 0 {
            promote(
                &mut authority,
                TemporalAuthority::HoldForReview,
                &mut reasons,
                "command_response_contradiction",
            );
        } else if causal_ambiguities > 0 {
            promote(
                &mut authority,
                TemporalAuthority::ProbeOnly,
                &mut reasons,
                "command_response_ambiguous",
            );
        }

        if authority == TemporalAuthority::HoldForReview {
            self.hold_latched = true;
            self.clean_dwell_steps = 0;
        } else if self.hold_latched {
            if authority == TemporalAuthority::Nominal && at_safe_service_location {
                self.clean_dwell_steps = self.clean_dwell_steps.saturating_add(1);
                if self.clean_dwell_steps >= TEMPORAL_REVIEW_CLEAN_DWELL_STEPS {
                    self.hold_latched = false;
                    self.clean_dwell_steps = 0;
                }
            } else {
                self.clean_dwell_steps = 0;
            }
        }
        if self.hold_latched {
            promote(
                &mut authority,
                TemporalAuthority::HoldForReview,
                &mut reasons,
                "temporal_review_hold_latched",
            );
        }
        reasons.truncate(MAX_TEMPORAL_REASONS);
        self.last = TemporalAssuranceAssessment {
            authority,
            control_time_ns: self.control_time_ns,
            worst_clock,
            observation_timing,
            plan,
            late_events,
            concurrent_events,
            rejected_events,
            causal_contradictions,
            causal_ambiguities,
            hold_latched: self.hold_latched,
            return_feasible,
            reasons,
        };
        self.last.clone()
    }

    pub fn constrain_command(&self, mut command: SubterraneanCommand) -> SubterraneanCommand {
        match self.last.authority {
            TemporalAuthority::Nominal => {}
            TemporalAuthority::ProbeOnly => {
                command.set_cutter_head(command.cutter_head().clamp(-0.12, 0.12));
                command.set_auger_feed(command.auger_feed().clamp(-0.1, 0.1));
                command.set_left_track(command.left_track().clamp(-0.25, 0.25));
                command.set_right_track(command.right_track().clamp(-0.25, 0.25));
            }
            TemporalAuthority::ReturnOnly => {
                command.set_cutter_head(0.0);
                command.set_auger_feed(0.0);
                command.set_ballast_trim(0.0);
                if self.last.return_feasible {
                    command.set_left_track(command.left_track().min(-0.2));
                    command.set_right_track(command.right_track().min(-0.2));
                } else {
                    command.set_left_track(0.0);
                    command.set_right_track(0.0);
                }
            }
            TemporalAuthority::HoldForReview => {
                command.set_cutter_head(0.0);
                command.set_auger_feed(0.0);
                command.set_left_track(0.0);
                command.set_right_track(0.0);
                command.set_ballast_trim(0.0);
            }
        }
        command.sanitize();
        command
    }

    pub fn last(&self) -> &TemporalAssuranceAssessment {
        &self.last
    }

    pub const fn control_time_ns(&self) -> u64 {
        self.control_time_ns
    }

    pub fn events(&self) -> &CausalEventLedger {
        &self.events
    }

    pub fn attribution(&self) -> &CausalAttributionLedger {
        &self.attribution
    }
}

fn clock_rank(disposition: ClockDisposition) -> u8 {
    match disposition {
        ClockDisposition::Accepted => 0,
        ClockDisposition::Degraded => 1,
        ClockDisposition::Rejected => 2,
    }
}

fn promote(
    current: &mut TemporalAuthority,
    candidate: TemporalAuthority,
    reasons: &mut Vec<String>,
    reason: impl Into<String>,
) {
    if candidate > *current {
        *current = candidate;
    }
    if candidate != TemporalAuthority::Nominal {
        let reason = reason.into();
        if !reasons.contains(&reason) {
            reasons.push(reason);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delayed_observation::ObservationPurpose;
    use crate::temporal_clock::{ClockDomain, ClockSourceId};

    fn nominal_frame(step: u64, time_ns: u64, revisions: RuntimeRevisions) -> TemporalRuntimeFrame {
        TemporalRuntimeFrame {
            schema_version: TEMPORAL_RUNTIME_FRAME_SCHEMA_VERSION,
            clock_samples: vec![ClockSample {
                source: ClockSourceId(0),
                domain: ClockDomain::Control,
                boot_epoch: 1,
                sequence: step + 1,
                event_time_ns: time_ns,
                uncertainty_ns: 1,
                received_step: step,
            }],
            observations: vec![TimedObservation {
                source: 0,
                purpose: ObservationPurpose::ImmediateControl,
                observed_time_ns: time_ns,
                received_time_ns: time_ns,
                uncertainty_ns: 1,
                freshness_limit_ns: 20_000_000,
                sequence: step + 1,
            }],
            plan: Some(PlanBasis {
                plan_id: step + 1,
                created_step: step,
                expires_step: step,
                revisions,
                permits_productive_work: true,
            }),
            ..TemporalRuntimeFrame::default()
        }
    }

    #[test]
    fn fresh_frame_keeps_nominal_authority() {
        let revisions = RuntimeRevisions::default();
        let mut supervisor = TemporalAssuranceSupervisor::default();
        let frame = nominal_frame(0, 5_000_000, revisions);
        let assessment = supervisor.assess(0.005, 0, revisions, &frame, true, false);
        assert_eq!(assessment.authority, TemporalAuthority::Nominal);
    }

    #[test]
    fn stale_control_observation_removes_productive_authority_same_frame() {
        let revisions = RuntimeRevisions::default();
        let mut supervisor = TemporalAssuranceSupervisor::default();
        let mut frame = nominal_frame(0, 5_000_000, revisions);
        frame.observations[0].observed_time_ns = 0;
        frame.observations[0].freshness_limit_ns = 1_000_000;
        let assessment = supervisor.assess(0.005, 0, revisions, &frame, true, false);
        assert!(assessment.authority >= TemporalAuthority::ReturnOnly);
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(1.0);
        let constrained = supervisor.constrain_command(command);
        assert_eq!(constrained.cutter_head(), 0.0);
    }
}
