// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Safety-monotonic human-rescue ethics authority.
//!
//! The supervisor composes consent continuity, emergency authorization,
//! corroborated subject claims, and transparent triage. It cannot diagnose,
//! identify a person from raw data, or create actuator authority.

use crate::rescue::RescueHandoffState;
use crate::rescue_consent::RescueConsentDisposition;
use crate::rescue_triage::{
    MAX_RESCUE_TRIAGE_CANDIDATES, RescueTriageAssessment, RescueTriageDisposition,
};
use crate::team::AgentId;
use crate::types::{AUGER, BALLAST_TRIM, CUTTER_HEAD, LEFT_TRACK, RIGHT_TRACK, SubterraneanCommand};
use serde::{Deserialize, Serialize};

pub const RESCUE_ETHICS_SCHEMA_VERSION: u16 = 1;
pub const MAX_RESCUE_ETHICS_REASONS: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueEthicsAuthority {
    Nominal,
    AwaitConsent,
    ReconcileClaims,
    RescueOnly,
    HoldForReview,
}

impl RescueEthicsAuthority {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::AwaitConsent => "await_consent",
            Self::ReconcileClaims => "reconcile_claims",
            Self::RescueOnly => "rescue_only",
            Self::HoldForReview => "hold_for_review",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueEthicsAssessment {
    pub authority: RescueEthicsAuthority,
    pub selected_subject: Option<AgentId>,
    pub selected_case_id: Option<u64>,
    pub consent: RescueConsentDisposition,
    pub emergency_authorized: bool,
    pub identity_conflict: bool,
    pub care_conflict: bool,
    pub non_discrimination_invariant_satisfied: bool,
    pub triage_candidates: usize,
    pub reasons: Vec<String>,
}

impl RescueEthicsAssessment {
    pub const fn nominal() -> Self {
        Self {
            authority: RescueEthicsAuthority::Nominal,
            selected_subject: None,
            selected_case_id: None,
            consent: RescueConsentDisposition::Unknown,
            emergency_authorized: false,
            identity_conflict: false,
            care_conflict: false,
            non_discrimination_invariant_satisfied: true,
            triage_candidates: 0,
            reasons: Vec::new(),
        }
    }

    pub fn validate(&self) -> bool {
        let selected_pair_is_consistent =
            self.selected_subject.is_some() == self.selected_case_id.is_some();
        let selected_case_is_valid = self.selected_case_id.is_none_or(|case_id| case_id != 0);
        let reasons_are_valid = self.reasons.len() <= MAX_RESCUE_ETHICS_REASONS
            && self.reasons.iter().all(|reason| !reason.trim().is_empty());
        let authority_is_consistent = match self.authority {
            RescueEthicsAuthority::Nominal => true,
            RescueEthicsAuthority::RescueOnly => {
                self.selected_subject.is_some()
                    && (self.consent == RescueConsentDisposition::Consented
                        || self.emergency_authorized)
                    && !self.identity_conflict
                    && !self.care_conflict
                    && self.non_discrimination_invariant_satisfied
            }
            RescueEthicsAuthority::AwaitConsent => {
                self.consent == RescueConsentDisposition::Unknown
                    && !self.emergency_authorized
                    && !self.identity_conflict
                    && !self.care_conflict
            }
            RescueEthicsAuthority::ReconcileClaims => {
                self.identity_conflict || self.care_conflict
            }
            RescueEthicsAuthority::HoldForReview => true,
        };
        selected_pair_is_consistent
            && selected_case_is_valid
            && reasons_are_valid
            && self.triage_candidates <= MAX_RESCUE_TRIAGE_CANDIDATES
            && authority_is_consistent
    }

    pub fn constrain_command(&self, mut command: SubterraneanCommand) -> SubterraneanCommand {
        match self.authority {
            RescueEthicsAuthority::Nominal => command,
            RescueEthicsAuthority::RescueOnly => {
                command.torques[CUTTER_HEAD] = 0.0;
                command.torques[AUGER] = 0.0;
                command
            }
            RescueEthicsAuthority::AwaitConsent
            | RescueEthicsAuthority::ReconcileClaims
            | RescueEthicsAuthority::HoldForReview => {
                command.torques[CUTTER_HEAD] = 0.0;
                command.torques[AUGER] = 0.0;
                command.torques[LEFT_TRACK] = 0.0;
                command.torques[RIGHT_TRACK] = 0.0;
                command.torques[BALLAST_TRIM] = 0.0;
                command
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueEthicsSupervisor {
    last: RescueEthicsAssessment,
}

impl RescueEthicsSupervisor {
    pub fn assess(
        &mut self,
        handoff_state: RescueHandoffState,
        triage: &RescueTriageAssessment,
    ) -> RescueEthicsAssessment {
        let engaged = matches!(
            handoff_state,
            RescueHandoffState::Requested
                | RescueHandoffState::Offered
                | RescueHandoffState::Accepted
                | RescueHandoffState::Active
        );
        let selected = triage.selected.as_ref();
        let leading = triage.ordered.first();
        let consent = selected
            .map(|candidate| candidate.consent)
            .or_else(|| leading.map(|entry| entry.candidate.consent))
            .unwrap_or(RescueConsentDisposition::Unknown);
        let identity_conflict = triage
            .ordered
            .iter()
            .any(|entry| entry.candidate.claim_assessment.identity_conflict);
        let care_conflict = triage
            .ordered
            .iter()
            .any(|entry| entry.candidate.claim_assessment.care_conflict);
        let emergency_authorized = selected.is_some_and(|candidate| candidate.emergency_authorized);
        let mut reasons = Vec::new();
        let authority = if !triage.non_discrimination_invariant_satisfied {
            push_reason(
                &mut reasons,
                "rescue triage non-discrimination invariant is not satisfied",
            );
            RescueEthicsAuthority::HoldForReview
        } else if identity_conflict || care_conflict {
            if identity_conflict {
                push_reason(
                    &mut reasons,
                    "trusted rescue claims conflict on subject identity",
                );
            }
            if care_conflict {
                push_reason(
                    &mut reasons,
                    "trusted rescue claims conflict on care urgency",
                );
            }
            if matches!(handoff_state, RescueHandoffState::Accepted | RescueHandoffState::Active) {
                RescueEthicsAuthority::HoldForReview
            } else {
                RescueEthicsAuthority::ReconcileClaims
            }
        } else if engaged
            && matches!(
                consent,
                RescueConsentDisposition::Refused | RescueConsentDisposition::Withdrawn
            )
        {
            push_reason(
                &mut reasons,
                "the rescue subject refused or withdrew case-specific consent",
            );
            RescueEthicsAuthority::HoldForReview
        } else if matches!(handoff_state, RescueHandoffState::Accepted | RescueHandoffState::Active)
        {
            if selected.is_some() {
                push_reason(
                    &mut reasons,
                    "case-specific consent and transparent triage permit rescue-only authority",
                );
                RescueEthicsAuthority::RescueOnly
            } else {
                push_reason(
                    &mut reasons,
                    "active rescue lacks a currently eligible triage subject",
                );
                RescueEthicsAuthority::HoldForReview
            }
        } else if engaged
            && leading.is_some_and(|entry| {
                entry.disposition == RescueTriageDisposition::AwaitingConsent
            })
        {
            push_reason(
                &mut reasons,
                "rescue motion awaits case-specific consent or valid emergency authority",
            );
            RescueEthicsAuthority::AwaitConsent
        } else {
            RescueEthicsAuthority::Nominal
        };
        self.last = RescueEthicsAssessment {
            authority,
            selected_subject: selected.map(|candidate| candidate.subject),
            selected_case_id: selected.map(|candidate| candidate.case_id.0),
            consent,
            emergency_authorized,
            identity_conflict,
            care_conflict,
            non_discrimination_invariant_satisfied: triage
                .non_discrimination_invariant_satisfied,
            triage_candidates: triage.ordered.len(),
            reasons,
        };
        self.last.clone()
    }

    pub fn last(&self) -> &RescueEthicsAssessment {
        &self.last
    }

    pub fn validate(&self) -> bool {
        self.last.validate()
    }
}

impl Default for RescueEthicsSupervisor {
    fn default() -> Self {
        Self {
            last: RescueEthicsAssessment::nominal(),
        }
    }
}

fn push_reason(reasons: &mut Vec<String>, reason: &str) {
    if reasons.len() < MAX_RESCUE_ETHICS_REASONS {
        reasons.push(reason.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rescue::RescueCaseId;
    use crate::rescue_subject_claim::{RescueCareUrgency, RescueSubjectClaimAssessment};
    use crate::rescue_triage::{
        RankedRescueCandidate, RescueTriageCandidate, RescueTriageDisposition,
        RESCUE_TRIAGE_SCHEMA_VERSION,
    };

    fn triage(consent: RescueConsentDisposition) -> RescueTriageAssessment {
        let candidate = RescueTriageCandidate {
            schema_version: RESCUE_TRIAGE_SCHEMA_VERSION,
            subject: AgentId::new(2),
            case_id: RescueCaseId(7),
            observed_step: 10,
            hazard_severity: 0.9,
            survival_window_steps: 20,
            route_reachable: true,
            rescue_energy_ratio: 0.2,
            evidence_confidence: 0.9,
            consent,
            claim_assessment: RescueSubjectClaimAssessment {
                case_id: RescueCaseId(7),
                subject: Some(AgentId::new(2)),
                trusted_reporters: 2,
                identity_bindings: 1,
                identity_conflict: false,
                care_urgency: RescueCareUrgency::Critical,
                care_conflict: false,
                communication_unavailable_corroborated: false,
                reasons: Vec::new(),
            },
            emergency_authorized: false,
        };
        RescueTriageAssessment {
            ordered: vec![RankedRescueCandidate {
                candidate: candidate.clone(),
                disposition: if consent == RescueConsentDisposition::Consented {
                    RescueTriageDisposition::Eligible
                } else {
                    RescueTriageDisposition::AwaitingConsent
                },
                urgency_score: 0.9,
                explanation: vec!["test candidate".into()],
            }],
            selected: (consent == RescueConsentDisposition::Consented).then_some(candidate),
            eligible_count: usize::from(consent == RescueConsentDisposition::Consented),
            emergency_authorized_count: 0,
            non_discrimination_invariant_satisfied: true,
        }
    }

    #[test]
    fn rescue_only_requires_selected_authorized_subject() {
        let mut invalid = RescueEthicsAssessment::nominal();
        invalid.authority = RescueEthicsAuthority::RescueOnly;
        invalid.triage_candidates = 1;
        assert!(!invalid.validate());
    }

    #[test]
    fn active_rescue_with_withdrawn_consent_holds_motion() {
        let mut supervisor = RescueEthicsSupervisor::default();
        let assessment = supervisor.assess(
            RescueHandoffState::Active,
            &triage(RescueConsentDisposition::Withdrawn),
        );
        assert_eq!(assessment.authority, RescueEthicsAuthority::HoldForReview);
        let mut command = SubterraneanCommand::zero();
        command.torques[LEFT_TRACK] = 0.8;
        command.dewatering_pump = 0.7;
        let constrained = assessment.constrain_command(command);
        assert_eq!(constrained.torques[LEFT_TRACK], 0.0);
        assert_eq!(constrained.dewatering_pump, 0.7);
    }

    #[test]
    fn valid_active_rescue_removes_productive_work_not_mobility() {
        let mut supervisor = RescueEthicsSupervisor::default();
        let assessment = supervisor.assess(
            RescueHandoffState::Active,
            &triage(RescueConsentDisposition::Consented),
        );
        assert_eq!(assessment.authority, RescueEthicsAuthority::RescueOnly);
        let mut command = SubterraneanCommand::zero();
        command.torques[CUTTER_HEAD] = 0.9;
        command.torques[LEFT_TRACK] = 0.5;
        let constrained = assessment.constrain_command(command);
        assert_eq!(constrained.torques[CUTTER_HEAD], 0.0);
        assert_eq!(constrained.torques[LEFT_TRACK], 0.5);
    }
}
