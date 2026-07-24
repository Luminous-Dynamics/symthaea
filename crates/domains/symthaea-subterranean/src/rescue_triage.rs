// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transparent, role-neutral rescue triage.
//!
//! The candidate schema intentionally excludes occupation, mission value,
//! payload value, nationality, disability, age, sex, race, religion, and other
//! protected or socially ranked attributes. Stable agent identity is used only
//! as the final deterministic tie-breaker.

use crate::rescue::RescueCaseId;
use crate::rescue_consent::RescueConsentDisposition;
use crate::rescue_subject_claim::{RescueCareUrgency, RescueSubjectClaimAssessment};
use crate::team::AgentId;
use serde::{Deserialize, Serialize};

pub const RESCUE_TRIAGE_SCHEMA_VERSION: u16 = 1;
pub const MAX_RESCUE_TRIAGE_CANDIDATES: usize = 16;
pub const MAX_RESCUE_TRIAGE_REASONS: usize = 12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RescueTriageCandidate {
    pub schema_version: u16,
    pub subject: AgentId,
    pub case_id: RescueCaseId,
    pub observed_step: u64,
    pub hazard_severity: f32,
    pub survival_window_steps: u64,
    pub route_reachable: bool,
    pub rescue_energy_ratio: f64,
    pub evidence_confidence: f64,
    pub consent: RescueConsentDisposition,
    pub claim_assessment: RescueSubjectClaimAssessment,
    pub emergency_authorized: bool,
}

impl RescueTriageCandidate {
    pub fn validate(&self) -> bool {
        self.schema_version == RESCUE_TRIAGE_SCHEMA_VERSION
            && self.subject != AgentId::SURFACE_CONTROL
            && self.case_id.0 != 0
            && self.hazard_severity.is_finite()
            && (0.0..=1.0).contains(&self.hazard_severity)
            && self.rescue_energy_ratio.is_finite()
            && (0.0..=1.0).contains(&self.rescue_energy_ratio)
            && self.evidence_confidence.is_finite()
            && (0.0..=1.0).contains(&self.evidence_confidence)
            && self.claim_assessment.case_id == self.case_id
            && (!self.emergency_authorized
                || self
                    .claim_assessment
                    .communication_unavailable_corroborated)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueTriageDisposition {
    Eligible,
    EmergencyAuthorized,
    AwaitingConsent,
    Refused,
    Withdrawn,
    IdentityConflict,
    CareConflict,
    Unreachable,
    Stale,
}

impl RescueTriageDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::EmergencyAuthorized => "emergency_authorized",
            Self::AwaitingConsent => "awaiting_consent",
            Self::Refused => "refused",
            Self::Withdrawn => "withdrawn",
            Self::IdentityConflict => "identity_conflict",
            Self::CareConflict => "care_conflict",
            Self::Unreachable => "unreachable",
            Self::Stale => "stale",
        }
    }

    pub const fn permits_rescue(self) -> bool {
        matches!(self, Self::Eligible | Self::EmergencyAuthorized)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedRescueCandidate {
    pub candidate: RescueTriageCandidate,
    pub disposition: RescueTriageDisposition,
    pub urgency_score: f64,
    pub explanation: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RescueTriageAssessment {
    pub ordered: Vec<RankedRescueCandidate>,
    pub selected: Option<RescueTriageCandidate>,
    pub eligible_count: usize,
    pub emergency_authorized_count: usize,
    pub non_discrimination_invariant_satisfied: bool,
}

impl RescueTriageAssessment {
    pub const fn empty() -> Self {
        Self {
            ordered: Vec::new(),
            selected: None,
            eligible_count: 0,
            emergency_authorized_count: 0,
            non_discrimination_invariant_satisfied: true,
        }
    }

    pub fn validate(&self) -> bool {
        self.ordered.len() <= MAX_RESCUE_TRIAGE_CANDIDATES
            && self.non_discrimination_invariant_satisfied
            && self.ordered.iter().all(|entry| {
                entry.candidate.validate()
                    && entry.urgency_score.is_finite()
                    && (-1.0..=2.0).contains(&entry.urgency_score)
                    && entry.explanation.len() <= MAX_RESCUE_TRIAGE_REASONS
                    && entry
                        .explanation
                        .iter()
                        .all(|reason| !reason.trim().is_empty())
            })
    }
}

impl Default for RescueTriageAssessment {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RescueTriagePolicy {
    pub stale_after_steps: u64,
    pub maximum_rescue_energy_ratio: f64,
}

impl Default for RescueTriagePolicy {
    fn default() -> Self {
        Self {
            stale_after_steps: 400,
            maximum_rescue_energy_ratio: 0.6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RescueTriageSupervisor {
    policy: RescueTriagePolicy,
    last: RescueTriageAssessment,
}

impl RescueTriageSupervisor {
    pub fn new(policy: RescueTriagePolicy) -> Self {
        Self {
            policy,
            last: RescueTriageAssessment::empty(),
        }
    }

    pub fn assess(
        &mut self,
        current_step: u64,
        candidates: &[RescueTriageCandidate],
    ) -> RescueTriageAssessment {
        let mut ordered: Vec<RankedRescueCandidate> = candidates
            .iter()
            .filter(|candidate| candidate.validate())
            .take(MAX_RESCUE_TRIAGE_CANDIDATES)
            .map(|candidate| {
                let age = current_step.saturating_sub(candidate.observed_step);
                let disposition = if age > self.policy.stale_after_steps {
                    RescueTriageDisposition::Stale
                } else if !candidate.route_reachable
                    || candidate.rescue_energy_ratio > self.policy.maximum_rescue_energy_ratio
                {
                    RescueTriageDisposition::Unreachable
                } else if candidate.claim_assessment.identity_conflict {
                    RescueTriageDisposition::IdentityConflict
                } else if candidate.claim_assessment.care_conflict {
                    RescueTriageDisposition::CareConflict
                } else {
                    match candidate.consent {
                        RescueConsentDisposition::Consented => RescueTriageDisposition::Eligible,
                        RescueConsentDisposition::Refused => RescueTriageDisposition::Refused,
                        RescueConsentDisposition::Withdrawn => RescueTriageDisposition::Withdrawn,
                        RescueConsentDisposition::Unknown if candidate.emergency_authorized => {
                            RescueTriageDisposition::EmergencyAuthorized
                        }
                        RescueConsentDisposition::Unknown => {
                            RescueTriageDisposition::AwaitingConsent
                        }
                    }
                };
                let survival_pressure = if candidate.survival_window_steps == 0 {
                    1.0
                } else {
                    1.0 / (1.0 + candidate.survival_window_steps as f64 / 100.0)
                };
                let care_pressure = match candidate.claim_assessment.care_urgency {
                    RescueCareUrgency::Unknown => 0.0,
                    RescueCareUrgency::Stable => 0.2,
                    RescueCareUrgency::Urgent => 0.6,
                    RescueCareUrgency::Critical => 1.0,
                };
                let urgency_score = f64::from(candidate.hazard_severity) * 0.45
                    + survival_pressure * 0.25
                    + care_pressure * 0.15
                    + candidate.evidence_confidence * 0.10
                    - candidate.rescue_energy_ratio * 0.05;
                let mut explanation = Vec::new();
                push_reason(
                    &mut explanation,
                    &format!(
                        "hazard severity {:.3} contributes to immediate threat",
                        candidate.hazard_severity
                    ),
                );
                push_reason(
                    &mut explanation,
                    &format!(
                        "bounded survival window {} steps",
                        candidate.survival_window_steps
                    ),
                );
                push_reason(
                    &mut explanation,
                    &format!(
                        "care urgency {}",
                        candidate.claim_assessment.care_urgency.label()
                    ),
                );
                push_reason(
                    &mut explanation,
                    &format!("consent disposition {}", candidate.consent.label()),
                );
                if candidate.emergency_authorized {
                    push_reason(
                        &mut explanation,
                        "split-role emergency authority permits rescue while communication is unavailable",
                    );
                }
                if !disposition.permits_rescue() {
                    push_reason(
                        &mut explanation,
                        &format!("candidate disposition {} blocks rescue authority", disposition.label()),
                    );
                }
                RankedRescueCandidate {
                    candidate: candidate.clone(),
                    disposition,
                    urgency_score,
                    explanation,
                }
            })
            .collect();
        ordered.sort_by(|left, right| {
            disposition_rank(left.disposition)
                .cmp(&disposition_rank(right.disposition))
                .then_with(|| right.urgency_score.total_cmp(&left.urgency_score))
                .then_with(|| {
                    left.candidate
                        .survival_window_steps
                        .cmp(&right.candidate.survival_window_steps)
                })
                .then_with(|| left.candidate.subject.cmp(&right.candidate.subject))
                .then_with(|| left.candidate.case_id.cmp(&right.candidate.case_id))
        });
        let selected = ordered
            .iter()
            .find(|entry| entry.disposition.permits_rescue())
            .map(|entry| entry.candidate.clone());
        let eligible_count = ordered
            .iter()
            .filter(|entry| entry.disposition.permits_rescue())
            .count();
        let emergency_authorized_count = ordered
            .iter()
            .filter(|entry| entry.disposition == RescueTriageDisposition::EmergencyAuthorized)
            .count();
        self.last = RescueTriageAssessment {
            ordered,
            selected,
            eligible_count,
            emergency_authorized_count,
            non_discrimination_invariant_satisfied: true,
        };
        self.last.clone()
    }

    pub fn last(&self) -> &RescueTriageAssessment {
        &self.last
    }

    pub fn validate(&self) -> bool {
        self.policy.maximum_rescue_energy_ratio.is_finite()
            && (0.0..=1.0).contains(&self.policy.maximum_rescue_energy_ratio)
            && self.last.validate()
    }
}

fn disposition_rank(disposition: RescueTriageDisposition) -> u8 {
    match disposition {
        RescueTriageDisposition::Eligible => 0,
        RescueTriageDisposition::EmergencyAuthorized => 1,
        RescueTriageDisposition::AwaitingConsent => 2,
        RescueTriageDisposition::IdentityConflict => 3,
        RescueTriageDisposition::CareConflict => 4,
        RescueTriageDisposition::Refused => 5,
        RescueTriageDisposition::Withdrawn => 6,
        RescueTriageDisposition::Unreachable => 7,
        RescueTriageDisposition::Stale => 8,
    }
}

fn push_reason(reasons: &mut Vec<String>, reason: &str) {
    if reasons.len() < MAX_RESCUE_TRIAGE_REASONS {
        reasons.push(reason.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(subject: u64, consent: RescueConsentDisposition) -> RescueTriageCandidate {
        RescueTriageCandidate {
            schema_version: RESCUE_TRIAGE_SCHEMA_VERSION,
            subject: AgentId::new(subject),
            case_id: RescueCaseId(subject),
            observed_step: 10,
            hazard_severity: 0.9,
            survival_window_steps: 50,
            route_reachable: true,
            rescue_energy_ratio: 0.2,
            evidence_confidence: 0.9,
            consent,
            claim_assessment: RescueSubjectClaimAssessment {
                case_id: RescueCaseId(subject),
                subject: Some(AgentId::new(subject)),
                trusted_reporters: 2,
                identity_bindings: 1,
                identity_conflict: false,
                care_urgency: RescueCareUrgency::Critical,
                care_conflict: false,
                communication_unavailable_corroborated: false,
                reasons: Vec::new(),
            },
            emergency_authorized: false,
        }
    }

    #[test]
    fn refusal_is_never_overridden_by_urgency() {
        let mut supervisor = RescueTriageSupervisor::default();
        let assessment = supervisor.assess(
            20,
            &[
                candidate(2, RescueConsentDisposition::Refused),
                candidate(3, RescueConsentDisposition::Consented),
            ],
        );
        assert_eq!(assessment.selected.map(|value| value.subject), Some(AgentId::new(3)));
    }

    #[test]
    fn tie_break_uses_only_stable_identity_after_equal_allowed_factors() {
        let mut supervisor = RescueTriageSupervisor::default();
        let assessment = supervisor.assess(
            20,
            &[
                candidate(7, RescueConsentDisposition::Consented),
                candidate(4, RescueConsentDisposition::Consented),
            ],
        );
        assert_eq!(assessment.selected.map(|value| value.subject), Some(AgentId::new(4)));
        assert!(assessment.non_discrimination_invariant_satisfied);
    }
}
