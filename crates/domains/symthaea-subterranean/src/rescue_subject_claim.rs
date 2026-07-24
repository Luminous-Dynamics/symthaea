// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Corroborated, non-diagnostic rescue-subject claims.
//!
//! This module does not identify people, diagnose illness, or decide legal
//! personhood. It keeps opaque subject bindings and coarse care-urgency claims
//! so contradictory reports cannot silently redirect rescue authority.

use crate::rescue::RescueCaseId;
use crate::team::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RESCUE_SUBJECT_CLAIM_SCHEMA_VERSION: u16 = 1;
pub const MAX_RESCUE_SUBJECT_CLAIMS: usize = 64;
pub const MAX_RESCUE_SUBJECT_REASONS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RescueCareUrgency {
    Unknown,
    Stable,
    Urgent,
    Critical,
}

impl RescueCareUrgency {
    pub const fn rank(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Stable => 1,
            Self::Urgent => 2,
            Self::Critical => 3,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Stable => "stable",
            Self::Urgent => "urgent",
            Self::Critical => "critical",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueSubjectClaim {
    pub schema_version: u16,
    pub case_id: RescueCaseId,
    pub subject: AgentId,
    pub reporter: AgentId,
    /// Opaque externally generated binding. This crate does not interpret it.
    pub identity_binding: u64,
    pub care_urgency: RescueCareUrgency,
    pub communication_unavailable: bool,
    pub epoch: u32,
    pub sequence: u64,
    pub issued_step: u64,
    pub expires_step: u64,
}

impl RescueSubjectClaim {
    pub fn validate(self) -> bool {
        self.schema_version == RESCUE_SUBJECT_CLAIM_SCHEMA_VERSION
            && self.case_id.0 != 0
            && self.subject != AgentId::SURFACE_CONTROL
            && self.reporter != AgentId::SURFACE_CONTROL
            && self.identity_binding != 0
            && self.expires_step >= self.issued_step
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RescueSubjectClaimRejection {
    InvalidClaim,
    UntrustedReporter,
    Replay,
    EpochRegression,
    Capacity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueSubjectClaimAssessment {
    pub case_id: RescueCaseId,
    pub subject: Option<AgentId>,
    pub trusted_reporters: usize,
    pub identity_bindings: usize,
    pub identity_conflict: bool,
    pub care_urgency: RescueCareUrgency,
    pub care_conflict: bool,
    pub communication_unavailable_corroborated: bool,
    pub reasons: Vec<String>,
}

impl RescueSubjectClaimAssessment {
    pub const fn empty(case_id: RescueCaseId) -> Self {
        Self {
            case_id,
            subject: None,
            trusted_reporters: 0,
            identity_bindings: 0,
            identity_conflict: false,
            care_urgency: RescueCareUrgency::Unknown,
            care_conflict: false,
            communication_unavailable_corroborated: false,
            reasons: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueSubjectClaimLedger {
    claims: BTreeMap<(RescueCaseId, AgentId), RescueSubjectClaim>,
}

impl RescueSubjectClaimLedger {
    pub fn new() -> Self {
        Self {
            claims: BTreeMap::new(),
        }
    }

    pub fn ingest(
        &mut self,
        claim: RescueSubjectClaim,
        reporter_trusted: bool,
    ) -> Result<(), RescueSubjectClaimRejection> {
        if !claim.validate() {
            return Err(RescueSubjectClaimRejection::InvalidClaim);
        }
        if !reporter_trusted {
            return Err(RescueSubjectClaimRejection::UntrustedReporter);
        }
        let key = (claim.case_id, claim.reporter);
        if let Some(previous) = self.claims.get(&key) {
            if claim.epoch < previous.epoch {
                return Err(RescueSubjectClaimRejection::EpochRegression);
            }
            if claim.epoch == previous.epoch && claim.sequence <= previous.sequence {
                return Err(RescueSubjectClaimRejection::Replay);
            }
        }
        if self.claims.len() >= MAX_RESCUE_SUBJECT_CLAIMS && !self.claims.contains_key(&key) {
            return Err(RescueSubjectClaimRejection::Capacity);
        }
        self.claims.insert(key, claim);
        Ok(())
    }

    pub fn assess(
        &mut self,
        case_id: RescueCaseId,
        current_step: u64,
    ) -> RescueSubjectClaimAssessment {
        self.claims
            .retain(|_, claim| current_step <= claim.expires_step);
        let current: Vec<RescueSubjectClaim> = self
            .claims
            .values()
            .copied()
            .filter(|claim| claim.case_id == case_id)
            .collect();
        if current.is_empty() {
            return RescueSubjectClaimAssessment::empty(case_id);
        }
        let subjects: BTreeSet<AgentId> = current.iter().map(|claim| claim.subject).collect();
        let bindings: BTreeSet<u64> = current
            .iter()
            .map(|claim| claim.identity_binding)
            .collect();
        let min_urgency = current
            .iter()
            .map(|claim| claim.care_urgency.rank())
            .min()
            .unwrap_or_default();
        let max_urgency = current
            .iter()
            .map(|claim| claim.care_urgency.rank())
            .max()
            .unwrap_or_default();
        let care_urgency = current
            .iter()
            .map(|claim| claim.care_urgency)
            .max()
            .unwrap_or(RescueCareUrgency::Unknown);
        let communication_reporters = current
            .iter()
            .filter(|claim| claim.communication_unavailable)
            .map(|claim| claim.reporter)
            .collect::<BTreeSet<_>>()
            .len();
        let identity_conflict = subjects.len() > 1 || bindings.len() > 1;
        let care_conflict = current.len() >= 2 && max_urgency.saturating_sub(min_urgency) >= 2;
        let mut reasons = Vec::new();
        if identity_conflict {
            push_reason(
                &mut reasons,
                "trusted rescue reports disagree on the opaque subject binding",
            );
        }
        if care_conflict {
            push_reason(
                &mut reasons,
                "trusted rescue reports materially disagree on care urgency",
            );
        }
        RescueSubjectClaimAssessment {
            case_id,
            subject: if subjects.len() == 1 {
                subjects.iter().next().copied()
            } else {
                None
            },
            trusted_reporters: current.len(),
            identity_bindings: bindings.len(),
            identity_conflict,
            care_urgency,
            care_conflict,
            communication_unavailable_corroborated: communication_reporters >= 2,
            reasons,
        }
    }

    pub fn validate(&self) -> bool {
        self.claims.len() <= MAX_RESCUE_SUBJECT_CLAIMS
            && self.claims.iter().all(|((case_id, reporter), claim)| {
                *case_id == claim.case_id && *reporter == claim.reporter && claim.validate()
            })
    }
}

impl Default for RescueSubjectClaimLedger {
    fn default() -> Self {
        Self::new()
    }
}

fn push_reason(reasons: &mut Vec<String>, reason: &str) {
    if reasons.len() < MAX_RESCUE_SUBJECT_REASONS {
        reasons.push(reason.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim(reporter: u64, binding: u64, urgency: RescueCareUrgency) -> RescueSubjectClaim {
        RescueSubjectClaim {
            schema_version: RESCUE_SUBJECT_CLAIM_SCHEMA_VERSION,
            case_id: RescueCaseId(7),
            subject: AgentId::new(2),
            reporter: AgentId::new(reporter),
            identity_binding: binding,
            care_urgency: urgency,
            communication_unavailable: true,
            epoch: 1,
            sequence: 1,
            issued_step: 10,
            expires_step: 100,
        }
    }

    #[test]
    fn conflicting_identity_bindings_require_reconciliation() {
        let mut ledger = RescueSubjectClaimLedger::new();
        assert!(ledger.ingest(claim(3, 10, RescueCareUrgency::Critical), true).is_ok());
        assert!(ledger.ingest(claim(4, 11, RescueCareUrgency::Critical), true).is_ok());
        let assessment = ledger.assess(RescueCaseId(7), 20);
        assert!(assessment.identity_conflict);
    }

    #[test]
    fn inability_to_communicate_requires_two_distinct_reporters() {
        let mut ledger = RescueSubjectClaimLedger::new();
        assert!(ledger.ingest(claim(3, 10, RescueCareUrgency::Critical), true).is_ok());
        assert!(!ledger
            .assess(RescueCaseId(7), 20)
            .communication_unavailable_corroborated);
        assert!(ledger.ingest(claim(4, 10, RescueCareUrgency::Urgent), true).is_ok());
        assert!(ledger
            .assess(RescueCaseId(7), 20)
            .communication_unavailable_corroborated);
    }
}
