// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded restoration obligations created by underground disturbance.
//!
//! Completion requires both physical progress and externally attributable
//! evidence. The ledger never treats elapsed time or mission completion as
//! restoration.

use crate::tunnel_graph::TunnelNodeId;
use serde::{Deserialize, Serialize};

pub const RESTORATION_STEWARDSHIP_SCHEMA_VERSION: u16 = 1;
pub const MAX_RESTORATION_OBLIGATIONS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestorationObligationKind {
    SealBore,
    StabilizeRoof,
    TreatWater,
    RemoveSpoil,
    RestoreHabitat,
    MonitorGroundwater,
}

impl RestorationObligationKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::SealBore => "seal_bore",
            Self::StabilizeRoof => "stabilize_roof",
            Self::TreatWater => "treat_water",
            Self::RemoveSpoil => "remove_spoil",
            Self::RestoreHabitat => "restore_habitat",
            Self::MonitorGroundwater => "monitor_groundwater",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestorationState {
    Open,
    InProgress,
    AwaitingEvidence,
    Complete,
}

impl RestorationState {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::InProgress => "in_progress",
            Self::AwaitingEvidence => "awaiting_evidence",
            Self::Complete => "complete",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestorationObligation {
    pub id: u64,
    pub kind: RestorationObligationKind,
    pub node: Option<TunnelNodeId>,
    pub required_quantity: f64,
    pub completed_quantity: f64,
    pub due_step: u64,
    pub state: RestorationState,
    pub authority_reference: String,
    pub completion_evidence_id: Option<String>,
    pub last_update_step: u64,
}

impl RestorationObligation {
    pub fn validate(&self) -> bool {
        self.id > 0
            && self.required_quantity.is_finite()
            && self.required_quantity > 0.0
            && self.completed_quantity.is_finite()
            && self.completed_quantity >= 0.0
            && self.completed_quantity <= self.required_quantity + f64::EPSILON
            && self.due_step > 0
            && !self.authority_reference.trim().is_empty()
            && self
                .completion_evidence_id
                .as_ref()
                .is_none_or(|value| !value.trim().is_empty())
            && (self.state != RestorationState::Complete
                || (self.completed_quantity + f64::EPSILON >= self.required_quantity
                    && self.completion_evidence_id.is_some()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestorationDisposition {
    Clear,
    ObligationsOpen,
    RestorationDue,
    RestorationOverdue,
}

impl RestorationDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Clear => "clear",
            Self::ObligationsOpen => "obligations_open",
            Self::RestorationDue => "restoration_due",
            Self::RestorationOverdue => "restoration_overdue",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RestorationAssessment {
    pub disposition: RestorationDisposition,
    pub open_obligations: usize,
    pub overdue_obligations: usize,
    pub completion_fraction: f64,
    pub new_productive_work_allowed: bool,
    pub return_required: bool,
}

impl RestorationAssessment {
    pub const fn clear() -> Self {
        Self {
            disposition: RestorationDisposition::Clear,
            open_obligations: 0,
            overdue_obligations: 0,
            completion_fraction: 1.0,
            new_productive_work_allowed: true,
            return_required: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestorationError {
    UnsupportedSchema,
    CapacityExceeded,
    DuplicateId,
    UnknownObligation,
    InvalidObligation,
    InvalidProgress,
    TerminalObligation,
    EvidenceBeforeCompletion,
    InvalidEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestorationLedger {
    pub schema_version: u16,
    obligations: Vec<RestorationObligation>,
}

impl RestorationLedger {
    pub fn new() -> Self {
        Self {
            schema_version: RESTORATION_STEWARDSHIP_SCHEMA_VERSION,
            obligations: Vec::new(),
        }
    }

    pub fn obligations(&self) -> &[RestorationObligation] {
        &self.obligations
    }

    pub fn add(&mut self, obligation: RestorationObligation) -> Result<(), RestorationError> {
        if self.obligations.len() >= MAX_RESTORATION_OBLIGATIONS {
            return Err(RestorationError::CapacityExceeded);
        }
        if !obligation.validate() || obligation.state == RestorationState::Complete {
            return Err(RestorationError::InvalidObligation);
        }
        if self
            .obligations
            .iter()
            .any(|existing| existing.id == obligation.id)
        {
            return Err(RestorationError::DuplicateId);
        }
        self.obligations.push(obligation);
        self.obligations.sort_by_key(|entry| entry.id);
        Ok(())
    }

    pub fn record_progress(
        &mut self,
        id: u64,
        quantity: f64,
        step: u64,
    ) -> Result<(), RestorationError> {
        if !quantity.is_finite() || quantity <= 0.0 {
            return Err(RestorationError::InvalidProgress);
        }
        let obligation = self
            .obligations
            .iter_mut()
            .find(|entry| entry.id == id)
            .ok_or(RestorationError::UnknownObligation)?;
        if obligation.state == RestorationState::Complete {
            return Err(RestorationError::TerminalObligation);
        }
        obligation.completed_quantity =
            (obligation.completed_quantity + quantity).min(obligation.required_quantity);
        obligation.state =
            if obligation.completed_quantity + f64::EPSILON >= obligation.required_quantity {
                RestorationState::AwaitingEvidence
            } else {
                RestorationState::InProgress
            };
        obligation.last_update_step = step;
        Ok(())
    }

    pub fn attest_completion(
        &mut self,
        id: u64,
        evidence_id: impl Into<String>,
        externally_verified: bool,
        step: u64,
    ) -> Result<(), RestorationError> {
        let evidence_id = evidence_id.into();
        if !externally_verified || evidence_id.trim().is_empty() {
            return Err(RestorationError::InvalidEvidence);
        }
        let obligation = self
            .obligations
            .iter_mut()
            .find(|entry| entry.id == id)
            .ok_or(RestorationError::UnknownObligation)?;
        if obligation.state == RestorationState::Complete {
            return Err(RestorationError::TerminalObligation);
        }
        if obligation.completed_quantity + f64::EPSILON < obligation.required_quantity {
            return Err(RestorationError::EvidenceBeforeCompletion);
        }
        obligation.completion_evidence_id = Some(evidence_id);
        obligation.state = RestorationState::Complete;
        obligation.last_update_step = step;
        Ok(())
    }

    pub fn assess(&self, step: u64) -> RestorationAssessment {
        if self.obligations.is_empty() {
            return RestorationAssessment::clear();
        }
        let open = self
            .obligations
            .iter()
            .filter(|entry| entry.state != RestorationState::Complete)
            .count();
        let overdue = self
            .obligations
            .iter()
            .filter(|entry| entry.state != RestorationState::Complete && step > entry.due_step)
            .count();
        let required = self
            .obligations
            .iter()
            .map(|entry| entry.required_quantity)
            .sum::<f64>();
        let completed = self
            .obligations
            .iter()
            .map(|entry| entry.completed_quantity)
            .sum::<f64>();
        let completion_fraction = if required > 0.0 {
            (completed / required).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let due = self.obligations.iter().any(|entry| {
            entry.state != RestorationState::Complete
                && step.saturating_add(1_000) >= entry.due_step
        });
        let disposition = if overdue > 0 {
            RestorationDisposition::RestorationOverdue
        } else if due {
            RestorationDisposition::RestorationDue
        } else if open > 0 {
            RestorationDisposition::ObligationsOpen
        } else {
            RestorationDisposition::Clear
        };
        RestorationAssessment {
            disposition,
            open_obligations: open,
            overdue_obligations: overdue,
            completion_fraction,
            new_productive_work_allowed: !matches!(
                disposition,
                RestorationDisposition::RestorationDue | RestorationDisposition::RestorationOverdue
            ),
            return_required: disposition == RestorationDisposition::RestorationOverdue,
        }
    }

    pub fn validate(&self) -> bool {
        self.schema_version == RESTORATION_STEWARDSHIP_SCHEMA_VERSION
            && self.obligations.len() <= MAX_RESTORATION_OBLIGATIONS
            && self.obligations.iter().all(RestorationObligation::validate)
            && self
                .obligations
                .windows(2)
                .all(|window| window[0].id < window[1].id)
    }
}

impl Default for RestorationLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obligation(id: u64, due_step: u64) -> RestorationObligation {
        RestorationObligation {
            id,
            kind: RestorationObligationKind::TreatWater,
            node: Some(TunnelNodeId(1)),
            required_quantity: 10.0,
            completed_quantity: 0.0,
            due_step,
            state: RestorationState::Open,
            authority_reference: "permit-restoration".into(),
            completion_evidence_id: None,
            last_update_step: 0,
        }
    }

    #[test]
    fn completion_requires_physical_progress_and_external_evidence() {
        let mut ledger = RestorationLedger::new();
        ledger.add(obligation(1, 100)).expect("valid obligation");
        assert_eq!(
            ledger.attest_completion(1, "evidence", true, 2),
            Err(RestorationError::EvidenceBeforeCompletion)
        );
        ledger.record_progress(1, 10.0, 3).expect("progress");
        ledger
            .attest_completion(1, "observer-evidence", true, 4)
            .expect("completion");
        assert_eq!(ledger.assess(4).disposition, RestorationDisposition::Clear);
    }

    #[test]
    fn overdue_obligation_blocks_new_work_and_requires_return() {
        let mut ledger = RestorationLedger::new();
        ledger.add(obligation(1, 10)).expect("valid obligation");
        let assessment = ledger.assess(11);
        assert_eq!(
            assessment.disposition,
            RestorationDisposition::RestorationOverdue
        );
        assert!(!assessment.new_productive_work_allowed);
        assert!(assessment.return_required);
    }
}
