// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Replay-resistant rescue-consent continuity.
//!
//! Acceptance of a rescue offer is explicit consent for that case. A later,
//! fresher refusal or withdrawal removes rescue authority. This module does not
//! infer consent from distress, identity, role, medical condition, or silence.
//! Authentication of the subject statement remains an upstream responsibility.

use crate::rescue::{RescueCaseId, RescueHandoffState};
use crate::team::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RESCUE_CONSENT_SCHEMA_VERSION: u16 = 1;
pub const MAX_RESCUE_CONSENT_RECORDS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueConsentDecision {
    Consent,
    Refuse,
    Withdraw,
}

impl RescueConsentDecision {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Consent => "consent",
            Self::Refuse => "refuse",
            Self::Withdraw => "withdraw",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueConsentStatement {
    pub schema_version: u16,
    pub subject: AgentId,
    pub case_id: RescueCaseId,
    pub epoch: u32,
    pub sequence: u64,
    pub issued_step: u64,
    pub expires_step: u64,
    pub decision: RescueConsentDecision,
}

impl RescueConsentStatement {
    pub fn validate(self) -> bool {
        self.schema_version == RESCUE_CONSENT_SCHEMA_VERSION
            && self.subject != AgentId::SURFACE_CONTROL
            && self.case_id.0 != 0
            && self.expires_step >= self.issued_step
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueConsentDisposition {
    Unknown,
    Consented,
    Refused,
    Withdrawn,
}

impl RescueConsentDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Consented => "consented",
            Self::Refused => "refused",
            Self::Withdrawn => "withdrawn",
        }
    }

    pub const fn permits_rescue(self) -> bool {
        matches!(self, Self::Consented)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RescueConsentRejection {
    InvalidStatement,
    Replay,
    EpochRegression,
    Expired,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RescueConsentLedger {
    records: BTreeMap<(AgentId, RescueCaseId), RescueConsentStatement>,
}

impl RescueConsentLedger {
    pub fn new() -> Self {
        Self {
            records: BTreeMap::new(),
        }
    }

    pub fn ingest(
        &mut self,
        statement: RescueConsentStatement,
        current_step: u64,
    ) -> Result<(), RescueConsentRejection> {
        if !statement.validate() {
            return Err(RescueConsentRejection::InvalidStatement);
        }
        if current_step > statement.expires_step {
            return Err(RescueConsentRejection::Expired);
        }
        let key = (statement.subject, statement.case_id);
        if let Some(previous) = self.records.get(&key) {
            if statement.epoch < previous.epoch {
                return Err(RescueConsentRejection::EpochRegression);
            }
            if statement.epoch == previous.epoch && statement.sequence <= previous.sequence {
                return Err(RescueConsentRejection::Replay);
            }
        }
        if self.records.len() >= MAX_RESCUE_CONSENT_RECORDS && !self.records.contains_key(&key) {
            let oldest = self
                .records
                .iter()
                .min_by_key(|(_, record)| (record.expires_step, record.issued_step))
                .map(|(key, _)| *key);
            if let Some(oldest) = oldest {
                self.records.remove(&oldest);
            }
        }
        self.records.insert(key, statement);
        Ok(())
    }

    pub fn disposition(
        &self,
        subject: AgentId,
        case_id: RescueCaseId,
        current_step: u64,
        handoff_state: RescueHandoffState,
    ) -> RescueConsentDisposition {
        if let Some(statement) = self.records.get(&(subject, case_id)) {
            if current_step <= statement.expires_step {
                return match statement.decision {
                    RescueConsentDecision::Consent => RescueConsentDisposition::Consented,
                    RescueConsentDecision::Refuse => RescueConsentDisposition::Refused,
                    RescueConsentDecision::Withdraw => RescueConsentDisposition::Withdrawn,
                };
            }
        }
        if matches!(
            handoff_state,
            RescueHandoffState::Accepted | RescueHandoffState::Active
        ) {
            RescueConsentDisposition::Consented
        } else {
            RescueConsentDisposition::Unknown
        }
    }

    pub fn expire(&mut self, current_step: u64) {
        self.records
            .retain(|_, statement| current_step <= statement.expires_step);
    }

    pub fn validate(&self) -> bool {
        self.records.len() <= MAX_RESCUE_CONSENT_RECORDS
            && self.records.iter().all(|((subject, case_id), statement)| {
                *subject == statement.subject
                    && *case_id == statement.case_id
                    && statement.validate()
            })
    }
}

impl Default for RescueConsentLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn statement(sequence: u64, decision: RescueConsentDecision) -> RescueConsentStatement {
        RescueConsentStatement {
            schema_version: RESCUE_CONSENT_SCHEMA_VERSION,
            subject: AgentId::new(2),
            case_id: RescueCaseId(9),
            epoch: 1,
            sequence,
            issued_step: 10 + sequence,
            expires_step: 200,
            decision,
        }
    }

    #[test]
    fn withdrawal_overrides_prior_acceptance() {
        let mut ledger = RescueConsentLedger::new();
        ledger
            .ingest(statement(1, RescueConsentDecision::Consent), 20)
            .assert_ok();
        ledger
            .ingest(statement(2, RescueConsentDecision::Withdraw), 21)
            .assert_ok();
        assert_eq!(
            ledger.disposition(
                AgentId::new(2),
                RescueCaseId(9),
                22,
                RescueHandoffState::Active,
            ),
            RescueConsentDisposition::Withdrawn
        );
    }

    #[test]
    fn accepted_handoff_counts_as_case_specific_consent() {
        let ledger = RescueConsentLedger::new();
        assert_eq!(
            ledger.disposition(
                AgentId::new(2),
                RescueCaseId(9),
                22,
                RescueHandoffState::Accepted,
            ),
            RescueConsentDisposition::Consented
        );
    }

    trait AssertOk {
        fn assert_ok(self);
    }

    impl<T, E: core::fmt::Debug> AssertOk for Result<T, E> {
        fn assert_ok(self) {
            assert!(self.is_ok(), "expected Ok, got {self:?}");
        }
    }
}
