// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Replay-resistant rescue-consent continuity.
//!
//! Acceptance of a rescue offer is explicit consent for that case. A later,
//! fresher refusal or withdrawal removes rescue authority and remains effective
//! until superseded by a fresher explicit consent statement or the case reaches
//! a terminal handoff state. This module does not infer consent from distress,
//! identity, role, medical condition, or silence. Authentication of the subject
//! statement remains an upstream responsibility.

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

    pub const fn is_negative(self) -> bool {
        matches!(self, Self::Refuse | Self::Withdraw)
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
    CapacityExceeded,
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
            // Never silently evict a refusal/withdrawal barrier. Losing the latest
            // case-specific negative decision could make an Accepted/Active handoff
            // appear consented again. Admission therefore fails closed at capacity;
            // terminal cases must be released explicitly.
            return Err(RescueConsentRejection::CapacityExceeded);
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
            match statement.decision {
                // Positive statement authority remains time-bounded. Once it
                // expires, the handoff state may still carry its own explicit
                // acceptance semantics.
                RescueConsentDecision::Consent if current_step <= statement.expires_step => {
                    return RescueConsentDisposition::Consented;
                }
                RescueConsentDecision::Consent => {}
                // A refusal/withdrawal is a revocation barrier, not temporary
                // positive authority. It must not expire back into a previous
                // Accepted/Active handoff. A fresher Consent statement can
                // supersede it through the normal epoch/sequence rules.
                RescueConsentDecision::Refuse => {
                    return RescueConsentDisposition::Refused;
                }
                RescueConsentDecision::Withdraw => {
                    return RescueConsentDisposition::Withdrawn;
                }
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

    /// Drop expired positive consent statements while retaining negative
    /// revocation barriers. Refusals and withdrawals are removed only when
    /// superseded by a fresher statement for the same case or when a caller
    /// explicitly releases a terminal case.
    pub fn expire(&mut self, current_step: u64) {
        self.records.retain(|_, statement| {
            statement.decision.is_negative() || current_step <= statement.expires_step
        });
    }

    /// Release consent continuity only after the rescue handoff has reached a
    /// terminal state. This provides bounded cleanup without allowing active
    /// authority to resurrect by eviction or expiry.
    pub fn release_terminal_case(
        &mut self,
        subject: AgentId,
        case_id: RescueCaseId,
        handoff_state: RescueHandoffState,
    ) -> bool {
        if !matches!(
            handoff_state,
            RescueHandoffState::Completed | RescueHandoffState::Aborted
        ) {
            return false;
        }
        self.records.remove(&(subject, case_id)).is_some()
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
    fn withdrawal_does_not_expire_back_into_active_handoff() {
        let mut ledger = RescueConsentLedger::new();
        ledger
            .ingest(statement(1, RescueConsentDecision::Withdraw), 20)
            .assert_ok();
        assert_eq!(
            ledger.disposition(
                AgentId::new(2),
                RescueCaseId(9),
                201,
                RescueHandoffState::Active,
            ),
            RescueConsentDisposition::Withdrawn
        );
    }

    #[test]
    fn expiration_retains_negative_continuity() {
        let mut ledger = RescueConsentLedger::new();
        ledger
            .ingest(statement(1, RescueConsentDecision::Refuse), 20)
            .assert_ok();
        ledger.expire(201);
        assert_eq!(
            ledger.disposition(
                AgentId::new(2),
                RescueCaseId(9),
                201,
                RescueHandoffState::Accepted,
            ),
            RescueConsentDisposition::Refused
        );
        assert!(ledger.validate());
    }

    #[test]
    fn fresh_consent_supersedes_prior_withdrawal() {
        let mut ledger = RescueConsentLedger::new();
        ledger
            .ingest(statement(1, RescueConsentDecision::Withdraw), 20)
            .assert_ok();
        ledger
            .ingest(statement(2, RescueConsentDecision::Consent), 21)
            .assert_ok();
        assert_eq!(
            ledger.disposition(
                AgentId::new(2),
                RescueCaseId(9),
                22,
                RescueHandoffState::Active,
            ),
            RescueConsentDisposition::Consented
        );
    }

    #[test]
    fn terminal_release_is_required_for_negative_cleanup() {
        let mut ledger = RescueConsentLedger::new();
        ledger
            .ingest(statement(1, RescueConsentDecision::Withdraw), 20)
            .assert_ok();
        assert!(!ledger.release_terminal_case(
            AgentId::new(2),
            RescueCaseId(9),
            RescueHandoffState::Active,
        ));
        assert_eq!(
            ledger.disposition(
                AgentId::new(2),
                RescueCaseId(9),
                201,
                RescueHandoffState::Active,
            ),
            RescueConsentDisposition::Withdrawn
        );
        assert!(ledger.release_terminal_case(
            AgentId::new(2),
            RescueCaseId(9),
            RescueHandoffState::Aborted,
        ));
        assert_eq!(
            ledger.disposition(
                AgentId::new(2),
                RescueCaseId(9),
                201,
                RescueHandoffState::Aborted,
            ),
            RescueConsentDisposition::Unknown
        );
    }

    #[test]
    fn ledger_capacity_fails_closed_without_evicting_negative_record() {
        let mut ledger = RescueConsentLedger::new();
        for i in 0..MAX_RESCUE_CONSENT_RECORDS {
            let statement = RescueConsentStatement {
                schema_version: RESCUE_CONSENT_SCHEMA_VERSION,
                subject: AgentId::new((i + 2) as u64),
                case_id: RescueCaseId((i + 1) as u64),
                epoch: 1,
                sequence: 1,
                issued_step: 10,
                expires_step: 200,
                decision: RescueConsentDecision::Withdraw,
            };
            ledger.ingest(statement, 20).assert_ok();
        }
        let overflow = RescueConsentStatement {
            schema_version: RESCUE_CONSENT_SCHEMA_VERSION,
            subject: AgentId::new(200),
            case_id: RescueCaseId(200),
            epoch: 1,
            sequence: 1,
            issued_step: 10,
            expires_step: 200,
            decision: RescueConsentDecision::Consent,
        };
        assert_eq!(
            ledger.ingest(overflow, 20),
            Err(RescueConsentRejection::CapacityExceeded)
        );
        assert_eq!(
            ledger.disposition(
                AgentId::new(2),
                RescueCaseId(1),
                201,
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
