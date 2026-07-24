// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Replay-resistant operator challenge workflow for recorded autonomy decisions.
//!
//! Challenges can demand explanation, correction of evidence, or independent
//! review. They cannot mutate the recorded command or weaken safety authority.

use crate::counterfactual_explanation::CounterfactualQuestion;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

pub const OPERATOR_CHALLENGE_SCHEMA_VERSION: u16 = 1;
pub const MAX_OPERATOR_CHALLENGES: usize = 128;
pub const MAX_CHALLENGE_TEXT: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChallengeRole {
    Operator,
    SafetyOfficer,
    VerificationAuthority,
    CommunityObserver,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChallengeKind {
    WhyDecision,
    WhyNotAlternative,
    EvidenceCorrection,
    NearMissReview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChallengeDisposition {
    Open,
    Upheld,
    ExplanationConfirmed,
    EvidenceCorrected,
    InsufficientEvidence,
    Rejected,
}

impl ChallengeDisposition {
    pub const fn is_closed(self) -> bool {
        !matches!(self, Self::Open)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChallengeEnvelope {
    pub schema_version: u16,
    pub challenge_id: u64,
    pub challenger_id: String,
    pub role: ChallengeRole,
    pub hardware_backed: bool,
    pub externally_authenticated: bool,
    pub epoch: u64,
    pub sequence: u64,
    pub issued_step: u64,
    pub expires_step: u64,
    pub decision_step: u64,
    pub kind: ChallengeKind,
    pub question: Option<CounterfactualQuestion>,
    pub statement: String,
}

impl ChallengeEnvelope {
    pub fn validate(&self) -> bool {
        self.schema_version == OPERATOR_CHALLENGE_SCHEMA_VERSION
            && self.challenge_id > 0
            && !self.challenger_id.trim().is_empty()
            && self.hardware_backed
            && self.externally_authenticated
            && self.sequence > 0
            && self.expires_step >= self.issued_step
            && self.decision_step <= self.issued_step
            && !self.statement.trim().is_empty()
            && self.statement.len() <= MAX_CHALLENGE_TEXT
            && (self.kind != ChallengeKind::WhyNotAlternative || self.question.is_some())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChallengeResponse {
    pub responder_id: String,
    pub role: ChallengeRole,
    pub hardware_backed: bool,
    pub externally_authenticated: bool,
    pub response_step: u64,
    pub disposition: ChallengeDisposition,
    pub explanation_reference: String,
    pub corrective_evidence_reference: Option<String>,
    pub rationale: String,
}

impl ChallengeResponse {
    pub fn validate(&self) -> bool {
        !self.responder_id.trim().is_empty()
            && self.hardware_backed
            && self.externally_authenticated
            && self.disposition.is_closed()
            && !self.explanation_reference.trim().is_empty()
            && !self.rationale.trim().is_empty()
            && self.rationale.len() <= MAX_CHALLENGE_TEXT
            && (self.disposition != ChallengeDisposition::EvidenceCorrected
                || self
                    .corrective_evidence_reference
                    .as_ref()
                    .is_some_and(|reference| !reference.trim().is_empty()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChallengeRecord {
    pub envelope: ChallengeEnvelope,
    pub disposition: ChallengeDisposition,
    pub response: Option<ChallengeResponse>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeRejection {
    InvalidEnvelope,
    Expired,
    Replay,
    DuplicateId,
    Capacity,
    UnknownChallenge,
    AlreadyClosed,
    InvalidResponse,
    SelfReviewNotIndependent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorChallengeLedger {
    pub schema_version: u16,
    records: VecDeque<ChallengeRecord>,
    last_epoch_sequence: BTreeMap<String, (u64, u64)>,
    total_accepted: u64,
    total_rejected: u64,
    total_corrected: u64,
}

impl Default for OperatorChallengeLedger {
    fn default() -> Self {
        Self {
            schema_version: OPERATOR_CHALLENGE_SCHEMA_VERSION,
            records: VecDeque::new(),
            last_epoch_sequence: BTreeMap::new(),
            total_accepted: 0,
            total_rejected: 0,
            total_corrected: 0,
        }
    }
}

impl OperatorChallengeLedger {
    pub fn submit(
        &mut self,
        envelope: ChallengeEnvelope,
        now_step: u64,
    ) -> Result<(), ChallengeRejection> {
        if !envelope.validate() {
            return self.reject(ChallengeRejection::InvalidEnvelope);
        }
        if now_step > envelope.expires_step {
            return self.reject(ChallengeRejection::Expired);
        }
        if self
            .records
            .iter()
            .any(|record| record.envelope.challenge_id == envelope.challenge_id)
        {
            return self.reject(ChallengeRejection::DuplicateId);
        }
        if self.records.len() >= MAX_OPERATOR_CHALLENGES {
            return self.reject(ChallengeRejection::Capacity);
        }
        if let Some((epoch, sequence)) = self.last_epoch_sequence.get(&envelope.challenger_id) {
            if envelope.epoch < *epoch
                || (envelope.epoch == *epoch && envelope.sequence <= *sequence)
            {
                return self.reject(ChallengeRejection::Replay);
            }
        }
        self.last_epoch_sequence.insert(
            envelope.challenger_id.clone(),
            (envelope.epoch, envelope.sequence),
        );
        self.records.push_back(ChallengeRecord {
            envelope,
            disposition: ChallengeDisposition::Open,
            response: None,
        });
        self.total_accepted = self.total_accepted.saturating_add(1);
        Ok(())
    }

    pub fn respond(
        &mut self,
        challenge_id: u64,
        response: ChallengeResponse,
    ) -> Result<(), ChallengeRejection> {
        if !response.validate() {
            return self.reject(ChallengeRejection::InvalidResponse);
        }
        let Some(record) = self
            .records
            .iter_mut()
            .find(|record| record.envelope.challenge_id == challenge_id)
        else {
            return self.reject(ChallengeRejection::UnknownChallenge);
        };
        if record.disposition.is_closed() {
            return self.reject(ChallengeRejection::AlreadyClosed);
        }
        if response.responder_id == record.envelope.challenger_id
            && !matches!(record.envelope.kind, ChallengeKind::EvidenceCorrection)
        {
            return self.reject(ChallengeRejection::SelfReviewNotIndependent);
        }
        if response.disposition == ChallengeDisposition::EvidenceCorrected {
            self.total_corrected = self.total_corrected.saturating_add(1);
        }
        record.disposition = response.disposition;
        record.response = Some(response);
        Ok(())
    }

    pub fn record(&self, challenge_id: u64) -> Option<&ChallengeRecord> {
        self.records
            .iter()
            .find(|record| record.envelope.challenge_id == challenge_id)
    }

    pub fn open_count(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.disposition == ChallengeDisposition::Open)
            .count()
    }

    pub fn total_accepted(&self) -> u64 {
        self.total_accepted
    }

    pub fn total_rejected(&self) -> u64 {
        self.total_rejected
    }

    pub fn total_corrected(&self) -> u64 {
        self.total_corrected
    }

    pub fn validate(&self) -> bool {
        if self.schema_version != OPERATOR_CHALLENGE_SCHEMA_VERSION
            || self.records.len() > MAX_OPERATOR_CHALLENGES
        {
            return false;
        }
        let mut ids = std::collections::BTreeSet::new();
        self.records.iter().all(|record| {
            ids.insert(record.envelope.challenge_id)
                && record.envelope.validate()
                && match (&record.response, record.disposition) {
                    (None, ChallengeDisposition::Open) => true,
                    (Some(response), disposition) => {
                        disposition == response.disposition && response.validate()
                    }
                    _ => false,
                }
        })
    }

    fn reject<T>(&mut self, rejection: ChallengeRejection) -> Result<T, ChallengeRejection> {
        self.total_rejected = self.total_rejected.saturating_add(1);
        Err(rejection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(sequence: u64) -> ChallengeEnvelope {
        ChallengeEnvelope {
            schema_version: OPERATOR_CHALLENGE_SCHEMA_VERSION,
            challenge_id: sequence,
            challenger_id: "operator-a".into(),
            role: ChallengeRole::Operator,
            hardware_backed: true,
            externally_authenticated: true,
            epoch: 1,
            sequence,
            issued_step: 100,
            expires_step: 200,
            decision_step: 90,
            kind: ChallengeKind::WhyDecision,
            question: None,
            statement: "Explain the hold decision".into(),
        }
    }

    #[test]
    fn stale_sequence_is_rejected() {
        let mut ledger = OperatorChallengeLedger::default();
        assert!(ledger.submit(envelope(2), 100).is_ok());
        let mut stale = envelope(1);
        stale.challenge_id = 3;
        assert_eq!(ledger.submit(stale, 100), Err(ChallengeRejection::Replay));
    }

    #[test]
    fn independent_response_closes_challenge() {
        let mut ledger = OperatorChallengeLedger::default();
        ledger.submit(envelope(1), 100).unwrap();
        ledger
            .respond(
                1,
                ChallengeResponse {
                    responder_id: "safety-b".into(),
                    role: ChallengeRole::SafetyOfficer,
                    hardware_backed: true,
                    externally_authenticated: true,
                    response_step: 110,
                    disposition: ChallengeDisposition::ExplanationConfirmed,
                    explanation_reference: "trace:90".into(),
                    corrective_evidence_reference: None,
                    rationale: "The recorded hazard required hold.".into(),
                },
            )
            .unwrap();
        assert_eq!(
            ledger.record(1).unwrap().disposition,
            ChallengeDisposition::ExplanationConfirmed
        );
        assert!(ledger.validate());
    }
}
