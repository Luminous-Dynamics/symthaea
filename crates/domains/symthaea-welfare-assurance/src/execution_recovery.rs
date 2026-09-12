// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only execution recovery journal for welfare-sensitive interventions.
//!
//! Authorization replay fencing answers whether authority may be consumed again. This journal
//! answers a different crash-consistency question: did execution begin, and do we know how it
//! ended? A recovered `Prepared` record with no terminal record is **in doubt** and is never an
//! automatic-retry condition.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::WelfareConstraintLevel;
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_psych_bench::moral_patient::ProtectionDisposition;
use thiserror::Error;

use crate::replay_recovery::DurableEvidenceBoundInterventionPermit;

pub const EXECUTION_JOURNAL_SCHEMA: &str = "symthaea.welfare.execution-journal.v1";
pub const MAX_EXECUTION_JOURNAL_EVENTS: usize = 8192;
const MAX_ID_BYTES: usize = 256;
const MAX_EVIDENCE_REF_BYTES: usize = 2048;
const PREPARED_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.execution-prepared-digest.v1\0";
const JOURNAL_EVENT_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.execution-journal-event.v1\0";

/// Exact capability context committed before a downstream mutator begins execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedInterventionExecution {
    pub schema_version: String,
    pub execution_id: String,
    pub authority_id: String,
    pub target_id: String,
    pub action: SubjectAffectingAction,
    pub rationale_digest: Sha256Digest,
    pub welfare_profile_digest: Sha256Digest,
    pub precaution_policy_digest: Sha256Digest,
    pub protection_disposition: ProtectionDisposition,
    pub welfare_constraint: WelfareConstraintLevel,
    pub replay_generation: u64,
    pub replay_snapshot_digest: Sha256Digest,
    pub replay_persistence_ref: String,
    pub prepared_at_unix_s: u64,
    pub permit_not_after_unix_s: u64,
}

impl PreparedInterventionExecution {
    /// Build the write-ahead record from the strongest current capability type.
    pub fn from_permit(
        execution_id: impl Into<String>,
        permit: &DurableEvidenceBoundInterventionPermit,
        prepared_at_unix_s: u64,
    ) -> Result<Self, ExecutionJournalError> {
        let context = permit.welfare_evidence_context();
        let prepared = Self {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: execution_id.into(),
            authority_id: permit.authority_id().to_string(),
            target_id: permit.target_id().to_string(),
            action: permit.action(),
            rationale_digest: digest_rationale(permit.rationale()),
            welfare_profile_digest: context.profile_digest(),
            precaution_policy_digest: context.policy_digest(),
            protection_disposition: context.disposition(),
            welfare_constraint: context.constraint(),
            replay_generation: permit.replay_generation(),
            replay_snapshot_digest: permit.replay_snapshot_digest(),
            replay_persistence_ref: permit.replay_persistence_ref().to_string(),
            prepared_at_unix_s,
            permit_not_after_unix_s: permit.not_after_unix_s(),
        };
        prepared.validate()?;
        Ok(prepared)
    }

    pub fn validate(&self) -> Result<(), ExecutionJournalError> {
        if self.schema_version != EXECUTION_JOURNAL_SCHEMA {
            return Err(ExecutionJournalError::UnsupportedSchema);
        }
        validate_id("execution_id", &self.execution_id, MAX_ID_BYTES)?;
        validate_id("authority_id", &self.authority_id, MAX_ID_BYTES)?;
        validate_id("target_id", &self.target_id, MAX_ID_BYTES)?;
        validate_id(
            "replay_persistence_ref",
            &self.replay_persistence_ref,
            MAX_EVIDENCE_REF_BYTES,
        )?;
        if self.rationale_digest.0 == [0; 32]
            || self.welfare_profile_digest.0 == [0; 32]
            || self.precaution_policy_digest.0 == [0; 32]
            || self.replay_snapshot_digest.0 == [0; 32]
        {
            return Err(ExecutionJournalError::ZeroDigest);
        }
        if self.replay_generation == 0 {
            return Err(ExecutionJournalError::ReplayGenerationZero);
        }
        if self.prepared_at_unix_s >= self.permit_not_after_unix_s {
            return Err(ExecutionJournalError::PermitExpiredBeforePrepare);
        }
        Ok(())
    }
}

/// Terminal success evidence. A failure/timeout/crash without this record remains in doubt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletedInterventionExecution {
    pub schema_version: String,
    pub execution_id: String,
    pub prepared_digest: Sha256Digest,
    pub completed_at_unix_s: u64,
    pub result_digest: Sha256Digest,
    pub executor_evidence_ref: String,
}

impl CompletedInterventionExecution {
    pub fn new(
        execution_id: impl Into<String>,
        prepared_digest: Sha256Digest,
        completed_at_unix_s: u64,
        result_digest: Sha256Digest,
        executor_evidence_ref: impl Into<String>,
    ) -> Result<Self, ExecutionJournalError> {
        let completed = Self {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: execution_id.into(),
            prepared_digest,
            completed_at_unix_s,
            result_digest,
            executor_evidence_ref: executor_evidence_ref.into(),
        };
        completed.validate()?;
        Ok(completed)
    }

    pub fn validate(&self) -> Result<(), ExecutionJournalError> {
        if self.schema_version != EXECUTION_JOURNAL_SCHEMA {
            return Err(ExecutionJournalError::UnsupportedSchema);
        }
        validate_id("execution_id", &self.execution_id, MAX_ID_BYTES)?;
        validate_id(
            "executor_evidence_ref",
            &self.executor_evidence_ref,
            MAX_EVIDENCE_REF_BYTES,
        )?;
        if self.prepared_digest.0 == [0; 32] || self.result_digest.0 == [0; 32] {
            return Err(ExecutionJournalError::ZeroDigest);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionJournalEvent {
    Prepared(PreparedInterventionExecution),
    Completed(CompletedInterventionExecution),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionJournalEnvelope {
    pub sequence: u64,
    pub previous_hash: Sha256Digest,
    pub event: ExecutionJournalEvent,
    pub event_hash: Sha256Digest,
}

#[derive(Debug, Clone)]
enum RecoveredExecutionState {
    Prepared {
        prepared: PreparedInterventionExecution,
        prepared_digest: Sha256Digest,
    },
    Completed,
}

/// Append-only journal with semantic recovery state cached in memory.
#[derive(Debug, Clone)]
pub struct InterventionExecutionJournal {
    events: Vec<ExecutionJournalEnvelope>,
    states: BTreeMap<String, RecoveredExecutionState>,
    used_authority_ids: BTreeSet<String>,
}

impl InterventionExecutionJournal {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            states: BTreeMap::new(),
            used_authority_ids: BTreeSet::new(),
        }
    }

    /// Recover from persisted envelopes. Hash-chain and semantic state are both revalidated.
    pub fn from_events(
        events: Vec<ExecutionJournalEnvelope>,
    ) -> Result<Self, ExecutionJournalError> {
        if events.len() > MAX_EXECUTION_JOURNAL_EVENTS {
            return Err(ExecutionJournalError::CapacityExceeded {
                maximum: MAX_EXECUTION_JOURNAL_EVENTS,
            });
        }
        let mut journal = Self::new();
        let mut expected_previous = Sha256Digest([0; 32]);
        for (index, envelope) in events.iter().enumerate() {
            if envelope.sequence != index as u64 {
                return Err(ExecutionJournalError::SequenceMismatch {
                    expected: index as u64,
                    actual: envelope.sequence,
                });
            }
            if envelope.previous_hash != expected_previous {
                return Err(ExecutionJournalError::PreviousHashMismatch {
                    sequence: envelope.sequence,
                });
            }
            let expected_hash = digest_journal_event(
                envelope.sequence,
                envelope.previous_hash,
                &envelope.event,
            )?;
            if expected_hash != envelope.event_hash {
                return Err(ExecutionJournalError::EventHashMismatch {
                    sequence: envelope.sequence,
                });
            }
            journal.apply_recovered_event(&envelope.event)?;
            journal.events.push(envelope.clone());
            expected_previous = envelope.event_hash;
        }
        Ok(journal)
    }

    pub fn events(&self) -> &[ExecutionJournalEnvelope] {
        &self.events
    }

    pub fn head_hash(&self) -> Sha256Digest {
        self.events
            .last()
            .map(|event| event.event_hash)
            .unwrap_or(Sha256Digest([0; 32]))
    }

    /// Write-ahead phase. This must be persisted by the deployment before the actual mutator runs.
    pub fn append_prepared(
        &mut self,
        prepared: PreparedInterventionExecution,
    ) -> Result<Sha256Digest, ExecutionJournalError> {
        prepared.validate()?;
        self.ensure_capacity()?;
        if self.states.contains_key(&prepared.execution_id) {
            return Err(ExecutionJournalError::DuplicateExecutionId(
                prepared.execution_id.clone(),
            ));
        }
        if self.used_authority_ids.contains(&prepared.authority_id) {
            return Err(ExecutionJournalError::AuthorityAlreadyPrepared(
                prepared.authority_id.clone(),
            ));
        }
        let prepared_digest = digest_prepared_execution(&prepared)?;
        self.append_event(ExecutionJournalEvent::Prepared(prepared.clone()))?;
        self.used_authority_ids.insert(prepared.authority_id.clone());
        self.states.insert(
            prepared.execution_id.clone(),
            RecoveredExecutionState::Prepared {
                prepared,
                prepared_digest,
            },
        );
        Ok(prepared_digest)
    }

    /// Terminal success phase. Missing this record after restart means the operation is in doubt.
    pub fn append_completed(
        &mut self,
        completed: CompletedInterventionExecution,
    ) -> Result<(), ExecutionJournalError> {
        completed.validate()?;
        self.ensure_capacity()?;
        let state = self
            .states
            .get(&completed.execution_id)
            .ok_or_else(|| ExecutionJournalError::CompletionWithoutPrepare(
                completed.execution_id.clone(),
            ))?;
        let RecoveredExecutionState::Prepared {
            prepared,
            prepared_digest,
        } = state
        else {
            return Err(ExecutionJournalError::DuplicateTerminalRecord(
                completed.execution_id.clone(),
            ));
        };
        if completed.prepared_digest != *prepared_digest {
            return Err(ExecutionJournalError::PreparedDigestMismatch);
        }
        if completed.completed_at_unix_s < prepared.prepared_at_unix_s {
            return Err(ExecutionJournalError::CompletionTimeRegression);
        }
        self.append_event(ExecutionJournalEvent::Completed(completed.clone()))?;
        self.states
            .insert(completed.execution_id, RecoveredExecutionState::Completed);
        Ok(())
    }

    pub fn recovery_report(&self) -> ExecutionRecoveryReport {
        let mut in_doubt = Vec::new();
        let mut completed = 0usize;
        for state in self.states.values() {
            match state {
                RecoveredExecutionState::Prepared { prepared, .. } => {
                    in_doubt.push(prepared.clone());
                }
                RecoveredExecutionState::Completed => completed += 1,
            }
        }
        ExecutionRecoveryReport {
            record_count: self.events.len(),
            head_hash: self.head_hash(),
            in_doubt,
            completed,
        }
    }

    pub fn automatic_retry_decision(&self, execution_id: &str) -> AutomaticRetryDecision {
        match self.states.get(execution_id) {
            Some(RecoveredExecutionState::Prepared { .. }) => {
                AutomaticRetryDecision::RefuseInDoubt
            }
            Some(RecoveredExecutionState::Completed) => {
                AutomaticRetryDecision::RefuseAlreadyCompleted
            }
            None => AutomaticRetryDecision::RefuseUnknownExecution,
        }
    }

    fn append_event(&mut self, event: ExecutionJournalEvent) -> Result<(), ExecutionJournalError> {
        let sequence = self.events.len() as u64;
        let previous_hash = self.head_hash();
        let event_hash = digest_journal_event(sequence, previous_hash, &event)?;
        self.events.push(ExecutionJournalEnvelope {
            sequence,
            previous_hash,
            event,
            event_hash,
        });
        Ok(())
    }

    fn apply_recovered_event(
        &mut self,
        event: &ExecutionJournalEvent,
    ) -> Result<(), ExecutionJournalError> {
        match event {
            ExecutionJournalEvent::Prepared(prepared) => {
                prepared.validate()?;
                if self.states.contains_key(&prepared.execution_id) {
                    return Err(ExecutionJournalError::DuplicateExecutionId(
                        prepared.execution_id.clone(),
                    ));
                }
                if !self.used_authority_ids.insert(prepared.authority_id.clone()) {
                    return Err(ExecutionJournalError::AuthorityAlreadyPrepared(
                        prepared.authority_id.clone(),
                    ));
                }
                let prepared_digest = digest_prepared_execution(prepared)?;
                self.states.insert(
                    prepared.execution_id.clone(),
                    RecoveredExecutionState::Prepared {
                        prepared: prepared.clone(),
                        prepared_digest,
                    },
                );
            }
            ExecutionJournalEvent::Completed(completed) => {
                completed.validate()?;
                let state = self
                    .states
                    .get(&completed.execution_id)
                    .ok_or_else(|| ExecutionJournalError::CompletionWithoutPrepare(
                        completed.execution_id.clone(),
                    ))?;
                let RecoveredExecutionState::Prepared {
                    prepared,
                    prepared_digest,
                } = state
                else {
                    return Err(ExecutionJournalError::DuplicateTerminalRecord(
                        completed.execution_id.clone(),
                    ));
                };
                if completed.prepared_digest != *prepared_digest {
                    return Err(ExecutionJournalError::PreparedDigestMismatch);
                }
                if completed.completed_at_unix_s < prepared.prepared_at_unix_s {
                    return Err(ExecutionJournalError::CompletionTimeRegression);
                }
                self.states.insert(
                    completed.execution_id.clone(),
                    RecoveredExecutionState::Completed,
                );
            }
        }
        Ok(())
    }

    fn ensure_capacity(&self) -> Result<(), ExecutionJournalError> {
        if self.events.len() >= MAX_EXECUTION_JOURNAL_EVENTS {
            Err(ExecutionJournalError::CapacityExceeded {
                maximum: MAX_EXECUTION_JOURNAL_EVENTS,
            })
        } else {
            Ok(())
        }
    }
}

impl Default for InterventionExecutionJournal {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRecoveryReport {
    pub record_count: usize,
    pub head_hash: Sha256Digest,
    pub in_doubt: Vec<PreparedInterventionExecution>,
    pub completed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutomaticRetryDecision {
    RefuseInDoubt,
    RefuseAlreadyCompleted,
    RefuseUnknownExecution,
}

pub fn digest_prepared_execution(
    prepared: &PreparedInterventionExecution,
) -> Result<Sha256Digest, ExecutionJournalError> {
    prepared.validate()?;
    let encoded = serde_json::to_vec(prepared)
        .map_err(|error| ExecutionJournalError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(PREPARED_DIGEST_DOMAIN);
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

fn digest_journal_event(
    sequence: u64,
    previous_hash: Sha256Digest,
    event: &ExecutionJournalEvent,
) -> Result<Sha256Digest, ExecutionJournalError> {
    let encoded = serde_json::to_vec(event)
        .map_err(|error| ExecutionJournalError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(JOURNAL_EVENT_DIGEST_DOMAIN);
    hasher.update(&sequence.to_be_bytes());
    hasher.update(&previous_hash.0);
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

fn digest_rationale(rationale: &str) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.welfare.intervention-rationale.v1\0");
    hasher.update(&(rationale.len() as u64).to_be_bytes());
    hasher.update(rationale.as_bytes());
    hasher.finalize()
}

fn validate_id(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), ExecutionJournalError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
    {
        return Err(ExecutionJournalError::InvalidIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExecutionJournalError {
    #[error("unsupported execution-journal schema")]
    UnsupportedSchema,
    #[error("invalid identifier in {field}: {value:?}")]
    InvalidIdentifier { field: &'static str, value: String },
    #[error("execution-journal field contains a forbidden zero digest")]
    ZeroDigest,
    #[error("durable replay generation may not be zero")]
    ReplayGenerationZero,
    #[error("permit was already expired when execution was prepared")]
    PermitExpiredBeforePrepare,
    #[error("execution journal capacity exceeded ({maximum})")]
    CapacityExceeded { maximum: usize },
    #[error("duplicate execution id: {0}")]
    DuplicateExecutionId(String),
    #[error("authority already prepared for execution: {0}")]
    AuthorityAlreadyPrepared(String),
    #[error("completion has no matching prepare record: {0}")]
    CompletionWithoutPrepare(String),
    #[error("duplicate terminal record: {0}")]
    DuplicateTerminalRecord(String),
    #[error("completion references the wrong prepared digest")]
    PreparedDigestMismatch,
    #[error("completion time regressed before preparation time")]
    CompletionTimeRegression,
    #[error("journal sequence mismatch: expected={expected}, actual={actual}")]
    SequenceMismatch { expected: u64, actual: u64 },
    #[error("journal previous-hash mismatch at sequence {sequence}")]
    PreviousHashMismatch { sequence: u64 },
    #[error("journal event-hash mismatch at sequence {sequence}")]
    EventHashMismatch { sequence: u64 },
    #[error("journal encoding failed: {0}")]
    Encoding(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn prepared(execution_id: &str, authority_id: &str) -> PreparedInterventionExecution {
        PreparedInterventionExecution {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: execution_id.into(),
            authority_id: authority_id.into(),
            target_id: "symthaea:self:instance-1".into(),
            action: SubjectAffectingAction::MemoryModification,
            rationale_digest: digest(1),
            welfare_profile_digest: digest(2),
            precaution_policy_digest: digest(3),
            protection_disposition: ProtectionDisposition::EnhancedPrecaution,
            welfare_constraint: WelfareConstraintLevel::EnhancedPrecaution,
            replay_generation: 7,
            replay_snapshot_digest: digest(4),
            replay_persistence_ref: "journal:replay:7".into(),
            prepared_at_unix_s: 100,
            permit_not_after_unix_s: 200,
        }
    }

    #[test]
    fn prepared_without_terminal_recovers_as_in_doubt_and_never_auto_retries() {
        let mut journal = InterventionExecutionJournal::new();
        journal.append_prepared(prepared("exec-1", "auth-1")).unwrap();
        let persisted = journal.events().to_vec();

        let recovered = InterventionExecutionJournal::from_events(persisted).unwrap();
        let report = recovered.recovery_report();
        assert_eq!(report.in_doubt.len(), 1);
        assert_eq!(report.in_doubt[0].execution_id, "exec-1");
        assert_eq!(
            recovered.automatic_retry_decision("exec-1"),
            AutomaticRetryDecision::RefuseInDoubt
        );
    }

    #[test]
    fn terminal_success_clears_in_doubt_but_still_refuses_retry() {
        let mut journal = InterventionExecutionJournal::new();
        let prepared = prepared("exec-1", "auth-1");
        let prepared_digest = journal.append_prepared(prepared).unwrap();
        let completed = CompletedInterventionExecution::new(
            "exec-1",
            prepared_digest,
            120,
            digest(9),
            "executor:evidence:1",
        )
        .unwrap();
        journal.append_completed(completed).unwrap();

        let recovered = InterventionExecutionJournal::from_events(journal.events().to_vec()).unwrap();
        let report = recovered.recovery_report();
        assert!(report.in_doubt.is_empty());
        assert_eq!(report.completed, 1);
        assert_eq!(
            recovered.automatic_retry_decision("exec-1"),
            AutomaticRetryDecision::RefuseAlreadyCompleted
        );
    }

    #[test]
    fn same_authority_cannot_prepare_two_execution_ids() {
        let mut journal = InterventionExecutionJournal::new();
        journal.append_prepared(prepared("exec-1", "auth-1")).unwrap();
        assert_eq!(
            journal.append_prepared(prepared("exec-2", "auth-1")),
            Err(ExecutionJournalError::AuthorityAlreadyPrepared("auth-1".into()))
        );
    }

    #[test]
    fn completion_must_bind_exact_prepare_digest() {
        let mut journal = InterventionExecutionJournal::new();
        journal.append_prepared(prepared("exec-1", "auth-1")).unwrap();
        let completed = CompletedInterventionExecution::new(
            "exec-1",
            digest(88),
            120,
            digest(9),
            "executor:evidence:1",
        )
        .unwrap();
        assert_eq!(
            journal.append_completed(completed),
            Err(ExecutionJournalError::PreparedDigestMismatch)
        );
    }

    #[test]
    fn hash_chain_tampering_is_detected_on_recovery() {
        let mut journal = InterventionExecutionJournal::new();
        journal.append_prepared(prepared("exec-1", "auth-1")).unwrap();
        let mut persisted = journal.events().to_vec();
        persisted[0].event_hash.0[0] ^= 0xff;
        assert!(matches!(
            InterventionExecutionJournal::from_events(persisted),
            Err(ExecutionJournalError::EventHashMismatch { .. })
        ));
    }

    #[test]
    fn expired_permit_cannot_enter_prepared_state() {
        let mut value = prepared("exec-1", "auth-1");
        value.prepared_at_unix_s = value.permit_not_after_unix_s;
        assert_eq!(
            value.validate(),
            Err(ExecutionJournalError::PermitExpiredBeforePrepare)
        );
    }
}
