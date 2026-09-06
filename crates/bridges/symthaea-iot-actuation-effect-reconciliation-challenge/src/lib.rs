// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh, non-authorizing reconciliation challenge for one unresolved physical-effect attempt.
//!
//! This crate deliberately does **not** decide whether an effect happened. It issues a short-lived
//! nonce-bound challenge only from a currently rollback-protected unresolved attempt journal. A
//! later device-class-specific verifier must authenticate outcome evidence for this exact challenge,
//! and a later reconciliation writer must still prove that the protected journal head has not moved
//! before closing the unresolved state.
//!
//! Generic command-sequence advancement, arbitrary telemetry, adapter acknowledgement and this
//! challenge itself are all insufficient to establish physical realization.

#![deny(unsafe_code)]

use std::error::Error as StdError;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use symthaea_authority::{Digest32, Operation, PrincipalId, ResourceRef};
use symthaea_iot_actuation_effect_attempt_journal::{
    DurableEffectAttemptStateV1, EffectAttemptJournalError, IndependentEffectAttemptHeadAnchor,
    RollbackProtectedEffectAttemptJournal, RollbackProtectedEffectAttemptJournalError,
};
use thiserror::Error;

pub const EFFECT_RECONCILIATION_CHALLENGE_SCHEMA_VERSION: u16 = 1;
pub const MAX_EFFECT_RECONCILIATION_CHALLENGE_LIFETIME_MS: u64 = 5_000;
pub const MAX_EFFECT_RECONCILIATION_ID_BYTES: usize = 1_024;

const CHALLENGE_DOMAIN: &[u8] = b"symthaea-iot-effect-reconciliation-challenge-v1\0";

/// Exact unresolved journal state for which fresh outcome evidence is requested.
///
/// `AdapterAcknowledged` carries the exact adapter-local evidence commitment already present in the
/// protected journal. It still does not mean the requested physical state was realized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ReconciliationSourceStateV1 {
    Prepared,
    AdapterAcknowledged {
        adapter_evidence_digest: Digest32,
    },
    AdapterIndeterminate,
}

/// Short-lived challenge naming one exact rollback-protected unresolved attempt.
///
/// Fields are private and this type intentionally implements `Serialize` but not `Deserialize`.
/// Production construction requires a live rollback-protected journal wrapper, OS entropy and the
/// guard-local wall clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EffectReconciliationChallengeV1 {
    schema_version: u16,
    nonce: [u8; 32],
    journal_generation: u64,
    journal_digest: Digest32,
    correlation_digest: Digest32,
    command_digest: Digest32,
    envelope_digest: Digest32,
    composition_digest: Digest32,
    device: ResourceRef,
    operation: Operation,
    executor: PrincipalId,
    sequence: u64,
    adapter_id: String,
    source_state: ReconciliationSourceStateV1,
    attempt_common_fenced_at_unix_ms: u64,
    attempt_wall_valid_until_unix_ms: u64,
    issued_at_unix_ms: u64,
    expires_at_unix_ms: u64,
}

impl EffectReconciliationChallengeV1 {
    pub fn validate(&self) -> Result<(), EffectReconciliationChallengeValidationError> {
        if self.schema_version != EFFECT_RECONCILIATION_CHALLENGE_SCHEMA_VERSION {
            return Err(EffectReconciliationChallengeValidationError::UnsupportedSchema);
        }
        if self.nonce == [0; 32] {
            return Err(EffectReconciliationChallengeValidationError::ZeroNonce);
        }
        if self.journal_generation == 0 || self.sequence == 0 {
            return Err(EffectReconciliationChallengeValidationError::ZeroGenerationOrSequence);
        }
        for digest in [
            self.journal_digest,
            self.correlation_digest,
            self.command_digest,
            self.envelope_digest,
            self.composition_digest,
        ] {
            if digest == Digest32([0; 32]) {
                return Err(EffectReconciliationChallengeValidationError::ZeroCommitment);
            }
        }
        if let ReconciliationSourceStateV1::AdapterAcknowledged {
            adapter_evidence_digest,
        } = self.source_state
        {
            if adapter_evidence_digest == Digest32([0; 32]) {
                return Err(EffectReconciliationChallengeValidationError::ZeroAdapterEvidence);
            }
        }
        for value in [
            self.device.0.as_str(),
            self.operation.0.as_str(),
            self.executor.0.as_str(),
            self.adapter_id.as_str(),
        ] {
            if !valid_id(value) {
                return Err(EffectReconciliationChallengeValidationError::InvalidIdentity);
            }
        }
        if self.attempt_wall_valid_until_unix_ms <= self.attempt_common_fenced_at_unix_ms {
            return Err(EffectReconciliationChallengeValidationError::InvalidAttemptWindow);
        }
        if self.issued_at_unix_ms < self.attempt_common_fenced_at_unix_ms {
            return Err(EffectReconciliationChallengeValidationError::IssuedBeforeAttemptFence);
        }
        let lifetime = self
            .expires_at_unix_ms
            .checked_sub(self.issued_at_unix_ms)
            .ok_or(EffectReconciliationChallengeValidationError::InvalidChallengeWindow)?;
        if lifetime == 0 || lifetime > MAX_EFFECT_RECONCILIATION_CHALLENGE_LIFETIME_MS {
            return Err(EffectReconciliationChallengeValidationError::InvalidChallengeWindow);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, EffectReconciliationChallengeValidationError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(CHALLENGE_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.nonce);
        h.update(&self.journal_generation.to_be_bytes());
        update_digest(&mut h, self.journal_digest);
        update_digest(&mut h, self.correlation_digest);
        update_digest(&mut h, self.command_digest);
        update_digest(&mut h, self.envelope_digest);
        update_digest(&mut h, self.composition_digest);
        update_string(&mut h, &self.device.0);
        update_string(&mut h, &self.operation.0);
        update_string(&mut h, &self.executor.0);
        h.update(&self.sequence.to_be_bytes());
        update_string(&mut h, &self.adapter_id);
        match self.source_state {
            ReconciliationSourceStateV1::Prepared => {
                h.update(&[0]);
            }
            ReconciliationSourceStateV1::AdapterAcknowledged {
                adapter_evidence_digest,
            } => {
                h.update(&[1]);
                update_digest(&mut h, adapter_evidence_digest);
            }
            ReconciliationSourceStateV1::AdapterIndeterminate => {
                h.update(&[2]);
            }
        }
        h.update(&self.attempt_common_fenced_at_unix_ms.to_be_bytes());
        h.update(&self.attempt_wall_valid_until_unix_ms.to_be_bytes());
        h.update(&self.issued_at_unix_ms.to_be_bytes());
        h.update(&self.expires_at_unix_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn canonical_bytes(
        &self,
    ) -> Result<Vec<u8>, EffectReconciliationChallengeValidationError> {
        self.validate()?;
        bincode::serialize(self)
            .map_err(|_| EffectReconciliationChallengeValidationError::Encoding)
    }

    pub fn is_fresh_at(&self, now_unix_ms: u64) -> bool {
        now_unix_ms >= self.issued_at_unix_ms && now_unix_ms < self.expires_at_unix_ms
    }

    pub const fn nonce(&self) -> [u8; 32] {
        self.nonce
    }

    pub const fn journal_generation(&self) -> u64 {
        self.journal_generation
    }

    pub const fn journal_digest(&self) -> Digest32 {
        self.journal_digest
    }

    pub const fn correlation_digest(&self) -> Digest32 {
        self.correlation_digest
    }

    pub const fn command_digest(&self) -> Digest32 {
        self.command_digest
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub const fn composition_digest(&self) -> Digest32 {
        self.composition_digest
    }

    pub fn device(&self) -> &ResourceRef {
        &self.device
    }

    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn executor(&self) -> &PrincipalId {
        &self.executor
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn adapter_id(&self) -> &str {
        &self.adapter_id
    }

    pub const fn source_state(&self) -> ReconciliationSourceStateV1 {
        self.source_state
    }

    pub const fn attempt_common_fenced_at_unix_ms(&self) -> u64 {
        self.attempt_common_fenced_at_unix_ms
    }

    pub const fn attempt_wall_valid_until_unix_ms(&self) -> u64 {
        self.attempt_wall_valid_until_unix_ms
    }

    pub const fn issued_at_unix_ms(&self) -> u64 {
        self.issued_at_unix_ms
    }

    pub const fn expires_at_unix_ms(&self) -> u64 {
        self.expires_at_unix_ms
    }
}

/// Issue one fresh challenge from the exact currently rollback-protected unresolved attempt.
///
/// `current_checkpoint()` re-reads both the independent anchor and the local checkpoint before the
/// challenge is constructed. The challenge therefore names the exact protected head current at
/// issuance. A later reconciliation writer must compare that head again before any terminal state
/// transition; challenge issuance alone does not freeze the journal indefinitely.
pub fn issue_effect_reconciliation_challenge<A>(
    journal: &mut RollbackProtectedEffectAttemptJournal<A>,
) -> Result<EffectReconciliationChallengeV1, EffectReconciliationChallengeIssueError<A::Error>>
where
    A: IndependentEffectAttemptHeadAnchor,
{
    let checkpoint = journal
        .current_checkpoint()
        .map_err(EffectReconciliationChallengeIssueError::Journal)?;
    let checkpoint_head = checkpoint
        .head()
        .map_err(EffectReconciliationChallengeIssueError::JournalState)?;
    if checkpoint_head != journal.anchored_head() {
        return Err(EffectReconciliationChallengeIssueError::ProtectedHeadMismatch);
    }
    let latest = checkpoint
        .latest()
        .ok_or(EffectReconciliationChallengeIssueError::NoUnresolvedAttempt)?;
    if !latest.requires_reconciliation() {
        return Err(EffectReconciliationChallengeIssueError::NoUnresolvedAttempt);
    }

    let source_state = match latest {
        DurableEffectAttemptStateV1::Prepared { .. } => ReconciliationSourceStateV1::Prepared,
        DurableEffectAttemptStateV1::AdapterAcknowledged {
            adapter_evidence_digest,
            ..
        } => ReconciliationSourceStateV1::AdapterAcknowledged {
            adapter_evidence_digest: *adapter_evidence_digest,
        },
        DurableEffectAttemptStateV1::AdapterIndeterminate { .. } => {
            ReconciliationSourceStateV1::AdapterIndeterminate
        }
        DurableEffectAttemptStateV1::AbandonedBeforePort { .. } => {
            return Err(EffectReconciliationChallengeIssueError::NoUnresolvedAttempt);
        }
    };

    let correlation = latest.correlation();
    let correlation_digest = correlation
        .digest()
        .map_err(EffectReconciliationChallengeIssueError::JournalState)?;
    let mut nonce = [0u8; 32];
    getrandom::getrandom(&mut nonce)
        .map_err(|_| EffectReconciliationChallengeIssueError::EntropyUnavailable)?;
    let issued_at_unix_ms = system_unix_ms()
        .map_err(EffectReconciliationChallengeIssueError::Clock)?;
    let expires_at_unix_ms = issued_at_unix_ms
        .checked_add(MAX_EFFECT_RECONCILIATION_CHALLENGE_LIFETIME_MS)
        .ok_or(EffectReconciliationChallengeIssueError::TimeOverflow)?;

    let challenge = EffectReconciliationChallengeV1 {
        schema_version: EFFECT_RECONCILIATION_CHALLENGE_SCHEMA_VERSION,
        nonce,
        journal_generation: checkpoint_head.generation(),
        journal_digest: checkpoint_head.digest(),
        correlation_digest,
        command_digest: correlation.command_digest(),
        envelope_digest: correlation.envelope_digest(),
        composition_digest: correlation.composition_digest(),
        device: ResourceRef(correlation.device().to_owned()),
        operation: Operation(correlation.operation().to_owned()),
        executor: PrincipalId(correlation.executor().to_owned()),
        sequence: correlation.sequence(),
        adapter_id: correlation.adapter_id().to_owned(),
        source_state,
        attempt_common_fenced_at_unix_ms: correlation.common_fenced_at_unix_ms(),
        attempt_wall_valid_until_unix_ms: correlation.wall_valid_until_unix_ms(),
        issued_at_unix_ms,
        expires_at_unix_ms,
    };
    challenge
        .validate()
        .map_err(EffectReconciliationChallengeIssueError::Validation)?;
    Ok(challenge)
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_EFFECT_RECONCILIATION_ID_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_digest(h: &mut blake3::Hasher, Digest32(bytes): Digest32) {
    h.update(&bytes);
}

fn system_unix_ms() -> Result<u64, EffectReconciliationClockError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| EffectReconciliationClockError::BeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| EffectReconciliationClockError::Overflow)
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum EffectReconciliationChallengeValidationError {
    #[error("unsupported effect-reconciliation challenge schema")]
    UnsupportedSchema,
    #[error("effect-reconciliation challenge nonce is zero")]
    ZeroNonce,
    #[error("effect-reconciliation challenge journal generation or command sequence is zero")]
    ZeroGenerationOrSequence,
    #[error("effect-reconciliation challenge contains a zero security commitment")]
    ZeroCommitment,
    #[error("acknowledged source state contains a zero adapter evidence commitment")]
    ZeroAdapterEvidence,
    #[error("effect-reconciliation challenge contains an invalid identity")]
    InvalidIdentity,
    #[error("original actuation attempt window is malformed")]
    InvalidAttemptWindow,
    #[error("reconciliation challenge appears to have been issued before the original attempt fence")]
    IssuedBeforeAttemptFence,
    #[error("effect-reconciliation challenge lifetime is invalid")]
    InvalidChallengeWindow,
    #[error("effect-reconciliation challenge encoding failed")]
    Encoding,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum EffectReconciliationClockError {
    #[error("system wall clock is before the Unix epoch")]
    BeforeUnixEpoch,
    #[error("system wall clock does not fit in u64 milliseconds")]
    Overflow,
}

#[derive(Debug, Error)]
pub enum EffectReconciliationChallengeIssueError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("rollback-protected attempt journal could not be read: {0}")]
    Journal(#[source] RollbackProtectedEffectAttemptJournalError<E>),
    #[error("protected journal state could not reproduce its commitment: {0}")]
    JournalState(#[source] EffectAttemptJournalError),
    #[error("protected journal checkpoint does not match its independently anchored head")]
    ProtectedHeadMismatch,
    #[error("current protected journal state does not require reconciliation")]
    NoUnresolvedAttempt,
    #[error("OS entropy unavailable for reconciliation challenge nonce")]
    EntropyUnavailable,
    #[error("wall clock unavailable for reconciliation challenge: {0}")]
    Clock(#[source] EffectReconciliationClockError),
    #[error("reconciliation challenge expiry overflow")]
    TimeOverflow,
    #[error("constructed reconciliation challenge is invalid: {0}")]
    Validation(#[source] EffectReconciliationChallengeValidationError),
}
