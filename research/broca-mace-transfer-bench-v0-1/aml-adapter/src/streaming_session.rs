// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Event-sourced temporal session memory for streaming update evaluation.
//!
//! The ledger separates event/valid time from trusted observation time,
//! applies one sequence at a time, supports exact idempotent replay, and keeps
//! corrections, quarantine, revalidation, revocation and erasure requests as
//! explicit state transitions. It is an in-memory reference state machine,
//! not a durable hosted Add/Search service or a secure-erasure implementation.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use crate::raw_session::RawSessionMessageV1;

const MAX_SESSION_ID_BYTES_V1: usize = 256;
const MAX_EVENT_ID_BYTES_V1: usize = 512;
const MAX_TURN_ID_BYTES_V1: usize = 512;
const MAX_MESSAGE_BYTES_V1: usize = 256 * 1024;
const MAX_MUTATIONS_V1: usize = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingSessionRejectV1 {
    InvalidSessionIdentity,
    InvalidEventId,
    InvalidMessage,
    InvalidDigest,
    InvalidMessageTime,
    DuplicateTurnId,
    SequenceMismatch { expected: u64, actual: u64 },
    EventIdentityConflict,
    ClockRollback,
    MutationLimitExceeded,
    UnknownTarget,
    TargetNotActive,
    TargetNotQuarantined,
    CorrectionTurnMismatch,
    AlreadyErasureRequested,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamingMutationKindV1 {
    Append {
        message: RawSessionMessageV1,
    },
    Correct {
        target_event_id: String,
        message: RawSessionMessageV1,
    },
    Quarantine {
        target_event_id: String,
        reason_sha256: String,
    },
    Revalidate {
        target_event_id: String,
        validation_receipt_sha256: String,
    },
    Revoke {
        target_event_id: String,
        revocation_receipt_sha256: String,
    },
    ErasureRequest {
        target_event_id: String,
        rights_receipt_sha256: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingSessionMutationV1 {
    /// Unique idempotency identity for this mutation, not the turn ID.
    pub event_id: String,
    /// Strictly contiguous consumer sequence, starting at one.
    pub sequence: u64,
    /// Effective/valid time of the mutation. May be earlier than prior events
    /// when a correction or late-arriving observation is applied.
    pub effective_at_ms: i64,
    pub kind: StreamingMutationKindV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StoredEventStatusV1 {
    Active,
    Superseded,
    Quarantined,
    Revoked,
    ErasureRequested,
    Control,
}

impl StoredEventStatusV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Active => 1,
            Self::Superseded => 2,
            Self::Quarantined => 3,
            Self::Revoked => 4,
            Self::ErasureRequested => 5,
            Self::Control => 6,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredSessionEventV1 {
    event_id: String,
    sequence: u64,
    effective_at_ms: i64,
    observed_at_ms: u64,
    mutation_sha256: String,
    message_sha256: Option<String>,
    message: Option<RawSessionMessageV1>,
    status: StoredEventStatusV1,
    target_event_id: Option<String>,
    transition_receipt_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingInvalidationCauseV1 {
    Correction,
    Quarantine,
    Revocation,
    RightsErasure,
}

impl StreamingInvalidationCauseV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Correction => 1,
            Self::Quarantine => 2,
            Self::Revocation => 3,
            Self::RightsErasure => 4,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingSessionMutationReceiptV1 {
    pub user_id: String,
    pub session_id: String,
    pub event_id: String,
    pub sequence: u64,
    pub effective_at_ms: i64,
    pub observed_at_ms: u64,
    pub mutation_sha256: String,
    pub previous_state_sha256: String,
    pub resulting_state_sha256: String,
    /// Ordered by valid timestamp, then original ingestion sequence.
    pub visible_event_ids: Vec<String>,
    /// Previously visible event IDs no longer visible after this transition.
    pub invalidated_event_ids: Vec<String>,
    /// Source event identities targeted by this invalidating mutation, even if
    /// they were already hidden by an earlier quarantine transition.
    pub affected_event_ids: Vec<String>,
    /// Why the active-view dependencies became invalid. None for non-invalidating events.
    pub invalidation_cause: Option<StreamingInvalidationCauseV1>,
    pub receipt_sha256: String,
}

impl StreamingSessionMutationReceiptV1 {
    /// Verify this transition's exact deterministic commitment.
    pub fn verify_integrity(&self) -> bool {
        self.receipt_sha256 == receipt_digest(
            &self.user_id,
            &self.session_id,
            &self.event_id,
            self.sequence,
            self.effective_at_ms,
            self.observed_at_ms,
            &self.mutation_sha256,
            &self.previous_state_sha256,
            &self.resulting_state_sha256,
            &self.visible_event_ids,
            &self.invalidated_event_ids,
            &self.affected_event_ids,
            self.invalidation_cause,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingMutationApplyStatusV1 {
    Applied,
    IdempotentReplay,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingMutationApplyResultV1 {
    pub status: StreamingMutationApplyStatusV1,
    pub receipt: StreamingSessionMutationReceiptV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingVisibleMessageV1 {
    pub event_id: String,
    pub sequence: u64,
    pub observed_at_ms: u64,
    pub message: RawSessionMessageV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingSessionSnapshotV1 {
    pub user_id: String,
    pub session_id: String,
    pub last_sequence: u64,
    pub observed_at_ms: Option<u64>,
    pub messages: Vec<StreamingVisibleMessageV1>,
    pub snapshot_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingSessionLedgerV1 {
    user_id: String,
    session_id: String,
    events: Vec<StoredSessionEventV1>,
    mutation_digests: BTreeMap<String, String>,
    receipts: BTreeMap<String, StreamingSessionMutationReceiptV1>,
    turn_ids: BTreeSet<String>,
    last_sequence: u64,
    last_observed_at_ms: Option<u64>,
}

impl StreamingSessionLedgerV1 {
    pub fn new(user_id: impl Into<String>, session_id: impl Into<String>)
        -> Result<Self, StreamingSessionRejectV1>
    {
        let user_id = user_id.into();
        let session_id = session_id.into();
        if !valid_atom(&user_id, MAX_SESSION_ID_BYTES_V1)
            || !valid_atom(&session_id, MAX_SESSION_ID_BYTES_V1)
        {
            return Err(StreamingSessionRejectV1::InvalidSessionIdentity);
        }
        Ok(Self {
            user_id,
            session_id,
            events: Vec::new(),
            mutation_digests: BTreeMap::new(),
            receipts: BTreeMap::new(),
            turn_ids: BTreeSet::new(),
            last_sequence: 0,
            last_observed_at_ms: None,
        })
    }

    pub fn user_id(&self) -> &str { &self.user_id }
    pub fn session_id(&self) -> &str { &self.session_id }
    pub fn last_sequence(&self) -> u64 { self.last_sequence }

    pub fn mutation_receipts(&self) -> impl Iterator<Item = &StreamingSessionMutationReceiptV1> {
        self.events.iter().filter_map(|event| self.receipts.get(&event.event_id))
    }

    /// Apply one mutation atomically. `observed_at_ms` is supplied by the
    /// consumer clock, not copied from the user's event timestamp.
    ///
    /// Exact replay of the same event ID and mutation digest is idempotent.
    /// Reusing the event ID with changed content, skipping a sequence, or
    /// moving the trusted observation clock backward fails closed.
    pub fn apply(
        &mut self,
        mutation: StreamingSessionMutationV1,
        observed_at_ms: u64,
    ) -> Result<StreamingMutationApplyResultV1, StreamingSessionRejectV1> {
        validate_mutation(&mutation)?;
        let mutation_sha256 = mutation_digest(&self.user_id, &self.session_id, &mutation);

        if let Some(previous_digest) = self.mutation_digests.get(&mutation.event_id) {
            if previous_digest != &mutation_sha256 {
                return Err(StreamingSessionRejectV1::EventIdentityConflict);
            }
            let receipt = self.receipts.get(&mutation.event_id)
                .cloned().ok_or(StreamingSessionRejectV1::EventIdentityConflict)?;
            return Ok(StreamingMutationApplyResultV1 {
                status: StreamingMutationApplyStatusV1::IdempotentReplay,
                receipt,
            });
        }

        let expected = self.last_sequence.checked_add(1)
            .ok_or(StreamingSessionRejectV1::MutationLimitExceeded)?;
        if mutation.sequence != expected {
            return Err(StreamingSessionRejectV1::SequenceMismatch {
                expected,
                actual: mutation.sequence,
            });
        }
        if self.last_observed_at_ms.is_some_and(|previous| observed_at_ms < previous) {
            return Err(StreamingSessionRejectV1::ClockRollback);
        }
        if self.events.len() >= MAX_MUTATIONS_V1 {
            return Err(StreamingSessionRejectV1::MutationLimitExceeded);
        }

        // Stage against a clone so any transition failure leaves the live
        // ledger byte-for-byte unchanged.
        let mut staged = self.clone();
        let receipt = staged.apply_new(mutation, mutation_sha256, observed_at_ms)?;
        *self = staged;
        Ok(StreamingMutationApplyResultV1 {
            status: StreamingMutationApplyStatusV1::Applied,
            receipt,
        })
    }

    fn apply_new(
        &mut self,
        mutation: StreamingSessionMutationV1,
        mutation_sha256: String,
        observed_at_ms: u64,
    ) -> Result<StreamingSessionMutationReceiptV1, StreamingSessionRejectV1> {
        let previous_state_sha256 = self.state_sha256();
        let previous_visible_event_ids = self.visible_event_ids();
        let invalidation_cause = match &mutation.kind {
            StreamingMutationKindV1::Correct { .. } => Some(StreamingInvalidationCauseV1::Correction),
            StreamingMutationKindV1::Quarantine { .. } => Some(StreamingInvalidationCauseV1::Quarantine),
            StreamingMutationKindV1::Revoke { .. } => Some(StreamingInvalidationCauseV1::Revocation),
            StreamingMutationKindV1::ErasureRequest { .. } => Some(StreamingInvalidationCauseV1::RightsErasure),
            StreamingMutationKindV1::Append { .. } | StreamingMutationKindV1::Revalidate { .. } => None,
        };
        let affected_event_ids = match &mutation.kind {
            StreamingMutationKindV1::Correct { target_event_id, .. }
            | StreamingMutationKindV1::Quarantine { target_event_id, .. }
            | StreamingMutationKindV1::Revoke { target_event_id, .. }
            | StreamingMutationKindV1::ErasureRequest { target_event_id, .. } => {
                vec![target_event_id.clone()]
            }
            StreamingMutationKindV1::Append { .. } | StreamingMutationKindV1::Revalidate { .. } => Vec::new(),
        };
        let event_id = mutation.event_id.clone();
        let sequence = mutation.sequence;
        let effective_at_ms = mutation.effective_at_ms;

        let new_event = match mutation.kind {
            StreamingMutationKindV1::Append { message } => {
                validate_message(&message)?;
                if message.timestamp_ms != effective_at_ms {
                    return Err(StreamingSessionRejectV1::InvalidMessageTime);
                }
                if !self.turn_ids.insert(message.turn_id.clone()) {
                    return Err(StreamingSessionRejectV1::DuplicateTurnId);
                }
                stored_event(
                    event_id.clone(), sequence, effective_at_ms, observed_at_ms,
                    mutation_sha256.clone(), Some(message), StoredEventStatusV1::Active,
                    None, None,
                )
            }
            StreamingMutationKindV1::Correct { target_event_id, message } => {
                validate_message(&message)?;
                if message.timestamp_ms != effective_at_ms {
                    return Err(StreamingSessionRejectV1::InvalidMessageTime);
                }
                let target_index = self.event_index(&target_event_id)?;
                let target = &self.events[target_index];
                if target.status != StoredEventStatusV1::Active {
                    return Err(StreamingSessionRejectV1::TargetNotActive);
                }
                let target_message = target.message.as_ref()
                    .ok_or(StreamingSessionRejectV1::TargetNotActive)?;
                if target_message.turn_id != message.turn_id {
                    return Err(StreamingSessionRejectV1::CorrectionTurnMismatch);
                }
                self.events[target_index].status = StoredEventStatusV1::Superseded;
                stored_event(
                    event_id.clone(), sequence, effective_at_ms, observed_at_ms,
                    mutation_sha256.clone(), Some(message), StoredEventStatusV1::Active,
                    Some(target_event_id), None,
                )
            }
            StreamingMutationKindV1::Quarantine { target_event_id, reason_sha256 } => {
                validate_digest(&reason_sha256)?;
                let target_index = self.event_index(&target_event_id)?;
                if self.events[target_index].status != StoredEventStatusV1::Active {
                    return Err(StreamingSessionRejectV1::TargetNotActive);
                }
                self.events[target_index].status = StoredEventStatusV1::Quarantined;
                stored_event(
                    event_id.clone(), sequence, effective_at_ms, observed_at_ms,
                    mutation_sha256.clone(), None, StoredEventStatusV1::Control,
                    Some(target_event_id), Some(reason_sha256),
                )
            }
            StreamingMutationKindV1::Revalidate { target_event_id, validation_receipt_sha256 } => {
                validate_digest(&validation_receipt_sha256)?;
                let target_index = self.event_index(&target_event_id)?;
                if self.events[target_index].status != StoredEventStatusV1::Quarantined {
                    return Err(StreamingSessionRejectV1::TargetNotQuarantined);
                }
                self.events[target_index].status = StoredEventStatusV1::Active;
                stored_event(
                    event_id.clone(), sequence, effective_at_ms, observed_at_ms,
                    mutation_sha256.clone(), None, StoredEventStatusV1::Control,
                    Some(target_event_id), Some(validation_receipt_sha256),
                )
            }
            StreamingMutationKindV1::Revoke { target_event_id, revocation_receipt_sha256 } => {
                validate_digest(&revocation_receipt_sha256)?;
                let target_index = self.event_index(&target_event_id)?;
                if !matches!(
                    self.events[target_index].status,
                    StoredEventStatusV1::Active | StoredEventStatusV1::Quarantined
                ) {
                    return Err(StreamingSessionRejectV1::TargetNotActive);
                }
                self.events[target_index].status = StoredEventStatusV1::Revoked;
                stored_event(
                    event_id.clone(), sequence, effective_at_ms, observed_at_ms,
                    mutation_sha256.clone(), None, StoredEventStatusV1::Control,
                    Some(target_event_id), Some(revocation_receipt_sha256),
                )
            }
            StreamingMutationKindV1::ErasureRequest { target_event_id, rights_receipt_sha256 } => {
                validate_digest(&rights_receipt_sha256)?;
                let target_index = self.event_index(&target_event_id)?;
                if self.events[target_index].status == StoredEventStatusV1::ErasureRequested
                    || self.events[target_index].message.is_none()
                {
                    return Err(StreamingSessionRejectV1::AlreadyErasureRequested);
                }
                // Remove the payload from the live in-memory ledger projection.
                // This is not proof of physical deletion from backups, replicas,
                // process memory snapshots or external stores.
                self.events[target_index].message = None;
                self.events[target_index].message_sha256 = None;
                self.events[target_index].status = StoredEventStatusV1::ErasureRequested;
                stored_event(
                    event_id.clone(), sequence, effective_at_ms, observed_at_ms,
                    mutation_sha256.clone(), None, StoredEventStatusV1::Control,
                    Some(target_event_id), Some(rights_receipt_sha256),
                )
            }
        };

        self.events.push(new_event);
        self.mutation_digests.insert(event_id.clone(), mutation_sha256.clone());
        self.last_sequence = sequence;
        self.last_observed_at_ms = Some(observed_at_ms);

        let resulting_state_sha256 = self.state_sha256();
        let visible_event_ids = self.visible_event_ids();
        let visible_after = visible_event_ids.iter().collect::<BTreeSet<_>>();
        let invalidated_event_ids = previous_visible_event_ids.into_iter()
            .filter(|id| !visible_after.contains(id))
            .collect::<Vec<_>>();
        let receipt_sha256 = receipt_digest(
            &self.user_id,
            &self.session_id,
            &event_id,
            sequence,
            effective_at_ms,
            observed_at_ms,
            &mutation_sha256,
            &previous_state_sha256,
            &resulting_state_sha256,
            &visible_event_ids,
            &invalidated_event_ids,
            &affected_event_ids,
            invalidation_cause,
        );
        let receipt = StreamingSessionMutationReceiptV1 {
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            event_id: event_id.clone(),
            sequence,
            effective_at_ms,
            observed_at_ms,
            mutation_sha256,
            previous_state_sha256,
            resulting_state_sha256,
            visible_event_ids,
            invalidated_event_ids,
            affected_event_ids,
            invalidation_cause,
            receipt_sha256,
        };
        self.receipts.insert(event_id, receipt.clone());
        Ok(receipt)
    }

    fn event_index(&self, event_id: &str) -> Result<usize, StreamingSessionRejectV1> {
        self.events.iter().position(|event| event.event_id == event_id)
            .ok_or(StreamingSessionRejectV1::UnknownTarget)
    }

    pub fn snapshot(&self) -> StreamingSessionSnapshotV1 {
        let mut messages = self.events.iter()
            .filter_map(|event| {
                if event.status == StoredEventStatusV1::Active {
                    event.message.clone().map(|message| StreamingVisibleMessageV1 {
                        event_id: event.event_id.clone(),
                        sequence: event.sequence,
                        observed_at_ms: event.observed_at_ms,
                        message,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        messages.sort_by(|a, b| {
            (a.message.timestamp_ms, a.sequence).cmp(&(b.message.timestamp_ms, b.sequence))
        });
        let snapshot_sha256 = snapshot_digest(&self.user_id, &self.session_id, &messages);
        StreamingSessionSnapshotV1 {
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            last_sequence: self.last_sequence,
            observed_at_ms: self.last_observed_at_ms,
            messages,
            snapshot_sha256,
        }
    }

    fn visible_event_ids(&self) -> Vec<String> {
        self.snapshot().messages.into_iter().map(|item| item.event_id).collect()
    }

    fn state_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"luminous:mace:streaming-session-state:v1\0");
        hash_field(&mut h, &self.user_id);
        hash_field(&mut h, &self.session_id);
        h.update(self.last_sequence.to_be_bytes());
        h.update((self.events.len() as u64).to_be_bytes());
        for event in &self.events {
            hash_field(&mut h, &event.event_id);
            h.update(event.sequence.to_be_bytes());
            h.update(event.effective_at_ms.to_be_bytes());
            h.update(event.observed_at_ms.to_be_bytes());
            hash_field(&mut h, &event.mutation_sha256);
            h.update([event.status.tag()]);
            match &event.message_sha256 {
                Some(digest) => {
                    h.update([1]);
                    hash_field(&mut h, digest);
                }
                None => h.update([0]),
            }
            if let Some(target) = &event.target_event_id {
                h.update([1]);
                hash_field(&mut h, target);
            } else {
                h.update([0]);
            }
            if let Some(transition) = &event.transition_receipt_sha256 {
                h.update([1]);
                hash_field(&mut h, transition);
            } else {
                h.update([0]);
            }
        }
        hex_digest(&h.finalize())
    }
}

fn stored_event(
    event_id: String,
    sequence: u64,
    effective_at_ms: i64,
    observed_at_ms: u64,
    mutation_sha256: String,
    message: Option<RawSessionMessageV1>,
    status: StoredEventStatusV1,
    target_event_id: Option<String>,
    transition_receipt_sha256: Option<String>,
) -> StoredSessionEventV1 {
    let message_sha256 = message.as_ref().map(message_digest);
    StoredSessionEventV1 {
        event_id,
        sequence,
        effective_at_ms,
        observed_at_ms,
        mutation_sha256,
        message_sha256,
        message,
        status,
        target_event_id,
        transition_receipt_sha256,
    }
}

fn validate_mutation(mutation: &StreamingSessionMutationV1)
    -> Result<(), StreamingSessionRejectV1>
{
    if !valid_atom(&mutation.event_id, MAX_EVENT_ID_BYTES_V1) {
        return Err(StreamingSessionRejectV1::InvalidEventId);
    }
    if mutation.sequence == 0 {
        return Err(StreamingSessionRejectV1::SequenceMismatch {
            expected: 1,
            actual: 0,
        });
    }
    match &mutation.kind {
        StreamingMutationKindV1::Append { message } => {
            validate_message(message)?;
            if message.timestamp_ms != mutation.effective_at_ms {
                return Err(StreamingSessionRejectV1::InvalidMessageTime);
            }
        }
        StreamingMutationKindV1::Correct { target_event_id, message } => {
            validate_target(target_event_id)?;
            validate_message(message)?;
            if message.timestamp_ms != mutation.effective_at_ms {
                return Err(StreamingSessionRejectV1::InvalidMessageTime);
            }
        }
        StreamingMutationKindV1::Quarantine { target_event_id, reason_sha256 } => {
            validate_target(target_event_id)?;
            validate_digest(reason_sha256)?;
        }
        StreamingMutationKindV1::Revalidate { target_event_id, validation_receipt_sha256 } => {
            validate_target(target_event_id)?;
            validate_digest(validation_receipt_sha256)?;
        }
        StreamingMutationKindV1::Revoke { target_event_id, revocation_receipt_sha256 } => {
            validate_target(target_event_id)?;
            validate_digest(revocation_receipt_sha256)?;
        }
        StreamingMutationKindV1::ErasureRequest { target_event_id, rights_receipt_sha256 } => {
            validate_target(target_event_id)?;
            validate_digest(rights_receipt_sha256)?;
        }
    }
    Ok(())
}

fn validate_message(message: &RawSessionMessageV1) -> Result<(), StreamingSessionRejectV1> {
    if !valid_atom(&message.turn_id, MAX_TURN_ID_BYTES_V1)
        || message.text.trim().is_empty()
        || message.text.len() > MAX_MESSAGE_BYTES_V1
        || !valid_digest(&message.raw_line_sha256)
    {
        return Err(StreamingSessionRejectV1::InvalidMessage);
    }
    Ok(())
}

fn validate_target(value: &str) -> Result<(), StreamingSessionRejectV1> {
    if valid_atom(value, MAX_EVENT_ID_BYTES_V1) {
        Ok(())
    } else {
        Err(StreamingSessionRejectV1::InvalidEventId)
    }
}

fn validate_digest(value: &str) -> Result<(), StreamingSessionRejectV1> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(StreamingSessionRejectV1::InvalidDigest)
    }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_atom(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= maximum
        && !value.chars().any(char::is_control)
}

fn role_tag(role: crate::Role) -> u8 {
    match role {
        crate::Role::User => 1,
        crate::Role::Assistant => 2,
    }
}

fn message_digest(message: &RawSessionMessageV1) -> String {
    let mut h = Sha256::new();
    h.update(b"luminous:mace:streaming-session-message:v1\0");
    hash_field(&mut h, &message.turn_id);
    h.update([role_tag(message.role)]);
    h.update(message.timestamp_ms.to_be_bytes());
    hash_field(&mut h, &message.raw_line_sha256);
    hash_field(&mut h, &message.text);
    hex_digest(&h.finalize())
}

fn mutation_digest(
    user_id: &str,
    session_id: &str,
    mutation: &StreamingSessionMutationV1,
) -> String {
    let mut h = Sha256::new();
    h.update(b"luminous:mace:streaming-session-mutation:v1\0");
    hash_field(&mut h, user_id);
    hash_field(&mut h, session_id);
    hash_field(&mut h, &mutation.event_id);
    h.update(mutation.sequence.to_be_bytes());
    h.update(mutation.effective_at_ms.to_be_bytes());
    match &mutation.kind {
        StreamingMutationKindV1::Append { message } => {
            h.update([1]);
            hash_field(&mut h, &message_digest(message));
        }
        StreamingMutationKindV1::Correct { target_event_id, message } => {
            h.update([2]);
            hash_field(&mut h, target_event_id);
            hash_field(&mut h, &message_digest(message));
        }
        StreamingMutationKindV1::Quarantine { target_event_id, reason_sha256 } => {
            h.update([3]);
            hash_field(&mut h, target_event_id);
            hash_field(&mut h, reason_sha256);
        }
        StreamingMutationKindV1::Revalidate { target_event_id, validation_receipt_sha256 } => {
            h.update([4]);
            hash_field(&mut h, target_event_id);
            hash_field(&mut h, validation_receipt_sha256);
        }
        StreamingMutationKindV1::Revoke { target_event_id, revocation_receipt_sha256 } => {
            h.update([5]);
            hash_field(&mut h, target_event_id);
            hash_field(&mut h, revocation_receipt_sha256);
        }
        StreamingMutationKindV1::ErasureRequest { target_event_id, rights_receipt_sha256 } => {
            h.update([6]);
            hash_field(&mut h, target_event_id);
            hash_field(&mut h, rights_receipt_sha256);
        }
    }
    hex_digest(&h.finalize())
}

fn snapshot_digest(
    user_id: &str,
    session_id: &str,
    messages: &[StreamingVisibleMessageV1],
) -> String {
    let mut h = Sha256::new();
    h.update(b"luminous:mace:streaming-session-snapshot:v1\0");
    hash_field(&mut h, user_id);
    hash_field(&mut h, session_id);
    h.update((messages.len() as u64).to_be_bytes());
    for item in messages {
        hash_field(&mut h, &item.event_id);
        h.update(item.sequence.to_be_bytes());
        h.update(item.observed_at_ms.to_be_bytes());
        hash_field(&mut h, &message_digest(&item.message));
    }
    hex_digest(&h.finalize())
}

fn receipt_digest(
    user_id: &str,
    session_id: &str,
    event_id: &str,
    sequence: u64,
    effective_at_ms: i64,
    observed_at_ms: u64,
    mutation_sha256: &str,
    previous_state_sha256: &str,
    resulting_state_sha256: &str,
    visible_event_ids: &[String],
    invalidated_event_ids: &[String],
    affected_event_ids: &[String],
    invalidation_cause: Option<StreamingInvalidationCauseV1>,
) -> String {
    let mut h = Sha256::new();
    h.update(b"luminous:mace:streaming-session-receipt:v1\0");
    hash_field(&mut h, user_id);
    hash_field(&mut h, session_id);
    hash_field(&mut h, event_id);
    h.update(sequence.to_be_bytes());
    h.update(effective_at_ms.to_be_bytes());
    h.update(observed_at_ms.to_be_bytes());
    hash_field(&mut h, mutation_sha256);
    hash_field(&mut h, previous_state_sha256);
    hash_field(&mut h, resulting_state_sha256);
    h.update((visible_event_ids.len() as u64).to_be_bytes());
    for id in visible_event_ids { hash_field(&mut h, id); }
    h.update((invalidated_event_ids.len() as u64).to_be_bytes());
    for id in invalidated_event_ids { hash_field(&mut h, id); }
    h.update((affected_event_ids.len() as u64).to_be_bytes());
    for id in affected_event_ids { hash_field(&mut h, id); }
    h.update([invalidation_cause.map_or(0, StreamingInvalidationCauseV1::tag)]);
    hex_digest(&h.finalize())
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Role;

    fn message(turn_id: &str, timestamp_ms: i64, text: &str) -> RawSessionMessageV1 {
        RawSessionMessageV1 {
            turn_id: turn_id.to_owned(),
            role: Role::User,
            timestamp_ms,
            text: text.to_owned(),
            raw_line_sha256: "a".repeat(64),
        }
    }

    fn append(event_id: &str, sequence: u64, timestamp_ms: i64, text: &str) -> StreamingSessionMutationV1 {
        StreamingSessionMutationV1 {
            event_id: event_id.to_owned(),
            sequence,
            effective_at_ms: timestamp_ms,
            kind: StreamingMutationKindV1::Append {
                message: message(&format!("turn-{sequence}"), timestamp_ms, text),
            },
        }
    }

    fn ledger() -> StreamingSessionLedgerV1 {
        StreamingSessionLedgerV1::new("user-1", "session-1").unwrap()
    }

    #[test]
    fn append_is_visible_and_receipt_binds_before_after_state() {
        let mut ledger = ledger();
        let result = ledger.apply(append("event-1", 1, 100, "first"), 1_000).unwrap();
        assert_eq!(result.status, StreamingMutationApplyStatusV1::Applied);
        assert_ne!(result.receipt.previous_state_sha256, result.receipt.resulting_state_sha256);
        assert_eq!(ledger.snapshot().messages.len(), 1);
        assert_eq!(ledger.snapshot().messages[0].message.text, "first");
        assert_eq!(result.receipt.visible_event_ids, vec!["event-1"]);
    }

    #[test]
    fn exact_event_replay_is_idempotent_without_advancing_sequence() {
        let mut ledger = ledger();
        let event = append("event-1", 1, 100, "first");
        let first = ledger.apply(event.clone(), 1_000).unwrap();
        let replay = ledger.apply(event, 1_100).unwrap();
        assert_eq!(replay.status, StreamingMutationApplyStatusV1::IdempotentReplay);
        assert_eq!(replay.receipt, first.receipt);
        assert_eq!(ledger.last_sequence(), 1);
        assert_eq!(ledger.snapshot().messages.len(), 1);
    }

    #[test]
    fn reused_event_identity_with_changed_payload_is_rejected() {
        let mut ledger = ledger();
        ledger.apply(append("event-1", 1, 100, "first"), 1_000).unwrap();
        let changed = append("event-1", 1, 100, "tampered");
        assert_eq!(
            ledger.apply(changed, 1_100),
            Err(StreamingSessionRejectV1::EventIdentityConflict)
        );
        assert_eq!(ledger.snapshot().messages[0].message.text, "first");
    }

    #[test]
    fn malformed_correction_target_is_rejected_before_lookup() {
        let mutation = StreamingSessionMutationV1 {
            event_id: "event-1".into(),
            sequence: 1,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: " bad-target".into(),
                message: message("turn-1", 100, "replacement"),
            },
        };
        assert_eq!(
            validate_mutation(&mutation),
            Err(StreamingSessionRejectV1::InvalidEventId)
        );
    }

    #[test]
    fn sequence_gaps_and_observation_clock_rollback_fail_closed() {
        let mut ledger = ledger();
        assert_eq!(
            ledger.apply(append("event-2", 2, 100, "gap"), 1_000),
            Err(StreamingSessionRejectV1::SequenceMismatch { expected: 1, actual: 2 })
        );
        ledger.apply(append("event-1", 1, 100, "first"), 1_000).unwrap();
        assert_eq!(
            ledger.apply(append("event-2", 2, 200, "rollback"), 999),
            Err(StreamingSessionRejectV1::ClockRollback)
        );
        assert_eq!(ledger.last_sequence(), 1);
    }

    #[test]
    fn transition_receipt_scope_and_invalidation_cause_are_committed() {
        let mut ledger = ledger();
        let append = ledger.apply(append("event-1", 1, 100, "fact"), 1_000).unwrap();
        assert!(append.receipt.verify_integrity());
        assert_eq!(append.receipt.user_id, "user-1");
        assert_eq!(append.receipt.session_id, "session-1");
        assert_eq!(append.receipt.invalidation_cause, None);
        let quarantine = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::Quarantine {
                target_event_id: "event-1".into(),
                reason_sha256: "b".repeat(64),
            },
        };
        let receipt = ledger.apply(quarantine, 1_100).unwrap().receipt;
        assert!(receipt.verify_integrity());
        assert_eq!(receipt.invalidation_cause, Some(StreamingInvalidationCauseV1::Quarantine));
        let mut altered = receipt.clone();
        altered.session_id = "different-session".into();
        assert!(!altered.verify_integrity());
        altered = receipt.clone();
        altered.invalidation_cause = Some(StreamingInvalidationCauseV1::Revocation);
        assert!(!altered.verify_integrity());
    }

    #[test]
    fn correction_receipt_names_the_event_removed_from_the_active_projection() {
        let mut ledger = ledger();
        ledger.apply(append("event-1", 1, 100, "old fact"), 1_000).unwrap();
        let correction = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "event-1".into(),
                message: message("turn-1", 100, "corrected fact"),
            },
        };
        let receipt = ledger.apply(correction, 1_100).unwrap().receipt;
        assert_eq!(receipt.visible_event_ids, vec!["event-2"]);
        assert_eq!(receipt.invalidated_event_ids, vec!["event-1"]);
    }

    #[test]
    fn quarantine_and_revocation_receipts_report_newly_hidden_events() {
        let mut ledger = ledger();
        ledger.apply(append("event-1", 1, 100, "fact"), 1_000).unwrap();
        let quarantine = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::Quarantine {
                target_event_id: "event-1".into(),
                reason_sha256: "b".repeat(64),
            },
        };
        let receipt = ledger.apply(quarantine, 1_100).unwrap().receipt;
        assert_eq!(receipt.invalidated_event_ids, vec!["event-1"]);
        let revalidate = StreamingSessionMutationV1 {
            event_id: "event-3".into(),
            sequence: 3,
            effective_at_ms: 300,
            kind: StreamingMutationKindV1::Revalidate {
                target_event_id: "event-1".into(),
                validation_receipt_sha256: "c".repeat(64),
            },
        };
        let receipt = ledger.apply(revalidate, 1_200).unwrap().receipt;
        assert!(receipt.invalidated_event_ids.is_empty());
        let revoke = StreamingSessionMutationV1 {
            event_id: "event-4".into(),
            sequence: 4,
            effective_at_ms: 400,
            kind: StreamingMutationKindV1::Revoke {
                target_event_id: "event-1".into(),
                revocation_receipt_sha256: "d".repeat(64),
            },
        };
        let receipt = ledger.apply(revoke, 1_300).unwrap().receipt;
        assert_eq!(receipt.invalidated_event_ids, vec!["event-1"]);
    }

    #[test]
    fn correction_supersedes_previous_content_without_overwriting_history() {
        let mut ledger = ledger();
        ledger.apply(append("event-1", 1, 100, "wrong value"), 1_000).unwrap();
        let correction = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "event-1".into(),
                message: message("turn-1", 100, "correct value"),
            },
        };
        let result = ledger.apply(correction, 1_100).unwrap();
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.messages.len(), 1);
        assert_eq!(snapshot.messages[0].event_id, "event-2");
        assert_eq!(snapshot.messages[0].message.text, "correct value");
        assert_eq!(result.receipt.visible_event_ids, vec!["event-2"]);
    }

    #[test]
    fn quarantine_hides_evidence_and_revalidation_restores_it() {
        let mut ledger = ledger();
        ledger.apply(append("event-1", 1, 100, "fact"), 1_000).unwrap();
        let quarantine = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::Quarantine {
                target_event_id: "event-1".into(),
                reason_sha256: "b".repeat(64),
            },
        };
        ledger.apply(quarantine, 1_100).unwrap();
        assert!(ledger.snapshot().messages.is_empty());
        let revalidate = StreamingSessionMutationV1 {
            event_id: "event-3".into(),
            sequence: 3,
            effective_at_ms: 300,
            kind: StreamingMutationKindV1::Revalidate {
                target_event_id: "event-1".into(),
                validation_receipt_sha256: "c".repeat(64),
            },
        };
        ledger.apply(revalidate, 1_200).unwrap();
        assert_eq!(ledger.snapshot().messages.len(), 1);
    }

    #[test]
    fn revocation_is_terminal_for_revalidation() {
        let mut ledger = ledger();
        ledger.apply(append("event-1", 1, 100, "fact"), 1_000).unwrap();
        let revoke = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::Revoke {
                target_event_id: "event-1".into(),
                revocation_receipt_sha256: "b".repeat(64),
            },
        };
        ledger.apply(revoke, 1_100).unwrap();
        let revalidate = StreamingSessionMutationV1 {
            event_id: "event-3".into(),
            sequence: 3,
            effective_at_ms: 300,
            kind: StreamingMutationKindV1::Revalidate {
                target_event_id: "event-1".into(),
                validation_receipt_sha256: "c".repeat(64),
            },
        };
        assert_eq!(
            ledger.apply(revalidate, 1_200),
            Err(StreamingSessionRejectV1::TargetNotQuarantined)
        );
        assert!(ledger.snapshot().messages.is_empty());
    }

    #[test]
    fn rights_erasure_request_removes_payload_from_active_ledger_projection() {
        let mut ledger = ledger();
        ledger.apply(append("event-1", 1, 100, "private fact"), 1_000).unwrap();
        let erase = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::ErasureRequest {
                target_event_id: "event-1".into(),
                rights_receipt_sha256: "b".repeat(64),
            },
        };
        ledger.apply(erase, 1_100).unwrap();
        assert!(ledger.snapshot().messages.is_empty());
        assert_eq!(ledger.events[0].message, None);
        assert_eq!(ledger.events[0].status, StoredEventStatusV1::ErasureRequested);
    }

    #[test]
    fn late_arriving_correction_preserves_valid_time_and_observed_time() {
        let mut ledger = ledger();
        ledger.apply(append("event-1", 1, 500, "future-valid fact"), 2_000).unwrap();
        let correction = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 300,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "event-1".into(),
                message: message("turn-1", 300, "retroactive correction"),
            },
        };
        let receipt = ledger.apply(correction, 2_500).unwrap().receipt;
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.messages[0].message.timestamp_ms, 300);
        assert_eq!(snapshot.messages[0].observed_at_ms, 2_500);
        assert_eq!(receipt.effective_at_ms, 300);
        assert_eq!(receipt.observed_at_ms, 2_500);
    }

    #[test]
    fn failed_mutation_leaves_state_digest_and_sequence_unchanged() {
        let mut ledger = ledger();
        ledger.apply(append("event-1", 1, 100, "fact"), 1_000).unwrap();
        let before = ledger.state_sha256();
        let invalid = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "missing-event".into(),
                message: message("turn-1", 200, "replacement"),
            },
        };
        assert_eq!(
            ledger.apply(invalid, 1_100),
            Err(StreamingSessionRejectV1::UnknownTarget)
        );
        assert_eq!(ledger.state_sha256(), before);
        assert_eq!(ledger.last_sequence(), 1);
    }
}
