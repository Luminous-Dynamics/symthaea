// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure restart activation gate for canonical episodic occurrences.
//!
//! This module deliberately performs no memory mutation. It decides which exact persisted
//! occurrence UUIDs are eligible to become active only after the caller has already recovered the
//! quarantine intent/state ledgers against their trusted external head anchors.
//!
//! The governing rule is fail closed:
//!
//! > No persisted episodic occurrence becomes cognitively active after restart while an exact
//! > pending quarantine intent, unresolved quarantine state, identity/content contradiction, or
//! > missing escrow dependency remains.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_memory::episodic_replay::EpisodeInstanceId;
use thiserror::Error;

use crate::memory_identity::EpisodeContentId;
use crate::quarantine_intent_ledger::{
    EpisodicQuarantineIntentLedger, PendingQuarantineIntent,
};
use crate::quarantine_state_ledger::{
    EpisodicQuarantineStateLedger, QuarantineLedgerState,
};

const MAX_REF_BYTES: usize = 2048;
const MAX_TARGET_BYTES: usize = 256;

/// One canonical persisted active-record projection available at restart.
///
/// `record_ref` is an opaque durable locator, not authority. `content_id` must be recomputed from
/// the persisted episode envelope before constructing this descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedOccurrenceDescriptor {
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub record_ref: String,
}

impl PersistedOccurrenceDescriptor {
    pub fn new(
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        record_ref: impl Into<String>,
    ) -> Result<Self, RestartActivationError> {
        let record_ref = record_ref.into();
        validate_ref("record_ref", &record_ref)?;
        Ok(Self {
            instance_id,
            content_id,
            record_ref,
        })
    }
}

/// Exact reversible escrow evidence available at restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedQuarantineEscrowDescriptor {
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub escrow_digest: Sha256Digest,
    pub escrow_persistence_ref: String,
}

impl PersistedQuarantineEscrowDescriptor {
    pub fn new(
        instance_id: EpisodeInstanceId,
        content_id: EpisodeContentId,
        escrow_digest: Sha256Digest,
        escrow_persistence_ref: impl Into<String>,
    ) -> Result<Self, RestartActivationError> {
        if escrow_digest.0 == [0; 32] {
            return Err(RestartActivationError::ZeroDigest {
                field: "escrow_digest",
            });
        }
        let escrow_persistence_ref = escrow_persistence_ref.into();
        validate_ref("escrow_persistence_ref", &escrow_persistence_ref)?;
        Ok(Self {
            instance_id,
            content_id,
            escrow_digest,
            escrow_persistence_ref,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InactiveRestartReason {
    /// A durable write-ahead quarantine intent exists but final state may not yet be committed.
    PendingQuarantineIntent,
    /// The quarantine lifecycle ledger independently requires this UUID to remain inactive.
    Quarantined,
    /// Both the write-ahead intent and lifecycle ledger require inactivity.
    PendingIntentAndQuarantined,
    /// A restore was prepared but never durably completed; quarantine remains authoritative.
    RestorePendingReconciliation,
}

/// An exact occurrence that may be inserted into the active replay heap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedActiveOccurrence {
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub record_ref: String,
}

/// An exact occurrence that must remain outside replay/retrieval/training after restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedInactiveOccurrence {
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub escrow_persistence_ref: String,
    pub reason: InactiveRestartReason,
    /// True when an operator/reconciler must resolve an incomplete transition rather than treating
    /// the state as an ordinary stable quarantine.
    pub reconciliation_required: bool,
}

/// Historical escrow with no current quarantine or pending intent.
///
/// Escrow is durable audit/recovery evidence and may legitimately outlive a completed restore. It
/// must not, by itself, reactivate or re-quarantine an occurrence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalEscrowOccurrence {
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub escrow_persistence_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodicRestartActivationPlan {
    pub active: Vec<PlannedActiveOccurrence>,
    pub inactive: Vec<PlannedInactiveOccurrence>,
    pub historical_escrow_only: Vec<HistoricalEscrowOccurrence>,
}

impl EpisodicRestartActivationPlan {
    pub fn reconciliation_required(&self) -> bool {
        self.inactive
            .iter()
            .any(|entry| entry.reconciliation_required)
    }
}

/// Build a deterministic fail-closed activation plan.
///
/// The supplied ledgers are assumed to have already passed their own `recover_anchored(...)`
/// checks. This function cross-validates their *semantic agreement* with persisted occurrence and
/// escrow material before any UUID may become active.
pub fn build_episodic_restart_activation_plan(
    store_target_id: &str,
    persisted_occurrences: &[PersistedOccurrenceDescriptor],
    persisted_escrows: &[PersistedQuarantineEscrowDescriptor],
    intent_ledger: &EpisodicQuarantineIntentLedger,
    quarantine_ledger: &EpisodicQuarantineStateLedger,
) -> Result<EpisodicRestartActivationPlan, RestartActivationError> {
    validate_store_target(store_target_id)?;

    let records = index_records(persisted_occurrences)?;
    let escrows = index_escrows(persisted_escrows)?;

    let pending_intents: BTreeMap<_, _> = intent_ledger
        .pending_states()
        .into_iter()
        .map(|state| (state.instance_id, state))
        .collect();
    let quarantine_states: BTreeMap<_, _> = quarantine_ledger
        .unresolved_states()
        .into_iter()
        .map(|state| (state.instance_id, state))
        .collect();

    let mut ids = BTreeSet::new();
    ids.extend(records.keys().copied());
    ids.extend(escrows.keys().copied());
    ids.extend(pending_intents.keys().copied());
    ids.extend(quarantine_states.keys().copied());

    let mut active = Vec::new();
    let mut inactive = Vec::new();
    let mut historical = Vec::new();

    for instance_id in ids {
        let record = records.get(&instance_id);
        let escrow = escrows.get(&instance_id);
        let pending = pending_intents.get(&instance_id);
        let quarantined = quarantine_states.get(&instance_id);

        cross_check_content(instance_id, record, escrow, pending, quarantined)?;
        if let Some(pending) = pending {
            validate_pending(store_target_id, pending, escrow)?;
        }
        if let Some(quarantined) = quarantined {
            validate_quarantine_state(store_target_id, quarantined, escrow)?;
        }
        if let (Some(pending), Some(quarantined)) = (pending, quarantined) {
            if pending.content_id != quarantined.content_id {
                return Err(RestartActivationError::IntentStateContentMismatch {
                    instance_id,
                    intent: pending.content_id,
                    state: quarantined.content_id,
                });
            }
            if pending.target_id != quarantined.target_id {
                return Err(RestartActivationError::IntentStateTargetMismatch {
                    instance_id,
                    intent: pending.target_id.clone(),
                    state: quarantined.target_id.clone(),
                });
            }
            if pending.escrow_digest != quarantined.escrow_digest
                || pending.escrow_persistence_ref != quarantined.escrow_persistence_ref
            {
                return Err(RestartActivationError::IntentStateEscrowMismatch(instance_id));
            }
        }

        match (record, escrow, pending, quarantined) {
            (Some(record), _, None, None) => {
                active.push(PlannedActiveOccurrence {
                    instance_id,
                    content_id: record.content_id,
                    record_ref: record.record_ref.clone(),
                });
            }
            (_, Some(escrow), pending, quarantined)
                if pending.is_some() || quarantined.is_some() =>
            {
                let restore_pending = quarantined
                    .and_then(|state| state.restore_pending.as_ref())
                    .is_some();
                let reason = if restore_pending {
                    InactiveRestartReason::RestorePendingReconciliation
                } else {
                    match (pending.is_some(), quarantined.is_some()) {
                        (true, true) => InactiveRestartReason::PendingIntentAndQuarantined,
                        (true, false) => InactiveRestartReason::PendingQuarantineIntent,
                        (false, true) => InactiveRestartReason::Quarantined,
                        (false, false) => unreachable!("guarded above"),
                    }
                };
                inactive.push(PlannedInactiveOccurrence {
                    instance_id,
                    content_id: escrow.content_id,
                    escrow_persistence_ref: escrow.escrow_persistence_ref.clone(),
                    reason,
                    reconciliation_required: pending.is_some() || restore_pending,
                });
            }
            (None, Some(escrow), None, None) => {
                historical.push(HistoricalEscrowOccurrence {
                    instance_id,
                    content_id: escrow.content_id,
                    escrow_persistence_ref: escrow.escrow_persistence_ref.clone(),
                });
            }
            (Some(_), None, pending, quarantined)
                if pending.is_some() || quarantined.is_some() =>
            {
                // An unresolved lifecycle restriction without its reversible escrow is not safe to
                // activate and not sufficiently complete to reconstruct quarantine state.
                return Err(RestartActivationError::MissingRequiredEscrow(instance_id));
            }
            (None, None, pending, quarantined)
                if pending.is_some() || quarantined.is_some() =>
            {
                return Err(RestartActivationError::MissingOccurrenceAndEscrow(instance_id));
            }
            (None, None, None, None) => unreachable!("ID came from union"),
            (Some(_), Some(escrow), None, None) => {
                // A historical escrow can coexist with a currently active restored record. The
                // current lifecycle ledgers, not escrow existence alone, decide activation.
                let record = record.expect("matched Some above");
                active.push(PlannedActiveOccurrence {
                    instance_id,
                    content_id: record.content_id,
                    record_ref: record.record_ref.clone(),
                });
                let _ = escrow;
            }
        }
    }

    active.sort_by_key(|entry| entry.instance_id);
    inactive.sort_by_key(|entry| entry.instance_id);
    historical.sort_by_key(|entry| entry.instance_id);

    Ok(EpisodicRestartActivationPlan {
        active,
        inactive,
        historical_escrow_only: historical,
    })
}

fn index_records(
    records: &[PersistedOccurrenceDescriptor],
) -> Result<BTreeMap<EpisodeInstanceId, PersistedOccurrenceDescriptor>, RestartActivationError> {
    let mut indexed = BTreeMap::new();
    for record in records {
        validate_ref("record_ref", &record.record_ref)?;
        if indexed.insert(record.instance_id, record.clone()).is_some() {
            return Err(RestartActivationError::DuplicatePersistedOccurrence(
                record.instance_id,
            ));
        }
    }
    Ok(indexed)
}

fn index_escrows(
    escrows: &[PersistedQuarantineEscrowDescriptor],
) -> Result<
    BTreeMap<EpisodeInstanceId, PersistedQuarantineEscrowDescriptor>,
    RestartActivationError,
> {
    let mut indexed = BTreeMap::new();
    for escrow in escrows {
        if escrow.escrow_digest.0 == [0; 32] {
            return Err(RestartActivationError::ZeroDigest {
                field: "escrow_digest",
            });
        }
        validate_ref("escrow_persistence_ref", &escrow.escrow_persistence_ref)?;
        if indexed.insert(escrow.instance_id, escrow.clone()).is_some() {
            return Err(RestartActivationError::DuplicateEscrowOccurrence(
                escrow.instance_id,
            ));
        }
    }
    Ok(indexed)
}

fn cross_check_content(
    instance_id: EpisodeInstanceId,
    record: Option<&PersistedOccurrenceDescriptor>,
    escrow: Option<&PersistedQuarantineEscrowDescriptor>,
    pending: Option<&PendingQuarantineIntent>,
    quarantined: Option<&QuarantineLedgerState>,
) -> Result<(), RestartActivationError> {
    let mut expected: Option<EpisodeContentId> = None;
    for candidate in [
        record.map(|value| value.content_id),
        escrow.map(|value| value.content_id),
        pending.map(|value| value.content_id),
        quarantined.map(|value| value.content_id),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(existing) = expected {
            if existing != candidate {
                return Err(RestartActivationError::ContentIdentityContradiction {
                    instance_id,
                    expected: existing,
                    actual: candidate,
                });
            }
        } else {
            expected = Some(candidate);
        }
    }
    Ok(())
}

fn validate_pending(
    store_target_id: &str,
    pending: &PendingQuarantineIntent,
    escrow: Option<&PersistedQuarantineEscrowDescriptor>,
) -> Result<(), RestartActivationError> {
    let expected_target = exact_target(store_target_id, pending.instance_id)?;
    if pending.target_id != expected_target {
        return Err(RestartActivationError::UnexpectedTarget {
            instance_id: pending.instance_id,
            expected: expected_target,
            actual: pending.target_id.clone(),
        });
    }
    let escrow = escrow.ok_or(RestartActivationError::MissingRequiredEscrow(
        pending.instance_id,
    ))?;
    if pending.escrow_digest != escrow.escrow_digest
        || pending.escrow_persistence_ref != escrow.escrow_persistence_ref
    {
        return Err(RestartActivationError::IntentEscrowMismatch(
            pending.instance_id,
        ));
    }
    Ok(())
}

fn validate_quarantine_state(
    store_target_id: &str,
    state: &QuarantineLedgerState,
    escrow: Option<&PersistedQuarantineEscrowDescriptor>,
) -> Result<(), RestartActivationError> {
    let expected_target = exact_target(store_target_id, state.instance_id)?;
    if state.target_id != expected_target {
        return Err(RestartActivationError::UnexpectedTarget {
            instance_id: state.instance_id,
            expected: expected_target,
            actual: state.target_id.clone(),
        });
    }
    let escrow = escrow.ok_or(RestartActivationError::MissingRequiredEscrow(
        state.instance_id,
    ))?;
    if state.escrow_digest != escrow.escrow_digest
        || state.escrow_persistence_ref != escrow.escrow_persistence_ref
    {
        return Err(RestartActivationError::StateEscrowMismatch(
            state.instance_id,
        ));
    }
    Ok(())
}

fn exact_target(
    store_target_id: &str,
    instance_id: EpisodeInstanceId,
) -> Result<String, RestartActivationError> {
    validate_store_target(store_target_id)?;
    let target = format!("{store_target_id}:instance:{instance_id}");
    if target.len() > MAX_TARGET_BYTES {
        return Err(RestartActivationError::InvalidStoreTarget);
    }
    Ok(target)
}

fn validate_store_target(value: &str) -> Result<(), RestartActivationError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_TARGET_BYTES
        || value.chars().any(char::is_control)
    {
        Err(RestartActivationError::InvalidStoreTarget)
    } else {
        Ok(())
    }
}

fn validate_ref(field: &'static str, value: &str) -> Result<(), RestartActivationError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_REF_BYTES
        || value.chars().any(char::is_control)
    {
        Err(RestartActivationError::InvalidReference { field })
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RestartActivationError {
    #[error("episodic restart store target is invalid")]
    InvalidStoreTarget,
    #[error("invalid restart reference field `{field}`")]
    InvalidReference { field: &'static str },
    #[error("restart digest `{field}` must not be zero")]
    ZeroDigest { field: &'static str },
    #[error("duplicate persisted record for occurrence {0}")]
    DuplicatePersistedOccurrence(EpisodeInstanceId),
    #[error("duplicate persisted escrow for occurrence {0}")]
    DuplicateEscrowOccurrence(EpisodeInstanceId),
    #[error("content identity contradiction for {instance_id}: expected={expected:?}, actual={actual:?}")]
    ContentIdentityContradiction {
        instance_id: EpisodeInstanceId,
        expected: EpisodeContentId,
        actual: EpisodeContentId,
    },
    #[error("unexpected restart target for {instance_id}: expected={expected:?}, actual={actual:?}")]
    UnexpectedTarget {
        instance_id: EpisodeInstanceId,
        expected: String,
        actual: String,
    },
    #[error("pending quarantine intent for {0} does not match persisted escrow")]
    IntentEscrowMismatch(EpisodeInstanceId),
    #[error("quarantine lifecycle state for {0} does not match persisted escrow")]
    StateEscrowMismatch(EpisodeInstanceId),
    #[error("pending intent and quarantine state disagree on content for {instance_id}: intent={intent:?}, state={state:?}")]
    IntentStateContentMismatch {
        instance_id: EpisodeInstanceId,
        intent: EpisodeContentId,
        state: EpisodeContentId,
    },
    #[error("pending intent and quarantine state disagree on target for {instance_id}: intent={intent:?}, state={state:?}")]
    IntentStateTargetMismatch {
        instance_id: EpisodeInstanceId,
        intent: String,
        state: String,
    },
    #[error("pending intent and quarantine state disagree on escrow for {0}")]
    IntentStateEscrowMismatch(EpisodeInstanceId),
    #[error("restricted occurrence {0} is missing required reversible escrow")]
    MissingRequiredEscrow(EpisodeInstanceId),
    #[error("restricted occurrence {0} has neither persisted active record nor reversible escrow")]
    MissingOccurrenceAndEscrow(EpisodeInstanceId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};

    use crate::memory_identity::episode_content_id;

    const STORE: &str = "symthaea:self:episodic-memory";

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn two_duplicate_identities() -> (
        EpisodeInstanceId,
        EpisodeInstanceId,
        EpisodeContentId,
    ) {
        let episode = Episode::new(
            ContinuousHV::from_values(vec![1.0, 2.0]),
            ContinuousHV::from_values(vec![3.0, 4.0]),
            0.9,
            42,
        );
        let content_id = episode_content_id(&episode).unwrap();
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let first = memory.store_if_significant_with_id(episode.clone()).unwrap();
        let second = memory.store_if_significant_with_id(episode).unwrap();
        (first, second, content_id)
    }

    #[test]
    fn one_duplicate_can_remain_quarantined_while_the_other_activates() {
        let (first, second, content_id) = two_duplicate_identities();
        let first_target = exact_target(STORE, first).unwrap();
        let escrow = PersistedQuarantineEscrowDescriptor::new(
            first,
            content_id,
            digest(1),
            "escrow:first",
        )
        .unwrap();
        let records = vec![
            PersistedOccurrenceDescriptor::new(first, content_id, "record:first").unwrap(),
            PersistedOccurrenceDescriptor::new(second, content_id, "record:second").unwrap(),
        ];
        let mut quarantine = EpisodicQuarantineStateLedger::new();
        quarantine
            .append_quarantined(
                &first_target,
                first,
                content_id,
                100,
                digest(1),
                "escrow:first",
            )
            .unwrap();
        let plan = build_episodic_restart_activation_plan(
            STORE,
            &records,
            &[escrow],
            &EpisodicQuarantineIntentLedger::new(),
            &quarantine,
        )
        .unwrap();
        assert_eq!(plan.active.len(), 1);
        assert_eq!(plan.active[0].instance_id, second);
        assert_eq!(plan.inactive.len(), 1);
        assert_eq!(plan.inactive[0].instance_id, first);
        assert_eq!(plan.inactive[0].reason, InactiveRestartReason::Quarantined);
    }

    #[test]
    fn pending_intent_without_final_state_is_still_inactive_and_needs_reconciliation() {
        let (first, _, content_id) = two_duplicate_identities();
        let target = exact_target(STORE, first).unwrap();
        let record = PersistedOccurrenceDescriptor::new(first, content_id, "record:first").unwrap();
        let escrow = PersistedQuarantineEscrowDescriptor::new(
            first,
            content_id,
            digest(2),
            "escrow:first",
        )
        .unwrap();
        let mut intent = EpisodicQuarantineIntentLedger::new();
        intent
            .append_prepared(
                "exec:q:pending",
                target,
                first,
                content_id,
                100,
                digest(3),
                digest(2),
                "escrow:first",
            )
            .unwrap();
        let plan = build_episodic_restart_activation_plan(
            STORE,
            &[record],
            &[escrow],
            &intent,
            &EpisodicQuarantineStateLedger::new(),
        )
        .unwrap();
        assert!(plan.active.is_empty());
        assert_eq!(plan.inactive.len(), 1);
        assert_eq!(
            plan.inactive[0].reason,
            InactiveRestartReason::PendingQuarantineIntent
        );
        assert!(plan.inactive[0].reconciliation_required);
    }

    #[test]
    fn pending_restore_never_becomes_active_implicitly() {
        let (first, _, content_id) = two_duplicate_identities();
        let target = exact_target(STORE, first).unwrap();
        let escrow = PersistedQuarantineEscrowDescriptor::new(
            first,
            content_id,
            digest(4),
            "escrow:first",
        )
        .unwrap();
        let mut quarantine = EpisodicQuarantineStateLedger::new();
        quarantine
            .append_quarantined(
                &target,
                first,
                content_id,
                100,
                digest(4),
                "escrow:first",
            )
            .unwrap();
        quarantine
            .append_restore_prepared(&target, first, content_id, 110, "exec:restore:1")
            .unwrap();
        let plan = build_episodic_restart_activation_plan(
            STORE,
            &[],
            &[escrow],
            &EpisodicQuarantineIntentLedger::new(),
            &quarantine,
        )
        .unwrap();
        assert!(plan.active.is_empty());
        assert_eq!(
            plan.inactive[0].reason,
            InactiveRestartReason::RestorePendingReconciliation
        );
        assert!(plan.reconciliation_required());
    }

    #[test]
    fn duplicate_uuid_rows_fail_closed_even_when_content_matches() {
        let (first, _, content_id) = two_duplicate_identities();
        let records = vec![
            PersistedOccurrenceDescriptor::new(first, content_id, "record:a").unwrap(),
            PersistedOccurrenceDescriptor::new(first, content_id, "record:b").unwrap(),
        ];
        assert!(matches!(
            build_episodic_restart_activation_plan(
                STORE,
                &records,
                &[],
                &EpisodicQuarantineIntentLedger::new(),
                &EpisodicQuarantineStateLedger::new(),
            ),
            Err(RestartActivationError::DuplicatePersistedOccurrence(id)) if id == first
        ));
    }

    #[test]
    fn content_mismatch_between_record_and_quarantine_fails_closed() {
        let (first, _, content_id) = two_duplicate_identities();
        let other_episode = Episode::new(
            ContinuousHV::from_values(vec![9.0, 9.0]),
            ContinuousHV::from_values(vec![8.0, 8.0]),
            0.9,
            99,
        );
        let other_content = episode_content_id(&other_episode).unwrap();
        assert_ne!(content_id, other_content);
        let target = exact_target(STORE, first).unwrap();
        let record = PersistedOccurrenceDescriptor::new(first, other_content, "record:first").unwrap();
        let escrow = PersistedQuarantineEscrowDescriptor::new(
            first,
            content_id,
            digest(5),
            "escrow:first",
        )
        .unwrap();
        let mut quarantine = EpisodicQuarantineStateLedger::new();
        quarantine
            .append_quarantined(
                target,
                first,
                content_id,
                100,
                digest(5),
                "escrow:first",
            )
            .unwrap();
        assert!(matches!(
            build_episodic_restart_activation_plan(
                STORE,
                &[record],
                &[escrow],
                &EpisodicQuarantineIntentLedger::new(),
                &quarantine,
            ),
            Err(RestartActivationError::ContentIdentityContradiction { instance_id, .. })
                if instance_id == first
        ));
    }

    #[test]
    fn historical_escrow_does_not_override_completed_restore_state() {
        let (first, _, content_id) = two_duplicate_identities();
        let record = PersistedOccurrenceDescriptor::new(first, content_id, "record:first").unwrap();
        let escrow = PersistedQuarantineEscrowDescriptor::new(
            first,
            content_id,
            digest(6),
            "escrow:historical",
        )
        .unwrap();
        let plan = build_episodic_restart_activation_plan(
            STORE,
            &[record],
            &[escrow],
            &EpisodicQuarantineIntentLedger::new(),
            &EpisodicQuarantineStateLedger::new(),
        )
        .unwrap();
        assert_eq!(plan.active.len(), 1);
        assert!(plan.inactive.is_empty());
    }
}
