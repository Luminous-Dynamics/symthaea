// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! External monotonic anchor protocol for exact episodic restart continuity.
//!
//! Local SQLite + hash chains provide durable integrity, but they cannot by themselves detect a
//! rollback in which the database and its internal hashes are restored together. This crate makes
//! the missing trust boundary explicit: an external anchor stores one atomic commitment covering
//! the *entire* restart-relevant continuity state, and advances through compare-and-swap (CAS).
//!
//! The anchor commits more than quarantine ledger heads. It also commits the exact active episode
//! envelopes and reversible escrow set. This prevents an attacker or stale backup from restoring
//! older cognitive/lifecycle state while leaving governance heads unchanged.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_episodic_continuity::{
    ContinuityStoreError, RecoveredEpisodicContinuity, SqliteEpisodicContinuityStore,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_welfare_assurance::quarantine_intent_ledger::EpisodicQuarantineIntentLedger;
use symthaea_welfare_assurance::quarantine_state_ledger::EpisodicQuarantineStateLedger;
use thiserror::Error;

pub const CONTINUITY_ANCHOR_SCHEMA: &str = "symthaea.episodic-continuity.anchor.v1";
const MANIFEST_DOMAIN: &[u8] = b"symthaea.episodic-continuity.manifest.v1\0";
const ANCHOR_COMMITMENT_DOMAIN: &[u8] = b"symthaea.episodic-continuity.anchor-commitment.v1\0";
const MAX_TARGET_BYTES: usize = 256;
const MAX_REF_BYTES: usize = 2048;

/// Atomic external commitment to one fully durable continuity state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuityAnchorSnapshot {
    pub schema_version: String,
    pub store_target_id: String,
    pub revision: u64,
    pub intent_generation: u64,
    pub intent_head: Sha256Digest,
    pub quarantine_generation: u64,
    pub quarantine_head: Sha256Digest,
    pub continuity_manifest_digest: Sha256Digest,
    pub committed_at_unix_s: u64,
    pub previous_anchor_commitment: Option<Sha256Digest>,
}

impl ContinuityAnchorSnapshot {
    pub fn validate(&self) -> Result<(), ContinuityAnchorError> {
        if self.schema_version != CONTINUITY_ANCHOR_SCHEMA {
            return Err(ContinuityAnchorError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        validate_text("store_target_id", &self.store_target_id, MAX_TARGET_BYTES)?;
        if self.revision == 0 {
            return Err(ContinuityAnchorError::ZeroRevision);
        }
        for (field, digest) in [
            ("intent_head", self.intent_head),
            ("quarantine_head", self.quarantine_head),
            ("continuity_manifest_digest", self.continuity_manifest_digest),
        ] {
            if digest.0 == [0; 32] {
                return Err(ContinuityAnchorError::ZeroDigest { field });
            }
        }
        if self.revision == 1 && self.previous_anchor_commitment.is_some() {
            return Err(ContinuityAnchorError::UnexpectedPreviousCommitment);
        }
        if self.revision > 1 && self.previous_anchor_commitment.is_none() {
            return Err(ContinuityAnchorError::MissingPreviousCommitment);
        }
        Ok(())
    }

    pub fn commitment(&self) -> Result<Sha256Digest, ContinuityAnchorError> {
        self.validate()?;
        let encoded = bincode::serialize(self)
            .map_err(|error| ContinuityAnchorError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(ANCHOR_COMMITMENT_DOMAIN);
        hasher.update(&(encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
        Ok(hasher.finalize())
    }
}

/// External monotonic/trusted anchor boundary.
///
/// Implementations may use TPM NV counters, a TEE, Xenia/Mycelix consensus, or another independent
/// root. `compare_and_swap` must atomically reject if the currently stored commitment differs from
/// `expected_current`. The returned reference is audit evidence, not authority by itself.
pub trait ContinuityHeadAnchor {
    type Error: StdError + Send + Sync + 'static;

    fn load(
        &self,
        store_target_id: &str,
    ) -> Result<Option<ContinuityAnchorSnapshot>, Self::Error>;

    fn compare_and_swap(
        &mut self,
        store_target_id: &str,
        expected_current: Option<Sha256Digest>,
        next: &ContinuityAnchorSnapshot,
    ) -> Result<String, Self::Error>;
}

/// Successful recovery tied to an independently loaded anchor snapshot.
pub struct AnchoredRecoveredContinuity {
    pub anchor: ContinuityAnchorSnapshot,
    pub anchor_commitment: Sha256Digest,
    pub recovered: RecoveredEpisodicContinuity,
}

/// Explicit one-time trust ceremony for a store that does not yet have an external anchor.
///
/// This may bootstrap a non-empty store and therefore must be treated as a provisioning decision,
/// not ordinary recovery. After revision 1 exists, callers must use CAS advancement.
pub fn bootstrap_anchor_from_store<A: ContinuityHeadAnchor>(
    store: &SqliteEpisodicContinuityStore,
    anchor: &mut A,
    store_target_id: &str,
    committed_at_unix_s: u64,
) -> Result<(ContinuityAnchorSnapshot, String), AnchorProtocolError<A::Error>> {
    validate_target(store_target_id)?;
    if anchor
        .load(store_target_id)
        .map_err(AnchorProtocolError::Anchor)?
        .is_some()
    {
        return Err(AnchorProtocolError::AlreadyAnchored);
    }

    let intent_genesis = EpisodicQuarantineIntentLedger::new();
    let quarantine_genesis = EpisodicQuarantineStateLedger::new();
    let recovered = store.recover_restart(
        store_target_id,
        intent_genesis.head_hash(),
        quarantine_genesis.head_hash(),
    )?;
    let manifest = continuity_manifest_digest(
        store,
        store_target_id,
        &recovered.intent_ledger,
        &recovered.quarantine_ledger,
    )?;
    let snapshot = ContinuityAnchorSnapshot {
        schema_version: CONTINUITY_ANCHOR_SCHEMA.into(),
        store_target_id: store_target_id.into(),
        revision: 1,
        intent_generation: recovered.intent_ledger.generation(),
        intent_head: recovered.intent_ledger.head_hash(),
        quarantine_generation: recovered.quarantine_ledger.generation(),
        quarantine_head: recovered.quarantine_ledger.head_hash(),
        continuity_manifest_digest: manifest,
        committed_at_unix_s,
        previous_anchor_commitment: None,
    };
    snapshot.validate()?;
    let reference = anchor
        .compare_and_swap(store_target_id, None, &snapshot)
        .map_err(AnchorProtocolError::Anchor)?;
    validate_ref(&reference)?;
    Ok((snapshot, reference))
}

/// Recover exact continuity state only when the external anchor and local durable state agree.
pub fn recover_with_anchor<A: ContinuityHeadAnchor>(
    store: &SqliteEpisodicContinuityStore,
    anchor: &A,
    store_target_id: &str,
) -> Result<AnchoredRecoveredContinuity, AnchorProtocolError<A::Error>> {
    validate_target(store_target_id)?;
    let snapshot = anchor
        .load(store_target_id)
        .map_err(AnchorProtocolError::Anchor)?
        .ok_or(AnchorProtocolError::MissingAnchor)?;
    snapshot.validate()?;
    if snapshot.store_target_id != store_target_id {
        return Err(AnchorProtocolError::AnchorTargetMismatch);
    }
    let commitment = snapshot.commitment()?;
    let recovered = store.recover_restart(
        store_target_id,
        snapshot.intent_head,
        snapshot.quarantine_head,
    )?;
    if recovered.intent_ledger.generation() != snapshot.intent_generation {
        return Err(AnchorProtocolError::GenerationMismatch {
            field: "intent_generation",
            anchored: snapshot.intent_generation,
            recovered: recovered.intent_ledger.generation(),
        });
    }
    if recovered.quarantine_ledger.generation() != snapshot.quarantine_generation {
        return Err(AnchorProtocolError::GenerationMismatch {
            field: "quarantine_generation",
            anchored: snapshot.quarantine_generation,
            recovered: recovered.quarantine_ledger.generation(),
        });
    }
    let manifest = continuity_manifest_digest(
        store,
        store_target_id,
        &recovered.intent_ledger,
        &recovered.quarantine_ledger,
    )?;
    if manifest != snapshot.continuity_manifest_digest {
        return Err(AnchorProtocolError::ManifestMismatch {
            anchored: snapshot.continuity_manifest_digest,
            recovered: manifest,
        });
    }
    Ok(AnchoredRecoveredContinuity {
        anchor: snapshot,
        anchor_commitment: commitment,
        recovered,
    })
}

/// Advance the external anchor *after* the new local state is already durably persisted.
///
/// The local store is fully recovered against the proposed heads first. If the process dies after
/// the local commit but before this CAS, subsequent recovery fails closed against the older anchor
/// and requires reconciliation. The reverse ordering (anchor first, local persistence second) is
/// intentionally not provided.
#[allow(clippy::too_many_arguments)]
pub fn advance_anchor_after_durable_store<A: ContinuityHeadAnchor>(
    store: &SqliteEpisodicContinuityStore,
    anchor: &mut A,
    previous: &ContinuityAnchorSnapshot,
    new_intent_head: Sha256Digest,
    new_quarantine_head: Sha256Digest,
    committed_at_unix_s: u64,
) -> Result<(ContinuityAnchorSnapshot, String), AnchorProtocolError<A::Error>> {
    previous.validate()?;
    let current = anchor
        .load(&previous.store_target_id)
        .map_err(AnchorProtocolError::Anchor)?
        .ok_or(AnchorProtocolError::MissingAnchor)?;
    let expected_commitment = previous.commitment()?;
    let current_commitment = current.commitment()?;
    if current_commitment != expected_commitment {
        return Err(AnchorProtocolError::StaleWriter {
            expected: expected_commitment,
            actual: current_commitment,
        });
    }

    let recovered = store.recover_restart(
        &previous.store_target_id,
        new_intent_head,
        new_quarantine_head,
    )?;
    if recovered.intent_ledger.generation() < previous.intent_generation {
        return Err(AnchorProtocolError::GenerationRegression {
            field: "intent_generation",
            previous: previous.intent_generation,
            next: recovered.intent_ledger.generation(),
        });
    }
    if recovered.quarantine_ledger.generation() < previous.quarantine_generation {
        return Err(AnchorProtocolError::GenerationRegression {
            field: "quarantine_generation",
            previous: previous.quarantine_generation,
            next: recovered.quarantine_ledger.generation(),
        });
    }
    let manifest = continuity_manifest_digest(
        store,
        &previous.store_target_id,
        &recovered.intent_ledger,
        &recovered.quarantine_ledger,
    )?;
    if recovered.intent_ledger.generation() == previous.intent_generation
        && recovered.quarantine_ledger.generation() == previous.quarantine_generation
        && manifest == previous.continuity_manifest_digest
    {
        return Err(AnchorProtocolError::NoContinuityChange);
    }

    let next = ContinuityAnchorSnapshot {
        schema_version: CONTINUITY_ANCHOR_SCHEMA.into(),
        store_target_id: previous.store_target_id.clone(),
        revision: previous
            .revision
            .checked_add(1)
            .ok_or(AnchorProtocolError::RevisionOverflow)?,
        intent_generation: recovered.intent_ledger.generation(),
        intent_head: recovered.intent_ledger.head_hash(),
        quarantine_generation: recovered.quarantine_ledger.generation(),
        quarantine_head: recovered.quarantine_ledger.head_hash(),
        continuity_manifest_digest: manifest,
        committed_at_unix_s,
        previous_anchor_commitment: Some(expected_commitment),
    };
    next.validate()?;
    let reference = anchor
        .compare_and_swap(
            &previous.store_target_id,
            Some(expected_commitment),
            &next,
        )
        .map_err(AnchorProtocolError::Anchor)?;
    validate_ref(&reference)?;
    Ok((next, reference))
}

/// Digest the complete exact continuity state relevant to restart activation.
pub fn continuity_manifest_digest(
    store: &SqliteEpisodicContinuityStore,
    store_target_id: &str,
    intent_ledger: &EpisodicQuarantineIntentLedger,
    quarantine_ledger: &EpisodicQuarantineStateLedger,
) -> Result<Sha256Digest, ContinuityAnchorError> {
    validate_target(store_target_id)?;
    let mut occurrences = store
        .load_validated_occurrences()
        .map_err(|error| ContinuityAnchorError::Store(error.to_string()))?;
    occurrences.sort_by_key(|record| record.instance_id());
    let mut escrows = store
        .load_escrow_descriptors()
        .map_err(|error| ContinuityAnchorError::Store(error.to_string()))?;
    escrows.sort_by_key(|escrow| escrow.instance_id);

    let mut hasher = Sha256::new();
    hasher.update(MANIFEST_DOMAIN);
    hash_text(&mut hasher, store_target_id);
    hasher.update(&(occurrences.len() as u64).to_le_bytes());
    for record in occurrences {
        if record.store_target_id() != store_target_id {
            return Err(ContinuityAnchorError::RecordTargetMismatch {
                instance_id: record.instance_id().to_string(),
            });
        }
        hasher.update(&record.instance_id().as_uuid().as_u128().to_le_bytes());
        hasher.update(&record.content_id().digest().0);
        hash_text(&mut hasher, record.record_ref());
        let episode_bytes = bincode::serialize(record.episode())
            .map_err(|error| ContinuityAnchorError::Encoding(error.to_string()))?;
        hasher.update(&(episode_bytes.len() as u64).to_le_bytes());
        hasher.update(&episode_bytes);
    }

    hasher.update(&(escrows.len() as u64).to_le_bytes());
    for escrow in escrows {
        hasher.update(&escrow.instance_id.as_uuid().as_u128().to_le_bytes());
        hasher.update(&escrow.content_id.digest().0);
        hasher.update(&escrow.escrow_digest.0);
        hash_text(&mut hasher, &escrow.escrow_persistence_ref);
    }

    hasher.update(&intent_ledger.generation().to_le_bytes());
    hasher.update(&intent_ledger.head_hash().0);
    hasher.update(&quarantine_ledger.generation().to_le_bytes());
    hasher.update(&quarantine_ledger.head_hash().0);
    Ok(hasher.finalize())
}

fn validate_target(value: &str) -> Result<(), ContinuityAnchorError> {
    validate_text("store_target_id", value, MAX_TARGET_BYTES)
}

fn validate_ref(value: &str) -> Result<(), ContinuityAnchorError> {
    validate_text("anchor_persistence_ref", value, MAX_REF_BYTES)
}

fn validate_text(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), ContinuityAnchorError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > max
        || value.chars().any(char::is_control)
    {
        Err(ContinuityAnchorError::InvalidText { field })
    } else {
        Ok(())
    }
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ContinuityAnchorError {
    #[error("unsupported continuity anchor schema: {0:?}")]
    UnsupportedSchema(String),
    #[error("continuity anchor revision must be nonzero")]
    ZeroRevision,
    #[error("continuity anchor digest `{field}` must not be zero")]
    ZeroDigest { field: &'static str },
    #[error("revision 1 anchor must not contain a previous commitment")]
    UnexpectedPreviousCommitment,
    #[error("revision >1 anchor must contain the previous commitment")]
    MissingPreviousCommitment,
    #[error("invalid continuity anchor text field `{field}`")]
    InvalidText { field: &'static str },
    #[error("continuity anchor encoding failed: {0}")]
    Encoding(String),
    #[error("continuity store validation failed: {0}")]
    Store(String),
    #[error("persisted occurrence belongs to the wrong continuity target: {instance_id}")]
    RecordTargetMismatch { instance_id: String },
}

#[derive(Debug, Error)]
pub enum AnchorProtocolError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Validation(#[from] ContinuityAnchorError),
    #[error(transparent)]
    Store(#[from] ContinuityStoreError),
    #[error("external continuity anchor failed: {0}")]
    Anchor(#[source] E),
    #[error("continuity target has no external anchor")]
    MissingAnchor,
    #[error("continuity target is already externally anchored")]
    AlreadyAnchored,
    #[error("external anchor belongs to a different continuity target")]
    AnchorTargetMismatch,
    #[error("anchored `{field}` disagrees with recovered generation: anchored={anchored}, recovered={recovered}")]
    GenerationMismatch {
        field: &'static str,
        anchored: u64,
        recovered: u64,
    },
    #[error("continuity manifest disagrees with external anchor")]
    ManifestMismatch {
        anchored: Sha256Digest,
        recovered: Sha256Digest,
    },
    #[error("anchor compare-and-swap observed a stale writer")]
    StaleWriter {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("continuity `{field}` regressed: previous={previous}, next={next}")]
    GenerationRegression {
        field: &'static str,
        previous: u64,
        next: u64,
    },
    #[error("continuity anchor revision overflow")]
    RevisionOverflow,
    #[error("continuity state has not changed")]
    NoContinuityChange,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fmt;

    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
    use symthaea_welfare_assurance::persisted_episode_envelope::PersistedEpisodicEnvelope;

    const STORE: &str = "symthaea:self:episodic-memory";

    #[derive(Debug)]
    struct MemoryAnchorError(&'static str);

    impl fmt::Display for MemoryAnchorError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl StdError for MemoryAnchorError {}

    #[derive(Default)]
    struct MemoryAnchor {
        snapshots: HashMap<String, ContinuityAnchorSnapshot>,
    }

    impl ContinuityHeadAnchor for MemoryAnchor {
        type Error = MemoryAnchorError;

        fn load(
            &self,
            store_target_id: &str,
        ) -> Result<Option<ContinuityAnchorSnapshot>, Self::Error> {
            Ok(self.snapshots.get(store_target_id).cloned())
        }

        fn compare_and_swap(
            &mut self,
            store_target_id: &str,
            expected_current: Option<Sha256Digest>,
            next: &ContinuityAnchorSnapshot,
        ) -> Result<String, Self::Error> {
            let actual = self
                .snapshots
                .get(store_target_id)
                .map(|snapshot| snapshot.commitment())
                .transpose()
                .map_err(|_| MemoryAnchorError("invalid current snapshot"))?;
            if actual != expected_current {
                return Err(MemoryAnchorError("compare-and-swap mismatch"));
            }
            next.validate()
                .map_err(|_| MemoryAnchorError("invalid next snapshot"))?;
            self.snapshots
                .insert(store_target_id.to_string(), next.clone());
            Ok(format!("memory-anchor:revision:{}", next.revision))
        }
    }

    fn stored_episode(seed: f32) -> Episode {
        let source = Episode::new(
            ContinuousHV::from_values(vec![seed, seed + 1.0]),
            ContinuousHV::from_values(vec![seed + 2.0, seed + 3.0]),
            0.8,
            seed as u64 + 10,
        );
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let id = memory.store_if_significant_with_id(source).unwrap();
        memory
            .get_top_episode_instances(1)
            .into_iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, episode)| episode)
            .unwrap()
    }

    #[test]
    fn bootstrap_and_recover_empty_store() {
        let store = SqliteEpisodicContinuityStore::in_memory().unwrap();
        let mut anchor = MemoryAnchor::default();
        let (snapshot, reference) =
            bootstrap_anchor_from_store(&store, &mut anchor, STORE, 100).unwrap();
        assert_eq!(snapshot.revision, 1);
        assert_eq!(reference, "memory-anchor:revision:1");
        let recovered = recover_with_anchor(&store, &anchor, STORE).unwrap();
        assert_eq!(recovered.anchor_commitment, snapshot.commitment().unwrap());
        assert!(recovered.recovered.import_batch.active().is_empty());
    }

    #[test]
    fn local_occurrence_change_without_anchor_advance_fails_recovery() {
        let mut store = SqliteEpisodicContinuityStore::in_memory().unwrap();
        let mut anchor = MemoryAnchor::default();
        let (snapshot, _) =
            bootstrap_anchor_from_store(&store, &mut anchor, STORE, 100).unwrap();

        let episode = stored_episode(1.0);
        store
            .upsert_occurrence(&PersistedEpisodicEnvelope::new(STORE, episode, 110, 1).unwrap())
            .unwrap();
        assert!(matches!(
            recover_with_anchor(&store, &anchor, STORE),
            Err(AnchorProtocolError::ManifestMismatch { .. })
        ));

        let (advanced, _) = advance_anchor_after_durable_store(
            &store,
            &mut anchor,
            &snapshot,
            snapshot.intent_head,
            snapshot.quarantine_head,
            120,
        )
        .unwrap();
        assert_eq!(advanced.revision, 2);
        assert!(recover_with_anchor(&store, &anchor, STORE).is_ok());
    }

    #[test]
    fn stale_writer_cannot_overwrite_newer_anchor_revision() {
        let mut store = SqliteEpisodicContinuityStore::in_memory().unwrap();
        let mut anchor = MemoryAnchor::default();
        let (revision_one, _) =
            bootstrap_anchor_from_store(&store, &mut anchor, STORE, 100).unwrap();
        let stale_copy = revision_one.clone();

        store
            .upsert_occurrence(
                &PersistedEpisodicEnvelope::new(STORE, stored_episode(2.0), 110, 1).unwrap(),
            )
            .unwrap();
        let (revision_two, _) = advance_anchor_after_durable_store(
            &store,
            &mut anchor,
            &revision_one,
            revision_one.intent_head,
            revision_one.quarantine_head,
            120,
        )
        .unwrap();
        assert_eq!(revision_two.revision, 2);

        store
            .upsert_occurrence(
                &PersistedEpisodicEnvelope::new(STORE, stored_episode(3.0), 130, 2).unwrap(),
            )
            .unwrap();
        assert!(matches!(
            advance_anchor_after_durable_store(
                &store,
                &mut anchor,
                &stale_copy,
                stale_copy.intent_head,
                stale_copy.quarantine_head,
                140,
            ),
            Err(AnchorProtocolError::StaleWriter { .. })
        ));
    }

    #[test]
    fn anchor_manifest_binds_lifecycle_state_not_only_content_identity() {
        let mut store = SqliteEpisodicContinuityStore::in_memory().unwrap();
        let mut episode = stored_episode(4.0);
        let original_content = symthaea_welfare_assurance::memory_identity::episode_content_id(&episode)
            .unwrap();
        store
            .upsert_occurrence(
                &PersistedEpisodicEnvelope::new(STORE, episode.clone(), 100, 1).unwrap(),
            )
            .unwrap();
        let mut anchor = MemoryAnchor::default();
        let (revision_one, _) =
            bootstrap_anchor_from_store(&store, &mut anchor, STORE, 110).unwrap();

        episode.replay_count += 1;
        episode.retrieval_count += 1;
        episode.consolidation_strength += 0.5;
        assert_eq!(
            original_content,
            symthaea_welfare_assurance::memory_identity::episode_content_id(&episode).unwrap()
        );
        store
            .upsert_occurrence(
                &PersistedEpisodicEnvelope::new(STORE, episode, 120, 2).unwrap(),
            )
            .unwrap();

        assert!(matches!(
            recover_with_anchor(&store, &anchor, STORE),
            Err(AnchorProtocolError::ManifestMismatch { .. })
        ));
        let (revision_two, _) = advance_anchor_after_durable_store(
            &store,
            &mut anchor,
            &revision_one,
            revision_one.intent_head,
            revision_one.quarantine_head,
            130,
        )
        .unwrap();
        assert_ne!(
            revision_one.continuity_manifest_digest,
            revision_two.continuity_manifest_digest
        );
    }
}
