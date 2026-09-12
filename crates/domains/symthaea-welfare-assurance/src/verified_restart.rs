// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Positive restart capability minted only from externally anchored lifecycle evidence.
//!
//! The lower restart planner remains useful as a pure theorem/test surface, but production restart
//! should not rely on a comment saying "the caller already verified the anchors". This module
//! performs both ledger recoveries itself, validates all durable episode/escrow material, and then
//! returns a non-Clone, non-serializable permit containing the resulting activation plan.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use thiserror::Error;

use crate::episodic_persistence_envelope::{
    EpisodicPersistenceEnvelopeError, digest_persisted_episode_envelope,
};
use crate::quarantine_intent_ledger::{
    EpisodicQuarantineIntentLedger, QuarantineIntentEnvelope, QuarantineIntentLedgerError,
};
use crate::quarantine_state_ledger::{
    EpisodicQuarantineStateLedger, QuarantineLedgerEnvelope, QuarantineLedgerError,
};
use crate::restart_activation_gate::{
    EpisodicRestartActivationPlan, PlannedActiveOccurrence, PlannedInactiveOccurrence,
};
use crate::restart_materialization::{
    DurableEpisodeRecord, DurableEscrowRecord, RestartMaterializationError,
    build_restart_plan_from_durable_material,
};

pub const VERIFIED_EPISODIC_RESTART_SCHEMA: &str =
    "symthaea.welfare.verified-episodic-restart.v1";
const PLAN_DOMAIN: &[u8] = b"symthaea.welfare.restart-plan.v1\0";
const MATERIAL_DOMAIN: &[u8] = b"symthaea.welfare.restart-material.v1\0";
const MAX_TARGET_BYTES: usize = 256;
const MAX_REF_BYTES: usize = 2048;

/// Non-authoritative evidence describing exactly what a verified permit committed to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedRestartReceipt {
    pub schema_version: String,
    pub store_target_id: String,
    pub trusted_intent_head: Sha256Digest,
    pub trusted_quarantine_head: Sha256Digest,
    pub activation_plan_digest: Sha256Digest,
    pub durable_material_digest: Sha256Digest,
    pub active_count: usize,
    pub inactive_count: usize,
    pub historical_escrow_count: usize,
    pub reconciliation_required: bool,
}

/// Positive restart capability.
///
/// Deliberately not `Clone`, `Serialize`, or constructible outside this module. Holding a plan-like
/// struct or a pair of raw ledgers is not equivalent to holding this permit.
#[derive(Debug)]
pub struct VerifiedEpisodicRestartPermit {
    plan: EpisodicRestartActivationPlan,
    receipt: VerifiedRestartReceipt,
}

impl VerifiedEpisodicRestartPermit {
    pub fn active_occurrences(&self) -> &[PlannedActiveOccurrence] {
        &self.plan.active
    }

    pub fn inactive_occurrences(&self) -> &[PlannedInactiveOccurrence] {
        &self.plan.inactive
    }

    pub fn reconciliation_required(&self) -> bool {
        self.plan.reconciliation_required()
    }

    pub fn receipt(&self) -> &VerifiedRestartReceipt {
        &self.receipt
    }

    /// Consume the capability into its validated activation plan.
    ///
    /// The eventual canonical importer should accept this permit by value through the assurance
    /// adapter; exposing consumption here supports that adapter without making the permit clonable.
    pub fn into_activation_plan(self) -> EpisodicRestartActivationPlan {
        self.plan
    }
}

/// Recover both lifecycle ledgers against caller-supplied trusted external heads, validate every
/// durable episode/escrow object, and mint the positive restart capability.
#[allow(clippy::too_many_arguments)]
pub fn verify_episodic_restart(
    store_target_id: &str,
    episode_records: &[DurableEpisodeRecord],
    escrow_records: &[DurableEscrowRecord],
    intent_events: &[QuarantineIntentEnvelope],
    trusted_intent_head: Sha256Digest,
    quarantine_events: &[QuarantineLedgerEnvelope],
    trusted_quarantine_head: Sha256Digest,
) -> Result<VerifiedEpisodicRestartPermit, VerifiedRestartError> {
    validate_target(store_target_id)?;

    let intent_ledger = EpisodicQuarantineIntentLedger::recover_anchored(
        intent_events,
        trusted_intent_head,
    )?;
    let quarantine_ledger = EpisodicQuarantineStateLedger::recover_anchored(
        quarantine_events,
        trusted_quarantine_head,
    )?;

    let plan = build_restart_plan_from_durable_material(
        store_target_id,
        episode_records,
        escrow_records,
        &intent_ledger,
        &quarantine_ledger,
    )?;

    let activation_plan_digest = digest_plan(&plan)?;
    let durable_material_digest = digest_material(
        store_target_id,
        episode_records,
        escrow_records,
        trusted_intent_head,
        trusted_quarantine_head,
        activation_plan_digest,
    )?;

    let receipt = VerifiedRestartReceipt {
        schema_version: VERIFIED_EPISODIC_RESTART_SCHEMA.into(),
        store_target_id: store_target_id.to_string(),
        trusted_intent_head,
        trusted_quarantine_head,
        activation_plan_digest,
        durable_material_digest,
        active_count: plan.active.len(),
        inactive_count: plan.inactive.len(),
        historical_escrow_count: plan.historical_escrow_only.len(),
        reconciliation_required: plan.reconciliation_required(),
    };

    Ok(VerifiedEpisodicRestartPermit { plan, receipt })
}

fn digest_plan(
    plan: &EpisodicRestartActivationPlan,
) -> Result<Sha256Digest, VerifiedRestartError> {
    let encoded = bincode::serialize(plan)
        .map_err(|error| VerifiedRestartError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(PLAN_DOMAIN);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

fn digest_material(
    store_target_id: &str,
    episode_records: &[DurableEpisodeRecord],
    escrow_records: &[DurableEscrowRecord],
    trusted_intent_head: Sha256Digest,
    trusted_quarantine_head: Sha256Digest,
    activation_plan_digest: Sha256Digest,
) -> Result<Sha256Digest, VerifiedRestartError> {
    let mut episodes = Vec::with_capacity(episode_records.len());
    for record in episode_records {
        validate_ref(&record.durable_record_ref)?;
        episodes.push((
            record.envelope.instance_id,
            digest_persisted_episode_envelope(&record.envelope)?,
            record.durable_record_ref.clone(),
        ));
    }
    episodes.sort_by_key(|entry| entry.0);

    let mut escrows = Vec::with_capacity(escrow_records.len());
    for record in escrow_records {
        validate_ref(&record.durable_escrow_ref)?;
        if record.escrow_digest.0 == [0; 32] {
            return Err(VerifiedRestartError::ZeroDigest("escrow_digest"));
        }
        escrows.push((
            record.escrow.instance_id,
            record.escrow_digest,
            record.durable_escrow_ref.clone(),
        ));
    }
    escrows.sort_by_key(|entry| entry.0);

    let mut hasher = Sha256::new();
    hasher.update(MATERIAL_DOMAIN);
    hash_text(&mut hasher, store_target_id);
    hasher.update(&trusted_intent_head.0);
    hasher.update(&trusted_quarantine_head.0);
    hasher.update(&activation_plan_digest.0);
    hasher.update(&(episodes.len() as u64).to_le_bytes());
    for (instance_id, digest, reference) in episodes {
        hasher.update(&instance_id.as_uuid().as_u128().to_le_bytes());
        hasher.update(&digest.0);
        hash_text(&mut hasher, &reference);
    }
    hasher.update(&(escrows.len() as u64).to_le_bytes());
    for (instance_id, digest, reference) in escrows {
        hasher.update(&instance_id.as_uuid().as_u128().to_le_bytes());
        hasher.update(&digest.0);
        hash_text(&mut hasher, &reference);
    }
    Ok(hasher.finalize())
}

fn validate_target(value: &str) -> Result<(), VerifiedRestartError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_TARGET_BYTES
        || value.chars().any(char::is_control)
    {
        Err(VerifiedRestartError::InvalidStoreTarget)
    } else {
        Ok(())
    }
}

fn validate_ref(value: &str) -> Result<(), VerifiedRestartError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_REF_BYTES
        || value.chars().any(char::is_control)
    {
        Err(VerifiedRestartError::InvalidDurableReference)
    } else {
        Ok(())
    }
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[derive(Debug, Error)]
pub enum VerifiedRestartError {
    #[error("episodic restart store target is invalid")]
    InvalidStoreTarget,
    #[error("episodic restart durable reference is invalid")]
    InvalidDurableReference,
    #[error("episodic restart digest `{0}` may not be zero")]
    ZeroDigest(&'static str),
    #[error("episodic restart evidence encoding failed: {0}")]
    Encoding(String),
    #[error("quarantine-intent anchor verification failed: {0}")]
    IntentAnchor(#[from] QuarantineIntentLedgerError),
    #[error("quarantine-state anchor verification failed: {0}")]
    QuarantineAnchor(#[from] QuarantineLedgerError),
    #[error("durable episodic material validation failed: {0}")]
    Material(#[from] RestartMaterializationError),
    #[error("episode envelope validation failed: {0}")]
    Envelope(#[from] EpisodicPersistenceEnvelopeError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};

    use crate::episodic_persistence_envelope::PersistedEpisodeEnvelope;
    use crate::memory_identity::episode_content_id;
    use crate::memory_quarantine::{
        EPISODIC_QUARANTINE_ESCROW_SCHEMA, EpisodicQuarantineEscrow,
        digest_episodic_quarantine_escrow, episodic_instance_target_id,
    };

    const STORE: &str = "symthaea:self:episodic-memory";

    fn canonical_episode(seed: f32) -> Episode {
        let source = Episode::new(
            ContinuousHV::from_values(vec![seed, seed + 1.0]),
            ContinuousHV::from_values(vec![seed + 2.0, seed + 3.0]),
            0.9,
            seed as u64 + 10,
        );
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        memory.store_if_significant_with_id(source).unwrap();
        let mut episodes = memory.get_top_episode_instances(1);
        episodes.remove(0).1
    }

    #[test]
    fn only_anchored_validated_material_mints_restart_permit() {
        let episode = canonical_episode(1.0);
        let id = episode.instance_id.unwrap();
        let record = DurableEpisodeRecord::new(
            PersistedEpisodeEnvelope::capture(&episode, 100).unwrap(),
            "sqlite:episode:1",
        )
        .unwrap();
        let intent = EpisodicQuarantineIntentLedger::new();
        let quarantine = EpisodicQuarantineStateLedger::new();

        let permit = verify_episodic_restart(
            STORE,
            &[record],
            &[],
            intent.events(),
            intent.head_hash(),
            quarantine.events(),
            quarantine.head_hash(),
        )
        .unwrap();
        assert_eq!(permit.active_occurrences().len(), 1);
        assert_eq!(permit.active_occurrences()[0].instance_id, id);
        assert!(permit.inactive_occurrences().is_empty());
        assert_ne!(permit.receipt().activation_plan_digest.0, [0; 32]);
        assert_ne!(permit.receipt().durable_material_digest.0, [0; 32]);
    }

    #[test]
    fn trusted_later_head_rejects_validly_hashed_suffix_rollback() {
        let episode = canonical_episode(2.0);
        let id = episode.instance_id.unwrap();
        let content_id = episode_content_id(&episode).unwrap();
        let target = episodic_instance_target_id(STORE, id).unwrap();
        let escrow = EpisodicQuarantineEscrow {
            schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: target.clone(),
            instance_id: id,
            content_id,
            captured_at_unix_s: 100,
            pre_active_state_digest: Sha256Digest([1; 32]),
            episode,
        };
        let escrow_digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
        let durable_escrow = DurableEscrowRecord::new(
            escrow,
            escrow_digest,
            "escrow:anchored:1",
        )
        .unwrap();

        let mut intent = EpisodicQuarantineIntentLedger::new();
        intent
            .append_prepared(
                "exec:q:1",
                &target,
                id,
                content_id,
                100,
                Sha256Digest([2; 32]),
                escrow_digest,
                "escrow:anchored:1",
            )
            .unwrap();
        let first_event = intent.events()[0].clone();
        intent
            .append_committed(
                "exec:q:1",
                &target,
                id,
                content_id,
                101,
                Sha256Digest([3; 32]),
            )
            .unwrap();
        let trusted_later_head = intent.head_hash();
        let quarantine = EpisodicQuarantineStateLedger::new();

        assert!(matches!(
            verify_episodic_restart(
                STORE,
                &[],
                &[durable_escrow],
                &[first_event],
                trusted_later_head,
                quarantine.events(),
                quarantine.head_hash(),
            ),
            Err(VerifiedRestartError::IntentAnchor(
                QuarantineIntentLedgerError::HeadAnchorMismatch { .. }
            ))
        ));
    }

    #[test]
    fn anchored_pending_quarantine_produces_inactive_capability_state() {
        let episode = canonical_episode(3.0);
        let id = episode.instance_id.unwrap();
        let content_id = episode_content_id(&episode).unwrap();
        let target = episodic_instance_target_id(STORE, id).unwrap();
        let escrow = EpisodicQuarantineEscrow {
            schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: target.clone(),
            instance_id: id,
            content_id,
            captured_at_unix_s: 100,
            pre_active_state_digest: Sha256Digest([4; 32]),
            episode,
        };
        let escrow_digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
        let durable_escrow = DurableEscrowRecord::new(
            escrow,
            escrow_digest,
            "escrow:pending:1",
        )
        .unwrap();
        let mut intent = EpisodicQuarantineIntentLedger::new();
        intent
            .append_prepared(
                "exec:q:pending",
                &target,
                id,
                content_id,
                100,
                Sha256Digest([5; 32]),
                escrow_digest,
                "escrow:pending:1",
            )
            .unwrap();
        let quarantine = EpisodicQuarantineStateLedger::new();

        let permit = verify_episodic_restart(
            STORE,
            &[],
            &[durable_escrow],
            intent.events(),
            intent.head_hash(),
            quarantine.events(),
            quarantine.head_hash(),
        )
        .unwrap();
        assert!(permit.active_occurrences().is_empty());
        assert_eq!(permit.inactive_occurrences().len(), 1);
        assert_eq!(permit.inactive_occurrences()[0].instance_id, id);
        assert!(permit.reconciliation_required());
    }
}
