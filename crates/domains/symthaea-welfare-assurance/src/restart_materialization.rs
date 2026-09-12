// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation bridge from durable episodic material to the pure restart activation gate.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use thiserror::Error;

use crate::episodic_persistence_envelope::{
    EpisodicPersistenceEnvelopeError, PersistedEpisodeEnvelope,
};
use crate::memory_quarantine::{
    EpisodicQuarantineEscrow, EpisodicQuarantineInterventionError,
    digest_episodic_quarantine_escrow, episodic_instance_target_id,
};
use crate::quarantine_intent_ledger::EpisodicQuarantineIntentLedger;
use crate::quarantine_state_ledger::EpisodicQuarantineStateLedger;
use crate::restart_activation_gate::{
    EpisodicRestartActivationPlan, PersistedOccurrenceDescriptor,
    PersistedQuarantineEscrowDescriptor, RestartActivationError,
    build_episodic_restart_activation_plan,
};

const MAX_REF_BYTES: usize = 2048;

#[derive(Debug, Clone)]
pub struct DurableEpisodeRecord {
    pub envelope: PersistedEpisodeEnvelope,
    pub durable_record_ref: String,
}

impl DurableEpisodeRecord {
    pub fn new(
        envelope: PersistedEpisodeEnvelope,
        durable_record_ref: impl Into<String>,
    ) -> Result<Self, RestartMaterializationError> {
        let durable_record_ref = durable_record_ref.into();
        validate_ref("durable_record_ref", &durable_record_ref)?;
        envelope.validate()?;
        Ok(Self {
            envelope,
            durable_record_ref,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DurableEscrowRecord {
    pub escrow: EpisodicQuarantineEscrow,
    pub escrow_digest: Sha256Digest,
    pub durable_escrow_ref: String,
}

impl DurableEscrowRecord {
    pub fn new(
        escrow: EpisodicQuarantineEscrow,
        escrow_digest: Sha256Digest,
        durable_escrow_ref: impl Into<String>,
    ) -> Result<Self, RestartMaterializationError> {
        let durable_escrow_ref = durable_escrow_ref.into();
        validate_ref("durable_escrow_ref", &durable_escrow_ref)?;
        escrow.validate()?;
        let actual = digest_episodic_quarantine_escrow(&escrow)?;
        if actual != escrow_digest {
            return Err(RestartMaterializationError::EscrowDigestMismatch {
                expected: escrow_digest,
                actual,
            });
        }
        Ok(Self {
            escrow,
            escrow_digest,
            durable_escrow_ref,
        })
    }
}

/// Validate all durable episode/escrow material, then derive the fail-closed activation plan.
pub fn build_restart_plan_from_durable_material(
    store_target_id: &str,
    episode_records: &[DurableEpisodeRecord],
    escrow_records: &[DurableEscrowRecord],
    intent_ledger: &EpisodicQuarantineIntentLedger,
    quarantine_ledger: &EpisodicQuarantineStateLedger,
) -> Result<EpisodicRestartActivationPlan, RestartMaterializationError> {
    let mut record_refs = BTreeSet::new();
    let mut occurrence_descriptors = Vec::with_capacity(episode_records.len());
    for record in episode_records {
        validate_ref("durable_record_ref", &record.durable_record_ref)?;
        if !record_refs.insert(record.durable_record_ref.clone()) {
            return Err(RestartMaterializationError::DuplicateDurableRecordReference(
                record.durable_record_ref.clone(),
            ));
        }
        record.envelope.validate()?;
        occurrence_descriptors.push(
            record
                .envelope
                .restart_descriptor(record.durable_record_ref.clone())?,
        );
    }

    let mut escrow_refs = BTreeSet::new();
    let mut escrow_descriptors = Vec::with_capacity(escrow_records.len());
    for record in escrow_records {
        validate_ref("durable_escrow_ref", &record.durable_escrow_ref)?;
        if !escrow_refs.insert(record.durable_escrow_ref.clone()) {
            return Err(RestartMaterializationError::DuplicateDurableEscrowReference(
                record.durable_escrow_ref.clone(),
            ));
        }
        record.escrow.validate()?;
        let actual_digest = digest_episodic_quarantine_escrow(&record.escrow)?;
        if actual_digest != record.escrow_digest {
            return Err(RestartMaterializationError::EscrowDigestMismatch {
                expected: record.escrow_digest,
                actual: actual_digest,
            });
        }
        let expected_target = episodic_instance_target_id(
            store_target_id,
            record.escrow.instance_id,
        )?;
        if record.escrow.target_id != expected_target {
            return Err(RestartMaterializationError::EscrowTargetMismatch {
                expected: expected_target,
                actual: record.escrow.target_id.clone(),
            });
        }
        // The durable reference supplied to the restart gate must be the exact reference committed
        // by the quarantine lifecycle/intent ledgers. This prevents swapping equivalent escrow
        // bytes under a different locator after authorization.
        escrow_descriptors.push(PersistedQuarantineEscrowDescriptor::new(
            record.escrow.instance_id,
            record.escrow.content_id,
            record.escrow_digest,
            record.durable_escrow_ref.clone(),
        )?);
    }

    build_episodic_restart_activation_plan(
        store_target_id,
        &occurrence_descriptors,
        &escrow_descriptors,
        intent_ledger,
        quarantine_ledger,
    )
    .map_err(RestartMaterializationError::Activation)
}

fn validate_ref(field: &'static str, value: &str) -> Result<(), RestartMaterializationError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_REF_BYTES
        || value.chars().any(char::is_control)
    {
        Err(RestartMaterializationError::InvalidReference { field })
    } else {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum RestartMaterializationError {
    #[error("invalid durable restart reference `{field}`")]
    InvalidReference { field: &'static str },
    #[error("duplicate durable episode-record reference {0:?}")]
    DuplicateDurableRecordReference(String),
    #[error("duplicate durable escrow reference {0:?}")]
    DuplicateDurableEscrowReference(String),
    #[error("persisted escrow digest mismatch: expected={expected:?}, actual={actual:?}")]
    EscrowDigestMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("persisted escrow target mismatch: expected={expected:?}, actual={actual:?}")]
    EscrowTargetMismatch { expected: String, actual: String },
    #[error(transparent)]
    Envelope(#[from] EpisodicPersistenceEnvelopeError),
    #[error("persisted quarantine escrow is invalid: {0}")]
    Escrow(#[from] EpisodicQuarantineInterventionError),
    #[error("restart activation material is inconsistent: {0}")]
    Activation(#[source] RestartActivationError),
}

impl From<RestartActivationError> for RestartMaterializationError {
    fn from(value: RestartActivationError) -> Self {
        Self::Activation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};

    use crate::episodic_persistence_envelope::PersistedEpisodeEnvelope;
    use crate::memory_identity::episode_content_id;
    use crate::memory_quarantine::EPISODIC_QUARANTINE_ESCROW_SCHEMA;

    const STORE: &str = "symthaea:self:episodic-memory";

    fn episode_and_id() -> (Episode, symthaea_memory::episodic_replay::EpisodeInstanceId) {
        let episode = Episode::new(
            ContinuousHV::from_values(vec![1.0, 2.0]),
            ContinuousHV::from_values(vec![3.0, 4.0]),
            0.9,
            42,
        );
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let id = memory.store_if_significant_with_id(episode).unwrap();
        let stored = memory.get_top_episode_instances(1).remove(0).1;
        (stored, id)
    }

    #[test]
    fn validated_envelope_flows_to_active_restart_plan() {
        let (episode, id) = episode_and_id();
        let envelope = PersistedEpisodeEnvelope::capture(&episode, 100).unwrap();
        let record = DurableEpisodeRecord::new(envelope, "sqlite:episode:1").unwrap();
        let plan = build_restart_plan_from_durable_material(
            STORE,
            &[record],
            &[],
            &EpisodicQuarantineIntentLedger::new(),
            &EpisodicQuarantineStateLedger::new(),
        )
        .unwrap();
        assert_eq!(plan.active.len(), 1);
        assert_eq!(plan.active[0].instance_id, id);
    }

    #[test]
    fn escrow_target_mismatch_is_rejected_before_activation_gate() {
        let (episode, id) = episode_and_id();
        let content_id = episode_content_id(&episode).unwrap();
        let escrow = EpisodicQuarantineEscrow {
            schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: format!("wrong-store:instance:{id}"),
            instance_id: id,
            content_id,
            captured_at_unix_s: 100,
            pre_active_state_digest: Sha256Digest([1; 32]),
            episode,
        };
        let digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
        let record = DurableEscrowRecord::new(escrow, digest, "escrow:1").unwrap();
        assert!(matches!(
            build_restart_plan_from_durable_material(
                STORE,
                &[],
                &[record],
                &EpisodicQuarantineIntentLedger::new(),
                &EpisodicQuarantineStateLedger::new(),
            ),
            Err(RestartMaterializationError::EscrowTargetMismatch { .. })
        ));
    }

    #[test]
    fn tampered_episode_state_is_rejected_before_descriptor_creation() {
        let (episode, _) = episode_and_id();
        let mut envelope = PersistedEpisodeEnvelope::capture(&episode, 100).unwrap();
        envelope.episode.replay_count += 1;
        let record = DurableEpisodeRecord {
            envelope,
            durable_record_ref: "sqlite:episode:tampered".into(),
        };
        assert!(matches!(
            build_restart_plan_from_durable_material(
                STORE,
                &[record],
                &[],
                &EpisodicQuarantineIntentLedger::new(),
                &EpisodicQuarantineStateLedger::new(),
            ),
            Err(RestartMaterializationError::Envelope(
                EpisodicPersistenceEnvelopeError::StateDigestMismatch { .. }
            ))
        ));
    }

    #[test]
    fn duplicate_durable_reference_is_rejected_even_for_distinct_uuids() {
        let (first_episode, _) = episode_and_id();
        let second_source = Episode::new(
            ContinuousHV::from_values(vec![5.0, 6.0]),
            ContinuousHV::from_values(vec![7.0, 8.0]),
            0.91,
            43,
        );
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        memory.store_if_significant_with_id(second_source).unwrap();
        let second_episode = memory.get_top_episode_instances(1).remove(0).1;
        let first = DurableEpisodeRecord::new(
            PersistedEpisodeEnvelope::capture(&first_episode, 100).unwrap(),
            "sqlite:same-ref",
        )
        .unwrap();
        let second = DurableEpisodeRecord::new(
            PersistedEpisodeEnvelope::capture(&second_episode, 100).unwrap(),
            "sqlite:same-ref",
        )
        .unwrap();
        assert!(matches!(
            build_restart_plan_from_durable_material(
                STORE,
                &[first, second],
                &[],
                &EpisodicQuarantineIntentLedger::new(),
                &EpisodicQuarantineStateLedger::new(),
            ),
            Err(RestartMaterializationError::DuplicateDurableRecordReference(_))
        ));
    }
}
