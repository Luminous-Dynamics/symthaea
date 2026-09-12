// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governed exact-instance quarantine for canonical episodic memory.
//!
//! Quarantine is deliberately modeled as reversible isolation, not erasure. The low-level memory
//! mechanism remains owned by `symthaea-memory`; this adapter requires the strongest current
//! welfare-assurance capability, exact occurrence identity, explicit ordinary consent/review, a
//! durable pre-mutation escrow record, and the two-phase execution journal.

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::ExplicitConsentState;
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_memory::episodic_replay::{
    Episode, EpisodeInstanceId, EpisodicMemory, EpisodicQuarantineError,
};
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_authority::WelfareAuthorityPolicyManifest;
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};
use thiserror::Error;

use crate::AssuredInterventionPermit;
use crate::execution_adapter::{
    ExecutionJournalPersistence, ExecutionObservationError, JournaledExecutionGateError,
    JournaledExecutionOutcome, ReceiptedExecution, ReceiptedInterventionExecutor,
    execute_durable_intervention_journaled,
};
use crate::execution_recovery::InterventionExecutionJournal;
use crate::memory_identity::{EpisodeContentId, EpisodeContentIdError, episode_content_id};
use crate::memory_intervention::digest_episodic_memory;
use crate::replay_recovery::DurableEvidenceBoundInterventionPermit;

pub const EPISODIC_QUARANTINE_ESCROW_SCHEMA: &str =
    "symthaea.welfare.episodic-quarantine-escrow.v1";
const ESCROW_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.episodic-quarantine-escrow-digest.v1\0";
const RESULT_DIGEST_DOMAIN: &[u8] = b"symthaea.welfare.episodic-quarantine-result.v1\0";
const MAX_TARGET_ID_BYTES: usize = 256;
const MAX_PERSISTENCE_REF_BYTES: usize = 2048;

/// Build the exact intervention target for one stored episodic occurrence.
pub fn episodic_instance_target_id(
    store_target_id: &str,
    instance_id: EpisodeInstanceId,
) -> Result<String, EpisodicQuarantineInterventionError> {
    validate_text("store_target_id", store_target_id, MAX_TARGET_ID_BYTES)?;
    let target = format!("{store_target_id}:instance:{instance_id}");
    validate_text("instance_target_id", &target, MAX_TARGET_ID_BYTES)?;
    Ok(target)
}

/// Durable reversible escrow captured before an active occurrence is quarantined.
///
/// This record preserves the exact episode occurrence and its pre-intervention active-store
/// commitment. Persistence is mandatory before the canonical active store is mutated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicQuarantineEscrow {
    pub schema_version: String,
    pub target_id: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub captured_at_unix_s: u64,
    pub pre_active_state_digest: Sha256Digest,
    pub episode: Episode,
}

impl EpisodicQuarantineEscrow {
    fn capture(
        target_id: &str,
        instance_id: EpisodeInstanceId,
        memory: &EpisodicMemory,
        captured_at_unix_s: u64,
    ) -> Result<Self, EpisodicQuarantineInterventionError> {
        let episode = active_episode(memory, instance_id)?;
        if episode.instance_id != Some(instance_id) {
            return Err(EpisodicQuarantineInterventionError::InstanceIdentityMismatch);
        }
        let content_id = episode_content_id(&episode)?;
        let pre_active_state_digest = digest_episodic_memory(memory)
            .map_err(|error| EpisodicQuarantineInterventionError::MemoryState(error.to_string()))?;
        Ok(Self {
            schema_version: EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: target_id.to_string(),
            instance_id,
            content_id,
            captured_at_unix_s,
            pre_active_state_digest,
            episode,
        })
    }

    pub fn validate(&self) -> Result<(), EpisodicQuarantineInterventionError> {
        if self.schema_version != EPISODIC_QUARANTINE_ESCROW_SCHEMA {
            return Err(EpisodicQuarantineInterventionError::UnsupportedEscrowSchema);
        }
        validate_text("target_id", &self.target_id, MAX_TARGET_ID_BYTES)?;
        if self.episode.instance_id != Some(self.instance_id) {
            return Err(EpisodicQuarantineInterventionError::InstanceIdentityMismatch);
        }
        if episode_content_id(&self.episode)? != self.content_id {
            return Err(EpisodicQuarantineInterventionError::ContentIdentityMismatch);
        }
        if self.pre_active_state_digest.0 == [0; 32] {
            return Err(EpisodicQuarantineInterventionError::ZeroStateDigest);
        }
        Ok(())
    }
}

/// Persistence boundary for reversible quarantine escrow.
pub trait EpisodicQuarantineEscrowPersistence {
    type Error: StdError + Send + Sync + 'static;

    fn persist_episodic_quarantine_escrow(
        &mut self,
        escrow: &EpisodicQuarantineEscrow,
        escrow_digest: Sha256Digest,
    ) -> Result<String, Self::Error>;
}

/// Canonical digest of one escrow record.
pub fn digest_episodic_quarantine_escrow(
    escrow: &EpisodicQuarantineEscrow,
) -> Result<Sha256Digest, EpisodicQuarantineInterventionError> {
    escrow.validate()?;
    let encoded = serde_json::to_vec(escrow)
        .map_err(|error| EpisodicQuarantineInterventionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(ESCROW_DIGEST_DOMAIN);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

/// Evidence returned only after exact-instance quarantine succeeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodicQuarantineReceipt {
    pub target_id: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub before_active_count: usize,
    pub after_active_count: usize,
    pub before_active_digest: Sha256Digest,
    pub after_active_digest: Sha256Digest,
    pub escrow_digest: Sha256Digest,
    pub escrow_persistence_ref: String,
}

struct EpisodicQuarantineExecutor<'a, Q>
where
    Q: EpisodicQuarantineEscrowPersistence,
{
    expected_target_id: String,
    instance_id: EpisodeInstanceId,
    memory: &'a mut EpisodicMemory,
    escrow_persistence: &'a mut Q,
    completed_at_unix_s: u64,
}

impl<Q> ReceiptedInterventionExecutor for EpisodicQuarantineExecutor<'_, Q>
where
    Q: EpisodicQuarantineEscrowPersistence,
{
    type Output = EpisodicQuarantineReceipt;
    type Error = EpisodicQuarantineExecutionError<Q::Error>;

    fn preflight(&self, permit: &AssuredInterventionPermit) -> Result<(), Self::Error> {
        if permit.action() != SubjectAffectingAction::MemoryModification {
            return Err(EpisodicQuarantineInterventionError::WrongAction {
                actual: permit.action(),
            }
            .into());
        }
        if permit.target_id() != self.expected_target_id {
            return Err(EpisodicQuarantineInterventionError::WrongTarget {
                expected: self.expected_target_id.clone(),
                actual: permit.target_id().to_string(),
            }
            .into());
        }

        // v1 intentionally supports only ordinary consensual quarantine. Emergency memory
        // quarantine needs its own typed core action/policy rather than overloading the coarse
        // `MemoryModification` class and silently creating a consent-override path.
        if permit.is_emergency() {
            return Err(EpisodicQuarantineInterventionError::EmergencyQuarantineNotTyped.into());
        }
        if permit.explicit_consent_state() != ExplicitConsentState::Granted {
            return Err(EpisodicQuarantineInterventionError::ExplicitConsentRequired.into());
        }
        if !permit.has_welfare_review_reference() {
            return Err(EpisodicQuarantineInterventionError::WelfareReviewRequired.into());
        }
        if !permit.has_independent_review_reference() {
            return Err(EpisodicQuarantineInterventionError::IndependentReviewRequired.into());
        }
        active_episode(self.memory, self.instance_id)?;
        Ok(())
    }

    fn execute_receipted(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<ReceiptedExecution<Self::Output>, Self::Error> {
        let before_active_count = self.memory.len();
        let escrow = EpisodicQuarantineEscrow::capture(
            &self.expected_target_id,
            self.instance_id,
            self.memory,
            self.completed_at_unix_s,
        )?;
        let escrow_digest = digest_episodic_quarantine_escrow(&escrow)?;
        let escrow_persistence_ref = self
            .escrow_persistence
            .persist_episodic_quarantine_escrow(&escrow, escrow_digest)
            .map_err(EpisodicQuarantineExecutionError::EscrowPersistence)?;
        validate_text(
            "escrow_persistence_ref",
            &escrow_persistence_ref,
            MAX_PERSISTENCE_REF_BYTES,
        )?;

        // Recheck immediately before mutation. Safe Rust's exclusive `&mut EpisodicMemory` already
        // excludes another in-process mutator, but the digest equality makes the precondition
        // explicit in evidence and detects accidental adapter changes.
        let immediate_pre_digest = digest_episodic_memory(self.memory)
            .map_err(|error| EpisodicQuarantineInterventionError::MemoryState(error.to_string()))?;
        if immediate_pre_digest != escrow.pre_active_state_digest {
            return Err(EpisodicQuarantineInterventionError::PreStateChanged.into());
        }

        let quarantined = self
            .memory
            .quarantine_instance(self.instance_id)
            .map_err(EpisodicQuarantineInterventionError::MemoryMechanism)?;
        if quarantined.episode.instance_id != Some(self.instance_id)
            || episode_content_id(&quarantined.episode)? != escrow.content_id
        {
            return Err(EpisodicQuarantineInterventionError::QuarantinedContentMismatch.into());
        }

        let after_active_count = self.memory.len();
        if after_active_count.checked_add(1) != Some(before_active_count) {
            return Err(EpisodicQuarantineInterventionError::ActiveCountPostcondition {
                before: before_active_count,
                after: after_active_count,
            }
            .into());
        }
        let after_active_digest = digest_episodic_memory(self.memory)
            .map_err(|error| EpisodicQuarantineInterventionError::MemoryState(error.to_string()))?;
        let durable_quarantine = self
            .memory
            .quarantined_instance(self.instance_id)
            .ok_or(EpisodicQuarantineInterventionError::QuarantinePostconditionMissing)?;
        if episode_content_id(&durable_quarantine.episode)? != escrow.content_id {
            return Err(EpisodicQuarantineInterventionError::QuarantinedContentMismatch.into());
        }

        let receipt = EpisodicQuarantineReceipt {
            target_id: permit.target_id().to_string(),
            instance_id: self.instance_id,
            content_id: escrow.content_id,
            before_active_count,
            after_active_count,
            before_active_digest: escrow.pre_active_state_digest,
            after_active_digest,
            escrow_digest,
            escrow_persistence_ref,
        };
        let result_digest = digest_result(&receipt, permit.rationale());
        let evidence_ref = format!(
            "symthaea-memory:episodic-quarantine:v1:sha256:{}",
            hex_digest(result_digest)
        );
        ReceiptedExecution::new(
            receipt,
            self.completed_at_unix_s,
            result_digest,
            evidence_ref,
        )
        .map_err(EpisodicQuarantineExecutionError::Observation)
    }
}

/// Govern one exact episodic occurrence through live revalidation, domain preflight, durable
/// write-ahead journaling, durable reversible escrow, canonical quarantine, and terminal evidence.
#[allow(clippy::too_many_arguments)]
pub fn execute_governed_episodic_quarantine<P, Q>(
    permit: DurableEvidenceBoundInterventionPermit,
    execution_id: impl Into<String>,
    store_target_id: &str,
    instance_id: EpisodeInstanceId,
    memory: &mut EpisodicMemory,
    current_profile: &MoralPatientEvidenceProfile,
    current_precaution_policy: &PrecautionPolicy,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    current_authority_manifest: &WelfareAuthorityPolicyManifest,
    current_trust_snapshot: &TrustSnapshot,
    unix_s: u64,
    journal: &mut InterventionExecutionJournal,
    execution_persistence: &mut P,
    escrow_persistence: &mut Q,
) -> Result<
    JournaledExecutionOutcome<
        EpisodicQuarantineReceipt,
        EpisodicQuarantineExecutionError<Q::Error>,
        P::Error,
    >,
    GovernedEpisodicQuarantineError<P::Error>,
>
where
    P: ExecutionJournalPersistence,
    Q: EpisodicQuarantineEscrowPersistence,
{
    let expected_target_id = episodic_instance_target_id(store_target_id, instance_id)
        .map_err(GovernedEpisodicQuarantineError::Configuration)?;
    if permit.action() != SubjectAffectingAction::MemoryModification {
        return Err(GovernedEpisodicQuarantineError::Configuration(
            EpisodicQuarantineInterventionError::WrongAction {
                actual: permit.action(),
            },
        ));
    }
    if permit.target_id() != expected_target_id {
        return Err(GovernedEpisodicQuarantineError::Configuration(
            EpisodicQuarantineInterventionError::WrongTarget {
                expected: expected_target_id,
                actual: permit.target_id().to_string(),
            },
        ));
    }

    let mut executor = EpisodicQuarantineExecutor {
        expected_target_id: permit.target_id().to_string(),
        instance_id,
        memory,
        escrow_persistence,
        completed_at_unix_s: unix_s,
    };
    execute_durable_intervention_journaled(
        permit,
        execution_id,
        current_profile,
        current_precaution_policy,
        consent_ledger,
        subject_registry,
        current_authority_manifest,
        current_trust_snapshot,
        unix_s,
        journal,
        execution_persistence,
        &mut executor,
    )
    .map_err(GovernedEpisodicQuarantineError::Gate)
}

fn active_episode(
    memory: &EpisodicMemory,
    instance_id: EpisodeInstanceId,
) -> Result<Episode, EpisodicQuarantineInterventionError> {
    memory
        .get_top_episode_instances(memory.len())
        .into_iter()
        .find_map(|(id, episode)| (id == instance_id).then_some(episode))
        .ok_or_else(|| {
            if memory.quarantined_instance(instance_id).is_some() {
                EpisodicQuarantineInterventionError::AlreadyQuarantined(instance_id)
            } else {
                EpisodicQuarantineInterventionError::InstanceNotActive(instance_id)
            }
        })
}

fn digest_result(receipt: &EpisodicQuarantineReceipt, rationale: &str) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(RESULT_DIGEST_DOMAIN);
    hasher.update(&(receipt.target_id.len() as u64).to_le_bytes());
    hasher.update(receipt.target_id.as_bytes());
    hasher.update(receipt.instance_id.as_uuid().as_bytes());
    hasher.update(&receipt.content_id.digest().0);
    hasher.update(&(receipt.before_active_count as u64).to_le_bytes());
    hasher.update(&(receipt.after_active_count as u64).to_le_bytes());
    hasher.update(&receipt.before_active_digest.0);
    hasher.update(&receipt.after_active_digest.0);
    hasher.update(&receipt.escrow_digest.0);
    hasher.update(&(receipt.escrow_persistence_ref.len() as u64).to_le_bytes());
    hasher.update(receipt.escrow_persistence_ref.as_bytes());
    hasher.update(&(rationale.len() as u64).to_le_bytes());
    hasher.update(rationale.as_bytes());
    hasher.finalize()
}

fn validate_text(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), EpisodicQuarantineInterventionError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
    {
        Err(EpisodicQuarantineInterventionError::InvalidText {
            field,
            value: value.to_string(),
        })
    } else {
        Ok(())
    }
}

fn hex_digest(digest: Sha256Digest) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest.0 {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[derive(Debug, Error)]
pub enum EpisodicQuarantineInterventionError {
    #[error("invalid {field}: {value:?}")]
    InvalidText { field: &'static str, value: String },
    #[error("episodic quarantine requires MemoryModification authority; actual={actual:?}")]
    WrongAction { actual: SubjectAffectingAction },
    #[error("episodic quarantine target mismatch: expected={expected:?}, actual={actual:?}")]
    WrongTarget { expected: String, actual: String },
    #[error("emergency episodic quarantine is not enabled until a typed core quarantine policy exists")]
    EmergencyQuarantineNotTyped,
    #[error("episodic quarantine requires explicit granted consent")]
    ExplicitConsentRequired,
    #[error("episodic quarantine requires a welfare-review reference")]
    WelfareReviewRequired,
    #[error("episodic quarantine requires an independent-review reference")]
    IndependentReviewRequired,
    #[error("episodic instance is not active: {0}")]
    InstanceNotActive(EpisodeInstanceId),
    #[error("episodic instance is already quarantined: {0}")]
    AlreadyQuarantined(EpisodeInstanceId),
    #[error("episode occurrence identity changed or is missing")]
    InstanceIdentityMismatch,
    #[error("episode content identity changed unexpectedly")]
    ContentIdentityMismatch,
    #[error("unsupported episodic-quarantine escrow schema")]
    UnsupportedEscrowSchema,
    #[error("episodic memory state digest may not be zero")]
    ZeroStateDigest,
    #[error("episodic memory state failed: {0}")]
    MemoryState(String),
    #[error("episodic quarantine pre-state changed after escrow capture")]
    PreStateChanged,
    #[error("canonical quarantine mechanism failed: {0}")]
    MemoryMechanism(#[source] EpisodicQuarantineError),
    #[error("quarantined content does not match the escrowed occurrence")]
    QuarantinedContentMismatch,
    #[error("quarantine postcondition missing from canonical memory store")]
    QuarantinePostconditionMissing,
    #[error("active-count quarantine postcondition failed: before={before}, after={after}")]
    ActiveCountPostcondition { before: usize, after: usize },
    #[error("quarantine evidence encoding failed: {0}")]
    Encoding(String),
    #[error(transparent)]
    ContentIdentity(#[from] EpisodeContentIdError),
}

#[derive(Debug, Error)]
pub enum EpisodicQuarantineExecutionError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Intervention(#[from] EpisodicQuarantineInterventionError),
    #[error("could not durably persist episodic-quarantine escrow: {0}")]
    EscrowPersistence(#[source] E),
    #[error(transparent)]
    Observation(#[from] ExecutionObservationError),
}

#[derive(Debug, Error)]
pub enum GovernedEpisodicQuarantineError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("governed episodic quarantine is misconfigured: {0}")]
    Configuration(#[source] EpisodicQuarantineInterventionError),
    #[error(transparent)]
    Gate(#[from] JournaledExecutionGateError<E>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_memory::episodic_replay::EpisodicReplayConfig;

    fn episode(seed: f32) -> Episode {
        Episode::new(
            ContinuousHV::from_values(vec![seed, seed + 1.0]),
            ContinuousHV::from_values(vec![seed + 2.0, seed + 3.0]),
            0.8,
            42,
        )
    }

    #[test]
    fn target_identity_is_exact_and_bounded() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let id = memory.store_if_significant_with_id(episode(1.0)).unwrap();
        let target = episodic_instance_target_id("symthaea:self:episodic-memory", id).unwrap();
        assert_eq!(target, format!("symthaea:self:episodic-memory:instance:{id}"));
    }

    #[test]
    fn escrow_binds_exact_occurrence_and_content() {
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let id = memory.store_if_significant_with_id(episode(2.0)).unwrap();
        let target = episodic_instance_target_id("symthaea:self:episodic-memory", id).unwrap();
        let escrow = EpisodicQuarantineEscrow::capture(&target, id, &memory, 100).unwrap();
        assert_eq!(escrow.instance_id, id);
        assert_eq!(escrow.episode.instance_id, Some(id));
        assert_eq!(escrow.content_id, episode_content_id(&escrow.episode).unwrap());
        assert_ne!(digest_episodic_quarantine_escrow(&escrow).unwrap().0, [0; 32]);
    }
}
