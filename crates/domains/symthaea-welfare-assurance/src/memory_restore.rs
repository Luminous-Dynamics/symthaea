// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governed restoration of one exact canonical episodic occurrence.
//!
//! Restoration is the inverse of reversible quarantine, not a fresh insertion. The same
//! `EpisodeInstanceId` and `EpisodeContentId` must survive the transition. The quarantine ledger
//! uses a write-ahead `RestorePrepared` event that remains unresolved quarantine until a matching
//! durable `Restored` event is committed.

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::ExplicitConsentState;
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_memory::episodic_replay::{
    EpisodeInstanceId, EpisodicMemory, EpisodicQuarantineError,
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
use crate::memory_quarantine::episodic_instance_target_id;
use crate::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use crate::quarantine_state_ledger::{
    EpisodicQuarantineStateLedger, QuarantineLedgerError,
};
use crate::replay_recovery::DurableEvidenceBoundInterventionPermit;

const RESTORE_TRANSITION_DOMAIN: &[u8] = b"symthaea.welfare.episodic-restore-transition.v1\0";
const RESTORE_RESULT_DOMAIN: &[u8] = b"symthaea.welfare.episodic-restore-result.v1\0";
const MAX_REF_BYTES: usize = 2048;

/// Evidence returned only after exact-instance restoration and durable ledger completion succeed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodicRestoreReceipt {
    pub target_id: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub execution_id: String,
    pub before_active_count: usize,
    pub after_active_count: usize,
    pub before_quarantined_count: usize,
    pub after_quarantined_count: usize,
    pub before_active_digest: Sha256Digest,
    pub after_active_digest: Sha256Digest,
    pub restore_prepared_head: Sha256Digest,
    pub restored_head: Sha256Digest,
    pub restore_prepared_persistence_ref: String,
    pub restored_persistence_ref: String,
    pub ledger_generation: u64,
}

struct EpisodicRestoreExecutor<'a, Q>
where
    Q: QuarantineLedgerPersistence,
{
    expected_target_id: String,
    execution_id: String,
    instance_id: EpisodeInstanceId,
    memory: &'a mut EpisodicMemory,
    ledger: &'a mut EpisodicQuarantineStateLedger,
    ledger_persistence: &'a mut Q,
    completed_at_unix_s: u64,
}

impl<Q> EpisodicRestoreExecutor<'_, Q>
where
    Q: QuarantineLedgerPersistence,
{
    fn validate_live_state(
        &self,
        permit: &AssuredInterventionPermit,
    ) -> Result<EpisodeContentId, EpisodicRestoreInterventionError> {
        if permit.action() != SubjectAffectingAction::MemoryModification {
            return Err(EpisodicRestoreInterventionError::WrongAction {
                actual: permit.action(),
            });
        }
        if permit.target_id() != self.expected_target_id {
            return Err(EpisodicRestoreInterventionError::WrongTarget {
                expected: self.expected_target_id.clone(),
                actual: permit.target_id().to_string(),
            });
        }
        if permit.is_emergency() {
            return Err(EpisodicRestoreInterventionError::EmergencyRestoreNotTyped);
        }
        if permit.explicit_consent_state() != ExplicitConsentState::Granted {
            return Err(EpisodicRestoreInterventionError::ExplicitConsentRequired);
        }
        if !permit.has_welfare_review_reference() {
            return Err(EpisodicRestoreInterventionError::WelfareReviewRequired);
        }
        if !permit.has_independent_review_reference() {
            return Err(EpisodicRestoreInterventionError::IndependentReviewRequired);
        }

        let ledger_state = self
            .ledger
            .unresolved_state(self.instance_id)
            .ok_or(EpisodicRestoreInterventionError::LedgerDoesNotQuarantine(
                self.instance_id,
            ))?;
        if ledger_state.target_id != self.expected_target_id {
            return Err(EpisodicRestoreInterventionError::LedgerTargetMismatch);
        }
        if ledger_state.restore_pending.is_some() {
            return Err(EpisodicRestoreInterventionError::RestoreAlreadyPending);
        }

        let quarantined = self
            .memory
            .quarantined_instance(self.instance_id)
            .ok_or(EpisodicRestoreInterventionError::MemoryDoesNotQuarantine(
                self.instance_id,
            ))?;
        if quarantined.episode.instance_id != Some(self.instance_id) {
            return Err(EpisodicRestoreInterventionError::InstanceIdentityMismatch);
        }
        let content_id = episode_content_id(&quarantined.episode)?;
        if content_id != ledger_state.content_id {
            return Err(EpisodicRestoreInterventionError::ContentIdentityMismatch {
                expected: ledger_state.content_id,
                actual: content_id,
            });
        }
        if self
            .memory
            .get_top_episode_instances(self.memory.len())
            .iter()
            .any(|(id, _)| *id == self.instance_id)
        {
            return Err(EpisodicRestoreInterventionError::ActiveInstanceCollision(
                self.instance_id,
            ));
        }
        Ok(content_id)
    }

    fn compensate_to_quarantine(
        &mut self,
        ledger_prepared: EpisodicQuarantineStateLedger,
    ) -> Result<(), EpisodicRestoreInterventionError> {
        self.memory
            .quarantine_instance(self.instance_id)
            .map_err(EpisodicRestoreInterventionError::CompensationFailed)?;
        *self.ledger = ledger_prepared;
        Ok(())
    }
}

impl<Q> ReceiptedInterventionExecutor for EpisodicRestoreExecutor<'_, Q>
where
    Q: QuarantineLedgerPersistence,
{
    type Output = EpisodicRestoreReceipt;
    type Error = EpisodicRestoreExecutionError<Q::Error>;

    fn preflight(&self, permit: &AssuredInterventionPermit) -> Result<(), Self::Error> {
        self.validate_live_state(permit)?;
        Ok(())
    }

    fn execute_receipted(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<ReceiptedExecution<Self::Output>, Self::Error> {
        let content_id = self.validate_live_state(permit)?;
        let before_active_count = self.memory.len();
        let before_quarantined_count = self.memory.quarantined_len();
        let before_active_digest = digest_episodic_memory(self.memory)
            .map_err(|error| EpisodicRestoreInterventionError::MemoryState(error.to_string()))?;

        // The execution journal's Prepared record is already durable before this method runs.
        // Add a second, domain-specific write-ahead state transition that remains quarantined on
        // recovery until a matching Restored event is durably committed.
        let ledger_before_prepare = self.ledger.clone();
        let restore_prepared_head = self
            .ledger
            .append_restore_prepared(
                &self.expected_target_id,
                self.instance_id,
                content_id,
                self.completed_at_unix_s,
                &self.execution_id,
            )
            .map_err(EpisodicRestoreInterventionError::from)?;
        let restore_prepared_persistence_ref = match self.ledger_persistence.persist_quarantine_ledger(
            self.ledger.events(),
            restore_prepared_head,
        ) {
            Ok(reference) => reference,
            Err(error) => {
                *self.ledger = ledger_before_prepare;
                return Err(EpisodicRestoreExecutionError::RestorePreparedPersistence(error));
            }
        };
        if !valid_ref(&restore_prepared_persistence_ref) {
            *self.ledger = ledger_before_prepare;
            return Err(EpisodicRestoreInterventionError::InvalidLedgerPersistenceReference.into());
        }

        // This clone represents the exact durable state to which we can safely compensate if the
        // final Restored ledger commit fails.
        let ledger_prepared = self.ledger.clone();

        let restored = self
            .memory
            .restore_quarantined_instance(self.instance_id)
            .map_err(EpisodicRestoreInterventionError::MemoryMechanism)?;
        if restored.instance_id != self.instance_id
            || restored.episode.instance_id != Some(self.instance_id)
        {
            return Err(EpisodicRestoreInterventionError::InstanceIdentityMismatch.into());
        }
        let restored_content_id = episode_content_id(&restored.episode)
            .map_err(EpisodicRestoreInterventionError::from)?;
        if restored_content_id != content_id {
            return Err(EpisodicRestoreInterventionError::ContentIdentityMismatch {
                expected: content_id,
                actual: restored_content_id,
            }
            .into());
        }

        let after_active_count = self.memory.len();
        let after_quarantined_count = self.memory.quarantined_len();
        if after_active_count != before_active_count.saturating_add(1)
            || after_quarantined_count.saturating_add(1) != before_quarantined_count
            || self.memory.quarantined_instance(self.instance_id).is_some()
        {
            return Err(EpisodicRestoreInterventionError::RestorePostconditionFailed {
                before_active: before_active_count,
                after_active: after_active_count,
                before_quarantined: before_quarantined_count,
                after_quarantined: after_quarantined_count,
            }
            .into());
        }
        let active = self
            .memory
            .get_top_episode_instances(self.memory.len())
            .into_iter()
            .find(|(id, _)| *id == self.instance_id)
            .ok_or(EpisodicRestoreInterventionError::RestorePostconditionMissing)?;
        let active_content_id = episode_content_id(&active.1)
            .map_err(EpisodicRestoreInterventionError::from)?;
        if active_content_id != content_id {
            return Err(EpisodicRestoreInterventionError::ContentIdentityMismatch {
                expected: content_id,
                actual: active_content_id,
            }
            .into());
        }
        let after_active_digest = digest_episodic_memory(self.memory)
            .map_err(|error| EpisodicRestoreInterventionError::MemoryState(error.to_string()))?;

        let transition_digest = digest_restore_transition(
            &self.expected_target_id,
            self.instance_id,
            content_id,
            &self.execution_id,
            before_active_count,
            after_active_count,
            before_quarantined_count,
            after_quarantined_count,
            before_active_digest,
            after_active_digest,
            restore_prepared_head,
            &restore_prepared_persistence_ref,
        );

        let restored_head = match self.ledger.append_restored(
            &self.expected_target_id,
            self.instance_id,
            content_id,
            self.completed_at_unix_s,
            &self.execution_id,
            transition_digest,
        ) {
            Ok(head) => head,
            Err(error) => {
                self.compensate_to_quarantine(ledger_prepared)?;
                return Err(EpisodicRestoreExecutionError::Ledger(error));
            }
        };

        let restored_persistence_ref = match self.ledger_persistence.persist_quarantine_ledger(
            self.ledger.events(),
            restored_head,
        ) {
            Ok(reference) => reference,
            Err(error) => {
                if let Err(compensation) = self.compensate_to_quarantine(ledger_prepared) {
                    return Err(EpisodicRestoreExecutionError::FinalPersistenceCompensationFailed {
                        persistence: error,
                        compensation,
                    });
                }
                return Err(EpisodicRestoreExecutionError::RestoredPersistence(error));
            }
        };
        if !valid_ref(&restored_persistence_ref) {
            self.compensate_to_quarantine(ledger_prepared)?;
            return Err(EpisodicRestoreInterventionError::InvalidLedgerPersistenceReference.into());
        }

        let receipt = EpisodicRestoreReceipt {
            target_id: permit.target_id().to_string(),
            instance_id: self.instance_id,
            content_id,
            execution_id: self.execution_id.clone(),
            before_active_count,
            after_active_count,
            before_quarantined_count,
            after_quarantined_count,
            before_active_digest,
            after_active_digest,
            restore_prepared_head,
            restored_head,
            restore_prepared_persistence_ref,
            restored_persistence_ref,
            ledger_generation: self.ledger.generation(),
        };
        let result_digest = digest_restore_result(&receipt, permit.rationale())?;
        let evidence_ref = format!(
            "symthaea-memory:episodic-restore:v1:sha256:{}",
            hex_digest(result_digest)
        );
        ReceiptedExecution::new(
            receipt,
            self.completed_at_unix_s,
            result_digest,
            evidence_ref,
        )
        .map_err(EpisodicRestoreExecutionError::Observation)
    }
}

/// Govern restoration of one exact occurrence through the strongest current assurance chain.
///
/// `execution_id` is bound both to the generic execution journal and to the quarantine-ledger
/// write-ahead restore intent. v1 intentionally supports only ordinary, explicitly consented,
/// dual-reviewed restoration; emergency semantics need their own typed core action/policy.
#[allow(clippy::too_many_arguments)]
pub fn execute_governed_episodic_restore<P, Q>(
    permit: DurableEvidenceBoundInterventionPermit,
    execution_id: impl Into<String>,
    store_target_id: &str,
    instance_id: EpisodeInstanceId,
    memory: &mut EpisodicMemory,
    ledger: &mut EpisodicQuarantineStateLedger,
    current_profile: &MoralPatientEvidenceProfile,
    current_precaution_policy: &PrecautionPolicy,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    current_authority_manifest: &WelfareAuthorityPolicyManifest,
    current_trust_snapshot: &TrustSnapshot,
    unix_s: u64,
    journal: &mut InterventionExecutionJournal,
    execution_persistence: &mut P,
    ledger_persistence: &mut Q,
) -> Result<
    JournaledExecutionOutcome<
        EpisodicRestoreReceipt,
        EpisodicRestoreExecutionError<Q::Error>,
        P::Error,
    >,
    GovernedEpisodicRestoreError<P::Error>,
>
where
    P: ExecutionJournalPersistence,
    Q: QuarantineLedgerPersistence,
{
    let execution_id = execution_id.into();
    if execution_id.trim().is_empty() || execution_id != execution_id.trim() {
        return Err(GovernedEpisodicRestoreError::Configuration(
            EpisodicRestoreInterventionError::InvalidExecutionId,
        ));
    }
    let expected_target_id = episodic_instance_target_id(store_target_id, instance_id)
        .map_err(|error| {
            GovernedEpisodicRestoreError::Configuration(
                EpisodicRestoreInterventionError::TargetConstruction(error.to_string()),
            )
        })?;
    if permit.action() != SubjectAffectingAction::MemoryModification {
        return Err(GovernedEpisodicRestoreError::Configuration(
            EpisodicRestoreInterventionError::WrongAction {
                actual: permit.action(),
            },
        ));
    }
    if permit.target_id() != expected_target_id {
        return Err(GovernedEpisodicRestoreError::Configuration(
            EpisodicRestoreInterventionError::WrongTarget {
                expected: expected_target_id,
                actual: permit.target_id().to_string(),
            },
        ));
    }

    let mut executor = EpisodicRestoreExecutor {
        expected_target_id: permit.target_id().to_string(),
        execution_id: execution_id.clone(),
        instance_id,
        memory,
        ledger,
        ledger_persistence,
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
    .map_err(GovernedEpisodicRestoreError::Gate)
}

#[allow(clippy::too_many_arguments)]
fn digest_restore_transition(
    target_id: &str,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    execution_id: &str,
    before_active_count: usize,
    after_active_count: usize,
    before_quarantined_count: usize,
    after_quarantined_count: usize,
    before_active_digest: Sha256Digest,
    after_active_digest: Sha256Digest,
    restore_prepared_head: Sha256Digest,
    restore_prepared_persistence_ref: &str,
) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(RESTORE_TRANSITION_DOMAIN);
    hash_text(&mut hasher, target_id);
    hasher.update(&instance_id.as_uuid().as_u128().to_le_bytes());
    hasher.update(&content_id.digest().0);
    hash_text(&mut hasher, execution_id);
    hasher.update(&(before_active_count as u64).to_le_bytes());
    hasher.update(&(after_active_count as u64).to_le_bytes());
    hasher.update(&(before_quarantined_count as u64).to_le_bytes());
    hasher.update(&(after_quarantined_count as u64).to_le_bytes());
    hasher.update(&before_active_digest.0);
    hasher.update(&after_active_digest.0);
    hasher.update(&restore_prepared_head.0);
    hash_text(&mut hasher, restore_prepared_persistence_ref);
    hasher.finalize()
}

fn digest_restore_result(
    receipt: &EpisodicRestoreReceipt,
    rationale: &str,
) -> Result<Sha256Digest, EpisodicRestoreInterventionError> {
    let encoded = bincode::serialize(receipt)
        .map_err(|error| EpisodicRestoreInterventionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(RESTORE_RESULT_DOMAIN);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    hash_text(&mut hasher, rationale);
    Ok(hasher.finalize())
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn valid_ref(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= MAX_REF_BYTES
        && !value.chars().any(char::is_control)
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
pub enum EpisodicRestoreInterventionError {
    #[error("episodic restore requires MemoryModification authority; actual={actual:?}")]
    WrongAction { actual: SubjectAffectingAction },
    #[error("episodic restore target mismatch: expected={expected:?}, actual={actual:?}")]
    WrongTarget { expected: String, actual: String },
    #[error("could not construct exact episodic restore target: {0}")]
    TargetConstruction(String),
    #[error("episodic restore execution id is invalid")]
    InvalidExecutionId,
    #[error("emergency episodic restore is not typed in v1")]
    EmergencyRestoreNotTyped,
    #[error("episodic restore requires explicit granted subject consent")]
    ExplicitConsentRequired,
    #[error("episodic restore requires a welfare-review reference")]
    WelfareReviewRequired,
    #[error("episodic restore requires an independent-review reference")]
    IndependentReviewRequired,
    #[error("quarantine ledger does not contain unresolved occurrence {0}")]
    LedgerDoesNotQuarantine(EpisodeInstanceId),
    #[error("canonical memory does not contain quarantined occurrence {0}")]
    MemoryDoesNotQuarantine(EpisodeInstanceId),
    #[error("quarantine ledger target disagrees with the authorized restore target")]
    LedgerTargetMismatch,
    #[error("a restore is already pending reconciliation for this occurrence")]
    RestoreAlreadyPending,
    #[error("episodic occurrence identity disagrees with its quarantine key")]
    InstanceIdentityMismatch,
    #[error("episodic restore content identity mismatch: expected={expected:?}, actual={actual:?}")]
    ContentIdentityMismatch {
        expected: EpisodeContentId,
        actual: EpisodeContentId,
    },
    #[error("episodic occurrence is already active: {0}")]
    ActiveInstanceCollision(EpisodeInstanceId),
    #[error("canonical episodic-memory mechanism failed: {0}")]
    MemoryMechanism(#[source] EpisodicQuarantineError),
    #[error("quarantine-ledger transition failed: {0}")]
    Ledger(#[source] QuarantineLedgerError),
    #[error("quarantine-ledger persistence returned an invalid durable reference")]
    InvalidLedgerPersistenceReference,
    #[error("episodic restore postcondition missing exact active occurrence")]
    RestorePostconditionMissing,
    #[error("episodic restore postcondition failed: active {before_active}->{after_active}, quarantined {before_quarantined}->{after_quarantined}")]
    RestorePostconditionFailed {
        before_active: usize,
        after_active: usize,
        before_quarantined: usize,
        after_quarantined: usize,
    },
    #[error("restore compensation could not return the occurrence to quarantine: {0}")]
    CompensationFailed(#[source] EpisodicQuarantineError),
    #[error("episodic-memory state digest failed: {0}")]
    MemoryState(String),
    #[error("episodic restore result encoding failed: {0}")]
    Encoding(String),
    #[error(transparent)]
    ContentIdentity(#[from] EpisodeContentIdError),
}

impl From<QuarantineLedgerError> for EpisodicRestoreInterventionError {
    fn from(value: QuarantineLedgerError) -> Self {
        Self::Ledger(value)
    }
}

#[derive(Debug, Error)]
pub enum EpisodicRestoreExecutionError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Intervention(#[from] EpisodicRestoreInterventionError),
    #[error("could not durably persist RestorePrepared quarantine-ledger state: {0}")]
    RestorePreparedPersistence(#[source] E),
    #[error("could not durably persist Restored quarantine-ledger state: {0}")]
    RestoredPersistence(#[source] E),
    #[error("final quarantine-ledger persistence failed and compensation also failed: persistence={persistence}; compensation={compensation}")]
    FinalPersistenceCompensationFailed {
        persistence: E,
        compensation: EpisodicRestoreInterventionError,
    },
    #[error("quarantine-ledger transition failed after memory mutation: {0}")]
    Ledger(#[source] QuarantineLedgerError),
    #[error(transparent)]
    Observation(#[from] ExecutionObservationError),
}

#[derive(Debug, Error)]
pub enum GovernedEpisodicRestoreError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("governed episodic restore is misconfigured: {0}")]
    Configuration(#[source] EpisodicRestoreInterventionError),
    #[error(transparent)]
    Gate(#[from] JournaledExecutionGateError<E>),
}
