// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governed restoration of one exact persisted episodic occurrence after restart.
//!
//! Restart intentionally reconstructs only the active canonical replay heap. Quarantined episode
//! payloads remain outside live memory and may re-enter only through this exact point-lookup path.
//! The lookup result is treated as untrusted transport until it is rebound to the recovered
//! quarantine ledger's exact UUID, content ID, escrow digest, persistence reference and target.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::ExplicitConsentState;
use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_memory::episodic_replay::{
    EpisodeInstanceId, EpisodicMemory,
    persisted_import::PersistedEpisodicImportError,
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
use crate::memory_quarantine::{
    EpisodicQuarantineEscrow, EpisodicQuarantineInterventionError,
    digest_episodic_quarantine_escrow, episodic_instance_target_id,
};
use crate::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use crate::quarantine_state_ledger::{
    EpisodicQuarantineStateLedger, QuarantineLedgerError, QuarantineLedgerState,
};
use crate::replay_recovery::DurableEvidenceBoundInterventionPermit;

const PERSISTED_RESTORE_TRANSITION_DOMAIN: &[u8] =
    b"symthaea.welfare.persisted-episodic-restore-transition.v1\0";
const PERSISTED_RESTORE_RESULT_DOMAIN: &[u8] =
    b"symthaea.welfare.persisted-episodic-restore-result.v1\0";
const MAX_REF_BYTES: usize = 2048;

/// Raw point-lookup result from a durable quarantine store.
///
/// The fields are intentionally public transport data, not trusted evidence. The governed restore
/// adapter independently recomputes and compares every binding against the recovered quarantine
/// ledger before any live-memory state is constructed.
#[derive(Debug, Clone)]
pub struct PersistedEpisodicEscrowRow {
    pub escrow: EpisodicQuarantineEscrow,
    pub stored_digest: Sha256Digest,
    pub persistence_ref: String,
}

/// Purpose-separated durable point-lookup boundary.
///
/// There is deliberately no bulk enumeration method here. Implementations resolve one exact
/// `EpisodeInstanceId` or return `None`.
pub trait EpisodicQuarantineEscrowLookup {
    type Error: StdError + Send + Sync + 'static;

    fn load_episodic_quarantine_escrow(
        &self,
        instance_id: EpisodeInstanceId,
    ) -> Result<Option<PersistedEpisodicEscrowRow>, Self::Error>;
}

/// Evidence returned only after the persisted occurrence is durably released by the quarantine
/// ledger and atomically swapped into the canonical live replay state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedEpisodicRestoreReceipt {
    pub target_id: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub execution_id: String,
    pub escrow_digest: Sha256Digest,
    pub escrow_persistence_ref: String,
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

#[derive(Debug, Clone)]
struct ValidatedPersistedEscrow {
    escrow: EpisodicQuarantineEscrow,
    digest: Sha256Digest,
    persistence_ref: String,
}

struct VerifiedCandidateState {
    memory: EpisodicMemory,
    after_active_count: usize,
    after_quarantined_count: usize,
    after_active_digest: Sha256Digest,
}

struct PersistedEpisodicRestoreExecutor<'a, L, Q>
where
    L: EpisodicQuarantineEscrowLookup,
    Q: QuarantineLedgerPersistence,
{
    expected_target_id: String,
    execution_id: String,
    instance_id: EpisodeInstanceId,
    memory: &'a mut EpisodicMemory,
    ledger: &'a mut EpisodicQuarantineStateLedger,
    lookup: &'a L,
    ledger_persistence: &'a mut Q,
    completed_at_unix_s: u64,
}

impl<L, Q> PersistedEpisodicRestoreExecutor<'_, L, Q>
where
    L: EpisodicQuarantineEscrowLookup,
    Q: QuarantineLedgerPersistence,
{
    fn validate_permit(
        &self,
        permit: &AssuredInterventionPermit,
    ) -> Result<(), PersistedEpisodicRestoreInterventionError> {
        if permit.action() != SubjectAffectingAction::MemoryModification {
            return Err(PersistedEpisodicRestoreInterventionError::WrongAction {
                actual: permit.action(),
            });
        }
        if permit.target_id() != self.expected_target_id {
            return Err(PersistedEpisodicRestoreInterventionError::WrongTarget {
                expected: self.expected_target_id.clone(),
                actual: permit.target_id().to_string(),
            });
        }
        if permit.is_emergency() {
            return Err(PersistedEpisodicRestoreInterventionError::EmergencyRestoreNotTyped);
        }
        if permit.explicit_consent_state() != ExplicitConsentState::Granted {
            return Err(PersistedEpisodicRestoreInterventionError::ExplicitConsentRequired);
        }
        if !permit.has_welfare_review_reference() {
            return Err(PersistedEpisodicRestoreInterventionError::WelfareReviewRequired);
        }
        if !permit.has_independent_review_reference() {
            return Err(PersistedEpisodicRestoreInterventionError::IndependentReviewRequired);
        }
        Ok(())
    }

    fn validate_live_absence(
        &self,
    ) -> Result<(), PersistedEpisodicRestoreInterventionError> {
        if self
            .memory
            .get_top_episode_instances(self.memory.len())
            .iter()
            .any(|(id, _)| *id == self.instance_id)
        {
            return Err(PersistedEpisodicRestoreInterventionError::ActiveInstanceCollision(
                self.instance_id,
            ));
        }
        if self.memory.quarantined_instance(self.instance_id).is_some() {
            return Err(
                PersistedEpisodicRestoreInterventionError::InMemoryQuarantineCollision(
                    self.instance_id,
                ),
            );
        }
        Ok(())
    }

    fn load_validated_escrow(
        &self,
    ) -> Result<ValidatedPersistedEscrow, PersistedEpisodicRestoreExecutionError<L::Error, Q::Error>>
    {
        let state = self
            .ledger
            .unresolved_state(self.instance_id)
            .ok_or(PersistedEpisodicRestoreInterventionError::LedgerDoesNotQuarantine(
                self.instance_id,
            ))?;
        if state.target_id != self.expected_target_id {
            return Err(PersistedEpisodicRestoreInterventionError::LedgerTargetMismatch.into());
        }
        if state.restore_pending.is_some() {
            return Err(PersistedEpisodicRestoreInterventionError::RestoreAlreadyPending.into());
        }
        let row = self
            .lookup
            .load_episodic_quarantine_escrow(self.instance_id)
            .map_err(PersistedEpisodicRestoreExecutionError::Lookup)?
            .ok_or(PersistedEpisodicRestoreInterventionError::EscrowNotFound(
                self.instance_id,
            ))?;
        validate_persisted_escrow_row(
            row,
            &self.expected_target_id,
            state,
            self.completed_at_unix_s,
        )
        .map_err(Into::into)
    }

    fn build_candidate(
        &self,
        material: &ValidatedPersistedEscrow,
    ) -> Result<VerifiedCandidateState, PersistedEpisodicRestoreInterventionError> {
        let before_active_count = self.memory.len();
        let before_quarantined_count = self.memory.quarantined_len();
        let mut candidate = self.memory.clone();
        let restored_id = candidate
            .restore_validated_persisted_occurrence(material.escrow.episode.clone())
            .map_err(PersistedEpisodicRestoreInterventionError::MemoryMechanism)?;
        if restored_id != self.instance_id {
            return Err(PersistedEpisodicRestoreInterventionError::InstanceIdentityMismatch);
        }
        let after_active_count = candidate.len();
        let after_quarantined_count = candidate.quarantined_len();
        if after_active_count != before_active_count.saturating_add(1)
            || after_quarantined_count != before_quarantined_count
        {
            return Err(PersistedEpisodicRestoreInterventionError::RestorePostconditionFailed {
                before_active: before_active_count,
                after_active: after_active_count,
                before_quarantined: before_quarantined_count,
                after_quarantined: after_quarantined_count,
            });
        }
        let active = candidate
            .get_top_episode_instances(candidate.len())
            .into_iter()
            .find(|(id, _)| *id == self.instance_id)
            .ok_or(PersistedEpisodicRestoreInterventionError::RestorePostconditionMissing)?;
        if episode_content_id(&active.1)? != material.escrow.content_id {
            return Err(PersistedEpisodicRestoreInterventionError::ContentIdentityMismatch);
        }
        let after_active_digest = digest_episodic_memory(&candidate)
            .map_err(|error| PersistedEpisodicRestoreInterventionError::MemoryState(error.to_string()))?;
        Ok(VerifiedCandidateState {
            memory: candidate,
            after_active_count,
            after_quarantined_count,
            after_active_digest,
        })
    }
}

impl<L, Q> ReceiptedInterventionExecutor for PersistedEpisodicRestoreExecutor<'_, L, Q>
where
    L: EpisodicQuarantineEscrowLookup,
    Q: QuarantineLedgerPersistence,
{
    type Output = PersistedEpisodicRestoreReceipt;
    type Error = PersistedEpisodicRestoreExecutionError<L::Error, Q::Error>;

    fn preflight(&self, permit: &AssuredInterventionPermit) -> Result<(), Self::Error> {
        self.validate_permit(permit)?;
        self.validate_live_absence()?;
        self.load_validated_escrow()?;
        Ok(())
    }

    fn execute_receipted(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<ReceiptedExecution<Self::Output>, Self::Error> {
        self.validate_permit(permit)?;
        self.validate_live_absence()?;
        let material = self.load_validated_escrow()?;

        let before_active_count = self.memory.len();
        let before_quarantined_count = self.memory.quarantined_len();
        let before_active_digest = digest_episodic_memory(self.memory)
            .map_err(|error| PersistedEpisodicRestoreInterventionError::MemoryState(error.to_string()))?;

        // Persist domain write-ahead state before constructing any candidate active memory.
        let ledger_before_prepare = self.ledger.clone();
        let restore_prepared_head = self
            .ledger
            .append_restore_prepared(
                &self.expected_target_id,
                self.instance_id,
                material.escrow.content_id,
                self.completed_at_unix_s,
                &self.execution_id,
            )
            .map_err(PersistedEpisodicRestoreInterventionError::from)?;
        let restore_prepared_persistence_ref = match self
            .ledger_persistence
            .persist_quarantine_ledger(self.ledger.events(), restore_prepared_head)
        {
            Ok(reference) => reference,
            Err(error) => {
                *self.ledger = ledger_before_prepare;
                return Err(PersistedEpisodicRestoreExecutionError::RestorePreparedPersistence(
                    error,
                ));
            }
        };
        if !valid_ref(&restore_prepared_persistence_ref) {
            *self.ledger = ledger_before_prepare;
            return Err(
                PersistedEpisodicRestoreInterventionError::InvalidLedgerPersistenceReference.into(),
            );
        }

        // Copy-on-write activation: every fallible memory validation happens on a clone while the
        // canonical live heap remains unchanged and the durable ledger still says quarantined.
        let candidate = self.build_candidate(&material)?;
        let transition_digest = digest_persisted_restore_transition(
            &self.expected_target_id,
            self.instance_id,
            material.escrow.content_id,
            &self.execution_id,
            material.digest,
            &material.persistence_ref,
            before_active_count,
            candidate.after_active_count,
            before_quarantined_count,
            candidate.after_quarantined_count,
            before_active_digest,
            candidate.after_active_digest,
            restore_prepared_head,
            &restore_prepared_persistence_ref,
        );

        // Commit terminal domain state while the actual canonical heap is still unchanged. Once
        // this durable record exists, the remaining operation is an infallible Rust assignment of
        // an already validated candidate. A crash in that tiny window recovers from the durable
        // `Restored` ledger state and reconstructs the occurrence active on the next restart.
        let ledger_prepared = self.ledger.clone();
        let restored_head = match self.ledger.append_restored(
            &self.expected_target_id,
            self.instance_id,
            material.escrow.content_id,
            self.completed_at_unix_s,
            &self.execution_id,
            transition_digest,
        ) {
            Ok(head) => head,
            Err(error) => {
                *self.ledger = ledger_prepared;
                return Err(PersistedEpisodicRestoreExecutionError::Ledger(error));
            }
        };
        let restored_persistence_ref = match self
            .ledger_persistence
            .persist_quarantine_ledger(self.ledger.events(), restored_head)
        {
            Ok(reference) => reference,
            Err(error) => {
                *self.ledger = ledger_prepared;
                return Err(PersistedEpisodicRestoreExecutionError::RestoredPersistence(
                    error,
                ));
            }
        };
        if !valid_ref(&restored_persistence_ref) {
            *self.ledger = ledger_prepared;
            return Err(
                PersistedEpisodicRestoreInterventionError::InvalidLedgerPersistenceReference.into(),
            );
        }

        // No fallible state transition follows this assignment. The exact candidate already passed
        // all memory-level validation and the terminal quarantine-ledger state is durable.
        *self.memory = candidate.memory;

        let receipt = PersistedEpisodicRestoreReceipt {
            target_id: permit.target_id().to_string(),
            instance_id: self.instance_id,
            content_id: material.escrow.content_id,
            execution_id: self.execution_id.clone(),
            escrow_digest: material.digest,
            escrow_persistence_ref: material.persistence_ref,
            before_active_count,
            after_active_count: candidate.after_active_count,
            before_quarantined_count,
            after_quarantined_count: candidate.after_quarantined_count,
            before_active_digest,
            after_active_digest: candidate.after_active_digest,
            restore_prepared_head,
            restored_head,
            restore_prepared_persistence_ref,
            restored_persistence_ref,
            ledger_generation: self.ledger.generation(),
        };
        let result_digest = digest_persisted_restore_result(&receipt, permit.rationale());
        let evidence_ref = format!(
            "symthaea-memory:persisted-episodic-restore:v1:sha256:{}",
            hex_digest(result_digest)
        );
        ReceiptedExecution::new(
            receipt,
            self.completed_at_unix_s,
            result_digest,
            evidence_ref,
        )
        .map_err(PersistedEpisodicRestoreExecutionError::Observation)
    }
}

/// Govern one exact post-restart restoration through evidence-bound authority, explicit consent,
/// independent review, an exact durable escrow point lookup, two-phase quarantine-ledger state and
/// copy-on-write canonical heap activation.
#[allow(clippy::too_many_arguments)]
pub fn execute_governed_persisted_episodic_restore<P, L, Q>(
    permit: DurableEvidenceBoundInterventionPermit,
    execution_id: impl Into<String>,
    store_target_id: &str,
    instance_id: EpisodeInstanceId,
    memory: &mut EpisodicMemory,
    ledger: &mut EpisodicQuarantineStateLedger,
    lookup: &L,
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
        PersistedEpisodicRestoreReceipt,
        PersistedEpisodicRestoreExecutionError<L::Error, Q::Error>,
        P::Error,
    >,
    GovernedPersistedEpisodicRestoreError<P::Error>,
>
where
    P: ExecutionJournalPersistence,
    L: EpisodicQuarantineEscrowLookup,
    Q: QuarantineLedgerPersistence,
{
    let execution_id = execution_id.into();
    if execution_id.trim().is_empty() || execution_id != execution_id.trim() {
        return Err(GovernedPersistedEpisodicRestoreError::Configuration(
            PersistedEpisodicRestoreInterventionError::InvalidExecutionId,
        ));
    }
    let expected_target_id = episodic_instance_target_id(store_target_id, instance_id)
        .map_err(|error| {
            GovernedPersistedEpisodicRestoreError::Configuration(
                PersistedEpisodicRestoreInterventionError::TargetConstruction(error.to_string()),
            )
        })?;
    if permit.action() != SubjectAffectingAction::MemoryModification {
        return Err(GovernedPersistedEpisodicRestoreError::Configuration(
            PersistedEpisodicRestoreInterventionError::WrongAction {
                actual: permit.action(),
            },
        ));
    }
    if permit.target_id() != expected_target_id {
        return Err(GovernedPersistedEpisodicRestoreError::Configuration(
            PersistedEpisodicRestoreInterventionError::WrongTarget {
                expected: expected_target_id,
                actual: permit.target_id().to_string(),
            },
        ));
    }

    let mut executor = PersistedEpisodicRestoreExecutor {
        expected_target_id: permit.target_id().to_string(),
        execution_id: execution_id.clone(),
        instance_id,
        memory,
        ledger,
        lookup,
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
    .map_err(GovernedPersistedEpisodicRestoreError::Gate)
}

fn validate_persisted_escrow_row(
    row: PersistedEpisodicEscrowRow,
    expected_target_id: &str,
    ledger_state: &QuarantineLedgerState,
    restore_unix_s: u64,
) -> Result<ValidatedPersistedEscrow, PersistedEpisodicRestoreInterventionError> {
    row.escrow
        .validate()
        .map_err(PersistedEpisodicRestoreInterventionError::EscrowValidation)?;
    if row.escrow.instance_id != ledger_state.instance_id {
        return Err(PersistedEpisodicRestoreInterventionError::InstanceIdentityMismatch);
    }
    if row.escrow.target_id != expected_target_id || row.escrow.target_id != ledger_state.target_id {
        return Err(PersistedEpisodicRestoreInterventionError::EscrowTargetMismatch);
    }
    if row.escrow.content_id != ledger_state.content_id
        || episode_content_id(&row.escrow.episode)? != ledger_state.content_id
    {
        return Err(PersistedEpisodicRestoreInterventionError::ContentIdentityMismatch);
    }
    if row.escrow.captured_at_unix_s > restore_unix_s {
        return Err(PersistedEpisodicRestoreInterventionError::EscrowCapturedAfterRestore);
    }
    if !valid_ref(&row.persistence_ref) {
        return Err(PersistedEpisodicRestoreInterventionError::InvalidEscrowPersistenceReference);
    }
    let actual_digest = digest_episodic_quarantine_escrow(&row.escrow)
        .map_err(PersistedEpisodicRestoreInterventionError::EscrowValidation)?;
    if actual_digest != row.stored_digest || actual_digest != ledger_state.escrow_digest {
        return Err(PersistedEpisodicRestoreInterventionError::EscrowDigestMismatch);
    }
    if row.persistence_ref != ledger_state.escrow_persistence_ref {
        return Err(PersistedEpisodicRestoreInterventionError::EscrowPersistenceRefMismatch);
    }
    Ok(ValidatedPersistedEscrow {
        escrow: row.escrow,
        digest: actual_digest,
        persistence_ref: row.persistence_ref,
    })
}

#[allow(clippy::too_many_arguments)]
fn digest_persisted_restore_transition(
    target_id: &str,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    execution_id: &str,
    escrow_digest: Sha256Digest,
    escrow_persistence_ref: &str,
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
    hasher.update(PERSISTED_RESTORE_TRANSITION_DOMAIN);
    hash_text(&mut hasher, target_id);
    hasher.update(&instance_id.as_uuid().as_u128().to_le_bytes());
    hasher.update(&content_id.digest().0);
    hash_text(&mut hasher, execution_id);
    hasher.update(&escrow_digest.0);
    hash_text(&mut hasher, escrow_persistence_ref);
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

fn digest_persisted_restore_result(
    receipt: &PersistedEpisodicRestoreReceipt,
    rationale: &str,
) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(PERSISTED_RESTORE_RESULT_DOMAIN);
    hash_text(&mut hasher, &receipt.target_id);
    hasher.update(&receipt.instance_id.as_uuid().as_u128().to_le_bytes());
    hasher.update(&receipt.content_id.digest().0);
    hash_text(&mut hasher, &receipt.execution_id);
    hasher.update(&receipt.escrow_digest.0);
    hash_text(&mut hasher, &receipt.escrow_persistence_ref);
    hasher.update(&(receipt.before_active_count as u64).to_le_bytes());
    hasher.update(&(receipt.after_active_count as u64).to_le_bytes());
    hasher.update(&(receipt.before_quarantined_count as u64).to_le_bytes());
    hasher.update(&(receipt.after_quarantined_count as u64).to_le_bytes());
    hasher.update(&receipt.before_active_digest.0);
    hasher.update(&receipt.after_active_digest.0);
    hasher.update(&receipt.restore_prepared_head.0);
    hasher.update(&receipt.restored_head.0);
    hash_text(&mut hasher, &receipt.restore_prepared_persistence_ref);
    hash_text(&mut hasher, &receipt.restored_persistence_ref);
    hasher.update(&receipt.ledger_generation.to_le_bytes());
    hash_text(&mut hasher, rationale);
    hasher.finalize()
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
pub enum PersistedEpisodicRestoreInterventionError {
    #[error("persisted episodic restore requires MemoryModification authority; actual={actual:?}")]
    WrongAction { actual: SubjectAffectingAction },
    #[error("persisted episodic restore target mismatch: expected={expected:?}, actual={actual:?}")]
    WrongTarget { expected: String, actual: String },
    #[error("could not construct exact persisted episodic restore target: {0}")]
    TargetConstruction(String),
    #[error("persisted episodic restore execution id is invalid")]
    InvalidExecutionId,
    #[error("emergency persisted episodic restore is not typed in v1")]
    EmergencyRestoreNotTyped,
    #[error("persisted episodic restore requires explicit granted subject consent")]
    ExplicitConsentRequired,
    #[error("persisted episodic restore requires a welfare-review reference")]
    WelfareReviewRequired,
    #[error("persisted episodic restore requires an independent-review reference")]
    IndependentReviewRequired,
    #[error("quarantine ledger does not contain unresolved occurrence {0}")]
    LedgerDoesNotQuarantine(EpisodeInstanceId),
    #[error("quarantine ledger target disagrees with the authorized persisted restore target")]
    LedgerTargetMismatch,
    #[error("a persisted restore is already pending reconciliation for this occurrence")]
    RestoreAlreadyPending,
    #[error("persisted quarantine escrow not found for occurrence {0}")]
    EscrowNotFound(EpisodeInstanceId),
    #[error("persisted escrow occurrence identity mismatch")]
    InstanceIdentityMismatch,
    #[error("persisted escrow target mismatch")]
    EscrowTargetMismatch,
    #[error("persisted escrow content identity mismatch")]
    ContentIdentityMismatch,
    #[error("persisted escrow was captured after the requested restore time")]
    EscrowCapturedAfterRestore,
    #[error("persisted escrow digest disagrees with durable row or quarantine ledger")]
    EscrowDigestMismatch,
    #[error("persisted escrow persistence reference disagrees with quarantine ledger")]
    EscrowPersistenceRefMismatch,
    #[error("persisted escrow persistence reference is invalid")]
    InvalidEscrowPersistenceReference,
    #[error("persisted escrow validation failed: {0}")]
    EscrowValidation(#[source] EpisodicQuarantineInterventionError),
    #[error("persisted occurrence is already active: {0}")]
    ActiveInstanceCollision(EpisodeInstanceId),
    #[error("persisted occurrence is already present in the live quarantine map: {0}")]
    InMemoryQuarantineCollision(EpisodeInstanceId),
    #[error("canonical persisted-memory restore mechanism failed: {0}")]
    MemoryMechanism(#[source] PersistedEpisodicImportError),
    #[error("persisted restore postcondition missing exact active occurrence")]
    RestorePostconditionMissing,
    #[error("persisted restore postcondition failed: active {before_active}->{after_active}, quarantined {before_quarantined}->{after_quarantined}")]
    RestorePostconditionFailed {
        before_active: usize,
        after_active: usize,
        before_quarantined: usize,
        after_quarantined: usize,
    },
    #[error("persisted restore memory-state digest failed: {0}")]
    MemoryState(String),
    #[error("quarantine-ledger transition failed: {0}")]
    Ledger(#[source] QuarantineLedgerError),
    #[error("quarantine-ledger persistence returned an invalid durable reference")]
    InvalidLedgerPersistenceReference,
    #[error(transparent)]
    ContentIdentity(#[from] EpisodeContentIdError),
}

impl From<QuarantineLedgerError> for PersistedEpisodicRestoreInterventionError {
    fn from(value: QuarantineLedgerError) -> Self {
        Self::Ledger(value)
    }
}

#[derive(Debug, Error)]
pub enum PersistedEpisodicRestoreExecutionError<LE, QE>
where
    LE: StdError + Send + Sync + 'static,
    QE: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Intervention(#[from] PersistedEpisodicRestoreInterventionError),
    #[error("persisted escrow point lookup failed: {0}")]
    Lookup(#[source] LE),
    #[error("could not durably persist RestorePrepared quarantine-ledger state: {0}")]
    RestorePreparedPersistence(#[source] QE),
    #[error("could not durably persist Restored quarantine-ledger state: {0}")]
    RestoredPersistence(#[source] QE),
    #[error("quarantine-ledger transition failed after candidate construction: {0}")]
    Ledger(#[source] QuarantineLedgerError),
    #[error(transparent)]
    Observation(#[from] ExecutionObservationError),
}

#[derive(Debug, Error)]
pub enum GovernedPersistedEpisodicRestoreError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("governed persisted episodic restore is misconfigured: {0}")]
    Configuration(#[source] PersistedEpisodicRestoreInterventionError),
    #[error(transparent)]
    Gate(#[from] JournaledExecutionGateError<E>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    fn episode(id: EpisodeInstanceId) -> symthaea_memory::episodic_replay::Episode {
        let mut episode = symthaea_memory::episodic_replay::Episode::new(
            ContinuousHV::from_vec(vec![0.25; 8]),
            ContinuousHV::from_vec(vec![0.75; 8]),
            0.8,
            10,
        );
        episode.instance_id = Some(id);
        episode
    }

    fn fixture() -> (
        PersistedEpisodicEscrowRow,
        QuarantineLedgerState,
        String,
    ) {
        let mut origin = EpisodicMemory::new(
            symthaea_memory::episodic_replay::EpisodicReplayConfig {
                psi_threshold: 0.0,
                ..Default::default()
            },
        );
        let id = origin
            .store_if_significant_with_id(symthaea_memory::episodic_replay::Episode::new(
                ContinuousHV::from_vec(vec![0.25; 8]),
                ContinuousHV::from_vec(vec![0.75; 8]),
                0.8,
                10,
            ))
            .unwrap();
        let exact = origin
            .get_top_episode_instances(1)
            .into_iter()
            .next()
            .unwrap()
            .1;
        let content_id = episode_content_id(&exact).unwrap();
        let target = episodic_instance_target_id("symthaea:self:episodic-memory", id).unwrap();
        let escrow = EpisodicQuarantineEscrow {
            schema_version: crate::memory_quarantine::EPISODIC_QUARANTINE_ESCROW_SCHEMA.into(),
            target_id: target.clone(),
            instance_id: id,
            content_id,
            captured_at_unix_s: 20,
            pre_active_state_digest: Sha256Digest([7; 32]),
            episode: exact,
        };
        let digest = digest_episodic_quarantine_escrow(&escrow).unwrap();
        let reference = format!("sqlite-continuity:escrow:{id}:fixture");
        let row = PersistedEpisodicEscrowRow {
            escrow,
            stored_digest: digest,
            persistence_ref: reference.clone(),
        };
        let state = QuarantineLedgerState {
            target_id: target.clone(),
            instance_id: id,
            content_id,
            quarantined_at_unix_s: 20,
            escrow_digest: digest,
            escrow_persistence_ref: reference,
            quarantine_generation: 1,
            restore_pending: None,
        };
        (row, state, target)
    }

    #[test]
    fn exact_row_binds_uuid_content_digest_ref_and_target() {
        let (row, state, target) = fixture();
        let validated = validate_persisted_escrow_row(row, &target, &state, 30).unwrap();
        assert_eq!(validated.escrow.instance_id, state.instance_id);
        assert_eq!(validated.escrow.content_id, state.content_id);
        assert_eq!(validated.digest, state.escrow_digest);
        assert_eq!(validated.persistence_ref, state.escrow_persistence_ref);
    }

    #[test]
    fn row_digest_substitution_fails_closed() {
        let (mut row, state, target) = fixture();
        row.stored_digest = Sha256Digest([9; 32]);
        assert!(matches!(
            validate_persisted_escrow_row(row, &target, &state, 30),
            Err(PersistedEpisodicRestoreInterventionError::EscrowDigestMismatch)
        ));
    }

    #[test]
    fn row_reference_substitution_fails_closed() {
        let (mut row, state, target) = fixture();
        row.persistence_ref = "sqlite-continuity:escrow:other".into();
        assert!(matches!(
            validate_persisted_escrow_row(row, &target, &state, 30),
            Err(PersistedEpisodicRestoreInterventionError::EscrowPersistenceRefMismatch)
        ));
    }

    #[test]
    fn row_content_substitution_fails_closed() {
        let (mut row, state, target) = fixture();
        row.escrow.episode.psi = 0.1;
        assert!(validate_persisted_escrow_row(row, &target, &state, 30).is_err());
    }

    #[test]
    fn future_escrow_fails_closed() {
        let (mut row, state, target) = fixture();
        row.escrow.captured_at_unix_s = 31;
        row.stored_digest = digest_episodic_quarantine_escrow(&row.escrow).unwrap();
        assert!(matches!(
            validate_persisted_escrow_row(row, &target, &state, 30),
            Err(PersistedEpisodicRestoreInterventionError::EscrowCapturedAfterRestore)
        ));
    }

    #[test]
    fn raw_episode_helper_keeps_exact_id() {
        let (_, state, _) = fixture();
        assert_eq!(episode(state.instance_id).instance_id, Some(state.instance_id));
    }
}
