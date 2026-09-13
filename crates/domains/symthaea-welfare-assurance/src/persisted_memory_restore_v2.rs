// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prepared-digest-bound governed restoration of one exact persisted episodic occurrence.
//!
//! This is an additive V2 execution path. The V1 persisted restore remains available for historical
//! compatibility. V2 opts into the generic `PreparedExecutionContextV2` boundary and stores an
//! independently recomputable cross-ledger commitment in the existing
//! `Restored.restore_result_digest` field. No quarantine-ledger enum or bincode discriminant changes.

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
use crate::execution_adapter::{ExecutionJournalPersistence, JournaledExecutionOutcome};
use crate::execution_adapter_v2::{
    ContextualReceiptedExecution, JournaledExecutionV2GateError, ReceiptedInterventionExecutorV2,
    execute_durable_intervention_journaled_v2,
};
use crate::execution_recovery::InterventionExecutionJournal;
use crate::memory_identity::{EpisodeContentId, episode_content_id};
use crate::memory_intervention::digest_episodic_memory;
use crate::memory_quarantine::{
    EpisodicQuarantineEscrow, digest_episodic_quarantine_escrow, episodic_instance_target_id,
};
use crate::persisted_memory_restore::{
    EpisodicQuarantineEscrowLookup, PersistedEpisodicEscrowRow, PersistedEpisodicRestoreExecutionError,
    PersistedEpisodicRestoreInterventionError, PersistedEpisodicRestoreReceipt,
};
use crate::persisted_restore_correlation_v2::{
    PersistedRestoreCorrelationV2Error, digest_persisted_restore_correlation_from_context_v2,
};
use crate::prepared_execution_context_v2::PreparedExecutionContextV2;
use crate::quarantine_ledger_persistence::QuarantineLedgerPersistence;
use crate::quarantine_state_ledger::{EpisodicQuarantineStateLedger, QuarantineLedgerState};
use crate::replay_recovery::DurableEvidenceBoundInterventionPermit;
use crate::restored_continuity_promotion::{
    RestoredContinuityPromotionBarrier, RestoredContinuityPromotionError,
    RestoredContinuityPromotionFailure, RestoredContinuityPromotionRequest,
};

const PERSISTED_RESTORE_RESULT_V3_DOMAIN: &[u8] =
    b"symthaea.welfare.persisted-episodic-restore-result.v3\0";
const MAX_REF_BYTES: usize = 2048;
const MAX_EXECUTION_ID_BYTES: usize = 256;

/// V2 terminal receipt. The historical V1 receipt remains intact and is augmented with the exact
/// generic Prepared identity used by the cross-ledger correlation commitment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedEpisodicRestoreReceiptV2 {
    pub restore: PersistedEpisodicRestoreReceipt,
    pub generic_prepared_digest: Sha256Digest,
    pub generic_prepared_persistence_ref: String,
    pub restore_correlation_digest: Sha256Digest,
}

#[derive(Debug, Clone)]
struct ValidatedPersistedEscrowV2 {
    escrow: EpisodicQuarantineEscrow,
    digest: Sha256Digest,
    persistence_ref: String,
}

struct VerifiedCandidateStateV2 {
    memory: EpisodicMemory,
    after_active_count: usize,
    after_quarantined_count: usize,
    after_active_digest: Sha256Digest,
}

struct PersistedEpisodicRestoreExecutorV2<'a, L, Q, B>
where
    L: EpisodicQuarantineEscrowLookup,
    Q: QuarantineLedgerPersistence,
    B: RestoredContinuityPromotionBarrier,
{
    store_target_id: String,
    expected_target_id: String,
    execution_id: String,
    instance_id: EpisodeInstanceId,
    memory: &'a mut EpisodicMemory,
    ledger: &'a mut EpisodicQuarantineStateLedger,
    lookup: &'a L,
    ledger_persistence: &'a mut Q,
    promotion_barrier: &'a mut B,
    completed_at_unix_s: u64,
}

impl<L, Q, B> PersistedEpisodicRestoreExecutorV2<'_, L, Q, B>
where
    L: EpisodicQuarantineEscrowLookup,
    Q: QuarantineLedgerPersistence,
    B: RestoredContinuityPromotionBarrier,
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

    fn validate_context(
        &self,
        context: &PreparedExecutionContextV2,
    ) -> Result<(), PersistedEpisodicRestoreExecutionV2Error<L::Error, Q::Error>> {
        if context.execution_id() != self.execution_id {
            return Err(PersistedEpisodicRestoreExecutionV2Error::PreparedExecutionIdMismatch {
                expected: self.execution_id.clone(),
                actual: context.execution_id().to_string(),
            });
        }
        if context.target_id() != self.expected_target_id {
            return Err(PersistedEpisodicRestoreExecutionV2Error::PreparedTargetMismatch {
                expected: self.expected_target_id.clone(),
                actual: context.target_id().to_string(),
            });
        }
        Ok(())
    }

    fn validate_live_absence(&self) -> Result<(), PersistedEpisodicRestoreInterventionError> {
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
    ) -> Result<
        ValidatedPersistedEscrowV2,
        PersistedEpisodicRestoreExecutionV2Error<L::Error, Q::Error>,
    > {
        let state = self.ledger.unresolved_state(self.instance_id).ok_or_else(|| {
            intervention_v2(PersistedEpisodicRestoreInterventionError::LedgerDoesNotQuarantine(
                self.instance_id,
            ))
        })?;
        if state.target_id != self.expected_target_id {
            return Err(intervention_v2(
                PersistedEpisodicRestoreInterventionError::LedgerTargetMismatch,
            ));
        }
        if state.restore_pending.is_some() {
            return Err(intervention_v2(
                PersistedEpisodicRestoreInterventionError::RestoreAlreadyPending,
            ));
        }
        let row = self
            .lookup
            .load_episodic_quarantine_escrow(self.instance_id)
            .map_err(|error| {
                PersistedEpisodicRestoreExecutionV2Error::Legacy(
                    PersistedEpisodicRestoreExecutionError::Lookup(error),
                )
            })?
            .ok_or_else(|| {
                intervention_v2(PersistedEpisodicRestoreInterventionError::EscrowNotFound(
                    self.instance_id,
                ))
            })?;
        validate_persisted_escrow_row_v2(
            row,
            &self.expected_target_id,
            state,
            self.completed_at_unix_s,
        )
        .map_err(intervention_v2)
    }

    fn build_candidate(
        &self,
        material: &ValidatedPersistedEscrowV2,
    ) -> Result<VerifiedCandidateStateV2, PersistedEpisodicRestoreInterventionError> {
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
        let after_active_digest = digest_episodic_memory(&candidate).map_err(|error| {
            PersistedEpisodicRestoreInterventionError::MemoryState(error.to_string())
        })?;
        Ok(VerifiedCandidateStateV2 {
            memory: candidate,
            after_active_count,
            after_quarantined_count,
            after_active_digest,
        })
    }
}

impl<L, Q, B> ReceiptedInterventionExecutorV2 for PersistedEpisodicRestoreExecutorV2<'_, L, Q, B>
where
    L: EpisodicQuarantineEscrowLookup,
    Q: QuarantineLedgerPersistence,
    B: RestoredContinuityPromotionBarrier,
{
    type Output = PersistedEpisodicRestoreReceiptV2;
    type Error = PersistedEpisodicRestoreExecutionV2Error<L::Error, Q::Error>;

    fn preflight(&self, permit: &AssuredInterventionPermit) -> Result<(), Self::Error> {
        self.validate_permit(permit).map_err(intervention_v2)?;
        self.validate_live_absence().map_err(intervention_v2)?;
        self.load_validated_escrow()?;
        Ok(())
    }

    fn execute_receipted_v2(
        &mut self,
        permit: &AssuredInterventionPermit,
        context: &PreparedExecutionContextV2,
    ) -> Result<ContextualReceiptedExecution<Self::Output>, Self::Error> {
        self.validate_context(context)?;
        self.validate_permit(permit).map_err(intervention_v2)?;
        self.validate_live_absence().map_err(intervention_v2)?;
        let material = self.load_validated_escrow()?;

        let before_active_count = self.memory.len();
        let before_quarantined_count = self.memory.quarantined_len();
        let before_active_digest = digest_episodic_memory(self.memory).map_err(|error| {
            intervention_v2(PersistedEpisodicRestoreInterventionError::MemoryState(
                error.to_string(),
            ))
        })?;

        // Domain write-ahead remains the existing V1 event for byte-level compatibility. The exact
        // generic Prepared identity is committed by the following Restored correlation digest.
        let restore_prepared_head = self
            .ledger
            .append_restore_prepared(
                &self.expected_target_id,
                self.instance_id,
                material.escrow.content_id,
                self.completed_at_unix_s,
                &self.execution_id,
            )
            .map_err(|error| intervention_v2(error.into()))?;
        let restore_prepared_persistence_ref = self
            .ledger_persistence
            .persist_quarantine_ledger(self.ledger.events(), restore_prepared_head)
            .map_err(|error| {
                PersistedEpisodicRestoreExecutionV2Error::Legacy(
                    PersistedEpisodicRestoreExecutionError::RestorePreparedPersistence(error),
                )
            })?;
        if !valid_ref(&restore_prepared_persistence_ref) {
            return Err(intervention_v2(
                PersistedEpisodicRestoreInterventionError::InvalidLedgerPersistenceReference,
            ));
        }

        let candidate = self.build_candidate(&material).map_err(intervention_v2)?;
        let restore_correlation_digest = digest_persisted_restore_correlation_from_context_v2(
            context,
            self.instance_id,
            material.escrow.content_id,
            self.completed_at_unix_s,
            restore_prepared_head,
        )?;

        let restored_head = self
            .ledger
            .append_restored(
                &self.expected_target_id,
                self.instance_id,
                material.escrow.content_id,
                self.completed_at_unix_s,
                &self.execution_id,
                restore_correlation_digest,
            )
            .map_err(|error| {
                PersistedEpisodicRestoreExecutionV2Error::Legacy(
                    PersistedEpisodicRestoreExecutionError::Ledger(error),
                )
            })?;
        let restored_persistence_ref = self
            .ledger_persistence
            .persist_quarantine_ledger(self.ledger.events(), restored_head)
            .map_err(|error| {
                PersistedEpisodicRestoreExecutionV2Error::Legacy(
                    PersistedEpisodicRestoreExecutionError::RestoredPersistence(error),
                )
            })?;
        if !valid_ref(&restored_persistence_ref) {
            return Err(intervention_v2(
                PersistedEpisodicRestoreInterventionError::InvalidLedgerPersistenceReference,
            ));
        }

        let promotion_request = RestoredContinuityPromotionRequest::try_new(
            &self.store_target_id,
            &self.expected_target_id,
            self.instance_id,
            material.escrow.content_id,
            &self.execution_id,
            restored_head,
        )
        .map_err(PersistedEpisodicRestoreExecutionV2Error::PromotionRequest)?;
        let continuity_promotion = self
            .promotion_barrier
            .commit_restored_continuity(&promotion_request)
            .map_err(PersistedEpisodicRestoreExecutionV2Error::PromotionBarrier)?;
        continuity_promotion
            .validate_for(&promotion_request)
            .map_err(PersistedEpisodicRestoreExecutionV2Error::PromotionEvidence)?;

        let restore = PersistedEpisodicRestoreReceipt {
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
            continuity_promotion,
        };
        let receipt = PersistedEpisodicRestoreReceiptV2 {
            restore,
            generic_prepared_digest: context.prepared_digest(),
            generic_prepared_persistence_ref: context.prepared_persistence_ref().to_string(),
            restore_correlation_digest,
        };
        let result_digest = digest_persisted_restore_result_v3(&receipt, permit.rationale());
        let evidence_ref = format!(
            "symthaea-memory:persisted-episodic-restore:v3:sha256:{}",
            hex_digest(result_digest)
        );
        let receipted = ContextualReceiptedExecution::new(
            receipt,
            self.completed_at_unix_s,
            result_digest,
            evidence_ref,
        )
        .map_err(|error| {
            PersistedEpisodicRestoreExecutionV2Error::Legacy(
                PersistedEpisodicRestoreExecutionError::Observation(error),
            )
        })?;

        // No fallible operation follows canonical activation.
        *self.memory = candidate.memory;
        Ok(receipted)
    }
}

/// Stronger additive counterpart of `execute_governed_persisted_episodic_restore`.
#[allow(clippy::too_many_arguments)]
pub fn execute_governed_persisted_episodic_restore_v2<P, L, Q, B>(
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
    promotion_barrier: &mut B,
) -> Result<
    JournaledExecutionOutcome<
        PersistedEpisodicRestoreReceiptV2,
        PersistedEpisodicRestoreExecutionV2Error<L::Error, Q::Error>,
        P::Error,
    >,
    GovernedPersistedEpisodicRestoreV2Error<P::Error>,
>
where
    P: ExecutionJournalPersistence,
    L: EpisodicQuarantineEscrowLookup,
    Q: QuarantineLedgerPersistence,
    B: RestoredContinuityPromotionBarrier,
{
    let execution_id = execution_id.into();
    if execution_id.trim().is_empty()
        || execution_id != execution_id.trim()
        || execution_id.len() > MAX_EXECUTION_ID_BYTES
        || execution_id.chars().any(char::is_control)
    {
        return Err(GovernedPersistedEpisodicRestoreV2Error::Configuration(
            PersistedEpisodicRestoreInterventionError::InvalidExecutionId,
        ));
    }
    let expected_target_id = episodic_instance_target_id(store_target_id, instance_id).map_err(
        |error| {
            GovernedPersistedEpisodicRestoreV2Error::Configuration(
                PersistedEpisodicRestoreInterventionError::TargetConstruction(error.to_string()),
            )
        },
    )?;
    if permit.action() != SubjectAffectingAction::MemoryModification {
        return Err(GovernedPersistedEpisodicRestoreV2Error::Configuration(
            PersistedEpisodicRestoreInterventionError::WrongAction {
                actual: permit.action(),
            },
        ));
    }
    if permit.target_id() != expected_target_id {
        return Err(GovernedPersistedEpisodicRestoreV2Error::Configuration(
            PersistedEpisodicRestoreInterventionError::WrongTarget {
                expected: expected_target_id,
                actual: permit.target_id().to_string(),
            },
        ));
    }

    let mut executor = PersistedEpisodicRestoreExecutorV2 {
        store_target_id: store_target_id.to_string(),
        expected_target_id: permit.target_id().to_string(),
        execution_id: execution_id.clone(),
        instance_id,
        memory,
        ledger,
        lookup,
        ledger_persistence,
        promotion_barrier,
        completed_at_unix_s: unix_s,
    };

    execute_durable_intervention_journaled_v2(
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
    .map_err(GovernedPersistedEpisodicRestoreV2Error::Gate)
}

fn validate_persisted_escrow_row_v2(
    row: PersistedEpisodicEscrowRow,
    expected_target_id: &str,
    ledger_state: &QuarantineLedgerState,
    restore_unix_s: u64,
) -> Result<ValidatedPersistedEscrowV2, PersistedEpisodicRestoreInterventionError> {
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
    Ok(ValidatedPersistedEscrowV2 {
        escrow: row.escrow,
        digest: actual_digest,
        persistence_ref: row.persistence_ref,
    })
}

fn digest_persisted_restore_result_v3(
    receipt: &PersistedEpisodicRestoreReceiptV2,
    rationale: &str,
) -> Sha256Digest {
    let restore = &receipt.restore;
    let mut hasher = Sha256::new();
    hasher.update(PERSISTED_RESTORE_RESULT_V3_DOMAIN);
    hasher.update(&receipt.generic_prepared_digest.0);
    hash_text(&mut hasher, &receipt.generic_prepared_persistence_ref);
    hasher.update(&receipt.restore_correlation_digest.0);
    hash_text(&mut hasher, &restore.target_id);
    hasher.update(&restore.instance_id.as_uuid().as_u128().to_le_bytes());
    hasher.update(&restore.content_id.digest().0);
    hash_text(&mut hasher, &restore.execution_id);
    hasher.update(&restore.escrow_digest.0);
    hash_text(&mut hasher, &restore.escrow_persistence_ref);
    hasher.update(&(restore.before_active_count as u64).to_le_bytes());
    hasher.update(&(restore.after_active_count as u64).to_le_bytes());
    hasher.update(&(restore.before_quarantined_count as u64).to_le_bytes());
    hasher.update(&(restore.after_quarantined_count as u64).to_le_bytes());
    hasher.update(&restore.before_active_digest.0);
    hasher.update(&restore.after_active_digest.0);
    hasher.update(&restore.restore_prepared_head.0);
    hasher.update(&restore.restored_head.0);
    hash_text(&mut hasher, &restore.restore_prepared_persistence_ref);
    hash_text(&mut hasher, &restore.restored_persistence_ref);
    hasher.update(&restore.ledger_generation.to_le_bytes());
    hasher.update(&restore.continuity_promotion.previous_anchor_commitment().0);
    hasher.update(&restore.continuity_promotion.next_anchor_commitment().0);
    hasher.update(&restore.continuity_promotion.next_anchor_revision().to_le_bytes());
    hasher.update(&restore.continuity_promotion.continuity_manifest_digest().0);
    hash_text(
        &mut hasher,
        restore.continuity_promotion.anchor_reference(),
    );
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

fn intervention_v2<LE, QE>(
    error: PersistedEpisodicRestoreInterventionError,
) -> PersistedEpisodicRestoreExecutionV2Error<LE, QE>
where
    LE: StdError + Send + Sync + 'static,
    QE: StdError + Send + Sync + 'static,
{
    PersistedEpisodicRestoreExecutionV2Error::Legacy(
        PersistedEpisodicRestoreExecutionError::Intervention(error),
    )
}

#[derive(Debug, Error)]
pub enum PersistedEpisodicRestoreExecutionV2Error<LE, QE>
where
    LE: StdError + Send + Sync + 'static,
    QE: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Legacy(#[from] PersistedEpisodicRestoreExecutionError<LE, QE>),
    #[error("V2 generic Prepared execution id mismatch: expected={expected:?}, actual={actual:?}")]
    PreparedExecutionIdMismatch { expected: String, actual: String },
    #[error("V2 generic Prepared target mismatch: expected={expected:?}, actual={actual:?}")]
    PreparedTargetMismatch { expected: String, actual: String },
    #[error(transparent)]
    Correlation(#[from] PersistedRestoreCorrelationV2Error),
    #[error("could not construct restored-continuity promotion request: {0}")]
    PromotionRequest(#[source] RestoredContinuityPromotionError),
    #[error("independent restored-continuity promotion failed: {0}")]
    PromotionBarrier(#[source] RestoredContinuityPromotionFailure),
    #[error("returned restored-continuity promotion evidence did not bind the exact restore: {0}")]
    PromotionEvidence(#[source] RestoredContinuityPromotionError),
}

#[derive(Debug, Error)]
pub enum GovernedPersistedEpisodicRestoreV2Error<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("governed persisted episodic restore V2 is misconfigured: {0}")]
    Configuration(#[source] PersistedEpisodicRestoreInterventionError),
    #[error(transparent)]
    Gate(#[from] JournaledExecutionV2GateError<E>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_core::intervention_interlock::WelfareConstraintLevel;
    use symthaea_psych_bench::moral_patient::ProtectionDisposition;

    use crate::execution_recovery::{
        EXECUTION_JOURNAL_SCHEMA, PreparedInterventionExecution, digest_prepared_execution,
    };

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn prepared(authority_id: &str, target_id: &str) -> PreparedInterventionExecution {
        PreparedInterventionExecution {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: "exec:restore:v2:test".into(),
            authority_id: authority_id.into(),
            target_id: target_id.into(),
            action: SubjectAffectingAction::MemoryModification,
            rationale_digest: digest(1),
            welfare_profile_digest: digest(2),
            precaution_policy_digest: digest(3),
            protection_disposition: ProtectionDisposition::Baseline,
            welfare_constraint: WelfareConstraintLevel::Baseline,
            replay_generation: 1,
            replay_snapshot_digest: digest(4),
            replay_persistence_ref: "replay:v2:test".into(),
            prepared_at_unix_s: 100,
            permit_not_after_unix_s: 200,
        }
    }

    #[test]
    fn exact_prepared_context_identity_is_required() {
        let target = "symthaea:self:episodic-memory:instance:v2-test";
        let prepared = prepared("authority:v2:test", target);
        let prepared_digest = digest_prepared_execution(&prepared).unwrap();
        PreparedExecutionContextV2::verify_exact_prepared_digest(&prepared, prepared_digest).unwrap();
        let context = PreparedExecutionContextV2::from_verified_durable(
            &prepared,
            prepared_digest,
            "execution-journal:prepared:v2:test".into(),
        );
        assert_eq!(context.execution_id(), "exec:restore:v2:test");
        assert_eq!(context.target_id(), target);
    }

    #[test]
    fn V2_result_domain_is_distinct_from_legacy_result_domain() {
        assert_eq!(
            PERSISTED_RESTORE_RESULT_V3_DOMAIN,
            b"symthaea.welfare.persisted-episodic-restore-result.v3\0"
        );
    }

    #[test]
    fn candidate_fixture_can_preserve_exact_instance_identity() {
        let mut origin = EpisodicMemory::new(
            symthaea_memory::episodic_replay::EpisodicReplayConfig {
                psi_threshold: 0.0,
                ..Default::default()
            },
        );
        let id = origin
            .store_if_significant_with_id(symthaea_memory::episodic_replay::Episode::new(
                ContinuousHV::from_vec(vec![0.2; 8]),
                ContinuousHV::from_vec(vec![0.8; 8]),
                0.9,
                10,
            ))
            .unwrap();
        let exact = origin
            .get_top_episode_instances(1)
            .into_iter()
            .next()
            .unwrap()
            .1;
        assert_eq!(exact.instance_id, Some(id));
    }
}
