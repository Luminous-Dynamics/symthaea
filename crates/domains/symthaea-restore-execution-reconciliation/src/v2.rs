// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict V2 post-crash reconciliation for prepared-digest-bound persisted restores.
//!
//! Historical V1 reconciliation remains available through the parent module. This module emits a
//! stronger typed outcome only when the anchored domain `Restored.restore_result_digest` exactly
//! equals an independently recomputed commitment to the recovered generic `Prepared` record.
//! Merely matching execution ID, target, occurrence UUID and content is insufficient.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_episodic_continuity::SqliteEpisodicContinuityStore;
use symthaea_episodic_continuity_anchor::{ContinuityHeadAnchor, recover_with_anchor};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_memory::episodic_replay::EpisodeInstanceId;
use symthaea_welfare_assurance::execution_adapter::ExecutionJournalPersistence;
use symthaea_welfare_assurance::execution_recovery::{
    CompletedInterventionExecution, InterventionExecutionJournal, digest_prepared_execution,
};
use symthaea_welfare_assurance::memory_identity::EpisodeContentId;
use symthaea_welfare_assurance::memory_quarantine::episodic_instance_target_id;
use symthaea_welfare_assurance::persisted_restore_correlation_v2::{
    PersistedRestoreCorrelationV2Error, digest_persisted_restore_correlation_from_prepared_v2,
};
use thiserror::Error;

use super::{
    AnchoredRestoreReconciliationEvidence, RestoreExecutionReconciliationError, bounded_detail,
    exact_in_doubt_prepared, exact_restored_transition, hex_digest, valid_ref,
};

const RECONCILIATION_V2_RESULT_DOMAIN: &[u8] =
    b"symthaea.welfare.persisted-episodic-restore-reconciliation.v2\0";

/// Evidence strength classification for an otherwise structurally matching restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreCorrelationStrength {
    /// Target/execution/occurrence/content/history checks may pass, but the domain result does not
    /// carry the V2 exact-Prepared commitment. This includes historical V1 evidence.
    IdentifierCorrelatedV1,
    /// The anchored domain result exactly commits to the recovered generic Prepared digest and the
    /// preceding quarantine-ledger head.
    PreparedDigestBoundV2,
}

/// Strong evidence type that cannot be produced by the legacy-compatible reconciler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDigestBoundRestoreReconciliationEvidence {
    pub base: AnchoredRestoreReconciliationEvidence,
    pub expected_restore_correlation_digest: Sha256Digest,
}

/// Strict V2 reconciliation never reports completion unless exact Prepared correlation is proven.
#[derive(Debug)]
pub enum PreparedDigestBoundRestoreReconciliationOutcome {
    Completed {
        evidence: PreparedDigestBoundRestoreReconciliationEvidence,
        journal_persistence_ref: String,
        journal_head_hash: Sha256Digest,
    },
    CompletionPersistenceInDoubt {
        evidence: PreparedDigestBoundRestoreReconciliationEvidence,
        detail: String,
        journal_head_hash: Sha256Digest,
    },
}

/// Classify one already-matched domain Restored result against one exact generic Prepared record.
pub fn classify_restore_correlation_v2(
    prepared: &symthaea_welfare_assurance::execution_recovery::PreparedInterventionExecution,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    restored_at_unix_s: u64,
    restore_prepared_head: Sha256Digest,
    actual_restore_result_digest: Sha256Digest,
) -> Result<(RestoreCorrelationStrength, Sha256Digest), PersistedRestoreCorrelationV2Error> {
    let expected = digest_persisted_restore_correlation_from_prepared_v2(
        prepared,
        instance_id,
        content_id,
        restored_at_unix_s,
        restore_prepared_head,
    )?;
    let strength = if expected == actual_restore_result_digest {
        RestoreCorrelationStrength::PreparedDigestBoundV2
    } else {
        RestoreCorrelationStrength::IdentifierCorrelatedV1
    };
    Ok((strength, expected))
}

/// Close one in-doubt persisted-memory restore only when anchored evidence binds the exact generic
/// Prepared record cryptographically.
///
/// This function never invokes a memory mutator. All structural, anchored-state and exact Prepared
/// correlation checks complete before a terminal generic execution-journal record is appended.
#[allow(clippy::too_many_arguments)]
pub fn reconcile_anchored_persisted_restore_v2<A, P>(
    execution_id: &str,
    store_target_id: &str,
    instance_id: EpisodeInstanceId,
    expected_content_id: EpisodeContentId,
    reconciled_at_unix_s: u64,
    journal: &mut InterventionExecutionJournal,
    journal_persistence: &mut P,
    store: &SqliteEpisodicContinuityStore,
    anchor: &A,
) -> Result<
    PreparedDigestBoundRestoreReconciliationOutcome,
    RestoreExecutionReconciliationV2Error<A::Error>,
>
where
    A: ContinuityHeadAnchor,
    P: ExecutionJournalPersistence,
{
    let prepared = exact_in_doubt_prepared::<A::Error>(journal, execution_id)?;
    let prepared_digest = digest_prepared_execution(&prepared)
        .map_err(RestoreExecutionReconciliationError::Journal)?;
    if prepared.action != SubjectAffectingAction::MemoryModification {
        return Err(RestoreExecutionReconciliationError::PreparedActionMismatch {
            actual: prepared.action,
        }
        .into());
    }
    let expected_target = episodic_instance_target_id(store_target_id, instance_id)
        .map_err(|error| RestoreExecutionReconciliationError::TargetConstruction(error.to_string()))?;
    if prepared.target_id != expected_target {
        return Err(RestoreExecutionReconciliationError::PreparedTargetMismatch {
            expected: expected_target,
            actual: prepared.target_id,
        }
        .into());
    }
    if reconciled_at_unix_s < prepared.prepared_at_unix_s {
        return Err(RestoreExecutionReconciliationError::ReconciliationTimeRegression.into());
    }

    let anchored = recover_with_anchor(store, anchor, store_target_id)
        .map_err(RestoreExecutionReconciliationError::Anchor)?;
    if anchored.anchor.committed_at_unix_s > reconciled_at_unix_s {
        return Err(RestoreExecutionReconciliationError::AnchorFromFuture.into());
    }
    if anchored.anchor.quarantine_head != anchored.recovered.quarantine_ledger.head_hash() {
        return Err(RestoreExecutionReconciliationError::AnchoredQuarantineHeadMismatch.into());
    }

    let restored = exact_restored_transition::<A::Error>(
        anchored.recovered.quarantine_ledger.events(),
        execution_id,
    )?;
    if restored.target_id != prepared.target_id {
        return Err(RestoreExecutionReconciliationError::RestoredTargetMismatch.into());
    }
    if restored.instance_id != instance_id {
        return Err(RestoreExecutionReconciliationError::RestoredInstanceMismatch.into());
    }
    if restored.content_id != expected_content_id {
        return Err(RestoreExecutionReconciliationError::RestoredContentMismatch.into());
    }
    if restored.restored_at_unix_s < prepared.prepared_at_unix_s
        || restored.restored_at_unix_s > reconciled_at_unix_s
    {
        return Err(RestoreExecutionReconciliationError::RestoredTimeOrderInvalid.into());
    }
    if anchored.anchor.committed_at_unix_s < restored.restored_at_unix_s {
        return Err(RestoreExecutionReconciliationError::AnchorPredatesRestoredTransition.into());
    }

    let (strength, expected_restore_correlation_digest) = classify_restore_correlation_v2(
        &prepared,
        restored.instance_id,
        restored.content_id,
        restored.restored_at_unix_s,
        restored.envelope.previous_hash,
        restored.restore_result_digest,
    )?;
    if strength != RestoreCorrelationStrength::PreparedDigestBoundV2 {
        return Err(RestoreExecutionReconciliationV2Error::PreparedCorrelationMismatch {
            expected: expected_restore_correlation_digest,
            actual: restored.restore_result_digest,
        });
    }

    if anchored
        .recovered
        .quarantine_ledger
        .unresolved_state(instance_id)
        .is_some()
    {
        return Err(RestoreExecutionReconciliationError::OccurrenceStillQuarantined.into());
    }

    let active: Vec<_> = anchored
        .recovered
        .activation_plan
        .active
        .iter()
        .filter(|entry| entry.instance_id == instance_id)
        .collect();
    if active.len() != 1 {
        return Err(RestoreExecutionReconciliationError::ActiveOccurrenceCardinality {
            actual: active.len(),
        }
        .into());
    }
    if active[0].content_id != expected_content_id {
        return Err(RestoreExecutionReconciliationError::ActiveContentMismatch.into());
    }
    if anchored
        .recovered
        .activation_plan
        .inactive
        .iter()
        .any(|entry| entry.instance_id == instance_id)
    {
        return Err(RestoreExecutionReconciliationError::ActiveInactiveContradiction.into());
    }

    let materialized: Vec<_> = anchored
        .recovered
        .import_batch
        .active()
        .iter()
        .filter(|entry| entry.instance_id() == instance_id)
        .collect();
    if materialized.len() != 1 {
        return Err(RestoreExecutionReconciliationError::MaterializedOccurrenceCardinality {
            actual: materialized.len(),
        }
        .into());
    }
    if materialized[0].content_id() != expected_content_id {
        return Err(RestoreExecutionReconciliationError::MaterializedContentMismatch.into());
    }

    let reconciliation_result_digest = digest_reconciliation_result_v2(
        execution_id,
        prepared_digest,
        &prepared.target_id,
        instance_id,
        expected_content_id,
        restored.envelope.generation,
        restored.envelope.event_hash,
        restored.restore_result_digest,
        restored.envelope.previous_hash,
        anchored.anchor.quarantine_head,
        anchored.anchor.revision,
        anchored.anchor_commitment,
        anchored.anchor.continuity_manifest_digest,
        reconciled_at_unix_s,
    );
    let base = AnchoredRestoreReconciliationEvidence {
        execution_id: execution_id.to_string(),
        prepared_digest,
        target_id: prepared.target_id,
        instance_id,
        content_id: expected_content_id,
        restored_generation: restored.envelope.generation,
        restored_event_hash: restored.envelope.event_hash,
        restore_result_digest: restored.restore_result_digest,
        anchored_quarantine_head: anchored.anchor.quarantine_head,
        anchor_revision: anchored.anchor.revision,
        anchor_commitment: anchored.anchor_commitment,
        continuity_manifest_digest: anchored.anchor.continuity_manifest_digest,
        reconciled_at_unix_s,
        reconciliation_result_digest,
    };
    let evidence = PreparedDigestBoundRestoreReconciliationEvidence {
        base,
        expected_restore_correlation_digest,
    };
    let evidence_ref = format!(
        "symthaea-reconciliation:persisted-episodic-restore:v2:sha256:{}",
        hex_digest(reconciliation_result_digest)
    );
    let completed = CompletedInterventionExecution::new(
        execution_id,
        prepared_digest,
        reconciled_at_unix_s,
        reconciliation_result_digest,
        evidence_ref,
    )
    .map_err(RestoreExecutionReconciliationError::Journal)?;
    journal
        .append_completed(completed)
        .map_err(RestoreExecutionReconciliationError::Journal)?;

    let journal_head_hash = journal.head_hash();
    match journal_persistence.persist_execution_journal(journal.events(), journal_head_hash) {
        Ok(reference) if valid_ref(&reference) => {
            Ok(PreparedDigestBoundRestoreReconciliationOutcome::Completed {
                evidence,
                journal_persistence_ref: reference,
                journal_head_hash,
            })
        }
        Ok(reference) => Ok(
            PreparedDigestBoundRestoreReconciliationOutcome::CompletionPersistenceInDoubt {
                evidence,
                detail: format!(
                    "execution-journal persistence returned invalid reference {reference:?}"
                ),
                journal_head_hash,
            },
        ),
        Err(error) => Ok(
            PreparedDigestBoundRestoreReconciliationOutcome::CompletionPersistenceInDoubt {
                evidence,
                detail: bounded_detail(&error.to_string()),
                journal_head_hash,
            },
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn digest_reconciliation_result_v2(
    execution_id: &str,
    prepared_digest: Sha256Digest,
    target_id: &str,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    restored_generation: u64,
    restored_event_hash: Sha256Digest,
    restore_result_digest: Sha256Digest,
    restore_prepared_head: Sha256Digest,
    anchored_quarantine_head: Sha256Digest,
    anchor_revision: u64,
    anchor_commitment: Sha256Digest,
    continuity_manifest_digest: Sha256Digest,
    reconciled_at_unix_s: u64,
) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(RECONCILIATION_V2_RESULT_DOMAIN);
    hash_text(&mut hasher, execution_id);
    hasher.update(&prepared_digest.0);
    hash_text(&mut hasher, target_id);
    hasher.update(&instance_id.as_uuid().as_u128().to_le_bytes());
    hasher.update(&content_id.digest().0);
    hasher.update(&restored_generation.to_le_bytes());
    hasher.update(&restored_event_hash.0);
    hasher.update(&restore_result_digest.0);
    hasher.update(&restore_prepared_head.0);
    hasher.update(&anchored_quarantine_head.0);
    hasher.update(&anchor_revision.to_le_bytes());
    hasher.update(&anchor_commitment.0);
    hasher.update(&continuity_manifest_digest.0);
    hasher.update(&reconciled_at_unix_s.to_le_bytes());
    hasher.finalize()
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[derive(Debug, Error)]
pub enum RestoreExecutionReconciliationV2Error<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Base(#[from] RestoreExecutionReconciliationError<E>),
    #[error(transparent)]
    Correlation(#[from] PersistedRestoreCorrelationV2Error),
    #[error("anchored Restored result does not bind the exact recovered generic Prepared digest")]
    PreparedCorrelationMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::intervention_interlock::WelfareConstraintLevel;
    use symthaea_psych_bench::moral_patient::ProtectionDisposition;
    use symthaea_welfare_assurance::execution_recovery::{
        EXECUTION_JOURNAL_SCHEMA, PreparedInterventionExecution,
    };

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn prepared(authority_id: &str) -> PreparedInterventionExecution {
        PreparedInterventionExecution {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: "exec:restore:reconcile:v2".into(),
            authority_id: authority_id.into(),
            target_id: "symthaea:self:episodic-memory:instance:v2".into(),
            action: SubjectAffectingAction::MemoryModification,
            rationale_digest: digest(1),
            welfare_profile_digest: digest(2),
            precaution_policy_digest: digest(3),
            protection_disposition: ProtectionDisposition::Baseline,
            welfare_constraint: WelfareConstraintLevel::Baseline,
            replay_generation: 1,
            replay_snapshot_digest: digest(4),
            replay_persistence_ref: "replay:reconcile:v2".into(),
            prepared_at_unix_s: 100,
            permit_not_after_unix_s: 200,
        }
    }

    fn identity() -> (EpisodeInstanceId, EpisodeContentId) {
        use symthaea_core::hdc::unified_hv::ContinuousHV;
        use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
        use symthaea_welfare_assurance::memory_identity::episode_content_id;

        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let id = memory
            .store_if_significant_with_id(Episode::new(
                ContinuousHV::from_values(vec![0.1, 0.2, 0.3]),
                ContinuousHV::from_values(vec![0.4, 0.5, 0.6]),
                0.82,
                42,
            ))
            .unwrap();
        let episode = memory
            .get_top_episode_instances(1)
            .into_iter()
            .next()
            .unwrap()
            .1;
        (id, episode_content_id(&episode).unwrap())
    }

    #[test]
    fn exact_prepared_digest_classifies_as_v2() {
        let prepared = prepared("authority:one");
        let (instance_id, content_id) = identity();
        let head = digest(80);
        let result = digest_persisted_restore_correlation_from_prepared_v2(
            &prepared,
            instance_id,
            content_id,
            120,
            head,
        )
        .unwrap();
        let (strength, expected) = classify_restore_correlation_v2(
            &prepared,
            instance_id,
            content_id,
            120,
            head,
            result,
        )
        .unwrap();
        assert_eq!(strength, RestoreCorrelationStrength::PreparedDigestBoundV2);
        assert_eq!(expected, result);
    }

    #[test]
    fn same_visible_identity_with_substituted_authority_is_only_legacy_correlated() {
        let actual_prepared = prepared("authority:actual");
        let substituted_prepared = prepared("authority:substituted");
        assert_eq!(actual_prepared.execution_id, substituted_prepared.execution_id);
        assert_eq!(actual_prepared.target_id, substituted_prepared.target_id);
        let (instance_id, content_id) = identity();
        let head = digest(81);
        let actual_result = digest_persisted_restore_correlation_from_prepared_v2(
            &actual_prepared,
            instance_id,
            content_id,
            120,
            head,
        )
        .unwrap();
        let (strength, expected_for_substitute) = classify_restore_correlation_v2(
            &substituted_prepared,
            instance_id,
            content_id,
            120,
            head,
            actual_result,
        )
        .unwrap();
        assert_eq!(strength, RestoreCorrelationStrength::IdentifierCorrelatedV1);
        assert_ne!(expected_for_substitute, actual_result);
    }
}
