// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict post-crash reconciliation for Prepared-event-bound persisted restores.
//!
//! Historical V1 reconciliation remains available through the parent module. This module emits a
//! stronger typed outcome only when the anchored domain `Restored.restore_result_digest` exactly
//! equals an independently recomputed commitment to the recovered generic `Prepared` payload,
//! its exact validated generic journal envelope, and the exact matching domain `RestorePrepared`
//! event. Merely matching execution ID, target, occurrence UUID and content is insufficient.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_episodic_continuity::SqliteEpisodicContinuityStore;
use symthaea_episodic_continuity_anchor::{ContinuityHeadAnchor, recover_with_anchor};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_memory::episodic_replay::EpisodeInstanceId;
use symthaea_welfare_assurance::execution_adapter::ExecutionJournalPersistence;
use symthaea_welfare_assurance::execution_recovery::{
    CompletedInterventionExecution, ExecutionJournalEnvelope, ExecutionJournalEvent,
    InterventionExecutionJournal, PreparedInterventionExecution, digest_prepared_execution,
};
use symthaea_welfare_assurance::memory_identity::EpisodeContentId;
use symthaea_welfare_assurance::memory_quarantine::episodic_instance_target_id;
use symthaea_welfare_assurance::persisted_restore_correlation_v2::{
    PersistedRestoreCorrelationV2Error, digest_persisted_restore_correlation_from_prepared_v2,
};
use symthaea_welfare_assurance::quarantine_state_ledger::{
    QuarantineLedgerEnvelope, QuarantineLedgerEventKind,
};
use thiserror::Error;

use super::{
    AnchoredRestoreReconciliationEvidence, RestoreExecutionReconciliationError, bounded_detail,
    exact_in_doubt_prepared, exact_restored_transition, hex_digest, valid_ref,
};

// V3 because the terminal reconciliation digest now binds the generic Prepared envelope identity.
const RECONCILIATION_V3_RESULT_DOMAIN: &[u8] =
    b"symthaea.welfare.persisted-episodic-restore-reconciliation.v3\0";

/// Evidence strength classification for an otherwise structurally matching restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreCorrelationStrength {
    /// Structural identity may match, but exact Prepared-event correlation is not established.
    /// This may represent historical evidence, another correlation scheme, substitution or damage;
    /// callers must not infer which merely from a digest mismatch.
    NotPreparedDigestBound,
    /// The anchored domain result exactly commits to the recovered generic Prepared payload digest,
    /// its exact generic journal-event hash, and the exact matching domain RestorePrepared hash.
    PreparedDigestBoundV2,
}

/// Strong evidence type that cannot be produced by the legacy-compatible reconciler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDigestBoundRestoreReconciliationEvidence {
    pub base: AnchoredRestoreReconciliationEvidence,
    pub generic_prepared_sequence: u64,
    pub generic_prepared_event_hash: Sha256Digest,
    pub restore_prepared_generation: u64,
    pub restore_prepared_event_hash: Sha256Digest,
    pub expected_restore_correlation_digest: Sha256Digest,
}

/// Strict reconciliation never reports completion unless exact Prepared-event correlation is proven.
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

struct GenericPreparedTransition<'a> {
    envelope: &'a ExecutionJournalEnvelope,
    prepared: &'a PreparedInterventionExecution,
}

struct RestorePreparedTransition<'a> {
    envelope: &'a QuarantineLedgerEnvelope,
    prepared_at_unix_s: u64,
}

/// Classify one already-matched domain Restored result against one exact generic Prepared record,
/// its exact generic journal-event hash, and one exact domain RestorePrepared event hash.
pub fn classify_restore_correlation_v2(
    prepared: &PreparedInterventionExecution,
    generic_prepared_event_hash: Sha256Digest,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    restored_at_unix_s: u64,
    restore_prepared_event_hash: Sha256Digest,
    actual_restore_result_digest: Sha256Digest,
) -> Result<(RestoreCorrelationStrength, Sha256Digest), PersistedRestoreCorrelationV2Error> {
    let expected = digest_persisted_restore_correlation_from_prepared_v2(
        prepared,
        generic_prepared_event_hash,
        instance_id,
        content_id,
        restored_at_unix_s,
        restore_prepared_event_hash,
    )?;
    let strength = if expected == actual_restore_result_digest {
        RestoreCorrelationStrength::PreparedDigestBoundV2
    } else {
        RestoreCorrelationStrength::NotPreparedDigestBound
    };
    Ok((strength, expected))
}

/// Close one in-doubt persisted-memory restore only when anchored evidence binds the exact generic
/// Prepared journal event cryptographically.
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
    let generic_prepared = exact_generic_prepared_transition::<A::Error>(journal.events(), execution_id)?;
    if generic_prepared.prepared != &prepared {
        return Err(RestoreExecutionReconciliationV2Error::GenericPreparedStateMismatch);
    }
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

    let events = anchored.recovered.quarantine_ledger.events();
    let restored = exact_restored_transition::<A::Error>(events, execution_id)?;
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

    let restore_prepared = exact_restore_prepared_transition(
        events,
        execution_id,
        restored.target_id,
        restored.instance_id,
        restored.content_id,
        restored.envelope.generation,
    )?;
    if restore_prepared.prepared_at_unix_s < prepared.prepared_at_unix_s
        || restore_prepared.prepared_at_unix_s > restored.restored_at_unix_s
    {
        return Err(RestoreExecutionReconciliationV2Error::RestorePreparedTimeOrderInvalid {
            generic_prepared_at_unix_s: prepared.prepared_at_unix_s,
            domain_prepared_at_unix_s: restore_prepared.prepared_at_unix_s,
            restored_at_unix_s: restored.restored_at_unix_s,
        });
    }

    let (strength, expected_restore_correlation_digest) = classify_restore_correlation_v2(
        &prepared,
        generic_prepared.envelope.event_hash,
        restored.instance_id,
        restored.content_id,
        restored.restored_at_unix_s,
        restore_prepared.envelope.event_hash,
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

    let reconciliation_result_digest = digest_reconciliation_result_v3(
        execution_id,
        prepared_digest,
        generic_prepared.envelope.sequence,
        generic_prepared.envelope.event_hash,
        &prepared.target_id,
        instance_id,
        expected_content_id,
        restore_prepared.envelope.generation,
        restore_prepared.envelope.event_hash,
        restored.envelope.generation,
        restored.envelope.event_hash,
        restored.restore_result_digest,
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
        generic_prepared_sequence: generic_prepared.envelope.sequence,
        generic_prepared_event_hash: generic_prepared.envelope.event_hash,
        restore_prepared_generation: restore_prepared.envelope.generation,
        restore_prepared_event_hash: restore_prepared.envelope.event_hash,
        expected_restore_correlation_digest,
    };
    let evidence_ref = format!(
        "symthaea-reconciliation:persisted-episodic-restore:v3:sha256:{}",
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

fn exact_generic_prepared_transition<'a, E>(
    events: &'a [ExecutionJournalEnvelope],
    execution_id: &str,
) -> Result<GenericPreparedTransition<'a>, RestoreExecutionReconciliationV2Error<E>>
where
    E: StdError + Send + Sync + 'static,
{
    let matches: Vec<_> = events
        .iter()
        .filter_map(|envelope| match &envelope.event {
            ExecutionJournalEvent::Prepared(prepared) if prepared.execution_id == execution_id => {
                Some(GenericPreparedTransition { envelope, prepared })
            }
            _ => None,
        })
        .collect();
    match matches.len() {
        1 => Ok(matches.into_iter().next().expect("length checked")),
        0 => Err(RestoreExecutionReconciliationV2Error::MissingGenericPreparedEnvelope),
        actual => Err(RestoreExecutionReconciliationV2Error::AmbiguousGenericPreparedEnvelope {
            actual,
        }),
    }
}

fn exact_restore_prepared_transition<'a, E>(
    events: &'a [QuarantineLedgerEnvelope],
    execution_id: &str,
    target_id: &str,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    restored_generation: u64,
) -> Result<RestorePreparedTransition<'a>, RestoreExecutionReconciliationV2Error<E>>
where
    E: StdError + Send + Sync + 'static,
{
    let matches: Vec<_> = events
        .iter()
        .filter_map(|envelope| match &envelope.event {
            QuarantineLedgerEventKind::RestorePrepared {
                target_id: candidate_target_id,
                instance_id: candidate_instance_id,
                content_id: candidate_content_id,
                prepared_at_unix_s,
                execution_id: candidate_execution_id,
            } if envelope.generation < restored_generation
                && candidate_execution_id == execution_id
                && candidate_target_id == target_id
                && *candidate_instance_id == instance_id
                && *candidate_content_id == content_id => Some(RestorePreparedTransition {
                    envelope,
                    prepared_at_unix_s: *prepared_at_unix_s,
                }),
            _ => None,
        })
        .collect();
    match matches.len() {
        1 => Ok(matches.into_iter().next().expect("length checked")),
        0 => Err(RestoreExecutionReconciliationV2Error::MissingRestorePreparedEvidence),
        actual => Err(RestoreExecutionReconciliationV2Error::AmbiguousRestorePreparedEvidence {
            actual,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn digest_reconciliation_result_v3(
    execution_id: &str,
    prepared_digest: Sha256Digest,
    generic_prepared_sequence: u64,
    generic_prepared_event_hash: Sha256Digest,
    target_id: &str,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    restore_prepared_generation: u64,
    restore_prepared_event_hash: Sha256Digest,
    restored_generation: u64,
    restored_event_hash: Sha256Digest,
    restore_result_digest: Sha256Digest,
    anchored_quarantine_head: Sha256Digest,
    anchor_revision: u64,
    anchor_commitment: Sha256Digest,
    continuity_manifest_digest: Sha256Digest,
    reconciled_at_unix_s: u64,
) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(RECONCILIATION_V3_RESULT_DOMAIN);
    hash_text(&mut hasher, execution_id);
    hasher.update(&prepared_digest.0);
    hasher.update(&generic_prepared_sequence.to_le_bytes());
    hasher.update(&generic_prepared_event_hash.0);
    hash_text(&mut hasher, target_id);
    hasher.update(&instance_id.as_uuid().as_u128().to_le_bytes());
    hasher.update(&content_id.digest().0);
    hasher.update(&restore_prepared_generation.to_le_bytes());
    hasher.update(&restore_prepared_event_hash.0);
    hasher.update(&restored_generation.to_le_bytes());
    hasher.update(&restored_event_hash.0);
    hasher.update(&restore_result_digest.0);
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
    #[error("validated execution journal has no Prepared envelope for the requested in-doubt execution")]
    MissingGenericPreparedEnvelope,
    #[error("validated execution journal has {actual} Prepared envelopes for the requested execution")]
    AmbiguousGenericPreparedEnvelope { actual: usize },
    #[error("generic Prepared recovery state disagrees with its validated journal envelope")]
    GenericPreparedStateMismatch,
    #[error("anchored quarantine ledger has no exact RestorePrepared event for the V2 restore")]
    MissingRestorePreparedEvidence,
    #[error("anchored quarantine ledger has {actual} exact RestorePrepared events for the V2 restore")]
    AmbiguousRestorePreparedEvidence { actual: usize },
    #[error(
        "domain RestorePrepared time is inconsistent with generic Prepared and Restored: generic={generic_prepared_at_unix_s}, domain={domain_prepared_at_unix_s}, restored={restored_at_unix_s}"
    )]
    RestorePreparedTimeOrderInvalid {
        generic_prepared_at_unix_s: u64,
        domain_prepared_at_unix_s: u64,
        restored_at_unix_s: u64,
    },
    #[error("anchored Restored result does not bind the exact recovered generic Prepared event")]
    PreparedCorrelationMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_core::intervention_interlock::WelfareConstraintLevel;
    use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
    use symthaea_psych_bench::moral_patient::ProtectionDisposition;
    use symthaea_welfare_assurance::execution_recovery::{
        EXECUTION_JOURNAL_SCHEMA, InterventionExecutionJournal,
    };
    use symthaea_welfare_assurance::memory_identity::episode_content_id;
    use symthaea_welfare_assurance::quarantine_state_ledger::EpisodicQuarantineStateLedger;

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

    fn prepared_event_hash(prepared: &PreparedInterventionExecution) -> Sha256Digest {
        let mut journal = InterventionExecutionJournal::new();
        journal.append_prepared(prepared.clone()).unwrap();
        journal.head_hash()
    }

    #[test]
    fn exact_prepared_event_classifies_as_v2() {
        let prepared = prepared("authority:one");
        let generic_event_hash = prepared_event_hash(&prepared);
        let (instance_id, content_id) = identity();
        let domain_event_hash = digest(80);
        let result = digest_persisted_restore_correlation_from_prepared_v2(
            &prepared,
            generic_event_hash,
            instance_id,
            content_id,
            120,
            domain_event_hash,
        )
        .unwrap();
        let (strength, expected) = classify_restore_correlation_v2(
            &prepared,
            generic_event_hash,
            instance_id,
            content_id,
            120,
            domain_event_hash,
            result,
        )
        .unwrap();
        assert_eq!(strength, RestoreCorrelationStrength::PreparedDigestBoundV2);
        assert_eq!(expected, result);
    }

    #[test]
    fn same_visible_identity_with_substituted_authority_is_not_prepared_digest_bound() {
        let actual_prepared = prepared("authority:actual");
        let substituted_prepared = prepared("authority:substituted");
        assert_eq!(actual_prepared.execution_id, substituted_prepared.execution_id);
        assert_eq!(actual_prepared.target_id, substituted_prepared.target_id);
        let actual_event_hash = prepared_event_hash(&actual_prepared);
        let substituted_event_hash = prepared_event_hash(&substituted_prepared);
        let (instance_id, content_id) = identity();
        let domain_event_hash = digest(81);
        let actual_result = digest_persisted_restore_correlation_from_prepared_v2(
            &actual_prepared,
            actual_event_hash,
            instance_id,
            content_id,
            120,
            domain_event_hash,
        )
        .unwrap();
        let (strength, expected_for_substitute) = classify_restore_correlation_v2(
            &substituted_prepared,
            substituted_event_hash,
            instance_id,
            content_id,
            120,
            domain_event_hash,
            actual_result,
        )
        .unwrap();
        assert_eq!(strength, RestoreCorrelationStrength::NotPreparedDigestBound);
        assert_ne!(expected_for_substitute, actual_result);
    }

    #[test]
    fn same_payload_at_different_generic_journal_position_is_not_prepared_digest_bound() {
        let prepared = prepared("authority:journal-position");
        let mut first = InterventionExecutionJournal::new();
        first.append_prepared(prepared.clone()).unwrap();
        let first_hash = first.head_hash();

        let mut other = prepared("authority:other-prefix");
        other.execution_id = "exec:other-prefix".into();
        let mut second = InterventionExecutionJournal::new();
        second.append_prepared(other).unwrap();
        second.append_prepared(prepared.clone()).unwrap();
        let second_hash = second.head_hash();
        assert_ne!(first_hash, second_hash);

        let (instance_id, content_id) = identity();
        let domain_event_hash = digest(82);
        let actual = digest_persisted_restore_correlation_from_prepared_v2(
            &prepared,
            first_hash,
            instance_id,
            content_id,
            120,
            domain_event_hash,
        )
        .unwrap();
        let (strength, expected) = classify_restore_correlation_v2(
            &prepared,
            second_hash,
            instance_id,
            content_id,
            120,
            domain_event_hash,
            actual,
        )
        .unwrap();
        assert_eq!(strength, RestoreCorrelationStrength::NotPreparedDigestBound);
        assert_ne!(expected, actual);
    }

    #[test]
    fn exact_restore_prepared_event_is_selected_even_with_interleaved_ledger_event() {
        let (instance_id, content_id) = identity();
        let target = "symthaea:self:episodic-memory:instance:v2";
        let mut ledger = EpisodicQuarantineStateLedger::new();
        ledger
            .append_quarantined(target, instance_id, content_id, 90, digest(10), "escrow:v2")
            .unwrap();
        let expected_hash = ledger
            .append_restore_prepared(target, instance_id, content_id, 110, "exec:restore:reconcile:v2")
            .unwrap();

        let (other_instance, other_content) = identity();
        ledger
            .append_quarantined(
                "symthaea:self:episodic-memory:instance:other",
                other_instance,
                other_content,
                111,
                digest(11),
                "escrow:other",
            )
            .unwrap();
        ledger
            .append_restored(
                target,
                instance_id,
                content_id,
                120,
                "exec:restore:reconcile:v2",
                digest(12),
            )
            .unwrap();

        let restored_generation = ledger.events().last().unwrap().generation;
        let matched = exact_restore_prepared_transition::<std::io::Error>(
            ledger.events(),
            "exec:restore:reconcile:v2",
            target,
            instance_id,
            content_id,
            restored_generation,
        )
        .unwrap();
        assert_eq!(matched.envelope.event_hash, expected_hash);
        assert_ne!(ledger.events().last().unwrap().previous_hash, expected_hash);
    }
}
