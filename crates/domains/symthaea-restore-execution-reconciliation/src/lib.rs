// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Post-crash reconciliation for one exact persisted episodic restore.
//!
//! This crate never re-executes the restore. It may append terminal execution-journal evidence only
//! when independently anchored durable continuity proves that the exact in-doubt intervention has
//! already completed for the exact target, occurrence UUID and content lineage.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_core::welfare::SubjectAffectingAction;
use symthaea_episodic_continuity::SqliteEpisodicContinuityStore;
use symthaea_episodic_continuity_anchor::{
    AnchorProtocolError, ContinuityHeadAnchor, recover_with_anchor,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_memory::episodic_replay::EpisodeInstanceId;
use symthaea_welfare_assurance::execution_adapter::ExecutionJournalPersistence;
use symthaea_welfare_assurance::execution_recovery::{
    AutomaticRetryDecision, CompletedInterventionExecution, ExecutionJournalError,
    InterventionExecutionJournal, PreparedInterventionExecution, digest_prepared_execution,
};
use symthaea_welfare_assurance::memory_identity::EpisodeContentId;
use symthaea_welfare_assurance::memory_quarantine::episodic_instance_target_id;
use symthaea_welfare_assurance::quarantine_state_ledger::{
    QuarantineLedgerEnvelope, QuarantineLedgerEventKind,
};
use thiserror::Error;

const RECONCILIATION_RESULT_DOMAIN: &[u8] =
    b"symthaea.welfare.persisted-episodic-restore-reconciliation.v1\0";
const MAX_REF_BYTES: usize = 2048;

/// Independently reconstructible evidence used to close one previously in-doubt restore.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchoredRestoreReconciliationEvidence {
    pub execution_id: String,
    pub prepared_digest: Sha256Digest,
    pub target_id: String,
    pub instance_id: EpisodeInstanceId,
    pub content_id: EpisodeContentId,
    pub restored_generation: u64,
    pub restored_event_hash: Sha256Digest,
    pub restore_result_digest: Sha256Digest,
    pub anchored_quarantine_head: Sha256Digest,
    pub anchor_revision: u64,
    pub anchor_commitment: Sha256Digest,
    pub continuity_manifest_digest: Sha256Digest,
    pub reconciled_at_unix_s: u64,
    pub reconciliation_result_digest: Sha256Digest,
}

/// Reconciliation never reports a persistence failure as permission to retry the intervention.
#[derive(Debug)]
pub enum AnchoredRestoreReconciliationOutcome {
    Completed {
        evidence: AnchoredRestoreReconciliationEvidence,
        journal_persistence_ref: String,
        journal_head_hash: Sha256Digest,
    },
    CompletionPersistenceInDoubt {
        evidence: AnchoredRestoreReconciliationEvidence,
        detail: String,
        journal_head_hash: Sha256Digest,
    },
}

struct RestoredTransition<'a> {
    envelope: &'a QuarantineLedgerEnvelope,
    target_id: &'a str,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    restored_at_unix_s: u64,
    restore_result_digest: Sha256Digest,
}

/// Close one in-doubt persisted-memory restore from externally anchored durable evidence.
///
/// This function never invokes a memory mutator. A successful reconciliation appends a normal
/// terminal `CompletedInterventionExecution`, but its evidence reference is namespaced as
/// post-crash reconciliation rather than direct executor evidence.
#[allow(clippy::too_many_arguments)]
pub fn reconcile_anchored_persisted_restore<A, P>(
    execution_id: &str,
    store_target_id: &str,
    instance_id: EpisodeInstanceId,
    expected_content_id: EpisodeContentId,
    reconciled_at_unix_s: u64,
    journal: &mut InterventionExecutionJournal,
    journal_persistence: &mut P,
    store: &SqliteEpisodicContinuityStore,
    anchor: &A,
) -> Result<AnchoredRestoreReconciliationOutcome, RestoreExecutionReconciliationError<A::Error>>
where
    A: ContinuityHeadAnchor,
    P: ExecutionJournalPersistence,
{
    let prepared = exact_in_doubt_prepared(journal, execution_id)?;
    let prepared_digest = digest_prepared_execution(&prepared)
        .map_err(RestoreExecutionReconciliationError::Journal)?;
    if prepared.action != SubjectAffectingAction::MemoryModification {
        return Err(RestoreExecutionReconciliationError::PreparedActionMismatch {
            actual: prepared.action,
        });
    }
    let expected_target = episodic_instance_target_id(store_target_id, instance_id)
        .map_err(|error| RestoreExecutionReconciliationError::TargetConstruction(error.to_string()))?;
    if prepared.target_id != expected_target {
        return Err(RestoreExecutionReconciliationError::PreparedTargetMismatch {
            expected: expected_target,
            actual: prepared.target_id,
        });
    }
    if reconciled_at_unix_s < prepared.prepared_at_unix_s {
        return Err(RestoreExecutionReconciliationError::ReconciliationTimeRegression);
    }

    let anchored = recover_with_anchor(store, anchor, store_target_id)
        .map_err(RestoreExecutionReconciliationError::Anchor)?;
    if anchored.anchor.committed_at_unix_s > reconciled_at_unix_s {
        return Err(RestoreExecutionReconciliationError::AnchorFromFuture);
    }
    if anchored.anchor.quarantine_head != anchored.recovered.quarantine_ledger.head_hash() {
        return Err(RestoreExecutionReconciliationError::AnchoredQuarantineHeadMismatch);
    }

    let restored = exact_restored_transition(
        anchored.recovered.quarantine_ledger.events(),
        execution_id,
    )?;
    if restored.target_id != prepared.target_id {
        return Err(RestoreExecutionReconciliationError::RestoredTargetMismatch);
    }
    if restored.instance_id != instance_id {
        return Err(RestoreExecutionReconciliationError::RestoredInstanceMismatch);
    }
    if restored.content_id != expected_content_id {
        return Err(RestoreExecutionReconciliationError::RestoredContentMismatch);
    }
    if restored.restored_at_unix_s < prepared.prepared_at_unix_s
        || restored.restored_at_unix_s > reconciled_at_unix_s
    {
        return Err(RestoreExecutionReconciliationError::RestoredTimeOrderInvalid);
    }
    if anchored.anchor.committed_at_unix_s < restored.restored_at_unix_s {
        return Err(RestoreExecutionReconciliationError::AnchorPredatesRestoredTransition);
    }
    if anchored
        .recovered
        .quarantine_ledger
        .unresolved_state(instance_id)
        .is_some()
    {
        return Err(RestoreExecutionReconciliationError::OccurrenceStillQuarantined);
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
        });
    }
    if active[0].content_id != expected_content_id {
        return Err(RestoreExecutionReconciliationError::ActiveContentMismatch);
    }
    if anchored
        .recovered
        .activation_plan
        .inactive
        .iter()
        .any(|entry| entry.instance_id == instance_id)
    {
        return Err(RestoreExecutionReconciliationError::ActiveInactiveContradiction);
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
        });
    }
    if materialized[0].content_id() != expected_content_id {
        return Err(RestoreExecutionReconciliationError::MaterializedContentMismatch);
    }

    let reconciliation_result_digest = digest_reconciliation_result(
        execution_id,
        prepared_digest,
        &prepared.target_id,
        instance_id,
        expected_content_id,
        restored.envelope.generation,
        restored.envelope.event_hash,
        restored.restore_result_digest,
        anchored.anchor.quarantine_head,
        anchored.anchor.revision,
        anchored.anchor_commitment,
        anchored.anchor.continuity_manifest_digest,
        reconciled_at_unix_s,
    );
    let evidence = AnchoredRestoreReconciliationEvidence {
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
    let evidence_ref = format!(
        "symthaea-reconciliation:persisted-episodic-restore:v1:sha256:{}",
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
        Ok(reference) if valid_ref(&reference) => Ok(AnchoredRestoreReconciliationOutcome::Completed {
            evidence,
            journal_persistence_ref: reference,
            journal_head_hash,
        }),
        Ok(reference) => Ok(
            AnchoredRestoreReconciliationOutcome::CompletionPersistenceInDoubt {
                evidence,
                detail: format!("execution-journal persistence returned invalid reference {reference:?}"),
                journal_head_hash,
            },
        ),
        Err(error) => Ok(
            AnchoredRestoreReconciliationOutcome::CompletionPersistenceInDoubt {
                evidence,
                detail: bounded_detail(&error.to_string()),
                journal_head_hash,
            },
        ),
    }
}

fn exact_in_doubt_prepared(
    journal: &InterventionExecutionJournal,
    execution_id: &str,
) -> Result<PreparedInterventionExecution, RestoreExecutionReconciliationError<std::io::Error>> {
    let matches: Vec<_> = journal
        .recovery_report()
        .in_doubt
        .into_iter()
        .filter(|prepared| prepared.execution_id == execution_id)
        .collect();
    match matches.len() {
        1 => Ok(matches.into_iter().next().expect("length checked")),
        0 => match journal.automatic_retry_decision(execution_id) {
            AutomaticRetryDecision::RefuseAlreadyCompleted => {
                Err(RestoreExecutionReconciliationError::ExecutionAlreadyCompleted)
            }
            _ => Err(RestoreExecutionReconciliationError::MissingInDoubtExecution),
        },
        actual => Err(RestoreExecutionReconciliationError::AmbiguousInDoubtExecution { actual }),
    }
}

fn exact_restored_transition<'a, E>(
    events: &'a [QuarantineLedgerEnvelope],
    execution_id: &str,
) -> Result<RestoredTransition<'a>, RestoreExecutionReconciliationError<E>>
where
    E: StdError + Send + Sync + 'static,
{
    let matches: Vec<_> = events
        .iter()
        .filter_map(|envelope| match &envelope.event {
            QuarantineLedgerEventKind::Restored {
                target_id,
                instance_id,
                content_id,
                restored_at_unix_s,
                execution_id: candidate_execution_id,
                restore_result_digest,
            } if candidate_execution_id == execution_id => Some(RestoredTransition {
                envelope,
                target_id,
                instance_id: *instance_id,
                content_id: *content_id,
                restored_at_unix_s: *restored_at_unix_s,
                restore_result_digest: *restore_result_digest,
            }),
            _ => None,
        })
        .collect();
    match matches.len() {
        1 => Ok(matches.into_iter().next().expect("length checked")),
        0 => Err(RestoreExecutionReconciliationError::MissingRestoredEvidence),
        actual => Err(RestoreExecutionReconciliationError::AmbiguousRestoredEvidence { actual }),
    }
}

#[allow(clippy::too_many_arguments)]
fn digest_reconciliation_result(
    execution_id: &str,
    prepared_digest: Sha256Digest,
    target_id: &str,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
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
    hasher.update(RECONCILIATION_RESULT_DOMAIN);
    hash_text(&mut hasher, execution_id);
    hasher.update(&prepared_digest.0);
    hash_text(&mut hasher, target_id);
    hasher.update(&instance_id.as_uuid().as_u128().to_le_bytes());
    hasher.update(&content_id.digest().0);
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

fn valid_ref(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= MAX_REF_BYTES
        && !value.chars().any(char::is_control)
}

fn bounded_detail(value: &str) -> String {
    if value.len() <= 2048 {
        return value.to_string();
    }
    let mut end = 2048;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
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
pub enum RestoreExecutionReconciliationError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("anchored continuity recovery failed: {0}")]
    Anchor(#[source] AnchorProtocolError<E>),
    #[error("execution journal reconciliation failed: {0}")]
    Journal(#[source] ExecutionJournalError),
    #[error("could not construct exact persisted restore target: {0}")]
    TargetConstruction(String),
    #[error("execution is already terminally completed")]
    ExecutionAlreadyCompleted,
    #[error("execution journal does not contain the requested in-doubt Prepared record")]
    MissingInDoubtExecution,
    #[error("execution journal contains {actual} matching in-doubt Prepared records")]
    AmbiguousInDoubtExecution { actual: usize },
    #[error("in-doubt Prepared action is not MemoryModification: {actual:?}")]
    PreparedActionMismatch { actual: SubjectAffectingAction },
    #[error("in-doubt Prepared target mismatch: expected={expected:?}, actual={actual:?}")]
    PreparedTargetMismatch { expected: String, actual: String },
    #[error("reconciliation time predates execution preparation")]
    ReconciliationTimeRegression,
    #[error("external anchor timestamp is in the future relative to reconciliation")]
    AnchorFromFuture,
    #[error("anchored quarantine head disagrees with recovered quarantine ledger")]
    AnchoredQuarantineHeadMismatch,
    #[error("anchored quarantine ledger has no Restored event for the execution")]
    MissingRestoredEvidence,
    #[error("anchored quarantine ledger has {actual} Restored events for the execution")]
    AmbiguousRestoredEvidence { actual: usize },
    #[error("Restored target disagrees with the prepared execution")]
    RestoredTargetMismatch,
    #[error("Restored occurrence UUID disagrees with reconciliation target")]
    RestoredInstanceMismatch,
    #[error("Restored content identity disagrees with reconciliation target")]
    RestoredContentMismatch,
    #[error("Restored transition timestamp is inconsistent with prepare/reconciliation time")]
    RestoredTimeOrderInvalid,
    #[error("external anchor predates the Restored transition")]
    AnchorPredatesRestoredTransition,
    #[error("occurrence remains unresolved/quarantined in anchored recovery")]
    OccurrenceStillQuarantined,
    #[error("anchored activation plan contains {actual} active entries for the occurrence")]
    ActiveOccurrenceCardinality { actual: usize },
    #[error("anchored active occurrence has the wrong content identity")]
    ActiveContentMismatch,
    #[error("anchored activation plan classifies the occurrence both active and inactive")]
    ActiveInactiveContradiction,
    #[error("anchored import batch contains {actual} materialized active entries for the occurrence")]
    MaterializedOccurrenceCardinality { actual: usize },
    #[error("anchored materialized occurrence has the wrong content identity")]
    MaterializedContentMismatch,
}
