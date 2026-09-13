// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independently recomputable cross-ledger commitment for one persisted episodic restore.
//!
//! The execution side derives its inputs from `PreparedExecutionContextV2`, which exists only after
//! the exact generic execution-journal `Prepared` record has crossed the durable write-ahead
//! boundary. The reconciliation side independently derives the same generic Prepared digest from
//! the recovered `PreparedInterventionExecution`.
//!
//! The resulting digest is correlation evidence, not action authority. It is designed to fit in the
//! existing `Restored.restore_result_digest` field so stronger V2 evidence does not require changing
//! the legacy quarantine-ledger enum or its bincode discriminants.

#![deny(unsafe_code)]

use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_memory::episodic_replay::EpisodeInstanceId;
use thiserror::Error;

use crate::execution_recovery::{
    ExecutionJournalError, PreparedInterventionExecution, digest_prepared_execution,
};
use crate::memory_identity::EpisodeContentId;
use crate::prepared_execution_context_v2::PreparedExecutionContextV2;
use crate::quarantine_restore_binding_v2::{
    RestorePreparedBindingV2, RestorePreparedBindingV2Error,
};

const CORRELATION_DOMAIN: &[u8] =
    b"symthaea.welfare.persisted-episodic-restore-correlation.v2\0";
const MAX_TARGET_ID_BYTES: usize = 256;
const MAX_EXECUTION_ID_BYTES: usize = 256;

/// Compute the V2 correlation commitment at the live execution boundary.
///
/// `restore_prepared_event_hash` is the hash returned when the exact domain `RestorePrepared`
/// event is appended. It identifies that write-ahead event directly; unrelated later ledger events
/// may interleave without changing the correlation theorem.
pub fn digest_persisted_restore_correlation_from_context_v2(
    context: &PreparedExecutionContextV2,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    restored_at_unix_s: u64,
    restore_prepared_event_hash: Sha256Digest,
) -> Result<Sha256Digest, PersistedRestoreCorrelationV2Error> {
    let binding = RestorePreparedBindingV2::from_context(context, restored_at_unix_s)?;
    digest_from_parts(
        binding.generic_prepared_digest(),
        binding.generic_prepared_at_unix_s(),
        binding.execution_id(),
        binding.target_id(),
        instance_id,
        content_id,
        restored_at_unix_s,
        restore_prepared_event_hash,
    )
}

/// Independently recompute the same V2 correlation commitment during post-crash reconciliation.
///
/// No execution-side binding object is trusted here. The exact Prepared digest is recomputed from
/// the recovered generic journal record, while `restore_prepared_event_hash` is taken from the
/// uniquely matching anchored domain `RestorePrepared` envelope.
pub fn digest_persisted_restore_correlation_from_prepared_v2(
    prepared: &PreparedInterventionExecution,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    restored_at_unix_s: u64,
    restore_prepared_event_hash: Sha256Digest,
) -> Result<Sha256Digest, PersistedRestoreCorrelationV2Error> {
    prepared.validate()?;
    let prepared_digest = digest_prepared_execution(prepared)?;
    digest_from_parts(
        prepared_digest,
        prepared.prepared_at_unix_s,
        &prepared.execution_id,
        &prepared.target_id,
        instance_id,
        content_id,
        restored_at_unix_s,
        restore_prepared_event_hash,
    )
}

#[allow(clippy::too_many_arguments)]
fn digest_from_parts(
    prepared_digest: Sha256Digest,
    prepared_at_unix_s: u64,
    execution_id: &str,
    target_id: &str,
    instance_id: EpisodeInstanceId,
    content_id: EpisodeContentId,
    restored_at_unix_s: u64,
    restore_prepared_event_hash: Sha256Digest,
) -> Result<Sha256Digest, PersistedRestoreCorrelationV2Error> {
    validate_text("execution_id", execution_id, MAX_EXECUTION_ID_BYTES)?;
    validate_text("target_id", target_id, MAX_TARGET_ID_BYTES)?;
    if prepared_digest.0 == [0; 32] {
        return Err(PersistedRestoreCorrelationV2Error::ZeroPreparedDigest);
    }
    if restore_prepared_event_hash.0 == [0; 32] {
        return Err(PersistedRestoreCorrelationV2Error::ZeroRestorePreparedEventHash);
    }
    if restored_at_unix_s < prepared_at_unix_s {
        return Err(PersistedRestoreCorrelationV2Error::RestoredBeforeGenericPrepare {
            prepared_at_unix_s,
            restored_at_unix_s,
        });
    }

    let mut hasher = Sha256::new();
    hasher.update(CORRELATION_DOMAIN);
    hasher.update(&prepared_digest.0);
    hasher.update(&prepared_at_unix_s.to_le_bytes());
    hash_text(&mut hasher, execution_id);
    hash_text(&mut hasher, target_id);
    hasher.update(&instance_id.as_uuid().as_u128().to_le_bytes());
    hasher.update(&content_id.digest().0);
    hasher.update(&restored_at_unix_s.to_le_bytes());
    hasher.update(&restore_prepared_event_hash.0);
    Ok(hasher.finalize())
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn validate_text(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), PersistedRestoreCorrelationV2Error> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
    {
        return Err(PersistedRestoreCorrelationV2Error::InvalidText {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum PersistedRestoreCorrelationV2Error {
    #[error(transparent)]
    Journal(#[from] ExecutionJournalError),
    #[error(transparent)]
    Binding(#[from] RestorePreparedBindingV2Error),
    #[error("invalid persisted-restore correlation field `{field}`: {value:?}")]
    InvalidText { field: &'static str, value: String },
    #[error("generic Prepared digest must not be zero")]
    ZeroPreparedDigest,
    #[error("restore-prepared event hash must not be zero")]
    ZeroRestorePreparedEventHash,
    #[error(
        "Restored transition predates generic Prepared: prepared={prepared_at_unix_s}, restored={restored_at_unix_s}"
    )]
    RestoredBeforeGenericPrepare {
        prepared_at_unix_s: u64,
        restored_at_unix_s: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_core::intervention_interlock::WelfareConstraintLevel;
    use symthaea_core::welfare::SubjectAffectingAction;
    use symthaea_memory::episodic_replay::{Episode, EpisodicMemory, EpisodicReplayConfig};
    use symthaea_psych_bench::moral_patient::ProtectionDisposition;

    use crate::execution_recovery::EXECUTION_JOURNAL_SCHEMA;
    use crate::memory_identity::episode_content_id;

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn prepared(authority_id: &str) -> PreparedInterventionExecution {
        PreparedInterventionExecution {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: "exec:restore:correlation:v2".into(),
            authority_id: authority_id.into(),
            target_id: "symthaea:self:episodic-memory:instance:correlation-v2".into(),
            action: SubjectAffectingAction::MemoryModification,
            rationale_digest: digest(1),
            welfare_profile_digest: digest(2),
            precaution_policy_digest: digest(3),
            protection_disposition: ProtectionDisposition::Baseline,
            welfare_constraint: WelfareConstraintLevel::Baseline,
            replay_generation: 1,
            replay_snapshot_digest: digest(4),
            replay_persistence_ref: "replay:correlation:v2".into(),
            prepared_at_unix_s: 100,
            permit_not_after_unix_s: 200,
        }
    }

    fn identity() -> (EpisodeInstanceId, EpisodeContentId) {
        let source = Episode::new(
            ContinuousHV::from_values(vec![0.12, 0.34, 0.56]),
            ContinuousHV::from_values(vec![0.65, 0.43, 0.21]),
            0.83,
            42,
        );
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        let instance_id = memory.store_if_significant_with_id(source).unwrap();
        let stored = memory
            .get_top_episode_instances(1)
            .into_iter()
            .next()
            .unwrap()
            .1;
        let content_id = episode_content_id(&stored).unwrap();
        (instance_id, content_id)
    }

    fn context(prepared: &PreparedInterventionExecution) -> PreparedExecutionContextV2 {
        let prepared_digest = digest_prepared_execution(prepared).unwrap();
        PreparedExecutionContextV2::verify_exact_prepared_digest(prepared, prepared_digest).unwrap();
        PreparedExecutionContextV2::from_verified_durable(
            prepared,
            prepared_digest,
            "execution-journal:prepared:correlation:v2".into(),
        )
    }

    #[test]
    fn execution_and_reconciliation_paths_compute_identical_commitment() {
        let prepared = prepared("authority:correlation:v2");
        let context = context(&prepared);
        let (instance_id, content_id) = identity();
        let event_hash = digest(80);

        let execution = digest_persisted_restore_correlation_from_context_v2(
            &context,
            instance_id,
            content_id,
            120,
            event_hash,
        )
        .unwrap();
        let reconciliation = digest_persisted_restore_correlation_from_prepared_v2(
            &prepared,
            instance_id,
            content_id,
            120,
            event_hash,
        )
        .unwrap();

        assert_eq!(execution, reconciliation);
    }

    #[test]
    fn same_visible_restore_identity_with_different_authority_has_different_commitment() {
        let first = prepared("authority:first");
        let second = prepared("authority:second");
        assert_eq!(first.execution_id, second.execution_id);
        assert_eq!(first.target_id, second.target_id);
        let (instance_id, content_id) = identity();
        let event_hash = digest(81);

        let first_digest = digest_persisted_restore_correlation_from_prepared_v2(
            &first,
            instance_id,
            content_id,
            120,
            event_hash,
        )
        .unwrap();
        let second_digest = digest_persisted_restore_correlation_from_prepared_v2(
            &second,
            instance_id,
            content_id,
            120,
            event_hash,
        )
        .unwrap();

        assert_ne!(first_digest, second_digest);
    }

    #[test]
    fn changing_restore_prepared_event_hash_changes_commitment() {
        let prepared = prepared("authority:event-binding");
        let (instance_id, content_id) = identity();
        let first = digest_persisted_restore_correlation_from_prepared_v2(
            &prepared,
            instance_id,
            content_id,
            120,
            digest(82),
        )
        .unwrap();
        let second = digest_persisted_restore_correlation_from_prepared_v2(
            &prepared,
            instance_id,
            content_id,
            120,
            digest(83),
        )
        .unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn changing_occurrence_or_content_changes_commitment() {
        let prepared = prepared("authority:identity-binding");
        let (first_instance, first_content) = identity();
        let (second_instance, second_content) = identity();
        let event_hash = digest(84);
        let first = digest_persisted_restore_correlation_from_prepared_v2(
            &prepared,
            first_instance,
            first_content,
            120,
            event_hash,
        )
        .unwrap();
        let second = digest_persisted_restore_correlation_from_prepared_v2(
            &prepared,
            second_instance,
            second_content,
            120,
            event_hash,
        )
        .unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn restored_time_regression_and_zero_event_hash_fail_closed() {
        let prepared = prepared("authority:time-binding");
        let (instance_id, content_id) = identity();
        assert!(matches!(
            digest_persisted_restore_correlation_from_prepared_v2(
                &prepared,
                instance_id,
                content_id,
                99,
                digest(85),
            ),
            Err(PersistedRestoreCorrelationV2Error::RestoredBeforeGenericPrepare { .. })
        ));
        assert!(matches!(
            digest_persisted_restore_correlation_from_prepared_v2(
                &prepared,
                instance_id,
                content_id,
                120,
                Sha256Digest([0; 32]),
            ),
            Err(PersistedRestoreCorrelationV2Error::ZeroRestorePreparedEventHash)
        ));
    }
}
