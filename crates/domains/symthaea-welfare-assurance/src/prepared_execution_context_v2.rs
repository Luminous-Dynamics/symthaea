// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact generic-journal context handed to digest-bound high-assurance domain executors.
//!
//! This is evidence correlation, not additional action authority. The public type has private
//! fields and no public constructor. The V2 journal adapter verifies the exact generic Prepared
//! digest before crossing the durable write-ahead boundary, then uses the crate-private infallible
//! constructor only after the Prepared journal and its durable reference have been accepted.

#![deny(unsafe_code)]

use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use thiserror::Error;

use crate::execution_recovery::{
    ExecutionJournalError, PreparedInterventionExecution, digest_prepared_execution,
};

const MAX_PREPARED_PERSISTENCE_REF_BYTES: usize = 2048;

/// Opaque V2 correlation context for a generic execution-journal `Prepared` record.
///
/// High-assurance domain executors may inspect the exact execution identity, canonical prepared
/// digest, timestamp and durable reference, but external callers cannot construct this value from
/// a permit, execution ID or arbitrary digest alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedExecutionContextV2 {
    execution_id: String,
    prepared_digest: Sha256Digest,
    prepared_at_unix_s: u64,
    prepared_persistence_ref: String,
}

impl PreparedExecutionContextV2 {
    /// Verify a caller-supplied digest against the exact generic write-ahead record.
    ///
    /// The V2 adapter performs this check before any Prepared persistence is attempted so a
    /// serialization/validation error cannot occur after the durable in-doubt boundary.
    pub(crate) fn verify_exact_prepared_digest(
        prepared: &PreparedInterventionExecution,
        prepared_digest: Sha256Digest,
    ) -> Result<(), PreparedExecutionContextV2Error> {
        prepared.validate()?;
        let recomputed = digest_prepared_execution(prepared)?;
        if recomputed != prepared_digest {
            return Err(PreparedExecutionContextV2Error::PreparedDigestMismatch {
                expected: recomputed,
                actual: prepared_digest,
            });
        }
        Ok(())
    }

    /// Validate the durable Prepared-journal reference before the context is constructed.
    pub(crate) fn validate_persistence_ref(
        value: &str,
    ) -> Result<(), PreparedExecutionContextV2Error> {
        if value.trim().is_empty()
            || value != value.trim()
            || value.len() > MAX_PREPARED_PERSISTENCE_REF_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(PreparedExecutionContextV2Error::InvalidPreparedPersistenceReference);
        }
        Ok(())
    }

    /// Construct from materials already verified by the V2 journal adapter.
    ///
    /// This is intentionally infallible: once the Prepared journal has been durably persisted,
    /// context construction must not introduce a new retry-shaped error path.
    pub(crate) fn from_verified_durable(
        prepared: &PreparedInterventionExecution,
        prepared_digest: Sha256Digest,
        prepared_persistence_ref: String,
    ) -> Self {
        Self {
            execution_id: prepared.execution_id.clone(),
            prepared_digest,
            prepared_at_unix_s: prepared.prepared_at_unix_s,
            prepared_persistence_ref,
        }
    }

    pub fn execution_id(&self) -> &str {
        &self.execution_id
    }

    pub fn prepared_digest(&self) -> Sha256Digest {
        self.prepared_digest
    }

    pub fn prepared_at_unix_s(&self) -> u64 {
        self.prepared_at_unix_s
    }

    pub fn prepared_persistence_ref(&self) -> &str {
        &self.prepared_persistence_ref
    }
}

#[derive(Debug, Error)]
pub enum PreparedExecutionContextV2Error {
    #[error(transparent)]
    Journal(#[from] ExecutionJournalError),
    #[error("caller-supplied prepared digest does not match the exact generic Prepared record")]
    PreparedDigestMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("invalid prepared execution-journal persistence reference")]
    InvalidPreparedPersistenceReference,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::intervention_interlock::WelfareConstraintLevel;
    use symthaea_core::welfare::SubjectAffectingAction;
    use symthaea_psych_bench::moral_patient::ProtectionDisposition;

    use crate::execution_recovery::EXECUTION_JOURNAL_SCHEMA;

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn prepared() -> PreparedInterventionExecution {
        PreparedInterventionExecution {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: "exec:restore:v2:1".into(),
            authority_id: "authority:restore:v2:1".into(),
            target_id: "symthaea:self:episodic-memory:instance:v2".into(),
            action: SubjectAffectingAction::MemoryModification,
            rationale_digest: digest(1),
            welfare_profile_digest: digest(2),
            precaution_policy_digest: digest(3),
            protection_disposition: ProtectionDisposition::Baseline,
            welfare_constraint: WelfareConstraintLevel::Baseline,
            replay_generation: 1,
            replay_snapshot_digest: digest(4),
            replay_persistence_ref: "replay:v2:1".into(),
            prepared_at_unix_s: 100,
            permit_not_after_unix_s: 200,
        }
    }

    #[test]
    fn verified_material_builds_read_only_context_without_post_persist_validation() {
        let prepared = prepared();
        let prepared_digest = digest_prepared_execution(&prepared).unwrap();
        PreparedExecutionContextV2::verify_exact_prepared_digest(&prepared, prepared_digest)
            .unwrap();
        PreparedExecutionContextV2::validate_persistence_ref(
            "execution-journal:prepared:v2:1",
        )
        .unwrap();

        let context = PreparedExecutionContextV2::from_verified_durable(
            &prepared,
            prepared_digest,
            "execution-journal:prepared:v2:1".into(),
        );

        assert_eq!(context.execution_id(), prepared.execution_id);
        assert_eq!(context.prepared_digest(), prepared_digest);
        assert_eq!(context.prepared_at_unix_s(), prepared.prepared_at_unix_s);
        assert_eq!(
            context.prepared_persistence_ref(),
            "execution-journal:prepared:v2:1"
        );
    }

    #[test]
    fn substituted_prepared_digest_is_rejected_before_durable_context_construction() {
        let prepared = prepared();
        let actual = digest_prepared_execution(&prepared).unwrap();
        let substituted = digest(99);
        assert_ne!(actual, substituted);

        let error = PreparedExecutionContextV2::verify_exact_prepared_digest(
            &prepared,
            substituted,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            PreparedExecutionContextV2Error::PreparedDigestMismatch { expected, actual }
                if expected == digest_prepared_execution(&prepared).unwrap() && actual == substituted
        ));
    }

    #[test]
    fn malformed_persistence_reference_is_rejected() {
        let error = PreparedExecutionContextV2::validate_persistence_ref(
            " execution-journal:prepared:v2:1",
        )
        .unwrap_err();

        assert!(matches!(
            error,
            PreparedExecutionContextV2Error::InvalidPreparedPersistenceReference
        ));
    }

    #[test]
    fn changing_exact_prepared_record_changes_required_digest() {
        let original = prepared();
        let original_digest = digest_prepared_execution(&original).unwrap();
        let mut changed = original.clone();
        changed.authority_id = "authority:restore:v2:other".into();
        let changed_digest = digest_prepared_execution(&changed).unwrap();
        assert_ne!(original_digest, changed_digest);

        let error = PreparedExecutionContextV2::verify_exact_prepared_digest(
            &changed,
            original_digest,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PreparedExecutionContextV2Error::PreparedDigestMismatch { .. }
        ));
    }
}
