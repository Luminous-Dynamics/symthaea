// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Additive V2 journal adapter for domain executors that must bind the exact generic Prepared digest.
//!
//! V1 executors remain unchanged. V2 is opt-in. Canonical Prepared digest computation and context
//! validation happen before the durable write-ahead boundary. After generic live revalidation and
//! domain preflight, the exact `PreparedInterventionExecution` is appended and its persistence is
//! attempted. Any acknowledgement ambiguity is returned as an explicit in-doubt outcome and the
//! domain executor is not called. Only an accepted Prepared reference can produce the opaque V2
//! context and reach domain execution.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_authority::WelfareAuthorityPolicyManifest;
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};
use thiserror::Error;

use crate::{AssuredInterventionExecutor, AssuredInterventionPermit};
use crate::evidence_context::EvidenceBoundExecutionError;
use crate::execution_adapter::{
    ExecutionJournalPersistence, ExecutionObservationError, JournaledExecutionOutcome,
};
use crate::execution_recovery::{
    CompletedInterventionExecution, ExecutionJournalError, InterventionExecutionJournal,
    PreparedInterventionExecution, digest_prepared_execution,
};
use crate::prepared_execution_context_v2::{
    PreparedExecutionContextV2, PreparedExecutionContextV2Error,
};
use crate::replay_recovery::{
    DurableEvidenceBoundInterventionPermit, execute_durable_evidence_bound_intervention_once,
};

const MAX_EXECUTION_REF_BYTES: usize = 2048;

/// V2 downstream mutation result with evidence needed for the terminal generic completion record.
#[derive(Debug)]
pub struct ContextualReceiptedExecution<O> {
    output: O,
    completed_at_unix_s: u64,
    result_digest: Sha256Digest,
    executor_evidence_ref: String,
}

impl<O> ContextualReceiptedExecution<O> {
    pub fn new(
        output: O,
        completed_at_unix_s: u64,
        result_digest: Sha256Digest,
        executor_evidence_ref: impl Into<String>,
    ) -> Result<Self, ExecutionObservationError> {
        if result_digest.0 == [0; 32] {
            return Err(ExecutionObservationError::ZeroResultDigest);
        }
        let executor_evidence_ref = executor_evidence_ref.into();
        validate_ref(&executor_evidence_ref)
            .map_err(|_| ExecutionObservationError::InvalidExecutorEvidenceReference)?;
        Ok(Self {
            output,
            completed_at_unix_s,
            result_digest,
            executor_evidence_ref,
        })
    }
}

/// Opt-in domain executor contract for cryptographically correlated Prepared evidence.
pub trait ReceiptedInterventionExecutorV2 {
    type Output;
    type Error: StdError + Send + Sync + 'static;

    /// Deterministic read-only domain checks before any generic `Prepared` journal record exists.
    fn preflight(&self, _permit: &AssuredInterventionPermit) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Execute only after the generic Prepared record has an accepted durable persistence reference
    /// and is exactly rebound into `context`. Errors remain in doubt because the adapter cannot
    /// assume downstream atomicity.
    fn execute_receipted_v2(
        &mut self,
        permit: &AssuredInterventionPermit,
        context: &PreparedExecutionContextV2,
    ) -> Result<ContextualReceiptedExecution<Self::Output>, Self::Error>;
}

struct JournaledExecutorAdapterV2<'a, X, P>
where
    X: ReceiptedInterventionExecutorV2,
    P: ExecutionJournalPersistence,
{
    prepared: PreparedInterventionExecution,
    journal: &'a mut InterventionExecutionJournal,
    persistence: &'a mut P,
    executor: &'a mut X,
}

impl<X, P> AssuredInterventionExecutor for JournaledExecutorAdapterV2<'_, X, P>
where
    X: ReceiptedInterventionExecutorV2,
    P: ExecutionJournalPersistence,
{
    type Output = JournaledExecutionOutcome<X::Output, X::Error, P::Error>;
    type Error = PreExecutionJournalV2Error<P::Error>;

    fn execute(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<Self::Output, Self::Error> {
        if let Err(error) = self.executor.preflight(permit) {
            return Ok(JournaledExecutionOutcome::PreflightRejected { error });
        }

        // Compute and validate the exact correlation digest before any durable Prepared state exists.
        // If serialization or record validation fails, this is still a clean pre-execution failure.
        let expected_prepared_digest = digest_prepared_execution(&self.prepared)
            .map_err(PreExecutionJournalV2Error::Journal)?;
        PreparedExecutionContextV2::verify_exact_prepared_digest(
            &self.prepared,
            expected_prepared_digest,
        )
        .map_err(PreExecutionJournalV2Error::ContextVerification)?;

        let prepared_digest = self
            .journal
            .append_prepared(self.prepared.clone())
            .map_err(PreExecutionJournalV2Error::Journal)?;
        if prepared_digest != expected_prepared_digest {
            // No persistence has been attempted yet. Treat disagreement between the generic journal
            // and the independently computed canonical digest as an internal invariant violation.
            return Err(PreExecutionJournalV2Error::PreparedDigestInvariant {
                expected: expected_prepared_digest,
                actual: prepared_digest,
            });
        }

        // A persistence error does not prove the write failed: the backend may commit before an
        // acknowledgement is lost. Treat both failure and an unusable success reference as explicit
        // write-ahead ambiguity, and never call the domain executor in either case.
        let prepared_persistence_ref = match self
            .persistence
            .persist_execution_journal(self.journal.events(), self.journal.head_hash())
        {
            Ok(reference) => reference,
            Err(error) => {
                return Ok(JournaledExecutionOutcome::PreparedPersistenceInDoubt {
                    error,
                    prepared_digest,
                });
            }
        };
        if validate_ref(&prepared_persistence_ref).is_err() {
            return Ok(
                JournaledExecutionOutcome::PreparedPersistenceReferenceInDoubt {
                    prepared_digest,
                    prepared_persistence_ref,
                },
            );
        }

        // All digest/context verification completed before durability. Context construction is
        // intentionally infallible once the persistence boundary has returned an accepted reference.
        let context = PreparedExecutionContextV2::from_verified_durable(
            &self.prepared,
            prepared_digest,
            prepared_persistence_ref.clone(),
        );

        let execution = match self.executor.execute_receipted_v2(permit, &context) {
            Ok(execution) => execution,
            Err(error) => {
                return Ok(JournaledExecutionOutcome::ExecutorInDoubt {
                    error,
                    prepared_digest,
                    prepared_persistence_ref,
                });
            }
        };

        let ContextualReceiptedExecution {
            output,
            completed_at_unix_s,
            result_digest,
            executor_evidence_ref,
        } = execution;

        let completed = match CompletedInterventionExecution::new(
            self.prepared.execution_id.clone(),
            prepared_digest,
            completed_at_unix_s,
            result_digest,
            executor_evidence_ref,
        ) {
            Ok(completed) => completed,
            Err(error) => {
                return Ok(JournaledExecutionOutcome::CompletionJournalInDoubt {
                    output,
                    error,
                    prepared_digest,
                    prepared_persistence_ref,
                });
            }
        };

        if let Err(error) = self.journal.append_completed(completed.clone()) {
            return Ok(JournaledExecutionOutcome::CompletionJournalInDoubt {
                output,
                error,
                prepared_digest,
                prepared_persistence_ref,
            });
        }

        let completion_persistence_ref = match self
            .persistence
            .persist_execution_journal(self.journal.events(), self.journal.head_hash())
        {
            Ok(reference) => reference,
            Err(error) => {
                return Ok(JournaledExecutionOutcome::CompletionPersistenceInDoubt {
                    output,
                    completed,
                    error,
                    prepared_digest,
                    prepared_persistence_ref,
                });
            }
        };

        if validate_ref(&completion_persistence_ref).is_err() {
            return Ok(
                JournaledExecutionOutcome::CompletionPersistenceReferenceInDoubt {
                    output,
                    completed,
                    prepared_digest,
                    prepared_persistence_ref,
                },
            );
        }

        Ok(JournaledExecutionOutcome::Completed {
            output,
            prepared_digest,
            prepared_persistence_ref,
            completion_persistence_ref,
            journal_head_hash: self.journal.head_hash(),
        })
    }
}

/// V2 counterpart of `execute_durable_intervention_journaled` for only those executors that need
/// exact generic-Prepared correlation evidence inside their domain write-ahead history.
#[allow(clippy::too_many_arguments)]
pub fn execute_durable_intervention_journaled_v2<X, P>(
    permit: DurableEvidenceBoundInterventionPermit,
    execution_id: impl Into<String>,
    current_profile: &MoralPatientEvidenceProfile,
    current_precaution_policy: &PrecautionPolicy,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    current_authority_manifest: &WelfareAuthorityPolicyManifest,
    current_trust_snapshot: &TrustSnapshot,
    unix_s: u64,
    journal: &mut InterventionExecutionJournal,
    persistence: &mut P,
    executor: &mut X,
) -> Result<
    JournaledExecutionOutcome<X::Output, X::Error, P::Error>,
    JournaledExecutionV2GateError<P::Error>,
>
where
    X: ReceiptedInterventionExecutorV2,
    P: ExecutionJournalPersistence,
{
    let prepared = PreparedInterventionExecution::from_permit(execution_id, &permit, unix_s)
        .map_err(JournaledExecutionV2GateError::Prepare)?;
    let mut adapter = JournaledExecutorAdapterV2 {
        prepared,
        journal,
        persistence,
        executor,
    };

    execute_durable_evidence_bound_intervention_once(
        permit,
        current_profile,
        current_precaution_policy,
        consent_ledger,
        subject_registry,
        current_authority_manifest,
        current_trust_snapshot,
        unix_s,
        &mut adapter,
    )
    .map_err(JournaledExecutionV2GateError::Live)
}

fn validate_ref(value: &str) -> Result<(), ()> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_EXECUTION_REF_BYTES
        || value.chars().any(char::is_control)
    {
        Err(())
    } else {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum PreExecutionJournalV2Error<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("could not append or precompute write-ahead execution record: {0}")]
    Journal(#[source] ExecutionJournalError),
    /// Retained for API compatibility. Persistence-attempt failures are represented by
    /// `JournaledExecutionOutcome::PreparedPersistenceInDoubt`.
    #[error("could not persist write-ahead execution journal: {0}")]
    PreparedPersistence(#[source] E),
    /// Retained for API compatibility. Invalid success references are represented by
    /// `JournaledExecutionOutcome::PreparedPersistenceReferenceInDoubt`.
    #[error("execution-journal persistence returned an invalid durable reference")]
    InvalidPersistenceReference,
    #[error("could not verify exact generic Prepared evidence before durability: {0}")]
    ContextVerification(#[source] PreparedExecutionContextV2Error),
    #[error("generic execution journal returned a Prepared digest inconsistent with canonical precomputation")]
    PreparedDigestInvariant {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
}

#[derive(Debug, Error)]
pub enum JournaledExecutionV2GateError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("could not construct prepared execution record: {0}")]
    Prepare(#[source] ExecutionJournalError),
    #[error(transparent)]
    Live(EvidenceBoundExecutionError<PreExecutionJournalV2Error<E>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextual_observation_rejects_zero_result_digest() {
        assert!(matches!(
            ContextualReceiptedExecution::new(
                (),
                100,
                Sha256Digest([0; 32]),
                "executor:evidence:v2"
            ),
            Err(ExecutionObservationError::ZeroResultDigest)
        ));
    }

    #[test]
    fn contextual_observation_rejects_noncanonical_evidence_reference() {
        assert!(matches!(
            ContextualReceiptedExecution::new(
                (),
                100,
                Sha256Digest([7; 32]),
                " bad "
            ),
            Err(ExecutionObservationError::InvalidExecutorEvidenceReference)
        ));
    }
}
