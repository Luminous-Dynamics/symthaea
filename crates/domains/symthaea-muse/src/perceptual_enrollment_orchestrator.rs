// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1EILR-G: normal production orchestration for durable enrollment ->
//! independently witnessed enrollment authority.
//!
//! This module deliberately uses a two-phase CAS protocol rather than holding a
//! filesystem lock across remote network I/O:
//!
//! ```text
//! local observe/classify [+ optional allocator CAS]
//!     -> exact pending request
//!     -> remote idempotent signer/witness operation
//!     -> durable-state reconstruction + authority verification
//!     -> exact coordinator CAS confirmation
//! ```
//!
//! A fresh allocation that survives locally but encounters any later provider
//! or confirmation failure remains exactly one pending durable allocation. The
//! next recovery invocation re-derives the same P1EILR-C `request_sha256`.
//! No provider response, candidate bundle, raw participant token, or boolean
//! witness flag is ever returned as enrollment authority.

use crate::evidence_digest::{
    perceptual_collection_authenticity::FrozenPerceptualCollectionAuthenticityPolicyV1,
    perceptual_enrollment_coordinator::{
        classify_enrollment_coordinator_recovery, DurableEnrollmentCoordinatorErrorV1,
        DurableEnrollmentCoordinatorStoreV1, DurablePerceptualEnrollmentCoordinatorStateV1,
        EnrollmentCoordinatorRecoveryDispositionV1, PendingEnrollmentWitnessRequestV1,
        PerceptualEnrollmentCoordinatorIssueV1,
    },
    perceptual_enrollment_lifecycle::{
        FrozenPerceptualEligibilityGateReceiptV1, FrozenPerceptualEnrollmentPolicyV1,
    },
    perceptual_enrollment_store::{
        DurableEnrollmentAllocationErrorV1, DurableEnrollmentAllocationStateV1,
        DurableEnrollmentAllocationStoreV1,
    },
    perceptual_enrollment_store_observation::inspect_validated_current_from_confirmed_head,
    perceptual_enrollment_witness::FrozenPerceptualEnrollmentWitnessPolicyV1,
    perceptual_enrollment_witness_provider::{
        append_external_enrollment_witness_candidate,
        assemble_external_enrollment_witness_receipt, build_collection_signing_request,
        build_witness_service_request, prepare_external_enrollment_witness,
        ExternalEnrollmentCollectionSignatureV1, ExternalEnrollmentCollectionSigningRequestV1,
        ExternalEnrollmentWitnessServiceRequestV1, ExternalEnrollmentWitnessServiceResponseV1,
        PerceptualEnrollmentWitnessProviderIssueV1,
    },
    perceptual_enrollment_witness_recovery::{
        confirm_pending_enrollment_witness_authority,
        recover_verified_witnessed_enrollment_allocation,
        PerceptualEnrollmentWitnessRecoveryIssueV1,
        RecoveredVerifiedWitnessedEnrollmentAllocationV1,
    },
    perceptual_participant_identity::{
        FrozenParticipantIdentityBoundaryPolicyV1,
        FrozenPerceptualParticipantTokenGenerationReceiptV1,
    },
    perceptual_participant_schedule::{
        PerceptualCohortSlotsV1, PerceptualParticipantScheduleBookV1,
    },
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};

/// Provider failure semantics that matter to the orchestration theorem.
///
/// `Uncertain` means the remote operation may have been accepted but its
/// response was not durably observed by this process. The core never invents a
/// retry successor in that case; the next invocation reuses the same request
/// identity and lets the provider adapter recover the prior semantic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnrollmentWitnessAdapterFailureKindV1 {
    Unavailable,
    Rejected,
    Uncertain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrollmentWitnessAdapterFailureV1 {
    pub kind: EnrollmentWitnessAdapterFailureKindV1,
    /// Optional non-sensitive provider-local diagnostic code. This is runtime
    /// diagnostics only and never enters scientific evidence.
    pub diagnostic_code: Option<String>,
}

impl EnrollmentWitnessAdapterFailureV1 {
    pub fn new(kind: EnrollmentWitnessAdapterFailureKindV1) -> Self {
        Self {
            kind,
            diagnostic_code: None,
        }
    }

    pub fn with_diagnostic_code(
        kind: EnrollmentWitnessAdapterFailureKindV1,
        diagnostic_code: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            diagnostic_code: Some(diagnostic_code.into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnrollmentWitnessProviderStageV1 {
    CollectionSigner,
    IndependentWitness,
}

/// Provider-neutral runtime adapter over the exact P1EILR-E DTOs.
///
/// Implementations may use HTTP, a local signing agent, HSM-backed RPC, or
/// another separately qualified transport. The orchestrator never trusts a
/// successful adapter return directly: every response is fed back through the
/// P1EILR-E verifier/assembler and then P1EILR-D durable reconstruction.
pub trait EnrollmentWitnessProviderAdapterV1 {
    fn sign_collection(
        &mut self,
        request: &ExternalEnrollmentCollectionSigningRequestV1,
    ) -> Result<ExternalEnrollmentCollectionSignatureV1, EnrollmentWitnessAdapterFailureV1>;

    fn witness_enrollment(
        &mut self,
        request: &ExternalEnrollmentWitnessServiceRequestV1,
    ) -> Result<ExternalEnrollmentWitnessServiceResponseV1, EnrollmentWitnessAdapterFailureV1>;
}

#[derive(Debug)]
pub enum PerceptualEnrollmentOrchestratorErrorV1 {
    Coordinator(DurableEnrollmentCoordinatorErrorV1),
    Allocator(DurableEnrollmentAllocationErrorV1),
    CoordinatorDomain(Vec<PerceptualEnrollmentCoordinatorIssueV1>),
    ProviderProtocol(Vec<PerceptualEnrollmentWitnessProviderIssueV1>),
    ProviderAdapter {
        stage: EnrollmentWitnessProviderStageV1,
        failure: EnrollmentWitnessAdapterFailureV1,
    },
    WitnessRecovery(Vec<PerceptualEnrollmentWitnessRecoveryIssueV1>),
    PendingRecoveryRequired { request_sha256: String },
    AllocationDidNotBecomePending,
}

impl std::fmt::Display for PerceptualEnrollmentOrchestratorErrorV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coordinator(error) => write!(formatter, "coordinator operation failed: {error}"),
            Self::Allocator(error) => write!(formatter, "allocator operation failed: {error}"),
            Self::CoordinatorDomain(issues) => write!(
                formatter,
                "coordinator domain validation failed with {} issue(s)",
                issues.len()
            ),
            Self::ProviderProtocol(issues) => write!(
                formatter,
                "enrollment witness provider protocol failed with {} issue(s)",
                issues.len()
            ),
            Self::ProviderAdapter { stage, failure } => write!(
                formatter,
                "enrollment witness provider adapter failed at {stage:?}: {:?}",
                failure.kind
            ),
            Self::WitnessRecovery(issues) => write!(
                formatter,
                "witness authority recovery failed with {} issue(s)",
                issues.len()
            ),
            Self::PendingRecoveryRequired { request_sha256 } => write!(
                formatter,
                "a prior enrollment witness must be recovered before allocating another slot ({request_sha256})"
            ),
            Self::AllocationDidNotBecomePending => write!(
                formatter,
                "successful durable allocation did not classify as exactly one pending witness"
            ),
        }
    }
}

impl std::error::Error for PerceptualEnrollmentOrchestratorErrorV1 {}

#[derive(Debug)]
pub enum PerceptualEnrollmentRecoveryOutcomeV1 {
    Synchronized,
    Recovered(RecoveredVerifiedWitnessedEnrollmentAllocationV1),
}

/// Immutable frozen-study context plus the two owner-local durable stores.
/// Every operation revalidates the underlying P1ENR/P1EIR evidence; this struct
/// is convenience, not authority.
pub struct PerceptualEnrollmentWitnessOrchestratorV1<'a> {
    pub protocol: &'a FrozenPerceptualStudyProtocolV1,
    pub stimulus_pack: &'a FrozenPerceptualStimulusPackV1,
    pub render_binding: &'a FrozenC6fRenderSubjectBindingV1,
    pub cohort: &'a PerceptualCohortSlotsV1,
    pub token_receipt: &'a FrozenPerceptualParticipantTokenGenerationReceiptV1,
    pub identity_policy: &'a FrozenParticipantIdentityBoundaryPolicyV1,
    pub schedule: &'a PerceptualParticipantScheduleBookV1,
    pub enrollment_policy: &'a FrozenPerceptualEnrollmentPolicyV1,
    pub authenticity_policy: &'a FrozenPerceptualCollectionAuthenticityPolicyV1,
    pub witness_policy: &'a FrozenPerceptualEnrollmentWitnessPolicyV1,
    pub allocation_store: &'a DurableEnrollmentAllocationStoreV1,
    pub coordinator_store: &'a DurableEnrollmentCoordinatorStoreV1,
}

impl PerceptualEnrollmentWitnessOrchestratorV1<'_> {
    /// Reconcile restart/provider uncertainty without allocating a new slot.
    ///
    /// `Synchronized` is only an observation. `Recovered` is returned only after
    /// the recovered evidence has reconstructed sealed authority and the exact
    /// coordinator successor has survived durable CAS confirmation/read-back.
    pub fn recover_pending<P: EnrollmentWitnessProviderAdapterV1>(
        &self,
        provider: &mut P,
    ) -> Result<PerceptualEnrollmentRecoveryOutcomeV1, PerceptualEnrollmentOrchestratorErrorV1> {
        let prior = self
            .coordinator_store
            .inspect()
            .map_err(PerceptualEnrollmentOrchestratorErrorV1::Coordinator)?;
        let current = self.observe_current(&prior)?;
        let pending = match self.classify(&current, &prior)? {
            EnrollmentCoordinatorRecoveryDispositionV1::Synchronized => {
                return Ok(PerceptualEnrollmentRecoveryOutcomeV1::Synchronized);
            }
            EnrollmentCoordinatorRecoveryDispositionV1::ExactlyOnePending(pending) => pending,
        };
        let (authority, next) = self.complete_pending(&prior, &pending, provider)?;
        let expected = prior.coordinator_state_sha256.clone();
        let authority = self
            .coordinator_store
            .transition(
                &expected,
                move |_| -> Result<
                    (
                        DurablePerceptualEnrollmentCoordinatorStateV1,
                        RecoveredVerifiedWitnessedEnrollmentAllocationV1,
                    ),
                    DurableEnrollmentCoordinatorErrorV1,
                > { Ok((next, authority)) },
            )
            .map_err(PerceptualEnrollmentOrchestratorErrorV1::Coordinator)?;
        Ok(PerceptualEnrollmentRecoveryOutcomeV1::Recovered(authority))
    }

    /// Allocate exactly one new deterministic P1ENR slot and independently
    /// witness it before returning runtime enrollment authority.
    ///
    /// If any earlier durable allocation is pending, this operation refuses the
    /// new eligibility gate. The caller must invoke `recover_pending` first so
    /// the earlier participant's authority handoff cannot be silently skipped.
    pub fn allocate_and_witness_one<P: EnrollmentWitnessProviderAdapterV1>(
        &self,
        gate: &FrozenPerceptualEligibilityGateReceiptV1,
        provider: &mut P,
    ) -> Result<
        RecoveredVerifiedWitnessedEnrollmentAllocationV1,
        PerceptualEnrollmentOrchestratorErrorV1,
    > {
        let prior = self
            .coordinator_store
            .inspect()
            .map_err(PerceptualEnrollmentOrchestratorErrorV1::Coordinator)?;
        let current = self.observe_current(&prior)?;
        match self.classify(&current, &prior)? {
            EnrollmentCoordinatorRecoveryDispositionV1::ExactlyOnePending(pending) => {
                return Err(
                    PerceptualEnrollmentOrchestratorErrorV1::PendingRecoveryRequired {
                        request_sha256: pending.request_sha256,
                    },
                );
            }
            EnrollmentCoordinatorRecoveryDispositionV1::Synchronized => {}
        }

        // The allocator's own cross-process lock + expected-head CAS decides the
        // race if two synchronized callers try to enroll concurrently. Exactly
        // one can advance from this predecessor; the loser fails closed.
        self.allocation_store
            .allocate_next(
                self.protocol,
                self.stimulus_pack,
                self.render_binding,
                self.cohort,
                self.token_receipt,
                self.identity_policy,
                self.schedule,
                self.enrollment_policy,
                &current.ledger.ledger_sha256,
                gate,
            )
            .map_err(PerceptualEnrollmentOrchestratorErrorV1::Allocator)?;

        // Do not trust the returned runtime allocation as the source of the
        // provider request. Re-open through the P1EILR-F observer and re-run the
        // coordinator theorem from durable state.
        let current = self.observe_current(&prior)?;
        let pending = match self.classify(&current, &prior)? {
            EnrollmentCoordinatorRecoveryDispositionV1::ExactlyOnePending(pending) => pending,
            EnrollmentCoordinatorRecoveryDispositionV1::Synchronized => {
                return Err(PerceptualEnrollmentOrchestratorErrorV1::AllocationDidNotBecomePending);
            }
        };

        let (authority, next) = self.complete_pending(&prior, &pending, provider)?;
        let expected = prior.coordinator_state_sha256.clone();
        self.coordinator_store
            .transition(
                &expected,
                move |_| -> Result<
                    (
                        DurablePerceptualEnrollmentCoordinatorStateV1,
                        RecoveredVerifiedWitnessedEnrollmentAllocationV1,
                    ),
                    DurableEnrollmentCoordinatorErrorV1,
                > { Ok((next, authority)) },
            )
            .map_err(PerceptualEnrollmentOrchestratorErrorV1::Coordinator)
    }

    fn observe_current(
        &self,
        prior: &DurablePerceptualEnrollmentCoordinatorStateV1,
    ) -> Result<DurableEnrollmentAllocationStateV1, PerceptualEnrollmentOrchestratorErrorV1> {
        inspect_validated_current_from_confirmed_head(
            self.allocation_store,
            self.protocol,
            self.stimulus_pack,
            self.render_binding,
            self.cohort,
            self.token_receipt,
            self.identity_policy,
            self.schedule,
            self.enrollment_policy,
            &prior.confirmed_enrollment_ledger.ledger_sha256,
        )
        .map_err(PerceptualEnrollmentOrchestratorErrorV1::Allocator)
    }

    fn classify(
        &self,
        current: &DurableEnrollmentAllocationStateV1,
        prior: &DurablePerceptualEnrollmentCoordinatorStateV1,
    ) -> Result<EnrollmentCoordinatorRecoveryDispositionV1, PerceptualEnrollmentOrchestratorErrorV1> {
        classify_enrollment_coordinator_recovery(
            self.protocol,
            self.stimulus_pack,
            self.render_binding,
            self.cohort,
            self.token_receipt,
            self.identity_policy,
            self.schedule,
            self.enrollment_policy,
            self.witness_policy,
            self.authenticity_policy,
            current,
            prior,
        )
        .map_err(PerceptualEnrollmentOrchestratorErrorV1::CoordinatorDomain)
    }

    fn complete_pending<P: EnrollmentWitnessProviderAdapterV1>(
        &self,
        prior: &DurablePerceptualEnrollmentCoordinatorStateV1,
        pending: &PendingEnrollmentWitnessRequestV1,
        provider: &mut P,
    ) -> Result<
        (
            RecoveredVerifiedWitnessedEnrollmentAllocationV1,
            DurablePerceptualEnrollmentCoordinatorStateV1,
        ),
        PerceptualEnrollmentOrchestratorErrorV1,
    > {
        let prepared = prepare_external_enrollment_witness(pending)
            .map_err(PerceptualEnrollmentOrchestratorErrorV1::ProviderProtocol)?;
        let collection_request = build_collection_signing_request(&prepared, self.witness_policy)
            .map_err(PerceptualEnrollmentOrchestratorErrorV1::ProviderProtocol)?;
        let collection_response = provider
            .sign_collection(&collection_request)
            .map_err(|failure| PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter {
                stage: EnrollmentWitnessProviderStageV1::CollectionSigner,
                failure,
            })?;
        let witness_request = build_witness_service_request(
            &prepared,
            &collection_request,
            &collection_response,
            self.witness_policy,
        )
        .map_err(PerceptualEnrollmentOrchestratorErrorV1::ProviderProtocol)?;
        let witness_response = provider
            .witness_enrollment(&witness_request)
            .map_err(|failure| PerceptualEnrollmentOrchestratorErrorV1::ProviderAdapter {
                stage: EnrollmentWitnessProviderStageV1::IndependentWitness,
                failure,
            })?;
        let receipt = assemble_external_enrollment_witness_receipt(
            &prepared,
            &collection_request,
            &collection_response,
            &witness_request,
            &witness_response,
            self.witness_policy,
        )
        .map_err(PerceptualEnrollmentOrchestratorErrorV1::ProviderProtocol)?;
        let candidate_bundle = append_external_enrollment_witness_candidate(
            &prior.confirmed_witness_bundle,
            receipt,
            self.witness_policy,
        )
        .map_err(PerceptualEnrollmentOrchestratorErrorV1::ProviderProtocol)?;

        let authority = recover_verified_witnessed_enrollment_allocation(
            self.protocol,
            self.stimulus_pack,
            self.render_binding,
            self.cohort,
            self.token_receipt,
            self.identity_policy,
            self.schedule,
            self.enrollment_policy,
            self.authenticity_policy,
            self.witness_policy,
            self.allocation_store,
            &candidate_bundle,
        )
        .map_err(PerceptualEnrollmentOrchestratorErrorV1::WitnessRecovery)?;

        let next = confirm_pending_enrollment_witness_authority(
            self.protocol,
            self.stimulus_pack,
            self.render_binding,
            self.cohort,
            self.token_receipt,
            self.identity_policy,
            self.schedule,
            self.enrollment_policy,
            self.authenticity_policy,
            self.witness_policy,
            self.allocation_store,
            prior,
            &authority,
            &candidate_bundle,
        )
        .map_err(PerceptualEnrollmentOrchestratorErrorV1::WitnessRecovery)?;

        Ok((authority, next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AlwaysUnavailable;

    impl EnrollmentWitnessProviderAdapterV1 for AlwaysUnavailable {
        fn sign_collection(
            &mut self,
            _request: &ExternalEnrollmentCollectionSigningRequestV1,
        ) -> Result<ExternalEnrollmentCollectionSignatureV1, EnrollmentWitnessAdapterFailureV1> {
            Err(EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Unavailable,
            ))
        }

        fn witness_enrollment(
            &mut self,
            _request: &ExternalEnrollmentWitnessServiceRequestV1,
        ) -> Result<ExternalEnrollmentWitnessServiceResponseV1, EnrollmentWitnessAdapterFailureV1> {
            Err(EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Unavailable,
            ))
        }
    }

    fn assert_adapter<T: EnrollmentWitnessProviderAdapterV1>() {}

    #[test]
    fn provider_adapter_surface_is_provider_neutral() {
        assert_adapter::<AlwaysUnavailable>();
    }

    #[test]
    fn provider_uncertainty_is_distinct_from_rejection_and_unavailability() {
        assert_ne!(
            EnrollmentWitnessAdapterFailureKindV1::Uncertain,
            EnrollmentWitnessAdapterFailureKindV1::Rejected
        );
        assert_ne!(
            EnrollmentWitnessAdapterFailureKindV1::Uncertain,
            EnrollmentWitnessAdapterFailureKindV1::Unavailable
        );
    }

    #[test]
    fn recovery_success_type_is_explicitly_witnessed_runtime_authority() {
        let name = std::any::type_name::<RecoveredVerifiedWitnessedEnrollmentAllocationV1>();
        assert!(name.contains("RecoveredVerifiedWitnessedEnrollmentAllocationV1"));
    }
}
