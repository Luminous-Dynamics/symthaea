// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Computing-continuity kernel.
//!
//! This crate deliberately separates observed facts from dependency claims,
//! continuity requirements, verification evidence, and execution authority.
//! None of these values grants migration authority by itself.

#![deny(unsafe_code)]

pub mod auth_wire;
mod compose;
pub mod contract;
pub mod distributed;
pub mod distributed_state;
pub mod exact_policy;
pub mod failure_domain;
pub mod observation;
pub mod profile_adoption;
pub mod scope;
pub mod subject_contract;
pub mod subject_witness;
pub mod verifier;
mod witness;

pub use auth_wire::{
    CONTINUITY_VERIFICATION_CLAIM_AUTH_SCHEMA, CONTINUITY_VERIFICATION_CLAIM_HASH_ALGORITHM,
    CONTINUITY_VERIFICATION_XENIA_PURPOSE, canonical_verification_claim_bytes,
    canonical_verification_claim_digest,
};
pub use contract::{
    ApprovalBasis, ContinuityContractId, ContinuityContractV1, ContinuityRequirementId,
    ContinuityRequirementV1, ContractError, EquivalencePredicate, RequirementCriticality,
    ValidatedContinuityContractV1,
};
pub use distributed::{
    DISTRIBUTED_CHANGE_BUDGET_SCHEMA_V1, DistributedChangeBudgetError, DistributedChangeBudgetId,
    DistributedChangeBudgetV1, MutualExclusionSetV1, RecoveryPathClassV1,
    ValidatedDistributedChangeBudgetV1,
};
pub use distributed_state::{
    DISTRIBUTED_STATE_CONTEXT_SCHEMA_V1, PARTICIPANT_STATE_CLAIM_SCHEMA_V1,
    AuthenticatedParticipantStateEvidenceId, DistributedStateContextId,
    DistributedStateContextV1, DistributedStateError, ParticipantOperationalStateV1,
    ParticipantSetDigest, ParticipantStateClaimId, ParticipantStateClaimV1,
    ValidatedDistributedStateContextV1,
};
pub use exact_policy::{
    ExactVerificationPolicyError, ExactVerificationPolicyId, ExactVerificationPolicyV1,
};
pub use failure_domain::{
    FAILURE_DOMAIN_POLICY_SCHEMA_V1, FailureDomainGroupV1, FailureDomainKindV1,
    FailureDomainPolicyError, FailureDomainPolicyId, FailureDomainPolicyV1,
    ValidatedFailureDomainPolicyV1,
};
pub use observation::{
    DependencyBasis, DependencyClaimId, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
    ObservationEnvelopeV1, ObservationError, ObservationId,
};
pub use profile_adoption::{
    VERIFIER_PROFILE_ADOPTION_SUBJECT_SCHEMA_V1, VERIFIER_PROFILE_ADOPTION_TRANSITION_SCHEMA_V1,
    VerifierAdoptionScopeV1, VerifierProfileAdoptionError, VerifierProfileAdoptionPredecessorV1,
    VerifierProfileAdoptionSubjectId, VerifierProfileAdoptionSubjectV1,
    VerifierProfileAdoptionTransitionDigest, VerifierProfileAdoptionTransitionV1,
};
pub use scope::{
    CONTINUITY_SUBJECT_SCHEMA_V1, ContinuityScopeV1, ContinuitySubjectError,
    ContinuitySubjectId, ContinuitySubjectV1,
};
pub use subject_contract::{
    SubjectBoundContinuityContractId, SubjectBoundContinuityContractV1,
    SubjectContractBindingError,
};
pub use subject_witness::{
    SubjectBoundQualifiedContinuityWitnessId, SubjectBoundQualifiedContinuityWitnessV1,
    SubjectWitnessBindingError,
};
pub use verifier::{
    AuthenticatedVerificationEvidenceId, VerificationAdmissionError, VerificationEvidenceClaimId,
    VerificationEvidenceClaimV1, VerificationOutcomeV1, VerifierProfileId, VerifierProfileV1,
};
pub use witness::{
    EvidenceClass, ObligationDispositionV1, QualifiedContinuityWitnessV1, TargetRealizationId,
    VerificationObligationId, VerificationPolicyEntryV1, WitnessError, WitnessId,
    WitnessManifestId,
};
