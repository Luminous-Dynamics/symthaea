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
pub mod exact_policy;
pub mod observation;
pub mod profile_adoption;
pub mod scope;
pub mod subject_snapshot;
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
pub use exact_policy::{
    ExactVerificationPolicyError, ExactVerificationPolicyId, ExactVerificationPolicyV1,
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
pub use subject_snapshot::{
    CONTINUITY_SUBJECT_SNAPSHOT_SCHEMA_V1, ContinuitySubjectSnapshotError,
    ContinuitySubjectSnapshotId, ContinuitySubjectSnapshotV1,
    ValidatedContinuitySubjectSnapshotV1,
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
