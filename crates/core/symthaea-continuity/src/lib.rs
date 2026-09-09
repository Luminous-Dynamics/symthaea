// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Computing-continuity kernel.
//!
//! This crate deliberately separates observed facts from dependency claims,
//! continuity requirements, verification evidence, freshness/currentness,
//! commit eligibility, and execution authority.

#![deny(unsafe_code)]

mod compose;
pub mod commit;
pub mod contract;
pub mod freshness;
pub mod observation;
pub mod verifier;
mod witness;

pub use commit::{
    CommitPreconditionsError, CommitPreconditionsSatisfiedV1, CurrentVerifierRootV1,
    TransitionCommitPreconditionsId, TransitionCommitPreconditionsV1,
    TransitionStateSnapshotId, TransitionStateSnapshotV1,
};
pub use contract::{
    ApprovalBasis, ContinuityContractId, ContinuityContractV1, ContinuityRequirementId,
    ContinuityRequirementV1, ContractError, EquivalencePredicate, RequirementCriticality,
    ValidatedContinuityContractV1,
};
pub use freshness::{
    EvidenceFreshnessRequirementV1, EvidenceStabilityClassV1, FreshnessError,
    FreshnessPolicyEntryV1, FreshnessPolicyId, FreshnessPolicyV1,
    QualifiedContinuityWitnessContextV1, QualifiedWitnessContextId,
    TrustedClockLineageV1, TrustedClockSnapshotId, TrustedClockSnapshotV1,
    VerifierRootDependencyV1,
};
pub use observation::{
    DependencyBasis, DependencyClaimId, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
    ObservationEnvelopeV1, ObservationError, ObservationId,
};
pub use verifier::{
    AuthenticatedVerificationEvidenceId, VerificationAdmissionError,
    VerificationEvidenceClaimId, VerificationEvidenceClaimV1, VerificationOutcomeV1,
    VerifierProfileId, VerifierProfileV1,
};
pub use witness::{
    EvidenceClass, ObligationDispositionV1, QualifiedContinuityWitnessV1,
    TargetRealizationId, VerificationObligationId, VerificationPolicyEntryV1,
    VerificationPolicyId, VerificationPolicyV1, WitnessError, WitnessId, WitnessManifestId,
};
