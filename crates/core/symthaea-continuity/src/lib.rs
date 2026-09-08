// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Computing-continuity kernel.
//!
//! This crate deliberately separates observed facts from dependency claims,
//! continuity requirements, verification evidence, and execution authority.
//! None of these values grants migration authority by itself.

#![deny(unsafe_code)]

pub mod compose;
pub mod contract;
pub mod observation;
mod witness;

pub use compose::{compose_qualified_witness, ComposeError, ContinuityEvidenceSubmissionV1};
pub use contract::{
    ApprovalBasis, ContinuityContractId, ContinuityContractV1, ContinuityRequirementId,
    ContinuityRequirementV1, ContractError, EquivalencePredicate, RequirementCriticality,
    ValidatedContinuityContractV1,
};
pub use observation::{
    DependencyBasis, DependencyClaimId, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
    ObservationEnvelopeV1, ObservationError, ObservationId,
};
pub use witness::{
    EvidenceClass, ObligationDispositionV1, QualifiedContinuityWitnessV1, TargetRealizationId,
    VerificationObligationId, VerificationPolicyEntryV1, VerificationPolicyId,
    VerificationPolicyV1, WitnessError, WitnessId, WitnessManifestId,
};
