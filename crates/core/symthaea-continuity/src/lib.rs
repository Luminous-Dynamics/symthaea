// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Computing-continuity kernel.
//!
//! This crate deliberately separates observed facts from dependency claims,
//! continuity requirements, verification evidence, counterfactual analysis,
//! and execution authority. None of these values grants migration authority by
//! itself.

#![deny(unsafe_code)]

pub mod auth_wire;
pub mod capability;
pub mod capability_activation;
pub mod capability_analysis_provenance;
pub mod capability_analysis_receipt;
pub mod capability_counterfactual;
pub mod capability_graph;
mod compose;
pub mod contract;
pub mod exact_policy;
pub mod observation;
pub mod profile_adoption;
pub mod verifier;
mod witness;

pub use auth_wire::{
    CONTINUITY_VERIFICATION_CLAIM_AUTH_SCHEMA, CONTINUITY_VERIFICATION_CLAIM_HASH_ALGORITHM,
    CONTINUITY_VERIFICATION_XENIA_PURPOSE, canonical_verification_claim_bytes,
    canonical_verification_claim_digest,
};
pub use capability::{
    CAPABILITY_DEFINITION_SCHEMA_V1, CapabilityDefinitionId, CapabilityDefinitionV1,
    CapabilityError, CapabilityId, CapabilityRequirementV1,
};
pub use capability_activation::{
    BlockedCapabilityActivationV1, CAPABILITY_ACTIVATION_ASSUMPTIONS_SCHEMA_V1,
    CapabilityActivationAssumptionsId, CapabilityActivationAssumptionsV1,
    CapabilityActivationClosureV1, CapabilityActivationError, CapabilityActivationRoundV1,
    UnsatisfiedCapabilityRequirementV1, ValidatedCapabilityActivationAssumptionsV1,
    derive_capability_activation_closure,
};
pub use capability_analysis_provenance::{
    BoundCapabilityActivationProvenanceV1, BoundCapabilityCounterfactualProvenanceV1,
    CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1, CAPABILITY_ACTIVATION_PROVENANCE_SCHEMA_V1,
    CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1,
    CAPABILITY_COUNTERFACTUAL_PROVENANCE_SCHEMA_V1, CapabilityActivationAlgorithmId,
    CapabilityActivationProvenanceId, CapabilityActivationProvenanceV1,
    CapabilityAnalysisProvenanceError, CapabilityCounterfactualAlgorithmId,
    CapabilityCounterfactualProvenanceId, CapabilityCounterfactualProvenanceV1,
    capability_activation_algorithm_id_v1, capability_counterfactual_algorithm_id_v1,
};
pub use capability_analysis_receipt::{
    CAPABILITY_ACTIVATION_CLOSURE_RECEIPT_SCHEMA_V1,
    CAPABILITY_COUNTERFACTUAL_FRONTIER_RECEIPT_SCHEMA_V1, CapabilityActivationClosureReceiptId,
    CapabilityActivationClosureReceiptV1, CapabilityAnalysisReceiptError,
    CapabilityCounterfactualFrontierReceiptId, CapabilityCounterfactualFrontierReceiptV1,
};
pub use capability_counterfactual::{
    CAPABILITY_COUNTERFACTUAL_CONFIG_SCHEMA_V1, CAPABILITY_COUNTERFACTUAL_QUERY_SCHEMA_V1,
    CapabilityCounterfactualConfigId, CapabilityCounterfactualConfigV1,
    CapabilityCounterfactualError, CapabilityCounterfactualFrontierV1,
    CapabilityCounterfactualOptionV1, CapabilityCounterfactualQueryId,
    CapabilityCounterfactualQueryV1, CapabilityCounterfactualTargetV1,
    ValidatedCapabilityCounterfactualConfigV1, ValidatedCapabilityCounterfactualQueryV1,
    derive_capability_counterfactual_frontier,
};
pub use capability_graph::{
    CAPABILITY_GRAPH_SNAPSHOT_SCHEMA_V1, CapabilityGraphError, CapabilityGraphSnapshotId,
    CapabilityGraphSnapshotV1, ValidatedCapabilityGraphV1,
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
pub use verifier::{
    AuthenticatedVerificationEvidenceId, VerificationAdmissionError, VerificationEvidenceClaimId,
    VerificationEvidenceClaimV1, VerificationOutcomeV1, VerifierProfileId, VerifierProfileV1,
};
pub use witness::{
    EvidenceClass, ObligationDispositionV1, QualifiedContinuityWitnessV1, TargetRealizationId,
    VerificationObligationId, VerificationPolicyEntryV1, WitnessError, WitnessId,
    WitnessManifestId,
};
