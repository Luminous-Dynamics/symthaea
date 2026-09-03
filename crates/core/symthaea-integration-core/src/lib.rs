// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Domain-neutral integration contracts for Symthaea's infrastructure fabric.
//!
//! This crate intentionally separates two concepts:
//! - runtime **observations** about external infrastructure, represented by
//!   [`ObservationEnvelope`]; and
//! - research/qualification **evidence** about whether an integration behaved
//!   as declared, represented by `symthaea-evidence-plane`.
//!
//! v0.1 is deliberately read-only. Observation and discovery contracts are
//! present; mutation/execution traits are deferred to a later, separately
//! qualified tranche so importing this crate cannot accidentally confer
//! actuation authority.

#![forbid(unsafe_code)]

pub mod conformance;
mod identity;
pub mod identity_normalization;
pub mod identity_provider;
pub mod independence;
pub mod independence_authority;
pub mod independence_reasoning;
pub mod limits;
pub mod manifest;
pub mod observation;
pub mod registry;
pub mod resolution;
pub mod resolution_pipeline;
pub mod semantic_identity;
pub mod state;
pub mod state_history;
pub mod state_origin;
mod state_registry;
pub mod state_temporal;
pub mod topology;
pub mod topology_limits;
mod topology_registry;
pub mod traits;

pub use conformance::{
    ReadOnlyConformanceCounters, evaluate_read_only_conformance,
};
pub use identity::{
    EntityPair, EntityResolutionProposal, ExternalIdentifier, IdentifierMatchEvidence,
    IdentifierStability, IdentifierUniqueness, IdentityClaim, IdentityClaimSource,
    IdentityStrength, IdentityValidationError, ResolutionStatus, SeparationClaim,
    assess_entity_pair,
};
pub use identity_normalization::{
    IdentityNormalizationError, LEGACY_K8S_UID_SCHEME, kubernetes_cluster_uid_from_topology,
    normalize_kubernetes_uid_snapshot,
};
pub use identity_provider::{
    IDENTITY_DISCOVERY_CAPABILITY, IdentityAdmissionError, IdentityLimits, IdentityProvider,
    IdentityRequest, IdentitySnapshot, IdentitySnapshotError,
};
pub use independence::{
    IndependenceAssessment, IndependenceAssessmentError, IndependenceMetadataConflict,
    assess_independence,
};
pub use independence_authority::{
    EvidenceLineageRef, IndependenceAttestation, IndependenceAttestationError,
    IndependenceAttestationSet, IndependenceAttestationSetError, IndependenceAuthorityPolicy,
    IndependenceAuthorityQualification, IndependenceBasis,
    INDEPENDENCE_ATTESTATION_SCHEMA_VERSION,
};
pub use independence_reasoning::{
    IndependenceCliqueWitness, QualifiedIndependenceAssessment, QualifiedIndependenceError,
    QualifiedIndependenceReasoningPolicy, assess_qualified_independence,
};
pub use limits::{ObservationBudgetError, ObservationLimits};
pub use manifest::{
    AccessMode, CapabilityClass, CapabilityDeclaration, CredentialKind,
    CredentialRequirement, IntegrationId, IntegrationManifest, ManifestValidationError,
    MaturityLevel, RiskClass, INTEGRATION_MANIFEST_SCHEMA_VERSION,
};
pub use observation::{
    BatchValidationError, EntityRef, LineageRelationship, ObservationBatch,
    ObservationEnvelope, ObservationId, ObservationKind, ObservationLineage,
    ObservationQuality, ObservationSource, ObservationState, ObservationValidationError,
    ObservationValue, TransformStep, OBSERVATION_SCHEMA_VERSION,
};
pub use registry::{IntegrationRegistry, RegistryError};
pub use resolution::{
    EntityResolutionBatch, ResolutionError, ResolutionLimits, attention_required,
    resolve_identity_claims, resolve_identity_claims_with_limits,
};
pub use resolution_pipeline::{
    ResolutionPipelineError, resolve_registry_identity_snapshots,
    resolve_registry_identity_snapshots_with_limits, resolve_registry_kubernetes_uid_snapshots,
    resolve_registry_kubernetes_uid_snapshots_with_limits, source_qualified_claim_id,
};
pub use semantic_identity::{
    K8S_CLUSTER_UID_SCHEME, K8S_CRONJOB_UID_SCHEME, K8S_DAEMONSET_UID_SCHEME,
    K8S_DEPLOYMENT_UID_SCHEME, K8S_JOB_UID_SCHEME, K8S_NODE_UID_SCHEME,
    K8S_OBJECT_UID_SCHEME, K8S_POD_UID_SCHEME, K8S_REPLICASET_UID_SCHEME,
    K8S_SERVICE_UID_SCHEME, K8S_STATEFULSET_UID_SCHEME,
    kubernetes_cluster_uid_identifier, kubernetes_cluster_uid_scope,
    kubernetes_object_uid_identifier, kubernetes_uid_scheme,
};
pub use state::{
    StateAssessment, StateAssessmentError, StateAssessmentStatus, StateAssertion,
    StateAssertionSource, StateBudgetError, StateComparisonPolicy, StateLimits, StateRole,
    StateSnapshot, StateValidationError, StateValue, assess_state_dimension,
};
pub use state_history::{
    DesiredStateContinuity, DriftContinuity, HistoricalStateAssessment, StateHistory,
    StateHistoryError, StateHistoryLimits, assess_state_dimension_with_history,
    desired_state_continuity, drift_state_continuity,
};
pub use state_origin::{
    DESIRED_STATE_ORIGIN_ATTRIBUTE, DesiredStateOrigin, DesiredStateOriginEvidence,
    StateOriginError, validate_state_snapshot_origins,
};
pub use state_temporal::{
    TemporalStateAssessment, TemporalStateAssessmentError, TemporalStatePolicy,
    TemporalStateStatus, assess_state_dimension_temporally,
};
pub use topology::{
    DiscoveredEntity, DiscoverySnapshot, EntityRelation, RelationBasis, RelationKind,
    TopologyValidationError,
};
pub use topology_limits::{TopologyBudgetError, TopologyLimits};
pub use traits::{
    COMPLETE_DISCOVERY_CAPABILITY, Discoverer, DiscoveryRequest, IntegrationError,
    IntegrationFuture, IntegrationIdentity, ObservationRequest, Observer,
};
