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
pub mod identity_provider;
pub mod independence;
pub mod limits;
pub mod manifest;
pub mod observation;
pub mod registry;
pub mod resolution;
pub mod resolution_pipeline;
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
pub use identity_provider::{
    IDENTITY_DISCOVERY_CAPABILITY, IdentityAdmissionError, IdentityLimits, IdentityProvider,
    IdentityRequest, IdentitySnapshot, IdentitySnapshotError,
};
pub use independence::{
    IndependenceAssessment, IndependenceAssessmentError, IndependenceMetadataConflict,
    assess_independence,
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
    resolve_registry_identity_snapshots_with_limits, source_qualified_claim_id,
};
pub use topology::{
    DiscoveredEntity, DiscoverySnapshot, EntityRelation, RelationBasis, RelationKind,
    TopologyValidationError,
};
pub use topology_limits::{TopologyBudgetError, TopologyLimits};
pub use traits::{
    Discoverer, DiscoveryRequest, IntegrationError, IntegrationFuture, IntegrationIdentity,
    ObservationRequest, Observer,
};
