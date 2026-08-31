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
pub mod manifest;
pub mod observation;
pub mod registry;
pub mod topology;
pub mod traits;

pub use conformance::{
    ReadOnlyConformanceCounters, evaluate_read_only_conformance,
};
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
pub use topology::{
    DiscoveredEntity, DiscoverySnapshot, EntityRelation, RelationKind,
};
pub use traits::{
    Discoverer, DiscoveryRequest, IntegrationError, IntegrationFuture, IntegrationIdentity,
    ObservationRequest, Observer,
};
