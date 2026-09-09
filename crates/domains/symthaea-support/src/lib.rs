// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Support — IT support intelligence sub-crate
//!
//! Provides triage, diagnostics, knowledge management, privacy scrubbing,
//! action engine, predictive engine, and evidence-bound IT system intelligence
//! for universal IT support.

pub mod actions;
pub mod change_timeline;
pub mod dependency_impact;
pub mod diagnostic_beliefs;
pub mod diagnostics;
pub mod federation;
pub mod knowledge;
pub mod knowledge_source;
#[cfg(feature = "logparse-adapter")]
pub mod logparse_adapter;
pub mod predictive;
pub mod privacy;
pub mod scrubber;
pub mod standards_registry;
pub mod system_state;
pub mod technology;
pub mod telemetry;
pub mod telemetry_adapter;
pub mod triage;
pub mod types;

pub use change_timeline::{
    temporal_relation, ChangeClockV1, ChangeId, ChangeKindV1, ChangeTimelineError,
    ChangeTimelineV1, FailureWindowV1, SystemChangeV1, TemporalRelationV1,
};
pub use dependency_impact::{
    analyze_dependency_impact_v1, DependencyImpactAnalysisV1, DependencyImpactErrorV1,
    DependencyImpactPolicyV1, ImpactEvidenceRequirementV1, ImpactEvidenceSummaryV1,
    ImpactPathStepV1, ImpactPropagationRuleV1, ImpactTraversalV1, PotentialImpactV1,
};
pub use diagnostic_beliefs::{
    rank_tests_by_information_gain, CausalHypothesisV1, DiagnosticBeliefError,
    DiagnosticOutcomeId, DiagnosticTestId, DiagnosticTestModelV1, ExpectedInformationGainV1,
    HypothesisDistributionV1, HypothesisId, HypothesisStatusV1,
};
pub use knowledge_source::{
    merge_knowledge_hits_v1, KnowledgeAuthorityClassV1, KnowledgeLifecycleV1,
    KnowledgeOriginV1, KnowledgeQueryPurposeV1, KnowledgeShareabilityV1,
    KnowledgeSourceErrorV1, KnowledgeStabilityV1, SupportKnowledgeHitV1,
    SupportKnowledgeQueryV1, SupportKnowledgeSourceV1,
};
#[cfg(feature = "logparse-adapter")]
pub use logparse_adapter::{
    LogObservationAdapterConfigV1, LogObservationAdapterError, LogObservationAdapterV1,
    LogObservationPolicyV1,
};
pub use standards_registry::{
    ClaimModalityV1, SourceCaptureV1, SourceDocumentIdV1, SourceDocumentKindV1,
    SourceRelationKindV1, SourceRelationV1, SourceSnapshotIdV1, StandardsRegistryErrorV1,
    TechnicalClaimIdV1, TechnicalKnowledgeClaimV1, TechnicalPublisherV1,
    TechnicalSourceDocumentV1, TechnicalSourceLocatorV1, TechnicalSourceSnapshotV1,
    TechnicalStandardsRegistryV1,
};
pub use system_state::{
    CurrentnessStatusV1, EntityId, EntityKindV1, ObservationClockV1, ObservationId,
    ObservationProvenanceV1, ObservationSourceKindV1, RelationId, RelationKindV1, StateValueV1,
    SystemEntityV1, SystemObservationV1, SystemRelationV1, SystemStateGraphError,
    SystemStateGraphV1,
};
pub use technology::{
    ApplicabilityAssessmentV1, ApplicabilityScopeV1, ApplicabilityStatusV1, StringSelectorV1,
    TechnologyIdentityError, TechnologyIdentityV1,
};
pub use telemetry_adapter::{
    ExternalTelemetryRecordV1, TelemetryAdapterErrorV1, TelemetryObservationAdapterConfigV1,
    TelemetryObservationAdapterV1, TelemetryRetentionPolicyV1, TelemetrySchemaIdentityV1,
    TelemetrySchemaStabilityV1, TelemetrySignalKindV1,
};
