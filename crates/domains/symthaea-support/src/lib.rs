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
pub mod it_coverage;
pub mod it_qualification;
pub mod knowledge;
pub mod knowledge_source;
pub mod legacy_artifact_qualification;
pub mod legacy_computing;
#[cfg(test)]
mod legacy_computing_pack_tests;
pub mod legacy_platform_identity;
pub mod legacy_qualification_profile;
pub mod legacy_source_artifacts;
pub mod legacy_source_readiness;
#[cfg(feature = "logparse-adapter")]
pub mod logparse_adapter;
pub mod predictive;
pub mod privacy;
pub mod protocol_evidence;
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
pub use it_coverage::{
    seed_it_coverage_inventory_v1, ItCoverageErrorV1, ItCoverageInventoryV1,
    ItCoverageSignalKindV1, ItCoverageSignalV1, ItCoverageSourceV1, ItDomainCoverageV1,
    ItImplementationFootprintV1, ALL_IT_DOMAINS_V1, IT_COVERAGE_INVENTORY_SCHEMA_V1,
};
pub use it_qualification::{
    case_digest_v1, qualification_failures_v1, AdversarialConditionV1,
    DomainQualificationAssessmentV1, DomainQualificationPolicyV1, DomainQualificationStatusV1,
    FailingQualificationCaseV1, ItCompetencyLevelV1, ItCoverageCellV1, ItDomainV1,
    ItQualificationCaseV1, ItQualificationErrorV1, ItQualificationMatrixV1,
    ItQualificationResultV1, QualificationCaseIdV1, QualificationCaseKeyV1,
    QualificationEvidenceClassV1, QualificationFailureDimensionV1, QualificationMetricsV1,
    QualificationResultIdV1, QualificationRunContextV1, QualificationRunIdV1,
    QualificationThresholdV1,
};
pub use knowledge_source::{
    merge_knowledge_hits_v1, KnowledgeAuthorityClassV1, KnowledgeLifecycleV1,
    KnowledgeOriginV1, KnowledgeQueryPurposeV1, KnowledgeShareabilityV1,
    KnowledgeSourceErrorV1, KnowledgeStabilityV1, SupportKnowledgeHitV1,
    SupportKnowledgeQueryV1, SupportKnowledgeSourceV1,
};
pub use legacy_artifact_qualification::{
    assess_legacy_strong_qualification_v1, LegacyArtifactQualificationErrorV1,
    LegacyStrongQualificationAssessmentV1, LegacyStrongQualificationBlockerV1,
};
pub use legacy_computing::{
    seed_legacy_computing_pack_v1, LegacyComputingErrorV1, LegacyComputingPackV1,
    LegacyCoverageStateV1, LegacyKnowledgeAreaV1, LegacyPlatformProfileV1, LegacyPlatformV1,
    LegacyProcedureAuthorityV1, LegacyProcedureKindV1, LegacyProcedureStepV1,
    LegacyProcedureV1, LEGACY_COMPUTING_PACK_SCHEMA_V1,
};
pub use legacy_platform_identity::{
    assess_legacy_platform_identity_v1, legacy_platform_identity_spec_v1,
    legacy_platform_identity_v1, legacy_platform_scope_v1, LegacyPlatformIdentitySpecV1,
    LEGACY_PLATFORM_ECOSYSTEM_V1, LEGACY_PLATFORM_IDENTITY_SCHEMA_V1,
};
pub use legacy_qualification_profile::{
    area_tag, assess_legacy_qualification_profile_v1, exhaustive_legacy_qualification_profile_v1,
    platform_tag, LegacyQualificationBlockerV1, LegacyQualificationProfileAssessmentV1,
    LegacyQualificationProfileErrorV1, LegacyQualificationProfileV1,
    LegacyQualificationRequirementAssessmentV1, LegacyQualificationRequirementV1,
    LEGACY_QUALIFICATION_PROFILE_SCHEMA_V1,
};
pub use legacy_source_artifacts::{
    assess_legacy_source_artifact_readiness_v1, LegacyArtifactAccessPolicyV1,
    LegacyArtifactStorageClassV1, LegacySourceArtifactErrorV1, LegacySourceArtifactLedgerV1,
    LegacySourceArtifactReadinessV1, LegacySourceArtifactRefV1,
    LEGACY_SOURCE_ARTIFACT_LEDGER_SCHEMA_V1,
};
pub use legacy_source_readiness::{
    admit_legacy_knowledge_use_v1, assess_legacy_source_readiness_v1,
    LegacyKnowledgeUseClassV1, LegacySourceCaptureClassV1, LegacySourceReadinessAssessmentV1,
    LegacySourceReadinessErrorV1, LegacySourceReadinessItemV1,
};
#[cfg(feature = "logparse-adapter")]
pub use logparse_adapter::{
    LogObservationAdapterConfigV1, LogObservationAdapterError, LogObservationAdapterV1,
    LogObservationPolicyV1,
};
pub use protocol_evidence::{
    DnsProtocolEventV1, IcmpProtocolEventV1, ProtocolEvidenceBasisV1,
    ProtocolEvidenceErrorV1, ProtocolEvidenceRecordV1, ProtocolEventV1,
    ProtocolObservationAdapterConfigV1, ProtocolObservationAdapterV1,
    ProtocolObservationRetentionPolicyV1, ProtocolSubjectRoleV1, QuicProtocolEventV1,
    TcpProtocolEventV1, TlsProtocolEventV1, TransportProtocolV1,
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
