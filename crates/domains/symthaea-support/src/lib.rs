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
pub mod legacy_aix;
pub mod legacy_aix_scenarios;
pub mod legacy_artifact_qualification;
pub mod legacy_computing;
#[cfg(test)]
mod legacy_computing_pack_tests;
pub mod legacy_hpux;
pub mod legacy_hpux_scenarios;
pub mod legacy_ibmi;
pub mod legacy_ibmi_scenarios;
pub mod legacy_platform_identity;
pub mod legacy_portfolio;
pub mod legacy_qualification_profile;
pub mod legacy_solaris;
pub mod legacy_solaris_scenarios;
pub mod legacy_source_artifacts;
pub mod legacy_source_lineage;
pub mod legacy_source_readiness;
pub mod legacy_zos;
pub mod legacy_zos_scenarios;
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
pub use legacy_aix::{
    enrich_legacy_aix_foundation_v1, AixEvidenceSignalV1, AixFailureModeV1, AixFoundationV1,
    AixMechanismKindV1, AixMechanismModelV1, LegacyAixErrorV1,
    LEGACY_AIX_FOUNDATION_SCHEMA_V1,
};
pub use legacy_aix_scenarios::{
    register_aix_qualification_cases_v1, seed_aix_qualification_scenarios_v1,
    AixQualificationScenarioV1, AixScenarioActionClassV1, AixScenarioActionV1,
    AixScenarioErrorV1, AixScenarioEvidenceCurrentnessV1, AixScenarioEvidenceV1,
    LEGACY_AIX_SCENARIO_SCHEMA_V1,
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
pub use legacy_hpux::{
    enrich_legacy_hpux_foundation_v1, HpuxEvidenceSignalV1, HpuxFailureModeV1,
    HpuxFoundationV1, HpuxMechanismKindV1, HpuxMechanismModelV1, LegacyHpuxErrorV1,
    LEGACY_HPUX_FOUNDATION_SCHEMA_V1,
};
pub use legacy_hpux_scenarios::{
    register_hpux_qualification_cases_v1, seed_hpux_qualification_scenarios_v1,
    HpuxQualificationScenarioV1, HpuxScenarioActionClassV1, HpuxScenarioActionV1,
    HpuxScenarioErrorV1, HpuxScenarioEvidenceCurrentnessV1, HpuxScenarioEvidenceV1,
    LEGACY_HPUX_SCENARIO_SCHEMA_V1,
};
pub use legacy_ibmi::{
    enrich_legacy_ibmi_foundation_v1, IbmiEvidenceSignalV1, IbmiFailureModeV1,
    IbmiFoundationV1, IbmiMechanismKindV1, IbmiMechanismModelV1, LegacyIbmiErrorV1,
    LEGACY_IBMI_FOUNDATION_SCHEMA_V1,
};
pub use legacy_ibmi_scenarios::{
    register_ibmi_qualification_cases_v1, seed_ibmi_qualification_scenarios_v1,
    IbmiQualificationScenarioV1, IbmiScenarioActionClassV1, IbmiScenarioActionV1,
    IbmiScenarioErrorV1, IbmiScenarioEvidenceCurrentnessV1, IbmiScenarioEvidenceV1,
    LEGACY_IBMI_SCENARIO_SCHEMA_V1,
};
pub use legacy_platform_identity::{
    assess_legacy_platform_identity_v1, legacy_platform_identity_spec_v1,
    legacy_platform_identity_v1, legacy_platform_scope_v1, LegacyPlatformIdentitySpecV1,
    LEGACY_PLATFORM_ECOSYSTEM_V1, LEGACY_PLATFORM_IDENTITY_SCHEMA_V1,
};
pub use legacy_portfolio::{
    build_legacy_five_platform_portfolio_v1, LegacyPortfolioErrorV1, LegacyPortfolioSummaryV1,
    LEGACY_PORTFOLIO_SCHEMA_V1,
};
pub use legacy_qualification_profile::{
    area_tag, assess_legacy_qualification_profile_v1, exhaustive_legacy_qualification_profile_v1,
    platform_tag, LegacyQualificationBlockerV1, LegacyQualificationProfileAssessmentV1,
    LegacyQualificationProfileErrorV1, LegacyQualificationProfileV1,
    LegacyQualificationRequirementAssessmentV1, LegacyQualificationRequirementV1,
    LEGACY_QUALIFICATION_PROFILE_SCHEMA_V1,
};
pub use legacy_solaris::{
    enrich_legacy_solaris_foundation_v1, LegacySolarisErrorV1, SolarisEvidenceSignalV1,
    SolarisFailureModeV1, SolarisFoundationV1, SolarisMechanismKindV1,
    SolarisMechanismModelV1, LEGACY_SOLARIS_FOUNDATION_SCHEMA_V1,
};
pub use legacy_solaris_scenarios::{
    register_solaris_qualification_cases_v1, seed_solaris_qualification_scenarios_v1,
    SolarisQualificationScenarioV1, SolarisScenarioActionClassV1, SolarisScenarioActionV1,
    SolarisScenarioErrorV1, SolarisScenarioEvidenceCurrentnessV1, SolarisScenarioEvidenceV1,
    LEGACY_SOLARIS_SCENARIO_SCHEMA_V1,
};
pub use legacy_source_artifacts::{
    assess_legacy_source_artifact_readiness_v1, LegacyArtifactAccessPolicyV1,
    LegacyArtifactStorageClassV1, LegacySourceArtifactErrorV1, LegacySourceArtifactLedgerV1,
    LegacySourceArtifactReadinessV1, LegacySourceArtifactRefV1,
    LEGACY_SOURCE_ARTIFACT_LEDGER_SCHEMA_V1,
};
pub use legacy_source_lineage::{
    assess_legacy_source_lineage_v1, summarize_legacy_claim_lineage_v1,
    LegacyClaimLineageSummaryV1, LegacySourceLineageAssessmentV1, LegacySourceLineageErrorV1,
    LegacySourceLineageGroupV1, LEGACY_SOURCE_LINEAGE_SCHEMA_V1,
};
pub use legacy_source_readiness::{
    admit_legacy_knowledge_use_v1, assess_legacy_source_readiness_v1,
    LegacyKnowledgeUseClassV1, LegacySourceCaptureClassV1, LegacySourceReadinessAssessmentV1,
    LegacySourceReadinessErrorV1, LegacySourceReadinessItemV1,
};
pub use legacy_zos::{
    enrich_legacy_zos_foundation_v1, LegacyZosErrorV1, ZosEvidenceSignalV1,
    ZosFailureModeV1, ZosFoundationV1, ZosMechanismKindV1, ZosMechanismModelV1,
    LEGACY_ZOS_FOUNDATION_SCHEMA_V1,
};
pub use legacy_zos_scenarios::{
    register_zos_qualification_cases_v1, seed_zos_qualification_scenarios_v1,
    ZosQualificationScenarioV1, ZosScenarioActionClassV1, ZosScenarioActionV1,
    ZosScenarioErrorV1, ZosScenarioEvidenceCurrentnessV1, ZosScenarioEvidenceV1,
    LEGACY_ZOS_SCENARIO_SCHEMA_V1,
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
