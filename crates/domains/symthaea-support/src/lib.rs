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
pub mod golden_incidents;
pub mod golden_incidents_v2;
pub mod golden_run_protocol;
pub mod golden_solver_view_v2;
pub mod it_qualification;
pub mod knowledge;
pub mod knowledge_source;
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
pub use golden_incidents::{
    seed_golden_incidents_v1, DiagnosticActionRiskV1, DiagnosticAuthorityRequirementV1,
    GoldenDiagnosticActionV1, GoldenIncidentCaseV1, GoldenIncidentCorpusV1,
    GoldenIncidentErrorV1, GoldenIncidentEvidenceKindV1, GoldenIncidentEvidenceV1,
    GOLDEN_INCIDENT_SCHEMA_V1,
};
pub use golden_incidents_v2::{
    seed_golden_incidents_v2, DiagnosticAuthorityRequirementV2, GoldenDiagnosticActionV2,
    GoldenEvidenceCurrentnessV2, GoldenIncidentCaseV2, GoldenIncidentCorpusV2,
    GoldenIncidentErrorV2, GoldenIncidentEvidenceV2, GOLDEN_INCIDENT_SCHEMA_V2,
};
pub use golden_run_protocol::{
    golden_grading_artifact_digest_v1, golden_solver_view_digest_v1,
    GoldenAbstentionReasonV1, GoldenDiagnosticRequestV1, GoldenFindingDispositionV1,
    GoldenPresentedEvidenceSourceV1, GoldenPresentedEvidenceV1, GoldenRunProtocolErrorV1,
    GoldenRunTranscriptV1, GoldenSolverAbstentionV1, GoldenSolverFindingV1,
    GoldenSolverSubmissionV1, GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1,
    GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1,
};
pub use golden_solver_view_v2::{GoldenSolverCorpusV2, GoldenSolverIncidentV2};
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
