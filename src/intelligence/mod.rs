// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Intelligence Module
//!
//! HDC-powered intelligence features:
//! - Bivariate causal discovery (71.3% accuracy on Tübingen benchmark)
//! - Causal consciousness integration (HSIC, attention, LTC bridge)
//! - Evidence-first reasoning qualification records, evaluation, and receipts
//! - Cross-domain capability matrix with explicit holdout, contamination, and resource policy
//! - Portable lane bundles and content-bound publication artifacts for capability evidence
//! - Decomposed metacognitive calibration, abstention, assumption, and revision qualification
//! - Public development probe for evidence-revision direction and epistemic remediation
//! - Public development probe for meta-reasoning control-flow and state persistence
//! - Public development probe for context-optimizer objective plumbing
//! - Canonical typed objective selection over measured primitive coordinates
//! - Two-phase longitudinal metacognitive state with pre-outcome freezing
//! - Canonical per-subject reasoning orchestration with plasticity held at baseline
//! - Evidence-backed competing-context assessment with ambiguity-preserving robust selection
//! - Canonical kernel V2 with robust multi-context selection and content-bound decisions
//! - Automatic RQ episode publication from validated canonical decisions
//! - Live measurement-only shadowing of the production meta-reasoner through canonical V2
//! - Evidence-backed objective intervals that preserve unknown axes instead of fabricating neutrality
//! - Evidence-seeking V3 planning with Select / NeedEvidence / Abstain outcomes
//! - Live primitive identity recovery with explicit objective-evidence gaps
//! - Append-only evidence acquisition sessions with hash-chained deterministic replanning
//! - Direct ActivePrimitive evidence preservation with HDC identity auditing
//! - Live V3 ActivePrimitive shadow telemetry without behavior authority

pub mod athena;
pub mod causal_consciousness;
pub mod causal_discovery;
pub mod nixos_causal;
pub mod reasoning_active_primitive_evidence;
pub mod reasoning_capability_artifact;
pub mod reasoning_capability_bundle;
pub mod reasoning_capability_matrix;
pub mod reasoning_context_competition;
pub mod reasoning_episode_emitter;
pub mod reasoning_evaluator;
pub mod reasoning_evaluator_policy;
pub mod reasoning_evidence_revision_probe;
pub mod reasoning_evidence_seeking;
pub mod reasoning_evidence_session;
pub mod reasoning_kernel;
pub mod reasoning_kernel_v2;
pub mod reasoning_live_primitive_evidence;
pub mod reasoning_meta_control_probe;
pub mod reasoning_meta_state;
pub mod reasoning_metacognition;
pub mod reasoning_objective_core;
pub mod reasoning_objective_evidence;
pub mod reasoning_objective_plumbing_probe;
pub mod reasoning_qualification;
pub mod reasoning_shadow_meta;

pub use causal_consciousness::{
    CausalAnalysisResult, CausalAttention, CausalConsciousness, CausalLTCBridge, GridSearchResult,
    HSICTest, LiveLearningRouter, RandomThresholdSearch, ThresholdTuner,
};
pub use causal_discovery::{CausalDirection, CausalDiscoveryEngine, MetaFeatures};
pub use nixos_causal::NixOSCausalAnalyzer;
pub use reasoning_active_primitive_evidence::{
    adapt_active_primitive_evidence, plan_active_primitive_evidence,
    ActivePrimitiveActivationEvidence, ActivePrimitiveEvidenceError, ActivePrimitiveEvidenceProfile,
    ActivePrimitiveEvidenceReport, ActivePrimitiveIdentityAudit, ObservedPrimitiveMetadata,
    ACTIVE_PRIMITIVE_EVIDENCE_ADAPTER_VERSION,
};
pub use reasoning_capability_artifact::{
    CapabilityArtifactError, ReasoningCapabilityArtifact,
    REASONING_CAPABILITY_ARTIFACT_SCHEMA_VERSION, REASONING_CAPABILITY_ARTIFACT_SERIALIZATION,
};
pub use reasoning_capability_bundle::{
    CapabilityLaneBundle, CapabilityLaneBundleError,
    REASONING_CAPABILITY_LANE_BUNDLE_SCHEMA_VERSION,
};
pub use reasoning_capability_matrix::{
    build_capability_lane, build_capability_matrix, BaselineMetric, CapabilityLaneDescriptor,
    CapabilityLaneReport, CapabilityMatrixError, ContaminationStatus, HoldoutPolicy,
    ObservedResources, ReasoningCapabilityMatrix, ResourceBudget,
    REASONING_CAPABILITY_MATRIX_SCHEMA_VERSION,
};
pub use reasoning_context_competition::{
    assess_and_select, assess_contexts, select_candidate_robustly, ContextAssessmentReport,
    ContextCompetitionError, ContextCompetitionPolicy, ContextHypothesis, ContextResolution,
    PerContextCandidateScore, RobustCandidateEvaluation, RobustContextSelectionReport,
    CONTEXT_COMPETITION_VERSION,
};
pub use reasoning_episode_emitter::{
    emit_reasoning_episode, EmittedReasoningEpisode, EpisodeEmissionError,
    ReasoningEpisodePublicationInput, CANONICAL_RQ_EMITTER_VERSION,
};
pub use reasoning_evaluator::{
    aggregate_receipts, evaluate_episode, wilson_interval_95, CapabilitySlice, EpisodeJudgment,
    ProportionInterval, ReasoningEvaluatorError, TaskScore, REASONING_EVALUATOR_VERSION,
};
pub use reasoning_evaluator_policy::{
    evaluate_episode_with_policy, ExactScoringPolicy,
    REASONING_EVALUATOR_OUTCOME_SEMANTIC_VERSION,
};
pub use reasoning_evidence_revision_probe::{
    run_evidence_revision_probe, ConflictActionObservation, EvidenceRevisionProbeReport,
    EVIDENCE_REVISION_PROBE_VERSION,
};
pub use reasoning_evidence_seeking::{
    plan_with_evidence, EvidenceRequest, EvidenceRequestKind, EvidenceSeekingAbstention,
    EvidenceSeekingOutcome, EvidenceSeekingPlanReport, EvidenceSeekingPlannerError,
    EVIDENCE_SEEKING_PLANNER_VERSION,
};
pub use reasoning_evidence_session::{
    EvidenceAcquisitionError, EvidenceAcquisitionEvent, EvidenceAcquisitionEventRecord,
    EvidenceAcquisitionSession, EvidenceAcquisitionSnapshot,
    EVIDENCE_ACQUISITION_SESSION_SCHEMA_VERSION,
};
pub use reasoning_kernel::{
    CanonicalReasoningDecision, CanonicalReasoningInput, CanonicalReasoningKernel,
    CanonicalSubjectCheckpoint, ReasoningKernelError, BASELINE_PLASTICITY_MULTIPLIER,
    CANONICAL_REASONING_KERNEL_VERSION,
};
pub use reasoning_kernel_v2::{
    CanonicalReasoningDecisionV2, CanonicalReasoningInputV2, CanonicalReasoningKernelV2,
    CanonicalSubjectCheckpointV2, ReasoningKernelV2Error,
    CANONICAL_REASONING_BASELINE_PLASTICITY, CANONICAL_REASONING_KERNEL_V2_VERSION,
};
pub use reasoning_live_primitive_evidence::{
    plan_live_primitive_evidence, recover_live_primitive_evidence, LiveCandidateIdentityAudit,
    LivePrimitiveEvidenceError, LivePrimitiveEvidenceProfile, LivePrimitiveEvidenceReport,
    RegistryPrimitiveMetadata, LIVE_PRIMITIVE_EVIDENCE_BRIDGE_VERSION,
};
pub use reasoning_meta_control_probe::{
    run_meta_control_probe, MetaControlObservation, MetaControlProbeReport,
    META_CONTROL_PROBE_VERSION,
};
pub use reasoning_meta_state::{
    MetaCommitReport, MetaEpistemicState, MetaEpisodeObservation, MetaEpisodeRecord,
    MetaOutcomeFeedback, MetaSignalRevision, MetaStateAuthority, MetaStateCheckpoint,
    MetaStateError, MetaSupportSignals, OutcomeCommitReport,
    META_EPISTEMIC_STATE_SCHEMA_VERSION,
};
pub use reasoning_metacognition::{
    evaluate_metacognition, BinaryDetectionReport, ConfidenceRevisionDirection,
    ConfidenceRevisionObservation, CorrectnessPrediction, MetacognitionEvaluationError,
    MetacognitionReport, SelectiveRiskPoint, WeakAssumptionObservation,
    METACOGNITION_EVALUATOR_VERSION,
};
pub use reasoning_objective_core::{
    select_candidate_by_objectives, CandidateObjectiveEvaluation, ObjectiveCoreError,
    ObjectiveKind, ObjectiveSelectionReport, ObjectiveVector,
    ObjectiveWeights as CanonicalObjectiveWeights, REASONING_OBJECTIVE_CORE_VERSION,
};
pub use reasoning_objective_evidence::{
    select_from_objective_evidence, CandidateEvidenceEvaluation, CandidateObjectiveEvidence,
    ContextualEvidenceScore, ObjectiveEvidence, ObjectiveEvidenceError,
    ObjectiveEvidenceSelection, ObjectiveEvidenceSelectionReport, ObjectiveEvidenceStatus,
    ObjectiveUnknownReason, ScoreInterval, OBJECTIVE_EVIDENCE_SELECTOR_VERSION,
};
pub use reasoning_objective_plumbing_probe::{
    run_objective_plumbing_probe, ObjectivePlumbingObservation,
    ObjectivePlumbingProbeReport, OBJECTIVE_PLUMBING_PROBE_VERSION,
};
pub use reasoning_qualification::{
    AbstentionReason, AssumptionRecord, EvidenceRef, QualificationMetric,
    QualificationValidationError, ReasoningDecisionRecord, ReasoningDomain, ReasoningEpisode,
    ReasoningEpisodeId, ReasoningOutcome, ReasoningProblemRef, ReasoningQualificationReceipt,
    ResourceUsage, REASONING_EPISODE_SCHEMA_VERSION,
};
pub use reasoning_shadow_meta::{
    ObjectiveSpread, ShadowMetaObservation, ShadowMetaStats, ShadowQualifiedMetaReasoner,
    V3ShadowObservation, V3ShadowOutcomeKind, LIVE_META_SHADOW_VERSION,
};
