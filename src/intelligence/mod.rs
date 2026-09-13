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
//! - Content-bound publication artifact for capability evidence

pub mod athena;
pub mod causal_consciousness;
pub mod causal_discovery;
pub mod nixos_causal;
pub mod reasoning_capability_artifact;
pub mod reasoning_capability_matrix;
pub mod reasoning_evaluator;
pub mod reasoning_qualification;

pub use causal_consciousness::{
    CausalAnalysisResult, CausalAttention, CausalConsciousness, CausalLTCBridge, GridSearchResult,
    HSICTest, LiveLearningRouter, RandomThresholdSearch, ThresholdTuner,
};
pub use causal_discovery::{CausalDirection, CausalDiscoveryEngine, MetaFeatures};
pub use nixos_causal::NixOSCausalAnalyzer;
pub use reasoning_capability_artifact::{
    CapabilityArtifactError, ReasoningCapabilityArtifact,
    REASONING_CAPABILITY_ARTIFACT_SCHEMA_VERSION,
};
pub use reasoning_capability_matrix::{
    build_capability_lane, build_capability_matrix, BaselineMetric, CapabilityLaneDescriptor,
    CapabilityLaneReport, CapabilityMatrixError, ContaminationStatus, HoldoutPolicy,
    ObservedResources, ReasoningCapabilityMatrix, ResourceBudget,
    REASONING_CAPABILITY_MATRIX_SCHEMA_VERSION,
};
pub use reasoning_evaluator::{
    aggregate_receipts, evaluate_episode, wilson_interval_95, CapabilitySlice, EpisodeJudgment,
    ProportionInterval, ReasoningEvaluatorError, TaskScore, REASONING_EVALUATOR_VERSION,
};
pub use reasoning_qualification::{
    AbstentionReason, AssumptionRecord, EvidenceRef, QualificationMetric,
    QualificationValidationError, ReasoningDecisionRecord, ReasoningDomain, ReasoningEpisode,
    ReasoningEpisodeId, ReasoningOutcome, ReasoningProblemRef, ReasoningQualificationReceipt,
    ResourceUsage, REASONING_EPISODE_SCHEMA_VERSION,
};
