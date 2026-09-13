// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Intelligence Module
//!
//! HDC-powered intelligence features:
//! - Bivariate causal discovery (71.3% accuracy on Tübingen benchmark)
//! - Causal consciousness integration (HSIC, attention, LTC bridge)
//! - Evidence-first reasoning qualification records, evaluation, and receipts

pub mod athena;
pub mod causal_consciousness;
pub mod causal_discovery;
pub mod nixos_causal;
pub mod reasoning_evaluator;
pub mod reasoning_evaluator_policy;
pub mod reasoning_qualification;

pub use causal_consciousness::{
    CausalAnalysisResult, CausalAttention, CausalConsciousness, CausalLTCBridge, GridSearchResult,
    HSICTest, LiveLearningRouter, RandomThresholdSearch, ThresholdTuner,
};
pub use causal_discovery::{CausalDirection, CausalDiscoveryEngine, MetaFeatures};
pub use nixos_causal::NixOSCausalAnalyzer;
pub use reasoning_evaluator::{
    aggregate_receipts, evaluate_episode, wilson_interval_95, CapabilitySlice, EpisodeJudgment,
    ProportionInterval, ReasoningEvaluatorError, TaskScore, REASONING_EVALUATOR_VERSION,
};
pub use reasoning_evaluator_policy::{
    evaluate_episode_with_policy, ExactScoringPolicy,
    REASONING_EVALUATOR_OUTCOME_SEMANTIC_VERSION,
};
pub use reasoning_qualification::{
    AbstentionReason, AssumptionRecord, EvidenceRef, QualificationMetric,
    QualificationValidationError, ReasoningDecisionRecord, ReasoningDomain, ReasoningEpisode,
    ReasoningEpisodeId, ReasoningOutcome, ReasoningProblemRef, ReasoningQualificationReceipt,
    ResourceUsage, REASONING_EPISODE_SCHEMA_VERSION,
};
