// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WisdomEngine: Full Stack Wisdom / Holistic Epistemics Architecture
//!
//! This module implements the most advanced epistemic architecture for decentralized
//! networks: a Self-Correcting Civilizational Operating System that moves beyond
//! "Truth" to "Justified, Wise Belief."
//!
//! ## The Four Core Components
//!
//! 1. **The Lens** (HarmonicWeights) - The Eight Harmonies as epistemic lenses
//! 2. **The Setting** (CommunityProfile) - Per-community harmonic weight profiles
//! 3. **The Mirror** (DiversityAuditor) - Bias detection and diversity metrics
//! 4. **The Hand** (ReparationsManager) - Power corrections for marginalized voices
//!
//! ## Extended Features
//!
//! 5. **Emergent Weights** - Adaptive learning from outcomes
//! 6. **Multi-Epistemology** - Support for diverse knowledge systems
//! 7. **Causal Graph** - Prediction tracking and reality-checking
//! 8. **SymthaeaCausalBridge** - Scientific Method for AI pattern learning
//!
//! ## Module Organization
//!
//! - `core` - WisdomEngine, basic types, traits, epistemology
//! - `domain` - Domain hierarchy and cross-domain discovery
//! - `composition` - Pattern composition and synergy tracking
//! - `trust` - Agent trust weighting for patterns
//! - `collective` - Collective pattern integration
//! - `lifecycle` - Pattern lifecycle management
//! - `versioning` - Pattern version history
//! - `dependencies` - Pattern dependencies and prerequisites
//! - `explainability` - Human-readable pattern explanations
//! - `symthaea` - SymthaeaCausalBridge and scientific method for AI
//! - `recommendations` - Pattern recommendations engine
//! - `anomaly` - Anomaly detection system
//! - `succession` - Pattern succession automation
//! - `similarity` - Pattern similarity metrics and clustering
//! - `associative_learner` - HDC-grounded associative learning

pub mod anomaly;
pub mod associative_learner;
pub mod collective;
pub mod composition;
pub mod core;
pub mod dependencies;
pub mod domain;
pub mod explainability;
pub mod lifecycle;
pub mod recommendations;
pub mod similarity;
pub mod succession;
pub mod symthaea;
pub mod trust;
pub mod versioning;

// Python bindings (only compiled when python feature is enabled)
#[cfg(feature = "python")]
pub mod python_bindings;

// ==============================================================================
// Common Type Aliases
// ==============================================================================

/// Unique identifier for communities
pub type CommunityId = u64;

/// Unique identifier for predictions
pub type PredictionId = u64;

/// Unique identifier for causal nodes
pub type CausalNodeId = u64;

/// Unique identifier for domains
pub type DomainId = u64;

/// Unique identifier for composites
pub type CompositeId = u64;

/// Unique identifier for patterns in the Symthaea system
pub type PatternId = u64;

/// Unique identifier for Symthaea agents/sources
pub type SymthaeaId = u64;

/// Unique identifier for agents in the trust system
pub type AgentId = u64;

// ==============================================================================
// Re-exports from submodules
// ==============================================================================

// Core module
pub use core::{
    BiasAlert, CausalAdjustment, CausalGraph, CausalNode, Claim, CommunityProfile,
    DiversityAuditor, DiversityMetrics, EmergentWeightLearner, EpistemicOutcome, Epistemology,
    EpistemologyCounts, Evaluation, EvaluationRecord, EvaluationWithPrediction, HarmonicWeights,
    LivingWisdomEngine, Oracle, OracleObservation, OracleVerificationLevel, OutcomeType,
    PositionCounts, PositionScores, Prediction, ReparationsManager, SimpleOracle,
    StructuralPosition, SystemHealth, SystemHealthReport, WeightAdjustmentSuggestion, WisdomEngine,
};

// Domain module
pub use domain::{Domain, DomainCriticality, DomainPath, DomainRegistry};

// Composition module
pub use composition::{
    CompositionStats, CompositionType, CooccurrenceTracker, PatternComposite, SynergyCandidate,
    SynergyReason,
};

// Trust module
pub use trust::{
    AgentTrustContext, AgentTrustRegistry, TrustLevel, TrustRegistryStats, TrustWeightConfig,
    TrustWeightedScore,
};

// Collective module
pub use collective::{
    CollectiveObservation, CollectivePatternConfig, CollectivePatternContext,
    CollectivePatternRegistry, CollectivePatternStats, CollectiveSignal,
};

// Lifecycle module
pub use lifecycle::{
    LifecycleConfig, LifecycleStats, LifecycleTransition, LifecycleTransitionReason,
    PatternLifecycleInfo, PatternLifecycleRegistry, PatternLifecycleState,
};

// Versioning module
pub use versioning::{
    PatternBranch, PatternEvolutionReason, PatternVersion, PatternVersionInfo,
    PatternVersionRegistry, VersioningConfig, VersioningStats,
};

// Dependencies module
pub use dependencies::{
    DependencyConfig, DependencyIssue, DependencyIssueType, DependencyResolution, DependencyStats,
    DependencyStrength, PatternDependency, PatternDependencyRegistry, PatternRelationType,
};

// Explainability module
pub use explainability::{
    ActionEffortLevel,
    ActionSuggestion,
    // Enhancement: Counterfactual explanations and actionable suggestions
    CounterfactualFactor,
    ExplainabilityConfig,
    ExplainabilityRegistry,
    ExplainabilityStats,
    ExplanationFactor,
    ExplanationFactorType,
    FactorImpact,
    ImprovementPlan,
    PatternComparison,
    PatternExplanation,
    PlanFeasibility,
    Recommendation,
};

// Symthaea module
pub use symthaea::{
    BridgeHealthReport,
    CalibrationBucket,
    CalibrationCurve,
    CausalDiscovery,
    // Causal discovery types (Component 9 / Enhancement 5)
    CausalPatternDependency,
    CounterfactualAlternative,
    CounterfactualAnalysis,
    DependencyType,
    EnhancedHealthReport,
    EstimationMethod,
    ExperimentPlan,
    ExperimentPlanner,
    ExperimentResult,
    LearningGuidance,
    PatternDecayStatus,
    PatternDowngrade,
    PatternEpistemics,
    PatternPrediction,
    PatternUpgrade,
    PatternUsageEvent,
    PatternUsageOracle,
    RiskLevel,
    SwarmInstanceResult,
    SwarmPatternOracle,
    SymthaeaCausalBridge,
    SymthaeaPattern,
    TemporalDecayConfig,
};

// Recommendations module (Component 18)
pub use recommendations::{
    PatternRecommendation, PatternSignals, RecommendationConfig, RecommendationContext,
    RecommendationRegistry, RecommendationSet, RecommendationStats, SignalBreakdown,
    SignalContribution,
};

// Anomaly module (Component 19)
pub use anomaly::{
    Anomaly, AnomalyConfig, AnomalyDetector, AnomalySeverity, AnomalyStats, AnomalyType, DataPoint,
    TimeSeries,
};

// Succession module (Component 20)
pub use succession::{
    DependencyChange, DependencyChangeType, MigrationEffort, MigrationInstruction, MigrationPlan,
    PatternSuccession, SuccessionConfig, SuccessionManager, SuccessionReason, SuccessionStats,
    SuccessionStatus,
};

// Similarity module (Component 21)
pub use similarity::{
    DuplicateCandidate, DuplicateDecision, DuplicateSuggestion, MergeSuggestion,
    MergeSuggestionStatus, PatternCluster, PatternFeatures, SimilarityCalculator, SimilarityConfig,
    SimilarityMetric, SimilarityRegistry, SimilarityScore, SimilarityStats, SimilarityWeights,
};

// Associative Learner module (Component 22)
pub use associative_learner::{
    ActionPrediction, ActionRegistry, AssociativeLearner, AssociativeLearnerConfig,
    AssociativeLearnerStats, BinaryHV, ContinuousHV, ExperienceEncoding, MemorySnapshot,
    SparseProjector, WisdomContext, HDC_BYTES, HDC_DIMENSION,
};

// Python bindings (Component 23 - PyO3 Integration)
#[cfg(feature = "python")]
pub use python_bindings::{
    register_wisdom_classes, PyActionPrediction, PyPattern, PyPatternEpistemics, PyWisdomBridge,
};

#[cfg(test)]
mod tests;
