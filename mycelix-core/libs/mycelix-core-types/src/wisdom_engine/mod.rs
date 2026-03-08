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

pub mod core;
pub mod domain;
pub mod composition;
pub mod trust;
pub mod collective;
pub mod lifecycle;
pub mod versioning;
pub mod dependencies;
pub mod explainability;
pub mod symthaea;
pub mod recommendations;
pub mod anomaly;
pub mod succession;
pub mod similarity;
pub mod associative_learner;

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
    HarmonicWeights, CommunityProfile, StructuralPosition,
    EvaluationRecord, DiversityMetrics, PositionCounts, EpistemologyCounts,
    PositionScores, DiversityAuditor, BiasAlert,
    ReparationsManager, EpistemicOutcome, OutcomeType,
    EmergentWeightLearner, WeightAdjustmentSuggestion,
    Epistemology, Claim, Evaluation, WisdomEngine,
    Prediction, CausalNode, Oracle, OracleObservation, OracleVerificationLevel,
    SimpleOracle, CausalAdjustment, CausalGraph,
    LivingWisdomEngine, EvaluationWithPrediction, SystemHealthReport, SystemHealth,
};

// Domain module
pub use domain::{
    DomainCriticality, Domain, DomainRegistry, DomainPath,
};

// Composition module
pub use composition::{
    CompositionType, PatternComposite, SynergyCandidate, SynergyReason,
    CooccurrenceTracker, CompositionStats,
};

// Trust module
pub use trust::{
    TrustWeightConfig, AgentTrustContext, TrustWeightedScore, TrustLevel,
    AgentTrustRegistry, TrustRegistryStats,
};

// Collective module
pub use collective::{
    CollectivePatternConfig, CollectivePatternContext, CollectiveSignal,
    CollectiveObservation, CollectivePatternRegistry, CollectivePatternStats,
};

// Lifecycle module
pub use lifecycle::{
    PatternLifecycleState, LifecycleTransitionReason, LifecycleTransition,
    LifecycleConfig, PatternLifecycleInfo, PatternLifecycleRegistry, LifecycleStats,
};

// Versioning module
pub use versioning::{
    PatternVersion, PatternEvolutionReason, PatternVersionInfo,
    VersioningConfig, PatternBranch, PatternVersionRegistry, VersioningStats,
};

// Dependencies module
pub use dependencies::{
    PatternRelationType, DependencyStrength, PatternDependency,
    DependencyConfig, DependencyResolution, DependencyIssue, DependencyIssueType,
    PatternDependencyRegistry, DependencyStats,
};

// Explainability module
pub use explainability::{
    ExplainabilityConfig, ExplanationFactorType, FactorImpact,
    ExplanationFactor, Recommendation, PatternExplanation, PatternComparison,
    ExplainabilityRegistry, ExplainabilityStats,
    // Enhancement: Counterfactual explanations and actionable suggestions
    CounterfactualFactor, ActionEffortLevel, ActionSuggestion, ImprovementPlan, PlanFeasibility,
};

// Symthaea module
pub use symthaea::{
    SymthaeaPattern, PatternEpistemics, PatternPrediction,
    PatternUsageOracle, PatternUsageEvent, SwarmPatternOracle, SwarmInstanceResult,
    LearningGuidance, PatternDowngrade, PatternUpgrade,
    SymthaeaCausalBridge, BridgeHealthReport,
    TemporalDecayConfig, PatternDecayStatus,
    CalibrationBucket, CalibrationCurve,
    CounterfactualAnalysis, CounterfactualAlternative, EstimationMethod,
    ExperimentPlan, RiskLevel, ExperimentResult, ExperimentPlanner,
    EnhancedHealthReport,
    // Causal discovery types (Component 9 / Enhancement 5)
    CausalPatternDependency, DependencyType, CausalDiscovery,
};

// Recommendations module (Component 18)
pub use recommendations::{
    RecommendationConfig, SignalBreakdown, SignalContribution,
    RecommendationContext, PatternRecommendation, RecommendationSet,
    PatternSignals, RecommendationStats, RecommendationRegistry,
};

// Anomaly module (Component 19)
pub use anomaly::{
    AnomalyType, AnomalySeverity, Anomaly, AnomalyConfig,
    DataPoint, TimeSeries, AnomalyStats, AnomalyDetector,
};

// Succession module (Component 20)
pub use succession::{
    SuccessionReason, SuccessionStatus, PatternSuccession,
    MigrationPlan, MigrationEffort, MigrationInstruction,
    DependencyChange, DependencyChangeType,
    SuccessionConfig, SuccessionStats, SuccessionManager,
};

// Similarity module (Component 21)
pub use similarity::{
    SimilarityMetric, SimilarityScore, PatternFeatures,
    SimilarityCalculator, PatternCluster, DuplicateCandidate,
    DuplicateSuggestion, DuplicateDecision, MergeSuggestion, MergeSuggestionStatus,
    SimilarityWeights, SimilarityConfig, SimilarityRegistry, SimilarityStats,
};

// Associative Learner module (Component 22)
pub use associative_learner::{
    HDC_DIMENSION, HDC_BYTES,
    BinaryHV, ContinuousHV, SparseProjector,
    ExperienceEncoding, ActionRegistry,
    AssociativeLearnerConfig, AssociativeLearner, ActionPrediction,
    MemorySnapshot, AssociativeLearnerStats, WisdomContext,
};

// Python bindings (Component 23 - PyO3 Integration)
#[cfg(feature = "python")]
pub use python_bindings::{
    PyWisdomBridge, PyPattern, PyPatternEpistemics, PyActionPrediction,
    register_wisdom_classes,
};

#[cfg(test)]
mod tests;
