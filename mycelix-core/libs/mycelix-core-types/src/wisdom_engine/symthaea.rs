//! SymthaeaCausalBridge: Scientific Method for AI
//!
//! The SymthaeaCausalBridge implements a complete scientific method cycle for
//! AI pattern learning, integrating all other components into a unified system.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{
    PatternId, SymthaeaId, DomainId, CompositeId,
    CommunityId, PredictionId, CausalNodeId,
    DomainRegistry,
    CompositionType, PatternComposite, SynergyCandidate, SynergyReason, CooccurrenceTracker, CompositionStats,
    TrustWeightedScore, TrustLevel, AgentTrustRegistry, TrustRegistryStats,
    CollectivePatternRegistry, CollectivePatternStats,
    PatternLifecycleState, PatternLifecycleRegistry, LifecycleStats,
    LifecycleTransitionReason, LifecycleTransition,
    PatternVersion, VersioningConfig, PatternVersionInfo, PatternBranch, PatternVersionRegistry, VersioningStats, PatternEvolutionReason,
    PatternRelationType, DependencyStrength, PatternDependency, DependencyConfig, DependencyResolution, DependencyIssue, DependencyIssueType, PatternDependencyRegistry, DependencyStats,
    ExplainabilityRegistry, PatternExplanation, PatternComparison, ExplainabilityStats,
    ImprovementPlan, ActionSuggestion, PlanFeasibility,
    // Component 18: Recommendations
    RecommendationRegistry, RecommendationContext, PatternRecommendation, RecommendationSet, PatternSignals,
    // Component 19: Anomaly Detection
    AnomalyDetector, Anomaly,
    // Component 20: Succession
    SuccessionManager, SuccessionReason, MigrationInstruction,
    // Component 21: Pattern Similarity
    SimilarityRegistry, SimilarityScore, PatternCluster, DuplicateCandidate,
    // Component 22: Associative Learner
    AssociativeLearner, WisdomContext,
};
// Note: CausalPatternDependency, DependencyType, CausalDiscovery are defined locally in this file
use super::core::{Oracle, OracleObservation, OracleVerificationLevel, CausalGraph, HarmonicWeights};
use crate::epistemic::EpistemicContext;
use crate::KVector;

// ==============================================================================
// COMPONENT 8: SYMTHAEA CAUSAL BRIDGE - Scientific Method for AI
// ==============================================================================
//
// This component integrates Symthaea pattern learning with the CausalGraph,
// creating a bidirectional feedback loop where:
//
// 1. Symthaea learns patterns (implicit predictions)
// 2. Patterns are used in the wild
// 3. Oracles observe outcomes
// 4. CausalGraph computes prediction errors
// 5. Errors flow back to update pattern confidence
// 6. Symthaea becomes wiser, not just "smarter"
//
// This is the "Scientific Method for AI" - testing its own hypotheses
// against reality and learning from the consequences of its advice.

/// Unique identifier for a Symthaea pattern

/// Unique identifier for a Symthaea instance

/// A Symthaea-learned pattern that maps problems to solutions
///
/// Every pattern is implicitly a prediction:
/// "If you encounter problem X, solution Y will work with probability P"
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SymthaeaPattern {
    /// Unique pattern identifier
    pub pattern_id: PatternId,

    /// The problem this pattern addresses (e.g., "memory_safety_concern")
    pub problem_domain: String,

    /// The solution this pattern proposes (e.g., "use_rust_instead_of_c")
    pub solution: String,

    /// Success rate observed so far (0.0-1.0)
    pub success_rate: f32,

    /// Number of times this pattern has been applied
    pub usage_count: u64,

    /// Number of successful applications
    pub success_count: u64,

    /// Epistemic classification of this pattern
    pub epistemic_classification: PatternEpistemics,

    /// Consciousness level (Φ) when this pattern was learned
    pub phi_at_learning: f32,

    /// Timestamp when pattern was created
    pub created_at: u64,

    /// Last time the pattern was used
    pub last_used: u64,

    /// Description of the pattern for humans
    pub description: String,

    /// The Symthaea instance that learned this pattern
    pub learned_by: SymthaeaId,

    /// Whether this pattern has been tested on real systems
    pub validated_in_production: bool,

    /// Structured domain references (Component 10)
    /// Patterns can belong to multiple domains for cross-domain discovery
    pub domain_ids: Vec<DomainId>,
}

impl SymthaeaPattern {
    /// Create a new pattern
    pub fn new(
        pattern_id: PatternId,
        problem_domain: &str,
        solution: &str,
        phi_at_learning: f32,
        timestamp: u64,
        learned_by: SymthaeaId,
    ) -> Self {
        Self {
            pattern_id,
            problem_domain: problem_domain.to_string(),
            solution: solution.to_string(),
            success_rate: 0.5, // Start neutral
            usage_count: 0,
            success_count: 0,
            epistemic_classification: PatternEpistemics::default(),
            phi_at_learning,
            created_at: timestamp,
            last_used: timestamp,
            description: String::new(),
            learned_by,
            validated_in_production: false,
            domain_ids: Vec::new(),
        }
    }

    /// Create a pattern with structured domains
    pub fn with_domains(
        pattern_id: PatternId,
        domain_ids: Vec<DomainId>,
        solution: &str,
        phi_at_learning: f32,
        timestamp: u64,
        learned_by: SymthaeaId,
    ) -> Self {
        let mut pattern = Self::new(pattern_id, "", solution, phi_at_learning, timestamp, learned_by);
        pattern.domain_ids = domain_ids;
        pattern
    }

    /// Add a domain to this pattern
    pub fn add_domain(&mut self, domain_id: DomainId) {
        if !self.domain_ids.contains(&domain_id) {
            self.domain_ids.push(domain_id);
        }
    }

    /// Check if pattern belongs to a domain
    pub fn in_domain(&self, domain_id: DomainId) -> bool {
        self.domain_ids.contains(&domain_id)
    }

    /// Check if pattern is related to a domain (including ancestors/descendants)
    pub fn related_to_domain(&self, domain_id: DomainId, registry: &DomainRegistry) -> bool {
        for &my_domain in &self.domain_ids {
            if my_domain == domain_id || registry.are_related(my_domain, domain_id) {
                return true;
            }
        }
        false
    }

    /// Calculate domain similarity to another pattern
    pub fn domain_similarity(&self, other: &SymthaeaPattern, registry: &DomainRegistry) -> f32 {
        if self.domain_ids.is_empty() || other.domain_ids.is_empty() {
            // Fall back to string comparison for legacy patterns
            return if self.problem_domain == other.problem_domain { 1.0 } else { 0.0 };
        }

        let mut max_similarity = 0.0_f32;

        for &my_domain in &self.domain_ids {
            for &other_domain in &other.domain_ids {
                let sim = registry.similarity(my_domain, other_domain);
                max_similarity = max_similarity.max(sim);
            }
        }

        max_similarity
    }

    /// Record a usage outcome
    pub fn record_outcome(&mut self, success: bool, timestamp: u64) {
        self.usage_count += 1;
        if success {
            self.success_count += 1;
        }
        self.success_rate = self.success_count as f32 / self.usage_count as f32;
        self.last_used = timestamp;

        // First production use validates the pattern
        if self.usage_count == 1 {
            self.validated_in_production = true;
        }
    }

    /// Get confidence-adjusted success rate
    ///
    /// Higher Φ at learning time = higher confidence in the pattern
    pub fn confidence_adjusted_success_rate(&self) -> f32 {
        // Bayesian adjustment: more usage = more confidence
        let usage_confidence = 1.0 - (1.0 / (self.usage_count as f32 + 1.0).sqrt());

        // Consciousness bonus: higher Φ = better learning
        let phi_bonus = (self.phi_at_learning - 0.3).max(0.0) * 0.1;

        let adjusted = self.success_rate * usage_confidence + phi_bonus;
        adjusted.min(1.0)
    }

    /// Convert this pattern to a prediction for the CausalGraph
    pub fn as_prediction(&self, community_id: CommunityId, expected_observation_time: u64) -> PatternPrediction {
        PatternPrediction {
            pattern_id: self.pattern_id,
            community_id,
            target_variable: format!("pattern_{}_success", self.pattern_id),
            predicted_value: self.confidence_adjusted_success_rate(),
            confidence: self.phi_at_learning.min(1.0),
            timestamp: self.last_used,
            expected_observation_time,
            problem_domain: self.problem_domain.clone(),
            solution: self.solution.clone(),
        }
    }
}

/// Epistemic classification for a pattern
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternEpistemics {
    /// E-level: How was this pattern verified?
    /// E0 = Theoretical, E1 = Personal test, E2 = Reviewed, E3 = Proven, E4 = Public
    pub e_level: u8,

    /// N-level: Who agrees this pattern works?
    /// N0 = Just me, N1 = My team, N2 = Network, N3 = Universal
    pub n_level: u8,

    /// M-level: How persistent is this knowledge?
    /// M0 = Ephemeral, M1 = Temporal, M2 = Persistent, M3 = Foundational
    pub m_level: u8,
}

impl PatternEpistemics {
    /// Create new epistemic classification
    pub fn new(e_level: u8, n_level: u8, m_level: u8) -> Self {
        Self {
            e_level: e_level.min(4),
            n_level: n_level.min(3),
            m_level: m_level.min(3),
        }
    }

    /// Calculate quality score (for compatibility with EpistemicClassification)
    pub fn quality_score(&self) -> f32 {
        let e = self.e_level as f32 / 4.0;
        let n = self.n_level as f32 / 3.0;
        let m = self.m_level as f32 / 3.0;
        e * 0.4 + n * 0.35 + m * 0.25
    }

    /// Upgrade E-level based on verification
    pub fn upgrade_verification(&mut self, level: OracleVerificationLevel) {
        self.e_level = match level {
            OracleVerificationLevel::Testimonial => self.e_level.max(1),
            OracleVerificationLevel::Audited => self.e_level.max(2),
            OracleVerificationLevel::Cryptographic => self.e_level.max(3),
            OracleVerificationLevel::PubliclyReproducible => 4,
        };
    }

    /// Upgrade N-level based on consensus
    pub fn upgrade_consensus(&mut self, users: usize) {
        if users >= 100 {
            self.n_level = 3; // Network consensus
        } else if users >= 10 {
            self.n_level = self.n_level.max(2);
        } else if users >= 2 {
            self.n_level = self.n_level.max(1);
        }
    }
}

/// A prediction derived from a Symthaea pattern
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternPrediction {
    /// Source pattern
    pub pattern_id: PatternId,

    /// Community where prediction applies
    pub community_id: CommunityId,

    /// Target variable being predicted
    pub target_variable: String,

    /// Predicted success rate
    pub predicted_value: f32,

    /// Confidence (based on Φ and usage)
    pub confidence: f32,

    /// When prediction was made
    pub timestamp: u64,

    /// When to check outcome
    pub expected_observation_time: u64,

    /// Problem domain
    pub problem_domain: String,

    /// Proposed solution
    pub solution: String,
}

/// An oracle that observes pattern usage outcomes
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternUsageOracle {
    /// Unique identifier
    pub id: String,

    /// Human description
    pub description: String,

    /// Pattern being observed
    pub pattern_id: PatternId,

    /// Usage events recorded
    pub usage_events: Vec<PatternUsageEvent>,

    /// Trust level (0.0-1.0)
    pub trust_level: f32,

    /// Verification level
    pub verification_level: OracleVerificationLevel,
}

/// A single pattern usage event
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternUsageEvent {
    /// When the pattern was used
    pub timestamp: u64,

    /// Whether the usage was successful
    pub success: bool,

    /// Who used the pattern
    pub user_id: Option<String>,

    /// Context of usage
    pub context: String,

    /// Verification level of this observation
    pub verification: OracleVerificationLevel,
}

impl PatternUsageOracle {
    /// Create a new pattern usage oracle
    pub fn new(pattern_id: PatternId, description: &str) -> Self {
        Self {
            id: format!("pattern_oracle_{}", pattern_id),
            description: description.to_string(),
            pattern_id,
            usage_events: Vec::new(),
            trust_level: 0.8,
            verification_level: OracleVerificationLevel::Audited,
        }
    }

    /// Record a usage event
    pub fn record_usage(&mut self, success: bool, timestamp: u64, context: &str, verification: OracleVerificationLevel) {
        self.usage_events.push(PatternUsageEvent {
            timestamp,
            success,
            user_id: None,
            context: context.to_string(),
            verification,
        });

        // Upgrade verification level if we have publicly reproducible evidence
        if verification == OracleVerificationLevel::PubliclyReproducible {
            self.verification_level = OracleVerificationLevel::PubliclyReproducible;
        }
    }

    /// Get consensus observation from all usage events
    pub fn consensus_observation(&self) -> Option<OracleObservation> {
        if self.usage_events.is_empty() {
            return None;
        }

        let success_count = self.usage_events.iter().filter(|e| e.success).count();
        let total = self.usage_events.len();
        let success_rate = success_count as f32 / total as f32;

        // Confidence increases with more observations
        let confidence = 1.0 - (1.0 / (total as f32 + 1.0).sqrt());

        let latest_timestamp = self.usage_events
            .iter()
            .map(|e| e.timestamp)
            .max()
            .unwrap_or(0);

        Some(OracleObservation {
            variable: format!("pattern_{}_success", self.pattern_id),
            value: success_rate,
            confidence,
            timestamp: latest_timestamp,
            oracle_id: self.id.clone(),
            metadata: None,
        })
    }

    /// Get number of unique users
    pub fn unique_users(&self) -> usize {
        self.usage_events
            .iter()
            .filter_map(|e| e.user_id.as_ref())
            .collect::<std::collections::HashSet<_>>()
            .len()
    }
}

impl Oracle for PatternUsageOracle {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn observe(&self, variable: &str, _timestamp: u64) -> Option<OracleObservation> {
        let expected_var = format!("pattern_{}_success", self.pattern_id);
        if variable != expected_var {
            return None;
        }
        self.consensus_observation()
    }

    fn can_observe(&self, variable: &str) -> bool {
        let expected_var = format!("pattern_{}_success", self.pattern_id);
        variable == expected_var && !self.usage_events.is_empty()
    }

    fn trust_level(&self) -> f32 {
        self.trust_level
    }

    fn verification_level(&self) -> OracleVerificationLevel {
        self.verification_level
    }
}

/// Multi-Symthaea oracle for collective pattern validation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SwarmPatternOracle {
    /// Pattern being validated
    pub pattern_id: PatternId,

    /// Results from different Symthaea instances
    pub instance_results: HashMap<SymthaeaId, SwarmInstanceResult>,

    /// Minimum instances for consensus
    pub quorum: usize,
}

/// Result from a single Symthaea instance
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SwarmInstanceResult {
    /// Instance ID
    pub instance_id: SymthaeaId,

    /// Observed success rate
    pub success_rate: f32,

    /// Number of usages
    pub usage_count: u64,

    /// Φ level of this instance
    pub phi_level: f32,

    /// Timestamp of last update
    pub timestamp: u64,
}

impl SwarmPatternOracle {
    /// Create a new swarm oracle
    pub fn new(pattern_id: PatternId, quorum: usize) -> Self {
        Self {
            pattern_id,
            instance_results: HashMap::new(),
            quorum,
        }
    }

    /// Add result from a Symthaea instance
    pub fn add_instance_result(&mut self, result: SwarmInstanceResult) {
        self.instance_results.insert(result.instance_id, result);
    }

    /// Check if we have quorum
    pub fn has_quorum(&self) -> bool {
        self.instance_results.len() >= self.quorum
    }

    /// Calculate Φ-weighted consensus
    ///
    /// Higher Φ instances have more influence on the consensus
    pub fn phi_weighted_consensus(&self) -> Option<f32> {
        if !self.has_quorum() {
            return None;
        }

        let total_phi: f32 = self.instance_results.values().map(|r| r.phi_level).sum();
        if total_phi == 0.0 {
            return None;
        }

        let weighted_sum: f32 = self.instance_results
            .values()
            .map(|r| r.success_rate * r.phi_level)
            .sum();

        Some(weighted_sum / total_phi)
    }

    /// Get consensus observation
    pub fn consensus_observation(&self) -> Option<OracleObservation> {
        let consensus = self.phi_weighted_consensus()?;

        let latest_timestamp = self.instance_results
            .values()
            .map(|r| r.timestamp)
            .max()
            .unwrap_or(0);

        // Confidence increases with more instances and higher Φ
        let total_phi: f32 = self.instance_results.values().map(|r| r.phi_level).sum::<f32>();
        let confidence = (self.instance_results.len() as f32 * total_phi.sqrt()).min(1.0) / 10.0 + 0.5;

        Some(OracleObservation {
            variable: format!("pattern_{}_success", self.pattern_id),
            value: consensus,
            confidence: confidence.min(0.99),
            timestamp: latest_timestamp,
            oracle_id: "swarm_oracle".to_string(),
            metadata: None,
        })
    }
}

impl Oracle for SwarmPatternOracle {
    fn id(&self) -> &str {
        "swarm_oracle"
    }

    fn description(&self) -> &str {
        "Collective pattern validation from Symthaea swarm"
    }

    fn observe(&self, variable: &str, _timestamp: u64) -> Option<OracleObservation> {
        let expected_var = format!("pattern_{}_success", self.pattern_id);
        if variable != expected_var {
            return None;
        }
        self.consensus_observation()
    }

    fn can_observe(&self, variable: &str) -> bool {
        let expected_var = format!("pattern_{}_success", self.pattern_id);
        variable == expected_var && self.has_quorum()
    }

    fn trust_level(&self) -> f32 {
        // Trust increases with more instances and higher average Φ
        if self.instance_results.is_empty() {
            return 0.0;
        }
        let avg_phi: f32 = self.instance_results.values().map(|r| r.phi_level).sum::<f32>()
            / self.instance_results.len() as f32;
        (0.5 + avg_phi * 0.5).min(0.99)
    }

    fn verification_level(&self) -> OracleVerificationLevel {
        if self.instance_results.len() >= 10 {
            OracleVerificationLevel::PubliclyReproducible
        } else if self.instance_results.len() >= 3 {
            OracleVerificationLevel::Audited
        } else {
            OracleVerificationLevel::Testimonial
        }
    }
}

/// Guidance for Symthaea learning based on causal feedback
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LearningGuidance {
    /// Should Symthaea prefer patterns learned at higher Φ?
    pub favor_high_phi_patterns: bool,

    /// Suggested weight adjustments per context
    pub context_effectiveness: HashMap<String, f32>,

    /// Patterns that should be downgraded
    pub patterns_to_downgrade: Vec<PatternDowngrade>,

    /// Patterns that should be upgraded
    pub patterns_to_upgrade: Vec<PatternUpgrade>,

    /// Overall learning rate adjustment
    pub learning_rate_multiplier: f32,

    /// Explanation of guidance
    pub explanation: String,
}

/// Instruction to downgrade a pattern's confidence
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternDowngrade {
    /// Pattern to downgrade
    pub pattern_id: PatternId,

    /// How much to reduce confidence (0.0-1.0)
    pub downgrade_factor: f32,

    /// Reason for downgrade
    pub reason: String,
}

/// Instruction to upgrade a pattern's confidence
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternUpgrade {
    /// Pattern to upgrade
    pub pattern_id: PatternId,

    /// How much to increase confidence (0.0-1.0)
    pub upgrade_factor: f32,

    /// Reason for upgrade
    pub reason: String,
}

impl Default for LearningGuidance {
    fn default() -> Self {
        Self {
            favor_high_phi_patterns: true,
            context_effectiveness: HashMap::new(),
            patterns_to_downgrade: Vec::new(),
            patterns_to_upgrade: Vec::new(),
            learning_rate_multiplier: 1.0,
            explanation: String::new(),
        }
    }
}

/// The Symthaea-Causal Bridge: Scientific Method for AI
///
/// This is the integration point between Symthaea pattern learning
/// and the CausalGraph reality checking system.
///
/// The cycle:
/// 1. Symthaea learns pattern → Bridge creates prediction
/// 2. Human/Agent uses pattern → Oracle records outcome
/// 3. Bridge compares prediction vs outcome → Finds error
/// 4. WisdomEngine suggests weight adjustment
/// 5. Bridge generates LearningGuidance → Symthaea updates
/// 6. Symthaea becomes wiser → Loop continues
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SymthaeaCausalBridge {
    /// The CausalGraph for reality checking
    pub causal: CausalGraph,

    /// Consciousness (Φ) as a causal node
    pub consciousness_node: Option<CausalNodeId>,

    /// Mapping from pattern IDs to prediction IDs
    pub pattern_predictions: HashMap<PatternId, PredictionId>,

    /// Registered patterns being tracked
    pub patterns: HashMap<PatternId, SymthaeaPattern>,

    /// Pattern usage oracles
    pub pattern_oracles: HashMap<PatternId, PatternUsageOracle>,

    /// Swarm oracle for collective validation
    pub swarm_oracle: Option<SwarmPatternOracle>,

    /// Historical learning guidance generated
    pub guidance_history: Vec<LearningGuidance>,

    /// Whether to auto-generate predictions when patterns are registered
    pub auto_predict: bool,

    /// Default community for predictions
    pub default_community: CommunityId,

    /// Symthaea instance ID (if applicable)
    pub symthaea_id: Option<SymthaeaId>,

    // ==========================================================================
    // Component 9: Advanced Learning Enhancements (Integrated)
    // ==========================================================================

    /// Temporal decay configuration for pattern staleness tracking
    pub temporal_decay: TemporalDecayConfig,

    /// Calibration tracking for prediction accuracy
    pub calibration: CalibrationCurve,

    /// Causal discovery for finding pattern dependencies
    pub causal_discovery: CausalDiscovery,

    /// Experiment planner for active learning
    pub experiment_planner: ExperimentPlanner,

    /// Whether to automatically run causal discovery after pattern usage
    pub auto_discover_causality: bool,

    /// Minimum number of pattern usages before running causal discovery
    pub discovery_threshold: usize,

    // ==========================================================================
    // Component 10: Domain System (Integrated)
    // ==========================================================================

    /// Domain registry for hierarchical knowledge organization
    pub domain_registry: DomainRegistry,

    /// Minimum domain similarity for cross-domain causal discovery (0.0-1.0)
    pub cross_domain_threshold: f32,

    // ==========================================================================
    // Component 11: Pattern Composition (Integrated)
    // ==========================================================================

    /// Registered pattern composites
    pub composites: HashMap<CompositeId, PatternComposite>,

    /// Co-occurrence tracking for synergy discovery
    pub cooccurrence_tracker: CooccurrenceTracker,

    /// Next available composite ID
    next_composite_id: CompositeId,

    /// Whether to auto-discover synergies
    pub auto_discover_synergies: bool,

    /// Minimum co-occurrences before suggesting a composite
    pub synergy_discovery_threshold: u32,

    // ==========================================================================
    // Component 12: Trust-Integrated Patterns
    // ==========================================================================

    /// Registry for tracking agent trust scores
    pub trust_registry: AgentTrustRegistry,

    /// Whether to use trust-weighted scoring for patterns
    pub use_trust_weighting: bool,

    // ==========================================================================
    // Component 13: Collective Pattern Integration
    // ==========================================================================

    /// Registry for collective pattern observations
    pub collective_registry: CollectivePatternRegistry,

    /// Whether to use collective observation for patterns
    pub use_collective_observation: bool,

    // ==========================================================================
    // Component 14: Pattern Lifecycle Management
    // ==========================================================================

    /// Registry for tracking pattern lifecycle states
    pub lifecycle_registry: PatternLifecycleRegistry,

    /// Whether to automatically deprecate stale/unused patterns
    pub auto_lifecycle_management: bool,

    // ==========================================================================
    // Component 15: Pattern Versioning & Evolution
    // ==========================================================================

    /// Registry for tracking pattern versions and evolution
    pub version_registry: PatternVersionRegistry,

    /// Whether to automatically version patterns when they evolve
    pub auto_versioning: bool,

    // ==========================================================================
    // Component 16: Pattern Dependencies & Prerequisites
    // ==========================================================================

    /// Registry for tracking pattern dependencies and prerequisites
    pub dependency_registry: PatternDependencyRegistry,

    /// Whether to auto-discover dependencies from co-occurrence patterns
    pub auto_discover_dependencies: bool,

    /// Whether to enforce required dependencies before pattern use
    pub enforce_dependencies: bool,

    // ==========================================================================
    // Component 17: Pattern Explainability
    // ==========================================================================

    /// Registry for generating human-readable pattern explanations
    pub explainability: ExplainabilityRegistry,

    // ==========================================================================
    // Component 18: Pattern Recommendations
    // ==========================================================================

    /// Registry for generating pattern recommendations
    pub recommendations: RecommendationRegistry,

    // ==========================================================================
    // Component 19: Anomaly Detection
    // ==========================================================================

    /// Detector for identifying unusual patterns in the system
    pub anomaly_detector: AnomalyDetector,

    // ==========================================================================
    // Component 20: Pattern Succession
    // ==========================================================================

    /// Manager for pattern succession and migration
    pub succession_manager: SuccessionManager,

    // ==========================================================================
    // Component 21: Pattern Similarity & Clustering
    // ==========================================================================

    /// Registry for pattern similarity metrics and clustering
    pub similarity_registry: SimilarityRegistry,

    /// Whether to auto-detect duplicates when patterns are learned
    pub auto_detect_duplicates: bool,

    /// Whether to auto-cluster patterns periodically
    pub auto_cluster_patterns: bool,

    // ==========================================================================
    // Component 22: HDC-Grounded Associative Learner
    // ==========================================================================

    /// Associative learner for zero-shot generalization
    pub associative_learner: AssociativeLearner,

    /// Whether to use HDC learning for pattern recommendations
    pub use_associative_learning: bool,
}

impl Default for SymthaeaCausalBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl SymthaeaCausalBridge {
    /// Create a new bridge with all Component 9 enhancements
    pub fn new() -> Self {
        let mut bridge = Self {
            causal: CausalGraph::new(),
            consciousness_node: None,
            pattern_predictions: HashMap::new(),
            patterns: HashMap::new(),
            pattern_oracles: HashMap::new(),
            swarm_oracle: None,
            guidance_history: Vec::new(),
            auto_predict: true,
            default_community: 1,
            symthaea_id: None,
            // Component 9: Advanced Learning Enhancements
            temporal_decay: TemporalDecayConfig::default(),
            calibration: CalibrationCurve::default(),
            causal_discovery: CausalDiscovery::new(),
            experiment_planner: ExperimentPlanner::new(),
            auto_discover_causality: true,
            discovery_threshold: 10, // Discover dependencies after 10 pattern usages
            // Component 10: Domain System
            domain_registry: DomainRegistry::new(),
            cross_domain_threshold: 0.5, // Discover across domains with >= 50% similarity
            // Component 11: Pattern Composition
            composites: HashMap::new(),
            cooccurrence_tracker: CooccurrenceTracker::new(),
            next_composite_id: 1,
            auto_discover_synergies: true,
            synergy_discovery_threshold: 5, // Suggest composites after 5 co-occurrences
            // Component 12: Trust-Integrated Patterns
            trust_registry: AgentTrustRegistry::new(),
            use_trust_weighting: false, // Opt-in for trust weighting
            // Component 13: Collective Pattern Integration
            collective_registry: CollectivePatternRegistry::new(),
            use_collective_observation: true, // Enable by default
            // Component 14: Pattern Lifecycle Management
            lifecycle_registry: PatternLifecycleRegistry::new(),
            auto_lifecycle_management: true, // Auto-deprecate stale patterns by default
            // Component 15: Pattern Versioning & Evolution
            version_registry: PatternVersionRegistry::new(),
            auto_versioning: true, // Automatically version patterns when they evolve
            // Component 16: Pattern Dependencies & Prerequisites
            dependency_registry: PatternDependencyRegistry::new(),
            auto_discover_dependencies: true, // Auto-discover dependencies from co-occurrence
            enforce_dependencies: false, // Don't enforce by default (opt-in)
            // Component 17: Pattern Explainability
            explainability: ExplainabilityRegistry::new(),
            // Component 18: Pattern Recommendations
            recommendations: RecommendationRegistry::new(),
            // Component 19: Anomaly Detection
            anomaly_detector: AnomalyDetector::new(),
            // Component 20: Pattern Succession
            succession_manager: SuccessionManager::new(),
            // Component 21: Pattern Similarity & Clustering
            similarity_registry: SimilarityRegistry::new(),
            auto_detect_duplicates: true,
            auto_cluster_patterns: true,
            // Component 22: HDC-Grounded Associative Learner
            associative_learner: AssociativeLearner::new(),
            use_associative_learning: true,
        };

        // Add consciousness as a causal node
        let phi_node = bridge.causal.add_node("phi", "Consciousness level (Φ)");
        bridge.consciousness_node = Some(phi_node);

        bridge
    }

    /// Create with a specific Symthaea instance ID
    pub fn with_symthaea_id(symthaea_id: SymthaeaId) -> Self {
        let mut bridge = Self::new();
        bridge.symthaea_id = Some(symthaea_id);
        bridge
    }

    /// Register a pattern learned by Symthaea
    ///
    /// This creates:
    /// 1. A causal node for the pattern
    /// 2. A link from Φ to the pattern (consciousness influences learning quality)
    /// 3. A prediction tracking the pattern's success
    /// 4. An oracle for recording usage outcomes
    pub fn on_pattern_learned(&mut self, pattern: SymthaeaPattern) -> PatternId {
        let pattern_id = pattern.pattern_id;

        // Create causal node for this pattern
        let pattern_node = self.causal.add_node(
            &format!("pattern_{}", pattern_id),
            &format!("Pattern: {} -> {}", pattern.problem_domain, pattern.solution),
        );

        // Link consciousness to pattern (Φ influences pattern quality)
        if let Some(phi_node) = self.consciousness_node {
            self.causal.add_causal_link(phi_node, pattern_node, pattern.phi_at_learning);
        }

        // Create prediction if auto_predict is enabled
        if self.auto_predict {
            let prediction = pattern.as_prediction(
                self.default_community,
                pattern.last_used + 30 * 24 * 60 * 60, // 30 days
            );

            let pred_id = self.causal.predict(
                pattern_id,
                prediction.community_id,
                prediction.target_variable.clone(),
                prediction.predicted_value,
                prediction.confidence,
                prediction.timestamp,
                prediction.expected_observation_time,
                EpistemicContext::Standard,
                HarmonicWeights::default_weights(),
            );

            self.pattern_predictions.insert(pattern_id, pred_id);
        }

        // Create oracle for this pattern
        let oracle = PatternUsageOracle::new(
            pattern_id,
            &format!("Usage oracle for pattern {}", pattern_id),
        );
        self.pattern_oracles.insert(pattern_id, oracle);

        // Store the pattern
        self.patterns.insert(pattern_id, pattern);

        pattern_id
    }

    /// Record a pattern usage outcome
    ///
    /// This feeds the observation into the oracle, resolves predictions,
    /// and integrates with Component 9 enhancements:
    /// - Tracks calibration (prediction accuracy)
    /// - Records co-occurrence for causal discovery
    /// - Auto-discovers pattern dependencies when threshold reached
    pub fn on_pattern_used(
        &mut self,
        pattern_id: PatternId,
        success: bool,
        timestamp: u64,
        context: &str,
        verification: OracleVerificationLevel,
    ) {
        // Get the pattern's predicted success rate for calibration tracking
        let predicted_confidence = self.patterns.get(&pattern_id)
            .map(|p| p.success_rate)
            .unwrap_or(0.5);

        // Update the pattern itself
        if let Some(pattern) = self.patterns.get_mut(&pattern_id) {
            pattern.record_outcome(success, timestamp);
        }

        // Update the oracle
        if let Some(oracle) = self.pattern_oracles.get_mut(&pattern_id) {
            oracle.record_usage(success, timestamp, context, verification);
        }

        // Try to resolve predictions
        self.resolve_pattern_predictions(pattern_id, timestamp);

        // =====================================================================
        // Component 9: Integrated Learning Enhancements
        // =====================================================================

        // Track calibration: compare prediction to actual outcome
        let actual_outcome = if success { 1.0 } else { 0.0 };
        self.calibration.add_result(predicted_confidence, predicted_confidence, actual_outcome);

        // Record co-occurrence for causal discovery
        // (pattern usage in this context)
        self.causal_discovery.record_usage(pattern_id, success, timestamp);

        // Auto-discover pattern dependencies when enough data accumulated
        if self.auto_discover_causality {
            let total_usages: usize = self.patterns.values().map(|p| p.usage_count as usize).sum();
            if total_usages >= self.discovery_threshold && total_usages % self.discovery_threshold == 0 {
                self.run_causal_discovery();
            }
        }

        // =====================================================================
        // Component 12: Trust-Integrated Patterns
        // =====================================================================
        // Update the learner agent's trust based on pattern outcome
        if self.use_trust_weighting {
            self.update_agent_trust_from_outcome(pattern_id, success, timestamp);
        }
    }

    /// Run causal discovery to find pattern dependencies
    ///
    /// This identifies patterns that tend to co-occur or conflict,
    /// helping Symthaea understand which patterns work together.
    /// Now supports cross-domain discovery using domain similarity (Component 10).
    pub fn run_causal_discovery(&mut self) {
        // Collect pattern IDs and their success rates
        let patterns: Vec<(PatternId, f32)> = self.patterns.iter()
            .map(|(id, p)| (*id, p.success_rate))
            .collect();

        // For each pair of patterns, check for potential dependencies
        for i in 0..patterns.len() {
            for j in (i + 1)..patterns.len() {
                let (id_a, rate_a) = patterns[i];
                let (id_b, rate_b) = patterns[j];

                // Calculate domain similarity
                let domain_similarity = match (self.patterns.get(&id_a), self.patterns.get(&id_b)) {
                    (Some(a), Some(b)) => {
                        // Use structured domains if available
                        if !a.domain_ids.is_empty() && !b.domain_ids.is_empty() {
                            a.domain_similarity(b, &self.domain_registry)
                        } else {
                            // Fall back to string comparison for legacy patterns
                            if a.problem_domain == b.problem_domain { 1.0 } else { 0.0 }
                        }
                    },
                    _ => continue,
                };

                // Skip if domains are too different
                if domain_similarity < self.cross_domain_threshold {
                    continue;
                }

                // Calculate correlation based on success rate similarity
                // Weight by domain similarity for cross-domain discoveries
                let rate_diff = (rate_a - rate_b).abs();
                let base_strength = 1.0 - rate_diff;
                let strength = base_strength * domain_similarity; // Scale by domain relevance

                if strength >= self.causal_discovery.min_strength {
                    // Determine dependency type based on rates
                    let dep_type = if (rate_a > 0.7 && rate_b > 0.7) || (rate_a < 0.3 && rate_b < 0.3) {
                        // Both succeed or both fail together
                        DependencyType::Synergistic
                    } else if (rate_a > 0.7 && rate_b < 0.3) || (rate_a < 0.3 && rate_b > 0.7) {
                        // Opposite success rates suggest conflict
                        DependencyType::Conflicting
                    } else {
                        // One typically follows the other
                        DependencyType::Sequential
                    };

                    self.causal_discovery.add_dependency(id_a, id_b, strength, dep_type);
                }
            }
        }
    }

    /// Register a domain path and return its ID
    pub fn register_domain(&mut self, path: &str, timestamp: u64) -> DomainId {
        self.domain_registry.register_path(path, timestamp)
    }

    /// Find patterns in a domain (including related domains)
    pub fn patterns_in_domain(&self, domain_id: DomainId, include_related: bool) -> Vec<PatternId> {
        self.patterns.iter()
            .filter(|(_, p)| {
                if include_related {
                    p.related_to_domain(domain_id, &self.domain_registry)
                } else {
                    p.in_domain(domain_id)
                }
            })
            .map(|(id, _)| *id)
            .collect()
    }

    // ==========================================================================
    // Component 11: Pattern Composition Methods
    // ==========================================================================

    /// Create a new pattern composite
    pub fn create_composite(
        &mut self,
        name: &str,
        pattern_ids: Vec<PatternId>,
        composition_type: CompositionType,
        timestamp: u64,
        created_by: SymthaeaId,
    ) -> CompositeId {
        let id = self.next_composite_id;
        self.next_composite_id += 1;

        let mut composite = PatternComposite::new(
            id,
            name,
            pattern_ids.clone(),
            composition_type,
            timestamp,
            created_by,
        );

        // Calculate expected rate based on individual patterns
        let individual_rates: Vec<f32> = pattern_ids
            .iter()
            .filter_map(|pid| self.patterns.get(pid))
            .map(|p| p.success_rate)
            .collect();

        composite.calculate_expected_rate(&individual_rates);

        // Inherit domains from constituent patterns
        let mut domain_set: std::collections::HashSet<DomainId> = std::collections::HashSet::new();
        for pid in &pattern_ids {
            if let Some(p) = self.patterns.get(pid) {
                domain_set.extend(p.domain_ids.iter().copied());
            }
        }
        composite.domain_ids = domain_set.into_iter().collect();

        self.composites.insert(id, composite);
        id
    }

    /// Create a composite with a rationale
    pub fn create_composite_with_rationale(
        &mut self,
        name: &str,
        pattern_ids: Vec<PatternId>,
        composition_type: CompositionType,
        rationale: &str,
        timestamp: u64,
        created_by: SymthaeaId,
    ) -> CompositeId {
        let id = self.create_composite(name, pattern_ids, composition_type, timestamp, created_by);
        if let Some(composite) = self.composites.get_mut(&id) {
            composite.rationale = rationale.to_string();
        }
        id
    }

    /// Record that a composite was used
    pub fn on_composite_used(
        &mut self,
        composite_id: CompositeId,
        success: bool,
        timestamp: u64,
    ) {
        // Update composite stats
        if let Some(composite) = self.composites.get_mut(&composite_id) {
            composite.record_outcome(success, timestamp);

            // Also record co-occurrence for synergy tracking
            let pattern_ids = composite.pattern_ids.clone();
            self.cooccurrence_tracker.record_cooccurrence(&pattern_ids, success);
        }
    }

    /// Record that multiple patterns were used together (for synergy discovery)
    pub fn on_patterns_used_together(
        &mut self,
        pattern_ids: &[PatternId],
        success: bool,
        _timestamp: u64,
    ) {
        self.cooccurrence_tracker.record_cooccurrence(pattern_ids, success);
    }

    /// Get a composite by ID
    pub fn get_composite(&self, id: CompositeId) -> Option<&PatternComposite> {
        self.composites.get(&id)
    }

    /// Get a mutable composite by ID
    pub fn get_composite_mut(&mut self, id: CompositeId) -> Option<&mut PatternComposite> {
        self.composites.get_mut(&id)
    }

    /// Find composites containing a specific pattern
    pub fn composites_containing(&self, pattern_id: PatternId) -> Vec<CompositeId> {
        self.composites
            .iter()
            .filter(|(_, c)| c.contains_pattern(pattern_id))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Find composites in a specific domain
    pub fn composites_in_domain(&self, domain_id: DomainId) -> Vec<CompositeId> {
        self.composites
            .iter()
            .filter(|(_, c)| c.in_domain(domain_id))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get composites showing synergy
    pub fn synergistic_composites(&self) -> Vec<&PatternComposite> {
        self.composites
            .values()
            .filter(|c| c.has_synergy())
            .collect()
    }

    /// Get composites showing interference (anti-patterns)
    pub fn interfering_composites(&self) -> Vec<&PatternComposite> {
        self.composites
            .values()
            .filter(|c| c.has_interference())
            .collect()
    }

    /// Discover potential synergies based on co-occurrence data
    pub fn discover_synergies(&self) -> Vec<SynergyCandidate> {
        let mut candidates = Vec::new();

        // Find co-occurring patterns with high success rates
        for (a, b, success_rate) in self.cooccurrence_tracker.find_synergy_candidates() {
            // Skip if composite already exists
            let already_exists = self.composites.values().any(|c| {
                c.pattern_ids.len() == 2
                    && c.pattern_ids.contains(&a)
                    && c.pattern_ids.contains(&b)
            });

            if already_exists {
                continue;
            }

            // Calculate expected rate if independent
            let rate_a = self.patterns.get(&a).map(|p| p.success_rate).unwrap_or(0.5);
            let rate_b = self.patterns.get(&b).map(|p| p.success_rate).unwrap_or(0.5);
            let expected_parallel = (rate_a + rate_b) / 2.0;

            // Estimate synergy
            let estimated_synergy = if expected_parallel > 0.0 {
                success_rate / expected_parallel
            } else {
                1.0
            };

            // Check if domains are related
            let domain_a: Vec<DomainId> = self.patterns.get(&a)
                .map(|p| p.domain_ids.clone())
                .unwrap_or_default();
            let domain_b: Vec<DomainId> = self.patterns.get(&b)
                .map(|p| p.domain_ids.clone())
                .unwrap_or_default();

            let reason = if !domain_a.is_empty() && !domain_b.is_empty() {
                // Check domain affinity
                let has_affinity = domain_a.iter().any(|da| {
                    domain_b.iter().any(|db| {
                        self.domain_registry.similarity(*da, *db) > 0.5
                    })
                });
                if has_affinity {
                    SynergyReason::DomainAffinity
                } else {
                    SynergyReason::CooccurrenceSuccess
                }
            } else {
                SynergyReason::CooccurrenceSuccess
            };

            // Calculate confidence based on co-occurrence count
            let (count, _, _) = self.cooccurrence_tracker.get_stats(a, b).unwrap_or((0, 0, 0.0));
            let confidence = (count as f32 / 20.0).min(1.0);

            candidates.push(SynergyCandidate {
                pattern_ids: vec![a, b],
                estimated_synergy,
                reason,
                confidence,
                suggested_type: CompositionType::Parallel,
            });
        }

        // Sort by estimated synergy (highest first)
        candidates.sort_by(|a, b| {
            b.estimated_synergy.partial_cmp(&a.estimated_synergy).unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Auto-create composites from discovered synergies
    pub fn auto_create_composites(&mut self, timestamp: u64, created_by: SymthaeaId) -> Vec<CompositeId> {
        let candidates = self.discover_synergies();
        let mut created = Vec::new();

        for candidate in candidates {
            if candidate.estimated_synergy > 1.1 && candidate.confidence > 0.5 {
                // Generate name from pattern names
                let names: Vec<String> = candidate.pattern_ids
                    .iter()
                    .filter_map(|id| self.patterns.get(id))
                    .map(|p| p.solution.clone())
                    .collect();
                let name = format!("{} + {}",
                    names.get(0).map(|s| s.as_str()).unwrap_or("?"),
                    names.get(1).map(|s| s.as_str()).unwrap_or("?")
                );

                let rationale = format!(
                    "Auto-discovered: patterns {} (estimated synergy: {:.2})",
                    candidate.reason,
                    candidate.estimated_synergy
                );

                let id = self.create_composite(
                    &name,
                    candidate.pattern_ids,
                    candidate.suggested_type,
                    timestamp,
                    created_by,
                );

                if let Some(composite) = self.composites.get_mut(&id) {
                    composite.rationale = rationale;
                    composite.auto_discovered = true;
                }

                created.push(id);
            }
        }

        created
    }

    /// Get the best composite for a domain based on synergy score
    pub fn best_composite_for_domain(&self, domain_id: DomainId) -> Option<&PatternComposite> {
        self.composites
            .values()
            .filter(|c| c.in_domain(domain_id) && c.usage_count >= 3)
            .max_by(|a, b| {
                // Prefer high synergy with good confidence
                let score_a = a.synergy_score * a.synergy_confidence;
                let score_b = b.synergy_score * b.synergy_confidence;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get composition statistics summary
    pub fn composition_stats(&self) -> CompositionStats {
        let total = self.composites.len();
        let synergistic = self.composites.values().filter(|c| c.has_synergy()).count();
        let interfering = self.composites.values().filter(|c| c.has_interference()).count();
        let auto_discovered = self.composites.values().filter(|c| c.auto_discovered).count();
        let tracked_pairs = self.cooccurrence_tracker.pair_count();

        CompositionStats {
            total_composites: total,
            synergistic_composites: synergistic,
            interfering_composites: interfering,
            auto_discovered_composites: auto_discovered,
            tracked_pattern_pairs: tracked_pairs,
            pending_candidates: self.discover_synergies().len(),
        }
    }

    // ==========================================================================
    // Component 12: Trust-Integrated Pattern Methods
    // ==========================================================================

    /// Register an agent with a known K-Vector trust profile
    pub fn register_agent_trust(&mut self, agent_id: SymthaeaId, k_vector: KVector, timestamp: u64) {
        self.trust_registry.register_agent(agent_id, k_vector, timestamp);
    }

    /// Register a new agent with neutral trust
    pub fn register_new_agent(&mut self, agent_id: SymthaeaId, timestamp: u64) {
        self.trust_registry.register_new_agent(agent_id, timestamp);
    }

    /// Vouch for an agent (increases their trust)
    pub fn vouch_for_agent(&mut self, agent_id: SymthaeaId, strength: f32, timestamp: u64) {
        let ctx = self.trust_registry.get_or_create(agent_id, timestamp);
        ctx.apply_vouch(strength);
    }

    /// Revoke a vouch for an agent
    pub fn revoke_vouch(&mut self, agent_id: SymthaeaId) {
        if let Some(ctx) = self.trust_registry.get_mut(agent_id) {
            ctx.revoke_vouch();
        }
    }

    /// Get the trust score for an agent
    pub fn agent_trust_score(&self, agent_id: SymthaeaId) -> f32 {
        self.trust_registry.trust_score(agent_id)
    }

    /// Get trust-weighted score for a pattern
    pub fn trust_weighted_pattern_score(&self, pattern_id: PatternId) -> Option<TrustWeightedScore> {
        let pattern = self.patterns.get(&pattern_id)?;
        Some(self.trust_registry.calculate_weighted_score(
            pattern.success_rate,
            pattern.usage_count,
            pattern.learned_by,
        ))
    }

    /// Get all patterns sorted by trust-weighted score
    pub fn patterns_by_trust_weighted_score(&self) -> Vec<(PatternId, TrustWeightedScore)> {
        let mut scored: Vec<_> = self.patterns
            .iter()
            .filter_map(|(id, pattern)| {
                let score = self.trust_registry.calculate_weighted_score(
                    pattern.success_rate,
                    pattern.usage_count,
                    pattern.learned_by,
                );
                if score.meets_threshold {
                    Some((*id, score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.weighted_score.partial_cmp(&a.1.weighted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored
    }

    /// Get the best pattern for a domain considering trust
    pub fn best_trusted_pattern_for_domain(
        &self,
        domain: &str,
        current_time: u64,
    ) -> Option<(PatternId, TrustWeightedScore)> {
        self.patterns
            .iter()
            .filter(|(_, p)| p.problem_domain == domain)
            .filter_map(|(id, pattern)| {
                let score = self.trust_registry.calculate_weighted_score(
                    pattern.success_rate,
                    pattern.usage_count,
                    pattern.learned_by,
                );

                if !score.meets_threshold {
                    return None;
                }

                // Apply temporal decay if configured
                let decay = self.temporal_decay.calculate_decay(pattern.last_used, current_time);
                let effective_score = TrustWeightedScore {
                    weighted_score: score.weighted_score * decay,
                    ..score
                };

                Some((*id, effective_score))
            })
            .max_by(|a, b| {
                a.1.weighted_score.partial_cmp(&b.1.weighted_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Update agent trust after a pattern outcome
    /// This is called automatically by on_pattern_used if trust weighting is enabled
    fn update_agent_trust_from_outcome(&mut self, pattern_id: PatternId, success: bool, timestamp: u64) {
        if let Some(pattern) = self.patterns.get(&pattern_id) {
            self.trust_registry.update_from_outcome(pattern.learned_by, success, timestamp);
        }
    }

    /// Apply trust decay to all agents
    pub fn apply_trust_decay(&mut self, current_time: u64) {
        self.trust_registry.apply_decay_all(current_time);
    }

    /// Get trust statistics for the system
    pub fn trust_stats(&self) -> TrustRegistryStats {
        self.trust_registry.stats()
    }

    /// Filter patterns by minimum trust level
    pub fn patterns_with_trust_level(&self, min_level: TrustLevel) -> Vec<PatternId> {
        self.patterns
            .iter()
            .filter(|(_, pattern)| {
                let trust = self.trust_registry.trust_score(pattern.learned_by);
                let level = TrustLevel::from_score(trust);
                match (min_level, level) {
                    (TrustLevel::VeryHigh, TrustLevel::VeryHigh) => true,
                    (TrustLevel::High, TrustLevel::VeryHigh | TrustLevel::High) => true,
                    (TrustLevel::Neutral, TrustLevel::VeryHigh | TrustLevel::High | TrustLevel::Neutral) => true,
                    (TrustLevel::Low, TrustLevel::VeryHigh | TrustLevel::High | TrustLevel::Neutral | TrustLevel::Low) => true,
                    (TrustLevel::Untrusted, _) => true, // All patterns pass
                    _ => false,
                }
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get patterns from high-trust agents (trust > 0.7)
    pub fn high_trust_patterns(&self) -> Vec<PatternId> {
        self.patterns_with_trust_level(TrustLevel::High)
    }

    /// Resolve predictions for a pattern using its oracle
    fn resolve_pattern_predictions(&mut self, pattern_id: PatternId, timestamp: u64) {
        let oracle = match self.pattern_oracles.get(&pattern_id) {
            Some(o) => o,
            None => return,
        };

        let observation = match oracle.consensus_observation() {
            Some(o) => o,
            None => return,
        };

        // Update the prediction in the causal graph
        if let Some(pred_id) = self.pattern_predictions.get(&pattern_id) {
            for prediction in &mut self.causal.predictions {
                if prediction.id == *pred_id && !prediction.resolved {
                    // Apply trust-adjusted observation
                    let trust_multiplier = oracle.verification_level().trust_multiplier();
                    let adjusted_value = observation.value * trust_multiplier
                        + prediction.predicted_value * (1.0 - trust_multiplier);

                    prediction.resolve(adjusted_value, timestamp);

                    // Track accuracy
                    if let Some(success) = prediction.is_successful(self.causal.error_threshold) {
                        if success {
                            self.causal.successful_predictions += 1;
                        }
                    }
                }
            }
        }
    }

    /// Add swarm validation results from another Symthaea instance
    pub fn add_swarm_result(&mut self, pattern_id: PatternId, result: SwarmInstanceResult) {
        if self.swarm_oracle.is_none() {
            self.swarm_oracle = Some(SwarmPatternOracle::new(pattern_id, 3));
        }

        if let Some(ref mut oracle) = self.swarm_oracle {
            if oracle.pattern_id == pattern_id {
                oracle.add_instance_result(result);
            }
        }
    }

    /// Update consciousness level
    ///
    /// This should be called when Symthaea's Φ measurement changes
    pub fn update_phi(&mut self, phi: f32, timestamp: u64) {
        if let Some(node_id) = self.consciousness_node {
            if let Some(node) = self.causal.nodes.get_mut(&node_id) {
                node.update(phi, timestamp);
            }
        }
    }

    /// Generate learning guidance based on causal feedback
    ///
    /// This is the key output: guidance that tells Symthaea how to improve
    pub fn generate_guidance(&mut self) -> LearningGuidance {
        let mut guidance = LearningGuidance::default();

        // Analyze patterns by their prediction errors
        let mut patterns_with_errors: Vec<(PatternId, f32)> = Vec::new();

        for (pattern_id, pred_id) in &self.pattern_predictions {
            for prediction in &self.causal.predictions {
                if prediction.id == *pred_id && prediction.resolved {
                    if let Some(error) = prediction.error {
                        patterns_with_errors.push((*pattern_id, error));
                    }
                }
            }
        }

        // Categorize patterns
        for (pattern_id, error) in patterns_with_errors {
            if error > 0.2 {
                // Over-predicted success: downgrade
                guidance.patterns_to_downgrade.push(PatternDowngrade {
                    pattern_id,
                    downgrade_factor: error.min(0.5),
                    reason: format!(
                        "Pattern over-predicted success by {:.1}%. Reality was worse than expected.",
                        error * 100.0
                    ),
                });
            } else if error < -0.2 {
                // Under-predicted success: upgrade
                guidance.patterns_to_upgrade.push(PatternUpgrade {
                    pattern_id,
                    upgrade_factor: (-error).min(0.5),
                    reason: format!(
                        "Pattern under-predicted success by {:.1}%. Reality was better than expected.",
                        -error * 100.0
                    ),
                });
            }
        }

        // Check if high-Φ patterns perform better
        let mut high_phi_success = 0.0;
        let mut high_phi_count = 0;
        let mut low_phi_success = 0.0;
        let mut low_phi_count = 0;

        for pattern in self.patterns.values() {
            if pattern.usage_count > 0 {
                if pattern.phi_at_learning > 0.5 {
                    high_phi_success += pattern.success_rate;
                    high_phi_count += 1;
                } else {
                    low_phi_success += pattern.success_rate;
                    low_phi_count += 1;
                }
            }
        }

        if high_phi_count > 0 && low_phi_count > 0 {
            let high_avg = high_phi_success / high_phi_count as f32;
            let low_avg = low_phi_success / low_phi_count as f32;
            guidance.favor_high_phi_patterns = high_avg > low_avg;

            if high_avg > low_avg + 0.1 {
                guidance.explanation = format!(
                    "Patterns learned at higher consciousness (Φ>{:.1}) have {:.1}% higher success rate. \
                     Favor learning when more conscious.",
                    0.5,
                    (high_avg - low_avg) * 100.0
                );
            }
        }

        // Adjust learning rate based on overall accuracy
        let accuracy = self.causal.current_accuracy();
        if accuracy > 0.8 {
            guidance.learning_rate_multiplier = 0.8; // Slow down, we're doing well
        } else if accuracy < 0.5 {
            guidance.learning_rate_multiplier = 1.5; // Speed up, need to adapt
        }

        // Store in history
        self.guidance_history.push(guidance.clone());

        guidance
    }

    /// Apply learning guidance to patterns
    ///
    /// This actually modifies pattern confidence based on guidance
    pub fn apply_guidance(&mut self, guidance: &LearningGuidance) {
        for downgrade in &guidance.patterns_to_downgrade {
            if let Some(pattern) = self.patterns.get_mut(&downgrade.pattern_id) {
                // Reduce success rate based on downgrade factor
                pattern.success_rate = (pattern.success_rate - downgrade.downgrade_factor).max(0.0);
            }
        }

        for upgrade in &guidance.patterns_to_upgrade {
            if let Some(pattern) = self.patterns.get_mut(&upgrade.pattern_id) {
                // Increase success rate based on upgrade factor
                pattern.success_rate = (pattern.success_rate + upgrade.upgrade_factor).min(1.0);
            }
        }
    }

    /// Get prediction accuracy for patterns
    pub fn pattern_accuracy(&self) -> f32 {
        self.causal.current_accuracy()
    }

    /// Get list of patterns needing review (large prediction errors)
    pub fn patterns_needing_review(&self) -> Vec<(PatternId, f32)> {
        let mut results = Vec::new();

        for (pattern_id, pred_id) in &self.pattern_predictions {
            for prediction in &self.causal.predictions {
                if prediction.id == *pred_id && prediction.resolved {
                    if let Some(error) = prediction.error {
                        if error.abs() > self.causal.error_threshold {
                            results.push((*pattern_id, error));
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get system health report
    pub fn health_report(&self) -> BridgeHealthReport {
        let total_patterns = self.patterns.len();
        let patterns_with_usage = self.patterns.values().filter(|p| p.usage_count > 0).count();
        let patterns_validated = self.patterns.values().filter(|p| p.validated_in_production).count();

        let avg_success_rate = if patterns_with_usage > 0 {
            self.patterns.values()
                .filter(|p| p.usage_count > 0)
                .map(|p| p.success_rate)
                .sum::<f32>() / patterns_with_usage as f32
        } else {
            0.0
        };

        BridgeHealthReport {
            total_patterns,
            patterns_with_usage,
            patterns_validated,
            prediction_accuracy: self.causal.current_accuracy(),
            average_pattern_success: avg_success_rate,
            guidance_generated: self.guidance_history.len(),
            swarm_instances: self.swarm_oracle.as_ref().map(|o| o.instance_results.len()).unwrap_or(0),
        }
    }

    // ==========================================================================
    // Component 13: Collective Pattern Integration Methods
    // ==========================================================================

    /// Record a pattern discovery in a collective context
    ///
    /// This is called when a pattern is learned while observing the collective field.
    /// Tracks independent discoveries to detect emergence.
    pub fn record_collective_discovery(
        &mut self,
        pattern_id: PatternId,
        discoverer: SymthaeaId,
        timestamp: u64,
    ) {
        if !self.use_collective_observation {
            return;
        }

        // Ensure pattern has a context
        if !self.collective_registry.has_context(pattern_id) {
            self.collective_registry.register_pattern(pattern_id, timestamp);
        }

        // Record the discovery
        self.collective_registry.record_discovery(pattern_id, discoverer, timestamp);
    }

    /// Record pattern usage with collective agreement level
    ///
    /// Tracks whether patterns are being used during high-agreement periods
    /// (potential echo chamber) or diverse contexts.
    pub fn record_collective_usage(
        &mut self,
        pattern_id: PatternId,
        agreement_level: f32,
        timestamp: u64,
    ) {
        if !self.use_collective_observation {
            return;
        }

        // Ensure pattern has a context
        if !self.collective_registry.has_context(pattern_id) {
            self.collective_registry.register_pattern(pattern_id, timestamp);
        }

        self.collective_registry.record_usage_agreement(pattern_id, agreement_level, timestamp);
    }

    /// Record tension change after pattern usage
    ///
    /// Tracks whether pattern usage reduces or increases group tension.
    pub fn record_tension_change(
        &mut self,
        pattern_id: PatternId,
        tension_before: f32,
        tension_after: f32,
        timestamp: u64,
    ) {
        if !self.use_collective_observation {
            return;
        }

        // Ensure pattern has a context
        if !self.collective_registry.has_context(pattern_id) {
            self.collective_registry.register_pattern(pattern_id, timestamp);
        }

        self.collective_registry.record_tension_change(
            pattern_id,
            tension_before,
            tension_after,
            timestamp,
        );
    }

    /// Mark a pattern as being in the "shadow" (needed but absent from collective)
    pub fn mark_pattern_in_shadow(&mut self, pattern_id: PatternId, in_shadow: bool, timestamp: u64) {
        if !self.use_collective_observation {
            return;
        }

        if !self.collective_registry.has_context(pattern_id) {
            self.collective_registry.register_pattern(pattern_id, timestamp);
        }

        self.collective_registry.set_shadow_status(pattern_id, in_shadow, timestamp);
    }

    /// Get collective-adjusted score for a pattern
    ///
    /// Combines the base success rate with collective observations:
    /// - Emergent patterns get a boost
    /// - Echo chamber patterns get a penalty
    /// - Tension-resolving patterns get a boost
    /// - Shadow patterns get a penalty
    pub fn collective_pattern_score(&self, pattern_id: PatternId, _timestamp: u64) -> Option<f32> {
        let pattern = self.patterns.get(&pattern_id)?;
        let base_score = pattern.success_rate;

        if !self.use_collective_observation {
            return Some(base_score);
        }

        let modifier = self.collective_registry.get_modifier(pattern_id);
        let weight = self.collective_registry.config.collective_weight;

        // Blend base score with collective modifier
        // modifier ranges from -1.0 to +1.0
        // positive modifier boosts, negative modifier penalizes
        let adjusted = base_score * (1.0 + modifier * weight);
        Some(adjusted.clamp(0.0, 1.0))
    }

    /// Get patterns that show emergence (independently discovered by multiple agents)
    pub fn emergent_patterns(&self) -> Vec<PatternId> {
        self.collective_registry.emergent_patterns()
    }

    /// Get patterns with high echo chamber risk
    pub fn echo_chamber_patterns(&self) -> Vec<PatternId> {
        self.collective_registry.echo_chamber_patterns()
    }

    /// Get patterns that resolve tension
    pub fn tension_resolving_patterns(&self) -> Vec<PatternId> {
        self.collective_registry.tension_resolving_patterns()
    }

    /// Get patterns currently in the shadow
    pub fn shadow_patterns(&self) -> Vec<PatternId> {
        self.collective_registry.shadow_patterns()
    }

    /// Get collective statistics
    pub fn collective_stats(&self) -> CollectivePatternStats {
        self.collective_registry.stats()
    }

    /// Best pattern for a domain with collective awareness
    ///
    /// Like `best_pattern_for_domain` but factors in collective observations.
    pub fn best_collective_pattern_for_domain(
        &self,
        domain: &str,
        timestamp: u64,
    ) -> Option<PatternDecayStatus> {
        // First, collect candidates with decay status
        let decay_config = self.temporal_decay.clone();
        let collective_weight = self.collective_registry.config.collective_weight;
        let use_collective = self.use_collective_observation;

        let mut best_id: Option<PatternId> = None;
        let mut best_score: f32 = -1.0;

        for (id, pattern) in &self.patterns {
            if pattern.problem_domain != domain {
                continue;
            }

            // Calculate decay
            let decay_factor = decay_config.calculate_decay(pattern.last_used, timestamp);
            let base_rate = pattern.success_rate * decay_factor;

            // Apply collective modifier
            let collective_modifier = if use_collective {
                self.collective_registry.get_modifier(*id)
            } else {
                0.0
            };

            let adjusted = base_rate * (1.0 + collective_modifier * collective_weight);
            let score = adjusted.clamp(0.0, 1.0);

            if score > best_score {
                best_score = score;
                best_id = Some(*id);
            }
        }

        best_id.map(|id| {
            let pattern = self.patterns.get(&id).unwrap();
            let decay_factor = decay_config.calculate_decay(pattern.last_used, timestamp);
            let days_since = (timestamp.saturating_sub(pattern.last_used)) as f32 / (24.0 * 60.0 * 60.0);
            let is_stale = decay_config.is_stale(pattern.last_used, timestamp);

            PatternDecayStatus {
                pattern_id: id,
                original_success_rate: pattern.success_rate,
                effective_success_rate: best_score,
                decay_factor,
                days_since_validation: days_since,
                is_stale,
                needs_revalidation: is_stale || decay_factor < 0.5,
            }
        })
    }

    /// Process collective field reading and update pattern modifiers
    ///
    /// This should be called periodically with the current collective field state.
    /// Updates all pattern modifiers based on collective metrics.
    pub fn observe_collective_field(
        &mut self,
        coherence: f32,
        tension: f32,
        wisdom_density: f32,
        emergence_level: f32,
        _timestamp: u64,
    ) {
        if !self.use_collective_observation {
            return;
        }

        // Update all pattern modifiers based on current collective state
        // High coherence + low tension = patterns are working well
        // High emergence = collective is innovating
        let field_health = coherence * (1.0 - tension.min(1.0)) * (1.0 + emergence_level * 0.5);

        // If field is very healthy, boost all pattern modifiers slightly
        // If field is stressed, reduce modifiers
        if field_health > 0.7 {
            // Healthy field - update modifiers positively
            self.collective_registry.update_all_modifiers();
        }

        // Store wisdom density for future reference
        // Higher wisdom density suggests patterns are crystallizing well
        let _ = wisdom_density; // Used for future expansion
    }

    // ==========================================================================
    // Component 14: Pattern Lifecycle Management Methods
    // ==========================================================================

    /// Manually deprecate a pattern with a given reason
    pub fn deprecate_pattern(
        &mut self,
        pattern_id: PatternId,
        reason: LifecycleTransitionReason,
        timestamp: u64,
    ) -> Result<(), &'static str> {
        if !self.patterns.contains_key(&pattern_id) {
            return Err("Pattern not found");
        }
        if self.lifecycle_registry.deprecate(pattern_id, reason, timestamp, "manual") {
            Ok(())
        } else {
            Err("Failed to deprecate pattern")
        }
    }

    /// Manually archive a pattern
    pub fn archive_pattern(
        &mut self,
        pattern_id: PatternId,
        reason: LifecycleTransitionReason,
        timestamp: u64,
    ) -> Result<(), &'static str> {
        if !self.patterns.contains_key(&pattern_id) {
            return Err("Pattern not found");
        }
        if self.lifecycle_registry.archive(pattern_id, reason, timestamp, "manual") {
            Ok(())
        } else {
            Err("Failed to archive pattern")
        }
    }

    /// Manually retire a pattern (permanent removal)
    pub fn retire_pattern(
        &mut self,
        pattern_id: PatternId,
        reason: LifecycleTransitionReason,
        timestamp: u64,
    ) -> Result<(), &'static str> {
        if !self.patterns.contains_key(&pattern_id) {
            return Err("Pattern not found");
        }
        if self.lifecycle_registry.retire(pattern_id, reason, timestamp, "manual") {
            Ok(())
        } else {
            Err("Failed to retire pattern")
        }
    }

    /// Resurrect a deprecated or archived pattern back to active
    pub fn resurrect_pattern(
        &mut self,
        pattern_id: PatternId,
        reason: String,
        timestamp: u64,
    ) -> Result<(), &'static str> {
        if !self.patterns.contains_key(&pattern_id) {
            return Err("Pattern not found");
        }
        if self.lifecycle_registry.resurrect(pattern_id, reason, timestamp, "manual") {
            Ok(())
        } else {
            Err("Failed to resurrect pattern")
        }
    }

    /// Get the current lifecycle state of a pattern
    pub fn pattern_lifecycle_state(&self, pattern_id: PatternId) -> Option<PatternLifecycleState> {
        if self.lifecycle_registry.get(pattern_id).is_some() {
            Some(self.lifecycle_registry.state(pattern_id))
        } else {
            None
        }
    }

    /// Check if a pattern is active (can be used normally)
    pub fn is_pattern_active(&self, pattern_id: PatternId) -> bool {
        self.lifecycle_registry.state(pattern_id) == PatternLifecycleState::Active
    }

    /// Check if a pattern is deprecated (still usable but warns)
    pub fn is_pattern_deprecated(&self, pattern_id: PatternId) -> bool {
        self.lifecycle_registry.state(pattern_id) == PatternLifecycleState::Deprecated
    }

    /// Get all active patterns
    pub fn active_patterns(&self) -> Vec<PatternId> {
        self.lifecycle_registry.patterns_by_state(PatternLifecycleState::Active)
    }

    /// Get all deprecated patterns
    pub fn deprecated_patterns(&self) -> Vec<PatternId> {
        self.lifecycle_registry.patterns_by_state(PatternLifecycleState::Deprecated)
    }

    /// Get all archived patterns
    pub fn archived_patterns(&self) -> Vec<PatternId> {
        self.lifecycle_registry.patterns_by_state(PatternLifecycleState::Archived)
    }

    /// Get lifecycle statistics
    pub fn lifecycle_stats(&self) -> LifecycleStats {
        self.lifecycle_registry.stats()
    }

    /// Run auto-deprecation check for all patterns
    /// Returns list of patterns that were deprecated
    pub fn run_auto_deprecation(&mut self, timestamp: u64) -> Vec<(PatternId, LifecycleTransitionReason)> {
        if !self.auto_lifecycle_management {
            return Vec::new();
        }

        let mut deprecated = Vec::new();

        // Collect pattern data for analysis
        let pattern_data: Vec<(PatternId, u32, u64, f32, bool)> = self
            .patterns
            .iter()
            .map(|(&id, p)| {
                let trust_score = if self.use_trust_weighting {
                    // Get trust from the pattern's creator if tracked
                    self.trust_registry
                        .get(p.phi_at_learning as u64) // Use phi as pseudo-ID for now
                        .map(|ctx| ctx.k_vector.trust_score())
                        .unwrap_or(0.5)
                } else {
                    p.success_rate
                };
                let is_active = self.lifecycle_registry.state(id) == PatternLifecycleState::Active;
                (id, p.usage_count as u32, p.last_used, trust_score, is_active)
            })
            .collect();

        // Check each active pattern for auto-deprecation
        for (pattern_id, usage_count, last_used, trust_score, is_active) in pattern_data {
            if !is_active {
                continue;
            }

            if let Some(reason) = self.lifecycle_registry.check_auto_deprecation(
                pattern_id,
                usage_count,
                last_used,
                trust_score,
                timestamp,
            ) {
                if self.lifecycle_registry.deprecate(pattern_id, reason.clone(), timestamp, "auto") {
                    deprecated.push((pattern_id, reason));
                }
            }
        }

        deprecated
    }

    /// Check and auto-archive patterns whose deprecation period has expired
    pub fn run_auto_archive(&mut self, timestamp: u64) -> Vec<PatternId> {
        if !self.auto_lifecycle_management {
            return Vec::new();
        }
        self.lifecycle_registry.auto_archive_expired(timestamp)
    }

    /// Get the best pattern for a domain, excluding non-active patterns
    pub fn best_active_pattern_for_domain(
        &self,
        domain: &str,
        timestamp: u64,
    ) -> Option<PatternDecayStatus> {
        // Get all patterns in the domain that are active
        let domain_patterns: Vec<_> = self
            .patterns
            .iter()
            .filter(|(id, p)| {
                p.problem_domain == domain && self.lifecycle_registry.state(**id) == PatternLifecycleState::Active
            })
            .collect();

        if domain_patterns.is_empty() {
            return None;
        }

        // Find the best one using temporal decay
        let decay_config = &self.temporal_decay;
        domain_patterns
            .into_iter()
            .map(|(&id, pattern)| {
                let decay = decay_config.calculate_decay(pattern.last_used, timestamp);
                PatternDecayStatus {
                    pattern_id: id,
                    original_success_rate: pattern.success_rate,
                    effective_success_rate: pattern.success_rate * decay,
                    decay_factor: decay,
                    days_since_validation: (timestamp.saturating_sub(pattern.last_used) as f32)
                        / (24.0 * 60.0 * 60.0),
                    is_stale: decay < 0.5,
                    needs_revalidation: decay < 0.7,
                }
            })
            .max_by(|a, b| {
                a.effective_success_rate
                    .partial_cmp(&b.effective_success_rate)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
    }

    /// Get lifecycle transition history for a pattern
    pub fn pattern_lifecycle_history(&self, pattern_id: PatternId) -> Vec<&LifecycleTransition> {
        self.lifecycle_registry.transitions_for(pattern_id)
    }

    /// Register a pattern with lifecycle tracking
    /// This is called automatically when on_pattern_learned is used
    pub fn register_pattern_lifecycle(&mut self, pattern_id: PatternId, timestamp: u64) {
        self.lifecycle_registry.register(pattern_id, timestamp);
    }

    /// Mark pattern as superseded by another pattern
    pub fn supersede_pattern(
        &mut self,
        old_pattern_id: PatternId,
        new_pattern_id: PatternId,
        timestamp: u64,
    ) -> Result<(), &'static str> {
        if !self.patterns.contains_key(&old_pattern_id) {
            return Err("Old pattern not found");
        }
        if !self.patterns.contains_key(&new_pattern_id) {
            return Err("New pattern not found");
        }

        let reason = LifecycleTransitionReason::Superseded {
            replacement_id: new_pattern_id,
        };
        if self.lifecycle_registry.deprecate(old_pattern_id, reason, timestamp, "supersede") {
            Ok(())
        } else {
            Err("Failed to deprecate pattern")
        }
    }

    // ==========================================================================
    // Component 15: Pattern Versioning & Evolution Bridge Methods
    // ==========================================================================

    /// Register an initial version for a pattern
    pub fn register_pattern_version(
        &mut self,
        pattern_id: PatternId,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Result<PatternVersion, &'static str> {
        let pattern = self.patterns.get(&pattern_id).ok_or("Pattern not found")?;
        self.version_registry.register_initial(
            pattern_id,
            pattern.solution.clone(),
            timestamp,
            creator,
        );
        Ok(PatternVersion::initial())
    }

    /// Get the current version of a pattern
    pub fn pattern_version(&self, pattern_id: PatternId) -> Option<PatternVersion> {
        self.version_registry.current_version(pattern_id).map(|v| v.version)
    }

    /// Get full version info for a pattern
    pub fn pattern_version_info(&self, pattern_id: PatternId) -> Option<&PatternVersionInfo> {
        self.version_registry.current_version(pattern_id)
    }

    /// Evolve a pattern to a new version
    pub fn evolve_pattern(
        &mut self,
        pattern_id: PatternId,
        reason: PatternEvolutionReason,
        new_solution: impl Into<String>,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Result<PatternVersion, &'static str> {
        let pattern = self.patterns.get(&pattern_id).ok_or("Pattern not found")?;
        let success_rate = pattern.success_rate;
        let usage_count = pattern.usage_count as u64;

        self.version_registry.evolve(
            pattern_id,
            reason,
            new_solution.into(),
            success_rate,
            usage_count,
            timestamp,
            creator,
        ).ok_or("Failed to evolve pattern - no current version found")
    }

    /// Get all versions of a pattern
    pub fn pattern_versions(&self, pattern_id: PatternId) -> Vec<&PatternVersionInfo> {
        self.version_registry.all_versions(pattern_id)
    }

    /// Get a specific version of a pattern
    pub fn get_pattern_version(
        &self,
        pattern_id: PatternId,
        version: PatternVersion,
    ) -> Option<&PatternVersionInfo> {
        self.version_registry.get_version(pattern_id, version)
    }

    /// Create an experimental branch for a pattern
    pub fn create_pattern_branch(
        &mut self,
        pattern_id: PatternId,
        branch_name: impl Into<String>,
        timestamp: u64,
        creator: impl Into<String>,
    ) -> Result<(), &'static str> {
        if !self.patterns.contains_key(&pattern_id) {
            return Err("Pattern not found");
        }
        self.version_registry.create_branch(pattern_id, branch_name, timestamp, creator)
    }

    /// Get active branches for a pattern
    pub fn pattern_branches(&self, pattern_id: PatternId) -> Vec<&PatternBranch> {
        self.version_registry.active_branches(pattern_id)
    }

    /// Merge a branch into the main line
    pub fn merge_pattern_branch(
        &mut self,
        pattern_id: PatternId,
        branch_name: &str,
        timestamp: u64,
        merger: impl Into<String>,
    ) -> Result<PatternVersion, &'static str> {
        if !self.patterns.contains_key(&pattern_id) {
            return Err("Pattern not found");
        }
        self.version_registry.merge_branch(pattern_id, branch_name, timestamp, merger)
    }

    /// Rollback a pattern to a previous version
    pub fn rollback_pattern_version(
        &mut self,
        pattern_id: PatternId,
        target_version: PatternVersion,
        reason: impl Into<String>,
        timestamp: u64,
        initiator: impl Into<String>,
    ) -> Result<PatternVersion, &'static str> {
        if !self.patterns.contains_key(&pattern_id) {
            return Err("Pattern not found");
        }
        self.version_registry.rollback(pattern_id, target_version, reason, timestamp, initiator)
    }

    /// Get the version lineage (history) for a pattern
    /// Returns (version, parent_version) pairs
    pub fn pattern_version_lineage(&self, pattern_id: PatternId) -> Vec<(PatternVersion, Option<PatternVersion>)> {
        self.version_registry.version_lineage(pattern_id)
    }

    /// Check if a pattern should be auto-versioned based on performance changes
    pub fn should_auto_version(
        &self,
        pattern_id: PatternId,
        new_success_rate: f32,
    ) -> bool {
        self.auto_versioning && self.version_registry.should_auto_version(pattern_id, new_success_rate)
    }

    /// Auto-evolve a pattern if it meets the threshold
    pub fn auto_evolve_if_needed(
        &mut self,
        pattern_id: PatternId,
        new_success_rate: f32,
        timestamp: u64,
    ) -> Option<PatternVersion> {
        if !self.auto_versioning {
            return None;
        }

        let pattern = self.patterns.get(&pattern_id)?;
        let old_success_rate = self.version_registry
            .current_version(pattern_id)
            .map(|v| v.success_rate_snapshot)
            .unwrap_or(0.0);

        if !self.version_registry.should_auto_version(pattern_id, new_success_rate) {
            return None;
        }

        let reason = PatternEvolutionReason::PerformanceImprovement {
            old_success_rate,
            new_success_rate,
        };

        self.version_registry.evolve(
            pattern_id,
            reason,
            pattern.solution.clone(),
            new_success_rate,
            pattern.usage_count as u64,
            timestamp,
            "auto",
        )
    }

    /// Prune old versions for all patterns
    pub fn prune_pattern_versions(&mut self, current_time: u64) {
        // Get all pattern IDs first to avoid borrow issues
        let pattern_ids: Vec<PatternId> = self.version_registry.versions().keys().copied().collect();
        for pattern_id in pattern_ids {
            self.version_registry.prune_versions(pattern_id, current_time);
        }
    }

    /// Get versioning statistics
    pub fn versioning_stats(&self) -> VersioningStats {
        self.version_registry.stats()
    }

    /// Enable or disable automatic versioning
    pub fn set_auto_versioning(&mut self, enabled: bool) {
        self.auto_versioning = enabled;
    }

    /// Configure versioning settings
    pub fn configure_versioning(&mut self, config: VersioningConfig) {
        self.version_registry.config = config;
    }

    /// Get patterns with the most versions (potentially over-evolved)
    pub fn patterns_with_many_versions(&self, threshold: usize) -> Vec<(PatternId, usize)> {
        self.version_registry.versions()
            .iter()
            .filter_map(|(id, versions)| {
                if versions.len() >= threshold {
                    Some((*id, versions.len()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get patterns that have branches
    pub fn patterns_with_branches(&self) -> Vec<(PatternId, usize)> {
        self.version_registry.branches()
            .iter()
            .filter_map(|(id, branches)| {
                let active = branches.iter().filter(|b| b.is_active).count();
                if active > 0 {
                    Some((*id, active))
                } else {
                    None
                }
            })
            .collect()
    }

    // ==========================================================================
    // Component 16: Pattern Dependencies & Prerequisites - Bridge Methods
    // ==========================================================================

    /// Add a dependency between two patterns
    pub fn add_pattern_dependency(&mut self, dependency: PatternDependency) {
        self.dependency_registry.add_dependency(dependency);
    }

    /// Add a prerequisite dependency (from_pattern must be used before to_pattern)
    pub fn add_prerequisite(
        &mut self,
        prerequisite: PatternId,
        dependent: PatternId,
        strength: DependencyStrength,
        timestamp: u64,
        creator: impl Into<String>,
    ) {
        let dep = PatternDependency::new(
            prerequisite,
            dependent,
            PatternRelationType::Prerequisite,
            strength,
            timestamp,
            creator,
        );
        self.dependency_registry.add_dependency(dep);
    }

    /// Add a requirement dependency (to_pattern requires from_pattern)
    pub fn add_requirement(
        &mut self,
        required: PatternId,
        requiring: PatternId,
        strength: DependencyStrength,
        timestamp: u64,
        creator: impl Into<String>,
    ) {
        let dep = PatternDependency::new(
            required,
            requiring,
            PatternRelationType::Requires,
            strength,
            timestamp,
            creator,
        );
        self.dependency_registry.add_dependency(dep);
    }

    /// Add a conflict between two patterns
    pub fn add_conflict(
        &mut self,
        pattern_a: PatternId,
        pattern_b: PatternId,
        timestamp: u64,
        creator: impl Into<String>,
    ) {
        let dep = PatternDependency::new(
            pattern_a,
            pattern_b,
            PatternRelationType::ConflictsWith,
            DependencyStrength::Required,
            timestamp,
            creator,
        );
        self.dependency_registry.add_dependency(dep);
    }

    /// Add an enhancement relationship
    pub fn add_enhancement(
        &mut self,
        enhancing: PatternId,
        enhanced: PatternId,
        strength: DependencyStrength,
        timestamp: u64,
        creator: impl Into<String>,
    ) {
        let dep = PatternDependency::new(
            enhancing,
            enhanced,
            PatternRelationType::EnhancedBy,
            strength,
            timestamp,
            creator,
        );
        self.dependency_registry.add_dependency(dep);
    }

    /// Check if a specific dependency exists between two patterns
    pub fn has_dependency(&self, from: PatternId, to: PatternId, relation_type: PatternRelationType) -> bool {
        self.dependency_registry.has_dependency(from, to, relation_type)
    }

    /// Check if any dependency exists between two patterns
    pub fn has_any_dependency(&self, from: PatternId, to: PatternId) -> bool {
        !self.dependency_registry.dependencies_from(from)
            .iter()
            .filter(|d| d.to_pattern == to)
            .collect::<Vec<_>>()
            .is_empty()
    }

    /// Get all dependencies from a pattern
    pub fn dependencies_from(&self, pattern_id: PatternId) -> Vec<&PatternDependency> {
        self.dependency_registry.dependencies_from(pattern_id)
    }

    /// Get all dependencies to a pattern
    pub fn dependencies_to(&self, pattern_id: PatternId) -> Vec<&PatternDependency> {
        self.dependency_registry.dependencies_to(pattern_id)
    }

    /// Get all dependencies of a specific type from a pattern
    pub fn dependencies_by_type(&self, pattern_id: PatternId, relation_type: PatternRelationType) -> Vec<&PatternDependency> {
        self.dependency_registry.dependencies_of_type(pattern_id, relation_type)
    }

    /// Get required patterns for a given pattern
    pub fn required_patterns(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.dependency_registry.required_patterns(pattern_id)
    }

    /// Get conflicting patterns for a given pattern
    pub fn conflicting_patterns(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.dependency_registry.conflicting_patterns(pattern_id)
    }

    /// Get prerequisites for a pattern
    pub fn pattern_prerequisites(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.dependency_registry.prerequisites(pattern_id)
    }

    /// Resolve all dependencies for a pattern (transitive)
    pub fn resolve_dependencies(&self, pattern_id: PatternId, active_patterns: &[PatternId]) -> DependencyResolution {
        self.dependency_registry.resolve(pattern_id, active_patterns)
    }

    /// Resolve dependencies with empty active patterns
    pub fn resolve_dependencies_simple(&self, pattern_id: PatternId) -> DependencyResolution {
        self.dependency_registry.resolve(pattern_id, &[])
    }

    /// Check if a pattern has circular dependencies
    pub fn has_circular_dependency(&self, pattern_id: PatternId) -> bool {
        self.dependency_registry.has_circular_dependency(pattern_id)
    }

    /// Record pattern co-occurrence for dependency discovery
    pub fn record_dependency_usage(
        &mut self,
        patterns_used: &[PatternId],
        was_successful: bool,
        timestamp: u64,
    ) {
        self.dependency_registry.record_usage(patterns_used, was_successful, timestamp);
    }

    /// Record two patterns used together
    pub fn record_dependency_cooccurrence(
        &mut self,
        first: PatternId,
        second: PatternId,
        was_successful: bool,
        timestamp: u64,
    ) {
        self.dependency_registry.record_usage(&[first, second], was_successful, timestamp);
    }

    /// Get discovered dependencies
    pub fn discovered_dependencies(&self) -> Vec<&PatternDependency> {
        self.dependency_registry.all_dependencies()
            .iter()
            .filter(|d| d.is_discovered)
            .collect()
    }

    /// Get patterns that depend on a given pattern (dependents)
    pub fn pattern_dependents(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.dependency_registry.dependents(pattern_id)
    }

    /// Check impact of deprecating a pattern
    pub fn deprecation_impact(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.dependency_registry.deprecation_impact(pattern_id)
    }

    /// Remove all dependencies for a pattern
    pub fn remove_pattern_dependencies(&mut self, pattern_id: PatternId) {
        self.dependency_registry.remove_pattern(pattern_id);
    }

    /// Get dependency statistics
    pub fn dependency_stats(&self) -> DependencyStats {
        self.dependency_registry.stats()
    }

    /// Get all dependencies
    pub fn all_dependencies(&self) -> &[PatternDependency] {
        self.dependency_registry.all_dependencies()
    }

    /// Enable or disable automatic dependency discovery
    pub fn set_auto_discover_dependencies(&mut self, enabled: bool) {
        self.auto_discover_dependencies = enabled;
    }

    /// Enable or disable dependency enforcement
    pub fn set_enforce_dependencies(&mut self, enabled: bool) {
        self.enforce_dependencies = enabled;
    }

    /// Configure dependency settings
    pub fn configure_dependencies(&mut self, config: DependencyConfig) {
        self.dependency_registry.config = config;
    }

    /// Check if a pattern can be used (all dependencies satisfied)
    pub fn can_use_pattern(&self, pattern_id: PatternId, used_patterns: &[PatternId]) -> Result<(), Vec<DependencyIssue>> {
        if !self.enforce_dependencies {
            return Ok(());
        }

        let resolution = self.resolve_dependencies(pattern_id, used_patterns);

        // Start with any issues found during resolution
        let mut issues = resolution.issues;

        // Check if all prerequisites have been used
        let missing_prereqs: Vec<DependencyIssue> = resolution.prerequisites
            .iter()
            .filter(|p| !used_patterns.contains(p))
            .map(|p| DependencyIssue {
                issue_type: DependencyIssueType::UnmetPrerequisite,
                patterns_involved: vec![*p],
                description: format!("Prerequisite pattern {} has not been used", p),
                is_blocking: true,
            })
            .collect();

        // Check if any conflicting patterns have been used
        let conflicts: Vec<DependencyIssue> = resolution.conflicts
            .iter()
            .filter(|c| used_patterns.contains(c))
            .map(|c| DependencyIssue {
                issue_type: DependencyIssueType::Conflict,
                patterns_involved: vec![*c],
                description: format!("Conflicting pattern {} has been used", c),
                is_blocking: true,
            })
            .collect();

        issues.extend(missing_prereqs);
        issues.extend(conflicts);

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    /// Get patterns with the most dependencies
    pub fn patterns_with_many_dependencies(&self, threshold: usize) -> Vec<(PatternId, usize)> {
        let mut counts: HashMap<PatternId, usize> = HashMap::new();

        for dep in self.dependency_registry.all_dependencies() {
            *counts.entry(dep.from_pattern).or_insert(0) += 1;
        }

        counts.into_iter()
            .filter(|(_, count)| *count >= threshold)
            .collect()
    }

    /// Get patterns that are most depended upon
    pub fn most_depended_patterns(&self, threshold: usize) -> Vec<(PatternId, usize)> {
        let mut counts: HashMap<PatternId, usize> = HashMap::new();

        for dep in self.dependency_registry.all_dependencies() {
            *counts.entry(dep.to_pattern).or_insert(0) += 1;
        }

        counts.into_iter()
            .filter(|(_, count)| *count >= threshold)
            .collect()
    }
}

/// Health report for the Symthaea-Causal bridge
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BridgeHealthReport {
    /// Total patterns registered
    pub total_patterns: usize,

    /// Patterns that have been used at least once
    pub patterns_with_usage: usize,

    /// Patterns validated in production
    pub patterns_validated: usize,

    /// Prediction accuracy (from CausalGraph)
    pub prediction_accuracy: f32,

    /// Average pattern success rate
    pub average_pattern_success: f32,

    /// Number of learning guidance generations
    pub guidance_generated: usize,

    /// Number of Symthaea instances in swarm
    pub swarm_instances: usize,
}

// ==============================================================================
// COMPONENT 9: SCIENTIFIC METHOD ENHANCEMENTS
// ==============================================================================
//
// Five enhancements that complete the "Scientific Method for AI":
//
// 1. TEMPORAL DECAY - Patterns must stay fresh; knowledge decays without revalidation
// 2. CONFIDENCE CALIBRATION - Are high-confidence predictions actually more accurate?
// 3. COUNTERFACTUAL REASONING - What would have happened with a different pattern?
// 4. ACTIVE EXPERIMENTATION - Deliberately test uncertain patterns
// 5. CAUSAL DISCOVERY - Automatically find pattern dependencies

// -----------------------------------------------------------------------------
// Enhancement 1: TEMPORAL PATTERN DECAY
// -----------------------------------------------------------------------------

/// Configuration for temporal decay of pattern confidence
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TemporalDecayConfig {
    /// Half-life in seconds (confidence halves after this time without validation)
    pub half_life_secs: u64,

    /// Minimum confidence after decay (never goes below this)
    pub floor_confidence: f32,

    /// Time threshold for "stale" warning (in seconds)
    pub staleness_threshold_secs: u64,

    /// Whether decay is enabled
    pub enabled: bool,
}

impl Default for TemporalDecayConfig {
    fn default() -> Self {
        Self {
            half_life_secs: 30 * 24 * 60 * 60, // 30 days
            floor_confidence: 0.1,
            staleness_threshold_secs: 90 * 24 * 60 * 60, // 90 days
            enabled: true,
        }
    }
}

/// Result of decay calculation for a pattern
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternDecayStatus {
    /// Pattern ID
    pub pattern_id: PatternId,

    /// Original (non-decayed) success rate
    pub original_success_rate: f32,

    /// Effective success rate after decay
    pub effective_success_rate: f32,

    /// Decay factor applied (0.0-1.0, where 1.0 = no decay)
    pub decay_factor: f32,

    /// Days since last validation
    pub days_since_validation: f32,

    /// Whether this pattern is considered stale
    pub is_stale: bool,

    /// Whether this pattern needs revalidation
    pub needs_revalidation: bool,
}

impl TemporalDecayConfig {
    /// Calculate decay factor based on time since last use
    pub fn calculate_decay(&self, last_used: u64, current_time: u64) -> f32 {
        if !self.enabled || current_time <= last_used {
            return 1.0;
        }

        let elapsed = current_time - last_used;
        let half_lives = elapsed as f64 / self.half_life_secs as f64;

        // Exponential decay: factor = 0.5^(elapsed / half_life)
        let decay = (0.5_f64).powf(half_lives) as f32;

        // Apply floor
        decay.max(self.floor_confidence)
    }

    /// Check if a pattern is stale
    pub fn is_stale(&self, last_used: u64, current_time: u64) -> bool {
        if !self.enabled || current_time <= last_used {
            return false;
        }
        (current_time - last_used) > self.staleness_threshold_secs
    }
}

// -----------------------------------------------------------------------------
// Enhancement 2: CONFIDENCE CALIBRATION
// -----------------------------------------------------------------------------

/// A single bucket in the calibration curve
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CalibrationBucket {
    /// Lower bound of confidence range (inclusive)
    pub confidence_min: f32,

    /// Upper bound of confidence range (exclusive)
    pub confidence_max: f32,

    /// Number of predictions in this bucket
    pub count: u32,

    /// Sum of predicted values
    pub predicted_sum: f32,

    /// Sum of actual values
    pub actual_sum: f32,

    /// Number of successes (actual matched predicted within threshold)
    pub successes: u32,
}

impl CalibrationBucket {
    /// Create a new bucket for a confidence range
    pub fn new(confidence_min: f32, confidence_max: f32) -> Self {
        Self {
            confidence_min,
            confidence_max,
            count: 0,
            predicted_sum: 0.0,
            actual_sum: 0.0,
            successes: 0,
        }
    }

    /// Add a prediction to this bucket
    pub fn add(&mut self, predicted: f32, actual: f32, success_threshold: f32) {
        self.count += 1;
        self.predicted_sum += predicted;
        self.actual_sum += actual;
        if (predicted - actual).abs() <= success_threshold {
            self.successes += 1;
        }
    }

    /// Get average predicted value in this bucket
    pub fn avg_predicted(&self) -> f32 {
        if self.count == 0 { 0.0 } else { self.predicted_sum / self.count as f32 }
    }

    /// Get average actual value in this bucket
    pub fn avg_actual(&self) -> f32 {
        if self.count == 0 { 0.0 } else { self.actual_sum / self.count as f32 }
    }

    /// Get calibration error for this bucket
    pub fn calibration_error(&self) -> f32 {
        if self.count == 0 { 0.0 } else { (self.avg_predicted() - self.avg_actual()).abs() }
    }
}

/// Calibration curve tracking prediction accuracy by confidence level
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CalibrationCurve {
    /// Calibration buckets (typically 10, for 0-10%, 10-20%, etc.)
    pub buckets: Vec<CalibrationBucket>,

    /// Brier score (lower is better, 0 = perfect)
    pub brier_score: f32,

    /// Overconfidence bias (positive = overconfident, negative = underconfident)
    pub overconfidence_bias: f32,

    /// Total predictions tracked
    pub total_predictions: u32,

    /// Success threshold for binary calibration
    pub success_threshold: f32,
}

impl Default for CalibrationCurve {
    fn default() -> Self {
        Self::new(10, 0.15)
    }
}

impl CalibrationCurve {
    /// Create a new calibration curve with N buckets
    pub fn new(num_buckets: usize, success_threshold: f32) -> Self {
        let bucket_size = 1.0 / num_buckets as f32;
        let buckets = (0..num_buckets)
            .map(|i| {
                let min = i as f32 * bucket_size;
                let max = (i + 1) as f32 * bucket_size;
                CalibrationBucket::new(min, max)
            })
            .collect();

        Self {
            buckets,
            brier_score: 0.0,
            overconfidence_bias: 0.0,
            total_predictions: 0,
            success_threshold,
        }
    }

    /// Add a prediction result to the calibration curve
    pub fn add_result(&mut self, confidence: f32, predicted: f32, actual: f32) {
        // Find the right bucket
        let confidence_clamped = confidence.clamp(0.0, 0.999);
        let bucket_idx = (confidence_clamped * self.buckets.len() as f32) as usize;
        let bucket_idx = bucket_idx.min(self.buckets.len() - 1);

        self.buckets[bucket_idx].add(predicted, actual, self.success_threshold);
        self.total_predictions += 1;

        // Update Brier score (running average)
        let error_sq = (predicted - actual).powi(2);
        let prev_total = (self.total_predictions - 1) as f32;
        let new_total = self.total_predictions as f32;
        self.brier_score = (self.brier_score * prev_total + error_sq) / new_total;

        // Update overconfidence bias
        let bias = predicted - actual;
        self.overconfidence_bias = (self.overconfidence_bias * prev_total + bias) / new_total;
    }

    /// Get calibration quality (0.0-1.0, higher is better)
    pub fn calibration_quality(&self) -> f32 {
        if self.total_predictions == 0 {
            return 0.5; // Unknown
        }

        // Average calibration error across buckets with data
        let (total_error, count) = self.buckets.iter()
            .filter(|b| b.count > 0)
            .fold((0.0, 0), |(sum, cnt), b| {
                (sum + b.calibration_error(), cnt + 1)
            });

        if count == 0 {
            return 0.5;
        }

        let avg_error = total_error / count as f32;
        // Convert to quality score (0 error = 1.0 quality)
        (1.0 - avg_error * 2.0).max(0.0)
    }

    /// Get confidence adjustment factor
    ///
    /// If system is overconfident, this will be < 1.0
    /// If system is underconfident, this will be > 1.0
    pub fn confidence_adjustment(&self) -> f32 {
        if self.total_predictions < 10 {
            return 1.0; // Not enough data
        }

        // If overconfident by 10%, multiply future confidence by 0.9
        1.0 - self.overconfidence_bias
    }

    /// Get reliability diagram data for visualization
    pub fn reliability_diagram(&self) -> Vec<(f32, f32, u32)> {
        self.buckets.iter()
            .filter(|b| b.count > 0)
            .map(|b| {
                let midpoint = (b.confidence_min + b.confidence_max) / 2.0;
                (midpoint, b.avg_actual(), b.count)
            })
            .collect()
    }
}

// -----------------------------------------------------------------------------
// Enhancement 3: COUNTERFACTUAL REASONING
// -----------------------------------------------------------------------------

/// A counterfactual analysis comparing what happened vs alternatives
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CounterfactualAnalysis {
    /// Pattern that was actually used
    pub actual_pattern_id: PatternId,

    /// Outcome that actually occurred
    pub actual_outcome: f32,

    /// Context in which the pattern was used
    pub context: String,

    /// Alternative patterns that could have been used
    pub alternatives: Vec<CounterfactualAlternative>,

    /// Best alternative (if any was better)
    pub best_alternative: Option<PatternId>,

    /// Opportunity cost (best alternative outcome - actual outcome)
    pub opportunity_cost: f32,

    /// Regret score (how much worse was our choice?)
    pub regret: f32,

    /// Timestamp of analysis
    pub timestamp: u64,
}

/// An alternative pattern in counterfactual analysis
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CounterfactualAlternative {
    /// Alternative pattern ID
    pub pattern_id: PatternId,

    /// Estimated outcome if this pattern had been used
    pub estimated_outcome: f32,

    /// Confidence in this estimate
    pub confidence: f32,

    /// How the estimate was calculated
    pub estimation_method: EstimationMethod,

    /// Whether this was a better choice
    pub would_have_been_better: bool,
}

/// Method used to estimate counterfactual outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EstimationMethod {
    /// Based on historical success rate in similar contexts
    HistoricalAverage,
    /// Based on causal model prediction
    CausalModel,
    /// Based on similar patterns' performance
    SimilarPatterns,
    /// Expert-provided estimate
    Expert,
}

impl CounterfactualAnalysis {
    /// Create a new counterfactual analysis
    pub fn new(
        actual_pattern_id: PatternId,
        actual_outcome: f32,
        context: &str,
        timestamp: u64,
    ) -> Self {
        Self {
            actual_pattern_id,
            actual_outcome,
            context: context.to_string(),
            alternatives: Vec::new(),
            best_alternative: None,
            opportunity_cost: 0.0,
            regret: 0.0,
            timestamp,
        }
    }

    /// Add an alternative pattern analysis
    pub fn add_alternative(&mut self, alt: CounterfactualAlternative) {
        let would_have_been_better = alt.estimated_outcome > self.actual_outcome;
        let mut alt = alt;
        alt.would_have_been_better = would_have_been_better;

        if would_have_been_better {
            let opportunity = alt.estimated_outcome - self.actual_outcome;
            if opportunity > self.opportunity_cost {
                self.opportunity_cost = opportunity;
                self.best_alternative = Some(alt.pattern_id);
            }
        }

        self.alternatives.push(alt);
    }

    /// Calculate regret (normalized opportunity cost)
    pub fn calculate_regret(&mut self) {
        if self.alternatives.is_empty() {
            self.regret = 0.0;
            return;
        }

        // Regret = (best possible - actual) / (best possible - worst possible)
        let best_outcome = self.alternatives.iter()
            .map(|a| a.estimated_outcome)
            .fold(self.actual_outcome, f32::max);

        let worst_outcome = self.alternatives.iter()
            .map(|a| a.estimated_outcome)
            .fold(self.actual_outcome, f32::min);

        let range = best_outcome - worst_outcome;
        if range > 0.0 {
            self.regret = (best_outcome - self.actual_outcome) / range;
        } else {
            self.regret = 0.0;
        }
    }

    /// Get the lesson learned from this analysis
    pub fn lesson(&self) -> String {
        if self.opportunity_cost <= 0.0 {
            format!(
                "Good choice! Pattern {} was optimal for context '{}'.",
                self.actual_pattern_id, self.context
            )
        } else if let Some(best_id) = self.best_alternative {
            format!(
                "Pattern {} would have been {:.1}% better in context '{}'. Consider it next time.",
                best_id,
                self.opportunity_cost * 100.0,
                self.context
            )
        } else {
            "Analysis inconclusive.".to_string()
        }
    }
}

// -----------------------------------------------------------------------------
// Enhancement 4: ACTIVE EXPERIMENTATION
// -----------------------------------------------------------------------------

/// An experiment plan for testing uncertain patterns
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExperimentPlan {
    /// Unique experiment ID
    pub id: u64,

    /// Pattern to test
    pub pattern_id: PatternId,

    /// Hypothesis being tested
    pub hypothesis: String,

    /// Recommended context for the experiment
    pub recommended_context: String,

    /// Expected information gain (entropy reduction)
    pub expected_information_gain: f32,

    /// Risk level of running this experiment
    pub risk_level: RiskLevel,

    /// Priority score (higher = more urgent to test)
    pub priority: f32,

    /// Whether this experiment has been run
    pub completed: bool,

    /// Result if completed
    pub result: Option<ExperimentResult>,

    /// When the experiment was planned
    pub planned_at: u64,
}

/// Risk level of an experiment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RiskLevel {
    /// Safe to experiment - low stakes
    Low,
    /// Some caution needed
    Medium,
    /// Careful consideration required
    High,
    /// Requires explicit approval
    Critical,
}

/// Result of a completed experiment
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExperimentResult {
    /// Did the hypothesis hold?
    pub hypothesis_supported: bool,

    /// Actual outcome observed
    pub observed_outcome: f32,

    /// How confident are we in this result?
    pub confidence: f32,

    /// Information actually gained
    pub information_gained: f32,

    /// Lessons learned
    pub insights: String,

    /// When the experiment was completed
    pub completed_at: u64,
}

/// Experiment planner for active learning
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExperimentPlanner {
    /// Planned experiments
    pub pending_experiments: Vec<ExperimentPlan>,

    /// Completed experiments
    pub completed_experiments: Vec<ExperimentPlan>,

    /// Maximum concurrent experiments
    pub max_concurrent: usize,

    /// Minimum information gain to warrant experiment
    pub min_information_gain: f32,

    /// Maximum risk level to auto-approve
    pub auto_approve_max_risk: RiskLevel,

    /// Next experiment ID
    next_id: u64,
}

impl Default for ExperimentPlanner {
    fn default() -> Self {
        Self {
            pending_experiments: Vec::new(),
            completed_experiments: Vec::new(),
            max_concurrent: 3,
            min_information_gain: 0.1,
            auto_approve_max_risk: RiskLevel::Medium,
            next_id: 1,
        }
    }
}

impl ExperimentPlanner {
    /// Create a new experiment planner
    pub fn new() -> Self {
        Self::default()
    }

    /// Plan an experiment for a pattern
    pub fn plan_experiment(
        &mut self,
        pattern_id: PatternId,
        hypothesis: &str,
        context: &str,
        uncertainty: f32,
        risk: RiskLevel,
        timestamp: u64,
    ) -> Option<u64> {
        // Calculate expected information gain from uncertainty
        // Higher uncertainty = more to learn
        let info_gain = uncertainty * (1.0 - uncertainty.min(0.5));

        if info_gain < self.min_information_gain {
            return None; // Not worth experimenting
        }

        // Calculate priority (high uncertainty + low risk = high priority)
        let risk_penalty = match risk {
            RiskLevel::Low => 0.0,
            RiskLevel::Medium => 0.2,
            RiskLevel::High => 0.5,
            RiskLevel::Critical => 0.8,
        };
        let priority = info_gain * (1.0 - risk_penalty);

        let experiment = ExperimentPlan {
            id: self.next_id,
            pattern_id,
            hypothesis: hypothesis.to_string(),
            recommended_context: context.to_string(),
            expected_information_gain: info_gain,
            risk_level: risk,
            priority,
            completed: false,
            result: None,
            planned_at: timestamp,
        };

        self.next_id += 1;
        let id = experiment.id;
        self.pending_experiments.push(experiment);

        // Sort by priority (highest first)
        self.pending_experiments.sort_by(|a, b| {
            b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
        });

        Some(id)
    }

    /// Get the next experiment to run (respecting max concurrent)
    pub fn next_experiment(&self) -> Option<&ExperimentPlan> {
        let running = self.pending_experiments.iter()
            .filter(|e| !e.completed)
            .count();

        if running >= self.max_concurrent {
            return None;
        }

        self.pending_experiments.iter()
            .find(|e| !e.completed && self.can_auto_approve(e))
    }

    /// Check if experiment can be auto-approved
    fn can_auto_approve(&self, experiment: &ExperimentPlan) -> bool {
        match (experiment.risk_level, self.auto_approve_max_risk) {
            (RiskLevel::Low, _) => true,
            (RiskLevel::Medium, RiskLevel::Medium | RiskLevel::High | RiskLevel::Critical) => true,
            (RiskLevel::High, RiskLevel::High | RiskLevel::Critical) => true,
            (RiskLevel::Critical, RiskLevel::Critical) => true,
            _ => false,
        }
    }

    /// Complete an experiment with results
    pub fn complete_experiment(
        &mut self,
        experiment_id: u64,
        hypothesis_supported: bool,
        observed_outcome: f32,
        confidence: f32,
        insights: &str,
        timestamp: u64,
    ) -> bool {
        let idx = self.pending_experiments.iter()
            .position(|e| e.id == experiment_id);

        if let Some(idx) = idx {
            let mut experiment = self.pending_experiments.remove(idx);

            // Calculate actual information gained
            let info_gained = if hypothesis_supported {
                experiment.expected_information_gain * confidence
            } else {
                // Disconfirmation is also information!
                experiment.expected_information_gain * confidence * 0.8
            };

            experiment.completed = true;
            experiment.result = Some(ExperimentResult {
                hypothesis_supported,
                observed_outcome,
                confidence,
                information_gained: info_gained,
                insights: insights.to_string(),
                completed_at: timestamp,
            });

            self.completed_experiments.push(experiment);
            true
        } else {
            false
        }
    }

    /// Get experiments for a specific pattern
    pub fn experiments_for_pattern(&self, pattern_id: PatternId) -> Vec<&ExperimentPlan> {
        self.pending_experiments.iter()
            .chain(self.completed_experiments.iter())
            .filter(|e| e.pattern_id == pattern_id)
            .collect()
    }
}

// -----------------------------------------------------------------------------
// Enhancement 5: CAUSAL DISCOVERY
// -----------------------------------------------------------------------------

/// A discovered dependency between patterns (legacy causal discovery type)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalPatternDependency {
    /// Prerequisite pattern (must be used first)
    pub prerequisite_id: PatternId,

    /// Dependent pattern (works better after prerequisite)
    pub dependent_id: PatternId,

    /// Strength of dependency (0.0-1.0)
    pub strength: f32,

    /// Confidence in this dependency
    pub confidence: f32,

    /// How many observations support this
    pub observation_count: u32,

    /// Type of dependency
    pub dependency_type: DependencyType,

    /// When this was discovered
    pub discovered_at: u64,
}

/// Type of dependency between patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DependencyType {
    /// Prerequisite must come before
    Sequential,
    /// Both patterns enhance each other
    Synergistic,
    /// Patterns should not be used together
    Conflicting,
    /// One pattern includes the other
    Subsumes,
}

/// A co-occurrence record for dependency discovery
#[derive(Debug, Clone)]
struct PatternCooccurrence {
    /// First pattern ID (smaller of the two, for debugging)
    _pattern_a: PatternId,
    /// Second pattern ID (larger of the two, for debugging)
    _pattern_b: PatternId,
    a_then_b_success: u32,
    a_then_b_fail: u32,
    b_then_a_success: u32,
    b_then_a_fail: u32,
    a_alone_success: u32,
    a_alone_fail: u32,
    b_alone_success: u32,
    b_alone_fail: u32,
}

impl PatternCooccurrence {
    fn new(a: PatternId, b: PatternId) -> Self {
        Self {
            _pattern_a: a.min(b),
            _pattern_b: a.max(b),
            a_then_b_success: 0,
            a_then_b_fail: 0,
            b_then_a_success: 0,
            b_then_a_fail: 0,
            a_alone_success: 0,
            a_alone_fail: 0,
            b_alone_success: 0,
            b_alone_fail: 0,
        }
    }

    fn a_then_b_rate(&self) -> f32 {
        let total = self.a_then_b_success + self.a_then_b_fail;
        if total == 0 { 0.5 } else { self.a_then_b_success as f32 / total as f32 }
    }

    fn b_then_a_rate(&self) -> f32 {
        let total = self.b_then_a_success + self.b_then_a_fail;
        if total == 0 { 0.5 } else { self.b_then_a_success as f32 / total as f32 }
    }

    fn a_alone_rate(&self) -> f32 {
        let total = self.a_alone_success + self.a_alone_fail;
        if total == 0 { 0.5 } else { self.a_alone_success as f32 / total as f32 }
    }

    fn b_alone_rate(&self) -> f32 {
        let total = self.b_alone_success + self.b_alone_fail;
        if total == 0 { 0.5 } else { self.b_alone_success as f32 / total as f32 }
    }

    fn total_observations(&self) -> u32 {
        self.a_then_b_success + self.a_then_b_fail +
        self.b_then_a_success + self.b_then_a_fail +
        self.a_alone_success + self.a_alone_fail +
        self.b_alone_success + self.b_alone_fail
    }
}

/// Causal discovery engine for finding pattern relationships
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalDiscovery {
    /// Discovered dependencies
    pub dependencies: Vec<CausalPatternDependency>,

    /// Minimum observations before considering a dependency
    pub min_observations: u32,

    /// Minimum strength to report a dependency
    pub min_strength: f32,

    /// Minimum confidence to report
    pub min_confidence: f32,

    /// Co-occurrence data (internal)
    #[cfg_attr(feature = "serde", serde(skip))]
    cooccurrences: HashMap<(PatternId, PatternId), PatternCooccurrence>,

    /// Usage history per session for detecting sequences
    #[cfg_attr(feature = "serde", serde(skip))]
    session_history: Vec<(PatternId, bool, u64)>, // (pattern, success, timestamp)
}

impl Default for CausalDiscovery {
    fn default() -> Self {
        Self {
            dependencies: Vec::new(),
            min_observations: 5,
            min_strength: 0.15,
            min_confidence: 0.6,
            cooccurrences: HashMap::new(),
            session_history: Vec::new(),
        }
    }
}

impl CausalDiscovery {
    /// Create a new causal discovery engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Get read-only access to session history for testing and inspection
    pub fn session_history(&self) -> &[(PatternId, bool, u64)] {
        &self.session_history
    }

    /// Record a pattern usage for causal discovery
    pub fn record_usage(&mut self, pattern_id: PatternId, success: bool, timestamp: u64) {
        // Look for recent patterns (within 1 hour)
        let recent_threshold = timestamp.saturating_sub(3600);
        let recent_patterns: Vec<(PatternId, bool)> = self.session_history.iter()
            .filter(|(_, _, t)| *t > recent_threshold && *t < timestamp)
            .map(|(p, s, _)| (*p, *s))
            .collect();

        // Update co-occurrences
        for (recent_id, _recent_success) in &recent_patterns {
            if *recent_id == pattern_id {
                continue; // Same pattern
            }

            let key = ((*recent_id).min(pattern_id), (*recent_id).max(pattern_id));
            let entry = self.cooccurrences
                .entry(key)
                .or_insert_with(|| PatternCooccurrence::new(*recent_id, pattern_id));

            // Record sequence
            if *recent_id < pattern_id {
                // A then B
                if success { entry.a_then_b_success += 1; }
                else { entry.a_then_b_fail += 1; }
            } else {
                // B then A
                if success { entry.b_then_a_success += 1; }
                else { entry.b_then_a_fail += 1; }
            }
        }

        // Record as isolated if no recent patterns
        if recent_patterns.is_empty() {
            // Find all patterns this could be "alone" for
            for (key, cooc) in &mut self.cooccurrences {
                if key.0 == pattern_id {
                    if success { cooc.a_alone_success += 1; }
                    else { cooc.a_alone_fail += 1; }
                } else if key.1 == pattern_id {
                    if success { cooc.b_alone_success += 1; }
                    else { cooc.b_alone_fail += 1; }
                }
            }
        }

        // Add to history
        self.session_history.push((pattern_id, success, timestamp));

        // Prune old history (keep last 1000)
        if self.session_history.len() > 1000 {
            self.session_history.remove(0);
        }
    }

    /// Analyze co-occurrences and discover dependencies
    pub fn discover_dependencies(&mut self, timestamp: u64) -> Vec<CausalPatternDependency> {
        let mut new_deps = Vec::new();

        for ((a, b), cooc) in &self.cooccurrences {
            if cooc.total_observations() < self.min_observations {
                continue;
            }

            // Calculate dependency metrics
            let a_then_b = cooc.a_then_b_rate();
            let b_then_a = cooc.b_then_a_rate();
            let a_alone = cooc.a_alone_rate();
            let b_alone = cooc.b_alone_rate();

            // Check for sequential dependency: A→B better than B alone
            if a_then_b > b_alone + self.min_strength {
                let strength = a_then_b - b_alone;
                let confidence = (cooc.a_then_b_success + cooc.a_then_b_fail) as f32
                    / cooc.total_observations() as f32;

                if confidence >= self.min_confidence {
                    let dep = CausalPatternDependency {
                        prerequisite_id: *a,
                        dependent_id: *b,
                        strength,
                        confidence,
                        observation_count: cooc.total_observations(),
                        dependency_type: DependencyType::Sequential,
                        discovered_at: timestamp,
                    };

                    // Check if we already have this dependency
                    let exists = self.dependencies.iter()
                        .any(|d| d.prerequisite_id == *a && d.dependent_id == *b);

                    if !exists {
                        new_deps.push(dep.clone());
                        self.dependencies.push(dep);
                    }
                }
            }

            // Check reverse: B→A better than A alone
            if b_then_a > a_alone + self.min_strength {
                let strength = b_then_a - a_alone;
                let confidence = (cooc.b_then_a_success + cooc.b_then_a_fail) as f32
                    / cooc.total_observations() as f32;

                if confidence >= self.min_confidence {
                    let dep = CausalPatternDependency {
                        prerequisite_id: *b,
                        dependent_id: *a,
                        strength,
                        confidence,
                        observation_count: cooc.total_observations(),
                        dependency_type: DependencyType::Sequential,
                        discovered_at: timestamp,
                    };

                    let exists = self.dependencies.iter()
                        .any(|d| d.prerequisite_id == *b && d.dependent_id == *a);

                    if !exists {
                        new_deps.push(dep.clone());
                        self.dependencies.push(dep);
                    }
                }
            }

            // Check for synergy: both orders better than either alone
            let combined_rate = (a_then_b + b_then_a) / 2.0;
            let alone_rate = (a_alone + b_alone) / 2.0;
            if combined_rate > alone_rate + self.min_strength {
                // Check if both orderings are good (not just one)
                if a_then_b > b_alone && b_then_a > a_alone {
                    let strength = combined_rate - alone_rate;
                    let confidence = 0.7; // Lower confidence for synergy

                    // Record as synergistic
                    let exists = self.dependencies.iter()
                        .any(|d| {
                            d.dependency_type == DependencyType::Synergistic &&
                            ((d.prerequisite_id == *a && d.dependent_id == *b) ||
                             (d.prerequisite_id == *b && d.dependent_id == *a))
                        });

                    if !exists && strength >= self.min_strength && confidence >= self.min_confidence {
                        let dep = CausalPatternDependency {
                            prerequisite_id: *a,
                            dependent_id: *b,
                            strength,
                            confidence,
                            observation_count: cooc.total_observations(),
                            dependency_type: DependencyType::Synergistic,
                            discovered_at: timestamp,
                        };
                        new_deps.push(dep.clone());
                        self.dependencies.push(dep);
                    }
                }
            }

            // Check for conflict: using both is worse than either alone
            if combined_rate < alone_rate - self.min_strength {
                let strength = alone_rate - combined_rate;
                let confidence = 0.6;

                let exists = self.dependencies.iter()
                    .any(|d| {
                        d.dependency_type == DependencyType::Conflicting &&
                        ((d.prerequisite_id == *a && d.dependent_id == *b) ||
                         (d.prerequisite_id == *b && d.dependent_id == *a))
                    });

                if !exists && strength >= self.min_strength && confidence >= self.min_confidence {
                    let dep = CausalPatternDependency {
                        prerequisite_id: *a,
                        dependent_id: *b,
                        strength,
                        confidence,
                        observation_count: cooc.total_observations(),
                        dependency_type: DependencyType::Conflicting,
                        discovered_at: timestamp,
                    };
                    new_deps.push(dep.clone());
                    self.dependencies.push(dep);
                }
            }
        }

        new_deps
    }

    /// Get dependencies for a specific pattern
    pub fn dependencies_for(&self, pattern_id: PatternId) -> Vec<&CausalPatternDependency> {
        self.dependencies.iter()
            .filter(|d| d.prerequisite_id == pattern_id || d.dependent_id == pattern_id)
            .collect()
    }

    /// Get prerequisites for a pattern
    pub fn prerequisites_for(&self, pattern_id: PatternId) -> Vec<&CausalPatternDependency> {
        self.dependencies.iter()
            .filter(|d| d.dependent_id == pattern_id && d.dependency_type == DependencyType::Sequential)
            .collect()
    }

    /// Get patterns that conflict with a given pattern
    pub fn conflicts_with(&self, pattern_id: PatternId) -> Vec<&CausalPatternDependency> {
        self.dependencies.iter()
            .filter(|d| {
                d.dependency_type == DependencyType::Conflicting &&
                (d.prerequisite_id == pattern_id || d.dependent_id == pattern_id)
            })
            .collect()
    }

    /// Clear session history (call when context changes)
    pub fn clear_session(&mut self) {
        self.session_history.clear();
    }

    /// Manually add a dependency (used by bridge's run_causal_discovery)
    pub fn add_dependency(
        &mut self,
        pattern_a: PatternId,
        pattern_b: PatternId,
        strength: f32,
        dep_type: DependencyType,
    ) {
        // Check if we already have this dependency
        let exists = self.dependencies.iter().any(|d| {
            (d.prerequisite_id == pattern_a && d.dependent_id == pattern_b) ||
            (d.prerequisite_id == pattern_b && d.dependent_id == pattern_a && dep_type != DependencyType::Sequential)
        });

        if !exists {
            let dep = CausalPatternDependency {
                prerequisite_id: pattern_a,
                dependent_id: pattern_b,
                strength,
                confidence: 0.7, // Default confidence for manually added
                observation_count: self.min_observations,
                dependency_type: dep_type,
                discovered_at: 0, // Will be set properly by caller if needed
            };
            self.dependencies.push(dep);
        }
    }
}

// -----------------------------------------------------------------------------
// Enhanced SymthaeaCausalBridge with all 5 improvements
// -----------------------------------------------------------------------------

impl SymthaeaCausalBridge {
    /// Get pattern with temporal decay applied
    pub fn get_pattern_with_decay(
        &self,
        pattern_id: PatternId,
        current_time: u64,
        decay_config: &TemporalDecayConfig,
    ) -> Option<PatternDecayStatus> {
        let pattern = self.patterns.get(&pattern_id)?;

        let decay_factor = decay_config.calculate_decay(pattern.last_used, current_time);
        let effective_rate = pattern.success_rate * decay_factor;
        let days_since = (current_time.saturating_sub(pattern.last_used)) as f32 / (24.0 * 60.0 * 60.0);
        let is_stale = decay_config.is_stale(pattern.last_used, current_time);

        Some(PatternDecayStatus {
            pattern_id,
            original_success_rate: pattern.success_rate,
            effective_success_rate: effective_rate,
            decay_factor,
            days_since_validation: days_since,
            is_stale,
            needs_revalidation: is_stale || decay_factor < 0.5,
        })
    }

    /// Get all patterns needing revalidation
    pub fn patterns_needing_revalidation(
        &self,
        current_time: u64,
        decay_config: &TemporalDecayConfig,
    ) -> Vec<PatternDecayStatus> {
        self.patterns.keys()
            .filter_map(|id| self.get_pattern_with_decay(*id, current_time, decay_config))
            .filter(|status| status.needs_revalidation)
            .collect()
    }

    /// Perform counterfactual analysis for a pattern usage
    pub fn analyze_counterfactual(
        &self,
        used_pattern_id: PatternId,
        actual_outcome: f32,
        context: &str,
        timestamp: u64,
    ) -> CounterfactualAnalysis {
        let mut analysis = CounterfactualAnalysis::new(
            used_pattern_id,
            actual_outcome,
            context,
            timestamp,
        );

        // Find alternative patterns in the same problem domain
        if let Some(used_pattern) = self.patterns.get(&used_pattern_id) {
            let alternatives: Vec<_> = self.patterns.values()
                .filter(|p| {
                    p.pattern_id != used_pattern_id &&
                    p.problem_domain == used_pattern.problem_domain &&
                    p.usage_count > 0
                })
                .collect();

            for alt_pattern in alternatives {
                let alt = CounterfactualAlternative {
                    pattern_id: alt_pattern.pattern_id,
                    estimated_outcome: alt_pattern.confidence_adjusted_success_rate(),
                    confidence: (alt_pattern.usage_count as f32).sqrt().min(1.0) / 10.0 + 0.5,
                    estimation_method: EstimationMethod::HistoricalAverage,
                    would_have_been_better: false, // Will be set by add_alternative
                };
                analysis.add_alternative(alt);
            }
        }

        analysis.calculate_regret();
        analysis
    }

    /// Suggest experiments based on pattern uncertainty
    pub fn suggest_experiments(
        &self,
        planner: &mut ExperimentPlanner,
        current_time: u64,
    ) -> Vec<u64> {
        let mut experiment_ids = Vec::new();

        for pattern in self.patterns.values() {
            // Calculate uncertainty from usage count
            let uncertainty = if pattern.usage_count < 5 {
                0.8 // High uncertainty for new patterns
            } else if pattern.usage_count < 20 {
                0.5
            } else {
                0.3
            };

            // Skip well-tested patterns
            if uncertainty < planner.min_information_gain {
                continue;
            }

            // Assess risk based on pattern domain
            let risk = if pattern.problem_domain.contains("critical") ||
                        pattern.problem_domain.contains("security") {
                RiskLevel::High
            } else if pattern.problem_domain.contains("production") {
                RiskLevel::Medium
            } else {
                RiskLevel::Low
            };

            let hypothesis = format!(
                "Pattern '{}' has success rate {:.1}% (based on {} uses)",
                pattern.solution,
                pattern.success_rate * 100.0,
                pattern.usage_count
            );

            if let Some(id) = planner.plan_experiment(
                pattern.pattern_id,
                &hypothesis,
                &pattern.problem_domain,
                uncertainty,
                risk,
                current_time,
            ) {
                experiment_ids.push(id);
            }
        }

        experiment_ids
    }

    /// Enhanced health report including new metrics
    pub fn enhanced_health_report(
        &self,
        current_time: u64,
        decay_config: &TemporalDecayConfig,
        calibration: &CalibrationCurve,
        discovery: &CausalDiscovery,
        planner: &ExperimentPlanner,
    ) -> EnhancedHealthReport {
        let basic = self.health_report();

        let stale_patterns = self.patterns_needing_revalidation(current_time, decay_config);

        EnhancedHealthReport {
            basic,
            stale_pattern_count: stale_patterns.len(),
            calibration_quality: calibration.calibration_quality(),
            overconfidence_bias: calibration.overconfidence_bias,
            brier_score: calibration.brier_score,
            discovered_dependencies: discovery.dependencies.len(),
            pending_experiments: planner.pending_experiments.len(),
            completed_experiments: planner.completed_experiments.len(),
        }
    }

    /// Quick health report using the bridge's integrated Component 9 fields
    ///
    /// This is a convenience method that uses the internally stored
    /// TemporalDecayConfig, CalibrationCurve, CausalDiscovery, and ExperimentPlanner.
    pub fn quick_health_report(&self, current_time: u64) -> EnhancedHealthReport {
        self.enhanced_health_report(
            current_time,
            &self.temporal_decay,
            &self.calibration,
            &self.causal_discovery,
            &self.experiment_planner,
        )
    }

    /// Get the best pattern for a domain, considering temporal decay
    ///
    /// Returns the pattern with the highest decay-adjusted success rate
    /// that hasn't become stale.
    pub fn best_pattern_for_domain(
        &self,
        domain: &str,
        current_time: u64,
    ) -> Option<PatternDecayStatus> {
        self.patterns.values()
            .filter(|p| p.problem_domain == domain)
            .filter_map(|p| self.get_pattern_with_decay(p.pattern_id, current_time, &self.temporal_decay))
            .filter(|status| !status.is_stale)
            .max_by(|a, b| {
                a.effective_success_rate
                    .partial_cmp(&b.effective_success_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Manually trigger causal discovery if not using auto-discovery
    pub fn discover_patterns(&mut self) {
        self.run_causal_discovery();
    }

    // ==========================================================================
    // Component 17: Pattern Explainability Methods
    // ==========================================================================

    /// Generate a human-readable explanation for why a pattern is recommended
    ///
    /// This method combines all available information about a pattern (success rate,
    /// trust, lifecycle, dependencies, etc.) to produce a justified explanation
    /// suitable for human review. This aligns with the project's emphasis on
    /// "Justified, Wise Belief" - explanations are core to epistemic justification.
    ///
    /// # Returns
    /// `Some(PatternExplanation)` if the pattern exists, `None` otherwise.
    pub fn explain_pattern(&mut self, pattern_id: PatternId, timestamp: u64) -> Option<PatternExplanation> {
        let pattern = self.patterns.get(&pattern_id)?.clone();

        Some(self.explainability.explain_pattern(
            &pattern,
            &self.trust_registry,
            &self.lifecycle_registry,
            &self.dependency_registry,
            &self.domain_registry,
            &self.collective_registry,
            &self.calibration,
            timestamp,
        ))
    }

    /// Get a brief, user-friendly explanation for why a pattern is recommended
    ///
    /// This is a convenience method that returns just the formatted recommendation
    /// text, suitable for display in a UI or command-line interface.
    ///
    /// # Returns
    /// `Some(String)` with the formatted explanation, or `None` if pattern not found.
    pub fn explain_recommendation(&mut self, pattern_id: PatternId, timestamp: u64) -> Option<String> {
        let explanation = self.explain_pattern(pattern_id, timestamp)?;
        Some(self.explainability.explain_recommendation(&explanation))
    }

    /// Get the most user-friendly explanation for why a pattern is recommended
    ///
    /// This returns a simple question-answering format: "Why should I use this pattern?"
    /// The answer includes the recommendation, key reasons to use it, and any risks.
    ///
    /// # Returns
    /// `Some(String)` with the user-friendly explanation, or `None` if pattern not found.
    pub fn why_pattern_recommended(&mut self, pattern_id: PatternId, timestamp: u64) -> Option<String> {
        let explanation = self.explain_pattern(pattern_id, timestamp)?;
        Some(explanation.format_full())
    }

    /// Compare two patterns and explain which is better and why
    ///
    /// This is useful when users are deciding between alternative approaches
    /// and want to understand the tradeoffs.
    pub fn compare_patterns(
        &mut self,
        pattern_a_id: PatternId,
        pattern_b_id: PatternId,
        timestamp: u64,
    ) -> Option<PatternComparison> {
        let pattern_a = self.patterns.get(&pattern_a_id)?.clone();
        let pattern_b = self.patterns.get(&pattern_b_id)?.clone();

        Some(self.explainability.compare_patterns(
            &pattern_a,
            &pattern_b,
            &self.trust_registry,
            &self.lifecycle_registry,
            &self.dependency_registry,
            &self.domain_registry,
            &self.collective_registry,
            &self.calibration,
            timestamp,
        ))
    }

    /// Get explainability statistics
    pub fn explainability_stats(&self) -> ExplainabilityStats {
        self.explainability.stats()
    }

    /// Invalidate cached explanation for a pattern
    ///
    /// Call this after a pattern's data changes significantly to ensure
    /// the next explanation request generates fresh data.
    pub fn invalidate_explanation_cache(&mut self, pattern_id: PatternId) {
        self.explainability.invalidate_cache(pattern_id);
    }

    /// Clear all explanation caches
    pub fn clear_explanation_cache(&mut self) {
        self.explainability.clear_cache();
    }

    // ==========================================================================
    // IMPROVEMENT PLAN METHODS
    // ==========================================================================

    /// Generate an improvement plan for a pattern
    ///
    /// This analyzes the pattern and generates:
    /// - Counterfactual factors (what would need to change)
    /// - Action suggestions (concrete steps to improve)
    /// - Feasibility assessment
    ///
    /// # Example
    /// ```ignore
    /// let plan = bridge.generate_improvement_plan(pattern_id, timestamp);
    /// if let Some(plan) = plan {
    ///     println!("{}", plan.format());
    ///     for action in plan.quick_wins() {
    ///         println!("Quick win: {}", action.action);
    ///     }
    /// }
    /// ```
    pub fn generate_improvement_plan(
        &mut self,
        pattern_id: PatternId,
        timestamp: u64,
    ) -> Option<ImprovementPlan> {
        let pattern = self.patterns.get(&pattern_id)?.clone();
        let explanation = self.explainability.explain_pattern(
            &pattern,
            &self.trust_registry,
            &self.lifecycle_registry,
            &self.dependency_registry,
            &self.domain_registry,
            &self.collective_registry,
            &self.calibration,
            timestamp,
        );

        Some(self.explainability.generate_improvement_plan(&pattern, &explanation, timestamp))
    }

    /// Get quick wins for improving a pattern
    ///
    /// Returns a list of easy, high-impact actions that can be taken
    /// immediately to improve the pattern's recommendation.
    pub fn get_quick_wins(&mut self, pattern_id: PatternId, timestamp: u64) -> Vec<ActionSuggestion> {
        self.generate_improvement_plan(pattern_id, timestamp)
            .map(|plan| plan.quick_wins().into_iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get top N actions by ROI for a pattern
    ///
    /// Returns the actions that provide the best return on investment,
    /// balancing impact against effort required.
    pub fn get_top_actions(
        &mut self,
        pattern_id: PatternId,
        n: usize,
        timestamp: u64,
    ) -> Vec<ActionSuggestion> {
        self.generate_improvement_plan(pattern_id, timestamp)
            .map(|plan| plan.top_actions(n).into_iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get a formatted improvement plan summary
    ///
    /// Returns a human-readable string describing what needs to change
    /// and what actions to take to improve the pattern's recommendation.
    pub fn format_improvement_plan(&mut self, pattern_id: PatternId, timestamp: u64) -> Option<String> {
        self.generate_improvement_plan(pattern_id, timestamp)
            .map(|plan| plan.format())
    }

    /// Check if improvement is feasible for a pattern
    ///
    /// Returns the feasibility assessment for improving the pattern,
    /// considering the gap to target and effort required.
    pub fn check_improvement_feasibility(
        &mut self,
        pattern_id: PatternId,
        timestamp: u64,
    ) -> Option<PlanFeasibility> {
        self.generate_improvement_plan(pattern_id, timestamp)
            .map(|plan| plan.feasibility)
    }

    // ==========================================================================
    // Component 18: Pattern Recommendations Bridge Methods
    // ==========================================================================

    /// Generate recommendations for patterns matching a context
    ///
    /// Synthesizes signals from all components (trust, lifecycle, collective,
    /// calibration, dependencies, domain relevance) to recommend optimal patterns.
    pub fn get_recommendations(
        &mut self,
        context: RecommendationContext,
    ) -> RecommendationSet {
        // Build pattern signals from all sources
        let patterns_with_signals: Vec<(PatternId, PatternSignals)> = self.patterns
            .iter()
            .map(|(&pattern_id, pattern)| {
                let signals = self.build_pattern_signals(pattern_id, pattern, &context);
                (pattern_id, signals)
            })
            .collect();

        self.recommendations.generate_recommendations(&patterns_with_signals, &context)
    }

    /// Build signals for a pattern from all component sources
    fn build_pattern_signals(
        &self,
        pattern_id: PatternId,
        pattern: &SymthaeaPattern,
        context: &RecommendationContext,
    ) -> PatternSignals {
        // Get trust score and level
        let trust_score = self.trust_registry.trust_score(pattern.learned_by);
        let trust_level = TrustLevel::from_score(trust_score);

        // Get lifecycle state
        let lifecycle_state = self.lifecycle_registry.state(pattern_id);

        // Get collective modifier
        let collective_modifier = self.collective_registry
            .get(pattern_id)
            .map(|ctx| ctx.collective_modifier)
            .unwrap_or(0.0);

        let collective_ctx = self.collective_registry.get(pattern_id);
        let is_emergent = collective_ctx.map(|c| c.independent_discoveries >= 3).unwrap_or(false);
        let resolves_tension = collective_ctx.map(|c| c.tension_resolutions > c.tension_increases).unwrap_or(false);

        // Get calibration accuracy
        let calibration_accuracy = self.calibration.calibration_quality();

        // Get dependency satisfaction
        let resolution = self.dependency_registry.resolve(pattern_id, &context.active_patterns);
        let total_deps = resolution.prerequisites.len() + resolution.required.len();
        let satisfied_deps = resolution.prerequisites.len() + resolution.required.len() - resolution.issues.len();
        let dependencies_satisfied = if total_deps > 0 {
            satisfied_deps as f32 / total_deps as f32
        } else {
            1.0
        };
        let has_unmet_requirements = resolution.issues.iter()
            .any(|i| matches!(i.issue_type, DependencyIssueType::MissingRequired));

        // Get domain relevance
        let domain_relevance = if context.target_domains.is_empty() {
            0.5 // Neutral if no target domains
        } else {
            let matches = pattern.domain_ids.iter()
                .filter(|d| context.target_domains.contains(d))
                .count();
            if pattern.domain_ids.is_empty() {
                0.3
            } else {
                matches as f32 / pattern.domain_ids.len().max(1) as f32
            }
        };

        // Calculate time since last use
        let time_since_last_use = context.timestamp.saturating_sub(pattern.last_used);

        PatternSignals {
            success_rate: pattern.success_rate,
            usage_count: pattern.usage_count,
            trust_score,
            trust_level: Some(trust_level),
            lifecycle_state: Some(lifecycle_state),
            collective_modifier,
            is_emergent,
            resolves_tension,
            calibration_accuracy,
            dependencies_satisfied,
            has_unmet_requirements,
            domain_relevance,
            time_since_last_use,
            production_validated: pattern.validated_in_production,
            age: context.timestamp.saturating_sub(pattern.created_at),
        }
    }

    /// Get top recommended patterns for a domain
    pub fn top_patterns_for_domain(
        &mut self,
        domain_id: DomainId,
        limit: usize,
        timestamp: u64,
    ) -> Vec<PatternRecommendation> {
        let context = RecommendationContext::for_domain(domain_id, timestamp);
        let set = self.get_recommendations(context);
        set.top_n(limit).into_iter().cloned().collect()
    }

    /// Get the single best pattern for a context
    pub fn best_pattern(&mut self, context: RecommendationContext) -> Option<PatternRecommendation> {
        self.get_recommendations(context).top().cloned()
    }

    // ==========================================================================
    // Component 19: Anomaly Detection Bridge Methods
    // ==========================================================================

    /// Record pattern metrics for anomaly detection
    ///
    /// Should be called periodically or after pattern usage to enable anomaly detection.
    pub fn record_metrics_for_anomaly_detection(&mut self, timestamp: u64) {
        // Record success rates for all patterns
        for (&pattern_id, pattern) in &self.patterns {
            self.anomaly_detector.record_success_rate(pattern_id, pattern.success_rate, timestamp);
            self.anomaly_detector.record_usage(pattern_id, pattern.usage_count, timestamp);
        }

        // Record trust scores - we get agents from patterns
        let mut recorded_agents = std::collections::HashSet::new();
        for pattern in self.patterns.values() {
            if !recorded_agents.contains(&pattern.learned_by) {
                let trust = self.trust_registry.trust_score(pattern.learned_by);
                self.anomaly_detector.record_trust(pattern.learned_by, trust, timestamp);
                recorded_agents.insert(pattern.learned_by);
            }
        }

        // Record calibration
        self.anomaly_detector.record_calibration(self.calibration.calibration_quality(), timestamp);

        // Record system health
        let health = self.health_report();
        let health_score = health.average_pattern_success.max(0.0).min(1.0);
        self.anomaly_detector.record_system_health(health_score, timestamp);
    }

    /// Run anomaly detection on all patterns
    ///
    /// Returns any new anomalies detected.
    pub fn detect_anomalies(&mut self, timestamp: u64) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        // Check each pattern for success rate anomalies
        for &pattern_id in self.patterns.keys() {
            if let Some(anomaly) = self.anomaly_detector.detect_success_rate_anomaly(pattern_id, timestamp) {
                anomalies.push(anomaly);
            }
        }

        // Check for calibration drift
        if let Some(anomaly) = self.anomaly_detector.detect_calibration_drift(timestamp) {
            anomalies.push(anomaly);
        }

        // Check for system health decline
        if let Some(anomaly) = self.anomaly_detector.detect_system_health_decline(timestamp) {
            anomalies.push(anomaly);
        }

        // Check for orphaned dependencies
        for (&pattern_id, _) in &self.patterns {
            let lifecycle = self.lifecycle_registry.state(pattern_id);
            if lifecycle == PatternLifecycleState::Active {
                // Check if this active pattern depends on deprecated patterns
                for dep in self.dependency_registry.dependencies_from(pattern_id) {
                    let dep_state = self.lifecycle_registry.state(dep.to_pattern);
                    if dep_state == PatternLifecycleState::Deprecated {
                        let anomaly = self.anomaly_detector.create_orphaned_dependency_anomaly(
                            pattern_id,
                            dep.to_pattern,
                            timestamp,
                        );
                        anomalies.push(anomaly);
                    }
                }
            }
        }

        // Add all detected anomalies to the detector
        for anomaly in &anomalies {
            self.anomaly_detector.add_anomaly(anomaly.clone());
        }

        anomalies
    }

    /// Get all active anomalies
    pub fn active_anomalies(&self) -> Vec<&Anomaly> {
        self.anomaly_detector.active_anomalies()
    }

    /// Get anomalies for a specific pattern
    pub fn anomalies_for_pattern(&self, pattern_id: PatternId) -> Vec<&Anomaly> {
        self.anomaly_detector.anomalies_for_pattern(pattern_id)
    }

    /// Acknowledge an anomaly
    pub fn acknowledge_anomaly(&mut self, anomaly_id: u64) -> bool {
        self.anomaly_detector.acknowledge_anomaly(anomaly_id)
    }

    /// Resolve an anomaly
    pub fn resolve_anomaly(&mut self, anomaly_id: u64, notes: &str) -> bool {
        self.anomaly_detector.resolve_anomaly(anomaly_id, notes)
    }

    /// Get anomaly summary report
    pub fn anomaly_summary(&self) -> String {
        self.anomaly_detector.summary_report()
    }

    // ==========================================================================
    // Component 20: Pattern Succession Bridge Methods
    // ==========================================================================

    /// Declare that one pattern supersedes another
    ///
    /// This starts the succession process, optionally with auto-activation
    /// and migration of dependents.
    pub fn declare_succession(
        &mut self,
        predecessor_id: PatternId,
        successor_id: PatternId,
        reason: SuccessionReason,
        explanation: &str,
        timestamp: u64,
    ) -> Option<u64> {
        // Validate both patterns exist
        if !self.patterns.contains_key(&predecessor_id) || !self.patterns.contains_key(&successor_id) {
            return None;
        }

        let declared_by = self.symthaea_id.unwrap_or(0);
        let mut succession = self.succession_manager.declare_succession(
            predecessor_id,
            successor_id,
            reason.clone(),
            declared_by,
            timestamp,
        ).with_explanation(explanation);

        // Add affected domains
        if let Some(pattern) = self.patterns.get(&predecessor_id) {
            succession = succession.with_domains(pattern.domain_ids.clone());
        }

        let succession_id = self.succession_manager.register(succession, timestamp);

        // Create migration plan for dependents
        let dependents = self.dependency_registry.dependents(predecessor_id);
        if !dependents.is_empty() {
            self.succession_manager.create_migration_plan(succession_id, dependents, timestamp);
        }

        // Deprecate predecessor if succession is active
        if let Some(succession) = self.succession_manager.get_succession(succession_id) {
            if succession.status == super::SuccessionStatus::Active {
                let initiator = match &reason {
                    SuccessionReason::BetterPerformance => "Succession: better pattern",
                    SuccessionReason::BugFix => "Succession: bug fix",
                    SuccessionReason::SecurityFix => "Succession: security fix",
                    _ => "Succession",
                };
                self.lifecycle_registry.deprecate(
                    predecessor_id,
                    LifecycleTransitionReason::Superseded { replacement_id: successor_id },
                    timestamp,
                    initiator,
                );
            }
        }

        Some(succession_id)
    }

    /// Get the current successor for a pattern
    pub fn current_successor(&self, pattern_id: PatternId) -> Option<PatternId> {
        self.succession_manager.current_successor(pattern_id)
    }

    /// Get the full lineage of a pattern (all predecessors)
    pub fn pattern_lineage(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.succession_manager.get_lineage(pattern_id)
    }

    /// Get all descendants of a pattern (all successors recursively)
    pub fn pattern_descendants(&self, pattern_id: PatternId) -> Vec<PatternId> {
        self.succession_manager.get_descendants(pattern_id)
    }

    /// Get migration instructions for a dependent pattern
    pub fn migration_instructions(
        &self,
        succession_id: u64,
        pattern_id: PatternId,
    ) -> Option<MigrationInstruction> {
        self.succession_manager.generate_migration_instruction(succession_id, pattern_id)
    }

    /// Complete a succession
    pub fn complete_succession(&mut self, succession_id: u64, timestamp: u64) -> bool {
        if self.succession_manager.complete(succession_id, timestamp) {
            // Retire the predecessor
            if let Some(succession) = self.succession_manager.get_succession(succession_id) {
                self.lifecycle_registry.retire(
                    succession.predecessor_id,
                    LifecycleTransitionReason::Superseded { replacement_id: succession.successor_id },
                    timestamp,
                    "Succession completed",
                );
            }
            true
        } else {
            false
        }
    }

    /// Get succession summary report
    pub fn succession_summary(&self) -> String {
        self.succession_manager.summary_report()
    }

    /// Process all pending grace periods and auto-complete successions
    pub fn process_succession_grace_periods(&mut self, timestamp: u64) -> Vec<u64> {
        self.succession_manager.process_grace_periods(timestamp)
    }

    // ==========================================================================
    // Component 21: Pattern Similarity & Clustering Bridge Methods
    // ==========================================================================

    /// Register pattern features for similarity analysis
    ///
    /// Call this when a pattern is learned to enable similarity matching.
    pub fn register_pattern_for_similarity(&mut self, pattern_id: PatternId, timestamp: u64) {
        if let Some(pattern) = self.patterns.get(&pattern_id) {
            self.similarity_registry.register_pattern(pattern, timestamp);
        }
    }

    /// Find patterns similar to a given pattern
    ///
    /// Returns patterns ordered by similarity score (highest first).
    pub fn find_similar_patterns(
        &mut self,
        pattern_id: PatternId,
        min_similarity: f32,
        timestamp: u64,
    ) -> Vec<SimilarityScore> {
        self.similarity_registry.find_similar(pattern_id, min_similarity, timestamp)
    }

    /// Get similarity between two specific patterns
    pub fn pattern_similarity(
        &mut self,
        pattern_a: PatternId,
        pattern_b: PatternId,
        timestamp: u64,
    ) -> Option<SimilarityScore> {
        self.similarity_registry.get_similarity(pattern_a, pattern_b, timestamp)
    }

    /// Detect potential duplicate patterns
    ///
    /// Returns pairs of patterns that may be duplicates.
    pub fn detect_duplicate_patterns(&mut self, timestamp: u64) -> Vec<DuplicateCandidate> {
        self.similarity_registry.detect_duplicates(timestamp)
    }

    /// Cluster all patterns into related groups
    pub fn cluster_patterns(&mut self, timestamp: u64) -> Vec<PatternCluster> {
        self.similarity_registry.cluster_patterns(timestamp)
    }

    /// Get the cluster containing a pattern
    pub fn pattern_cluster(&self, pattern_id: PatternId) -> Option<&PatternCluster> {
        self.similarity_registry.get_cluster(pattern_id)
    }

    /// Find patterns that might be good candidates for merging
    pub fn suggest_pattern_merges(&mut self, timestamp: u64) -> Vec<super::MergeSuggestion> {
        self.similarity_registry.suggest_merges(timestamp)
    }

    /// Get similarity registry statistics
    pub fn similarity_stats(&self) -> &super::SimilarityStats {
        self.similarity_registry.stats()
    }

    // ==========================================================================
    // Component 22: HDC-Grounded Associative Learner Bridge Methods
    // ==========================================================================

    /// Learn from a pattern usage experience using HDC
    ///
    /// This encodes the context-action-outcome triplet into the associative memory,
    /// enabling zero-shot generalization to similar future contexts.
    pub fn learn_pattern_experience(
        &mut self,
        pattern_id: PatternId,
        domain_id: u64,
        success: bool,
        timestamp: u64,
    ) {
        if !self.use_associative_learning {
            return;
        }

        // Build context from pattern metadata
        let context = WisdomContext::new()
            .with_domain(domain_id)
            .with_attribute("pattern_type", "symthaea");

        let context_pairs = context.build();
        let action = format!("use_pattern_{}", pattern_id);
        let outcome = if success { 1.0 } else { -0.5 };

        self.associative_learner.learn(&context_pairs, &action, outcome, timestamp);
    }

    /// Predict best patterns for a given context using HDC
    ///
    /// Returns pattern IDs ranked by predicted success, even for novel contexts.
    pub fn predict_patterns_for_context(
        &mut self,
        domain_id: u64,
        anomaly_type: Option<&str>,
    ) -> Vec<(PatternId, f32)> {
        if !self.use_associative_learning {
            return Vec::new();
        }

        let mut context = WisdomContext::new()
            .with_domain(domain_id);

        if let Some(anomaly) = anomaly_type {
            context = context.with_anomaly(anomaly);
        }

        let context_pairs = context.build();
        let predictions = self.associative_learner.predict(&context_pairs);

        // Convert action names back to pattern IDs
        predictions.into_iter()
            .filter_map(|pred| {
                // Parse pattern ID from action name "use_pattern_X"
                pred.action
                    .strip_prefix("use_pattern_")
                    .and_then(|s| s.parse::<PatternId>().ok())
                    .map(|id| (id, pred.confidence))
            })
            .collect()
    }

    /// Query how similar a context is to past successful experiences
    ///
    /// Returns a similarity score where higher = more similar to success cases.
    pub fn context_experience_similarity(&mut self, domain_id: u64) -> f32 {
        let context = WisdomContext::new()
            .with_domain(domain_id);
        let context_pairs = context.build();
        self.associative_learner.query_experience_similarity(&context_pairs)
    }

    /// Register an action type for HDC learning
    pub fn register_learning_action(&mut self, action_name: &str) {
        self.associative_learner.register_action(action_name);
    }

    /// Get associative learner statistics
    pub fn associative_learner_stats(&self) -> &super::AssociativeLearnerStats {
        self.associative_learner.stats()
    }

    /// Reset the associative learner (clear all learned experiences)
    pub fn reset_associative_learner(&mut self) {
        self.associative_learner.reset();
    }

    /// Get recent learning experiences
    pub fn recent_learning_experiences(&self) -> &[super::ExperienceEncoding] {
        self.associative_learner.recent_experiences()
    }

    /// Export the HDC memory state for persistence
    pub fn export_hdc_memory(&self) -> super::MemorySnapshot {
        self.associative_learner.export_memory()
    }

    /// Import a previously exported HDC memory state
    pub fn import_hdc_memory(&mut self, snapshot: super::MemorySnapshot) {
        self.associative_learner.import_memory(snapshot);
    }

    // ==========================================================================
    // Combined Methods (Using Multiple Components)
    // ==========================================================================

    /// Enhanced pattern learning that uses both similarity and HDC
    ///
    /// When a pattern is learned:
    /// 1. Registers for similarity analysis
    /// 2. Checks for duplicates
    /// 3. Prepares HDC learning actions
    pub fn enhanced_on_pattern_learned(&mut self, pattern: SymthaeaPattern, timestamp: u64) -> PatternId {
        let pattern_id = self.on_pattern_learned(pattern);

        // Register for similarity
        self.register_pattern_for_similarity(pattern_id, timestamp);

        // Check for duplicates if enabled
        if self.auto_detect_duplicates {
            let duplicates = self.similarity_registry.find_similar(
                pattern_id,
                0.85, // High threshold for duplicates
                timestamp,
            );

            // If we found a potential duplicate, log it but still keep the pattern
            // The user/system can decide what to do with duplicates
            if !duplicates.is_empty() {
                // Could emit an event or warning here
            }
        }

        // Register pattern action for HDC learning
        let action_name = format!("use_pattern_{}", pattern_id);
        self.associative_learner.register_action(&action_name);

        pattern_id
    }

    /// Enhanced pattern usage that records to both standard and HDC systems
    pub fn enhanced_on_pattern_used(
        &mut self,
        pattern_id: PatternId,
        success: bool,
        context: &str,
        verification: OracleVerificationLevel,
        timestamp: u64,
    ) {
        // Standard usage recording
        self.on_pattern_used(pattern_id, success, timestamp, context, verification);

        // HDC learning
        if let Some(pattern) = self.patterns.get(&pattern_id) {
            let domain_id = pattern.domain_ids.first().copied().unwrap_or(0);
            self.learn_pattern_experience(pattern_id, domain_id, success, timestamp);
        }
    }

    /// Get comprehensive pattern analysis combining all sources
    pub fn comprehensive_pattern_analysis(
        &mut self,
        pattern_id: PatternId,
        timestamp: u64,
    ) -> Option<ComprehensivePatternAnalysis> {
        let pattern = self.patterns.get(&pattern_id)?.clone();

        // Get explanation
        let explanation = self.explain_pattern(pattern_id, timestamp);

        // Get similar patterns
        let similar = self.find_similar_patterns(pattern_id, 0.5, timestamp);

        // Get cluster info
        let cluster = self.pattern_cluster(pattern_id).cloned();

        // Get HDC predictions for this pattern's domain
        let domain_id = pattern.domain_ids.first().copied().unwrap_or(0);
        let hdc_confidence = self.context_experience_similarity(domain_id);

        Some(ComprehensivePatternAnalysis {
            pattern_id,
            pattern_name: pattern.solution.clone(),
            success_rate: pattern.success_rate,
            usage_count: pattern.usage_count,
            explanation,
            similar_patterns: similar.into_iter().map(|s| (s.pattern_b, s.overall)).collect(),
            cluster_id: cluster.map(|c| c.cluster_id),
            hdc_confidence,
            lifecycle_state: Some(self.lifecycle_registry.state(pattern_id)),
        })
    }
}

/// Comprehensive analysis of a pattern combining all available data sources
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ComprehensivePatternAnalysis {
    /// Pattern ID
    pub pattern_id: PatternId,

    /// Pattern name/solution
    pub pattern_name: String,

    /// Success rate from usage history
    pub success_rate: f32,

    /// Total usage count
    pub usage_count: u64,

    /// Explanation from explainability registry
    pub explanation: Option<PatternExplanation>,

    /// Similar patterns (id, similarity_score)
    pub similar_patterns: Vec<(PatternId, f32)>,

    /// Cluster this pattern belongs to
    pub cluster_id: Option<u64>,

    /// HDC-based confidence in this pattern's domain
    pub hdc_confidence: f32,

    /// Current lifecycle state
    pub lifecycle_state: Option<PatternLifecycleState>,
}

/// Enhanced health report including all scientific method metrics
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EnhancedHealthReport {
    /// Basic health metrics
    pub basic: BridgeHealthReport,

    /// Number of patterns needing revalidation (stale)
    pub stale_pattern_count: usize,

    /// Calibration quality (0.0-1.0)
    pub calibration_quality: f32,

    /// Overconfidence bias (positive = overconfident)
    pub overconfidence_bias: f32,

    /// Brier score for predictions
    pub brier_score: f32,

    /// Number of discovered pattern dependencies
    pub discovered_dependencies: usize,

    /// Number of pending experiments
    pub pending_experiments: usize,

    /// Number of completed experiments
    pub completed_experiments: usize,
}

