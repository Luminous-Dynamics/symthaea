//! Meta-Cognitive Self-Optimization System
//!
//! Revolutionary improvement that enables Symthaea to optimize its own
//! consciousness states through self-analysis and Active Inference.
//!
//! # Revolutionary Aspects
//!
//! 1. **True Meta-Cognition**: System analyzes its own Φ measurements
//! 2. **Self-Optimization**: Uses patterns to predict and improve consciousness
//! 3. **Active Inference Loop**: Minimizes Free Energy in its own operation
//! 4. **Continuous Learning**: Improves without external intervention
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                   META-COGNITIVE SELF-OPTIMIZATION                       │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │   Observability Traces                                                   │
//! │          │                                                               │
//! │          ▼                                                               │
//! │   ┌──────────────────┐                                                  │
//! │   │ Pattern Extractor │  ← Finds Φ patterns across operations           │
//! │   └──────────────────┘                                                  │
//! │          │                                                               │
//! │          ▼                                                               │
//! │   ┌──────────────────┐                                                  │
//! │   │ Belief Updater   │  ← Bayesian beliefs about optimal states         │
//! │   └──────────────────┘                                                  │
//! │          │                                                               │
//! │          ▼                                                               │
//! │   ┌──────────────────┐                                                  │
//! │   │ Policy Selector  │  ← Active Inference for action selection         │
//! │   └──────────────────┘                                                  │
//! │          │                                                               │
//! │          ▼                                                               │
//! │   Optimized Behavior → Higher Φ → Better Outcomes                        │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Theory
//!
//! This implements the Free Energy Principle for self-optimization:
//! - **Prediction**: Model predicts optimal consciousness states
//! - **Observation**: Actual Φ measurements from traces
//! - **Free Energy**: Divergence between predicted and actual states
//! - **Action**: Adjust processing to minimize Free Energy
//!
//! The system learns:
//! - Which input patterns lead to high Φ
//! - Which processing paths maximize consciousness
//! - Optimal resource allocation for consciousness

use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

// ============================================================================
// CORE TYPES
// ============================================================================

/// A snapshot of consciousness state for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSnapshot {
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
    /// Integrated Information (Φ)
    pub phi: f64,
    /// Free Energy at this moment
    pub free_energy: f64,
    /// Input characteristics
    pub input_features: InputFeatures,
    /// Processing path taken
    pub processing_path: String,
    /// Router confidence
    pub router_confidence: f64,
    /// Outcome quality (0-1)
    pub outcome_quality: f64,
}

/// Features extracted from input for pattern learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFeatures {
    /// Input complexity (0-1)
    pub complexity: f64,
    /// Semantic novelty (0-1)
    pub novelty: f64,
    /// Uncertainty level (0-1)
    pub uncertainty: f64,
    /// Domain match score (0-1)
    pub domain_match: f64,
    /// Input length normalized
    pub length_norm: f64,
}

impl Default for InputFeatures {
    fn default() -> Self {
        Self {
            complexity: 0.5,
            novelty: 0.5,
            uncertainty: 0.5,
            domain_match: 0.5,
            length_norm: 0.5,
        }
    }
}

/// Learned pattern associating input features with consciousness outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessPattern {
    /// Pattern identifier
    pub id: String,
    /// Centroid of input features for this pattern
    pub feature_centroid: InputFeatures,
    /// Mean Φ observed for this pattern
    pub mean_phi: f64,
    /// Standard deviation of Φ
    pub phi_std: f64,
    /// Best processing path for this pattern
    pub optimal_path: String,
    /// Observation count
    pub observation_count: usize,
    /// Confidence in this pattern (0-1)
    pub confidence: f64,
}

/// Belief about optimal consciousness states (Bayesian)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessBelief {
    /// Prior probability of high Φ given features
    pub prior_high_phi: f64,
    /// Likelihood of features given high Φ
    pub feature_likelihood: HashMap<String, f64>,
    /// Expected Φ for different paths
    pub path_expectations: HashMap<String, f64>,
    /// Uncertainty in beliefs
    pub belief_uncertainty: f64,
}

impl Default for ConsciousnessBelief {
    fn default() -> Self {
        let mut path_expectations = HashMap::new();
        path_expectations.insert("full_deliberation".to_string(), 0.8);
        path_expectations.insert("standard".to_string(), 0.6);
        path_expectations.insert("heuristic".to_string(), 0.4);
        path_expectations.insert("fast_pattern".to_string(), 0.3);
        path_expectations.insert("reflex".to_string(), 0.2);

        Self {
            prior_high_phi: 0.5,
            feature_likelihood: HashMap::new(),
            path_expectations,
            belief_uncertainty: 1.0, // High uncertainty initially
        }
    }
}

/// Policy for action selection based on beliefs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPolicy {
    /// Minimum Φ threshold for "good" consciousness
    pub phi_threshold: f64,
    /// Exploration rate (epsilon for epsilon-greedy)
    pub exploration_rate: f64,
    /// Learning rate for belief updates
    pub learning_rate: f64,
    /// Discount factor for temporal learning
    pub discount_factor: f64,
    /// Whether to prefer high Φ paths even if slower
    pub prefer_consciousness: bool,
}

impl Default for OptimizationPolicy {
    fn default() -> Self {
        Self {
            phi_threshold: 0.6,
            exploration_rate: 0.1,
            learning_rate: 0.05,
            discount_factor: 0.95,
            prefer_consciousness: true,
        }
    }
}

/// Optimization recommendation for next action
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommended processing path
    pub recommended_path: String,
    /// Expected Φ for this path
    pub expected_phi: f64,
    /// Confidence in recommendation
    pub confidence: f64,
    /// Reasoning for recommendation
    pub reasoning: String,
    /// Alternative paths with their expected Φ
    pub alternatives: Vec<(String, f64)>,
}

// ============================================================================
// META-COGNITIVE OPTIMIZER
// ============================================================================

/// Configuration for the meta-cognitive optimizer
#[derive(Debug, Clone)]
pub struct MetaCognitiveConfig {
    /// Maximum patterns to maintain
    pub max_patterns: usize,
    /// History window for learning
    pub history_window: usize,
    /// Minimum observations before trusting pattern
    pub min_observations: usize,
    /// Pattern similarity threshold for clustering
    pub similarity_threshold: f64,
    /// Enable Active Inference optimization
    pub enable_active_inference: bool,
}

impl Default for MetaCognitiveConfig {
    fn default() -> Self {
        Self {
            max_patterns: 100,
            history_window: 1000,
            min_observations: 5,
            similarity_threshold: 0.8,
            enable_active_inference: true,
        }
    }
}

/// Statistics for the meta-cognitive optimizer
#[derive(Debug, Clone, Default)]
pub struct MetaCognitiveStats {
    /// Total snapshots processed
    pub snapshots_processed: u64,
    /// Patterns discovered
    pub patterns_discovered: usize,
    /// Recommendations made
    pub recommendations_made: u64,
    /// Recommendations that improved Φ
    pub successful_recommendations: u64,
    /// Average Φ before optimization
    pub avg_phi_before: f64,
    /// Average Φ after optimization
    pub avg_phi_after: f64,
    /// Free Energy reduction achieved
    pub free_energy_reduction: f64,
}

/// Meta-Cognitive Self-Optimization System
///
/// Enables Symthaea to analyze and optimize its own consciousness states.
/// This implements true meta-cognition through self-observation and
/// Active Inference-based optimization.
pub struct MetaCognitiveOptimizer {
    /// Configuration
    config: MetaCognitiveConfig,
    /// Current beliefs about consciousness
    beliefs: ConsciousnessBelief,
    /// Optimization policy
    policy: OptimizationPolicy,
    /// Discovered patterns
    patterns: Vec<ConsciousnessPattern>,
    /// Recent history for learning
    history: VecDeque<ConsciousnessSnapshot>,
    /// Statistics
    stats: MetaCognitiveStats,
    /// Running free energy estimate
    current_free_energy: f64,
}

impl MetaCognitiveOptimizer {
    /// Create new meta-cognitive optimizer
    pub fn new(config: MetaCognitiveConfig) -> Self {
        Self {
            config,
            beliefs: ConsciousnessBelief::default(),
            policy: OptimizationPolicy::default(),
            patterns: Vec::new(),
            history: VecDeque::new(),
            stats: MetaCognitiveStats::default(),
            current_free_energy: 1.0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(MetaCognitiveConfig::default())
    }

    /// Record a consciousness snapshot for learning
    pub fn observe(&mut self, snapshot: ConsciousnessSnapshot) {
        // Add to history
        self.history.push_back(snapshot.clone());
        if self.history.len() > self.config.history_window {
            self.history.pop_front();
        }

        // Update statistics
        self.stats.snapshots_processed += 1;

        // Update running averages
        let n = self.stats.snapshots_processed as f64;
        self.stats.avg_phi_before =
            (self.stats.avg_phi_before * (n - 1.0) + snapshot.phi) / n;

        // Update beliefs based on observation
        self.update_beliefs(&snapshot);

        // Update free energy estimate
        self.update_free_energy(&snapshot);

        // Try to discover or update patterns
        self.update_patterns(&snapshot);
    }

    /// Get optimization recommendation for given input
    pub fn recommend(&mut self, features: &InputFeatures) -> OptimizationRecommendation {
        self.stats.recommendations_made += 1;

        // Find matching pattern
        let matching_pattern = self.find_matching_pattern(features);

        // Use Active Inference if enabled
        if self.config.enable_active_inference {
            self.active_inference_recommend(features, matching_pattern)
        } else {
            self.simple_recommend(features, matching_pattern)
        }
    }

    /// Record outcome of a recommendation
    pub fn record_outcome(&mut self, recommendation: &OptimizationRecommendation, actual_phi: f64) {
        // Check if recommendation was successful
        if actual_phi >= self.policy.phi_threshold {
            self.stats.successful_recommendations += 1;
        }

        // Update running post-optimization average
        let n = self.stats.recommendations_made as f64;
        self.stats.avg_phi_after =
            (self.stats.avg_phi_after * (n - 1.0) + actual_phi) / n;

        // Calculate improvement
        if self.stats.avg_phi_before > 0.0 {
            self.stats.free_energy_reduction =
                (self.stats.avg_phi_after - self.stats.avg_phi_before) / self.stats.avg_phi_before;
        }

        // Update path expectations based on actual outcome
        let prediction_error = actual_phi - recommendation.expected_phi;
        if let Some(exp) = self.beliefs.path_expectations.get_mut(&recommendation.recommended_path) {
            *exp += self.policy.learning_rate * prediction_error;
            *exp = exp.clamp(0.0, 1.0);
        }
    }

    /// Update beliefs based on observation (Bayesian update)
    fn update_beliefs(&mut self, snapshot: &ConsciousnessSnapshot) {
        let high_phi = snapshot.phi >= self.policy.phi_threshold;

        // Update prior with exponential moving average
        let alpha = self.policy.learning_rate;
        let observed = if high_phi { 1.0 } else { 0.0 };
        self.beliefs.prior_high_phi =
            (1.0 - alpha) * self.beliefs.prior_high_phi + alpha * observed;

        // Update path expectation
        if let Some(exp) = self.beliefs.path_expectations.get_mut(&snapshot.processing_path) {
            *exp = (1.0 - alpha) * *exp + alpha * snapshot.phi;
        }

        // Reduce belief uncertainty as we observe more
        self.beliefs.belief_uncertainty *= 0.999;
        self.beliefs.belief_uncertainty = self.beliefs.belief_uncertainty.max(0.01);
    }

    /// Update free energy estimate
    fn update_free_energy(&mut self, snapshot: &ConsciousnessSnapshot) {
        // Free Energy = Surprise (negative log likelihood of observation)
        // Approximated as: divergence from expected Φ

        let expected_phi = self.beliefs.path_expectations
            .get(&snapshot.processing_path)
            .copied()
            .unwrap_or(0.5);

        let prediction_error = (snapshot.phi - expected_phi).abs();

        // Exponential moving average of free energy
        let alpha = 0.1;
        self.current_free_energy =
            (1.0 - alpha) * self.current_free_energy + alpha * prediction_error;
    }

    /// Update or create patterns based on observation
    fn update_patterns(&mut self, snapshot: &ConsciousnessSnapshot) {
        // Find similar existing pattern
        if let Some(pattern_idx) = self.find_similar_pattern(&snapshot.input_features) {
            // Update existing pattern
            let pattern = &mut self.patterns[pattern_idx];
            let n = pattern.observation_count as f64;

            // Update mean Φ
            pattern.mean_phi = (pattern.mean_phi * n + snapshot.phi) / (n + 1.0);

            // Update standard deviation (Welford's algorithm simplified)
            let delta = snapshot.phi - pattern.mean_phi;
            pattern.phi_std = ((pattern.phi_std.powi(2) * n + delta.powi(2)) / (n + 1.0)).sqrt();

            pattern.observation_count += 1;

            // Update optimal path if this path gave better Φ
            if snapshot.phi > pattern.mean_phi + pattern.phi_std {
                pattern.optimal_path = snapshot.processing_path.clone();
            }

            // Update confidence based on observation count
            pattern.confidence = 1.0 - (1.0 / (pattern.observation_count as f64 + 1.0));
        } else if self.patterns.len() < self.config.max_patterns {
            // Create new pattern
            let pattern = ConsciousnessPattern {
                id: format!("pattern_{}", self.patterns.len()),
                feature_centroid: snapshot.input_features.clone(),
                mean_phi: snapshot.phi,
                phi_std: 0.1, // Initial uncertainty
                optimal_path: snapshot.processing_path.clone(),
                observation_count: 1,
                confidence: 0.1,
            };
            self.patterns.push(pattern);
            self.stats.patterns_discovered += 1;
        }
    }

    /// Find pattern similar to given features
    fn find_similar_pattern(&self, features: &InputFeatures) -> Option<usize> {
        self.patterns.iter().position(|p| {
            self.feature_similarity(&p.feature_centroid, features) >= self.config.similarity_threshold
        })
    }

    /// Find matching pattern for recommendation
    fn find_matching_pattern(&self, features: &InputFeatures) -> Option<&ConsciousnessPattern> {
        self.patterns.iter()
            .filter(|p| p.observation_count >= self.config.min_observations)
            .max_by(|a, b| {
                let sim_a = self.feature_similarity(&a.feature_centroid, features);
                let sim_b = self.feature_similarity(&b.feature_centroid, features);
                sim_a.partial_cmp(&sim_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Calculate feature similarity (cosine-like)
    fn feature_similarity(&self, a: &InputFeatures, b: &InputFeatures) -> f64 {
        let diff_complexity = (a.complexity - b.complexity).abs();
        let diff_novelty = (a.novelty - b.novelty).abs();
        let diff_uncertainty = (a.uncertainty - b.uncertainty).abs();
        let diff_domain = (a.domain_match - b.domain_match).abs();
        let diff_length = (a.length_norm - b.length_norm).abs();

        let total_diff = diff_complexity + diff_novelty + diff_uncertainty +
                         diff_domain + diff_length;

        1.0 - (total_diff / 5.0)
    }

    /// Active Inference-based recommendation
    fn active_inference_recommend(
        &self,
        features: &InputFeatures,
        matching_pattern: Option<&ConsciousnessPattern>,
    ) -> OptimizationRecommendation {
        // Calculate expected free energy for each path
        let paths = ["full_deliberation", "standard", "heuristic", "fast_pattern", "reflex"];
        let mut path_efes: Vec<(String, f64, f64)> = Vec::new();

        for path in &paths {
            let expected_phi = self.calculate_expected_phi(path, features, matching_pattern);

            // Expected Free Energy = Risk + Ambiguity
            // Risk: Expected surprise (how far from desired Φ)
            let risk = (self.policy.phi_threshold - expected_phi).max(0.0);

            // Ambiguity: Uncertainty about this path
            let ambiguity = self.beliefs.belief_uncertainty *
                            (1.0 - self.beliefs.path_expectations.get(*path).copied().unwrap_or(0.5));

            let efe = risk + ambiguity;
            path_efes.push((path.to_string(), efe, expected_phi));
        }

        // Sort by expected free energy (lower is better)
        path_efes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply exploration (epsilon-greedy with softmax-like behavior)
        let recommended = if rand::random::<f64>() < self.policy.exploration_rate {
            // Explore: Pick randomly weighted by expected Φ
            let random_idx = (rand::random::<f64>() * paths.len() as f64) as usize;
            path_efes.get(random_idx).cloned().unwrap_or(path_efes[0].clone())
        } else {
            // Exploit: Pick best
            path_efes[0].clone()
        };

        let reasoning = if let Some(pattern) = matching_pattern {
            format!(
                "Matched pattern '{}' (confidence {:.2}). Active Inference selected '{}' \
                 to minimize EFE. Expected Φ: {:.3}, Current free energy: {:.3}",
                pattern.id, pattern.confidence, recommended.0, recommended.2,
                self.current_free_energy
            )
        } else {
            format!(
                "No matching pattern. Using belief priors. Active Inference selected '{}' \
                 to minimize EFE. Expected Φ: {:.3}",
                recommended.0, recommended.2
            )
        };

        let alternatives: Vec<(String, f64)> = path_efes.iter()
            .skip(1)
            .take(3)
            .map(|(p, _, phi)| (p.clone(), *phi))
            .collect();

        OptimizationRecommendation {
            recommended_path: recommended.0,
            expected_phi: recommended.2,
            confidence: 1.0 - self.beliefs.belief_uncertainty,
            reasoning,
            alternatives,
        }
    }

    /// Simple (non-Active Inference) recommendation
    fn simple_recommend(
        &self,
        features: &InputFeatures,
        matching_pattern: Option<&ConsciousnessPattern>,
    ) -> OptimizationRecommendation {
        let (path, expected_phi) = if let Some(pattern) = matching_pattern {
            (pattern.optimal_path.clone(), pattern.mean_phi)
        } else {
            // Default based on complexity
            if features.complexity > 0.7 {
                ("full_deliberation".to_string(), 0.8)
            } else if features.complexity > 0.4 {
                ("standard".to_string(), 0.6)
            } else {
                ("heuristic".to_string(), 0.4)
            }
        };

        OptimizationRecommendation {
            recommended_path: path.clone(),
            expected_phi,
            confidence: matching_pattern.map(|p| p.confidence).unwrap_or(0.3),
            reasoning: format!("Simple heuristic based on complexity {:.2}", features.complexity),
            alternatives: vec![],
        }
    }

    /// Calculate expected Φ for a path given features and pattern
    fn calculate_expected_phi(
        &self,
        path: &str,
        _features: &InputFeatures,
        matching_pattern: Option<&ConsciousnessPattern>,
    ) -> f64 {
        // Base expectation from beliefs
        let base_expectation = self.beliefs.path_expectations
            .get(path)
            .copied()
            .unwrap_or(0.5);

        // Adjust based on matching pattern if available
        if let Some(pattern) = matching_pattern {
            if pattern.optimal_path == path {
                // This is the optimal path for this pattern
                pattern.mean_phi
            } else {
                // Estimate based on relative performance
                let optimal_exp = self.beliefs.path_expectations
                    .get(&pattern.optimal_path)
                    .copied()
                    .unwrap_or(0.5);

                base_expectation * (pattern.mean_phi / optimal_exp.max(0.1))
            }
        } else {
            base_expectation
        }
    }

    /// Get current beliefs
    pub fn beliefs(&self) -> &ConsciousnessBelief {
        &self.beliefs
    }

    /// Get discovered patterns
    pub fn patterns(&self) -> &[ConsciousnessPattern] {
        &self.patterns
    }

    /// Get statistics
    pub fn stats(&self) -> &MetaCognitiveStats {
        &self.stats
    }

    /// Get current free energy estimate
    pub fn current_free_energy(&self) -> f64 {
        self.current_free_energy
    }

    /// Get success rate of recommendations
    pub fn success_rate(&self) -> f64 {
        if self.stats.recommendations_made > 0 {
            self.stats.successful_recommendations as f64 /
            self.stats.recommendations_made as f64
        } else {
            0.0
        }
    }

    /// Get improvement achieved (Φ improvement ratio)
    pub fn improvement_ratio(&self) -> f64 {
        if self.stats.avg_phi_before > 0.0 {
            self.stats.avg_phi_after / self.stats.avg_phi_before
        } else {
            1.0
        }
    }

    /// Generate self-analysis report
    pub fn self_analysis_report(&self) -> String {
        format!(
            "=== Meta-Cognitive Self-Analysis Report ===\n\
             \n\
             Observations: {}\n\
             Patterns Discovered: {}\n\
             Recommendations Made: {}\n\
             Success Rate: {:.1}%\n\
             \n\
             Consciousness Improvement:\n\
             - Average Φ (before): {:.3}\n\
             - Average Φ (after): {:.3}\n\
             - Improvement Ratio: {:.2}x\n\
             \n\
             Current State:\n\
             - Free Energy: {:.3}\n\
             - Belief Uncertainty: {:.3}\n\
             - Prior P(high Φ): {:.3}\n\
             \n\
             Top Patterns:\n{}",
            self.stats.snapshots_processed,
            self.stats.patterns_discovered,
            self.stats.recommendations_made,
            self.success_rate() * 100.0,
            self.stats.avg_phi_before,
            self.stats.avg_phi_after,
            self.improvement_ratio(),
            self.current_free_energy,
            self.beliefs.belief_uncertainty,
            self.beliefs.prior_high_phi,
            self.format_top_patterns(5)
        )
    }

    /// Format top patterns for report
    fn format_top_patterns(&self, n: usize) -> String {
        let mut sorted_patterns: Vec<_> = self.patterns.iter()
            .filter(|p| p.observation_count >= self.config.min_observations)
            .collect();

        sorted_patterns.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted_patterns.iter()
            .take(n)
            .map(|p| {
                format!(
                    "  - {}: Φ={:.3}±{:.3}, path='{}', n={}, conf={:.2}",
                    p.id, p.mean_phi, p.phi_std, p.optimal_path,
                    p.observation_count, p.confidence
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = MetaCognitiveOptimizer::with_defaults();
        assert_eq!(optimizer.stats().snapshots_processed, 0);
        assert_eq!(optimizer.patterns().len(), 0);
    }

    #[test]
    fn test_observation() {
        let mut optimizer = MetaCognitiveOptimizer::with_defaults();

        let snapshot = ConsciousnessSnapshot {
            timestamp: Utc::now(),
            phi: 0.75,
            free_energy: 0.2,
            input_features: InputFeatures::default(),
            processing_path: "standard".to_string(),
            router_confidence: 0.9,
            outcome_quality: 0.8,
        };

        optimizer.observe(snapshot);

        assert_eq!(optimizer.stats().snapshots_processed, 1);
        assert_eq!(optimizer.patterns().len(), 1);
    }

    #[test]
    fn test_recommendation() {
        let mut optimizer = MetaCognitiveOptimizer::with_defaults();

        // Train with some observations
        for i in 0..10 {
            let snapshot = ConsciousnessSnapshot {
                timestamp: Utc::now(),
                phi: 0.7 + (i as f64 * 0.02),
                free_energy: 0.3 - (i as f64 * 0.02),
                input_features: InputFeatures {
                    complexity: 0.6,
                    novelty: 0.4,
                    ..Default::default()
                },
                processing_path: "standard".to_string(),
                router_confidence: 0.85,
                outcome_quality: 0.75,
            };
            optimizer.observe(snapshot);
        }

        // Get recommendation
        let features = InputFeatures {
            complexity: 0.6,
            novelty: 0.4,
            ..Default::default()
        };
        let rec = optimizer.recommend(&features);

        assert!(rec.expected_phi > 0.0);
        assert!(rec.confidence > 0.0);
    }

    #[test]
    fn test_pattern_discovery() {
        let mut optimizer = MetaCognitiveOptimizer::with_defaults();

        // Different feature clusters should create different patterns
        // Cluster 1: High complexity, high novelty, high uncertainty (deliberation needed)
        for _ in 0..10 {
            optimizer.observe(ConsciousnessSnapshot {
                timestamp: Utc::now(),
                phi: 0.8,
                free_energy: 0.2,
                input_features: InputFeatures {
                    complexity: 0.95,
                    novelty: 0.9,
                    uncertainty: 0.85,
                    domain_match: 0.8,
                    length_norm: 0.9,
                },
                processing_path: "full_deliberation".to_string(),
                router_confidence: 0.9,
                outcome_quality: 0.9,
            });
        }

        // Cluster 2: Low complexity, low novelty, low uncertainty (reflexive)
        // Features differ enough to exceed the 1.0 total_diff threshold for distinct patterns
        // (similarity_threshold = 0.8 means total_diff >= 1.0 for different pattern)
        for _ in 0..10 {
            optimizer.observe(ConsciousnessSnapshot {
                timestamp: Utc::now(),
                phi: 0.4,
                free_energy: 0.5,
                input_features: InputFeatures {
                    complexity: 0.1,
                    novelty: 0.15,
                    uncertainty: 0.1,
                    domain_match: 0.9,
                    length_norm: 0.2,
                },
                processing_path: "reflex".to_string(),
                router_confidence: 0.95,
                outcome_quality: 0.7,
            });
        }

        // Should have discovered 2 patterns
        // total_diff = |0.95-0.1| + |0.9-0.15| + |0.85-0.1| + |0.8-0.9| + |0.9-0.2|
        //            = 0.85 + 0.75 + 0.75 + 0.1 + 0.7 = 3.15
        // similarity = 1.0 - 3.15/5.0 = 1.0 - 0.63 = 0.37 < 0.8 threshold
        assert_eq!(optimizer.patterns().len(), 2);
    }

    #[test]
    fn test_belief_update() {
        let mut optimizer = MetaCognitiveOptimizer::with_defaults();

        let initial_prior = optimizer.beliefs().prior_high_phi;

        // Observe many high Φ events
        for _ in 0..100 {
            optimizer.observe(ConsciousnessSnapshot {
                timestamp: Utc::now(),
                phi: 0.85, // Above threshold
                free_energy: 0.15,
                input_features: InputFeatures::default(),
                processing_path: "standard".to_string(),
                router_confidence: 0.9,
                outcome_quality: 0.85,
            });
        }

        // Prior should have increased
        assert!(optimizer.beliefs().prior_high_phi > initial_prior);
    }

    #[test]
    fn test_self_analysis_report() {
        let mut optimizer = MetaCognitiveOptimizer::with_defaults();

        for i in 0..20 {
            optimizer.observe(ConsciousnessSnapshot {
                timestamp: Utc::now(),
                phi: 0.5 + (i as f64 * 0.02),
                free_energy: 0.4 - (i as f64 * 0.01),
                input_features: InputFeatures::default(),
                processing_path: "standard".to_string(),
                router_confidence: 0.8,
                outcome_quality: 0.7,
            });
        }

        let report = optimizer.self_analysis_report();
        assert!(report.contains("Observations: 20"));
        assert!(report.contains("Meta-Cognitive Self-Analysis Report"));
    }
}
