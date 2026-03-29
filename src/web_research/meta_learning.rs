// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meta-Learning Module for Epistemic Self-Improvement
//!
//! This module implements the third level of epistemic consciousness:
//! the system learns to improve its own verification process over time.
//! It tracks verification outcomes, learns source trustworthiness per
//! domain, and develops domain-specific expertise.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use super::types::{ResearchSource, VerificationOutcome};

/// Configuration for the epistemic learner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerConfig {
    /// Learning rate for updating source credibility
    pub learning_rate: f32,
    /// Minimum outcomes needed before trusting learned values
    pub min_outcomes_threshold: usize,
    /// Decay rate for old outcomes
    pub decay_rate: f32,
    /// Maximum outcomes to track per source
    pub max_outcomes_per_source: usize,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            min_outcomes_threshold: 5,
            decay_rate: 0.01,
            max_outcomes_per_source: 100,
        }
    }
}

/// Learned credibility for a source domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCredibility {
    /// The source domain (e.g., "rust-lang.org")
    pub domain: String,
    /// Overall credibility score (0.0 - 1.0)
    pub credibility: f32,
    /// Per-domain credibility (e.g., "programming" -> 0.95)
    pub domain_credibility: HashMap<String, f32>,
    /// Number of outcomes tracked
    pub outcome_count: usize,
    /// Number of correct outcomes
    pub correct_count: usize,
    /// Source type
    pub source_type: ResearchSource,
}

impl SourceCredibility {
    /// Create a new source credibility entry
    pub fn new(domain: String, source_type: ResearchSource) -> Self {
        Self {
            domain,
            credibility: 0.5, // Start neutral
            domain_credibility: HashMap::new(),
            outcome_count: 0,
            correct_count: 0,
            source_type,
        }
    }

    /// Get accuracy rate
    pub fn accuracy(&self) -> f32 {
        if self.outcome_count == 0 {
            return 0.5;
        }
        self.correct_count as f32 / self.outcome_count as f32
    }

    /// Get credibility for a specific topic domain
    pub fn credibility_for_domain(&self, topic_domain: &str) -> f32 {
        self.domain_credibility
            .get(topic_domain)
            .copied()
            .unwrap_or(self.credibility)
    }
}

/// Strategy for verification based on learned patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStrategy {
    /// Strategy name
    pub name: String,
    /// Topic domain this strategy applies to
    pub domain: String,
    /// Preferred source types for this domain
    pub preferred_sources: Vec<ResearchSource>,
    /// Minimum sources required
    pub min_sources: usize,
    /// Confidence threshold for high confidence
    pub high_confidence_threshold: f32,
    /// Success rate with this strategy
    pub success_rate: f32,
    /// Number of times this strategy was used
    pub use_count: usize,
}

/// Meta-Phi: a measure of epistemic self-awareness
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaPhi {
    /// Overall meta-phi value
    pub value: f64,
    /// Improvement rate (positive = improving)
    pub improvement_rate: f64,
    /// Number of meta-learning cycles
    pub cycles: usize,
    /// Initial accuracy (for comparison)
    pub initial_accuracy: f32,
    /// Current accuracy
    pub current_accuracy: f32,
    /// Accuracy improvement
    pub accuracy_improvement: f32,
}

impl MetaPhi {
    /// Calculate meta-phi based on learning metrics
    pub fn calculate(
        initial_accuracy: f32,
        current_accuracy: f32,
        domain_count: usize,
        strategy_count: usize,
    ) -> Self {
        let accuracy_improvement = current_accuracy - initial_accuracy;

        // Meta-Phi combines:
        // 1. Accuracy improvement
        // 2. Domain expertise development
        // 3. Strategy library growth
        let domain_factor = (domain_count as f64 / 10.0).min(1.0);
        let strategy_factor = (strategy_count as f64 / 5.0).min(1.0);
        let improvement_factor = (accuracy_improvement as f64 + 0.5).clamp(0.0, 1.0);

        let value =
            (improvement_factor * 0.5 + domain_factor * 0.3 + strategy_factor * 0.2).min(1.0);

        Self {
            value,
            improvement_rate: accuracy_improvement as f64,
            cycles: 0,
            initial_accuracy,
            current_accuracy,
            accuracy_improvement,
        }
    }

    /// Update meta-phi after a learning cycle
    pub fn update(&mut self, new_accuracy: f32, domain_count: usize, strategy_count: usize) {
        self.cycles += 1;
        self.current_accuracy = new_accuracy;
        self.accuracy_improvement = new_accuracy - self.initial_accuracy;

        // Recalculate value
        let domain_factor = (domain_count as f64 / 10.0).min(1.0);
        let strategy_factor = (strategy_count as f64 / 5.0).min(1.0);
        let improvement_factor = (self.accuracy_improvement as f64 + 0.5).clamp(0.0, 1.0);

        let new_value =
            (improvement_factor * 0.5 + domain_factor * 0.3 + strategy_factor * 0.2).min(1.0);

        // Calculate improvement rate (smoothed)
        self.improvement_rate = self.improvement_rate * 0.9 + (new_value - self.value) * 0.1;
        self.value = new_value;
    }
}

/// Epistemic learner for meta-learning verification
#[derive(Debug)]
pub struct EpistemicLearner {
    /// Configuration
    config: LearnerConfig,
    /// Source credibility tracking
    source_credibility: HashMap<String, SourceCredibility>,
    /// Verification strategies
    strategies: Vec<VerificationStrategy>,
    /// Outcome history
    outcomes: VecDeque<VerificationOutcome>,
    /// Meta-Phi tracking
    meta_phi: MetaPhi,
    /// Domain expertise scores
    domain_expertise: HashMap<String, f32>,
}

impl Default for EpistemicLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicLearner {
    /// Create a new epistemic learner
    pub fn new() -> Self {
        Self {
            config: LearnerConfig::default(),
            source_credibility: HashMap::new(),
            strategies: Vec::new(),
            outcomes: VecDeque::new(),
            meta_phi: MetaPhi::default(),
            domain_expertise: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: LearnerConfig) -> Self {
        Self {
            config,
            source_credibility: HashMap::new(),
            strategies: Vec::new(),
            outcomes: VecDeque::new(),
            meta_phi: MetaPhi::default(),
            domain_expertise: HashMap::new(),
        }
    }

    /// Record a verification outcome for learning
    pub fn record_outcome(&mut self, outcome: VerificationOutcome) -> anyhow::Result<()> {
        // Update source credibility
        self.update_source_credibility(&outcome);

        // Update domain expertise
        self.update_domain_expertise(&outcome);

        // Store outcome
        self.outcomes.push_back(outcome);

        // Trim old outcomes
        if self.outcomes.len() > 1000 {
            self.outcomes.pop_front();
        }

        // Periodically update meta-phi
        if self.outcomes.len() % 10 == 0 {
            self.update_meta_phi();
        }

        Ok(())
    }

    /// Update source credibility based on outcome
    fn update_source_credibility(&mut self, outcome: &VerificationOutcome) {
        let domain = &outcome.source_domain;

        let credibility = self
            .source_credibility
            .entry(domain.clone())
            .or_insert_with(|| SourceCredibility::new(domain.clone(), outcome.source_type));

        // Update counts
        credibility.outcome_count += 1;
        if outcome.was_correct {
            credibility.correct_count += 1;
        }

        // Update overall credibility (exponential moving average)
        let target = if outcome.was_correct { 1.0 } else { 0.0 };
        credibility.credibility += self.config.learning_rate * (target - credibility.credibility);
        credibility.credibility = credibility.credibility.clamp(0.1, 0.99);

        // Update domain-specific credibility
        let topic_domain = &outcome.domain;
        let current = credibility
            .domain_credibility
            .get(topic_domain)
            .copied()
            .unwrap_or(0.5);
        let new_value = current + self.config.learning_rate * (target - current);
        credibility
            .domain_credibility
            .insert(topic_domain.clone(), new_value.clamp(0.1, 0.99));
    }

    /// Update domain expertise based on outcome
    fn update_domain_expertise(&mut self, outcome: &VerificationOutcome) {
        let domain = &outcome.domain;

        let current = self.domain_expertise.get(domain).copied().unwrap_or(0.5);

        // Expertise increases faster with correct outcomes
        let adjustment = if outcome.was_correct {
            self.config.learning_rate * 1.5
        } else {
            -self.config.learning_rate * 0.5
        };

        let new_value = (current + adjustment).clamp(0.0, 1.0);
        self.domain_expertise.insert(domain.clone(), new_value);
    }

    /// Update meta-phi calculation
    fn update_meta_phi(&mut self) {
        let current_accuracy = self.calculate_overall_accuracy();
        let domain_count = self.domain_expertise.len();
        let strategy_count = self.strategies.len();

        self.meta_phi
            .update(current_accuracy, domain_count, strategy_count);
    }

    /// Calculate overall accuracy across all outcomes
    fn calculate_overall_accuracy(&self) -> f32 {
        if self.outcomes.is_empty() {
            return 0.5;
        }

        let correct = self.outcomes.iter().filter(|o| o.was_correct).count();
        correct as f32 / self.outcomes.len() as f32
    }

    /// Get source credibility for a domain
    pub fn get_source_credibility(&self, domain: &str) -> Option<&SourceCredibility> {
        self.source_credibility.get(domain)
    }

    /// Get recommended sources for a topic domain
    pub fn recommend_sources(&self, topic_domain: &str) -> Vec<(String, f32)> {
        let mut recommendations: Vec<_> = self
            .source_credibility
            .values()
            .map(|s| {
                let credibility = s.credibility_for_domain(topic_domain);
                (s.domain.clone(), credibility)
            })
            .collect();

        // Sort by credibility (descending)
        recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        recommendations.truncate(10);
        recommendations
    }

    /// Get domain expertise level
    pub fn get_domain_expertise(&self, domain: &str) -> f32 {
        self.domain_expertise.get(domain).copied().unwrap_or(0.5)
    }

    /// Get current meta-phi
    pub fn meta_phi(&self) -> &MetaPhi {
        &self.meta_phi
    }

    /// Get learning statistics
    pub fn stats(&self) -> LearnerStats {
        LearnerStats {
            total_outcomes: self.outcomes.len(),
            accuracy: self.calculate_overall_accuracy(),
            source_count: self.source_credibility.len(),
            domain_count: self.domain_expertise.len(),
            strategy_count: self.strategies.len(),
            meta_phi: self.meta_phi.value,
            improvement_rate: self.meta_phi.improvement_rate,
        }
    }

    /// Check if the learner has enough data to make recommendations
    pub fn is_ready(&self) -> bool {
        self.outcomes.len() >= self.config.min_outcomes_threshold
    }

    /// Get verification strategy recommendation for a domain
    pub fn recommend_strategy(&self, topic_domain: &str) -> Option<&VerificationStrategy> {
        self.strategies
            .iter()
            .filter(|s| s.domain == topic_domain)
            .max_by(|a, b| {
                a.success_rate
                    .partial_cmp(&b.success_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Add a verification strategy
    pub fn add_strategy(&mut self, strategy: VerificationStrategy) {
        self.strategies.push(strategy);
    }

    /// Export learner state for persistence
    pub fn export_state(&self) -> LearnerState {
        LearnerState {
            source_credibility: self.source_credibility.clone(),
            strategies: self.strategies.clone(),
            meta_phi: self.meta_phi.clone(),
            domain_expertise: self.domain_expertise.clone(),
        }
    }

    /// Import learner state from persistence
    pub fn import_state(&mut self, state: LearnerState) {
        self.source_credibility = state.source_credibility;
        self.strategies = state.strategies;
        self.meta_phi = state.meta_phi;
        self.domain_expertise = state.domain_expertise;
    }
}

/// Statistics about the epistemic learner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerStats {
    /// Total outcomes recorded
    pub total_outcomes: usize,
    /// Overall accuracy
    pub accuracy: f32,
    /// Number of tracked sources
    pub source_count: usize,
    /// Number of domains with expertise
    pub domain_count: usize,
    /// Number of strategies
    pub strategy_count: usize,
    /// Current meta-phi value
    pub meta_phi: f64,
    /// Meta-phi improvement rate
    pub improvement_rate: f64,
}

/// Exportable learner state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerState {
    /// Source credibility data
    pub source_credibility: HashMap<String, SourceCredibility>,
    /// Learned strategies
    pub strategies: Vec<VerificationStrategy>,
    /// Meta-phi tracking
    pub meta_phi: MetaPhi,
    /// Domain expertise
    pub domain_expertise: HashMap<String, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outcome(domain: &str, source: &str, correct: bool) -> VerificationOutcome {
        VerificationOutcome {
            claim: "Test claim".to_string(),
            source_domain: source.to_string(),
            source_type: ResearchSource::Documentation,
            was_correct: correct,
            predicted_confidence: 0.8,
            domain: domain.to_string(),
        }
    }

    #[test]
    fn test_record_outcomes() {
        let mut learner = EpistemicLearner::new();

        // Record some outcomes: 7 correct, 3 wrong
        for i in 0..10 {
            let correct = i % 3 != 0; // i=0,3,6,9 are wrong, rest are correct (6/10)
            learner
                .record_outcome(make_outcome("programming", "rust-lang.org", correct))
                .unwrap();
        }

        let stats = learner.stats();
        assert_eq!(stats.total_outcomes, 10);
        // 6 out of 10 are correct (i=1,2,4,5,7,8)
        assert!(
            stats.accuracy >= 0.5 && stats.accuracy <= 0.7,
            "Expected accuracy between 0.5 and 0.7, got {}",
            stats.accuracy
        );
        assert_eq!(stats.source_count, 1);
    }

    #[test]
    fn test_source_credibility_learning() {
        let mut learner = EpistemicLearner::new();

        // Source that is mostly correct
        for _ in 0..20 {
            learner
                .record_outcome(make_outcome("programming", "docs.rust-lang.org", true))
                .unwrap();
        }

        // Source that is often wrong
        for _ in 0..20 {
            learner
                .record_outcome(make_outcome("programming", "random-blog.com", false))
                .unwrap();
        }

        let good_source = learner
            .get_source_credibility("docs.rust-lang.org")
            .unwrap();
        let bad_source = learner.get_source_credibility("random-blog.com").unwrap();

        assert!(good_source.credibility > bad_source.credibility);
        assert!(good_source.accuracy() > 0.9);
        assert!(bad_source.accuracy() < 0.1);
    }

    #[test]
    fn test_domain_expertise() {
        let mut learner = EpistemicLearner::new();

        // Build expertise in programming
        for _ in 0..30 {
            learner
                .record_outcome(make_outcome("programming", "rust-lang.org", true))
                .unwrap();
        }

        let expertise = learner.get_domain_expertise("programming");
        assert!(expertise > 0.7);

        // No expertise in unknown domain
        let unknown = learner.get_domain_expertise("underwater_basket_weaving");
        assert_eq!(unknown, 0.5);
    }

    #[test]
    fn test_meta_phi_calculation() {
        let meta_phi = MetaPhi::calculate(0.5, 0.8, 5, 3);

        assert!(meta_phi.value > 0.0);
        assert_eq!(meta_phi.accuracy_improvement, 0.3);
    }

    #[test]
    fn test_recommend_sources() {
        let mut learner = EpistemicLearner::new();

        // Build history with different sources
        for _ in 0..10 {
            learner
                .record_outcome(make_outcome("programming", "rust-lang.org", true))
                .unwrap();
        }
        for _ in 0..10 {
            learner
                .record_outcome(make_outcome("programming", "stackoverflow.com", true))
                .unwrap();
            learner
                .record_outcome(make_outcome("programming", "stackoverflow.com", false))
                .unwrap();
        }

        let recommendations = learner.recommend_sources("programming");
        assert!(!recommendations.is_empty());

        // rust-lang.org should be ranked higher
        if recommendations.len() >= 2 {
            assert!(recommendations[0].0 == "rust-lang.org" || recommendations[0].1 > 0.7);
        }
    }
}
