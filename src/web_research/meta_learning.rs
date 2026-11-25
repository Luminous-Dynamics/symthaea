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

use super::types::{AdjudicatedVerificationOutcome, ResearchSource, VerificationOutcome};

const ADJUDICATED_LEARNER_STATE_VERSION: u16 = 1;
const MAX_PERSISTED_ADJUDICATED_OUTCOMES: usize = 1_000;
#[cfg(test)]
use super::types::AdjudicationKind;

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

    /// Record an outcome only when its independent-adjudication metadata is present.
    ///
    /// The wrapper validates required provenance fields and confidence bounds.
    /// It does not authenticate the adjudicator or resolve the referenced
    /// evidence; those remain responsibilities of the independent custody path.
    pub fn record_adjudicated_outcome(
        &mut self,
        adjudicated: AdjudicatedVerificationOutcome,
    ) -> anyhow::Result<()> {
        let outcome = adjudicated
            .into_outcome()
            .map_err(anyhow::Error::msg)?;
        self.record_outcome_verified(outcome)
    }

    /// Internal sink for outcomes already validated by the adjudication boundary.
    fn record_outcome_verified(&mut self, outcome: VerificationOutcome) -> anyhow::Result<()> {
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

    /// Iterate over outcomes that retain independent adjudication provenance.
    ///
    /// Unadjudicated predictions are intentionally excluded from this view and
    /// must not be used as correctness labels.
    pub fn adjudicated_outcomes(
        &self,
    ) -> impl Iterator<Item = &VerificationOutcome> {
        self.outcomes
            .iter()
            .filter(|outcome| outcome.adjudication_provenance.is_some())
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

    /// Export the adjudicated observations needed to deterministically rebuild
    /// calibration state. Cached aggregates are deliberately not persisted as
    /// an authority; they are recomputed from the provenance-bearing records.
    pub fn export_state(&self) -> LearnerState {
        LearnerState {
            adjudication_schema_version: ADJUDICATED_LEARNER_STATE_VERSION,
            adjudicated_outcomes: self.adjudicated_outcomes().cloned().collect(),
        }
    }

    fn reset_calibration_state(&mut self) {
        self.source_credibility.clear();
        self.strategies.clear();
        self.outcomes.clear();
        self.meta_phi = MetaPhi::default();
        self.domain_expertise.clear();
    }

    fn valid_persisted_adjudication(outcome: &VerificationOutcome) -> bool {
        let Some(provenance) = &outcome.adjudication_provenance else {
            return false;
        };
        !outcome.claim.trim().is_empty()
            && !outcome.source_domain.trim().is_empty()
            && !outcome.domain.trim().is_empty()
            && outcome.predicted_confidence.is_finite()
            && (0.0..=1.0).contains(&outcome.predicted_confidence)
            && !provenance.authority_id.trim().is_empty()
            && !provenance.evidence_reference.trim().is_empty()
    }

    /// Import learner state from persistence, rebuilding aggregates from
    /// provenance-bearing outcome records.
    ///
    /// Unversioned legacy snapshots are rejected fail-closed because their
    /// calibration aggregates predate adjudication provenance and may include
    /// confidence-derived correctness labels. This also rejects versioned
    /// snapshots containing any unadjudicated or malformed outcome. In either
    /// case all calibration-derived state is reset.
    ///
    /// Returns true only when a compatible state was imported successfully.
    /// This validates provenance structure, not authority signatures or the
    /// truth of the referenced evidence.
    pub fn import_state(&mut self, state: LearnerState) -> bool {
        self.reset_calibration_state();
        if state.adjudication_schema_version != ADJUDICATED_LEARNER_STATE_VERSION
            || state.adjudicated_outcomes.len() > MAX_PERSISTED_ADJUDICATED_OUTCOMES
            || state
                .adjudicated_outcomes
                .iter()
                .any(|outcome| !Self::valid_persisted_adjudication(outcome))
        {
            return false;
        }

        for outcome in state.adjudicated_outcomes {
            if self.record_outcome_verified(outcome).is_err() {
                self.reset_calibration_state();
                return false;
            }
        }
        true
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
    /// Version of the independent-adjudication contract used to build this state.
    /// Missing values deserialize as zero and are rejected by import_state().
    #[serde(default)]
    pub adjudication_schema_version: u16,
    /// Ground-truth records used to rebuild source/domain/meta-learning state.
    /// Every entry must carry non-empty adjudication provenance.
    #[serde(default)]
    pub adjudicated_outcomes: Vec<VerificationOutcome>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outcome(domain: &str, source: &str, correct: bool) -> AdjudicatedVerificationOutcome {
        AdjudicatedVerificationOutcome {
            outcome: VerificationOutcome {
                claim: "Test claim".to_string(),
                source_domain: source.to_string(),
                source_type: ResearchSource::Documentation,
                was_correct: correct,
                predicted_confidence: 0.8,
                domain: domain.to_string(),
                adjudication_provenance: None,
            },
            adjudication_kind: AdjudicationKind::Other,
            authority_id: "synthetic-unit-test-authority".to_owned(),
            evidence_reference: "test-fixture://epistemic-outcome".to_owned(),
        }
    }

    #[test]
    fn persisted_state_rebuilds_calibration_from_adjudicated_records() {
        let mut learner = EpistemicLearner::new();
        for index in 0..12 {
            learner
                .record_adjudicated_outcome(make_outcome(
                    "programming",
                    "rust-lang.org",
                    index % 4 != 0,
                ))
                .unwrap();
        }

        let state = learner.export_state();
        assert_eq!(state.adjudication_schema_version, ADJUDICATED_LEARNER_STATE_VERSION);
        assert_eq!(state.adjudicated_outcomes.len(), 12);
        assert!(state
            .adjudicated_outcomes
            .iter()
            .all(|outcome| outcome.adjudication_provenance.is_some()));

        let serialized = serde_json::to_string(&state).unwrap();
        let state: LearnerState = serde_json::from_str(&serialized).unwrap();
        let mut restored = EpistemicLearner::new();
        assert!(restored.import_state(state));
        assert_eq!(restored.stats().total_outcomes, 12);
        assert_eq!(restored.adjudicated_outcomes().count(), 12);
        assert_eq!(
            restored
                .get_source_credibility("rust-lang.org")
                .unwrap()
                .outcome_count,
            12
        );
    }

    #[test]
    fn legacy_or_malformed_persisted_state_resets_calibration() {
        let mut seeded = EpistemicLearner::new();
        for _ in 0..12 {
            seeded
                .record_adjudicated_outcome(make_outcome("programming", "rust-lang.org", true))
                .unwrap();
        }
        let legacy_json = r#"{
            "source_credibility": {},
            "strategies": [],
            "meta_phi": {},
            "domain_expertise": {}
        }"#;
        let legacy_state: LearnerState = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(legacy_state.adjudication_schema_version, 0);

        let mut restored = EpistemicLearner::new();
        restored
            .record_adjudicated_outcome(make_outcome("history", "example.org", false))
            .unwrap();
        assert!(!restored.import_state(legacy_state));
        assert_eq!(restored.stats().total_outcomes, 0);
        assert_eq!(restored.stats().source_count, 0);
        assert_eq!(restored.stats().domain_count, 0);
        assert_eq!(restored.adjudicated_outcomes().count(), 0);

        let mut malformed_state = seeded.export_state();
        malformed_state.adjudicated_outcomes[0].adjudication_provenance = None;
        assert!(!restored.import_state(malformed_state));
        assert_eq!(restored.stats().total_outcomes, 0);
        assert_eq!(restored.stats().source_count, 0);

        let mut invalid_confidence_state = seeded.export_state();
        invalid_confidence_state.adjudicated_outcomes[0].predicted_confidence = f32::INFINITY;
        assert!(!restored.import_state(invalid_confidence_state));
        assert_eq!(restored.stats().total_outcomes, 0);
        assert_eq!(restored.stats().source_count, 0);

        let mut oversized_state = seeded.export_state();
        let duplicate = oversized_state.adjudicated_outcomes[0].clone();
        oversized_state
            .adjudicated_outcomes
            .extend(std::iter::repeat_n(duplicate, MAX_PERSISTED_ADJUDICATED_OUTCOMES + 1));
        assert!(!restored.import_state(oversized_state));
        assert_eq!(restored.stats().total_outcomes, 0);
        assert_eq!(restored.stats().source_count, 0);
    }

    #[test]
    fn test_record_outcomes() {
        let mut learner = EpistemicLearner::new();

        // Record some outcomes: 7 correct, 3 wrong
        for i in 0..10 {
            let correct = i % 3 != 0; // i=0,3,6,9 are wrong, rest are correct (6/10)
            learner
                .record_adjudicated_outcome(make_outcome("programming", "rust-lang.org", correct))
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
                .record_adjudicated_outcome(make_outcome("programming", "docs.rust-lang.org", true))
                .unwrap();
        }

        // Source that is often wrong
        for _ in 0..20 {
            learner
                .record_adjudicated_outcome(make_outcome("programming", "random-blog.com", false))
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
                .record_adjudicated_outcome(make_outcome("programming", "rust-lang.org", true))
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
                .record_adjudicated_outcome(make_outcome("programming", "rust-lang.org", true))
                .unwrap();
        }
        for _ in 0..10 {
            learner
                .record_adjudicated_outcome(make_outcome("programming", "stackoverflow.com", true))
                .unwrap();
            learner
                .record_adjudicated_outcome(make_outcome("programming", "stackoverflow.com", false))
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
