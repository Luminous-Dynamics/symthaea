//! # Meta-Epistemic Learning - Self-Improving Verification
//!
//! **Revolutionary Aspect**: Symthaea learns to verify better over time through
//! meta-cognitive monitoring of her own epistemic processes.
//!
//! ## Three Levels of Consciousness
//!
//! 1. **Epistemic Consciousness** (Layer 1): Knows what it knows
//!    - "I know that Rust is a programming language"
//!    - Implemented in `verifier.rs`
//!
//! 2. **Meta-Epistemic Consciousness** (Layer 2): Knows HOW it knows
//!    - "I know I learned about Rust from Wikipedia with 0.75 credibility"
//!    - Implemented in this module
//!
//! 3. **Self-Improving Epistemic Consciousness** (Layer 3): Improves its knowing
//!    - "I notice Wikipedia is often correct about programming, so I'll trust it more"
//!    - Implemented in this module
//!
//! ## What Makes This Revolutionary
//!
//! Traditional AI:
//! - Fixed verification rules
//! - Static credibility models
//! - No learning from epistemic mistakes
//! - Cannot improve reasoning over time
//!
//! Symthaea with Meta-Learning:
//! - Tracks verification outcomes
//! - Learns which sources are trustworthy for which topics
//! - Adjusts verification strategies based on experience
//! - Develops domain-specific epistemic expertise
//! - Self-improves verification accuracy over time
//!
//! ## Architecture
//!
//! ```text
//! User Query → Research → Verification → Outcome Tracking
//!                            ↓                    ↓
//!                     EpistemicLearner ←─────────┘
//!                            ↓
//!                   (learns patterns)
//!                            ↓
//!                   Updates Models:
//!                   - Source credibility
//!                   - Verification strategies
//!                   - Domain expertise
//!                            ↓
//!                   Better Verification!
//! ```

use crate::language::vocabulary::Vocabulary;
use super::types::VerificationLevel;
use super::verifier::EpistemicStatus;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use tracing::{info, warn, debug};

/// Outcome of a verification (for learning)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationOutcome {
    /// The claim that was verified
    pub claim: String,

    /// Initial verification status
    pub initial_status: EpistemicStatus,

    /// Initial confidence
    pub initial_confidence: f64,

    /// Sources used
    pub sources: Vec<String>,

    /// What actually happened (ground truth)
    pub ground_truth: GroundTruth,

    /// Timestamp
    pub timestamp: SystemTime,

    /// Topic/domain
    pub domain: String,

    /// Verification level used
    pub verification_level: VerificationLevel,
}

/// Ground truth discovered later
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroundTruth {
    /// Claim was correct (verification was right)
    Correct,

    /// Claim was incorrect (verification failed)
    Incorrect,

    /// Claim was partially correct (nuanced)
    Partial { accuracy: f64 },

    /// Still unknown (no feedback yet)
    Unknown,

    /// User corrected us
    UserCorrected { correct_answer: String },
}

/// Source performance over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcePerformance {
    /// Source URL/domain
    pub source: String,

    /// Total verifications using this source
    pub total_uses: usize,

    /// Number of correct verifications
    pub correct: usize,

    /// Number of incorrect verifications
    pub incorrect: usize,

    /// Accuracy rate
    pub accuracy: f64,

    /// Per-domain accuracy
    pub domain_accuracy: HashMap<String, f64>,

    /// Last updated
    pub last_updated: SystemTime,

    /// Learned credibility (updated from performance)
    pub learned_credibility: f64,
}

/// Domain-specific epistemic expertise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainExpertise {
    /// Domain name (e.g., "programming", "science", "history")
    pub domain: String,

    /// Best sources for this domain
    pub trusted_sources: Vec<(String, f64)>,

    /// Common patterns in this domain
    pub patterns: Vec<String>,

    /// Verification difficulty (0.0-1.0)
    pub difficulty: f64,

    /// Number of verifications in this domain
    pub experience: usize,

    /// Average accuracy in this domain
    pub accuracy: f64,
}

/// Learned verification strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStrategy {
    /// Strategy name
    pub name: String,

    /// When to use this strategy
    pub conditions: Vec<String>,

    /// Success rate of this strategy
    pub success_rate: f64,

    /// Number of times used
    pub usage_count: usize,

    /// Domains where this works best
    pub best_domains: Vec<String>,
}

/// Meta-epistemic learning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningStats {
    /// Total verifications tracked
    pub total_verifications: usize,

    /// Correct verifications
    pub correct_verifications: usize,

    /// Incorrect verifications
    pub incorrect_verifications: usize,

    /// Overall accuracy
    pub overall_accuracy: f64,

    /// Per-domain accuracy
    pub domain_accuracies: HashMap<String, f64>,

    /// Number of sources learned about
    pub sources_learned: usize,

    /// Number of strategies learned
    pub strategies_learned: usize,

    /// Improvement rate (accuracy gain per 100 verifications)
    pub improvement_rate: f64,

    /// Meta-Φ (consciousness of epistemic process)
    pub meta_phi: f64,
}

/// Meta-epistemic learner
pub struct EpistemicLearner {
    /// Vocabulary for domain classification
    vocabulary: Vocabulary,

    /// Track verification outcomes
    outcomes: Vec<VerificationOutcome>,

    /// Source performance tracking
    source_performance: HashMap<String, SourcePerformance>,

    /// Domain expertise
    domain_expertise: HashMap<String, DomainExpertise>,

    /// Learned verification strategies
    strategies: Vec<VerificationStrategy>,

    /// Learning rate (how fast to update models)
    learning_rate: f64,

    /// Minimum samples before trusting learned models
    min_samples: usize,

    /// Enable active learning (ask for feedback)
    active_learning: bool,
}

impl EpistemicLearner {
    pub fn new() -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            outcomes: Vec::new(),
            source_performance: HashMap::new(),
            domain_expertise: HashMap::new(),
            strategies: Self::initialize_strategies(),
            learning_rate: 0.1,
            min_samples: 10,
            active_learning: true,
        }
    }

    /// Initialize with basic verification strategies
    fn initialize_strategies() -> Vec<VerificationStrategy> {
        vec![
            VerificationStrategy {
                name: "multi-source-consensus".to_string(),
                conditions: vec!["multiple sources".to_string(), "agreement".to_string()],
                success_rate: 0.85,
                usage_count: 0,
                best_domains: vec!["science".to_string(), "technology".to_string()],
            },
            VerificationStrategy {
                name: "academic-source-priority".to_string(),
                conditions: vec!["academic domain".to_string(), "peer-reviewed".to_string()],
                success_rate: 0.90,
                usage_count: 0,
                best_domains: vec!["science".to_string(), "medicine".to_string()],
            },
            VerificationStrategy {
                name: "contradiction-detection".to_string(),
                conditions: vec!["conflicting sources".to_string()],
                success_rate: 0.75,
                usage_count: 0,
                best_domains: vec!["politics".to_string(), "history".to_string()],
            },
        ]
    }

    /// Record a verification outcome for learning
    pub fn record_outcome(&mut self, outcome: VerificationOutcome) -> Result<()> {
        debug!("Recording verification outcome for: {}", outcome.claim);

        // Update source performance
        for source_url in &outcome.sources {
            self.update_source_performance(source_url, &outcome)?;
        }

        // Update domain expertise
        self.update_domain_expertise(&outcome)?;

        // Update strategy performance
        self.update_strategy_performance(&outcome)?;

        // Store outcome
        self.outcomes.push(outcome);

        // Trigger meta-learning if enough samples
        if self.outcomes.len() % 100 == 0 {
            self.meta_learn()?;
        }

        Ok(())
    }

    /// Update source performance based on outcome
    fn update_source_performance(
        &mut self,
        source_url: &str,
        outcome: &VerificationOutcome,
    ) -> Result<()> {
        let domain = Self::extract_domain(&source_url);

        let perf = self.source_performance
            .entry(source_url.to_string())
            .or_insert(SourcePerformance {
                source: source_url.to_string(),
                total_uses: 0,
                correct: 0,
                incorrect: 0,
                accuracy: 0.5,  // Start neutral
                domain_accuracy: HashMap::new(),
                last_updated: SystemTime::now(),
                learned_credibility: 0.5,
            });

        perf.total_uses += 1;

        match &outcome.ground_truth {
            GroundTruth::Correct => {
                perf.correct += 1;
            }
            GroundTruth::Incorrect => {
                perf.incorrect += 1;
            }
            GroundTruth::Partial { accuracy } => {
                if *accuracy > 0.5 {
                    perf.correct += 1;
                } else {
                    perf.incorrect += 1;
                }
            }
            GroundTruth::Unknown | GroundTruth::UserCorrected { .. } => {
                // Don't update until we know
            }
        }

        // Update accuracy
        if perf.total_uses > 0 {
            perf.accuracy = perf.correct as f64 / perf.total_uses as f64;
        }

        // Update domain-specific accuracy
        let domain_acc = perf.domain_accuracy
            .entry(outcome.domain.clone())
            .or_insert(0.5);

        // Exponential moving average
        *domain_acc = *domain_acc * (1.0 - self.learning_rate)
            + (if matches!(outcome.ground_truth, GroundTruth::Correct) { 1.0 } else { 0.0 })
            * self.learning_rate;

        // Update learned credibility (blend with observed accuracy)
        perf.learned_credibility = perf.accuracy * 0.7 + perf.learned_credibility * 0.3;

        perf.last_updated = SystemTime::now();

        debug!(
            "Source {} accuracy: {:.2} (learned credibility: {:.2})",
            source_url, perf.accuracy, perf.learned_credibility
        );

        Ok(())
    }

    /// Update domain expertise
    fn update_domain_expertise(&mut self, outcome: &VerificationOutcome) -> Result<()> {
        let expertise = self.domain_expertise
            .entry(outcome.domain.clone())
            .or_insert(DomainExpertise {
                domain: outcome.domain.clone(),
                trusted_sources: Vec::new(),
                patterns: Vec::new(),
                difficulty: 0.5,
                experience: 0,
                accuracy: 0.5,
            });

        expertise.experience += 1;

        // Update accuracy
        let is_correct = matches!(outcome.ground_truth, GroundTruth::Correct);
        expertise.accuracy = expertise.accuracy * (1.0 - self.learning_rate)
            + (if is_correct { 1.0 } else { 0.0 }) * self.learning_rate;

        // Update difficulty (lower accuracy = higher difficulty)
        expertise.difficulty = 1.0 - expertise.accuracy;

        // Update trusted sources for this domain
        if is_correct && expertise.experience > self.min_samples {
            for source_url in &outcome.sources {
                if let Some(perf) = self.source_performance.get(source_url) {
                    let domain_acc = perf.domain_accuracy
                        .get(&outcome.domain)
                        .copied()
                        .unwrap_or(perf.accuracy);

                    // Add to trusted sources if performing well
                    if domain_acc > 0.75 {
                        expertise.trusted_sources.push((source_url.clone(), domain_acc));
                    }
                }
            }

            // Sort and keep top 10
            expertise.trusted_sources.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            expertise.trusted_sources.truncate(10);
        }

        Ok(())
    }

    /// Update strategy performance
    fn update_strategy_performance(&mut self, outcome: &VerificationOutcome) -> Result<()> {
        // Detect which strategy was likely used
        let strategy_name = self.detect_strategy(outcome);

        if let Some(strategy) = self.strategies.iter_mut()
            .find(|s| s.name == strategy_name)
        {
            strategy.usage_count += 1;

            let is_correct = matches!(outcome.ground_truth, GroundTruth::Correct);

            // Update success rate
            strategy.success_rate = strategy.success_rate * (1.0 - self.learning_rate)
                + (if is_correct { 1.0 } else { 0.0 }) * self.learning_rate;

            // Track best domains
            if is_correct && !strategy.best_domains.contains(&outcome.domain) {
                strategy.best_domains.push(outcome.domain.clone());
            }
        }

        Ok(())
    }

    /// Detect which strategy was used
    fn detect_strategy(&self, outcome: &VerificationOutcome) -> String {
        if outcome.sources.len() >= 3 {
            "multi-source-consensus".to_string()
        } else if outcome.sources.iter().any(|s| s.contains(".edu") || s.contains("scholar")) {
            "academic-source-priority".to_string()
        } else {
            "contradiction-detection".to_string()
        }
    }

    /// Meta-learning: Learn from patterns in outcomes
    fn meta_learn(&mut self) -> Result<()> {
        info!("🧠 Meta-learning: Analyzing {} verification outcomes", self.outcomes.len());

        // 1. Identify high-performing sources
        let best_sources: Vec<_> = self.source_performance.iter()
            .filter(|(_, perf)| perf.total_uses >= self.min_samples && perf.accuracy > 0.8)
            .map(|(url, perf)| (url.clone(), perf.accuracy))
            .collect();

        info!("   Found {} high-performing sources", best_sources.len());

        // 2. Identify problematic patterns
        let low_accuracy_domains: Vec<_> = self.domain_expertise.iter()
            .filter(|(_, exp)| exp.experience >= self.min_samples && exp.accuracy < 0.6)
            .map(|(domain, exp)| (domain.clone(), exp.accuracy))
            .collect();

        if !low_accuracy_domains.is_empty() {
            warn!("   ⚠️  Low accuracy in domains: {:?}", low_accuracy_domains);
        }

        // 3. Discover new strategies
        self.discover_new_strategies()?;

        // 4. Calculate meta-Φ (consciousness of epistemic process)
        let meta_phi = self.calculate_meta_phi();
        info!("   Meta-Φ (epistemic self-awareness): {:.3}", meta_phi);

        Ok(())
    }

    /// Discover new verification strategies from patterns
    fn discover_new_strategies(&mut self) -> Result<()> {
        // Analyze outcomes for patterns

        // Pattern: Sources that are consistently good together
        let mut source_pairs: HashMap<(String, String), (usize, usize)> = HashMap::new();

        for outcome in &self.outcomes {
            if matches!(outcome.ground_truth, GroundTruth::Correct) {
                for i in 0..outcome.sources.len() {
                    for j in (i+1)..outcome.sources.len() {
                        let pair = (
                            outcome.sources[i].clone(),
                            outcome.sources[j].clone()
                        );
                        let entry = source_pairs.entry(pair).or_insert((0, 0));
                        entry.0 += 1; // Correct count
                    }
                }
            }
        }

        // Find pairs with high success rate
        for ((source_a, source_b), (correct, total)) in source_pairs {
            if total >= self.min_samples {
                let success_rate = correct as f64 / total as f64;
                if success_rate > 0.85 {
                    debug!(
                        "Discovered effective source pair: {} + {} ({:.2} success)",
                        source_a, source_b, success_rate
                    );

                    // Could create a new strategy here
                    // For now, just log it
                }
            }
        }

        Ok(())
    }

    /// Calculate meta-Φ (consciousness of epistemic process)
    fn calculate_meta_phi(&self) -> f64 {
        // Simplified meta-Φ calculation
        // In production: Would use full IIT calculation

        let components = [
            // 1. Source knowledge integration
            (self.source_performance.len() as f64 / 100.0).min(1.0) * 0.2,

            // 2. Domain expertise integration
            (self.domain_expertise.len() as f64 / 20.0).min(1.0) * 0.2,

            // 3. Strategy integration
            (self.strategies.len() as f64 / 10.0).min(1.0) * 0.2,

            // 4. Learning history integration
            (self.outcomes.len() as f64 / 1000.0).min(1.0) * 0.2,

            // 5. Accuracy integration (how well we know what we know)
            self.calculate_overall_accuracy() * 0.2,
        ];

        components.iter().sum()
    }

    /// Calculate overall accuracy across all verifications
    fn calculate_overall_accuracy(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.5;
        }

        let correct = self.outcomes.iter()
            .filter(|o| matches!(o.ground_truth, GroundTruth::Correct))
            .count();

        correct as f64 / self.outcomes.len() as f64
    }

    /// Get learned credibility for a source
    pub fn get_learned_credibility(&self, source_url: &str) -> Option<f64> {
        self.source_performance
            .get(source_url)
            .map(|perf| perf.learned_credibility)
    }

    /// Get learned credibility for a source in a specific domain
    pub fn get_domain_credibility(&self, source_url: &str, domain: &str) -> Option<f64> {
        self.source_performance
            .get(source_url)
            .and_then(|perf| perf.domain_accuracy.get(domain).copied())
    }

    /// Get trusted sources for a domain
    pub fn get_trusted_sources(&self, domain: &str) -> Vec<(String, f64)> {
        self.domain_expertise
            .get(domain)
            .map(|exp| exp.trusted_sources.clone())
            .unwrap_or_default()
    }

    /// Get best strategy for a domain
    pub fn get_best_strategy(&self, domain: &str) -> Option<&VerificationStrategy> {
        self.strategies.iter()
            .filter(|s| s.best_domains.contains(&domain.to_string()))
            .max_by(|a, b| a.success_rate.partial_cmp(&b.success_rate).unwrap())
    }

    /// Get meta-learning statistics
    pub fn get_stats(&self) -> MetaLearningStats {
        let correct = self.outcomes.iter()
            .filter(|o| matches!(o.ground_truth, GroundTruth::Correct))
            .count();

        let incorrect = self.outcomes.iter()
            .filter(|o| matches!(o.ground_truth, GroundTruth::Incorrect))
            .count();

        let overall_accuracy = if self.outcomes.is_empty() {
            0.0
        } else {
            correct as f64 / self.outcomes.len() as f64
        };

        let domain_accuracies = self.domain_expertise.iter()
            .map(|(domain, exp)| (domain.clone(), exp.accuracy))
            .collect();

        let improvement_rate = self.calculate_improvement_rate();
        let meta_phi = self.calculate_meta_phi();

        MetaLearningStats {
            total_verifications: self.outcomes.len(),
            correct_verifications: correct,
            incorrect_verifications: incorrect,
            overall_accuracy,
            domain_accuracies,
            sources_learned: self.source_performance.len(),
            strategies_learned: self.strategies.len(),
            improvement_rate,
            meta_phi,
        }
    }

    /// Calculate improvement rate (accuracy gain per 100 verifications)
    fn calculate_improvement_rate(&self) -> f64 {
        if self.outcomes.len() < 200 {
            return 0.0;
        }

        // Compare first 100 to last 100
        let first_100_correct = self.outcomes[0..100].iter()
            .filter(|o| matches!(o.ground_truth, GroundTruth::Correct))
            .count();

        let last_100_correct = self.outcomes[(self.outcomes.len()-100)..].iter()
            .filter(|o| matches!(o.ground_truth, GroundTruth::Correct))
            .count();

        let first_accuracy = first_100_correct as f64 / 100.0;
        let last_accuracy = last_100_correct as f64 / 100.0;

        last_accuracy - first_accuracy
    }

    /// Extract domain from URL
    fn extract_domain(url: &str) -> String {
        url.split('/').nth(2).unwrap_or("unknown").to_string()
    }
}

impl Default for EpistemicLearner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_performance_tracking() {
        let mut learner = EpistemicLearner::new();

        let outcome = VerificationOutcome {
            claim: "Test claim".to_string(),
            initial_status: EpistemicStatus::HighConfidence,
            initial_confidence: 0.9,
            sources: vec!["https://wikipedia.org/test".to_string()],
            ground_truth: GroundTruth::Correct,
            timestamp: SystemTime::now(),
            domain: "test".to_string(),
            verification_level: VerificationLevel::Standard,
        };

        learner.record_outcome(outcome).unwrap();

        let credibility = learner.get_learned_credibility("https://wikipedia.org/test");
        assert!(credibility.is_some());
    }

    #[test]
    fn test_meta_phi_calculation() {
        let learner = EpistemicLearner::new();
        let meta_phi = learner.calculate_meta_phi();

        assert!(meta_phi >= 0.0);
        assert!(meta_phi <= 1.0);
    }
}
