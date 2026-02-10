//! Adaptive Threshold Learning
//!
//! Implements online learning algorithms for dynamically adjusting
//! trust thresholds based on system feedback.
//!
//! Features:
//! - Multi-armed bandit (UCB1) for threshold selection
//! - Gradient estimation with momentum
//! - Combined adaptive engine
//! - Feedback processing for learning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for adaptive thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// UCB exploration parameter
    pub ucb_c: f64,
    /// Minimum threshold value
    pub min_threshold: f64,
    /// Maximum threshold value
    pub max_threshold: f64,
    /// Number of arms for bandit
    pub num_arms: usize,
    /// Window size for recent feedback
    pub feedback_window: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            ucb_c: 2.0,
            min_threshold: 0.1,
            max_threshold: 0.9,
            num_arms: 10,
            feedback_window: 100,
        }
    }
}

/// Threshold type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThresholdType {
    /// Trust acceptance threshold
    TrustAcceptance,
    /// Quarantine threshold
    Quarantine,
    /// Slashing threshold
    Slashing,
    /// Reward threshold
    Reward,
    /// Byzantine detection threshold
    ByzantineDetection,
    /// Coherence threshold (Phi)
    Coherence,
    /// Consensus threshold
    Consensus,
}

/// Current state of a threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdState {
    /// Threshold type
    pub threshold_type: ThresholdType,
    /// Current value
    pub value: f64,
    /// Number of updates
    pub updates: usize,
    /// Cumulative reward
    pub cumulative_reward: f64,
    /// Last update timestamp
    pub last_update: u64,
}

/// Feedback for threshold adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdFeedback {
    /// Threshold type
    pub threshold_type: ThresholdType,
    /// Threshold value used
    pub threshold_value: f64,
    /// Outcome
    pub outcome: FeedbackOutcome,
    /// Context
    pub context: FeedbackContext,
    /// Timestamp
    pub timestamp: u64,
}

/// Feedback outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackOutcome {
    /// Threshold correctly accepted good behavior
    TruePositive,
    /// Threshold correctly rejected bad behavior
    TrueNegative,
    /// Threshold incorrectly accepted bad behavior
    FalsePositive,
    /// Threshold incorrectly rejected good behavior
    FalseNegative,
}

impl FeedbackOutcome {
    /// Convert to reward value
    pub fn to_reward(&self) -> f64 {
        match self {
            FeedbackOutcome::TruePositive => 1.0,
            FeedbackOutcome::TrueNegative => 1.0,
            FeedbackOutcome::FalsePositive => -1.0,
            FeedbackOutcome::FalseNegative => -0.5, // Less penalty for being too cautious
        }
    }

    /// Whether this is a correct decision
    pub fn is_correct(&self) -> bool {
        matches!(
            self,
            FeedbackOutcome::TruePositive | FeedbackOutcome::TrueNegative
        )
    }
}

/// Context for feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackContext {
    /// Network health at time of decision
    pub network_health: f64,
    /// Number of active agents
    pub active_agents: usize,
    /// Current threat level
    pub threat_level: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for FeedbackContext {
    fn default() -> Self {
        Self {
            network_health: 1.0,
            active_agents: 0,
            threat_level: 0.0,
            metadata: HashMap::new(),
        }
    }
}

/// Bandit arm state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditArm {
    /// Arm index
    pub index: usize,
    /// Threshold value for this arm
    pub threshold: f64,
    /// Number of pulls
    pub pulls: usize,
    /// Total reward
    pub total_reward: f64,
    /// Average reward
    pub average_reward: f64,
}

impl BanditArm {
    /// Create new arm
    pub fn new(index: usize, threshold: f64) -> Self {
        Self {
            index,
            threshold,
            pulls: 0,
            total_reward: 0.0,
            average_reward: 0.0,
        }
    }

    /// Update arm with reward
    pub fn update(&mut self, reward: f64) {
        self.pulls += 1;
        self.total_reward += reward;
        self.average_reward = self.total_reward / self.pulls as f64;
    }

    /// Calculate UCB value
    pub fn ucb_value(&self, total_pulls: usize, c: f64) -> f64 {
        if self.pulls == 0 {
            return f64::INFINITY; // Ensure unexplored arms are tried
        }

        let exploitation = self.average_reward;
        let exploration = c * ((total_pulls as f64).ln() / self.pulls as f64).sqrt();

        exploitation + exploration
    }
}

/// Multi-armed bandit for threshold selection
pub struct ThresholdBandit {
    /// Configuration
    config: AdaptiveConfig,
    /// Arms for each threshold type
    arms: HashMap<ThresholdType, Vec<BanditArm>>,
    /// Total pulls per threshold type
    total_pulls: HashMap<ThresholdType, usize>,
}

impl ThresholdBandit {
    /// Create new bandit
    pub fn new(config: AdaptiveConfig) -> Self {
        let mut arms = HashMap::new();
        let total_pulls = HashMap::new();

        // Initialize arms for common threshold types
        let threshold_types = [
            ThresholdType::TrustAcceptance,
            ThresholdType::Quarantine,
            ThresholdType::Slashing,
            ThresholdType::ByzantineDetection,
            ThresholdType::Coherence,
        ];

        for threshold_type in threshold_types {
            let type_arms = Self::create_arms(&config);
            arms.insert(threshold_type, type_arms);
        }

        Self {
            config,
            arms,
            total_pulls,
        }
    }

    /// Create arms for a threshold type
    fn create_arms(config: &AdaptiveConfig) -> Vec<BanditArm> {
        let step = (config.max_threshold - config.min_threshold) / (config.num_arms - 1) as f64;

        (0..config.num_arms)
            .map(|i| {
                let threshold = config.min_threshold + step * i as f64;
                BanditArm::new(i, threshold)
            })
            .collect()
    }

    /// Select threshold using UCB1
    pub fn select(&self, threshold_type: ThresholdType) -> f64 {
        let arms = match self.arms.get(&threshold_type) {
            Some(a) => a,
            None => return 0.5, // Default
        };

        let total = self.total_pulls.get(&threshold_type).copied().unwrap_or(0);

        // Find arm with highest UCB value
        let best_arm = arms.iter().max_by(|a, b| {
            let ucb_a = a.ucb_value(total.max(1), self.config.ucb_c);
            let ucb_b = b.ucb_value(total.max(1), self.config.ucb_c);
            ucb_a
                .partial_cmp(&ucb_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        best_arm.map(|a| a.threshold).unwrap_or(0.5)
    }

    /// Update with feedback
    pub fn update(&mut self, feedback: &ThresholdFeedback) {
        let threshold_type = feedback.threshold_type;
        let reward = feedback.outcome.to_reward();

        // Find the arm closest to the used threshold
        if let Some(arms) = self.arms.get_mut(&threshold_type) {
            let arm_idx = arms
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let dist_a = (a.threshold - feedback.threshold_value).abs();
                    let dist_b = (b.threshold - feedback.threshold_value).abs();
                    dist_a
                        .partial_cmp(&dist_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);

            if let Some(idx) = arm_idx {
                arms[idx].update(reward);
                *self.total_pulls.entry(threshold_type).or_insert(0) += 1;
            }
        }
    }

    /// Get current best threshold for each type
    pub fn best_thresholds(&self) -> HashMap<ThresholdType, f64> {
        self.arms
            .iter()
            .map(|(t, arms)| {
                let best = arms
                    .iter()
                    .filter(|a| a.pulls > 0)
                    .max_by(|a, b| {
                        a.average_reward
                            .partial_cmp(&b.average_reward)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|a| a.threshold)
                    .unwrap_or(0.5);
                (*t, best)
            })
            .collect()
    }

    /// Get arm statistics for a threshold type
    pub fn arm_stats(&self, threshold_type: ThresholdType) -> Option<Vec<(f64, f64, usize)>> {
        self.arms.get(&threshold_type).map(|arms| {
            arms.iter()
                .map(|a| (a.threshold, a.average_reward, a.pulls))
                .collect()
        })
    }
}

/// Gradient estimator with momentum
pub struct GradientEstimator {
    /// Configuration
    config: AdaptiveConfig,
    /// Current threshold values
    thresholds: HashMap<ThresholdType, f64>,
    /// Velocity (momentum)
    velocity: HashMap<ThresholdType, f64>,
    /// Recent feedback window
    recent_feedback: Vec<ThresholdFeedback>,
}

impl GradientEstimator {
    /// Create new gradient estimator
    pub fn new(config: AdaptiveConfig) -> Self {
        let mut thresholds = HashMap::new();
        let velocity = HashMap::new();

        // Initialize default thresholds
        thresholds.insert(ThresholdType::TrustAcceptance, 0.5);
        thresholds.insert(ThresholdType::Quarantine, 0.3);
        thresholds.insert(ThresholdType::Slashing, 0.2);
        thresholds.insert(ThresholdType::ByzantineDetection, 0.4);
        thresholds.insert(ThresholdType::Coherence, 0.6);

        Self {
            config,
            thresholds,
            velocity,
            recent_feedback: vec![],
        }
    }

    /// Get current threshold
    pub fn get_threshold(&self, threshold_type: ThresholdType) -> f64 {
        self.thresholds.get(&threshold_type).copied().unwrap_or(0.5)
    }

    /// Update with feedback
    pub fn update(&mut self, feedback: ThresholdFeedback) {
        // Add to recent feedback
        self.recent_feedback.push(feedback.clone());
        if self.recent_feedback.len() > self.config.feedback_window {
            self.recent_feedback.remove(0);
        }

        // Estimate gradient
        let gradient = self.estimate_gradient(feedback.threshold_type);

        // Update velocity with momentum
        let v = self.velocity.entry(feedback.threshold_type).or_insert(0.0);
        *v = self.config.momentum * *v + self.config.learning_rate * gradient;

        // Update threshold
        let threshold = self
            .thresholds
            .entry(feedback.threshold_type)
            .or_insert(0.5);
        *threshold += *v;

        // Clamp to valid range
        *threshold = threshold.clamp(self.config.min_threshold, self.config.max_threshold);
    }

    /// Estimate gradient from recent feedback
    fn estimate_gradient(&self, threshold_type: ThresholdType) -> f64 {
        let relevant: Vec<_> = self
            .recent_feedback
            .iter()
            .filter(|f| f.threshold_type == threshold_type)
            .collect();

        if relevant.is_empty() {
            return 0.0;
        }

        // Count false positives vs false negatives
        let false_positives = relevant
            .iter()
            .filter(|f| matches!(f.outcome, FeedbackOutcome::FalsePositive))
            .count();

        let false_negatives = relevant
            .iter()
            .filter(|f| matches!(f.outcome, FeedbackOutcome::FalseNegative))
            .count();

        // If too many false positives, increase threshold
        // If too many false negatives, decrease threshold
        let fp_rate = false_positives as f64 / relevant.len() as f64;
        let fn_rate = false_negatives as f64 / relevant.len() as f64;

        // Gradient: positive = increase threshold, negative = decrease
        fp_rate - fn_rate
    }

    /// Get all current thresholds
    pub fn all_thresholds(&self) -> &HashMap<ThresholdType, f64> {
        &self.thresholds
    }
}

/// Combined adaptive threshold engine
pub struct AdaptiveThresholdEngine {
    /// Configuration
    config: AdaptiveConfig,
    /// Bandit for exploration
    bandit: ThresholdBandit,
    /// Gradient estimator for refinement
    gradient: GradientEstimator,
    /// Current mode per threshold
    mode: HashMap<ThresholdType, AdaptiveMode>,
    /// Feedback count per threshold
    feedback_count: HashMap<ThresholdType, usize>,
    /// Exploration threshold (switch to gradient after this many samples)
    exploration_threshold: usize,
}

/// Adaptive mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdaptiveMode {
    /// Exploration via bandit
    Exploration,
    /// Exploitation via gradient descent
    Exploitation,
}

impl AdaptiveThresholdEngine {
    /// Create new engine
    pub fn new(config: AdaptiveConfig) -> Self {
        Self {
            bandit: ThresholdBandit::new(config.clone()),
            gradient: GradientEstimator::new(config.clone()),
            mode: HashMap::new(),
            feedback_count: HashMap::new(),
            exploration_threshold: 50,
            config,
        }
    }

    /// Get recommended threshold
    pub fn get_threshold(&self, threshold_type: ThresholdType) -> f64 {
        let mode = self
            .mode
            .get(&threshold_type)
            .copied()
            .unwrap_or(AdaptiveMode::Exploration);

        match mode {
            AdaptiveMode::Exploration => self.bandit.select(threshold_type),
            AdaptiveMode::Exploitation => self.gradient.get_threshold(threshold_type),
        }
    }

    /// Process feedback
    pub fn process_feedback(&mut self, feedback: ThresholdFeedback) {
        let threshold_type = feedback.threshold_type;

        // Update counters
        *self.feedback_count.entry(threshold_type).or_insert(0) += 1;
        let count = self.feedback_count[&threshold_type];

        // Check if should switch modes
        if count > self.exploration_threshold {
            self.mode.insert(threshold_type, AdaptiveMode::Exploitation);
        } else {
            self.mode
                .entry(threshold_type)
                .or_insert(AdaptiveMode::Exploration);
        }

        // Update both algorithms
        self.bandit.update(&feedback);
        self.gradient.update(feedback);
    }

    /// Get recommendation with explanation
    pub fn recommend(&self, threshold_type: ThresholdType) -> ThresholdRecommendation {
        let mode = self
            .mode
            .get(&threshold_type)
            .copied()
            .unwrap_or(AdaptiveMode::Exploration);
        let value = self.get_threshold(threshold_type);

        let (confidence, direction) = match mode {
            AdaptiveMode::Exploration => {
                let count = self
                    .feedback_count
                    .get(&threshold_type)
                    .copied()
                    .unwrap_or(0);
                let conf = (count as f64 / self.exploration_threshold as f64).min(1.0);
                (conf, RecommendationDirection::Exploring)
            }
            AdaptiveMode::Exploitation => {
                // High confidence if gradient is small
                let gradient_velocity = self
                    .gradient
                    .velocity
                    .get(&threshold_type)
                    .copied()
                    .unwrap_or(0.0)
                    .abs();
                let conf = (1.0 - gradient_velocity * 10.0).max(0.5);

                let dir = if gradient_velocity.abs() < 0.001 {
                    RecommendationDirection::Stable
                } else if gradient_velocity > 0.0 {
                    RecommendationDirection::Increasing
                } else {
                    RecommendationDirection::Decreasing
                };

                (conf, dir)
            }
        };

        ThresholdRecommendation {
            threshold_type,
            value,
            confidence,
            direction,
            samples: self
                .feedback_count
                .get(&threshold_type)
                .copied()
                .unwrap_or(0),
        }
    }

    /// Adapt all thresholds based on context
    pub fn adapt_thresholds(&mut self, context: &FeedbackContext) -> HashMap<ThresholdType, f64> {
        let mut thresholds = HashMap::new();

        // Get base thresholds
        for threshold_type in [
            ThresholdType::TrustAcceptance,
            ThresholdType::Quarantine,
            ThresholdType::Slashing,
            ThresholdType::ByzantineDetection,
            ThresholdType::Coherence,
        ] {
            let mut value = self.get_threshold(threshold_type);

            // Adjust based on context
            // Higher threat = stricter thresholds
            if context.threat_level > 0.5 {
                let adjustment = (context.threat_level - 0.5) * 0.2;
                match threshold_type {
                    ThresholdType::TrustAcceptance => value += adjustment,
                    ThresholdType::Quarantine => value -= adjustment,
                    ThresholdType::ByzantineDetection => value -= adjustment,
                    _ => {}
                }
            }

            // Lower health = more cautious
            if context.network_health < 0.7 {
                let adjustment = (0.7 - context.network_health) * 0.1;
                match threshold_type {
                    ThresholdType::Slashing => value += adjustment,
                    ThresholdType::Quarantine => value -= adjustment,
                    _ => {}
                }
            }

            // Clamp final value
            value = value.clamp(self.config.min_threshold, self.config.max_threshold);
            thresholds.insert(threshold_type, value);
        }

        thresholds
    }

    /// Get bandit arm statistics
    pub fn arm_stats(&self, threshold_type: ThresholdType) -> Option<Vec<(f64, f64, usize)>> {
        self.bandit.arm_stats(threshold_type)
    }
}

/// Threshold recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdRecommendation {
    /// Threshold type
    pub threshold_type: ThresholdType,
    /// Recommended value
    pub value: f64,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
    /// Direction of change
    pub direction: RecommendationDirection,
    /// Number of samples used
    pub samples: usize,
}

/// Direction of threshold change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationDirection {
    /// Still exploring, direction uncertain
    Exploring,
    /// Threshold is stable
    Stable,
    /// Threshold trending upward
    Increasing,
    /// Threshold trending downward
    Decreasing,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandit_arm() {
        let mut arm = BanditArm::new(0, 0.5);

        arm.update(1.0);
        arm.update(0.5);
        arm.update(0.8);

        assert_eq!(arm.pulls, 3);
        assert!((arm.average_reward - 0.766).abs() < 0.01);
    }

    #[test]
    fn test_bandit_selection() {
        let config = AdaptiveConfig::default();
        let bandit = ThresholdBandit::new(config);

        // Initially should explore (return various values)
        let threshold = bandit.select(ThresholdType::TrustAcceptance);
        assert!(threshold >= 0.1 && threshold <= 0.9);
    }

    #[test]
    fn test_bandit_update() {
        let config = AdaptiveConfig::default();
        let mut bandit = ThresholdBandit::new(config);

        let feedback = ThresholdFeedback {
            threshold_type: ThresholdType::TrustAcceptance,
            threshold_value: 0.5,
            outcome: FeedbackOutcome::TruePositive,
            context: FeedbackContext::default(),
            timestamp: 1000,
        };

        bandit.update(&feedback);

        let stats = bandit.arm_stats(ThresholdType::TrustAcceptance).unwrap();
        let pulled: Vec<_> = stats.iter().filter(|(_, _, pulls)| *pulls > 0).collect();
        assert!(!pulled.is_empty());
    }

    #[test]
    fn test_gradient_estimator() {
        let config = AdaptiveConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut estimator = GradientEstimator::new(config);

        // Add false positive feedback (should increase threshold)
        for _ in 0..5 {
            let feedback = ThresholdFeedback {
                threshold_type: ThresholdType::TrustAcceptance,
                threshold_value: 0.5,
                outcome: FeedbackOutcome::FalsePositive,
                context: FeedbackContext::default(),
                timestamp: 1000,
            };
            estimator.update(feedback);
        }

        // Threshold should have increased
        let threshold = estimator.get_threshold(ThresholdType::TrustAcceptance);
        assert!(
            threshold > 0.5,
            "Threshold should increase after false positives"
        );
    }

    #[test]
    fn test_adaptive_engine() {
        let config = AdaptiveConfig::default();
        let mut engine = AdaptiveThresholdEngine::new(config);

        // Process some feedback
        for i in 0..60 {
            let outcome = if i % 3 == 0 {
                FeedbackOutcome::FalsePositive
            } else {
                FeedbackOutcome::TruePositive
            };

            let feedback = ThresholdFeedback {
                threshold_type: ThresholdType::TrustAcceptance,
                threshold_value: 0.5,
                outcome,
                context: FeedbackContext::default(),
                timestamp: 1000 + i,
            };
            engine.process_feedback(feedback);
        }

        let recommendation = engine.recommend(ThresholdType::TrustAcceptance);
        assert!(recommendation.samples >= 60);
        assert!(recommendation.confidence > 0.0);
    }

    #[test]
    fn test_context_adaptation() {
        let config = AdaptiveConfig::default();
        let mut engine = AdaptiveThresholdEngine::new(config);

        // Normal context
        let normal_context = FeedbackContext {
            network_health: 1.0,
            threat_level: 0.1,
            ..Default::default()
        };

        let normal_thresholds = engine.adapt_thresholds(&normal_context);

        // High threat context
        let threat_context = FeedbackContext {
            network_health: 0.5,
            threat_level: 0.8,
            ..Default::default()
        };

        let threat_thresholds = engine.adapt_thresholds(&threat_context);

        // Trust acceptance should be higher under threat
        assert!(
            threat_thresholds[&ThresholdType::TrustAcceptance]
                >= normal_thresholds[&ThresholdType::TrustAcceptance]
        );
    }

    #[test]
    fn test_feedback_outcome_rewards() {
        assert_eq!(FeedbackOutcome::TruePositive.to_reward(), 1.0);
        assert_eq!(FeedbackOutcome::TrueNegative.to_reward(), 1.0);
        assert_eq!(FeedbackOutcome::FalsePositive.to_reward(), -1.0);
        assert_eq!(FeedbackOutcome::FalseNegative.to_reward(), -0.5);

        assert!(FeedbackOutcome::TruePositive.is_correct());
        assert!(!FeedbackOutcome::FalsePositive.is_correct());
    }
}
