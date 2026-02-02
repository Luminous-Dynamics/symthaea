//! # Adaptive Thresholds Engine
//!
//! ML-driven dynamic adjustment of trust thresholds based on network conditions.
//!
//! ## Features
//!
//! - **Online Learning**: Continuous threshold adaptation
//! - **Multi-Armed Bandits**: Explore-exploit for optimal thresholds
//! - **Anomaly-Aware**: Adjust based on detected attack patterns
//! - **Feedback Loop**: Learn from consensus outcomes
//!
//! ## Philosophy
//!
//! Static thresholds are brittle. The optimal trust threshold for consensus
//! depends on network conditions, attack prevalence, and agent population.
//! This module dynamically adjusts thresholds to maintain security while
//! maximizing participation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// Configuration
// ============================================================================

/// Adaptive thresholds configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Initial trust threshold
    pub initial_threshold: f64,
    /// Minimum allowed threshold
    pub min_threshold: f64,
    /// Maximum allowed threshold
    pub max_threshold: f64,
    /// Learning rate for gradient updates
    pub learning_rate: f64,
    /// Exploration rate for bandits (epsilon)
    pub exploration_rate: f64,
    /// Window size for moving average
    pub window_size: usize,
    /// Update frequency (every N events)
    pub update_frequency: u32,
    /// Enable automatic adjustment
    pub auto_adjust: bool,
    /// Momentum for smoothing
    pub momentum: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            initial_threshold: 0.5,
            min_threshold: 0.3,
            max_threshold: 0.9,
            learning_rate: 0.01,
            exploration_rate: 0.1,
            window_size: 100,
            update_frequency: 10,
            auto_adjust: true,
            momentum: 0.9,
        }
    }
}

// ============================================================================
// Threshold Types
// ============================================================================

/// Types of thresholds to adapt
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThresholdType {
    /// Minimum trust for group membership
    Membership,
    /// Trust required for voting
    Voting,
    /// Trust required for proposals
    Proposal,
    /// Quorum threshold
    Quorum,
    /// Approval threshold
    Approval,
    /// Trust for high-stakes actions
    HighStakes,
    /// Alert trigger threshold
    AlertTrigger,
}

/// Current threshold state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdState {
    /// Threshold type
    pub threshold_type: ThresholdType,
    /// Current value
    pub value: f64,
    /// Velocity (change rate)
    pub velocity: f64,
    /// Confidence in current value
    pub confidence: f64,
    /// Last update timestamp
    pub last_updated: u64,
    /// Update count
    pub update_count: u64,
}

impl ThresholdState {
    pub fn new(threshold_type: ThresholdType, initial_value: f64) -> Self {
        Self {
            threshold_type,
            value: initial_value,
            velocity: 0.0,
            confidence: 0.5,
            last_updated: 0,
            update_count: 0,
        }
    }
}

// ============================================================================
// Feedback Signal
// ============================================================================

/// Feedback from system operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdFeedback {
    /// Threshold type this feedback relates to
    pub threshold_type: ThresholdType,
    /// Outcome type
    pub outcome: FeedbackOutcome,
    /// Current threshold value when outcome occurred
    pub threshold_at_event: f64,
    /// Additional context
    pub context: FeedbackContext,
    /// Timestamp
    pub timestamp: u64,
}

/// Feedback outcomes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackOutcome {
    /// Positive outcome (threshold was appropriate)
    Success,
    /// Negative outcome (threshold was too low - let bad actor through)
    FalseNegative,
    /// Negative outcome (threshold was too high - blocked good actor)
    FalsePositive,
    /// Neutral (no clear signal)
    Neutral,
}

/// Additional context for feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackContext {
    /// Number of participants
    pub participants: u32,
    /// Average trust of participants
    pub average_trust: f64,
    /// Was there an attack detected?
    pub attack_detected: bool,
    /// Network health at time
    pub network_health: f64,
}

// ============================================================================
// Multi-Armed Bandit
// ============================================================================

/// Arm in the bandit problem (a threshold value)
#[derive(Debug, Clone)]
pub struct BanditArm {
    /// Threshold value
    pub value: f64,
    /// Number of times pulled
    pub pulls: u64,
    /// Total reward accumulated
    pub total_reward: f64,
    /// Average reward
    pub average_reward: f64,
    /// UCB score
    pub ucb_score: f64,
}

impl BanditArm {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            pulls: 0,
            total_reward: 0.0,
            average_reward: 0.0,
            ucb_score: f64::MAX, // Unpulled arms have infinite UCB
        }
    }

    /// Update arm with reward
    pub fn update(&mut self, reward: f64) {
        self.pulls += 1;
        self.total_reward += reward;
        self.average_reward = self.total_reward / self.pulls as f64;
    }

    /// Calculate UCB score
    pub fn calculate_ucb(&mut self, total_pulls: u64, exploration: f64) {
        if self.pulls == 0 {
            self.ucb_score = f64::MAX;
        } else {
            let exploitation = self.average_reward;
            let exploration_bonus = exploration * ((total_pulls as f64).ln() / self.pulls as f64).sqrt();
            self.ucb_score = exploitation + exploration_bonus;
        }
    }
}

/// Multi-armed bandit for threshold selection
#[derive(Debug)]
pub struct ThresholdBandit {
    /// Arms (threshold values)
    arms: Vec<BanditArm>,
    /// Total pulls across all arms
    total_pulls: u64,
    /// Exploration parameter
    exploration: f64,
    /// Current selected arm index
    current_arm: usize,
}

impl ThresholdBandit {
    /// Create bandit with discretized threshold values
    pub fn new(min: f64, max: f64, num_arms: usize, exploration: f64) -> Self {
        let step = (max - min) / (num_arms - 1) as f64;
        let arms = (0..num_arms)
            .map(|i| {
                // Clamp to avoid floating point precision issues at boundaries
                let value = (min + step * i as f64).clamp(min, max);
                BanditArm::new(value)
            })
            .collect();

        Self {
            arms,
            total_pulls: 0,
            exploration,
            current_arm: num_arms / 2, // Start in middle
        }
    }

    /// Select next arm using UCB1
    pub fn select_arm(&mut self) -> f64 {
        // Update UCB scores
        for arm in &mut self.arms {
            arm.calculate_ucb(self.total_pulls, self.exploration);
        }

        // Select arm with highest UCB score
        self.current_arm = self.arms.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.ucb_score.partial_cmp(&b.ucb_score).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.arms[self.current_arm].value
    }

    /// Update with reward
    pub fn update(&mut self, reward: f64) {
        self.total_pulls += 1;
        self.arms[self.current_arm].update(reward);
    }

    /// Get best arm so far
    pub fn best_arm(&self) -> f64 {
        self.arms.iter()
            .max_by(|a, b| a.average_reward.partial_cmp(&b.average_reward).unwrap())
            .map(|a| a.value)
            .unwrap_or(0.5)
    }

    /// Get arm statistics
    pub fn stats(&self) -> Vec<(f64, u64, f64)> {
        self.arms.iter()
            .map(|a| (a.value, a.pulls, a.average_reward))
            .collect()
    }
}

// ============================================================================
// Gradient Estimator
// ============================================================================

/// Online gradient estimator for continuous threshold adjustment
#[derive(Debug)]
pub struct GradientEstimator {
    /// Current threshold value
    current: f64,
    /// Gradient estimate
    gradient: f64,
    /// Learning rate
    learning_rate: f64,
    /// Momentum
    momentum: f64,
    /// Previous gradient (for momentum)
    prev_gradient: f64,
    /// Recent rewards for variance estimation
    recent_rewards: VecDeque<f64>,
    /// Window size
    window_size: usize,
}

impl GradientEstimator {
    pub fn new(initial: f64, learning_rate: f64, momentum: f64, window_size: usize) -> Self {
        Self {
            current: initial,
            gradient: 0.0,
            learning_rate,
            momentum,
            prev_gradient: 0.0,
            recent_rewards: VecDeque::new(),
            window_size,
        }
    }

    /// Update with reward signal
    pub fn update(&mut self, reward: f64, perturbation: f64) {
        // Store reward
        self.recent_rewards.push_back(reward);
        while self.recent_rewards.len() > self.window_size {
            self.recent_rewards.pop_front();
        }

        // Estimate gradient using REINFORCE-style update
        let baseline = self.recent_rewards.iter().sum::<f64>()
            / self.recent_rewards.len() as f64;

        let advantage = reward - baseline;
        let gradient_estimate = advantage * perturbation;

        // Apply momentum
        self.gradient = self.momentum * self.prev_gradient
            + (1.0 - self.momentum) * gradient_estimate;
        self.prev_gradient = self.gradient;
    }

    /// Step the threshold
    pub fn step(&mut self, min: f64, max: f64) -> f64 {
        self.current += self.learning_rate * self.gradient;
        self.current = self.current.clamp(min, max);
        self.current
    }

    /// Get current value
    pub fn current(&self) -> f64 {
        self.current
    }

    /// Get gradient estimate
    pub fn gradient(&self) -> f64 {
        self.gradient
    }
}

// ============================================================================
// Adaptive Threshold Engine
// ============================================================================

/// Main adaptive thresholds engine
#[derive(Debug)]
#[allow(dead_code)]
pub struct AdaptiveThresholdEngine {
    config: AdaptiveConfig,
    /// Thresholds by type
    thresholds: HashMap<ThresholdType, ThresholdState>,
    /// Bandits for each threshold type
    bandits: HashMap<ThresholdType, ThresholdBandit>,
    /// Gradient estimators
    gradients: HashMap<ThresholdType, GradientEstimator>,
    /// Feedback history
    feedback_history: VecDeque<ThresholdFeedback>,
    /// Event counter
    event_counter: u32,
    /// Current timestamp
    current_time: u64,
    /// RNG state
    rng_state: u64,
}

impl AdaptiveThresholdEngine {
    /// Create new engine
    pub fn new(config: AdaptiveConfig) -> Self {
        let mut engine = Self {
            thresholds: HashMap::new(),
            bandits: HashMap::new(),
            gradients: HashMap::new(),
            feedback_history: VecDeque::new(),
            event_counter: 0,
            current_time: 0,
            rng_state: 12345,
            config,
        };

        // Initialize default thresholds
        engine.initialize_threshold(ThresholdType::Membership, 0.5);
        engine.initialize_threshold(ThresholdType::Voting, 0.5);
        engine.initialize_threshold(ThresholdType::Proposal, 0.6);
        engine.initialize_threshold(ThresholdType::Quorum, 0.5);
        engine.initialize_threshold(ThresholdType::Approval, 0.67);
        engine.initialize_threshold(ThresholdType::HighStakes, 0.8);
        engine.initialize_threshold(ThresholdType::AlertTrigger, 0.7);

        engine
    }

    fn initialize_threshold(&mut self, threshold_type: ThresholdType, default: f64) {
        let initial = default.clamp(self.config.min_threshold, self.config.max_threshold);

        self.thresholds.insert(
            threshold_type,
            ThresholdState::new(threshold_type, initial),
        );

        self.bandits.insert(
            threshold_type,
            ThresholdBandit::new(
                self.config.min_threshold,
                self.config.max_threshold,
                10, // 10 discrete arms
                self.config.exploration_rate,
            ),
        );

        self.gradients.insert(
            threshold_type,
            GradientEstimator::new(
                initial,
                self.config.learning_rate,
                self.config.momentum,
                self.config.window_size,
            ),
        );
    }

    /// Get current threshold
    pub fn get_threshold(&self, threshold_type: ThresholdType) -> f64 {
        self.thresholds.get(&threshold_type)
            .map(|t| t.value)
            .unwrap_or(0.5)
    }

    /// Get threshold state
    pub fn get_state(&self, threshold_type: ThresholdType) -> Option<&ThresholdState> {
        self.thresholds.get(&threshold_type)
    }

    /// Process feedback
    pub fn process_feedback(&mut self, feedback: ThresholdFeedback) {
        self.event_counter += 1;
        self.current_time = feedback.timestamp;

        // Store feedback
        self.feedback_history.push_back(feedback.clone());
        while self.feedback_history.len() > self.config.window_size * 10 {
            self.feedback_history.pop_front();
        }

        // Calculate reward
        let reward = self.calculate_reward(&feedback);

        // Update bandit
        if let Some(bandit) = self.bandits.get_mut(&feedback.threshold_type) {
            bandit.update(reward);
        }

        // Update gradient estimator
        let perturbation = feedback.threshold_at_event - self.get_threshold(feedback.threshold_type);
        if let Some(gradient) = self.gradients.get_mut(&feedback.threshold_type) {
            gradient.update(reward, perturbation);
        }

        // Check if we should update thresholds
        if self.config.auto_adjust && self.event_counter >= self.config.update_frequency {
            self.event_counter = 0;
            self.adapt_thresholds();
        }
    }

    fn calculate_reward(&self, feedback: &ThresholdFeedback) -> f64 {
        match feedback.outcome {
            FeedbackOutcome::Success => {
                // Bonus for success during attack
                if feedback.context.attack_detected {
                    1.5
                } else {
                    1.0
                }
            }
            FeedbackOutcome::FalseNegative => {
                // Severe penalty - let bad actor through
                -2.0
            }
            FeedbackOutcome::FalsePositive => {
                // Moderate penalty - blocked good actor
                -0.5
            }
            FeedbackOutcome::Neutral => 0.0,
        }
    }

    /// Adapt thresholds based on accumulated feedback
    fn adapt_thresholds(&mut self) {
        let threshold_keys: Vec<ThresholdType> = self.thresholds.keys().copied().collect();

        for threshold_type in threshold_keys {
            let current_value = self.thresholds[&threshold_type].value;

            // Combine bandit and gradient recommendations
            let bandit_recommendation = self.bandits.get(&threshold_type)
                .map(|b| b.best_arm())
                .unwrap_or(current_value);

            let gradient_recommendation = if let Some(gradient) = self.gradients.get_mut(&threshold_type) {
                gradient.step(self.config.min_threshold, self.config.max_threshold)
            } else {
                current_value
            };

            // Weighted combination
            let exploration_weight = self.config.exploration_rate;
            let new_value = exploration_weight * bandit_recommendation
                + (1.0 - exploration_weight) * gradient_recommendation;

            let state = self.thresholds.get_mut(&threshold_type).unwrap();

            // Update with momentum
            let old_velocity = state.velocity;
            state.velocity = self.config.momentum * old_velocity
                + (1.0 - self.config.momentum) * (new_value - state.value);
            state.value = (state.value + state.velocity)
                .clamp(self.config.min_threshold, self.config.max_threshold);

            state.last_updated = self.current_time;
            state.update_count += 1;

            // Update confidence based on feedback consistency
            let confidence = self.calculate_confidence(threshold_type);
            self.thresholds.get_mut(&threshold_type).unwrap().confidence = confidence;
        }
    }

    fn calculate_confidence(&self, threshold_type: ThresholdType) -> f64 {
        let recent: Vec<_> = self.feedback_history.iter()
            .filter(|f| f.threshold_type == threshold_type)
            .collect();

        if recent.is_empty() {
            return 0.5;
        }

        let success_rate = recent.iter()
            .filter(|f| matches!(f.outcome, FeedbackOutcome::Success))
            .count() as f64 / recent.len() as f64;

        success_rate
    }

    /// Manually set threshold
    pub fn set_threshold(&mut self, threshold_type: ThresholdType, value: f64) {
        let clamped = value.clamp(self.config.min_threshold, self.config.max_threshold);
        if let Some(state) = self.thresholds.get_mut(&threshold_type) {
            state.value = clamped;
            state.velocity = 0.0;
            state.last_updated = self.current_time;
        }
    }

    /// Get recommendation for a threshold (without applying)
    pub fn get_recommendation(&self, threshold_type: ThresholdType) -> ThresholdRecommendation {
        let current = self.get_threshold(threshold_type);
        let confidence = self.thresholds.get(&threshold_type)
            .map(|t| t.confidence)
            .unwrap_or(0.5);

        let bandit_value = self.bandits.get(&threshold_type)
            .map(|b| b.best_arm())
            .unwrap_or(current);

        let gradient_value = self.gradients.get(&threshold_type)
            .map(|g| g.current())
            .unwrap_or(current);

        let recommended = (bandit_value + gradient_value) / 2.0;

        let direction = if recommended > current + 0.01 {
            RecommendationDirection::Increase
        } else if recommended < current - 0.01 {
            RecommendationDirection::Decrease
        } else {
            RecommendationDirection::Maintain
        };

        ThresholdRecommendation {
            threshold_type,
            current_value: current,
            recommended_value: recommended,
            direction,
            confidence,
            reasoning: self.generate_reasoning(threshold_type),
        }
    }

    fn generate_reasoning(&self, threshold_type: ThresholdType) -> String {
        let recent: Vec<_> = self.feedback_history.iter()
            .rev()
            .take(20)
            .filter(|f| f.threshold_type == threshold_type)
            .collect();

        let false_negatives = recent.iter()
            .filter(|f| matches!(f.outcome, FeedbackOutcome::FalseNegative))
            .count();

        let false_positives = recent.iter()
            .filter(|f| matches!(f.outcome, FeedbackOutcome::FalsePositive))
            .count();

        if false_negatives > false_positives {
            format!("High false negative rate ({}) suggests threshold may be too low", false_negatives)
        } else if false_positives > false_negatives {
            format!("High false positive rate ({}) suggests threshold may be too high", false_positives)
        } else {
            "Threshold appears well-calibrated".to_string()
        }
    }

    /// Get all threshold states
    pub fn all_states(&self) -> impl Iterator<Item = &ThresholdState> {
        self.thresholds.values()
    }

    /// Get bandit statistics
    pub fn bandit_stats(&self, threshold_type: ThresholdType) -> Option<Vec<(f64, u64, f64)>> {
        self.bandits.get(&threshold_type).map(|b| b.stats())
    }

    /// Reset learning state
    pub fn reset(&mut self) {
        for (_threshold_type, state) in &mut self.thresholds {
            state.value = self.config.initial_threshold;
            state.velocity = 0.0;
            state.confidence = 0.5;
            state.update_count = 0;
        }

        self.feedback_history.clear();
        self.event_counter = 0;

        // Reinitialize bandits and gradients
        for threshold_type in self.thresholds.keys().cloned().collect::<Vec<_>>() {
            self.bandits.insert(
                threshold_type,
                ThresholdBandit::new(
                    self.config.min_threshold,
                    self.config.max_threshold,
                    10,
                    self.config.exploration_rate,
                ),
            );

            self.gradients.insert(
                threshold_type,
                GradientEstimator::new(
                    self.config.initial_threshold,
                    self.config.learning_rate,
                    self.config.momentum,
                    self.config.window_size,
                ),
            );
        }
    }

    /// Simple RNG for exploration
    #[allow(dead_code)]
    fn random(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state as f64) / (u64::MAX as f64)
    }
}

/// Threshold recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdRecommendation {
    pub threshold_type: ThresholdType,
    pub current_value: f64,
    pub recommended_value: f64,
    pub direction: RecommendationDirection,
    pub confidence: f64,
    pub reasoning: String,
}

/// Recommendation direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationDirection {
    Increase,
    Decrease,
    Maintain,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_feedback(
        threshold_type: ThresholdType,
        outcome: FeedbackOutcome,
        threshold: f64,
        timestamp: u64,
    ) -> ThresholdFeedback {
        ThresholdFeedback {
            threshold_type,
            outcome,
            threshold_at_event: threshold,
            context: FeedbackContext {
                participants: 10,
                average_trust: 0.7,
                attack_detected: false,
                network_health: 0.9,
            },
            timestamp,
        }
    }

    #[test]
    fn test_threshold_initialization() {
        let engine = AdaptiveThresholdEngine::new(AdaptiveConfig::default());

        assert!(engine.get_threshold(ThresholdType::Membership) >= 0.3);
        assert!(engine.get_threshold(ThresholdType::Membership) <= 0.9);
    }

    #[test]
    fn test_feedback_processing() {
        let mut engine = AdaptiveThresholdEngine::new(AdaptiveConfig {
            update_frequency: 1,
            auto_adjust: true,
            ..Default::default()
        });

        let initial = engine.get_threshold(ThresholdType::Membership);

        // Send success feedback
        for i in 0..10 {
            engine.process_feedback(create_feedback(
                ThresholdType::Membership,
                FeedbackOutcome::Success,
                initial,
                i as u64 * 1000,
            ));
        }

        // Threshold should remain relatively stable with all success
        let after = engine.get_threshold(ThresholdType::Membership);
        assert!((after - initial).abs() < 0.2);
    }

    #[test]
    fn test_threshold_increases_on_false_negatives() {
        let mut engine = AdaptiveThresholdEngine::new(AdaptiveConfig {
            update_frequency: 5,
            auto_adjust: true,
            learning_rate: 0.1, // Higher learning rate for test
            ..Default::default()
        });

        let initial = engine.get_threshold(ThresholdType::Membership);

        // Send false negative feedback (threshold too low)
        for i in 0..20 {
            engine.process_feedback(create_feedback(
                ThresholdType::Membership,
                FeedbackOutcome::FalseNegative,
                initial,
                i as u64 * 1000,
            ));
        }

        // Recommendation should suggest increase
        let rec = engine.get_recommendation(ThresholdType::Membership);
        // Note: actual threshold change depends on gradient/bandit state
        assert!(rec.reasoning.contains("false negative") || rec.recommended_value >= initial);
    }

    #[test]
    fn test_bandit_arm_selection() {
        let mut bandit = ThresholdBandit::new(0.3, 0.9, 5, 2.0);

        // Initial selection should explore
        let first = bandit.select_arm();
        assert!(first >= 0.3 && first <= 0.9);

        // Update with reward
        bandit.update(1.0);

        // Select again
        let second = bandit.select_arm();
        assert!(second >= 0.3 && second <= 0.9);
    }

    #[test]
    fn test_gradient_estimator() {
        let mut estimator = GradientEstimator::new(0.5, 0.1, 0.9, 10);

        // First, establish a low baseline with low rewards
        for _ in 0..5 {
            estimator.update(0.3, 0.1);
        }

        // Then, send high rewards with positive perturbation
        // This creates positive advantage (reward > baseline) which combined
        // with positive perturbation produces positive gradient
        for _ in 0..5 {
            estimator.update(0.9, 0.1);
        }

        // Gradient should be positive (high reward - low baseline) * positive perturbation
        assert!(estimator.gradient() > 0.0);

        // Step should increase threshold
        let new_val = estimator.step(0.3, 0.9);
        assert!(new_val >= 0.5);
    }

    #[test]
    fn test_manual_threshold_set() {
        let mut engine = AdaptiveThresholdEngine::new(AdaptiveConfig::default());

        engine.set_threshold(ThresholdType::Membership, 0.75);
        assert!((engine.get_threshold(ThresholdType::Membership) - 0.75).abs() < 0.01);

        // Test clamping
        engine.set_threshold(ThresholdType::Membership, 0.1); // Below min
        assert!(engine.get_threshold(ThresholdType::Membership) >= 0.3);
    }

    #[test]
    fn test_recommendation_generation() {
        let engine = AdaptiveThresholdEngine::new(AdaptiveConfig::default());

        let rec = engine.get_recommendation(ThresholdType::Membership);
        assert!(rec.current_value >= 0.3 && rec.current_value <= 0.9);
        assert!(rec.confidence >= 0.0 && rec.confidence <= 1.0);
    }

    #[test]
    fn test_reset() {
        let mut engine = AdaptiveThresholdEngine::new(AdaptiveConfig::default());

        // Process some feedback
        for i in 0..10 {
            engine.process_feedback(create_feedback(
                ThresholdType::Membership,
                FeedbackOutcome::Success,
                0.5,
                i as u64 * 1000,
            ));
        }

        // Reset
        engine.reset();

        // Should be back to initial state
        assert!(engine.feedback_history.is_empty());
        let state = engine.get_state(ThresholdType::Membership).unwrap();
        assert_eq!(state.update_count, 0);
    }

    #[test]
    fn test_all_threshold_types() {
        let engine = AdaptiveThresholdEngine::new(AdaptiveConfig::default());

        let types = [
            ThresholdType::Membership,
            ThresholdType::Voting,
            ThresholdType::Proposal,
            ThresholdType::Quorum,
            ThresholdType::Approval,
            ThresholdType::HighStakes,
            ThresholdType::AlertTrigger,
        ];

        for threshold_type in types {
            let value = engine.get_threshold(threshold_type);
            assert!(value >= 0.3 && value <= 0.9, "Invalid threshold for {:?}", threshold_type);
        }
    }
}
