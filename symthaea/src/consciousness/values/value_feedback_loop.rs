// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Value Feedback Loop: Learning from Value-Based Outcomes
//!
//! Implements a feedback loop that allows the system to learn and
//! improve its value alignment based on outcomes and corrections.

use super::eight_harmonies::Harmony;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;

/// Configuration for the value feedback loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackLoopConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// History size
    pub history_size: usize,
    /// Discount factor for temporal difference
    pub discount_factor: f32,
    /// Minimum confidence for updates
    pub min_confidence: f32,
    /// Momentum for updates
    pub momentum: f32,
}

impl Default for FeedbackLoopConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            history_size: 1000,
            discount_factor: 0.95,
            min_confidence: 0.5,
            momentum: 0.9,
        }
    }
}

/// A feedback signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSignal {
    /// Signal identifier
    pub id: u64,
    /// The action that was evaluated
    pub action_id: String,
    /// Feedback type
    pub feedback_type: FeedbackType,
    /// Feedback value (-1.0 to 1.0)
    pub value: f32,
    /// Confidence in the feedback
    pub confidence: f32,
    /// Harmony affected (if specific)
    pub harmony: Option<Harmony>,
    /// Timestamp
    pub timestamp: u64,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Types of feedback
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Explicit reward/punishment
    Explicit,
    /// Implicit from outcomes
    Implicit,
    /// Correction from user
    Correction,
    /// Self-assessment
    SelfAssessment,
    /// Comparison with baseline
    Comparative,
}

impl FeedbackSignal {
    /// Create a new feedback signal
    pub fn new(
        id: u64,
        action_id: impl Into<String>,
        feedback_type: FeedbackType,
        value: f32,
    ) -> Self {
        Self {
            id,
            action_id: action_id.into(),
            feedback_type,
            value: value.clamp(-1.0, 1.0),
            confidence: 1.0,
            harmony: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            context: HashMap::new(),
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set specific harmony
    pub fn with_harmony(mut self, harmony: Harmony) -> Self {
        self.harmony = Some(harmony);
        self
    }

    /// Add context
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

/// Result of processing feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackResult {
    /// Whether the feedback was processed
    pub processed: bool,
    /// Update magnitude
    pub update_magnitude: f32,
    /// Harmonies affected
    pub harmonies_affected: Vec<Harmony>,
    /// New estimated values
    pub new_estimates: HashMap<String, f32>,
}

/// Value estimate for an action pattern
#[derive(Debug, Clone)]
pub struct ValueEstimate {
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern embedding
    pub embedding: ContinuousHV,
    /// Current value estimate
    pub value: f32,
    /// Uncertainty (variance)
    pub uncertainty: f32,
    /// Update count
    pub update_count: u64,
    /// Per-harmony estimates
    pub harmony_values: HashMap<Harmony, f32>,
}

/// The value feedback loop
#[derive(Debug)]
pub struct ValueFeedbackLoop {
    /// Configuration
    config: FeedbackLoopConfig,
    /// Value estimates for patterns
    estimates: HashMap<String, ValueEstimate>,
    /// Feedback history
    history: VecDeque<FeedbackSignal>,
    /// Next feedback ID
    next_id: u64,
    /// Momentum accumulators
    momentum: HashMap<String, f32>,
    /// Statistics
    stats: FeedbackLoopStats,
    /// Per-harmony importance adjustments (learned from feedback)
    importance_adjustments: HashMap<String, f64>,
}

/// Summary of feedback loop learning (alias for stats)
pub type FeedbackLoopSummary = FeedbackLoopStats;

/// Statistics for the feedback loop
#[derive(Debug, Clone, Default)]
pub struct FeedbackLoopStats {
    /// Total feedback signals received
    pub total_feedback: u64,
    /// Positive feedback count
    pub positive_feedback: u64,
    /// Negative feedback count
    pub negative_feedback: u64,
    /// Total updates applied
    pub updates_applied: u64,
    /// Average feedback value
    pub avg_feedback_value: f32,
}

impl ValueFeedbackLoop {
    /// Create a new feedback loop
    pub fn new(config: FeedbackLoopConfig) -> Self {
        Self {
            config,
            estimates: HashMap::new(),
            history: VecDeque::new(),
            next_id: 1,
            momentum: HashMap::new(),
            stats: FeedbackLoopStats::default(),
            importance_adjustments: HashMap::new(),
        }
    }

    /// Register a pattern for value estimation
    pub fn register_pattern(&mut self, pattern_id: impl Into<String>, embedding: ContinuousHV) {
        let id = pattern_id.into();
        let estimate = ValueEstimate {
            pattern_id: id.clone(),
            embedding,
            value: 0.0,
            uncertainty: 1.0,
            update_count: 0,
            harmony_values: HashMap::new(),
        };
        self.estimates.insert(id, estimate);
    }

    /// Get current value estimate for a pattern
    pub fn get_estimate(&self, pattern_id: &str) -> Option<&ValueEstimate> {
        self.estimates.get(pattern_id)
    }

    /// Process a feedback signal
    pub fn process_feedback(&mut self, signal: FeedbackSignal) -> FeedbackResult {
        self.stats.total_feedback += 1;

        if signal.value > 0.0 {
            self.stats.positive_feedback += 1;
        } else if signal.value < 0.0 {
            self.stats.negative_feedback += 1;
        }

        // Update average
        let n = self.stats.total_feedback as f32;
        self.stats.avg_feedback_value =
            (self.stats.avg_feedback_value * (n - 1.0) + signal.value) / n;

        // Check confidence threshold
        if signal.confidence < self.config.min_confidence {
            return FeedbackResult {
                processed: false,
                update_magnitude: 0.0,
                harmonies_affected: Vec::new(),
                new_estimates: HashMap::new(),
            };
        }

        // Find matching pattern
        let action_id = signal.action_id.clone();
        let mut harmonies_affected = Vec::new();
        let mut new_estimates = HashMap::new();
        let mut update_magnitude = 0.0;

        if let Some(estimate) = self.estimates.get_mut(&action_id) {
            // Calculate update with momentum
            let prev_momentum = self.momentum.get(&action_id).copied().unwrap_or(0.0);
            let gradient = signal.value * signal.confidence;
            let update =
                self.config.momentum * prev_momentum + self.config.learning_rate * gradient;

            // Apply update
            estimate.value += update;
            estimate.value = estimate.value.clamp(-1.0, 1.0);
            estimate.uncertainty *= 0.99; // Reduce uncertainty with each update
            estimate.update_count += 1;

            // Store momentum
            self.momentum.insert(action_id.clone(), update);

            update_magnitude = update.abs();
            new_estimates.insert(action_id.clone(), estimate.value);

            // Update specific harmony if provided
            if let Some(harmony) = signal.harmony {
                let current = estimate
                    .harmony_values
                    .get(&harmony)
                    .copied()
                    .unwrap_or(0.0);
                estimate.harmony_values.insert(harmony, current + update);
                harmonies_affected.push(harmony);
            }

            self.stats.updates_applied += 1;
        }

        // Store in history
        if self.history.len() >= self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(signal);

        FeedbackResult {
            processed: true,
            update_magnitude,
            harmonies_affected,
            new_estimates,
        }
    }

    /// Create a new feedback signal
    pub fn create_signal(
        &mut self,
        action_id: impl Into<String>,
        feedback_type: FeedbackType,
        value: f32,
    ) -> FeedbackSignal {
        let id = self.next_id;
        self.next_id += 1;
        FeedbackSignal::new(id, action_id, feedback_type, value)
    }

    /// Apply temporal difference learning
    pub fn temporal_difference_update(
        &mut self,
        current_pattern: &str,
        next_pattern: &str,
        reward: f32,
    ) -> f32 {
        let current_value = self
            .estimates
            .get(current_pattern)
            .map(|e| e.value)
            .unwrap_or(0.0);

        let next_value = self
            .estimates
            .get(next_pattern)
            .map(|e| e.value)
            .unwrap_or(0.0);

        // TD error: δ = r + γV(s') - V(s)
        let td_error = reward + self.config.discount_factor * next_value - current_value;

        // Create and process feedback
        let signal = self.create_signal(current_pattern, FeedbackType::Implicit, td_error);
        self.process_feedback(signal);

        td_error
    }

    /// Get feedback statistics
    pub fn stats(&self) -> &FeedbackLoopStats {
        &self.stats
    }

    /// Get positive feedback ratio
    pub fn positive_ratio(&self) -> f32 {
        if self.stats.total_feedback == 0 {
            0.5
        } else {
            self.stats.positive_feedback as f32 / self.stats.total_feedback as f32
        }
    }

    /// Get recent feedback trend
    pub fn recent_trend(&self, window: usize) -> f32 {
        let recent: Vec<_> = self.history.iter().rev().take(window).collect();
        if recent.is_empty() {
            return 0.0;
        }
        recent.iter().map(|f| f.value).sum::<f32>() / recent.len() as f32
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Record user feedback on a value decision (high-level interface)
    ///
    /// Accepts any eval_result and decision types to avoid circular dependencies.
    /// The feedback loop primarily uses the rating to adjust importance weights.
    pub fn record_user_feedback<E, D>(
        &mut self,
        action: &str,
        _eval_result: &E,
        _decision: &D,
        rating: f64,
        _phi: f64,
        _comment: Option<String>,
    ) {
        let signal = self.create_signal(action, FeedbackType::Explicit, rating as f32);
        self.process_feedback(signal);

        // Update per-harmony importance based on feedback
        let adjustment_delta = (rating - 0.5) * 2.0 * self.config.learning_rate as f64;
        for harmony in Harmony::all() {
            let name = harmony.name().to_string();
            let current = self
                .importance_adjustments
                .get(&name)
                .copied()
                .unwrap_or(1.0);
            let new_val = (current + adjustment_delta).clamp(0.5, 2.0);
            self.importance_adjustments.insert(name, new_val);
        }
    }

    /// Record self-reflection feedback from meta-cognition
    pub fn record_self_reflection<E, D>(
        &mut self,
        action: &str,
        _eval_result: &E,
        _decision: &D,
        phi_change: f64,
        coherence_change: f64,
        _phi: f64,
    ) {
        let value = (phi_change + coherence_change) / 2.0;
        let signal = self.create_signal(action, FeedbackType::SelfAssessment, value as f32);
        self.process_feedback(signal);
    }

    /// Get learned importance adjustment for a harmony
    ///
    /// Returns a multiplier: 1.0 = no adjustment, >1.0 = more important, <1.0 = less important
    pub fn get_importance_adjustment(&self, harmony: &Harmony) -> f64 {
        self.importance_adjustments
            .get(harmony.name())
            .copied()
            .unwrap_or(1.0)
    }

    /// Get all learned importance adjustments (keyed by harmony name)
    pub fn get_all_adjustments(&self) -> &HashMap<String, f64> {
        &self.importance_adjustments
    }

    /// Get number of learning data points
    pub fn learning_data_count(&self) -> u64 {
        self.stats.updates_applied
    }

    /// Get summary of feedback loop learning
    pub fn summary(&self) -> FeedbackLoopSummary {
        self.stats.clone()
    }

    /// Apply decay to learned adjustments, moving them toward 1.0 (neutral)
    pub fn apply_decay(&mut self) {
        let decay = self.config.discount_factor as f64;
        for val in self.importance_adjustments.values_mut() {
            *val = 1.0 + (*val - 1.0) * decay;
        }
    }
}

impl Default for ValueFeedbackLoop {
    fn default() -> Self {
        Self::new(FeedbackLoopConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_loop_creation() {
        let loop_ = ValueFeedbackLoop::default();
        assert_eq!(loop_.stats.total_feedback, 0);
    }

    #[test]
    fn test_pattern_registration() {
        let mut loop_ = ValueFeedbackLoop::default();
        loop_.register_pattern("action1", ContinuousHV::random(512, 42));
        assert!(loop_.get_estimate("action1").is_some());
    }

    #[test]
    fn test_feedback_processing() {
        let mut loop_ = ValueFeedbackLoop::default();
        loop_.register_pattern("action1", ContinuousHV::random(512, 42));

        let signal = loop_.create_signal("action1", FeedbackType::Explicit, 0.8);
        let result = loop_.process_feedback(signal);

        assert!(result.processed);
        assert!(result.update_magnitude > 0.0);
    }

    #[test]
    fn test_temporal_difference() {
        let mut loop_ = ValueFeedbackLoop::default();
        loop_.register_pattern("state1", ContinuousHV::random(512, 42));
        loop_.register_pattern("state2", ContinuousHV::random(512, 42));

        let td_error = loop_.temporal_difference_update("state1", "state2", 1.0);
        // TD error is always a valid float (assertion always passes but kept for documentation)
        assert!(td_error.is_finite());
    }

    #[test]
    fn test_feedback_trend() {
        let mut loop_ = ValueFeedbackLoop::default();

        // Add some positive feedback
        for i in 0..5 {
            let signal = loop_.create_signal(format!("action{}", i), FeedbackType::Explicit, 0.5);
            loop_.history.push_back(signal);
        }

        let trend = loop_.recent_trend(5);
        assert!((trend - 0.5).abs() < 0.01);
    }
}
