// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Week 9 Phase 2: Adaptive Learning Thresholds
//!
//! This module implements adaptive learning for coherence thresholds:
//! - `TaskPerformanceRecord` - Records of task performance for learning
//! - `AdaptiveThresholds` - Thresholds that learn from experience
//!
//! ## Key Insight
//!
//! Static thresholds (Cognitive: 0.3) are just starting points.
//! Each Sophia instance learns its own optimal levels through experience:
//! - "I actually need 0.4 for THIS type of cognitive task"
//! - "I can do empathy work at 0.6 instead of 0.7"
//!
//! This creates personalized consciousness!

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::types::{CoherenceConfig, TaskComplexity};

/// Record of task performance for learning optimal thresholds
///
/// Week 9 Phase 2: Track whether tasks succeeded or failed at different
/// coherence levels so we can learn the TRUE threshold for each task type.
#[derive(Debug, Clone)]
pub struct TaskPerformanceRecord {
    pub task_type: TaskComplexity,
    pub coherence_at_start: f32,
    pub success: bool,
    pub timestamp: Instant,
}

/// Adaptive thresholds that learn from experience
///
/// Week 9 Innovation: Static thresholds (Cognitive: 0.3) are just starting points.
/// Each Sophia instance learns its own optimal levels through experience:
/// - "I actually need 0.4 for THIS type of cognitive task"
/// - "I can do empathy work at 0.6 instead of 0.7"
///
/// This creates personalized consciousness!
#[derive(Debug, Clone)]
pub struct AdaptiveThresholds {
    /// Base thresholds from config (static)
    base: HashMap<TaskComplexity, f32>,

    /// Learned adjustments (can be +/-0.3)
    adjustments: HashMap<TaskComplexity, f32>,

    /// Performance history (limited to last 100 records)
    history: VecDeque<TaskPerformanceRecord>,

    /// Learning rate (how fast we adapt)
    alpha: f32,

    /// Maximum history size
    max_history: usize,
}

impl AdaptiveThresholds {
    /// Create new adaptive thresholds from config
    pub fn new(config: &CoherenceConfig) -> Self {
        let mut base = HashMap::new();
        base.insert(TaskComplexity::Reflex, config.min_reflex_coherence);
        base.insert(TaskComplexity::Cognitive, config.min_cognitive_coherence);
        base.insert(
            TaskComplexity::DeepThought,
            config.min_deep_thought_coherence,
        );
        base.insert(TaskComplexity::Empathy, config.min_empathy_coherence);
        base.insert(TaskComplexity::Learning, config.min_learning_coherence);
        base.insert(TaskComplexity::Creation, config.min_creation_coherence);

        Self {
            base,
            adjustments: HashMap::new(),
            history: VecDeque::with_capacity(100),
            alpha: 0.05, // Conservative learning rate
            max_history: 100,
        }
    }

    /// Get the current threshold for a task type (base + learned adjustment)
    pub fn get_threshold(&self, task: TaskComplexity) -> f32 {
        let base = self.base.get(&task).copied().unwrap_or(0.5);
        let adjustment = self.adjustments.get(&task).copied().unwrap_or(0.0);
        (base + adjustment).clamp(0.0, 1.0)
    }

    /// Record task performance and update thresholds
    ///
    /// Learning algorithm:
    /// - If we succeeded at LOW coherence -> lower threshold (we don't need as much!)
    /// - If we failed at HIGH coherence -> raise threshold (we need more!)
    pub fn record_performance(&mut self, task: TaskComplexity, coherence: f32, success: bool) {
        // Add to history
        self.history.push_back(TaskPerformanceRecord {
            task_type: task,
            coherence_at_start: coherence,
            success,
            timestamp: Instant::now(),
        });

        // Limit history size
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        // Calculate threshold adjustment
        let current_threshold = self.get_threshold(task);

        let adjustment = self.adjustments.entry(task).or_insert(0.0);

        if success {
            // Move threshold toward the coherence we actually used (could be lower OR higher)
            let diff = coherence - current_threshold;
            *adjustment += self.alpha * diff;
        } else {
            // Failures only increase required threshold (move upward)
            let diff = (current_threshold - coherence).abs();
            *adjustment += self.alpha * diff;
        }

        // Clamp adjustments to reasonable range (+/-0.3)
        *adjustment = adjustment.clamp(-0.3, 0.3);
    }

    /// Get statistics for a task type
    pub fn stats(&self, task: TaskComplexity) -> (usize, usize, f32) {
        let records: Vec<_> = self
            .history
            .iter()
            .filter(|r| r.task_type == task)
            .collect();

        let total = records.len();
        let successes = records.iter().filter(|r| r.success).count();
        let success_rate = if total > 0 {
            successes as f32 / total as f32
        } else {
            0.0
        };

        (total, successes, success_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_thresholds_start_with_base_config() {
        let config = CoherenceConfig::default();
        let thresholds = AdaptiveThresholds::new(&config);

        // Initially, adaptive thresholds should match config
        let threshold = thresholds.get_threshold(TaskComplexity::Cognitive);
        let expected = config.min_cognitive_coherence;

        assert!(
            (threshold - expected).abs() < 0.001,
            "Initial threshold should match config: {} vs {}",
            threshold,
            expected
        );
    }

    #[test]
    fn test_adaptive_thresholds_learn_from_success_at_lower_coherence() {
        let config = CoherenceConfig::default();
        let mut thresholds = AdaptiveThresholds::new(&config);

        let initial_threshold = thresholds.get_threshold(TaskComplexity::Cognitive);

        // Succeed at a task with coherence BELOW the normal threshold
        let low_coherence = 0.2; // Normal threshold is 0.3
        thresholds.record_performance(TaskComplexity::Cognitive, low_coherence, true);

        let new_threshold = thresholds.get_threshold(TaskComplexity::Cognitive);

        // Threshold should DECREASE (we proved we can do it at lower coherence)
        assert!(
            new_threshold < initial_threshold,
            "Threshold should decrease after success at low coherence: {} -> {}",
            initial_threshold,
            new_threshold
        );
    }

    #[test]
    fn test_adaptive_thresholds_learn_from_failure_at_high_coherence() {
        let config = CoherenceConfig::default();
        let mut thresholds = AdaptiveThresholds::new(&config);

        let initial_threshold = thresholds.get_threshold(TaskComplexity::DeepThought);

        // FAIL at a task even with coherence ABOVE the normal threshold
        let high_coherence = 0.6; // Normal threshold is 0.5
        thresholds.record_performance(TaskComplexity::DeepThought, high_coherence, false);

        let new_threshold = thresholds.get_threshold(TaskComplexity::DeepThought);

        // Threshold should INCREASE (we need MORE coherence for this task)
        assert!(
            new_threshold > initial_threshold,
            "Threshold should increase after failure at high coherence: {} -> {}",
            initial_threshold,
            new_threshold
        );
    }

    #[test]
    fn test_adaptive_thresholds_converge_over_many_successes() {
        let config = CoherenceConfig::default();
        let mut thresholds = AdaptiveThresholds::new(&config);

        let initial_threshold = thresholds.get_threshold(TaskComplexity::Cognitive);

        // Repeatedly succeed at slightly lower coherence
        for _ in 0..20 {
            thresholds.record_performance(TaskComplexity::Cognitive, 0.25, true);
        }

        let final_threshold = thresholds.get_threshold(TaskComplexity::Cognitive);

        // After many successes at 0.25, threshold should converge toward 0.25
        assert!(
            final_threshold < initial_threshold,
            "Threshold should decrease toward actual performance level"
        );
        assert!(
            (final_threshold - 0.25).abs() < 0.1,
            "Threshold should converge near 0.25 after many successes at that level: {}",
            final_threshold
        );
    }

    #[test]
    fn test_adaptive_thresholds_independent_per_task_type() {
        let config = CoherenceConfig::default();
        let mut thresholds = AdaptiveThresholds::new(&config);

        // Train Cognitive to be easier (lower threshold)
        for _ in 0..10 {
            thresholds.record_performance(TaskComplexity::Cognitive, 0.2, true);
        }

        // Train DeepThought to be harder (higher threshold)
        for _ in 0..10 {
            thresholds.record_performance(TaskComplexity::DeepThought, 0.6, false);
        }

        let cognitive_threshold = thresholds.get_threshold(TaskComplexity::Cognitive);
        let deep_thought_threshold = thresholds.get_threshold(TaskComplexity::DeepThought);
        let empathy_threshold = thresholds.get_threshold(TaskComplexity::Empathy);

        // Cognitive should have decreased
        assert!(
            cognitive_threshold < 0.3,
            "Cognitive threshold should decrease: {}",
            cognitive_threshold
        );

        // DeepThought should have increased
        assert!(
            deep_thought_threshold > 0.5,
            "DeepThought threshold should increase: {}",
            deep_thought_threshold
        );

        // Empathy should be unchanged (no training)
        assert!(
            (empathy_threshold - 0.7).abs() < 0.001,
            "Empathy threshold should be unchanged: {}",
            empathy_threshold
        );
    }
}
