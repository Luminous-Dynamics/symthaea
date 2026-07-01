// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Surprise-Driven Exploration using Free Energy Principle
//!
//! This module implements exploration strategies based on prediction errors (surprise)
//! following the Free Energy Principle (FEP). When predictions consistently fail,
//! the system triggers exploration to reduce uncertainty.
//!
//! ## Core Concepts
//!
//! - **Surprise**: Divergence between predicted and actual states (prediction error)
//! - **Adaptive Threshold**: Threshold that adapts to running statistics
//! - **Exploration Actions**: Perturbations to current state when surprise is high
//!
//! ## Mathematical Foundation
//!
//! Surprise is computed as:
//! - **MSE**: Mean Squared Error between predicted and actual
//! - **KL Divergence**: Information-theoretic measure of distribution difference
//!
//! The system explores when surprise exceeds an adaptive threshold:
//! ```text
//! threshold = mean + k * std_dev
//! explore_if surprise > threshold
//! ```
//!
//! ## Integration with Cognitive Loop
//!
//! The `SurpriseTracker` can be integrated into the cognitive loop to:
//! 1. Track prediction errors over time
//! 2. Trigger exploration when predictions are poor
//! 3. Perturb state to seek novel configurations
//! 4. Track which explorations reduced surprise
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea::exploration::surprise_driven::{SurpriseTracker, SurpriseTrackerConfig};
//!
//! let config = SurpriseTrackerConfig::default();
//! let mut tracker = SurpriseTracker::new(config);
//!
//! // In the cognitive loop:
//! let surprise = tracker.compute_surprise(&predicted, &actual);
//! if tracker.should_explore(surprise) {
//!     let exploration = tracker.generate_exploration_action(&current_state);
//!     // Apply exploration...
//! }
//! ```

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for the surprise tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseTrackerConfig {
    /// Window size for computing running statistics
    pub window_size: usize,

    /// Initial surprise threshold (before adaptive kicks in)
    pub initial_threshold: f64,

    /// Number of standard deviations above mean for adaptive threshold
    pub threshold_sigma: f64,

    /// Minimum threshold (floor)
    pub min_threshold: f64,

    /// Maximum threshold (ceiling)
    pub max_threshold: f64,

    /// Exploration noise scale (how much to perturb state)
    pub exploration_noise_scale: f32,

    /// Exploration decay rate (reduce noise over time if it helps)
    pub exploration_decay: f32,

    /// Minimum exploration noise
    pub min_exploration_noise: f32,

    /// Whether to use KL divergence (true) or MSE (false)
    pub use_kl_divergence: bool,

    /// Small constant to prevent division by zero in KL divergence
    pub epsilon: f64,

    /// Cooldown cycles after exploration before allowing another
    pub exploration_cooldown: usize,

    /// Exploration boost factor when in high-surprise streak
    pub streak_boost_factor: f32,

    /// Minimum streak length to activate boost
    pub min_streak_for_boost: usize,
}

impl Default for SurpriseTrackerConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            initial_threshold: 0.5,
            threshold_sigma: 1.5,
            min_threshold: 0.1,
            max_threshold: 2.0,
            exploration_noise_scale: 0.1,
            exploration_decay: 0.95,
            min_exploration_noise: 0.01,
            use_kl_divergence: false, // MSE by default (more stable)
            epsilon: 1e-10,
            exploration_cooldown: 5,
            streak_boost_factor: 1.5,
            min_streak_for_boost: 3,
        }
    }
}

/// Statistics tracked for surprise history
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SurpriseStats {
    /// Mean surprise over window
    pub mean: f64,

    /// Standard deviation of surprise
    pub std_dev: f64,

    /// Current adaptive threshold
    pub adaptive_threshold: f64,

    /// Number of surprises recorded
    pub count: usize,

    /// Maximum surprise seen
    pub max_surprise: f64,

    /// Minimum surprise seen
    pub min_surprise: f64,

    /// Current streak of high-surprise cycles
    pub high_surprise_streak: usize,

    /// Total exploration triggers
    pub exploration_triggers: usize,

    /// Successful explorations (reduced surprise afterward)
    pub successful_explorations: usize,
}

/// Record of an exploration event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationEvent {
    /// Surprise level that triggered exploration
    pub trigger_surprise: f64,

    /// Surprise level after exploration
    pub post_exploration_surprise: Option<f64>,

    /// Exploration action taken (perturbation vector)
    pub action: Vec<f32>,

    /// Cycle number when exploration occurred
    pub cycle: usize,

    /// Whether exploration was successful (reduced surprise)
    pub successful: Option<bool>,
}

/// Surprise-driven exploration tracker
///
/// Tracks prediction errors over time and triggers exploration when
/// surprise exceeds an adaptive threshold.
#[derive(Debug, Clone)]
pub struct SurpriseTracker {
    /// Configuration
    config: SurpriseTrackerConfig,

    /// History of surprise values (rolling window)
    surprise_history: VecDeque<f64>,

    /// Current statistics
    stats: SurpriseStats,

    /// Current exploration noise scale (adapts over time)
    current_noise_scale: f32,

    /// Cycles since last exploration
    cycles_since_exploration: usize,

    /// Current cycle number
    cycle_count: usize,

    /// History of exploration events
    exploration_history: VecDeque<ExplorationEvent>,

    /// Maximum exploration history size
    max_exploration_history: usize,

    /// Last exploration event (for tracking success)
    pending_exploration: Option<ExplorationEvent>,

    /// Random seed for deterministic testing
    rng_seed: u64,
}

impl SurpriseTracker {
    /// Create a new surprise tracker with default configuration
    pub fn new(config: SurpriseTrackerConfig) -> Self {
        let initial_threshold = config.initial_threshold;
        Self {
            config,
            surprise_history: VecDeque::with_capacity(100),
            stats: SurpriseStats {
                adaptive_threshold: initial_threshold,
                ..Default::default()
            },
            current_noise_scale: 0.1,
            cycles_since_exploration: 0,
            cycle_count: 0,
            exploration_history: VecDeque::with_capacity(100),
            max_exploration_history: 100,
            pending_exploration: None,
            rng_seed: 42,
        }
    }

    /// Create with a specific seed for deterministic behavior
    pub fn with_seed(config: SurpriseTrackerConfig, seed: u64) -> Self {
        let mut tracker = Self::new(config);
        tracker.rng_seed = seed;
        tracker
    }

    /// Compute surprise (prediction error) between predicted and actual states
    ///
    /// # Arguments
    /// * `predicted` - The predicted state vector
    /// * `actual` - The actual observed state vector
    ///
    /// # Returns
    /// Surprise value (higher = more unexpected)
    pub fn compute_surprise(&self, predicted: &[f32], actual: &[f32]) -> f64 {
        if predicted.is_empty() || actual.is_empty() {
            return 0.0;
        }

        if self.config.use_kl_divergence {
            self.compute_kl_divergence(predicted, actual)
        } else {
            self.compute_mse(predicted, actual)
        }
    }

    /// Compute Mean Squared Error between predicted and actual
    fn compute_mse(&self, predicted: &[f32], actual: &[f32]) -> f64 {
        let n = predicted.len().min(actual.len());
        if n == 0 {
            return 0.0;
        }

        let sum: f64 = predicted
            .iter()
            .zip(actual.iter())
            .take(n)
            .map(|(&p, &a)| {
                let diff = (p as f64) - (a as f64);
                diff * diff
            })
            .sum();

        let mse = sum / n as f64;
        if mse.is_finite() {
            mse
        } else {
            self.config.max_threshold
        }
    }

    /// Compute KL divergence (treating vectors as probability distributions)
    ///
    /// KL(P||Q) = sum(P(i) * log(P(i) / Q(i)))
    ///
    /// Note: This normalizes inputs to sum to 1 and adds epsilon for stability
    fn compute_kl_divergence(&self, predicted: &[f32], actual: &[f32]) -> f64 {
        let n = predicted.len().min(actual.len());
        if n == 0 {
            return 0.0;
        }

        let eps = self.config.epsilon;

        // Normalize to probability distributions
        let sum_p: f64 = actual.iter().take(n).map(|&x| (x as f64).abs()).sum();
        let sum_q: f64 = predicted.iter().take(n).map(|&x| (x as f64).abs()).sum();

        if sum_p < eps || sum_q < eps {
            // Degenerate case: use MSE instead
            return self.compute_mse(predicted, actual);
        }

        let kl: f64 = actual
            .iter()
            .zip(predicted.iter())
            .take(n)
            .map(|(&a, &p)| {
                let p_norm = ((a as f64).abs() / sum_p).max(eps);
                let q_norm = ((p as f64).abs() / sum_q).max(eps);
                p_norm * (p_norm / q_norm).ln()
            })
            .sum();

        if kl.is_finite() && kl >= 0.0 {
            kl
        } else {
            self.compute_mse(predicted, actual)
        }
    }

    /// Record a surprise observation and update statistics
    ///
    /// # Arguments
    /// * `surprise` - The surprise value to record
    pub fn record_surprise(&mut self, surprise: f64) {
        self.cycle_count += 1;
        self.cycles_since_exploration += 1;

        // Add to history
        if self.surprise_history.len() >= self.config.window_size {
            self.surprise_history.pop_front();
        }
        self.surprise_history.push_back(surprise);

        // Update statistics
        self.update_stats();

        // Check if this resolves a pending exploration
        if let Some(ref mut pending) = self.pending_exploration {
            pending.post_exploration_surprise = Some(surprise);
            pending.successful = Some(surprise < pending.trigger_surprise);

            if pending.successful == Some(true) {
                self.stats.successful_explorations += 1;
                // Decay noise on success
                self.current_noise_scale = (self.current_noise_scale
                    * self.config.exploration_decay)
                    .max(self.config.min_exploration_noise);
            }

            // Move to history
            let event = pending.clone();
            if self.exploration_history.len() >= self.max_exploration_history {
                self.exploration_history.pop_front();
            }
            self.exploration_history.push_back(event);
            self.pending_exploration = None;
        }

        // Update high-surprise streak
        if surprise > self.stats.adaptive_threshold {
            self.stats.high_surprise_streak += 1;
        } else {
            self.stats.high_surprise_streak = 0;
        }
    }

    /// Update running statistics from history
    fn update_stats(&mut self) {
        let count = self.surprise_history.len();
        self.stats.count = count;

        if count == 0 {
            self.stats.mean = 0.0;
            self.stats.std_dev = 0.0;
            self.stats.adaptive_threshold = self.config.initial_threshold;
            return;
        }

        // Compute mean
        let sum: f64 = self.surprise_history.iter().sum();
        self.stats.mean = sum / count as f64;

        // Compute std dev
        if count > 1 {
            let variance: f64 = self
                .surprise_history
                .iter()
                .map(|&x| {
                    let diff = x - self.stats.mean;
                    diff * diff
                })
                .sum::<f64>()
                / (count - 1) as f64;
            self.stats.std_dev = variance.sqrt();
        } else {
            self.stats.std_dev = 0.0;
        }

        // Update min/max
        for &s in &self.surprise_history {
            if s > self.stats.max_surprise {
                self.stats.max_surprise = s;
            }
            if self.stats.min_surprise == 0.0 || s < self.stats.min_surprise {
                self.stats.min_surprise = s;
            }
        }

        // Compute adaptive threshold
        self.stats.adaptive_threshold = (self.stats.mean
            + self.config.threshold_sigma * self.stats.std_dev)
            .clamp(self.config.min_threshold, self.config.max_threshold);
    }

    /// Determine if exploration should be triggered based on surprise level
    ///
    /// # Arguments
    /// * `surprise` - Current surprise value
    ///
    /// # Returns
    /// true if exploration should occur
    pub fn should_explore(&self, surprise: f64) -> bool {
        // Check cooldown
        if self.cycles_since_exploration < self.config.exploration_cooldown {
            return false;
        }

        // Check threshold
        surprise > self.stats.adaptive_threshold
    }

    /// Generate an exploration action by perturbing the current state
    ///
    /// The perturbation is scaled by the current noise level and can be
    /// boosted if there's a streak of high-surprise cycles.
    ///
    /// # Arguments
    /// * `current_state` - The current state vector to perturb
    ///
    /// # Returns
    /// Exploration action vector (perturbation to apply)
    pub fn generate_exploration_action(&mut self, current_state: &[f32]) -> Vec<f32> {
        self.stats.exploration_triggers += 1;
        self.cycles_since_exploration = 0;

        // Compute noise scale with optional streak boost
        let mut noise_scale = self.current_noise_scale;
        if self.stats.high_surprise_streak >= self.config.min_streak_for_boost {
            noise_scale *= self.config.streak_boost_factor;
        }

        // Generate perturbation using simple LCG for reproducibility
        let mut seed = self.rng_seed.wrapping_add(self.cycle_count as u64);
        let action: Vec<f32> = current_state
            .iter()
            .map(|&s| {
                // Simple LCG: x = (a * x + c) mod m
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = ((seed >> 33) as f32) / (u32::MAX as f32);
                let noise = (r - 0.5) * 2.0 * noise_scale;

                // Clamp perturbation to reasonable range relative to state
                let max_perturb = s.abs().max(0.1) * noise_scale * 2.0;
                noise.clamp(-max_perturb, max_perturb)
            })
            .collect();

        // Record pending exploration
        self.pending_exploration = Some(ExplorationEvent {
            trigger_surprise: self
                .surprise_history
                .back()
                .copied()
                .unwrap_or(self.stats.adaptive_threshold),
            post_exploration_surprise: None,
            action: action.clone(),
            cycle: self.cycle_count,
            successful: None,
        });

        // Increase noise for next time if this exploration was triggered
        // (will be decayed if successful)
        self.current_noise_scale =
            (self.current_noise_scale * 1.1).min(self.config.exploration_noise_scale * 2.0);

        action
    }

    /// Apply exploration action to a state
    ///
    /// # Arguments
    /// * `state` - State to perturb
    /// * `action` - Exploration action (perturbation)
    ///
    /// # Returns
    /// New state with exploration applied
    pub fn apply_exploration(&self, state: &[f32], action: &[f32]) -> Vec<f32> {
        state
            .iter()
            .zip(action.iter())
            .map(|(&s, &a)| s + a)
            .collect()
    }

    /// Get current statistics
    pub fn stats(&self) -> &SurpriseStats {
        &self.stats
    }

    /// Get current adaptive threshold
    pub fn threshold(&self) -> f64 {
        self.stats.adaptive_threshold
    }

    /// Get exploration success rate
    pub fn exploration_success_rate(&self) -> f64 {
        if self.stats.exploration_triggers == 0 {
            return 0.0;
        }
        self.stats.successful_explorations as f64 / self.stats.exploration_triggers as f64
    }

    /// Get exploration history
    pub fn exploration_history(&self) -> &VecDeque<ExplorationEvent> {
        &self.exploration_history
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.surprise_history.clear();
        self.exploration_history.clear();
        self.pending_exploration = None;
        self.stats = SurpriseStats {
            adaptive_threshold: self.config.initial_threshold,
            ..Default::default()
        };
        self.current_noise_scale = self.config.exploration_noise_scale;
        self.cycles_since_exploration = self.config.exploration_cooldown; // Allow immediate exploration
        self.cycle_count = 0;
    }

    /// Get a summary of the tracker state
    pub fn summary(&self) -> SurpriseTrackerSummary {
        SurpriseTrackerSummary {
            mean_surprise: self.stats.mean,
            std_dev_surprise: self.stats.std_dev,
            adaptive_threshold: self.stats.adaptive_threshold,
            high_surprise_streak: self.stats.high_surprise_streak,
            exploration_triggers: self.stats.exploration_triggers,
            successful_explorations: self.stats.successful_explorations,
            exploration_success_rate: self.exploration_success_rate(),
            current_noise_scale: self.current_noise_scale,
            cycle_count: self.cycle_count,
        }
    }
}

/// Summary of surprise tracker state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseTrackerSummary {
    /// Mean surprise over window
    pub mean_surprise: f64,
    /// Standard deviation of surprise
    pub std_dev_surprise: f64,
    /// Current adaptive threshold
    pub adaptive_threshold: f64,
    /// Current high-surprise streak
    pub high_surprise_streak: usize,
    /// Total exploration triggers
    pub exploration_triggers: usize,
    /// Successful explorations
    pub successful_explorations: usize,
    /// Success rate (0.0 to 1.0)
    pub exploration_success_rate: f64,
    /// Current noise scale
    pub current_noise_scale: f32,
    /// Total cycles
    pub cycle_count: usize,
}

// =============================================================================
// INTEGRATION WITH COGNITIVE LOOP
// =============================================================================

/// Bridge for integrating surprise-driven exploration with the cognitive loop
///
/// This struct provides a higher-level interface for using surprise tracking
/// within the main cognitive loop.
#[derive(Debug, Clone)]
pub struct SurpriseExplorationBridge {
    /// The underlying surprise tracker
    tracker: SurpriseTracker,

    /// Whether exploration is currently active
    pub exploring: bool,

    /// Current exploration action (if exploring)
    current_action: Option<Vec<f32>>,

    /// Last predicted state (for computing surprise)
    last_prediction: Option<Vec<f32>>,

    /// Exploration factor for integration with learning rate
    pub exploration_factor: f32,
}

impl SurpriseExplorationBridge {
    /// Create a new bridge with default configuration
    pub fn new() -> Self {
        Self::with_config(SurpriseTrackerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SurpriseTrackerConfig) -> Self {
        Self {
            tracker: SurpriseTracker::new(config),
            exploring: false,
            current_action: None,
            last_prediction: None,
            exploration_factor: 0.0,
        }
    }

    /// Process a cycle with prediction and actual state
    ///
    /// # Arguments
    /// * `predicted` - The predicted state from the model
    /// * `actual` - The actual observed state
    /// * `current_state` - Current state (for generating exploration)
    ///
    /// # Returns
    /// `(surprise, should_explore, exploration_action)`
    pub fn cycle(
        &mut self,
        predicted: &[f32],
        actual: &[f32],
        current_state: &[f32],
    ) -> (f64, bool, Option<Vec<f32>>) {
        // Compute surprise
        let surprise = self.tracker.compute_surprise(predicted, actual);

        // Record
        self.tracker.record_surprise(surprise);

        // Check if should explore
        let should_explore = self.tracker.should_explore(surprise);

        let action = if should_explore {
            self.exploring = true;
            let action = self.tracker.generate_exploration_action(current_state);
            self.current_action = Some(action.clone());

            // Set exploration factor based on surprise intensity
            let threshold = self.tracker.threshold();
            self.exploration_factor = if threshold > 0.0 {
                ((surprise / threshold) as f32 - 1.0).clamp(0.0, 1.0)
            } else {
                0.5
            };

            Some(action)
        } else {
            self.exploring = false;
            self.current_action = None;
            self.exploration_factor = 0.0;
            None
        };

        self.last_prediction = Some(predicted.to_vec());

        (surprise, should_explore, action)
    }

    /// Get the underlying tracker
    pub fn tracker(&self) -> &SurpriseTracker {
        &self.tracker
    }

    /// Get mutable reference to tracker
    pub fn tracker_mut(&mut self) -> &mut SurpriseTracker {
        &mut self.tracker
    }

    /// Get summary
    pub fn summary(&self) -> SurpriseTrackerSummary {
        self.tracker.summary()
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.tracker.reset();
        self.exploring = false;
        self.current_action = None;
        self.last_prediction = None;
        self.exploration_factor = 0.0;
    }
}

impl Default for SurpriseExplorationBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mse() {
        let config = SurpriseTrackerConfig::default();
        let tracker = SurpriseTracker::new(config);

        let predicted = vec![1.0, 2.0, 3.0];
        let actual = vec![1.0, 2.0, 3.0];

        let surprise = tracker.compute_surprise(&predicted, &actual);
        assert!(
            surprise < 0.001,
            "Same vectors should have near-zero surprise"
        );

        let actual_diff = vec![2.0, 3.0, 4.0];
        let surprise_diff = tracker.compute_surprise(&predicted, &actual_diff);
        assert!(
            surprise_diff > 0.5,
            "Different vectors should have high surprise"
        );
    }

    #[test]
    fn test_high_surprise_triggers_exploration() {
        let config = SurpriseTrackerConfig {
            initial_threshold: 0.3,
            exploration_cooldown: 0, // No cooldown for testing
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);

        // Record low surprise values to establish baseline
        for _ in 0..10 {
            tracker.record_surprise(0.1);
        }

        // High surprise should trigger exploration
        let high_surprise = 1.0;
        tracker.record_surprise(high_surprise);
        assert!(
            tracker.should_explore(high_surprise),
            "High surprise should trigger exploration"
        );
    }

    #[test]
    fn test_exploration_reduces_surprise_over_time() {
        let config = SurpriseTrackerConfig {
            initial_threshold: 0.5,
            exploration_cooldown: 0,
            window_size: 20,
            // Use low threshold_sigma so the threshold tracks the mean closely.
            // With the default 1.5, mixing high and low surprise values increases
            // the std_dev which can push the threshold UP even though the mean drops.
            threshold_sigma: 0.5,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);

        // Simulate high surprise period
        for _ in 0..10 {
            tracker.record_surprise(0.8);
        }

        let initial_threshold = tracker.threshold();

        // Trigger exploration
        let state = vec![0.5, 0.5, 0.5];
        let _action = tracker.generate_exploration_action(&state);

        // Simulate successful exploration (lower surprise afterward)
        // Record enough low-surprise values to push out the old high values
        for _ in 0..20 {
            tracker.record_surprise(0.2);
        }

        let final_threshold = tracker.threshold();

        // Threshold should have adapted downward
        assert!(
            final_threshold < initial_threshold,
            "Threshold should decrease when surprise drops: final={:.4} vs initial={:.4}",
            final_threshold,
            initial_threshold
        );
    }

    #[test]
    fn test_no_exploration_when_predictions_good() {
        let config = SurpriseTrackerConfig {
            initial_threshold: 0.5,
            threshold_sigma: 2.0,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);

        // Record consistently low surprise
        for _ in 0..50 {
            tracker.record_surprise(0.1);
        }

        // Should not trigger exploration
        let low_surprise = 0.1;
        assert!(
            !tracker.should_explore(low_surprise),
            "Low surprise should not trigger exploration"
        );
    }

    #[test]
    fn test_exploration_success_tracking() {
        let config = SurpriseTrackerConfig {
            initial_threshold: 0.3,
            exploration_cooldown: 0,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);

        // Record some baseline
        for _ in 0..5 {
            tracker.record_surprise(0.2);
        }

        // High surprise triggers exploration
        tracker.record_surprise(0.8);
        let state = vec![0.5, 0.5];
        tracker.generate_exploration_action(&state);

        // Record lower surprise (successful exploration)
        tracker.record_surprise(0.3);

        // Check success was recorded
        assert_eq!(
            tracker.stats().successful_explorations,
            1,
            "Should track successful exploration"
        );
    }

    #[test]
    fn test_adaptive_threshold() {
        let config = SurpriseTrackerConfig {
            window_size: 10,
            threshold_sigma: 1.0,
            min_threshold: 0.1,
            max_threshold: 2.0,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);

        // Record values with known statistics
        // Mean = 0.5, std_dev ~= 0.158
        let values = vec![0.3, 0.4, 0.5, 0.6, 0.7];
        for v in values {
            tracker.record_surprise(v);
        }

        let threshold = tracker.threshold();
        let mean = tracker.stats().mean;
        let std_dev = tracker.stats().std_dev;

        // Threshold should be mean + 1.0 * std_dev (clamped)
        let expected = (mean + std_dev).clamp(0.1, 2.0);
        assert!(
            (threshold - expected).abs() < 0.01,
            "Adaptive threshold should follow mean + sigma * std_dev"
        );
    }

    #[test]
    fn test_bridge_integration() {
        let config = SurpriseTrackerConfig {
            initial_threshold: 0.3,
            exploration_cooldown: 0,
            // Use low threshold_sigma so the adaptive threshold tracks the mean
            // closely rather than being inflated by std_dev from the variance
            // between the low-surprise and high-surprise cycles.
            threshold_sigma: 0.5,
            ..Default::default()
        };
        let mut bridge = SurpriseExplorationBridge::with_config(config);

        let predicted = vec![0.5, 0.5, 0.5];
        let actual = vec![0.5, 0.5, 0.5];
        let state = vec![0.5, 0.5, 0.5];

        // Low surprise - no exploration
        let (surprise, should_explore, action) = bridge.cycle(&predicted, &actual, &state);
        assert!(surprise < 0.01);
        assert!(!should_explore);
        assert!(action.is_none());

        // High surprise - should explore
        let actual_diff = vec![1.5, 1.5, 1.5];
        let (surprise, should_explore, action) = bridge.cycle(&predicted, &actual_diff, &state);
        assert!(surprise > 0.3);
        assert!(should_explore);
        assert!(action.is_some());
    }

    #[test]
    fn test_kl_divergence() {
        let config = SurpriseTrackerConfig {
            use_kl_divergence: true,
            ..Default::default()
        };
        let tracker = SurpriseTracker::new(config);

        // Similar distributions
        let p = vec![0.3, 0.4, 0.3];
        let q = vec![0.33, 0.34, 0.33];
        let kl_similar = tracker.compute_surprise(&q, &p);
        assert!(kl_similar < 0.1, "Similar distributions should have low KL");

        // Different distributions
        let p_diff = vec![0.9, 0.05, 0.05];
        let kl_diff = tracker.compute_surprise(&q, &p_diff);
        assert!(
            kl_diff > kl_similar,
            "Different distributions should have higher KL"
        );
    }

    #[test]
    fn test_surprise_tracker_config_default() {
        let config = SurpriseTrackerConfig::default();
        assert_eq!(config.window_size, 100);
        assert_eq!(config.initial_threshold, 0.5);
        assert!(!config.use_kl_divergence);
        assert_eq!(config.exploration_cooldown, 5);
    }

    #[test]
    fn test_surprise_tracker_with_seed_deterministic() {
        let config = SurpriseTrackerConfig {
            exploration_cooldown: 0,
            ..Default::default()
        };
        let mut t1 = SurpriseTracker::with_seed(config.clone(), 42);
        let mut t2 = SurpriseTracker::with_seed(config, 42);
        t1.record_surprise(0.8);
        t2.record_surprise(0.8);
        let state = vec![0.5; 10];
        let a1 = t1.generate_exploration_action(&state);
        let a2 = t2.generate_exploration_action(&state);
        assert_eq!(a1, a2, "Same seed should produce same exploration action");
    }

    #[test]
    fn test_compute_surprise_empty_inputs() {
        let tracker = SurpriseTracker::new(SurpriseTrackerConfig::default());
        assert_eq!(tracker.compute_surprise(&[], &[]), 0.0);
        assert_eq!(tracker.compute_surprise(&[1.0], &[]), 0.0);
    }

    #[test]
    fn test_compute_surprise_identical_vectors_zero() {
        let tracker = SurpriseTracker::new(SurpriseTrackerConfig::default());
        let v = vec![0.3, 0.5, 0.7, 0.9];
        let surprise = tracker.compute_surprise(&v, &v);
        assert!(
            surprise < 1e-6,
            "Identical vectors should have ~0 surprise, got {}",
            surprise
        );
    }

    #[test]
    fn test_compute_surprise_different_lengths() {
        let tracker = SurpriseTracker::new(SurpriseTrackerConfig::default());
        let short = vec![1.0, 2.0];
        let long = vec![1.0, 2.0, 3.0, 4.0];
        assert!(tracker.compute_surprise(&short, &long).is_finite());
    }

    #[test]
    fn test_record_surprise_updates_stats() {
        let mut tracker = SurpriseTracker::new(SurpriseTrackerConfig::default());
        assert_eq!(tracker.stats().count, 0);
        tracker.record_surprise(0.5);
        assert_eq!(tracker.stats().count, 1);
        assert!((tracker.stats().mean - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_record_surprise_max_min_tracking() {
        let mut tracker = SurpriseTracker::new(SurpriseTrackerConfig::default());
        tracker.record_surprise(0.3);
        tracker.record_surprise(0.8);
        tracker.record_surprise(0.5);
        assert!((tracker.stats().max_surprise - 0.8).abs() < 1e-10);
        assert!((tracker.stats().min_surprise - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_should_explore_respects_cooldown() {
        let config = SurpriseTrackerConfig {
            initial_threshold: 0.3,
            exploration_cooldown: 5,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);
        for _ in 0..5 {
            tracker.record_surprise(0.1);
        }
        tracker.record_surprise(0.9);
        let state = vec![0.5; 4];
        tracker.generate_exploration_action(&state);
        assert!(
            !tracker.should_explore(1.0),
            "Should not explore during cooldown"
        );
        for _ in 0..5 {
            tracker.record_surprise(0.1);
        }
        assert!(tracker.should_explore(1.0), "Should explore after cooldown");
    }

    #[test]
    fn test_generate_exploration_action_length_matches_state() {
        let config = SurpriseTrackerConfig {
            exploration_cooldown: 0,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);
        tracker.record_surprise(0.8);
        let state = vec![0.5; 20];
        let action = tracker.generate_exploration_action(&state);
        assert_eq!(action.len(), 20);
    }

    #[test]
    fn test_exploration_action_finite_values() {
        let config = SurpriseTrackerConfig {
            exploration_cooldown: 0,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);
        tracker.record_surprise(0.8);
        let action = tracker.generate_exploration_action(&vec![0.5; 10]);
        for &v in &action {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_apply_exploration_produces_perturbed_state() {
        let tracker = SurpriseTracker::new(SurpriseTrackerConfig::default());
        let state = vec![0.5, 0.5, 0.5];
        let action = vec![0.1, -0.1, 0.05];
        let result = tracker.apply_exploration(&state, &action);
        assert!((result[0] - 0.6).abs() < 1e-6);
        assert!((result[1] - 0.4).abs() < 1e-6);
        assert!((result[2] - 0.55).abs() < 1e-6);
    }

    #[test]
    fn test_exploration_success_rate_no_explorations() {
        let tracker = SurpriseTracker::new(SurpriseTrackerConfig::default());
        assert_eq!(tracker.exploration_success_rate(), 0.0);
    }

    #[test]
    fn test_high_surprise_streak_tracking() {
        // Use threshold_sigma=2.0 with varied low values so the adaptive threshold
        // (mean + 2*std) is clearly above all low values, avoiding FP boundary
        // issues that arise when all values are identical and threshold_sigma=0.
        let config = SurpriseTrackerConfig {
            initial_threshold: 0.5,
            threshold_sigma: 2.0,
            min_threshold: 0.01,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);
        // Low values 0.05..0.14: mean~0.095, std~0.030, threshold~0.156
        for i in 0..10 {
            tracker.record_surprise(0.05 + (i as f64) * 0.01);
        }
        assert_eq!(tracker.stats().high_surprise_streak, 0);
        // 0.9 clearly exceeds threshold
        tracker.record_surprise(0.9);
        assert_eq!(tracker.stats().high_surprise_streak, 1);
        tracker.record_surprise(0.9);
        assert_eq!(tracker.stats().high_surprise_streak, 2);
        // Low value resets streak
        tracker.record_surprise(0.01);
        assert_eq!(tracker.stats().high_surprise_streak, 0);
    }

    #[test]
    fn test_tracker_reset() {
        let config = SurpriseTrackerConfig {
            exploration_cooldown: 0,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);
        for _ in 0..20 {
            tracker.record_surprise(0.5);
        }
        tracker.generate_exploration_action(&vec![0.5; 4]);
        tracker.reset();
        assert_eq!(tracker.stats().count, 0);
        assert_eq!(tracker.stats().exploration_triggers, 0);
    }

    #[test]
    fn test_summary_fields() {
        let mut tracker = SurpriseTracker::new(SurpriseTrackerConfig::default());
        tracker.record_surprise(0.5);
        let summary = tracker.summary();
        assert_eq!(summary.cycle_count, 1);
        assert!(summary.mean_surprise > 0.0);
    }

    #[test]
    fn test_bridge_default_state() {
        let bridge = SurpriseExplorationBridge::new();
        assert!(!bridge.exploring);
        assert_eq!(bridge.exploration_factor, 0.0);
    }

    #[test]
    fn test_bridge_reset() {
        let config = SurpriseTrackerConfig {
            exploration_cooldown: 0,
            threshold_sigma: 0.0,
            ..Default::default()
        };
        let mut bridge = SurpriseExplorationBridge::with_config(config);
        bridge.cycle(&vec![0.5; 4], &vec![1.5; 4], &vec![0.5; 4]);
        bridge.reset();
        assert!(!bridge.exploring);
        assert_eq!(bridge.exploration_factor, 0.0);
    }

    #[test]
    fn test_kl_divergence_with_zero_distributions() {
        let config = SurpriseTrackerConfig {
            use_kl_divergence: true,
            ..Default::default()
        };
        let tracker = SurpriseTracker::new(config);
        let surprise = tracker.compute_surprise(&vec![0.0; 3], &vec![1.0; 3]);
        assert!(surprise.is_finite(), "KL with zero dist should be finite");
    }

    #[test]
    fn test_adaptive_threshold_clamped_to_bounds() {
        let config = SurpriseTrackerConfig {
            window_size: 5,
            threshold_sigma: 100.0,
            min_threshold: 0.1,
            max_threshold: 2.0,
            ..Default::default()
        };
        let mut tracker = SurpriseTracker::new(config);
        tracker.record_surprise(0.0);
        tracker.record_surprise(10.0);
        tracker.record_surprise(0.0);
        tracker.record_surprise(10.0);
        let threshold = tracker.threshold();
        assert!(
            threshold <= 2.0 && threshold >= 0.1,
            "Threshold should be clamped: {}",
            threshold
        );
    }
}
