// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ported from fl-aggregator/src/adaptive.rs
//! Adaptive Defense System for Federated Learning.
//!
//! Provides dynamic defense strategy adaptation based on observed Byzantine behavior.
//! Monitors Byzantine detection rates and automatically escalates or de-escalates
//! defense mechanisms through a configurable escalation ladder.
//!
//! # Escalation Ladder (default)
//!
//! 1. **FedAvg** - Fastest, no Byzantine tolerance (baseline)
//! 2. **Median** - Coordinate-wise median, moderate protection
//! 3. **TrimmedMean** - Removes extremes, good Byzantine tolerance
//! 4. **Krum** - Single best gradient selection
//! 5. **MultiKrum** - Average of k best gradients (strongest protection)

use crate::types::AggregationMethod;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for the adaptive defense system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveDefenseConfig {
    /// Starting defense strategy.
    pub initial_method: AggregationMethod,

    /// Threshold for escalating to a stronger defense (e.g., 0.2 = 20% Byzantine detected).
    pub attack_threshold: f32,

    /// Number of recent rounds to consider when computing Byzantine rate.
    pub adaptation_window: usize,

    /// Defense strategies to escalate through, from weakest to strongest.
    pub escalation_levels: Vec<AggregationMethod>,

    /// Number of consecutive clean rounds required before de-escalation.
    pub deescalation_rounds: usize,

    /// Threshold below which we consider de-escalation (e.g., 0.05 = 5%).
    pub deescalation_threshold: f32,

    /// Minimum rounds to stay at a level before allowing de-escalation.
    pub min_rounds_at_level: usize,

    /// Enable hysteresis to prevent rapid oscillation.
    pub enable_hysteresis: bool,

    /// Hysteresis factor - rate must exceed threshold * factor to escalate.
    pub hysteresis_factor: f32,
}

impl Default for AdaptiveDefenseConfig {
    fn default() -> Self {
        Self {
            initial_method: AggregationMethod::FedAvg,
            attack_threshold: 0.2,
            adaptation_window: 5,
            escalation_levels: vec![
                AggregationMethod::FedAvg,
                AggregationMethod::Median,
                AggregationMethod::TrimmedMean,
                AggregationMethod::Krum,
                AggregationMethod::MultiKrum,
            ],
            deescalation_rounds: 10,
            deescalation_threshold: 0.05,
            min_rounds_at_level: 5,
            enable_hysteresis: true,
            hysteresis_factor: 1.2,
        }
    }
}

impl AdaptiveDefenseConfig {
    /// Set the initial defense strategy.
    pub fn with_initial_method(mut self, method: AggregationMethod) -> Self {
        self.initial_method = method;
        self
    }

    /// Set the attack threshold for escalation.
    pub fn with_attack_threshold(mut self, threshold: f32) -> Self {
        self.attack_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the adaptation window size.
    pub fn with_adaptation_window(mut self, window: usize) -> Self {
        self.adaptation_window = window.max(1);
        self
    }

    /// Set custom escalation levels.
    pub fn with_escalation_levels(mut self, levels: Vec<AggregationMethod>) -> Self {
        self.escalation_levels = levels;
        self
    }

    /// Set de-escalation parameters.
    pub fn with_deescalation(mut self, rounds: usize, threshold: f32) -> Self {
        self.deescalation_rounds = rounds;
        self.deescalation_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set minimum rounds at a level before de-escalation.
    pub fn with_min_rounds_at_level(mut self, rounds: usize) -> Self {
        self.min_rounds_at_level = rounds;
        self
    }

    /// Enable or disable hysteresis.
    pub fn with_hysteresis(mut self, enabled: bool, factor: f32) -> Self {
        self.enable_hysteresis = enabled;
        self.hysteresis_factor = factor.max(1.0);
        self
    }
}

/// Current state of the adaptive defense system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveDefenseState {
    /// Current active defense strategy.
    pub current_method: AggregationMethod,

    /// Current escalation level index (0 = base level).
    pub escalation_level: usize,

    /// Rolling window of recent Byzantine detection rates.
    pub recent_byzantine_rates: VecDeque<f32>,

    /// Rounds spent at current escalation level.
    pub rounds_at_current_level: usize,

    /// Consecutive clean rounds (rate below de-escalation threshold).
    pub consecutive_clean_rounds: usize,

    /// Total escalations performed.
    pub total_escalations: usize,

    /// Total de-escalations performed.
    pub total_deescalations: usize,

    /// Total rounds processed.
    pub total_rounds: usize,
}

impl Default for AdaptiveDefenseState {
    fn default() -> Self {
        Self {
            current_method: AggregationMethod::FedAvg,
            escalation_level: 0,
            recent_byzantine_rates: VecDeque::new(),
            rounds_at_current_level: 0,
            consecutive_clean_rounds: 0,
            total_escalations: 0,
            total_deescalations: 0,
            total_rounds: 0,
        }
    }
}

/// Statistics about the adaptive defense system's behavior.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveStats {
    /// Current defense being used.
    pub current_method: AggregationMethod,
    /// Current escalation level (0 = base).
    pub escalation_level: usize,
    /// Maximum possible escalation level.
    pub max_escalation_level: usize,
    /// Average Byzantine rate over the adaptation window.
    pub avg_byzantine_rate: f32,
    /// Peak Byzantine rate observed in recent window.
    pub peak_byzantine_rate: f32,
    /// Rounds spent at current level.
    pub rounds_at_current_level: usize,
    /// Consecutive clean rounds.
    pub consecutive_clean_rounds: usize,
    /// Total escalations performed.
    pub total_escalations: usize,
    /// Total de-escalations performed.
    pub total_deescalations: usize,
    /// Total rounds processed.
    pub total_rounds: usize,
    /// Whether the system is at maximum defense level.
    pub at_max_defense: bool,
    /// Whether the system is at minimum defense level.
    pub at_min_defense: bool,
}

/// Manager for adaptive defense strategy selection.
pub struct AdaptiveDefenseManager {
    config: AdaptiveDefenseConfig,
    state: AdaptiveDefenseState,
}

impl AdaptiveDefenseManager {
    /// Create a new adaptive defense manager.
    pub fn new(config: AdaptiveDefenseConfig) -> Self {
        let initial_level = config
            .escalation_levels
            .iter()
            .position(|d| *d == config.initial_method)
            .unwrap_or(0);

        let initial_method = config
            .escalation_levels
            .get(initial_level)
            .copied()
            .unwrap_or(AggregationMethod::FedAvg);

        let state = AdaptiveDefenseState {
            current_method: initial_method,
            escalation_level: initial_level,
            recent_byzantine_rates: VecDeque::with_capacity(config.adaptation_window),
            ..Default::default()
        };

        Self { config, state }
    }

    /// Record the outcome of a round.
    pub fn record_round_result(&mut self, byzantine_count: usize, total_nodes: usize) {
        let rate = if total_nodes == 0 {
            0.0
        } else {
            (byzantine_count as f32 / total_nodes as f32).clamp(0.0, 1.0)
        };

        if self.state.recent_byzantine_rates.len() >= self.config.adaptation_window {
            self.state.recent_byzantine_rates.pop_front();
        }
        self.state.recent_byzantine_rates.push_back(rate);

        self.state.rounds_at_current_level += 1;
        self.state.total_rounds += 1;

        if rate <= self.config.deescalation_threshold {
            self.state.consecutive_clean_rounds += 1;
        } else {
            self.state.consecutive_clean_rounds = 0;
        }
    }

    /// Check if the defense should be escalated.
    pub fn should_escalate(&self) -> bool {
        if self.state.escalation_level >= self.config.escalation_levels.len().saturating_sub(1) {
            return false;
        }
        if self.state.recent_byzantine_rates.is_empty() {
            return false;
        }

        let avg_rate = self.average_byzantine_rate();
        let effective_threshold = if self.config.enable_hysteresis {
            self.config.attack_threshold * self.config.hysteresis_factor
        } else {
            self.config.attack_threshold
        };

        avg_rate > effective_threshold
    }

    /// Check if the defense can be de-escalated.
    pub fn should_deescalate(&self) -> bool {
        if self.state.escalation_level == 0 {
            return false;
        }
        if self.state.rounds_at_current_level < self.config.min_rounds_at_level {
            return false;
        }
        if self.state.consecutive_clean_rounds < self.config.deescalation_rounds {
            return false;
        }
        self.average_byzantine_rate() <= self.config.deescalation_threshold
    }

    /// Adapt the defense strategy. Returns `Some(method)` if changed.
    pub fn adapt(&mut self) -> Option<AggregationMethod> {
        if self.should_escalate() {
            self.escalate()
        } else if self.should_deescalate() {
            self.deescalate()
        } else {
            None
        }
    }

    /// Get the current defense strategy.
    pub fn current_method(&self) -> AggregationMethod {
        self.state.current_method
    }

    /// Get statistics about the adaptive defense system.
    pub fn get_stats(&self) -> AdaptiveStats {
        let max_level = self.config.escalation_levels.len().saturating_sub(1);

        AdaptiveStats {
            current_method: self.state.current_method,
            escalation_level: self.state.escalation_level,
            max_escalation_level: max_level,
            avg_byzantine_rate: self.average_byzantine_rate(),
            peak_byzantine_rate: self.peak_byzantine_rate(),
            rounds_at_current_level: self.state.rounds_at_current_level,
            consecutive_clean_rounds: self.state.consecutive_clean_rounds,
            total_escalations: self.state.total_escalations,
            total_deescalations: self.state.total_deescalations,
            total_rounds: self.state.total_rounds,
            at_max_defense: self.state.escalation_level >= max_level,
            at_min_defense: self.state.escalation_level == 0,
        }
    }

    /// Get the current state (for serialization/persistence).
    pub fn get_state(&self) -> &AdaptiveDefenseState {
        &self.state
    }

    /// Restore from a saved state.
    pub fn restore_state(&mut self, state: AdaptiveDefenseState) {
        self.state = state;
    }

    /// Reset the manager to initial state.
    pub fn reset(&mut self) {
        let initial_level = self
            .config
            .escalation_levels
            .iter()
            .position(|d| *d == self.config.initial_method)
            .unwrap_or(0);

        let initial_method = self
            .config
            .escalation_levels
            .get(initial_level)
            .copied()
            .unwrap_or(AggregationMethod::FedAvg);

        self.state = AdaptiveDefenseState {
            current_method: initial_method,
            escalation_level: initial_level,
            recent_byzantine_rates: VecDeque::with_capacity(self.config.adaptation_window),
            ..Default::default()
        };
    }

    /// Get the configuration.
    pub fn config(&self) -> &AdaptiveDefenseConfig {
        &self.config
    }

    fn average_byzantine_rate(&self) -> f32 {
        if self.state.recent_byzantine_rates.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.state.recent_byzantine_rates.iter().sum();
        sum / self.state.recent_byzantine_rates.len() as f32
    }

    fn peak_byzantine_rate(&self) -> f32 {
        self.state
            .recent_byzantine_rates
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
    }

    fn escalate(&mut self) -> Option<AggregationMethod> {
        let next_level = self.state.escalation_level + 1;
        if next_level >= self.config.escalation_levels.len() {
            return None;
        }

        let new_method = self.config.escalation_levels[next_level];
        self.state.escalation_level = next_level;
        self.state.current_method = new_method;
        self.state.rounds_at_current_level = 0;
        self.state.total_escalations += 1;

        Some(new_method)
    }

    fn deescalate(&mut self) -> Option<AggregationMethod> {
        if self.state.escalation_level == 0 {
            return None;
        }

        let prev_level = self.state.escalation_level - 1;
        let new_method = self.config.escalation_levels[prev_level];

        self.state.escalation_level = prev_level;
        self.state.current_method = new_method;
        self.state.rounds_at_current_level = 0;
        self.state.consecutive_clean_rounds = 0;
        self.state.total_deescalations += 1;

        Some(new_method)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AdaptiveDefenseConfig {
        AdaptiveDefenseConfig::default()
            .with_attack_threshold(0.2)
            .with_adaptation_window(3)
            .with_deescalation(5, 0.05)
            .with_min_rounds_at_level(3)
            .with_hysteresis(false, 1.0)
    }

    #[test]
    fn test_new_manager() {
        let config = test_config();
        let manager = AdaptiveDefenseManager::new(config);
        assert_eq!(manager.current_method(), AggregationMethod::FedAvg);
        assert_eq!(manager.state.escalation_level, 0);
        assert_eq!(manager.state.total_rounds, 0);
    }

    #[test]
    fn test_record_round_result() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);
        manager.record_round_result(2, 10);
        assert_eq!(manager.state.total_rounds, 1);
        assert_eq!(manager.state.rounds_at_current_level, 1);
        assert_eq!(manager.state.recent_byzantine_rates.len(), 1);
        assert!((manager.state.recent_byzantine_rates[0] - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_escalation_when_byzantine_rate_high() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);
        manager.record_round_result(3, 10);
        manager.record_round_result(3, 10);
        manager.record_round_result(4, 10);
        assert!(manager.should_escalate());

        let new = manager.adapt();
        assert!(new.is_some());
        assert_eq!(manager.current_method(), AggregationMethod::Median);
        assert_eq!(manager.state.escalation_level, 1);
        assert_eq!(manager.state.total_escalations, 1);
    }

    #[test]
    fn test_deescalation_after_clean_rounds() {
        let mut config = test_config();
        config.min_rounds_at_level = 2;
        config.deescalation_rounds = 3;
        let mut manager = AdaptiveDefenseManager::new(config);

        // Escalate first
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.adapt();
        assert_eq!(manager.state.escalation_level, 1);

        // Clean rounds
        manager.record_round_result(0, 10);
        manager.record_round_result(0, 10);
        assert!(!manager.should_deescalate()); // Not enough rounds at level

        manager.record_round_result(0, 10);
        assert!(manager.should_deescalate());

        let new = manager.adapt();
        assert!(new.is_some());
        assert_eq!(manager.current_method(), AggregationMethod::FedAvg);
        assert_eq!(manager.state.total_deescalations, 1);
    }

    #[test]
    fn test_no_oscillation_with_min_rounds() {
        let mut config = test_config();
        config.min_rounds_at_level = 5;
        config.deescalation_rounds = 3;
        let mut manager = AdaptiveDefenseManager::new(config);

        // Escalate
        for _ in 0..3 {
            manager.record_round_result(3, 10);
        }
        manager.adapt();
        assert_eq!(manager.state.escalation_level, 1);

        // Try to de-escalate immediately
        for _ in 0..3 {
            manager.record_round_result(0, 10);
        }
        assert!(!manager.should_deescalate());
        assert!(manager.adapt().is_none());

        for _ in 0..2 {
            manager.record_round_result(0, 10);
        }
        assert!(manager.should_deescalate());
    }

    #[test]
    fn test_cannot_escalate_beyond_max() {
        let config = AdaptiveDefenseConfig {
            initial_method: AggregationMethod::MultiKrum,
            attack_threshold: 0.2,
            adaptation_window: 3,
            escalation_levels: vec![
                AggregationMethod::FedAvg,
                AggregationMethod::Median,
                AggregationMethod::MultiKrum,
            ],
            deescalation_rounds: 5,
            deescalation_threshold: 0.05,
            min_rounds_at_level: 3,
            enable_hysteresis: false,
            hysteresis_factor: 1.0,
        };

        let mut manager = AdaptiveDefenseManager::new(config);
        assert_eq!(manager.state.escalation_level, 2);

        for _ in 0..3 {
            manager.record_round_result(5, 10);
        }
        assert!(!manager.should_escalate());
        assert!(manager.adapt().is_none());
    }

    #[test]
    fn test_cannot_deescalate_below_min() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);
        assert_eq!(manager.state.escalation_level, 0);

        for _ in 0..10 {
            manager.record_round_result(0, 10);
        }
        assert!(!manager.should_deescalate());
    }

    #[test]
    fn test_empty_round_handling() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);
        manager.record_round_result(0, 0);
        assert_eq!(manager.state.recent_byzantine_rates[0], 0.0);
    }

    #[test]
    fn test_100_percent_byzantine() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        manager.record_round_result(10, 10);
        manager.record_round_result(10, 10);
        manager.record_round_result(10, 10);
        assert!(manager.should_escalate());

        while manager.adapt().is_some() {
            manager.record_round_result(10, 10);
        }

        let stats = manager.get_stats();
        assert!(stats.at_max_defense);
        assert_eq!(stats.escalation_level, stats.max_escalation_level);
    }

    #[test]
    fn test_rolling_window() {
        let mut config = test_config();
        config.adaptation_window = 3;
        let mut manager = AdaptiveDefenseManager::new(config);

        manager.record_round_result(1, 10);
        manager.record_round_result(2, 10);
        manager.record_round_result(3, 10);
        assert_eq!(manager.state.recent_byzantine_rates.len(), 3);

        manager.record_round_result(4, 10);
        assert_eq!(manager.state.recent_byzantine_rates.len(), 3);
        assert!((manager.state.recent_byzantine_rates[0] - 0.2).abs() < 0.001);
        assert!((manager.state.recent_byzantine_rates[2] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_hysteresis() {
        let mut config = test_config();
        config.attack_threshold = 0.2;
        config.enable_hysteresis = true;
        config.hysteresis_factor = 1.5;
        let mut manager = AdaptiveDefenseManager::new(config);

        // 25% - above threshold but below hysteresis threshold (30%)
        manager.record_round_result(25, 100);
        manager.record_round_result(25, 100);
        manager.record_round_result(25, 100);
        assert!(!manager.should_escalate());

        // 35% - above hysteresis threshold
        manager.record_round_result(35, 100);
        manager.record_round_result(35, 100);
        manager.record_round_result(35, 100);
        assert!(manager.should_escalate());
    }

    #[test]
    fn test_get_stats() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        manager.record_round_result(3, 10);
        manager.record_round_result(4, 10);
        manager.record_round_result(2, 10);

        let stats = manager.get_stats();
        assert_eq!(stats.current_method, AggregationMethod::FedAvg);
        assert_eq!(stats.escalation_level, 0);
        assert!((stats.avg_byzantine_rate - 0.3).abs() < 0.001);
        assert!((stats.peak_byzantine_rate - 0.4).abs() < 0.001);
        assert_eq!(stats.total_rounds, 3);
        assert!(stats.at_min_defense);
        assert!(!stats.at_max_defense);
    }

    #[test]
    fn test_reset() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.adapt();
        assert_eq!(manager.state.escalation_level, 1);

        manager.reset();
        assert_eq!(manager.state.escalation_level, 0);
        assert_eq!(manager.state.total_rounds, 0);
        assert_eq!(manager.current_method(), AggregationMethod::FedAvg);
        assert!(manager.state.recent_byzantine_rates.is_empty());
    }

    #[test]
    fn test_state_persistence() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config.clone());

        manager.record_round_result(3, 10);
        manager.record_round_result(3, 10);
        manager.record_round_result(3, 10);
        manager.adapt();

        let saved_state = manager.get_state().clone();

        let mut new_manager = AdaptiveDefenseManager::new(config);
        new_manager.restore_state(saved_state);

        assert_eq!(new_manager.state.escalation_level, 1);
        assert_eq!(new_manager.state.total_rounds, 3);
        assert_eq!(new_manager.current_method(), AggregationMethod::Median);
    }

    #[test]
    fn test_consecutive_clean_rounds_reset() {
        let mut config = test_config();
        config.deescalation_rounds = 5;
        let mut manager = AdaptiveDefenseManager::new(config);

        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.adapt();

        manager.record_round_result(0, 10);
        manager.record_round_result(0, 10);
        manager.record_round_result(0, 10);
        assert_eq!(manager.state.consecutive_clean_rounds, 3);

        manager.record_round_result(2, 10); // 20% resets counter
        assert_eq!(manager.state.consecutive_clean_rounds, 0);
    }

    #[test]
    fn test_progressive_escalation() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        let expected = vec![
            AggregationMethod::FedAvg,
            AggregationMethod::Median,
            AggregationMethod::TrimmedMean,
            AggregationMethod::Krum,
            AggregationMethod::MultiKrum,
        ];

        for (i, expected_method) in expected.iter().enumerate() {
            assert_eq!(manager.current_method(), *expected_method);
            assert_eq!(manager.state.escalation_level, i);

            manager.record_round_result(5, 10);
            manager.record_round_result(5, 10);
            manager.record_round_result(5, 10);
            manager.adapt();
        }

        let stats = manager.get_stats();
        assert!(stats.at_max_defense);
        assert_eq!(stats.total_escalations, 4);
    }

    #[test]
    fn test_config_builders() {
        let config = AdaptiveDefenseConfig::default()
            .with_initial_method(AggregationMethod::Median)
            .with_attack_threshold(0.3)
            .with_adaptation_window(10)
            .with_deescalation(15, 0.1)
            .with_min_rounds_at_level(7)
            .with_hysteresis(true, 1.3);

        assert_eq!(config.initial_method, AggregationMethod::Median);
        assert!((config.attack_threshold - 0.3).abs() < 0.001);
        assert_eq!(config.adaptation_window, 10);
        assert_eq!(config.deescalation_rounds, 15);
        assert!((config.deescalation_threshold - 0.1).abs() < 0.001);
        assert_eq!(config.min_rounds_at_level, 7);
        assert!(config.enable_hysteresis);
        assert!((config.hysteresis_factor - 1.3).abs() < 0.001);
    }

    #[test]
    fn test_boundary_thresholds() {
        let config = AdaptiveDefenseConfig::default()
            .with_attack_threshold(1.5) // clamped to 1.0
            .with_deescalation(5, -0.1); // clamped to 0.0

        assert!((config.attack_threshold - 1.0).abs() < 0.001);
        assert!((config.deescalation_threshold - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_custom_escalation_levels() {
        let config = AdaptiveDefenseConfig::default()
            .with_escalation_levels(vec![
                AggregationMethod::Median,
                AggregationMethod::Krum,
                AggregationMethod::MultiKrum,
            ])
            .with_initial_method(AggregationMethod::Median);

        let mut manager = AdaptiveDefenseManager::new(config);
        assert_eq!(manager.current_method(), AggregationMethod::Median);
        assert_eq!(manager.state.escalation_level, 0);

        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.adapt();

        assert_eq!(manager.current_method(), AggregationMethod::Krum);
    }

    #[test]
    fn test_initial_method_not_in_levels() {
        let config = AdaptiveDefenseConfig {
            initial_method: AggregationMethod::GeometricMedian,
            escalation_levels: vec![AggregationMethod::FedAvg, AggregationMethod::Median],
            ..Default::default()
        };

        let manager = AdaptiveDefenseManager::new(config);
        assert_eq!(manager.state.escalation_level, 0);
        assert_eq!(manager.current_method(), AggregationMethod::FedAvg);
    }
}
