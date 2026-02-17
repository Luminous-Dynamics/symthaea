//! Adaptive Defense System for Federated Learning.
//!
//! This module provides dynamic defense strategy adaptation based on observed
//! Byzantine behavior. The system monitors Byzantine detection rates and
//! automatically escalates or de-escalates defense mechanisms.
//!
//! # Escalation Ladder
//!
//! The system follows a defense escalation ladder:
//! 1. **FedAvg** - Fastest, no Byzantine tolerance (baseline)
//! 2. **Median** - Coordinate-wise median, moderate protection
//! 3. **TrimmedMean** - Removes extremes, good Byzantine tolerance
//! 4. **Krum** - Single best gradient selection
//! 5. **MultiKrum** - Average of k best gradients (strongest protection)
//!
//! # Example
//!
//! ```rust,ignore
//! use fl_aggregator::adaptive::{AdaptiveDefenseManager, AdaptiveDefenseConfig};
//! use fl_aggregator::Defense;
//!
//! let config = AdaptiveDefenseConfig::default();
//! let mut manager = AdaptiveDefenseManager::new(config);
//!
//! // After each round, record the Byzantine detection result
//! manager.record_round_result(2, 10); // 2 Byzantine out of 10 nodes
//!
//! // Check if we need to adapt
//! if let Some(new_defense) = manager.adapt() {
//!     println!("Switching to defense: {}", new_defense);
//! }
//! ```

use crate::byzantine::Defense;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for the adaptive defense system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveDefenseConfig {
    /// Starting defense strategy.
    pub initial_defense: Defense,

    /// Threshold for escalating to a stronger defense (e.g., 0.2 = 20% Byzantine detected).
    pub attack_threshold: f32,

    /// Number of recent rounds to consider when computing Byzantine rate.
    pub adaptation_window: usize,

    /// Defense strategies to escalate through, from weakest to strongest.
    pub escalation_levels: Vec<Defense>,

    /// Number of consecutive clean rounds required before de-escalation.
    pub deescalation_rounds: usize,

    /// Threshold below which we consider de-escalation (e.g., 0.05 = 5%).
    pub deescalation_threshold: f32,

    /// Minimum rounds to stay at a level before allowing de-escalation.
    pub min_rounds_at_level: usize,

    /// Enable hysteresis to prevent rapid oscillation.
    pub enable_hysteresis: bool,

    /// Hysteresis factor - how much the rate must exceed threshold before escalating.
    /// E.g., 1.2 means rate must be 20% above threshold to trigger escalation.
    pub hysteresis_factor: f32,
}

impl Default for AdaptiveDefenseConfig {
    fn default() -> Self {
        Self {
            initial_defense: Defense::FedAvg,
            attack_threshold: 0.2,
            adaptation_window: 5,
            escalation_levels: vec![
                Defense::FedAvg,
                Defense::Median,
                Defense::TrimmedMean { beta: 0.1 },
                Defense::Krum { f: 1 },
                Defense::MultiKrum { f: 1, k: 3 },
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
    /// Create a new config with custom initial defense.
    pub fn with_initial_defense(mut self, defense: Defense) -> Self {
        self.initial_defense = defense;
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
    pub fn with_escalation_levels(mut self, levels: Vec<Defense>) -> Self {
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
    pub current_defense: Defense,

    /// Current escalation level index (0 = base level).
    pub escalation_level: usize,

    /// Rolling window of recent Byzantine detection rates.
    pub recent_byzantine_rates: VecDeque<f32>,

    /// Number of rounds spent at current escalation level.
    pub rounds_at_current_level: usize,

    /// Number of consecutive clean rounds (rate below de-escalation threshold).
    pub consecutive_clean_rounds: usize,

    /// Total number of escalations performed.
    pub total_escalations: usize,

    /// Total number of de-escalations performed.
    pub total_deescalations: usize,

    /// Total rounds processed.
    pub total_rounds: usize,
}

impl Default for AdaptiveDefenseState {
    fn default() -> Self {
        Self {
            current_defense: Defense::FedAvg,
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
    pub current_defense: String,

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
    /// Create a new adaptive defense manager with the given configuration.
    pub fn new(config: AdaptiveDefenseConfig) -> Self {
        let initial_level = config
            .escalation_levels
            .iter()
            .position(|d| *d == config.initial_defense)
            .unwrap_or(0);

        let initial_defense = config
            .escalation_levels
            .get(initial_level)
            .cloned()
            .unwrap_or(Defense::FedAvg);

        let state = AdaptiveDefenseState {
            current_defense: initial_defense,
            escalation_level: initial_level,
            recent_byzantine_rates: VecDeque::with_capacity(config.adaptation_window),
            ..Default::default()
        };

        Self { config, state }
    }

    /// Record the outcome of a round.
    ///
    /// # Arguments
    /// * `byzantine_count` - Number of nodes detected as Byzantine in this round
    /// * `total_nodes` - Total number of nodes that participated in this round
    pub fn record_round_result(&mut self, byzantine_count: usize, total_nodes: usize) {
        // Handle edge cases
        let rate = if total_nodes == 0 {
            0.0
        } else {
            byzantine_count as f32 / total_nodes as f32
        };

        // Clamp rate to valid range
        let rate = rate.clamp(0.0, 1.0);

        // Add to rolling window
        if self.state.recent_byzantine_rates.len() >= self.config.adaptation_window {
            self.state.recent_byzantine_rates.pop_front();
        }
        self.state.recent_byzantine_rates.push_back(rate);

        // Update state counters
        self.state.rounds_at_current_level += 1;
        self.state.total_rounds += 1;

        // Track consecutive clean rounds
        if rate <= self.config.deescalation_threshold {
            self.state.consecutive_clean_rounds += 1;
        } else {
            self.state.consecutive_clean_rounds = 0;
        }

        tracing::debug!(
            "Recorded round result: {}/{} Byzantine ({:.1}%), level={}, clean_rounds={}",
            byzantine_count,
            total_nodes,
            rate * 100.0,
            self.state.escalation_level,
            self.state.consecutive_clean_rounds
        );
    }

    /// Check if the defense should be escalated to a stronger strategy.
    pub fn should_escalate(&self) -> bool {
        // Can't escalate beyond max level
        if self.state.escalation_level >= self.config.escalation_levels.len().saturating_sub(1) {
            return false;
        }

        // Need at least some data to make a decision
        if self.state.recent_byzantine_rates.is_empty() {
            return false;
        }

        // Calculate average rate
        let avg_rate = self.average_byzantine_rate();

        // Apply hysteresis if enabled
        let effective_threshold = if self.config.enable_hysteresis {
            self.config.attack_threshold * self.config.hysteresis_factor
        } else {
            self.config.attack_threshold
        };

        avg_rate > effective_threshold
    }

    /// Check if the defense can be de-escalated to a weaker (faster) strategy.
    pub fn should_deescalate(&self) -> bool {
        // Can't de-escalate below level 0
        if self.state.escalation_level == 0 {
            return false;
        }

        // Must stay at level for minimum rounds
        if self.state.rounds_at_current_level < self.config.min_rounds_at_level {
            return false;
        }

        // Need enough consecutive clean rounds
        if self.state.consecutive_clean_rounds < self.config.deescalation_rounds {
            return false;
        }

        // Check that the average rate is below threshold
        let avg_rate = self.average_byzantine_rate();
        avg_rate <= self.config.deescalation_threshold
    }

    /// Adapt the defense strategy if needed.
    ///
    /// Returns `Some(Defense)` if the defense was changed, `None` if no change was made.
    pub fn adapt(&mut self) -> Option<Defense> {
        if self.should_escalate() {
            self.escalate()
        } else if self.should_deescalate() {
            self.deescalate()
        } else {
            None
        }
    }

    /// Get the current defense strategy.
    pub fn current_defense(&self) -> &Defense {
        &self.state.current_defense
    }

    /// Get statistics about the adaptive defense system.
    pub fn get_stats(&self) -> AdaptiveStats {
        let max_level = self.config.escalation_levels.len().saturating_sub(1);

        AdaptiveStats {
            current_defense: self.state.current_defense.to_string(),
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
            .position(|d| *d == self.config.initial_defense)
            .unwrap_or(0);

        let initial_defense = self
            .config
            .escalation_levels
            .get(initial_level)
            .cloned()
            .unwrap_or(Defense::FedAvg);

        self.state = AdaptiveDefenseState {
            current_defense: initial_defense,
            escalation_level: initial_level,
            recent_byzantine_rates: VecDeque::with_capacity(self.config.adaptation_window),
            ..Default::default()
        };
    }

    /// Get the configuration.
    pub fn config(&self) -> &AdaptiveDefenseConfig {
        &self.config
    }

    // -------------------------------------------------------------------------
    // Private methods
    // -------------------------------------------------------------------------

    /// Calculate the average Byzantine rate over the adaptation window.
    fn average_byzantine_rate(&self) -> f32 {
        if self.state.recent_byzantine_rates.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.state.recent_byzantine_rates.iter().sum();
        sum / self.state.recent_byzantine_rates.len() as f32
    }

    /// Get the peak Byzantine rate in the recent window.
    fn peak_byzantine_rate(&self) -> f32 {
        self.state
            .recent_byzantine_rates
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
    }

    /// Escalate to the next defense level.
    fn escalate(&mut self) -> Option<Defense> {
        let next_level = self.state.escalation_level + 1;
        if next_level >= self.config.escalation_levels.len() {
            return None;
        }

        let new_defense = self.config.escalation_levels[next_level].clone();

        tracing::info!(
            "Escalating defense: {} -> {} (level {} -> {}), avg_rate={:.1}%",
            self.state.current_defense,
            new_defense,
            self.state.escalation_level,
            next_level,
            self.average_byzantine_rate() * 100.0
        );

        self.state.escalation_level = next_level;
        self.state.current_defense = new_defense.clone();
        self.state.rounds_at_current_level = 0;
        self.state.total_escalations += 1;

        Some(new_defense)
    }

    /// De-escalate to the previous defense level.
    fn deescalate(&mut self) -> Option<Defense> {
        if self.state.escalation_level == 0 {
            return None;
        }

        let prev_level = self.state.escalation_level - 1;
        let new_defense = self.config.escalation_levels[prev_level].clone();

        tracing::info!(
            "De-escalating defense: {} -> {} (level {} -> {}), clean_rounds={}",
            self.state.current_defense,
            new_defense,
            self.state.escalation_level,
            prev_level,
            self.state.consecutive_clean_rounds
        );

        self.state.escalation_level = prev_level;
        self.state.current_defense = new_defense.clone();
        self.state.rounds_at_current_level = 0;
        self.state.consecutive_clean_rounds = 0;
        self.state.total_deescalations += 1;

        Some(new_defense)
    }
}

// =============================================================================
// Integration with UnifiedAggregator
// =============================================================================

/// Extension trait for integrating adaptive defense with UnifiedAggregator.
pub trait AdaptiveDefenseIntegration {
    /// Update the defense strategy based on adaptive manager recommendation.
    fn apply_adaptive_defense(&mut self, defense: &Defense);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AdaptiveDefenseConfig {
        AdaptiveDefenseConfig::default()
            .with_attack_threshold(0.2)
            .with_adaptation_window(3)
            .with_deescalation(5, 0.05)
            .with_min_rounds_at_level(3)
            .with_hysteresis(false, 1.0) // Disable hysteresis for predictable tests
    }

    #[test]
    fn test_new_manager() {
        let config = test_config();
        let manager = AdaptiveDefenseManager::new(config);

        assert_eq!(*manager.current_defense(), Defense::FedAvg);
        assert_eq!(manager.state.escalation_level, 0);
        assert_eq!(manager.state.total_rounds, 0);
    }

    #[test]
    fn test_record_round_result() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        // Record a round with 20% Byzantine
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

        // Record several high Byzantine rate rounds
        manager.record_round_result(3, 10); // 30%
        manager.record_round_result(3, 10); // 30%
        manager.record_round_result(4, 10); // 40%

        assert!(manager.should_escalate());

        let new_defense = manager.adapt();
        assert!(new_defense.is_some());
        assert_eq!(*manager.current_defense(), Defense::Median);
        assert_eq!(manager.state.escalation_level, 1);
        assert_eq!(manager.state.total_escalations, 1);
    }

    #[test]
    fn test_deescalation_after_clean_rounds() {
        let mut config = test_config();
        config.min_rounds_at_level = 2;
        config.deescalation_rounds = 3;
        let mut manager = AdaptiveDefenseManager::new(config);

        // First escalate
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.adapt();
        assert_eq!(manager.state.escalation_level, 1);

        // Now record clean rounds
        manager.record_round_result(0, 10); // 0%
        manager.record_round_result(0, 10); // 0%
        assert!(!manager.should_deescalate()); // Not enough rounds at level

        manager.record_round_result(0, 10); // 0% - now 3 clean rounds, 3 rounds at level
        assert!(manager.should_deescalate());

        let new_defense = manager.adapt();
        assert!(new_defense.is_some());
        assert_eq!(*manager.current_defense(), Defense::FedAvg);
        assert_eq!(manager.state.escalation_level, 0);
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
            manager.record_round_result(3, 10); // 30%
        }
        manager.adapt();
        assert_eq!(manager.state.escalation_level, 1);

        // Try to de-escalate immediately with clean rounds
        for _ in 0..3 {
            manager.record_round_result(0, 10);
        }

        // Should not de-escalate - haven't been at level long enough
        assert!(!manager.should_deescalate());
        assert!(manager.adapt().is_none());

        // After more rounds, should be able to de-escalate
        for _ in 0..2 {
            manager.record_round_result(0, 10);
        }
        assert!(manager.should_deescalate());
    }

    #[test]
    fn test_cannot_escalate_beyond_max() {
        let config = AdaptiveDefenseConfig {
            initial_defense: Defense::MultiKrum { f: 1, k: 3 },
            attack_threshold: 0.2,
            adaptation_window: 3,
            escalation_levels: vec![
                Defense::FedAvg,
                Defense::Median,
                Defense::MultiKrum { f: 1, k: 3 },
            ],
            deescalation_rounds: 5,
            deescalation_threshold: 0.05,
            min_rounds_at_level: 3,
            enable_hysteresis: false,
            hysteresis_factor: 1.0,
        };

        let mut manager = AdaptiveDefenseManager::new(config);
        assert_eq!(manager.state.escalation_level, 2); // Starts at max

        // Record high Byzantine rate
        for _ in 0..3 {
            manager.record_round_result(5, 10);
        }

        // Cannot escalate - already at max
        assert!(!manager.should_escalate());
        assert!(manager.adapt().is_none());
    }

    #[test]
    fn test_cannot_deescalate_below_min() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        // Already at level 0
        assert_eq!(manager.state.escalation_level, 0);

        // Record clean rounds
        for _ in 0..10 {
            manager.record_round_result(0, 10);
        }

        // Cannot de-escalate - already at min
        assert!(!manager.should_deescalate());
        assert!(manager.adapt().is_none());
    }

    #[test]
    fn test_empty_round_handling() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        // Empty round (0 total nodes)
        manager.record_round_result(0, 0);

        // Should handle gracefully with 0% rate
        assert_eq!(manager.state.recent_byzantine_rates.len(), 1);
        assert_eq!(manager.state.recent_byzantine_rates[0], 0.0);
    }

    #[test]
    fn test_100_percent_byzantine() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        // All nodes are Byzantine
        manager.record_round_result(10, 10); // 100%
        manager.record_round_result(10, 10);
        manager.record_round_result(10, 10);

        // Should escalate immediately
        assert!(manager.should_escalate());

        // Keep escalating until max
        while let Some(_) = manager.adapt() {
            manager.record_round_result(10, 10);
        }

        // Should be at max level
        let stats = manager.get_stats();
        assert!(stats.at_max_defense);
        assert_eq!(stats.escalation_level, stats.max_escalation_level);
    }

    #[test]
    fn test_rolling_window() {
        let mut config = test_config();
        config.adaptation_window = 3;
        let mut manager = AdaptiveDefenseManager::new(config);

        // Fill the window
        manager.record_round_result(1, 10); // 10%
        manager.record_round_result(2, 10); // 20%
        manager.record_round_result(3, 10); // 30%

        assert_eq!(manager.state.recent_byzantine_rates.len(), 3);

        // Add one more - should push out the first
        manager.record_round_result(4, 10); // 40%

        assert_eq!(manager.state.recent_byzantine_rates.len(), 3);
        assert!((manager.state.recent_byzantine_rates[0] - 0.2).abs() < 0.001); // 20% is now first
        assert!((manager.state.recent_byzantine_rates[2] - 0.4).abs() < 0.001); // 40% is last
    }

    #[test]
    fn test_hysteresis() {
        let mut config = test_config();
        config.attack_threshold = 0.2;
        config.enable_hysteresis = true;
        config.hysteresis_factor = 1.5; // Need 30% to escalate
        let mut manager = AdaptiveDefenseManager::new(config);

        // 25% - above threshold but below hysteresis threshold
        manager.record_round_result(25, 100);
        manager.record_round_result(25, 100);
        manager.record_round_result(25, 100);

        assert!(!manager.should_escalate()); // Hysteresis prevents escalation

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

        assert_eq!(stats.current_defense, "FedAvg");
        assert_eq!(stats.escalation_level, 0);
        assert!((stats.avg_byzantine_rate - 0.3).abs() < 0.001); // (0.3+0.4+0.2)/3
        assert!((stats.peak_byzantine_rate - 0.4).abs() < 0.001);
        assert_eq!(stats.total_rounds, 3);
        assert!(stats.at_min_defense);
        assert!(!stats.at_max_defense);
    }

    #[test]
    fn test_reset() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        // Build up some state
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.adapt();

        assert_eq!(manager.state.escalation_level, 1);
        assert_eq!(manager.state.total_rounds, 3);

        // Reset
        manager.reset();

        assert_eq!(manager.state.escalation_level, 0);
        assert_eq!(manager.state.total_rounds, 0);
        assert_eq!(*manager.current_defense(), Defense::FedAvg);
        assert!(manager.state.recent_byzantine_rates.is_empty());
    }

    #[test]
    fn test_state_persistence() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config.clone());

        // Build up some state
        manager.record_round_result(3, 10);
        manager.record_round_result(3, 10);
        manager.record_round_result(3, 10);
        manager.adapt();

        let saved_state = manager.get_state().clone();

        // Create new manager and restore
        let mut new_manager = AdaptiveDefenseManager::new(config);
        new_manager.restore_state(saved_state);

        assert_eq!(new_manager.state.escalation_level, 1);
        assert_eq!(new_manager.state.total_rounds, 3);
        assert_eq!(*new_manager.current_defense(), Defense::Median);
    }

    #[test]
    fn test_consecutive_clean_rounds_reset() {
        let mut config = test_config();
        config.deescalation_rounds = 5;
        let mut manager = AdaptiveDefenseManager::new(config);

        // First escalate
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.adapt();

        // Some clean rounds
        manager.record_round_result(0, 10);
        manager.record_round_result(0, 10);
        manager.record_round_result(0, 10);
        assert_eq!(manager.state.consecutive_clean_rounds, 3);

        // One bad round resets the counter
        manager.record_round_result(2, 10); // 20% - above de-escalation threshold
        assert_eq!(manager.state.consecutive_clean_rounds, 0);
    }

    #[test]
    fn test_progressive_escalation() {
        let config = test_config();
        let mut manager = AdaptiveDefenseManager::new(config);

        // Keep recording high Byzantine rates and adapting
        let defenses = vec![
            Defense::FedAvg,
            Defense::Median,
            Defense::TrimmedMean { beta: 0.1 },
            Defense::Krum { f: 1 },
            Defense::MultiKrum { f: 1, k: 3 },
        ];

        for (i, expected_defense) in defenses.iter().enumerate() {
            assert_eq!(*manager.current_defense(), *expected_defense);
            assert_eq!(manager.state.escalation_level, i);

            // Record high Byzantine rate to trigger escalation
            manager.record_round_result(5, 10);
            manager.record_round_result(5, 10);
            manager.record_round_result(5, 10);

            manager.adapt();
        }

        // Should be at max now
        let stats = manager.get_stats();
        assert!(stats.at_max_defense);
        assert_eq!(stats.total_escalations, 4);
    }

    #[test]
    fn test_config_builders() {
        let config = AdaptiveDefenseConfig::default()
            .with_initial_defense(Defense::Median)
            .with_attack_threshold(0.3)
            .with_adaptation_window(10)
            .with_deescalation(15, 0.1)
            .with_min_rounds_at_level(7)
            .with_hysteresis(true, 1.3);

        assert_eq!(config.initial_defense, Defense::Median);
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
            .with_attack_threshold(1.5) // Should be clamped to 1.0
            .with_deescalation(5, -0.1); // Should be clamped to 0.0

        assert!((config.attack_threshold - 1.0).abs() < 0.001);
        assert!((config.deescalation_threshold - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_custom_escalation_levels() {
        let custom_levels = vec![
            Defense::Median,
            Defense::Krum { f: 2 },
            Defense::MultiKrum { f: 3, k: 5 },
        ];

        let config = AdaptiveDefenseConfig::default()
            .with_escalation_levels(custom_levels.clone())
            .with_initial_defense(Defense::Median);

        let mut manager = AdaptiveDefenseManager::new(config);

        assert_eq!(*manager.current_defense(), Defense::Median);
        assert_eq!(manager.state.escalation_level, 0);

        // Escalate
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.record_round_result(5, 10);
        manager.adapt();

        assert_eq!(*manager.current_defense(), Defense::Krum { f: 2 });
    }

    #[test]
    fn test_initial_defense_not_in_levels() {
        // If initial defense is not in escalation levels, start at level 0
        let config = AdaptiveDefenseConfig {
            initial_defense: Defense::GeometricMedian {
                max_iterations: 100,
                tolerance: 1e-6,
            },
            escalation_levels: vec![Defense::FedAvg, Defense::Median],
            ..Default::default()
        };

        let manager = AdaptiveDefenseManager::new(config);

        // Should fall back to level 0
        assert_eq!(manager.state.escalation_level, 0);
        assert_eq!(*manager.current_defense(), Defense::FedAvg);
    }
}
