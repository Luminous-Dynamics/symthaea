// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Describes the exploration mutation that [`CuriosityDrive::update`] wants to
/// apply. The caller routes this through the feedback proposal system.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ExplorationUpdate {
    /// Set exploration_urge to an absolute value (boredom * curiosity).
    Set(f32),
    /// Scale exploration_urge by a decay factor (e.g. 0.9).
    Scale(f32),
}

/// Curiosity drive - novelty-seeking mechanism to prevent stagnation
///
/// When predictions become too accurate/predictable, curiosity triggers
/// exploration mode to discover new patterns and prevent cognitive stagnation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CuriosityDrive {
    /// Recent prediction errors (rolling window).
    /// Capacity bound: HISTORY_SIZE (50) elements -- evict before push via pop_front.
    error_history: std::collections::VecDeque<f32>,

    /// Boredom level (0.0 to 1.0)
    /// High when predictions are too accurate for too long
    pub boredom: f32,

    /// Curiosity level (0.0 to 1.0)
    /// High boredom triggers high curiosity
    pub curiosity: f32,

    /// Exploration urge (0.0 to 1.0)
    /// Direct measure of desire to explore new patterns
    pub exploration_urge: f64,

    /// Novelty bonus for learning rate
    /// Higher when exploring new territory
    pub novelty_bonus: f32,

    /// Consecutive cycles of low error (boredom buildup)
    low_error_streak: u32,

    /// Threshold below which error is "too good"
    boredom_threshold: f32,
}

impl Default for CuriosityDrive {
    fn default() -> Self {
        Self {
            error_history: std::collections::VecDeque::with_capacity(50),
            boredom: 0.0,
            curiosity: 0.3, // Start with some curiosity
            exploration_urge: 0.0,
            novelty_bonus: 1.0,
            low_error_streak: 0,
            boredom_threshold: 0.1,
        }
    }
}

impl CuriosityDrive {
    /// Window size for error history
    const HISTORY_SIZE: usize = 50;
    /// Boredom streak threshold
    const BOREDOM_STREAK: u32 = 10;
    /// Maximum novelty bonus
    const MAX_NOVELTY_BONUS: f32 = 1.5;

    /// Update curiosity drive based on prediction error.
    ///
    /// Returns an [`ExplorationUpdate`] describing the intended mutation to
    /// `exploration_urge`. The **caller** is responsible for applying this
    /// through the feedback proposal system (via `set_exploration` /
    /// `scale_exploration`) so that divergence tracking captures it.
    pub fn update(&mut self, prediction_error: f32) -> ExplorationUpdate {
        // Track error history -- evict before push to prevent transient over-capacity
        if self.error_history.len() >= Self::HISTORY_SIZE {
            self.error_history.pop_front();
        }
        self.error_history.push_back(prediction_error);

        // Compute average error (safe division with max(1))
        let avg_error = if !self.error_history.is_empty() {
            self.error_history.iter().sum::<f32>() / self.error_history.len().max(1) as f32
        } else {
            0.5
        };

        // Detect boredom (consistently low error)
        if prediction_error < self.boredom_threshold {
            self.low_error_streak += 1;
        } else {
            self.low_error_streak = self.low_error_streak.saturating_sub(2);
        }

        // Boredom grows with low error streak (safe cast via f64)
        let streak_factor =
            ((self.low_error_streak as f64 / Self::BOREDOM_STREAK.max(1) as f64) as f32).min(1.0);
        self.boredom = 0.9 * self.boredom + 0.1 * streak_factor;

        // Curiosity is inverse of average error (interesting when things are predictable)
        let error_curiosity = (1.0 - avg_error.min(1.0)).max(0.0);
        self.curiosity = 0.8 * self.curiosity + 0.2 * error_curiosity;

        // Compute exploration intent (caller applies through feedback system)
        let exploration_update = if self.boredom > 0.5 && self.curiosity > 0.5 {
            ExplorationUpdate::Set((self.boredom * self.curiosity).min(1.0))
        } else {
            ExplorationUpdate::Scale(0.9) // Decay
        };

        // Novelty bonus: higher when exploring after boredom
        // Reads current exploration_urge (set by feedback system in previous cycle)
        if self.exploration_urge > 0.3 {
            self.novelty_bonus =
                1.0 + (Self::MAX_NOVELTY_BONUS - 1.0) * self.exploration_urge as f32;
        } else if prediction_error > 0.5 {
            // High error = novel situation, boost learning
            self.novelty_bonus = 1.0 + 0.3 * (prediction_error - 0.5);
        } else {
            self.novelty_bonus = 1.0;
        }

        exploration_update
    }

    /// Check if exploration should be triggered
    pub fn should_explore(&self) -> bool {
        self.exploration_urge > 0.4 || (self.boredom > 0.7 && self.curiosity > 0.6)
    }

    /// Get effective learning rate with novelty bonus
    pub fn effective_learning_rate(&self, base_rate: f32) -> f32 {
        base_rate * self.novelty_bonus
    }

    /// Reset curiosity drive
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Set boredom threshold from self-reflection
    ///
    /// This allows the meta-learning system to adjust when boredom triggers.
    pub fn set_boredom_threshold(&mut self, threshold: f32) {
        self.boredom_threshold = threshold.clamp(0.05, 0.3);
    }

    /// Get current boredom threshold
    pub fn get_boredom_threshold(&self) -> f32 {
        self.boredom_threshold
    }

    /// Apply an exploration update directly (test-only).
    ///
    /// In production code, the feedback helper system routes ExplorationUpdate
    /// through ProposalCollector. This method exists only for unit-testing
    /// CuriosityDrive in isolation.
    #[cfg(test)]
    pub fn apply_exploration_update(&mut self, upd: ExplorationUpdate) {
        match upd {
            ExplorationUpdate::Set(val) => {
                self.exploration_urge = (val as f64).clamp(0.0, 1.0);
            }
            ExplorationUpdate::Scale(factor) => {
                self.exploration_urge = (self.exploration_urge * factor as f64).clamp(0.0, 1.0);
            }
        }
        self.novelty_bonus = 1.0 + (Self::MAX_NOVELTY_BONUS - 1.0) * self.exploration_urge as f32;
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn curiosity_default_state() {
        let cd = CuriosityDrive::default();
        assert_eq!(cd.boredom, 0.0);
        assert!((cd.curiosity - 0.3).abs() < f32::EPSILON);
        assert_eq!(cd.exploration_urge, 0.0);
        assert_eq!(cd.novelty_bonus, 1.0);
    }

    #[test]
    fn curiosity_low_error_builds_boredom() {
        let mut cd = CuriosityDrive::default();
        // Feed many low-error updates
        for _ in 0..30 {
            let upd = cd.update(0.01);
            cd.apply_exploration_update(upd);
        }
        assert!(cd.boredom > 0.3, "boredom should build: {}", cd.boredom);
    }

    #[test]
    fn curiosity_high_error_resets_streak() {
        let mut cd = CuriosityDrive::default();
        for _ in 0..20 {
            let upd = cd.update(0.01);
            cd.apply_exploration_update(upd);
        }
        let boredom_before = cd.boredom;
        // High error should reduce low_error_streak
        for _ in 0..10 {
            let upd = cd.update(0.5);
            cd.apply_exploration_update(upd);
        }
        assert!(
            cd.boredom < boredom_before + 0.1,
            "boredom should not keep growing after high error"
        );
    }

    #[test]
    fn curiosity_should_explore_after_boredom() {
        let mut cd = CuriosityDrive::default();
        // Build up enough boredom + curiosity
        for _ in 0..100 {
            let upd = cd.update(0.01); // Very low error -> boredom + high curiosity
            cd.apply_exploration_update(upd);
        }
        assert!(
            cd.should_explore(),
            "should want to explore after prolonged low error"
        );
    }

    #[test]
    fn curiosity_effective_lr_baseline() {
        let cd = CuriosityDrive::default();
        let lr = cd.effective_learning_rate(0.01);
        assert!(
            (lr - 0.01).abs() < 1e-6,
            "baseline novelty_bonus should be 1.0"
        );
    }

    #[test]
    fn curiosity_effective_lr_bounded() {
        let mut cd = CuriosityDrive::default();
        // Drive exploration to max
        for _ in 0..200 {
            let upd = cd.update(0.01);
            cd.apply_exploration_update(upd);
        }
        let lr = cd.effective_learning_rate(1.0);
        assert!(
            lr <= CuriosityDrive::MAX_NOVELTY_BONUS,
            "LR bonus {} exceeds max {}",
            lr,
            CuriosityDrive::MAX_NOVELTY_BONUS
        );
        assert!(lr >= 1.0, "LR bonus should be >= 1.0");
    }

    #[test]
    fn curiosity_high_error_gives_novelty_bonus() {
        let mut cd = CuriosityDrive::default();
        let _upd = cd.update(0.8); // High error = novel situation
        // Check novelty_bonus right after update (before apply_exploration_update
        // resets it based on exploration_urge which is still 0.0 at this point).
        assert!(
            cd.novelty_bonus > 1.0,
            "high error should boost novelty: {}",
            cd.novelty_bonus
        );
    }

    #[test]
    fn curiosity_history_bounded() {
        let mut cd = CuriosityDrive::default();
        for i in 0..200 {
            let upd = cd.update(i as f32 * 0.005);
            cd.apply_exploration_update(upd);
        }
        assert!(cd.error_history.len() <= CuriosityDrive::HISTORY_SIZE);
    }

    #[test]
    fn curiosity_set_boredom_threshold_clamped() {
        let mut cd = CuriosityDrive::default();
        cd.set_boredom_threshold(0.0);
        assert!((cd.get_boredom_threshold() - 0.05).abs() < f32::EPSILON);
        cd.set_boredom_threshold(1.0);
        assert!((cd.get_boredom_threshold() - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn curiosity_reset_restores_default() {
        let mut cd = CuriosityDrive::default();
        for _ in 0..50 {
            let upd = cd.update(0.01);
            cd.apply_exploration_update(upd);
        }
        cd.reset();
        assert_eq!(cd.boredom, 0.0);
        assert_eq!(cd.exploration_urge, 0.0);
    }
}
