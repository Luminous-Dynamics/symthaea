// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conversation coherence tracking with degradation detection and correction urgency.

use std::collections::VecDeque;

/// Tracks coherence trajectory across conversation turns.
/// Detects degradation and produces a correction urgency signal.
#[derive(Debug, Clone)]
pub struct ConversationCoherenceTracker {
    history: VecDeque<f32>,
    ema_coherence: f32,
    alpha: f32,
    degradation_threshold: f32,
    degradation_streak: u32,
    degradation_count: u32,
    total_turns: u64,
    min_coherence: f32,
    max_coherence: f32,
    sum_sq_diff: f64,
}

/// Statistics about the coherence trajectory.
#[derive(Debug, Clone)]
pub struct CoherenceTrajectoryStats {
    pub current_ema: f32,
    pub min_coherence: f32,
    pub max_coherence: f32,
    pub degradation_count: u32,
    pub stability: f32,
    pub total_turns: u64,
}

impl ConversationCoherenceTracker {
    /// Create a new tracker with the given degradation threshold.
    pub fn new(degradation_threshold: f32) -> Self {
        Self {
            history: VecDeque::with_capacity(200),
            ema_coherence: 0.5,
            alpha: 0.1,
            degradation_threshold,
            degradation_streak: 0,
            degradation_count: 0,
            total_turns: 0,
            min_coherence: f32::MAX,
            max_coherence: f32::MIN,
            sum_sq_diff: 0.0,
        }
    }

    /// Record a turn's coherence value. Returns true if degradation detected.
    pub fn record_turn(&mut self, coherence: f32) -> bool {
        self.total_turns += 1;
        self.history.push_back(coherence);
        if self.history.len() > 200 {
            self.history.pop_front();
        }

        // Update min/max
        if coherence < self.min_coherence {
            self.min_coherence = coherence;
        }
        if coherence > self.max_coherence {
            self.max_coherence = coherence;
        }

        // Update EMA
        let prev_ema = self.ema_coherence;
        self.ema_coherence =
            symthaea_core::math::ema_update(self.ema_coherence, coherence, self.alpha);

        // Track variance via running sum of squared differences from EMA
        let diff = coherence - self.ema_coherence;
        self.sum_sq_diff += (diff * diff) as f64;

        // Degradation detection: EMA dropped below threshold or consecutive drops

        if self.ema_coherence < self.degradation_threshold {
            self.degradation_streak += 1;
            self.degradation_count += 1;
            true
        } else if self.ema_coherence < prev_ema - 0.05 {
            // Significant single-step drop
            self.degradation_streak += 1;
            if self.degradation_streak >= 3 {
                self.degradation_count += 1;
                true
            } else {
                false
            }
        } else {
            self.degradation_streak = 0;
            false
        }
    }

    /// Returns correction urgency in [0.0, 1.0].
    /// Higher values mean more urgent need for self-correction.
    pub fn correction_urgency(&self) -> f32 {
        if self.total_turns == 0 {
            return 0.0;
        }
        // Combine: low EMA + high streak + many degradations
        let ema_urgency = (1.0 - self.ema_coherence).clamp(0.0, 1.0);
        let streak_urgency = (self.degradation_streak as f32 / 5.0).clamp(0.0, 1.0);
        let count_urgency = (self.degradation_count as f32 / 10.0).clamp(0.0, 1.0);
        (0.5 * ema_urgency + 0.3 * streak_urgency + 0.2 * count_urgency).clamp(0.0, 1.0)
    }

    /// Get trajectory statistics.
    pub fn stats(&self) -> CoherenceTrajectoryStats {
        let variance = if self.total_turns > 1 {
            (self.sum_sq_diff / self.total_turns as f64) as f32
        } else {
            0.0
        };
        // Stability = 1.0 - normalized variance (capped at 1.0)
        let stability = (1.0 - variance.sqrt().min(1.0)).clamp(0.0, 1.0);

        CoherenceTrajectoryStats {
            current_ema: self.ema_coherence,
            min_coherence: if self.total_turns > 0 {
                self.min_coherence
            } else {
                0.0
            },
            max_coherence: if self.total_turns > 0 {
                self.max_coherence
            } else {
                0.0
            },
            degradation_count: self.degradation_count,
            stability,
            total_turns: self.total_turns,
        }
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.history.clear();
        self.ema_coherence = 0.5;
        self.degradation_streak = 0;
        self.degradation_count = 0;
        self.total_turns = 0;
        self.min_coherence = f32::MAX;
        self.max_coherence = f32::MIN;
        self.sum_sq_diff = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_coherence_no_degradation() {
        let mut tracker = ConversationCoherenceTracker::new(0.3);
        for _ in 0..50 {
            assert!(!tracker.record_turn(0.7));
        }
        assert_eq!(tracker.stats().degradation_count, 0);
        assert!(tracker.correction_urgency() < 0.5);
    }

    #[test]
    fn test_degradation_detection() {
        let mut tracker = ConversationCoherenceTracker::new(0.3);
        // Start high
        for _ in 0..20 {
            tracker.record_turn(0.8);
        }
        // Drop low
        for _ in 0..10 {
            tracker.record_turn(0.1);
        }
        assert!(tracker.stats().degradation_count > 0);
        assert!(tracker.correction_urgency() > 0.3);
    }

    #[test]
    fn test_reset() {
        let mut tracker = ConversationCoherenceTracker::new(0.3);
        for _ in 0..10 {
            tracker.record_turn(0.5);
        }
        tracker.reset();
        assert_eq!(tracker.stats().total_turns, 0);
        assert_eq!(tracker.stats().degradation_count, 0);
    }

    #[test]
    fn test_stability_computation() {
        let mut tracker = ConversationCoherenceTracker::new(0.3);
        // Very stable input
        for _ in 0..100 {
            tracker.record_turn(0.6);
        }
        let stats = tracker.stats();
        assert!(
            stats.stability > 0.9,
            "Stability should be high for constant input: {}",
            stats.stability
        );
    }
}
