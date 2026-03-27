// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-node PoGQ state machine.
//!
//! This state machine tracks EMA scores, violation/clear counters, and
//! quarantine status for each node across FL rounds. The transition logic
//! is designed to be compatible with the winterfell-pogq STARK trace
//! definition (`vsv-stark/winterfell-pogq/src/trace.rs`), which constrains:
//!
//! - **EMA update**: `ema_{t+1} = beta * ema_t + (1 - beta) * x_t`
//! - **Violation counter**: increments on violation, resets on clear
//! - **Clear counter**: increments on clear, resets on violation
//! - **Quarantine entry**: `violation_count >= k` (and not already quarantined)
//! - **Quarantine release**: `clear_count >= m` (while quarantined)
//! - **Warm-up override**: quarantine suppressed during first `W` rounds

use serde::{Deserialize, Serialize};

use super::config::PoGQv41Config;

/// Per-node PoGQ state tracked across FL rounds.
///
/// This struct represents a single row in the per-node state table.
/// The STARK prover verifies that these transitions are correct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodePoGQState {
    /// Unique node identifier.
    pub node_id: String,

    /// Exponential moving average of hybrid scores.
    /// Updated each round: `ema = beta * ema + (1 - beta) * hybrid_score`.
    pub ema_score: f64,

    /// Consecutive violation count (resets on clear).
    pub violation_count: usize,

    /// Consecutive clear count since last violation (resets on violation).
    pub clear_count: usize,

    /// Whether this node is currently quarantined.
    pub quarantined: bool,

    /// Total rounds this node has participated in.
    pub rounds_seen: usize,

    /// Lifetime total violations (never resets).
    pub total_violations: usize,

    /// Lifetime total clears (never resets).
    pub total_clears: usize,
}

impl NodePoGQState {
    /// Create a new node state with default (clean) values.
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            ema_score: 0.0,
            violation_count: 0,
            clear_count: 0,
            quarantined: false,
            rounds_seen: 0,
            total_violations: 0,
            total_clears: 0,
        }
    }

    /// Update state with a new hybrid score for this round.
    ///
    /// Implements the full PoGQ state machine transition:
    ///
    /// 1. **EMA update**: `ema = beta * ema + (1 - beta) * hybrid_score`
    /// 2. **Violation check**: `ema < threshold`
    /// 3. **Counter update**: violation increments `violation_count` and resets
    ///    `clear_count`; clear does the reverse
    /// 4. **Hysteresis**: quarantine if `violation_count >= k`, release if
    ///    `clear_count >= m`
    /// 5. **Warm-up override**: never quarantine during first `warm_up_rounds`
    ///
    /// Returns whether the node is quarantined after this update.
    pub fn update(
        &mut self,
        hybrid_score: f64,
        threshold: f64,
        config: &PoGQv41Config,
    ) -> bool {
        self.rounds_seen += 1;

        // 1. EMA update
        if self.rounds_seen == 1 {
            // First round: initialize EMA to the raw score
            self.ema_score = hybrid_score;
        } else {
            self.ema_score =
                config.beta * self.ema_score + (1.0 - config.beta) * hybrid_score;
        }

        // 2. Violation check
        let is_violation = self.ema_score < threshold;

        // 3. Counter update
        if is_violation {
            self.violation_count += 1;
            self.clear_count = 0;
            self.total_violations += 1;
        } else {
            self.clear_count += 1;
            self.violation_count = 0;
            self.total_clears += 1;
        }

        // 4. Hysteresis: enter quarantine or release
        if self.violation_count >= config.k_quarantine {
            self.quarantined = true;
        } else if self.clear_count >= config.m_release {
            self.quarantined = false;
        }
        // Otherwise: maintain current quarantine state

        // 5. Warm-up override: suppress quarantine during grace period
        if self.rounds_seen <= config.warm_up_rounds {
            self.quarantined = false;
        }

        self.quarantined
    }

    /// Reset quarantine state (manual admin override).
    pub fn reset_quarantine(&mut self) {
        self.quarantined = false;
        self.violation_count = 0;
        self.clear_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PoGQv41Config {
        PoGQv41Config::default()
    }

    #[test]
    fn test_new_state() {
        let state = NodePoGQState::new("node-0");
        assert_eq!(state.node_id, "node-0");
        assert_eq!(state.ema_score, 0.0);
        assert_eq!(state.violation_count, 0);
        assert_eq!(state.clear_count, 0);
        assert!(!state.quarantined);
        assert_eq!(state.rounds_seen, 0);
        assert_eq!(state.total_violations, 0);
        assert_eq!(state.total_clears, 0);
    }

    #[test]
    fn test_first_round_initializes_ema() {
        let mut state = NodePoGQState::new("n");
        let cfg = default_config();
        state.update(0.8, 0.5, &cfg);
        assert!((state.ema_score - 0.8).abs() < 1e-9);
        assert_eq!(state.rounds_seen, 1);
    }

    #[test]
    fn test_ema_converges_toward_recent() {
        let mut state = NodePoGQState::new("n");
        let cfg = default_config();

        // Give it a high initial score
        state.update(1.0, 0.5, &cfg);
        assert!((state.ema_score - 1.0).abs() < 1e-9);

        // Feed low scores — EMA should decrease toward them
        for _ in 0..20 {
            state.update(0.2, 0.5, &cfg);
        }

        // After 20 rounds with beta=0.85, EMA should be close to 0.2
        assert!(
            (state.ema_score - 0.2).abs() < 0.05,
            "EMA should converge toward 0.2, got {}",
            state.ema_score
        );
    }

    #[test]
    fn test_violation_then_quarantine() {
        let mut state = NodePoGQState::new("n");
        let cfg = PoGQv41Config {
            warm_up_rounds: 0, // disable warm-up for this test
            k_quarantine: 2,
            ..default_config()
        };
        let threshold = 0.5;

        // Two consecutive violations should trigger quarantine (k=2)
        state.update(0.1, threshold, &cfg); // violation 1
        assert!(!state.quarantined);
        assert_eq!(state.violation_count, 1);

        state.update(0.1, threshold, &cfg); // violation 2 -> quarantine
        assert!(state.quarantined);
        assert_eq!(state.violation_count, 2);
    }

    #[test]
    fn test_clear_releases_quarantine() {
        let mut state = NodePoGQState::new("n");
        let cfg = PoGQv41Config {
            warm_up_rounds: 0,
            k_quarantine: 1,
            m_release: 3,
            beta: 0.0, // No EMA smoothing — raw scores pass through
            ..default_config()
        };

        // Force quarantine with a low score
        state.update(0.1, 0.5, &cfg);
        assert!(state.quarantined);

        // Three consecutive high scores should release (m=3)
        // With beta=0.0, EMA = raw score
        state.update(0.9, 0.5, &cfg); // clear 1
        assert!(state.quarantined); // still quarantined
        state.update(0.9, 0.5, &cfg); // clear 2
        assert!(state.quarantined); // still quarantined
        state.update(0.9, 0.5, &cfg); // clear 3 -> release
        assert!(!state.quarantined);
    }

    #[test]
    fn test_warmup_suppresses_quarantine() {
        let mut state = NodePoGQState::new("n");
        let cfg = PoGQv41Config {
            warm_up_rounds: 3,
            k_quarantine: 1, // would normally quarantine on first violation
            ..default_config()
        };

        // Violations during warm-up should NOT quarantine
        state.update(0.1, 0.5, &cfg); // round 1 (warm-up)
        assert!(!state.quarantined);
        state.update(0.1, 0.5, &cfg); // round 2 (warm-up)
        assert!(!state.quarantined);
        state.update(0.1, 0.5, &cfg); // round 3 (warm-up)
        assert!(!state.quarantined);

        // Round 4: warm-up over, should now quarantine
        state.update(0.1, 0.5, &cfg);
        assert!(state.quarantined);
    }

    #[test]
    fn test_lifetime_counters() {
        let mut state = NodePoGQState::new("n");
        let cfg = PoGQv41Config {
            warm_up_rounds: 0,
            beta: 0.0, // No EMA smoothing — raw scores used for threshold comparison
            ..default_config()
        };

        state.update(0.1, 0.5, &cfg); // violation (0.1 < 0.5)
        state.update(0.9, 0.5, &cfg); // clear (0.9 >= 0.5)
        state.update(0.1, 0.5, &cfg); // violation
        state.update(0.1, 0.5, &cfg); // violation

        assert_eq!(state.total_violations, 3);
        assert_eq!(state.total_clears, 1);
        assert_eq!(state.rounds_seen, 4);
    }

    #[test]
    fn test_clear_resets_violation_count() {
        let mut state = NodePoGQState::new("n");
        let cfg = PoGQv41Config {
            warm_up_rounds: 0,
            k_quarantine: 3,
            beta: 0.0, // No EMA smoothing — raw scores pass through
            ..default_config()
        };

        state.update(0.1, 0.5, &cfg); // violation 1 (0.1 < 0.5)
        assert_eq!(state.violation_count, 1);

        state.update(0.9, 0.5, &cfg); // clear -> resets violation count
        assert_eq!(state.violation_count, 0);
        assert_eq!(state.clear_count, 1);

        state.update(0.1, 0.5, &cfg); // violation 1 again
        assert_eq!(state.violation_count, 1);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut state = NodePoGQState::new("test-node");
        let cfg = default_config();
        state.update(0.7, 0.5, &cfg);
        state.update(0.3, 0.5, &cfg);

        let json = serde_json::to_string(&state).unwrap();
        let state2: NodePoGQState = serde_json::from_str(&json).unwrap();

        assert_eq!(state.node_id, state2.node_id);
        assert!((state.ema_score - state2.ema_score).abs() < 1e-12);
        assert_eq!(state.violation_count, state2.violation_count);
        assert_eq!(state.clear_count, state2.clear_count);
        assert_eq!(state.quarantined, state2.quarantined);
        assert_eq!(state.rounds_seen, state2.rounds_seen);
        assert_eq!(state.total_violations, state2.total_violations);
        assert_eq!(state.total_clears, state2.total_clears);
    }

    #[test]
    fn test_reset_quarantine() {
        let mut state = NodePoGQState::new("n");
        state.quarantined = true;
        state.violation_count = 5;
        state.clear_count = 1;

        state.reset_quarantine();
        assert!(!state.quarantined);
        assert_eq!(state.violation_count, 0);
        assert_eq!(state.clear_count, 0);
    }
}
