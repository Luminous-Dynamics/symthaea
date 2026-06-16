// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Closed learning loop for the cognitive loop.
//!
//! Implements Q-learning guided strategy selection with Φ-gating,
//! enabling behavioral adaptation based on prediction error feedback.

use rand::Rng;
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::ShakeRng;

use super::flow::ResponseStrategy;
use super::thresholds::{
    EXPLORATION_DECAY_RATE, EXPLORATION_RATE_INITIAL, EXPLORATION_RATE_MIN,
    PHI_INTEGRATIVE_THRESHOLD, PHI_REACTIVE_THRESHOLD, Q_LEARNING_RATE, Q_VALUE_INITIAL,
    REWARD_NEGATIVE_THRESHOLD, REWARD_POSITIVE_THRESHOLD,
};

/// Learning result from a cycle (for closed loop)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleLearningResult {
    /// Reward from cycle (-1.0 to 1.0)
    /// Based on prediction error (lower error = higher reward)
    pub reward: f32,

    /// Strategy that was used
    pub strategy_used: ResponseStrategy,

    /// Whether the cycle was successful (low error, in flow, etc.)
    pub successful: bool,

    /// Prediction error during this cycle
    pub prediction_error: f32,

    /// Coherence during this cycle
    pub coherence: f32,
}

/// Closed Learning Loop Manager
///
/// Implements the paradigm shift from CLOSED_LEARNING_LOOP.md:
/// - Learning → Behavioral Change (not just compute and discard)
/// - Q-learning guided strategy selection
/// - Φ-gated strategy preferences
#[derive(Debug)]
pub struct ClosedLearningLoop {
    /// Current selected strategy
    pub current_strategy: ResponseStrategy,

    /// Last learning result (influences next strategy)
    pub last_result: Option<CycleLearningResult>,

    /// Q-values for each strategy (estimated long-term reward)
    q_values: [f32; 5],

    /// Learning rate for Q-updates
    q_learning_rate: f32,

    /// Exploration rate (epsilon for epsilon-greedy)
    exploration_rate: f32,

    /// Total interactions
    total_interactions: u64,

    /// Total accumulated reward
    total_reward: f32,

    /// Strategy usage counts
    strategy_counts: [u64; 5],

    /// Optional genesis-seeded RNG for deterministic exploration
    rng: Option<ShakeRng>,
}

impl Default for ClosedLearningLoop {
    fn default() -> Self {
        Self {
            current_strategy: ResponseStrategy::default(),
            last_result: None,
            q_values: [Q_VALUE_INITIAL; 5], // Start neutral
            q_learning_rate: Q_LEARNING_RATE,
            exploration_rate: EXPLORATION_RATE_INITIAL,
            total_interactions: 0,
            total_reward: 0.0,
            strategy_counts: [0; 5],
            rng: None,
        }
    }
}

impl ClosedLearningLoop {
    /// Create with a genesis-seeded RNG for deterministic exploration.
    pub fn with_rng(rng: ShakeRng) -> Self {
        Self {
            rng: Some(rng),
            ..Default::default()
        }
    }

    /// Select strategy based on Q-learning + previous result + Φ
    ///
    /// This is the core of the closed learning loop:
    /// 1. Start with Q-learning policy (greedy or explore)
    /// 2. Modify based on previous result
    /// 3. Gate based on consciousness level (Φ)
    pub fn select_strategy(&mut self, phi: f64, _previous_reward: Option<f32>) -> ResponseStrategy {
        // Step 1: Q-learning selection (epsilon-greedy)
        let (explore_val, variant_val): (f32, u8) = match self.rng.as_mut() {
            Some(rng) => (rng.r#gen::<f32>(), rng.r#gen::<u8>()),
            None => (rand::random::<f32>(), rand::random::<u8>()),
        };
        let explore = explore_val < self.exploration_rate;
        let base_strategy = if explore {
            // Random exploration
            match variant_val % 5 {
                0 => ResponseStrategy::Detailed,
                1 => ResponseStrategy::Concise,
                2 => ResponseStrategy::Clarifying,
                3 => ResponseStrategy::Supportive,
                _ => ResponseStrategy::Exploratory,
            }
        } else {
            // Greedy: select best Q-value
            let best_idx = self
                .q_values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(3); // Default to Supportive

            match best_idx {
                0 => ResponseStrategy::Detailed,
                1 => ResponseStrategy::Concise,
                2 => ResponseStrategy::Clarifying,
                3 => ResponseStrategy::Supportive,
                _ => ResponseStrategy::Exploratory,
            }
        };

        // Step 2: Modify based on previous result
        let strategy = if let Some(ref last) = self.last_result {
            if last.reward > REWARD_POSITIVE_THRESHOLD {
                // Strong positive - stick with what worked
                last.strategy_used
            } else if last.reward < REWARD_NEGATIVE_THRESHOLD {
                // Negative - try opposite strategy
                last.strategy_used.opposite()
            } else {
                base_strategy
            }
        } else {
            base_strategy
        };

        // Step 3: Φ-gating (consciousness influences strategy)
        let final_strategy = if phi >= PHI_INTEGRATIVE_THRESHOLD {
            // Integrative mode - favor Exploratory/Detailed
            match strategy {
                ResponseStrategy::Supportive => ResponseStrategy::Exploratory,
                ResponseStrategy::Concise => ResponseStrategy::Detailed,
                other => other,
            }
        } else if phi < PHI_REACTIVE_THRESHOLD {
            // Reactive mode - favor Supportive/Concise
            match strategy {
                ResponseStrategy::Exploratory => ResponseStrategy::Supportive,
                ResponseStrategy::Detailed => ResponseStrategy::Concise,
                other => other,
            }
        } else {
            // Reflective mode - use Q-learning selection as-is
            strategy
        };

        self.current_strategy = final_strategy;
        final_strategy
    }

    /// Update Q-values with cycle result.
    pub fn update(&mut self, result: CycleLearningResult) {
        self.update_with_plasticity(result, 1.0);
    }

    /// Update Q-values, modulated by FEP-derived plasticity factor.
    ///
    /// `plasticity` scales the Q-learning rate: >1.0 = faster learning (high FEP
    /// learning signal), <1.0 = slower (low signal). Clamped to [0.5, 2.0].
    /// Friston (2010): free energy drives learning rate via expected information gain.
    pub fn update_with_plasticity(&mut self, result: CycleLearningResult, plasticity: f32) {
        let plasticity = plasticity.clamp(0.5, 2.0);

        // Update strategy count
        let strategy_idx = self.strategy_index(result.strategy_used);
        self.strategy_counts[strategy_idx] += 1;

        // Q-learning update: Q(s,a) <- Q(s,a) + α*plasticity * (r - Q(s,a))
        let old_q = self.q_values[strategy_idx];
        let effective_lr = self.q_learning_rate * plasticity;
        let new_q = old_q + effective_lr * (result.reward - old_q);
        self.q_values[strategy_idx] = new_q;

        // Update totals
        self.total_interactions += 1;
        self.total_reward += result.reward;

        // Store for next selection
        self.last_result = Some(result);

        // Decay exploration rate over time (but keep minimum of 5%)
        self.exploration_rate =
            (self.exploration_rate * EXPLORATION_DECAY_RATE).max(EXPLORATION_RATE_MIN);
    }

    /// Get strategy index for Q-value lookup
    fn strategy_index(&self, strategy: ResponseStrategy) -> usize {
        match strategy {
            ResponseStrategy::Detailed => 0,
            ResponseStrategy::Concise => 1,
            ResponseStrategy::Clarifying => 2,
            ResponseStrategy::Supportive => 3,
            ResponseStrategy::Exploratory => 4,
        }
    }

    /// Get average reward
    pub fn average_reward(&self) -> f32 {
        if self.total_interactions == 0 {
            0.0
        } else {
            self.total_reward / self.total_interactions as f32
        }
    }

    /// Get best strategy according to Q-values
    pub fn best_strategy(&self) -> ResponseStrategy {
        let best_idx = self
            .q_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(3);

        match best_idx {
            0 => ResponseStrategy::Detailed,
            1 => ResponseStrategy::Concise,
            2 => ResponseStrategy::Clarifying,
            3 => ResponseStrategy::Supportive,
            _ => ResponseStrategy::Exploratory,
        }
    }

    /// Get Q-values for each strategy
    pub fn q_values(&self) -> &[f32; 5] {
        &self.q_values
    }

    /// Get strategy usage counts
    pub fn strategy_counts(&self) -> &[u64; 5] {
        &self.strategy_counts
    }

    /// Get exploration rate
    pub fn exploration_rate(&self) -> f32 {
        self.exploration_rate
    }

    /// Set exploration rate (clamped to [EXPLORATION_RATE_MIN, 1.0]).
    pub fn set_exploration_rate(&mut self, rate: f32) {
        self.exploration_rate = rate.clamp(EXPLORATION_RATE_MIN, 1.0);
    }

    /// Get total interactions
    pub fn total_interactions(&self) -> u64 {
        self.total_interactions
    }

    /// Reset the learning loop
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Force the CLL to deterministically select a specific strategy by
    /// setting its Q-value to 1.0 and all others to 0.0 with no exploration.
    #[cfg(test)]
    pub(crate) fn force_strategy(&mut self, strategy: ResponseStrategy) {
        let idx = self.strategy_index(strategy);
        self.q_values = [0.0; 5];
        self.q_values[idx] = 1.0;
        self.exploration_rate = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::genesis::{GenesisSeed, ShakeRng};

    /// Helper: build a seeded ShakeRng from a u64 seed.
    fn rng_from_seed(seed: u64) -> ShakeRng {
        let phrase = format!("test-seed-{seed}");
        GenesisSeed::from_phrase(&phrase).domain("learning-tests")
    }

    /// Helper: build a CycleLearningResult with the given strategy and reward.
    fn make_result(strategy: ResponseStrategy, reward: f32) -> CycleLearningResult {
        CycleLearningResult {
            reward,
            strategy_used: strategy,
            successful: reward > 0.0,
            prediction_error: (1.0 - reward).abs(),
            coherence: 0.5,
        }
    }

    #[test]
    fn test_default_values() {
        let cll = ClosedLearningLoop::default();
        assert_eq!(
            cll.q_values, [Q_VALUE_INITIAL; 5],
            "default Q-values should be Q_VALUE_INITIAL"
        );
        assert!(
            (cll.exploration_rate() - EXPLORATION_RATE_INITIAL).abs() < f32::EPSILON,
            "default exploration_rate should be EXPLORATION_RATE_INITIAL"
        );
        assert_eq!(cll.total_interactions(), 0);
    }

    #[test]
    fn test_q_value_update() {
        let mut cll = ClosedLearningLoop::default();
        let old_q = cll.q_values()[0]; // Detailed index 0
        assert!((old_q - Q_VALUE_INITIAL).abs() < f32::EPSILON);

        // Give a strong positive reward for Detailed
        cll.update(make_result(ResponseStrategy::Detailed, 0.9));

        let new_q = cll.q_values()[0];
        assert!(
            new_q > old_q,
            "Q-value for Detailed should increase after positive reward: {new_q} > {old_q}"
        );
        // Q = 0.5 + 0.1 * (0.9 - 0.5) = 0.54
        assert!(
            (new_q - 0.54).abs() < 1e-5,
            "Q-update should follow Q(s,a) <- Q(s,a) + α*(r - Q(s,a)): got {new_q}"
        );
    }

    #[test]
    fn test_exploration_rate_decay() {
        let mut cll = ClosedLearningLoop::default();
        let initial_rate = cll.exploration_rate();
        assert!((initial_rate - EXPLORATION_RATE_INITIAL).abs() < f32::EPSILON);

        for _ in 0..100 {
            cll.update(make_result(ResponseStrategy::Supportive, 0.5));
        }

        let decayed = cll.exploration_rate();
        assert!(
            decayed < initial_rate,
            "exploration_rate should decay: {decayed} < {initial_rate}"
        );
        assert!(
            decayed >= EXPLORATION_RATE_MIN,
            "exploration_rate should not drop below EXPLORATION_RATE_MIN: {decayed}"
        );
    }

    #[test]
    fn test_average_reward() {
        let mut cll = ClosedLearningLoop::default();
        assert!(
            cll.average_reward().abs() < f32::EPSILON,
            "average_reward should be 0.0 with no interactions"
        );

        cll.update(make_result(ResponseStrategy::Concise, 0.6));
        cll.update(make_result(ResponseStrategy::Concise, 0.4));
        cll.update(make_result(ResponseStrategy::Concise, 0.8));

        let avg = cll.average_reward();
        let expected = (0.6 + 0.4 + 0.8) / 3.0;
        assert!(
            (avg - expected).abs() < 1e-5,
            "average_reward should be {expected}, got {avg}"
        );
    }

    #[test]
    fn test_best_strategy() {
        let mut cll = ClosedLearningLoop::with_rng(rng_from_seed(42));

        // Repeatedly reward Exploratory (index 4) to make its Q-value dominant.
        for _ in 0..50 {
            cll.update(make_result(ResponseStrategy::Exploratory, 1.0));
        }

        assert_eq!(
            cll.best_strategy(),
            ResponseStrategy::Exploratory,
            "best_strategy should be Exploratory after many positive updates"
        );
    }

    #[test]
    fn test_phi_gating_high_phi() {
        // High phi (>=0.6): Supportive→Exploratory, Concise→Detailed
        let mut cll = ClosedLearningLoop::with_rng(rng_from_seed(100));

        // Seed a strong positive result for Supportive so step 2 sticks with it
        cll.update(make_result(ResponseStrategy::Supportive, 0.8));
        let strategy = cll.select_strategy(0.8, None);
        // Step 2 sticks with Supportive (reward > 0.5), then phi-gate upgrades
        assert_eq!(
            strategy,
            ResponseStrategy::Exploratory,
            "high phi should upgrade Supportive→Exploratory"
        );

        // Now test Concise→Detailed
        let mut cll2 = ClosedLearningLoop::with_rng(rng_from_seed(200));
        cll2.update(make_result(ResponseStrategy::Concise, 0.8));
        let strategy2 = cll2.select_strategy(0.8, None);
        assert_eq!(
            strategy2,
            ResponseStrategy::Detailed,
            "high phi should upgrade Concise→Detailed"
        );
    }

    #[test]
    fn test_phi_gating_low_phi() {
        // Low phi (<0.3): Exploratory→Supportive, Detailed→Concise
        let mut cll = ClosedLearningLoop::with_rng(rng_from_seed(300));

        // Seed a strong positive result for Exploratory so step 2 sticks with it
        cll.update(make_result(ResponseStrategy::Exploratory, 0.8));
        let strategy = cll.select_strategy(0.2, None);
        assert_eq!(
            strategy,
            ResponseStrategy::Supportive,
            "low phi should downgrade Exploratory→Supportive"
        );

        // Now test Detailed→Concise
        let mut cll2 = ClosedLearningLoop::with_rng(rng_from_seed(400));
        cll2.update(make_result(ResponseStrategy::Detailed, 0.8));
        let strategy2 = cll2.select_strategy(0.2, None);
        assert_eq!(
            strategy2,
            ResponseStrategy::Concise,
            "low phi should downgrade Detailed→Concise"
        );
    }

    #[test]
    fn test_positive_reward_repeats_strategy() {
        let mut cll = ClosedLearningLoop::with_rng(rng_from_seed(500));

        // Give a strong positive reward for Clarifying
        cll.update(make_result(ResponseStrategy::Clarifying, 0.8));

        // Select with neutral phi so phi-gating is a no-op
        let strategy = cll.select_strategy(0.5, None);
        assert_eq!(
            strategy,
            ResponseStrategy::Clarifying,
            "after reward > 0.5, select_strategy should repeat the last successful strategy"
        );
    }

    #[test]
    fn test_negative_reward_switches_strategy() {
        let mut cll = ClosedLearningLoop::with_rng(rng_from_seed(600));

        // Give a negative reward for Clarifying
        cll.update(make_result(ResponseStrategy::Clarifying, -0.5));

        // Select with neutral phi
        let strategy = cll.select_strategy(0.5, None);
        let expected = ResponseStrategy::Clarifying.opposite();
        assert_eq!(
            strategy, expected,
            "after reward < -0.2, select_strategy should use opposite(): expected {expected:?}, got {strategy:?}"
        );
    }

    #[test]
    fn test_reset() {
        let mut cll = ClosedLearningLoop::default();

        // Modify state
        for _ in 0..20 {
            cll.update(make_result(ResponseStrategy::Exploratory, 0.9));
        }
        assert!(cll.total_interactions() > 0);
        assert!(cll.q_values()[4] > Q_VALUE_INITIAL);

        // Reset
        cll.reset();

        assert_eq!(
            cll.q_values, [Q_VALUE_INITIAL; 5],
            "q_values should reset to Q_VALUE_INITIAL"
        );
        assert!(
            (cll.exploration_rate() - EXPLORATION_RATE_INITIAL).abs() < f32::EPSILON,
            "exploration_rate should reset to EXPLORATION_RATE_INITIAL"
        );
        assert_eq!(
            cll.total_interactions(),
            0,
            "total_interactions should reset to 0"
        );
        assert!(
            cll.average_reward().abs() < f32::EPSILON,
            "average_reward should reset to 0.0"
        );
        assert_eq!(
            cll.strategy_counts(),
            &[0; 5],
            "strategy_counts should reset to zeros"
        );
        assert!(
            cll.last_result.is_none(),
            "last_result should reset to None"
        );
    }

    #[test]
    fn test_strategy_counts_tracked() {
        let mut cll = ClosedLearningLoop::default();

        cll.update(make_result(ResponseStrategy::Detailed, 0.5));
        cll.update(make_result(ResponseStrategy::Detailed, 0.5));
        cll.update(make_result(ResponseStrategy::Concise, 0.5));
        cll.update(make_result(ResponseStrategy::Exploratory, 0.5));
        cll.update(make_result(ResponseStrategy::Exploratory, 0.5));
        cll.update(make_result(ResponseStrategy::Exploratory, 0.5));

        let counts = cll.strategy_counts();
        assert_eq!(counts[0], 2, "Detailed count should be 2");
        assert_eq!(counts[1], 1, "Concise count should be 1");
        assert_eq!(counts[2], 0, "Clarifying count should be 0");
        assert_eq!(counts[3], 0, "Supportive count should be 0");
        assert_eq!(counts[4], 3, "Exploratory count should be 3");
    }

    #[test]
    fn test_with_rng_deterministic() {
        let mut cll_a = ClosedLearningLoop::with_rng(rng_from_seed(42));
        let mut cll_b = ClosedLearningLoop::with_rng(rng_from_seed(42));

        let mut strategies_a = Vec::new();
        let mut strategies_b = Vec::new();

        for i in 0..20 {
            let phi = 0.5; // neutral phi to avoid gating interference
            let sa = cll_a.select_strategy(phi, None);
            let sb = cll_b.select_strategy(phi, None);
            strategies_a.push(sa);
            strategies_b.push(sb);

            // Feed identical results so state stays in sync
            let reward = (i as f32 % 5.0) * 0.2 - 0.4; // varying rewards
            cll_a.update(make_result(sa, reward));
            cll_b.update(make_result(sb, reward));
        }

        assert_eq!(
            strategies_a, strategies_b,
            "two loops with identical ShakeRng seeds should produce identical strategy sequences"
        );
    }

    #[test]
    fn test_plasticity_modulates_learning_rate() {
        // High plasticity should update Q-values faster than low plasticity.
        let mut cll_high = ClosedLearningLoop::with_rng(rng_from_seed(42));
        let mut cll_low = ClosedLearningLoop::with_rng(rng_from_seed(42));
        let mut cll_neutral = ClosedLearningLoop::with_rng(rng_from_seed(42));

        let result = make_result(ResponseStrategy::Exploratory, 0.9);
        cll_high.update_with_plasticity(result.clone(), 2.0);
        cll_low.update_with_plasticity(result.clone(), 0.5);
        cll_neutral.update_with_plasticity(result, 1.0);

        let q_high = cll_high.q_values()[4];
        let q_low = cll_low.q_values()[4];
        let q_neutral = cll_neutral.q_values()[4];

        assert!(
            q_high > q_neutral,
            "high plasticity should learn faster: {q_high} > {q_neutral}"
        );
        assert!(
            q_neutral > q_low,
            "low plasticity should learn slower: {q_neutral} > {q_low}"
        );
    }

    #[test]
    fn test_plasticity_clamped() {
        // Extreme plasticity values should be clamped to [0.5, 2.0].
        let mut cll_extreme = ClosedLearningLoop::with_rng(rng_from_seed(42));
        let mut cll_max = ClosedLearningLoop::with_rng(rng_from_seed(42));

        let result = make_result(ResponseStrategy::Detailed, 0.7);
        cll_extreme.update_with_plasticity(result.clone(), 100.0); // Should clamp to 2.0
        cll_max.update_with_plasticity(result, 2.0);

        assert!(
            (cll_extreme.q_values()[0] - cll_max.q_values()[0]).abs() < 1e-6,
            "plasticity > 2.0 should clamp to 2.0"
        );
    }
}
