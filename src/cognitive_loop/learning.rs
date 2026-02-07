//! Closed learning loop for the cognitive loop.
//!
//! Implements Q-learning guided strategy selection with Φ-gating,
//! enabling behavioral adaptation based on prediction error feedback.

use rand::Rng;
use serde::{Serialize, Deserialize};
use symthaea_core::genesis::ShakeRng;

use super::flow::ResponseStrategy;

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
            q_values: [0.5; 5], // Start neutral
            q_learning_rate: 0.1,
            exploration_rate: 0.2,
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
            Some(rng) => (rng.gen::<f32>(), rng.gen::<u8>()),
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
            let best_idx = self.q_values.iter()
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
            if last.reward > 0.5 {
                // Strong positive - stick with what worked
                last.strategy_used
            } else if last.reward < -0.2 {
                // Negative - try opposite strategy
                last.strategy_used.opposite()
            } else {
                base_strategy
            }
        } else {
            base_strategy
        };

        // Step 3: Φ-gating (consciousness influences strategy)
        let final_strategy = if phi >= 0.6 {
            // Integrative mode - favor Exploratory/Detailed
            match strategy {
                ResponseStrategy::Supportive => ResponseStrategy::Exploratory,
                ResponseStrategy::Concise => ResponseStrategy::Detailed,
                other => other,
            }
        } else if phi < 0.3 {
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

    /// Update Q-values with cycle result
    pub fn update(&mut self, result: CycleLearningResult) {
        // Update strategy count
        let strategy_idx = self.strategy_index(result.strategy_used);
        self.strategy_counts[strategy_idx] += 1;

        // Q-learning update: Q(s,a) <- Q(s,a) + α * (r - Q(s,a))
        let old_q = self.q_values[strategy_idx];
        let new_q = old_q + self.q_learning_rate * (result.reward - old_q);
        self.q_values[strategy_idx] = new_q;

        // Update totals
        self.total_interactions += 1;
        self.total_reward += result.reward;

        // Store for next selection
        self.last_result = Some(result);

        // Decay exploration rate over time (but keep minimum of 5%)
        self.exploration_rate = (self.exploration_rate * 0.999).max(0.05);
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
        let best_idx = self.q_values.iter()
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

    /// Get total interactions
    pub fn total_interactions(&self) -> u64 {
        self.total_interactions
    }

    /// Reset the learning loop
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
