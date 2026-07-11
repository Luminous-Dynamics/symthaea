// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Planning Types
//!
//! Core types for the temporal planning subsystem: ForkedState, MCTS
//! configuration, budget tiers, and planning results.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Budget Tiers (hard realtime scheduling)
// ─────────────────────────────────────────────────────────────────────────────

/// Budget tiers for tiered degradation.
///
/// Each tier has a hard time budget. The engine always completes Tier 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BudgetTier {
    /// ≤2ms: conflict + R + Φ_eff + gate + fallback (always completes).
    Tier0,
    /// ≤8ms: + micro-MCTS (K=3-5, N=50, short horizon).
    Tier1,
    /// ≤20ms: + full MCTS + counterfactuals (narrative is best-effort).
    Tier2,
}

impl BudgetTier {
    /// Select tier based on available budget, reliability, and consciousness level (phi).
    pub fn select(available_us: u64, reliability: f64, phi: f64) -> Self {
        if phi < 0.3 || available_us < 2_000 || reliability < 0.2 {
            BudgetTier::Tier0
        } else if phi <= 0.6 || available_us < 8_000 || reliability < 0.5 {
            BudgetTier::Tier1
        } else {
            BudgetTier::Tier2
        }
    }

    /// Maximum allowed microseconds for this tier.
    pub fn budget_us(self) -> u64 {
        match self {
            BudgetTier::Tier0 => 2_000,
            BudgetTier::Tier1 => 8_000,
            BudgetTier::Tier2 => 20_000,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reasoning Budget
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks budget consumption during a reasoning cycle.
#[derive(Debug, Clone)]
pub struct ReasoningBudget {
    /// Selected tier.
    pub tier: BudgetTier,
    /// Start time (microseconds since epoch).
    start_us: u64,
    /// Maximum allowed microseconds.
    max_us: u64,
}

impl ReasoningBudget {
    /// Create a new budget for a reasoning cycle.
    pub fn new(available_us: u64, reliability: f64, phi: f64) -> Self {
        let tier = BudgetTier::select(available_us, reliability, phi);
        Self {
            tier,
            start_us: Self::now_us(),
            max_us: tier.budget_us().min(available_us),
        }
    }

    /// Whether the budget has been exceeded.
    pub fn exceeded(&self) -> bool {
        self.elapsed_us() >= self.max_us
    }

    /// Remaining microseconds.
    pub fn remaining_us(&self) -> u64 {
        let elapsed = self.elapsed_us();
        if elapsed >= self.max_us {
            0
        } else {
            self.max_us - elapsed
        }
    }

    /// Elapsed microseconds.
    pub fn elapsed_us(&self) -> u64 {
        Self::now_us().saturating_sub(self.start_us)
    }

    fn now_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MCTS Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Monte Carlo Tree Search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MctsConfig {
    /// Maximum actions to consider per node (branching factor).
    pub max_actions: usize,
    /// Maximum rollout iterations.
    pub max_iterations: usize,
    /// Maximum rollout depth (horizon).
    pub max_depth: usize,
    /// UCB1 exploration constant.
    pub exploration_constant: f64,
    /// Whether to use dream priors for action selection.
    pub use_dream_priors: bool,
}

impl MctsConfig {
    /// Tier 1 config: micro-MCTS (K=3-5, N=50, short horizon).
    pub fn tier1() -> Self {
        Self {
            max_actions: 5,
            max_iterations: 50,
            max_depth: 3,
            exploration_constant: 1.414,
            use_dream_priors: true,
        }
    }

    /// Tier 2 config: full MCTS (K=10, N=200, longer horizon).
    pub fn tier2() -> Self {
        Self {
            max_actions: 10,
            max_iterations: 200,
            max_depth: 8,
            exploration_constant: 1.414,
            use_dream_priors: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MCTS Result
// ─────────────────────────────────────────────────────────────────────────────

/// Result of MCTS planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MctsResult {
    /// Index of the best action (into the available actions array).
    pub best_action_idx: Option<usize>,
    /// Confidence in the best action [0, 1].
    pub confidence: f64,
    /// Number of iterations completed.
    pub iterations: u32,
    /// Tree size (number of nodes explored).
    pub tree_size: u32,
    /// Expected value of the best action.
    pub expected_value: f64,
    /// Whether the planner ran (vs returned no-plan).
    pub did_plan: bool,
}

impl MctsResult {
    /// No planning was done (EVS below threshold or budget exceeded).
    pub fn no_plan() -> Self {
        Self {
            best_action_idx: None,
            confidence: 0.0,
            iterations: 0,
            tree_size: 0,
            expected_value: 0.0,
            did_plan: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forked State (lightweight simulation state)
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight state fork for MCTS rollouts.
///
/// Shares immutable weights via `Arc` (zero-copy) and clones only
/// the mutable state vectors. This enables O(1) fork creation.
#[derive(Debug, Clone)]
pub struct ForkedState {
    /// Shared network weights (immutable, zero-copy via Arc).
    pub weights: Arc<Vec<f32>>,
    /// Cloned state vectors (mutable per fork).
    pub states: Vec<Vec<f32>>,
    /// Time constants (mutable per fork).
    pub taus: Vec<f32>,
    /// RNG state for deterministic rollouts (INV-3).
    pub rng_state: [u8; 32],
}

impl ForkedState {
    /// Create a new forked state.
    pub fn new(
        weights: Arc<Vec<f32>>,
        states: Vec<Vec<f32>>,
        taus: Vec<f32>,
        seed: [u8; 32],
    ) -> Self {
        Self {
            weights,
            states,
            taus,
            rng_state: seed,
        }
    }

    /// Fork this state (clone states/taus, share weights).
    pub fn fork(&self) -> Self {
        Self {
            weights: Arc::clone(&self.weights),
            states: self.states.clone(),
            taus: self.taus.clone(),
            rng_state: self.rng_state,
        }
    }

    /// Evolve the state by one step with the given input.
    ///
    /// Uses closed-form CfC solution: x(t+dt) = x(t) + f(x, input) * dt
    /// This is O(1) in dt — the key property enabling temporal planning.
    pub fn evolve(&mut self, dt: f32, input: &[f32]) {
        for (state_vec, tau) in self.states.iter_mut().zip(self.taus.iter()) {
            for (i, s) in state_vec.iter_mut().enumerate() {
                let inp = input.get(i).copied().unwrap_or(0.0);
                // Simplified CfC dynamics: dx/dt = (-x + tanh(w*inp)) / tau
                let activation = (inp).tanh();
                let dx = (-*s + activation) / tau;
                *s += dx * dt;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Planned Action
// ─────────────────────────────────────────────────────────────────────────────

/// A candidate action for MCTS evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedAction {
    /// Action identifier.
    pub id: String,
    /// Description for narrative.
    pub description: String,
    /// Action embedding (for dream prior matching).
    pub embedding: Vec<f32>,
    /// Prior probability (from dream feedback, default uniform).
    pub prior: f64,
    /// Whether this action is epistemic (information-gathering).
    pub is_epistemic: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_tier_selection() {
        assert_eq!(BudgetTier::select(1_000, 0.8, 0.8), BudgetTier::Tier0);
        assert_eq!(BudgetTier::select(5_000, 0.8, 0.8), BudgetTier::Tier1);
        assert_eq!(BudgetTier::select(15_000, 0.8, 0.8), BudgetTier::Tier2);
        // Low reliability forces Tier0 even with budget
        assert_eq!(BudgetTier::select(20_000, 0.1, 0.8), BudgetTier::Tier0);
        // Low phi forces Tier0/Tier1
        assert_eq!(BudgetTier::select(20_000, 0.8, 0.2), BudgetTier::Tier0);
        assert_eq!(BudgetTier::select(20_000, 0.8, 0.5), BudgetTier::Tier1);
    }

    #[test]
    fn test_reasoning_budget() {
        let budget = ReasoningBudget::new(10_000, 0.8, 0.8);
        assert_eq!(budget.tier, BudgetTier::Tier2); // 10ms budget + high R + high phi → Tier2
        assert!(!budget.exceeded());
        assert!(budget.remaining_us() > 0);

        // Tier1 with lower budget
        let budget1 = ReasoningBudget::new(5_000, 0.8, 0.8);
        assert_eq!(budget1.tier, BudgetTier::Tier1);
    }

    #[test]
    fn test_forked_state_fork_shares_weights() {
        let weights = Arc::new(vec![1.0, 2.0, 3.0]);
        let states = vec![vec![0.0, 0.0]];
        let taus = vec![1.0];
        let state = ForkedState::new(weights.clone(), states, taus, [42u8; 32]);

        let fork = state.fork();
        // Weights are shared (same Arc)
        assert!(Arc::ptr_eq(&state.weights, &fork.weights));
        // States are independent
        assert_eq!(state.states, fork.states);
    }

    #[test]
    fn test_forked_state_evolve_independence() {
        // TEST-FORK-2: Fork A and Fork B from same snapshot, evolve independently
        let weights = Arc::new(vec![1.0]);
        let states = vec![vec![0.0]];
        let taus = vec![1.0];
        let state = ForkedState::new(weights, states, taus, [42u8; 32]);

        let mut fork_a = state.fork();
        let mut fork_b = state.fork();

        fork_a.evolve(0.1, &[1.0]);
        fork_b.evolve(0.1, &[-1.0]);

        // A and B should diverge
        assert_ne!(fork_a.states[0][0], fork_b.states[0][0]);
        // Original unchanged
        assert_eq!(state.states[0][0], 0.0);
    }

    #[test]
    fn test_mcts_result_no_plan() {
        let result = MctsResult::no_plan();
        assert!(!result.did_plan);
        assert!(result.best_action_idx.is_none());
    }
}
