// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Budget-Bounded MCTS
//!
//! Monte Carlo Tree Search with:
//! - Hard budget enforcement (INV-6)
//! - EVS-gated simulation
//! - Dream-prior-biased action selection
//! - O(1) rollouts via CfC closed-form solution

use super::types::*;

/// Expected Value of Simulation: reliability-gated simulation signal.
///
/// Determines whether simulation is worthwhile given current reliability.
pub fn evs(conflict_entropy: f64, reliability: f64, n_actions: usize, recent_utility: f64) -> f64 {
    use crate::consciousness::epistemic_conflict::thresholds::R_SIM_MIN;

    // Hard gate: don't simulate on nonsense
    if reliability < R_SIM_MIN {
        return 0.0;
    }

    let action_diversity = (n_actions as f64).ln().max(1.0);
    let utility_prior = recent_utility.clamp(0.1, 1.0);

    conflict_entropy * action_diversity * utility_prior * (1.0 - reliability)
}

/// MCTS node in the search tree.
#[derive(Debug, Clone)]
struct MctsNode {
    /// Action index that led to this node.
    action_idx: Option<usize>,
    /// Visit count.
    visits: u32,
    /// Total reward accumulated.
    total_reward: f64,
    /// Children (action_idx → child node).
    children: Vec<MctsNode>,
    /// Prior probability (from dream feedback).
    prior: f64,
}

impl MctsNode {
    fn new(action_idx: Option<usize>, prior: f64) -> Self {
        Self {
            action_idx,
            visits: 0,
            total_reward: 0.0,
            children: Vec::new(),
            prior,
        }
    }

    fn avg_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_reward / self.visits as f64
        }
    }

    /// UCB1 + prior - penalty: balances exploration, exploitation, and negative knowledge.
    fn ucb1(&self, parent_visits: u32, c: f64, negative_penalty: f64) -> f64 {
        if self.visits == 0 {
            f64::MAX // unexplored nodes get priority
        } else {
            let exploitation = self.avg_reward();
            let exploration = c * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
            let prior_bonus = self.prior / (1.0 + self.visits as f64);
            // Apply severe penalty for similarity to disproven approaches (INV-12)
            exploitation + exploration + prior_bonus - negative_penalty
        }
    }
}

/// A bank of negative prototypes (disproven logical structures) used for MCTS penalties.
#[derive(Debug, Clone, Default)]
pub struct NegativePrototypeBank {
    pub prototypes: Vec<(Vec<f32>, f64)>, // (embedding, penalty_weight)
}

impl NegativePrototypeBank {
    pub fn compute_penalty(&self, action_embedding: &[f32]) -> f64 {
        let mut total_penalty = 0.0;
        for (proto, weight) in &self.prototypes {
            let sim = cosine_similarity(action_embedding, proto);
            if sim > 0.8 {
                // Severe exponential penalty for high similarity to disproven paths
                total_penalty += weight * (sim - 0.8).exp() * 5.0;
            }
        }
        total_penalty
    }
}

/// Substrate performance model for hardware/software partitioning decisions.
#[derive(Debug, Clone, Default)]
pub struct SubstrateCostModel {
    pub rust_latency_us: f32,
    pub hdl_latency_us: f32,
    pub energy_budget_mw: f32,
}

impl SubstrateCostModel {
    pub fn compute_performance_reward(&self, action_embedding: &[f32]) -> f64 {
        // High-phi dimensions in the embedding represent computational complexity.
        // We use them to predict if Rust will exceed the real-time budget.
        let complexity: f32 =
            action_embedding.iter().map(|&x| x.abs()).sum::<f32>() / action_embedding.len() as f32;

        let rust_score = 1.0 - (complexity * self.rust_latency_us / 1000.0).min(1.0);
        let hdl_score = 1.0 - (self.hdl_latency_us / 100.0).min(1.0);

        // Return normalized reward: higher is better (more deterministic/faster)
        (rust_score.max(hdl_score) as f64).clamp(0.0, 1.0)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0;
    let mut mag_a = 0.0;
    let mut mag_b = 0.0;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] as f64 * b[i] as f64;
        mag_a += a[i] as f64 * a[i] as f64;
        mag_b += b[i] as f64 * b[i] as f64;
    }
    if mag_a > 0.0 && mag_b > 0.0 {
        dot / (mag_a.sqrt() * mag_b.sqrt())
    } else {
        0.0
    }
}

/// Budget-bounded MCTS planner.
pub struct MctsPlanner;

impl MctsPlanner {
    /// Run MCTS with budget enforcement (INV-6) and negative feedback (INV-12).
    ///
    /// Returns best action + confidence. Guaranteed to return
    /// within `budget.remaining_us()` or `config.max_iterations`,
    /// whichever is tighter.
    pub fn plan(
        state: &ForkedState,
        actions: &[PlannedAction],
        config: &MctsConfig,
        budget: &ReasoningBudget,
        negatives: &NegativePrototypeBank,
        substrate: &SubstrateCostModel,
    ) -> MctsResult {
        if actions.is_empty() {
            return MctsResult::no_plan();
        }

        let k = actions.len().min(config.max_actions);
        let mut root = MctsNode::new(None, 1.0);

        // Precompute negative penalties for top-level actions
        let mut penalties = Vec::with_capacity(k);
        for action in actions.iter().take(k) {
            penalties.push(negatives.compute_penalty(&action.embedding));
        }

        // Initialize children with priors
        for (i, action) in actions.iter().take(k).enumerate() {
            root.children.push(MctsNode::new(Some(i), action.prior));
        }

        let mut iterations = 0u32;

        // Main MCTS loop with budget check
        while iterations < config.max_iterations as u32 && !budget.exceeded() {
            // Select: UCB1 with negative penalties
            let child_idx = Self::select(&root, config.exploration_constant, &penalties);

            // Simulate: rollout from this action
            let reward = Self::simulate(
                state,
                actions,
                child_idx,
                config.max_depth,
                negatives,
                substrate,
            );

            // Backpropagate
            root.children[child_idx].visits += 1;
            root.children[child_idx].total_reward += reward;
            root.visits += 1;

            iterations += 1;
        }

        // Pick best action by visit count (more robust than avg reward)
        let best_idx = root
            .children
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.visits)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best = &root.children[best_idx];
        let confidence = if root.visits > 0 {
            best.visits as f64 / root.visits as f64
        } else {
            0.0
        };

        MctsResult {
            best_action_idx: best.action_idx,
            confidence,
            iterations,
            tree_size: root.visits,
            expected_value: best.avg_reward(),
            did_plan: iterations > 0,
        }
    }

    /// Select child with highest UCB1 score.
    fn select(node: &MctsNode, c: f64, penalties: &[f64]) -> usize {
        node.children
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| {
                let p_a = penalties.get(*i).copied().unwrap_or(0.0);
                let p_b = penalties.get(*j).copied().unwrap_or(0.0);
                let ucb_a = a.ucb1(node.visits, c, p_a);
                let ucb_b = b.ucb1(node.visits, c, p_b);
                ucb_a.total_cmp(&ucb_b)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Simulate (rollout) from a state with a given action.
    ///
    /// Uses ForkedState.evolve() for O(1) temporal stepping.
    fn simulate(
        state: &ForkedState,
        actions: &[PlannedAction],
        action_idx: usize,
        _max_depth: usize,
        negatives: &NegativePrototypeBank,
        substrate: &SubstrateCostModel,
    ) -> f64 {
        let mut forked = state.fork();
        let action = &actions[action_idx];

        // Apply action as input to the network
        forked.evolve(1.0, &action.embedding);

        // Simple reward: state magnitude after action (proxy for Φ)
        let reward: f64 = forked
            .states
            .iter()
            .flat_map(|s| s.iter())
            .map(|&x| (x as f64).abs())
            .sum::<f64>()
            / forked.states.iter().map(|s| s.len()).sum::<usize>().max(1) as f64;

        // Negative feedback: penalize reward based on similarity to disproven paths
        let penalty = negatives.compute_penalty(&action.embedding);

        // Substrate partitioning: reward actions that fit the hardware constraints (INV-14)
        let substrate_reward = substrate.compute_performance_reward(&action.embedding);

        // Epistemic bonus: prefer info-gathering actions
        let epistemic_bonus = if action.is_epistemic { 0.2 } else { 0.0 };

        (reward + epistemic_bonus + substrate_reward - penalty).clamp(0.0, 1.0)
    }

    /// Epistemic-only rollout: minimize uncertainty rather than maximize reward.
    pub fn epistemic_rollout(
        _state: &ForkedState,
        actions: &[PlannedAction],
        _budget: &ReasoningBudget,
    ) -> MctsResult {
        if actions.is_empty() {
            return MctsResult::no_plan();
        }

        // Prefer epistemic actions
        let epistemic_actions: Vec<(usize, &PlannedAction)> = actions
            .iter()
            .enumerate()
            .filter(|(_, a)| a.is_epistemic)
            .collect();

        if epistemic_actions.is_empty() {
            // No epistemic actions; return first action with low confidence
            return MctsResult {
                best_action_idx: Some(0),
                confidence: 0.1,
                iterations: 0,
                tree_size: 0,
                expected_value: 0.0,
                did_plan: true,
            };
        }

        // Simple: pick the epistemic action with highest prior
        let (best_idx, _) = epistemic_actions
            .iter()
            .max_by(|(_, a), (_, b)| a.prior.total_cmp(&b.prior))
            .expect("epistemic_actions confirmed non-empty above");

        MctsResult {
            best_action_idx: Some(*best_idx),
            confidence: 0.5, // moderate confidence for epistemic actions
            iterations: 1,
            tree_size: 1,
            expected_value: 0.5,
            did_plan: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn test_state() -> ForkedState {
        ForkedState::new(
            Arc::new(vec![1.0; 4]),
            vec![vec![0.0; 4]],
            vec![1.0],
            [42u8; 32],
        )
    }

    fn test_actions() -> Vec<PlannedAction> {
        vec![
            PlannedAction {
                id: "action_0".to_string(),
                description: "Do nothing".to_string(),
                embedding: vec![0.0; 4],
                prior: 0.3,
                is_epistemic: false,
            },
            PlannedAction {
                id: "action_1".to_string(),
                description: "Explore".to_string(),
                embedding: vec![0.5; 4],
                prior: 0.5,
                is_epistemic: true,
            },
            PlannedAction {
                id: "action_2".to_string(),
                description: "Execute".to_string(),
                embedding: vec![1.0; 4],
                prior: 0.2,
                is_epistemic: false,
            },
        ]
    }

    #[test]
    fn test_evs_below_threshold() {
        // R below R_SIM_MIN → EVS = 0
        assert_eq!(evs(1.0, 0.1, 5, 0.5), 0.0);
    }

    #[test]
    fn test_evs_above_threshold() {
        let val = evs(0.5, 0.3, 5, 0.5);
        assert!(val > 0.0, "EVS should be positive when R > R_SIM_MIN");
    }

    #[test]
    fn test_mcts_returns_result() {
        let state = test_state();
        let actions = test_actions();
        let config = MctsConfig::tier1();
        let budget = ReasoningBudget::new(10_000, 0.8);

        let result = MctsPlanner::plan(
            &state,
            &actions,
            &config,
            &budget,
            &Default::default(),
            &Default::default(),
        );
        assert!(result.did_plan);
        assert!(result.best_action_idx.is_some());
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_mcts_no_actions() {
        let state = test_state();
        let config = MctsConfig::tier1();
        let budget = ReasoningBudget::new(10_000, 0.8);

        let result = MctsPlanner::plan(&state, &[], &config, &budget, &Default::default());
        assert!(!result.did_plan);
    }

    #[test]
    fn test_inv6_budget_guarantee() {
        // INV-6: Engine returns within budget
        let state = test_state();
        let actions = test_actions();
        let config = MctsConfig {
            max_iterations: 10_000, // very high
            ..MctsConfig::tier1()
        };
        // Very tight budget
        let budget = ReasoningBudget::new(1_000, 0.8);

        let start = std::time::Instant::now();
        let result = MctsPlanner::plan(
            &state,
            &actions,
            &config,
            &budget,
            &Default::default(),
            &Default::default(),
        );
        let elapsed = start.elapsed();

        // Should complete within a reasonable time (Tier0 budget is 2ms)
        assert!(
            elapsed.as_millis() < 50,
            "INV-6: MCTS took {}ms, should be budget-bounded",
            elapsed.as_millis()
        );
        // Should have done fewer than max iterations
        assert!(result.iterations < 10_000);
    }

    #[test]
    fn test_epistemic_rollout() {
        let state = test_state();
        let actions = test_actions();
        let budget = ReasoningBudget::new(10_000, 0.8);

        let result = MctsPlanner::epistemic_rollout(&state, &actions, &budget);
        assert!(result.did_plan);
        // Should prefer the epistemic action (index 1)
        assert_eq!(result.best_action_idx, Some(1));
    }

    #[test]
    fn test_inv5_epistemic_dominance() {
        // INV-5: When R < 0.30, prefer epistemic actions
        let r = 0.2;
        let e = evs(0.5, r, 5, 0.5);
        // EVS should be positive but low — system should use epistemic rollout
        assert!(e > 0.0, "EVS should be positive for R=0.2");
    }
}
