// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Applied ToM helpers: FEP-based behavioral prediction from mental states.
//!
//! Instead of just attributing beliefs via HDC geometry, these helpers use
//! the FEP `ActiveInferenceAgent` to predict *behavior* from (false) beliefs.
//! `select_action()` returns what an agent would DO given their belief state,
//! proving behavioral ToM that LLMs lack.
//!
//! Gated behind `#[cfg(feature = "symthaea-backend")]`.

use symthaea_fep::{
    ActionSelectionResult, ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation,
};

/// Creates an FEP agent modeling a social character.
///
/// The agent operates in a small state space suitable for social reasoning:
/// e.g., 2 states (object-at-A vs object-at-B), 2 observations, 2 actions.
pub fn social_agent(state_dim: usize, obs_dim: usize, num_actions: usize) -> ActiveInferenceAgent {
    let config = ActiveInferenceAgentConfig {
        state_dim,
        obs_dim,
        num_actions,
        inference_iterations: 5,
        belief_learning_rate: 0.1,
        planning_horizon: 3,
        action_temperature: 0.5, // Sharper action selection for clearer behavioral predictions
        enable_model_learning: false, // Don't learn during social simulation
        enable_td_learning: false,
        ..Default::default()
    };
    ActiveInferenceAgent::new(config)
}

/// Injects a belief state without updating from observations.
///
/// This models the agent having stale/false beliefs — directly setting
/// `belief.mean` to the given values bypasses the perception update.
pub fn inject_belief(agent: &mut ActiveInferenceAgent, belief_mean: Vec<f64>) {
    assert_eq!(
        belief_mean.len(),
        agent.config.state_dim,
        "belief_mean length must match state_dim"
    );
    agent.belief.mean = belief_mean;
    // Set high precision to indicate confidence in the injected belief
    agent.belief.precision = vec![4.0; agent.config.state_dim];
}

/// Ask the agent for its preferred (MAP) action given current beliefs + goals.
///
/// Returns `(map_action, action_probabilities)`.
///
/// Uses the maximum a posteriori action from the softmax distribution rather
/// than a stochastic sample. For Theory of Mind tasks we care about the agent's
/// *preferred* action — not whether a random draw happened to match.
pub fn predict_behavior(agent: &mut ActiveInferenceAgent) -> (usize, Vec<f64>) {
    let result: ActionSelectionResult = agent.select_action();
    let map_action = result
        .action_probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    (map_action, result.action_probabilities)
}

/// Convenience: create an observation vector from a slice.
pub fn make_observation(values: Vec<f64>, modality: &str) -> Observation {
    Observation::new(values, 1.0, modality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_social_agent_creation() {
        let agent = social_agent(2, 2, 2);
        assert_eq!(agent.config.state_dim, 2);
        assert_eq!(agent.config.obs_dim, 2);
        assert_eq!(agent.config.num_actions, 2);
    }

    #[test]
    fn test_inject_belief() {
        let mut agent = social_agent(2, 2, 2);
        inject_belief(&mut agent, vec![0.9, 0.1]);
        assert!((agent.belief.mean[0] - 0.9).abs() < 1e-10);
        assert!((agent.belief.mean[1] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_predict_behavior_returns_valid() {
        let mut agent = social_agent(2, 2, 2);
        inject_belief(&mut agent, vec![0.9, 0.1]);
        agent.set_goals(vec![1.0, 0.0], 4.0);
        let (action, probs) = predict_behavior(&mut agent);
        assert!(action < 2);
        assert_eq!(probs.len(), 2);
        // Probabilities should sum to ~1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "probs sum = {}", sum);
    }
}
