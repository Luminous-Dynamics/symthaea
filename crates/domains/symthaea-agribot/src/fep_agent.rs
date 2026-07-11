// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{AgribotState, NUM_STATE_CHANNELS};
use symthaea_fep::Observation;

pub struct AgribotFepResult {
    pub tau_factor: f32,
    pub free_energy: f64,
}

/// See SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md Tier 1 — mirrors the
/// subterranean reference fix: `tick()` runs the agent's real `perceive()`
/// step instead of a hand-rolled L2 norm, and `obs_dim` matches
/// `NUM_STATE_CHANNELS` so the full state feeds the belief update (the
/// previous `obs_dim: 7` against a 30-channel state silently underused it).
pub struct ActiveInferenceAgribotAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    tick_count: u64,
}

impl ActiveInferenceAgribotAgent {
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: NUM_STATE_CHANNELS,
            num_actions: 1, // unused: no select_action/act in this wiring
            belief_learning_rate: 0.1,
            planning_horizon: 1,
            action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(config),
            tick_count: 0,
        }
    }
    pub fn tick(&mut self, state: &AgribotState) -> AgribotFepResult {
        self.tick_count += 1;
        let observation = Observation {
            values: state.channels.to_vec(),
            precision: 1.0,
            timestamp: self.tick_count,
            modality: "agribot".to_string(),
        };
        self.agent.perceive(&observation);
        let fe = self.agent.current_free_energy();
        let tau = if self.agent.is_surprised() { 0.85 } else { 1.0 };
        AgribotFepResult {
            tau_factor: tau,
            free_energy: fe,
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
impl Default for ActiveInferenceAgribotAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fep_home() {
        let mut a = ActiveInferenceAgribotAgent::new();
        let r = a.tick(&AgribotState::home());
        assert!(r.free_energy.is_finite());
    }

    #[test]
    fn test_fep_free_energy_is_not_a_raw_l2_norm() {
        let mut a = ActiveInferenceAgribotAgent::new();
        let state = AgribotState::home();
        let fake_l2_norm: f64 = state.channels.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        let r = a.tick(&state);
        assert!(
            (r.free_energy - fake_l2_norm).abs() > 1e-6,
            "free_energy ({}) must not equal the old fabricated L2 norm ({})",
            r.free_energy,
            fake_l2_norm
        );
    }

    #[test]
    fn test_fep_perceiving_changes_agent_belief() {
        let mut a = ActiveInferenceAgribotAgent::new();
        let belief_before = a.agent.belief.mean.clone();
        let mut hot_state = AgribotState::home();
        for v in &mut hot_state.channels {
            *v = 1.0;
        }
        for _ in 0..10 {
            a.tick(&hot_state);
        }
        assert_ne!(
            belief_before, a.agent.belief.mean,
            "belief must change after repeated perception of an extreme observation"
        );
    }
}
