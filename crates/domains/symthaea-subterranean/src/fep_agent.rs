// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::encoder::normalized_channels;
use crate::types::{NUM_STATE_CHANNELS, SubterraneanState};
use symthaea_fep::Observation;

pub struct SubterraneanFepResult {
    pub tau_factor: f32,
    pub free_energy: f64,
    pub observation_precision: f64,
}

/// Reference implementation for Tier 1 of
/// SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md: `tick()` now runs the
/// agent's real perception step (`perceive()` — variational belief update,
/// genuine free-energy computation) instead of a hand-rolled L2 norm of the
/// raw state vector. `free_energy`/`tau_factor` are read from the agent's
/// own `current_free_energy()`/`is_surprised()`, not fabricated locally.
/// `select_action`/`act`/`learn_from_outcome` are NOT used here — this
/// platform's motor command comes from a separate HDC-LTC controller, not
/// from FEP action selection, so only the perception half of the agent
/// applies. `obs_dim` matches `NUM_STATE_CHANNELS` so the full normalized
/// state feeds the belief update (the previous `obs_dim: 6` against a 32-channel state
/// silently used only the first 6 channels' worth of likelihood-matrix
/// rows/columns — dimension mismatch that made even a "real" wiring
/// attempt here quietly wrong).
pub struct ActiveInferenceSubterraneanAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    tick_count: u64,
}

impl ActiveInferenceSubterraneanAgent {
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
    pub fn tick(&mut self, state: &SubterraneanState) -> SubterraneanFepResult {
        self.tick_with_precision(state, 1.0)
    }

    pub fn tick_with_precision(
        &mut self,
        state: &SubterraneanState,
        observation_precision: f64,
    ) -> SubterraneanFepResult {
        self.tick_count += 1;
        let observation_precision = if observation_precision.is_finite() {
            observation_precision.clamp(0.05, 1.0)
        } else {
            0.05
        };
        let observation = Observation {
            values: normalized_channels(state).to_vec(),
            precision: observation_precision,
            timestamp: self.tick_count,
            modality: "subterranean".to_string(),
        };
        self.agent.perceive(&observation);
        let fe = self.agent.current_free_energy();
        let tau = if self.agent.is_surprised() { 0.85 } else { 1.0 };
        SubterraneanFepResult {
            tau_factor: tau,
            free_energy: fe,
            observation_precision,
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
impl Default for ActiveInferenceSubterraneanAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fep_home() {
        let mut a = ActiveInferenceSubterraneanAgent::new();
        let r = a.tick(&SubterraneanState::home());
        assert!(r.free_energy.is_finite());
    }

    #[test]
    fn test_fep_free_energy_is_not_a_raw_l2_norm() {
        // Regression: the old fake implementation was exactly
        // `state.channels.iter().map(|v| v.powi(2)).sum::<f64>().sqrt()`.
        // The real agent's variational free energy must differ from that
        // quantity (it starts from near-zero belief-prediction error at a
        // fresh agent's prior, not from the raw observation magnitude).
        let mut a = ActiveInferenceSubterraneanAgent::new();
        let state = SubterraneanState::home();
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
        // The agent's belief state must actually update in response to
        // perceive() — proof this isn't a no-op wrapper around a fake calc.
        let mut a = ActiveInferenceSubterraneanAgent::new();
        let belief_before = a.agent.belief.mean.clone();
        let mut hot_state = SubterraneanState::home();
        // Push every channel to an extreme value, far from any sane prior.
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

    #[test]
    fn supplied_observation_precision_is_bounded_and_reported() {
        let mut agent = ActiveInferenceSubterraneanAgent::new();
        let result = agent.tick_with_precision(&SubterraneanState::home(), 0.2);
        assert_eq!(result.observation_precision, 0.2);
        let invalid = agent.tick_with_precision(&SubterraneanState::home(), f64::NAN);
        assert_eq!(invalid.observation_precision, 0.05);
    }
}
