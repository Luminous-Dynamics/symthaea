// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{ExoskeletonState, NUM_STATE_CHANNELS};
use symthaea_fep::Observation;

pub struct ExoFepResult {
    pub tau_factor: f32,
    pub learning_rate_factor: f32,
    pub free_energy: f64,
    pub assistance_recommendation: f32,
}

/// See SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md -- same fix
/// pattern as the quadruped reference: `tick()` now runs a genuine
/// perception step instead of a hand-rolled heuristic, and `obs_dim` matches
/// `NUM_STATE_CHANNELS` (was mismatched at 6 against 28). `tau_factor`
/// (center-of-pressure gating) and `assistance_recommendation` (effort-based)
/// are legitimate domain-specific signals kept as-is; only
/// `learning_rate_factor`'s threshold logic depended on the old fabricated
/// free energy, so it now uses a plainly-named local deviation estimate
/// instead of masquerading as FEP output.
pub struct ActiveInferenceExoAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    tick_count: u64,
}

impl ActiveInferenceExoAgent {
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
    pub fn tick(&mut self, state: &ExoskeletonState) -> ExoFepResult {
        self.tick_count += 1;
        let values: Vec<f64> = state.to_channels().iter().map(|v| *v as f64).collect();
        let observation = Observation {
            values,
            precision: 1.0,
            timestamp: self.tick_count,
            modality: "exoskeleton".to_string(),
        };
        self.agent.perceive(&observation);

        let effort: f64 = state.human_torques.iter().map(|t| t.abs()).sum();
        let cop =
            (state.center_of_pressure[0].powi(2) + state.center_of_pressure[1].powi(2)).sqrt();
        // Local deviation estimate for the learning-rate gate below -- kept
        // separate from the agent's real free_energy field.
        let deviation_estimate = effort * 0.01 + cop * 10.0;
        let tau = if cop > 0.08 {
            0.85
        } else if cop < 0.03 {
            1.15
        } else {
            1.0
        };
        let lr = if deviation_estimate > 10.0 {
            1.5
        } else if deviation_estimate < 2.0 {
            0.6
        } else {
            1.0
        };
        let assist = (effort as f32 / 50.0).clamp(0.2, 0.8);
        ExoFepResult {
            tau_factor: tau,
            learning_rate_factor: lr,
            free_energy: self.agent.current_free_energy(),
            assistance_recommendation: assist,
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
impl Default for ActiveInferenceExoAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_home() {
        let mut a = ActiveInferenceExoAgent::new();
        let r = a.tick(&ExoskeletonState::standing());
        assert!(r.free_energy.is_finite());
    }

    #[test]
    fn test_fep_free_energy_is_not_the_old_fabricated_heuristic() {
        let mut a = ActiveInferenceExoAgent::new();
        let state = ExoskeletonState::standing();
        let effort: f64 = state.human_torques.iter().map(|t| t.abs()).sum();
        let cop =
            (state.center_of_pressure[0].powi(2) + state.center_of_pressure[1].powi(2)).sqrt();
        let fake_fe = effort * 0.01 + cop * 10.0;
        let r = a.tick(&state);
        assert!(
            (r.free_energy - fake_fe).abs() > 1e-6,
            "free_energy ({}) must not equal the old fabricated heuristic ({})",
            r.free_energy,
            fake_fe
        );
    }

    #[test]
    fn test_fep_perceiving_changes_agent_belief() {
        let mut a = ActiveInferenceExoAgent::new();
        let belief_before = a.agent.belief.mean.clone();
        let mut strained_state = ExoskeletonState::standing();
        for v in &mut strained_state.human_torques {
            *v = 50.0;
        }
        strained_state.center_of_pressure = [0.5, 0.5];
        for _ in 0..10 {
            a.tick(&strained_state);
        }
        assert_ne!(
            belief_before, a.agent.belief.mean,
            "belief must change after repeated perception of an extreme observation"
        );
    }
}
