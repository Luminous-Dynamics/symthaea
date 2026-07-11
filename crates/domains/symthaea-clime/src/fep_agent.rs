// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{
    AIR_QUALITY, CO2_LOAD, ClimeState, NUM_STATE_CHANNELS, THERMAL_STRESS, UTILITY_RESERVE,
};
use symthaea_fep::Observation;

pub struct ClimeFepResult {
    pub tau_factor: f32,
    pub free_energy: f64,
}

/// See SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md Tier 1 — mirrors the
/// subterranean reference fix for `free_energy` (real `perceive()` instead of
/// a hand-rolled L2 norm; `obs_dim` matches `NUM_STATE_CHANNELS`, was
/// mismatched at 8 against a 28-channel state). `tau_factor`'s domain-specific
/// CO2/thermal/air-quality/utility-reserve gating is untouched — that's
/// legitimate sensing logic, not part of the FEP-theater bug.
pub struct ActiveInferenceClimeAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    tick_count: u64,
}

impl ActiveInferenceClimeAgent {
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
    pub fn tick(&mut self, state: &ClimeState) -> ClimeFepResult {
        self.tick_count += 1;
        let observation = Observation {
            values: state.channels.to_vec(),
            precision: 1.0,
            timestamp: self.tick_count,
            modality: "clime".to_string(),
        };
        self.agent.perceive(&observation);
        let fe = self.agent.current_free_energy();
        let tau = if state.channels[CO2_LOAD] > 0.6
            || state.channels[THERMAL_STRESS] > 0.6
            || state.channels[AIR_QUALITY] < 0.4
            || state.channels[UTILITY_RESERVE] < 0.2
        {
            0.82
        } else {
            1.0
        };
        ClimeFepResult {
            tau_factor: tau,
            free_energy: fe,
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
impl Default for ActiveInferenceClimeAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fep_home() {
        let mut a = ActiveInferenceClimeAgent::new();
        let r = a.tick(&ClimeState::home());
        assert!(r.free_energy.is_finite());
    }

    #[test]
    fn test_fep_free_energy_is_not_a_raw_l2_norm() {
        let mut a = ActiveInferenceClimeAgent::new();
        let state = ClimeState::home();
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
        let mut a = ActiveInferenceClimeAgent::new();
        let belief_before = a.agent.belief.mean.clone();
        let mut hot_state = ClimeState::home();
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
