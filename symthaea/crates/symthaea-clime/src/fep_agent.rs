// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{ClimeState, AIR_QUALITY, CO2_LOAD, THERMAL_STRESS, UTILITY_RESERVE};

pub struct ClimeFepResult {
    pub tau_factor: f32,
    pub free_energy: f64,
}

pub struct ActiveInferenceClimeAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
}

impl ActiveInferenceClimeAgent {
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 8,
            num_actions: 8,
            belief_learning_rate: 0.1,
            planning_horizon: 1,
            action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(config),
        }
    }
    pub fn tick(&mut self, state: &ClimeState) -> ClimeFepResult {
        let deviation: f64 = state.channels.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        let fe = deviation;
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
}
