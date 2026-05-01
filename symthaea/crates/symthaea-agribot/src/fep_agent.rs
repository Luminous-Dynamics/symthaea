// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::AgribotState;

pub struct AgribotFepResult {
    pub tau_factor: f32,
    pub free_energy: f64,
}

pub struct ActiveInferenceAgribotAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
}

impl ActiveInferenceAgribotAgent {
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 7,
            obs_dim: 7,
            num_actions: 7,
            belief_learning_rate: 0.1,
            planning_horizon: 1,
            action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(config),
        }
    }
    pub fn tick(&mut self, state: &AgribotState) -> AgribotFepResult {
        let deviation: f64 = state.channels.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        let fe = deviation;
        let tau = if deviation > 2.0 { 0.85 } else { 1.0 };
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
}
