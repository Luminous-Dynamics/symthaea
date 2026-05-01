// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::BiotaState;

pub struct BiotaFepResult {
    pub tau_factor: f32,
    pub free_energy: f64,
}

pub struct ActiveInferenceBiotaAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
}

impl ActiveInferenceBiotaAgent {
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 6,
            obs_dim: 6,
            num_actions: 6,
            belief_learning_rate: 0.1,
            planning_horizon: 1,
            action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(config),
        }
    }

    pub fn tick(&mut self, state: &BiotaState) -> BiotaFepResult {
        let deviation: f64 = state.channels.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        let fe = deviation;
        let tau = if state.distress_signal() > 0.6 || state.path_conflict_risk() > 0.6 {
            0.8
        } else {
            1.0
        };
        BiotaFepResult {
            tau_factor: tau,
            free_energy: fe,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for ActiveInferenceBiotaAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_home() {
        let mut a = ActiveInferenceBiotaAgent::new();
        let r = a.tick(&BiotaState::home());
        assert!(r.free_energy.is_finite());
    }
}
