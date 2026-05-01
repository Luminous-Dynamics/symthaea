// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::SurgicalState;
pub struct SurgicalFepResult {
    pub tau_factor: f32,
    pub learning_rate_factor: f32,
    pub free_energy: f64,
    pub anomaly_detected: bool,
}
pub struct ActiveInferenceSurgicalAgent {
    #[allow(dead_code)]
    agent: symthaea_fep::ActiveInferenceAgent,
}
impl ActiveInferenceSurgicalAgent {
    pub fn new() -> Self {
        let c = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 6,
            obs_dim: 6,
            num_actions: 8,
            belief_learning_rate: 0.05,
            planning_horizon: 1,
            action_temperature: 0.3,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(c),
        }
    }
    pub fn tick(&mut self, s: &SurgicalState) -> SurgicalFepResult {
        let f = s.force_magnitude();
        let cd = s.critical_structure_distance;
        let tc = s.trocar_compliance;
        let fe = f * 2.0 + (20.0 - cd).max(0.0) + tc * 50.0;
        let anom = f > 3.0 || cd < 5.0 || tc > 0.3;
        let tau = if anom {
            0.7
        } else if fe < 5.0 {
            1.2
        } else {
            1.0
        };
        let lr = if anom {
            2.0
        } else if fe < 2.0 {
            0.5
        } else {
            1.0
        };
        SurgicalFepResult {
            tau_factor: tau,
            learning_rate_factor: lr,
            free_energy: fe,
            anomaly_detected: anom,
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
impl Default for ActiveInferenceSurgicalAgent {
    fn default() -> Self {
        Self::new()
    }
}
