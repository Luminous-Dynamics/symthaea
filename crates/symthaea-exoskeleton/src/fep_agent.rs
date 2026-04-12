// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::ExoskeletonState;

pub struct ExoFepResult { pub tau_factor: f32, pub learning_rate_factor: f32, pub free_energy: f64, pub assistance_recommendation: f32 }

pub struct ActiveInferenceExoAgent { agent: symthaea_fep::ActiveInferenceAgent }

impl ActiveInferenceExoAgent {
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 6, obs_dim: 6, num_actions: 6, belief_learning_rate: 0.1,
            planning_horizon: 1, action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self { agent: symthaea_fep::ActiveInferenceAgent::new(config) }
    }
    pub fn tick(&mut self, state: &ExoskeletonState) -> ExoFepResult {
        let effort: f64 = state.human_torques.iter().map(|t| t.abs()).sum();
        let cop = (state.center_of_pressure[0].powi(2) + state.center_of_pressure[1].powi(2)).sqrt();
        let fe = effort * 0.01 + cop * 10.0;
        let tau = if cop > 0.08 { 0.85 } else if cop < 0.03 { 1.15 } else { 1.0 };
        let lr = if fe > 10.0 { 1.5 } else if fe < 2.0 { 0.6 } else { 1.0 };
        let assist = (effort as f32 / 50.0).clamp(0.2, 0.8);
        ExoFepResult { tau_factor: tau, learning_rate_factor: lr, free_energy: fe, assistance_recommendation: assist }
    }
    pub fn reset(&mut self) { *self = Self::new(); }
}
impl Default for ActiveInferenceExoAgent { fn default() -> Self { Self::new() } }
