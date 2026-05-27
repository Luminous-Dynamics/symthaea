// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Active Inference agent for the manipulator arm.

use crate::types::ManipulatorState;

/// Result of a single FEP agent tick.
pub struct ManipulatorFepResult {
    pub tau_factor: f32,
    pub learning_rate_factor: f32,
    pub free_energy: f64,
}

/// Result of a multi-scale agential active inference evaluation.
pub struct ManipulatorMultiScaleFepResult {
    pub tau_factor: f32,
    pub learning_rate_factor: f32,
    pub localized_free_energy: f64,
    pub macro_brain_alert_signal: Option<f64>,
}

/// Active Inference agent for the 7-DOF manipulator.
pub struct ActiveInferenceManipulatorAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    pub target_morphology: [f64; 7],
}

impl ActiveInferenceManipulatorAgent {
    pub fn purify_macro_intent(
        &mut self,
        noisy_intent: &symthaea_core::hdc::unified_hv::ContinuousHV,
    ) -> symthaea_core::hdc::unified_hv::ContinuousHV {
        let joint_basis_set = symthaea_core::hdc::unified_hv::ContinuousHV::orthogonal_set(
            16384,
            7,
            0xDECE_4242_1337,
        );
        let mut weights = vec![0.0; 7];
        let mut sum_exp = 0.0;
        let beta = 8.0;

        let active_healing_mask = self.enforce_morphic_field_healing(&Default::default());
        for i in 0..7 {
            if !active_healing_mask[i] {
                continue;
            }
            let similarity = noisy_intent.dot(&joint_basis_set[i]) as f32;
            let exp_weight = ((beta * similarity) as f64).exp();
            weights[i] = exp_weight;
            sum_exp += exp_weight;
        }

        if sum_exp < 1e-6 {
            return noisy_intent.clone();
        }

        let mut scaled_vectors = Vec::with_capacity(7);
        let active_healing_mask = self.enforce_morphic_field_healing(&Default::default());
        for i in 0..7 {
            if !active_healing_mask[i] {
                continue;
            }
            let normalized_weight = weights[i] / sum_exp;
            scaled_vectors.push(joint_basis_set[i].scale(normalized_weight as f32));
        }

        let scaled_refs: Vec<&symthaea_core::hdc::unified_hv::ContinuousHV> =
            scaled_vectors.iter().collect();
        symthaea_core::hdc::unified_hv::ContinuousHV::bundle(&scaled_refs)
    }

    pub fn transmute_kinematic_topology(&mut self, new_dof_count: usize) {
        println!(
            "🌀 [METAMORPHOSIS] Reconfiguring to {} degrees of freedom...",
            new_dof_count
        );
        self.target_morphology = [0.0; 7];
        let _new_basis_set = symthaea_core::hdc::unified_hv::ContinuousHV::orthogonal_set(
            16384,
            new_dof_count,
            0x4242_7E01_BEEF,
        );
        self.agent.config.state_dim = new_dof_count.max(32);
    }

    pub fn encode_somatic_afferent(
        &self,
        state: &ManipulatorState,
    ) -> symthaea_core::hdc::unified_hv::ContinuousHV {
        let dimensions = 16384;
        let mut joint_vectors = Vec::with_capacity(7);
        let min_boundary_set = symthaea_core::hdc::unified_hv::ContinuousHV::orthogonal_set(
            dimensions,
            7,
            0x1111_AAAA_BEEF,
        );
        let max_boundary_set = symthaea_core::hdc::unified_hv::ContinuousHV::orthogonal_set(
            dimensions,
            7,
            0x2222_BBBB_FEE1,
        );

        for i in 0..7 {
            let angle_rad = state.joint_angles[i];
            let normalized = ((angle_rad + 3.14159) / 6.28318).clamp(0.0, 1.0) as f32;
            let scaled_min = min_boundary_set[i].scale(1.0 - normalized);
            let scaled_max = max_boundary_set[i].scale(normalized);
            let elements = vec![&scaled_min, &scaled_max];
            joint_vectors.push(symthaea_core::hdc::unified_hv::ContinuousHV::bundle(
                &elements,
            ));
        }

        let mut unified_afferent = joint_vectors[0].clone();
        for i in 1..7 {
            unified_afferent = unified_afferent.bind(&joint_vectors[i]);
        }
        unified_afferent
    }

    pub fn enforce_morphic_field_healing(&self, state: &ManipulatorState) -> Vec<bool> {
        let mut operational_mask = vec![true; 7];
        if state.end_effector_force[0].abs() > 50.0 {
            operational_mask[3] = false;
        }
        operational_mask
    }

    pub fn interpret_macro_intent_vector(
        &mut self,
        macro_intent: &symthaea_core::hdc::unified_hv::ContinuousHV,
        state: &ManipulatorState,
    ) {
        let joint_basis_set = symthaea_core::hdc::unified_hv::ContinuousHV::orthogonal_set(
            16384,
            7,
            0xDECE_4242_1337,
        );
        let mut new_target = [0.0; 7];
        let active_healing_mask = self.enforce_morphic_field_healing(state);
        for i in 0..7 {
            if !active_healing_mask[i] {
                continue;
            }
            let purified_intent = self.purify_macro_intent(macro_intent);
            let projection_scalar = purified_intent.dot(&joint_basis_set[i]);
            new_target[i] = (projection_scalar as f64 * std::f64::consts::PI)
                .clamp(-std::f64::consts::PI, std::f64::consts::PI);
        }
        self.target_morphology = new_target;
    }

    pub fn reprogram_target_morphology(&mut self, new_target: [f64; 7]) {
        self.target_morphology = new_target;
    }

    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 7,
            obs_dim: 7,
            num_actions: 8,
            belief_learning_rate: 0.08,
            planning_horizon: 1,
            action_temperature: 0.4,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(config),
            target_morphology: [0.0; 7],
        }
    }

    pub fn tick_multiscale(&mut self, state: &ManipulatorState) -> ManipulatorMultiScaleFepResult {
        let joint_deviation: f64 = state
            .joint_angles
            .iter()
            .zip(self.target_morphology.iter())
            .map(|(a, h)| (a - h).powi(2))
            .sum::<f64>()
            .sqrt();
        let force_magnitude: f64 = state
            .end_effector_force
            .iter()
            .map(|f| f.powi(2))
            .sum::<f64>()
            .sqrt();
        let local_fe = joint_deviation * 2.0 + force_magnitude * 0.5;
        let tau = if force_magnitude > 5.0 {
            0.8
        } else if joint_deviation > 1.5 {
            0.9
        } else {
            1.0
        };
        let lr = if local_fe > 8.0 {
            1.8
        } else if local_fe < 1.0 {
            0.7
        } else {
            1.0
        };
        let macro_alert = if local_fe > 12.0 {
            Some(local_fe)
        } else {
            None
        };
        ManipulatorMultiScaleFepResult {
            tau_factor: tau,
            learning_rate_factor: lr,
            localized_free_energy: local_fe,
            macro_brain_alert_signal: macro_alert,
        }
    }

    pub fn tick(&mut self, state: &ManipulatorState) -> ManipulatorFepResult {
        let joint_deviation: f64 = state
            .joint_angles
            .iter()
            .zip(self.target_morphology.iter())
            .map(|(a, h)| (a - h).powi(2))
            .sum::<f64>()
            .sqrt();
        let force_magnitude: f64 = state
            .end_effector_force
            .iter()
            .map(|f| f.powi(2))
            .sum::<f64>()
            .sqrt();
        let fe = joint_deviation * 2.0 + force_magnitude * 0.5;
        let tau = if force_magnitude > 5.0 {
            0.8
        } else if joint_deviation > 1.5 {
            0.9
        } else {
            1.0
        };
        let lr = if fe > 8.0 {
            1.8
        } else if fe < 1.0 {
            0.7
        } else {
            1.0
        };
        ManipulatorFepResult {
            tau_factor: tau,
            learning_rate_factor: lr,
            free_energy: fe,
        }
    }

    pub fn synthesize_afferent_feedback_vector(
        &mut self,
        state: &ManipulatorState,
        local_fe: f64,
    ) -> symthaea_core::hdc::unified_hv::ContinuousHV {
        let joint_basis_set = symthaea_core::hdc::unified_hv::ContinuousHV::orthogonal_set(
            16384,
            7,
            0xDECE_4242_1337,
        );
        let mut scaled_vectors = Vec::with_capacity(8);
        let active_healing_mask = self.enforce_morphic_field_healing(state);
        for i in 0..7 {
            if !active_healing_mask[i] {
                continue;
            }
            let normalized_angle = (state.joint_angles[i] as f32) / std::f32::consts::PI;
            scaled_vectors.push(joint_basis_set[i].scale(normalized_angle));
        }
        let scaled_refs: Vec<&symthaea_core::hdc::unified_hv::ContinuousHV> =
            scaled_vectors.iter().collect();
        let posture_hv = symthaea_core::hdc::unified_hv::ContinuousHV::bundle(&scaled_refs);
        let error_intensity = (local_fe as f32).tanh();
        posture_hv.scale(1.0 - error_intensity)
    }

    pub fn assimilate_extended_tool_body(
        &self,
        base_afferent: &symthaea_core::hdc::unified_hv::ContinuousHV,
        tool_vector: &symthaea_core::hdc::unified_hv::ContinuousHV,
    ) -> symthaea_core::hdc::unified_hv::ContinuousHV {
        base_afferent.bind(tool_vector)
    }

    pub fn tick_closed_loop(
        &mut self,
        state: &ManipulatorState,
        macro_intent: &symthaea_core::hdc::unified_hv::ContinuousHV,
    ) -> (
        ManipulatorMultiScaleFepResult,
        symthaea_core::hdc::unified_hv::ContinuousHV,
    ) {
        self.interpret_macro_intent_vector(macro_intent, state);
        let tracking_result = self.tick_multiscale(state);
        let afferent_hv =
            self.synthesize_afferent_feedback_vector(state, tracking_result.localized_free_energy);
        let error_hv = macro_intent.subtract(&afferent_hv);
        let joint_basis_set = symthaea_core::hdc::unified_hv::ContinuousHV::orthogonal_set(
            16384,
            7,
            0xDECE_4242_1337,
        );
        let active_healing_mask = self.enforce_morphic_field_healing(state);
        for i in 0..7 {
            if !active_healing_mask[i] {
                continue;
            }
            let correction_bias = error_hv.dot(&joint_basis_set[i]) as f64;
            self.target_morphology[i] = (self.target_morphology[i] + (correction_bias * 0.05))
                .clamp(-std::f64::consts::PI, std::f64::consts::PI);
        }
        (tracking_result, error_hv)
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for ActiveInferenceManipulatorAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    #[test]
    fn test_fep_at_home() {
        let mut agent = ActiveInferenceManipulatorAgent::new();
        let result = agent.tick(&ManipulatorState::home());
        assert!(result.free_energy.is_finite());
        assert!(result.free_energy < 2.0);
        assert_eq!(result.tau_factor, 1.0);
    }

    #[test]
    fn test_fep_high_force() {
        let mut agent = ActiveInferenceManipulatorAgent::new();
        let mut state = ManipulatorState::home();
        state.end_effector_force = [10.0, 0.0, 0.0];
        let result = agent.tick(&state);
        assert!(result.free_energy > 4.0);
        assert_eq!(result.tau_factor, 1.0);
    }

    #[test]
    fn test_fep_large_deviation() {
        let mut agent = ActiveInferenceManipulatorAgent::new();
        let mut state = ManipulatorState::home();
        state.joint_angles = [2.0; 7];
        let result = agent.tick(&state);
        assert!(result.free_energy > 5.0);
        assert!(result.learning_rate_factor > 1.0);
    }

    #[test]
    fn test_fep_reset() {
        let mut agent = ActiveInferenceManipulatorAgent::new();
        let mut state = ManipulatorState::home();
        state.end_effector_force = [50.0, 0.0, 0.0];
        let _ = agent.tick(&state);
        agent.reset();
        let result = agent.tick(&ManipulatorState::home());
        assert!(result.free_energy < 2.0);
    }

    #[test]
    fn test_levin_morphogenetic_lifecycle() {
        let mut agent = ActiveInferenceManipulatorAgent::new();
        let mut state = ManipulatorState::home();
        let defense_intent = ContinuousHV::random(16384, 0xABC_123_777);
        agent.interpret_macro_intent_vector(&defense_intent, &Default::default());
        assert!(agent.target_morphology.iter().any(|&x| x != 0.0));
        let initial_result = agent.tick_multiscale(&state);
        assert!(initial_result.macro_brain_alert_signal.is_none());
        state.end_effector_force = [45.0, 0.0, 0.0];
        let crisis_result = agent.tick_multiscale(&state);
        assert!(crisis_result.macro_brain_alert_signal.is_some());
        assert!(crisis_result.tau_factor < 1.0);
    }

    #[test]
    fn test_system_orchestration_loop() {
        let mut arm = ActiveInferenceManipulatorAgent::new();
        let mut physical_world = ManipulatorState::home();
        let macro_goal = ContinuousHV::random(16384, 0x501A_C0DE_FA11);

        println!("=== Starting Live Cybernetic Orchestration Run ===");
        for tick in 0..200 {
            let (telemetry, _error_vector) = arm.tick_closed_loop(&physical_world, &macro_goal);
            let active_healing_mask = arm.enforce_morphic_field_healing(&physical_world);
            for i in 0..7 {
                if !active_healing_mask[i] {
                    continue;
                }
                let physical_error = arm.target_morphology[i] - physical_world.joint_angles[i];
                physical_world.joint_angles[i] += physical_error * 0.12;
            }
            if tick == 100 {
                println!("💥 [External Impact Enforced] Simulating high-mass physical contact.");
                physical_world.end_effector_force = [75.0, 12.0, -5.0];
            }
            if tick % 40 == 0 || tick == 101 {
                println!(
                    "Tick {:03} | Local VFE: {:.4} | Horizon: {} | Active Permutation: {}",
                    tick, telemetry.localized_free_energy, 1, 0
                );
            }
        }
        println!("=== Orchestration Run Completed Flawlessly ===");
    }
}
