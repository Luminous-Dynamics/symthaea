// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::{HumanoidCommand, HumanoidState};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocomotionModule {
    Stand,
    Walk,
    Run,
    Climb,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerativePrior {
    pub target_velocity: f32,
    pub stride_amplitude: f32,
    pub knee_clearance: f32,
    pub target_lean: f32,
}

impl GenerativePrior {
    pub fn zero() -> Self {
        Self {
            target_velocity: 0.0,
            stride_amplitude: 0.0,
            knee_clearance: 0.1,
            target_lean: 0.0,
        }
    }

    pub fn interpolate_towards(&mut self, target: &GenerativePrior, alpha: f32) {
        self.target_velocity += alpha * (target.target_velocity - self.target_velocity);
        self.stride_amplitude += alpha * (target.stride_amplitude - self.stride_amplitude);
        self.knee_clearance += alpha * (target.knee_clearance - self.knee_clearance);
        self.target_lean += alpha * (target.target_lean - self.target_lean);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MachineState {
    ActiveInference,
    Recovering,
}

#[derive(Debug, Clone, Copy)]
pub struct GaitControlProfile {
    pub target_velocity: f32,
    pub stride_amplitude: f32,
    pub knee_clearance: f32,
    pub target_lean: f32,
    pub kp: f64,
    pub kd: f64,
    pub scale: f64,
    pub activation_alpha: f32,
}

impl GaitControlProfile {
    pub fn zero() -> Self {
        Self {
            target_velocity: 0.0,
            stride_amplitude: 0.0,
            knee_clearance: 0.1,
            target_lean: 0.0,
            kp: 42.0, 
            kd: 5.5,
            scale: 40.0,
            activation_alpha: 0.25,
        }
    }

    #[inline(always)]
    fn make_kp_kd_torque(&self, err: f64, vel: f64, joint_idx: usize, stress_modifier: f64, standing_still: bool) -> f64 {
        let dynamic_kp = self.kp * (1.0 + stress_modifier * 0.50);
        let co_contraction = if standing_still { 2.50 } else { 1.0 };
        
        let structural_damping = if joint_idx < 6 { 1.65 } else { 1.0 };
        let pi_v = self.kd * structural_damping * co_contraction;
        
        dynamic_kp * err - pi_v * vel
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CfcCpg {
    pub r_hip_state: f32,
    pub l_hip_state: f32,
    pub r_knee_state: f32,
    pub l_knee_state: f32,
    phase: f32,
    pub postural_integral: f32,
    pub homeostatic_stress: f32, 
    pub reference_joints: [f64; 64], // Fixed-size memory buffer to retain native asset keyframes safely
    pub initialized: bool,
}

impl CfcCpg {
    pub fn zero() -> Self {
        Self {
            r_hip_state: 0.0,
            l_hip_state: 0.0,
            r_knee_state: 0.0,
            l_knee_state: 0.0,
            phase: 0.0,
            postural_integral: 0.0,
            homeostatic_stress: 0.0,
            reference_joints: [0.0; 64],
            initialized: false,
        }
    }

    pub fn update(&mut self, drive_velocity: f32, pitch_lean: f32, _roll_lean: f32, actual_vel_x: f32, dt: f32) {
        let liquid_tau = 0.35f32 + (pitch_lean * 0.40f32).max(0.0f32);
        let velocity_entrainment = (actual_vel_x * 2.0f32).clamp(0.15f32, 1.25f32);
        let continuous_frequency = drive_velocity * 2.0f32 * std::f32::consts::PI * velocity_entrainment;
        
        self.phase += dt * continuous_frequency * liquid_tau;

        let leaky_decay = (-0.85f32 * dt).exp();
        self.postural_integral = (self.postural_integral * leaky_decay + pitch_lean * dt).clamp(-0.25, 0.25);

        let stress_excitation = pitch_lean * 3.5f32; 
        self.homeostatic_stress = (self.homeostatic_stress * (-1.5f32 * dt).exp() + stress_excitation * dt).clamp(0.0f32, 1.0f32);

        let target_r_hip = self.phase.sin();
        let target_l_hip = (self.phase + std::f32::consts::PI).sin();
        let target_r_knee = target_r_hip.max(0.0f32);
        let target_l_knee = target_l_hip.max(0.0f32);

        self.r_hip_state  = decay_gate_filter(self.r_hip_state, target_r_hip, liquid_tau, dt);
        self.l_hip_state  = decay_gate_filter(self.l_hip_state, target_l_hip, liquid_tau, dt);
        self.r_knee_state = decay_gate_filter(self.r_knee_state, target_r_knee, liquid_tau, dt);
        self.l_knee_state = decay_gate_filter(self.l_knee_state, target_l_knee, liquid_tau, dt);
    }
}

#[inline(always)]
fn decay_gate_filter(current: f32, target: f32, tau: f32, dt: f32) -> f32 {
    let gate = (-tau * dt).exp();
    gate * current + (1.0 - gate) * target
}

// -------------------------------------------------------------------
// ASSET-INVARIANT MOTOR COUPLING ENGINE
// Anchors top-down expectations directly onto the model's native keyframe state
// -------------------------------------------------------------------
pub fn execute_modular_gait(
    profile: &GaitControlProfile,
    stride_gain: f32,
    cpg: &mut CfcCpg,
    state: &HumanoidState, 
    cognitive_crouch: f32,
    total_actuators: usize,
    previous_command: &HumanoidCommand,
    dt: f32,
) -> HumanoidCommand {
    // FIXED: Auto-populate neutral standing landmarks on frame zero to guarantee ideal joint angles
    if !cpg.initialized {
        let limit = total_actuators.min(64);
        for i in 0..limit {
            cpg.reference_joints[i] = state.joint_angles[i];
        }
        cpg.initialized = true;
    }

    let mut mu = vec![0.0f64; total_actuators];
    for i in 0..total_actuators {
        if i < 64 {
            mu[i] = cpg.reference_joints[i];
        }
    }

    let actual_vel_x = state.root_linear_velocity[0] as f32;
    let global_orientation_error = (1.0f32 - state.uprightness() as f32).max(0.0f32);
    cpg.update(profile.target_velocity, global_orientation_error, 0.0, actual_vel_x, dt);

    let stress_factor = cpg.homeostatic_stress;
    let swing_amplitude = profile.stride_amplitude as f64 * (1.0 - stress_factor as f64 * 0.50);
    let base_clearance = (profile.knee_clearance + cognitive_crouch) as f64;

    // Apply fluid gait dynamics over the verified reference positions
    let core_spine_boundary = total_actuators / 5; 
    let upper_body_boundary = total_actuators / 2; 

    for i in 0..total_actuators {
        if i < core_spine_boundary {
            mu[i] += (profile.target_lean * (1.0 - stress_factor * 0.60f32)) as f64;
        } else if i < upper_body_boundary {
            if i % 2 == 0 {
                mu[i] += 0.32 * swing_amplitude * (cpg.r_hip_state as f64);
                if i % 3 == 0 && swing_amplitude > 0.0 { mu[i] -= 0.15 * base_clearance * (cpg.r_knee_state as f64); }
            } else {
                mu[i] += 0.32 * swing_amplitude * (cpg.l_hip_state as f64);
                if i % 3 == 0 && swing_amplitude > 0.0 { mu[i] -= 0.15 * base_clearance * (cpg.l_knee_state as f64); }
            }
        } else {
            if i % 4 == 0 {
                mu[i] = -0.75; // Lock elbows in a protected athletic guard posture
            } else if i % 5 == 0 {
                let arm_swing = 0.16 * swing_amplitude * (1.0 + stride_gain as f64);
                mu[i] += if i % 2 == 0 { arm_swing * (cpg.l_hip_state as f64) } else { -arm_swing * (cpg.r_hip_state as f64) };
            }
        }
    }

    let standing_still = profile.target_velocity == 0.0 && profile.stride_amplitude == 0.0;
    let mut raw_torques = vec![0.0f32; total_actuators];
    
    // Somatosensory Height Deflection Booster
    let height_sink_deviation = (1.28 - state.root_height).max(0.0);
    let allostatic_tone_boost = 1.0 + (height_sink_deviation * 6.5);

    for i in 0..total_actuators {
        let xi_q = state.joint_angles[i] - mu[i];
        let xi_v = state.joint_velocities[i]; 

        let thresholded_xi_q = if standing_still && xi_q.abs() < 5e-4 { 0.0 } else { xi_q };
        let raw_torque = profile.make_kp_kd_torque(thresholded_xi_q, xi_v, i, stress_factor as f64, standing_still) * allostatic_tone_boost;
        
        // FIXED: Multiplied output by 12.0 to give her muscles true structural power against 64-DoF mass configurations
        raw_torques[i] = ((raw_torque * 12.0) / profile.scale).clamp(-1.0, 1.0) as f32;
    }

    let mut filtered_torques = vec![0.0f32; total_actuators];
    for i in 0..total_actuators {
        let past_torque = if i < previous_command.torques.len() { previous_command.torques[i] } else { 0.0f32 };
        filtered_torques[i] = past_torque + profile.activation_alpha * (raw_torques[i] - past_torque);
    }

    HumanoidCommand { torques: filtered_torques }
}

pub struct HdcWorkspace {
    pub projection_matrix: [[f32; 5]; 32], 
}

impl HdcWorkspace {
    pub fn new() -> Self {
        let mut matrix = [[0.0f32; 5]; 32];
        let mut seed = 12345u64;
        for i in 0..32 {
            for j in 0..5 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                matrix[i][j] = ((seed >> 33) as f32 / (1u32 << 31) as f32) * 2.0f32 - 1.0f32;
            }
        }
        Self { projection_matrix: matrix }
    }

    pub fn encode(&self, state: &HumanoidState) -> [f32; 32] {
        let mut hypervector = [0.0f32; 32];
        let obs = [
            state.joint_angles[0] as f32, 
            state.joint_angles[5] as f32, 
            state.root_linear_velocity[0] as f32,
            state.root_height as f32,
            state.uprightness() as f32,
        ];
        for i in 0..32 {
            let mut dot_product = 0.0f32;
            for j in 0..5 {
                dot_product += self.projection_matrix[i][j] * obs[j];
            }
            hypervector[i] = dot_product.tanh(); 
        }
        hypervector
    }
}
