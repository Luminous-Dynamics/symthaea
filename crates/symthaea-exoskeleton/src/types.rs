// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

pub const NUM_JOINTS: usize = 6;
pub const NUM_STATE_CHANNELS: usize = 28;
pub const NUM_ACTUATORS: usize = 6;

/// Assistance mode — determined by consciousness level (Phi).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistanceMode {
    Predictive,          // Phi > 0.6
    Responsive,          // 0.3-0.6
    Transparent,         // 0.1-0.3
    GravityCompensation, // < 0.1
}

impl AssistanceMode {
    pub fn from_phi(phi: f64) -> Self {
        if phi > 0.6 { Self::Predictive }
        else if phi > 0.3 { Self::Responsive }
        else if phi > 0.1 { Self::Transparent }
        else { Self::GravityCompensation }
    }
    pub fn torque_factor(&self) -> f32 {
        match self { Self::Predictive => 1.0, Self::Responsive => 0.6, Self::Transparent => 0.2, Self::GravityCompensation => 0.0 }
    }
    pub fn stiffness_factor(&self) -> f64 {
        match self { Self::Predictive => 1.0, Self::Responsive => 0.6, Self::Transparent => 0.15, Self::GravityCompensation => 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExoskeletonState {
    pub joint_angles: [f64; NUM_JOINTS],
    pub joint_velocities: [f64; NUM_JOINTS],
    pub human_torques: [f64; NUM_JOINTS],
    pub exo_torques: [f64; NUM_JOINTS],
    pub center_of_pressure: [f64; 2],
    pub ground_reaction_force: f64,
    pub battery_soc: f64,
}

impl ExoskeletonState {
    pub fn standing() -> Self {
        Self {
            joint_angles: [0.05, 0.1, 0.0, 0.05, 0.1, 0.0],
            joint_velocities: [0.0; NUM_JOINTS],
            human_torques: [0.0; NUM_JOINTS],
            exo_torques: [0.0; NUM_JOINTS],
            center_of_pressure: [0.0, 0.0],
            ground_reaction_force: 800.0,
            battery_soc: 1.0,
        }
    }
    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        let mut c = [0.0f32; NUM_STATE_CHANNELS];
        for i in 0..NUM_JOINTS {
            c[i] = self.joint_angles[i] as f32;
            c[i + 6] = self.joint_velocities[i] as f32;
            c[i + 12] = self.human_torques[i] as f32;
            c[i + 18] = self.exo_torques[i] as f32;
        }
        c[24] = self.center_of_pressure[0] as f32;
        c[25] = self.center_of_pressure[1] as f32;
        c[26] = self.ground_reaction_force as f32;
        c[27] = self.battery_soc as f32;
        c
    }
    pub fn is_finite(&self) -> bool {
        self.joint_angles.iter().chain(self.joint_velocities.iter())
            .chain(self.human_torques.iter()).chain(self.exo_torques.iter())
            .all(|v| v.is_finite())
            && self.center_of_pressure.iter().all(|v| v.is_finite())
            && self.ground_reaction_force.is_finite() && self.battery_soc.is_finite()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExoskeletonCommand {
    pub joint_torques: [f32; NUM_ACTUATORS],
    pub stiffness_gain: f32,
    pub damping_gain: f32,
}

impl ExoskeletonCommand {
    pub fn zero() -> Self { Self { joint_torques: [0.0; NUM_ACTUATORS], stiffness_gain: 0.5, damping_gain: 0.3 } }
    pub fn control_effort(&self) -> f32 { self.joint_torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32 }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExoskeletonConfig {
    pub max_torques: [f64; NUM_JOINTS],
    pub human_mass: f64,
    pub exo_mass: f64,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub learning_rate: f32,
}

impl Default for ExoskeletonConfig {
    fn default() -> Self {
        Self {
            max_torques: [40.0, 30.0, 20.0, 40.0, 30.0, 20.0],
            human_mass: 80.0, exo_mass: 12.0, physics_hz: 200.0,
            cognitive_interval: 10, steps_per_episode: 2000,
            network_layers: 3, neurons_per_layer: 8, learning_rate: 0.001,
        }
    }
}

impl ExoskeletonConfig {
    pub fn physics_dt(&self) -> f64 { 1.0 / self.physics_hz }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_standing_valid() { let s = ExoskeletonState::standing(); assert!(s.is_finite()); assert_eq!(s.to_channels().len(), NUM_STATE_CHANNELS); }
    #[test]
    fn test_assistance_from_phi() {
        assert_eq!(AssistanceMode::from_phi(0.8), AssistanceMode::Predictive);
        assert_eq!(AssistanceMode::from_phi(0.4), AssistanceMode::Responsive);
        assert_eq!(AssistanceMode::from_phi(0.05), AssistanceMode::GravityCompensation);
    }
    #[test]
    fn test_zero_cmd() { assert_eq!(ExoskeletonCommand::zero().control_effort(), 0.0); }
}
