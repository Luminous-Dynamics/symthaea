// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};
use symthaea_core::embodiment::MotorSafetyLevel;

pub const NUM_JOINTS: usize = 6;
pub const NUM_STATE_CHANNELS: usize = 24;
pub const NUM_ACTUATORS: usize = 8;

// ── Surgical safety, unified under the shared MotorSafetyLevel contract ────
//
// This used to be a standalone `SurgicalSafetyLevel` enum with its own
// `from_phi()` threshold function — a byte-for-byte parallel copy of
// `MotorSafetyLevel`'s own 4-tier thresholds (0.6/0.3/0.1), duplicated so
// the two ladders had to be hand-kept in sync. Per the SafeFallback audit
// (2026-07), it's unified here: `MotorSafetyLevel` is the single source of
// the tier decision (computed once, incorporating phi + safety_override +
// moral gate — see `embodiment.rs`), and these two free functions derive
// surgical-specific *behavior* from that one shared tier. The STRUCTURE
// (4 tiers, Phi-derived) is shared with every other platform; only the
// per-tier numeric tuning (torque gain, cautery gate) is surgical-specific,
// which is exactly what the unification was meant to make auditable.

/// Surgical-specific torque gain per shared safety tier.
///
/// Stricter than the generic `MotorSafetyLevel::motor_gain()` (0.6/0.3 for
/// Yellow/Orange) because this platform cuts human tissue: Orange freezes
/// completely rather than retaining 30% authority.
pub fn surgical_torque_gain(level: MotorSafetyLevel) -> f32 {
    match level {
        MotorSafetyLevel::Green => 1.0,
        MotorSafetyLevel::Yellow => 0.4,
        MotorSafetyLevel::Orange | MotorSafetyLevel::Red => 0.0,
    }
}

/// Cautery is only ever allowed at full (Green) confidence.
pub fn surgical_cautery_allowed(level: MotorSafetyLevel) -> bool {
    matches!(level, MotorSafetyLevel::Green)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgicalState {
    pub joint_angles: [f64; NUM_JOINTS],
    pub joint_velocities: [f64; NUM_JOINTS],
    pub tip_position: [f64; 3],
    pub tip_force: [f64; 3],
    pub jaw_angle: f64,
    pub cautery_power: f64,
    pub critical_structure_distance: f64,
    pub trocar_compliance: f64,
}
impl SurgicalState {
    pub fn home() -> Self {
        Self {
            joint_angles: [0.0, 0.3, 0.0, -0.5, 0.0, 0.0],
            joint_velocities: [0.0; NUM_JOINTS],
            tip_position: [0.0, 0.0, -50.0],
            tip_force: [0.0; 3],
            jaw_angle: 0.0,
            cautery_power: 0.0,
            critical_structure_distance: 20.0,
            trocar_compliance: 0.0,
        }
    }
    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        let mut c = [0.0f32; NUM_STATE_CHANNELS];
        for i in 0..NUM_JOINTS {
            c[i] = self.joint_angles[i] as f32;
            c[i + 6] = self.joint_velocities[i] as f32;
        }
        c[12] = self.tip_position[0] as f32;
        c[13] = self.tip_position[1] as f32;
        c[14] = self.tip_position[2] as f32;
        c[15] = self.tip_force[0] as f32;
        c[16] = self.tip_force[1] as f32;
        c[17] = self.tip_force[2] as f32;
        c[18] = self.jaw_angle as f32;
        c[19] = self.cautery_power as f32;
        c[20] = self.critical_structure_distance as f32;
        c[21] = self.trocar_compliance as f32;
        c[22] = (self.joint_velocities.iter().map(|v| v * v).sum::<f64>()).sqrt() as f32;
        c[23] = (self.tip_force.iter().map(|f| f * f).sum::<f64>()).sqrt() as f32;
        c
    }
    pub fn is_finite(&self) -> bool {
        self.joint_angles
            .iter()
            .chain(self.joint_velocities.iter())
            .chain(self.tip_position.iter())
            .chain(self.tip_force.iter())
            .all(|v| v.is_finite())
            && self.jaw_angle.is_finite()
            && self.cautery_power.is_finite()
    }
    pub fn force_magnitude(&self) -> f64 {
        (self.tip_force.iter().map(|f| f * f).sum::<f64>()).sqrt()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SurgicalCommand {
    pub joint_torques: [f32; NUM_JOINTS],
    pub jaw: f32,
    pub cautery: f32,
}
impl SurgicalCommand {
    pub fn zero() -> Self {
        Self {
            joint_torques: [0.0; NUM_JOINTS],
            jaw: 0.0,
            cautery: 0.0,
        }
    }
    pub fn control_effort(&self) -> f32 {
        (self.joint_torques.iter().map(|t| t.abs()).sum::<f32>()
            + self.jaw.abs()
            + self.cautery.abs())
            / NUM_ACTUATORS as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgicalConfig {
    pub max_joint_torques: [f64; NUM_JOINTS],
    pub motion_scaling: f64,
    pub tremor_filter_hz: f64,
    pub rcm_stiffness: f64,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub learning_rate: f32,
}
impl Default for SurgicalConfig {
    fn default() -> Self {
        Self {
            max_joint_torques: [5.0, 5.0, 3.0, 2.0, 1.0, 0.5],
            motion_scaling: 0.2,
            tremor_filter_hz: 8.0,
            rcm_stiffness: 1000.0,
            physics_hz: 1000.0,
            cognitive_interval: 50,
            steps_per_episode: 5000,
            network_layers: 3,
            neurons_per_layer: 8,
            learning_rate: 0.0005,
        }
    }
}
impl SurgicalConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_home() {
        let s = SurgicalState::home();
        assert!(s.is_finite());
        assert_eq!(s.to_channels().len(), NUM_STATE_CHANNELS);
    }
    #[test]
    fn test_safety() {
        assert!(surgical_cautery_allowed(MotorSafetyLevel::from_phi(0.8)));
        assert!(!surgical_cautery_allowed(MotorSafetyLevel::from_phi(0.4)));
    }

    #[test]
    fn test_command_zero_effort() {
        let c = SurgicalCommand::zero();
        assert_eq!(c.control_effort(), 0.0);
        assert_eq!(c.joint_torques, [0.0; NUM_JOINTS]);
    }

    #[test]
    fn test_command_effort_bounds() {
        let c = SurgicalCommand {
            joint_torques: [1.0; NUM_JOINTS],
            jaw: 1.0,
            cautery: 1.0,
        };
        // 6 joints + jaw + cautery = 8 unit contributions / 8 actuators = 1.0.
        assert!((c.control_effort() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_safety_torque_gain_ordering() {
        // Gain must decrease monotonically as safety tightens.
        let full = surgical_torque_gain(MotorSafetyLevel::Green);
        let reduced = surgical_torque_gain(MotorSafetyLevel::Yellow);
        let freeze = surgical_torque_gain(MotorSafetyLevel::Orange);
        let retract = surgical_torque_gain(MotorSafetyLevel::Red);
        assert!(full > reduced);
        assert!(reduced > freeze);
        assert_eq!(
            freeze, retract,
            "orange (freeze) and red (retract) both zero torque"
        );
        assert_eq!(freeze, 0.0);
    }

    #[test]
    fn test_safety_from_phi_boundaries() {
        // MotorSafetyLevel::from_phi is the single shared source of the
        // 4-tier boundary now — verified thoroughly in symthaea-core itself;
        // this just confirms the surgical-specific gain/cautery functions
        // agree with it at the documented thresholds.
        assert_eq!(MotorSafetyLevel::from_phi(0.61), MotorSafetyLevel::Green);
        assert_eq!(MotorSafetyLevel::from_phi(0.60), MotorSafetyLevel::Yellow);
        assert_eq!(MotorSafetyLevel::from_phi(0.31), MotorSafetyLevel::Yellow);
        assert_eq!(MotorSafetyLevel::from_phi(0.30), MotorSafetyLevel::Orange);
        assert_eq!(MotorSafetyLevel::from_phi(0.11), MotorSafetyLevel::Orange);
        assert_eq!(MotorSafetyLevel::from_phi(0.10), MotorSafetyLevel::Red);
        assert_eq!(MotorSafetyLevel::from_phi(0.0), MotorSafetyLevel::Red);
    }

    #[test]
    fn test_state_force_magnitude() {
        let mut s = SurgicalState::home();
        assert_eq!(s.force_magnitude(), 0.0);
        s.tip_force = [3.0, 4.0, 0.0];
        assert!((s.force_magnitude() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_physics_dt() {
        let cfg = SurgicalConfig::default();
        assert!((cfg.physics_dt() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_cautery_only_at_full_control() {
        assert!(surgical_cautery_allowed(MotorSafetyLevel::Green));
        assert!(!surgical_cautery_allowed(MotorSafetyLevel::Yellow));
        assert!(!surgical_cautery_allowed(MotorSafetyLevel::Orange));
        assert!(!surgical_cautery_allowed(MotorSafetyLevel::Red));
    }
}
