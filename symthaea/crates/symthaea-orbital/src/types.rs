use serde::{Deserialize, Serialize};
pub const NUM_JOINTS: usize = 7;
pub const NUM_STATE_CHANNELS: usize = 26;
pub const NUM_ACTUATORS: usize = 7;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalState {
    pub joint_angles: [f64; NUM_JOINTS],
    pub joint_velocities: [f64; NUM_JOINTS],
    pub ee_position: [f64; 3],
    pub spacecraft_angular_velocity: [f64; 3],
    pub solar_exposure: f64,
    pub comm_window: f64,
}
impl OrbitalState {
    pub fn stowed() -> Self {
        Self {
            joint_angles: [0.0; NUM_JOINTS],
            joint_velocities: [0.0; NUM_JOINTS],
            ee_position: [0.0, 0.0, 2.0],
            spacecraft_angular_velocity: [0.0; 3],
            solar_exposure: 1.0,
            comm_window: 1.0,
        }
    }
    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        let mut c = [0.0f32; NUM_STATE_CHANNELS];
        for i in 0..NUM_JOINTS {
            c[i] = self.joint_angles[i] as f32;
            c[i + 7] = self.joint_velocities[i] as f32;
        }
        c[14] = self.ee_position[0] as f32;
        c[15] = self.ee_position[1] as f32;
        c[16] = self.ee_position[2] as f32;
        c[17] = self.spacecraft_angular_velocity[0] as f32;
        c[18] = self.spacecraft_angular_velocity[1] as f32;
        c[19] = self.spacecraft_angular_velocity[2] as f32;
        c[20] = self.solar_exposure as f32;
        c[21] = self.comm_window as f32;
        c[22] = (self
            .spacecraft_angular_velocity
            .iter()
            .map(|v| v * v)
            .sum::<f64>())
        .sqrt() as f32;
        c[23] = (self.ee_position.iter().map(|p| p * p).sum::<f64>()).sqrt() as f32;
        c[24] = self.joint_velocities.iter().map(|v| v * v).sum::<f64>() as f32;
        c[25] = (self.solar_exposure * self.comm_window) as f32;
        c
    }
    pub fn is_finite(&self) -> bool {
        self.joint_angles
            .iter()
            .chain(self.joint_velocities.iter())
            .chain(self.ee_position.iter())
            .chain(self.spacecraft_angular_velocity.iter())
            .all(|v| v.is_finite())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrbitalCommand {
    pub joint_torques: [f32; NUM_ACTUATORS],
}
impl OrbitalCommand {
    pub fn zero() -> Self {
        Self {
            joint_torques: [0.0; NUM_ACTUATORS],
        }
    }
    pub fn control_effort(&self) -> f32 {
        self.joint_torques.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalConfig {
    pub max_joint_torques: [f64; NUM_JOINTS],
    pub spacecraft_inertia: [f64; 3],
    pub reaction_wheel_capacity: f64,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub learning_rate: f32,
}
impl Default for OrbitalConfig {
    fn default() -> Self {
        Self {
            max_joint_torques: [20.0, 20.0, 15.0, 10.0, 5.0, 3.0, 2.0],
            spacecraft_inertia: [5000.0, 5000.0, 3000.0],
            reaction_wheel_capacity: 50.0,
            physics_hz: 100.0,
            cognitive_interval: 5,
            steps_per_episode: 3000,
            network_layers: 3,
            neurons_per_layer: 8,
            learning_rate: 0.001,
        }
    }
}
impl OrbitalConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stowed() {
        let s = OrbitalState::stowed();
        assert!(s.is_finite());
        assert_eq!(s.to_channels().len(), NUM_STATE_CHANNELS);
    }

    #[test]
    fn test_command_zero_effort() {
        let c = OrbitalCommand::zero();
        assert_eq!(c.control_effort(), 0.0);
        assert_eq!(c.joint_torques, [0.0; NUM_ACTUATORS]);
    }

    #[test]
    fn test_command_effort_max() {
        // All torques saturated → effort = 1.0 (mean of absolute values / N_actuators).
        let c = OrbitalCommand {
            joint_torques: [1.0; NUM_ACTUATORS],
        };
        assert!((c.control_effort() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_command_effort_symmetric_in_sign() {
        // Negative torques contribute equally to effort (magnitude-based).
        let c = OrbitalCommand {
            joint_torques: [-0.5; NUM_ACTUATORS],
        };
        assert!((c.control_effort() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_config_physics_dt() {
        // Default physics_hz = 100 Hz → dt = 10 ms.
        let cfg = OrbitalConfig::default();
        assert!((cfg.physics_dt() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_stowed_ee_position_above_origin() {
        // Stowed pose documentation claims the end-effector sits 2 m above
        // the spacecraft origin along z. Verify.
        let s = OrbitalState::stowed();
        assert_eq!(s.ee_position, [0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_stowed_exposure_and_comm_both_nominal() {
        // Stowed initial state should have full solar exposure and open
        // comm window — the "nominal" orbital starting condition.
        let s = OrbitalState::stowed();
        assert_eq!(s.solar_exposure, 1.0);
        assert_eq!(s.comm_window, 1.0);
    }

    #[test]
    fn test_finite_rejects_nan() {
        let mut s = OrbitalState::stowed();
        s.joint_angles[0] = f64::NAN;
        assert!(!s.is_finite());
    }
}
