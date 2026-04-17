use serde::{Deserialize, Serialize};
pub const NUM_LEGS: usize = 4;
pub const JOINTS_PER_LEG: usize = 3;
pub const NUM_JOINTS: usize = NUM_LEGS * JOINTS_PER_LEG;
pub const NUM_STATE_CHANNELS: usize = 37;
pub const NUM_ACTUATORS: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaitType {
    Trot,
    Walk,
    Freeze,
    Collapse,
}
impl GaitType {
    pub fn from_phi(phi: f64) -> Self {
        if phi > 0.6 {
            Self::Trot
        } else if phi > 0.3 {
            Self::Walk
        } else if phi > 0.1 {
            Self::Freeze
        } else {
            Self::Collapse
        }
    }
    pub fn frequency(&self) -> f64 {
        match self {
            Self::Trot => 2.0,
            Self::Walk => 1.0,
            _ => 0.0,
        }
    }
    pub fn step_height(&self) -> f64 {
        match self {
            Self::Trot => 0.08,
            Self::Walk => 0.04,
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadrupedState {
    pub base_position: [f64; 3],
    pub base_quaternion: [f64; 4],
    pub base_linear_velocity: [f64; 3],
    pub base_angular_velocity: [f64; 3],
    pub joint_angles: [f64; NUM_JOINTS],
    pub joint_velocities: [f64; NUM_JOINTS],
    pub foot_contacts: [f64; NUM_LEGS],
}
impl QuadrupedState {
    pub fn standing() -> Self {
        Self {
            base_position: [0.0, 0.0, 0.35],
            base_quaternion: [1.0, 0.0, 0.0, 0.0],
            base_linear_velocity: [0.0; 3],
            base_angular_velocity: [0.0; 3],
            joint_angles: [
                0.0, 0.5, -1.0, 0.0, 0.5, -1.0, 0.0, 0.5, -1.0, 0.0, 0.5, -1.0,
            ],
            joint_velocities: [0.0; NUM_JOINTS],
            foot_contacts: [1.0; NUM_LEGS],
        }
    }
    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        let mut c = [0.0f32; NUM_STATE_CHANNELS];
        for i in 0..3 {
            c[i] = self.base_position[i] as f32;
        }
        for i in 0..4 {
            c[3 + i] = self.base_quaternion[i] as f32;
        }
        for i in 0..3 {
            c[7 + i] = self.base_linear_velocity[i] as f32;
            c[10 + i] = self.base_angular_velocity[i] as f32;
        }
        for i in 0..NUM_JOINTS {
            c[13 + i] = self.joint_angles[i] as f32;
        }
        // remaining: 13+12=25, need 37-25=12 more for velocities
        for i in 0..NUM_JOINTS {
            c[25 + i] = self.joint_velocities[i] as f32;
        }
        c
    }
    pub fn is_finite(&self) -> bool {
        self.base_position
            .iter()
            .chain(self.base_quaternion.iter())
            .chain(self.joint_angles.iter())
            .chain(self.joint_velocities.iter())
            .all(|v| v.is_finite())
    }
    pub fn height(&self) -> f64 {
        self.base_position[2]
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QuadrupedCommand {
    pub joint_torques: [f32; NUM_ACTUATORS],
}
impl QuadrupedCommand {
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
pub struct QuadrupedConfig {
    pub max_joint_torques: [f64; NUM_JOINTS],
    pub body_mass: f64,
    pub physics_hz: f64,
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub learning_rate: f32,
}
impl Default for QuadrupedConfig {
    fn default() -> Self {
        Self {
            max_joint_torques: [
                30.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0, 30.0, 20.0,
            ],
            body_mass: 12.0,
            physics_hz: 200.0,
            cognitive_interval: 10,
            steps_per_episode: 4000,
            network_layers: 3,
            neurons_per_layer: 8,
            learning_rate: 0.001,
        }
    }
}
impl QuadrupedConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_standing() {
        let s = QuadrupedState::standing();
        assert!(s.is_finite());
        assert_eq!(s.to_channels().len(), NUM_STATE_CHANNELS);
    }
    #[test]
    fn test_gait() {
        assert_eq!(GaitType::from_phi(0.8), GaitType::Trot);
        assert_eq!(GaitType::from_phi(0.05), GaitType::Collapse);
    }

    #[test]
    fn test_gait_boundaries() {
        // All four tier transitions at the documented phi thresholds.
        assert_eq!(GaitType::from_phi(0.61), GaitType::Trot);
        assert_eq!(GaitType::from_phi(0.60), GaitType::Walk);
        assert_eq!(GaitType::from_phi(0.31), GaitType::Walk);
        assert_eq!(GaitType::from_phi(0.30), GaitType::Freeze);
        assert_eq!(GaitType::from_phi(0.11), GaitType::Freeze);
        assert_eq!(GaitType::from_phi(0.10), GaitType::Collapse);
        assert_eq!(GaitType::from_phi(0.0), GaitType::Collapse);
    }

    #[test]
    fn test_gait_frequency_ordering() {
        // Trot is faster than walk; freeze/collapse produce no motion.
        assert!(GaitType::Trot.frequency() > GaitType::Walk.frequency());
        assert_eq!(GaitType::Freeze.frequency(), 0.0);
        assert_eq!(GaitType::Collapse.frequency(), 0.0);
    }

    #[test]
    fn test_gait_step_height_ordering() {
        // Trot lifts feet higher than walk; stationary gaits don't lift.
        assert!(GaitType::Trot.step_height() > GaitType::Walk.step_height());
        assert_eq!(GaitType::Freeze.step_height(), 0.0);
        assert_eq!(GaitType::Collapse.step_height(), 0.0);
    }

    #[test]
    fn test_command_zero_effort() {
        let c = QuadrupedCommand::zero();
        assert_eq!(c.control_effort(), 0.0);
        assert_eq!(c.joint_torques, [0.0; NUM_ACTUATORS]);
    }

    #[test]
    fn test_command_effort_max() {
        // All 12 actuators saturated → effort = 1.0 (mean of |torques|).
        let c = QuadrupedCommand {
            joint_torques: [1.0; NUM_ACTUATORS],
        };
        assert!((c.control_effort() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_standing_pose_fields() {
        // Stowed state: feet all in contact, base at 0.35 m, quaternion
        // identity, zero velocities everywhere.
        let s = QuadrupedState::standing();
        assert_eq!(s.base_position, [0.0, 0.0, 0.35]);
        assert_eq!(s.base_quaternion, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.foot_contacts, [1.0; NUM_LEGS]);
        assert_eq!(s.base_linear_velocity, [0.0; 3]);
        assert_eq!(s.base_angular_velocity, [0.0; 3]);
    }

    #[test]
    fn test_height_accessor() {
        let s = QuadrupedState::standing();
        assert_eq!(s.height(), 0.35);
    }

    #[test]
    fn test_config_physics_dt() {
        // Default physics_hz = 200 Hz → dt = 5 ms.
        let cfg = QuadrupedConfig::default();
        assert!((cfg.physics_dt() - 0.005).abs() < 1e-9);
    }

    #[test]
    fn test_num_joints_consistency() {
        // NUM_JOINTS must equal NUM_LEGS × JOINTS_PER_LEG.
        assert_eq!(NUM_JOINTS, NUM_LEGS * JOINTS_PER_LEG);
        assert_eq!(NUM_JOINTS, NUM_ACTUATORS, "one actuator per joint");
    }

    #[test]
    fn test_finite_rejects_nan() {
        let mut s = QuadrupedState::standing();
        s.joint_angles[0] = f64::NAN;
        assert!(!s.is_finite());
    }
}
