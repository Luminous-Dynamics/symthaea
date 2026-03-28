// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for helicopter control: state, command, config, task.

use serde::{Deserialize, Serialize};

// ── Constants ────────────────────────────────────────────────────────────────

/// Number of actuator channels in the helicopter command.
pub const NUM_ACTUATORS: usize = 6;

/// Number of sensor state channels.
pub const NUM_STATE_CHANNELS: usize = 18;

/// Joint/channel names for documentation and debugging.
pub const CHANNEL_NAMES: [&str; NUM_STATE_CHANNELS] = [
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "angvel_x",
    "angvel_y",
    "angvel_z",
    "main_rotor_rpm",
    "tail_rotor_rpm",
    "collective_pitch",
    "cyclic_lon_feedback",
    "cyclic_lat_feedback",
];

// ── State ────────────────────────────────────────────────────────────────────

/// Full helicopter state (18D): position + quaternion + velocities + rotor feedback.
///
/// Channels 0–12 are shared with quadrotor (enabling morphological transfer).
/// Channels 13–17 are helicopter-specific (rotor RPM, swashplate feedback).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelicopterState {
    /// World-frame position [x, y, z] in meters.
    pub position: [f64; 3],
    /// Orientation quaternion [w, x, y, z] (w-first convention).
    pub quaternion: [f64; 4],
    /// World-frame linear velocity [vx, vy, vz] in m/s.
    pub linear_velocity: [f64; 3],
    /// Body-frame angular velocity [wx, wy, wz] in rad/s.
    pub angular_velocity: [f64; 3],
    /// Main rotor RPM (0–6000 typical).
    pub main_rotor_rpm: f64,
    /// Tail rotor RPM (0–6000 typical).
    pub tail_rotor_rpm: f64,
    /// Current collective pitch angle (rad, feedback from swashplate).
    pub collective_pitch: f64,
    /// Cyclic longitudinal feedback (rad).
    pub cyclic_lon_feedback: f64,
    /// Cyclic lateral feedback (rad).
    pub cyclic_lat_feedback: f64,
}

impl HelicopterState {
    /// Pack state into 18 f32 channels for HDC encoding.
    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        [
            self.position[0] as f32,
            self.position[1] as f32,
            self.position[2] as f32,
            self.quaternion[0] as f32,
            self.quaternion[1] as f32,
            self.quaternion[2] as f32,
            self.quaternion[3] as f32,
            self.linear_velocity[0] as f32,
            self.linear_velocity[1] as f32,
            self.linear_velocity[2] as f32,
            self.angular_velocity[0] as f32,
            self.angular_velocity[1] as f32,
            self.angular_velocity[2] as f32,
            self.main_rotor_rpm as f32,
            self.tail_rotor_rpm as f32,
            self.collective_pitch as f32,
            self.cyclic_lon_feedback as f32,
            self.cyclic_lat_feedback as f32,
        ]
    }

    /// Altitude (z position) in meters.
    pub fn altitude(&self) -> f64 {
        self.position[2]
    }

    /// Linear speed magnitude in m/s.
    pub fn speed(&self) -> f64 {
        let [vx, vy, vz] = self.linear_velocity;
        (vx * vx + vy * vy + vz * vz).sqrt()
    }

    /// Horizontal speed (XY plane) in m/s.
    pub fn horizontal_speed(&self) -> f64 {
        let [vx, vy, _] = self.linear_velocity;
        (vx * vx + vy * vy).sqrt()
    }

    /// Angular speed magnitude in rad/s.
    pub fn angular_speed(&self) -> f64 {
        let [wx, wy, wz] = self.angular_velocity;
        (wx * wx + wy * wy + wz * wz).sqrt()
    }

    /// Approximate uprightness: z-component of body "up" vector via quaternion.
    /// 1.0 = perfectly upright, 0.0 = horizontal, -1.0 = inverted.
    pub fn uprightness(&self) -> f64 {
        let [w, x, y, _z] = self.quaternion;
        1.0 - 2.0 * (x * x + y * y) + 2.0 * w * w - 1.0 // Simplification; full rotation matrix z column
    }

    /// Euler angles (roll, pitch, yaw) from quaternion.
    pub fn euler_angles(&self) -> (f64, f64, f64) {
        let [w, x, y, z] = self.quaternion;
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// A hovering state at the given altitude.
    pub fn hover(altitude: f64) -> Self {
        Self {
            position: [0.0, 0.0, altitude],
            quaternion: [1.0, 0.0, 0.0, 0.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            main_rotor_rpm: 3500.0, // Typical hover RPM
            tail_rotor_rpm: 2000.0,
            collective_pitch: 0.15, // ~9 degrees for hover
            cyclic_lon_feedback: 0.0,
            cyclic_lat_feedback: 0.0,
        }
    }

    /// All-zero state (grounded, rotors off).
    pub fn grounded() -> Self {
        Self {
            position: [0.0; 3],
            quaternion: [1.0, 0.0, 0.0, 0.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            main_rotor_rpm: 0.0,
            tail_rotor_rpm: 0.0,
            collective_pitch: 0.0,
            cyclic_lon_feedback: 0.0,
            cyclic_lat_feedback: 0.0,
        }
    }

    /// Whether all state values are finite (no NaN/Inf).
    pub fn is_finite(&self) -> bool {
        self.position.iter().all(|v| v.is_finite())
            && self.quaternion.iter().all(|v| v.is_finite())
            && self.linear_velocity.iter().all(|v| v.is_finite())
            && self.angular_velocity.iter().all(|v| v.is_finite())
            && self.main_rotor_rpm.is_finite()
            && self.tail_rotor_rpm.is_finite()
    }
}

// ── Command ──────────────────────────────────────────────────────────────────

/// Motor command output (6D): collective + cyclic (lon/lat) + pedal + thrust + tail.
///
/// SAR helicopter control surfaces:
/// - `collective`: main rotor blade pitch [-1.0, 1.0] (normalized from min to max pitch)
/// - `cyclic_lon`: longitudinal cyclic (pitch control) [-1.0, 1.0]
/// - `cyclic_lat`: lateral cyclic (roll control) [-1.0, 1.0]
/// - `pedal`: tail rotor / yaw control [-1.0, 1.0]
/// - `thrust`: main rotor RPM command [0.0, 1.0] (fraction of max RPM)
/// - `tail_rotor`: tail rotor RPM command [0.0, 1.0]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HelicopterCommand {
    pub collective: f32,
    pub cyclic_lon: f32,
    pub cyclic_lat: f32,
    pub pedal: f32,
    pub thrust: f32,
    pub tail_rotor: f32,
}

impl HelicopterCommand {
    /// Hover command: collective at hover pitch, thrust at ~60% max RPM.
    pub const HOVER_COLLECTIVE: f32 = 0.3;
    pub const HOVER_THRUST: f32 = 0.6;
    pub const HOVER_TAIL: f32 = 0.5;

    /// Hover-stabilized command.
    pub fn hover() -> Self {
        Self {
            collective: Self::HOVER_COLLECTIVE,
            cyclic_lon: 0.0,
            cyclic_lat: 0.0,
            pedal: 0.0,
            thrust: Self::HOVER_THRUST,
            tail_rotor: Self::HOVER_TAIL,
        }
    }

    /// All-zero command (motors off).
    pub fn zero() -> Self {
        Self {
            collective: 0.0,
            cyclic_lon: 0.0,
            cyclic_lat: 0.0,
            pedal: 0.0,
            thrust: 0.0,
            tail_rotor: 0.0,
        }
    }

    /// Clamp all channels to valid ranges.
    pub fn clamped(self) -> Self {
        Self {
            collective: self.collective.clamp(-1.0, 1.0),
            cyclic_lon: self.cyclic_lon.clamp(-1.0, 1.0),
            cyclic_lat: self.cyclic_lat.clamp(-1.0, 1.0),
            pedal: self.pedal.clamp(-1.0, 1.0),
            thrust: self.thrust.clamp(0.0, 1.0),
            tail_rotor: self.tail_rotor.clamp(0.0, 1.0),
        }
    }

    /// Convert to f64 array for physics sim.
    pub fn to_ctrl(&self) -> [f64; NUM_ACTUATORS] {
        [
            self.collective as f64,
            self.cyclic_lon as f64,
            self.cyclic_lat as f64,
            self.pedal as f64,
            self.thrust as f64,
            self.tail_rotor as f64,
        ]
    }

    /// Construct from raw f32 array.
    pub fn from_raw(values: &[f32; NUM_ACTUATORS]) -> Self {
        Self {
            collective: values[0],
            cyclic_lon: values[1],
            cyclic_lat: values[2],
            pedal: values[3],
            thrust: values[4],
            tail_rotor: values[5],
        }
    }

    /// Mean absolute control effort (0.0–1.0).
    pub fn control_effort(&self) -> f32 {
        (self.collective.abs()
            + self.cyclic_lon.abs()
            + self.cyclic_lat.abs()
            + self.pedal.abs()
            + self.thrust.abs()
            + self.tail_rotor.abs())
            / NUM_ACTUATORS as f32
    }

    /// Add Gaussian noise to all channels.
    pub fn with_noise(mut self, std: f32, rng: &mut impl rand::Rng) -> Self {
        use rand::distributions::Distribution;
        let dist = rand::distributions::Uniform::new(-std, std);
        self.collective += dist.sample(rng);
        self.cyclic_lon += dist.sample(rng);
        self.cyclic_lat += dist.sample(rng);
        self.pedal += dist.sample(rng);
        self.thrust += dist.sample(rng);
        self.tail_rotor += dist.sample(rng);
        self.clamped()
    }
}

// ── Config ───────────────────────────────────────────────────────────────────

/// Configuration for helicopter controller and training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelicopterConfig {
    /// Genesis phrase for deterministic weight initialization.
    pub genesis_phrase: String,
    /// Learning rate for output projection (default: 0.0005).
    pub learning_rate: f32,
    /// Number of layers in HdcLtcUnifiedNetwork (default: 2).
    pub network_layers: usize,
    /// Neurons per layer (default: 8).
    pub neurons_per_layer: usize,
    /// Physics timestep in seconds (default: 1/300 = 3.33ms).
    pub physics_hz: f64,
    /// Cognitive tick interval in physics steps (default: 15 → 20Hz).
    pub cognitive_interval: usize,
    /// Steps per training episode (default: 3000 → 10s at 300Hz).
    pub steps_per_episode: usize,
    /// Number of training episodes (default: 200).
    pub num_episodes: usize,
    /// Actuator noise standard deviation (default: 0.02).
    pub actuator_noise_std: f32,
    /// Observation noise standard deviation (default: 0.01).
    pub observation_noise_std: f32,
}

impl Default for HelicopterConfig {
    fn default() -> Self {
        Self {
            genesis_phrase: "symthaea-helicopter-sar".to_string(),
            learning_rate: 0.0005,
            network_layers: 2,
            neurons_per_layer: 8,
            physics_hz: 300.0,
            cognitive_interval: 15, // 300/15 = 20Hz cognitive tick
            steps_per_episode: 3000,
            num_episodes: 200,
            actuator_noise_std: 0.02,
            observation_noise_std: 0.01,
        }
    }
}

impl HelicopterConfig {
    /// Physics timestep in seconds.
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }

    /// Cognitive tick rate in Hz.
    pub fn cognitive_hz(&self) -> f64 {
        self.physics_hz / self.cognitive_interval as f64
    }
}

// ── Task ─────────────────────────────────────────────────────────────────────

/// SAR helicopter mission tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HelicopterTask {
    /// Maintain stable hover at target altitude.
    Hover,
    /// Fly to target position and hover.
    GoToPosition,
    /// Search pattern over designated zone.
    SearchGrid,
    /// Hover over target with winch deploy clearance.
    HoverOverTarget,
    /// Emergency autorotation descent (engine failure).
    Autorotation,
}

impl HelicopterTask {
    /// Target altitude for this task (meters).
    pub fn target_altitude(&self) -> f64 {
        match self {
            Self::Hover => 20.0,
            Self::GoToPosition => 30.0,
            Self::SearchGrid => 50.0,
            Self::HoverOverTarget => 15.0,
            Self::Autorotation => 0.0,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hover_state_is_finite() {
        let state = HelicopterState::hover(20.0);
        assert!(state.is_finite());
        assert!((state.altitude() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_grounded_state() {
        let state = HelicopterState::grounded();
        assert!(state.is_finite());
        assert_eq!(state.altitude(), 0.0);
        assert_eq!(state.speed(), 0.0);
        assert_eq!(state.main_rotor_rpm, 0.0);
    }

    #[test]
    fn test_state_to_channels_roundtrip() {
        let state = HelicopterState::hover(25.0);
        let channels = state.to_channels();
        assert_eq!(channels.len(), NUM_STATE_CHANNELS);
        assert!((channels[2] - 25.0).abs() < 1e-5); // pos_z
        assert!((channels[0]).abs() < 1e-5); // pos_x = 0
    }

    #[test]
    fn test_hover_command() {
        let cmd = HelicopterCommand::hover();
        assert!(cmd.collective > 0.0);
        assert!(cmd.thrust > 0.0);
        assert_eq!(cmd.cyclic_lon, 0.0);
        assert_eq!(cmd.cyclic_lat, 0.0);
    }

    #[test]
    fn test_zero_command() {
        let cmd = HelicopterCommand::zero();
        assert_eq!(cmd.control_effort(), 0.0);
    }

    #[test]
    fn test_command_clamped() {
        let cmd = HelicopterCommand {
            collective: 2.0,
            cyclic_lon: -3.0,
            cyclic_lat: 0.5,
            pedal: 0.0,
            thrust: 1.5,
            tail_rotor: -0.5,
        };
        let clamped = cmd.clamped();
        assert_eq!(clamped.collective, 1.0);
        assert_eq!(clamped.cyclic_lon, -1.0);
        assert_eq!(clamped.cyclic_lat, 0.5);
        assert_eq!(clamped.thrust, 1.0);
        assert_eq!(clamped.tail_rotor, 0.0); // [0, 1] range
    }

    #[test]
    fn test_command_from_raw() {
        let raw = [0.3, 0.0, 0.0, 0.0, 0.6, 0.5];
        let cmd = HelicopterCommand::from_raw(&raw);
        assert_eq!(cmd.collective, 0.3);
        assert_eq!(cmd.thrust, 0.6);
    }

    #[test]
    fn test_command_to_ctrl() {
        let cmd = HelicopterCommand::hover();
        let ctrl = cmd.to_ctrl();
        assert_eq!(ctrl.len(), NUM_ACTUATORS);
    }

    #[test]
    fn test_euler_angles_identity() {
        let state = HelicopterState::hover(20.0);
        let (roll, pitch, yaw) = state.euler_angles();
        assert!(roll.abs() < 1e-10);
        assert!(pitch.abs() < 1e-10);
        assert!(yaw.abs() < 1e-10);
    }

    #[test]
    fn test_config_defaults() {
        let config = HelicopterConfig::default();
        assert_eq!(config.physics_hz, 300.0);
        assert!((config.physics_dt() - 1.0 / 300.0).abs() < 1e-10);
        assert!((config.cognitive_hz() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_task_target_altitudes() {
        assert_eq!(HelicopterTask::Hover.target_altitude(), 20.0);
        assert_eq!(HelicopterTask::Autorotation.target_altitude(), 0.0);
        assert!(
            HelicopterTask::SearchGrid.target_altitude() > HelicopterTask::Hover.target_altitude()
        );
    }

    #[test]
    fn test_serde_roundtrip_state() {
        let state = HelicopterState::hover(30.0);
        let json = serde_json::to_string(&state).unwrap();
        let restored: HelicopterState = serde_json::from_str(&json).unwrap();
        assert!((restored.altitude() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_serde_roundtrip_command() {
        let cmd = HelicopterCommand::hover();
        let json = serde_json::to_string(&cmd).unwrap();
        let restored: HelicopterCommand = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.collective, cmd.collective);
    }

    #[test]
    fn test_serde_roundtrip_config() {
        let config = HelicopterConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: HelicopterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.physics_hz, config.physics_hz);
    }

    #[test]
    fn test_control_effort() {
        let cmd = HelicopterCommand::hover();
        let effort = cmd.control_effort();
        assert!(effort > 0.0);
        assert!(effort <= 1.0);
    }
}
