// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core data types for quadrotor flight control.

use serde::{Deserialize, Serialize};

/// Full quadrotor state (13D): position + quaternion + velocities.
///
/// Extracted from MuJoCo `qpos` (7D: pos[3] + quat[4]) and `qvel` (6D: lin_vel[3] + ang_vel[3]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightState {
    /// World-frame position [x, y, z] in meters.
    pub position: [f64; 3],
    /// Orientation quaternion [w, x, y, z] (MuJoCo convention: w-first).
    pub quaternion: [f64; 4],
    /// World-frame linear velocity [vx, vy, vz] in m/s.
    pub linear_velocity: [f64; 3],
    /// Body-frame angular velocity [wx, wy, wz] in rad/s.
    pub angular_velocity: [f64; 3],
    /// Simulation time in seconds.
    pub timestamp: f64,
}

impl FlightState {
    /// Construct from MuJoCo qpos/qvel arrays.
    pub fn from_qpos_qvel(qpos: &[f64], qvel: &[f64], t: f64) -> Self {
        Self {
            position: [qpos[0], qpos[1], qpos[2]],
            quaternion: [qpos[3], qpos[4], qpos[5], qpos[6]],
            linear_velocity: [qvel[0], qvel[1], qvel[2]],
            angular_velocity: [qvel[3], qvel[4], qvel[5]],
            timestamp: t,
        }
    }

    /// Altitude (z position) in meters.
    pub fn altitude(&self) -> f64 {
        self.position[2]
    }

    /// Approximate Euler angles (roll, pitch, yaw) from quaternion.
    /// Uses the ZYX convention. Returns (roll, pitch, yaw) in radians.
    pub fn euler_angles(&self) -> (f64, f64, f64) {
        let [w, x, y, z] = self.quaternion;

        // Roll (x-axis rotation)
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (y-axis rotation)
        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        // Yaw (z-axis rotation)
        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// Linear speed magnitude in m/s.
    pub fn speed(&self) -> f64 {
        let [vx, vy, vz] = self.linear_velocity;
        (vx * vx + vy * vy + vz * vz).sqrt()
    }

    /// Angular speed magnitude in rad/s.
    pub fn angular_speed(&self) -> f64 {
        let [wx, wy, wz] = self.angular_velocity;
        (wx * wx + wy * wy + wz * wz).sqrt()
    }

    /// Whether all state values are finite (NaN/Inf check).
    pub fn is_finite(&self) -> bool {
        self.position.iter().all(|v| v.is_finite())
            && self.quaternion.iter().all(|v| v.is_finite())
            && self.linear_velocity.iter().all(|v| v.is_finite())
            && self.angular_velocity.iter().all(|v| v.is_finite())
    }

    /// Domain-specific bounds check for physically plausible flight state.
    ///
    /// Returns `true` if the state is within reasonable physical limits:
    /// - Altitude: [0, 10000] m (ground to ~33,000 ft)
    /// - Speed: [0, 50] m/s (~110 mph, reasonable for quadrotor)
    /// - Angular rate: [0, 20] rad/s
    pub fn is_bounded(&self) -> bool {
        self.is_finite()
            && self.position[2] >= 0.0
            && self.position[2] < 10_000.0
            && self.speed() < 50.0
            && self.angular_speed() < 20.0
    }

    /// Pack the 13 state channels into an f32 array for HDC encoding.
    pub fn to_channels(&self) -> [f32; 13] {
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
        ]
    }

    /// Reconstruct a FlightState from 13 packed channels.
    ///
    /// Inverse of `to_channels()`. Timestamp is set to 0.0 (not stored in channels).
    pub fn from_channels(channels: [f32; 13]) -> Self {
        Self {
            position: [channels[0] as f64, channels[1] as f64, channels[2] as f64],
            quaternion: [
                channels[3] as f64,
                channels[4] as f64,
                channels[5] as f64,
                channels[6] as f64,
            ],
            linear_velocity: [channels[7] as f64, channels[8] as f64, channels[9] as f64],
            angular_velocity: [
                channels[10] as f64,
                channels[11] as f64,
                channels[12] as f64,
            ],
            timestamp: 0.0,
        }
    }

    /// A hovering state at the origin at the given altitude.
    pub fn hover(altitude: f64) -> Self {
        Self {
            position: [0.0, 0.0, altitude],
            quaternion: [1.0, 0.0, 0.0, 0.0], // Identity rotation
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            timestamp: 0.0,
        }
    }
}

/// Motor command output (4D): thrust + 3 body moments.
///
/// Matches the Crazyflie 2 actuator layout from MuJoCo Menagerie:
/// - `thrust`: total vertical force [0, 0.6] N (hover ≈ 0.265)
/// - `roll_moment`: body x-axis torque [-0.005, 0.005] Nm
/// - `pitch_moment`: body y-axis torque [-0.005, 0.005] Nm
/// - `yaw_moment`: body z-axis torque [-0.002, 0.002] Nm
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QuadrotorCommand {
    pub thrust: f32,
    pub roll_moment: f32,
    pub pitch_moment: f32,
    pub yaw_moment: f32,
}

impl QuadrotorCommand {
    /// Hover thrust for Crazyflie 2 (mass × g ≈ 0.027 × 9.81).
    pub const HOVER_THRUST: f32 = 0.265;

    /// Maximum thrust (sum of 4 rotors).
    pub const MAX_THRUST: f32 = 0.6;

    /// Maximum roll/pitch moment magnitude.
    pub const MAX_MOMENT_RP: f32 = 0.005;

    /// Maximum yaw moment magnitude.
    pub const MAX_MOMENT_YAW: f32 = 0.002;

    /// A hover command (thrust at hover, zero moments).
    pub fn hover() -> Self {
        Self {
            thrust: Self::HOVER_THRUST,
            roll_moment: 0.0,
            pitch_moment: 0.0,
            yaw_moment: 0.0,
        }
    }

    /// Zero command (motors off).
    pub fn zero() -> Self {
        Self {
            thrust: 0.0,
            roll_moment: 0.0,
            pitch_moment: 0.0,
            yaw_moment: 0.0,
        }
    }

    /// Clamp all channels to actuator ranges.
    pub fn clamped(self) -> Self {
        Self {
            thrust: self.thrust.clamp(0.0, Self::MAX_THRUST),
            roll_moment: self
                .roll_moment
                .clamp(-Self::MAX_MOMENT_RP, Self::MAX_MOMENT_RP),
            pitch_moment: self
                .pitch_moment
                .clamp(-Self::MAX_MOMENT_RP, Self::MAX_MOMENT_RP),
            yaw_moment: self
                .yaw_moment
                .clamp(-Self::MAX_MOMENT_YAW, Self::MAX_MOMENT_YAW),
        }
    }

    /// Convert to f64 array for MuJoCo ctrl.
    pub fn to_ctrl(&self) -> [f64; 4] {
        [
            self.thrust as f64,
            self.roll_moment as f64,
            self.pitch_moment as f64,
            self.yaw_moment as f64,
        ]
    }

    /// Construct from raw 4D output (pre-clamped).
    pub fn from_raw(values: [f32; 4]) -> Self {
        Self {
            thrust: values[0],
            roll_moment: values[1],
            pitch_moment: values[2],
            yaw_moment: values[3],
        }
    }

    /// Add exploration noise to the command.
    pub fn with_noise(self, noise: &QuadrotorCommand) -> Self {
        Self {
            thrust: self.thrust + noise.thrust,
            roll_moment: self.roll_moment + noise.roll_moment,
            pitch_moment: self.pitch_moment + noise.pitch_moment,
            yaw_moment: self.yaw_moment + noise.yaw_moment,
        }
        .clamped()
    }
}

/// Target setpoint for the controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightSetpoint {
    /// Target position [x, y, z] in meters.
    pub position: [f64; 3],
    /// Target yaw angle in radians.
    pub yaw: f64,
}

impl FlightSetpoint {
    /// Default hover setpoint: 10cm altitude, zero yaw.
    pub fn hover() -> Self {
        Self {
            position: [0.0, 0.0, 0.1],
            yaw: 0.0,
        }
    }

    /// Position error between current state and setpoint.
    pub fn position_error(&self, state: &FlightState) -> [f64; 3] {
        [
            self.position[0] - state.position[0],
            self.position[1] - state.position[1],
            self.position[2] - state.position[2],
        ]
    }

    /// Scalar position error magnitude.
    pub fn position_error_magnitude(&self, state: &FlightState) -> f64 {
        let e = self.position_error(state);
        (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
    }
}

impl Default for FlightSetpoint {
    fn default() -> Self {
        Self::hover()
    }
}

/// Per-step telemetry from the flight controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightTelemetry {
    /// Current simulation step.
    pub step: usize,
    /// Simulation time.
    pub time: f64,
    /// Position error magnitude.
    pub position_error: f64,
    /// Attitude error (roll² + pitch²).
    pub attitude_error: f64,
    /// Speed.
    pub speed: f64,
    /// Altitude.
    pub altitude: f64,
    /// Free energy (from FEP agent, updated at cognitive rate).
    pub free_energy: f64,
    /// Current τ modulation factor.
    pub tau_factor: f32,
    /// Current learning rate.
    pub learning_rate: f32,
    /// Command sent to physics.
    pub command: QuadrotorCommand,
}

/// Configuration for the flight training system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightConfig {
    /// Motor loop frequency in Hz (default: 500).
    pub motor_hz: f64,
    /// Cognitive tick frequency in Hz (default: 25).
    pub cognitive_hz: f64,
    /// Training learning rate.
    pub learning_rate: f32,
    /// Number of HDC-LTC network layers.
    pub network_layers: usize,
    /// Neurons per layer.
    pub neurons_per_layer: usize,
    /// Number of HDC level codebook entries.
    pub num_levels: usize,
    /// Number of training episodes.
    pub num_episodes: usize,
    /// Steps per episode.
    pub steps_per_episode: usize,
    /// Training frequency divider (train every N motor steps).
    pub train_every: usize,
    /// Genesis seed phrase for deterministic initialization.
    pub genesis_phrase: String,
    /// Whether to collect per-step telemetry during episodes (default: false).
    pub collect_telemetry: bool,
    /// Experience replay buffer size (default: 64). Set to 0 to disable replay.
    pub replay_buffer_size: usize,
    /// Number of replay samples per training step (default: 3).
    pub replay_count: usize,
    /// Enable cosine annealing LR schedule across episodes (default: true).
    pub enable_lr_schedule: bool,
    /// Enable early termination on crash or divergence (default: true).
    /// Terminates episode if position error > 5m or altitude < 0.
    pub early_termination: bool,
    /// Custom curriculum of setpoints cycled across episodes.
    /// Empty (default) uses the built-in 5-setpoint cycle.
    pub curriculum: Vec<FlightSetpoint>,
    /// Use PID baseline (with integral term) instead of PD for training targets.
    /// When true, `pid_baseline()` generates targets; when false (default), `pd_baseline()`.
    pub use_pid: bool,
}

impl Default for FlightConfig {
    fn default() -> Self {
        Self {
            motor_hz: 500.0,
            cognitive_hz: 25.0,
            learning_rate: 0.001,
            network_layers: 2,
            neurons_per_layer: 4,
            num_levels: 32,
            num_episodes: 100,
            steps_per_episode: 2000, // 4s at 500Hz
            train_every: 4,          // 125Hz training
            genesis_phrase: "symthaea-multirotor-quadrotor".to_string(),
            collect_telemetry: false,
            replay_buffer_size: 64,
            replay_count: 3,
            enable_lr_schedule: true,
            early_termination: true,
            curriculum: Vec::new(),
            use_pid: false,
        }
    }
}

impl FlightConfig {
    /// Motor timestep in seconds.
    pub fn motor_dt(&self) -> f64 {
        1.0 / self.motor_hz
    }

    /// Cognitive tick interval in motor steps.
    pub fn cognitive_interval(&self) -> usize {
        (self.motor_hz / self.cognitive_hz) as usize
    }
}

/// PD controller gains for generating baseline training targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdGains {
    /// Position proportional gain.
    pub kp_pos: f64,
    /// Position derivative gain.
    pub kd_pos: f64,
    /// Attitude proportional gain.
    pub kp_att: f64,
    /// Attitude derivative gain.
    pub kd_att: f64,
    /// Yaw proportional gain.
    pub kp_yaw: f64,
    /// Yaw derivative gain.
    pub kd_yaw: f64,
    /// Position integral gain (used by `pid_baseline`).
    pub ki_pos: f64,
}

impl Default for PdGains {
    fn default() -> Self {
        Self {
            kp_pos: 8.0,
            kd_pos: 4.0,
            kp_att: 0.04,
            kd_att: 0.01,
            kp_yaw: 0.01,
            kd_yaw: 0.005,
            ki_pos: 2.0,
        }
    }
}

/// PID controller state (integral accumulators).
#[derive(Debug, Clone, Default)]
pub struct PidState {
    /// Integral accumulator for position error [x, y, z].
    pub integral: [f64; 3],
}

/// Compute PD baseline command for a given state and setpoint.
///
/// This serves as the training target: the CfC network learns to imitate
/// this PD controller, while the FEP layer modulates adaptation dynamics.
pub fn pd_baseline(
    state: &FlightState,
    setpoint: &FlightSetpoint,
    gains: &PdGains,
) -> QuadrotorCommand {
    // Position error → thrust adjustment
    let pos_err = setpoint.position_error(state);
    let z_err = pos_err[2];
    let vz = state.linear_velocity[2];
    let thrust_adj = gains.kp_pos * z_err - gains.kd_pos * vz;
    let thrust = (QuadrotorCommand::HOVER_THRUST as f64 + thrust_adj)
        .clamp(0.0, QuadrotorCommand::MAX_THRUST as f64);

    // Attitude error → moment commands
    let (roll, pitch, yaw) = state.euler_angles();
    let [wx, wy, wz] = state.angular_velocity;

    // Desired roll/pitch from lateral position error
    let desired_pitch =
        (gains.kp_pos * pos_err[0] - gains.kd_pos * state.linear_velocity[0]).clamp(-0.3, 0.3);
    let desired_roll =
        -(gains.kp_pos * pos_err[1] - gains.kd_pos * state.linear_velocity[1]).clamp(-0.3, 0.3);

    let roll_moment = gains.kp_att * (desired_roll - roll) - gains.kd_att * wx;
    let pitch_moment = gains.kp_att * (desired_pitch - pitch) - gains.kd_att * wy;

    // Yaw control
    let yaw_err = angle_wrap(setpoint.yaw - yaw);
    let yaw_moment = gains.kp_yaw * yaw_err - gains.kd_yaw * wz;

    QuadrotorCommand {
        thrust: thrust as f32,
        roll_moment: roll_moment as f32,
        pitch_moment: pitch_moment as f32,
        yaw_moment: yaw_moment as f32,
    }
    .clamped()
}

/// Compute PID baseline command (PD + integral term for zero steady-state error).
///
/// The integral term eliminates steady-state offset under sustained disturbances
/// (e.g., constant wind). Includes anti-windup clamping and zero-crossing reset.
pub fn pid_baseline(
    state: &FlightState,
    setpoint: &FlightSetpoint,
    gains: &PdGains,
    pid_state: &mut PidState,
    dt: f64,
) -> QuadrotorCommand {
    let pos_err = setpoint.position_error(state);

    // Update integral with anti-windup
    for i in 0..3 {
        // Zero-crossing reset: halve integral when error changes sign
        if pid_state.integral[i] != 0.0 && pid_state.integral[i].signum() != pos_err[i].signum() {
            pid_state.integral[i] *= 0.5;
        }
        pid_state.integral[i] += pos_err[i] * dt;
        pid_state.integral[i] = pid_state.integral[i].clamp(-0.5, 0.5);
    }

    // Thrust: PD + I on z-axis
    let z_err = pos_err[2];
    let vz = state.linear_velocity[2];
    let thrust_adj =
        gains.kp_pos * z_err - gains.kd_pos * vz + gains.ki_pos * pid_state.integral[2];
    let thrust = (QuadrotorCommand::HOVER_THRUST as f64 + thrust_adj)
        .clamp(0.0, QuadrotorCommand::MAX_THRUST as f64);

    // Attitude error → moment commands (same as pd_baseline)
    let (roll, pitch, yaw) = state.euler_angles();
    let [wx, wy, wz] = state.angular_velocity;

    let desired_pitch = (gains.kp_pos * pos_err[0] - gains.kd_pos * state.linear_velocity[0]
        + gains.ki_pos * pid_state.integral[0])
        .clamp(-0.3, 0.3);
    let desired_roll = -(gains.kp_pos * pos_err[1] - gains.kd_pos * state.linear_velocity[1]
        + gains.ki_pos * pid_state.integral[1])
        .clamp(-0.3, 0.3);

    let roll_moment = gains.kp_att * (desired_roll - roll) - gains.kd_att * wx;
    let pitch_moment = gains.kp_att * (desired_pitch - pitch) - gains.kd_att * wy;

    let yaw_err = angle_wrap(setpoint.yaw - yaw);
    let yaw_moment = gains.kp_yaw * yaw_err - gains.kd_yaw * wz;

    QuadrotorCommand {
        thrust: thrust as f32,
        roll_moment: roll_moment as f32,
        pitch_moment: pitch_moment as f32,
        yaw_moment: yaw_moment as f32,
    }
    .clamped()
}

/// Wrap angle to [-π, π].
fn angle_wrap(a: f64) -> f64 {
    let mut a = a % (2.0 * std::f64::consts::PI);
    if a > std::f64::consts::PI {
        a -= 2.0 * std::f64::consts::PI;
    } else if a < -std::f64::consts::PI {
        a += 2.0 * std::f64::consts::PI;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flight_state_hover() {
        let state = FlightState::hover(0.1);
        assert!((state.altitude() - 0.1).abs() < 1e-10);
        assert!(state.speed() < 1e-10);
        assert!(state.angular_speed() < 1e-10);
    }

    #[test]
    fn test_flight_state_euler_identity() {
        let state = FlightState::hover(0.0);
        let (roll, pitch, yaw) = state.euler_angles();
        assert!(roll.abs() < 1e-10);
        assert!(pitch.abs() < 1e-10);
        assert!(yaw.abs() < 1e-10);
    }

    #[test]
    fn test_flight_state_from_qpos_qvel() {
        let qpos = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0];
        let qvel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let state = FlightState::from_qpos_qvel(&qpos, &qvel, 1.5);
        assert_eq!(state.position, [1.0, 2.0, 3.0]);
        assert_eq!(state.quaternion, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(state.linear_velocity, [0.1, 0.2, 0.3]);
        assert_eq!(state.angular_velocity, [0.4, 0.5, 0.6]);
        assert!((state.timestamp - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_from_channels_roundtrip() {
        let state = FlightState {
            position: [1.5, -0.3, 0.7],
            quaternion: [0.9, 0.1, 0.2, 0.3],
            linear_velocity: [0.5, -1.0, 0.3],
            angular_velocity: [2.0, -1.5, 0.8],
            timestamp: 42.0,
        };
        let channels = state.to_channels();
        let restored = FlightState::from_channels(channels);
        for i in 0..3 {
            assert!(
                (state.position[i] - restored.position[i]).abs() < 1e-5,
                "position[{i}] mismatch"
            );
            assert!(
                (state.linear_velocity[i] - restored.linear_velocity[i]).abs() < 1e-5,
                "linear_velocity[{i}] mismatch"
            );
            assert!(
                (state.angular_velocity[i] - restored.angular_velocity[i]).abs() < 1e-5,
                "angular_velocity[{i}] mismatch"
            );
        }
        for i in 0..4 {
            assert!(
                (state.quaternion[i] - restored.quaternion[i]).abs() < 1e-5,
                "quaternion[{i}] mismatch"
            );
        }
        // Timestamp is lost in channels
        assert!((restored.timestamp - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadrotor_command_hover() {
        let cmd = QuadrotorCommand::hover();
        assert!((cmd.thrust - QuadrotorCommand::HOVER_THRUST).abs() < 1e-6);
        assert!(cmd.roll_moment.abs() < 1e-10);
        assert!(cmd.pitch_moment.abs() < 1e-10);
        assert!(cmd.yaw_moment.abs() < 1e-10);
    }

    #[test]
    fn test_quadrotor_command_clamped() {
        let cmd = QuadrotorCommand {
            thrust: 10.0,
            roll_moment: 1.0,
            pitch_moment: -1.0,
            yaw_moment: 0.5,
        };
        let clamped = cmd.clamped();
        assert!((clamped.thrust - QuadrotorCommand::MAX_THRUST).abs() < 1e-6);
        assert!((clamped.roll_moment - QuadrotorCommand::MAX_MOMENT_RP).abs() < 1e-6);
        assert!((clamped.pitch_moment + QuadrotorCommand::MAX_MOMENT_RP).abs() < 1e-6);
        assert!((clamped.yaw_moment - QuadrotorCommand::MAX_MOMENT_YAW).abs() < 1e-6);
    }

    #[test]
    fn test_quadrotor_command_to_ctrl() {
        let cmd = QuadrotorCommand::hover();
        let ctrl = cmd.to_ctrl();
        assert_eq!(ctrl.len(), 4);
        assert!((ctrl[0] - QuadrotorCommand::HOVER_THRUST as f64).abs() < 1e-6);
    }

    #[test]
    fn test_flight_setpoint_error() {
        let setpoint = FlightSetpoint::hover();
        let state = FlightState::hover(0.0);
        let err = setpoint.position_error(&state);
        assert!((err[2] - 0.1).abs() < 1e-10); // 10cm above
    }

    #[test]
    fn test_pd_baseline_at_hover() {
        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();
        let gains = PdGains::default();
        let cmd = pd_baseline(&state, &setpoint, &gains);
        // At setpoint: thrust ≈ hover, moments ≈ 0
        assert!((cmd.thrust - QuadrotorCommand::HOVER_THRUST).abs() < 0.01);
        assert!(cmd.roll_moment.abs() < 0.001);
        assert!(cmd.pitch_moment.abs() < 0.001);
        assert!(cmd.yaw_moment.abs() < 0.001);
    }

    #[test]
    fn test_pd_baseline_below_setpoint() {
        let state = FlightState::hover(0.05); // Below setpoint
        let setpoint = FlightSetpoint::hover(); // z=0.1
        let gains = PdGains::default();
        let cmd = pd_baseline(&state, &setpoint, &gains);
        // Should command more thrust to climb
        assert!(cmd.thrust > QuadrotorCommand::HOVER_THRUST);
    }

    #[test]
    fn test_flight_state_serde_roundtrip() {
        let state = FlightState::hover(0.1);
        let json = serde_json::to_string(&state).unwrap();
        let restored: FlightState = serde_json::from_str(&json).unwrap();
        assert_eq!(state.position, restored.position);
        assert_eq!(state.quaternion, restored.quaternion);
    }

    #[test]
    fn test_quadrotor_command_serde_roundtrip() {
        let cmd = QuadrotorCommand::hover();
        let json = serde_json::to_string(&cmd).unwrap();
        let restored: QuadrotorCommand = serde_json::from_str(&json).unwrap();
        assert!((cmd.thrust - restored.thrust).abs() < 1e-10);
    }

    #[test]
    fn test_flight_config_defaults() {
        let config = FlightConfig::default();
        assert!((config.motor_dt() - 0.002).abs() < 1e-10);
        assert_eq!(config.cognitive_interval(), 20);
        assert!(config.curriculum.is_empty());
    }

    #[test]
    fn test_to_channels() {
        let state = FlightState::hover(0.1);
        let ch = state.to_channels();
        assert_eq!(ch.len(), 13);
        assert!((ch[2] - 0.1).abs() < 1e-6); // z
        assert!((ch[3] - 1.0).abs() < 1e-6); // qw = 1
    }

    #[test]
    fn test_pid_reduces_steady_state_error() {
        // PID with pre-accumulated integral should command more thrust than PD
        let setpoint = FlightSetpoint::hover(); // z=0.1
        let gains = PdGains::default();
        let state = FlightState::hover(0.08); // 2cm below setpoint

        // PD baseline
        let pd_cmd = pd_baseline(&state, &setpoint, &gains);

        // PID with pre-accumulated integral (simulates persistent offset)
        let mut pid_state = PidState {
            integral: [0.0, 0.0, 0.02],
        };
        let pid_cmd = pid_baseline(&state, &setpoint, &gains, &mut pid_state, 0.002);

        // PID should command MORE thrust due to integral term
        // PD: thrust_adj = 8.0 * 0.02 = 0.16 → thrust = 0.425
        // PID: thrust_adj = 0.16 + 2.0 * 0.02 = 0.20 → thrust = 0.465
        assert!(
            pid_cmd.thrust > pd_cmd.thrust,
            "PID with integral should command more thrust: pid={:.6}, pd={:.6}",
            pid_cmd.thrust,
            pd_cmd.thrust
        );
    }

    #[test]
    fn test_pid_anti_windup() {
        let setpoint = FlightSetpoint {
            position: [0.0, 0.0, 10.0], // Huge offset to saturate integral
            yaw: 0.0,
        };
        let gains = PdGains::default();
        let mut pid_state = PidState::default();
        let state = FlightState::hover(0.1);
        let dt = 0.002;

        // Run many steps to try to blow up the integral
        for _ in 0..10_000 {
            pid_baseline(&state, &setpoint, &gains, &mut pid_state, dt);
        }

        // Integral should be clamped to ±0.5
        for i in 0..3 {
            assert!(
                pid_state.integral[i].abs() <= 0.5 + 1e-10,
                "Integral[{i}] should be clamped: {}",
                pid_state.integral[i]
            );
        }
    }

    #[test]
    fn test_curriculum_custom_cycle() {
        let config = FlightConfig {
            curriculum: vec![
                FlightSetpoint {
                    position: [0.0, 0.0, 0.2],
                    yaw: 0.0,
                },
                FlightSetpoint {
                    position: [0.1, 0.0, 0.1],
                    yaw: 0.0,
                },
            ],
            ..FlightConfig::default()
        };
        assert_eq!(config.curriculum.len(), 2);
        // Episode 0 → curriculum[0], episode 1 → curriculum[1], episode 2 → curriculum[0]
        assert!((config.curriculum[0 % 2].position[2] - 0.2).abs() < 1e-10);
        assert!((config.curriculum[1 % 2].position[0] - 0.1).abs() < 1e-10);
        assert!((config.curriculum[2 % 2].position[2] - 0.2).abs() < 1e-10);
    }
}
