// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core data types for autonomous vehicle control.

use serde::{Deserialize, Serialize};

/// Number of actuators: steering, throttle, brake.
pub const NUM_ACTUATORS: usize = 3;

/// Number of state channels for HDC encoding (20D: 14 proprioceptive + 2 road + 4 mesh).
pub const NUM_STATE_CHANNELS: usize = 20;

/// Actuator indices.
pub const ACT_STEERING: usize = 0;
pub const ACT_THROTTLE: usize = 1;
pub const ACT_BRAKE: usize = 2;

/// Actuator names.
pub const ACTUATOR_NAMES: [&str; NUM_ACTUATORS] = ["steering", "throttle", "brake"];

/// Cross-platform right-of-way advisory from civic wildlife / cohabitation systems.
///
/// This stays outside the core vehicle state packing so the civic signal can
/// drive safety composition without perturbing historical encoder geometry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BiotaRightOfWaySignal {
    /// Animal distress intensity (0..1).
    pub distress_signal: f32,
    /// Predicted ego-path conflict with a sanctuary or crossing zone (0..1).
    pub path_conflict_risk: f32,
    /// Strength of the active sanctuary / protected-crossing claim (0..1).
    pub sanctuary_signal: f32,
    /// Confidence that the route is clear without intervention (0..1).
    pub route_clear_confidence: f32,
    /// Confidence the upstream animal classification is valid (0..1).
    pub classification_confidence: f32,
}

impl BiotaRightOfWaySignal {
    pub fn clear() -> Self {
        Self {
            distress_signal: 0.0,
            path_conflict_risk: 0.0,
            sanctuary_signal: 0.0,
            route_clear_confidence: 1.0,
            classification_confidence: 1.0,
        }
    }

    pub fn should_hold_red(&self) -> bool {
        self.classification_confidence >= 0.45
            && ((self.distress_signal >= 0.75 && self.path_conflict_risk >= 0.6)
                || (self.sanctuary_signal >= 0.85 && self.path_conflict_risk >= 0.7))
    }

    pub fn should_yield_orange(&self) -> bool {
        self.classification_confidence >= 0.35
            && (self.sanctuary_signal >= 0.55
                || self.path_conflict_risk >= 0.45
                || self.route_clear_confidence <= 0.45)
    }

    pub fn should_caution_yellow(&self) -> bool {
        self.classification_confidence >= 0.25
            && (self.path_conflict_risk >= 0.25
                || self.sanctuary_signal >= 0.25
                || self.route_clear_confidence <= 0.7)
    }
}

/// Channel names matching the `to_channels()` layout.
pub const CHANNEL_NAMES: [&str; NUM_STATE_CHANNELS] = [
    // Proprioceptive core (14)
    "speed",
    "lateral_velocity",
    "yaw_rate",
    "heading",
    "position_x",
    "position_y",
    "steering_angle",
    "throttle_position",
    "brake_pressure",
    "longitudinal_accel",
    "lateral_accel",
    "tire_slip_front",
    "tire_slip_rear",
    "pitch",
    // Road vector (2)
    "lateral_offset",
    "curvature_ahead",
    // Swarm vector (4)
    "nearest_peer_distance",
    "peer_relative_speed",
    "mesh_brake_density",
    "mesh_confidence",
];

/// Full vehicle state (20D): proprioceptive + road + mesh channels.
///
/// Channel layout:
/// ── Proprioceptive Core (14) ──
/// [0]  speed             — longitudinal velocity (m/s)
/// [1]  lateral_velocity  — lateral velocity (m/s), positive = leftward
/// [2]  yaw_rate          — angular velocity around vertical (rad/s)
/// [3]  heading           — yaw angle in world frame (rad), 0 = east
/// [4]  position_x        — world x position (m)
/// [5]  position_y        — world y position (m)
/// [6]  steering_angle    — current front wheel angle (rad)
/// [7]  throttle_position — current throttle (0..1)
/// [8]  brake_pressure    — current brake (0..1)
/// [9]  longitudinal_accel — forward acceleration (m/s²)
/// [10] lateral_accel     — lateral acceleration (m/s²)
/// [11] tire_slip_front   — front tire slip angle magnitude (rad)
/// [12] tire_slip_rear    — rear tire slip angle magnitude (rad)
/// [13] pitch             — pitch angle from weight transfer (rad)
/// ── Road Vector (2) ──
/// [14] lateral_offset    — distance from lane center (m), positive = left
/// [15] curvature_ahead   — road curvature at look-ahead point (1/m)
/// ── Swarm Vector (4) ──
/// [16] nearest_peer_distance  — distance to nearest mesh peer (m)
/// [17] peer_relative_speed    — speed of nearest peer minus ego speed (m/s)
/// [18] mesh_brake_density     — fraction of swarm currently braking (0..1)
/// [19] mesh_confidence        — aggregate mesh signal integrity (0..1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleState {
    // ── Proprioceptive Core (14) ──
    /// Longitudinal velocity (m/s). Always non-negative for forward motion.
    pub speed: f64,
    /// Lateral velocity (m/s). Positive = leftward in body frame.
    pub lateral_velocity: f64,
    /// Yaw rate (rad/s). Positive = counter-clockwise.
    pub yaw_rate: f64,
    /// Heading angle in world frame (rad). 0 = east, pi/2 = north.
    pub heading: f64,
    /// World-frame x position (m).
    pub position_x: f64,
    /// World-frame y position (m).
    pub position_y: f64,
    /// Current front wheel steering angle (rad). Positive = left.
    pub steering_angle: f64,
    /// Current throttle position (0..1).
    pub throttle_position: f64,
    /// Current brake pressure (0..1).
    pub brake_pressure: f64,
    /// Longitudinal acceleration (m/s²).
    pub longitudinal_accel: f64,
    /// Lateral acceleration (m/s²).
    pub lateral_accel: f64,
    /// Front tire slip angle magnitude (rad).
    pub tire_slip_front: f64,
    /// Rear tire slip angle magnitude (rad).
    pub tire_slip_rear: f64,
    /// Pitch angle from weight transfer (rad). Positive = nose up (braking).
    pub pitch: f64,

    // ── Road Vector (2) ──
    /// Lateral offset from lane center (m). Positive = left of center.
    /// Set by the training loop from the road model. Zero when no road is defined.
    pub lateral_offset: f64,
    /// Road curvature at look-ahead point (1/m). Positive = curving left.
    /// Set by the training loop from the road model. Zero for straight roads.
    pub curvature_ahead: f64,

    // ── Swarm Vector (4) ──
    /// Distance to nearest mesh peer (m). 200.0 = no peers visible.
    pub nearest_peer_distance: f64,
    /// Speed of nearest peer minus ego speed (m/s). Negative = peer braking.
    pub peer_relative_speed: f64,
    /// Fraction of swarm currently applying brakes (0..1).
    pub mesh_brake_density: f64,
    /// Aggregate mesh signal integrity (0..1). 0 = no mesh, 1 = full confidence.
    pub mesh_confidence: f64,

    // ── Metadata ──
    /// Simulation time (s).
    pub timestamp: f64,
}

impl VehicleState {
    /// Pack all 20 state channels into an f32 array for HDC encoding.
    pub fn to_channels(&self) -> Vec<f32> {
        vec![
            // Proprioceptive (14)
            self.speed as f32,
            self.lateral_velocity as f32,
            self.yaw_rate as f32,
            self.heading as f32,
            self.position_x as f32,
            self.position_y as f32,
            self.steering_angle as f32,
            self.throttle_position as f32,
            self.brake_pressure as f32,
            self.longitudinal_accel as f32,
            self.lateral_accel as f32,
            self.tire_slip_front as f32,
            self.tire_slip_rear as f32,
            self.pitch as f32,
            // Road (2)
            self.lateral_offset as f32,
            self.curvature_ahead as f32,
            // Mesh (4)
            self.nearest_peer_distance as f32,
            self.peer_relative_speed as f32,
            self.mesh_brake_density as f32,
            self.mesh_confidence as f32,
        ]
    }

    /// Number of channels produced by `to_channels()`.
    pub fn num_channels() -> usize {
        NUM_STATE_CHANNELS
    }

    /// Default state: stopped, facing east, at origin. No road, no mesh.
    pub fn stopped() -> Self {
        Self {
            speed: 0.0,
            lateral_velocity: 0.0,
            yaw_rate: 0.0,
            heading: 0.0,
            position_x: 0.0,
            position_y: 0.0,
            steering_angle: 0.0,
            throttle_position: 0.0,
            brake_pressure: 0.0,
            longitudinal_accel: 0.0,
            lateral_accel: 0.0,
            tire_slip_front: 0.0,
            tire_slip_rear: 0.0,
            pitch: 0.0,
            lateral_offset: 0.0,
            curvature_ahead: 0.0,
            nearest_peer_distance: 200.0, // no peers
            peer_relative_speed: 0.0,
            mesh_brake_density: 0.0,
            mesh_confidence: 0.0,
            timestamp: 0.0,
        }
    }

    /// Default state: cruising at a given speed (m/s), heading east.
    pub fn cruising(speed: f64) -> Self {
        Self {
            speed,
            ..Self::stopped()
        }
    }

    /// Total speed magnitude (longitudinal + lateral).
    pub fn total_speed(&self) -> f64 {
        (self.speed * self.speed + self.lateral_velocity * self.lateral_velocity).sqrt()
    }

    /// Sideslip angle (body slip angle) in radians.
    pub fn sideslip_angle(&self) -> f64 {
        if self.speed.abs() < 0.1 {
            0.0
        } else {
            (self.lateral_velocity / self.speed).atan()
        }
    }

    /// Whether all state values are finite (NaN/Inf check).
    pub fn is_finite(&self) -> bool {
        self.speed.is_finite()
            && self.lateral_velocity.is_finite()
            && self.yaw_rate.is_finite()
            && self.heading.is_finite()
            && self.position_x.is_finite()
            && self.position_y.is_finite()
            && self.steering_angle.is_finite()
            && self.longitudinal_accel.is_finite()
            && self.lateral_accel.is_finite()
            && self.tire_slip_front.is_finite()
            && self.tire_slip_rear.is_finite()
            && self.pitch.is_finite()
            && self.lateral_offset.is_finite()
            && self.curvature_ahead.is_finite()
            && self.nearest_peer_distance.is_finite()
            && self.peer_relative_speed.is_finite()
            && self.mesh_brake_density.is_finite()
            && self.mesh_confidence.is_finite()
    }

    /// Domain-specific bounds check for physically plausible vehicle state.
    ///
    /// Returns `true` if the state is within reasonable physical limits:
    /// - Speed: [0, 80] m/s (~180 mph, absolute road vehicle limit)
    /// - Yaw rate: [-3, 3] rad/s (reasonable for any road vehicle)
    /// - Lateral velocity: [-20, 20] m/s (extreme drift scenarios)
    pub fn is_bounded(&self) -> bool {
        self.is_finite()
            && self.speed >= 0.0
            && self.speed < 80.0
            && self.yaw_rate.abs() < 3.0
            && self.lateral_velocity.abs() < 20.0
    }

    /// Inject straight-road context: lateral_offset = position_y, curvature = 0.
    ///
    /// Call this before encoding when no road model is defined (Phase A).
    pub fn apply_straight_road(&mut self) {
        self.lateral_offset = self.position_y;
        self.curvature_ahead = 0.0;
    }

    /// Apply sensor blindness: zero out the first `n` proprioceptive channels.
    ///
    /// Simulates camera/LiDAR degradation from rain, fog, or sensor failure.
    /// Channels are zeroed in order: speed, lateral_velocity, yaw_rate, heading,
    /// position_x/y, steering_angle, throttle/brake, accel, tire slip, pitch.
    pub fn apply_sensor_blindness(&mut self, n: usize) {
        if n >= 1 {
            self.speed = 0.0;
        }
        if n >= 2 {
            self.lateral_velocity = 0.0;
        }
        if n >= 3 {
            self.yaw_rate = 0.0;
        }
        if n >= 4 {
            self.heading = 0.0;
        }
        if n >= 5 {
            self.position_x = 0.0;
        }
        if n >= 6 {
            self.position_y = 0.0;
        }
        if n >= 7 {
            self.steering_angle = 0.0;
        }
        if n >= 8 {
            self.throttle_position = 0.0;
        }
        if n >= 9 {
            self.brake_pressure = 0.0;
        }
        if n >= 10 {
            self.longitudinal_accel = 0.0;
        }
        if n >= 11 {
            self.lateral_accel = 0.0;
        }
        if n >= 12 {
            self.tire_slip_front = 0.0;
        }
        if n >= 13 {
            self.tire_slip_rear = 0.0;
        }
        if n >= 14 {
            self.pitch = 0.0;
        }
    }

    /// Apply mesh spoof: inject false brake signals from spoofed peers.
    ///
    /// Each spoofed peer adds 0.25 to brake density (capped at 1.0) and
    /// inflates mesh_confidence (spoofed data claims high confidence).
    pub fn apply_mesh_spoof(&mut self, num_spoofed_peers: usize) {
        let spoof_fraction = (num_spoofed_peers as f64 * 0.25).min(1.0);
        self.mesh_brake_density = (self.mesh_brake_density + spoof_fraction).min(1.0);
        self.mesh_confidence = (self.mesh_confidence + spoof_fraction * 0.5).min(1.0);
    }
}

/// Motor command output (3D): steering, throttle, brake.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VehicleCommand {
    /// Steering command: -1.0 = full right, +1.0 = full left.
    pub steering: f32,
    /// Throttle command: 0.0 = off, 1.0 = full power.
    pub throttle: f32,
    /// Brake command: 0.0 = off, 1.0 = full braking force.
    pub brake: f32,
}

impl VehicleCommand {
    /// Zero command (coasting, no input).
    pub fn zero() -> Self {
        Self {
            steering: 0.0,
            throttle: 0.0,
            brake: 0.0,
        }
    }

    /// Construct from a raw f32 slice, clamped to valid ranges.
    pub fn from_raw(values: &[f32]) -> Self {
        Self {
            steering: values.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0),
            throttle: values.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0),
            brake: values.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0),
        }
    }

    /// Clamp all commands to valid ranges.
    pub fn clamped(self) -> Self {
        Self {
            steering: self.steering.clamp(-1.0, 1.0),
            throttle: self.throttle.clamp(0.0, 1.0),
            brake: self.brake.clamp(0.0, 1.0),
        }
    }

    /// Convert to f64 array [steering, throttle, brake].
    pub fn to_ctrl(&self) -> [f64; NUM_ACTUATORS] {
        [
            self.steering as f64,
            self.throttle as f64,
            self.brake as f64,
        ]
    }

    /// Mean absolute command (control effort proxy).
    pub fn control_effort(&self) -> f32 {
        (self.steering.abs() + self.throttle.abs() + self.brake.abs()) / NUM_ACTUATORS as f32
    }

    /// Add exploration noise, then clamp.
    pub fn with_noise(self, noise: [f32; NUM_ACTUATORS]) -> Self {
        Self {
            steering: (self.steering + noise[ACT_STEERING]).clamp(-1.0, 1.0),
            throttle: (self.throttle + noise[ACT_THROTTLE]).clamp(0.0, 1.0),
            brake: (self.brake + noise[ACT_BRAKE]).clamp(0.0, 1.0),
        }
    }
}

/// Driving task variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum VehicleTask {
    /// Maintain lane position at a target velocity.
    #[default]
    LaneKeep,
    /// Follow a lead vehicle (adaptive cruise).
    Follow { target_gap: f64 },
    /// Execute a lane change maneuver.
    LaneChange,
    /// Emergency stop from speed.
    EmergencyStop,
}

impl VehicleTask {
    /// Target speed for this task (m/s).
    pub fn target_speed(&self) -> f64 {
        match self {
            VehicleTask::LaneKeep => 13.4,      // ~30 mph / ~48 km/h
            VehicleTask::Follow { .. } => 13.4, // match lead
            VehicleTask::LaneChange => 13.4,    // maintain during maneuver
            VehicleTask::EmergencyStop => 0.0,  // stop
        }
    }
}

/// Swarm mesh signal from another vehicle in the flock.
///
/// This is the 2KB Wisdom Vector concept: a compressed state broadcast
/// from a peer vehicle that Symthaea incorporates into her world model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshSignal {
    /// Peer vehicle ID.
    pub peer_id: u64,
    /// Relative position from ego vehicle (m): [longitudinal, lateral].
    pub relative_position: [f64; 2],
    /// Peer's current speed (m/s).
    pub peer_speed: f64,
    /// Peer's current yaw rate (rad/s).
    pub peer_yaw_rate: f64,
    /// Peer's brake status (true = braking).
    pub peer_braking: bool,
    /// Peer's ABS engaged (true = emergency braking detected).
    pub peer_abs_engaged: bool,
    /// Signal age / latency (seconds).
    pub latency: f64,
    /// Signal confidence (0..1), degrades with distance and latency.
    pub confidence: f64,
}

/// Per-step telemetry from the vehicle controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleTelemetry {
    pub step: usize,
    pub time: f64,
    pub speed: f64,
    pub heading: f64,
    pub lateral_offset: f64,
    pub tire_slip_front: f64,
    pub tire_slip_rear: f64,
    pub episode_reward: f64,
    pub free_energy: f64,
    pub tau_factor: f32,
    pub control_effort: f32,
    pub mesh_peers_visible: usize,
}

/// Configuration for the vehicle training system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleConfig {
    /// Physics loop frequency in Hz (default: 200, vehicle dynamics needs higher rate).
    pub physics_hz: f64,
    /// Cognitive tick frequency in Hz (default: 50, Symthaea's standard loop).
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
    /// Training frequency divider (train every N physics steps).
    pub train_every: usize,
    /// Genesis seed phrase.
    pub genesis_phrase: String,
    /// Whether to collect per-step telemetry.
    pub collect_telemetry: bool,
    /// Experience replay buffer size.
    pub replay_buffer_size: usize,
    /// Number of replay samples per training step.
    pub replay_count: usize,
    /// Enable cosine annealing LR schedule.
    pub enable_lr_schedule: bool,
    /// Enable early termination on collision/instability.
    pub early_termination: bool,
    /// Current driving task.
    pub task: VehicleTask,
    /// Target speed override (None = use task default).
    pub target_speed: Option<f64>,
}

impl Default for VehicleConfig {
    fn default() -> Self {
        Self {
            physics_hz: 200.0,
            cognitive_hz: 50.0,
            learning_rate: 0.0003,
            network_layers: 3,
            neurons_per_layer: 8,
            num_levels: 32,
            num_episodes: 500,
            steps_per_episode: 2000,
            train_every: 4, // 50Hz training
            genesis_phrase: "symthaea-vehicle-sage".to_string(),
            collect_telemetry: false,
            replay_buffer_size: 256,
            replay_count: 4,
            enable_lr_schedule: true,
            early_termination: true,
            task: VehicleTask::LaneKeep,
            target_speed: None,
        }
    }
}

impl VehicleConfig {
    /// Physics timestep in seconds.
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }

    /// Cognitive tick interval in physics steps.
    pub fn cognitive_interval(&self) -> usize {
        (self.physics_hz / self.cognitive_hz) as usize
    }

    /// Effective target speed (override or task default).
    pub fn effective_target_speed(&self) -> f64 {
        self.target_speed
            .unwrap_or_else(|| self.task.target_speed())
    }

    /// Speed at which the simulator initializes.
    ///
    /// For EmergencyStop, defaults to 13.4 m/s (need to start at cruising speed
    /// to have something to stop from). For other tasks, same as `effective_target_speed()`.
    pub fn init_speed(&self) -> f64 {
        match self.task {
            VehicleTask::EmergencyStop => self.target_speed.unwrap_or(13.4),
            _ => self.effective_target_speed(),
        }
    }
}

/// Vehicle chassis parameters for the bicycle model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChassisParams {
    /// Total vehicle mass (kg).
    pub mass: f64,
    /// Yaw moment of inertia (kg*m²).
    pub yaw_inertia: f64,
    /// Distance from CG to front axle (m).
    pub front_axle_dist: f64,
    /// Distance from CG to rear axle (m).
    pub rear_axle_dist: f64,
    /// CG height (m) — for weight transfer.
    pub cg_height: f64,
    /// Front tire cornering stiffness (N/rad).
    pub front_cornering_stiffness: f64,
    /// Rear tire cornering stiffness (N/rad).
    pub rear_cornering_stiffness: f64,
    /// Maximum steering angle (rad).
    pub max_steering_angle: f64,
    /// Steering actuator rate limit (rad/s).
    pub steering_rate_limit: f64,
    /// Maximum engine force at wheels (N) at low speed.
    pub max_engine_force: f64,
    /// Maximum engine power (W). Force is min(max_engine_force, max_power / v).
    pub max_power: f64,
    /// Maximum braking force (N).
    pub max_brake_force: f64,
    /// Aerodynamic drag coefficient * frontal area (Cd * A).
    pub drag_coeff: f64,
    /// Rolling resistance coefficient.
    pub rolling_resistance: f64,
    /// Air density (kg/m³).
    pub air_density: f64,
    /// Wheelbase (computed: front_axle_dist + rear_axle_dist).
    pub wheelbase: f64,
}

impl Default for ChassisParams {
    /// Approximate 1,500 kg sedan (e.g., a compact SUV / "Sage" platform).
    fn default() -> Self {
        let front_axle_dist = 1.2;
        let rear_axle_dist = 1.5;
        Self {
            mass: 1500.0,
            yaw_inertia: 2500.0,
            front_axle_dist,
            rear_axle_dist,
            cg_height: 0.55,
            front_cornering_stiffness: 80_000.0,
            rear_cornering_stiffness: 90_000.0,
            max_steering_angle: 0.52, // ~30 degrees
            steering_rate_limit: 1.0, // rad/s
            max_engine_force: 5000.0, // peak torque force at low speed
            max_power: 110_000.0,     // 110kW (~150hp) modest sedan
            max_brake_force: 15000.0, // ~1g braking
            drag_coeff: 0.36 * 2.2,   // Cd * A for sedan
            rolling_resistance: 0.012,
            air_density: 1.225,
            wheelbase: front_axle_dist + rear_axle_dist,
        }
    }
}

/// PD cruise-control baseline: drive toward target speed with lane correction.
///
/// Generates a target VehicleCommand from the current state.
/// This is the "teacher" that the CfC network learns to imitate, then surpass.
/// Assumes target heading is 0.0 (east / straight road).
pub fn pd_cruise_baseline(state: &VehicleState, target_speed: f64) -> VehicleCommand {
    pd_cruise_baseline_with_road(state, target_speed, 0.0)
}

/// PD cruise-control baseline with explicit road heading and curvature feed-forward.
///
/// Same as `pd_cruise_baseline` but corrects toward `road_heading` instead of 0.0,
/// and adds a curvature feed-forward term from `state.curvature_ahead`.
pub fn pd_cruise_baseline_with_road(
    state: &VehicleState,
    target_speed: f64,
    road_heading: f64,
) -> VehicleCommand {
    // Longitudinal: PD on speed error
    let speed_error = target_speed - state.speed;
    let throttle = if speed_error > 0.0 {
        (speed_error * 0.3).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let brake = if speed_error < -1.0 {
        ((-speed_error - 1.0) * 0.2).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    // Lateral: PD on lane offset + heading error relative to road
    let lane_error = -state.lateral_offset; // negative feedback: left offset → steer right
    let heading_error = -(state.heading - road_heading);
    let curvature_ff = state.curvature_ahead * 5.0; // feed-forward for upcoming curvature
    let steering =
        ((lane_error * 0.8 + heading_error * 2.0 + curvature_ff) as f32).clamp(-1.0, 1.0);

    VehicleCommand {
        steering,
        throttle,
        brake,
    }
}

/// PD cruise-control for Follow task: modulates speed based on gap to nearest peer.
///
/// Same lateral/heading control as `pd_cruise_baseline_with_road`, but the speed
/// command is adjusted by the gap error: if the peer is too close, slow down;
/// if too far, speed up (up to 120% of target).
pub fn pd_cruise_with_follow(
    state: &VehicleState,
    target_speed: f64,
    road_heading: f64,
    target_gap: f64,
) -> VehicleCommand {
    // Gap-modulated speed: speed_cmd = target_speed + gap_error * 0.3
    let gap_error = state.nearest_peer_distance - target_gap;
    let speed_cmd = (target_speed + gap_error * 0.3).clamp(0.0, target_speed * 1.2);

    pd_cruise_baseline_with_road(state, speed_cmd, road_heading)
}

/// Training-level perturbations that interact with the swarm (distinct from physics-level
/// `VehiclePerturbation`). These trigger scenario events like cascading brake lights.
#[derive(Debug, Clone)]
pub enum TrainingPerturbation {
    /// No perturbation.
    None,
    /// Trigger a lead-brake event on a specific peer at a specific step.
    LeadBrake {
        trigger_step: usize,
        peer_idx: usize,
    },
}

/// Generate a cascading lead-brake sequence for a convoy.
///
/// Peer 0 brakes at step 50, peer 1 at step 80, peer 2 at step 110, etc.
/// Simulates cascading brake lights propagating through a convoy.
pub fn lead_brake_cascade(num_peers: usize) -> Vec<TrainingPerturbation> {
    (0..num_peers)
        .map(|i| TrainingPerturbation::LeadBrake {
            trigger_step: 50 + 30 * i,
            peer_idx: i,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicle_state_stopped() {
        let state = VehicleState::stopped();
        assert!((state.speed - 0.0).abs() < 1e-10);
        assert!((state.heading - 0.0).abs() < 1e-10);
        assert!(state.total_speed() < 1e-10);
        assert!(state.is_finite());
        assert!((state.lateral_offset - 0.0).abs() < 1e-10);
        assert!((state.nearest_peer_distance - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_vehicle_state_cruising() {
        let state = VehicleState::cruising(13.4);
        assert!((state.speed - 13.4).abs() < 1e-10);
        assert!((state.total_speed() - 13.4).abs() < 1e-6);
    }

    #[test]
    fn test_vehicle_state_to_channels() {
        let state = VehicleState::cruising(13.4);
        let channels = state.to_channels();
        assert_eq!(channels.len(), VehicleState::num_channels());
        assert_eq!(channels.len(), NUM_STATE_CHANNELS);
        assert_eq!(channels.len(), 20);
        assert!((channels[0] - 13.4).abs() < 1e-5);
        // Road channels should be zero
        assert!((channels[14] - 0.0).abs() < 1e-5);
        assert!((channels[15] - 0.0).abs() < 1e-5);
        // Mesh: nearest_peer_distance = 200
        assert!((channels[16] - 200.0).abs() < 1e-3);
    }

    #[test]
    fn test_vehicle_state_sideslip() {
        let mut state = VehicleState::cruising(10.0);
        state.lateral_velocity = 1.0;
        let beta = state.sideslip_angle();
        assert!((beta - 0.1_f64.atan()).abs() < 1e-6);
    }

    #[test]
    fn test_apply_straight_road() {
        let mut state = VehicleState::cruising(13.4);
        state.position_y = 0.5;
        state.apply_straight_road();
        assert!((state.lateral_offset - 0.5).abs() < 1e-10);
        assert!((state.curvature_ahead - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vehicle_command_zero() {
        let cmd = VehicleCommand::zero();
        assert!(cmd.steering.abs() < 1e-10);
        assert!(cmd.throttle.abs() < 1e-10);
        assert!(cmd.brake.abs() < 1e-10);
        assert!(cmd.control_effort() < 1e-10);
    }

    #[test]
    fn test_biota_right_of_way_thresholds() {
        let red = BiotaRightOfWaySignal {
            distress_signal: 0.9,
            path_conflict_risk: 0.8,
            sanctuary_signal: 0.2,
            route_clear_confidence: 0.2,
            classification_confidence: 0.8,
        };
        assert!(red.should_hold_red());

        let orange = BiotaRightOfWaySignal {
            distress_signal: 0.2,
            path_conflict_risk: 0.5,
            sanctuary_signal: 0.7,
            route_clear_confidence: 0.4,
            classification_confidence: 0.7,
        };
        assert!(orange.should_yield_orange());

        let yellow = BiotaRightOfWaySignal {
            distress_signal: 0.1,
            path_conflict_risk: 0.3,
            sanctuary_signal: 0.1,
            route_clear_confidence: 0.65,
            classification_confidence: 0.4,
        };
        assert!(yellow.should_caution_yellow());
    }

    #[test]
    fn test_vehicle_command_from_raw() {
        let raw = vec![2.0, -0.5, 1.5];
        let cmd = VehicleCommand::from_raw(&raw);
        assert!((cmd.steering - 1.0).abs() < 1e-6);
        assert!((cmd.throttle - 0.0).abs() < 1e-6);
        assert!((cmd.brake - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vehicle_command_clamped() {
        let cmd = VehicleCommand {
            steering: 5.0,
            throttle: -1.0,
            brake: 2.0,
        };
        let c = cmd.clamped();
        assert!((c.steering - 1.0).abs() < 1e-6);
        assert!((c.throttle - 0.0).abs() < 1e-6);
        assert!((c.brake - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vehicle_command_to_ctrl() {
        let cmd = VehicleCommand {
            steering: 0.5,
            throttle: 0.7,
            brake: 0.1,
        };
        let ctrl = cmd.to_ctrl();
        assert_eq!(ctrl.len(), NUM_ACTUATORS);
        assert!((ctrl[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_vehicle_task_target_speed() {
        assert!((VehicleTask::LaneKeep.target_speed() - 13.4).abs() < 1e-10);
        assert!((VehicleTask::EmergencyStop.target_speed() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vehicle_config_defaults() {
        let config = VehicleConfig::default();
        assert!((config.physics_dt() - 0.005).abs() < 1e-10);
        assert_eq!(config.cognitive_interval(), 4);
        assert!((config.effective_target_speed() - 13.4).abs() < 1e-10);
    }

    #[test]
    fn test_vehicle_config_target_speed_override() {
        let config = VehicleConfig {
            target_speed: Some(20.0),
            ..VehicleConfig::default()
        };
        assert!((config.effective_target_speed() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_chassis_params_default() {
        let params = ChassisParams::default();
        assert!((params.wheelbase - 2.7).abs() < 1e-10);
        assert!((params.mass - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_vehicle_state_serde_roundtrip() {
        let state = VehicleState::cruising(13.4);
        let json = serde_json::to_string(&state).unwrap();
        let restored: VehicleState = serde_json::from_str(&json).unwrap();
        assert!((state.speed - restored.speed).abs() < 1e-10);
        assert!((state.nearest_peer_distance - restored.nearest_peer_distance).abs() < 1e-10);
    }

    #[test]
    fn test_vehicle_command_serde_roundtrip() {
        let cmd = VehicleCommand {
            steering: 0.5,
            throttle: 0.8,
            brake: 0.0,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let restored: VehicleCommand = serde_json::from_str(&json).unwrap();
        assert!((cmd.steering - restored.steering).abs() < 1e-10);
    }

    #[test]
    fn test_channel_names_count() {
        assert_eq!(CHANNEL_NAMES.len(), NUM_STATE_CHANNELS);
    }

    #[test]
    fn test_actuator_names_count() {
        assert_eq!(ACTUATOR_NAMES.len(), NUM_ACTUATORS);
    }

    #[test]
    fn test_pd_cruise_baseline_at_target() {
        let state = VehicleState::cruising(13.4);
        let cmd = pd_cruise_baseline(&state, 13.4);
        assert!(cmd.throttle < 0.1, "At target: minimal throttle");
        assert!(cmd.brake < 0.01, "At target: no brake");
    }

    #[test]
    fn test_pd_cruise_baseline_too_slow() {
        let state = VehicleState::cruising(5.0);
        let cmd = pd_cruise_baseline(&state, 13.4);
        assert!(cmd.throttle > 0.1, "Below target: throttle");
        assert!(cmd.brake < 0.01, "Below target: no brake");
    }

    #[test]
    fn test_pd_cruise_baseline_lane_correction() {
        let mut state = VehicleState::cruising(13.4);
        state.position_y = 1.0; // 1m left of center
        state.apply_straight_road(); // copies position_y → lateral_offset
        let cmd = pd_cruise_baseline(&state, 13.4);
        assert!(
            cmd.steering < 0.0,
            "Left offset → steer right (negative): {}",
            cmd.steering
        );
    }

    #[test]
    fn test_sensor_blindness_zeros_channels() {
        let mut state = VehicleState::cruising(13.4);
        state.lateral_velocity = 1.0;
        state.yaw_rate = 0.5;
        state.heading = 0.3;

        state.apply_sensor_blindness(4);

        assert!((state.speed - 0.0).abs() < 1e-10, "speed should be zeroed");
        assert!(
            (state.lateral_velocity - 0.0).abs() < 1e-10,
            "lat_vel should be zeroed"
        );
        assert!(
            (state.yaw_rate - 0.0).abs() < 1e-10,
            "yaw_rate should be zeroed"
        );
        assert!(
            (state.heading - 0.0).abs() < 1e-10,
            "heading should be zeroed"
        );
        // position_x should NOT be zeroed (only 4 channels)
        assert!((state.position_x - 0.0).abs() < 1e-10); // was already 0
                                                         // mesh channels should be untouched
        assert!((state.nearest_peer_distance - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_sensor_blindness_clamps_to_14() {
        let mut state = VehicleState::cruising(13.4);
        state.nearest_peer_distance = 50.0;
        state.mesh_brake_density = 0.5;

        state.apply_sensor_blindness(20); // more than 14 → clamped

        assert!(
            (state.speed - 0.0).abs() < 1e-10,
            "all proprioceptive zeroed"
        );
        assert!(
            (state.pitch - 0.0).abs() < 1e-10,
            "pitch zeroed (channel 13)"
        );
        // Road and mesh channels should be untouched
        assert!(
            (state.nearest_peer_distance - 50.0).abs() < 1e-10,
            "mesh untouched"
        );
        assert!(
            (state.mesh_brake_density - 0.5).abs() < 1e-10,
            "mesh untouched"
        );
    }

    #[test]
    fn test_mesh_spoof_inflates_brake_density() {
        let mut state = VehicleState::cruising(13.4);
        state.mesh_brake_density = 0.1;
        state.mesh_confidence = 0.3;

        state.apply_mesh_spoof(2);

        // 2 spoofed peers → 0.25*2 = 0.5 added to brake_density
        assert!(
            (state.mesh_brake_density - 0.6).abs() < 1e-10,
            "brake_density: {}",
            state.mesh_brake_density
        );
        // confidence inflated by 0.5 * 0.5 = 0.25
        assert!(
            (state.mesh_confidence - 0.55).abs() < 1e-10,
            "confidence: {}",
            state.mesh_confidence
        );
    }

    #[test]
    fn test_mesh_spoof_caps_at_one() {
        let mut state = VehicleState::cruising(13.4);
        state.mesh_brake_density = 0.8;

        state.apply_mesh_spoof(4); // 4*0.25 = 1.0 added

        assert!(
            (state.mesh_brake_density - 1.0).abs() < 1e-10,
            "should cap at 1.0: {}",
            state.mesh_brake_density
        );
    }

    #[test]
    fn test_pd_follow_slows_for_close_peer() {
        let mut state = VehicleState::cruising(13.4);
        state.nearest_peer_distance = 15.0; // closer than target gap
        let cmd = pd_cruise_with_follow(&state, 13.4, 0.0, 30.0);
        // Gap error = 15 - 30 = -15 → speed_cmd = 13.4 + (-15)*0.3 = 8.9
        // At 13.4 m/s with target 8.9 → speed_error negative → should brake or reduce throttle
        let baseline = pd_cruise_baseline(&state, 13.4);
        assert!(
            cmd.throttle < baseline.throttle || cmd.brake > baseline.brake,
            "Close peer → reduced throttle or increased brake: follow={:?}, baseline={:?}",
            cmd,
            baseline
        );
    }

    #[test]
    fn test_pd_follow_speeds_for_far_peer() {
        let mut state = VehicleState::cruising(10.0);
        state.nearest_peer_distance = 60.0; // farther than target gap
        let cmd = pd_cruise_with_follow(&state, 13.4, 0.0, 30.0);
        // Gap error = 60 - 30 = 30 → speed_cmd = 13.4 + 30*0.3 = 22.4, clamped to 16.08
        // At 10.0 m/s with target 16.08 → should throttle harder than baseline at 13.4
        let baseline = pd_cruise_baseline_with_road(&state, 13.4, 0.0);
        assert!(
            cmd.throttle >= baseline.throttle,
            "Far peer → increased throttle: follow={:?}, baseline={:?}",
            cmd,
            baseline
        );
    }
}
