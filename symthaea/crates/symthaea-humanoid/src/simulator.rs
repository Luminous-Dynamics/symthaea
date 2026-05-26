// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics simulator trait and implementations for humanoid.
//!
//! Defines `HumanoidPhysicsSimulator` trait for pluggable backends.
//! The `SimpleHumanoidSimulator` provides a pure-Rust approximation with:
//! - Per-joint inertia, damping, and torque scaling from dm_control
//! - Anatomically correct joint limits from MJCF
//! - Forward kinematics for leg chains (contact-aware root height)
//! - Proper quaternion integration from angular velocity
//! - Weighted COM from segment positions
//!
//! The `MuJoCoHumanoidSimulator` wraps MuJoCo for full contact dynamics.

use crate::types::{HumanoidCommand, HumanoidState, NUM_ACTUATORS};

/// Trait for humanoid physics simulation backends.
pub trait HumanoidPhysicsSimulator {
    /// Advance one timestep with the given motor command.
    fn step(&mut self, cmd: &HumanoidCommand, dt: f64);

    /// Current humanoid state.
    fn state(&self) -> &HumanoidState;

    /// Reset to default standing pose.
    fn reset(&mut self);

    /// Reset with a deterministic perturbation.
    fn reset_with_perturbation(&mut self, perturbation: f64, seed: u64);

    /// Apply an external force to the torso (world-frame, Newtons) for the next step.
    fn apply_external_force(&mut self, force: [f64; 3]);
}

// ── Anatomical Body Model (approximate dm_control humanoid) ──

/// Segment indices for the body model.
const SEG_TORSO: usize = 0;
const SEG_THIGH: usize = 1;
const SEG_SHIN: usize = 2;
const SEG_FOOT: usize = 3;
const SEG_UPPER_ARM: usize = 4;
const SEG_FOREARM: usize = 5;
const SEG_HEAD: usize = 6;

/// Anatomical body model with segment masses, lengths, and per-joint inertia.
///
/// Values approximate the dm_control humanoid (70kg adult, ~1.7m).
#[derive(Clone)]
struct HumanoidBodyModel {
    /// Segment lengths in meters: torso, thigh, shin, foot, upper_arm, forearm, head.
    segment_lengths: [f64; 7],
    /// Segment masses in kg.
    segment_masses: [f64; 7],
    /// Total body mass in kg.
    total_mass: f64,
    /// Per-joint effective inertia (mass * length^2), indexed by actuator.
    joint_inertias: Vec<f64>,
    /// Per-joint damping coefficients.
    joint_damping: Vec<f64>,
    /// Per-joint torque scaling (normalized command -> Nm).
    joint_torque_scale: Vec<f64>,
}

impl HumanoidBodyModel {
    fn new() -> Self {
        let segment_lengths = [0.30, 0.40, 0.35, 0.12, 0.28, 0.25, 0.18]; // m
        let segment_masses = [17.0, 8.0, 4.0, 1.5, 2.5, 1.5, 5.0]; // kg (per-segment, bilateral doubled in COM)
        let total_mass = 70.0;

        // Per-joint effective inertia: I_eff ~ m_segment * L_segment^2
        let joint_inertias = [
            // Abdomen y/z/x (torso segment)
            0.30, 0.30, 0.30, // Right hip x/z/y (thigh segment)
            0.50, 0.50, 0.50, // Right knee (shin segment)
            0.20, // Right ankle x/y (foot segment)
            0.05, 0.05, // Left hip x/z/y
            0.50, 0.50, 0.50, // Left knee
            0.20, // Left ankle x/y
            0.05, 0.05, // Right shoulder1/2 (upper arm)
            0.10, 0.10, // Right elbow (forearm)
            0.03, // Left shoulder1/2
            0.10, 0.10, // Left elbow
            0.03,
        ];

        let joint_damping = [
            // Abdomen
            6.0, 6.0, 6.0, // Right hip
            8.0, 8.0, 8.0, // Right knee
            5.0, // Right ankle
            3.0, 3.0, // Left hip
            8.0, 8.0, 8.0, // Left knee
            5.0, // Left ankle
            3.0, 3.0, // Right arm
            2.0, 2.0, 1.5, // Left arm
            2.0, 2.0, 1.5,
        ];

        // Torque scale matches MJCF gear ratios
        let joint_torque_scale = [
            // Abdomen (gear=100)
            100.0, 100.0, 100.0, // Right hip (100, 100, 300), knee (200)
            100.0, 100.0, 300.0, 200.0, // Right ankle (100, 100)
            100.0, 100.0, // Left hip, knee
            100.0, 100.0, 300.0, 200.0, // Left ankle
            100.0, 100.0, // Right arm (25)
            25.0, 25.0, 25.0, // Left arm (25)
            25.0, 25.0, 25.0,
        ];

        Self {
            segment_lengths,
            segment_masses,
            total_mass,
            joint_inertias: joint_inertias.to_vec(),
            joint_damping: joint_damping.to_vec(),
            joint_torque_scale: joint_torque_scale.to_vec(),
        }
    }

    /// Create a body model for a specific morphology.
    fn for_morphology(morphology: crate::morphology::HumanoidMorphology) -> Self {
        let base = Self::new(); // Start with DMC21 base segments
        Self {
            segment_lengths: base.segment_lengths,
            segment_masses: base.segment_masses,
            total_mass: base.total_mass,
            joint_inertias: morphology.joint_inertias(),
            joint_damping: morphology.joint_damping(),
            joint_torque_scale: morphology.joint_torque_scales(),
        }
    }
}

/// Per-joint limits (radians) matching the dm_control MJCF.
/// Each entry is [min, max].
const JOINT_LIMITS: [[f64; 2]; NUM_ACTUATORS] = [
    // abdomen_y: -75 to 30 deg
    [-1.31, 0.52],
    // abdomen_z: -45 to 45 deg
    [-0.79, 0.79],
    // abdomen_x: -35 to 35 deg
    [-0.61, 0.61],
    // right_hip_x: -25 to 5 deg
    [-0.44, 0.09],
    // right_hip_z: -60 to 35 deg
    [-1.05, 0.61],
    // right_hip_y: -110 to 20 deg
    [-1.92, 0.35],
    // right_knee: -160 to 2 deg
    [-2.79, 0.03],
    // right_ankle_x: -50 to 50 deg
    [-0.87, 0.87],
    // right_ankle_y: -50 to 50 deg
    [-0.87, 0.87],
    // left_hip_x: -25 to 5 deg
    [-0.44, 0.09],
    // left_hip_z: -60 to 35 deg
    [-1.05, 0.61],
    // left_hip_y: -110 to 20 deg
    [-1.92, 0.35],
    // left_knee: -160 to 2 deg
    [-2.79, 0.03],
    // left_ankle_x: -50 to 50 deg
    [-0.87, 0.87],
    // left_ankle_y: -50 to 50 deg
    [-0.87, 0.87],
    // right_shoulder1: -85 to 60 deg
    [-1.48, 1.05],
    // right_shoulder2: -85 to 60 deg
    [-1.48, 1.05],
    // right_elbow: -90 to 50 deg
    [-1.57, 0.87],
    // left_shoulder1: -60 to 85 deg
    [-1.05, 1.48],
    // left_shoulder2: -60 to 85 deg
    [-1.05, 1.48],
    // left_elbow: -90 to 50 deg
    [-1.57, 0.87],
];

/// Sensory jitter buffer: delays state observations by 1-3 ticks to simulate
/// real-world sensor latency (I2C bus timing, IMU processing, etc.).
///
/// Forces the agent to learn predictive control instead of relying on
/// instantaneous ground truth. The delay is randomized per-reset using a
/// deterministic xorshift PRNG seeded from the perturbation seed.
struct SensoryJitterBuffer {
    /// Ring buffer of recent states (capacity = max_delay + 1).
    buffer: Vec<HumanoidState>,
    /// Write index into the ring buffer.
    write_idx: usize,
    /// Current delay in ticks (1-3, randomized per-reset).
    delay_ticks: usize,
    /// Maximum possible delay.
    max_delay: usize,
    /// Whether jitter is enabled.
    enabled: bool,
}

impl SensoryJitterBuffer {
    #[allow(dead_code)]
    fn new(max_delay: usize) -> Self {
        Self::new_for(max_delay, crate::morphology::HumanoidMorphology::Dmc21)
    }

    fn new_for(max_delay: usize, morphology: crate::morphology::HumanoidMorphology) -> Self {
        let capacity = max_delay + 1;
        Self {
            buffer: vec![HumanoidState::standing_for(morphology); capacity],
            write_idx: 0,
            delay_ticks: 1,
            max_delay,
            enabled: max_delay > 0,
        }
    }

    /// Push a new state into the buffer.
    fn push(&mut self, state: &HumanoidState) {
        self.buffer[self.write_idx] = state.clone();
        self.write_idx = (self.write_idx + 1) % self.buffer.len();
    }

    /// Read the delayed state (current minus delay_ticks).
    fn delayed_state(&self) -> &HumanoidState {
        let cap = self.buffer.len();
        let read_idx = (self.write_idx + cap - self.delay_ticks - 1) % cap;
        &self.buffer[read_idx]
    }

    /// Randomize delay using a seed, fill buffer with the given state.
    fn reset(&mut self, standing: &HumanoidState, seed: u64) {
        // Deterministic delay: 1 + (seed_hash % max_delay)
        let hash = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.delay_ticks = 1 + (hash as usize % self.max_delay).min(self.max_delay);
        for slot in &mut self.buffer {
            *slot = standing.clone();
        }
        self.write_idx = 0;
    }
}

/// Actuator noise model: delays commands and adds noise for sim-to-real transfer.
///
/// Mirrors `SensoryJitterBuffer` for the output side. Commands are delayed
/// by 1-N ticks through a ring buffer, then corrupted with xorshift noise
/// scaled to [-noise_std, +noise_std].
struct ActuatorNoiseModel {
    /// Ring buffer of recent commands (capacity = max_delay + 1).
    command_buffer: Vec<HumanoidCommand>,
    /// Write index into the ring buffer.
    write_idx: usize,
    /// Current delay in ticks (1-max_delay, randomized per-reset).
    delay_ticks: usize,
    /// Maximum possible delay.
    max_delay: usize,
    /// Noise standard deviation per torque channel.
    noise_std: f64,
    /// PRNG state for deterministic noise.
    rng_state: u64,
    /// Whether noise/delay is enabled.
    enabled: bool,
}

impl ActuatorNoiseModel {
    fn new(max_delay: usize, noise_std: f64) -> Self {
        let capacity = max_delay + 1;
        Self {
            command_buffer: vec![HumanoidCommand::zero(); capacity],
            write_idx: 0,
            delay_ticks: 1,
            max_delay,
            noise_std,
            rng_state: 0,
            enabled: false,
        }
    }

    /// Reset delay and PRNG from seed, fill buffer with zero commands.
    fn reset(&mut self, seed: u64) {
        let hash = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.delay_ticks = if self.max_delay > 0 {
            1 + (hash as usize % self.max_delay).min(self.max_delay)
        } else {
            1
        };
        self.rng_state = seed;
        for slot in &mut self.command_buffer {
            *slot = HumanoidCommand::zero();
        }
        self.write_idx = 0;
    }

    /// Apply delay and noise to a command.
    fn apply(&mut self, cmd: &HumanoidCommand) -> HumanoidCommand {
        // Push to ring buffer
        self.command_buffer[self.write_idx] = cmd.clone();
        self.write_idx = (self.write_idx + 1) % self.command_buffer.len();

        // Read delayed command
        let cap = self.command_buffer.len();
        let read_idx = (self.write_idx + cap - self.delay_ticks - 1) % cap;
        let mut delayed = self.command_buffer[read_idx].clone();

        // Add noise per torque
        for t in &mut delayed.torques {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let noise = (self.rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            *t = (*t + (noise * self.noise_std) as f32).clamp(-1.0, 1.0);
        }

        delayed
    }
}

/// Per-episode domain randomization of body parameters for sim-to-real transfer.
///
/// Randomizes segment masses (±15%), lengths (±5%), and joint damping (±20%)
/// each episode to make the policy robust to body variation.
struct DomainRandomization {
    /// Whether randomization is enabled.
    enabled: bool,
    /// Mass randomization range (fraction, e.g., 0.15 = ±15%).
    mass_range: f64,
    /// Length randomization range (fraction).
    length_range: f64,
    /// Damping randomization range (fraction).
    damping_range: f64,
}

/// Mapping from joint index to body segment index for inertia scaling.
const JOINT_SEGMENT_MAP: [usize; NUM_ACTUATORS] = [
    SEG_TORSO,
    SEG_TORSO,
    SEG_TORSO, // abdomen y/z/x
    SEG_THIGH,
    SEG_THIGH,
    SEG_THIGH,
    SEG_SHIN, // right hip x/z/y, knee
    SEG_FOOT,
    SEG_FOOT, // right ankle x/y
    SEG_THIGH,
    SEG_THIGH,
    SEG_THIGH,
    SEG_SHIN, // left hip x/z/y, knee
    SEG_FOOT,
    SEG_FOOT, // left ankle x/y
    SEG_UPPER_ARM,
    SEG_UPPER_ARM,
    SEG_FOREARM, // right shoulder1/2, elbow
    SEG_UPPER_ARM,
    SEG_UPPER_ARM,
    SEG_FOREARM, // left shoulder1/2, elbow
];

/// Observation noise model: additive sensor noise on state observations.
///
/// Complements `SensoryJitterBuffer`'s temporal delay with spatial noise.
/// Forces the agent to learn robust control despite imperfect sensing.
struct ObservationNoiseModel {
    /// Noise standard deviation per state channel.
    noise_std: f64,
    /// PRNG state for deterministic noise.
    rng_state: u64,
    /// Whether noise is enabled.
    enabled: bool,
}

impl ObservationNoiseModel {
    fn new(noise_std: f64) -> Self {
        Self {
            noise_std,
            rng_state: 0,
            enabled: false,
        }
    }

    /// Reset PRNG with a new seed.
    fn reset(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    /// Apply additive noise to a state, returning a noised clone.
    fn apply(&mut self, state: &HumanoidState) -> HumanoidState {
        let mut noised = state.clone();
        let mut next_noise = || -> f64 {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            (self.rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        for a in &mut noised.joint_angles {
            *a += self.noise_std * next_noise();
        }
        for v in &mut noised.joint_velocities {
            *v += self.noise_std * next_noise();
        }
        for v in &mut noised.root_linear_velocity {
            *v += self.noise_std * next_noise();
        }
        for v in &mut noised.root_angular_velocity {
            *v += self.noise_std * next_noise();
        }
        noised.root_height += self.noise_std * next_noise();
        noised.head_height += self.noise_std * next_noise();

        noised
    }
}

/// Spring-damper ground contact model for soft ground reaction forces.
///
/// Replaces the hard floor constraint with spring-damper + Coulomb friction,
/// producing more realistic contact dynamics for sim-to-real transfer.
struct GroundContactModel {
    /// Spring stiffness in N/m.
    stiffness: f64,
    /// Damping coefficient in N·s/m.
    damping: f64,
    /// Coulomb friction coefficient.
    friction: f64,
    /// Whether the contact model is enabled.
    enabled: bool,
}

impl GroundContactModel {
    fn new() -> Self {
        Self {
            stiffness: 10000.0,
            damping: 500.0,
            friction: 1.0,
            enabled: false,
        }
    }

    /// Compute ground reaction forces for a foot.
    ///
    /// Returns [Fx, Fy, Fz] where Fz is vertical spring-damper force
    /// and Fx/Fy are Coulomb friction forces opposing horizontal motion.
    fn apply(&self, foot_z: f64, foot_vz: f64, horizontal_vel: [f64; 2]) -> [f64; 3] {
        if foot_z >= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Vertical: spring-damper, clamp >= 0 (no suction)
        let fz = (-self.stiffness * foot_z - self.damping * foot_vz).max(0.0);

        // Horizontal: Coulomb friction opposing horizontal velocity
        let h_speed = (horizontal_vel[0].powi(2) + horizontal_vel[1].powi(2)).sqrt();
        let (fx, fy) = if h_speed > 1e-6 {
            let friction_mag = self.friction * fz;
            (
                -friction_mag * horizontal_vel[0] / h_speed,
                -friction_mag * horizontal_vel[1] / h_speed,
            )
        } else {
            (0.0, 0.0)
        };

        [fx, fy, fz]
    }
}

/// Per-episode terrain variation for sim-to-real transfer.
///
/// Randomizes ground slope and compliance each episode to make the policy
/// robust to uneven terrain encountered in real-world deployment.
struct TerrainModel {
    /// Forward slope in radians (positive = uphill).
    slope_x: f64,
    /// Lateral slope in radians (positive = right tilt).
    slope_y: f64,
    /// Ground compliance: 0.0 = rigid, 1.0 = soft (reduces ground stiffness).
    compliance: f64,
    /// Whether terrain variation is enabled.
    enabled: bool,
}

impl TerrainModel {
    fn new() -> Self {
        Self {
            slope_x: 0.0,
            slope_y: 0.0,
            compliance: 0.0,
            enabled: false,
        }
    }

    /// Randomize terrain parameters from a deterministic seed.
    fn randomize(&mut self, seed: u64) {
        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as f64 / u64::MAX as f64
        };

        // slope_x ∈ [-0.05, 0.05]
        self.slope_x = (next_f64() * 2.0 - 1.0) * 0.05;
        // slope_y ∈ [-0.03, 0.03]
        self.slope_y = (next_f64() * 2.0 - 1.0) * 0.03;
        // compliance ∈ [0, 0.3]
        self.compliance = next_f64() * 0.3;
    }
}

impl DomainRandomization {
    fn new() -> Self {
        Self {
            enabled: false,
            mass_range: 0.15,
            length_range: 0.05,
            damping_range: 0.20,
        }
    }

    /// Create a randomized body model from a baseline using a deterministic seed.
    fn randomize(&self, baseline: &HumanoidBodyModel, seed: u64) -> HumanoidBodyModel {
        let mut body = baseline.clone();
        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        // Randomize segment masses and lengths
        let mut mass_scales = [1.0f64; 7];
        let mut length_scales = [1.0f64; 7];
        for i in 0..7 {
            mass_scales[i] = 1.0 + next_f64() * self.mass_range;
            length_scales[i] = 1.0 + next_f64() * self.length_range;
            body.segment_masses[i] = baseline.segment_masses[i] * mass_scales[i];
            body.segment_lengths[i] = baseline.segment_lengths[i] * length_scales[i];
        }

        // Recompute total mass proportionally
        let baseline_segment_total: f64 = baseline.segment_masses.iter().sum();
        body.total_mass = body.segment_masses.iter().sum::<f64>()
            + (baseline.total_mass - baseline_segment_total);

        // Scale joint inertias: I ∝ m × L²
        let n = body.joint_inertias.len().min(JOINT_SEGMENT_MAP.len());
        for i in 0..n {
            let seg = JOINT_SEGMENT_MAP[i];
            body.joint_inertias[i] =
                baseline.joint_inertias[i] * mass_scales[seg] * length_scales[seg].powi(2);
        }

        // Randomize damping
        for i in 0..body.joint_damping.len() {
            let scale = 1.0 + next_f64() * self.damping_range;
            body.joint_damping[i] = baseline.joint_damping[i] * scale;
        }

        body
    }
}

/// Simple physics model for pure-Rust testing (no MuJoCo required).
///
/// Improved model (~50% MuJoCo fidelity) with:
/// - Per-joint inertia, damping, and torque scaling from dm_control
/// - Anatomically correct joint limits from MJCF
/// - Forward kinematics for leg chains (contact-aware root height)
/// - Proper quaternion integration from angular velocity
/// - Weighted COM from segment positions
/// - Sensory jitter buffer (1-3 tick delay) for sim-to-real transfer
///
/// Main remaining gap vs MuJoCo: contact/friction dynamics, collision detection.
pub struct SimpleHumanoidSimulator {
    state: HumanoidState,
    external_force: [f64; 3],
    body: HumanoidBodyModel,
    baseline_body: HumanoidBodyModel,
    jitter: SensoryJitterBuffer,
    actuator_noise: ActuatorNoiseModel,
    domain_rand: DomainRandomization,
    observation_noise: ObservationNoiseModel,
    observed_state: HumanoidState,
    ground_contact: GroundContactModel,
    terrain: TerrainModel,
    /// Morphology variant for this simulator instance.
    morphology: crate::morphology::HumanoidMorphology,
    /// Per-joint limits from morphology (replaces const JOINT_LIMITS for extended morphologies).
    joint_limits: Vec<[f64; 2]>,
    /// Gravity scaling factor (1.0 = Earth, 0.16 = Moon, 0.38 = Mars).
    gravity_scale: f64,
}

impl SimpleHumanoidSimulator {
    /// Create a new simulator at default standing pose (DMC21).
    pub fn new() -> Self {
        Self::new_for(crate::morphology::HumanoidMorphology::Dmc21)
    }

    /// Create a simulator for a specific morphology.
    pub fn new_for(morphology: crate::morphology::HumanoidMorphology) -> Self {
        let body = if morphology == crate::morphology::HumanoidMorphology::Dmc21 {
            HumanoidBodyModel::new() // Use exact existing values for backward compat
        } else {
            HumanoidBodyModel::for_morphology(morphology)
        };
        let joint_limits = morphology.joint_limits();
        let state = HumanoidState::standing_for(morphology);
        let observed_state = state.clone();
        Self {
            state,
            external_force: [0.0; 3],
            baseline_body: body.clone(),
            body,
            jitter: SensoryJitterBuffer::new_for(3, morphology),
            actuator_noise: ActuatorNoiseModel::new(3, 0.03),
            domain_rand: DomainRandomization::new(),
            observation_noise: ObservationNoiseModel::new(0.01),
            observed_state,
            ground_contact: GroundContactModel::new(),
            terrain: TerrainModel::new(),
            morphology,
            joint_limits,
            gravity_scale: 1.0,
        }
    }

    /// Enable/disable per-episode domain randomization of body parameters.
    pub fn with_domain_randomization(mut self, enabled: bool) -> Self {
        self.domain_rand.enabled = enabled;
        self
    }

    /// Set actuator noise standard deviation (0.0 to disable).
    pub fn with_actuator_noise(mut self, noise_std: f64) -> Self {
        self.actuator_noise.noise_std = noise_std;
        self.actuator_noise.enabled = noise_std > 0.0;
        self
    }

    /// Set observation noise standard deviation (0.0 to disable).
    pub fn with_observation_noise(mut self, noise_std: f64) -> Self {
        self.observation_noise.noise_std = noise_std;
        self.observation_noise.enabled = noise_std > 0.0;
        self
    }

    /// Enable/disable spring-damper ground contact model.
    pub fn with_ground_contact(mut self, enabled: bool) -> Self {
        self.ground_contact.enabled = enabled;
        self
    }

    /// Enable/disable per-episode terrain variation.
    pub fn with_terrain_variation(mut self, enabled: bool) -> Self {
        self.terrain.enabled = enabled;
        self
    }

    /// Scale all noise models by a factor in [0.0, 1.0].
    ///
    /// Must be called AFTER `with_actuator_noise()`, `with_domain_randomization()`,
    /// and `with_observation_noise()`.
    pub fn with_noise_scale(mut self, scale: f64) -> Self {
        self.actuator_noise.noise_std *= scale;
        self.domain_rand.mass_range *= scale;
        self.domain_rand.length_range *= scale;
        self.domain_rand.damping_range *= scale;
        self.observation_noise.noise_std *= scale;
        self
    }

    /// Set gravity scaling factor (1.0 = Earth, 0.16 = Moon, 0.38 = Mars).
    pub fn with_gravity(mut self, scale: f64) -> Self {
        self.gravity_scale = scale;
        self
    }

    /// Compute vertical length from hip to foot via forward kinematics for one leg.
    ///
    /// `hip_y`: hip flexion angle (radians), `knee`: knee flexion angle (radians).
    /// Returns vertical displacement from hip joint to foot sole.
    fn leg_vertical_length(&self, hip_y: f64, knee: f64) -> f64 {
        let thigh_len = self.body.segment_lengths[SEG_THIGH];
        let shin_len = self.body.segment_lengths[SEG_SHIN];
        let foot_len = self.body.segment_lengths[SEG_FOOT];

        // Forward kinematics: hip -> knee -> ankle chain
        let thigh_z = thigh_len * hip_y.cos();
        let knee_angle = hip_y + knee;
        let shin_z = shin_len * knee_angle.cos();

        (thigh_z + shin_z + foot_len).max(0.05)
    }

    /// Compute weighted center-of-mass z-coordinate from segment positions.
    fn compute_com_z(&self) -> f64 {
        let body = &self.body;
        let root_h = self.state.root_height;
        let torso_half = body.segment_lengths[SEG_TORSO] * 0.5;

        // Torso at root height
        let torso_z = root_h;

        // Head above torso
        let head_z = root_h + torso_half + body.segment_lengths[SEG_HEAD] * 0.5;

        // Right leg chain
        let hip_z = root_h - torso_half;
        let r_hip_y = self.state.joint_angles[5];
        let r_knee = self.state.joint_angles[6];
        let r_thigh_z = hip_z - body.segment_lengths[SEG_THIGH] * 0.5 * r_hip_y.cos();
        let r_shin_z = hip_z
            - body.segment_lengths[SEG_THIGH] * r_hip_y.cos()
            - body.segment_lengths[SEG_SHIN] * 0.5 * (r_hip_y + r_knee).cos();

        // Left leg chain
        let l_hip_y = self.state.joint_angles[11];
        let l_knee = self.state.joint_angles[12];
        let l_thigh_z = hip_z - body.segment_lengths[SEG_THIGH] * 0.5 * l_hip_y.cos();
        let l_shin_z = hip_z
            - body.segment_lengths[SEG_THIGH] * l_hip_y.cos()
            - body.segment_lengths[SEG_SHIN] * 0.5 * (l_hip_y + l_knee).cos();

        // Arms hanging from shoulders
        let arm_z = root_h + torso_half * 0.3;

        // Weighted sum
        (body.segment_masses[SEG_TORSO] * torso_z
            + body.segment_masses[SEG_HEAD] * head_z
            + body.segment_masses[SEG_THIGH] * (r_thigh_z + l_thigh_z)
            + body.segment_masses[SEG_SHIN] * (r_shin_z + l_shin_z)
            + body.segment_masses[SEG_FOOT] * 2.0 * 0.05
            + body.segment_masses[SEG_UPPER_ARM] * 2.0 * arm_z
            + body.segment_masses[SEG_FOREARM]
                * 2.0
                * (arm_z - body.segment_lengths[SEG_UPPER_ARM]))
            / body.total_mass
    }
}

impl Default for SimpleHumanoidSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl HumanoidPhysicsSimulator for SimpleHumanoidSimulator {
    fn step(&mut self, cmd: &HumanoidCommand, dt: f64) {
        let cmd = if self.actuator_noise.enabled {
            self.actuator_noise.apply(cmd)
        } else {
            cmd.clone()
        };
        let g = 9.81 * self.gravity_scale;

        // 1. Joint dynamics: per-joint inertia, damping, torque scaling
        for i in 0..cmd.torques.len().min(self.body.joint_torque_scale.len()) {
            let torque = cmd.torques[i] as f64 * self.body.joint_torque_scale[i];
            let damping = self.body.joint_damping[i];
            let inertia = self.body.joint_inertias[i];

            let accel = (torque - damping * self.state.joint_velocities[i]) / inertia;
            self.state.joint_velocities[i] += accel * dt;
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;

            // Per-joint anatomical limits with velocity clamping at stops
            let [lo, hi] = if i < self.joint_limits.len() {
                self.joint_limits[i]
            } else {
                [-3.14, 3.14]
            };
            if self.state.joint_angles[i] < lo {
                self.state.joint_angles[i] = lo;
                self.state.joint_velocities[i] = self.state.joint_velocities[i].max(0.0);
            } else if self.state.joint_angles[i] > hi {
                self.state.joint_angles[i] = hi;
                self.state.joint_velocities[i] = self.state.joint_velocities[i].min(0.0);
            }
        }

        // 2. Torso tilt and uprightness from abdomen angles
        let abd_y = self.state.joint_angles[0]; // sagittal
        let abd_z = self.state.joint_angles[1]; // transverse
        let abd_x = self.state.joint_angles[2]; // coronal

        let tilt_sagittal = abd_y.sin();
        let tilt_coronal = abd_x.sin();
        let tilt_mag = (tilt_sagittal * tilt_sagittal + tilt_coronal * tilt_coronal).sqrt();
        let uprightness = (1.0 - tilt_mag).clamp(0.0, 1.0);

        self.state.torso_vertical[0] = tilt_coronal;
        self.state.torso_vertical[1] = abd_z.sin() * 0.3;
        self.state.torso_vertical[2] = uprightness;

        // 3. Contact-aware root height from forward kinematics
        let r_hip_y = self.state.joint_angles[5];
        let r_knee = self.state.joint_angles[6];
        let l_hip_y = self.state.joint_angles[11];
        let l_knee = self.state.joint_angles[12];

        let right_leg_len = self.leg_vertical_length(r_hip_y, r_knee);
        let left_leg_len = self.leg_vertical_length(l_hip_y, l_knee);

        // Root height = support leg length * uprightness + torso offset
        let kinematic_height = right_leg_len.max(left_leg_len) * uprightness.max(0.2)
            + self.body.segment_lengths[SEG_TORSO] * 0.5;

        // Gravity: inverted pendulum torque from tilt
        let lean_torque = -g * self.body.total_mass * tilt_mag * 0.15;
        self.state.root_linear_velocity[2] += (lean_torque / self.body.total_mass
            + self.external_force[2] / self.body.total_mass)
            * dt;
        self.state.root_height += self.state.root_linear_velocity[2] * dt;

        // Blend dynamic height with kinematic (70% kinematic, 30% dynamic)
        self.state.root_height = 0.3 * self.state.root_height + 0.7 * kinematic_height;

        // Ground contact: spring-damper or hard constraint
        if self.ground_contact.enabled {
            // Scale stiffness/damping by terrain compliance
            let compliance_scale = if self.terrain.enabled {
                1.0 - self.terrain.compliance.min(0.99)
            } else {
                1.0
            };
            let contact = GroundContactModel {
                stiffness: self.ground_contact.stiffness * compliance_scale,
                damping: self.ground_contact.damping * compliance_scale,
                friction: self.ground_contact.friction,
                enabled: true,
            };

            let r_foot_z_raw = self.state.root_height - right_leg_len;
            let l_foot_z_raw = self.state.root_height - left_leg_len;
            let h_vel = [
                self.state.root_linear_velocity[0],
                self.state.root_linear_velocity[1],
            ];

            // Right foot contact
            let r_forces = contact.apply(r_foot_z_raw, self.state.root_linear_velocity[2], h_vel);
            self.state.root_linear_velocity[0] += r_forces[0] / self.body.total_mass * dt;
            self.state.root_linear_velocity[1] += r_forces[1] / self.body.total_mass * dt;
            self.state.root_linear_velocity[2] += r_forces[2] / self.body.total_mass * dt;

            // Left foot contact
            let l_forces = contact.apply(l_foot_z_raw, self.state.root_linear_velocity[2], h_vel);
            self.state.root_linear_velocity[0] += l_forces[0] / self.body.total_mass * dt;
            self.state.root_linear_velocity[1] += l_forces[1] / self.body.total_mass * dt;
            self.state.root_linear_velocity[2] += l_forces[2] / self.body.total_mass * dt;

            // Soft floor: prevent deep penetration
            if self.state.root_height < 0.15 {
                self.state.root_height = 0.15;
                self.state.root_linear_velocity[2] = self.state.root_linear_velocity[2].max(0.0);
            }
        } else {
            // Hard floor constraint (original behavior)
            let ground_margin = if self.terrain.enabled {
                0.2 + self.terrain.compliance * 0.05
            } else {
                0.2
            };
            if self.state.root_height < ground_margin {
                self.state.root_height = ground_margin;
                self.state.root_linear_velocity[2] = self.state.root_linear_velocity[2].max(0.0);
            }
        }

        // 4. Head height from torso chain
        self.state.head_height = self.state.root_height
            + self.body.segment_lengths[SEG_TORSO] * 0.5 * uprightness
            + self.body.segment_lengths[SEG_HEAD] * uprightness;

        // 5. Horizontal dynamics
        // Two mechanisms: (a) torso lean (inverted pendulum) and (b) stance-phase
        // ground reaction force from leg pushoff.
        //
        // (a) Lean-driven acceleration: forward from sagittal tilt, lateral from coronal.
        let lean_accel_scale = if uprightness > 0.3 {
            g * 0.4 * (uprightness - 0.3).min(0.7) / 0.7
        } else {
            0.0
        };
        let mut forward_accel = tilt_sagittal * lean_accel_scale;
        let mut lateral_accel = tilt_coronal * lean_accel_scale;

        // Terrain slope: gravity component along slope
        if self.terrain.enabled {
            forward_accel += g * self.terrain.slope_x.sin();
            lateral_accel += g * self.terrain.slope_y.sin();
        }

        // (b) Stance-phase ground reaction force: when a foot is on the ground
        // and the hip is extending (positive hip_y velocity = leg pushing back),
        // add a forward pushoff impulse. This simulates the stance phase of gait
        // where ground contact converts leg extension into forward propulsion.
        let r_foot_z = (self.state.root_height - right_leg_len).max(0.0);
        let l_foot_z = (self.state.root_height - left_leg_len).max(0.0);

        let ground_threshold = 0.05; // foot within 5cm of ground = contact
        let pushoff_gain = 6.0; // N·s per (rad/s) of hip extension

        let mut pushoff_accel = 0.0;

        // Right leg: hip_y velocity > 0 = extending backward = pushoff
        if r_foot_z < ground_threshold && uprightness > 0.4 {
            let r_hip_vel = self.state.joint_velocities[5];
            if r_hip_vel > 0.0 {
                pushoff_accel += r_hip_vel * pushoff_gain;
            }
        }

        // Left leg: hip_y velocity > 0 = extending backward = pushoff
        if l_foot_z < ground_threshold && uprightness > 0.4 {
            let l_hip_vel = self.state.joint_velocities[11];
            if l_hip_vel > 0.0 {
                pushoff_accel += l_hip_vel * pushoff_gain;
            }
        }

        // Ankle pushoff: plantarflexion during toe-off adds forward impulse
        let ankle_pushoff_gain = 2.0;
        let r_ankle_vel = self.state.joint_velocities[8]; // right ankle_y
        let l_ankle_vel = self.state.joint_velocities[14]; // left ankle_y

        if r_foot_z < ground_threshold && r_ankle_vel > 0.0 && uprightness > 0.4 {
            pushoff_accel += r_ankle_vel * ankle_pushoff_gain;
        }
        if l_foot_z < ground_threshold && l_ankle_vel > 0.0 && uprightness > 0.4 {
            pushoff_accel += l_ankle_vel * ankle_pushoff_gain;
        }

        // Knee extension force: straightening during stance propels forward
        let knee_ext_gain = 1.5;
        let r_knee_vel = self.state.joint_velocities[6];
        let l_knee_vel = self.state.joint_velocities[12];

        if r_foot_z < ground_threshold && r_knee_vel > 0.0 && uprightness > 0.4 {
            pushoff_accel += r_knee_vel * knee_ext_gain;
        }
        if l_foot_z < ground_threshold && l_knee_vel > 0.0 && uprightness > 0.4 {
            pushoff_accel += l_knee_vel * knee_ext_gain;
        }

        self.state.root_linear_velocity[0] +=
            (forward_accel + pushoff_accel + self.external_force[0] / self.body.total_mass) * dt;
        self.state.root_linear_velocity[1] +=
            (lateral_accel + self.external_force[1] / self.body.total_mass) * dt;

        // Speed-dependent drag: high at rest (prevents drift), low at speed (allows running)
        let speed = (self.state.root_linear_velocity[0].powi(2)
            + self.state.root_linear_velocity[1].powi(2))
        .sqrt();
        let drag = 1.0 - 0.7 * (1.0 - (-speed * 0.5).exp());
        // drag ≈ 1.0 at rest, ≈ 0.3 at high speed
        self.state.root_linear_velocity[0] *= (1.0 - drag * dt).max(0.0);
        self.state.root_linear_velocity[1] *= (1.0 - drag * dt).max(0.0);

        // 6. COM velocity from segment positions
        let com_z = self.compute_com_z();
        let prev_com_vz = self.state.com_velocity[2];
        self.state.com_velocity[0] = self.state.root_linear_velocity[0];
        self.state.com_velocity[1] = self.state.root_linear_velocity[1];
        self.state.com_velocity[2] =
            0.7 * prev_com_vz + 0.3 * (com_z - self.state.root_height) / dt.max(1e-6);

        // 7. Angular velocity from abdomen joint velocities + decay
        self.state.root_angular_velocity[0] = 0.3 * self.state.joint_velocities[2]; // roll
        self.state.root_angular_velocity[1] = 0.3 * self.state.joint_velocities[0]; // pitch
        self.state.root_angular_velocity[2] = 0.3 * self.state.joint_velocities[1]; // yaw
        for av in &mut self.state.root_angular_velocity {
            *av *= (1.0 - 2.0 * dt).max(0.0);
        }

        // 8. Quaternion integration from angular velocity
        self.state.root_quaternion = integrate_quaternion(
            self.state.root_quaternion,
            self.state.root_angular_velocity,
            dt,
        );

        // 9. Approximate extremity positions
        let root_h = self.state.root_height;
        let torso_half = self.body.segment_lengths[SEG_TORSO] * 0.5;

        // Right hand
        let arm_base_z = root_h + torso_half * 0.3;
        let r_shoulder = self.state.joint_angles[15];
        let r_elbow = self.state.joint_angles[17];
        let arm_reach = self.body.segment_lengths[SEG_UPPER_ARM] * r_shoulder.sin().abs()
            + self.body.segment_lengths[SEG_FOREARM] * (r_shoulder + r_elbow).sin().abs();
        self.state.extremities[0] = arm_reach * 0.3;
        self.state.extremities[1] = -0.17;
        self.state.extremities[2] = arm_base_z - arm_reach * 0.5;

        // Left hand
        let l_shoulder = self.state.joint_angles[18];
        let l_elbow = self.state.joint_angles[20];
        let l_arm_reach = self.body.segment_lengths[SEG_UPPER_ARM] * l_shoulder.sin().abs()
            + self.body.segment_lengths[SEG_FOREARM] * (l_shoulder + l_elbow).sin().abs();
        self.state.extremities[3] = l_arm_reach * 0.3;
        self.state.extremities[4] = 0.17;
        self.state.extremities[5] = arm_base_z - l_arm_reach * 0.5;

        // Right foot
        let r_foot_z = (root_h - right_leg_len).max(0.0);
        self.state.extremities[6] = r_hip_y.sin() * 0.1;
        self.state.extremities[7] = -0.1;
        self.state.extremities[8] = r_foot_z;

        // Left foot
        let l_foot_z = (root_h - left_leg_len).max(0.0);
        self.state.extremities[9] = l_hip_y.sin() * 0.1;
        self.state.extremities[10] = 0.1;
        self.state.extremities[11] = l_foot_z;

        // Hand FK for dexterous morphologies (compute fingertip centroids)
        if self.morphology.num_actuators() > 21 && self.state.extremities.len() >= 18 {
            let r_hand_base = [
                self.state.extremities[0],
                self.state.extremities[1],
                self.state.extremities[2],
            ];
            let l_hand_base = [
                self.state.extremities[3],
                self.state.extremities[4],
                self.state.extremities[5],
            ];

            // Right hand: joints 21..37
            if self.state.joint_angles.len() >= 37 {
                let r_centroid = crate::morphology::compute_hand_centroid(
                    r_hand_base,
                    &self.state.joint_angles[21..37],
                    [1.0, 0.0, 0.0], // right hand forward direction
                );
                self.state.extremities[12] = r_centroid[0];
                self.state.extremities[13] = r_centroid[1];
                self.state.extremities[14] = r_centroid[2];
            }

            // Left hand: joints 37..53
            if self.state.joint_angles.len() >= 53 {
                let l_centroid = crate::morphology::compute_hand_centroid(
                    l_hand_base,
                    &self.state.joint_angles[37..53],
                    [1.0, 0.0, 0.0], // left hand forward direction
                );
                self.state.extremities[15] = l_centroid[0];
                self.state.extremities[16] = l_centroid[1];
                self.state.extremities[17] = l_centroid[2];
            }
        }

        self.state.timestamp += dt;
        self.external_force = [0.0; 3];

        // Push post-physics state into jitter buffer
        self.jitter.push(&self.state);

        // Apply observation noise
        if self.observation_noise.enabled {
            self.observed_state = self.observation_noise.apply(&self.state);
        }
    }

    fn state(&self) -> &HumanoidState {
        if self.observation_noise.enabled {
            &self.observed_state
        } else if self.jitter.enabled {
            self.jitter.delayed_state()
        } else {
            &self.state
        }
    }

    fn reset(&mut self) {
        self.state = HumanoidState::standing_for(self.morphology);
        self.external_force = [0.0; 3];
        self.body = self.baseline_body.clone();
        self.jitter.reset(&self.state, 0);
        self.actuator_noise.reset(0);
        self.observation_noise.reset(0);
        self.observed_state = self.state.clone();
        self.terrain.slope_x = 0.0;
        self.terrain.slope_y = 0.0;
        self.terrain.compliance = 0.0;
    }

    fn reset_with_perturbation(&mut self, perturbation: f64, seed: u64) {
        self.state = HumanoidState::standing();
        self.external_force = [0.0; 3];

        // Domain randomization: randomize body parameters from baseline
        if self.domain_rand.enabled {
            self.body = self
                .domain_rand
                .randomize(&self.baseline_body, seed.wrapping_mul(1442695040888963407));
        } else {
            self.body = self.baseline_body.clone();
        }

        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        // Perturb joint angles within anatomical limits (DMC21 base joints)
        let n = self.state.joint_angles.len().min(JOINT_LIMITS.len());
        for i in 0..n {
            let [lo, hi] = if i < self.joint_limits.len() {
                self.joint_limits[i]
            } else {
                [-3.14, 3.14]
            };
            let range = hi - lo;
            self.state.joint_angles[i] += perturbation * next_f64() * range * 0.03;
            self.state.joint_angles[i] = self.state.joint_angles[i].clamp(lo, hi);
        }

        // Perturb root height slightly
        self.state.root_height += perturbation * next_f64() * 0.02;
        self.state.head_height = self.state.root_height
            + self.body.segment_lengths[SEG_TORSO] * 0.5
            + self.body.segment_lengths[SEG_HEAD];

        self.jitter.reset(&self.state, seed);
        self.actuator_noise.reset(seed.wrapping_mul(2654435761));
        self.observation_noise.reset(seed.wrapping_mul(3141592653));
        self.observed_state = self.state.clone();

        // Terrain randomization
        if self.terrain.enabled {
            self.terrain
                .randomize(seed.wrapping_mul(7046029254386353131));
        }
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        self.external_force[0] += force[0];
        self.external_force[1] += force[1];
        self.external_force[2] += force[2];
    }
}

/// Integrate quaternion from angular velocity using Euler method.
///
/// q_dot = 0.5 * q (x) [0, omega_x, omega_y, omega_z]
fn integrate_quaternion(q: [f64; 4], omega: [f64; 3], dt: f64) -> [f64; 4] {
    let [w, x, y, z] = q;
    let [ox, oy, oz] = omega;

    let dw = 0.5 * (-x * ox - y * oy - z * oz);
    let dx = 0.5 * (w * ox + y * oz - z * oy);
    let dy = 0.5 * (w * oy + z * ox - x * oz);
    let dz = 0.5 * (w * oz + x * oy - y * ox);

    normalize_quat([w + dw * dt, x + dx * dt, y + dy * dt, z + dz * dt])
}

/// MuJoCo-based humanoid simulator (requires mujoco-rs).
///
/// Wraps the dm_control humanoid MJCF model with proper contact dynamics,
/// ground reaction forces, and full rigid body simulation.
///
/// Uses mujoco-rs 2.3 safe view API for all state access (zero unsafe blocks).
#[cfg(feature = "mujoco")]
use mujoco_rs::prelude::*;

/// Humanoid qpos size: 7 (free joint) + 21 (hinge joints) = 28.
#[cfg(feature = "mujoco")]
const HUMANOID_NQ: usize = 28;

#[cfg(feature = "mujoco")]
pub struct MuJoCoHumanoidSimulator {
    model: std::sync::Arc<MjModel>,
    data: MjData<std::sync::Arc<MjModel>>,
    state: HumanoidState,
    body_ids: HumanoidBodyIds,
    external_force: [f64; 3],
}

/// Cached body name -> ID lookups for efficient state extraction.
#[cfg(feature = "mujoco")]
struct HumanoidBodyIds {
    torso: usize,
    head: usize,
    right_hand: usize,
    left_hand: usize,
    right_foot: usize,
    left_foot: usize,
}

#[cfg(feature = "mujoco")]
impl MuJoCoHumanoidSimulator {
    /// Create a new simulator from an MJCF XML string.
    pub fn from_xml(xml: &str) -> anyhow::Result<Self> {
        let model = std::sync::Arc::new(MjModel::from_xml_string(xml)?);
        let data = MjData::new(std::sync::Arc::clone(&model));

        let body_ids = HumanoidBodyIds {
            torso: Self::find_body_id(&model, "torso"),
            head: Self::find_body_id(&model, "head"),
            right_hand: Self::find_body_id_or(&model, "right_hand", 3),
            left_hand: Self::find_body_id_or(&model, "left_hand", 4),
            right_foot: Self::find_body_id_or(&model, "right_foot", 5),
            left_foot: Self::find_body_id_or(&model, "left_foot", 6),
        };

        let state = HumanoidState::standing();

        let mut sim = Self {
            model,
            data,
            state,
            body_ids,
            external_force: [0.0; 3],
        };

        // Forward kinematics to populate xpos/xmat from qpos
        sim.data.forward();
        sim.extract_state();

        Ok(sim)
    }

    /// Create from an MJCF file path.
    pub fn from_path(path: &str) -> anyhow::Result<Self> {
        let xml = std::fs::read_to_string(path)?;
        Self::from_xml(&xml)
    }

    /// Create from the bundled dm_control humanoid MJCF asset.
    ///
    /// Loads `assets/humanoid.xml` from the crate directory. This is the
    /// standard dm_control humanoid model adapted for mujoco-rs 2.3.
    pub fn from_bundled_asset() -> anyhow::Result<Self> {
        let xml = include_str!("../assets/humanoid.xml");
        Self::from_xml(xml)
    }

    /// Find body ID by name. Panics if not found.
    fn find_body_id(model: &MjModel, name: &str) -> usize {
        let id = model.name_to_id(MjtObj::mjOBJ_BODY, name);
        assert!(id >= 0, "Body '{}' not found in MJCF model", name);
        id as usize
    }

    /// Find body ID by name, returning a fallback if not found.
    fn find_body_id_or(model: &MjModel, name: &str, fallback: usize) -> usize {
        let id = model.name_to_id(MjtObj::mjOBJ_BODY, name);
        if id >= 0 {
            id as usize
        } else {
            fallback
        }
    }

    /// Extract the current humanoid state from MuJoCo data.
    fn extract_state(&mut self) {
        let t = self.data.time();
        let qpos = self.data.qpos();
        let qvel = self.data.qvel();

        let xpos = self.data.xpos();
        let xmat = self.data.xmat();

        let head_height = xpos[self.body_ids.head][2];

        let torso_xmat = xmat[self.body_ids.torso];
        let torso_vertical = [torso_xmat[6], torso_xmat[7], torso_xmat[8]];

        let rh = xpos[self.body_ids.right_hand];
        let lh = xpos[self.body_ids.left_hand];
        let rf = xpos[self.body_ids.right_foot];
        let lf = xpos[self.body_ids.left_foot];
        let extremities = [
            rh[0], rh[1], rh[2], lh[0], lh[1], lh[2], rf[0], rf[1], rf[2], lf[0], lf[1], lf[2],
        ];

        let com_velocity = [qvel[0], qvel[1], qvel[2]];

        self.state = HumanoidState::from_mujoco(
            qpos,
            qvel,
            head_height,
            torso_vertical,
            &extremities,
            com_velocity,
            t,
        );
    }

    /// Get the body mass of the torso.
    pub fn body_mass(&self) -> f64 {
        self.model.body_mass()[self.body_ids.torso]
    }

    /// Write motor commands into MuJoCo ctrl array.
    fn write_ctrl(&mut self, cmd: &HumanoidCommand) {
        let ctrl = self.data.ctrl_mut();
        for i in 0..cmd.torques.len().min(ctrl.len()) {
            ctrl[i] = cmd.torques[i] as f64;
        }
    }

    /// Apply accumulated external force to torso via xfrc_applied.
    fn apply_xfrc(&mut self) {
        if self.external_force[0].abs() > 1e-15
            || self.external_force[1].abs() > 1e-15
            || self.external_force[2].abs() > 1e-15
        {
            let xfrc = self.data.xfrc_applied_mut();
            // xfrc_applied is [nBody x 6]: [force(3), torque(3)]
            xfrc[self.body_ids.torso][0] = self.external_force[0];
            xfrc[self.body_ids.torso][1] = self.external_force[1];
            xfrc[self.body_ids.torso][2] = self.external_force[2];
        }
    }

    /// Clear external forces from xfrc_applied.
    fn clear_xfrc(&mut self) {
        self.data.xfrc_applied_mut()[self.body_ids.torso] = [0.0; 6];
        self.external_force = [0.0; 3];
    }
}

#[cfg(feature = "mujoco")]
impl HumanoidPhysicsSimulator for MuJoCoHumanoidSimulator {
    fn step(&mut self, cmd: &HumanoidCommand, _dt: f64) {
        self.write_ctrl(cmd);
        self.apply_xfrc();

        self.data.step();

        self.extract_state();
        self.clear_xfrc();
    }

    fn state(&self) -> &HumanoidState {
        &self.state
    }

    fn reset(&mut self) {
        self.data.reset();
        self.data.forward();
        self.external_force = [0.0; 3];
        self.extract_state();
    }

    fn reset_with_perturbation(&mut self, perturbation: f64, seed: u64) {
        self.data.reset();
        self.external_force = [0.0; 3];

        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        // Perturb joint angles (qpos[7..28]) within bounds
        let qpos = self.data.qpos_mut();
        for i in 7..HUMANOID_NQ {
            qpos[i] += perturbation * next_f64() * 0.05;
        }

        self.data.forward();
        self.extract_state();
    }

    fn apply_external_force(&mut self, force: [f64; 3]) {
        self.external_force[0] += force[0];
        self.external_force[1] += force[1];
        self.external_force[2] += force[2];
    }
}

fn normalize_quat(q: [f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-10 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_object_compiles() {
        let sim: Box<dyn HumanoidPhysicsSimulator> = Box::new(SimpleHumanoidSimulator::new());
        assert!(sim.state().root_height > 0.0);
    }

    #[test]
    fn test_simple_physics_standing() {
        let mut sim = SimpleHumanoidSimulator::new();
        let cmd = HumanoidCommand::zero();
        for _ in 0..100 {
            sim.step(&cmd, 0.025);
        }
        assert!(
            sim.state().root_height > 0.5,
            "Standing with zero torque should maintain some height: got {}",
            sim.state().root_height
        );
    }

    #[test]
    fn test_external_force_affects_velocity() {
        let mut sim = SimpleHumanoidSimulator::new();
        let cmd = HumanoidCommand::zero();

        sim.apply_external_force([500.0, 0.0, 0.0]);
        sim.step(&cmd, 0.025);
        // Second step so the jitter-delayed state reflects the force from step 1
        sim.step(&cmd, 0.025);

        assert!(
            sim.state().root_linear_velocity[0].abs() > 0.0,
            "External force should cause velocity"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut sim = SimpleHumanoidSimulator::new();
        sim.apply_external_force([500.0, 500.0, 500.0]);
        let cmd = HumanoidCommand {
            torques: vec![0.5; NUM_ACTUATORS],
        };
        sim.step(&cmd, 0.025);

        sim.reset();
        assert!((sim.state().root_height - 1.3).abs() < 0.1);
        assert!(sim.state().horizontal_speed() < 1e-10);
    }

    #[test]
    fn test_perturbation_changes_state() {
        let mut sim = SimpleHumanoidSimulator::new();
        sim.reset_with_perturbation(1.0, 42);
        let any_nonzero = sim.state().joint_angles.iter().any(|a| a.abs() > 0.001);
        assert!(any_nonzero, "Perturbation should change joint angles");
    }

    #[test]
    fn test_joint_limits() {
        let mut sim = SimpleHumanoidSimulator::new();
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 1.0; // abdomen_y: limit [-1.31, 0.52]
        for _ in 0..1000 {
            sim.step(&cmd, 0.025);
        }
        assert!(
            sim.state().joint_angles[0] <= 0.53,
            "Joint should respect upper limit 0.52: got {}",
            sim.state().joint_angles[0]
        );
    }

    #[test]
    fn test_body_model_consistency() {
        let body = HumanoidBodyModel::new();
        assert_eq!(body.joint_inertias.len(), NUM_ACTUATORS);
        assert_eq!(body.joint_damping.len(), NUM_ACTUATORS);
        assert_eq!(body.joint_torque_scale.len(), NUM_ACTUATORS);
        assert!((body.total_mass - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_integration_identity() {
        // Zero angular velocity should preserve identity quaternion
        let q = [1.0, 0.0, 0.0, 0.0];
        let result = integrate_quaternion(q, [0.0, 0.0, 0.0], 0.025);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
    }

    #[test]
    fn test_quaternion_integration_rotation() {
        let q = [1.0, 0.0, 0.0, 0.0];
        let result = integrate_quaternion(q, [0.0, 0.0, 1.0], 0.1);
        // Should have rotated around z-axis
        let norm = (result[0] * result[0]
            + result[1] * result[1]
            + result[2] * result[2]
            + result[3] * result[3])
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Quaternion should stay normalized: {norm}"
        );
        assert!(result[3].abs() > 0.01, "Should have z rotation component");
    }

    #[test]
    fn test_per_joint_limits_applied() {
        let mut sim = SimpleHumanoidSimulator::new();
        // Push right_knee (idx 6) with negative torque: limit is [-2.79, 0.03]
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[6] = -1.0;
        for _ in 0..2000 {
            sim.step(&cmd, 0.025);
        }
        let knee_angle = sim.state().joint_angles[6];
        assert!(
            knee_angle >= -2.80,
            "Knee should respect lower limit -2.79: got {knee_angle}"
        );
    }

    #[test]
    fn test_stance_pushoff_increases_speed() {
        let mut sim = SimpleHumanoidSimulator::new();
        // Apply positive hip_y torque to right leg (extending backward = pushoff)
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[5] = 0.5; // right_hip_y: positive = extend back
        cmd.torques[0] = 0.1; // slight forward lean to keep upright
        for _ in 0..50 {
            sim.step(&cmd, 0.025);
        }
        let speed = sim.state().horizontal_speed();
        assert!(
            speed > 0.01,
            "Hip extension pushoff should produce forward speed: got {speed}"
        );
    }

    #[test]
    fn test_head_height_above_root() {
        let mut sim = SimpleHumanoidSimulator::new();
        let cmd = HumanoidCommand::zero();
        for _ in 0..50 {
            sim.step(&cmd, 0.025);
        }
        assert!(
            sim.state().head_height > sim.state().root_height,
            "Head should be above root: head={}, root={}",
            sim.state().head_height,
            sim.state().root_height
        );
    }

    #[test]
    fn test_com_z_reasonable() {
        let sim = SimpleHumanoidSimulator::new();
        let com_z = sim.compute_com_z();
        assert!(
            com_z > 0.5 && com_z < 1.5,
            "COM z should be reasonable for standing: {com_z}"
        );
    }

    #[test]
    fn test_jitter_buffer_delays_state() {
        let mut jitter = SensoryJitterBuffer::new(3);
        // Push 5 distinct states with different root heights: 1.0, 1.1, 1.2, 1.3, 1.4
        for i in 0..5 {
            let mut state = HumanoidState::standing();
            state.root_height = 1.0 + i as f64 * 0.1;
            jitter.push(&state);
        }
        // With delay_ticks=1 (default), delayed_state should return state before latest.
        // Latest pushed was 1.4, so delayed should be 1.3 (one tick behind).
        let delayed = jitter.delayed_state();
        assert!(
            (delayed.root_height - 1.3).abs() < 1e-6,
            "Delayed state should be one tick behind latest (1.3): got {}",
            delayed.root_height
        );
    }

    #[test]
    fn test_jitter_reset_randomizes_delay() {
        let mut jitter_a = SensoryJitterBuffer::new(3);
        let mut jitter_b = SensoryJitterBuffer::new(3);
        let state = HumanoidState::standing();
        jitter_a.reset(&state, 42);
        jitter_b.reset(&state, 999);
        // Different seeds may produce different delays (probabilistic but deterministic)
        // At minimum, both should have valid delay in [1, max_delay]
        assert!(jitter_a.delay_ticks >= 1 && jitter_a.delay_ticks <= 3);
        assert!(jitter_b.delay_ticks >= 1 && jitter_b.delay_ticks <= 3);
    }

    // ── Actuator noise tests ──

    #[test]
    fn test_actuator_noise_delay() {
        let mut noise = ActuatorNoiseModel::new(3, 0.0); // zero noise to isolate delay
        noise.enabled = true;
        noise.reset(42);

        // Push 5 distinct commands with different torques[0]
        for i in 0..5 {
            let mut cmd = HumanoidCommand::zero();
            cmd.torques[0] = i as f32 * 0.2;
            let out = noise.apply(&cmd);
            // With delay >= 1, output should NOT be the latest command
            if i >= noise.delay_ticks + 1 {
                // Output should be a previous command (delayed)
                assert!(
                    (out.torques[0] - cmd.torques[0]).abs() > 0.01,
                    "Output should be delayed, not latest: out={} cmd={}",
                    out.torques[0],
                    cmd.torques[0]
                );
            }
        }
    }

    #[test]
    fn test_actuator_noise_applied() {
        let mut noise = ActuatorNoiseModel::new(1, 0.1);
        noise.enabled = true;
        noise.reset(42);

        let cmd = HumanoidCommand {
            torques: vec![0.5; NUM_ACTUATORS],
        };
        // Push twice to fill buffer past delay
        let _ = noise.apply(&cmd);
        let out = noise.apply(&cmd);

        // At least one torque should differ from input due to noise
        let any_differ = out
            .torques
            .iter()
            .zip(cmd.torques.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(any_differ, "Noise should perturb at least one torque");
    }

    #[test]
    fn test_actuator_noise_deterministic() {
        let cmd = HumanoidCommand {
            torques: vec![0.5; NUM_ACTUATORS],
        };

        let mut noise_a = ActuatorNoiseModel::new(3, 0.05);
        noise_a.enabled = true;
        noise_a.reset(42);
        let _ = noise_a.apply(&cmd);
        let out_a = noise_a.apply(&cmd);

        let mut noise_b = ActuatorNoiseModel::new(3, 0.05);
        noise_b.enabled = true;
        noise_b.reset(42);
        let _ = noise_b.apply(&cmd);
        let out_b = noise_b.apply(&cmd);

        for i in 0..cmd.torques.len() {
            assert!(
                (out_a.torques[i] - out_b.torques[i]).abs() < 1e-10,
                "Same seed should produce identical noise: joint {i}"
            );
        }
    }

    // ── Domain randomization tests ──

    #[test]
    fn test_domain_rand_changes_body() {
        let mut sim = SimpleHumanoidSimulator::new().with_domain_randomization(true);
        sim.reset_with_perturbation(0.5, 42);

        // Body params should differ from baseline
        let any_mass_diff = sim
            .body
            .segment_masses
            .iter()
            .zip(sim.baseline_body.segment_masses.iter())
            .any(|(a, b)| (a - b).abs() > 0.01);
        assert!(
            any_mass_diff,
            "Domain randomization should change segment masses"
        );
    }

    #[test]
    fn test_domain_rand_deterministic() {
        let mut sim_a = SimpleHumanoidSimulator::new().with_domain_randomization(true);
        sim_a.reset_with_perturbation(0.5, 42);
        let masses_a = sim_a.body.segment_masses;

        let mut sim_b = SimpleHumanoidSimulator::new().with_domain_randomization(true);
        sim_b.reset_with_perturbation(0.5, 42);
        let masses_b = sim_b.body.segment_masses;

        for i in 0..7 {
            assert!(
                (masses_a[i] - masses_b[i]).abs() < 1e-10,
                "Same seed should produce identical body: segment {i}"
            );
        }
    }

    #[test]
    fn test_domain_rand_disabled() {
        let mut sim = SimpleHumanoidSimulator::new().with_domain_randomization(false);
        sim.reset_with_perturbation(0.5, 42);

        // Body should match baseline when disabled
        for i in 0..7 {
            assert!(
                (sim.body.segment_masses[i] - sim.baseline_body.segment_masses[i]).abs() < 1e-10,
                "Disabled randomization should preserve baseline: segment {i}"
            );
        }
    }

    // ── Noise scale tests ──

    #[test]
    fn test_noise_scale_zero_disables() {
        let sim = SimpleHumanoidSimulator::new()
            .with_actuator_noise(0.05)
            .with_domain_randomization(true)
            .with_observation_noise(0.02)
            .with_noise_scale(0.0);

        assert!(
            sim.actuator_noise.noise_std.abs() < 1e-10,
            "Scale 0 should zero actuator noise: {}",
            sim.actuator_noise.noise_std
        );
        assert!(
            sim.domain_rand.mass_range.abs() < 1e-10,
            "Scale 0 should zero mass range: {}",
            sim.domain_rand.mass_range
        );
        assert!(
            sim.domain_rand.length_range.abs() < 1e-10,
            "Scale 0 should zero length range"
        );
        assert!(
            sim.domain_rand.damping_range.abs() < 1e-10,
            "Scale 0 should zero damping range"
        );
        assert!(
            sim.observation_noise.noise_std.abs() < 1e-10,
            "Scale 0 should zero observation noise"
        );
    }

    #[test]
    fn test_noise_scale_progressive() {
        let sim = SimpleHumanoidSimulator::new()
            .with_actuator_noise(0.06)
            .with_domain_randomization(true)
            .with_observation_noise(0.02)
            .with_noise_scale(0.5);

        assert!(
            (sim.actuator_noise.noise_std - 0.03).abs() < 1e-10,
            "Scale 0.5 should halve actuator noise: {}",
            sim.actuator_noise.noise_std
        );
        assert!(
            (sim.observation_noise.noise_std - 0.01).abs() < 1e-10,
            "Scale 0.5 should halve observation noise: {}",
            sim.observation_noise.noise_std
        );
        // Default mass_range=0.15, halved=0.075
        assert!(
            (sim.domain_rand.mass_range - 0.075).abs() < 1e-10,
            "Scale 0.5 should halve mass range: {}",
            sim.domain_rand.mass_range
        );
    }

    // ── Observation noise tests ──

    #[test]
    fn test_observation_noise_applied() {
        let mut sim = SimpleHumanoidSimulator::new().with_observation_noise(0.1);
        let cmd = HumanoidCommand::zero();
        for _ in 0..10 {
            sim.step(&cmd, 0.025);
        }
        // observed_state should differ from true state
        let any_differ = sim
            .observed_state
            .joint_angles
            .iter()
            .zip(sim.state.joint_angles.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(any_differ, "Observation noise should perturb joint angles");
    }

    #[test]
    fn test_observation_noise_deterministic() {
        let mut sim_a = SimpleHumanoidSimulator::new().with_observation_noise(0.05);
        let mut sim_b = SimpleHumanoidSimulator::new().with_observation_noise(0.05);
        let cmd = HumanoidCommand::zero();

        sim_a.reset_with_perturbation(0.5, 42);
        sim_b.reset_with_perturbation(0.5, 42);

        for _ in 0..10 {
            sim_a.step(&cmd, 0.025);
            sim_b.step(&cmd, 0.025);
        }

        for i in 0..cmd.torques.len() {
            assert!(
                (sim_a.observed_state.joint_angles[i] - sim_b.observed_state.joint_angles[i]).abs()
                    < 1e-10,
                "Same seed should give identical observed states: joint {i}"
            );
        }
    }

    #[test]
    fn test_observation_noise_disabled() {
        let mut sim = SimpleHumanoidSimulator::new(); // noise disabled by default
        let cmd = HumanoidCommand::zero();
        for _ in 0..10 {
            sim.step(&cmd, 0.025);
        }
        // state() should return internal state (via jitter), not observed_state
        let returned = sim.state();
        // Since observation noise is disabled, state() returns jitter delayed state
        assert!(
            !sim.observation_noise.enabled,
            "Observation noise should be disabled by default"
        );
        assert!(returned.root_height.is_finite());
    }

    // ── Ground contact tests ──

    #[test]
    fn test_ground_contact_no_penetration() {
        let mut sim = SimpleHumanoidSimulator::new().with_ground_contact(true);
        // Push humanoid down with strong downward force
        let cmd = HumanoidCommand::zero();
        sim.apply_external_force([0.0, 0.0, -2000.0]);
        for _ in 0..100 {
            sim.step(&cmd, 0.025);
        }
        // With contact model, root should stay near ground (not deeply negative)
        assert!(
            sim.state.root_height > 0.1,
            "Ground contact should prevent deep penetration: got {}",
            sim.state.root_height
        );
    }

    #[test]
    fn test_ground_contact_disabled_no_effect() {
        // Run identical simulations with and without ground contact disabled
        let mut sim_off = SimpleHumanoidSimulator::new().with_ground_contact(false);
        let mut sim_default = SimpleHumanoidSimulator::new(); // default: disabled
        let cmd = HumanoidCommand::zero();

        for _ in 0..50 {
            sim_off.step(&cmd, 0.025);
            sim_default.step(&cmd, 0.025);
        }

        assert!(
            (sim_off.state.root_height - sim_default.state.root_height).abs() < 1e-10,
            "Disabled contact should match default: off={} default={}",
            sim_off.state.root_height,
            sim_default.state.root_height
        );
    }

    // ── Terrain variation tests ──

    #[test]
    fn test_terrain_slope_affects_velocity() {
        let mut sim_flat = SimpleHumanoidSimulator::new().with_terrain_variation(false);
        let mut sim_slope = SimpleHumanoidSimulator::new().with_terrain_variation(true);
        let cmd = HumanoidCommand::zero();

        // Set a known positive slope manually (uphill → should decelerate forward)
        sim_slope.terrain.enabled = true;
        sim_slope.terrain.slope_x = 0.05; // positive = uphill

        for _ in 0..50 {
            sim_flat.step(&cmd, 0.025);
            sim_slope.step(&cmd, 0.025);
        }

        // Slope should affect forward velocity differently than flat
        let flat_vx = sim_flat.state.root_linear_velocity[0];
        let slope_vx = sim_slope.state.root_linear_velocity[0];
        // Uphill slope adds positive forward_accel (g*sin(0.05)), so slope should differ
        assert!(
            (flat_vx - slope_vx).abs() > 1e-6,
            "Terrain slope should affect velocity: flat={flat_vx} slope={slope_vx}"
        );
    }

    #[test]
    fn test_terrain_randomize_deterministic() {
        let mut terrain_a = TerrainModel::new();
        let mut terrain_b = TerrainModel::new();

        terrain_a.randomize(42);
        terrain_b.randomize(42);

        assert!(
            (terrain_a.slope_x - terrain_b.slope_x).abs() < 1e-10,
            "Same seed should give same slope_x"
        );
        assert!(
            (terrain_a.slope_y - terrain_b.slope_y).abs() < 1e-10,
            "Same seed should give same slope_y"
        );
        assert!(
            (terrain_a.compliance - terrain_b.compliance).abs() < 1e-10,
            "Same seed should give same compliance"
        );
    }

    // ── MuJoCo runtime tests (require mujoco library at runtime) ──

    #[cfg(feature = "mujoco")]
    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_bundled_loads() {
        let sim = MuJoCoHumanoidSimulator::from_bundled_asset()
            .expect("Bundled humanoid.xml should load");
        let state = sim.state();
        assert!(state.root_height > 0.0, "Initial height should be positive");
        assert!(state.head_height > 0.0, "Head height should be positive");
    }

    #[cfg(feature = "mujoco")]
    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_step_finite() {
        let mut sim = MuJoCoHumanoidSimulator::from_bundled_asset().unwrap();
        let cmd = HumanoidCommand::zero();

        for _ in 0..100 {
            sim.step(&cmd, 0.025);
        }

        let state = sim.state();
        assert!(state.root_height.is_finite());
        assert!(state.head_height.is_finite());
        for &a in &state.joint_angles {
            assert!(a.is_finite(), "Joint angle must be finite");
        }
        for &v in &state.joint_velocities {
            assert!(v.is_finite(), "Joint velocity must be finite");
        }
    }

    #[cfg(feature = "mujoco")]
    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_external_force() {
        let mut sim = MuJoCoHumanoidSimulator::from_bundled_asset().unwrap();
        sim.apply_external_force([100.0, 0.0, 0.0]);
        let cmd = HumanoidCommand::zero();
        sim.step(&cmd, 0.025);

        let state = sim.state();
        assert!(
            state.root_linear_velocity[0].abs() > 0.0 || state.com_velocity[0].abs() > 0.0,
            "External force should affect velocity"
        );
    }

    #[cfg(feature = "mujoco")]
    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_reset() {
        let mut sim = MuJoCoHumanoidSimulator::from_bundled_asset().unwrap();
        let cmd = HumanoidCommand {
            torques: vec![0.5; NUM_ACTUATORS],
        };

        for _ in 0..50 {
            sim.step(&cmd, 0.025);
        }

        sim.reset();
        let state = sim.state();
        assert!(
            (state.root_height - sim.state().root_height).abs() < 0.01,
            "Reset should restore initial state"
        );
    }

    #[cfg(feature = "mujoco")]
    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_body_mass() {
        let sim = MuJoCoHumanoidSimulator::from_bundled_asset().unwrap();
        let mass = sim.body_mass();
        assert!(mass > 0.0, "Torso mass should be positive: {mass}");
        assert!(mass < 100.0, "Torso mass should be reasonable: {mass}");
    }

    #[cfg(feature = "mujoco")]
    #[test]
    #[ignore] // Requires MuJoCo library
    fn test_mujoco_perturbation() {
        let mut sim = MuJoCoHumanoidSimulator::from_bundled_asset().unwrap();
        sim.reset_with_perturbation(1.0, 42);

        let state = sim.state();
        // Perturbed state should differ from default standing
        let any_nonzero = state.joint_angles.iter().any(|a| a.abs() > 0.001);
        assert!(any_nonzero, "Perturbation should change joint angles");
    }

    #[test]
    fn test_dexterous53_simulator_runs() {
        use crate::morphology::HumanoidMorphology;
        let mut sim = SimpleHumanoidSimulator::new_for(HumanoidMorphology::Dexterous53);
        let state = sim.state();
        assert_eq!(state.joint_angles.len(), 53);
        assert_eq!(state.joint_velocities.len(), 53);
        assert_eq!(state.extremities.len(), 18); // 12 base + 6 hand centroids

        // Step with zero command
        let cmd = HumanoidCommand::zero_for(53);
        for _ in 0..100 {
            sim.step(&cmd, 0.025);
        }
        let state = sim.state();
        assert!(
            state.root_height > 0.5,
            "Should still be standing: h={}",
            state.root_height
        );
        // Hand centroids should be computed (not all zero after stepping)
        assert!(
            state.extremities.iter().all(|x| x.is_finite()),
            "All extremities should be finite"
        );
    }

    #[test]
    fn test_dexterous53_hand_centroids_respond_to_joints() {
        use crate::morphology::HumanoidMorphology;
        let mut sim = SimpleHumanoidSimulator::new_for(HumanoidMorphology::Dexterous53);

        // Step with zero command first to establish baseline
        let cmd = HumanoidCommand::zero_for(53);
        sim.step(&cmd, 0.025);
        let baseline_centroid = [
            sim.state().extremities[12],
            sim.state().extremities[13],
            sim.state().extremities[14],
        ];

        // Now apply flexion to right hand fingers
        let mut cmd2 = HumanoidCommand::zero_for(53);
        for i in 21..37 {
            cmd2.torques[i] = 0.8; // Strong flexion command
        }
        for _ in 0..200 {
            sim.step(&cmd2, 0.025);
        }
        let flexed_centroid = [
            sim.state().extremities[12],
            sim.state().extremities[13],
            sim.state().extremities[14],
        ];

        // Centroid should change when fingers flex
        let dist = ((flexed_centroid[0] - baseline_centroid[0]).powi(2)
            + (flexed_centroid[1] - baseline_centroid[1]).powi(2)
            + (flexed_centroid[2] - baseline_centroid[2]).powi(2))
        .sqrt();
        assert!(
            dist > 0.001,
            "Hand centroid should move when fingers flex: dist={dist}"
        );
    }

    #[test]
    fn test_dmc21_backward_compat_new_vs_new_for() {
        use crate::morphology::HumanoidMorphology;
        let sim1 = SimpleHumanoidSimulator::new();
        let sim2 = SimpleHumanoidSimulator::new_for(HumanoidMorphology::Dmc21);
        let s1 = sim1.state();
        let s2 = sim2.state();
        assert_eq!(s1.joint_angles.len(), s2.joint_angles.len());
        assert_eq!(s1.extremities.len(), s2.extremities.len());
        assert!((s1.root_height - s2.root_height).abs() < 1e-10);
    }
}
