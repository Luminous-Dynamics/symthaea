// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core data types for DMC humanoid bipedal control.

use serde::{Deserialize, Serialize};

use crate::morphology::HumanoidMorphology;

/// Default number of actuated joints (DMC standard humanoid).
/// Kept for backward compatibility — prefer `HumanoidMorphology::num_actuators()`.
pub const NUM_ACTUATORS: usize = 21;

/// Number of state channels for HDC encoding (DMC21 default).
/// For other morphologies, use `HumanoidMorphology::num_channels()`.
pub const NUM_STATE_CHANNELS: usize = 72;

/// Joint names matching the dm_control humanoid MJCF (DMC21).
/// For other morphologies, use `HumanoidMorphology::joint_names()`.
pub const JOINT_NAMES: [&str; NUM_ACTUATORS] = [
    "abdomen_y",
    "abdomen_z",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "right_ankle_x",
    "right_ankle_y",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "left_ankle_x",
    "left_ankle_y",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
];

/// A structural or numerical violation in a humanoid observation frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateValidationError {
    JointAngleCount { expected: usize, actual: usize },
    JointVelocityCount { expected: usize, actual: usize },
    ExtremityChannelCount { expected: usize, actual: usize },
    NonFiniteValue,
    NegativeTimestamp,
}

/// Physical interpretation advertised by an embodiment backend.
///
/// A command must never silently change meaning between simulation and hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActuationMode {
    /// Unitless command in [-1, 1], scaled by a backend torque limit.
    NormalizedTorque,
    /// Torque in newton-metres.
    TorqueNewtonMetres,
    /// Unitless position target in [-1, 1], mapped through joint calibration.
    NormalizedPosition,
    /// Absolute joint position target in radians.
    PositionTargetRadians,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodimentCapabilities {
    pub backend_name: String,
    pub morphology: HumanoidMorphology,
    pub actuator_count: usize,
    pub observation_schema_id: String,
    pub actuation_mode: ActuationMode,
    pub privileged_truth_available: bool,
    pub command_deadlines_enforced: bool,
}

impl EmbodimentCapabilities {
    pub fn validate_for_morphology(&self, expected: HumanoidMorphology) -> Result<(), String> {
        if self.morphology != expected {
            return Err(format!(
                "backend {} exposes {:?}, expected {:?}",
                self.backend_name, self.morphology, expected
            ));
        }
        if self.actuator_count != expected.num_actuators() {
            return Err(format!(
                "backend {} exposes {} actuators, expected {}",
                self.backend_name,
                self.actuator_count,
                expected.num_actuators()
            ));
        }
        if self.observation_schema_id != expected.schema_id() {
            return Err(format!(
                "backend {} exposes schema {}, expected {}",
                self.backend_name,
                self.observation_schema_id,
                expected.schema_id()
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandValidationError {
    ActuatorCount { expected: usize, actual: usize },
    NonFiniteValue { index: usize },
    NormalizedValueOutOfRange { index: usize },
}

/// Full humanoid state: proprioceptive + computed features.
///
/// Joint arrays are dynamically sized to support different morphologies.
/// DMC21: 21 joints (72 channels). Dexterous53: 53 joints (142 channels).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidState {
    // ── Proprioceptive ──
    /// Root body height (qpos[2], z excluding global x/y).
    pub root_height: f64,
    /// Privileged root position in world coordinates. This is deliberately
    /// excluded from `to_channels()` so policy observations remain translation
    /// invariant while evaluators and recovery controllers can reason about
    /// support geometry.
    pub root_position: [f64; 3],
    /// Root orientation quaternion [w, x, y, z] (qpos[3..7]).
    pub root_quaternion: [f64; 4],
    /// Joint angles for all actuated joints. Length = morphology.num_actuators().
    pub joint_angles: Vec<f64>,
    /// Root linear velocity in world frame (qvel[0..3]).
    pub root_linear_velocity: [f64; 3],
    /// Root angular velocity in body frame (qvel[3..6]).
    pub root_angular_velocity: [f64; 3],
    /// Joint velocities for all actuated joints. Length = morphology.num_actuators().
    pub joint_velocities: Vec<f64>,

    // ── Computed features ──
    /// Head body height (xpos['head'][2]).
    pub head_height: f64,
    /// Torso vertical direction (z-column of torso rotation matrix).
    pub torso_vertical: [f64; 3],
    /// Extremity world positions: right_hand(3), left_hand(3), right_foot(3), left_foot(3).
    /// For Dexterous53+, additional hand centroid features are appended.
    pub extremities: Vec<f64>,
    /// Center-of-mass velocity (from subtree_com derivative).
    pub com_velocity: [f64; 3],

    // ── Metadata ──
    /// Simulation time in seconds.
    pub timestamp: f64,
}

impl HumanoidState {
    /// Neutral (all-zero pose, upright orientation) state sized for the given
    /// morphology's actuator/extremity counts — a standing baseline for tests
    /// and callers that don't care about a specific pose.
    pub fn default_for(morphology: HumanoidMorphology) -> Self {
        let num_actuators = morphology.num_actuators();
        let num_extremities = morphology.num_extremity_channels();
        Self {
            root_height: 0.0,
            root_position: [0.0; 3],
            root_quaternion: [1.0, 0.0, 0.0, 0.0],
            joint_angles: vec![0.0; num_actuators],
            root_linear_velocity: [0.0; 3],
            root_angular_velocity: [0.0; 3],
            joint_velocities: vec![0.0; num_actuators],
            head_height: 0.0,
            torso_vertical: [0.0, 0.0, 1.0],
            extremities: vec![0.0; num_extremities],
            com_velocity: [0.0; 3],
            timestamp: 0.0,
        }
    }

    /// Pack the translation-invariant observation channels into the stable
    /// schema order used by the encoder. `root_position` is privileged state
    /// and is intentionally not encoded.
    ///
    /// DMC21 layout (72D total):
    /// - root height: 1
    /// - root quaternion: 4
    /// - joint angles: 21
    /// - root linear velocity: 3
    /// - root angular velocity: 3
    /// - joint velocities: 21
    /// - head height: 1
    /// - torso vertical: 3
    /// - extremity positions: 12
    /// - center-of-mass velocity: 3
    ///
    /// Extended morphologies retain the same ordering while increasing the joint
    /// and extremity sections. Use [`HumanoidMorphology::num_observation_channels`]
    /// instead of hard-coding offsets.
    pub fn to_channels(&self) -> Vec<f32> {
        let mut channels = Vec::with_capacity(self.num_channels());

        // Proprioceptive (53D)
        channels.push(self.root_height as f32);
        for &q in &self.root_quaternion {
            channels.push(q as f32);
        }
        for &a in &self.joint_angles {
            channels.push(a as f32);
        }
        for &v in &self.root_linear_velocity {
            channels.push(v as f32);
        }
        for &w in &self.root_angular_velocity {
            channels.push(w as f32);
        }
        for &jv in &self.joint_velocities {
            channels.push(jv as f32);
        }

        // Computed features (19D)
        channels.push(self.head_height as f32);
        for &tv in &self.torso_vertical {
            channels.push(tv as f32);
        }
        for &ext in &self.extremities {
            channels.push(ext as f32);
        }
        for &cv in &self.com_velocity {
            channels.push(cv as f32);
        }

        channels
    }

    /// Number of channels produced by `to_channels()`.
    pub fn num_channels(&self) -> usize {
        // 1 + 4 + n_joints + 3 + 3 + n_joints + 1 + 3 + n_extremities + 3
        1 + 4
            + self.joint_angles.len()
            + 3
            + 3
            + self.joint_velocities.len()
            + 1
            + 3
            + self.extremities.len()
            + 3
    }

    /// Number of actuated joints in this state.
    pub fn num_actuators(&self) -> usize {
        self.joint_angles.len()
    }

    /// Validate this frame against a morphology before it enters the controller,
    /// replay buffer, reward function, or hardware boundary.
    pub fn validate_for(&self, morphology: HumanoidMorphology) -> Result<(), StateValidationError> {
        let expected_joints = morphology.num_actuators();
        if self.joint_angles.len() != expected_joints {
            return Err(StateValidationError::JointAngleCount {
                expected: expected_joints,
                actual: self.joint_angles.len(),
            });
        }
        if self.joint_velocities.len() != expected_joints {
            return Err(StateValidationError::JointVelocityCount {
                expected: expected_joints,
                actual: self.joint_velocities.len(),
            });
        }

        let expected_extremities = morphology.num_extremity_channels();
        if self.extremities.len() != expected_extremities {
            return Err(StateValidationError::ExtremityChannelCount {
                expected: expected_extremities,
                actual: self.extremities.len(),
            });
        }

        let finite = self.root_height.is_finite()
            && self.root_position.iter().all(|value| value.is_finite())
            && self.root_quaternion.iter().all(|value| value.is_finite())
            && self.joint_angles.iter().all(|value| value.is_finite())
            && self
                .root_linear_velocity
                .iter()
                .all(|value| value.is_finite())
            && self
                .root_angular_velocity
                .iter()
                .all(|value| value.is_finite())
            && self.joint_velocities.iter().all(|value| value.is_finite())
            && self.head_height.is_finite()
            && self.torso_vertical.iter().all(|value| value.is_finite())
            && self.extremities.iter().all(|value| value.is_finite())
            && self.com_velocity.iter().all(|value| value.is_finite())
            && self.timestamp.is_finite();
        if !finite {
            return Err(StateValidationError::NonFiniteValue);
        }
        if self.timestamp < 0.0 {
            return Err(StateValidationError::NegativeTimestamp);
        }
        Ok(())
    }

    /// Construct a default upright standing state (DMC21: 21 joints).
    pub fn standing() -> Self {
        Self::standing_for(HumanoidMorphology::Dmc21)
    }

    /// Construct a standing state for a specific morphology.
    pub fn standing_for(morphology: HumanoidMorphology) -> Self {
        let n = morphology.num_actuators();
        let n_extremities = morphology.num_extremity_channels();
        Self {
            root_height: 1.3,
            root_position: [0.0, 0.0, 1.3],
            root_quaternion: [1.0, 0.0, 0.0, 0.0],
            joint_angles: vec![0.0; n],
            root_linear_velocity: [0.0; 3],
            root_angular_velocity: [0.0; 3],
            joint_velocities: vec![0.0; n],
            head_height: 1.4,
            torso_vertical: [0.0, 0.0, 1.0],
            extremities: vec![0.0; n_extremities],
            com_velocity: [0.0; 3],
            timestamp: 0.0,
        }
    }

    /// Construct from MuJoCo qpos/qvel arrays + computed features.
    ///
    /// For DMC21: qpos has 28 entries (7 root + 21 joints), qvel has 27 (6 root + 21 joints).
    /// For extended morphologies: qpos and qvel are longer accordingly.
    pub fn from_mujoco(
        qpos: &[f64],
        qvel: &[f64],
        head_height: f64,
        torso_vertical: [f64; 3],
        extremities: &[f64],
        com_velocity: [f64; 3],
        t: f64,
    ) -> Self {
        let n_joints = qpos.len() - 7; // Root takes 7 qpos entries
        let joint_angles = qpos[7..7 + n_joints].to_vec();
        let joint_velocities = qvel[6..6 + n_joints].to_vec();

        Self {
            root_height: qpos[2],
            root_position: [qpos[0], qpos[1], qpos[2]],
            root_quaternion: [qpos[3], qpos[4], qpos[5], qpos[6]],
            joint_angles,
            root_linear_velocity: [qvel[0], qvel[1], qvel[2]],
            root_angular_velocity: [qvel[3], qvel[4], qvel[5]],
            joint_velocities,
            head_height,
            torso_vertical,
            extremities: extremities.to_vec(),
            com_velocity,
            timestamp: t,
        }
    }

    /// Horizontal speed magnitude (x/y components of COM velocity).
    pub fn horizontal_speed(&self) -> f64 {
        (self.com_velocity[0].powi(2) + self.com_velocity[1].powi(2)).sqrt()
    }

    /// Uprightness: z-component of torso vertical (1.0 = perfectly upright).
    pub fn uprightness(&self) -> f64 {
        self.torso_vertical[2]
    }

    /// Total angular momentum magnitude (from angular velocity).
    pub fn angular_momentum(&self) -> f64 {
        let [wx, wy, wz] = self.root_angular_velocity;
        (wx * wx + wy * wy + wz * wz).sqrt()
    }

    /// Mean absolute joint velocity (energy proxy).
    pub fn mean_joint_speed(&self) -> f64 {
        if self.joint_velocities.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.joint_velocities.iter().map(|v| v.abs()).sum();
        sum / self.joint_velocities.len() as f64
    }
}

/// Motor command output: joint torques in [-1, 1], mapped by actuator gear ratios.
///
/// Dynamically sized to match the morphology's actuator count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidCommand {
    /// Normalized torques for all actuated joints, each in [-1, 1].
    pub torques: Vec<f32>,
}

impl HumanoidCommand {
    /// Zero command for DMC21 (21 joints). Use `zero_for()` for other morphologies.
    pub fn zero() -> Self {
        Self::zero_for(NUM_ACTUATORS)
    }

    /// Zero command for a specific number of actuators.
    pub fn zero_for(num_actuators: usize) -> Self {
        Self {
            torques: vec![0.0; num_actuators],
        }
    }

    /// Construct from a slice of raw f32 values, clamped to [-1, 1].
    pub fn from_raw(values: &[f32]) -> Self {
        let torques: Vec<f32> = values.iter().map(|&v| v.clamp(-1.0, 1.0)).collect();
        Self { torques }
    }

    /// Number of actuators in this command.
    pub fn num_actuators(&self) -> usize {
        self.torques.len()
    }

    /// Validate the frame against backend capabilities before actuation.
    pub fn validate_for(
        &self,
        expected_actuators: usize,
        mode: ActuationMode,
    ) -> Result<(), CommandValidationError> {
        if self.torques.len() != expected_actuators {
            return Err(CommandValidationError::ActuatorCount {
                expected: expected_actuators,
                actual: self.torques.len(),
            });
        }

        for (index, value) in self.torques.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(CommandValidationError::NonFiniteValue { index });
            }
            if matches!(
                mode,
                ActuationMode::NormalizedTorque | ActuationMode::NormalizedPosition
            ) && !(-1.0..=1.0).contains(&value)
            {
                return Err(CommandValidationError::NormalizedValueOutOfRange { index });
            }
        }
        Ok(())
    }

    /// Clamp all torques to [-1, 1].
    pub fn clamped(self) -> Self {
        let torques: Vec<f32> = self
            .torques
            .into_iter()
            .map(|t| t.clamp(-1.0, 1.0))
            .collect();
        Self { torques }
    }

    /// Convert to f64 vec for MuJoCo ctrl.
    pub fn to_ctrl(&self) -> Vec<f64> {
        self.torques.iter().map(|&t| t as f64).collect()
    }

    /// Mean absolute torque (control effort proxy).
    pub fn control_effort(&self) -> f32 {
        if self.torques.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.torques.iter().map(|t| t.abs()).sum();
        sum / self.torques.len() as f32
    }

    /// Add exploration noise, then clamp.
    pub fn with_noise(self, noise: &[f32]) -> Self {
        let torques: Vec<f32> = self
            .torques
            .iter()
            .zip(noise.iter())
            .map(|(&t, &n)| (t + n).clamp(-1.0, 1.0))
            .collect();
        Self { torques }
    }
}

/// DMC task variants for the humanoid benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidTask {
    /// Stand upright with minimal movement.
    Stand,
    /// Walk forward at ~1 m/s.
    Walk,
    /// Run forward at ~10 m/s.
    Run,
    /// Extend an arm toward a target object. Available to all morphologies
    /// with the canonical shoulder/elbow chain; wrist-enabled bodies can refine
    /// end-effector orientation.
    Reach,
    /// Reach + close fingers around object (requires Dexterous53+).
    Grasp,
}

impl HumanoidTask {
    /// Target horizontal speed for this task.
    pub fn target_speed(&self) -> f64 {
        match self {
            HumanoidTask::Stand | HumanoidTask::Reach | HumanoidTask::Grasp => 0.0,
            HumanoidTask::Walk => 1.0,
            HumanoidTask::Run => 10.0,
        }
    }

    /// Whether this task requires independently actuated fingers.
    pub fn requires_dexterous(&self) -> bool {
        matches!(self, HumanoidTask::Grasp)
    }
}

/// Per-step telemetry from the humanoid controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidTelemetry {
    /// Current simulation step.
    pub step: usize,
    /// Simulation time.
    pub time: f64,
    /// Head height.
    pub head_height: f64,
    /// Uprightness (torso_vertical[2]).
    pub uprightness: f64,
    /// Horizontal speed (COM).
    pub horizontal_speed: f64,
    /// Standing reward.
    pub standing_reward: f64,
    /// Episode reward for current task.
    pub episode_reward: f64,
    /// Free energy (from FEP agent).
    pub free_energy: f64,
    /// Current tau modulation factor.
    pub tau_factor: f32,
    /// Current learning rate.
    pub learning_rate: f32,
    /// Mean absolute torque.
    pub control_effort: f32,
    /// Learned residual authority after stability and uncertainty gating.
    #[serde(default)]
    pub residual_authority: f32,
    /// Mean magnitude of deterministic balance correction.
    #[serde(default)]
    pub balance_effort: f32,
    /// Deterministic recovery mode selected from capture-point stability.
    #[serde(default)]
    pub recovery_mode: crate::recovery::RecoveryMode,
    /// Signed capture-point margin to the support polygon in meters.
    #[serde(default)]
    pub capture_margin_m: f64,
    /// Mean magnitude of capture/protective recovery correction.
    #[serde(default)]
    pub recovery_effort: f32,
    /// Predictive recovery footstep target, when one is active.
    #[serde(default)]
    pub planned_footstep_world_m: Option<[f64; 3]>,
    /// Terrain-shaped swing apex, when a recovery step is active.
    #[serde(default)]
    pub planned_swing_apex_world_m: Option<[f64; 3]>,
    /// Minimum confidence of terrain samples along the swing path.
    #[serde(default)]
    pub terrain_confidence: f64,
    /// Selected vertical foot clearance above the support surface.
    #[serde(default)]
    pub terrain_clearance_m: f64,
    /// Number of active whole-body projected dynamics constraints.
    #[serde(default)]
    pub whole_body_active_constraints: usize,
    /// Maximum normalized joint-range utilization.
    #[serde(default)]
    pub whole_body_joint_utilization: f64,
    /// Residual whole-body torso objective error.
    #[serde(default)]
    pub whole_body_objective_residual: f64,
    /// Whether the whole-body allocation was finite and feasible.
    #[serde(default = "default_true")]
    pub whole_body_feasible: bool,
    /// Sparse inverse-dynamics solver iterations used this step.
    #[serde(default)]
    pub inverse_dynamics_iterations: usize,
    /// Maximum remaining inverse-dynamics constraint violation.
    #[serde(default)]
    pub inverse_dynamics_max_violation: f64,
    /// Whether inverse dynamics used its deterministic fallback allocator.
    #[serde(default)]
    pub inverse_dynamics_fallback: bool,
    /// Protective fall/get-up phase active at this step.
    #[serde(default)]
    pub fall_protection_phase: crate::fall_protection::FallProtectionPhase,
    /// Estimated body orientation used by the protective controller.
    #[serde(default)]
    pub fall_orientation: crate::fall_protection::FallOrientation,
    /// Mean bounded authority used by the protective controller.
    #[serde(default)]
    pub protective_effort: f32,
    /// Progress through the deterministic get-up sequence.
    #[serde(default)]
    pub get_up_progress: f64,
    /// Total measured or modeled vertical ground reaction force.
    #[serde(default)]
    pub total_normal_force_n: f64,
    /// Force-weighted center of pressure, when available.
    #[serde(default)]
    pub center_of_pressure_world_m: Option<[f64; 2]>,
    /// Estimated bilateral support phase.
    #[serde(default = "default_support_phase")]
    pub support_phase: crate::hierarchical::SupportPhase,
    /// Contact source confidence and freshness gate used by control.
    #[serde(default)]
    pub contact_trust: f32,
    /// Current right foot z-height.
    pub r_foot_z: f64,
    /// Current left foot z-height.
    pub l_foot_z: f64,
}

fn default_support_phase() -> crate::hierarchical::SupportPhase {
    crate::hierarchical::SupportPhase::DoubleSupport
}

fn default_true() -> bool {
    true
}

/// Configuration for the humanoid training system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidConfig {
    /// Morphology variant (default: Dmc21 for backward compatibility).
    pub morphology: HumanoidMorphology,
    /// Physics loop frequency in Hz (default: 40, DMC standard 0.025s timestep).
    pub physics_hz: f64,
    /// Cognitive tick frequency in Hz (default: 10, FEP agent rate).
    pub cognitive_hz: f64,
    /// Training learning rate.
    pub learning_rate: f32,
    /// Number of HDC-LTC network layers (default: 4, deeper than drone's 2).
    pub network_layers: usize,
    /// Neurons per layer (default: 12, wider than drone's 4).
    pub neurons_per_layer: usize,
    /// Number of HDC level codebook entries.
    pub num_levels: usize,
    /// Number of training episodes.
    pub num_episodes: usize,
    /// Steps per episode (default: 1000, DMC standard).
    pub steps_per_episode: usize,
    /// Training frequency divider (train every N physics steps).
    pub train_every: usize,
    /// Genesis seed phrase for deterministic initialization.
    pub genesis_phrase: String,
    /// Whether to collect per-step telemetry.
    pub collect_telemetry: bool,
    /// Experience replay buffer size.
    pub replay_buffer_size: usize,
    /// Number of replay samples per training step.
    pub replay_count: usize,
    /// Enable cosine annealing LR schedule.
    pub enable_lr_schedule: bool,
    /// Enable early termination on unrecoverable fall.
    pub early_termination: bool,
    /// Attempt bounded protective falling and deterministic get-up before an
    /// episode is terminated.
    #[serde(default = "default_true")]
    pub enable_fall_recovery: bool,
    /// Current task.
    pub task: HumanoidTask,
    /// Target speed override (None = use task default).
    pub target_speed: Option<f64>,
    /// Enable adaptive curriculum (performance-based phase transitions).
    pub adaptive_curriculum: bool,
    /// Standing reward threshold to consider standing "mastered".
    pub standing_mastery_threshold: f64,
    /// Consecutive qualifying episodes needed to advance curriculum phase.
    pub mastery_streak_required: usize,
    /// Exploration decay rate for the FEP agent's leaky accumulator.
    pub exploration_decay_rate: f64,
    /// Standard deviation of actuator noise for sim-to-real transfer (default: 0.03).
    pub actuator_noise_std: f64,
    /// Enable per-episode domain randomization of body parameters (default: true).
    pub domain_randomization: bool,
    /// Standard deviation of observation noise for sim-to-real transfer (default: 0.01).
    pub observation_noise_std: f64,
    /// Enable progressive noise curriculum: noise ramps 0→100% over training (default: true).
    pub progressive_noise: bool,
    /// Enable per-episode terrain variation (slope + compliance) (default: false).
    pub terrain_variation: bool,
    /// Target object position for Reach/Grasp tasks [x, y, z] in meters.
    /// Default: [0.3, -0.2, 1.0] (natural right-hand reach).
    pub object_position: [f64; 3],
    /// Which hand to use for Reach/Grasp tasks.
    pub reach_hand: crate::morphology::HandSide,
    /// Fuse delayed/noisy proprioception before encoding and control.
    #[serde(default = "default_true")]
    pub enable_state_estimation: bool,
    /// Route every policy command through the deterministic safety projector.
    #[serde(default = "default_true")]
    pub enable_safety_projection: bool,
    /// Enable gradient updates inside the recurrent HDC-LTC network. Disabled by
    /// default because the current core API cannot serialize learned recurrent
    /// parameters. Head-only learning is fully checkpointable and reproducible.
    #[serde(default)]
    pub enable_recurrent_learning: bool,
    /// Permit rollback of only the output projection. This is coherent only when
    /// recurrent learning is disabled; validation and training enforce that rule.
    #[serde(default)]
    pub enable_head_only_rollback: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HumanoidConfigError {
    NonFiniteOrNonPositive(&'static str),
    CognitiveRateExceedsPhysicsRate,
    ZeroSizedNetwork,
    InvalidLearningRate,
    InvalidLevelCount,
    EmptyTrainingSchedule,
    InvalidTrainingDivider,
    ReplayRequestedWithoutBuffer,
    DexterousTaskRequiresDexterousMorphology,
    NonFiniteObjectPosition,
}

impl std::fmt::Display for HumanoidConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteOrNonPositive(field) => write!(f, "{field} must be finite and positive"),
            Self::CognitiveRateExceedsPhysicsRate => {
                write!(f, "cognitive_hz must not exceed physics_hz")
            }
            Self::ZeroSizedNetwork => {
                write!(f, "network_layers and neurons_per_layer must be non-zero")
            }
            Self::InvalidLearningRate => {
                write!(f, "learning_rate must be finite and in [1e-6, 0.1]")
            }
            Self::InvalidLevelCount => write!(f, "num_levels must be at least 2"),
            Self::EmptyTrainingSchedule => {
                write!(f, "num_episodes and steps_per_episode must be non-zero")
            }
            Self::InvalidTrainingDivider => write!(f, "train_every must be non-zero"),
            Self::ReplayRequestedWithoutBuffer => {
                write!(f, "replay_count requires a non-zero replay_buffer_size")
            }
            Self::DexterousTaskRequiresDexterousMorphology => {
                write!(f, "Grasp requires Dexterous53 or FullSpine morphology")
            }
            Self::NonFiniteObjectPosition => write!(f, "object_position must be finite"),
        }
    }
}

impl std::error::Error for HumanoidConfigError {}

impl Default for HumanoidConfig {
    fn default() -> Self {
        Self {
            morphology: HumanoidMorphology::Dmc21,
            physics_hz: 40.0,
            cognitive_hz: 10.0,
            learning_rate: 0.0005,
            network_layers: 4,
            neurons_per_layer: 12,
            num_levels: 32,
            num_episodes: 200,
            steps_per_episode: 1000,
            train_every: 2, // 20Hz training
            genesis_phrase: "symthaea-humanoid-dmc".to_string(),
            collect_telemetry: false,
            replay_buffer_size: 128,
            replay_count: 3,
            enable_lr_schedule: true,
            early_termination: true,
            enable_fall_recovery: true,
            task: HumanoidTask::Stand,
            target_speed: None,
            adaptive_curriculum: true,
            standing_mastery_threshold: 0.85,
            mastery_streak_required: 3,
            exploration_decay_rate: 0.5,
            actuator_noise_std: 0.03,
            domain_randomization: true,
            observation_noise_std: 0.01,
            progressive_noise: true,
            terrain_variation: false,
            object_position: [0.3, -0.2, 1.0],
            reach_hand: crate::morphology::HandSide::Right,
            enable_state_estimation: true,
            enable_safety_projection: true,
            enable_recurrent_learning: false,
            enable_head_only_rollback: false,
        }
    }
}

impl HumanoidConfig {
    /// Validate scientific and runtime invariants before constructing a trainer.
    pub fn validate(&self) -> Result<(), HumanoidConfigError> {
        if !self.physics_hz.is_finite() || self.physics_hz <= 0.0 {
            return Err(HumanoidConfigError::NonFiniteOrNonPositive("physics_hz"));
        }
        if !self.cognitive_hz.is_finite() || self.cognitive_hz <= 0.0 {
            return Err(HumanoidConfigError::NonFiniteOrNonPositive("cognitive_hz"));
        }
        if self.cognitive_hz > self.physics_hz {
            return Err(HumanoidConfigError::CognitiveRateExceedsPhysicsRate);
        }
        if self.network_layers == 0 || self.neurons_per_layer == 0 {
            return Err(HumanoidConfigError::ZeroSizedNetwork);
        }
        if !self.learning_rate.is_finite() || !(1.0e-6..=0.1).contains(&self.learning_rate) {
            return Err(HumanoidConfigError::InvalidLearningRate);
        }
        if self.num_levels < 2 {
            return Err(HumanoidConfigError::InvalidLevelCount);
        }
        if self.num_episodes == 0 || self.steps_per_episode == 0 {
            return Err(HumanoidConfigError::EmptyTrainingSchedule);
        }
        if self.train_every == 0 {
            return Err(HumanoidConfigError::InvalidTrainingDivider);
        }
        if self.replay_count > 0 && self.replay_buffer_size == 0 {
            return Err(HumanoidConfigError::ReplayRequestedWithoutBuffer);
        }
        if self.task.requires_dexterous()
            && !matches!(
                self.morphology,
                HumanoidMorphology::Dexterous53 | HumanoidMorphology::FullSpine
            )
        {
            return Err(HumanoidConfigError::DexterousTaskRequiresDexterousMorphology);
        }
        if self.object_position.iter().any(|value| !value.is_finite()) {
            return Err(HumanoidConfigError::NonFiniteObjectPosition);
        }
        Ok(())
    }

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
}

/// PD controller gains for generating baseline standing targets.
///
/// Dynamically sized to match the morphology. Use `default()` for DMC21
/// or `for_morphology()` for extended morphologies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidPdGains {
    /// Proportional gains per joint.
    pub kp: Vec<f64>,
    /// Derivative (damping) gains per joint.
    pub kd: Vec<f64>,
}

impl Default for HumanoidPdGains {
    fn default() -> Self {
        Self::for_morphology(HumanoidMorphology::Dmc21)
    }
}

impl HumanoidPdGains {
    /// Create PD gains for a specific morphology.
    pub fn for_morphology(morphology: HumanoidMorphology) -> Self {
        let (kp, kd) = morphology.pd_gains();
        Self { kp, kd }
    }
}

/// Compute PD baseline standing command: drive all joints to zero (upright).
///
/// Target: all joint angles = 0 (MuJoCo default pose is approximately upright).
/// Output: normalized torques in [-1, 1].
// 5 parallel arrays (torques, joint_angles, joint_velocities, kp, kd) in lockstep;
// a zip chain here is less readable than the indexed loop.
#[allow(clippy::needless_range_loop)]
pub fn pd_standing_baseline(state: &HumanoidState, gains: &HumanoidPdGains) -> HumanoidCommand {
    let n = state.num_actuators();
    let mut torques = vec![0.0f32; n];
    for i in 0..n.min(gains.kp.len()) {
        let angle_error = 0.0 - state.joint_angles[i];
        let vel_damping = state.joint_velocities[i];
        torques[i] = (gains.kp[i] * angle_error - gains.kd[i] * vel_damping) as f32;
        torques[i] = torques[i].clamp(-1.0, 1.0);
    }
    HumanoidCommand { torques }
}

/// Compute PD walking baseline: forward lean + alternating leg swing.
///
/// Target angles include:
/// - Abdomen forward lean (sagittal tilt for forward COM shift)
/// - Cyclic hip/knee flexion (sinusoidal gait pattern)
/// - Foot-contact-aware stance/swing modulation
/// - Arms swing opposite to legs (natural counterbalance)
///
/// The `phase` parameter (0.0..1.0) drives the gait cycle.
/// `target_speed` scales the amplitude of the gait pattern.
/// Foot contact is detected from `state.extremities[8]` (right foot z)
/// and `state.extremities[11]` (left foot z).
pub fn pd_walking_baseline(
    state: &HumanoidState,
    gains: &HumanoidPdGains,
    phase: f64,
    target_speed: f64,
) -> HumanoidCommand {
    let n = state.num_actuators();
    let mut target_angles = vec![0.0f64; n];

    // Scale gait amplitude with target speed (0 at speed=0, full at speed=1+)
    let amplitude = (target_speed / 1.0).clamp(0.0, 1.0);
    let cycle = (phase * 2.0 * std::f64::consts::PI).sin();
    let half_cycle = ((phase * 2.0 * std::f64::consts::PI) + std::f64::consts::FRAC_PI_2).sin();

    // Foot contact detection: z < 0.05m = on ground (stance phase)
    let contact_threshold = 0.05;
    let r_foot_z = state.extremities[8];
    let l_foot_z = state.extremities[11];
    let r_stance = r_foot_z < contact_threshold;
    let l_stance = l_foot_z < contact_threshold;

    // Stance/swing modulation: blend sinusoidal with contact-aware targets
    // Stance leg: extend hip back (pushoff), extend knee
    // Swing leg: flex hip forward (lift), flex knee (clearance)
    // During double-support (both feet down), reduce contact modulation
    // to avoid overpowering the sinusoidal gait pattern.
    let contact_blend = if r_stance && l_stance { 0.1 } else { 0.4 };
    let sin_blend = 1.0 - contact_blend;

    let r_contact_mod = if r_stance { 0.3 } else { -0.3 };
    let l_contact_mod = if l_stance { 0.3 } else { -0.3 };

    let r_hip_target = sin_blend * cycle + contact_blend * r_contact_mod;
    let l_hip_target = sin_blend * (-cycle) + contact_blend * l_contact_mod;

    // Knee: during swing (not in stance), flex more for foot clearance
    let r_knee_swing = if r_stance { 0.0 } else { -0.3 };
    let l_knee_swing = if l_stance { 0.0 } else { -0.3 };

    // Abdomen: forward lean for COM shift (sagittal tilt)
    target_angles[0] = 0.15 * amplitude; // abdomen_y: forward lean
    target_angles[1] = 0.0; // abdomen_z: no yaw
    target_angles[2] = 0.0; // abdomen_x: no lateral lean

    // Right leg: contact-aware hip + knee
    target_angles[3] = 0.0; // right_hip_x: abduction (minimal)
    target_angles[4] = 0.0; // right_hip_z: rotation (minimal)
    target_angles[5] = 0.3 * amplitude * r_hip_target; // right_hip_y: contact-aware swing
    target_angles[6] = -0.4 * amplitude * half_cycle.max(0.0) + amplitude * r_knee_swing;
    target_angles[7] = -0.1 * amplitude * cycle; // right_ankle_x
    target_angles[8] = 0.05 * amplitude; // right_ankle_y: slight dorsiflexion

    // Left leg: contact-aware
    target_angles[9] = 0.0; // left_hip_x
    target_angles[10] = 0.0; // left_hip_z
    target_angles[11] = 0.3 * amplitude * l_hip_target; // left_hip_y: contact-aware
    target_angles[12] = -0.4 * amplitude * (-half_cycle).max(0.0) + amplitude * l_knee_swing;
    target_angles[13] = 0.1 * amplitude * cycle; // left_ankle_x
    target_angles[14] = 0.05 * amplitude; // left_ankle_y

    // Arms: opposite swing to legs (natural counterbalance)
    target_angles[15] = -0.2 * amplitude * cycle; // right_shoulder1: opposite to right leg
    target_angles[16] = 0.0;
    target_angles[17] = -0.3 * amplitude; // right_elbow: slight bend
    target_angles[18] = 0.2 * amplitude * cycle; // left_shoulder1: opposite to left leg
    target_angles[19] = 0.0;
    target_angles[20] = -0.3 * amplitude; // left_elbow

    let mut torques = vec![0.0f32; n];
    for i in 0..n.min(gains.kp.len()) {
        let angle_error = target_angles[i] - state.joint_angles[i];
        let vel_damping = state.joint_velocities[i];
        torques[i] = (gains.kp[i] * angle_error - gains.kd[i] * vel_damping) as f32;
        torques[i] = torques[i].clamp(-1.0, 1.0);
    }
    HumanoidCommand { torques }
}

/// Compute PD running baseline: deeper lean + larger stride + faster cycle.
///
/// Amplified version of walking baseline with:
/// - Steeper forward lean
/// - Larger hip swing amplitude
/// - Higher knee lift + foot-contact-aware stance/swing
/// - More aggressive arm swing
pub fn pd_running_baseline(
    state: &HumanoidState,
    gains: &HumanoidPdGains,
    phase: f64,
    target_speed: f64,
) -> HumanoidCommand {
    let n = state.num_actuators();
    let mut target_angles = vec![0.0f64; n];

    let amplitude = (target_speed / 3.0).clamp(0.0, 1.0);
    let cycle = (phase * 2.0 * std::f64::consts::PI).sin();
    let half_cycle = ((phase * 2.0 * std::f64::consts::PI) + std::f64::consts::FRAC_PI_2).sin();

    // Foot contact detection
    let contact_threshold = 0.05;
    let r_foot_z = state.extremities[8];
    let l_foot_z = state.extremities[11];
    let r_stance = r_foot_z < contact_threshold;
    let l_stance = l_foot_z < contact_threshold;

    // Contact-aware hip targets: stance → extend (pushoff), swing → flex (lift)
    // Reduce modulation during double support to keep sinusoidal dominant.
    let contact_blend = if r_stance && l_stance { 0.1 } else { 0.4 };
    let sin_blend = 1.0 - contact_blend;

    let r_contact_mod = if r_stance { 0.4 } else { -0.4 };
    let l_contact_mod = if l_stance { 0.4 } else { -0.4 };
    let r_hip_target = sin_blend * cycle + contact_blend * r_contact_mod;
    let l_hip_target = sin_blend * (-cycle) + contact_blend * l_contact_mod;

    // Knee: swing phase needs higher clearance for running
    let r_knee_swing = if r_stance { 0.0 } else { -0.5 };
    let l_knee_swing = if l_stance { 0.0 } else { -0.5 };

    // Deeper forward lean for running
    target_angles[0] = 0.25 * amplitude; // abdomen_y: steeper lean
    target_angles[1] = 0.0;
    target_angles[2] = 0.0;

    // Right leg: contact-aware larger stride
    target_angles[3] = 0.0;
    target_angles[4] = 0.0;
    target_angles[5] = 0.5 * amplitude * r_hip_target; // right_hip_y: contact-aware
    target_angles[6] = -0.7 * amplitude * half_cycle.max(0.0) + amplitude * r_knee_swing;
    target_angles[7] = -0.15 * amplitude * cycle;
    target_angles[8] = 0.08 * amplitude;

    // Left leg: contact-aware
    target_angles[9] = 0.0;
    target_angles[10] = 0.0;
    target_angles[11] = 0.5 * amplitude * l_hip_target;
    target_angles[12] = -0.7 * amplitude * (-half_cycle).max(0.0) + amplitude * l_knee_swing;
    target_angles[13] = 0.15 * amplitude * cycle;
    target_angles[14] = 0.08 * amplitude;

    // Arms: more aggressive swing
    target_angles[15] = -0.4 * amplitude * cycle;
    target_angles[16] = 0.0;
    target_angles[17] = -0.5 * amplitude;
    target_angles[18] = 0.4 * amplitude * cycle;
    target_angles[19] = 0.0;
    target_angles[20] = -0.5 * amplitude;

    let mut torques = vec![0.0f32; n];
    for i in 0..n.min(gains.kp.len()) {
        let angle_error = target_angles[i] - state.joint_angles[i];
        let vel_damping = state.joint_velocities[i];
        torques[i] = (gains.kp[i] * angle_error - gains.kd[i] * vel_damping) as f32;
        torques[i] = torques[i].clamp(-1.0, 1.0);
    }
    HumanoidCommand { torques }
}

/// Compute PD reaching baseline: standing posture + arm IK toward target object.
///
/// Drives shoulder/elbow toward the object position using simplified 2-link IK.
/// Available to every morphology containing the canonical shoulder/elbow chain.
pub fn pd_reaching_baseline(
    state: &HumanoidState,
    gains: &HumanoidPdGains,
    object_pos: [f64; 3],
    hand: crate::morphology::HandSide,
) -> HumanoidCommand {
    // Start with standing baseline (keep body upright)
    let mut cmd = pd_standing_baseline(state, gains);
    let n = state.num_actuators();

    // Shoulder/elbow indices for the reaching hand
    let (s1, s2, elbow) = match hand {
        crate::morphology::HandSide::Right => (15, 16, 17),
        crate::morphology::HandSide::Left => (18, 19, 20),
    };

    // Simplified analytical arm IK toward object
    // Compute desired shoulder flexion from object height/distance
    let shoulder_base_z = state.root_height + 0.15; // approximate shoulder height
    let dz = object_pos[2] - shoulder_base_z;
    let dx = object_pos[0]; // forward distance
    let reach_dist = (dx * dx + dz * dz).sqrt();
    let arm_length = 0.28 + 0.25; // upper_arm + forearm

    // Shoulder1 (flexion): angle to point arm toward object
    let shoulder_target = (dz / reach_dist.max(0.01)).asin().clamp(-1.5, 1.5);
    // Elbow: bend to match reach distance
    let elbow_target = if reach_dist < arm_length {
        -((arm_length - reach_dist) / arm_length * 1.5).clamp(0.0, 1.5)
    } else {
        0.0 // fully extended
    };

    // Lateral component for shoulder2
    let dy = match hand {
        crate::morphology::HandSide::Right => object_pos[1] + 0.17, // offset from right shoulder
        crate::morphology::HandSide::Left => object_pos[1] - 0.17,
    };
    let shoulder2_target = (dy / reach_dist.max(0.01)).asin().clamp(-1.5, 1.5);

    // Apply PD control to shoulder/elbow
    if s1 < n {
        let err = shoulder_target - state.joint_angles[s1];
        cmd.torques[s1] = (gains.kp[s1] * err - gains.kd[s1] * state.joint_velocities[s1]) as f32;
        cmd.torques[s1] = cmd.torques[s1].clamp(-1.0, 1.0);
    }
    if s2 < n {
        let err = shoulder2_target - state.joint_angles[s2];
        cmd.torques[s2] = (gains.kp[s2] * err - gains.kd[s2] * state.joint_velocities[s2]) as f32;
        cmd.torques[s2] = cmd.torques[s2].clamp(-1.0, 1.0);
    }
    if elbow < n {
        let err = elbow_target - state.joint_angles[elbow];
        cmd.torques[elbow] =
            (gains.kp[elbow] * err - gains.kd[elbow] * state.joint_velocities[elbow]) as f32;
        cmd.torques[elbow] = cmd.torques[elbow].clamp(-1.0, 1.0);
    }

    cmd
}

/// Compute PD grasping baseline: reaching + sinusoidal finger closure.
///
/// `grasp_phase` controls finger closure progression (0.0=open, 1.0=fully closed).
pub fn pd_grasping_baseline(
    state: &HumanoidState,
    gains: &HumanoidPdGains,
    object_pos: [f64; 3],
    hand: crate::morphology::HandSide,
    grasp_phase: f64,
) -> HumanoidCommand {
    let mut cmd = pd_reaching_baseline(state, gains, object_pos, hand);
    let n = state.num_actuators();

    // Determine hand joint range
    let hand_start = match hand {
        crate::morphology::HandSide::Right => 21,
        crate::morphology::HandSide::Left => 37,
    };
    let hand_end = hand_start + 16;

    if hand_end > n {
        return cmd; // Not enough joints (DMC21)
    }

    // Target flexion ramps with grasp_phase
    let target_flexion = grasp_phase * 1.2; // ~70° at full closure

    for i in hand_start..hand_end.min(n) {
        if i < gains.kp.len() {
            let err = target_flexion - state.joint_angles[i];
            cmd.torques[i] = (gains.kp[i] * err - gains.kd[i] * state.joint_velocities[i]) as f32;
            cmd.torques[i] = cmd.torques[i].clamp(-1.0, 1.0);
        }
    }

    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_humanoid_state_standing() {
        let state = HumanoidState::standing();
        assert!((state.root_height - 1.3).abs() < 1e-10);
        assert!((state.head_height - 1.4).abs() < 1e-10);
        assert!((state.uprightness() - 1.0).abs() < 1e-10);
        assert!(state.horizontal_speed() < 1e-10);
        assert!(state.angular_momentum() < 1e-10);
    }

    #[test]
    fn test_humanoid_state_to_channels() {
        let state = HumanoidState::standing();
        let channels = state.to_channels();
        assert_eq!(channels.len(), state.num_channels());
        assert_eq!(channels.len(), 72);
        // root_height at index 0
        assert!((channels[0] - 1.3).abs() < 1e-6);
        // quat w at index 1
        assert!((channels[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_state_schema_matches_morphology() {
        for morphology in [
            HumanoidMorphology::Dmc21,
            HumanoidMorphology::Dexterous53,
            HumanoidMorphology::WithNeckWrist,
            HumanoidMorphology::FullSpine,
        ] {
            let state = HumanoidState::standing_for(morphology);
            assert_eq!(state.num_channels(), morphology.num_observation_channels());
            assert_eq!(
                state.to_channels().len(),
                morphology.num_observation_channels()
            );
            assert_eq!(state.validate_for(morphology), Ok(()));
        }
    }

    #[test]
    fn test_state_validation_rejects_non_finite_values() {
        let mut state = HumanoidState::standing();
        state.joint_angles[0] = f64::NAN;
        assert_eq!(
            state.validate_for(HumanoidMorphology::Dmc21),
            Err(StateValidationError::NonFiniteValue)
        );
    }

    #[test]
    fn test_humanoid_command_zero() {
        let cmd = HumanoidCommand::zero();
        for &t in &cmd.torques {
            assert!(t.abs() < 1e-10);
        }
        assert!(cmd.control_effort() < 1e-10);
    }

    #[test]
    fn test_command_validation_rejects_non_finite_values() {
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[3] = f32::NAN;
        assert_eq!(
            cmd.validate_for(NUM_ACTUATORS, ActuationMode::NormalizedTorque),
            Err(CommandValidationError::NonFiniteValue { index: 3 })
        );
    }

    #[test]
    fn test_command_validation_rejects_wrong_morphology() {
        let cmd = HumanoidCommand::zero_for(20);
        assert_eq!(
            cmd.validate_for(NUM_ACTUATORS, ActuationMode::NormalizedTorque),
            Err(CommandValidationError::ActuatorCount {
                expected: NUM_ACTUATORS,
                actual: 20,
            })
        );
    }

    #[test]
    fn test_humanoid_command_from_raw() {
        let raw = vec![2.0f32; NUM_ACTUATORS];
        let cmd = HumanoidCommand::from_raw(&raw);
        for &t in &cmd.torques {
            assert!((t - 1.0).abs() < 1e-6, "Should clamp to 1.0");
        }
    }

    #[test]
    fn test_humanoid_command_clamped() {
        let mut cmd = HumanoidCommand::zero();
        cmd.torques[0] = 5.0;
        cmd.torques[1] = -5.0;
        let clamped = cmd.clamped();
        assert!((clamped.torques[0] - 1.0).abs() < 1e-6);
        assert!((clamped.torques[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_humanoid_command_to_ctrl() {
        let cmd = HumanoidCommand::zero();
        let ctrl = cmd.to_ctrl();
        assert_eq!(ctrl.len(), NUM_ACTUATORS);
    }

    #[test]
    fn test_humanoid_task_target_speed() {
        assert!((HumanoidTask::Stand.target_speed() - 0.0).abs() < 1e-10);
        assert!((HumanoidTask::Walk.target_speed() - 1.0).abs() < 1e-10);
        assert!((HumanoidTask::Run.target_speed() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_humanoid_config_defaults() {
        let config = HumanoidConfig::default();
        assert!((config.physics_dt() - 0.025).abs() < 1e-10);
        assert_eq!(config.cognitive_interval(), 4); // 40/10
        assert!((config.effective_target_speed() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn reach_allows_wrist_morphology_but_grasp_requires_fingers() {
        let reach = HumanoidConfig {
            morphology: HumanoidMorphology::WithNeckWrist,
            task: HumanoidTask::Reach,
            ..HumanoidConfig::default()
        };
        assert!(reach.validate().is_ok());

        let grasp = HumanoidConfig {
            morphology: HumanoidMorphology::WithNeckWrist,
            task: HumanoidTask::Grasp,
            ..HumanoidConfig::default()
        };
        assert_eq!(
            grasp.validate(),
            Err(HumanoidConfigError::DexterousTaskRequiresDexterousMorphology)
        );
    }

    #[test]
    fn test_humanoid_config_target_speed_override() {
        let config = HumanoidConfig {
            target_speed: Some(5.0),
            ..HumanoidConfig::default()
        };
        assert!((config.effective_target_speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pd_standing_baseline_at_standing() {
        let state = HumanoidState::standing();
        let gains = HumanoidPdGains::default();
        let cmd = pd_standing_baseline(&state, &gains);
        // At standing (all joints at zero): all torques should be ~0
        for &t in &cmd.torques {
            assert!(
                t.abs() < 0.01,
                "At standing, torques should be near zero: {t}"
            );
        }
    }

    #[test]
    fn test_pd_baseline_corrects_deviation() {
        let mut state = HumanoidState::standing();
        state.joint_angles[0] = 0.1; // Abdomen tilted
        let gains = HumanoidPdGains::default();
        let cmd = pd_standing_baseline(&state, &gains);
        // Should apply corrective torque (negative, to reduce positive angle)
        assert!(
            cmd.torques[0] < -0.01,
            "Should correct positive angle deviation: {}",
            cmd.torques[0]
        );
    }

    #[test]
    fn test_humanoid_state_serde_roundtrip() {
        let state = HumanoidState::standing();
        let json = serde_json::to_string(&state).unwrap();
        let restored: HumanoidState = serde_json::from_str(&json).unwrap();
        assert!((state.root_height - restored.root_height).abs() < 1e-10);
        assert!((state.head_height - restored.head_height).abs() < 1e-10);
    }

    #[test]
    fn test_humanoid_command_serde_roundtrip() {
        let cmd = HumanoidCommand::zero();
        let json = serde_json::to_string(&cmd).unwrap();
        let restored: HumanoidCommand = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd.torques.len(), restored.torques.len());
        for i in 0..cmd.torques.len() {
            assert!((cmd.torques[i] - restored.torques[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_from_mujoco() {
        let mut qpos = vec![0.0f64; 28];
        qpos[2] = 1.3; // root height
        qpos[3] = 1.0; // quat w
        qpos[7] = 0.1; // first joint angle

        let mut qvel = vec![0.0f64; 27];
        qvel[0] = 0.5; // root lin vel x
        qvel[6] = 0.2; // first joint vel

        let state = HumanoidState::from_mujoco(
            &qpos,
            &qvel,
            1.4,
            [0.0, 0.0, 1.0],
            &[0.0; 12],
            [0.5, 0.0, 0.0],
            1.0,
        );

        assert!((state.root_height - 1.3).abs() < 1e-10);
        assert!((state.root_quaternion[0] - 1.0).abs() < 1e-10);
        assert!((state.joint_angles[0] - 0.1).abs() < 1e-10);
        assert!((state.root_linear_velocity[0] - 0.5).abs() < 1e-10);
        assert!((state.joint_velocities[0] - 0.2).abs() < 1e-10);
        assert!((state.head_height - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_horizontal_speed() {
        let mut state = HumanoidState::standing();
        state.com_velocity = [3.0, 4.0, 0.0];
        assert!((state.horizontal_speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_joint_names_count() {
        assert_eq!(JOINT_NAMES.len(), NUM_ACTUATORS);
    }

    #[test]
    fn test_pd_walking_baseline_produces_forward_lean() {
        let state = HumanoidState::standing();
        let gains = HumanoidPdGains::default();
        let cmd = pd_walking_baseline(&state, &gains, 0.0, 1.0);
        // Abdomen_y (joint 0) should have positive torque for forward lean
        assert!(
            cmd.torques[0] > 0.01,
            "Walking should produce forward lean torque: {}",
            cmd.torques[0]
        );
    }

    #[test]
    fn test_pd_walking_baseline_cyclic() {
        let state = HumanoidState::standing();
        // Unit P-gain, zero D-gain: default kp=200 saturates BOTH phases at
        // the ±1 torque clamp (hip targets 0.009 vs 0.279 × 200 → 1 vs 1),
        // which made this test fail vacuously for a long time (documented
        // pre-existing in commit 1a5b724bbd). With kp=1 the torque IS the
        // target error, so cyclicity is actually observable.
        let n = state.num_actuators();
        let gains = HumanoidPdGains {
            kp: vec![1.0; n],
            kd: vec![0.0; n],
        };
        let cmd_0 = pd_walking_baseline(&state, &gains, 0.0, 1.0);
        let cmd_quarter = pd_walking_baseline(&state, &gains, 0.25, 1.0);

        // Phase 0 (sin=0) vs phase 0.25 (sin=1): hip_y targets differ
        assert!(
            (cmd_0.torques[5] - cmd_quarter.torques[5]).abs() > 0.01,
            "Gait should be cyclic: phase 0 vs 0.25 should differ: {} vs {}",
            cmd_0.torques[5],
            cmd_quarter.torques[5]
        );
        // At phase 0.25 (peak), legs should be in anti-phase
        assert!(
            (cmd_quarter.torques[5].signum() != cmd_quarter.torques[11].signum()),
            "Legs should be in anti-phase at peak: right={} left={}",
            cmd_quarter.torques[5],
            cmd_quarter.torques[11]
        );
    }

    #[test]
    fn test_pd_walking_baseline_zero_speed() {
        let state = HumanoidState::standing();
        let gains = HumanoidPdGains::default();
        let cmd = pd_walking_baseline(&state, &gains, 0.0, 0.0);
        // At zero speed, amplitude = 0, so all torques should be ~0
        for &t in &cmd.torques {
            assert!(
                t.abs() < 0.01,
                "Zero speed should give near-zero torques: {t}"
            );
        }
    }

    #[test]
    fn test_pd_running_baseline_stronger_than_walking() {
        // At a non-zero phase, running should command more joints to non-zero targets
        // than walking (bigger arm swing, deeper knee lift, steeper lean).
        // We test at a state near the walking target, so residual errors reveal the
        // difference in amplitude that would otherwise be hidden by clamping.
        let mut state = HumanoidState::standing();
        // Set joints to walking-phase-0.25 targets (approx)
        state.joint_angles[0] = 0.15; // abdomen lean (walk amplitude at speed=1)
        state.joint_angles[5] = 0.3; // right_hip_y
        state.joint_angles[17] = -0.3; // right elbow

        let gains = HumanoidPdGains::default();
        let walk = pd_walking_baseline(&state, &gains, 0.25, 1.0);
        let run = pd_running_baseline(&state, &gains, 0.25, 3.0);

        // Running targets are larger than walking targets, so when state is AT
        // walking targets, running has positive residual torque and walking has ~0.
        // Abdomen: run wants 0.25, state is at 0.15 → run pushes forward more
        assert!(
            run.torques[0] > walk.torques[0],
            "Running should push abdomen further: run={} walk={}",
            run.torques[0],
            walk.torques[0]
        );
    }

    #[test]
    fn test_dexterous53_standing() {
        use crate::morphology::HumanoidMorphology;
        let state = HumanoidState::standing_for(HumanoidMorphology::Dexterous53);
        assert_eq!(state.num_actuators(), 53);
        assert_eq!(state.joint_angles.len(), 53);
        assert_eq!(state.joint_velocities.len(), 53);
        let channels = state.to_channels();
        // 1+4+53+3+3+53+1+3+18+3 = 142
        assert_eq!(channels.len(), state.num_channels());
    }

    #[test]
    fn test_dexterous53_pd_baseline() {
        use crate::morphology::HumanoidMorphology;
        let state = HumanoidState::standing_for(HumanoidMorphology::Dexterous53);
        let gains = HumanoidPdGains::for_morphology(HumanoidMorphology::Dexterous53);
        let cmd = pd_standing_baseline(&state, &gains);
        assert_eq!(cmd.torques.len(), 53);
        // At standing, all torques should be ~0
        for (i, &t) in cmd.torques.iter().enumerate() {
            assert!(t.abs() < 0.01, "Joint {i} should be near zero: {t}");
        }
    }

    #[test]
    fn test_command_zero_for_morphology() {
        let cmd = HumanoidCommand::zero_for(53);
        assert_eq!(cmd.torques.len(), 53);
        assert!(cmd.torques.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn standing_pose_is_inside_dmc_joint_limits() {
        let state = HumanoidState::standing();
        for (angle, [low, high]) in state
            .joint_angles
            .iter()
            .zip(HumanoidMorphology::Dmc21.joint_limits())
        {
            assert!((low..=high).contains(angle));
        }
    }

    #[test]
    fn knee_flexion_sign_matches_bundled_mujoco_model() {
        let limits = HumanoidMorphology::Dmc21.joint_limits();
        assert!(limits[6][0] < 0.0 && limits[6][1] >= 0.0);
        assert!(limits[12][0] < 0.0 && limits[12][1] >= 0.0);
    }
}
