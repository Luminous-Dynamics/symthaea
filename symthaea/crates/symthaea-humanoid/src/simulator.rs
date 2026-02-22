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
struct HumanoidBodyModel {
    /// Segment lengths in meters: torso, thigh, shin, foot, upper_arm, forearm, head.
    segment_lengths: [f64; 7],
    /// Segment masses in kg.
    segment_masses: [f64; 7],
    /// Total body mass in kg.
    total_mass: f64,
    /// Per-joint effective inertia (mass * length^2), indexed by actuator.
    joint_inertias: [f64; NUM_ACTUATORS],
    /// Per-joint damping coefficients.
    joint_damping: [f64; NUM_ACTUATORS],
    /// Per-joint torque scaling (normalized command -> Nm).
    joint_torque_scale: [f64; NUM_ACTUATORS],
}

impl HumanoidBodyModel {
    fn new() -> Self {
        let segment_lengths = [0.30, 0.40, 0.35, 0.12, 0.28, 0.25, 0.18]; // m
        let segment_masses = [17.0, 8.0, 4.0, 1.5, 2.5, 1.5, 5.0]; // kg (per-segment, bilateral doubled in COM)
        let total_mass = 70.0;

        // Per-joint effective inertia: I_eff ~ m_segment * L_segment^2
        let joint_inertias = [
            // Abdomen y/z/x (torso segment)
            0.30, 0.30, 0.30,
            // Right hip x/z/y (thigh segment)
            0.50, 0.50, 0.50,
            // Right knee (shin segment)
            0.20,
            // Right ankle x/y (foot segment)
            0.05, 0.05,
            // Left hip x/z/y
            0.50, 0.50, 0.50,
            // Left knee
            0.20,
            // Left ankle x/y
            0.05, 0.05,
            // Right shoulder1/2 (upper arm)
            0.10, 0.10,
            // Right elbow (forearm)
            0.03,
            // Left shoulder1/2
            0.10, 0.10,
            // Left elbow
            0.03,
        ];

        let joint_damping = [
            // Abdomen
            6.0, 6.0, 6.0,
            // Right hip
            8.0, 8.0, 8.0,
            // Right knee
            5.0,
            // Right ankle
            3.0, 3.0,
            // Left hip
            8.0, 8.0, 8.0,
            // Left knee
            5.0,
            // Left ankle
            3.0, 3.0,
            // Right arm
            2.0, 2.0, 1.5,
            // Left arm
            2.0, 2.0, 1.5,
        ];

        // Torque scale matches MJCF gear ratios
        let joint_torque_scale = [
            // Abdomen (gear=100)
            100.0, 100.0, 100.0,
            // Right hip (100, 100, 300), knee (200)
            100.0, 100.0, 300.0, 200.0,
            // Right ankle (100, 100)
            100.0, 100.0,
            // Left hip, knee
            100.0, 100.0, 300.0, 200.0,
            // Left ankle
            100.0, 100.0,
            // Right arm (25)
            25.0, 25.0, 25.0,
            // Left arm (25)
            25.0, 25.0, 25.0,
        ];

        Self {
            segment_lengths,
            segment_masses,
            total_mass,
            joint_inertias,
            joint_damping,
            joint_torque_scale,
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

/// Simple physics model for pure-Rust testing (no MuJoCo required).
///
/// Improved model (~50% MuJoCo fidelity) with:
/// - Per-joint inertia, damping, and torque scaling from dm_control
/// - Anatomically correct joint limits from MJCF
/// - Forward kinematics for leg chains (contact-aware root height)
/// - Proper quaternion integration from angular velocity
/// - Weighted COM from segment positions
///
/// Main remaining gap vs MuJoCo: contact/friction dynamics, collision detection.
pub struct SimpleHumanoidSimulator {
    state: HumanoidState,
    external_force: [f64; 3],
    body: HumanoidBodyModel,
}

impl SimpleHumanoidSimulator {
    /// Create a new simulator at default standing pose.
    pub fn new() -> Self {
        Self {
            state: HumanoidState::standing(),
            external_force: [0.0; 3],
            body: HumanoidBodyModel::new(),
        }
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
        let g = 9.81;

        // 1. Joint dynamics: per-joint inertia, damping, torque scaling
        for i in 0..NUM_ACTUATORS {
            let torque = cmd.torques[i] as f64 * self.body.joint_torque_scale[i];
            let damping = self.body.joint_damping[i];
            let inertia = self.body.joint_inertias[i];

            let accel =
                (torque - damping * self.state.joint_velocities[i]) / inertia;
            self.state.joint_velocities[i] += accel * dt;
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;

            // Per-joint anatomical limits with velocity clamping at stops
            let [lo, hi] = JOINT_LIMITS[i];
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
        self.state.root_linear_velocity[2] +=
            (lean_torque / self.body.total_mass + self.external_force[2] / self.body.total_mass)
                * dt;
        self.state.root_height += self.state.root_linear_velocity[2] * dt;

        // Blend dynamic height with kinematic (70% kinematic, 30% dynamic)
        self.state.root_height = 0.3 * self.state.root_height + 0.7 * kinematic_height;

        // Ground contact constraint
        if self.state.root_height < 0.2 {
            self.state.root_height = 0.2;
            self.state.root_linear_velocity[2] = self.state.root_linear_velocity[2].max(0.0);
        }

        // 4. Head height from torso chain
        self.state.head_height = self.state.root_height
            + self.body.segment_lengths[SEG_TORSO] * 0.5 * uprightness
            + self.body.segment_lengths[SEG_HEAD] * uprightness;

        // 5. Horizontal dynamics
        self.state.root_linear_velocity[0] +=
            self.external_force[0] / self.body.total_mass * dt;
        self.state.root_linear_velocity[1] +=
            self.external_force[1] / self.body.total_mass * dt;

        let drag = 0.5;
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

        self.state.timestamp += dt;
        self.external_force = [0.0; 3];
    }

    fn state(&self) -> &HumanoidState {
        &self.state
    }

    fn reset(&mut self) {
        self.state = HumanoidState::standing();
        self.external_force = [0.0; 3];
    }

    fn reset_with_perturbation(&mut self, perturbation: f64, seed: u64) {
        self.state = HumanoidState::standing();
        self.external_force = [0.0; 3];

        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        // Perturb joint angles within anatomical limits
        for i in 0..NUM_ACTUATORS {
            let [lo, hi] = JOINT_LIMITS[i];
            let range = hi - lo;
            self.state.joint_angles[i] += perturbation * next_f64() * range * 0.03;
            self.state.joint_angles[i] = self.state.joint_angles[i].clamp(lo, hi);
        }

        // Perturb root height slightly
        self.state.root_height += perturbation * next_f64() * 0.02;
        self.state.head_height = self.state.root_height
            + self.body.segment_lengths[SEG_TORSO] * 0.5
            + self.body.segment_lengths[SEG_HEAD];
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
#[cfg(feature = "mujoco")]
pub struct MuJoCoHumanoidSimulator {
    model: std::sync::Arc<mujoco_rs::MjModel>,
    data: mujoco_rs::MjData<std::sync::Arc<mujoco_rs::MjModel>>,
    state: HumanoidState,
    body_ids: HumanoidBodyIds,
    external_force: [f64; 3],
}

/// Cached body name -> ID lookups for efficient state extraction.
#[cfg(feature = "mujoco")]
struct HumanoidBodyIds {
    torso: i32,
    head: i32,
    right_hand: i32,
    left_hand: i32,
    right_foot: i32,
    left_foot: i32,
}

#[cfg(feature = "mujoco")]
impl MuJoCoHumanoidSimulator {
    /// Create a new simulator from an MJCF XML string.
    pub fn from_xml(xml: &str) -> anyhow::Result<Self> {
        let model = std::sync::Arc::new(mujoco_rs::MjModel::from_xml(xml)?);
        let data = mujoco_rs::MjData::new(model.clone());

        let body_ids = HumanoidBodyIds {
            torso: model.body_name2id("torso").unwrap_or(1),
            head: model.body_name2id("head").unwrap_or(2),
            right_hand: model.body_name2id("right_hand").unwrap_or(3),
            left_hand: model.body_name2id("left_hand").unwrap_or(4),
            right_foot: model.body_name2id("right_foot").unwrap_or(5),
            left_foot: model.body_name2id("left_foot").unwrap_or(6),
        };

        let state = HumanoidState::standing();

        Ok(Self {
            model,
            data,
            state,
            body_ids,
            external_force: [0.0; 3],
        })
    }

    /// Create from an MJCF file path.
    pub fn from_path(path: &str) -> anyhow::Result<Self> {
        let xml = std::fs::read_to_string(path)?;
        Self::from_xml(&xml)
    }

    /// Extract the current humanoid state from MuJoCo data.
    fn extract_state(&mut self) {
        let qpos = self.data.qpos();
        let qvel = self.data.qvel();

        let head_height = self.data.xpos(self.body_ids.head as usize)[2];

        let torso_xmat = self.data.xmat(self.body_ids.torso as usize);
        let torso_vertical = [torso_xmat[6], torso_xmat[7], torso_xmat[8]];

        let rh = self.data.xpos(self.body_ids.right_hand as usize);
        let lh = self.data.xpos(self.body_ids.left_hand as usize);
        let rf = self.data.xpos(self.body_ids.right_foot as usize);
        let lf = self.data.xpos(self.body_ids.left_foot as usize);
        let extremities = [
            rh[0], rh[1], rh[2], lh[0], lh[1], lh[2], rf[0], rf[1], rf[2], lf[0], lf[1], lf[2],
        ];

        let com_velocity = [qvel[0], qvel[1], qvel[2]];

        self.state = HumanoidState::from_mujoco(
            qpos,
            qvel,
            head_height,
            torso_vertical,
            extremities,
            com_velocity,
            self.data.time(),
        );
    }

    /// Get the body mass of the torso.
    pub fn body_mass(&self) -> f64 {
        self.model.body_mass(self.body_ids.torso as usize)
    }
}

#[cfg(feature = "mujoco")]
impl HumanoidPhysicsSimulator for MuJoCoHumanoidSimulator {
    fn step(&mut self, cmd: &HumanoidCommand, _dt: f64) {
        let ctrl = self.data.ctrl_mut();
        for i in 0..NUM_ACTUATORS.min(ctrl.len()) {
            ctrl[i] = cmd.torques[i] as f64;
        }

        if self.external_force.iter().any(|f| f.abs() > 1e-10) {
            let xfrc = self.data.xfrc_applied_mut();
            let torso_idx = self.body_ids.torso as usize;
            xfrc[torso_idx * 6] += self.external_force[0];
            xfrc[torso_idx * 6 + 1] += self.external_force[1];
            xfrc[torso_idx * 6 + 2] += self.external_force[2];
        }

        self.data.step();

        if self.external_force.iter().any(|f| f.abs() > 1e-10) {
            let xfrc = self.data.xfrc_applied_mut();
            let torso_idx = self.body_ids.torso as usize;
            xfrc[torso_idx * 6] = 0.0;
            xfrc[torso_idx * 6 + 1] = 0.0;
            xfrc[torso_idx * 6 + 2] = 0.0;
            self.external_force = [0.0; 3];
        }

        self.extract_state();
    }

    fn state(&self) -> &HumanoidState {
        &self.state
    }

    fn reset(&mut self) {
        self.data.reset();
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

        let qpos = self.data.qpos_mut();
        for i in 7..28 {
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
            torques: [0.5; NUM_ACTUATORS],
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
}
