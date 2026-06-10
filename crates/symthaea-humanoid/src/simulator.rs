// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics simulator trait and implementations for humanoid.

use crate::types::{HumanoidCommand, HumanoidState, NUM_ACTUATORS};

/// Trait for humanoid physics simulation backends.
pub trait HumanoidPhysicsSimulator {
    fn step(&mut self, cmd: &HumanoidCommand, dt: f64);
    fn state(&self) -> &HumanoidState;
    fn reset(&mut self);
    fn reset_with_perturbation(&mut self, perturbation: f64, seed: u64);
    fn apply_external_force(&mut self, force: [f64; 3]);
}

const SEG_TORSO: usize = 0;
const SEG_THIGH: usize = 1;
const SEG_SHIN: usize = 2;
const SEG_FOOT: usize = 3;
const SEG_UPPER_ARM: usize = 4;
const SEG_FOREARM: usize = 5;
const SEG_HEAD: usize = 6;

#[derive(Clone)]
struct HumanoidBodyModel {
    segment_lengths: [f64; 7],
    segment_masses: [f64; 7],
    total_mass: f64,
    joint_inertias: Vec<f64>,
    joint_damping: Vec<f64>,
    joint_torque_scale: Vec<f64>,
}

impl HumanoidBodyModel {
    fn new() -> Self {
        let segment_lengths = [0.30, 0.40, 0.35, 0.12, 0.28, 0.25, 0.18];
        let segment_masses = [17.0, 8.0, 4.0, 1.5, 2.5, 1.5, 5.0];
        let total_mass = 70.0;

        let joint_inertias = [
            0.30, 0.30, 0.30, 0.50, 0.50, 0.50, 0.20, 0.05, 0.05, 0.50, 0.50, 0.50, 0.20, 0.05,
            0.05, 0.10, 0.10, 0.06, 0.10, 0.10, 0.06,
        ];

        let joint_damping = [
            6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 5.0, 3.0, 3.0, 8.0, 8.0, 8.0, 5.0, 3.0, 3.0, 2.0, 2.0,
            1.5, 2.0, 2.0, 1.5,
        ];

        let joint_torque_scale = [
            100.0, 100.0, 100.0, 100.0, 100.0, 300.0, 200.0, 100.0, 100.0, 100.0, 100.0, 300.0,
            200.0, 100.0, 100.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0,
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

    fn for_morphology(morphology: crate::morphology::HumanoidMorphology) -> Self {
        let base = Self::new();
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

const JOINT_LIMITS: [[f64; 2]; NUM_ACTUATORS] = [
    [-1.31, 0.52],
    [-0.79, 0.79],
    [-0.61, 0.61],
    [-0.44, 0.09],
    [-1.05, 0.61],
    [-1.92, 0.35],
    [-2.79, 0.03],
    [-0.87, 0.87],
    [-0.87, 0.87],
    [-0.44, 0.09],
    [-1.05, 0.61],
    [-1.92, 0.35],
    [-2.79, 0.03],
    [-0.87, 0.87],
    [-0.87, 0.87],
    [-1.48, 1.05],
    [-1.48, 1.05],
    [-1.57, 0.87],
    [-1.05, 1.48],
    [-1.05, 1.48],
    [-1.57, 0.87],
];

struct SensoryJitterBuffer {
    buffer: Vec<HumanoidState>,
    write_idx: usize,
    delay_ticks: usize,
    max_delay: usize,
    enabled: bool,
}

impl SensoryJitterBuffer {
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

    fn push(&mut self, state: &HumanoidState) {
        self.buffer[self.write_idx] = state.clone();
        self.write_idx = (self.write_idx + 1) % self.buffer.len();
    }

    fn delayed_state(&self) -> &HumanoidState {
        let cap = self.buffer.len();
        let read_idx = (self.write_idx + cap - self.delay_ticks - 1) % cap;
        &self.buffer[read_idx]
    }

    fn reset(&mut self, standing: &HumanoidState, seed: u64) {
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

struct ActuatorNoiseModel {
    command_buffer: Vec<HumanoidCommand>,
    write_idx: usize,
    delay_ticks: usize,
    max_delay: usize,
    noise_std: f64,
    rng_state: u64,
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

    fn apply(&mut self, cmd: &HumanoidCommand) -> HumanoidCommand {
        self.command_buffer[self.write_idx] = cmd.clone();
        self.write_idx = (self.write_idx + 1) % self.command_buffer.len();

        let cap = self.command_buffer.len();
        let read_idx = (self.write_idx + cap - self.delay_ticks - 1) % cap;
        let mut delayed = self.command_buffer[read_idx].clone();

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

struct DomainRandomization {
    enabled: bool,
    mass_range: f64,
    length_range: f64,
    damping_range: f64,
}

const JOINT_SEGMENT_MAP: [usize; NUM_ACTUATORS] = [
    SEG_TORSO,
    SEG_TORSO,
    SEG_TORSO,
    SEG_THIGH,
    SEG_THIGH,
    SEG_THIGH,
    SEG_SHIN,
    SEG_FOOT,
    SEG_FOOT,
    SEG_THIGH,
    SEG_THIGH,
    SEG_THIGH,
    SEG_SHIN,
    SEG_FOOT,
    SEG_FOOT,
    SEG_UPPER_ARM,
    SEG_UPPER_ARM,
    SEG_FOREARM,
    SEG_UPPER_ARM,
    SEG_UPPER_ARM,
    SEG_FOREARM,
];

struct ObservationNoiseModel {
    noise_std: f64,
    rng_state: u64,
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

    fn reset(&mut self, seed: u64) {
        self.rng_state = seed;
    }

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

struct GroundContactModel {
    stiffness: f64,
    damping: f64,
    friction: f64,
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

    fn apply(&self, foot_z: f64, foot_vz: f64, horizontal_vel: [f64; 2]) -> [f64; 3] {
        if foot_z >= 0.0 {
            return [0.0, 0.0, 0.0];
        }
        let fz = (-self.stiffness * foot_z - self.damping * foot_vz).max(0.0);
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

struct TerrainModel {
    slope_x: f64,
    slope_y: f64,
    compliance: f64,
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

    fn randomize(&mut self, seed: u64) {
        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as f64 / u64::MAX as f64
        };
        self.slope_x = (next_f64() * 2.0 - 1.0) * 0.05;
        self.slope_y = (next_f64() * 2.0 - 1.0) * 0.03;
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

    fn randomize(&self, baseline: &HumanoidBodyModel, seed: u64) -> HumanoidBodyModel {
        let mut body = baseline.clone();
        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        let mut mass_scales = [1.0f64; 7];
        let mut length_scales = [1.0f64; 7];
        for i in 0..7 {
            mass_scales[i] = 1.0 + next_f64() * self.mass_range;
            length_scales[i] = 1.0 + next_f64() * self.length_range;
            body.segment_masses[i] = baseline.segment_masses[i] * mass_scales[i];
            body.segment_lengths[i] = baseline.segment_lengths[i] * length_scales[i];
        }

        let baseline_segment_total: f64 = baseline.segment_masses.iter().sum();
        body.total_mass = body.segment_masses.iter().sum::<f64>()
            + (baseline.total_mass - baseline_segment_total);

        let n = body.joint_inertias.len().min(JOINT_SEGMENT_MAP.len());
        for i in 0..n {
            let seg = JOINT_SEGMENT_MAP[i];
            body.joint_inertias[i] =
                baseline.joint_inertias[i] * mass_scales[seg] * length_scales[seg].powi(2);
        }

        for i in 0..body.joint_damping.len() {
            body.joint_damping[i] =
                baseline.joint_damping[i] * (1.0 + next_f64() * self.damping_range);
        }
        body
    }
}

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
    morphology: crate::morphology::HumanoidMorphology,
    joint_limits: Vec<[f64; 2]>,
    gravity_scale: f64,
}

impl SimpleHumanoidSimulator {
    pub fn new() -> Self {
        Self::new_for(crate::morphology::HumanoidMorphology::Dmc21)
    }

    pub fn new_for(morphology: crate::morphology::HumanoidMorphology) -> Self {
        let body = if morphology == crate::morphology::HumanoidMorphology::Dmc21 {
            HumanoidBodyModel::new()
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

    pub fn with_domain_randomization(mut self, enabled: bool) -> Self {
        self.domain_rand.enabled = enabled;
        self
    }
    pub fn with_actuator_noise(mut self, noise_std: f64) -> Self {
        self.actuator_noise.noise_std = noise_std;
        self.actuator_noise.enabled = noise_std > 0.0;
        self
    }
    pub fn with_observation_noise(mut self, noise_std: f64) -> Self {
        self.observation_noise.noise_std = noise_std;
        self.observation_noise.enabled = noise_std > 0.0;
        self
    }
    pub fn with_ground_contact(mut self, enabled: bool) -> Self {
        self.ground_contact.enabled = enabled;
        self
    }
    pub fn with_terrain_variation(mut self, enabled: bool) -> Self {
        self.terrain.enabled = enabled;
        self
    }
    pub fn with_noise_scale(mut self, scale: f64) -> Self {
        self.actuator_noise.noise_std *= scale;
        self.domain_rand.mass_range *= scale;
        self.domain_rand.length_range *= scale;
        self.domain_rand.damping_range *= scale;
        self.observation_noise.noise_std *= scale;
        self
    }
    pub fn with_gravity(mut self, scale: f64) -> Self {
        self.gravity_scale = scale;
        self
    }

    fn leg_vertical_length(&self, hip_y: f64, knee: f64) -> f64 {
        let thigh_len = self.body.segment_lengths[SEG_THIGH];
        let shin_len = self.body.segment_lengths[SEG_SHIN];
        let foot_len = self.body.segment_lengths[SEG_FOOT];
        (thigh_len * hip_y.cos() + shin_len * (hip_y + knee).cos() + foot_len).max(0.05)
    }

    fn compute_com_z(&self) -> f64 {
        let body = &self.body;
        let root_h = self.state.root_height;
        let torso_half = body.segment_lengths[SEG_TORSO] * 0.5;
        let torso_z = root_h;
        let head_z = root_h + torso_half + body.segment_lengths[SEG_HEAD] * 0.5;

        let hip_z = root_h - torso_half;
        let r_hip_y = if self.state.joint_angles.len() > 5 {
            self.state.joint_angles[5]
        } else {
            0.0
        };
        let r_knee = if self.state.joint_angles.len() > 6 {
            self.state.joint_angles[6]
        } else {
            0.0
        };
        let r_thigh_z = hip_z - body.segment_lengths[SEG_THIGH] * 0.5 * r_hip_y.cos();
        let r_shin_z = hip_z
            - body.segment_lengths[SEG_THIGH] * r_hip_y.cos()
            - body.segment_lengths[SEG_SHIN] * 0.5 * (r_hip_y + r_knee).cos();

        let l_hip_y = if self.state.joint_angles.len() > 11 {
            self.state.joint_angles[11]
        } else {
            0.0
        };
        let l_knee = if self.state.joint_angles.len() > 12 {
            self.state.joint_angles[12]
        } else {
            0.0
        };
        let l_thigh_z = hip_z - body.segment_lengths[SEG_THIGH] * 0.5 * l_hip_y.cos();
        let l_shin_z = hip_z
            - body.segment_lengths[SEG_THIGH] * l_hip_y.cos()
            - body.segment_lengths[SEG_SHIN] * 0.5 * (l_hip_y + l_knee).cos();

        let arm_z = root_h + torso_half * 0.3;

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

        for i in 0..cmd.torques.len().min(self.body.joint_torque_scale.len()) {
            let torque = cmd.torques[i] as f64 * self.body.joint_torque_scale[i];
            let base_damping = self.body.joint_damping[i];

            // Apply non-linear viscoelastic joint stiffening to high-DOF segments (spine/hands)
            // to automatically absorb high-frequency resonant velocity whips.
            let damping = if i >= 21 {
                base_damping + 5.0 * self.state.joint_velocities[i].abs()
            } else {
                base_damping
            };
            let inertia = self.body.joint_inertias[i];

            let accel = (torque - damping * self.state.joint_velocities[i]) / inertia;
            self.state.joint_velocities[i] += accel * dt;
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;

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

        let abd_y = if self.state.joint_angles.len() > 0 {
            self.state.joint_angles[0]
        } else {
            0.0
        };
        let abd_z = if self.state.joint_angles.len() > 1 {
            self.state.joint_angles[1]
        } else {
            0.0
        };
        let abd_x = if self.state.joint_angles.len() > 2 {
            self.state.joint_angles[2]
        } else {
            0.0
        };

        let tilt_sagittal = abd_y.sin();
        let tilt_coronal = abd_x.sin();
        let tilt_mag = (tilt_sagittal * tilt_sagittal + tilt_coronal * tilt_coronal).sqrt();
        let uprightness = (1.0 - tilt_mag).clamp(0.0, 1.0);

        self.state.torso_vertical[0] = tilt_coronal;
        self.state.torso_vertical[1] = abd_z.sin() * 0.3;
        self.state.torso_vertical[2] = uprightness;

        let r_hip_y = if self.state.joint_angles.len() > 5 {
            self.state.joint_angles[5]
        } else {
            0.0
        };
        let r_knee = if self.state.joint_angles.len() > 6 {
            self.state.joint_angles[6]
        } else {
            0.0
        };
        let l_hip_y = if self.state.joint_angles.len() > 11 {
            self.state.joint_angles[11]
        } else {
            0.0
        };
        let l_knee = if self.state.joint_angles.len() > 12 {
            self.state.joint_angles[12]
        } else {
            0.0
        };

        let right_leg_len = self.leg_vertical_length(r_hip_y, r_knee);
        let left_leg_len = self.leg_vertical_length(l_hip_y, l_knee);

        let kinematic_height = right_leg_len.max(left_leg_len) * uprightness.max(0.2)
            + self.body.segment_lengths[SEG_TORSO] * 0.5;

        let lean_torque = -g * self.body.total_mass * tilt_mag * 0.15;
        self.state.root_linear_velocity[2] += (lean_torque / self.body.total_mass
            + self.external_force[2] / self.body.total_mass)
            * dt;
        self.state.root_height += self.state.root_linear_velocity[2] * dt;
        self.state.root_height = 0.3 * self.state.root_height + 0.7 * kinematic_height;

        if self.ground_contact.enabled {
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

            let r_forces = contact.apply(r_foot_z_raw, self.state.root_linear_velocity[2], h_vel);
            self.state.root_linear_velocity[0] += r_forces[0] / self.body.total_mass * dt;
            self.state.root_linear_velocity[1] += r_forces[1] / self.body.total_mass * dt;
            self.state.root_linear_velocity[2] += r_forces[2] / self.body.total_mass * dt;

            let l_forces = contact.apply(l_foot_z_raw, self.state.root_linear_velocity[2], h_vel);
            self.state.root_linear_velocity[0] += l_forces[0] / self.body.total_mass * dt;
            self.state.root_linear_velocity[1] += l_forces[1] / self.body.total_mass * dt;
            self.state.root_linear_velocity[2] += l_forces[2] / self.body.total_mass * dt;

            if self.state.root_height < 0.15 {
                self.state.root_height = 0.15;
                self.state.root_linear_velocity[2] = self.state.root_linear_velocity[2].max(0.0);
            }
        } else {
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

        self.state.head_height = self.state.root_height
            + self.body.segment_lengths[SEG_TORSO] * 0.5 * uprightness
            + self.body.segment_lengths[SEG_HEAD] * uprightness;

        let lean_accel_scale = if uprightness > 0.3 {
            g * 0.4 * (uprightness - 0.3).min(0.7) / 0.7
        } else {
            0.0
        };
        let mut forward_accel = tilt_sagittal * lean_accel_scale;
        let mut lateral_accel = tilt_coronal * lean_accel_scale;

        if self.terrain.enabled {
            forward_accel += g * self.terrain.slope_x.sin();
            lateral_accel += g * self.terrain.slope_y.sin();
        }

        let r_foot_z = (self.state.root_height - right_leg_len).max(0.0);
        let l_foot_z = (self.state.root_height - left_leg_len).max(0.0);
        let ground_threshold = 0.05;
        let pushoff_gain = 6.0;
        let mut pushoff_accel = 0.0;

        if r_foot_z < ground_threshold && uprightness > 0.4 && self.state.joint_velocities.len() > 5
        {
            let r_hip_vel = self.state.joint_velocities[5];
            if r_hip_vel > 0.0 {
                pushoff_accel += r_hip_vel * pushoff_gain;
            }
        }
        if l_foot_z < ground_threshold
            && uprightness > 0.4
            && self.state.joint_velocities.len() > 11
        {
            let l_hip_vel = self.state.joint_velocities[11];
            if l_hip_vel > 0.0 {
                pushoff_accel += l_hip_vel * pushoff_gain;
            }
        }

        let ankle_pushoff_gain = 2.0;
        if r_foot_z < ground_threshold
            && self.state.joint_velocities.len() > 8
            && self.state.joint_velocities[8] > 0.0
            && uprightness > 0.4
        {
            pushoff_accel += self.state.joint_velocities[8] * ankle_pushoff_gain;
        }
        if l_foot_z < ground_threshold
            && self.state.joint_velocities.len() > 14
            && self.state.joint_velocities[14] > 0.0
            && uprightness > 0.4
        {
            pushoff_accel += self.state.joint_velocities[14] * ankle_pushoff_gain;
        }

        let knee_ext_gain = 1.5;
        if r_foot_z < ground_threshold
            && self.state.joint_velocities.len() > 6
            && self.state.joint_velocities[6] > 0.0
            && uprightness > 0.4
        {
            pushoff_accel += self.state.joint_velocities[6] * knee_ext_gain;
        }
        if l_foot_z < ground_threshold
            && self.state.joint_velocities.len() > 12
            && self.state.joint_velocities[12] > 0.0
            && uprightness > 0.4
        {
            pushoff_accel += self.state.joint_velocities[12] * knee_ext_gain;
        }

        self.state.root_linear_velocity[0] +=
            (forward_accel + pushoff_accel + self.external_force[0] / self.body.total_mass) * dt;
        self.state.root_linear_velocity[1] +=
            (lateral_accel + self.external_force[1] / self.body.total_mass) * dt;

        let speed = (self.state.root_linear_velocity[0].powi(2)
            + self.state.root_linear_velocity[1].powi(2))
        .sqrt();
        let drag = 1.0 - 0.7 * (1.0 - (-speed * 0.5).exp());
        self.state.root_linear_velocity[0] *= (1.0 - drag * dt).max(0.0);
        self.state.root_linear_velocity[1] *= (1.0 - drag * dt).max(0.0);

        let com_z = self.compute_com_z();
        let prev_com_vz = self.state.com_velocity[2];
        self.state.com_velocity[0] = self.state.root_linear_velocity[0];
        self.state.com_velocity[1] = self.state.root_linear_velocity[1];
        self.state.com_velocity[2] =
            0.7 * prev_com_vz + 0.3 * (com_z - self.state.root_height) / dt.max(1e-6);

        if self.state.joint_velocities.len() > 2 {
            self.state.root_angular_velocity[0] = 0.3 * self.state.joint_velocities[2];
            self.state.root_angular_velocity[1] = 0.3 * self.state.joint_velocities[0];
            self.state.root_angular_velocity[2] = 0.3 * self.state.joint_velocities[1];
        }
        for av in &mut self.state.root_angular_velocity {
            *av *= (1.0 - 2.0 * dt).max(0.0);
        }

        self.state.root_quaternion = integrate_quaternion(
            self.state.root_quaternion,
            self.state.root_angular_velocity,
            dt,
        );

        let root_h = self.state.root_height;
        let torso_half = self.body.segment_lengths[SEG_TORSO] * 0.5;

        let arm_base_z = root_h + torso_half * 0.3;
        let r_shoulder = if self.state.joint_angles.len() > 15 {
            self.state.joint_angles[15]
        } else {
            0.0
        };
        let r_elbow = if self.state.joint_angles.len() > 17 {
            self.state.joint_angles[17]
        } else {
            0.0
        };
        let arm_reach = self.body.segment_lengths[SEG_UPPER_ARM] * r_shoulder.sin().abs()
            + self.body.segment_lengths[SEG_FOREARM] * (r_shoulder + r_elbow).sin().abs();
        self.state.extremities[0] = arm_reach * 0.3;
        self.state.extremities[1] = -0.17;
        self.state.extremities[2] = arm_base_z - arm_reach * 0.5;

        let l_shoulder = if self.state.joint_angles.len() > 18 {
            self.state.joint_angles[18]
        } else {
            0.0
        };
        let l_elbow = if self.state.joint_angles.len() > 20 {
            self.state.joint_angles[20]
        } else {
            0.0
        };
        let l_arm_reach = self.body.segment_lengths[SEG_UPPER_ARM] * l_shoulder.sin().abs()
            + self.body.segment_lengths[SEG_FOREARM] * (l_shoulder + l_elbow).sin().abs();
        self.state.extremities[3] = l_arm_reach * 0.3;
        self.state.extremities[4] = 0.17;
        self.state.extremities[5] = arm_base_z - l_arm_reach * 0.5;

        let r_foot_z = (root_h - right_leg_len).max(0.0);
        self.state.extremities[6] = r_hip_y.sin() * 0.1;
        self.state.extremities[7] = -0.1;
        self.state.extremities[8] = r_foot_z;

        let l_foot_z = (root_h - left_leg_len).max(0.0);
        self.state.extremities[9] = l_hip_y.sin() * 0.1;
        self.state.extremities[10] = 0.1;
        self.state.extremities[11] = l_foot_z;

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

            if self.state.joint_angles.len() >= 37 {
                let r_centroid = crate::morphology::compute_hand_centroid(
                    r_hand_base,
                    &self.state.joint_angles[21..37],
                    [1.0, 0.0, 0.0],
                );
                self.state.extremities[12] = r_centroid[0];
                self.state.extremities[13] = r_centroid[1];
                self.state.extremities[14] = r_centroid[2];
            }
            if self.state.joint_angles.len() >= 53 {
                let l_centroid = crate::morphology::compute_hand_centroid(
                    l_hand_base,
                    &self.state.joint_angles[37..53],
                    [1.0, 0.0, 0.0],
                );
                self.state.extremities[15] = l_centroid[0];
                self.state.extremities[16] = l_centroid[1];
                self.state.extremities[17] = l_centroid[2];
            }
        }

        self.state.timestamp += dt;
        self.external_force = [0.0; 3];

        self.jitter.push(&self.state);
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
        self.state = HumanoidState::standing_for(self.morphology);
        self.external_force = [0.0; 3];

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

        let n = self.state.joint_angles.len();
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

        self.state.root_height += perturbation * next_f64() * 0.02;
        self.state.head_height = self.state.root_height
            + self.body.segment_lengths[SEG_TORSO] * 0.5
            + self.body.segment_lengths[SEG_HEAD];

        self.jitter.reset(&self.state, seed);
        self.actuator_noise.reset(seed.wrapping_mul(2654435761));
        self.observation_noise.reset(seed.wrapping_mul(3141592653));
        self.observed_state = self.state.clone();

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

fn integrate_quaternion(q: [f64; 4], omega: [f64; 3], dt: f64) -> [f64; 4] {
    let [w, x, y, z] = q;
    let [ox, oy, oz] = omega;
    let dw = 0.5 * (-x * ox - y * oy - z * oz);
    let dx = 0.5 * (w * ox + y * oz - z * oy);
    let dy = 0.5 * (w * oy + z * ox - x * oz);
    let dz = 0.5 * (w * oz + x * oy - y * ox);
    normalize_quat([w + dw * dt, x + dx * dt, y + dy * dt, z + dz * dt])
}

fn normalize_quat(q: [f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-10 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
    }
}

/// MuJoCo-based humanoid simulator (requires mujoco-rs).
#[cfg(feature = "mujoco")]
use mujoco_rs::prelude::*;

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
        sim.data.forward();
        sim.extract_state();
        Ok(sim)
    }

    pub fn from_path(path: &str) -> anyhow::Result<Self> {
        let xml = std::fs::read_to_string(path)?;
        Self::from_xml(&xml)
    }

    pub fn from_bundled_asset() -> anyhow::Result<Self> {
        let xml = include_str!("../assets/humanoid.xml");
        Self::from_xml(xml)
    }

    /// Spins up a native high-fidelity MuJoCo instance using the procedural MJCF compiler
    pub fn for_morphology(
        morphology: crate::morphology::HumanoidMorphology,
    ) -> anyhow::Result<Self> {
        let xml = morphology.to_mjcf();
        Self::from_xml(&xml)
    }

    fn find_body_id(model: &MjModel, name: &str) -> usize {
        let id = model.name_to_id(MjtObj::mjOBJ_BODY, name);
        assert!(id >= 0, "Body '{}' not found in MJCF model", name);
        id as usize
    }

    fn find_body_id_or(model: &MjModel, name: &str, fallback: usize) -> usize {
        let id = model.name_to_id(MjtObj::mjOBJ_BODY, name);
        if id >= 0 { id as usize } else { fallback }
    }

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

    pub fn body_mass(&self) -> f64 {
        self.model.body_mass()[self.body_ids.torso]
    }

    fn write_ctrl(&mut self, cmd: &HumanoidCommand) {
        let ctrl = self.data.ctrl_mut();
        for i in 0..cmd.torques.len().min(ctrl.len()) {
            ctrl[i] = cmd.torques[i] as f64;
        }
    }

    fn apply_xfrc(&mut self) {
        if self.external_force[0].abs() > 1e-15
            || self.external_force[1].abs() > 1e-15
            || self.external_force[2].abs() > 1e-15
        {
            let xfrc = self.data.xfrc_applied_mut();
            xfrc[self.body_ids.torso][0] = self.external_force[0];
            xfrc[self.body_ids.torso][1] = self.external_force[1];
            xfrc[self.body_ids.torso][2] = self.external_force[2];
        }
    }

    fn clear_xfrc(&mut self) {
        self.data.xfrc_applied_mut()[self.body_ids.torso] = [0.0; 6];
        self.external_force = [0.0; 3];
    }

    pub fn model_arc(&self) -> &std::sync::Arc<MjModel> {
        &self.model
    }
    pub fn data_mut(&mut self) -> &mut MjData<std::sync::Arc<MjModel>> {
        &mut self.data
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
        assert!(sim.state().root_height > 0.5);
    }
}
