// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Human-exoskeleton coupled impedance dynamics simulator.
use crate::types::*;

const G: f64 = 9.81;

pub trait ExoskeletonPhysicsSimulator {
    fn step(&mut self, cmd: &ExoskeletonCommand, dt: f64);
    fn state(&self) -> &ExoskeletonState;
    fn reset(&mut self);
}

pub struct SimpleExoskeletonSimulator {
    state: ExoskeletonState,
    config: ExoskeletonConfig,
    inertias: [f64; NUM_JOINTS],
    joint_damping: [f64; NUM_JOINTS],
    segment_masses: [f64; NUM_JOINTS],
    segment_lengths: [f64; NUM_JOINTS],
    rng_state: u64,
    gait_phase: f64,
    gait_frequency: f64,
    walking: bool,
    /// Neutral (standing) posture the exo impedance spring pulls toward.
    neutral_pose: [f64; NUM_JOINTS],
    /// Disturbance-observer estimate of the HUMAN's joint torques,
    /// inferred from the dynamics residual (I·q̈ + d·q̇ + g − τ_exo) —
    /// never by reading the scripted human generator. This is what a real
    /// exo can actually sense; see `intent_estimate()`.
    intent_estimate: [f64; NUM_JOINTS],
}

impl SimpleExoskeletonSimulator {
    pub fn new() -> Self {
        let config = ExoskeletonConfig::default();
        let total = config.human_mass + config.exo_mass;
        Self {
            state: ExoskeletonState::standing(),
            inertias: [
                0.12 * total,
                0.08 * total,
                0.03 * total,
                0.12 * total,
                0.08 * total,
                0.03 * total,
            ],
            joint_damping: [3.0, 2.5, 1.5, 3.0, 2.5, 1.5],
            segment_masses: [
                0.10 * total,
                0.05 * total,
                0.015 * total,
                0.10 * total,
                0.05 * total,
                0.015 * total,
            ],
            segment_lengths: [0.45, 0.43, 0.08, 0.45, 0.43, 0.08],
            config,
            rng_state: 42,
            gait_phase: 0.0,
            gait_frequency: 1.0,
            walking: true,
            neutral_pose: ExoskeletonState::standing().joint_angles,
            intent_estimate: [0.0; NUM_JOINTS],
        }
    }

    pub fn set_walking(&mut self, w: bool) {
        self.walking = w;
    }

    /// Disturbance-observer estimate of the human's joint torques (N·m).
    ///
    /// Computed each step as the dynamics residual: the torque left over
    /// after accounting for observed acceleration, damping, gravity, and
    /// the exo's own (commanded + impedance) torque. A real exoskeleton
    /// estimates its wearer's intent exactly this way — from interaction
    /// dynamics, not from privileged access to the human's motor plan.
    pub fn intent_estimate(&self) -> [f64; NUM_JOINTS] {
        self.intent_estimate
    }

    fn random(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        ((self.rng_state >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0
    }

    fn gravity_torque(&self, joint: usize) -> f64 {
        let start = if joint < 3 { 0 } else { 3 };
        let end = start + 3;
        let mut tau = 0.0;
        for j in joint..end {
            tau += self.segment_masses[j]
                * G
                * self.segment_lengths[j]
                * self.state.joint_angles[joint].sin();
        }
        tau
    }

    fn human_gait_torques(&mut self) -> [f64; NUM_JOINTS] {
        if !self.walking {
            let mut t = [0.0; NUM_JOINTS];
            for v in &mut t {
                *v = self.random() * 2.0;
            }
            return t;
        }
        let p = self.gait_phase;
        let rp = p + std::f64::consts::PI;
        let n = 3.0;
        [
            30.0 * p.sin() + self.random() * n,
            15.0 * (p * 2.0).sin() + self.random() * n,
            10.0 * (p + 0.5).cos() + self.random() * n,
            30.0 * rp.sin() + self.random() * n,
            15.0 * (rp * 2.0).sin() + self.random() * n,
            10.0 * (rp + 0.5).cos() + self.random() * n,
        ]
    }
}

impl Default for SimpleExoskeletonSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExoskeletonPhysicsSimulator for SimpleExoskeletonSimulator {
    fn step(&mut self, cmd: &ExoskeletonCommand, dt: f64) {
        let human = self.human_gait_torques();
        // Impedance gains: stiffness/damping were previously telemetry-only
        // (self-flagged in embodiment.rs) — the "compliant/transparent"
        // tiers had identical mechanics. They are now a real spring-damper
        // the exo applies toward the neutral posture (Tier 2.6, 2026-07).
        let k_imp = (cmd.stiffness_gain as f64).clamp(0.0, 1.0) * self.config.max_joint_stiffness;
        let d_imp = (cmd.damping_gain as f64).clamp(0.0, 1.0) * self.config.max_joint_damping;
        for i in 0..NUM_JOINTS {
            let exo_cmd = cmd.joint_torques[i] as f64 * self.config.max_torques[i];
            let impedance = -k_imp * (self.state.joint_angles[i] - self.neutral_pose[i])
                - d_imp * self.state.joint_velocities[i];
            let exo = exo_cmd + impedance;
            self.state.human_torques[i] = human[i];
            self.state.exo_torques[i] = exo;
            let gravity = self.gravity_torque(i);
            let v_prev = self.state.joint_velocities[i];
            let total = human[i] + exo - self.joint_damping[i] * v_prev - gravity;
            let ddq = total / self.inertias[i];
            self.state.joint_velocities[i] += ddq * dt;
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;
            // Intent observer: residual torque after everything the exo can
            // model/measure — I·q̈_obs + d·q̇ + g − τ_exo. (q̈_obs from the
            // velocity delta, i.e. what an encoder-differentiating observer
            // sees; NOT a read of the scripted human generator.)
            let ddq_obs = (self.state.joint_velocities[i] - v_prev) / dt.max(1e-9);
            self.intent_estimate[i] =
                self.inertias[i] * ddq_obs + self.joint_damping[i] * v_prev + gravity - exo;
            let limits = match i % 3 {
                0 => [-0.5, 1.5],
                1 => [0.0, 2.4],
                2 => [-0.5, 0.5],
                _ => unreachable!(),
            };
            if self.state.joint_angles[i] < limits[0] {
                self.state.joint_angles[i] = limits[0];
                self.state.joint_velocities[i] = self.state.joint_velocities[i].max(0.0);
            }
            if self.state.joint_angles[i] > limits[1] {
                self.state.joint_angles[i] = limits[1];
                self.state.joint_velocities[i] = self.state.joint_velocities[i].min(0.0);
            }
        }
        let total_weight = (self.config.human_mass + self.config.exo_mass) * G;
        self.state.ground_reaction_force = total_weight;
        self.state.center_of_pressure[0] = 0.10 * self.gait_phase.sin();
        self.state.center_of_pressure[1] = 0.02 * (self.gait_phase * 2.0).sin();
        if self.walking {
            self.gait_phase += std::f64::consts::TAU * self.gait_frequency * dt;
            if self.gait_phase > std::f64::consts::TAU {
                self.gait_phase -= std::f64::consts::TAU;
            }
        }
        self.state.battery_soc = (self.state.battery_soc - 0.00001 * dt).max(0.0);
    }
    fn state(&self) -> &ExoskeletonState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = ExoskeletonState::standing();
        self.gait_phase = 0.0;
        self.rng_state = 42;
        self.intent_estimate = [0.0; NUM_JOINTS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_standing_stable() {
        let mut sim = SimpleExoskeletonSimulator::new();
        sim.set_walking(false);
        for _ in 0..1000 {
            sim.step(&ExoskeletonCommand::zero(), 0.005);
        }
        assert!(sim.state().is_finite());
    }
    #[test]
    fn test_walking_moves() {
        let mut sim = SimpleExoskeletonSimulator::new();
        let init = sim.state().joint_angles;
        for _ in 0..200 {
            sim.step(&ExoskeletonCommand::zero(), 0.005);
        }
        let max_d = (0..NUM_JOINTS)
            .map(|i| (sim.state().joint_angles[i] - init[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(max_d > 0.01, "Walking should produce motion: {max_d}");
    }
    #[test]
    fn test_joint_limits() {
        let mut sim = SimpleExoskeletonSimulator::new();
        let mut cmd = ExoskeletonCommand::zero();
        cmd.joint_torques = [1.0; NUM_ACTUATORS];
        for _ in 0..10000 {
            sim.step(&cmd, 0.005);
        }
        assert!(sim.state().joint_angles[0] <= 1.51);
        assert!(sim.state().is_finite());
    }
    #[test]
    fn test_stiffness_tier_is_physically_real() {
        // High-impedance tier must resist the human's perturbing torques
        // more than the transparent tier — this fails against the old sim,
        // where stiffness_gain/damping_gain were telemetry-only and the
        // "compliant" tiers had identical mechanics. Same RNG seed in both
        // sims → identical human torque sequences.
        let mut stiff = SimpleExoskeletonSimulator::new();
        stiff.set_walking(false); // random perturbing human torques
        let mut transparent = SimpleExoskeletonSimulator::new();
        transparent.set_walking(false);

        let stiff_cmd = ExoskeletonCommand {
            joint_torques: [0.0; NUM_ACTUATORS],
            stiffness_gain: 1.0,
            damping_gain: 1.0,
        };
        let transparent_cmd = ExoskeletonCommand {
            joint_torques: [0.0; NUM_ACTUATORS],
            stiffness_gain: 0.0,
            damping_gain: 0.0,
        };

        let neutral = ExoskeletonState::standing().joint_angles;
        let mut dev_stiff = 0.0f64;
        let mut dev_transparent = 0.0f64;
        for _ in 0..2000 {
            stiff.step(&stiff_cmd, 0.005);
            transparent.step(&transparent_cmd, 0.005);
            for i in 0..NUM_JOINTS {
                dev_stiff += (stiff.state().joint_angles[i] - neutral[i]).powi(2);
                dev_transparent += (transparent.state().joint_angles[i] - neutral[i]).powi(2);
            }
        }
        assert!(
            dev_stiff < 0.5 * dev_transparent,
            "full impedance must at least halve posture deviation: stiff {dev_stiff:.4} vs transparent {dev_transparent:.4}"
        );
    }

    #[test]
    fn test_intent_estimator_tracks_human_torque() {
        // The disturbance-observer intent estimate must track the scripted
        // human's actual torque (ground truth it never reads directly).
        let mut sim = SimpleExoskeletonSimulator::new();
        let cmd = ExoskeletonCommand {
            joint_torques: [0.1; NUM_ACTUATORS],
            stiffness_gain: 0.3,
            damping_gain: 0.2,
        };
        let mut agree = 0usize;
        let mut total = 0usize;
        for _ in 0..1000 {
            sim.step(&cmd, 0.005);
            let est = sim.intent_estimate();
            let truth = sim.state().human_torques;
            for i in 0..NUM_JOINTS {
                // Only score confidently-nonzero ground truth (hip/knee
                // amplitudes reach 15-30 N·m; noise floor is ~3 N·m).
                if truth[i].abs() > 5.0 {
                    total += 1;
                    if est[i].signum() == truth[i].signum() {
                        agree += 1;
                    }
                }
            }
        }
        assert!(total > 500, "need a meaningful sample, got {total}");
        let rate = agree as f64 / total as f64;
        assert!(
            rate > 0.9,
            "intent estimate must track human torque sign: {rate:.3} agreement"
        );
    }

    mod proptest_physics {
        use super::*;
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn arbitrary_stay_finite(
                t0 in -1.0f32..1.0, t1 in -1.0f32..1.0, t2 in -1.0f32..1.0,
                t3 in -1.0f32..1.0, t4 in -1.0f32..1.0, t5 in -1.0f32..1.0,
                dt in 0.001f64..0.02, steps in 1usize..500,
            ) {
                let mut sim = SimpleExoskeletonSimulator::new();
                let cmd = ExoskeletonCommand { joint_torques: [t0,t1,t2,t3,t4,t5], stiffness_gain: 0.5, damping_gain: 0.3 };
                for _ in 0..steps { sim.step(&cmd, dt); }
                prop_assert!(sim.state().is_finite());
            }
        }
    }
}
