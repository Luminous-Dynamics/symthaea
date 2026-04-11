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
}

impl SimpleExoskeletonSimulator {
    pub fn new() -> Self {
        let config = ExoskeletonConfig::default();
        let total = config.human_mass + config.exo_mass;
        Self {
            state: ExoskeletonState::standing(),
            inertias: [0.12 * total, 0.08 * total, 0.03 * total, 0.12 * total, 0.08 * total, 0.03 * total],
            joint_damping: [3.0, 2.5, 1.5, 3.0, 2.5, 1.5],
            segment_masses: [0.10 * total, 0.05 * total, 0.015 * total, 0.10 * total, 0.05 * total, 0.015 * total],
            segment_lengths: [0.45, 0.43, 0.08, 0.45, 0.43, 0.08],
            config, rng_state: 42, gait_phase: 0.0, gait_frequency: 1.0, walking: true,
        }
    }

    pub fn set_walking(&mut self, w: bool) { self.walking = w; }

    fn random(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.rng_state >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0
    }

    fn gravity_torque(&self, joint: usize) -> f64 {
        let start = if joint < 3 { 0 } else { 3 };
        let end = start + 3;
        let mut tau = 0.0;
        for j in joint..end {
            tau += self.segment_masses[j] * G * self.segment_lengths[j] * self.state.joint_angles[joint].sin();
        }
        tau
    }

    fn human_gait_torques(&mut self) -> [f64; NUM_JOINTS] {
        if !self.walking {
            let mut t = [0.0; NUM_JOINTS];
            for v in &mut t { *v = self.random() * 2.0; }
            return t;
        }
        let p = self.gait_phase;
        let rp = p + std::f64::consts::PI;
        let n = 3.0;
        [
            30.0 * p.sin() + self.random() * n, 15.0 * (p * 2.0).sin() + self.random() * n,
            10.0 * (p + 0.5).cos() + self.random() * n,
            30.0 * rp.sin() + self.random() * n, 15.0 * (rp * 2.0).sin() + self.random() * n,
            10.0 * (rp + 0.5).cos() + self.random() * n,
        ]
    }
}

impl Default for SimpleExoskeletonSimulator { fn default() -> Self { Self::new() } }

impl ExoskeletonPhysicsSimulator for SimpleExoskeletonSimulator {
    fn step(&mut self, cmd: &ExoskeletonCommand, dt: f64) {
        let human = self.human_gait_torques();
        for i in 0..NUM_JOINTS {
            let exo = cmd.joint_torques[i] as f64 * self.config.max_torques[i];
            self.state.human_torques[i] = human[i];
            self.state.exo_torques[i] = exo;
            let gravity = self.gravity_torque(i);
            let total = human[i] + exo - self.joint_damping[i] * self.state.joint_velocities[i] - gravity;
            let ddq = total / self.inertias[i];
            self.state.joint_velocities[i] += ddq * dt;
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;
            let limits = match i % 3 { 0 => [-0.5, 1.5], 1 => [0.0, 2.4], 2 => [-0.5, 0.5], _ => unreachable!() };
            if self.state.joint_angles[i] < limits[0] { self.state.joint_angles[i] = limits[0]; self.state.joint_velocities[i] = self.state.joint_velocities[i].max(0.0); }
            if self.state.joint_angles[i] > limits[1] { self.state.joint_angles[i] = limits[1]; self.state.joint_velocities[i] = self.state.joint_velocities[i].min(0.0); }
        }
        let total_weight = (self.config.human_mass + self.config.exo_mass) * G;
        self.state.ground_reaction_force = total_weight;
        self.state.center_of_pressure[0] = 0.10 * self.gait_phase.sin();
        self.state.center_of_pressure[1] = 0.02 * (self.gait_phase * 2.0).sin();
        if self.walking {
            self.gait_phase += std::f64::consts::TAU * self.gait_frequency * dt;
            if self.gait_phase > std::f64::consts::TAU { self.gait_phase -= std::f64::consts::TAU; }
        }
        self.state.battery_soc = (self.state.battery_soc - 0.00001 * dt).max(0.0);
    }
    fn state(&self) -> &ExoskeletonState { &self.state }
    fn reset(&mut self) { self.state = ExoskeletonState::standing(); self.gait_phase = 0.0; self.rng_state = 42; }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_standing_stable() {
        let mut sim = SimpleExoskeletonSimulator::new();
        sim.set_walking(false);
        for _ in 0..1000 { sim.step(&ExoskeletonCommand::zero(), 0.005); }
        assert!(sim.state().is_finite());
    }
    #[test]
    fn test_walking_moves() {
        let mut sim = SimpleExoskeletonSimulator::new();
        let init = sim.state().joint_angles;
        for _ in 0..200 { sim.step(&ExoskeletonCommand::zero(), 0.005); }
        let max_d = (0..NUM_JOINTS).map(|i| (sim.state().joint_angles[i] - init[i]).abs()).fold(0.0f64, f64::max);
        assert!(max_d > 0.01, "Walking should produce motion: {max_d}");
    }
    #[test]
    fn test_joint_limits() {
        let mut sim = SimpleExoskeletonSimulator::new();
        let mut cmd = ExoskeletonCommand::zero(); cmd.joint_torques = [1.0; NUM_ACTUATORS];
        for _ in 0..10000 { sim.step(&cmd, 0.005); }
        assert!(sim.state().joint_angles[0] <= 1.51);
        assert!(sim.state().is_finite());
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
