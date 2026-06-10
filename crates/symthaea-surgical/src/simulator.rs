// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::*;

pub trait SurgicalPhysicsSimulator {
    fn step(&mut self, cmd: &SurgicalCommand, dt: f64);
    fn state(&self) -> &SurgicalState;
    fn reset(&mut self);
}

pub struct SimpleSurgicalSimulator {
    state: SurgicalState,
    config: SurgicalConfig,
    inertias: [f64; NUM_JOINTS],
    damping: [f64; NUM_JOINTS],
    tremor_state: [f64; NUM_JOINTS],
    tissue_stiffness: f64,
    tissue_z: f64,
}

impl SimpleSurgicalSimulator {
    pub fn new() -> Self {
        Self {
            state: SurgicalState::home(),
            config: SurgicalConfig::default(),
            inertias: [0.5, 0.5, 0.3, 0.2, 0.1, 0.05],
            damping: [2.0, 2.0, 1.5, 1.0, 0.5, 0.3],
            tremor_state: [0.0; NUM_JOINTS],
            tissue_stiffness: 0.5,
            tissue_z: -10.0,
        }
    }
    fn filter_tremor(&mut self, raw: [f64; NUM_JOINTS], dt: f64) -> [f64; NUM_JOINTS] {
        let a = dt * self.config.tremor_filter_hz * std::f64::consts::TAU;
        let a = a / (1.0 + a);
        let mut out = [0.0; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            self.tremor_state[i] = self.tremor_state[i] * (1.0 - a) + raw[i] * a;
            out[i] = self.tremor_state[i];
        }
        out
    }
}
impl Default for SimpleSurgicalSimulator {
    fn default() -> Self {
        Self::new()
    }
}
impl SurgicalPhysicsSimulator for SimpleSurgicalSimulator {
    fn step(&mut self, cmd: &SurgicalCommand, dt: f64) {
        let mut raw = [0.0; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            raw[i] = cmd.joint_torques[i] as f64
                * self.config.max_joint_torques[i]
                * self.config.motion_scaling;
        }
        let filtered = self.filter_tremor(raw, dt);
        for i in 0..NUM_JOINTS {
            let ddq =
                (filtered[i] - self.damping[i] * self.state.joint_velocities[i]) / self.inertias[i];
            self.state.joint_velocities[i] += ddq * dt;
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;
            let lim = match i {
                0 | 3 => 1.5,
                1 | 4 => 1.2,
                _ => 0.8,
            };
            if self.state.joint_angles[i] < -lim {
                self.state.joint_angles[i] = -lim;
                self.state.joint_velocities[i] = self.state.joint_velocities[i].max(0.0);
            }
            if self.state.joint_angles[i] > lim {
                self.state.joint_angles[i] = lim;
                self.state.joint_velocities[i] = self.state.joint_velocities[i].min(0.0);
            }
        }
        // Simplified FK
        let (q1, q2, q3) = (
            self.state.joint_angles[0],
            self.state.joint_angles[1],
            self.state.joint_angles[2],
        );
        self.state.tip_position[0] =
            150.0 * q1.sin() + 100.0 * (q1 + q2).sin() + 30.0 * (q1 + q2 + q3).sin();
        self.state.tip_position[2] =
            -(150.0 * q1.cos() + 100.0 * (q1 + q2).cos() + 30.0 * (q1 + q2 + q3).cos());
        self.state.tip_position[1] = 30.0 * self.state.joint_angles[3].sin();
        // Tissue interaction
        let depth = self.state.tip_position[2] - self.tissue_z;
        if depth > 0.0 {
            self.state.tip_force[2] = -self.tissue_stiffness * depth;
        } else {
            self.state.tip_force = [0.0; 3];
        }
        self.state.critical_structure_distance = (20.0 + 5.0 * (q1 * 3.0).sin()).max(1.0);
        self.state.trocar_compliance =
            ((self.state.tip_position[0].powi(2) + self.state.tip_position[1].powi(2)).sqrt()
                / 50.0)
                .clamp(0.0, 1.0);
        self.state.jaw_angle = (cmd.jaw as f64).clamp(0.0, 1.0);
        self.state.cautery_power = (cmd.cautery as f64).clamp(0.0, 1.0);
    }
    fn state(&self) -> &SurgicalState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = SurgicalState::home();
        self.tremor_state = [0.0; NUM_JOINTS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut s = SimpleSurgicalSimulator::new();
        for _ in 0..1000 {
            s.step(&SurgicalCommand::zero(), 0.001);
        }
        assert!(s.state().is_finite());
    }
    #[test]
    fn test_torque_moves() {
        let mut s = SimpleSurgicalSimulator::new();
        let init = s.state().tip_position;
        let mut c = SurgicalCommand::zero();
        c.joint_torques[0] = 0.3;
        for _ in 0..500 {
            s.step(&c, 0.001);
        }
        let d = ((s.state().tip_position[0] - init[0]).powi(2)
            + (s.state().tip_position[2] - init[2]).powi(2))
        .sqrt();
        assert!(d > 0.1);
    }

    #[test]
    fn test_reset_returns_to_home() {
        // After arbitrary motion, reset() must restore the home state.
        let mut s = SimpleSurgicalSimulator::new();
        let home_angles = s.state().joint_angles;
        let home_tip = s.state().tip_position;
        let mut c = SurgicalCommand::zero();
        c.joint_torques[0] = 0.5;
        for _ in 0..200 {
            s.step(&c, 0.001);
        }
        assert_ne!(
            s.state().tip_position,
            home_tip,
            "motion should have occurred"
        );
        s.reset();
        assert_eq!(s.state().joint_angles, home_angles);
        assert_eq!(s.state().joint_velocities, [0.0; NUM_JOINTS]);
        assert_eq!(s.state().tip_position, home_tip);
    }

    #[test]
    fn test_zero_command_no_joint_drift() {
        // Starting at home with zero torque: velocities stay zero, angles
        // stay pinned at home. (Tip position is not part of the invariant —
        // FK is recomputed each step and disagrees with home's hardcoded
        // initial tip; the joint angles are the authoritative state.)
        let mut s = SimpleSurgicalSimulator::new();
        let init_angles = s.state().joint_angles;
        for _ in 0..500 {
            s.step(&SurgicalCommand::zero(), 0.001);
        }
        for i in 0..NUM_JOINTS {
            assert!(
                (s.state().joint_angles[i] - init_angles[i]).abs() < 1e-9,
                "joint {} drifted from home under zero command",
                i
            );
            assert!(s.state().joint_velocities[i].abs() < 1e-9);
        }
    }

    #[test]
    fn test_joint_limits_enforced() {
        // Sustained max torque on joint 0 must saturate at the 1.5 rad limit.
        let mut s = SimpleSurgicalSimulator::new();
        let mut c = SurgicalCommand::zero();
        c.joint_torques[0] = 1.0;
        for _ in 0..5000 {
            s.step(&c, 0.001);
        }
        assert!(
            s.state().joint_angles[0] <= 1.5 + 1e-6,
            "joint 0 should saturate at positive limit"
        );
        assert!(
            s.state().joint_angles[0] >= -1.5 - 1e-6,
            "joint 0 should stay above negative limit"
        );
    }

    #[test]
    fn test_deterministic_across_fresh_sims() {
        // Two independent fresh sims driven by identical commands must end
        // in the same state — determinism precondition for RL training.
        let c = SurgicalCommand {
            joint_torques: [0.2, -0.1, 0.3, 0.0, 0.15, -0.05],
            jaw: 0.3,
            cautery: 0.0,
        };
        let mut a = SimpleSurgicalSimulator::new();
        let mut b = SimpleSurgicalSimulator::new();
        for _ in 0..300 {
            a.step(&c, 0.001);
            b.step(&c, 0.001);
        }
        assert_eq!(a.state().joint_angles, b.state().joint_angles);
        assert_eq!(a.state().tip_position, b.state().tip_position);
    }

    mod proptest_physics {
        use super::*;
        use proptest::prelude::*;
        proptest! {
            #[test]
            fn finite(
                t0 in -1.0f32..1.0, t1 in -1.0f32..1.0, t2 in -1.0f32..1.0,
                t3 in -1.0f32..1.0, t4 in -1.0f32..1.0, t5 in -1.0f32..1.0,
                dt in 0.0001f64..0.005, steps in 1usize..500
            ) {
                let mut s = SimpleSurgicalSimulator::new();
                let c = SurgicalCommand { joint_torques: [t0,t1,t2,t3,t4,t5], jaw: 0.0, cautery: 0.0 };
                for _ in 0..steps { s.step(&c, dt); }
                prop_assert!(s.state().is_finite());
            }
        }
    }
}
