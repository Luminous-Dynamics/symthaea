use crate::types::*;
pub trait OrbitalPhysicsSimulator {
    fn step(&mut self, cmd: &OrbitalCommand, dt: f64);
    fn state(&self) -> &OrbitalState;
    fn reset(&mut self);
}

pub struct SimpleOrbitalSimulator {
    state: OrbitalState,
    config: OrbitalConfig,
    ji: [f64; NUM_JOINTS],
    jd: [f64; NUM_JOINTS],
    links: [f64; NUM_JOINTS],
    rwm: [f64; 3],
    orbit_phase: f64,
}
impl SimpleOrbitalSimulator {
    pub fn new() -> Self {
        Self {
            state: OrbitalState::stowed(),
            config: OrbitalConfig::default(),
            ji: [5.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.3],
            jd: [1.0, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1],
            links: [1.5, 1.5, 1.0, 0.8, 0.5, 0.3, 0.2],
            rwm: [0.0; 3],
            orbit_phase: 0.0,
        }
    }
}
impl Default for SimpleOrbitalSimulator {
    fn default() -> Self {
        Self::new()
    }
}
impl OrbitalPhysicsSimulator for SimpleOrbitalSimulator {
    fn step(&mut self, cmd: &OrbitalCommand, dt: f64) {
        let mut react = [0.0f64; 3];
        for i in 0..NUM_JOINTS {
            let torque = cmd.joint_torques[i] as f64 * self.config.max_joint_torques[i];
            let ddq = (torque - self.jd[i] * self.state.joint_velocities[i]) / self.ji[i];
            self.state.joint_velocities[i] += ddq * dt;
            self.state.joint_angles[i] += self.state.joint_velocities[i] * dt;
            self.state.joint_angles[i] = self.state.joint_angles[i].clamp(-2.9, 2.9);
            if self.state.joint_angles[i].abs() >= 2.89 {
                self.state.joint_velocities[i] = 0.0;
            }
            react[i % 3] -= torque;
        }
        for a in 0..3 {
            let alpha = react[a] / self.config.spacecraft_inertia[a];
            self.state.spacecraft_angular_velocity[a] += alpha * dt;
            let wt = (-50.0 * self.state.spacecraft_angular_velocity[a]).clamp(-5.0, 5.0);
            self.rwm[a] = (self.rwm[a] + wt * dt).clamp(
                -self.config.reaction_wheel_capacity,
                self.config.reaction_wheel_capacity,
            );
            self.state.spacecraft_angular_velocity[a] +=
                wt * dt / self.config.spacecraft_inertia[a];
        }
        // FK
        let (mut x, mut z, mut ca) = (0.0, 0.0, 0.0);
        for i in 0..NUM_JOINTS.min(4) {
            ca += self.state.joint_angles[i];
            x += self.links[i] * ca.sin();
            z += self.links[i] * ca.cos();
        }
        self.state.ee_position = [x, 0.3 * self.state.joint_angles[4].sin(), z];
        // Orbit
        self.orbit_phase += dt / 5400.0 * std::f64::consts::TAU;
        if self.orbit_phase > std::f64::consts::TAU {
            self.orbit_phase -= std::f64::consts::TAU;
        }
        self.state.solar_exposure = (self.orbit_phase.sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        self.state.comm_window = if self.orbit_phase.cos() > -0.3 {
            1.0
        } else {
            0.0
        };
    }
    fn state(&self) -> &OrbitalState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = OrbitalState::stowed();
        self.rwm = [0.0; 3];
        self.orbit_phase = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut s = SimpleOrbitalSimulator::new();
        for _ in 0..1000 {
            s.step(&OrbitalCommand::zero(), 0.01);
        }
        assert!(s.state().is_finite());
    }
    #[test]
    fn test_reaction() {
        let mut s = SimpleOrbitalSimulator::new();
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 0.5;
        for _ in 0..100 {
            s.step(&c, 0.01);
        }
        assert!(
            s.state()
                .spacecraft_angular_velocity
                .iter()
                .map(|v| v.abs())
                .sum::<f64>()
                > 0.0
        );
    }

    #[test]
    fn test_reset_returns_to_stowed() {
        // After arbitrary motion, reset() must restore the stowed state.
        let mut s = SimpleOrbitalSimulator::new();
        let stowed_angles = s.state().joint_angles;
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 0.5;
        for _ in 0..200 {
            s.step(&c, 0.01);
        }
        assert_ne!(s.state().joint_angles, stowed_angles, "motion should occur");
        s.reset();
        assert_eq!(s.state().joint_angles, stowed_angles);
        assert_eq!(s.state().joint_velocities, [0.0; NUM_JOINTS]);
        assert_eq!(s.state().spacecraft_angular_velocity, [0.0; 3]);
    }

    #[test]
    fn test_zero_command_no_joint_drift() {
        // Starting at stowed (all angles 0) with zero torque, joints must
        // stay at zero — no spontaneous motion from integration error.
        let mut s = SimpleOrbitalSimulator::new();
        for _ in 0..500 {
            s.step(&OrbitalCommand::zero(), 0.01);
        }
        for i in 0..NUM_JOINTS {
            assert!(
                s.state().joint_angles[i].abs() < 1e-9,
                "joint {} drifted under zero command",
                i
            );
            assert!(s.state().joint_velocities[i].abs() < 1e-9);
        }
    }

    #[test]
    fn test_joint_limits_clamp() {
        // Documented joint range is ±2.9 rad; sustained max torque must
        // saturate at that boundary without going over.
        let mut s = SimpleOrbitalSimulator::new();
        let mut c = OrbitalCommand::zero();
        c.joint_torques[0] = 1.0; // max positive
        for _ in 0..10_000 {
            s.step(&c, 0.01);
        }
        assert!(
            s.state().joint_angles[0] <= 2.9 + 1e-6,
            "joint 0 should saturate at +2.9 rad, got {}",
            s.state().joint_angles[0]
        );
    }

    #[test]
    fn test_deterministic_across_fresh_sims() {
        // Two independent fresh simulators driven by identical commands
        // must end in the same state — RL training precondition.
        let c = OrbitalCommand {
            joint_torques: [0.2, -0.1, 0.3, 0.0, 0.15, -0.05, 0.1],
        };
        let mut a = SimpleOrbitalSimulator::new();
        let mut b = SimpleOrbitalSimulator::new();
        for _ in 0..300 {
            a.step(&c, 0.01);
            b.step(&c, 0.01);
        }
        assert_eq!(a.state().joint_angles, b.state().joint_angles);
        assert_eq!(
            a.state().spacecraft_angular_velocity,
            b.state().spacecraft_angular_velocity
        );
    }

    #[test]
    fn test_orbit_phase_varies_solar_exposure() {
        // Integrate over one full LEO orbit (5400 s, dt = 1 s). Orbit phase
        // drives `solar_exposure = sin(phase) * 0.5 + 0.5`, so it must sweep
        // through both extremes: phase=π/2 → ~1.0, phase=3π/2 → ~0.0 (clamped).
        // Half an orbit is NOT enough — sin stays non-negative through π.
        let mut s = SimpleOrbitalSimulator::new();
        let mut saw_low = false;
        let mut saw_high = false;
        for _ in 0..5400 {
            s.step(&OrbitalCommand::zero(), 1.0);
            if s.state().solar_exposure < 0.1 {
                saw_low = true;
            }
            if s.state().solar_exposure > 0.9 {
                saw_high = true;
            }
        }
        assert!(
            saw_low && saw_high,
            "over a full orbit solar_exposure should sweep through low and high values \
             (saw_low={saw_low}, saw_high={saw_high})"
        );
    }

    mod proptest_physics {
        use super::*;
        use proptest::prelude::*;
        proptest! { #[test] fn finite(t0 in -1.0f32..1.0, t1 in -1.0f32..1.0, t2 in -1.0f32..1.0, t3 in -1.0f32..1.0, t4 in -1.0f32..1.0, t5 in -1.0f32..1.0, t6 in -1.0f32..1.0, dt in 0.001f64..0.05, steps in 1usize..300) {
            let mut s = SimpleOrbitalSimulator::new(); let c = OrbitalCommand { joint_torques: [t0,t1,t2,t3,t4,t5,t6] };
            for _ in 0..steps { s.step(&c, dt); } prop_assert!(s.state().is_finite());
        }}
    }
}
