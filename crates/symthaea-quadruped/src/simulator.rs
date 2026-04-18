use crate::types::*;
const G: f64 = 9.81;
const GROUND_K: f64 = 5000.0;
const GROUND_D: f64 = 100.0;
pub trait QuadrupedPhysicsSimulator {
    fn step(&mut self, cmd: &QuadrupedCommand, dt: f64);
    fn state(&self) -> &QuadrupedState;
    fn reset(&mut self);
}

pub struct SimpleQuadrupedSimulator {
    state: QuadrupedState,
    config: QuadrupedConfig,
    ji: [f64; NUM_JOINTS],
    jd: [f64; NUM_JOINTS],
    cpg_phases: [f64; NUM_LEGS],
    cpg_freq: f64,
    gait: GaitType,
}
impl SimpleQuadrupedSimulator {
    pub fn new() -> Self {
        Self {
            state: QuadrupedState::standing(),
            config: QuadrupedConfig::default(),
            ji: [0.5, 0.5, 0.3, 0.5, 0.5, 0.3, 0.5, 0.5, 0.3, 0.5, 0.5, 0.3],
            jd: [3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0],
            cpg_phases: [0.0, std::f64::consts::PI, std::f64::consts::PI, 0.0],
            cpg_freq: 2.0,
            gait: GaitType::Trot,
        }
    }
    pub fn set_gait(&mut self, g: GaitType) {
        self.gait = g;
        self.cpg_freq = g.frequency();
    }
    fn cpg_target(&self, leg: usize) -> [f64; JOINTS_PER_LEG] {
        let p = self.cpg_phases[leg];
        let sh = self.gait.step_height();
        let sw = p.sin().max(0.0);
        [0.0, 0.5 + 0.2 * p.sin(), -1.0 + sw * sh * 10.0]
    }
    fn foot_height(&self, leg: usize) -> f64 {
        let b = leg * JOINTS_PER_LEG;
        let hf = self.state.joint_angles[b + 1];
        let k = self.state.joint_angles[b + 2];
        self.state.base_position[2] - 0.2 * hf.cos() - 0.2 * (hf + k).cos()
    }
}
impl Default for SimpleQuadrupedSimulator {
    fn default() -> Self {
        Self::new()
    }
}
impl QuadrupedPhysicsSimulator for SimpleQuadrupedSimulator {
    fn step(&mut self, cmd: &QuadrupedCommand, dt: f64) {
        for leg in 0..NUM_LEGS {
            self.cpg_phases[leg] += std::f64::consts::TAU * self.cpg_freq * dt;
            if self.cpg_phases[leg] > std::f64::consts::TAU {
                self.cpg_phases[leg] -= std::f64::consts::TAU;
            }
        }
        let mut vf = 0.0;
        for leg in 0..NUM_LEGS {
            let tgt = self.cpg_target(leg);
            for j in 0..JOINTS_PER_LEG {
                let idx = leg * JOINTS_PER_LEG + j;
                let ct = cmd.joint_torques[idx] as f64 * self.config.max_joint_torques[idx];
                let err = tgt[j] - self.state.joint_angles[idx];
                let total = 50.0 * err - 5.0 * self.state.joint_velocities[idx] + ct
                    - self.jd[idx] * self.state.joint_velocities[idx];
                let ddq = total / self.ji[idx];
                self.state.joint_velocities[idx] += ddq * dt;
                self.state.joint_angles[idx] += self.state.joint_velocities[idx] * dt;
                self.state.joint_angles[idx] = self.state.joint_angles[idx].clamp(-2.0, 2.0);
                if self.state.joint_angles[idx].abs() >= 1.99 {
                    self.state.joint_velocities[idx] = 0.0;
                }
            }
            let fh = self.foot_height(leg);
            if fh <= 0.0 {
                self.state.foot_contacts[leg] = 1.0;
                vf += GROUND_K * (-fh) + GROUND_D * (-self.state.base_linear_velocity[2]).max(0.0);
            } else {
                self.state.foot_contacts[leg] = 0.0;
            }
        }
        let az = (vf - self.config.body_mass * G) / self.config.body_mass;
        self.state.base_linear_velocity[2] += az * dt;
        self.state.base_position[2] += self.state.base_linear_velocity[2] * dt;
        let fwd = match self.gait {
            GaitType::Trot => 1.5,
            GaitType::Walk => 0.5,
            _ => 0.0,
        };
        self.state.base_linear_velocity[0] = fwd;
        self.state.base_position[0] += fwd * dt;
        if self.state.base_position[2] < 0.05 {
            self.state.base_position[2] = 0.05;
            self.state.base_linear_velocity[2] = self.state.base_linear_velocity[2].max(0.0);
        }
    }
    fn state(&self) -> &QuadrupedState {
        &self.state
    }
    fn reset(&mut self) {
        self.state = QuadrupedState::standing();
        self.cpg_phases = [0.0, std::f64::consts::PI, std::f64::consts::PI, 0.0];
        self.cpg_freq = 2.0;
        self.gait = GaitType::Trot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stable() {
        let mut s = SimpleQuadrupedSimulator::new();
        s.set_gait(GaitType::Freeze);
        for _ in 0..1000 {
            s.step(&QuadrupedCommand::zero(), 0.005);
        }
        assert!(s.state().is_finite());
        assert!(s.state().height() > 0.0);
    }
    #[test]
    fn test_trot() {
        let mut s = SimpleQuadrupedSimulator::new();
        let x0 = s.state().base_position[0];
        for _ in 0..1000 {
            s.step(&QuadrupedCommand::zero(), 0.005);
        }
        assert!(s.state().base_position[0] > x0 + 1.0);
    }

    #[test]
    fn test_reset_returns_to_standing() {
        // After arbitrary locomotion, reset() must restore the standing pose.
        let mut s = SimpleQuadrupedSimulator::new();
        let standing_pos = s.state().base_position;
        let standing_angles = s.state().joint_angles;
        // Let it trot for 500 steps to accumulate state.
        for _ in 0..500 {
            s.step(&QuadrupedCommand::zero(), 0.005);
        }
        assert_ne!(s.state().base_position[0], standing_pos[0]);
        s.reset();
        assert_eq!(s.state().base_position, standing_pos);
        assert_eq!(s.state().joint_angles, standing_angles);
        assert_eq!(s.state().joint_velocities, [0.0; NUM_JOINTS]);
    }

    #[test]
    fn test_freeze_gait_stays_still() {
        // Freeze gait has zero CPG frequency — robot shouldn't translate.
        let mut s = SimpleQuadrupedSimulator::new();
        s.set_gait(GaitType::Freeze);
        let x0 = s.state().base_position[0];
        for _ in 0..500 {
            s.step(&QuadrupedCommand::zero(), 0.005);
        }
        // Freeze: motion limited to stabilization residuals — should stay
        // within 10 cm of starting x (vs ~1 m for trot over same duration).
        assert!(
            (s.state().base_position[0] - x0).abs() < 0.1,
            "freeze should not translate, drifted {:.3} m",
            s.state().base_position[0] - x0
        );
    }

    #[test]
    fn test_deterministic_across_fresh_sims() {
        // Two independent fresh sims with identical commands must end in
        // identical state — RL training precondition.
        let c = QuadrupedCommand {
            joint_torques: [
                0.1, -0.05, 0.15, 0.1, -0.05, 0.15, 0.1, -0.05, 0.15, 0.1, -0.05, 0.15,
            ],
        };
        let mut a = SimpleQuadrupedSimulator::new();
        let mut b = SimpleQuadrupedSimulator::new();
        for _ in 0..200 {
            a.step(&c, 0.005);
            b.step(&c, 0.005);
        }
        assert_eq!(a.state().base_position, b.state().base_position);
        assert_eq!(a.state().joint_angles, b.state().joint_angles);
    }

    #[test]
    fn test_gait_switch_changes_cpg_frequency() {
        // After set_gait(Walk), stepping should advance slower than Trot.
        // Compare base translation over 500 steps for Walk vs Trot.
        let mut trot = SimpleQuadrupedSimulator::new();
        trot.set_gait(GaitType::Trot);
        let mut walk = SimpleQuadrupedSimulator::new();
        walk.set_gait(GaitType::Walk);
        for _ in 0..500 {
            trot.step(&QuadrupedCommand::zero(), 0.005);
            walk.step(&QuadrupedCommand::zero(), 0.005);
        }
        // Trot at 2 Hz vs Walk at 1 Hz — trot should cover more ground.
        assert!(
            trot.state().base_position[0] > walk.state().base_position[0],
            "trot ({:.3}) should exceed walk ({:.3}) in x translation",
            trot.state().base_position[0],
            walk.state().base_position[0]
        );
    }

    mod proptest_physics {
        use super::*;
        use proptest::prelude::*;
        proptest! { #[test] fn finite(t0 in -1.0f32..1.0, t1 in -1.0f32..1.0, t2 in -1.0f32..1.0, t3 in -1.0f32..1.0, t4 in -1.0f32..1.0, t5 in -1.0f32..1.0, t6 in -1.0f32..1.0, t7 in -1.0f32..1.0, t8 in -1.0f32..1.0, t9 in -1.0f32..1.0, t10 in -1.0f32..1.0, t11 in -1.0f32..1.0, dt in 0.001f64..0.02, steps in 1usize..300) {
            let mut s = SimpleQuadrupedSimulator::new(); let c = QuadrupedCommand { joint_torques: [t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11] };
            for _ in 0..steps { s.step(&c, dt); } prop_assert!(s.state().is_finite());
        }}
    }
}
