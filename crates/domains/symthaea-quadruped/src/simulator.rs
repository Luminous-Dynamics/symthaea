//! Simple quadruped body model.
//!
//! Architecture (2026-07 robotics deep review rework): the simulator embeds a
//! low-level "spinal" layer — a CPG generating joint targets plus a PD reflex
//! tracking them — and the commanded torques from the learned/HDC controller
//! superimpose on that reflex (descending modulation, like biological motor
//! control). Base translation is NOT scripted: it emerges from stance-leg
//! traction — feet sweeping backward while in ground contact drag the body
//! forward through a no-slip friction model. Zero net leg motion (Freeze, or
//! commands pinning the legs) produces zero net translation.
use crate::types::*;
const G: f64 = 9.81;
const GROUND_K: f64 = 5000.0;
const GROUND_D: f64 = 100.0;
/// Per-stance-leg traction gain (N per m/s of slip). Pulls the base velocity
/// toward the no-slip target; time constant ≈ mass/(gain × stance legs)
/// ≈ 20 ms for the 12 kg default with two stance legs.
const TRACTION_GAIN: f64 = 300.0;
/// Leg link length used by both foot kinematics and traction (m).
const LINK_LEN: f64 = 0.2;
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
        // Hip in phase QUADRATURE with the knee cycle. Geometry note: the
        // `sw` knee term EXTENDS the leg (knee -1.0 → -0.2 lowers the foot
        // ~15 mm), so sin(p) > 0 is the stance-loading window and
        // sin(p) <= 0 is the (relatively) unloaded swing window. With hip
        // target 0.5 + 0.2·cos(p), the stance window (p: 0→π) sweeps the
        // hip monotonically 0.7 → 0.3 — a backward sweep that produces net
        // forward traction; the swing window returns the leg forward while
        // unloaded. (The previous in-phase form 0.5 + 0.2·sin(p) returned
        // the hip to its starting angle over the stance window, which
        // yields ZERO net displacement under honest no-slip contact — the
        // old scripted base velocity was masking that.)
        [0.0, 0.5 + 0.2 * p.cos(), -1.0 + sw * sh * 10.0]
    }
    fn foot_height(&self, leg: usize) -> f64 {
        let b = leg * JOINTS_PER_LEG;
        let hf = self.state.joint_angles[b + 1];
        let k = self.state.joint_angles[b + 2];
        self.state.base_position[2] - LINK_LEN * hf.cos() - LINK_LEN * (hf + k).cos()
    }
    /// Normalized torque command the internal CPG-PD reflex layer is applying
    /// right now — the imitation target for the learned controller.
    pub fn reflex_command(&self) -> QuadrupedCommand {
        let mut t = [0.0f32; NUM_ACTUATORS];
        for leg in 0..NUM_LEGS {
            let tgt = self.cpg_target(leg);
            for j in 0..JOINTS_PER_LEG {
                let idx = leg * JOINTS_PER_LEG + j;
                let err = tgt[j] - self.state.joint_angles[idx];
                let pd = 50.0 * err - 5.0 * self.state.joint_velocities[idx];
                t[idx] = (pd / self.config.max_joint_torques[idx]).clamp(-1.0, 1.0) as f32;
            }
        }
        QuadrupedCommand { joint_torques: t }
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
        let mut fx = 0.0;
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
                // ── Traction: stance foot sweeping backward drags the body
                // forward (no-slip friction model). Foot x-velocity relative
                // to the base is the time derivative of the sagittal foot
                // offset LINK_LEN·sin(hf) + LINK_LEN·sin(hf+k).
                let b = leg * JOINTS_PER_LEG;
                let hf = self.state.joint_angles[b + 1];
                let k = self.state.joint_angles[b + 2];
                let hf_dot = self.state.joint_velocities[b + 1];
                let k_dot = self.state.joint_velocities[b + 2];
                let vfx_rel =
                    LINK_LEN * hf.cos() * hf_dot + LINK_LEN * (hf + k).cos() * (hf_dot + k_dot);
                let no_slip_target = -vfx_rel;
                fx += TRACTION_GAIN * (no_slip_target - self.state.base_linear_velocity[0]);
            } else {
                self.state.foot_contacts[leg] = 0.0;
            }
        }
        let az = (vf - self.config.body_mass * G) / self.config.body_mass;
        self.state.base_linear_velocity[2] += az * dt;
        self.state.base_position[2] += self.state.base_linear_velocity[2] * dt;
        // Sagittal translation EMERGES from stance-leg traction. Airborne
        // (no stance legs): ballistic, velocity carries. (Previously the base
        // velocity was a hardcoded constant per gait, independent of joints,
        // torques, and contact — locomotion was scripted.)
        let ax = fx / self.config.body_mass;
        self.state.base_linear_velocity[0] += ax * dt;
        self.state.base_position[0] += self.state.base_linear_velocity[0] * dt;
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
    fn test_trot_locomotion_emerges_from_traction() {
        // Forward motion must EMERGE from stance-leg traction (spinal CPG
        // sweeps the legs; contact drags the body), not from a scripted
        // velocity. Distance is therefore gait-dynamics-dependent, not a
        // constant × time.
        let mut s = SimpleQuadrupedSimulator::new();
        let x0 = s.state().base_position[0];
        for _ in 0..1000 {
            s.step(&QuadrupedCommand::zero(), 0.005);
        }
        let dist = s.state().base_position[0] - x0;
        assert!(
            dist > 0.3,
            "trot should advance via traction, got {dist:.3} m over 5 s"
        );
        assert!(
            dist < 10.0,
            "implausible speed {dist:.3} m over 5 s — traction model broken"
        );
    }

    #[test]
    fn test_locomotion_is_torque_dependent() {
        // Commands that pin the legs must kill locomotion — this test FAILS
        // against the old scripted-velocity sim, where base motion was
        // completely independent of joint torques.
        let mut baseline = SimpleQuadrupedSimulator::new();
        let mut pinned = SimpleQuadrupedSimulator::new();
        // Full-authority torque folding every joint against the CPG sweep:
        // commanded 30/30/20 N·m outmuscles the 50·err reflex for the
        // CPG's ±0.2 rad targets.
        let fold = QuadrupedCommand {
            joint_torques: [-1.0; NUM_ACTUATORS],
        };
        let x0 = baseline.state().base_position[0];
        for _ in 0..1000 {
            baseline.step(&QuadrupedCommand::zero(), 0.005);
            pinned.step(&fold, 0.005);
        }
        let d_base = baseline.state().base_position[0] - x0;
        let d_pin = pinned.state().base_position[0] - x0;
        assert!(
            d_pin.abs() < 0.5 * d_base.abs(),
            "pinning the legs must slash locomotion: baseline {d_base:.3} m vs pinned {d_pin:.3} m"
        );
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
