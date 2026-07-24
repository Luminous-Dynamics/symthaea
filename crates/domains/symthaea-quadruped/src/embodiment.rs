use crate::controller::QuadrupedController;
use crate::encoder::QuadrupedHdcEncoder;
use crate::simulator::{QuadrupedBackend, QuadrupedPhysicsSimulator, SimpleQuadrupedSimulator};
use crate::types::{GaitType, NUM_ACTUATORS, QuadrupedCommand, QuadrupedConfig};
pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Emergency fallback posture for the quadruped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadrupedFallbackStage {
    /// Fold hip/thigh/knee joints into a stable, low crouch — a controlled
    /// sit/lie-down rather than a passive collapse under zero torque.
    SitDown,
}

pub struct QuadrupedEmbodiment {
    ctrl: QuadrupedController,
    sim: QuadrupedBackend,
    enc: QuadrupedHdcEncoder,
    last_p: Option<ContinuousHV>,
    steps: usize,
    safety: MotorSafetyLevel,
    safety_ov: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    effort: f32,
    pe: f32,
    gait: GaitType,
    fallback_stage: QuadrupedFallbackStage,
    fallback_cycles_in_stage: u32,
}
impl QuadrupedEmbodiment {
    pub fn new(g: &GenesisSeed) -> Self {
        Self::with_backend(g, QuadrupedBackend::Simple(SimpleQuadrupedSimulator::new()))
    }

    /// Construct with the real rigid-body physics backend (GJK/EPA contact
    /// via `symtropy-physics`) instead of the default scripted CPG-PD model.
    /// No gait scheduling — the network's torques are the only input.
    #[cfg(feature = "symtropy")]
    pub fn new_symtropy(g: &GenesisSeed) -> Self {
        Self::with_backend(
            g,
            QuadrupedBackend::Symtropy(crate::symtropy_sim::SymtropyQuadrupedSimulator::new()),
        )
    }

    fn with_backend(g: &GenesisSeed, sim: QuadrupedBackend) -> Self {
        let c = QuadrupedConfig::default();
        Self {
            ctrl: QuadrupedController::new(g, &c),
            sim,
            enc: QuadrupedHdcEncoder::new(g, 32),
            last_p: None,
            steps: 0,
            safety: MotorSafetyLevel::Green,
            safety_ov: None,
            moral_safety: None,
            effort: 0.0,
            pe: 0.0,
            gait: GaitType::Trot,
            fallback_stage: QuadrupedFallbackStage::SitDown,
            fallback_cycles_in_stage: 0,
        }
    }
    pub fn set_safety_override(&mut self, l: MotorSafetyLevel) {
        self.safety_ov = Some(l);
    }
    pub fn clear_safety_override(&mut self) {
        self.safety_ov = None;
    }
    pub fn fallback_stage(&self) -> QuadrupedFallbackStage {
        self.fallback_stage
    }
    /// Fold hip/thigh/knee joints into a stable low crouch instead of
    /// passively zeroing torque. Executes at full authority (not scaled by
    /// motor_gain) per the SafeFallback contract.
    ///
    /// Bias is strongly negative on thigh+knee: this pulls the simulated
    /// foot position up and away from the ground-contact plane, so the
    /// ground-reaction spring stops supporting body weight and the body
    /// settles to a genuinely lower height — not just a frozen stance.
    fn apply_sit_down(&self, cmd: &mut QuadrupedCommand) {
        const HIP_BIAS: f32 = 0.0;
        const THIGH_BIAS: f32 = -1.0;
        const KNEE_BIAS: f32 = -1.0;
        for leg in 0..4 {
            cmd.joint_torques[leg * 3] = HIP_BIAS;
            cmd.joint_torques[leg * 3 + 1] = THIGH_BIAS;
            cmd.joint_torques[leg * 3 + 2] = KNEE_BIAS;
        }
    }
    /// Apply moral gate from ethics engine.
    /// A quadruped can step on beings — ahimsa must force Red, consent violation Orange.
    pub fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.moral_safety =
            if gate.ahimsa_violated || gate.verdict == MoralGateInput::VERDICT_BLOCKED {
                Some(MotorSafetyLevel::Red)
            } else if gate.consent_violation {
                Some(MotorSafetyLevel::Orange)
            } else if gate.verdict == MoralGateInput::VERDICT_CAUTION {
                Some(MotorSafetyLevel::Yellow)
            } else {
                None
            };
    }
    pub fn gait(&self) -> GaitType {
        self.gait
    }
    pub fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.gait = GaitType::from_phi(phi);
        self.sim.set_gait(self.gait);
        let pl = MotorSafetyLevel::from_phi(phi);
        self.safety = pl;
        if let Some(ov) = self.safety_ov {
            self.safety = self.safety.max(ov);
        }
        if let Some(m) = self.moral_safety {
            self.safety = self.safety.max(m);
        }
        let gain = self.safety.motor_gain();
        let mut cmd = self.ctrl.forward(hv, dt);

        // ── SafeFallback: SitDown at Red ─────────────────────────────
        // Zero torque alone leaves the legs frozen mid-stance, not a
        // controlled sit/lie-down — a being under the robot's feet needs
        // the robot to actively lower itself, not just stop.
        if matches!(self.safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_sit_down(&mut cmd);
        } else {
            self.fallback_stage = QuadrupedFallbackStage::SitDown;
            self.fallback_cycles_in_stage = 0;
            if gain < 1.0 {
                for t in &mut cmd.joint_torques {
                    *t *= gain;
                }
            }
        }
        self.effort = cmd.control_effort();
        self.sim.step(&cmd, dt as f64);
        let p = self.enc.encode(self.sim.state());
        self.pe = if let Some(ref prev) = self.last_p {
            (1.0 - p.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0
        };
        self.last_p = Some(p);
        self.steps += 1;
        EmbodimentResult {
            num_actuators: NUM_ACTUATORS,
            control_effort: self.effort,
            success: self.sim.state().is_finite(),
            prediction_error: self.pe,
            safety_level: self.safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(self.pe),
        }
    }
    pub fn encode_perception(&mut self) -> ContinuousHV {
        let p = self.enc.encode(self.sim.state());
        self.last_p = Some(p.clone());
        p
    }
    pub fn reset(&mut self) {
        self.sim.reset();
        self.ctrl.reset();
        self.enc.reset();
        self.last_p = None;
        self.steps = 0;
        self.safety = MotorSafetyLevel::Green;
        self.safety_ov = None;
        self.moral_safety = None;
        self.effort = 0.0;
        self.pe = 0.0;
        self.gait = GaitType::Trot;
        self.fallback_stage = QuadrupedFallbackStage::SitDown;
        self.fallback_cycles_in_stage = 0;
    }
    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.safety
    }
    pub fn total_steps(&self) -> usize {
        self.steps
    }
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.steps as u64,
            control_effort: self.effort,
            prediction_error: self.pe,
            safety_level: self.safety,
            platform: "quadruped".to_string(),
            num_actuators: NUM_ACTUATORS,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.pe),
            platform_specific: Vec::new(),
        }
    }
}
impl symthaea_core::embodiment::EmbodimentBridge for QuadrupedEmbodiment {
    fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.step(hv, dt, phi)
    }
    fn encode_perception(&mut self) -> ContinuousHV {
        self.encode_perception()
    }
    fn reset(&mut self) {
        self.reset()
    }
    fn safety_level(&self) -> MotorSafetyLevel {
        self.safety_level()
    }
    fn set_safety_override(&mut self, l: MotorSafetyLevel) {
        self.set_safety_override(l)
    }
    fn clear_safety_override(&mut self) {
        self.clear_safety_override()
    }
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::Quadruped
    }
    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }
    fn total_steps(&self) -> usize {
        self.total_steps()
    }
    fn telemetry(&self) -> EmbodimentTelemetry {
        self.telemetry()
    }
    fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.apply_moral_gate(gate)
    }
}

impl SafeFallback for QuadrupedEmbodiment {
    fn platform_name(&self) -> &'static str {
        "quadruped"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        6 // Moderate-high: mobile, in contact with ground/beings
    }
    fn safe_fallback_description(&self) -> &'static str {
        "SitDown: fold hip/thigh/knee joints into a stable low crouch, halt locomotion"
    }
    fn safe_fallback_latency_cycles(&self) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_step() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert!(b.step(&ContinuousHV::random(16384, 42), 0.005, 0.7).success);
        assert_eq!(b.gait(), GaitType::Trot);
    }
    #[test]
    fn test_gait_switch() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.step(&ContinuousHV::random(16384, 42), 0.005, 0.05);
        assert_eq!(b.gait(), GaitType::Collapse);
    }
    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "ahimsa must force Red even at high phi"
        );
        // NOTE: this used to assert control_effort < 0.01 — that was the bug
        // this fix closes. A quadruped standing frozen mid-stance over
        // something it's stepped on is not safe; Red now actively commands
        // SitDown (a deliberate, non-zero fold-down torque), not a passive
        // freeze. See test_red_sit_down_* below for what this actually
        // asserts about the commanded torques.
        assert!(
            r.control_effort > 0.1,
            "Red must command SitDown (non-zero fold torque), got {}",
            r.control_effort
        );
    }
    #[test]
    fn test_moral_gate_clears() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_BLOCKED,
            consent_violation: false,
            ahimsa_violated: false,
        });
        b.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Green,
            "clearing moral gate must return to phi-derived safety"
        );
    }

    #[test]
    fn test_num_actuators() {
        use symthaea_core::embodiment::EmbodimentBridge;
        let b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert_eq!(b.num_actuators(), NUM_ACTUATORS);
        assert_eq!(NUM_ACTUATORS, 12, "quadruped: 4 legs × 3 joints = 12");
    }

    #[test]
    fn test_platform_is_quadruped() {
        use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform};
        let b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert_eq!(b.platform(), EmbodimentPlatform::Quadruped);
    }

    #[test]
    fn test_telemetry_populated_after_step() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let t0 = b.telemetry();
        assert_eq!(t0.total_steps, 0);
        assert_eq!(t0.platform, "quadruped");
        b.step(&ContinuousHV::random(16384, 42), 0.005, 0.7);
        let t1 = b.telemetry();
        assert_eq!(t1.total_steps, 1);
        assert_eq!(t1.num_actuators, NUM_ACTUATORS);
        assert!(t1.control_effort.is_finite());
    }

    #[test]
    fn test_safety_override_raises_tier() {
        // Phi 0.9 → Green normally; Red override forces Red.
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.set_safety_override(MotorSafetyLevel::Red);
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_safety_override_cleared() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.set_safety_override(MotorSafetyLevel::Red);
        b.step(&ContinuousHV::random(16384, 42), 0.005, 0.9);
        b.clear_safety_override();
        let r = b.step(&ContinuousHV::random(16384, 42), 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_total_steps_increments_monotonically() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        for expected in 1..=15 {
            b.step(&hv, 0.005, 0.7);
            assert_eq!(b.total_steps(), expected);
        }
    }

    #[test]
    fn test_reset_clears_accumulated_state() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        for _ in 0..50 {
            b.step(&hv, 0.005, 0.7);
        }
        b.set_safety_override(MotorSafetyLevel::Orange);
        assert_eq!(b.total_steps(), 50);
        b.reset();
        assert_eq!(b.total_steps(), 0);
        assert_eq!(b.gait(), GaitType::Trot, "reset restores default Trot gait");
        let r = b.step(&hv, 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_extended_run_finite_and_monotonic() {
        // Long run under nominal phi — every step produces finite state.
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("long_run"));
        let hv = ContinuousHV::random(16384, 42);
        for _ in 0..300 {
            let r = b.step(&hv, 0.005, 0.7);
            assert!(r.success);
            assert!(r.control_effort.is_finite());
            assert!(r.prediction_error.is_finite());
        }
        assert_eq!(b.total_steps(), 300);
    }

    #[test]
    fn test_red_triggers_sit_down_stage() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.step(&ContinuousHV::random(16384, 42), 0.005, 0.05);
        assert_eq!(b.fallback_stage(), QuadrupedFallbackStage::SitDown);
    }

    #[test]
    fn test_red_sit_down_lowers_body_height_over_time() {
        // The core safety fix: Red must not just freeze mid-stance — it must
        // actively fold the legs into a stable, lower crouch. Confirm the
        // body height genuinely drops from the standing height over
        // sustained Red steps (not just that torques are non-zero).
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        let standing_height = b.sim.state().height();
        for _ in 0..2000 {
            let r = b.step(&hv, 0.005, 0.05); // Red
            assert!(r.success, "state must stay finite while sitting down");
        }
        let final_height = b.sim.state().height();
        assert!(
            final_height < standing_height - 0.05,
            "SitDown should lower body height: standing={standing_height}, final={final_height}"
        );
    }

    #[test]
    fn test_red_sit_down_commands_fixed_fold_torques_not_scaled_output() {
        // At Red, joint torques must be the fixed SitDown bias
        // (hip=0, thigh=-1.0, knee=-1.0 per leg), not the gain-scaled
        // (and thus zeroed) neural controller output.
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        let r = b.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        // Mean |torque| over 12 joints with 4×(0,1,1) = 8/12.
        assert!(
            (r.control_effort - (8.0 / 12.0)).abs() < 1e-5,
            "expected fixed SitDown fold torque, got {}",
            r.control_effort
        );
    }

    #[test]
    fn test_safe_fallback_trait_impl() {
        let b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert_eq!(b.platform_name(), "quadruped");
        assert_eq!(b.safe_fallback_priority(), 6);
        assert_eq!(b.safe_fallback_latency_cycles(), 1);
        assert!(b.safe_fallback_description().contains("SitDown"));
    }

    #[test]
    fn test_safe_fallback_active_iff_red_or_orange() {
        let mut b = QuadrupedEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        b.step(&hv, 0.005, 0.9); // Green
        assert!(!b.safe_fallback_active());
        b.step(&hv, 0.005, 0.05); // Red
        assert!(b.safe_fallback_active());
    }

    #[cfg(feature = "symtropy")]
    #[test]
    fn test_symtropy_backend_is_actually_driven() {
        // Robotics plan 2026-07-10 Tier 4.4: SymtropyQuadrupedSimulator
        // existed and implemented the trait but nothing ever constructed
        // one through the embodiment bridge. Confirm it now does, and that
        // stepping it actually evolves real rigid-body state (not just
        // constructs successfully).
        let mut b = QuadrupedEmbodiment::new_symtropy(&GenesisSeed::from_phrase("symtropy-t"));
        let hv = ContinuousHV::random(16384, 42);
        let z0 = b.sim.state().base_position[2];
        for _ in 0..200 {
            let r = b.step(&hv, 0.005, 0.7);
            assert!(r.success, "symtropy backend must stay finite");
        }
        let z1 = b.sim.state().base_position[2];
        assert_ne!(
            z0, z1,
            "symtropy backend must actually evolve under gravity/contact, not sit frozen"
        );
    }

    #[cfg(feature = "symtropy")]
    #[test]
    fn test_symtropy_backend_safety_gating_still_applies() {
        // The Phi->safety gate composes above the backend choice — Red
        // SitDown must still engage on the real-physics backend.
        let mut b = QuadrupedEmbodiment::new_symtropy(&GenesisSeed::from_phrase("symtropy-t2"));
        let hv = ContinuousHV::random(16384, 42);
        let r = b.step(&hv, 0.005, 0.05); // Red
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert_eq!(b.fallback_stage(), QuadrupedFallbackStage::SitDown);
    }
}
