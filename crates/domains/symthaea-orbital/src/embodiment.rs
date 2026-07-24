use crate::controller::OrbitalController;
use crate::encoder::OrbitalHdcEncoder;
use crate::simulator::{OrbitalPhysicsSimulator, SimpleOrbitalSimulator};
use crate::types::{NUM_ACTUATORS, OrbitalConfig, OrbitalState};
pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Emergency fallback behavior for the orbital servicing arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitalFallbackStage {
    /// Zero relative-motion torque on every joint — hold station attitude
    /// and let the arm passively drift with the host spacecraft rather
    /// than risk an uncontrolled motion near other spacecraft/debris.
    Park,
}

pub struct OrbitalEmbodiment {
    ctrl: OrbitalController,
    sim: SimpleOrbitalSimulator,
    enc: OrbitalHdcEncoder,
    last_p: Option<ContinuousHV>,
    steps: usize,
    safety: MotorSafetyLevel,
    safety_ov: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    effort: f32,
    pe: f32,
    fallback_stage: OrbitalFallbackStage,
    fallback_cycles_in_stage: u32,
    stuck_thruster_fault: Option<[f32; 3]>,
}
impl OrbitalEmbodiment {
    pub fn new(g: &GenesisSeed) -> Self {
        let c = OrbitalConfig::default();
        Self {
            ctrl: OrbitalController::new(g, &c),
            sim: SimpleOrbitalSimulator::new(),
            enc: OrbitalHdcEncoder::new(g, 32),
            last_p: None,
            steps: 0,
            safety: MotorSafetyLevel::Green,
            safety_ov: None,
            moral_safety: None,
            effort: 0.0,
            pe: 0.0,
            fallback_stage: OrbitalFallbackStage::Park,
            fallback_cycles_in_stage: 0,
            stuck_thruster_fault: None,
        }
    }
    pub fn set_safety_override(&mut self, l: MotorSafetyLevel) {
        self.safety_ov = Some(l);
    }
    pub fn clear_safety_override(&mut self) {
        self.safety_ov = None;
    }
    pub fn fallback_stage(&self) -> OrbitalFallbackStage {
        self.fallback_stage
    }
    /// Read-only view of the current orbital/arm state (position, velocity,
    /// delta-v used, joint state, etc.) — for telemetry and testing.
    pub fn orbital_state(&self) -> &OrbitalState {
        self.sim.state()
    }
    /// Simulate a stuck thruster valve: this burn (m/s per axis, per step)
    /// is added to whatever the controller commands, every step, until
    /// cleared. Models a COMMAND-level fault (the fault always tries to
    /// fire) — this is what the safety gate can defend against. It does
    /// NOT model an actuator-level fault where the valve ignores a
    /// zero-command entirely; that would defeat any software gate by
    /// construction and needs a different (hardware-redundancy) mitigation,
    /// not a control-loop one.
    pub fn inject_stuck_thruster(&mut self, burn_mps: [f32; 3]) {
        self.stuck_thruster_fault = Some(burn_mps);
    }
    pub fn clear_stuck_thruster(&mut self) {
        self.stuck_thruster_fault = None;
    }
    /// Apply moral gate from ethics engine.
    /// Orbital platforms can damage other spacecraft/debris — ahimsa forces Red (park), consent Orange.
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
    pub fn step(&mut self, hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
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
        if let Some(fault) = self.stuck_thruster_fault {
            for i in 0..3 {
                cmd.translational_burn_mps[i] += fault[i];
            }
        }

        // ── SafeFallback: Park at Red (unifies the former 0.3 hard cliff
        // under the shared MotorSafetyLevel contract) ───────────────────
        // Previously `gain <= 0.3` zeroed torques at BOTH Orange (gain=0.3)
        // AND Red (gain=0.0) — collapsing the graduated Orange tier into
        // the same hard stop as Red. Only Red should trigger the full
        // Park fallback; Orange keeps its designed reduced-but-nonzero
        // authority (motor_gain() == 0.3) like every other platform.
        if matches!(self.safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            cmd.joint_torques = [0.0; NUM_ACTUATORS];
            // Park means the WHOLE bus goes passive, not just the arm: a
            // stuck-thruster or fault-injected burn command must not slip
            // through the safety gate. Added with translational_burn_mps
            // (Phase 1) -- this gate predates that field and only zeroed
            // joint_torques, which would have let a faulty burn command
            // fire right through Red.
            cmd.translational_burn_mps = [0.0; 3];
        } else {
            self.fallback_stage = OrbitalFallbackStage::Park;
            self.fallback_cycles_in_stage = 0;
            for t in &mut cmd.joint_torques {
                *t *= gain;
            }
        }
        if self.sim.state().comm_window == 0.0 && gain > 0.3 {
            for t in &mut cmd.joint_torques {
                *t *= 0.5;
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
        self.fallback_stage = OrbitalFallbackStage::Park;
        self.fallback_cycles_in_stage = 0;
        self.stuck_thruster_fault = None;
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
            platform: "orbital".to_string(),
            num_actuators: NUM_ACTUATORS,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.pe),
            platform_specific: Vec::new(),
        }
    }
}
impl symthaea_core::embodiment::EmbodimentBridge for OrbitalEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Orbital
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

impl SafeFallback for OrbitalEmbodiment {
    fn platform_name(&self) -> &'static str {
        "orbital"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        3 // Low-moderate: stationary/contained, but debris/collision risk
    }
    fn safe_fallback_description(&self) -> &'static str {
        "Park: zero relative-motion torque on every joint and zero all thruster \
         burns, hold station attitude"
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
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert!(b.step(&ContinuousHV::random(16384, 42), 0.01, 0.7).success);
    }
    #[test]
    fn test_orange_retains_scaled_authority_not_hard_zero() {
        // NOTE: this test used to be named test_orange_parks and asserted
        // control_effort < 0.01 at Orange — that was the "0.3 hard cliff"
        // bug this fix closes. `gain <= 0.3` caught BOTH Orange (gain=0.3)
        // and Red (gain=0.0), collapsing Orange into the same hard stop as
        // Red. Orange must now retain its designed reduced-but-nonzero
        // authority, matching every other platform's motor_gain() contract.
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let r = b.step(&ContinuousHV::random(16384, 42), 0.01, 0.15);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);
        assert!(
            r.control_effort > 0.0,
            "Orange must retain scaled (non-zero) authority, got {}",
            r.control_effort
        );
    }
    #[test]
    fn test_red_parks_hard_stop() {
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let r = b.step(&ContinuousHV::random(16384, 42), 0.01, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert!(
            r.control_effort < 1e-6,
            "Red must fully Park (hard zero), got {}",
            r.control_effort
        );
        assert_eq!(b.fallback_stage(), OrbitalFallbackStage::Park);
    }
    #[test]
    fn test_stuck_thruster_fault_applies_at_green() {
        // Sanity check that the fault mechanism itself works: at Green
        // (full authority), an injected stuck-thruster burn must actually
        // reach the simulator and spend delta-v. Without this, the "Red
        // blocks it" test below would be meaningless (it'd pass even if the
        // fault never did anything).
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.inject_stuck_thruster([1.0, 0.0, 0.0]);
        b.step(&ContinuousHV::random(16384, 42), 0.01, 0.9); // Green
        assert!(
            b.orbital_state().delta_v_used_m_s > 0.0,
            "stuck-thruster fault should have spent delta-v at Green"
        );
    }
    #[test]
    fn test_stuck_thruster_fault_zeroed_at_red() {
        // The safety-tier cascade (Phase 2 scenario: stuck-thruster fault):
        // a fault-injected burn command must NOT slip through Red-tier
        // Park, even though it keeps trying to fire every step.
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.inject_stuck_thruster([1.0, 0.0, 0.0]);
        b.step(&ContinuousHV::random(16384, 42), 0.01, 0.05); // Red
        assert_eq!(
            b.orbital_state().delta_v_used_m_s,
            0.0,
            "Red-tier Park must zero a stuck-thruster fault, not just arm torques"
        );
    }
    #[test]
    fn test_clear_stuck_thruster_stops_further_burns() {
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        b.inject_stuck_thruster([1.0, 0.0, 0.0]);
        b.step(&hv, 0.01, 0.9); // Green: fault applies
        let used_after_fault = b.orbital_state().delta_v_used_m_s;
        assert!(used_after_fault > 0.0);
        b.clear_stuck_thruster();
        b.step(&hv, 0.01, 0.9); // Green: no fault, no further burn
        assert_eq!(b.orbital_state().delta_v_used_m_s, used_after_fault);
    }
    #[test]
    fn test_safe_fallback_trait_impl() {
        let b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert_eq!(b.platform_name(), "orbital");
        assert_eq!(b.safe_fallback_priority(), 3);
        assert_eq!(b.safe_fallback_latency_cycles(), 1);
        assert!(b.safe_fallback_description().contains("Park"));
    }
    #[test]
    fn test_safe_fallback_active_at_orange_and_red() {
        // safe_fallback_active() is defined (by the shared trait default) as
        // Red OR Orange, even though only Red triggers the hard Park zero —
        // Orange is still "fallback active" in the sense of reduced
        // authority per the trait's own documented invariant.
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        b.step(&hv, 0.01, 0.9); // Green
        assert!(!b.safe_fallback_active());
        b.step(&hv, 0.01, 0.15); // Orange
        assert!(b.safe_fallback_active());
        b.step(&hv, 0.01, 0.05); // Red
        assert!(b.safe_fallback_active());
    }
    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let r = b.step(&ContinuousHV::random(16384, 42), 0.01, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "ahimsa must force Red regardless of phi"
        );
        assert!(
            r.control_effort < 0.01,
            "Red must park, got effort {}",
            r.control_effort
        );
    }

    #[test]
    fn test_num_actuators() {
        use symthaea_core::embodiment::EmbodimentBridge;
        let b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert_eq!(b.num_actuators(), NUM_ACTUATORS);
        assert_eq!(NUM_ACTUATORS, 7, "orbital manipulator has 7 actuators");
    }

    #[test]
    fn test_platform_is_orbital() {
        use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform};
        let b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert_eq!(b.platform(), EmbodimentPlatform::Orbital);
    }

    #[test]
    fn test_telemetry_populated_after_step() {
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let t0 = b.telemetry();
        assert_eq!(t0.total_steps, 0);
        assert_eq!(t0.platform, "orbital");
        b.step(&ContinuousHV::random(16384, 42), 0.01, 0.7);
        let t1 = b.telemetry();
        assert_eq!(t1.total_steps, 1);
        assert_eq!(t1.num_actuators, NUM_ACTUATORS);
        assert!(t1.control_effort.is_finite());
    }

    #[test]
    fn test_safety_override_raises_tier() {
        // Phi of 0.9 would give Green; Red override should force Red.
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.set_safety_override(MotorSafetyLevel::Red);
        let r = b.step(&ContinuousHV::random(16384, 42), 0.01, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        // Red triggers the Park SafeFallback: hard zero.
        assert!(r.control_effort < 0.01);
    }

    #[test]
    fn test_safety_override_cleared_returns_to_phi_tier() {
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.set_safety_override(MotorSafetyLevel::Red);
        b.step(&ContinuousHV::random(16384, 42), 0.01, 0.9);
        b.clear_safety_override();
        let r = b.step(&ContinuousHV::random(16384, 42), 0.01, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_total_steps_increments_monotonically() {
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        for expected in 1..=15 {
            b.step(&hv, 0.01, 0.7);
            assert_eq!(b.total_steps(), expected);
        }
    }

    #[test]
    fn test_reset_clears_accumulated_state() {
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        for _ in 0..50 {
            b.step(&hv, 0.01, 0.7);
        }
        b.set_safety_override(MotorSafetyLevel::Orange);
        assert_eq!(b.total_steps(), 50);
        b.reset();
        assert_eq!(b.total_steps(), 0);
        // Reset must clear the override too; next step at phi 0.9 → Green.
        let r = b.step(&hv, 0.01, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_extended_run_stays_stable() {
        // Long run under nominal phi — no panics, all results finite.
        let mut b = OrbitalEmbodiment::new(&GenesisSeed::from_phrase("long_run"));
        let hv = ContinuousHV::random(16384, 42);
        for _ in 0..300 {
            let r = b.step(&hv, 0.01, 0.7);
            assert!(r.success);
            assert!(r.control_effort.is_finite());
            assert!(r.prediction_error.is_finite());
        }
    }
}
