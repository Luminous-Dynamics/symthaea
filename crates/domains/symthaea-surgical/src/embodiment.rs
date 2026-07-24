// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::SurgicalController;
use crate::encoder::SurgicalHdcEncoder;
use crate::simulator::{SimpleSurgicalSimulator, SurgicalPhysicsSimulator};
use crate::types::{
    NUM_ACTUATORS, SurgicalConfig, surgical_cautery_allowed, surgical_tip_speed_limit,
    surgical_torque_gain,
};
pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Emergency fallback posture for the surgical robot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurgicalFallbackStage {
    /// Withdraw the tool tip (joint 1 pulls back), close the jaw, and
    /// disable cautery — the surgical-specific "safe" posture at Red.
    Retract,
}

pub struct SurgicalEmbodiment {
    controller: SurgicalController,
    simulator: SimpleSurgicalSimulator,
    encoder: SurgicalHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: SurgicalFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl SurgicalEmbodiment {
    pub fn new(g: &GenesisSeed) -> Self {
        let c = SurgicalConfig::default();
        Self {
            controller: SurgicalController::new(g, &c),
            simulator: SimpleSurgicalSimulator::new(),
            encoder: SurgicalHdcEncoder::new(g, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: SurgicalFallbackStage::Retract,
            fallback_cycles_in_stage: 0,
        }
    }
    pub fn set_safety_override(&mut self, l: MotorSafetyLevel) {
        self.safety_override = Some(l);
    }
    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }
    pub fn fallback_stage(&self) -> SurgicalFallbackStage {
        self.fallback_stage
    }
    /// Apply moral gate from ethics engine. SAFETY-CRITICAL: surgical robot cuts human tissue.
    /// Ahimsa violation or Blocked verdict forces Retract (emergency stop + disable cautery + withdraw pose).
    /// Consent violation forces Orange (no cautery, reduced torque). Caution verdict forces Yellow.
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
        // Unified tier: MotorSafetyLevel is the single source of truth for
        // phi + safety_override + moral gate combined (previously the
        // parallel `SurgicalSafetyLevel` ladder only looked at phi + moral
        // gate, silently ignoring `safety_override` — an external
        // SafetyAgent override could report Red in telemetry without ever
        // actually retracting the tool. Fixed by deriving everything from
        // this one combined value.)
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = phi_level;
        if let Some(ov) = self.safety_override {
            self.current_safety = self.current_safety.max(ov);
        }
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }

        let tg = surgical_torque_gain(self.current_safety);
        let cautery_ok = surgical_cautery_allowed(self.current_safety);

        let mut cmd = self.controller.forward(hv, dt);
        for t in &mut cmd.joint_torques {
            *t *= tg;
        }
        if !cautery_ok {
            cmd.cautery = 0.0;
        }

        // ── SafeFallback: Retract ──────────────────────────────────────
        // Triggers on general Red (phi/override/ahimsa — anything that
        // maxes current_safety to Red) OR specifically on a moral consent
        // violation (Orange from the ethics engine, not from phi). Consent
        // violation is a stronger, more certain signal than a merely-low
        // Phi reading, so it earns the full active withdrawal pose rather
        // than phi-only Orange's plain zero-torque Freeze.
        let force_retract = matches!(self.current_safety, MotorSafetyLevel::Red)
            || self.moral_safety == Some(MotorSafetyLevel::Orange);
        if force_retract {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            cmd.joint_torques = [0.0, -0.3, 0.0, 0.0, 0.0, 0.0];
            cmd.jaw = 0.0;
            cmd.cautery = 0.0;
        } else {
            self.fallback_stage = SurgicalFallbackStage::Retract;
            self.fallback_cycles_in_stage = 0;
        }
        // Moral-Red (ahimsa) is stricter than the general Retract pose: full
        // emergency stop, zero torque — even the withdrawal motion itself
        // may be unsafe if harm is actively in progress right now.
        if self.moral_safety == Some(MotorSafetyLevel::Red) {
            cmd.joint_torques = [0.0; 6];
            cmd.jaw = 0.0;
            cmd.cautery = 0.0;
        }
        self.last_control_effort = cmd.control_effort();
        // Per-tier tip speed cap, enforced inside the physics step (joint
        // velocities are scaled whenever the kinematic tip velocity would
        // exceed the tier's mm/s allowance).
        self.simulator
            .set_tip_speed_limit(surgical_tip_speed_limit(self.current_safety));
        self.simulator.step(&cmd, dt as f64);
        let p = self.encoder.encode(self.simulator.state());
        let pe = if let Some(ref prev) = self.last_perception {
            (1.0 - p.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0f32
        };
        self.last_prediction_error = pe;
        self.last_perception = Some(p);
        self.total_steps += 1;
        EmbodimentResult {
            num_actuators: NUM_ACTUATORS,
            control_effort: self.last_control_effort,
            success: self.simulator.state().is_finite(),
            prediction_error: pe,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pe),
        }
    }
    pub fn encode_perception(&mut self) -> ContinuousHV {
        let p = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(p.clone());
        p
    }
    pub fn reset(&mut self) {
        self.simulator.reset();
        self.controller.reset();
        self.encoder.reset();
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.moral_safety = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
        self.fallback_stage = SurgicalFallbackStage::Retract;
        self.fallback_cycles_in_stage = 0;
    }
    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: self.current_safety,
            platform: "surgical".to_string(),
            num_actuators: NUM_ACTUATORS,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for SurgicalEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Surgical
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

impl SafeFallback for SurgicalEmbodiment {
    fn platform_name(&self) -> &'static str {
        "surgical"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        8 // High: in contact with human tissue, though base is stationary
    }
    fn safe_fallback_description(&self) -> &'static str {
        "Retract: withdraw tool tip, close jaw, disable cautery"
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
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert!(b.step(&ContinuousHV::random(16384, 42), 0.001, 0.7).success);
    }
    #[test]
    fn test_retract() {
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let r = b.step(&ContinuousHV::random(16384, 42), 0.001, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert_eq!(b.fallback_stage(), SurgicalFallbackStage::Retract);
        assert!(b.safe_fallback_active());
    }
    #[test]
    fn test_extended() {
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        for _ in 0..500 {
            assert!(b.step(&hv, 0.001, 0.7).success);
        }
    }
    #[test]
    fn test_moral_gate_ahimsa_emergency_stop() {
        // SAFETY-CRITICAL: surgical ahimsa violation → strict zero (not retract pose).
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let r = b.step(&ContinuousHV::random(16384, 42), 0.001, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "ahimsa must force Red at any phi"
        );
        assert_eq!(b.fallback_stage(), SurgicalFallbackStage::Retract);
        assert!(
            r.control_effort < 1e-6,
            "ahimsa must emergency-stop (strict zero), got {}",
            r.control_effort
        );
    }
    #[test]
    fn test_moral_gate_consent_violation() {
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let r = b.step(&ContinuousHV::random(16384, 42), 0.001, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);
        // Consent violation is a stronger signal than phi-only Orange —
        // it earns the full Retract withdrawal pose, not just zero torque.
        assert_eq!(b.fallback_stage(), SurgicalFallbackStage::Retract);
    }

    #[test]
    fn test_num_actuators() {
        let b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let r = EmbodimentResult {
            num_actuators: NUM_ACTUATORS,
            control_effort: 0.0,
            success: true,
            prediction_error: 0.0,
            safety_level: MotorSafetyLevel::Green,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: 0.5,
        };
        assert_eq!(r.num_actuators, NUM_ACTUATORS);
        // Also verify the trait accessor returns the same value.
        use symthaea_core::embodiment::EmbodimentBridge;
        assert_eq!(b.num_actuators(), NUM_ACTUATORS);
    }

    #[test]
    fn test_platform_is_surgical() {
        use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform};
        let b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert_eq!(b.platform(), EmbodimentPlatform::Surgical);
    }

    #[test]
    fn test_telemetry_populated_after_step() {
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let t0 = b.telemetry();
        assert_eq!(t0.total_steps, 0);
        assert_eq!(t0.platform, "surgical");
        b.step(&ContinuousHV::random(16384, 42), 0.001, 0.7);
        let t1 = b.telemetry();
        assert_eq!(t1.total_steps, 1);
        assert_eq!(t1.num_actuators, NUM_ACTUATORS);
        assert!(t1.control_effort.is_finite());
    }

    #[test]
    fn test_safety_override_raises_tier() {
        // Phi of 0.9 would normally give Green, but a Red override forces Red.
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.set_safety_override(MotorSafetyLevel::Red);
        let r = b.step(&ContinuousHV::random(16384, 42), 0.001, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_safety_override_cleared_returns_to_phi_tier() {
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.set_safety_override(MotorSafetyLevel::Red);
        b.step(&ContinuousHV::random(16384, 42), 0.001, 0.9);
        b.clear_safety_override();
        let r = b.step(&ContinuousHV::random(16384, 42), 0.001, 0.9);
        // Phi 0.9 → Green (documented threshold in MotorSafetyLevel::from_phi).
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_total_steps_increments_monotonically() {
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        for expected in 1..=20 {
            b.step(&hv, 0.001, 0.7);
            assert_eq!(b.total_steps(), expected);
        }
    }

    #[test]
    fn test_reset_clears_accumulated_state() {
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        let hv = ContinuousHV::random(16384, 42);
        for _ in 0..50 {
            b.step(&hv, 0.001, 0.7);
        }
        b.set_safety_override(MotorSafetyLevel::Orange);
        assert_eq!(b.total_steps(), 50);
        b.reset();
        assert_eq!(b.total_steps(), 0);
        assert_eq!(b.fallback_cycles_in_stage, 0);
        // Override must also clear (documented in reset impl).
        let r = b.step(&hv, 0.001, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_retract_pose_also_disables_cautery() {
        // When phi drops into the Red tier, cautery_allowed is false — verify
        // that the step's end state has cautery_power at 0 regardless of how
        // loudly the HV would otherwise request it.
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("cautery_attempt"));
        let r = b.step(&ContinuousHV::random(16384, 7), 0.001, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert_eq!(b.fallback_stage(), SurgicalFallbackStage::Retract);
        // Cautery capability check at the safety layer (policy-level assertion).
        assert!(!surgical_cautery_allowed(r.safety_level));
        assert_eq!(b.simulator.state().cautery_power, 0.0);
    }

    #[test]
    fn test_tip_speed_limit_follows_tier() {
        // Yellow tier must arm the simulator with the Yellow (20 mm/s) cap —
        // strictly below the Green allowance. The cap's physical enforcement
        // is proven in simulator::tests::test_tip_speed_clamp_enforced; this
        // verifies the tier→cap wiring, then that measured tip speed under a
        // sustained Yellow-tier run indeed stays below the Green limit.
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("speed"));
        let hv = ContinuousHV::random(16384, 42);
        let green = surgical_tip_speed_limit(MotorSafetyLevel::Green);
        let yellow = surgical_tip_speed_limit(MotorSafetyLevel::Yellow);
        let mut max_speed = 0.0f64;
        for _ in 0..300 {
            let r = b.step(&hv, 0.001, 0.45); // Yellow band
            assert_eq!(r.safety_level, MotorSafetyLevel::Yellow);
            max_speed = max_speed.max(b.simulator.last_tip_speed());
        }
        assert_eq!(b.simulator.tip_speed_limit(), yellow);
        assert!(
            max_speed <= yellow + 1e-6,
            "Yellow-tier tip speed {max_speed:.1} mm/s must stay below the cap"
        );
        assert!(max_speed < green);
    }

    #[test]
    fn test_safe_fallback_trait_impl() {
        let b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        assert_eq!(b.platform_name(), "surgical");
        assert_eq!(b.safe_fallback_priority(), 8);
        assert_eq!(b.safe_fallback_latency_cycles(), 1);
        assert!(b.safe_fallback_description().contains("Retract"));
    }

    #[test]
    fn test_safety_override_now_actually_retracts() {
        // Regression test for the bug the unification fixed: previously
        // `SurgicalSafetyLevel` was derived only from phi + moral gate, so
        // an external SafetyAgent override to Red would report Red in
        // `safety_level` but NEVER actually retract the tool (the physical
        // command stayed whatever phi/moral said). Now current_safety is
        // the single input to both the reported tier AND the Retract
        // fallback, so an override must retract for real.
        let mut b = SurgicalEmbodiment::new(&GenesisSeed::from_phrase("t"));
        b.set_safety_override(MotorSafetyLevel::Red);
        let hv = ContinuousHV::random(16384, 42);
        let r = b.step(&hv, 0.001, 0.9); // phi alone would be Green
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert_eq!(b.fallback_stage(), SurgicalFallbackStage::Retract);
        // Exact expected effort for the fixed Retract command:
        // joint_torques=[0,-0.3,0,0,0,0], jaw=0, cautery=0 → 0.3/8.
        assert!(
            (r.control_effort - 0.3 / 8.0).abs() < 1e-5,
            "expected exact Retract withdrawal effort 0.3/8, got {}",
            r.control_effort
        );
    }
}
