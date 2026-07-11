// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::SubterraneanController;
use crate::encoder::SubterraneanHdcEncoder;
use crate::simulator::{SimpleSubterraneanSimulator, SubterraneanPhysicsSimulator};
use crate::types::SubterraneanConfig;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

/// Emergency fallback for a mole/boring scout.
///
/// Zeroing every actuator at Red (the trait's plain default, `motor_gain=0`)
/// is dangerous here: it kills `thermal_pump` (cooling) at exactly the
/// moment the cutter is most likely to be running hot, with no reflexive
/// behavior to compensate. (Found in the 2026-07-07 unaudited-platforms
/// review; this crate had no `SafeFallback` at all before this fix.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubterraneanFallbackStage {
    /// Stop boring/feed/drive (cutter_head, auger_feed, tracks all zero —
    /// no more heat generation, no further hazard exposure), thermal_pump at
    /// full authority (not gain-scaled) to arrest any developing overheat.
    VentAndRetreat,
}

pub struct SubterraneanEmbodiment {
    controller: SubterraneanController,
    simulator: SimpleSubterraneanSimulator,
    encoder: SubterraneanHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: SubterraneanFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl SubterraneanEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = SubterraneanConfig::default();
        Self {
            controller: SubterraneanController::new(genesis, &config),
            simulator: SimpleSubterraneanSimulator::new(),
            encoder: SubterraneanHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: SubterraneanFallbackStage::VentAndRetreat,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Apply moral gate from the ethics engine. Ahimsa forces Red
    /// (VentAndRetreat), a consent violation forces Orange, caution forces
    /// a Yellow cap. Previously this crate never overrode the trait's no-op
    /// default.
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

    /// SafeFallback: VentAndRetreat. Stops boring/feed/drive and commands
    /// full thermal_pump authority regardless of `motor_gain`.
    fn apply_vent_and_retreat(&self, cmd: &mut crate::types::SubterraneanCommand) {
        *cmd = crate::types::SubterraneanCommand::zero();
        cmd.torques[5] = 1.0; // thermal_pump: full authority, never zero
        // cutter_head/auger_feed/left_track/right_track/ballast_trim: 0.0 —
        // stop generating heat and stop further hazard exposure.
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(o) => phi_level.max(o),
            None => phi_level,
        };
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }
        let gain = self.current_safety.motor_gain();
        let mut cmd = self.controller.forward(thought_hv, dt);

        // ── SafeFallback: VentAndRetreat at Red ──────────────────────
        // Zeroing every actuator at Red (motor_gain=0) would kill the
        // thermal pump right when the cutter is most likely overheating.
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_vent_and_retreat(&mut cmd);
        } else {
            self.fallback_stage = SubterraneanFallbackStage::VentAndRetreat;
            self.fallback_cycles_in_stage = 0;
            if gain < 1.0 {
                for t in &mut cmd.torques {
                    *t *= gain;
                }
            }
        }
        self.last_control_effort = cmd.control_effort();
        self.simulator.step(&cmd, dt as f64);
        let perception = self.encoder.encode(self.simulator.state());
        let pe = if let Some(ref prev) = self.last_perception {
            (1.0 - perception.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0
        };
        self.last_prediction_error = pe;
        self.last_perception = Some(perception);
        self.total_steps += 1;
        EmbodimentResult {
            num_actuators: 6,
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
        self.fallback_stage = SubterraneanFallbackStage::VentAndRetreat;
        self.fallback_cycles_in_stage = 0;
    }
    pub fn fallback_stage(&self) -> SubterraneanFallbackStage {
        self.fallback_stage
    }
    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }
    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "subterranean".to_string(),
            num_actuators: 6,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for SubterraneanEmbodiment {
    fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        self.step(thought_hv, dt, phi)
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

    fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.set_safety_override(level)
    }

    fn clear_safety_override(&mut self) {
        self.clear_safety_override()
    }

    fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.apply_moral_gate(gate)
    }

    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::Subterranean
    }

    fn num_actuators(&self) -> usize {
        6
    }

    fn total_steps(&self) -> usize {
        self.total_steps()
    }

    fn telemetry(&self) -> EmbodimentTelemetry {
        self.telemetry()
    }
}

impl SafeFallback for SubterraneanEmbodiment {
    fn platform_name(&self) -> &'static str {
        "subterranean"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        5 // High: enclosed/underground, thermal risk to the vehicle itself
    }
    fn safe_fallback_description(&self) -> &'static str {
        "VentAndRetreat: stop boring/feed/drive, full thermal_pump authority to arrest overheat"
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
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.7);
        assert!(r.success);
    }
    #[test]
    fn test_red_halts() {
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_red_does_not_zero_thermal_pump() {
        // Regression: asserts the RESULTING COMMAND, not just the
        // safety-tier enum — the exact gap that let the "Red kills cooling"
        // bug ship silently.
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let mut cmd = crate::types::SubterraneanCommand::zero();
        cmd.torques = [0.9; crate::types::NUM_ACTUATORS];
        e.apply_vent_and_retreat(&mut cmd);
        assert_eq!(
            cmd.thermal_pump(),
            1.0,
            "thermal_pump must stay at full authority at Red"
        );
        assert_eq!(
            cmd.cutter_head(),
            0.0,
            "boring must stop during the fallback"
        );
    }

    #[test]
    fn test_red_arrests_cutter_overheat() {
        // End-to-end: starting from a hot cutter, sustained Red-tier
        // stepping must cool it down (thermal_pump active), not let it
        // continue climbing toward the 180 C clamp the way a
        // zero-everything fallback would (cooling=0, and if a stale
        // controller command still requests boring, heat only grows).
        let mut e = SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.simulator.reset();
        e.simulator.state_mut().channels[crate::types::CUTTER_TEMP_C] = 150.0;
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        for _ in 0..50 {
            e.step(&hv, 0.05, 0.05); // Phi < 0.1 -> Red
        }
        let temp = e.simulator.state().channels[crate::types::CUTTER_TEMP_C];
        assert!(
            temp < 150.0,
            "cutter must cool under sustained VentAndRetreat, started at 150.0, got {temp}"
        );
    }
}
