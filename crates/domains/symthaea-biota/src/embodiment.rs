// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::BiotaController;
use crate::encoder::BiotaHdcEncoder;
use crate::simulator::{BiotaPhysicsSimulator, SimpleBiotaSimulator};
use crate::types::BiotaConfig;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

/// Emergency fallback for an interspecies-bridge / sanctuary platform.
///
/// Zeroing every actuator at Red (the trait's plain default, `motor_gain=0`)
/// is the worst possible failure mode here: this platform's entire job is
/// signaling (sanctuary/right-of-way/distress), and Red is exactly when a
/// distress episode is most likely still active — going silent at that
/// moment defeats the platform's purpose. (Found in the 2026-07-07
/// unaudited-platforms review; this crate had no `SafeFallback` at all
/// before this fix.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiotaFallbackStage {
    /// Freeze drive/gaze_beacon (no further approach/disturbance),
    /// acoustic_chime/thermal_beacon/sanctuary_projector held at full
    /// authority (not gain-scaled) to keep signaling through the episode.
    SanctuaryHold,
}

pub struct BiotaEmbodiment {
    controller: BiotaController,
    simulator: SimpleBiotaSimulator,
    encoder: BiotaHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: BiotaFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl BiotaEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = BiotaConfig::default();
        Self {
            controller: BiotaController::new(genesis, &config),
            simulator: SimpleBiotaSimulator::new(genesis),
            encoder: BiotaHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: BiotaFallbackStage::SanctuaryHold,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Apply moral gate from the ethics engine. Ahimsa forces Red
    /// (SanctuaryHold), a consent violation forces Orange, caution forces
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

    /// SafeFallback: SanctuaryHold. Freezes drive/gaze and commands full
    /// signaling authority (acoustic/thermal/sanctuary) regardless of
    /// `motor_gain`.
    fn apply_sanctuary_hold(&self, cmd: &mut crate::types::BiotaCommand) {
        *cmd = crate::types::BiotaCommand::zero();
        cmd.torques[3] = 1.0; // acoustic_chime: full authority, never zero
        cmd.torques[4] = 1.0; // thermal_beacon: full authority, never zero
        cmd.torques[5] = 1.0; // sanctuary_projector: full authority, never zero
        // left_drive/right_drive/gaze_beacon: 0.0 — freeze approach/disturbance.
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

        // ── SafeFallback: SanctuaryHold at Red ────────────────────────
        // Zeroing every actuator at Red (motor_gain=0) would go silent
        // exactly when a distress episode is most likely still active.
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_sanctuary_hold(&mut cmd);
        } else {
            self.fallback_stage = BiotaFallbackStage::SanctuaryHold;
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
        self.fallback_stage = BiotaFallbackStage::SanctuaryHold;
        self.fallback_cycles_in_stage = 0;
    }
    pub fn fallback_stage(&self) -> BiotaFallbackStage {
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
            safety_level: self.current_safety,
            platform: "biota".to_string(),
            num_actuators: 6,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for BiotaEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Biota
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

impl SafeFallback for BiotaEmbodiment {
    fn platform_name(&self) -> &'static str {
        "biota"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        10 // Critical: the platform's entire job is signaling, and Red is
        // exactly when an animal-distress episode is most likely active.
    }
    fn safe_fallback_description(&self) -> &'static str {
        "SanctuaryHold: freeze drive/gaze, full acoustic/thermal/sanctuary signaling authority"
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
        let mut e = BiotaEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.7);
        assert!(r.success);
    }

    #[test]
    fn test_red_halts() {
        let mut e = BiotaEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_red_does_not_go_silent() {
        // Regression: asserts the RESULTING COMMAND, not just the
        // safety-tier enum — the highest-stakes gap in the whole review,
        // since this platform's entire job is signaling.
        let mut e = BiotaEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let mut cmd = crate::types::BiotaCommand::zero();
        cmd.torques = [0.9; crate::types::NUM_ACTUATORS];
        e.apply_sanctuary_hold(&mut cmd);
        assert_eq!(
            cmd.acoustic_chime(),
            1.0,
            "acoustic signal must not go silent at Red"
        );
        assert_eq!(
            cmd.thermal_beacon(),
            1.0,
            "thermal beacon must not go dark at Red"
        );
        assert_eq!(
            cmd.sanctuary_projector(),
            1.0,
            "sanctuary projection must not go dark at Red"
        );
        assert_eq!(cmd.torques[0], 0.0, "drive must freeze during the fallback");
    }

    #[test]
    fn test_red_sustains_sanctuary_signal() {
        // End-to-end: starting from a degraded sanctuary signal, sustained
        // Red-tier stepping must recover it (sanctuary_projector active),
        // not let it collapse to zero the way a zero-everything fallback
        // would.
        let mut e = BiotaEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.simulator.reset();
        e.simulator.state_mut().channels[crate::types::SANCTUARY_SIGNAL] = 0.1;
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        for _ in 0..300 {
            e.step(&hv, 0.05, 0.05); // Phi < 0.1 -> Red
        }
        let signal = e.simulator.state().channels[crate::types::SANCTUARY_SIGNAL];
        assert!(
            signal > 0.1,
            "sanctuary signal must recover under sustained SanctuaryHold, started at 0.1, got {signal}"
        );
    }
}
