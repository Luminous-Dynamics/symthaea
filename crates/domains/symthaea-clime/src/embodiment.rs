// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::ClimeController;
use crate::encoder::ClimeHdcEncoder;
use crate::simulator::{ClimePhysicsSimulator, SimpleClimeSimulator};
use crate::types::{ClimeConfig, THERMAL_STRESS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

/// Emergency fallback for a habitat life-safety platform.
///
/// Zeroing every actuator at Red (the trait's plain default, `motor_gain=0`)
/// is actively dangerous here: it shuts off ventilation and filtration
/// exactly when Phi has collapsed and the system is least able to correct
/// course — the opposite of what a life-safety habitat platform should do.
/// (Found in the 2026-07-07 unaudited-platforms review; this crate had no
/// `SafeFallback` at all before this fix.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClimeFallbackStage {
    /// Full ventilation + filtration authority (not gain-scaled), thermal
    /// loop held at a floor if stress is elevated, comfort actuators (light,
    /// humidity) held at zero.
    SafeVentilate,
}

/// Cooling/heating effort commanded during SafeVentilate when the
/// corresponding thermal-stress channel is already elevated.
const SAFE_VENTILATE_THERMAL_ELEVATED: f32 = 0.4;
/// Thermal stress threshold above which SafeVentilate commands active
/// cooling/heating rather than leaving the thermal loop idle.
const SAFE_VENTILATE_THERMAL_THRESHOLD: f64 = 0.5;

pub struct ClimeEmbodiment {
    controller: ClimeController,
    simulator: SimpleClimeSimulator,
    encoder: ClimeHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: ClimeFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl ClimeEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = ClimeConfig::default();
        Self {
            controller: ClimeController::new(genesis, &config),
            simulator: SimpleClimeSimulator::new(),
            encoder: ClimeHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: ClimeFallbackStage::SafeVentilate,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Apply moral gate from the ethics engine.
    ///
    /// Occupants cannot consent to being denied breathable air — ahimsa
    /// forces Red (SafeVentilate), a consent violation (e.g. overriding an
    /// occupant's air-quality preference) forces Orange, caution forces a
    /// Yellow cap. Previously this crate never overrode the trait's no-op
    /// default, so ethics-engine verdicts had zero effect on habitat output.
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

    /// SafeFallback: SafeVentilate. Overrides the command with a
    /// full-ventilation + full-filtration posture at full authority (not
    /// scaled by motor_gain), holding the thermal loop at a floor only if
    /// stress is already elevated, and zeroing comfort actuators.
    fn apply_safe_ventilate(&self, cmd: &mut crate::types::ClimeCommand) {
        let thermal_elevated =
            self.simulator.state().channels[THERMAL_STRESS] >= SAFE_VENTILATE_THERMAL_THRESHOLD;
        *cmd = crate::types::ClimeCommand::zero();
        cmd.torques[0] = 1.0; // ventilation_fan: full authority, never zero
        cmd.torques[1] = 1.0; // filtration_loop: full authority, never zero
        cmd.torques[2] = if thermal_elevated {
            SAFE_VENTILATE_THERMAL_ELEVATED
        } else {
            0.0
        }; // cooling_loop
        cmd.torques[3] = 0.0; // heating_loop
        // humidifier/dehumidifier/light_*: 0.0 — comfort actuators, not
        // safety-critical, held at zero during the fallback.
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

        // ── SafeFallback: SafeVentilate at Red ───────────────────────
        // Zeroing every actuator at Red (motor_gain=0) would shut off
        // ventilation/filtration exactly when Phi has collapsed — inverted
        // polarity for a life-safety habitat platform.
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_safe_ventilate(&mut cmd);
        } else {
            self.fallback_stage = ClimeFallbackStage::SafeVentilate;
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
            num_actuators: 8,
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
        self.fallback_stage = ClimeFallbackStage::SafeVentilate;
        self.fallback_cycles_in_stage = 0;
    }
    pub fn fallback_stage(&self) -> ClimeFallbackStage {
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
            platform: "clime".to_string(),
            num_actuators: 8,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for ClimeEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Clime
    }
    fn num_actuators(&self) -> usize {
        8
    }
    fn total_steps(&self) -> usize {
        self.total_steps()
    }
    fn telemetry(&self) -> EmbodimentTelemetry {
        self.telemetry()
    }
}

impl SafeFallback for ClimeEmbodiment {
    fn platform_name(&self) -> &'static str {
        "clime"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        5 // High: occupant life-safety (breathable air), stationary
    }
    fn safe_fallback_description(&self) -> &'static str {
        "SafeVentilate: full ventilation + filtration authority, thermal loop held at a floor only if stress is elevated, comfort actuators zeroed"
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
        let mut e = ClimeEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.7);
        assert!(r.success);
    }
    #[test]
    fn test_red_halts() {
        let mut e = ClimeEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_red_does_not_zero_ventilation_command() {
        // Regression: this test asserts the RESULTING COMMAND, not just the
        // safety-tier enum — the exact gap that let the inverted-polarity
        // bug (Red used to zero ventilation/filtration) ship silently.
        let mut e = ClimeEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let mut cmd = crate::types::ClimeCommand::zero();
        // Fill with nonzero torques first, to prove apply_safe_ventilate
        // actually overwrites rather than leaving a stale full command.
        cmd.torques = [0.9; crate::types::NUM_ACTUATORS];
        e.apply_safe_ventilate(&mut cmd);
        assert_eq!(
            cmd.ventilation_fan(),
            1.0,
            "ventilation must stay at full authority at Red"
        );
        assert_eq!(
            cmd.filtration_loop(),
            1.0,
            "filtration must stay at full authority at Red"
        );
        assert_eq!(
            cmd.light_brightness(),
            0.0,
            "comfort actuators zeroed during fallback"
        );
    }

    #[test]
    fn test_red_sustains_ventilation_authority_state() {
        // End-to-end: starting from a degraded ventilation_authority state
        // channel, sustained Red-tier stepping must recover it toward the
        // simulator's own fixed point for full ventilation input (0.5, given
        // the channel's own 0.998/0.001 leaky-integrator coefficients) — NOT
        // let it collapse further, as a hard motor_gain=0 zero-everything
        // fallback would (which drives it toward 0.0 instead).
        let mut e = ClimeEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.simulator.reset();
        e.simulator.state_mut().channels[crate::types::VENTILATION_AUTHORITY] = 0.2;
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        for _ in 0..500 {
            e.step(&hv, 0.05, 0.05); // Phi < 0.1 -> Red
        }
        let recovered = e.simulator.state().channels[crate::types::VENTILATION_AUTHORITY];
        assert!(
            recovered > 0.35,
            "ventilation authority must recover toward the 0.5 fixed point under sustained \
             SafeVentilate (started at 0.2), got {recovered}"
        );
    }
}
