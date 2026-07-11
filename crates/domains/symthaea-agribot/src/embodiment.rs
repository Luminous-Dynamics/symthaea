// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::AgribotController;
use crate::encoder::AgribotHdcEncoder;
use crate::simulator::{AgribotPhysicsSimulator, SimpleAgribotSimulator};
use crate::types::AgribotConfig;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

/// Emergency fallback for an agricultural stewardship platform.
///
/// Zeroing every actuator at Red (the trait's plain default, `motor_gain=0`)
/// would hard-cut irrigation instantly rather than holding a safe-idle drip
/// rate — undesirable for a drought-stressed field recovering from a
/// commanded watering cycle. (Found in the 2026-07-07 unaudited-platforms
/// review; this crate had no `SafeFallback` at all before this fix.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgribotFallbackStage {
    /// Freeze drive/tool/seed/mast (no further field disturbance),
    /// water_pump held at a safe-idle drip rate (not gain-scaled to zero).
    IrrigationHold,
}

/// Safe-idle drip rate for water_pump during IrrigationHold — enough to
/// avoid an abrupt hard-stop mid-cycle, well below full commanded flow.
const IRRIGATION_HOLD_DRIP_RATE: f32 = 0.3;

pub struct AgribotEmbodiment {
    controller: AgribotController,
    simulator: SimpleAgribotSimulator,
    encoder: AgribotHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: AgribotFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl AgribotEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = AgribotConfig::default();
        Self {
            controller: AgribotController::new(genesis, &config),
            simulator: SimpleAgribotSimulator::new(genesis),
            encoder: AgribotHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: AgribotFallbackStage::IrrigationHold,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Apply moral gate from the ethics engine. Ahimsa forces Red
    /// (IrrigationHold), a consent violation forces Orange, caution forces
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

    /// SafeFallback: IrrigationHold. Freezes drive/tool/seed/mast and holds
    /// water_pump at a safe-idle drip rate regardless of `motor_gain`.
    fn apply_irrigation_hold(&self, cmd: &mut crate::types::AgribotCommand) {
        *cmd = crate::types::AgribotCommand::zero();
        cmd.torques[4] = IRRIGATION_HOLD_DRIP_RATE; // water_pump: safe-idle drip, not zero
        // left_drive/right_drive/arm_lift/tool_head/seed_dispenser/
        // canopy_sensor_mast: 0.0 — freeze all other field disturbance.
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

        // ── SafeFallback: IrrigationHold at Red ───────────────────────
        // Zeroing every actuator at Red (motor_gain=0) would hard-cut
        // irrigation instantly instead of holding a safe-idle drip rate.
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_irrigation_hold(&mut cmd);
        } else {
            self.fallback_stage = AgribotFallbackStage::IrrigationHold;
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
            num_actuators: 7,
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
        self.fallback_stage = AgribotFallbackStage::IrrigationHold;
        self.fallback_cycles_in_stage = 0;
    }
    pub fn fallback_stage(&self) -> AgribotFallbackStage {
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
            platform: "agribot".to_string(),
            num_actuators: 7,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for AgribotEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Agribot
    }

    fn num_actuators(&self) -> usize {
        7
    }

    fn total_steps(&self) -> usize {
        self.total_steps()
    }

    fn telemetry(&self) -> EmbodimentTelemetry {
        self.telemetry()
    }
}

impl SafeFallback for AgribotEmbodiment {
    fn platform_name(&self) -> &'static str {
        "agribot"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        1 // Low: stationary/contained, no in-contact human safety stakes
    }
    fn safe_fallback_description(&self) -> &'static str {
        "IrrigationHold: freeze drive/tool/seed/mast, water_pump held at safe-idle drip rate"
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
        let mut e = AgribotEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.7);
        assert!(r.success);
    }
    #[test]
    fn test_red_halts() {
        let mut e = AgribotEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_red_holds_safe_idle_drip_not_hard_zero() {
        // Regression: asserts the RESULTING COMMAND, not just the
        // safety-tier enum.
        let mut e = AgribotEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let mut cmd = crate::types::AgribotCommand::zero();
        cmd.torques = [0.9; crate::types::NUM_ACTUATORS];
        e.apply_irrigation_hold(&mut cmd);
        assert_eq!(
            cmd.water_pump(),
            IRRIGATION_HOLD_DRIP_RATE,
            "water_pump must hold a safe-idle drip, not hard zero"
        );
        assert_eq!(
            cmd.tool_head(),
            0.0,
            "field disturbance must freeze during the fallback"
        );
    }

    #[test]
    fn test_red_still_relieves_drought() {
        // End-to-end: starting from a drought-stressed field, sustained
        // Red-tier stepping must still relieve drought risk via the
        // safe-idle drip, not let it persist the way a hard-zero fallback
        // would (watering=0 means no relief term at all).
        let mut e = AgribotEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.simulator.reset();
        e.simulator.state_mut().channels[crate::types::SOIL_MOISTURE] = 0.1;
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        for _ in 0..500 {
            e.step(&hv, 0.05, 0.05); // Phi < 0.1 -> Red
        }
        let moisture = e.simulator.state().channels[crate::types::SOIL_MOISTURE];
        assert!(
            moisture > 0.1,
            "soil moisture must improve under sustained IrrigationHold, started at 0.1, got {moisture}"
        );
    }
}
