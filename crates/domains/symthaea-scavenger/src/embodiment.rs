// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::ScavengerController;
use crate::encoder::ScavengerHdcEncoder;
use crate::simulator::{ScavengerPhysicsSimulator, SimpleScavengerSimulator};
use crate::types::ScavengerConfig;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

/// Emergency fallback for a disassembly/salvage platform.
///
/// Zeroing every actuator at Red (the trait's plain default, `motor_gain=0`)
/// is dangerous here: it kills `dust_suppression` at exactly the moment an
/// incident-risk trigger should be suppressing dust, not shutting the fan
/// off. (Found in the 2026-07-07 unaudited-platforms review; this crate had
/// no `SafeFallback` at all before this fix.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScavengerFallbackStage {
    /// Retract: stop cutting/gripping/sorting/compacting/driving (no
    /// further hazard exposure), dust_suppression at full authority (not
    /// gain-scaled) to keep clearing the incident's dust load.
    Retract,
}

pub struct ScavengerEmbodiment {
    controller: ScavengerController,
    simulator: SimpleScavengerSimulator,
    encoder: ScavengerHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: ScavengerFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl ScavengerEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = ScavengerConfig::default();
        Self {
            controller: ScavengerController::new(genesis, &config),
            simulator: SimpleScavengerSimulator::new(genesis),
            encoder: ScavengerHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: ScavengerFallbackStage::Retract,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Apply moral gate from the ethics engine. Ahimsa (e.g. a bystander
    /// detected in the workspace) forces Red (Retract), a consent violation
    /// forces Orange, caution forces a Yellow cap. Previously this crate
    /// never overrode the trait's no-op default.
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

    /// SafeFallback: Retract. Stops all disassembly/drive actuators and
    /// commands full dust_suppression authority regardless of `motor_gain`.
    fn apply_retract(&self, cmd: &mut crate::types::ScavengerCommand) {
        *cmd = crate::types::ScavengerCommand::zero();
        cmd.torques[9] = 1.0; // dust_suppression: full authority, never zero
        // left_track/right_track/arm_lift/arm_extend/gripper/cutter/sorter/
        // hopper_feed/compactor: 0.0 — stop all disassembly work and driving.
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

        // ── SafeFallback: Retract at Red ──────────────────────────────
        // Zeroing every actuator at Red (motor_gain=0) would kill dust
        // suppression right when an incident-risk trigger needs it most.
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_retract(&mut cmd);
        } else {
            self.fallback_stage = ScavengerFallbackStage::Retract;
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
            num_actuators: 10,
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
        self.fallback_stage = ScavengerFallbackStage::Retract;
        self.fallback_cycles_in_stage = 0;
    }
    pub fn fallback_stage(&self) -> ScavengerFallbackStage {
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
            platform: "scavenger".to_string(),
            num_actuators: 10,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for ScavengerEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Scavenger
    }

    fn num_actuators(&self) -> usize {
        10
    }

    fn total_steps(&self) -> usize {
        self.total_steps()
    }

    fn telemetry(&self) -> EmbodimentTelemetry {
        self.telemetry()
    }
}

impl SafeFallback for ScavengerEmbodiment {
    fn platform_name(&self) -> &'static str {
        "scavenger"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        5 // High: in-contact (cutter/gripper), operates near humans
    }
    fn safe_fallback_description(&self) -> &'static str {
        "Retract: stop all disassembly/drive actuators, full dust_suppression authority"
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
        let mut e = ScavengerEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.7);
        assert!(r.success);
    }
    #[test]
    fn test_red_halts() {
        let mut e = ScavengerEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_red_does_not_zero_dust_suppression() {
        // Regression: asserts the RESULTING COMMAND, not just the
        // safety-tier enum — the exact gap that let the "Red kills dust
        // suppression mid-incident" bug ship silently.
        let mut e = ScavengerEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let mut cmd = crate::types::ScavengerCommand::zero();
        cmd.torques = [0.9; crate::types::NUM_ACTUATORS];
        e.apply_retract(&mut cmd);
        assert_eq!(
            cmd.dust_suppression(),
            1.0,
            "dust_suppression must stay at full authority at Red"
        );
        assert_eq!(cmd.cutter(), 0.0, "cutting must stop during the fallback");
    }

    #[test]
    fn test_red_clears_dust_incident() {
        // End-to-end: starting from an elevated dust level, sustained
        // Red-tier stepping must clear it (dust_suppression active), not
        // let it persist the way a zero-everything fallback would
        // (suppression=0 means the dust_level decay term vanishes).
        let mut e = ScavengerEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.simulator.reset();
        e.simulator.state_mut().channels[9] = 0.8; // dust_level
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        for _ in 0..200 {
            e.step(&hv, 0.05, 0.05); // Phi < 0.1 -> Red
        }
        let dust = e.simulator.state().channels[9];
        assert!(
            dust < 0.8,
            "dust level must clear under sustained Retract, started at 0.8, got {dust}"
        );
    }
}
