// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for AUV commons steward.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::AuvController;
use crate::encoder::AuvHdcEncoder;
use crate::simulator::{AuvPhysicsSimulator, SimpleAuvSimulator};
use crate::types::{AuvCommand, AuvConfig};

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

/// Emergency fallback behavior for the AUV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuvFallbackStage {
    /// Vertical thrusters commanded to ascend toward the surface; all other
    /// thrust suppressed. Zero thrust alone would leave the vehicle
    /// drifting with current/tide at whatever depth it happened to be at —
    /// not safe near reefs, wrecks, or other vessels.
    BuoyancyAscent,
}

/// Fraction of full vertical thrust commanded during buoyancy ascent.
const ASCENT_THRUST: f32 = 0.5;

/// AUV embodiment bridge.
pub struct AuvEmbodiment {
    controller: AuvController,
    simulator: SimpleAuvSimulator,
    encoder: AuvHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: AuvFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl AuvEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = AuvConfig::default();
        Self {
            controller: AuvController::new(genesis, &config),
            simulator: SimpleAuvSimulator::new(),
            encoder: AuvHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: AuvFallbackStage::BuoyancyAscent,
            fallback_cycles_in_stage: 0,
        }
    }

    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    pub fn fallback_stage(&self) -> AuvFallbackStage {
        self.fallback_stage
    }

    /// SafeFallback: BuoyancyAscent. Overrides the command with a vertical
    /// ascent thrust, executed at full authority (not scaled by motor_gain).
    fn apply_buoyancy_ascent(&self, cmd: &mut AuvCommand) {
        *cmd = AuvCommand::ascend(ASCENT_THRUST);
    }

    /// Apply moral gate from ethics engine.
    /// An AUV operating near divers, fish, or reefs can cause harm — ahimsa
    /// forces Red (buoyancy ascent to surface, not merely zero thrust —
    /// drifting with current near reefs/vessels is not safe), consent
    /// violation forces Orange, caution forces Yellow.
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

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };
        // Ethics gate max-composes on top of phi + override.
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }
        let gain = self.current_safety.motor_gain();
        let mut cmd = self.controller.forward(thought_hv, dt);

        // ── SafeFallback: BuoyancyAscent at Red ──────────────────────
        // Zero thrust in open water is not safe — the vehicle drifts with
        // current/tide at whatever depth it happens to be. Actively ascend
        // to the surface instead, at full authority (not gain-scaled).
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_buoyancy_ascent(&mut cmd);
        } else {
            self.fallback_stage = AuvFallbackStage::BuoyancyAscent;
            self.fallback_cycles_in_stage = 0;
            if gain < 1.0 {
                for t in &mut cmd.thrusters {
                    *t *= gain;
                }
            }
        }
        self.last_control_effort = cmd.control_effort();
        self.simulator.step(&cmd, dt as f64);
        let perception = self.encoder.encode(self.simulator.state());
        let pred_error = if let Some(ref prev) = self.last_perception {
            (1.0 - perception.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0_f32
        };
        self.last_prediction_error = pred_error;
        self.last_perception = Some(perception);
        self.total_steps += 1;
        let success = self.simulator.state().is_finite();
        EmbodimentResult {
            num_actuators: 8,
            control_effort: self.last_control_effort,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pred_error),
        }
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        let p = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(p.clone());
        p
    }
    pub fn reset(&mut self) {
        self.simulator.reset(10.0);
        self.controller.reset();
        self.encoder.reset();
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.moral_safety = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
        self.fallback_stage = AuvFallbackStage::BuoyancyAscent;
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
            platform: "auv".to_string(),
            num_actuators: 8,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for AuvEmbodiment {
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
    fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.set_safety_override(level)
    }
    fn clear_safety_override(&mut self) {
        self.clear_safety_override()
    }
    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::Auv
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
    fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.apply_moral_gate(gate)
    }
}

impl SafeFallback for AuvEmbodiment {
    fn platform_name(&self) -> &'static str {
        "auv"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        6 // Moderate-high: mobile, operates near reefs/wrecks/other vessels
    }
    fn safe_fallback_description(&self) -> &'static str {
        "BuoyancyAscent: vertical thrusters commanded to surface, all other thrust suppressed"
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
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.01, 0.7);
        assert!(r.success);
        assert_eq!(r.num_actuators, 8);
    }

    #[test]
    fn test_safety_gating() {
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.01, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        // NOTE: this used to assert control_effort == 0.0, but that was the
        // bug this fix closes — zero thrust in open water just means
        // drifting with current at whatever depth. Red now commands
        // BuoyancyAscent (a deliberate, non-zero surfacing thrust).
        assert!(
            r.control_effort > 0.0,
            "Red must command BuoyancyAscent (non-zero), got {}",
            r.control_effort
        );
    }

    #[test]
    fn test_perception() {
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.01, 0.7);
        let p = bridge.encode_perception();
        assert_eq!(p.dim(), 16384);
    }

    #[test]
    fn test_telemetry() {
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.01, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "auv");
    }

    #[test]
    fn test_reset() {
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.01, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
    }

    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let hv = ContinuousHV::random(16384, 42);
        // Phi 0.9 → Green normally; ahimsa forces Red.
        let r = bridge.step(&hv, 0.01, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "ahimsa must force Red regardless of phi"
        );
        // NOTE: this used to assert control_effort == 0.0 ("Red tier zeros
        // thrust") — that was exactly the gap this fix closes. Ahimsa-Red
        // now commands BuoyancyAscent, a deliberate non-zero surfacing
        // thrust, not a passive drift.
        assert!(
            r.control_effort > 0.0,
            "ahimsa-Red must command BuoyancyAscent (non-zero), got {}",
            r.control_effort
        );
    }

    #[test]
    fn test_moral_gate_consent_violation_forces_orange() {
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.01, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Orange,
            "consent violation → Orange"
        );
    }

    #[test]
    fn test_red_triggers_buoyancy_ascent_stage() {
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.01, 0.05);
        assert_eq!(bridge.fallback_stage(), AuvFallbackStage::BuoyancyAscent);
    }

    #[test]
    fn test_red_ascends_toward_surface_over_time() {
        // The core safety fix: sustained Red must actually reduce depth
        // (ascend), not just leave the vehicle drifting at whatever depth
        // it happened to be.
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        // Start at neutral buoyancy, depth 10m (AuvEmbodiment::new default).
        let initial_depth = bridge.simulator.state().depth_m();
        for _ in 0..3000 {
            let r = bridge.step(&hv, 0.01, 0.05); // Red throughout
            assert!(r.success, "state must stay finite while ascending");
        }
        let final_depth = bridge.simulator.state().depth_m();
        assert!(
            final_depth < initial_depth,
            "BuoyancyAscent should reduce depth: initial={initial_depth}, final={final_depth}"
        );
    }

    #[test]
    fn test_red_commands_fixed_ascent_thrust_not_scaled_output() {
        // At Red, the vertical thrusters must be the fixed ascent command,
        // not the gain-scaled (and thus zeroed) neural controller output.
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.01, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        // Only thrusters 4 and 5 are commanded (ASCENT_THRUST each), all
        // others zero: mean |thrust| over 8 actuators = 2*ASCENT_THRUST/8.
        let expected = 2.0 * ASCENT_THRUST / 8.0;
        assert!(
            (r.control_effort - expected).abs() < 1e-5,
            "expected fixed ascent thrust {expected}, got {}",
            r.control_effort
        );
    }

    #[test]
    fn test_safe_fallback_trait_impl() {
        let bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        assert_eq!(bridge.platform_name(), "auv");
        assert_eq!(bridge.safe_fallback_priority(), 6);
        assert_eq!(bridge.safe_fallback_latency_cycles(), 1);
        assert!(
            bridge
                .safe_fallback_description()
                .contains("BuoyancyAscent")
        );
    }

    #[test]
    fn test_safe_fallback_active_iff_red_or_orange() {
        let mut bridge = AuvEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.01, 0.9); // Green
        assert!(!bridge.safe_fallback_active());
        bridge.step(&hv, 0.01, 0.05); // Red
        assert!(bridge.safe_fallback_active());
    }
}
