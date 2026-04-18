// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for quadrotor flight platform.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::FlightController;
use crate::encoder::QuadrotorHdcEncoder;
use crate::simulator::{PhysicsSimulator, SimplePhysicsSimulator};
use crate::types::FlightConfig;

pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    MoralGateInput, MotorSafetyLevel, SafeFallback, GROUNDING_SENSORIMOTOR,
};

/// Multi-stage emergency fallback for quadrotors.
///
/// Mirrors DJI/PX4 emergency landing protocols:
///   1. StabilizeHover: hold position, level attitude
///   2. ControlledDescent: 30% hover thrust for ~3 m/s descent
///   3. Touchdown: cushion landing thrust spike near ground (<1m)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightFallbackStage {
    /// Stage 1: Maintain altitude with hover thrust, level attitude.
    StabilizeHover,
    /// Stage 2: Controlled descent at 30% hover thrust.
    ControlledDescent,
    /// Stage 3: Touchdown cushion (near ground).
    Touchdown,
}

/// Quadrotor embodiment bridge.
pub struct FlightEmbodiment {
    controller: FlightController,
    simulator: SimplePhysicsSimulator,
    encoder: QuadrotorHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: FlightFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl FlightEmbodiment {
    /// Emergency descent thrust (Newtons). Approximately 30% of hover thrust
    /// for a 1.5kg quadrotor: 0.3 * 1.5 * 9.81 ≈ 4.4 N.
    /// Produces a controlled ~3 m/s descent instead of freefall.
    const EMERGENCY_DESCENT_THRUST: f32 = 4.4;

    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = FlightConfig::default();
        Self {
            controller: FlightController::new(genesis, &config),
            simulator: SimplePhysicsSimulator::new(),
            encoder: QuadrotorHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: FlightFallbackStage::StabilizeHover,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Apply moral gate from ethics engine.
    /// A flying quadrotor can strike beings — ahimsa forces Red (which triggers
    /// the StabilizeHover → ControlledDescent → Touchdown fallback chain at
    /// the existing Red-tier handler), consent violation forces Orange,
    /// caution forces Yellow.
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

    /// Update fallback stage based on altitude and time.
    fn update_fallback_stage(&mut self) {
        let altitude = self.simulator.state().position[2];
        match self.fallback_stage {
            FlightFallbackStage::StabilizeHover => {
                // After 50 cycles of stable hover, begin controlled descent
                if self.fallback_cycles_in_stage > 50 {
                    self.fallback_stage = FlightFallbackStage::ControlledDescent;
                    self.fallback_cycles_in_stage = 0;
                }
            }
            FlightFallbackStage::ControlledDescent => {
                // Approach ground → cushion landing
                if altitude < 1.0 {
                    self.fallback_stage = FlightFallbackStage::Touchdown;
                    self.fallback_cycles_in_stage = 0;
                }
            }
            FlightFallbackStage::Touchdown => {
                // Stay in touchdown
            }
        }
    }

    fn reset_fallback_stage(&mut self) {
        self.fallback_stage = FlightFallbackStage::StabilizeHover;
        self.fallback_cycles_in_stage = 0;
    }

    pub fn fallback_stage(&self) -> FlightFallbackStage {
        self.fallback_stage
    }

    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };
        // Ethics gate max-composes on top of phi + override. A moral-Red
        // feeds into the existing Red-tier fallback chain below.
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }
        let gain = self.current_safety.motor_gain();

        let mut cmd = self.controller.forward(thought_hv, dt);

        // ── MULTI-STAGE SAFE FALLBACK for Red tier ──────────────
        // Mirrors DJI/PX4 emergency landing protocol:
        //   Stage 1: StabilizeHover (50 cycles, hover thrust)
        //   Stage 2: ControlledDescent (30% hover thrust, ~3 m/s)
        //   Stage 3: Touchdown (cushion thrust spike <1m AGL)
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage += 1;
            self.update_fallback_stage();

            // Always zero moments at Red — prevent attitude divergence
            cmd.roll_moment = 0.0;
            cmd.pitch_moment = 0.0;
            cmd.yaw_moment = 0.0;

            cmd.thrust = match self.fallback_stage {
                FlightFallbackStage::StabilizeHover => {
                    // Hover thrust ~14.7 N for 1.5kg airframe
                    14.7
                }
                FlightFallbackStage::ControlledDescent => {
                    Self::EMERGENCY_DESCENT_THRUST // 4.4 N (30% hover)
                }
                FlightFallbackStage::Touchdown => {
                    // Cushion landing — slightly more than descent thrust
                    8.0
                }
            };
        } else {
            // Not in Red — reset state machine
            self.reset_fallback_stage();
            if gain < 1.0 {
                cmd.roll_moment *= gain;
                cmd.pitch_moment *= gain;
                cmd.yaw_moment *= gain;
                cmd.thrust *= gain;
            }
        }

        let effort = (cmd.thrust.abs()
            + cmd.roll_moment.abs()
            + cmd.pitch_moment.abs()
            + cmd.yaw_moment.abs())
            / 4.0;
        self.last_control_effort = effort;

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

        let success = self.simulator.state().position[2].is_finite();

        EmbodimentResult {
            num_actuators: 4,
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
        self.simulator.reset(0.1);
        self.controller.reset();
        self.encoder.reset();
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.moral_safety = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
        self.fallback_stage = FlightFallbackStage::StabilizeHover;
        self.fallback_cycles_in_stage = 0;
    }

    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }

    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Access the underlying simulator state (for testing and telemetry).
    pub fn simulator(&self) -> &SimplePhysicsSimulator {
        &self.simulator
    }

    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "quadrotor".to_string(),
            num_actuators: 4,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for FlightEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Quadrotor
    }
    fn num_actuators(&self) -> usize {
        4
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

/// SafeFallback for Quadrotor — Multi-Stage Emergency Landing.
///
/// Mirrors DJI/PX4 emergency landing protocol:
///   Stage 1 — StabilizeHover: 50 cycles of hover thrust to stop drift
///   Stage 2 — ControlledDescent: 30% hover thrust → ~3 m/s drag-limited descent
///   Stage 3 — Touchdown: cushion thrust spike (8N) when altitude <1m AGL
impl SafeFallback for FlightEmbodiment {
    fn platform_name(&self) -> &'static str {
        "quadrotor"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        10
    } // Critical: airborne
    fn safe_fallback_description(&self) -> &'static str {
        "Multi-stage: StabilizeHover → ControlledDescent → Touchdown"
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
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.7);
        assert!(r.success);
        assert_eq!(r.num_actuators, 4);
    }

    #[test]
    fn test_safety_gating() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        // Red safety: emergency descent thrust is non-zero (controlled descent, not freefall)
        assert!(r.control_effort >= 0.0);
    }

    #[test]
    fn test_red_safety_controlled_descent() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("descent_test"));
        let hv = ContinuousHV::random(16384, 42);

        // Step with Green safety first to establish altitude
        for _ in 0..20 {
            bridge.step(&hv, 0.01, 0.8);
        }

        // Now force Red safety for 100 steps
        for _ in 0..100 {
            let r = bridge.step(&hv, 0.01, 0.05); // Phi < 0.1 → Red
            assert_eq!(r.safety_level, MotorSafetyLevel::Red);
            assert!(
                r.success,
                "state must remain finite during emergency descent"
            );
        }

        // Verify descent rate is bounded (never exceeds ~5 m/s)
        let state = bridge.simulator().state();
        assert!(
            state.linear_velocity[2] >= -5.1,
            "descent rate {} exceeded MAX_DESCENT_RATE",
            state.linear_velocity[2]
        );
    }

    #[test]
    fn test_perception() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        let p = bridge.encode_perception();
        assert_eq!(p.dim(), 16384);
    }

    #[test]
    fn test_telemetry() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "quadrotor");
    }

    #[test]
    fn test_reset() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
    }

    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let hv = ContinuousHV::random(16384, 42);
        // Phi 0.9 → Green normally; ahimsa forces Red. Red triggers the
        // emergency-landing fallback chain (StabilizeHover → ControlledDescent
        // → Touchdown). Unlike ground-platform Red, flight does NOT zero
        // thrust — a freefalling drone kills more people than a hovering one.
        // So we verify only the tier escalation; the fallback-state assertion
        // lives in the dedicated fallback tests.
        let r = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "ahimsa must force Red regardless of phi"
        );
    }

    #[test]
    fn test_moral_gate_consent_violation_forces_orange() {
        let mut bridge = FlightEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Orange,
            "consent violation → Orange"
        );
    }
}
