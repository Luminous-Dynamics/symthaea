// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for SAR helicopter.
//!
//! Enables the cognitive loop to fly a helicopter via the proprioceptive loop:
//! thought → motor commands → rotor dynamics + physics → proprioceptive HV.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

use crate::controller::HelicopterController;
use crate::encoder::HelicopterHdcEncoder;
use crate::simulator::{HelicopterPhysicsSimulator, SimpleHelicopterSimulator};
use crate::types::HelicopterConfig;

/// Multi-stage emergency fallback for helicopter.
///
/// Real helicopter pilots follow a procedure:
///   1. Stabilize hover (try to hold position)
///   2. If hover impossible: controlled descent (autorotation profile)
///   3. Near ground: flare + touchdown
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelicopterFallbackStage {
    /// Stage 1: Try to hover in place. Stop drifting, maintain altitude.
    StabilizeHover,
    /// Stage 2: Controlled autorotation descent if hover failed.
    AutorotationDescent,
    /// Stage 3: Flare maneuver near ground (< 5m AGL).
    Touchdown,
}

/// Helicopter embodiment bridge.
///
/// Wraps controller + simulator + encoder into a single step function
/// that the cognitive loop can call each cycle.
pub struct HelicopterEmbodiment {
    controller: HelicopterController,
    simulator: SimpleHelicopterSimulator,
    encoder: HelicopterHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    /// Multi-stage fallback state.
    fallback_stage: HelicopterFallbackStage,
    /// Cycles spent in the current fallback stage (for timeout-based escalation).
    fallback_cycles_in_stage: u32,
}

impl HelicopterEmbodiment {
    /// Create from default config.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = HelicopterConfig::default();
        let controller = HelicopterController::new(genesis, &config);
        let simulator = SimpleHelicopterSimulator::new();
        let encoder = HelicopterHdcEncoder::new(genesis, 32);

        Self {
            controller,
            simulator,
            encoder,
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: HelicopterFallbackStage::StabilizeHover,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Apply moral gate from ethics engine.
    /// A helicopter on an SAR mission still has rotor blades — ahimsa forces
    /// Red (triggers autorotation descent), consent violation forces Orange,
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

    /// Update fallback stage based on current state and time-in-stage.
    fn update_fallback_stage(&mut self) {
        let state = self.simulator.state();
        let altitude = state.altitude();
        let vertical_speed = state.linear_velocity[2]; // negative = descending

        match self.fallback_stage {
            HelicopterFallbackStage::StabilizeHover => {
                // If we're descending fast (>2 m/s) for more than 100 cycles → escalate
                if vertical_speed < -2.0 && self.fallback_cycles_in_stage > 100 {
                    self.fallback_stage = HelicopterFallbackStage::AutorotationDescent;
                    self.fallback_cycles_in_stage = 0;
                }
                // If we're below 10m AGL → escalate to touchdown
                if altitude < 10.0 {
                    self.fallback_stage = HelicopterFallbackStage::Touchdown;
                    self.fallback_cycles_in_stage = 0;
                }
            }
            HelicopterFallbackStage::AutorotationDescent => {
                // Approach ground → flare for touchdown
                if altitude < 5.0 {
                    self.fallback_stage = HelicopterFallbackStage::Touchdown;
                    self.fallback_cycles_in_stage = 0;
                }
            }
            HelicopterFallbackStage::Touchdown => {
                // Stay in touchdown until on ground
            }
        }
    }

    /// Reset the fallback state machine (call when exiting Red tier).
    fn reset_fallback_stage(&mut self) {
        self.fallback_stage = HelicopterFallbackStage::StabilizeHover;
        self.fallback_cycles_in_stage = 0;
    }

    /// Current fallback stage (for telemetry/testing).
    pub fn fallback_stage(&self) -> HelicopterFallbackStage {
        self.fallback_stage
    }

    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    /// Step: thought → motor command (consciousness-gated) → physics → proprioceptive encoding.
    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };
        // Ethics gate max-composes on top of phi + override. Moral-Red
        // triggers the autorotation-descent fallback chain.
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }
        let gain = self.current_safety.motor_gain();

        let mut cmd = self.controller.forward(thought_hv, dt);

        // ── MULTI-STAGE SAFE FALLBACK for Red tier ──────────────
        // Real helicopter emergency procedure:
        //   Stage 1: Stabilize hover (try to hold position)
        //   Stage 2: Autorotation descent (if hover impossible)
        //   Stage 3: Touchdown flare (near ground)
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage += 1;
            self.update_fallback_stage();

            match self.fallback_stage {
                HelicopterFallbackStage::StabilizeHover => {
                    // Try to hover: maintain hover thrust, level cyclic
                    cmd.collective = 0.55; // Hover collective
                    cmd.cyclic_lon = 0.0; // Level pitch
                    cmd.cyclic_lat = 0.0; // Level roll
                    cmd.pedal = 0.0; // Centered yaw
                    cmd.thrust = 0.6; // Hover thrust
                    cmd.tail_rotor = 0.5; // Anti-torque for hover
                }
                HelicopterFallbackStage::AutorotationDescent => {
                    // Engine cut + windmilling rotor
                    cmd.collective = 0.15; // Low positive pitch → windmill
                    cmd.cyclic_lon = 0.0; // Level pitch
                    cmd.cyclic_lat = 0.0; // Level roll
                    cmd.pedal = 0.0; // Centered yaw
                    cmd.thrust = 0.0; // Engine OFF
                    cmd.tail_rotor = 0.1; // Minimal anti-torque
                }
                HelicopterFallbackStage::Touchdown => {
                    // Flare maneuver: increase collective to cushion landing
                    cmd.collective = 0.4; // Flare pitch
                    cmd.cyclic_lon = -0.1; // Slight nose-up
                    cmd.cyclic_lat = 0.0;
                    cmd.pedal = 0.0;
                    cmd.thrust = 0.2; // Restart cushion thrust
                    cmd.tail_rotor = 0.3;
                }
            }
        } else {
            // Not in Red — reset fallback state machine
            self.reset_fallback_stage();
            if gain < 1.0 {
                // Yellow/Orange: scaled authority but pilot retains control
                cmd.collective *= gain;
                cmd.cyclic_lon *= gain;
                cmd.cyclic_lat *= gain;
                cmd.pedal *= gain;
                cmd.thrust *= gain;
                cmd.tail_rotor *= gain;
            }
        }

        self.last_control_effort = cmd.control_effort();
        self.simulator.step(&cmd, dt as f64);

        let perception = self.encoder.encode(self.simulator.state());

        let pred_error = if let Some(ref prev) = self.last_perception {
            let sim = perception.similarity(prev);
            (1.0 - sim.max(0.0)).min(1.0)
        } else {
            0.0_f32
        };
        self.last_prediction_error = pred_error;
        self.last_perception = Some(perception);
        self.total_steps += 1;

        let success = self.simulator.state().is_finite();

        EmbodimentResult {
            num_actuators: 6,
            control_effort: self.last_control_effort,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pred_error),
        }
    }

    /// Encode current body state as proprioceptive HV.
    pub fn encode_perception(&mut self) -> ContinuousHV {
        let perception = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(perception.clone());
        perception
    }

    /// Reset to hover state.
    pub fn reset(&mut self) {
        self.simulator.reset(20.0);
        self.controller.reset();
        self.encoder.reset();
        self.last_perception = None;
        self.total_steps = 0;
        self.current_safety = MotorSafetyLevel::Green;
        self.safety_override = None;
        self.moral_safety = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
        self.fallback_stage = HelicopterFallbackStage::StabilizeHover;
        self.fallback_cycles_in_stage = 0;
    }

    pub fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
    pub fn last_perception(&self) -> Option<&ContinuousHV> {
        self.last_perception.as_ref()
    }

    /// Telemetry summary.
    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: format!("{:?}", self.current_safety),
            platform: "helicopter".to_string(),
            num_actuators: 6,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for HelicopterEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Helicopter
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
    fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        self.apply_moral_gate(gate)
    }
}

/// SafeFallback for Helicopter — Multi-Stage Emergency Procedure.
///
/// Real helicopter pilots follow a tiered procedure. Symthaea mirrors this:
///
///   Stage 1 — StabilizeHover: First try to hold position. Hover thrust + level cyclic.
///             Escalates if descent rate >2 m/s for >100 cycles, OR altitude <10m.
///   Stage 2 — AutorotationDescent: Engine cut, windmilling rotor for controlled descent.
///             Escalates to Touchdown when altitude <5m.
///   Stage 3 — Touchdown: Flare maneuver — increased collective + slight nose-up to
///             cushion the landing.
///
/// This mirrors FAA emergency procedure 27.143 (Engine failure in flight).
impl SafeFallback for HelicopterEmbodiment {
    fn platform_name(&self) -> &'static str {
        "helicopter"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        10
    } // Critical: airborne, human pilot/passengers
    fn safe_fallback_description(&self) -> &'static str {
        "Multi-stage: StabilizeHover → AutorotationDescent → Touchdown flare"
    }
    fn safe_fallback_latency_cycles(&self) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bridge() -> HelicopterEmbodiment {
        let genesis = GenesisSeed::from_phrase("test-heli-embodiment");
        HelicopterEmbodiment::new(&genesis)
    }

    #[test]
    fn test_step_produces_valid_result() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        let result = bridge.step(&hv, 1.0 / 300.0, 0.7);
        assert!(result.success);
        assert_eq!(result.num_actuators, 6);
        assert_eq!(result.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_proprioceptive_encoding() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 1.0 / 300.0, 0.7);
        let perception = bridge.encode_perception();
        assert_eq!(perception.dim(), 16384);
    }

    #[test]
    fn test_safety_gating() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);

        // Red safety does NOT zero the motors — a helicopter cannot safely
        // hard-cut power mid-air. It enters the staged emergency-landing
        // fallback (StabilizeHover → AutorotationDescent → Touchdown), so
        // control effort is the fallback's hover command, not 0.0.
        // (An earlier version of this test asserted effort == 0.0, which the
        // staged fallback can never produce — stale hard-zero design.)
        let result = bridge.step(&hv, 1.0 / 300.0, 0.05);
        assert_eq!(result.safety_level, MotorSafetyLevel::Red);
        assert!(
            result.control_effort > 0.0,
            "Red fallback must command a staged hover, not dead motors"
        );
        // Stage 1 hover command: collective 0.55, thrust 0.6, tail 0.5,
        // cyclic/pedal zeroed → effort = 1.65/6 = 0.275
        assert!(
            (result.control_effort - 0.275).abs() < 1e-4,
            "expected StabilizeHover effort 0.275, got {}",
            result.control_effort
        );
    }

    #[test]
    fn test_telemetry() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 1.0 / 300.0, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "helicopter");
        assert_eq!(t.num_actuators, 6);
    }

    #[test]
    fn test_reset() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 1.0 / 300.0, 0.7);
        assert_eq!(bridge.total_steps(), 1);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
        assert!(bridge.last_perception().is_none());
    }

    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        let mut bridge = make_bridge();
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let hv = ContinuousHV::random(16384, 42);
        // Phi 0.9 → Green normally; ahimsa forces Red.
        let r = bridge.step(&hv, 1.0 / 300.0, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "ahimsa must force Red regardless of phi"
        );
    }

    #[test]
    fn test_moral_gate_consent_violation_forces_orange() {
        let mut bridge = make_bridge();
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 1.0 / 300.0, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Orange,
            "consent violation → Orange"
        );
    }
}
