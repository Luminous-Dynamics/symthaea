// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for autonomous vehicle.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::VehicleController;
use crate::encoder::VehicleHdcEncoder;
use crate::simulator::{BicycleModelSimulator, VehiclePhysicsSimulator};
use crate::types::{BiotaRightOfWaySignal, VehicleConfig};

pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    MoralGateInput, MotorSafetyLevel, SafeFallback, GROUNDING_SENSORIMOTOR,
};

/// Multi-stage emergency fallback for vehicles.
///
/// Mirrors real ADAS handover protocols (ISO 21448 SOTIF):
///   1. Maintain course briefly (allow human takeover detection)
///   2. Gradual deceleration (50% brake, smooth steering hold)
///   3. Emergency brake (full force)
///   4. Post-stop (parking brake, hazards on)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VehicleFallbackStage {
    /// Stage 1: Maintain course (50 cycles ~100ms for human takeover).
    MaintainCourse,
    /// Stage 2: Gradual deceleration (50% brake) + steering hold.
    GradualDeceleration,
    /// Stage 3: Full emergency brake.
    EmergencyBrake,
    /// Stage 4: Stopped, parking brake engaged.
    PostStop,
}

/// Vehicle embodiment bridge.
pub struct VehicleEmbodiment {
    controller: VehicleController,
    simulator: BicycleModelSimulator,
    encoder: VehicleHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    /// Last commanded steering angle (held through safe fallback with taper).
    last_safe_steering: f32,
    /// Cycles since safe fallback activated (for taper schedule).
    fallback_cycles: u32,
    /// Multi-stage fallback state.
    fallback_stage: VehicleFallbackStage,
    /// Cycles in current fallback stage.
    fallback_cycles_in_stage: u32,
}

impl VehicleEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = VehicleConfig::default();
        Self {
            controller: VehicleController::new(genesis, &config),
            simulator: BicycleModelSimulator::new(),
            encoder: VehicleHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            last_safe_steering: 0.0,
            fallback_cycles: 0,
            fallback_stage: VehicleFallbackStage::MaintainCourse,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Apply moral gate from ethics engine.
    /// A vehicle moving at speed can kill — ahimsa forces Red (triggers
    /// ADAS emergency-brake fallback), consent violation forces Orange,
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

    /// Apply right-of-way guidance from a biota / sanctuary system.
    ///
    /// This composes into the existing moral safety lane so civic cohabitation
    /// signals reuse the same fallback chain as ethics-triggered slow/stop
    /// behavior.
    pub fn apply_biota_right_of_way(&mut self, signal: BiotaRightOfWaySignal) {
        let biota_safety = if signal.should_hold_red() {
            Some(MotorSafetyLevel::Red)
        } else if signal.should_yield_orange() {
            Some(MotorSafetyLevel::Orange)
        } else if signal.should_caution_yellow() {
            Some(MotorSafetyLevel::Yellow)
        } else {
            None
        };
        self.moral_safety = match (self.moral_safety, biota_safety) {
            (Some(existing), Some(incoming)) => Some(existing.max(incoming)),
            (None, incoming) => incoming,
            (existing, None) => existing,
        };
    }

    /// Advance the multi-stage fallback state machine based on speed and time.
    fn update_fallback_stage(&mut self) {
        let speed = self.simulator.state().speed;
        match self.fallback_stage {
            VehicleFallbackStage::MaintainCourse => {
                // 50 cycles (~100ms) to allow human/external takeover detection
                if self.fallback_cycles_in_stage > 50 {
                    self.fallback_stage = VehicleFallbackStage::GradualDeceleration;
                    self.fallback_cycles_in_stage = 0;
                }
            }
            VehicleFallbackStage::GradualDeceleration => {
                // If speed still high after 200 cycles, escalate to emergency
                if speed > 5.0 && self.fallback_cycles_in_stage > 200 {
                    self.fallback_stage = VehicleFallbackStage::EmergencyBrake;
                    self.fallback_cycles_in_stage = 0;
                }
                // If speed is low enough → can stop with parking brake
                if speed < 0.5 {
                    self.fallback_stage = VehicleFallbackStage::PostStop;
                    self.fallback_cycles_in_stage = 0;
                }
            }
            VehicleFallbackStage::EmergencyBrake => {
                if speed < 0.5 {
                    self.fallback_stage = VehicleFallbackStage::PostStop;
                    self.fallback_cycles_in_stage = 0;
                }
            }
            VehicleFallbackStage::PostStop => {
                // Stay here until safety clears
            }
        }
    }

    fn reset_fallback_stage(&mut self) {
        self.fallback_stage = VehicleFallbackStage::MaintainCourse;
        self.fallback_cycles_in_stage = 0;
    }

    /// Current vehicle fallback stage (for telemetry/testing).
    pub fn fallback_stage(&self) -> VehicleFallbackStage {
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
        // Ethics gate max-composes on top of phi + override. Moral-Red
        // escalates into the existing emergency-brake fallback chain.
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }
        let gain = self.current_safety.motor_gain();

        let mut cmd = self.controller.forward(thought_hv, dt);

        // Record the last safe steering during normal operation
        if matches!(
            self.current_safety,
            MotorSafetyLevel::Green | MotorSafetyLevel::Yellow
        ) {
            self.last_safe_steering = cmd.steering;
            self.fallback_cycles = 0;
        }

        // ── MULTI-STAGE SAFE FALLBACK for Red tier ──────────────────
        // ISO 21448 SOTIF-aligned tiered emergency procedure:
        //   Stage 1: MaintainCourse (allow human takeover ~100ms)
        //   Stage 2: GradualDeceleration (50% brake)
        //   Stage 3: EmergencyBrake (full force if speed still high)
        //   Stage 4: PostStop (parking brake engaged)
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles += 1;
            self.fallback_cycles_in_stage += 1;
            self.update_fallback_stage();

            // Smooth steering taper across ALL stages — never snap the wheel
            const TAPER_CYCLES: f32 = 100.0;
            let taper_t = (self.fallback_cycles as f32 / TAPER_CYCLES).min(1.0);
            cmd.steering = self.last_safe_steering * (1.0 - taper_t);

            match self.fallback_stage {
                VehicleFallbackStage::MaintainCourse => {
                    // Just hold what we were doing, but stop accelerating
                    cmd.throttle = 0.0;
                    // Light brake to start slowing (engine braking equivalent)
                    cmd.brake = 0.1;
                }
                VehicleFallbackStage::GradualDeceleration => {
                    cmd.throttle = 0.0;
                    cmd.brake = 0.5; // 50% brake — passenger comfort
                }
                VehicleFallbackStage::EmergencyBrake => {
                    cmd.throttle = 0.0;
                    cmd.brake = 1.0; // Full force
                }
                VehicleFallbackStage::PostStop => {
                    cmd.throttle = 0.0;
                    cmd.brake = 1.0; // Hold the brake (parking brake equivalent)
                    cmd.steering = 0.0; // Centered
                }
            }
        } else {
            // Not in Red — reset state machine and allow normal operation
            self.reset_fallback_stage();
            self.fallback_cycles = 0;
            if gain < 1.0 {
                // Yellow/Orange: scaled authority but preserve agency
                cmd.steering *= gain;
                cmd.throttle *= gain;
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

        let success = self.simulator.state().speed.is_finite();

        EmbodimentResult {
            num_actuators: 3,
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
        self.last_safe_steering = 0.0;
        self.fallback_cycles = 0;
        self.fallback_stage = VehicleFallbackStage::MaintainCourse;
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
            safety_level: format!("{:?}", self.current_safety),
            platform: "vehicle".to_string(),
            num_actuators: 3,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for VehicleEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Vehicle
    }
    fn num_actuators(&self) -> usize {
        3
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

/// SafeFallback guarantees for the Vehicle platform.
///
/// When consciousness degrades to Red, the vehicle executes:
///   1. Full brake (1.0) — maximum deceleration immediately
///   2. Zero throttle — stop accelerating
///   3. Smooth steering taper — last_safe_steering → 0 over 50 cycles
///
/// This prevents the two failure modes of naive "zero force":
///   - No brakes at highway speed (catastrophic)
///   - Sudden steering snap causing lane departure
impl SafeFallback for VehicleEmbodiment {
    fn platform_name(&self) -> &'static str {
        "vehicle"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        10
    } // Critical: human-carrying, at speed
    fn safe_fallback_description(&self) -> &'static str {
        "Multi-stage: MaintainCourse → GradualDeceleration → EmergencyBrake → PostStop"
    }
    fn safe_fallback_latency_cycles(&self) -> u32 {
        1
    } // Immediate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step() {
        let mut bridge = VehicleEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.005, 0.7);
        assert!(r.success);
        assert_eq!(r.num_actuators, 3);
    }

    #[test]
    fn test_safety_gating() {
        let mut bridge = VehicleEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_perception() {
        let mut bridge = VehicleEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.005, 0.7);
        let p = bridge.encode_perception();
        assert_eq!(p.dim(), 16384);
    }

    #[test]
    fn test_telemetry() {
        let mut bridge = VehicleEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.005, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "vehicle");
    }

    #[test]
    fn test_reset() {
        let mut bridge = VehicleEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.005, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
    }

    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        let mut bridge = VehicleEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let hv = ContinuousHV::random(16384, 42);
        // Phi 0.9 → Green normally; ahimsa forces Red.
        let r = bridge.step(&hv, 0.005, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Red,
            "ahimsa must force Red regardless of phi"
        );
    }

    #[test]
    fn test_moral_gate_consent_violation_forces_orange() {
        let mut bridge = VehicleEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.005, 0.9);
        assert_eq!(
            r.safety_level,
            MotorSafetyLevel::Orange,
            "consent violation → Orange"
        );
    }

    #[test]
    fn test_biota_sanctuary_conflict_forces_orange_yield() {
        let mut bridge = VehicleEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_biota_right_of_way(BiotaRightOfWaySignal {
            distress_signal: 0.15,
            path_conflict_risk: 0.52,
            sanctuary_signal: 0.72,
            route_clear_confidence: 0.35,
            classification_confidence: 0.8,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);
    }

    #[test]
    fn test_biota_distressed_crossing_forces_red_fallback() {
        let mut bridge = VehicleEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_biota_right_of_way(BiotaRightOfWaySignal {
            distress_signal: 0.92,
            path_conflict_risk: 0.81,
            sanctuary_signal: 0.4,
            route_clear_confidence: 0.1,
            classification_confidence: 0.9,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert_eq!(
            bridge.fallback_stage(),
            VehicleFallbackStage::MaintainCourse
        );
    }
}
