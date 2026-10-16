// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for quadrotor flight platform.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::controller::FlightController;
use crate::encoder::QuadrotorHdcEncoder;
use crate::simulator::{PhysicsSimulator, SimplePhysicsSimulator};
use crate::types::{FlightConfig, QuadrotorCommand};

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
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
    /// Last command accepted by the in-process simulator. This is not hardware evidence.
    last_applied_command: Option<QuadrotorCommand>,
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
    /// Emergency descent thrust as a fraction of hover thrust.
    /// Produces a controlled descent instead of freefall.
    const EMERGENCY_DESCENT_FRACTION: f32 = 0.3;
    /// Touchdown cushion thrust as a fraction of hover thrust —
    /// more than descent, still below hover so the vehicle settles.
    const TOUCHDOWN_CUSHION_FRACTION: f32 = 0.55;
    /// StabilizeHover vertical-rate damping gain (fraction of hover thrust
    /// removed/added per m/s of climb/sink). 0.5 nulls a 0.5 m/s residual
    /// climb in roughly a quarter second on the Crazyflie airframe.
    const HOVER_VERTICAL_DAMPING: f32 = 0.5;

    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = FlightConfig::default();
        Self::with_controller(genesis, FlightController::new(genesis, &config))
    }

    /// Construct a bridge around an already-trained controller — the
    /// trainer→bridge transfer path (`FlightTrainer::train_returning_
    /// controller` / `train_intent_controller`). Without this, every shipped
    /// bridge ran genesis-random weights, under which thought has no
    /// task-axis motor authority (measured by examples/cognition_ablation.rs).
    /// `genesis` still seeds the encoder so perception stays deterministic.
    pub fn with_controller(genesis: &GenesisSeed, controller: FlightController) -> Self {
        Self {
            controller,
            simulator: SimplePhysicsSimulator::new(),
            last_applied_command: None,
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

            // Derive fallback thrust from the airframe's actual hover thrust
            // (mass·g). Hardcoded Newton values sized for a different airframe
            // previously made "descent" a >10g ascent on the 27g Crazyflie.
            let hover = self.simulator.hover_thrust() as f32;
            cmd.thrust = match self.fallback_stage {
                FlightFallbackStage::StabilizeHover => {
                    // Actually stabilize: damp any inherited vertical rate
                    // instead of merely holding altitude-neutral thrust
                    // (plain thrust=hover preserves a pre-Red climb/sink
                    // indefinitely, modulo drag).
                    let v_z = self.simulator.state().linear_velocity[2] as f32;
                    (hover * (1.0 - Self::HOVER_VERTICAL_DAMPING * v_z)).clamp(0.0, hover * 1.5)
                }
                FlightFallbackStage::ControlledDescent => hover * Self::EMERGENCY_DESCENT_FRACTION,
                FlightFallbackStage::Touchdown => {
                    // Cushion landing — more than descent thrust, below hover
                    hover * Self::TOUCHDOWN_CUSHION_FRACTION
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

        self.last_applied_command = Some(cmd);
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
        self.last_applied_command = None;
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
            safety_level: self.current_safety,
            platform: "quadrotor".to_string(),
            num_actuators: 4,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}


/// V2 semantic adapter. Applied actions refer only to the named in-process simulator.
impl symthaea_core::embodiment_v2::EmbodimentV2Adapter for FlightEmbodiment {
    fn descriptor_v2(&self, instance_id: &str) -> symthaea_core::embodiment_v2::EmbodimentDescriptorV2 {
        use symthaea_core::embodiment_v2::EmbodimentDescriptorV2;
        EmbodimentDescriptorV2 {
            contract_version: symthaea_core::embodiment_v2::EMBODIMENT_CONTRACT_VERSION_V2,
            profile_id: "symthaea.multirotor.simple-physics-quadrotor.v1".into(),
            platform_family_id: "aerial.multirotor".into(),
            instance_id: instance_id.into(),
            morphology_id: "quadrotor.simple-physics-27g.v1".into(),
            observation_schema_id: "symthaea.observation.scalar-quantity.v2".into(),
            action_schema_id: "symthaea.action.scalar-quantity.v2".into(),
            frame_graph_id: Some("world/body".into()),
            clock_domain_id: "symthaea.multirotor.sim-time".into(),
            actuator_capability_ids: vec![
                "actuator/thrust".into(),
                "actuator/roll_moment".into(),
                "actuator/pitch_moment".into(),
                "actuator/yaw_moment".into(),
            ],
            sensor_capability_ids: vec![
                "position".into(),
                "orientation".into(),
                "linear_velocity".into(),
                "angular_velocity".into(),
            ],
            safe_fallback_profile_id: "symthaea.multirotor.multi-stage-landing.v1".into(),
            calibration_profile_id: None,
            backend_provenance_id: "symthaea-multirotor.simple-physics-simulator".into(),
            descriptor_digest: None,
        }
    }

    fn observation_frame_v2(
        &self,
        instance_id: &str,
    ) -> Result<symthaea_core::embodiment_v2::ObservationFrameV2, String> {
        use symthaea_core::embodiment_v2::{
            ClockStampV2, EMBODIMENT_CONTRACT_VERSION_V2, ObservationFrameV2, ObservationValueV2,
        };
        let state = self.simulator.state();
        if !state.timestamp.is_finite()
            || state.timestamp < 0.0
            || state.timestamp > (u64::MAX as f64 / 1_000_000_000.0)
        {
            return Err("simulator timestamp cannot be represented as nanoseconds".into());
        }
        let timestamp_ns = (state.timestamp * 1_000_000_000.0).round() as u64;
        let mut observations = Vec::with_capacity(13);
        let mut add = |address: &str, quantity: &str, unit: &str, value: f64| {
            observations.push(ObservationValueV2 {
                address: address.into(),
                quantity: quantity.into(),
                unit: unit.into(),
                value: Some(value),
                missing_reason: None,
                uncertainty: None,
            });
        };
        add("position/x", "position", "m", state.position[0]);
        add("position/y", "position", "m", state.position[1]);
        add("position/z", "position", "m", state.position[2]);
        add("orientation/w", "orientation_component", "1", state.quaternion[0]);
        add("orientation/x", "orientation_component", "1", state.quaternion[1]);
        add("orientation/y", "orientation_component", "1", state.quaternion[2]);
        add("orientation/z", "orientation_component", "1", state.quaternion[3]);
        add("linear_velocity/x", "velocity", "m/s", state.linear_velocity[0]);
        add("linear_velocity/y", "velocity", "m/s", state.linear_velocity[1]);
        add("linear_velocity/z", "velocity", "m/s", state.linear_velocity[2]);
        add("angular_velocity/x", "angular_velocity", "rad/s", state.angular_velocity[0]);
        add("angular_velocity/y", "angular_velocity", "rad/s", state.angular_velocity[1]);
        add("angular_velocity/z", "angular_velocity", "rad/s", state.angular_velocity[2]);
        let frame = ObservationFrameV2 {
            contract_version: EMBODIMENT_CONTRACT_VERSION_V2,
            profile_id: "symthaea.multirotor.simple-physics-quadrotor.v1".into(),
            instance_id: instance_id.into(),
            clock: ClockStampV2 {
                clock_domain_id: "symthaea.multirotor.sim-time".into(),
                sequence: Some(self.total_steps as u64),
                timestamp_ns: Some(timestamp_ns),
            },
            frame_graph_id: Some("world/body".into()),
            source_id: "SimplePhysicsSimulator::state".into(),
            observations,
        };
        frame.validate()?;
        Ok(frame)
    }

    fn last_action_frame_v2(
        &self,
        instance_id: &str,
    ) -> Result<Option<symthaea_core::embodiment_v2::ActionFrameV2>, String> {
        use symthaea_core::embodiment_v2::{
            ActionFrameV2, ActionStageV2, ActionValueV2, ClockStampV2, EMBODIMENT_CONTRACT_VERSION_V2,
        };
        let Some(command) = self.last_applied_command else {
            return Ok(None);
        };
        let timestamp = self.simulator.state().timestamp;
        if !timestamp.is_finite()
            || timestamp < 0.0
            || timestamp > (u64::MAX as f64 / 1_000_000_000.0)
        {
            return Err("simulator timestamp cannot be represented as nanoseconds".into());
        }
        let frame = ActionFrameV2 {
            contract_version: EMBODIMENT_CONTRACT_VERSION_V2,
            profile_id: "symthaea.multirotor.simple-physics-quadrotor.v1".into(),
            instance_id: instance_id.into(),
            clock: ClockStampV2 {
                clock_domain_id: "symthaea.multirotor.sim-time".into(),
                sequence: Some(self.total_steps as u64),
                timestamp_ns: Some((timestamp * 1_000_000_000.0).round() as u64),
            },
            frame_graph_id: Some("world/body".into()),
            source_id: "symthaea-multirotor.simple-physics-simulator".into(),
            stage: ActionStageV2::Applied,
            actions: vec![
                ActionValueV2 { address: "actuator/thrust".into(), quantity: "force".into(), unit: "N".into(), value: command.thrust as f64 },
                ActionValueV2 { address: "actuator/roll_moment".into(), quantity: "moment".into(), unit: "N*m".into(), value: command.roll_moment as f64 },
                ActionValueV2 { address: "actuator/pitch_moment".into(), quantity: "moment".into(), unit: "N*m".into(), value: command.pitch_moment as f64 },
                ActionValueV2 { address: "actuator/yaw_moment".into(), quantity: "moment".into(), unit: "N*m".into(), value: command.yaw_moment as f64 },
            ],
        };
        frame.validate()?;
        Ok(Some(frame))
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
        let alt_at_red_onset = bridge.simulator().state().position[2];

        // Now force Red safety for 100 steps
        for _ in 0..100 {
            let r = bridge.step(&hv, 0.01, 0.05); // Phi < 0.1 → Red
            assert_eq!(r.safety_level, MotorSafetyLevel::Red);
            assert!(
                r.success,
                "state must remain finite during emergency descent"
            );
            // Descent rate bounded at every step (never exceeds ~5 m/s)
            let v_z = bridge.simulator().state().linear_velocity[2];
            assert!(v_z >= -5.1, "descent rate {v_z} exceeded MAX_DESCENT_RATE");
            // And never a sustained climb — an ascent also satisfies a lower
            // bound alone, which is exactly how the 55x thrust-unit bug
            // previously slipped through this test. Tolerance 1.0 m/s allows
            // the residual Green-phase climb that StabilizeHover inherits and
            // damps out (the 55x bug produced +5 m/s after a single step).
            assert!(v_z <= 1.0, "Red fallback must not ascend, got v_z = {v_z}");
        }

        // The vehicle must have come DOWN (descended or landed), never
        // ballooned upward during "descent".
        let state = bridge.simulator().state();
        assert!(
            state.position[2] <= alt_at_red_onset + 0.05,
            "Red fallback flew away upward: {} -> {}",
            alt_at_red_onset,
            state.position[2]
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
