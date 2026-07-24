// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EmbodimentBridge implementation for 7-DOF industrial manipulator.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::admittance::apply_admittance;
use crate::controller::ManipulatorController;
use crate::encoder::ManipulatorHdcEncoder;
use crate::kitchen_scenario::{clamp_grip_command, hazard_tier, KitchenObject};
use crate::predictor::normalized_state_prediction_error;
use crate::safety_supervisor::{ManipulatorSafetySupervisor, SafetyIntervention};
use crate::sensorimotor::{ControllerInputSchema, SensorimotorInputBinder};
use crate::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};
use crate::types::ManipulatorConfig;

pub use symthaea_core::embodiment::{
    grounding_from_prediction_error, grounding_label, EmbodimentResult, EmbodimentTelemetry,
    GroundingEstimator, MoralGateInput, MotorSafetyLevel, SafeFallback, GROUNDING_SENSORIMOTOR,
};

/// Emergency fallback posture for the manipulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManipulatorFallbackStage {
    /// Hold current pose via per-joint gravity-compensation torque; gripper frozen.
    GravityHold,
}

/// Manipulator embodiment bridge.
pub struct ManipulatorEmbodiment {
    controller: ManipulatorController,
    simulator: SimpleManipulatorSimulator,
    encoder: ManipulatorHdcEncoder,
    sensorimotor_input: SensorimotorInputBinder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    moral_safety: Option<MotorSafetyLevel>,
    grounding: GroundingEstimator,
    last_grounding: u8,
    fallback_stage: ManipulatorFallbackStage,
    fallback_cycles_in_stage: u32,
    safety_supervisor: ManipulatorSafetySupervisor,
    last_safety_intervention: SafetyIntervention,
    last_workspace_clearance_m: f64,
    last_measured_force_n: f64,
    /// Object currently grasped, if any — see `kitchen_scenario.rs`. An
    /// environmental hazard (hot/sharp), not a Φ/confidence signal, so it
    /// composes into `current_safety` the same way `safety_override` and
    /// `moral_safety` already do: as an independent `.max()` floor.
    held_object: Option<KitchenObject>,
}

impl ManipulatorEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = ManipulatorConfig::default();
        let simulator = SimpleManipulatorSimulator::new();
        let safety_supervisor = ManipulatorSafetySupervisor::new(simulator.state().joint_angles);
        Self {
            controller: ManipulatorController::new(genesis, &config),
            simulator,
            encoder: ManipulatorHdcEncoder::new(genesis, 32),
            sensorimotor_input: SensorimotorInputBinder::new(genesis),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            moral_safety: None,
            grounding: GroundingEstimator::new(),
            last_grounding: GROUNDING_SENSORIMOTOR,
            fallback_stage: ManipulatorFallbackStage::GravityHold,
            fallback_cycles_in_stage: 0,
            safety_supervisor,
            last_safety_intervention: SafetyIntervention::None,
            last_workspace_clearance_m: f64::INFINITY,
            last_measured_force_n: 0.0,
            held_object: None,
        }
    }

    /// Current SafeFallback stage (always `GravityHold` for this platform).
    pub fn fallback_stage(&self) -> ManipulatorFallbackStage {
        self.fallback_stage
    }

    /// Apply moral gate from the ethics engine.
    ///
    /// For the manipulator (force on humans/objects), ethics gating is critical:
    /// - Blocked → Red (power cut)
    /// - Caution → cap at Yellow (reduced force)
    /// - consent_violation → Orange (retreat to home)
    /// - ahimsa_violated → Red (emergency stop)
    pub fn apply_moral_gate(&mut self, gate: MoralGateInput) {
        if gate.ahimsa_violated || gate.verdict == MoralGateInput::VERDICT_BLOCKED {
            self.moral_safety = Some(MotorSafetyLevel::Red);
        } else if gate.consent_violation {
            self.moral_safety = Some(MotorSafetyLevel::Orange);
        } else if gate.verdict == MoralGateInput::VERDICT_CAUTION {
            self.moral_safety = Some(MotorSafetyLevel::Yellow);
        } else {
            self.moral_safety = None;
        }
    }

    /// Install trained output-layer weights (from `ManipulatorTrainer::
    /// export_weights` / `train_intent_weights`, or a persisted snapshot)
    /// into the live bridge. This is the trainer→bridge transfer path:
    /// without it the shipped bridge can only ever run genesis-random
    /// weights, under which thought has no task-axis motor authority
    /// (measured by examples/cognition_ablation.rs).
    pub fn install_weights(
        &mut self,
        weights: &crate::controller::ControllerWeights,
    ) -> Result<(), String> {
        if weights.input_schema != ControllerInputSchema::SensorimotorBoundV1 {
            return Err(format!(
                "controller input schema mismatch: snapshot {:?}, runtime {:?}",
                weights.input_schema,
                ControllerInputSchema::SensorimotorBoundV1
            ));
        }
        self.controller.import_weights(weights)
    }

    /// Read access to the simulated body state (task-space observables for
    /// tests, benchmarks, and telemetry consumers).
    pub fn body_state(&self) -> &crate::types::ManipulatorState {
        self.simulator.state()
    }

    pub fn safety_supervisor(&self) -> &ManipulatorSafetySupervisor {
        &self.safety_supervisor
    }

    pub fn safety_supervisor_mut(&mut self) -> &mut ManipulatorSafetySupervisor {
        &mut self.safety_supervisor
    }

    pub fn last_applied_command(&self) -> crate::types::ManipulatorCommand {
        self.safety_supervisor.last_applied_command()
    }

    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    /// Report that `object` is now grasped — its own hazards (hot/sharp) will
    /// floor `current_safety` and cap grip force from the next `step()` on.
    pub fn set_held_object(&mut self, object: KitchenObject) {
        self.held_object = Some(object);
    }

    /// Report that nothing is grasped (dropped/released/never picked up).
    pub fn clear_held_object(&mut self) {
        self.held_object = None;
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        // Effective safety = max(phi_safety, safety_override, moral_safety, hazard_tier)
        // Higher enum variant = stricter (Green < Yellow < Orange < Red)
        self.current_safety = phi_level;
        if let Some(override_level) = self.safety_override {
            self.current_safety = self.current_safety.max(override_level);
        }
        if let Some(moral_level) = self.moral_safety {
            self.current_safety = self.current_safety.max(moral_level);
        }
        // A held object's own hazards (hot/sharp) floor the tier regardless
        // of Φ — see kitchen_scenario.rs's module doc for why this needs no
        // new gating machinery beyond MotorSafetyLevel's derived Ord.
        self.current_safety = self
            .current_safety
            .max(hazard_tier(self.held_object.as_ref()));
        let body_hv = self.encoder.encode(self.simulator.state());
        let control_input = self.sensorimotor_input.fuse(thought_hv, &body_hv);
        let mut cmd = self.controller.forward(&control_input, dt);

        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
        } else {
            self.fallback_stage = ManipulatorFallbackStage::GravityHold;
            self.fallback_cycles_in_stage = 0;

            apply_admittance(
                &mut cmd,
                self.simulator.kinematics(),
                &self.simulator.state().joint_angles,
                &self.simulator.state().end_effector_force,
                self.simulator.max_torques(),
                self.current_safety,
            );
            cmd.gripper =
                clamp_grip_command(cmd.gripper, self.current_safety, self.held_object.as_ref());
        }

        // Final authority: no command reaches physics without passing through
        // the stateful safety supervisor.
        let state_before = self.simulator.state().clone();
        let gravity_hold = self.simulator.gravity_compensation_torques();
        let max_torques = *self.simulator.max_torques();
        let decision = self.safety_supervisor.supervise(
            cmd,
            &state_before,
            self.simulator.kinematics(),
            &max_torques,
            gravity_hold,
            self.current_safety,
            dt as f64,
        );
        self.last_safety_intervention = decision.intervention;
        self.last_workspace_clearance_m = decision.workspace_clearance_m;
        self.last_measured_force_n = decision.measured_force_n;
        self.last_control_effort = decision.command.control_effort();
        let predicted_state = self
            .simulator
            .predict_next_state(&decision.command, dt as f64);
        self.simulator.step(&decision.command, dt as f64);

        let perception = self.encoder.encode(self.simulator.state());
        let pred_error =
            normalized_state_prediction_error(&predicted_state, self.simulator.state());
        self.last_prediction_error = pred_error;
        self.last_perception = Some(perception);
        self.total_steps += 1;

        // Compute epistemic grounding from prediction error trend.
        // Manipulator has no swarm peers, so social grounding is not possible.
        self.last_grounding = self.grounding.estimate(pred_error, None);

        let success = self.simulator.state().is_finite();

        EmbodimentResult {
            num_actuators: 8, // 7 joints + 1 gripper
            control_effort: self.last_control_effort,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: self.last_grounding,
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
        self.grounding.reset();
        self.last_grounding = GROUNDING_SENSORIMOTOR;
        self.fallback_stage = ManipulatorFallbackStage::GravityHold;
        self.fallback_cycles_in_stage = 0;
        self.safety_supervisor.reset();
        self.last_safety_intervention = SafetyIntervention::None;
        self.last_workspace_clearance_m = f64::INFINITY;
        self.last_measured_force_n = 0.0;
        self.held_object = None;
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
            platform: "manipulator".to_string(),
            num_actuators: 8,
            epistemic_grounding: grounding_label(self.last_grounding).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: self.platform_telemetry_bytes(),
        }
    }

    /// Serialize joint angles and end-effector force as JSON bytes.
    pub fn platform_telemetry_bytes(&self) -> Vec<u8> {
        let state = self.simulator.state();
        serde_json::to_vec(&serde_json::json!({
            "joint_angles": state.joint_angles,
            "ee_force": state.end_effector_force,
            "gripper": state.gripper_opening,
            "safety_intervention": format!("{:?}", self.last_safety_intervention),
            "workspace_clearance_m": self.last_workspace_clearance_m,
            "measured_force_n": self.last_measured_force_n,
            "prediction_model": "one_step_simple_dynamics",
            "controller_input_schema": "sensorimotor_bound_v1",
        }))
        .unwrap_or_default()
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for ManipulatorEmbodiment {
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
        symthaea_core::embodiment::EmbodimentPlatform::Manipulator
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
    fn apply_moral_gate(&mut self, gate: symthaea_core::embodiment::MoralGateInput) {
        self.apply_moral_gate(gate)
    }
    fn platform_telemetry_bytes(&self) -> Vec<u8> {
        self.platform_telemetry_bytes()
    }
}

impl SafeFallback for ManipulatorEmbodiment {
    fn platform_name(&self) -> &'static str {
        "manipulator"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        5 // High: grasping, in-contact (per SafeFallback trait's own scale)
    }
    fn safe_fallback_description(&self) -> &'static str {
        "GravityHold: per-joint gravity-compensation torque holds current pose; gripper frozen"
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
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.7);
        assert!(r.success);
        assert_eq!(r.num_actuators, 8);
    }

    #[test]
    fn test_safety_gating() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    /// The actual "done" proof for kitchen-scenario Phase 1: high Φ alone
    /// would report Green, but holding a hazardous object floors the live
    /// `step()` output at the hazard tier regardless — this exercises the
    /// real composed path (`step()`'s output), not just `hazard_tier()` in
    /// isolation (already covered in `kitchen_scenario.rs`'s own tests).
    #[test]
    fn test_held_hazard_floors_safety_even_at_high_phi() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);

        // High Φ alone: Green.
        let baseline = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(baseline.safety_level, MotorSafetyLevel::Green);

        // Same high Φ, but now holding a sharp+hot object: Red, despite Φ
        // never having changed — proving the hazard tier is a genuine,
        // independent floor on the live path, not a decorative field.
        bridge.set_held_object(crate::kitchen_scenario::KitchenObject {
            temperature_c: 90.0,
            max_safe_grip_force_n: 50.0,
            is_sharp: true,
        });
        let hazardous = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(hazardous.safety_level, MotorSafetyLevel::Red);

        // Releasing the object at the same Φ returns to Green.
        bridge.clear_held_object();
        let released = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(released.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_perception() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        let p = bridge.encode_perception();
        assert_eq!(p.dim(), 16384);
    }

    #[test]
    fn test_telemetry() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        let t = bridge.telemetry();
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.platform, "manipulator");
    }

    #[test]
    fn test_moral_gate_blocked_forces_red() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_BLOCKED,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9); // High phi, but blocked
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        // NOTE: this used to assert control_effort == 0.0, but that was the
        // bug this fix closes: "no force" is not safe for a loaded arm — it
        // means "drop whatever it's holding". Red now commands GravityHold
        // (a deliberate, non-zero, purposeful hold torque) instead of a
        // passive zero. See test_red_triggers_gravity_hold below for the
        // behavior this actually asserts.
        assert!(r.control_effort.is_finite());
    }

    #[test]
    fn test_moral_gate_ahimsa_forces_red() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_moral_gate_consent_forces_orange() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);
    }

    #[test]
    fn test_moral_gate_caution_caps_yellow() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_CAUTION,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9); // Phi says Green, ethics caps at Yellow
        assert_eq!(r.safety_level, MotorSafetyLevel::Yellow);
    }

    #[test]
    fn test_moral_gate_safe_no_effect() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        bridge.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_reset() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.7);
        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
    }

    #[test]
    fn test_red_triggers_gravity_hold_stage() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.05);
        assert_eq!(
            bridge.fallback_stage(),
            ManipulatorFallbackStage::GravityHold
        );
    }

    #[test]
    fn test_red_gravity_hold_commands_nonzero_torque_not_passive_zero() {
        // The core safety bug: at Red, gain=0.0 alone means "drop the load".
        // GravityHold must command a real, non-zero holding torque derived
        // from the arm's own gravity model — not simply zero everything.
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert!(
            r.control_effort > 1e-4,
            "GravityHold must command non-zero holding torque, got {}",
            r.control_effort
        );
    }

    #[test]
    fn test_red_gravity_hold_matches_simulator_gravity_compensation() {
        // The commanded torque at Red must equal the simulator's own
        // gravity-compensation computation for the pre-step pose (full
        // authority, not scaled by motor_gain).
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let expected = bridge.simulator.gravity_compensation_torques();
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.05);
        assert_eq!(bridge.last_control_effort, {
            expected.iter().map(|t| t.abs()).sum::<f32>() / expected.len() as f32
        });
    }

    #[test]
    fn test_red_gravity_hold_freezes_gripper() {
        // Gripper must stay at whatever opening it was, not snap to a
        // default — freezing state is part of "hold current pose".
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        // First step at Green to let the gripper move away from its default.
        bridge.step(&hv, 0.002, 0.9);
        let opening_before_red = bridge.simulator.state().gripper_opening;
        bridge.step(&hv, 0.002, 0.05); // now Red
        let opening_after_red = bridge.simulator.state().gripper_opening;
        assert!(
            (opening_after_red - opening_before_red).abs() < 1e-6,
            "gripper should freeze at Red, went from {} to {}",
            opening_before_red,
            opening_after_red
        );
    }

    #[test]
    fn test_safe_fallback_trait_impl() {
        let bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        assert_eq!(bridge.platform_name(), "manipulator");
        assert_eq!(bridge.safe_fallback_priority(), 5);
        assert_eq!(bridge.safe_fallback_latency_cycles(), 1);
        assert!(bridge.safe_fallback_description().contains("GravityHold"));
    }

    #[test]
    fn test_safe_fallback_active_iff_red_or_orange() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        bridge.step(&hv, 0.002, 0.9); // Green
        assert!(!bridge.safe_fallback_active());
        bridge.step(&hv, 0.002, 0.05); // Red
        assert!(bridge.safe_fallback_active());
    }

    #[test]
    fn test_non_red_tiers_still_scale_by_gain_not_gravity_hold() {
        // Yellow/Orange must retain graduated authority (scaled command),
        // not jump straight to the Red fallback.
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(16384, 42);
        let r = bridge.step(&hv, 0.002, 0.5); // Yellow, gain 0.6
        assert_eq!(r.safety_level, MotorSafetyLevel::Yellow);
        assert_eq!(bridge.fallback_cycles_in_stage, 0);
    }
    #[test]
    fn live_human_zone_is_enforced_at_actuator_boundary() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("zone-test"));
        let position = bridge.body_state().end_effector_position;
        bridge
            .safety_supervisor_mut()
            .workspace_mut()
            .human_zones
            .push([position[0], position[1], position[2], 0.2]);

        let hv = ContinuousHV::random(16384, 44);
        let result = bridge.step(&hv, 0.002, 0.8);
        assert!(result.success);
        assert_ne!(
            bridge.safety_supervisor().last_intervention(),
            SafetyIntervention::None
        );
    }

    #[test]
    fn deterministic_digital_twin_prediction_is_low_error() {
        let mut bridge = ManipulatorEmbodiment::new(&GenesisSeed::from_phrase("prediction-test"));
        let hv = ContinuousHV::random(16384, 71);
        let result = bridge.step(&hv, 0.002, 0.8);
        assert!(
            result.prediction_error < 1e-6,
            "matching digital twin should predict its own next state: {}",
            result.prediction_error
        );
    }

    #[test]
    fn bridge_rejects_legacy_direct_input_weights() {
        let genesis = GenesisSeed::from_phrase("schema-test");
        let config = ManipulatorConfig::default();
        let controller = ManipulatorController::new(&genesis, &config);
        let weights = controller.export_weights();
        let mut bridge = ManipulatorEmbodiment::new(&genesis);
        assert!(bridge.install_weights(&weights).is_err());
    }
}
