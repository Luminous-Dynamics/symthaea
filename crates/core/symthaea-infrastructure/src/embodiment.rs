// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::controller::InfrastructureController;
use crate::encoder::InfrastructureHdcEncoder;
use crate::simulator::{InfrastructurePhysicsSimulator, SimpleInfrastructureSimulator};
use crate::types::{
    ClimeInfrastructureSignal, InfrastructureCommand, InfrastructureConfig, THERMAL_RUNAWAY_RISK,
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Construct the default simulator backend.
///
/// Behind the `grid_physics` feature, this is the real-electrical-physics
/// `GridPhysicsInfrastructureSimulator` (battery SoC, radial feeder power
/// flow, droop, trip envelope, islanding detection — see
/// `grid_physics_simulator.rs`). Without it, the fast coupled-heuristic
/// `SimpleInfrastructureSimulator`.
///
/// Previously `InfrastructureEmbodiment` hardcoded the heuristic simulator
/// as a concrete field type, so `GridPhysicsInfrastructureSimulator` was
/// unreachable from the actually-driven agent regardless of the
/// `grid_physics` feature (2026-07-07 unaudited-platforms review, Tier 4).
fn make_default_simulator() -> Box<dyn InfrastructurePhysicsSimulator> {
    #[cfg(feature = "grid_physics")]
    {
        Box::new(crate::grid_physics_simulator::GridPhysicsInfrastructureSimulator::new())
    }
    #[cfg(not(feature = "grid_physics"))]
    {
        Box::new(SimpleInfrastructureSimulator::new())
    }
}

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, SafeFallback, grounding_from_prediction_error, grounding_label,
};

/// Emergency fallback behavior for a stationary infrastructure node.
///
/// Unlike mobile platforms, infrastructure "does not abort — it escalates
/// operational posture" (see `docs/robotics/INFRASTRUCTURE_HAZARDS_AND_SENSORIUM.md`).
/// Zeroing every actuator at Red (the trait's plain default) is wrong here for
/// the same reason it was wrong for the AUV: a fault-condition node that stops
/// discharging stranded storage and stops isolating from a failing external tie
/// can leave dependents unserved *and* propagate the fault outward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfrastructureFallbackStage {
    /// Open all external routing (island from the broader grid/mesh), halt
    /// charging to protect the storage reserve, discharge storage to serve
    /// local critical load, and run active cooling if thermal risk is
    /// elevated. Full authority, not gain-scaled.
    SafeIsland,
}

/// Cooling effort commanded during SafeIsland when thermal runaway risk is
/// already elevated — enough to arrest a developing thermal event without
/// assuming full undamaged cooling-loop capacity.
const SAFE_ISLAND_COOLING_ELEVATED: f32 = 1.0;
/// Baseline cooling effort during SafeIsland when thermal risk is not yet
/// elevated — keep the loop live at reduced duty rather than fully idle.
const SAFE_ISLAND_COOLING_BASELINE: f32 = 0.3;
/// Thermal runaway risk threshold above which SafeIsland commands full cooling.
const SAFE_ISLAND_THERMAL_THRESHOLD: f64 = 0.3;

pub struct InfrastructureEmbodiment {
    controller: InfrastructureController,
    simulator: Box<dyn InfrastructurePhysicsSimulator>,
    encoder: InfrastructureHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    civic_safety: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
    fallback_stage: InfrastructureFallbackStage,
    fallback_cycles_in_stage: u32,
}

impl InfrastructureEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = InfrastructureConfig::default();
        Self {
            controller: InfrastructureController::new(genesis, &config),
            simulator: make_default_simulator(),
            encoder: InfrastructureHdcEncoder::new(genesis, 32),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            civic_safety: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
            fallback_stage: InfrastructureFallbackStage::SafeIsland,
            fallback_cycles_in_stage: 0,
        }
    }

    /// Apply habitat and atmosphere pressure from a Clime platform.
    ///
    /// This composes civic building-safety demand into the same safety ladder
    /// already used for phi degradation and manual overrides.
    pub fn apply_clime_habitat_signal(&mut self, signal: ClimeInfrastructureSignal) {
        let incoming = if signal.should_emergency_red() {
            Some(MotorSafetyLevel::Red)
        } else if signal.should_isolate_orange() {
            Some(MotorSafetyLevel::Orange)
        } else if signal.should_conserve_yellow() {
            Some(MotorSafetyLevel::Yellow)
        } else {
            None
        };
        self.civic_safety = match (self.civic_safety, incoming) {
            (Some(existing), Some(next)) => Some(existing.max(next)),
            (None, next) => next,
            (existing, None) => existing,
        };
    }

    /// Apply moral gate from the ethics engine.
    ///
    /// Infrastructure nodes serve dependents (critical-load customers) who
    /// cannot consent to being cut off — ahimsa forces Red (SafeIsland: serve
    /// critical load from local storage rather than merely halting), a
    /// consent violation (e.g. unauthorized load-shed of a protected
    /// customer) forces Orange, caution forces a Yellow cap.
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

    /// SafeFallback: SafeIsland. Overrides the command with an isolate +
    /// discharge-to-critical-load posture, executed at full authority (not
    /// scaled by motor_gain).
    fn apply_safe_island(&self, cmd: &mut InfrastructureCommand) {
        let thermal_elevated =
            self.simulator.state().channels[THERMAL_RUNAWAY_RISK] >= SAFE_ISLAND_THERMAL_THRESHOLD;
        *cmd = InfrastructureCommand::zero();
        cmd.torques[0] = 0.0; // charge_bus: stop charging, protect the storage reserve
        cmd.torques[1] = 1.0; // discharge_bus: full discharge to serve local critical load
        cmd.torques[2] = if thermal_elevated {
            SAFE_ISLAND_COOLING_ELEVATED
        } else {
            SAFE_ISLAND_COOLING_BASELINE
        };
        cmd.torques[3] = 0.0; // heating_loop: off — conserve reserve for critical service
        // routing_north/south/east/west: 0.0 — open external ties, island the node
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(o) => phi_level.max(o),
            None => phi_level,
        };
        if let Some(civic) = self.civic_safety {
            self.current_safety = self.current_safety.max(civic);
        }
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }
        let gain = self.current_safety.motor_gain();
        let mut cmd = self.controller.forward(thought_hv, dt);

        // ── SafeFallback: SafeIsland at Red ──────────────────────────
        // Zeroing every actuator at Red (motor_gain=0) would stop discharging
        // stranded storage into local critical load AND leave external
        // routing open to propagate the fault outward — the opposite of
        // "escalate operational posture" from the hazards/sensorium spec.
        if matches!(self.current_safety, MotorSafetyLevel::Red) {
            self.fallback_cycles_in_stage = self.fallback_cycles_in_stage.saturating_add(1);
            self.apply_safe_island(&mut cmd);
        } else {
            self.fallback_stage = InfrastructureFallbackStage::SafeIsland;
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
        self.civic_safety = None;
        self.moral_safety = None;
        self.last_control_effort = 0.0;
        self.last_prediction_error = 0.0;
        self.fallback_stage = InfrastructureFallbackStage::SafeIsland;
        self.fallback_cycles_in_stage = 0;
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
    pub fn fallback_stage(&self) -> InfrastructureFallbackStage {
        self.fallback_stage
    }

    /// Which simulator backend is actually driving this embodiment —
    /// "grid_physics" (real battery/feeder/droop/islanding physics) when the
    /// `grid_physics` feature is enabled, "heuristic" otherwise. See
    /// `make_default_simulator`.
    pub fn simulator_backend_name(&self) -> &'static str {
        self.simulator.backend_name()
    }

    /// Serialize operating mode and key grid-health channels as JSON bytes,
    /// for `robotics-dispatch` telemetry integration.
    pub fn platform_telemetry_bytes(&self) -> Vec<u8> {
        let state = self.simulator.state();
        serde_json::to_vec(&serde_json::json!({
            "operating_mode": format!("{:?}", state.inferred_mode()),
            "storage_ratio": state.storage_ratio(),
            "voltage_stability": state.voltage_stability(),
            "brownout_risk": state.brownout_risk(),
            "shed_load_ratio": state.shed_load_ratio(),
            "unserved_demand_ratio": state.unserved_demand_ratio(),
            "islanding_risk": state.islanding_risk(),
            "service_integrity": state.service_integrity(),
            "thermal_runaway_risk": state.thermal_runaway_risk(),
        }))
        .unwrap_or_default()
    }

    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: self.current_safety,
            platform: "infrastructure".to_string(),
            num_actuators: 8,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: self.platform_telemetry_bytes(),
        }
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for InfrastructureEmbodiment {
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

    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::Infrastructure
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

    fn platform_telemetry_bytes(&self) -> Vec<u8> {
        self.platform_telemetry_bytes()
    }
}

impl SafeFallback for InfrastructureEmbodiment {
    fn platform_name(&self) -> &'static str {
        "infrastructure"
    }
    fn current_safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }
    fn safe_fallback_priority(&self) -> u8 {
        5 // High: in-contact with dependents (critical-load customers), stationary
    }
    fn safe_fallback_description(&self) -> &'static str {
        "SafeIsland: open external routing, halt charging, discharge storage to critical load, active cooling if thermal risk elevated"
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
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.7);
        assert!(r.success);
    }
    #[test]
    fn test_red_halts() {
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    #[cfg(not(feature = "grid_physics"))]
    fn test_default_backend_is_heuristic() {
        let e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        assert_eq!(e.simulator_backend_name(), "heuristic");
    }

    #[test]
    #[cfg(feature = "grid_physics")]
    fn test_grid_physics_feature_actually_wires_the_real_backend() {
        // Regression for Tier 4 of SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md:
        // GridPhysicsInfrastructureSimulator existed and was well-tested in
        // isolation, but InfrastructureEmbodiment hardcoded the heuristic
        // simulator as a concrete field type, so the real backend was
        // unreachable from the actually-driven agent regardless of this
        // feature flag. This test fails unless that wiring is genuinely live.
        let e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        assert_eq!(e.simulator_backend_name(), "grid_physics");
    }

    #[test]
    fn test_clime_isolation_pressure_forces_orange() {
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.apply_clime_habitat_signal(ClimeInfrastructureSignal {
            smoke_risk: 0.52,
            contamination_risk: 0.61,
            public_health_risk: 0.42,
            utility_reserve: 0.28,
            zone_isolation_confidence: 0.73,
            sensor_confidence: 0.84,
        });
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);
    }

    #[test]
    fn test_clime_smoke_emergency_forces_red() {
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.apply_clime_habitat_signal(ClimeInfrastructureSignal {
            smoke_risk: 0.82,
            contamination_risk: 0.25,
            public_health_risk: 0.3,
            utility_reserve: 0.61,
            zone_isolation_confidence: 0.79,
            sensor_confidence: 0.87,
        });
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_red_commands_safe_island_not_zero() {
        // At Red, the fallback must command a deliberate discharge + island
        // posture (torques[1] discharge_bus = 1.0), NOT the trait's plain
        // zero-command default — zeroing everything would starve local
        // critical load AND leave external routing open.
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert!(
            r.control_effort > 0.0,
            "SafeIsland must command non-zero control effort (discharge to critical load), got {}",
            r.control_effort
        );
        assert_eq!(e.fallback_stage(), InfrastructureFallbackStage::SafeIsland);
    }

    #[test]
    fn test_safe_island_charging_halted_and_routing_opened() {
        let e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let mut cmd = InfrastructureCommand::zero();
        cmd.torques[0] = 1.0; // pretend controller wanted to charge
        cmd.torques[4] = 1.0; // pretend controller wanted external routing
        e.apply_safe_island(&mut cmd);
        assert_eq!(
            cmd.charge_bus(),
            0.0,
            "charging must halt to protect reserve"
        );
        assert_eq!(
            cmd.discharge_bus(),
            1.0,
            "must discharge to serve critical load"
        );
        assert_eq!(cmd.torques[4], 0.0, "external routing must open (island)");
    }

    #[test]
    fn test_ahimsa_violation_forces_red_safe_island() {
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.9); // high phi would otherwise be Green
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
        assert!(r.control_effort > 0.0, "must execute SafeIsland, not zero");
    }

    #[test]
    fn test_moral_blocked_forces_red() {
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_BLOCKED,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_moral_consent_violation_forces_orange() {
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: true,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);
    }

    #[test]
    fn test_moral_caution_caps_yellow() {
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_CAUTION,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.9); // phi alone -> Green
        assert_eq!(r.safety_level, MotorSafetyLevel::Yellow);
    }

    #[test]
    fn test_moral_safe_does_not_override_phi() {
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        e.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: false,
        });
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let r = e.step(&hv, 0.005, 0.9);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_platform_telemetry_bytes_populated() {
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        e.step(&hv, 0.005, 0.9);
        let bytes = e.platform_telemetry_bytes();
        assert!(!bytes.is_empty());
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(parsed.get("operating_mode").is_some());
        assert!(parsed.get("storage_ratio").is_some());
        let t = e.telemetry();
        assert!(!t.platform_specific.is_empty());
    }

    #[test]
    fn test_safe_fallback_trait_impl() {
        use symthaea_core::embodiment::SafeFallback;
        let mut e = InfrastructureEmbodiment::new(&GenesisSeed::from_phrase("test"));
        assert_eq!(e.safe_fallback_priority(), 5);
        assert_eq!(e.safe_fallback_latency_cycles(), 1);
        assert!(e.safe_fallback_description().contains("SafeIsland"));
        assert!(!e.safe_fallback_active());
        let hv = ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        e.step(&hv, 0.005, 0.05);
        assert!(e.safe_fallback_active());
    }
}
