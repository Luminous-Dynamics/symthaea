// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phi-gated metric-perturbation bridge for the gravcraft platform.
//!
//! Unlike all other platforms that apply forces, this applies metric
//! perturbations to spacetime. Consciousness (Phi) directly gates
//! how much spacetime warping is allowed.
//!
//! # This is NOT an `EmbodimentBridge` implementation
//!
//! This module's doc previously claimed to be one. It is not, and correcting that
//! matters because the false claim is exactly how this would get wired up by mistake
//! (corrected 2026-07-29 during the Phase 4 platform audit,
//! `SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md`). Concretely:
//! there is no `impl EmbodimentBridge for ...` anywhere in this crate, no
//! `src/plugin.rs`, no `EmbodimentPlatform` enum variant, and no workspace crate
//! depends on it. It is a duck-typed lookalike that reuses `MotorSafetyLevel`.
//!
//! The 2026-07-29 audit found two safety gaps here. **One is now closed; one
//! remains and needs a design decision before this is wired into the fleet.**
//!
//! 1. ~~No safety-override path and no moral gate.~~ **CLOSED 2026-07-29.** This
//!    was the only Phi-consuming platform in the workspace where a `SafetyAgent`
//!    override or an ahimsa/consent verdict had *no route to the actuators*. It
//!    now composes `max(phi_level, safety_override, moral_safety)` on the derived
//!    `Ord` — the same convention as the other 15 platforms — via
//!    [`GravcraftEmbodiment::set_safety_override`] and
//!    [`GravcraftEmbodiment::apply_moral_gate`]. Enforced by
//!    `scripts/check-embodiment-safety-composition.sh`, so a future platform
//!    cannot silently repeat the omission.
//! 2. **STILL OPEN: Orange collapses into Red** — both yield
//!    `MetricCommand::default()`. That hard cliff is the same shape already found
//!    and unified away for the orbital and surgical platforms; this crate was
//!    missed because it sits outside the trait's orbit. **Deliberately not fixed
//!    here**, because unlike gap 1 it is not a missing-wiring defect with an
//!    obvious fleet convention to copy — it requires deciding what a *partially*
//!    restricted metric authority should physically mean for spacetime
//!    perturbation (a reduced-amplitude single amplifier? a different amplifier
//!    set? drift-only with a nonzero floor?). That is a platform-physics
//!    question for an owner, not a mechanical fix.

use crate::controller::MetricController;
use crate::encoder::GravcraftHdcEncoder;
use crate::simulator::MetricPerturbationSimulator;
use crate::types::{GravcraftConfig, MetricCommand};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::ContinuousHV;

pub use symthaea_core::embodiment::{
    EmbodimentResult, EmbodimentTelemetry, GROUNDING_SENSORIMOTOR, MoralGateInput,
    MotorSafetyLevel, grounding_from_prediction_error, grounding_label,
};

/// Gravcraft embodiment bridge.
///
/// Safety gating unique to this platform:
/// - Green (Phi > 0.6): full metric authority, all 3 amplifiers
/// - Yellow (0.3-0.6): single amplifier only, reduced amplitude
/// - Orange (0.1-0.3): flat metric (zero perturbation), drift only
/// - Red (< 0.1): emergency metric neutralization
pub struct GravcraftEmbodiment {
    controller: MetricController,
    simulator: MetricPerturbationSimulator,
    encoder: GravcraftHdcEncoder,
    last_perception: Option<ContinuousHV>,
    total_steps: usize,
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    moral_safety: Option<MotorSafetyLevel>,
    last_control_effort: f32,
    last_prediction_error: f32,
}

impl GravcraftEmbodiment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = GravcraftConfig::default();
        Self {
            controller: MetricController::new(genesis, &config),
            simulator: MetricPerturbationSimulator::new(config),
            encoder: GravcraftHdcEncoder::new(genesis),
            last_perception: None,
            total_steps: 0,
            current_safety: MotorSafetyLevel::Green,
            safety_override: None,
            moral_safety: None,
            last_control_effort: 0.0,
            last_prediction_error: 0.0,
        }
    }

    /// Apply a moral gate from the ethics engine, matching the fleet convention
    /// (see `symthaea-scavenger`, `symthaea-biota`, and 13 others): ahimsa or a
    /// BLOCKED verdict forces Red, a consent violation forces Orange, caution caps
    /// at Yellow. Added 2026-07-29 -- this crate previously had no route at all
    /// from an ethics verdict to its metric amplifiers.
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

    /// Force a lower (more restrictive) tier from an external `SafetyAgent`.
    /// Composed via `max` on the derived `Ord`, so an override can only ever
    /// restrict, never grant, authority.
    pub fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    /// Clear the external safety override.
    pub fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    pub fn step(&mut self, thought_hv: &ContinuousHV, dt: f32, phi: f64) -> EmbodimentResult {
        // Fleet convention: max(phi_level, safety_override, moral_safety) on the
        // derived Ord -- any input can only make the tier MORE restrictive.
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(o) => phi_level.max(o),
            None => phi_level,
        };
        if let Some(m) = self.moral_safety {
            self.current_safety = self.current_safety.max(m);
        }

        // Get raw command from controller (Phi already gates amplitude internally)
        let mut cmd = self.controller.forward(thought_hv, phi);

        // Apply safety-level restrictions
        match self.current_safety {
            MotorSafetyLevel::Green => {
                // Full authority — all 3 amplifiers active
            }
            MotorSafetyLevel::Yellow => {
                // Single amplifier only, reduced amplitude
                cmd.amplifiers[1] = (0.0, 0.0, 0.0, 0.0);
                cmd.amplifiers[2] = (0.0, 0.0, 0.0, 0.0);
                let (amp, f, a, e) = cmd.amplifiers[0];
                cmd.amplifiers[0] = (amp * 0.5, f, a, e);
            }
            MotorSafetyLevel::Orange => {
                // Flat metric — zero perturbation, drift only
                cmd = MetricCommand::default();
            }
            MotorSafetyLevel::Red => {
                // Emergency metric neutralization
                cmd = MetricCommand::default();
            }
        }

        // Compute control effort (sum of absolute amplitudes)
        let effort: f64 = cmd.amplifiers.iter().map(|(a, _, _, _)| a.abs()).sum();
        self.last_control_effort = effort as f32;

        // Step the simulator
        self.simulator.step(&cmd, dt as f64);

        // Encode perception
        let perception = self.encoder.encode(self.simulator.state());

        let pred_error = if let Some(ref prev) = self.last_perception {
            (1.0 - perception.similarity(prev).max(0.0)).min(1.0)
        } else {
            0.0_f32
        };
        self.last_prediction_error = pred_error;
        self.last_perception = Some(perception);
        self.total_steps += 1;

        let success = self
            .simulator
            .state()
            .position
            .iter()
            .all(|p| p.is_finite());

        EmbodimentResult {
            num_actuators: 12, // 3 amplifiers × 4 controls
            control_effort: effort as f32,
            success,
            prediction_error: pred_error,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pred_error),
        }
    }

    pub fn encode_perception(&mut self) -> ContinuousHV {
        self.encoder.encode(self.simulator.state())
    }

    pub fn telemetry(&self) -> EmbodimentTelemetry {
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: self.last_control_effort,
            prediction_error: self.last_prediction_error,
            safety_level: self.current_safety,
            platform: "gravcraft".to_string(),
            num_actuators: 12,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::HDC_DIMENSION;

    #[test]
    fn test_embodiment_step_valid() {
        let genesis = GenesisSeed::from_phrase("gravcraft embodiment");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let thought = ContinuousHV::random(HDC_DIMENSION, 0xCAFE);

        let result = emb.step(&thought, 0.01, 0.7);
        assert!(result.success);
        assert_eq!(result.num_actuators, 12);
        assert!(result.control_effort.is_finite());
    }

    /// The gap closed on 2026-07-29: before this, a `SafetyAgent` override and an
    /// ethics verdict had NO route to the metric amplifiers on this platform. These
    /// assert the route exists and is restrictive-only, matching the fleet's
    /// `max(phi_level, safety_override, moral_safety)` convention.
    #[test]
    fn test_safety_override_restricts_high_phi() {
        let genesis = GenesisSeed::from_phrase("override test");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let thought = ContinuousHV::random(HDC_DIMENSION, 0xF00D);

        // Phi alone would be Green (full authority, all 3 amplifiers).
        emb.step(&thought, 0.01, 0.9);
        assert_eq!(emb.current_safety, MotorSafetyLevel::Green);

        // An external override must be able to force a lower tier.
        emb.set_safety_override(MotorSafetyLevel::Orange);
        let r = emb.step(&thought, 0.01, 0.9);
        assert_eq!(
            emb.current_safety,
            MotorSafetyLevel::Orange,
            "a SafetyAgent override must reach this platform's actuators"
        );
        assert_eq!(
            r.control_effort, 0.0,
            "Orange means flat metric -- the override must actually zero authority"
        );

        // Clearing it restores phi-derived authority (override restricts, never grants).
        emb.clear_safety_override();
        emb.step(&thought, 0.01, 0.9);
        assert_eq!(emb.current_safety, MotorSafetyLevel::Green);
    }

    #[test]
    fn test_moral_gate_forces_red_on_ahimsa() {
        let genesis = GenesisSeed::from_phrase("moral gate test");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let thought = ContinuousHV::random(HDC_DIMENSION, 0xBEEE);

        emb.apply_moral_gate(MoralGateInput {
            verdict: MoralGateInput::VERDICT_SAFE,
            consent_violation: false,
            ahimsa_violated: true,
        });

        let r = emb.step(&thought, 0.01, 0.95);
        assert_eq!(
            emb.current_safety,
            MotorSafetyLevel::Red,
            "an ahimsa violation must override even a maximal phi"
        );
        assert_eq!(r.control_effort, 0.0, "Red means metric neutralization");
    }

    #[test]
    fn test_phi_zero_no_metric_authority() {
        let genesis = GenesisSeed::from_phrase("safety test");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let thought = ContinuousHV::random(HDC_DIMENSION, 0xBEEF);

        let result = emb.step(&thought, 0.01, 0.0);
        assert_eq!(result.safety_level, MotorSafetyLevel::Red);
        // With Red safety, there should be no control effort
        assert!(
            result.control_effort < 1e-6,
            "Red safety should zero control effort: {}",
            result.control_effort
        );
    }

    #[test]
    fn test_safety_level_transitions() {
        let genesis = GenesisSeed::from_phrase("transition test");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let thought = ContinuousHV::random(HDC_DIMENSION, 0xFACE);

        // Green
        let r = emb.step(&thought, 0.01, 0.8);
        assert_eq!(r.safety_level, MotorSafetyLevel::Green);

        // Yellow
        let r = emb.step(&thought, 0.01, 0.4);
        assert_eq!(r.safety_level, MotorSafetyLevel::Yellow);

        // Orange
        let r = emb.step(&thought, 0.01, 0.2);
        assert_eq!(r.safety_level, MotorSafetyLevel::Orange);

        // Red
        let r = emb.step(&thought, 0.01, 0.05);
        assert_eq!(r.safety_level, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_encode_perception_returns_valid() {
        let genesis = GenesisSeed::from_phrase("perception test");
        let mut emb = GravcraftEmbodiment::new(&genesis);
        let hv = emb.encode_perception();
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }
}
