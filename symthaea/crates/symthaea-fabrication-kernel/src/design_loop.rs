// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Design-Production Loop — Genesis Mission Challenge 23
//!
//! HDC + CfC + FEP architecture for closed-loop design-to-production monitoring.
//! Predicts design intent, manufactured state, tolerance error, and quality
//! confidence across timescales from 0.1s (measure) to 1 day (production)
//! using O(1) CfC closed-form evolution.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Horizons ────────────────────────────────────────────────────────────

/// Design-production loop prediction horizons (seconds): 0.1s → 1s → 10s → 100s → 1 day.
pub const DESIGN_LOOP_HORIZONS: &[f32] = &[
    0.1,      // 0.1s — measure
    1.0,      // 1s — feedback
    10.0,     // 10s — adapt
    100.0,    // 100s — convergence
    86_400.0, // 1 day — production
];

pub const DESIGN_LOOP_HORIZON_LABELS: &[&str] = &[
    "0.1s (measure)",
    "1s (feedback)",
    "10s (adapt)",
    "100s (convergence)",
    "1 day (production)",
];

// ── Encoder ─────────────────────────────────────────────────────────────

/// Design-production loop observable reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignLoopReading {
    /// Design intent fidelity (0..1, 1 = perfectly captured).
    pub design_intent: f64,
    /// Manufactured state accuracy (0..1, 1 = matches design exactly).
    pub manufactured_state: f64,
    /// Tolerance error (0..1, 0 = within spec, 1 = max deviation).
    pub tolerance_error: f64,
    /// Quality confidence (0..1, 1 = full confidence in output).
    pub quality_confidence: f64,
}

/// HDC encoder for design-production loop state (4 observables -> ContinuousHV).
pub struct DesignLoopHdcEncoder {
    bases: [ContinuousHV; 4],
}

impl DesignLoopHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 4] = [
            0xD91_0001, // design_intent
            0xD91_0002, // manufactured_state
            0xD91_0003, // tolerance_error
            0xD91_0004, // quality_confidence
        ];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &DesignLoopReading) -> ContinuousHV {
        let weights = [
            reading.design_intent.clamp(0.0, 1.0) as f32,
            reading.manufactured_state.clamp(0.0, 1.0) as f32,
            reading.tolerance_error.clamp(0.0, 1.0) as f32,
            reading.quality_confidence.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for DesignLoopHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Predictor ───────────────────────────────────────────────────────────

/// Multi-scale predictor for design-production loop dynamics.
pub struct DesignLoopPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl DesignLoopPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 1.0,
            backbone_tau: 100.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xD91_10A0),
        }
    }

    pub fn predict_at_horizon(&self, current: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        assert!(horizon_seconds.is_finite() && horizon_seconds > 0.0);
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(horizon_seconds, current);
        neuron_copy.state().clone()
    }

    pub fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.neuron.evolve_closed_form(dt_seconds, state);
    }
}

impl Default for DesignLoopPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for DesignLoopPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }

    fn domain(&self) -> &'static str {
        "design_production"
    }

    fn tau_base(&self) -> f32 {
        1.0
    }

    fn default_horizons(&self) -> &'static [f32] {
        DESIGN_LOOP_HORIZONS
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        DESIGN_LOOP_HORIZON_LABELS
    }
}

// ── FEP Agent ───────────────────────────────────────────────────────────

/// Actions for design-production loop control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DesignLoopFepAction {
    ContinueProduction,
    AdjustDesign,
    RecalibrateProcess,
    HaltLine,
    EmergencyRecall,
}

impl DesignLoopFepAction {
    pub const ALL: [DesignLoopFepAction; 5] = [
        DesignLoopFepAction::ContinueProduction,
        DesignLoopFepAction::AdjustDesign,
        DesignLoopFepAction::RecalibrateProcess,
        DesignLoopFepAction::HaltLine,
        DesignLoopFepAction::EmergencyRecall,
    ];
}

const FE_RECALL: f64 = 0.7;
const FE_HALT: f64 = 0.5;
const FE_RECALIBRATE: f64 = 0.3;
const FE_ADJUST: f64 = 0.1;

/// FEP agent for design-production loop stability.
pub struct DesignLoopFepAgent {
    reference_state: ContinuousHV,
}

impl DesignLoopFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0xD91_BEEF),
        }
    }

    pub fn set_reference(&mut self, reference: ContinuousHV) {
        self.reference_state = reference;
    }

    pub fn compute_free_energy(&self, observed: &ContinuousHV) -> f64 {
        let sim = observed.similarity(&self.reference_state) as f64;
        if !sim.is_finite() {
            return 1.0;
        }
        (1.0 - sim).max(0.0)
    }

    pub fn select_action(&self, observed: &ContinuousHV) -> DesignLoopFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > FE_RECALL {
            DesignLoopFepAction::EmergencyRecall
        } else if fe > FE_HALT {
            DesignLoopFepAction::HaltLine
        } else if fe > FE_RECALIBRATE {
            DesignLoopFepAction::RecalibrateProcess
        } else if fe > FE_ADJUST {
            DesignLoopFepAction::AdjustDesign
        } else {
            DesignLoopFepAction::ContinueProduction
        }
    }
}

impl Default for DesignLoopFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ── Output ──────────────────────────────────────────────────────────────

/// Safety level for design-production loop operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DesignLoopSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

/// Output from a design-production loop prediction cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignLoopOutput {
    pub free_energy: f64,
    pub recommended_action: DesignLoopFepAction,
    pub safety_level: DesignLoopSafetyLevel,
    pub prediction_similarities: Vec<(f32, f32)>,
}

impl DesignLoopOutput {
    /// Bridge tuple for consciousness integration.
    ///
    /// Returns `(consciousness_level, prediction_error, coherence)` where:
    /// - `consciousness_level` = 1.0 - free_energy (higher FE => lower awareness)
    /// - `prediction_error` = free_energy (surprise signal)
    /// - `coherence` = average similarity across prediction horizons
    pub fn to_safety_tuple(&self) -> (f32, f32, f32) {
        let consciousness_level = (1.0 - self.free_energy) as f32;
        let prediction_error = self.free_energy as f32;
        let coherence = if self.prediction_similarities.is_empty() {
            0.0
        } else {
            let sum: f32 = self.prediction_similarities.iter().map(|(_, s)| *s).sum();
            sum / self.prediction_similarities.len() as f32
        };
        (consciousness_level, prediction_error, coherence)
    }
}

/// Design-production loop digital twin.
pub struct DesignLoopTwin {
    encoder: DesignLoopHdcEncoder,
    predictor: DesignLoopPredictor,
    agent: DesignLoopFepAgent,
    cycle_count: u64,
}

impl DesignLoopTwin {
    pub fn new() -> Self {
        Self {
            encoder: DesignLoopHdcEncoder::new(),
            predictor: DesignLoopPredictor::new(),
            agent: DesignLoopFepAgent::new(),
            cycle_count: 0,
        }
    }

    pub fn set_reference(&mut self, reading: &DesignLoopReading) {
        let hv = self.encoder.encode(reading);
        self.agent.set_reference(hv);
    }

    pub fn step(&mut self, reading: &DesignLoopReading, dt_seconds: f32) -> DesignLoopOutput {
        assert!(dt_seconds.is_finite() && dt_seconds > 0.0);
        self.cycle_count += 1;

        let hv = self.encoder.encode(reading);
        self.predictor.observe(&hv, dt_seconds);

        let sims: Vec<(f32, f32)> = DESIGN_LOOP_HORIZONS
            .iter()
            .map(|&h| {
                let pred = self.predictor.predict_at_horizon(&hv, h);
                (h, pred.similarity(&hv))
            })
            .collect();

        let fe = self.agent.compute_free_energy(&hv);
        let action = self.agent.select_action(&hv);
        let level = if fe > FE_RECALL {
            DesignLoopSafetyLevel::Red
        } else if fe > FE_HALT {
            DesignLoopSafetyLevel::Orange
        } else if fe > FE_ADJUST {
            DesignLoopSafetyLevel::Yellow
        } else {
            DesignLoopSafetyLevel::Green
        };

        DesignLoopOutput {
            free_energy: fe,
            recommended_action: action,
            safety_level: level,
            prediction_similarities: sims,
        }
    }

    pub fn predictor(&self) -> &DesignLoopPredictor {
        &self.predictor
    }

    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for DesignLoopTwin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy_reading() -> DesignLoopReading {
        DesignLoopReading {
            design_intent: 0.95,
            manufactured_state: 0.92,
            tolerance_error: 0.05,
            quality_confidence: 0.9,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..DESIGN_LOOP_HORIZONS.len() {
            assert!(DESIGN_LOOP_HORIZONS[i] > DESIGN_LOOP_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(DESIGN_LOOP_HORIZONS.len(), DESIGN_LOOP_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        let enc = DesignLoopHdcEncoder::new();
        let hv = enc.encode(&healthy_reading());
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_dimension() {
        let pred = DesignLoopPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let out = pred.predict_at_horizon(&input, 1.0);
        assert_eq!(out.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_o1_property() {
        let pred = DesignLoopPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let t1 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 0.1);
        }
        let short = t1.elapsed();

        let t2 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 86_400.0);
        }
        let long = t2.elapsed();

        let ratio = long.as_nanos() as f64 / short.as_nanos().max(1) as f64;
        assert!(ratio < 5.0 && ratio > 0.2, "O(1) violated: ratio={}", ratio);
    }

    #[test]
    fn test_fep_self_reference() {
        let mut agent = DesignLoopFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        let fe = agent.compute_free_energy(&hv);
        assert!(fe < 0.01, "Self-reference FE should be ~0, got {}", fe);
    }

    #[test]
    fn test_fep_action_escalation() {
        assert!(DesignLoopFepAction::ContinueProduction < DesignLoopFepAction::EmergencyRecall);
    }

    #[test]
    fn test_twin_healthy() {
        let reading = healthy_reading();
        let mut twin = DesignLoopTwin::new();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);
        assert_eq!(output.safety_level, DesignLoopSafetyLevel::Green);
        assert_eq!(
            output.recommended_action,
            DesignLoopFepAction::ContinueProduction
        );
    }

    #[test]
    fn test_twin_cycle_count() {
        let mut twin = DesignLoopTwin::new();
        twin.step(&healthy_reading(), 1.0);
        twin.step(&healthy_reading(), 1.0);
        assert_eq!(twin.cycle_count(), 2);
    }

    #[test]
    fn test_safety_level_ordering() {
        assert!(DesignLoopSafetyLevel::Green < DesignLoopSafetyLevel::Yellow);
        assert!(DesignLoopSafetyLevel::Yellow < DesignLoopSafetyLevel::Orange);
        assert!(DesignLoopSafetyLevel::Orange < DesignLoopSafetyLevel::Red);
    }

    #[test]
    fn test_safety_tuple_self_reference() {
        let reading = healthy_reading();
        let mut twin = DesignLoopTwin::new();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);
        let (consciousness, error, coherence) = output.to_safety_tuple();
        assert!(consciousness > 0.9, "consciousness={}", consciousness);
        assert!(error < 0.1, "error={}", error);
        assert!(coherence.is_finite(), "coherence must be finite");
    }

    #[test]
    fn test_safety_tuple_range() {
        let reading = healthy_reading();
        let mut twin = DesignLoopTwin::new();
        let output = twin.step(&reading, 1.0);
        let (consciousness, error, coherence) = output.to_safety_tuple();
        assert!(
            (0.0..=1.0).contains(&consciousness),
            "consciousness={}",
            consciousness
        );
        assert!((0.0..=1.0).contains(&error), "error={}", error);
        assert!(coherence.is_finite(), "coherence must be finite");
    }
}
