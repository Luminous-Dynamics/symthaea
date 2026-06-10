// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced Manufacturing — Genesis Mission Challenge 15
//!
//! HDC + CfC + FEP architecture for manufacturing process monitoring.
//! Predicts tolerance, surface quality, throughput, and energy cost
//! across timescales from 0.1s (tool pass) to 1 day (shift) using
//! O(1) CfC closed-form evolution.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Horizons ────────────────────────────────────────────────────────────

/// Manufacturing prediction horizons (seconds): 0.1s → 1s → 10s → 100s → 1 day.
pub const MANUFACTURING_HORIZONS: &[f32] = &[
    0.1,      // 0.1s — tool pass
    1.0,      // 1s — operation
    10.0,     // 10s — cycle
    100.0,    // 100s — batch
    86_400.0, // 1 day — shift
];

pub const MANUFACTURING_HORIZON_LABELS: &[&str] = &[
    "0.1s (tool pass)",
    "1s (operation)",
    "10s (cycle)",
    "100s (batch)",
    "1 day (shift)",
];

// ── Encoder ─────────────────────────────────────────────────────────────

/// Manufacturing observable reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManufacturingReading {
    /// Tolerance adherence (0..1, 1 = perfect).
    pub tolerance: f64,
    /// Surface quality (0..1, 1 = mirror finish).
    pub surface_quality: f64,
    /// Throughput (0..1, fraction of rated capacity).
    pub throughput: f64,
    /// Energy cost (0..1, fraction of budget).
    pub energy_cost: f64,
}

/// HDC encoder for manufacturing state (4 observables -> ContinuousHV).
pub struct ManufacturingHdcEncoder {
    bases: [ContinuousHV; 4],
}

impl ManufacturingHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 4] = [
            0xAF6_0001, // tolerance
            0xAF6_0002, // surface_quality
            0xAF6_0003, // throughput
            0xAF6_0004, // energy_cost
        ];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &ManufacturingReading) -> ContinuousHV {
        let weights = [
            reading.tolerance.clamp(0.0, 1.0) as f32,
            reading.surface_quality.clamp(0.0, 1.0) as f32,
            reading.throughput.clamp(0.0, 1.0) as f32,
            reading.energy_cost.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for ManufacturingHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Predictor ───────────────────────────────────────────────────────────

/// Multi-scale predictor for manufacturing process dynamics.
pub struct ManufacturingPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl ManufacturingPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 1.0,
            backbone_tau: 100.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xAF6_10A0),
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

impl Default for ManufacturingPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for ManufacturingPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }

    fn domain(&self) -> &'static str {
        "manufacturing"
    }

    fn tau_base(&self) -> f32 {
        1.0
    }

    fn default_horizons(&self) -> &'static [f32] {
        MANUFACTURING_HORIZONS
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        MANUFACTURING_HORIZON_LABELS
    }
}

// ── FEP Agent ───────────────────────────────────────────────────────────

/// Actions for manufacturing process control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ManufacturingFepAction {
    Maintain,
    AdjustTooling,
    RecalibrateProcess,
    ReduceSpeed,
    EmergencyHalt,
}

impl ManufacturingFepAction {
    pub const ALL: [ManufacturingFepAction; 5] = [
        ManufacturingFepAction::Maintain,
        ManufacturingFepAction::AdjustTooling,
        ManufacturingFepAction::RecalibrateProcess,
        ManufacturingFepAction::ReduceSpeed,
        ManufacturingFepAction::EmergencyHalt,
    ];
}

const FE_HALT: f64 = 0.7;
const FE_REDUCE: f64 = 0.5;
const FE_RECALIBRATE: f64 = 0.3;
const FE_ADJUST: f64 = 0.1;

/// FEP agent for manufacturing process stability.
pub struct ManufacturingFepAgent {
    reference_state: ContinuousHV,
}

impl ManufacturingFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0xAF6_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> ManufacturingFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > FE_HALT {
            ManufacturingFepAction::EmergencyHalt
        } else if fe > FE_REDUCE {
            ManufacturingFepAction::ReduceSpeed
        } else if fe > FE_RECALIBRATE {
            ManufacturingFepAction::RecalibrateProcess
        } else if fe > FE_ADJUST {
            ManufacturingFepAction::AdjustTooling
        } else {
            ManufacturingFepAction::Maintain
        }
    }
}

impl Default for ManufacturingFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ── Output ──────────────────────────────────────────────────────────────

/// Safety level for manufacturing operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ManufacturingSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

/// Output from a manufacturing prediction cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManufacturingOutput {
    pub free_energy: f64,
    pub recommended_action: ManufacturingFepAction,
    pub safety_level: ManufacturingSafetyLevel,
    pub prediction_similarities: Vec<(f32, f32)>,
}

impl ManufacturingOutput {
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

/// Manufacturing digital twin.
pub struct ManufacturingTwin {
    encoder: ManufacturingHdcEncoder,
    predictor: ManufacturingPredictor,
    agent: ManufacturingFepAgent,
    cycle_count: u64,
}

impl ManufacturingTwin {
    pub fn new() -> Self {
        Self {
            encoder: ManufacturingHdcEncoder::new(),
            predictor: ManufacturingPredictor::new(),
            agent: ManufacturingFepAgent::new(),
            cycle_count: 0,
        }
    }

    pub fn set_reference(&mut self, reading: &ManufacturingReading) {
        let hv = self.encoder.encode(reading);
        self.agent.set_reference(hv);
    }

    pub fn step(&mut self, reading: &ManufacturingReading, dt_seconds: f32) -> ManufacturingOutput {
        assert!(dt_seconds.is_finite() && dt_seconds > 0.0);
        self.cycle_count += 1;

        let hv = self.encoder.encode(reading);
        self.predictor.observe(&hv, dt_seconds);

        let sims: Vec<(f32, f32)> = MANUFACTURING_HORIZONS
            .iter()
            .map(|&h| {
                let pred = self.predictor.predict_at_horizon(&hv, h);
                (h, pred.similarity(&hv))
            })
            .collect();

        let fe = self.agent.compute_free_energy(&hv);
        let action = self.agent.select_action(&hv);
        let level = if fe > FE_HALT {
            ManufacturingSafetyLevel::Red
        } else if fe > FE_REDUCE {
            ManufacturingSafetyLevel::Orange
        } else if fe > FE_ADJUST {
            ManufacturingSafetyLevel::Yellow
        } else {
            ManufacturingSafetyLevel::Green
        };

        ManufacturingOutput {
            free_energy: fe,
            recommended_action: action,
            safety_level: level,
            prediction_similarities: sims,
        }
    }

    pub fn predictor(&self) -> &ManufacturingPredictor {
        &self.predictor
    }

    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for ManufacturingTwin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy_reading() -> ManufacturingReading {
        ManufacturingReading {
            tolerance: 0.95,
            surface_quality: 0.9,
            throughput: 0.8,
            energy_cost: 0.3,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..MANUFACTURING_HORIZONS.len() {
            assert!(MANUFACTURING_HORIZONS[i] > MANUFACTURING_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(
            MANUFACTURING_HORIZONS.len(),
            MANUFACTURING_HORIZON_LABELS.len()
        );
    }

    #[test]
    fn test_encoder_dimension() {
        let enc = ManufacturingHdcEncoder::new();
        let hv = enc.encode(&healthy_reading());
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_dimension() {
        let pred = ManufacturingPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let out = pred.predict_at_horizon(&input, 1.0);
        assert_eq!(out.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_o1_property() {
        let pred = ManufacturingPredictor::new();
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
        let mut agent = ManufacturingFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        let fe = agent.compute_free_energy(&hv);
        assert!(fe < 0.01, "Self-reference FE should be ~0, got {}", fe);
    }

    #[test]
    fn test_fep_action_escalation() {
        assert!(ManufacturingFepAction::Maintain < ManufacturingFepAction::EmergencyHalt);
    }

    #[test]
    fn test_twin_healthy() {
        let reading = healthy_reading();
        let mut twin = ManufacturingTwin::new();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);
        assert_eq!(output.safety_level, ManufacturingSafetyLevel::Green);
        assert_eq!(output.recommended_action, ManufacturingFepAction::Maintain);
    }

    #[test]
    fn test_twin_cycle_count() {
        let mut twin = ManufacturingTwin::new();
        twin.step(&healthy_reading(), 1.0);
        twin.step(&healthy_reading(), 1.0);
        assert_eq!(twin.cycle_count(), 2);
    }

    #[test]
    fn test_safety_level_ordering() {
        assert!(ManufacturingSafetyLevel::Green < ManufacturingSafetyLevel::Yellow);
        assert!(ManufacturingSafetyLevel::Yellow < ManufacturingSafetyLevel::Orange);
        assert!(ManufacturingSafetyLevel::Orange < ManufacturingSafetyLevel::Red);
    }

    #[test]
    fn test_safety_tuple_self_reference() {
        let reading = healthy_reading();
        let mut twin = ManufacturingTwin::new();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);
        let (consciousness, error, coherence) = output.to_safety_tuple();
        // Self-reference: low free energy => high consciousness, low error
        assert!(consciousness > 0.9, "consciousness={}", consciousness);
        assert!(error < 0.1, "error={}", error);
        assert!(coherence.is_finite(), "coherence must be finite");
    }

    #[test]
    fn test_safety_tuple_range() {
        let reading = healthy_reading();
        let mut twin = ManufacturingTwin::new();
        // Do NOT set reference — random reference will produce nonzero FE
        let output = twin.step(&reading, 1.0);
        let (consciousness, error, coherence) = output.to_safety_tuple();
        // All values should be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&consciousness),
            "consciousness={}",
            consciousness
        );
        assert!((0.0..=1.0).contains(&error), "error={}", error);
        assert!(coherence.is_finite(), "coherence must be finite");
    }
}
