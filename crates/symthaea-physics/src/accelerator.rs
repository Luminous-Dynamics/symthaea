// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Particle Accelerators — Genesis Mission Challenge 13
//!
//! HDC + CfC + FEP architecture for particle accelerator beam monitoring.
//! Predicts beam energy, luminosity, beam loss, tune drift, and emittance
//! across timescales from 1ms to 10hr using O(1) CfC closed-form evolution.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Horizons ────────────────────────────────────────────────────────────

/// Accelerator prediction horizons (seconds): 1ms → 10ms → 1s → 100s → 10hr.
pub const ACCELERATOR_HORIZONS: &[f32] = &[
    0.001,    // 1 ms — orbit
    0.01,     // 10 ms — injection
    1.0,      // 1 s — ramp
    100.0,    // 100 s — squeeze
    36_000.0, // 10 hr — fill
];

pub const ACCELERATOR_HORIZON_LABELS: &[&str] = &[
    "1ms (orbit)",
    "10ms (injection)",
    "1s (ramp)",
    "100s (squeeze)",
    "10hr (fill)",
];

// ── Encoder ─────────────────────────────────────────────────────────────

/// Accelerator observable reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorReading {
    /// Beam energy (GeV, normalized by 7000 GeV).
    pub beam_energy: f64,
    /// Luminosity (×10³⁴ cm⁻²s⁻¹, normalized by 2.0).
    pub luminosity: f64,
    /// Beam loss (fraction, 0..1).
    pub beam_loss: f64,
    /// Tune drift (fraction, 0..1).
    pub tune_drift: f64,
    /// Emittance (fraction, 0..1).
    pub emittance: f64,
}

/// HDC encoder for accelerator state (5 observables → ContinuousHV).
pub struct AcceleratorHdcEncoder {
    bases: [ContinuousHV; 5],
}

impl AcceleratorHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 5] = [
            0xACC_0001, // beam_energy
            0xACC_0002, // luminosity
            0xACC_0003, // beam_loss
            0xACC_0004, // tune_drift
            0xACC_0005, // emittance
        ];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &AcceleratorReading) -> ContinuousHV {
        let weights = [
            (reading.beam_energy / 7000.0).clamp(0.0, 1.0) as f32,
            (reading.luminosity / 2.0).clamp(0.0, 1.0) as f32,
            reading.beam_loss.clamp(0.0, 1.0) as f32,
            reading.tune_drift.clamp(0.0, 1.0) as f32,
            reading.emittance.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for AcceleratorHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Predictor ───────────────────────────────────────────────────────────

/// Multi-scale predictor for accelerator beam dynamics.
pub struct AcceleratorPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl AcceleratorPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 1.0, // 1 second base
            backbone_tau: 100.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xACC_10A0),
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

impl Default for AcceleratorPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for AcceleratorPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }

    fn domain(&self) -> &'static str {
        "accelerator"
    }

    fn tau_base(&self) -> f32 {
        1.0
    }

    fn default_horizons(&self) -> &'static [f32] {
        ACCELERATOR_HORIZONS
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        ACCELERATOR_HORIZON_LABELS
    }
}

// ── FEP Agent ───────────────────────────────────────────────────────────

/// Actions for accelerator beam control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AcceleratorFepAction {
    MaintainBeam,
    AdjustTune,
    CorrectOrbit,
    ReduceIntensity,
    BeamDump,
}

impl AcceleratorFepAction {
    pub const ALL: [AcceleratorFepAction; 5] = [
        AcceleratorFepAction::MaintainBeam,
        AcceleratorFepAction::AdjustTune,
        AcceleratorFepAction::CorrectOrbit,
        AcceleratorFepAction::ReduceIntensity,
        AcceleratorFepAction::BeamDump,
    ];
}

const FE_BEAM_DUMP: f64 = 0.7;
const FE_REDUCE_INTENSITY: f64 = 0.5;
const FE_CORRECT_ORBIT: f64 = 0.3;
const FE_ADJUST_TUNE: f64 = 0.1;

/// FEP agent for accelerator beam stability.
pub struct AcceleratorFepAgent {
    reference_state: ContinuousHV,
}

impl AcceleratorFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0xACC_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> AcceleratorFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > FE_BEAM_DUMP {
            AcceleratorFepAction::BeamDump
        } else if fe > FE_REDUCE_INTENSITY {
            AcceleratorFepAction::ReduceIntensity
        } else if fe > FE_CORRECT_ORBIT {
            AcceleratorFepAction::CorrectOrbit
        } else if fe > FE_ADJUST_TUNE {
            AcceleratorFepAction::AdjustTune
        } else {
            AcceleratorFepAction::MaintainBeam
        }
    }
}

impl Default for AcceleratorFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ── Output ──────────────────────────────────────────────────────────────

/// Accelerator safety level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AcceleratorSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

/// Output from an accelerator prediction cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorOutput {
    pub free_energy: f64,
    pub recommended_action: AcceleratorFepAction,
    pub safety_level: AcceleratorSafetyLevel,
    pub prediction_similarities: Vec<(f32, f32)>,
}

/// Accelerator digital twin.
pub struct AcceleratorTwin {
    encoder: AcceleratorHdcEncoder,
    predictor: AcceleratorPredictor,
    agent: AcceleratorFepAgent,
    cycle_count: u64,
}

impl AcceleratorTwin {
    pub fn new() -> Self {
        Self {
            encoder: AcceleratorHdcEncoder::new(),
            predictor: AcceleratorPredictor::new(),
            agent: AcceleratorFepAgent::new(),
            cycle_count: 0,
        }
    }

    pub fn set_reference(&mut self, reading: &AcceleratorReading) {
        let hv = self.encoder.encode(reading);
        self.agent.set_reference(hv);
    }

    pub fn step(&mut self, reading: &AcceleratorReading, dt_seconds: f32) -> AcceleratorOutput {
        assert!(dt_seconds.is_finite() && dt_seconds > 0.0);
        self.cycle_count += 1;

        let hv = self.encoder.encode(reading);
        self.predictor.observe(&hv, dt_seconds);

        let sims: Vec<(f32, f32)> = ACCELERATOR_HORIZONS
            .iter()
            .map(|&h| {
                let pred = self.predictor.predict_at_horizon(&hv, h);
                (h, pred.similarity(&hv))
            })
            .collect();

        let fe = self.agent.compute_free_energy(&hv);
        let action = self.agent.select_action(&hv);
        let level = if fe > FE_BEAM_DUMP {
            AcceleratorSafetyLevel::Red
        } else if fe > FE_REDUCE_INTENSITY {
            AcceleratorSafetyLevel::Orange
        } else if fe > FE_ADJUST_TUNE {
            AcceleratorSafetyLevel::Yellow
        } else {
            AcceleratorSafetyLevel::Green
        };

        AcceleratorOutput {
            free_energy: fe,
            recommended_action: action,
            safety_level: level,
            prediction_similarities: sims,
        }
    }

    pub fn predictor(&self) -> &AcceleratorPredictor {
        &self.predictor
    }

    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for AcceleratorTwin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy_reading() -> AcceleratorReading {
        AcceleratorReading {
            beam_energy: 6500.0,
            luminosity: 1.0,
            beam_loss: 0.01,
            tune_drift: 0.02,
            emittance: 0.1,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..ACCELERATOR_HORIZONS.len() {
            assert!(ACCELERATOR_HORIZONS[i] > ACCELERATOR_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(ACCELERATOR_HORIZONS.len(), ACCELERATOR_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        let enc = AcceleratorHdcEncoder::new();
        let hv = enc.encode(&healthy_reading());
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_dimension() {
        let pred = AcceleratorPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let out = pred.predict_at_horizon(&input, 1.0);
        assert_eq!(out.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_o1_property() {
        let pred = AcceleratorPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let t1 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 0.001);
        }
        let short = t1.elapsed();

        let t2 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 36_000.0);
        }
        let long = t2.elapsed();

        let ratio = long.as_nanos() as f64 / short.as_nanos().max(1) as f64;
        assert!(ratio < 5.0 && ratio > 0.2, "O(1) violated: ratio={}", ratio);
    }

    #[test]
    fn test_fep_self_reference() {
        let mut agent = AcceleratorFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        let fe = agent.compute_free_energy(&hv);
        assert!(fe < 0.01, "Self-reference FE should be ~0, got {}", fe);
    }

    #[test]
    fn test_fep_action_escalation() {
        assert!(AcceleratorFepAction::MaintainBeam < AcceleratorFepAction::BeamDump);
    }

    #[test]
    fn test_twin_healthy() {
        let reading = healthy_reading();
        let mut twin = AcceleratorTwin::new();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);
        assert_eq!(output.safety_level, AcceleratorSafetyLevel::Green);
        assert_eq!(
            output.recommended_action,
            AcceleratorFepAction::MaintainBeam
        );
    }

    #[test]
    fn test_twin_cycle_count() {
        let mut twin = AcceleratorTwin::new();
        twin.step(&healthy_reading(), 1.0);
        twin.step(&healthy_reading(), 1.0);
        assert_eq!(twin.cycle_count(), 2);
    }

    #[test]
    fn test_safety_level_ordering() {
        assert!(AcceleratorSafetyLevel::Green < AcceleratorSafetyLevel::Yellow);
        assert!(AcceleratorSafetyLevel::Yellow < AcceleratorSafetyLevel::Orange);
        assert!(AcceleratorSafetyLevel::Orange < AcceleratorSafetyLevel::Red);
    }
}
