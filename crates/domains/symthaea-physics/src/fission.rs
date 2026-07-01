// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fission Reactors — Genesis Mission Challenge 2
//!
//! HDC + CfC + FEP architecture for fission reactor monitoring.
//! Predicts power output, coolant temperature, neutron flux, pressure,
//! and control rod position across timescales from 0.1s to 1 day
//! using O(1) CfC closed-form evolution.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Horizons ────────────────────────────────────────────────────────────

/// Fission prediction horizons (seconds): 0.1s → 1s → 10s → 100s → 1day.
pub const FISSION_HORIZONS: &[f32] = &[
    0.1,      // 0.1s — prompt neutron response
    1.0,      // 1s — delayed neutron dynamics
    10.0,     // 10s — xenon poisoning onset
    100.0,    // 100s — thermal hydraulic transient
    86_400.0, // 1 day — fuel burnup
];

pub const FISSION_HORIZON_LABELS: &[&str] = &[
    "0.1s (prompt)",
    "1s (delayed neutron)",
    "10s (xenon)",
    "100s (thermal)",
    "1 day (burnup)",
];

// ── Encoder ─────────────────────────────────────────────────────────────

/// Fission reactor observable reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FissionReading {
    /// Power output (MW, normalized 0-1 by rated capacity).
    pub power_output: f64,
    /// Coolant temperature (°C, /400 for normalization).
    pub coolant_temp: f64,
    /// Neutron flux (normalized 0-1).
    pub neutron_flux: f64,
    /// Pressure (MPa, /15 for normalization).
    pub pressure: f64,
    /// Control rod position (fraction inserted, 0..1).
    pub control_rod_pos: f64,
}

/// HDC encoder for fission reactor state (5 observables → ContinuousHV).
pub struct FissionHdcEncoder {
    bases: [ContinuousHV; 5],
}

impl FissionHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 5] = [
            0xF15_0001, // power_output
            0xF15_0002, // coolant_temp
            0xF15_0003, // neutron_flux
            0xF15_0004, // pressure
            0xF15_0005, // control_rod_pos
        ];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &FissionReading) -> ContinuousHV {
        let weights = [
            reading.power_output.clamp(0.0, 1.0) as f32,
            (reading.coolant_temp / 400.0).clamp(0.0, 1.0) as f32,
            reading.neutron_flux.clamp(0.0, 1.0) as f32,
            (reading.pressure / 15.0).clamp(0.0, 1.0) as f32,
            reading.control_rod_pos.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for FissionHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Predictor ───────────────────────────────────────────────────────────

/// Multi-scale predictor for fission reactor dynamics.
pub struct FissionPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl FissionPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 10.0, // 10 second base
            backbone_tau: 100.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xF15_10A0),
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

impl Default for FissionPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for FissionPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }

    fn domain(&self) -> &'static str {
        "fission"
    }

    fn tau_base(&self) -> f32 {
        10.0
    }

    fn default_horizons(&self) -> &'static [f32] {
        FISSION_HORIZONS
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        FISSION_HORIZON_LABELS
    }
}

// ── FEP Agent ───────────────────────────────────────────────────────────

/// Actions for fission reactor control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FissionFepAction {
    MaintainReactor,
    AdjustRods,
    IncreaseCoolant,
    ReducePower,
    SCRAM,
}

impl FissionFepAction {
    pub const ALL: [FissionFepAction; 5] = [
        FissionFepAction::MaintainReactor,
        FissionFepAction::AdjustRods,
        FissionFepAction::IncreaseCoolant,
        FissionFepAction::ReducePower,
        FissionFepAction::SCRAM,
    ];
}

const FE_SCRAM: f64 = 0.7;
const FE_REDUCE_POWER: f64 = 0.5;
const FE_INCREASE_COOLANT: f64 = 0.3;
const FE_ADJUST_RODS: f64 = 0.1;

/// FEP agent for fission reactor safety.
pub struct FissionFepAgent {
    reference_state: ContinuousHV,
}

impl FissionFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0xF15_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> FissionFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > FE_SCRAM {
            FissionFepAction::SCRAM
        } else if fe > FE_REDUCE_POWER {
            FissionFepAction::ReducePower
        } else if fe > FE_INCREASE_COOLANT {
            FissionFepAction::IncreaseCoolant
        } else if fe > FE_ADJUST_RODS {
            FissionFepAction::AdjustRods
        } else {
            FissionFepAction::MaintainReactor
        }
    }
}

impl Default for FissionFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ── Output ──────────────────────────────────────────────────────────────

/// NRC-style fission reactor safety level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FissionSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

/// Output from a fission reactor prediction cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FissionOutput {
    pub free_energy: f64,
    pub recommended_action: FissionFepAction,
    pub safety_level: FissionSafetyLevel,
    pub prediction_similarities: Vec<(f32, f32)>,
}

/// Fission reactor digital twin.
pub struct FissionTwin {
    encoder: FissionHdcEncoder,
    predictor: FissionPredictor,
    agent: FissionFepAgent,
    cycle_count: u64,
}

impl FissionTwin {
    pub fn new() -> Self {
        Self {
            encoder: FissionHdcEncoder::new(),
            predictor: FissionPredictor::new(),
            agent: FissionFepAgent::new(),
            cycle_count: 0,
        }
    }

    pub fn set_reference(&mut self, reading: &FissionReading) {
        let hv = self.encoder.encode(reading);
        self.agent.set_reference(hv);
    }

    pub fn step(&mut self, reading: &FissionReading, dt_seconds: f32) -> FissionOutput {
        assert!(dt_seconds.is_finite() && dt_seconds > 0.0);
        self.cycle_count += 1;

        let hv = self.encoder.encode(reading);
        self.predictor.observe(&hv, dt_seconds);

        let sims: Vec<(f32, f32)> = FISSION_HORIZONS
            .iter()
            .map(|&h| {
                let pred = self.predictor.predict_at_horizon(&hv, h);
                (h, pred.similarity(&hv))
            })
            .collect();

        let fe = self.agent.compute_free_energy(&hv);
        let action = self.agent.select_action(&hv);
        let level = if fe > FE_SCRAM {
            FissionSafetyLevel::Red
        } else if fe > FE_REDUCE_POWER {
            FissionSafetyLevel::Orange
        } else if fe > FE_ADJUST_RODS {
            FissionSafetyLevel::Yellow
        } else {
            FissionSafetyLevel::Green
        };

        FissionOutput {
            free_energy: fe,
            recommended_action: action,
            safety_level: level,
            prediction_similarities: sims,
        }
    }

    pub fn predictor(&self) -> &FissionPredictor {
        &self.predictor
    }

    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for FissionTwin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy_reading() -> FissionReading {
        FissionReading {
            power_output: 0.8,
            coolant_temp: 300.0,
            neutron_flux: 0.5,
            pressure: 10.0,
            control_rod_pos: 0.5,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..FISSION_HORIZONS.len() {
            assert!(FISSION_HORIZONS[i] > FISSION_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(FISSION_HORIZONS.len(), FISSION_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        let enc = FissionHdcEncoder::new();
        let hv = enc.encode(&healthy_reading());
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_dimension() {
        let pred = FissionPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let out = pred.predict_at_horizon(&input, 10.0);
        assert_eq!(out.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_o1_property() {
        let pred = FissionPredictor::new();
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
        let mut agent = FissionFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        let fe = agent.compute_free_energy(&hv);
        assert!(fe < 0.01, "Self-reference FE should be ~0, got {}", fe);
    }

    #[test]
    fn test_fep_action_escalation() {
        assert!(FissionFepAction::MaintainReactor < FissionFepAction::SCRAM);
    }

    #[test]
    fn test_twin_healthy() {
        let reading = healthy_reading();
        let mut twin = FissionTwin::new();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 1.0);
        assert_eq!(output.safety_level, FissionSafetyLevel::Green);
        assert_eq!(output.recommended_action, FissionFepAction::MaintainReactor);
    }

    #[test]
    fn test_twin_cycle_count() {
        let mut twin = FissionTwin::new();
        twin.step(&healthy_reading(), 1.0);
        twin.step(&healthy_reading(), 1.0);
        assert_eq!(twin.cycle_count(), 2);
    }

    #[test]
    fn test_safety_level_ordering() {
        assert!(FissionSafetyLevel::Green < FissionSafetyLevel::Yellow);
        assert!(FissionSafetyLevel::Yellow < FissionSafetyLevel::Orange);
        assert!(FissionSafetyLevel::Orange < FissionSafetyLevel::Red);
    }
}
