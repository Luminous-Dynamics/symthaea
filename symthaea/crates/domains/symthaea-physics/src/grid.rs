// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Grid Scaling — Genesis Mission Challenge 1
//!
//! HDC + CfC + FEP architecture for electrical grid stability monitoring.
//! Predicts voltage, frequency, demand, reserve margin, and line flow
//! across timescales from 1 minute to 1 year using O(1) CfC closed-form evolution.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Horizons ────────────────────────────────────────────────────────────

/// Grid prediction horizons (seconds): 1min → 1hr → 1day → 1month → 1year.
pub const GRID_HORIZONS: &[f32] = &[
    60.0,         // 1 minute — frequency excursion
    3_600.0,      // 1 hour — demand ramp
    86_400.0,     // 1 day — diurnal cycle
    2_592_000.0,  // 1 month — seasonal load
    31_536_000.0, // 1 year — capacity planning
];

pub const GRID_HORIZON_LABELS: &[&str] = &[
    "1 min (freq excursion)",
    "1 hr (demand ramp)",
    "1 day (diurnal)",
    "1 month (seasonal)",
    "1 year (planning)",
];

// ── Encoder ─────────────────────────────────────────────────────────────

/// Grid observable reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridReading {
    /// Voltage (pu, nominal = 1.0).
    pub voltage: f64,
    /// Frequency (Hz, nominal = 60.0).
    pub frequency: f64,
    /// Demand (MW, normalized by capacity).
    pub demand: f64,
    /// Reserve margin (fraction, 0..1).
    pub reserve_margin: f64,
    /// Line flow (fraction of thermal limit, 0..1).
    pub line_flow: f64,
}

/// HDC encoder for grid state (5 observables → ContinuousHV).
pub struct GridHdcEncoder {
    bases: [ContinuousHV; 5],
}

impl GridHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 5] = [
            0x68D_0001, // voltage
            0x68D_0002, // frequency
            0x68D_0003, // demand
            0x68D_0004, // reserve margin
            0x68D_0005, // line flow
        ];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &GridReading) -> ContinuousHV {
        let weights = [
            reading.voltage.clamp(0.0, 2.0) as f32 / 2.0,
            (reading.frequency / 60.0).clamp(0.0, 2.0) as f32 / 2.0,
            reading.demand.clamp(0.0, 1.0) as f32,
            reading.reserve_margin.clamp(0.0, 1.0) as f32,
            reading.line_flow.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for GridHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Predictor ───────────────────────────────────────────────────────────

/// Multi-scale predictor for electrical grid dynamics.
pub struct GridPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl GridPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 60.0, // 1 minute base
            backbone_tau: 3600.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0x68D_10A0),
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

impl Default for GridPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for GridPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }

    fn domain(&self) -> &'static str {
        "grid"
    }

    fn tau_base(&self) -> f32 {
        60.0
    }

    fn default_horizons(&self) -> &'static [f32] {
        GRID_HORIZONS
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        GRID_HORIZON_LABELS
    }
}

// ── FEP Agent ───────────────────────────────────────────────────────────

/// Actions for grid stability control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GridFepAction {
    MaintainGrid,
    AdjustGeneration,
    ShiftLoad,
    ActivateReserve,
    EmergencyCurtailment,
}

impl GridFepAction {
    pub const ALL: [GridFepAction; 5] = [
        GridFepAction::MaintainGrid,
        GridFepAction::AdjustGeneration,
        GridFepAction::ShiftLoad,
        GridFepAction::ActivateReserve,
        GridFepAction::EmergencyCurtailment,
    ];
}

const FE_CURTAILMENT: f64 = 0.7;
const FE_RESERVE: f64 = 0.5;
const FE_SHIFT: f64 = 0.3;
const FE_ADJUST: f64 = 0.1;

/// FEP agent for grid stability.
pub struct GridFepAgent {
    reference_state: ContinuousHV,
}

impl GridFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0x68D_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> GridFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > FE_CURTAILMENT {
            GridFepAction::EmergencyCurtailment
        } else if fe > FE_RESERVE {
            GridFepAction::ActivateReserve
        } else if fe > FE_SHIFT {
            GridFepAction::ShiftLoad
        } else if fe > FE_ADJUST {
            GridFepAction::AdjustGeneration
        } else {
            GridFepAction::MaintainGrid
        }
    }
}

impl Default for GridFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ── Output ──────────────────────────────────────────────────────────────

/// NRC-style grid safety level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GridSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

/// Output from a grid prediction cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridOutput {
    pub free_energy: f64,
    pub recommended_action: GridFepAction,
    pub safety_level: GridSafetyLevel,
    pub prediction_similarities: Vec<(f32, f32)>,
}

/// Grid digital twin.
pub struct GridTwin {
    encoder: GridHdcEncoder,
    predictor: GridPredictor,
    agent: GridFepAgent,
    cycle_count: u64,
}

impl GridTwin {
    pub fn new() -> Self {
        Self {
            encoder: GridHdcEncoder::new(),
            predictor: GridPredictor::new(),
            agent: GridFepAgent::new(),
            cycle_count: 0,
        }
    }

    pub fn set_reference(&mut self, reading: &GridReading) {
        let hv = self.encoder.encode(reading);
        self.agent.set_reference(hv);
    }

    pub fn step(&mut self, reading: &GridReading, dt_seconds: f32) -> GridOutput {
        assert!(dt_seconds.is_finite() && dt_seconds > 0.0);
        self.cycle_count += 1;

        let hv = self.encoder.encode(reading);
        self.predictor.observe(&hv, dt_seconds);

        let sims: Vec<(f32, f32)> = GRID_HORIZONS
            .iter()
            .map(|&h| {
                let pred = self.predictor.predict_at_horizon(&hv, h);
                (h, pred.similarity(&hv))
            })
            .collect();

        let fe = self.agent.compute_free_energy(&hv);
        let action = self.agent.select_action(&hv);
        let level = if fe > FE_CURTAILMENT {
            GridSafetyLevel::Red
        } else if fe > FE_RESERVE {
            GridSafetyLevel::Orange
        } else if fe > FE_ADJUST {
            GridSafetyLevel::Yellow
        } else {
            GridSafetyLevel::Green
        };

        GridOutput {
            free_energy: fe,
            recommended_action: action,
            safety_level: level,
            prediction_similarities: sims,
        }
    }

    pub fn predictor(&self) -> &GridPredictor {
        &self.predictor
    }

    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for GridTwin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy_reading() -> GridReading {
        GridReading {
            voltage: 1.0,
            frequency: 60.0,
            demand: 0.6,
            reserve_margin: 0.2,
            line_flow: 0.5,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..GRID_HORIZONS.len() {
            assert!(GRID_HORIZONS[i] > GRID_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(GRID_HORIZONS.len(), GRID_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        let enc = GridHdcEncoder::new();
        let hv = enc.encode(&healthy_reading());
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_dimension() {
        let pred = GridPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let out = pred.predict_at_horizon(&input, 60.0);
        assert_eq!(out.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_o1_property() {
        let pred = GridPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let t1 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 60.0);
        }
        let short = t1.elapsed();

        let t2 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 31_536_000.0);
        }
        let long = t2.elapsed();

        let ratio = long.as_nanos() as f64 / short.as_nanos().max(1) as f64;
        assert!(ratio < 5.0 && ratio > 0.2, "O(1) violated: ratio={}", ratio);
    }

    #[test]
    fn test_fep_self_reference() {
        let mut agent = GridFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        let fe = agent.compute_free_energy(&hv);
        assert!(fe < 0.01, "Self-reference FE should be ~0, got {}", fe);
    }

    #[test]
    fn test_fep_action_escalation() {
        assert!(GridFepAction::MaintainGrid < GridFepAction::EmergencyCurtailment);
    }

    #[test]
    fn test_twin_healthy() {
        let reading = healthy_reading();
        let mut twin = GridTwin::new();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 60.0);
        assert_eq!(output.safety_level, GridSafetyLevel::Green);
        assert_eq!(output.recommended_action, GridFepAction::MaintainGrid);
    }

    #[test]
    fn test_twin_cycle_count() {
        let mut twin = GridTwin::new();
        twin.step(&healthy_reading(), 60.0);
        twin.step(&healthy_reading(), 60.0);
        assert_eq!(twin.cycle_count(), 2);
    }

    #[test]
    fn test_safety_level_ordering() {
        assert!(GridSafetyLevel::Green < GridSafetyLevel::Yellow);
        assert!(GridSafetyLevel::Yellow < GridSafetyLevel::Orange);
        assert!(GridSafetyLevel::Orange < GridSafetyLevel::Red);
    }
}
