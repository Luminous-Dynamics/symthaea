// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Building Systems — Genesis Mission Challenge 19
//!
//! HDC + CfC + FEP architecture for building systems monitoring.
//! Predicts thermal load, structural stress, occupancy, comfort, and
//! energy consumption across timescales from 1 hour (HVAC) to 30 years
//! (lifecycle) using O(1) CfC closed-form evolution.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Horizons ────────────────────────────────────────────────────────────

/// Building prediction horizons (seconds): 1hr → 1day → 1month → 1year → 30years.
pub const BUILDING_HORIZONS: &[f32] = &[
    3_600.0,       // 1 hour — HVAC
    86_400.0,      // 1 day — occupancy
    2_592_000.0,   // 1 month — season
    31_536_000.0,  // 1 year — maintenance
    946_080_000.0, // 30 years — lifecycle
];

pub const BUILDING_HORIZON_LABELS: &[&str] = &[
    "1 hr (HVAC)",
    "1 day (occupancy)",
    "1 month (season)",
    "1 year (maintenance)",
    "30 years (lifecycle)",
];

// ── Encoder ─────────────────────────────────────────────────────────────

/// Building observable reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingReading {
    /// Thermal load (0..1, fraction of HVAC capacity).
    pub thermal_load: f64,
    /// Structural stress (0..1, fraction of design limit).
    pub structural_stress: f64,
    /// Occupancy (0..1, fraction of rated capacity).
    pub occupancy: f64,
    /// Comfort index (0..1, 1 = ideal conditions).
    pub comfort: f64,
    /// Energy consumption (0..1, fraction of budget).
    pub energy_consumption: f64,
}

/// HDC encoder for building state (5 observables -> ContinuousHV).
pub struct BuildingHdcEncoder {
    bases: [ContinuousHV; 5],
}

impl BuildingHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 5] = [
            0xB1D_0001, // thermal_load
            0xB1D_0002, // structural_stress
            0xB1D_0003, // occupancy
            0xB1D_0004, // comfort
            0xB1D_0005, // energy_consumption
        ];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &BuildingReading) -> ContinuousHV {
        let weights = [
            reading.thermal_load.clamp(0.0, 1.0) as f32,
            reading.structural_stress.clamp(0.0, 1.0) as f32,
            reading.occupancy.clamp(0.0, 1.0) as f32,
            reading.comfort.clamp(0.0, 1.0) as f32,
            reading.energy_consumption.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for BuildingHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Predictor ───────────────────────────────────────────────────────────

/// Multi-scale predictor for building system dynamics.
pub struct BuildingPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl BuildingPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 3600.0,
            backbone_tau: 86400.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xB1D_10A0),
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

impl Default for BuildingPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for BuildingPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }

    fn domain(&self) -> &'static str {
        "building"
    }

    fn tau_base(&self) -> f32 {
        3600.0
    }

    fn default_horizons(&self) -> &'static [f32] {
        BUILDING_HORIZONS
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        BUILDING_HORIZON_LABELS
    }
}

// ── FEP Agent ───────────────────────────────────────────────────────────

/// Actions for building systems control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BuildingFepAction {
    Maintain,
    AdjustHVAC,
    LoadShed,
    ScheduleRepair,
    Evacuate,
}

impl BuildingFepAction {
    pub const ALL: [BuildingFepAction; 5] = [
        BuildingFepAction::Maintain,
        BuildingFepAction::AdjustHVAC,
        BuildingFepAction::LoadShed,
        BuildingFepAction::ScheduleRepair,
        BuildingFepAction::Evacuate,
    ];
}

const FE_EVACUATE: f64 = 0.7;
const FE_REPAIR: f64 = 0.5;
const FE_LOADSHED: f64 = 0.3;
const FE_HVAC: f64 = 0.1;

/// FEP agent for building systems stability.
pub struct BuildingFepAgent {
    reference_state: ContinuousHV,
}

impl BuildingFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0xB1D_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> BuildingFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > FE_EVACUATE {
            BuildingFepAction::Evacuate
        } else if fe > FE_REPAIR {
            BuildingFepAction::ScheduleRepair
        } else if fe > FE_LOADSHED {
            BuildingFepAction::LoadShed
        } else if fe > FE_HVAC {
            BuildingFepAction::AdjustHVAC
        } else {
            BuildingFepAction::Maintain
        }
    }
}

impl Default for BuildingFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ── Output ──────────────────────────────────────────────────────────────

/// Safety level for building operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BuildingSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

/// Output from a building prediction cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingOutput {
    pub free_energy: f64,
    pub recommended_action: BuildingFepAction,
    pub safety_level: BuildingSafetyLevel,
    pub prediction_similarities: Vec<(f32, f32)>,
}

impl BuildingOutput {
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

/// Building digital twin.
pub struct BuildingTwin {
    encoder: BuildingHdcEncoder,
    predictor: BuildingPredictor,
    agent: BuildingFepAgent,
    cycle_count: u64,
}

impl BuildingTwin {
    pub fn new() -> Self {
        Self {
            encoder: BuildingHdcEncoder::new(),
            predictor: BuildingPredictor::new(),
            agent: BuildingFepAgent::new(),
            cycle_count: 0,
        }
    }

    pub fn set_reference(&mut self, reading: &BuildingReading) {
        let hv = self.encoder.encode(reading);
        self.agent.set_reference(hv);
    }

    pub fn step(&mut self, reading: &BuildingReading, dt_seconds: f32) -> BuildingOutput {
        assert!(dt_seconds.is_finite() && dt_seconds > 0.0);
        self.cycle_count += 1;

        let hv = self.encoder.encode(reading);
        self.predictor.observe(&hv, dt_seconds);

        let sims: Vec<(f32, f32)> = BUILDING_HORIZONS
            .iter()
            .map(|&h| {
                let pred = self.predictor.predict_at_horizon(&hv, h);
                (h, pred.similarity(&hv))
            })
            .collect();

        let fe = self.agent.compute_free_energy(&hv);
        let action = self.agent.select_action(&hv);
        let level = if fe > FE_EVACUATE {
            BuildingSafetyLevel::Red
        } else if fe > FE_REPAIR {
            BuildingSafetyLevel::Orange
        } else if fe > FE_HVAC {
            BuildingSafetyLevel::Yellow
        } else {
            BuildingSafetyLevel::Green
        };

        BuildingOutput {
            free_energy: fe,
            recommended_action: action,
            safety_level: level,
            prediction_similarities: sims,
        }
    }

    pub fn predictor(&self) -> &BuildingPredictor {
        &self.predictor
    }

    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for BuildingTwin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy_reading() -> BuildingReading {
        BuildingReading {
            thermal_load: 0.4,
            structural_stress: 0.1,
            occupancy: 0.6,
            comfort: 0.85,
            energy_consumption: 0.35,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..BUILDING_HORIZONS.len() {
            assert!(BUILDING_HORIZONS[i] > BUILDING_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(BUILDING_HORIZONS.len(), BUILDING_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        let enc = BuildingHdcEncoder::new();
        let hv = enc.encode(&healthy_reading());
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_dimension() {
        let pred = BuildingPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let out = pred.predict_at_horizon(&input, 3600.0);
        assert_eq!(out.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_o1_property() {
        let pred = BuildingPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let t1 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 3_600.0);
        }
        let short = t1.elapsed();

        let t2 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 946_080_000.0);
        }
        let long = t2.elapsed();

        let ratio = long.as_nanos() as f64 / short.as_nanos().max(1) as f64;
        assert!(ratio < 5.0 && ratio > 0.2, "O(1) violated: ratio={}", ratio);
    }

    #[test]
    fn test_fep_self_reference() {
        let mut agent = BuildingFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        let fe = agent.compute_free_energy(&hv);
        assert!(fe < 0.01, "Self-reference FE should be ~0, got {}", fe);
    }

    #[test]
    fn test_fep_action_escalation() {
        assert!(BuildingFepAction::Maintain < BuildingFepAction::Evacuate);
    }

    #[test]
    fn test_twin_healthy() {
        let reading = healthy_reading();
        let mut twin = BuildingTwin::new();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 3600.0);
        assert_eq!(output.safety_level, BuildingSafetyLevel::Green);
        assert_eq!(output.recommended_action, BuildingFepAction::Maintain);
    }

    #[test]
    fn test_twin_cycle_count() {
        let mut twin = BuildingTwin::new();
        twin.step(&healthy_reading(), 3600.0);
        twin.step(&healthy_reading(), 3600.0);
        assert_eq!(twin.cycle_count(), 2);
    }

    #[test]
    fn test_safety_level_ordering() {
        assert!(BuildingSafetyLevel::Green < BuildingSafetyLevel::Yellow);
        assert!(BuildingSafetyLevel::Yellow < BuildingSafetyLevel::Orange);
        assert!(BuildingSafetyLevel::Orange < BuildingSafetyLevel::Red);
    }

    #[test]
    fn test_safety_tuple_self_reference() {
        let reading = healthy_reading();
        let mut twin = BuildingTwin::new();
        twin.set_reference(&reading);
        let output = twin.step(&reading, 3600.0);
        let (consciousness, error, coherence) = output.to_safety_tuple();
        assert!(consciousness > 0.9, "consciousness={}", consciousness);
        assert!(error < 0.1, "error={}", error);
        assert!(coherence.is_finite(), "coherence must be finite");
    }

    #[test]
    fn test_safety_tuple_range() {
        let reading = healthy_reading();
        let mut twin = BuildingTwin::new();
        let output = twin.step(&reading, 3600.0);
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
