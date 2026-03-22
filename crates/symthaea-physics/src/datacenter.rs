// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Data Center Operations — Genesis Mission Challenge 20
//!
//! HDC + CfC + FEP architecture for data center monitoring.
//! Predicts power draw, cooling load, latency, and throughput
//! across timescales from 1 minute to 1 year.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const DATACENTER_HORIZONS: &[f32] = &[
    60.0,         // 1 min — burst detection
    3_600.0,      // 1 hr — workload shift
    86_400.0,     // 1 day — diurnal pattern
    2_592_000.0,  // 1 month — capacity trend
    31_536_000.0, // 1 year — growth planning
];

pub const DATACENTER_HORIZON_LABELS: &[&str] = &[
    "1 min (burst)",
    "1 hr (workload)",
    "1 day (diurnal)",
    "1 month (trend)",
    "1 year (planning)",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatacenterReading {
    pub power_draw: f64,
    pub cooling_load: f64,
    pub latency: f64,
    pub throughput: f64,
}

pub struct DatacenterHdcEncoder {
    bases: [ContinuousHV; 4],
}

impl DatacenterHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 4] = [0xDC7_0001, 0xDC7_0002, 0xDC7_0003, 0xDC7_0004];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &DatacenterReading) -> ContinuousHV {
        let weights = [
            reading.power_draw.clamp(0.0, 1.0) as f32,
            reading.cooling_load.clamp(0.0, 1.0) as f32,
            (reading.latency / 100.0).clamp(0.0, 1.0) as f32,
            reading.throughput.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for DatacenterHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DatacenterPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl DatacenterPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 60.0,
            backbone_tau: 3600.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xDC7_10A0),
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

impl Default for DatacenterPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for DatacenterPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }
    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }
    fn domain(&self) -> &'static str {
        "datacenter"
    }
    fn tau_base(&self) -> f32 {
        60.0
    }
    fn default_horizons(&self) -> &'static [f32] {
        DATACENTER_HORIZONS
    }
    fn horizon_labels(&self) -> &'static [&'static str] {
        DATACENTER_HORIZON_LABELS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DatacenterFepAction {
    Maintain,
    AdjustCooling,
    MigrateWorkloads,
    ThrottleCompute,
    EmergencyCooldown,
}

impl DatacenterFepAction {
    pub const ALL: [DatacenterFepAction; 5] = [
        DatacenterFepAction::Maintain,
        DatacenterFepAction::AdjustCooling,
        DatacenterFepAction::MigrateWorkloads,
        DatacenterFepAction::ThrottleCompute,
        DatacenterFepAction::EmergencyCooldown,
    ];
}

const FE_EMERGENCY_COOL: f64 = 0.7;
const FE_THROTTLE: f64 = 0.5;
const FE_MIGRATE: f64 = 0.3;
const FE_COOL: f64 = 0.1;

pub struct DatacenterFepAgent {
    reference_state: ContinuousHV,
}

impl DatacenterFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0xDC7_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> DatacenterFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > FE_EMERGENCY_COOL {
            DatacenterFepAction::EmergencyCooldown
        } else if fe > FE_THROTTLE {
            DatacenterFepAction::ThrottleCompute
        } else if fe > FE_MIGRATE {
            DatacenterFepAction::MigrateWorkloads
        } else if fe > FE_COOL {
            DatacenterFepAction::AdjustCooling
        } else {
            DatacenterFepAction::Maintain
        }
    }
}

impl Default for DatacenterFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DatacenterSafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatacenterOutput {
    pub free_energy: f64,
    pub recommended_action: DatacenterFepAction,
    pub safety_level: DatacenterSafetyLevel,
    pub prediction_similarities: Vec<(f32, f32)>,
}

pub struct DatacenterTwin {
    encoder: DatacenterHdcEncoder,
    predictor: DatacenterPredictor,
    agent: DatacenterFepAgent,
    cycle_count: u64,
}

impl DatacenterTwin {
    pub fn new() -> Self {
        Self {
            encoder: DatacenterHdcEncoder::new(),
            predictor: DatacenterPredictor::new(),
            agent: DatacenterFepAgent::new(),
            cycle_count: 0,
        }
    }

    pub fn set_reference(&mut self, reading: &DatacenterReading) {
        let hv = self.encoder.encode(reading);
        self.agent.set_reference(hv);
    }

    pub fn step(&mut self, reading: &DatacenterReading, dt_seconds: f32) -> DatacenterOutput {
        assert!(dt_seconds.is_finite() && dt_seconds > 0.0);
        self.cycle_count += 1;
        let hv = self.encoder.encode(reading);
        self.predictor.observe(&hv, dt_seconds);
        let sims: Vec<(f32, f32)> = DATACENTER_HORIZONS
            .iter()
            .map(|&h| (h, self.predictor.predict_at_horizon(&hv, h).similarity(&hv)))
            .collect();
        let fe = self.agent.compute_free_energy(&hv);
        let action = self.agent.select_action(&hv);
        let level = if fe > FE_EMERGENCY_COOL {
            DatacenterSafetyLevel::Red
        } else if fe > FE_THROTTLE {
            DatacenterSafetyLevel::Orange
        } else if fe > FE_COOL {
            DatacenterSafetyLevel::Yellow
        } else {
            DatacenterSafetyLevel::Green
        };
        DatacenterOutput {
            free_energy: fe,
            recommended_action: action,
            safety_level: level,
            prediction_similarities: sims,
        }
    }

    pub fn predictor(&self) -> &DatacenterPredictor {
        &self.predictor
    }
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for DatacenterTwin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy() -> DatacenterReading {
        DatacenterReading {
            power_draw: 0.6,
            cooling_load: 0.4,
            latency: 5.0,
            throughput: 0.8,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..DATACENTER_HORIZONS.len() {
            assert!(DATACENTER_HORIZONS[i] > DATACENTER_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(DATACENTER_HORIZONS.len(), DATACENTER_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        assert_eq!(
            DatacenterHdcEncoder::new().encode(&healthy()).dim(),
            HDC_DIMENSION
        );
    }

    #[test]
    fn test_predictor_dimension() {
        let pred = DatacenterPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        assert_eq!(pred.predict_at_horizon(&input, 60.0).dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_o1_property() {
        let pred = DatacenterPredictor::new();
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
        let mut agent = DatacenterFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        assert!(agent.compute_free_energy(&hv) < 0.01);
    }

    #[test]
    fn test_twin_healthy() {
        let r = healthy();
        let mut twin = DatacenterTwin::new();
        twin.set_reference(&r);
        let out = twin.step(&r, 60.0);
        assert_eq!(out.safety_level, DatacenterSafetyLevel::Green);
        assert_eq!(out.recommended_action, DatacenterFepAction::Maintain);
    }

    #[test]
    fn test_safety_ordering() {
        assert!(DatacenterSafetyLevel::Green < DatacenterSafetyLevel::Red);
    }

    #[test]
    fn test_action_ordering() {
        assert!(DatacenterFepAction::Maintain < DatacenterFepAction::EmergencyCooldown);
    }
}
