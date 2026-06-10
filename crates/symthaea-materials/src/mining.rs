// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Critical Minerals — Genesis Mission Challenge 18
//!
//! HDC + CfC + FEP architecture for mineral extraction monitoring.
//! Predicts ore grade, extraction rate, environmental impact, and cost
//! across timescales from 1 day to 10 years.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const MINING_HORIZONS: &[f32] = &[
    86_400.0,      // 1 day — blast cycle
    604_800.0,     // 1 week — extraction batch
    2_592_000.0,   // 1 month — grade trend
    31_536_000.0,  // 1 year — reserve depletion
    315_360_000.0, // 10 years — mine life
];

pub const MINING_HORIZON_LABELS: &[&str] = &[
    "1 day (blast cycle)",
    "1 week (extraction)",
    "1 month (grade trend)",
    "1 year (depletion)",
    "10 years (mine life)",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningReading {
    pub ore_grade: f64,
    pub extraction_rate: f64,
    pub environmental_impact: f64,
    pub cost: f64,
}

pub struct MiningHdcEncoder {
    bases: [ContinuousHV; 4],
}

impl MiningHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 4] = [0x41A_0001, 0x41A_0002, 0x41A_0003, 0x41A_0004];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &MiningReading) -> ContinuousHV {
        let weights = [
            reading.ore_grade.clamp(0.0, 1.0) as f32,
            reading.extraction_rate.clamp(0.0, 1.0) as f32,
            reading.environmental_impact.clamp(0.0, 1.0) as f32,
            reading.cost.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for MiningHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MiningPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl MiningPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 86_400.0,
            backbone_tau: 31_536_000.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0x41A_10A0),
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

impl Default for MiningPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for MiningPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }
    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }
    fn domain(&self) -> &'static str {
        "critical_minerals"
    }
    fn tau_base(&self) -> f32 {
        86_400.0
    }
    fn default_horizons(&self) -> &'static [f32] {
        MINING_HORIZONS
    }
    fn horizon_labels(&self) -> &'static [&'static str] {
        MINING_HORIZON_LABELS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MiningFepAction {
    ContinueExtraction,
    AdjustProcess,
    ReduceRate,
    HaltExtraction,
    EmergencyRemediation,
}

impl MiningFepAction {
    pub const ALL: [MiningFepAction; 5] = [
        MiningFepAction::ContinueExtraction,
        MiningFepAction::AdjustProcess,
        MiningFepAction::ReduceRate,
        MiningFepAction::HaltExtraction,
        MiningFepAction::EmergencyRemediation,
    ];
}

pub struct MiningFepAgent {
    reference_state: ContinuousHV,
}

impl MiningFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0x41A_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> MiningFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > 0.7 {
            MiningFepAction::EmergencyRemediation
        } else if fe > 0.5 {
            MiningFepAction::HaltExtraction
        } else if fe > 0.3 {
            MiningFepAction::ReduceRate
        } else if fe > 0.1 {
            MiningFepAction::AdjustProcess
        } else {
            MiningFepAction::ContinueExtraction
        }
    }
}

impl Default for MiningFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy() -> MiningReading {
        MiningReading {
            ore_grade: 0.5,
            extraction_rate: 0.7,
            environmental_impact: 0.1,
            cost: 0.4,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..MINING_HORIZONS.len() {
            assert!(MINING_HORIZONS[i] > MINING_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(MINING_HORIZONS.len(), MINING_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        assert_eq!(
            MiningHdcEncoder::new().encode(&healthy()).dim(),
            HDC_DIMENSION
        );
    }

    #[test]
    fn test_o1_property() {
        let pred = MiningPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let t1 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 86_400.0);
        }
        let short = t1.elapsed();
        let t2 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 315_360_000.0);
        }
        let long = t2.elapsed();
        let ratio = long.as_nanos() as f64 / short.as_nanos().max(1) as f64;
        assert!(ratio < 5.0 && ratio > 0.2, "O(1) violated: ratio={}", ratio);
    }

    #[test]
    fn test_fep_self_reference() {
        let mut agent = MiningFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        assert!(agent.compute_free_energy(&hv) < 0.01);
    }

    #[test]
    fn test_action_ordering() {
        assert!(MiningFepAction::ContinueExtraction < MiningFepAction::EmergencyRemediation);
    }
}
