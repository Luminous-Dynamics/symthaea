// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proliferation Safeguards — Genesis Mission Challenge 24
//!
//! HDC + CfC + FEP architecture for nuclear material inventory monitoring.
//! Detects inventory discrepancies, sensor anomalies, and timeline inconsistencies
//! across timescales from 1 day to 1 year.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const SAFEGUARDS_HORIZONS: &[f32] = &[
    86_400.0,     // 1 day — routine check
    604_800.0,    // 1 week — inspection cycle
    2_592_000.0,  // 1 month — inventory period
    7_776_000.0,  // 3 months — quarterly audit
    31_536_000.0, // 1 year — annual verification
];

pub const SAFEGUARDS_HORIZON_LABELS: &[&str] = &[
    "1 day (routine)",
    "1 week (inspection)",
    "1 month (inventory)",
    "3 months (audit)",
    "1 year (verification)",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeguardsReading {
    pub inventory_discrepancy: f64,
    pub sensor_anomaly: f64,
    pub timeline_consistency: f64,
}

pub struct SafeguardsHdcEncoder {
    bases: [ContinuousHV; 3],
}

impl SafeguardsHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 3] = [0x5F6_0001, 0x5F6_0002, 0x5F6_0003];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &SafeguardsReading) -> ContinuousHV {
        let weights = [
            reading.inventory_discrepancy.clamp(0.0, 1.0) as f32,
            reading.sensor_anomaly.clamp(0.0, 1.0) as f32,
            reading.timeline_consistency.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for SafeguardsHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SafeguardsPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl SafeguardsPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 86_400.0,
            backbone_tau: 31_536_000.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0x5F6_10A0),
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

impl Default for SafeguardsPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for SafeguardsPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }
    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }
    fn domain(&self) -> &'static str {
        "safeguards"
    }
    fn tau_base(&self) -> f32 {
        86_400.0
    }
    fn default_horizons(&self) -> &'static [f32] {
        SAFEGUARDS_HORIZONS
    }
    fn horizon_labels(&self) -> &'static [&'static str] {
        SAFEGUARDS_HORIZON_LABELS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SafeguardsFepAction {
    ContinueMonitoring,
    FlagForReview,
    RequestInspection,
    EscalateToIAEA,
    EmergencyAlert,
}

impl SafeguardsFepAction {
    pub const ALL: [SafeguardsFepAction; 5] = [
        SafeguardsFepAction::ContinueMonitoring,
        SafeguardsFepAction::FlagForReview,
        SafeguardsFepAction::RequestInspection,
        SafeguardsFepAction::EscalateToIAEA,
        SafeguardsFepAction::EmergencyAlert,
    ];
}

pub struct SafeguardsFepAgent {
    reference_state: ContinuousHV,
}

impl SafeguardsFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0x5F6_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> SafeguardsFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > 0.7 {
            SafeguardsFepAction::EmergencyAlert
        } else if fe > 0.5 {
            SafeguardsFepAction::EscalateToIAEA
        } else if fe > 0.3 {
            SafeguardsFepAction::RequestInspection
        } else if fe > 0.1 {
            SafeguardsFepAction::FlagForReview
        } else {
            SafeguardsFepAction::ContinueMonitoring
        }
    }
}

impl Default for SafeguardsFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy() -> SafeguardsReading {
        SafeguardsReading {
            inventory_discrepancy: 0.01,
            sensor_anomaly: 0.02,
            timeline_consistency: 0.95,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..SAFEGUARDS_HORIZONS.len() {
            assert!(SAFEGUARDS_HORIZONS[i] > SAFEGUARDS_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(SAFEGUARDS_HORIZONS.len(), SAFEGUARDS_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        assert_eq!(
            SafeguardsHdcEncoder::new().encode(&healthy()).dim(),
            HDC_DIMENSION
        );
    }

    #[test]
    fn test_o1_property() {
        let pred = SafeguardsPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let t1 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 86_400.0);
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
        let mut agent = SafeguardsFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        assert!(agent.compute_free_energy(&hv) < 0.01);
    }

    #[test]
    fn test_action_ordering() {
        assert!(SafeguardsFepAction::ContinueMonitoring < SafeguardsFepAction::EmergencyAlert);
    }
}
