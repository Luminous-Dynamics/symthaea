// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strategic Materials — Genesis Mission Challenge 21
//!
//! HDC + CfC + FEP architecture for extreme-environment material performance.
//! Predicts resilience, radiation damage, and failure probability across
//! timescales from 1 day to 50 years.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const STRATEGIC_HORIZONS: &[f32] = &[
    86_400.0,        // 1 day — thermal cycle
    2_592_000.0,     // 1 month — irradiation campaign
    31_536_000.0,    // 1 year — annual inspection
    315_360_000.0,   // 10 years — mid-life review
    1_576_800_000.0, // 50 years — design lifetime
];

pub const STRATEGIC_HORIZON_LABELS: &[&str] = &[
    "1 day (thermal cycle)",
    "1 month (irradiation)",
    "1 year (inspection)",
    "10 years (mid-life)",
    "50 years (design life)",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategicReading {
    pub extreme_temp_resilience: f64,
    pub radiation_dose: f64,
    pub time_at_condition: f64,
    pub failure_probability: f64,
}

pub struct StrategicHdcEncoder {
    bases: [ContinuousHV; 4],
}

impl StrategicHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 4] = [0x578_0001, 0x578_0002, 0x578_0003, 0x578_0004];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &StrategicReading) -> ContinuousHV {
        let weights = [
            reading.extreme_temp_resilience.clamp(0.0, 1.0) as f32,
            reading.radiation_dose.clamp(0.0, 1.0) as f32,
            (reading.time_at_condition / 1_576_800_000.0).clamp(0.0, 1.0) as f32,
            reading.failure_probability.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for StrategicHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StrategicPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl StrategicPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 86_400.0,
            backbone_tau: 31_536_000.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0x578_10A0),
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

impl Default for StrategicPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for StrategicPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }
    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }
    fn domain(&self) -> &'static str {
        "strategic_materials"
    }
    fn tau_base(&self) -> f32 {
        86_400.0
    }
    fn default_horizons(&self) -> &'static [f32] {
        STRATEGIC_HORIZONS
    }
    fn horizon_labels(&self) -> &'static [&'static str] {
        STRATEGIC_HORIZON_LABELS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StrategicFepAction {
    ContinueService,
    IncreasedInspection,
    ReduceLoad,
    ScheduleReplacement,
    ImmediateWithdrawal,
}

impl StrategicFepAction {
    pub const ALL: [StrategicFepAction; 5] = [
        StrategicFepAction::ContinueService,
        StrategicFepAction::IncreasedInspection,
        StrategicFepAction::ReduceLoad,
        StrategicFepAction::ScheduleReplacement,
        StrategicFepAction::ImmediateWithdrawal,
    ];
}

pub struct StrategicFepAgent {
    reference_state: ContinuousHV,
}

impl StrategicFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0x578_BEEF),
        }
    }

    pub fn set_reference(&mut self, reference: ContinuousHV) {
        self.reference_state = reference;
    }

    /// Compute free energy between observed state and reference.
    pub fn compute_free_energy(&self, observed: &ContinuousHV) -> f64 {
        let sim = observed.similarity(&self.reference_state) as f64;
        if !sim.is_finite() {
            return 1.0;
        }
        (1.0 - sim).max(0.0)
    }

    /// Select strategic action based on observed free energy.
    pub fn select_action(&self, observed: &ContinuousHV) -> StrategicFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > 0.7 {
            StrategicFepAction::ImmediateWithdrawal
        } else if fe > 0.5 {
            StrategicFepAction::ScheduleReplacement
        } else if fe > 0.3 {
            StrategicFepAction::ReduceLoad
        } else if fe > 0.1 {
            StrategicFepAction::IncreasedInspection
        } else {
            StrategicFepAction::ContinueService
        }
    }
}

impl Default for StrategicFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy() -> StrategicReading {
        StrategicReading {
            extreme_temp_resilience: 0.9,
            radiation_dose: 0.1,
            time_at_condition: 86_400.0,
            failure_probability: 0.001,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..STRATEGIC_HORIZONS.len() {
            assert!(STRATEGIC_HORIZONS[i] > STRATEGIC_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(STRATEGIC_HORIZONS.len(), STRATEGIC_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        assert_eq!(
            StrategicHdcEncoder::new().encode(&healthy()).dim(),
            HDC_DIMENSION
        );
    }

    #[test]
    fn test_o1_property() {
        let pred = StrategicPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let t1 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 86_400.0);
        }
        let short = t1.elapsed();
        let t2 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 1_576_800_000.0);
        }
        let long = t2.elapsed();
        let ratio = long.as_nanos() as f64 / short.as_nanos().max(1) as f64;
        assert!(ratio < 5.0 && ratio > 0.2, "O(1) violated: ratio={}", ratio);
    }

    #[test]
    fn test_fep_self_reference() {
        let mut agent = StrategicFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        assert!(agent.compute_free_energy(&hv) < 0.01);
    }

    #[test]
    fn test_action_ordering() {
        assert!(StrategicFepAction::ContinueService < StrategicFepAction::ImmediateWithdrawal);
    }
}
