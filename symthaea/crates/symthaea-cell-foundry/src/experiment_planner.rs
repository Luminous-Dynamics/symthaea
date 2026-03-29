// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Experiment Planner — Genesis Mission Challenge 5
//!
//! HDC + CfC + FEP architecture for experimental capacity optimization.
//! Predicts experiment uncertainty, cost, time remaining, and information gain
//! across timescales from 1 hour to 1 month.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const EXPERIMENT_HORIZONS: &[f32] = &[
    3_600.0,     // 1 hr — batch completion
    86_400.0,    // 1 day — daily schedule
    604_800.0,   // 1 week — experiment cycle
    2_592_000.0, // 1 month — campaign
];

pub const EXPERIMENT_HORIZON_LABELS: &[&str] = &[
    "1 hr (batch)",
    "1 day (schedule)",
    "1 week (cycle)",
    "1 month (campaign)",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentReading {
    pub experiment_uncertainty: f64,
    pub cost: f64,
    pub time_remaining: f64,
    pub info_gain: f64,
}

pub struct ExperimentHdcEncoder {
    bases: [ContinuousHV; 4],
}

impl ExperimentHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 4] = [0xE4A_0001, 0xE4A_0002, 0xE4A_0003, 0xE4A_0004];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &ExperimentReading) -> ContinuousHV {
        let weights = [
            reading.experiment_uncertainty.clamp(0.0, 1.0) as f32,
            reading.cost.clamp(0.0, 1.0) as f32,
            (reading.time_remaining / 2_592_000.0).clamp(0.0, 1.0) as f32,
            reading.info_gain.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for ExperimentHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ExperimentPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl ExperimentPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 3_600.0,
            backbone_tau: 86_400.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xE4A_10A0),
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

impl Default for ExperimentPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for ExperimentPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }
    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }
    fn domain(&self) -> &'static str {
        "experiment"
    }
    fn tau_base(&self) -> f32 {
        3_600.0
    }
    fn default_horizons(&self) -> &'static [f32] {
        EXPERIMENT_HORIZONS
    }
    fn horizon_labels(&self) -> &'static [&'static str] {
        EXPERIMENT_HORIZON_LABELS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExperimentFepAction {
    Continue,
    Replicate,
    Reallocate,
    Pause,
    Abort,
}

impl ExperimentFepAction {
    pub const ALL: [ExperimentFepAction; 5] = [
        ExperimentFepAction::Continue,
        ExperimentFepAction::Replicate,
        ExperimentFepAction::Reallocate,
        ExperimentFepAction::Pause,
        ExperimentFepAction::Abort,
    ];
}

pub struct ExperimentFepAgent {
    reference_state: ContinuousHV,
}

impl ExperimentFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0xE4A_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> ExperimentFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > 0.7 {
            ExperimentFepAction::Abort
        } else if fe > 0.5 {
            ExperimentFepAction::Pause
        } else if fe > 0.3 {
            ExperimentFepAction::Reallocate
        } else if fe > 0.1 {
            ExperimentFepAction::Replicate
        } else {
            ExperimentFepAction::Continue
        }
    }
}

impl Default for ExperimentFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy() -> ExperimentReading {
        ExperimentReading {
            experiment_uncertainty: 0.2,
            cost: 0.3,
            time_remaining: 604_800.0,
            info_gain: 0.7,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..EXPERIMENT_HORIZONS.len() {
            assert!(EXPERIMENT_HORIZONS[i] > EXPERIMENT_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(EXPERIMENT_HORIZONS.len(), EXPERIMENT_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        assert_eq!(
            ExperimentHdcEncoder::new().encode(&healthy()).dim(),
            HDC_DIMENSION
        );
    }

    #[test]
    fn test_o1_property() {
        let pred = ExperimentPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let t1 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 3_600.0);
        }
        let short = t1.elapsed();
        let t2 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 2_592_000.0);
        }
        let long = t2.elapsed();
        let ratio = long.as_nanos() as f64 / short.as_nanos().max(1) as f64;
        assert!(ratio < 5.0 && ratio > 0.2, "O(1) violated: ratio={}", ratio);
    }

    #[test]
    fn test_fep_self_reference() {
        let mut agent = ExperimentFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        assert!(agent.compute_free_energy(&hv) < 0.01);
    }

    #[test]
    fn test_action_ordering() {
        assert!(ExperimentFepAction::Continue < ExperimentFepAction::Abort);
    }
}
