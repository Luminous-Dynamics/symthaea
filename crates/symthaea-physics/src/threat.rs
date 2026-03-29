// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Threat Assessment — Genesis Mission Challenge 22
//!
//! HDC + CfC + FEP architecture for anomaly detection and threat response.
//! Monitors sensor readings against expected baselines to detect surprise signals
//! across timescales from 10ms to 1 hour.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const THREAT_HORIZONS: &[f32] = &[
    0.01,    // 10 ms — immediate detection
    0.1,     // 100 ms — pattern confirmation
    1.0,     // 1 s — response initiation
    60.0,    // 1 min — situational assessment
    3_600.0, // 1 hr — threat evolution
];

pub const THREAT_HORIZON_LABELS: &[&str] = &[
    "10 ms (detection)",
    "100 ms (confirmation)",
    "1 s (response)",
    "1 min (assessment)",
    "1 hr (evolution)",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatReading {
    pub sensor_reading: f64,
    pub expected_value: f64,
    pub surprise_signal: f64,
    pub response_time: f64,
    pub confidence: f64,
}

pub struct ThreatHdcEncoder {
    bases: [ContinuousHV; 5],
}

impl ThreatHdcEncoder {
    pub fn new() -> Self {
        let seeds: [u64; 5] = [0x7E8_0001, 0x7E8_0002, 0x7E8_0003, 0x7E8_0004, 0x7E8_0005];
        Self {
            bases: seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s)),
        }
    }

    pub fn encode(&self, reading: &ThreatReading) -> ContinuousHV {
        let weights = [
            reading.sensor_reading.clamp(0.0, 1.0) as f32,
            reading.expected_value.clamp(0.0, 1.0) as f32,
            reading.surprise_signal.clamp(0.0, 1.0) as f32,
            (reading.response_time / 10.0).clamp(0.0, 1.0) as f32,
            reading.confidence.clamp(0.0, 1.0) as f32,
        ];
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

impl Default for ThreatHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ThreatPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl ThreatPredictor {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 1.0,
            backbone_tau: 60.0,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0x7E8_10A0),
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

impl Default for ThreatPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for ThreatPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }
    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }
    fn domain(&self) -> &'static str {
        "threat"
    }
    fn tau_base(&self) -> f32 {
        1.0
    }
    fn default_horizons(&self) -> &'static [f32] {
        THREAT_HORIZONS
    }
    fn horizon_labels(&self) -> &'static [&'static str] {
        THREAT_HORIZON_LABELS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatFepAction {
    Monitor,
    ElevateAlert,
    ActivateResponse,
    Evacuate,
    EmergencyProtocol,
}

impl ThreatFepAction {
    pub const ALL: [ThreatFepAction; 5] = [
        ThreatFepAction::Monitor,
        ThreatFepAction::ElevateAlert,
        ThreatFepAction::ActivateResponse,
        ThreatFepAction::Evacuate,
        ThreatFepAction::EmergencyProtocol,
    ];
}

const FE_EMERGENCY_PROTOCOL: f64 = 0.7;
const FE_EVACUATE: f64 = 0.5;
const FE_ACTIVATE: f64 = 0.3;
const FE_ELEVATE: f64 = 0.1;

pub struct ThreatFepAgent {
    reference_state: ContinuousHV,
}

impl ThreatFepAgent {
    pub fn new() -> Self {
        Self {
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0x7E8_BEEF),
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

    pub fn select_action(&self, observed: &ContinuousHV) -> ThreatFepAction {
        let fe = self.compute_free_energy(observed);
        if fe > FE_EMERGENCY_PROTOCOL {
            ThreatFepAction::EmergencyProtocol
        } else if fe > FE_EVACUATE {
            ThreatFepAction::Evacuate
        } else if fe > FE_ACTIVATE {
            ThreatFepAction::ActivateResponse
        } else if fe > FE_ELEVATE {
            ThreatFepAction::ElevateAlert
        } else {
            ThreatFepAction::Monitor
        }
    }
}

impl Default for ThreatFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatOutput {
    pub free_energy: f64,
    pub recommended_action: ThreatFepAction,
    pub threat_level: ThreatLevel,
    pub prediction_similarities: Vec<(f32, f32)>,
}

pub struct ThreatTwin {
    encoder: ThreatHdcEncoder,
    predictor: ThreatPredictor,
    agent: ThreatFepAgent,
    cycle_count: u64,
}

impl ThreatTwin {
    pub fn new() -> Self {
        Self {
            encoder: ThreatHdcEncoder::new(),
            predictor: ThreatPredictor::new(),
            agent: ThreatFepAgent::new(),
            cycle_count: 0,
        }
    }

    pub fn set_reference(&mut self, reading: &ThreatReading) {
        let hv = self.encoder.encode(reading);
        self.agent.set_reference(hv);
    }

    pub fn step(&mut self, reading: &ThreatReading, dt_seconds: f32) -> ThreatOutput {
        assert!(dt_seconds.is_finite() && dt_seconds > 0.0);
        self.cycle_count += 1;
        let hv = self.encoder.encode(reading);
        self.predictor.observe(&hv, dt_seconds);
        let sims: Vec<(f32, f32)> = THREAT_HORIZONS
            .iter()
            .map(|&h| (h, self.predictor.predict_at_horizon(&hv, h).similarity(&hv)))
            .collect();
        let fe = self.agent.compute_free_energy(&hv);
        let action = self.agent.select_action(&hv);
        let level = if fe > FE_EMERGENCY_PROTOCOL {
            ThreatLevel::Red
        } else if fe > FE_EVACUATE {
            ThreatLevel::Orange
        } else if fe > FE_ELEVATE {
            ThreatLevel::Yellow
        } else {
            ThreatLevel::Green
        };
        ThreatOutput {
            free_energy: fe,
            recommended_action: action,
            threat_level: level,
            prediction_similarities: sims,
        }
    }

    pub fn predictor(&self) -> &ThreatPredictor {
        &self.predictor
    }
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }
}

impl Default for ThreatTwin {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn healthy() -> ThreatReading {
        ThreatReading {
            sensor_reading: 0.5,
            expected_value: 0.5,
            surprise_signal: 0.0,
            response_time: 1.0,
            confidence: 0.9,
        }
    }

    #[test]
    fn test_horizons_ordered() {
        for i in 1..THREAT_HORIZONS.len() {
            assert!(THREAT_HORIZONS[i] > THREAT_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_horizons_labels_match() {
        assert_eq!(THREAT_HORIZONS.len(), THREAT_HORIZON_LABELS.len());
    }

    #[test]
    fn test_encoder_dimension() {
        let enc = ThreatHdcEncoder::new();
        assert_eq!(enc.encode(&healthy()).dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_dimension() {
        let pred = ThreatPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        assert_eq!(pred.predict_at_horizon(&input, 1.0).dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_o1_property() {
        let pred = ThreatPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let t1 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 0.01);
        }
        let short = t1.elapsed();
        let t2 = Instant::now();
        for _ in 0..100 {
            let _ = pred.predict_at_horizon(&input, 3600.0);
        }
        let long = t2.elapsed();
        let ratio = long.as_nanos() as f64 / short.as_nanos().max(1) as f64;
        assert!(
            ratio < 10.0 && ratio > 0.1,
            "O(1) violated: ratio={}",
            ratio
        );
    }

    #[test]
    fn test_fep_self_reference() {
        let mut agent = ThreatFepAgent::new();
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(hv.clone());
        assert!(agent.compute_free_energy(&hv) < 0.01);
    }

    #[test]
    fn test_twin_healthy() {
        let r = healthy();
        let mut twin = ThreatTwin::new();
        twin.set_reference(&r);
        let out = twin.step(&r, 1.0);
        assert_eq!(out.threat_level, ThreatLevel::Green);
        assert_eq!(out.recommended_action, ThreatFepAction::Monitor);
    }

    #[test]
    fn test_safety_ordering() {
        assert!(ThreatLevel::Green < ThreatLevel::Red);
    }

    #[test]
    fn test_action_ordering() {
        assert!(ThreatFepAction::Monitor < ThreatFepAction::EmergencyProtocol);
    }
}
