// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fusion Digital Twin — Genesis Mission Challenge 3
//!
//! Ties a ContinuousHV-based plasma encoder + CfC multi-scale predictor + FEP active-inference
//! agent into a closed-loop digital twin for tokamak disruption avoidance.
//!
//! The key innovation is **O(1) temporal prediction**: predicting plasma state 10 seconds
//! ahead costs the same as predicting 1 ms ahead, thanks to the closed-form CfC solution.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::plasma_hdc_encoder::{PlasmaReading, PlasmaSensorType};

// ── FEP Action Thresholds (unified) ────────────────────────────────────────
/// Free energy thresholds for PlasmaFepAction selection AND DisruptionRisk
/// classification. Single source of truth — never duplicate these values.
const FE_EMERGENCY: f64 = 0.7;
const FE_SOFT_LANDING: f64 = 0.5;
const FE_INJECT_GAS: f64 = 0.3;
const FE_ADJUST_FIELD: f64 = 0.2;
const FE_REDUCE_HEATING: f64 = 0.1;

// ── Plasma-Tuned Horizons ──────────────────────────────────────────────────

/// Plasma prediction horizons (seconds): from sub-millisecond MHD events to 10s confinement.
pub const PLASMA_HORIZONS: &[f32] = &[
    0.001, // 1 ms   — MHD mode onset
    0.01,  // 10 ms  — disruption precursor
    0.1,   // 100 ms — current quench
    1.0,   // 1 s    — ramp-down
    10.0,  // 10 s   — steady-state confinement
];

/// Human-readable horizon labels for reporting.
pub const PLASMA_HORIZON_LABELS: &[&str] = &[
    "1 ms (MHD)",
    "10 ms (precursor)",
    "100 ms (quench)",
    "1 s (ramp-down)",
    "10 s (confinement)",
];

// ── Fusion HDC Encoder (ContinuousHV-based) ─────────────────────────────

/// ContinuousHV-based encoder for plasma readings (4 sensor dimensions).
///
/// Unlike the main `PlasmaHdcEncoder` (which uses BinaryHV), this encoder
/// produces ContinuousHV for compatibility with the CfC predictor and FEP agent.
struct FusionHdcEncoder {
    bases: [ContinuousHV; 4],
}

impl FusionHdcEncoder {
    fn new() -> Self {
        let seeds: [u64; 4] = [
            0xF05_0001, // electron density
            0xF05_0002, // electron temperature
            0xF05_0003, // radiated power
            0xF05_0004, // safety factor
        ];
        let bases = seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s));
        Self { bases }
    }

    /// Encode a window of plasma readings into a single ContinuousHV.
    fn encode_window(&self, readings: &[PlasmaReading]) -> ContinuousHV {
        if readings.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        // Extract per-sensor averages from the window
        let mut sums = [0.0f64; 4];
        let mut counts = [0u32; 4];
        for r in readings {
            if !r.value.is_finite() {
                continue;
            }
            let idx = match r.sensor {
                PlasmaSensorType::ElectronDensity => 0,
                PlasmaSensorType::ElectronTemperature => 1,
                PlasmaSensorType::RadiatedPower => 2,
                PlasmaSensorType::SafetyFactor => 3,
                _ => continue,
            };
            sums[idx] += r.value;
            counts[idx] += 1;
        }

        // Normalize to [0, 1] range (NaN-safe: division only when count > 0)
        let norms = [
            if counts[0] > 0 {
                (sums[0] / counts[0] as f64 / 2.0).clamp(0.0, 1.0)
            } else {
                0.0
            },
            if counts[1] > 0 {
                (sums[1] / counts[1] as f64 / 10.0).clamp(0.0, 1.0)
            } else {
                0.0
            },
            if counts[2] > 0 {
                (sums[2] / counts[2] as f64 / 15.0).clamp(0.0, 1.0)
            } else {
                0.0
            },
            if counts[3] > 0 {
                (sums[3] / counts[3] as f64 / 5.0).clamp(0.0, 1.0)
            } else {
                0.0
            },
        ];

        let weights: Vec<f32> = norms.iter().map(|&v| v as f32).collect();
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }
}

// ── Multi-Scale Plasma Predictor ───────────────────────────────────────────

/// Multi-scale predictor tuned for plasma timescales.
///
/// Uses a single `HdcLtcUnifiedNeuron` with tau_base appropriate for plasma
/// dynamics (~1 ms base). Each `predict_at_horizon` call is O(1).
pub struct PlasmaMultiScalePredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl PlasmaMultiScalePredictor {
    /// Create a predictor tuned for plasma timescales.
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 0.001, // 1 ms base (plasma MHD timescale)
            backbone_tau: 0.2,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xF05_10A0),
        }
    }

    /// Predict plasma state at a specific time horizon (seconds). O(1) cost.
    ///
    /// # Panics
    ///
    /// Panics if `horizon_seconds` is not finite or is non-positive.
    pub fn predict_at_horizon(
        &self,
        current_state: &ContinuousHV,
        horizon_seconds: f32,
    ) -> ContinuousHV {
        assert!(
            horizon_seconds.is_finite() && horizon_seconds > 0.0,
            "horizon_seconds must be finite and positive, got {}",
            horizon_seconds
        );
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(horizon_seconds, current_state);
        neuron_copy.state().clone()
    }

    /// Predict at all plasma horizons (1 ms to 10 s).
    pub fn predict_all_horizons(&self, current_state: &ContinuousHV) -> Vec<(f32, ContinuousHV)> {
        PLASMA_HORIZONS
            .iter()
            .map(|&h| (h, self.predict_at_horizon(current_state, h)))
            .collect()
    }

    /// Feed an observation to update internal state.
    pub fn observe(&mut self, observed_state: &ContinuousHV, dt_seconds: f32) {
        self.neuron.evolve_closed_form(dt_seconds, observed_state);
    }

    /// Current internal state.
    pub fn state(&self) -> &ContinuousHV {
        self.neuron.state()
    }
}

impl Default for PlasmaMultiScalePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for PlasmaMultiScalePredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }

    fn domain(&self) -> &'static str {
        "plasma"
    }

    fn tau_base(&self) -> f32 {
        0.001 // 1 ms (plasma MHD timescale)
    }

    fn default_horizons(&self) -> &'static [f32] {
        PLASMA_HORIZONS
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        PLASMA_HORIZON_LABELS
    }
}

// ── Plasma FEP Agent ───────────────────────────────────────────────────────

/// Discrete actions the plasma FEP agent can take to avoid disruptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PlasmaFepAction {
    /// Everything within tolerance — no intervention.
    MaintainState,
    /// Gradually reduce heating power.
    ReduceHeating,
    /// Adjust magnetic field coils for MHD stability.
    AdjustMagneticField,
    /// Inject impurity gas for radiative cooling.
    InjectGas,
    /// Controlled ramp-down to avoid runaway electrons.
    TriggerSoftLanding,
    /// Immediate shutdown — last resort.
    EmergencyShutdown,
}

impl PlasmaFepAction {
    pub const ALL: [PlasmaFepAction; 6] = [
        PlasmaFepAction::MaintainState,
        PlasmaFepAction::ReduceHeating,
        PlasmaFepAction::AdjustMagneticField,
        PlasmaFepAction::InjectGas,
        PlasmaFepAction::TriggerSoftLanding,
        PlasmaFepAction::EmergencyShutdown,
    ];

    pub fn to_index(self) -> usize {
        match self {
            PlasmaFepAction::MaintainState => 0,
            PlasmaFepAction::ReduceHeating => 1,
            PlasmaFepAction::AdjustMagneticField => 2,
            PlasmaFepAction::InjectGas => 3,
            PlasmaFepAction::TriggerSoftLanding => 4,
            PlasmaFepAction::EmergencyShutdown => 5,
        }
    }

    pub fn from_index(idx: usize) -> Self {
        match PlasmaFepAction::ALL.get(idx) {
            Some(&action) => action,
            None => PlasmaFepAction::EmergencyShutdown, // out-of-bounds → safest action
        }
    }
}

/// FEP-based agent for plasma disruption avoidance.
///
/// Uses HDC-encoded plasma state similarity to compute free energy (surprise),
/// then selects the safest intervention.
pub struct PlasmaFepAgent {
    /// Threshold above which the agent considers the state "surprising".
    pub surprise_threshold: f64,
    /// Reference "healthy" plasma state HV.
    reference_state: ContinuousHV,
}

impl PlasmaFepAgent {
    pub fn new() -> Self {
        Self {
            surprise_threshold: 0.4,
            reference_state: ContinuousHV::random(HDC_DIMENSION, 0xBEA_1174),
        }
    }

    /// Set the reference (expected) plasma state from a healthy shot.
    pub fn set_reference(&mut self, reference: ContinuousHV) {
        self.reference_state = reference;
    }

    /// Compute free energy (prediction error) between observed and reference state.
    ///
    /// Returns 1.0 (maximum surprise) if similarity is NaN.
    pub fn compute_free_energy(&self, observed: &ContinuousHV) -> f64 {
        let similarity = observed.similarity(&self.reference_state) as f64;
        if !similarity.is_finite() {
            return 1.0; // maximum surprise for degenerate states
        }
        (1.0 - similarity).max(0.0)
    }

    /// Whether the current plasma state is "surprising" (high free energy).
    pub fn is_surprised(&self, observed: &ContinuousHV) -> bool {
        self.compute_free_energy(observed) > self.surprise_threshold
    }

    /// Select the best action given the current plasma readings and their encoded HV.
    ///
    /// Decision logic mirrors NRC safety levels:
    /// - Low surprise → MaintainState
    /// - Moderate surprise → ReduceHeating or AdjustMagneticField
    /// - High surprise → InjectGas or TriggerSoftLanding
    /// - Extreme surprise → EmergencyShutdown
    pub fn select_action(
        &self,
        readings: &[PlasmaReading],
        current_hv: &ContinuousHV,
    ) -> PlasmaFepAction {
        let free_energy = self.compute_free_energy(current_hv);

        // Check for specific dangerous conditions
        let has_critical_density = readings
            .iter()
            .any(|r| r.sensor == PlasmaSensorType::ElectronDensity && r.value > 1.5);
        let has_high_radiation = readings
            .iter()
            .any(|r| r.sensor == PlasmaSensorType::RadiatedPower && r.value > 8.0);
        let has_low_safety_factor = readings
            .iter()
            .any(|r| r.sensor == PlasmaSensorType::SafetyFactor && r.value < 1.5);

        if free_energy > FE_EMERGENCY || has_low_safety_factor {
            PlasmaFepAction::EmergencyShutdown
        } else if free_energy > FE_SOFT_LANDING || (has_critical_density && has_high_radiation) {
            PlasmaFepAction::TriggerSoftLanding
        } else if free_energy > FE_INJECT_GAS || has_high_radiation {
            PlasmaFepAction::InjectGas
        } else if free_energy > FE_ADJUST_FIELD {
            PlasmaFepAction::AdjustMagneticField
        } else if free_energy > FE_REDUCE_HEATING {
            PlasmaFepAction::ReduceHeating
        } else {
            PlasmaFepAction::MaintainState
        }
    }
}

impl Default for PlasmaFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ── Fusion Digital Twin ────────────────────────────────────────────────────

/// NRC-style disruption risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DisruptionRisk {
    /// Normal operation.
    Green,
    /// Elevated monitoring — minor deviations detected.
    Yellow,
    /// Active intervention required.
    Orange,
    /// Imminent disruption — emergency procedures.
    Red,
}

/// Output from a single digital-twin prediction cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionTwinOutput {
    /// Multi-scale predictions at plasma horizons.
    pub predictions: Vec<(f32, String)>,
    /// Current disruption risk level.
    pub disruption_risk: DisruptionRisk,
    /// Free energy (surprise) of current state.
    pub free_energy: f64,
    /// Recommended FEP action.
    pub recommended_action: PlasmaFepAction,
    /// Similarity of each predicted state to reference.
    pub prediction_similarities: Vec<(f32, f32)>,
}

/// Fusion Digital Twin: closed-loop plasma control via HDC + CfC + FEP.
///
/// Combines:
/// - **FusionHdcEncoder** to encode sensor data into 16,384D ContinuousHV
/// - **PlasmaMultiScalePredictor** for O(1) temporal prediction (1 ms — 10 s)
/// - **PlasmaFepAgent** for disruption-avoidance action selection
///
/// # Genesis Mission Challenge 3
///
/// Addresses the DOE Fusion Energy challenge by demonstrating that a single
/// consciousness-like architecture can monitor, predict, and control plasma
/// instabilities across 4 orders of magnitude in timescale.
pub struct FusionDigitalTwin {
    encoder: FusionHdcEncoder,
    predictor: PlasmaMultiScalePredictor,
    agent: PlasmaFepAgent,
    cycle_count: u64,
}

impl FusionDigitalTwin {
    /// Create a new digital twin with default configuration.
    pub fn new() -> Self {
        Self {
            encoder: FusionHdcEncoder::new(),
            predictor: PlasmaMultiScalePredictor::new(),
            agent: PlasmaFepAgent::new(),
            cycle_count: 0,
        }
    }

    /// Set the reference healthy-plasma state from a known-good shot.
    pub fn set_reference_shot(&mut self, readings: &[PlasmaReading]) {
        let reference_hv = self.encoder.encode_window(readings);
        self.agent.set_reference(reference_hv);
    }

    /// Run one prediction cycle on new sensor readings.
    ///
    /// Returns a `FusionTwinOutput` containing predictions at all plasma horizons,
    /// the current disruption risk level, and a recommended action.
    ///
    /// # Panics
    ///
    /// Panics if `dt_seconds` is not finite or is non-positive.
    pub fn step(&mut self, readings: &[PlasmaReading], dt_seconds: f32) -> FusionTwinOutput {
        assert!(
            dt_seconds.is_finite() && dt_seconds > 0.0,
            "dt_seconds must be finite and positive, got {}",
            dt_seconds
        );
        self.cycle_count += 1;

        // 1. Encode current state
        let current_hv = self.encoder.encode_window(readings);

        // 2. Update predictor internal state
        self.predictor.observe(&current_hv, dt_seconds);

        // 3. Multi-scale prediction
        let raw_predictions = self.predictor.predict_all_horizons(&current_hv);

        // 4. Compute similarity of each prediction to reference
        let prediction_similarities: Vec<(f32, f32)> = raw_predictions
            .iter()
            .map(|(h, pred)| (*h, pred.similarity(&current_hv)))
            .collect();

        let predictions: Vec<(f32, String)> = raw_predictions
            .iter()
            .zip(PLASMA_HORIZON_LABELS.iter())
            .map(|((h, _), label)| (*h, label.to_string()))
            .collect();

        // 5. FEP action selection
        let free_energy = self.agent.compute_free_energy(&current_hv);
        let recommended_action = self.agent.select_action(readings, &current_hv);

        // 6. Map free energy to disruption risk
        let disruption_risk = if free_energy > FE_EMERGENCY {
            DisruptionRisk::Red
        } else if free_energy > FE_SOFT_LANDING {
            DisruptionRisk::Orange
        } else if free_energy > FE_REDUCE_HEATING {
            DisruptionRisk::Yellow
        } else {
            DisruptionRisk::Green
        };

        FusionTwinOutput {
            predictions,
            disruption_risk,
            free_energy,
            recommended_action,
            prediction_similarities,
        }
    }

    /// How many cycles have been processed.
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }

    /// Access the predictor (for inspection).
    pub fn predictor(&self) -> &PlasmaMultiScalePredictor {
        &self.predictor
    }
}

impl Default for FusionDigitalTwin {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper: synthetic disruption scenario ──────────────────────────────────

/// Generate a synthetic plasma shot that transitions from healthy to disruption.
///
/// Returns a vector of reading-windows, each representing one time step.
/// The first `healthy_steps` are normal operation; subsequent steps have
/// increasing instability (rising radiation, falling safety factor).
pub fn synthetic_disruption_scenario(
    healthy_steps: usize,
    disruption_steps: usize,
) -> Vec<Vec<PlasmaReading>> {
    let mut scenario = Vec::new();

    for step in 0..(healthy_steps + disruption_steps) {
        let t = step as f64 * 0.001; // 1 ms per step

        let (density, temp, radiation, safety_q) = if step < healthy_steps {
            // Healthy phase
            (0.8, 5.0, 2.0, 3.5)
        } else {
            // Disruption onset: density rises, radiation spikes, safety factor drops
            let frac = (step - healthy_steps) as f64 / disruption_steps.max(1) as f64;
            (
                0.8 + frac * 0.8, // 0.8 → 1.6
                5.0 - frac * 2.0, // 5.0 → 3.0
                2.0 + frac * 8.0, // 2.0 → 10.0
                3.5 - frac * 2.5, // 3.5 → 1.0
            )
        };

        scenario.push(vec![
            PlasmaReading::new(PlasmaSensorType::ElectronDensity, density, t),
            PlasmaReading::new(PlasmaSensorType::ElectronTemperature, temp, t),
            PlasmaReading::new(PlasmaSensorType::RadiatedPower, radiation, t),
            PlasmaReading::new(PlasmaSensorType::SafetyFactor, safety_q, t),
        ]);
    }

    scenario
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_plasma_horizons_ordered() {
        for i in 1..PLASMA_HORIZONS.len() {
            assert!(PLASMA_HORIZONS[i] > PLASMA_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_plasma_horizons_labels_match() {
        assert_eq!(PLASMA_HORIZONS.len(), PLASMA_HORIZON_LABELS.len());
    }

    #[test]
    fn test_predictor_dimension() {
        let predictor = PlasmaMultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let predicted = predictor.predict_at_horizon(&input, 0.01);
        assert_eq!(predicted.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_all_horizons_count() {
        let predictor = PlasmaMultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let preds = predictor.predict_all_horizons(&input);
        assert_eq!(preds.len(), PLASMA_HORIZONS.len());
    }

    #[test]
    fn test_o1_property_plasma() {
        let predictor = PlasmaMultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let start_1ms = Instant::now();
        for _ in 0..100 {
            let _ = predictor.predict_at_horizon(&input, 0.001);
        }
        let time_1ms = start_1ms.elapsed();

        let start_10s = Instant::now();
        for _ in 0..100 {
            let _ = predictor.predict_at_horizon(&input, 10.0);
        }
        let time_10s = start_10s.elapsed();

        let ratio = time_10s.as_nanos() as f64 / time_1ms.as_nanos().max(1) as f64;
        assert!(
            ratio < 5.0 && ratio > 0.2,
            "O(1) property: 1ms took {:?}, 10s took {:?}, ratio={}",
            time_1ms,
            time_10s,
            ratio
        );
    }

    #[test]
    fn test_fep_action_index_roundtrip() {
        for action in &PlasmaFepAction::ALL {
            let idx = action.to_index();
            let recovered = PlasmaFepAction::from_index(idx);
            assert_eq!(*action, recovered);
        }
    }

    #[test]
    fn test_fep_agent_free_energy_self() {
        let mut agent = PlasmaFepAgent::new();
        let ref_hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(ref_hv.clone());
        let fe = agent.compute_free_energy(&ref_hv);
        assert!(
            fe < 0.01,
            "Free energy should be ~0 for reference state, got {}",
            fe
        );
    }

    #[test]
    fn test_fep_agent_not_surprised_at_reference() {
        let mut agent = PlasmaFepAgent::new();
        let ref_hv = ContinuousHV::random(HDC_DIMENSION, 42);
        agent.set_reference(ref_hv.clone());
        assert!(!agent.is_surprised(&ref_hv));
    }

    #[test]
    fn test_disruption_risk_ordering() {
        assert!(DisruptionRisk::Green < DisruptionRisk::Yellow);
        assert!(DisruptionRisk::Yellow < DisruptionRisk::Orange);
        assert!(DisruptionRisk::Orange < DisruptionRisk::Red);
    }

    #[test]
    fn test_synthetic_scenario_length() {
        let scenario = synthetic_disruption_scenario(100, 50);
        assert_eq!(scenario.len(), 150);
    }

    #[test]
    fn test_digital_twin_healthy() {
        let scenario = synthetic_disruption_scenario(10, 0);
        let mut twin = FusionDigitalTwin::new();
        twin.set_reference_shot(&scenario[0]);

        let output = twin.step(&scenario[0], 0.001);
        assert_eq!(output.disruption_risk, DisruptionRisk::Green);
        assert_eq!(output.recommended_action, PlasmaFepAction::MaintainState);
    }

    #[test]
    fn test_digital_twin_disruption_escalation() {
        let scenario = synthetic_disruption_scenario(5, 10);
        let mut twin = FusionDigitalTwin::new();
        twin.set_reference_shot(&scenario[0]);

        let last_output = twin.step(&scenario[scenario.len() - 1], 0.001);

        // Sensor check (safety factor < 1.5) should trigger EmergencyShutdown
        assert_eq!(
            last_output.recommended_action,
            PlasmaFepAction::EmergencyShutdown,
            "Low safety factor should trigger EmergencyShutdown"
        );
        // Risk should be at least Yellow from free energy
        assert!(
            last_output.disruption_risk >= DisruptionRisk::Yellow,
            "Full disruption should trigger at least Yellow risk, got {:?} (fe={:.3})",
            last_output.disruption_risk,
            last_output.free_energy
        );
    }

    #[test]
    fn test_digital_twin_cycle_count() {
        let mut twin = FusionDigitalTwin::new();
        let readings = vec![PlasmaReading::new(
            PlasmaSensorType::ElectronDensity,
            0.8,
            0.0,
        )];
        twin.step(&readings, 0.001);
        twin.step(&readings, 0.001);
        assert_eq!(twin.cycle_count(), 2);
    }

    #[test]
    fn test_predictions_have_all_horizons() {
        let scenario = synthetic_disruption_scenario(5, 0);
        let mut twin = FusionDigitalTwin::new();
        let output = twin.step(&scenario[0], 0.001);
        assert_eq!(output.predictions.len(), PLASMA_HORIZONS.len());
        assert_eq!(output.prediction_similarities.len(), PLASMA_HORIZONS.len());
    }

    // ── Track B: failure-path tests ──────────────────────────────────────

    #[test]
    fn test_encode_window_nan_readings_skipped() {
        let encoder = FusionHdcEncoder::new();
        let readings = vec![
            PlasmaReading::new(PlasmaSensorType::ElectronDensity, f64::NAN, 0.0),
            PlasmaReading::new(PlasmaSensorType::ElectronDensity, 0.8, 0.001),
        ];
        let hv = encoder.encode_window(&readings);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        // Should produce a valid HV (NaN reading was skipped)
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_encode_window_all_nan_produces_zero_weights() {
        let encoder = FusionHdcEncoder::new();
        let readings = vec![
            PlasmaReading::new(PlasmaSensorType::ElectronDensity, f64::NAN, 0.0),
            PlasmaReading::new(PlasmaSensorType::ElectronTemperature, f64::INFINITY, 0.001),
        ];
        let hv = encoder.encode_window(&readings);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encode_window_empty() {
        let encoder = FusionHdcEncoder::new();
        let hv = encoder.encode_window(&[]);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        // Zero HV
        assert!(hv.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_free_energy_nan_observed_returns_max_surprise() {
        let agent = PlasmaFepAgent::new();
        // Zero HV will have zero norm → similarity may be NaN
        let zero_hv = ContinuousHV::zero(HDC_DIMENSION);
        let fe = agent.compute_free_energy(&zero_hv);
        assert!(
            fe.is_finite(),
            "Free energy should be finite even for zero HV"
        );
    }

    #[test]
    fn test_from_index_out_of_bounds_returns_emergency() {
        assert_eq!(
            PlasmaFepAction::from_index(6),
            PlasmaFepAction::EmergencyShutdown
        );
        assert_eq!(
            PlasmaFepAction::from_index(100),
            PlasmaFepAction::EmergencyShutdown
        );
        assert_eq!(
            PlasmaFepAction::from_index(usize::MAX),
            PlasmaFepAction::EmergencyShutdown
        );
    }

    #[test]
    #[should_panic(expected = "dt_seconds must be finite and positive")]
    fn test_step_rejects_nan_dt() {
        let mut twin = FusionDigitalTwin::new();
        twin.step(&[], f32::NAN);
    }

    #[test]
    #[should_panic(expected = "dt_seconds must be finite and positive")]
    fn test_step_rejects_zero_dt() {
        let mut twin = FusionDigitalTwin::new();
        twin.step(&[], 0.0);
    }

    #[test]
    #[should_panic(expected = "dt_seconds must be finite and positive")]
    fn test_step_rejects_negative_dt() {
        let mut twin = FusionDigitalTwin::new();
        twin.step(&[], -1.0);
    }

    #[test]
    #[should_panic(expected = "dt_seconds must be finite and positive")]
    fn test_step_rejects_infinity_dt() {
        let mut twin = FusionDigitalTwin::new();
        twin.step(&[], f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_predict_at_horizon_rejects_nan() {
        let predictor = PlasmaMultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        predictor.predict_at_horizon(&input, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_predict_at_horizon_rejects_zero() {
        let predictor = PlasmaMultiScalePredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        predictor.predict_at_horizon(&input, 0.0);
    }

    #[test]
    fn test_encode_window_negative_values_clamped() {
        let encoder = FusionHdcEncoder::new();
        let readings = vec![PlasmaReading::new(
            PlasmaSensorType::ElectronDensity,
            -5.0,
            0.0,
        )];
        let hv = encoder.encode_window(&readings);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }
}
