// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Water Prediction — Genesis Mission Challenge 8
//!
//! HDC-encoded water quality monitoring with O(1) multi-scale temporal
//! prediction and FEP-based intervention selection.
//!
//! Mirrors Mycelix water zome data structures (QualityReading) without
//! Holochain dependencies, enabling offline prediction and planning.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Hydrological Horizons ──────────────────────────────────────────────────

/// Hydrological prediction horizons (seconds).
pub const WATER_HORIZONS: &[f32] = &[
    3_600.0,      // 1 hour   — contamination response
    86_400.0,     // 1 day    — daily monitoring
    604_800.0,    // 1 week   — weekly patterns
    2_592_000.0,  // 1 month  — seasonal trends
    31_536_000.0, // 1 year   — annual cycles
];

/// Human-readable horizon labels.
pub const WATER_HORIZON_LABELS: &[&str] = &[
    "1 hour (response)",
    "1 day (monitoring)",
    "1 week (patterns)",
    "1 month (seasonal)",
    "1 year (annual)",
];

// ── Water Quality Reading ──────────────────────────────────────────────────

/// A single water quality reading (mirrors Mycelix QualityReading).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterQualityReading {
    /// pH level (0-14, target: 6.5-8.5).
    pub ph: f64,
    /// Turbidity in NTU (target: < 1.0).
    pub turbidity_ntu: f64,
    /// Total Dissolved Solids in mg/L (target: < 500).
    pub tds_mg_l: f64,
    /// Dissolved Oxygen in mg/L (target: > 6.0).
    pub dissolved_oxygen_mg_l: f64,
    /// Nitrate level in mg/L (target: < 10.0).
    pub nitrates_mg_l: f64,
    /// Coliform count per 100mL (target: 0).
    pub coliform_per_100ml: f64,
    /// Temperature in Celsius.
    pub temperature_celsius: f64,
    /// Potability score (0.0 = unsafe, 1.0 = safe).
    pub potability: f64,
}

impl WaterQualityReading {
    /// Create a clean, potable water reading.
    pub fn clean() -> Self {
        Self {
            ph: 7.2,
            turbidity_ntu: 0.3,
            tds_mg_l: 250.0,
            dissolved_oxygen_mg_l: 8.0,
            nitrates_mg_l: 3.0,
            coliform_per_100ml: 0.0,
            temperature_celsius: 15.0,
            potability: 1.0,
        }
    }

    /// Create a contaminated water reading.
    pub fn contaminated() -> Self {
        Self {
            ph: 5.5,
            turbidity_ntu: 15.0,
            tds_mg_l: 800.0,
            dissolved_oxygen_mg_l: 3.0,
            nitrates_mg_l: 25.0,
            coliform_per_100ml: 50.0,
            temperature_celsius: 22.0,
            potability: 0.1,
        }
    }

    /// Deviation from potable standards (sum of squared fractional errors).
    pub fn deviation_from_potable(&self) -> f64 {
        let ref_reading = Self::clean();
        let ph_err = (self.ph - ref_reading.ph) / ref_reading.ph;
        let turb_err = (self.turbidity_ntu - ref_reading.turbidity_ntu).max(0.0) / 1.0;
        let tds_err = (self.tds_mg_l - ref_reading.tds_mg_l).max(0.0) / 500.0;
        let do_err =
            (ref_reading.dissolved_oxygen_mg_l - self.dissolved_oxygen_mg_l).max(0.0) / 8.0;
        let nit_err = (self.nitrates_mg_l - ref_reading.nitrates_mg_l).max(0.0) / 10.0;
        let col_err = self.coliform_per_100ml / 10.0;

        ph_err * ph_err
            + turb_err * turb_err
            + tds_err * tds_err
            + do_err * do_err
            + nit_err * nit_err
            + col_err * col_err
    }
}

// ── Water HDC Encoder ──────────────────────────────────────────────────────

/// HDC encoder for water quality readings.
///
/// Encodes 8 water parameters into a 16,384D hypervector using
/// the same binding + level-code pattern as PlasmaHdcEncoder.
pub struct WaterHdcEncoder {
    /// Base vectors for each parameter (deterministic seeds).
    bases: [ContinuousHV; 8],
}

impl WaterHdcEncoder {
    /// Create a new encoder with deterministic per-parameter base vectors.
    pub fn new() -> Self {
        let seeds: [u64; 8] = [
            0xDA7_E001, // pH
            0xDA7_E002, // turbidity
            0xDA7_E003, // TDS
            0xDA7_E004, // dissolved O2
            0xDA7_E005, // nitrates
            0xDA7_E006, // coliform
            0xDA7_E007, // temperature
            0xDA7_E008, // potability
        ];
        let bases = seeds.map(|s| ContinuousHV::random(HDC_DIMENSION, s));
        Self { bases }
    }

    /// Encode a single water reading into 16,384D.
    pub fn encode(&self, reading: &WaterQualityReading) -> ContinuousHV {
        let values = [
            reading.ph / 14.0,
            (reading.turbidity_ntu / 100.0).min(1.0),
            (reading.tds_mg_l / 2000.0).min(1.0),
            (reading.dissolved_oxygen_mg_l / 15.0).min(1.0),
            (reading.nitrates_mg_l / 50.0).min(1.0),
            (reading.coliform_per_100ml / 100.0).min(1.0),
            (reading.temperature_celsius / 40.0).min(1.0),
            reading.potability,
        ];

        let weights: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        ContinuousHV::encode_weighted(&self.bases, &weights)
    }

    /// Encode a window of readings (temporal bundling).
    pub fn encode_window(&self, readings: &[WaterQualityReading]) -> ContinuousHV {
        if readings.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }
        let encoded: Vec<ContinuousHV> = readings.iter().map(|r| self.encode(r)).collect();
        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        ContinuousHV::bundle(&refs)
    }
}

impl Default for WaterHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Hydrological Predictor ─────────────────────────────────────────────────

/// Multi-scale water quality predictor using O(1) closed-form temporal jumps.
pub struct HydrologicalPredictor {
    neuron: HdcLtcUnifiedNeuron,
}

impl HydrologicalPredictor {
    /// Create a predictor with a 1-hour base time constant suitable for hydrological data.
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 3600.0, // 1 hour base (hydrological timescale)
            backbone_tau: 0.1,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xB20_BEAD),
        }
    }

    /// Predict water state at a specific time horizon. O(1) cost.
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

    /// Predict at all hydrological horizons (1h to 1y).
    pub fn predict_all_horizons(&self, current_state: &ContinuousHV) -> Vec<(f32, ContinuousHV)> {
        WATER_HORIZONS
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

impl Default for HydrologicalPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for HydrologicalPredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_at_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }

    fn domain(&self) -> &'static str {
        "hydrology"
    }

    fn tau_base(&self) -> f32 {
        3600.0 // 1 hour (hydrological timescale)
    }

    fn default_horizons(&self) -> &'static [f32] {
        WATER_HORIZONS
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        WATER_HORIZON_LABELS
    }
}

// ── Water FEP Agent ────────────────────────────────────────────────────────

/// Discrete actions for water quality management.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Discrete interventions the water FEP agent can recommend.
pub enum WaterFepAction {
    /// Water quality within tolerance.
    Maintain,
    /// Add chlorine for disinfection.
    Chlorinate,
    /// Reduce water extraction rate.
    ReduceExtraction,
    /// Activate filtration systems.
    ActivateFilters,
    /// Issue public water quality advisory.
    IssueAdvisory,
    /// Emergency shutoff of water supply.
    EmergencyShutoff,
}

impl WaterFepAction {
    /// All available water management actions in a fixed-size array.
    pub const ALL: [WaterFepAction; 6] = [
        WaterFepAction::Maintain,
        WaterFepAction::Chlorinate,
        WaterFepAction::ReduceExtraction,
        WaterFepAction::ActivateFilters,
        WaterFepAction::IssueAdvisory,
        WaterFepAction::EmergencyShutoff,
    ];

    /// Convert this action to its canonical integer index.
    pub fn to_index(self) -> usize {
        match self {
            WaterFepAction::Maintain => 0,
            WaterFepAction::Chlorinate => 1,
            WaterFepAction::ReduceExtraction => 2,
            WaterFepAction::ActivateFilters => 3,
            WaterFepAction::IssueAdvisory => 4,
            WaterFepAction::EmergencyShutoff => 5,
        }
    }

    /// Convert an integer index back to an action; out-of-bounds returns `EmergencyShutoff`.
    pub fn from_index(idx: usize) -> Self {
        match WaterFepAction::ALL.get(idx) {
            Some(&action) => action,
            None => WaterFepAction::EmergencyShutoff, // out-of-bounds → safest action
        }
    }
}

/// FEP-based agent for water quality management.
pub struct WaterFepAgent {
    /// Free energy above which a reading is considered surprising (triggers intervention).
    pub surprise_threshold: f64,
    encoder: WaterHdcEncoder,
    reference_state: ContinuousHV,
}

impl WaterFepAgent {
    /// Create a new agent with a clean-water reference state and default surprise threshold.
    pub fn new() -> Self {
        let encoder = WaterHdcEncoder::new();
        let reference_state = encoder.encode(&WaterQualityReading::clean());
        Self {
            surprise_threshold: 0.4,
            encoder,
            reference_state,
        }
    }

    /// Set the reference from a known-good water reading.
    pub fn set_reference(&mut self, reading: &WaterQualityReading) {
        self.reference_state = self.encoder.encode(reading);
    }

    /// Compute free energy between observed and reference state.
    ///
    /// Returns 1.0 (maximum surprise) if similarity is NaN.
    pub fn compute_free_energy(&self, reading: &WaterQualityReading) -> f64 {
        let observed = self.encoder.encode(reading);
        let similarity = observed.similarity(&self.reference_state) as f64;
        if !similarity.is_finite() {
            return 1.0; // maximum surprise for degenerate states
        }
        (1.0 - similarity).max(0.0)
    }

    /// Whether the current reading is "surprising".
    pub fn is_surprised(&self, reading: &WaterQualityReading) -> bool {
        self.compute_free_energy(reading) > self.surprise_threshold
    }

    /// Select the best action given a water quality reading.
    pub fn select_action(&self, reading: &WaterQualityReading) -> WaterFepAction {
        let deviation = reading.deviation_from_potable();

        // Emergency: coliform or very low potability
        if reading.coliform_per_100ml > 20.0 || reading.potability < 0.2 {
            return WaterFepAction::EmergencyShutoff;
        }

        // High coliform → chlorinate
        if reading.coliform_per_100ml > 5.0 {
            return WaterFepAction::Chlorinate;
        }

        // High turbidity → activate filters
        if reading.turbidity_ntu > 5.0 {
            return WaterFepAction::ActivateFilters;
        }

        // High nitrates → reduce extraction
        if reading.nitrates_mg_l > 10.0 {
            return WaterFepAction::ReduceExtraction;
        }

        // Moderate deviation → advisory
        if deviation > 1.0 {
            return WaterFepAction::IssueAdvisory;
        }

        WaterFepAction::Maintain
    }
}

impl Default for WaterFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

// ── Synthetic Water Scenario ───────────────────────────────────────────────

/// Generate seasonal water quality data with optional contamination event.
///
/// Returns one reading per time step. If `contamination_step` is Some,
/// a coliform spike occurs at that step.
pub fn synthetic_water_scenario(
    steps: usize,
    contamination_step: Option<usize>,
) -> Vec<WaterQualityReading> {
    let mut readings = Vec::with_capacity(steps);

    for step in 0..steps {
        let seasonal = (step as f64 * std::f64::consts::PI * 2.0 / 365.0).sin();

        let (coliform, potability) = if contamination_step == Some(step) {
            (35.0, 0.1)
        } else {
            (0.0 + seasonal.abs() * 0.5, 0.95 + seasonal * 0.03)
        };

        readings.push(WaterQualityReading {
            ph: 7.2 + seasonal * 0.3,
            turbidity_ntu: 0.3 + seasonal.abs() * 0.5,
            tds_mg_l: 250.0 + seasonal * 30.0,
            dissolved_oxygen_mg_l: 8.0 - seasonal * 1.0,
            nitrates_mg_l: 3.0 + seasonal.abs() * 2.0,
            coliform_per_100ml: coliform,
            temperature_celsius: 15.0 + seasonal * 8.0,
            potability: potability.clamp(0.0, 1.0),
        });
    }

    readings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_water_horizons_ordered() {
        for i in 1..WATER_HORIZONS.len() {
            assert!(WATER_HORIZONS[i] > WATER_HORIZONS[i - 1]);
        }
    }

    #[test]
    fn test_water_horizons_labels_match() {
        assert_eq!(WATER_HORIZONS.len(), WATER_HORIZON_LABELS.len());
    }

    #[test]
    fn test_clean_reading_low_deviation() {
        let reading = WaterQualityReading::clean();
        assert!(reading.deviation_from_potable() < 0.01);
    }

    #[test]
    fn test_contaminated_reading_high_deviation() {
        let reading = WaterQualityReading::contaminated();
        assert!(reading.deviation_from_potable() > 1.0);
    }

    #[test]
    fn test_encoder_dimension() {
        let encoder = WaterHdcEncoder::new();
        let reading = WaterQualityReading::clean();
        let hv = encoder.encode(&reading);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encoder_similar_readings() {
        let encoder = WaterHdcEncoder::new();
        let r1 = WaterQualityReading::clean();
        let mut r2 = WaterQualityReading::clean();
        r2.ph = 7.3; // tiny change
        let hv1 = encoder.encode(&r1);
        let hv2 = encoder.encode(&r2);
        assert!(
            hv1.similarity(&hv2) > 0.9,
            "Similar readings should produce similar HVs"
        );
    }

    #[test]
    fn test_encoder_different_readings() {
        let encoder = WaterHdcEncoder::new();
        let hv_clean = encoder.encode(&WaterQualityReading::clean());
        let hv_dirty = encoder.encode(&WaterQualityReading::contaminated());
        assert!(
            hv_clean.similarity(&hv_dirty) < 0.9,
            "Clean vs contaminated should differ"
        );
    }

    #[test]
    fn test_predictor_dimension() {
        let predictor = HydrologicalPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let predicted = predictor.predict_at_horizon(&input, 3600.0);
        assert_eq!(predicted.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_predictor_all_horizons_count() {
        let predictor = HydrologicalPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let preds = predictor.predict_all_horizons(&input);
        assert_eq!(preds.len(), WATER_HORIZONS.len());
    }

    #[test]
    fn test_o1_property_water() {
        let predictor = HydrologicalPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);

        let start_1h = std::time::Instant::now();
        for _ in 0..100 {
            let _ = predictor.predict_at_horizon(&input, 3600.0);
        }
        let time_1h = start_1h.elapsed();

        let start_1y = std::time::Instant::now();
        for _ in 0..100 {
            let _ = predictor.predict_at_horizon(&input, 31_536_000.0);
        }
        let time_1y = start_1y.elapsed();

        let ratio = time_1y.as_nanos() as f64 / time_1h.as_nanos().max(1) as f64;
        assert!(
            ratio < 5.0 && ratio > 0.2,
            "O(1) property: 1h took {:?}, 1y took {:?}, ratio={}",
            time_1h,
            time_1y,
            ratio
        );
    }

    #[test]
    fn test_fep_action_clean_water() {
        let agent = WaterFepAgent::new();
        let reading = WaterQualityReading::clean();
        assert_eq!(agent.select_action(&reading), WaterFepAction::Maintain);
    }

    #[test]
    fn test_fep_action_contaminated() {
        let agent = WaterFepAgent::new();
        let reading = WaterQualityReading::contaminated();
        assert_eq!(
            agent.select_action(&reading),
            WaterFepAction::EmergencyShutoff
        );
    }

    #[test]
    fn test_fep_action_high_turbidity() {
        let agent = WaterFepAgent::new();
        let mut reading = WaterQualityReading::clean();
        reading.turbidity_ntu = 8.0;
        assert_eq!(
            agent.select_action(&reading),
            WaterFepAction::ActivateFilters
        );
    }

    #[test]
    fn test_fep_action_high_coliform() {
        let agent = WaterFepAgent::new();
        let mut reading = WaterQualityReading::clean();
        reading.coliform_per_100ml = 8.0;
        assert_eq!(agent.select_action(&reading), WaterFepAction::Chlorinate);
    }

    #[test]
    fn test_fep_action_high_nitrates() {
        let agent = WaterFepAgent::new();
        let mut reading = WaterQualityReading::clean();
        reading.nitrates_mg_l = 15.0;
        assert_eq!(
            agent.select_action(&reading),
            WaterFepAction::ReduceExtraction
        );
    }

    #[test]
    fn test_fep_surprise_clean() {
        let agent = WaterFepAgent::new();
        let reading = WaterQualityReading::clean();
        assert!(!agent.is_surprised(&reading));
    }

    #[test]
    fn test_fep_surprise_contaminated() {
        let agent = WaterFepAgent::new();
        let reading = WaterQualityReading::contaminated();
        assert!(agent.is_surprised(&reading));
    }

    #[test]
    fn test_action_index_roundtrip() {
        for action in &WaterFepAction::ALL {
            let idx = action.to_index();
            let recovered = WaterFepAction::from_index(idx);
            assert_eq!(*action, recovered);
        }
    }

    #[test]
    fn test_synthetic_scenario_length() {
        let scenario = synthetic_water_scenario(100, None);
        assert_eq!(scenario.len(), 100);
    }

    #[test]
    fn test_synthetic_scenario_contamination() {
        let scenario = synthetic_water_scenario(100, Some(50));
        assert!(scenario[50].coliform_per_100ml > 10.0);
        assert!(scenario[50].potability < 0.2);
        assert!(scenario[0].coliform_per_100ml < 5.0);
    }

    // ── Track B: failure-path tests ──────────────────────────────────────

    #[test]
    fn test_from_index_out_of_bounds_returns_shutoff() {
        assert_eq!(
            WaterFepAction::from_index(6),
            WaterFepAction::EmergencyShutoff
        );
        assert_eq!(
            WaterFepAction::from_index(100),
            WaterFepAction::EmergencyShutoff
        );
    }

    #[test]
    fn test_free_energy_finite_for_zero_hv() {
        let agent = WaterFepAgent::new();
        // Even an extreme reading should produce finite free energy
        let mut extreme = WaterQualityReading::clean();
        extreme.ph = 0.0;
        extreme.tds_mg_l = 0.0;
        extreme.dissolved_oxygen_mg_l = 0.0;
        let fe = agent.compute_free_energy(&extreme);
        assert!(fe.is_finite(), "Free energy should be finite, got {}", fe);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_predictor_rejects_nan_horizon() {
        let predictor = HydrologicalPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        predictor.predict_at_horizon(&input, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_predictor_rejects_zero_horizon() {
        let predictor = HydrologicalPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        predictor.predict_at_horizon(&input, 0.0);
    }

    #[test]
    fn test_deviation_is_zero_for_clean() {
        let reading = WaterQualityReading::clean();
        assert!(reading.deviation_from_potable() < 1e-10);
    }

    #[test]
    fn test_encode_window_empty_returns_zero() {
        let encoder = WaterHdcEncoder::new();
        let hv = encoder.encode_window(&[]);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|&v| v == 0.0));
    }
}
