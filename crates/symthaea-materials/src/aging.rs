// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Material aging prediction using O(1) CfC temporal jumps.

use crate::encoder::MaterialHdcEncoder;
use crate::properties::MaterialProperty;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Aging prediction horizons in seconds: 1 day, 1 month, 1 year, 10 years, 50 years.
pub const AGING_HORIZONS: &[f32] = &[
    86_400.0,
    2_592_000.0,
    31_536_000.0,
    315_360_000.0,
    1_576_800_000.0,
];
/// Human-readable labels matching [`AGING_HORIZONS`] in order.
pub const AGING_HORIZON_LABELS: &[&str] = &["1 day", "1 month", "1 year", "10 years", "50 years"];

/// Result of an O(1) aging prediction at a single time horizon.
#[derive(Debug, Clone)]
pub struct AgingPrediction {
    /// Prediction horizon in seconds.
    pub horizon_seconds: f32,
    /// Human-readable horizon label (e.g., "1 year").
    pub horizon_label: String,
    /// CfC-predicted material state at the target horizon.
    pub predicted_state: ContinuousHV,
    /// Cosine similarity between current and predicted state (1.0 = unchanged).
    pub state_similarity: f32,
    /// Estimated remaining structural integrity (clamped to [0, 1]).
    pub remaining_strength: f32,
}

/// O(1) material aging predictor using CfC closed-form temporal evolution.
///
/// Encodes a material's current properties into a 16,384D hypervector,
/// then uses [`HdcLtcUnifiedNeuron::evolve_closed_form`] to jump to any
/// future time in constant time. Prediction cost is identical for 1 day
/// and 50 years.
pub struct MaterialAgingModel {
    neuron: HdcLtcUnifiedNeuron,
    encoder: MaterialHdcEncoder,
}

impl MaterialAgingModel {
    /// Create a new aging model with a 1-day base timescale.
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 86_400.0,
            backbone_tau: 0.1,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xA61_0E00),
            encoder: MaterialHdcEncoder::new(),
        }
    }

    /// # Panics
    ///
    /// Panics if `horizon_seconds` is not finite or is non-positive.
    pub fn predict_at_horizon(
        &self,
        material: &MaterialProperty,
        horizon_seconds: f32,
    ) -> AgingPrediction {
        assert!(
            horizon_seconds.is_finite() && horizon_seconds > 0.0,
            "horizon_seconds must be finite and positive, got {}",
            horizon_seconds
        );
        let current_hv = self.encoder.encode(material);
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(horizon_seconds, &current_hv);
        let predicted = neuron_copy.state().clone();
        let state_similarity = current_hv.similarity(&predicted);
        let label = AGING_HORIZONS
            .iter()
            .position(|&h| (h - horizon_seconds).abs() < 1.0)
            .map(|i| AGING_HORIZON_LABELS[i].to_string())
            .unwrap_or_else(|| format!("{:.0}s", horizon_seconds));
        AgingPrediction {
            horizon_seconds,
            horizon_label: label,
            predicted_state: predicted,
            state_similarity,
            remaining_strength: state_similarity.max(0.0),
        }
    }

    /// Predict aging at all 5 standard horizons (1 day to 50 years).
    pub fn predict_all_horizons(&self, material: &MaterialProperty) -> Vec<AgingPrediction> {
        AGING_HORIZONS
            .iter()
            .map(|&h| self.predict_at_horizon(material, h))
            .collect()
    }
}

impl Default for MaterialAgingModel {
    fn default() -> Self {
        Self::new()
    }
}

impl symthaea_core::temporal::TemporalPredictor for MaterialAgingModel {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(horizon_seconds, current_state);
        neuron_copy.state().clone()
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.neuron.evolve_closed_form(dt_seconds, state);
    }

    fn domain(&self) -> &'static str {
        "materials"
    }

    fn tau_base(&self) -> f32 {
        86_400.0 // 1 day (materials aging timescale)
    }

    fn default_horizons(&self) -> &'static [f32] {
        AGING_HORIZONS
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        AGING_HORIZON_LABELS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aging_horizons_ordered() {
        for i in 1..AGING_HORIZONS.len() {
            assert!(AGING_HORIZONS[i] > AGING_HORIZONS[i - 1]);
        }
    }
    #[test]
    fn test_aging_horizons_labels_match() {
        assert_eq!(AGING_HORIZONS.len(), AGING_HORIZON_LABELS.len());
    }
    #[test]
    fn test_predict_dimension() {
        assert_eq!(
            MaterialAgingModel::new()
                .predict_at_horizon(&MaterialProperty::steel_a36(), 86_400.0)
                .predicted_state
                .dim(),
            HDC_DIMENSION
        );
    }
    #[test]
    fn test_predict_all_horizons_count() {
        assert_eq!(
            MaterialAgingModel::new()
                .predict_all_horizons(&MaterialProperty::steel_a36())
                .len(),
            AGING_HORIZONS.len()
        );
    }

    #[test]
    fn test_o1_property_aging() {
        let m = MaterialAgingModel::new();
        let s = MaterialProperty::steel_a36();
        let t1 = std::time::Instant::now();
        for _ in 0..100 {
            let _ = m.predict_at_horizon(&s, 86_400.0);
        }
        let d1 = t1.elapsed();
        let t2 = std::time::Instant::now();
        for _ in 0..100 {
            let _ = m.predict_at_horizon(&s, 1_576_800_000.0);
        }
        let d2 = t2.elapsed();
        let ratio = d2.as_nanos() as f64 / d1.as_nanos().max(1) as f64;
        assert!(
            ratio < 5.0 && ratio > 0.2,
            "O(1): 1d={:?}, 50y={:?}, ratio={}",
            d1,
            d2,
            ratio
        );
    }

    #[test]
    fn test_remaining_strength_bounded() {
        for p in MaterialAgingModel::new().predict_all_horizons(&MaterialProperty::steel_a36()) {
            assert!(p.remaining_strength >= 0.0 && p.remaining_strength <= 1.0);
        }
    }

    // ── Track B: failure-path tests ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_predict_rejects_nan() {
        MaterialAgingModel::new().predict_at_horizon(&MaterialProperty::steel_a36(), f32::NAN);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_predict_rejects_zero() {
        MaterialAgingModel::new().predict_at_horizon(&MaterialProperty::steel_a36(), 0.0);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_predict_rejects_negative() {
        MaterialAgingModel::new().predict_at_horizon(&MaterialProperty::steel_a36(), -86_400.0);
    }

    #[test]
    fn test_all_preset_materials_age_valid() {
        let model = MaterialAgingModel::new();
        for mat in MaterialProperty::presets() {
            let preds = model.predict_all_horizons(&mat);
            for p in &preds {
                assert!(
                    p.state_similarity.is_finite(),
                    "NaN state_similarity for {}",
                    mat.name
                );
                assert!(
                    p.remaining_strength.is_finite(),
                    "NaN remaining_strength for {}",
                    mat.name
                );
            }
        }
    }

    // ── Track C: trait coverage ───────────────────────────────────────────

    #[test]
    fn test_observe_changes_prediction() {
        use symthaea_core::temporal::TemporalPredictor;
        let mut m = MaterialAgingModel::new();
        let seed = ContinuousHV::random(HDC_DIMENSION, 0xBEEF);
        let before = m.predict_at(&seed, 86_400.0);
        m.observe(&seed, 86_400.0);
        let after = m.predict_at(&seed, 86_400.0);
        let sim = before.similarity(&after);
        assert!(
            sim < 1.0,
            "observe() should change predictions, sim={}",
            sim
        );
    }

    #[test]
    fn test_default_horizons_and_labels() {
        use symthaea_core::temporal::TemporalPredictor;
        let m = MaterialAgingModel::new();
        assert_eq!(m.default_horizons(), AGING_HORIZONS);
        assert_eq!(m.horizon_labels(), AGING_HORIZON_LABELS);
        assert_eq!(m.default_horizons().len(), m.horizon_labels().len());
    }
}
