// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Material-state drift research using O(1) CfC temporal jumps.
//!
//! The model below evolves an HDC/CfC representation of a material and reports
//! representation-space similarity. It is a research heuristic: it is **not**
//! a calibrated residual-strength, fatigue-life, damage, corrosion, creep, or
//! service-life model. Physical engineering claims require a separately
//! evidenced model and validation path.

use crate::encoder::MaterialHdcEncoder;
use crate::properties::MaterialProperty;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Claim ceiling for [`MaterialAgingModel`] outputs.
pub const MATERIAL_AGING_CLAIM_CLASS: &str = "research-representation-drift-only";

/// Aging research horizons in seconds: 1 day, 1 month, 1 year, 10 years, 50 years.
pub const AGING_HORIZONS: &[f32] = &[
    86_400.0,
    2_592_000.0,
    31_536_000.0,
    315_360_000.0,
    1_576_800_000.0,
];
/// Human-readable labels matching [`AGING_HORIZONS`] in order.
pub const AGING_HORIZON_LABELS: &[&str] = &["1 day", "1 month", "1 year", "10 years", "50 years"];

/// Result of an O(1) material-representation prediction at a single horizon.
///
/// This record deliberately carries no physically validated damage or lifetime
/// estimate. `state_similarity` is similarity in the learned HDC/CfC state
/// representation, not structural capacity.
#[derive(Debug, Clone)]
pub struct AgingPrediction {
    /// Prediction horizon in seconds.
    pub horizon_seconds: f32,
    /// Human-readable horizon label (e.g., "1 year").
    pub horizon_label: String,
    /// CfC-predicted material representation at the target horizon.
    pub predicted_state: ContinuousHV,
    /// Cosine similarity between current and predicted representation.
    ///
    /// This is a research drift signal only. It has no calibrated mapping to
    /// residual strength, fatigue life, damage fraction, or safe operation.
    pub state_similarity: f32,
    /// Compatibility-only neutral sentinel retained while downstream callers
    /// migrate away from the former physical-sounding API.
    ///
    /// This value is intentionally fixed to `1.0` and MUST NOT be interpreted
    /// as measured/predicted residual strength. It exists only so this narrow
    /// quarantine can stop the previous `state_similarity -> strength` semantic
    /// laundering without forcing an unrelated broad API migration in the same
    /// evidence subject. #4862 tracks removal/replacement with typed evidence.
    pub remaining_strength: f32,
}

impl AgingPrediction {
    /// Return the research-only representation-drift signal explicitly.
    pub fn representation_similarity(&self) -> f32 {
        self.state_similarity
    }

    /// Return the claim ceiling for this prediction.
    pub const fn claim_class(&self) -> &'static str {
        MATERIAL_AGING_CLAIM_CLASS
    }
}

/// O(1) material representation-drift predictor using CfC closed-form evolution.
///
/// Encodes a material's current properties into a 16,384D hypervector, then
/// uses [`HdcLtcUnifiedNeuron::evolve_closed_form`] to jump to a future
/// representation in constant time. Prediction cost is identical for 1 day
/// and 50 years.
///
/// This type is not a physical degradation model. In particular, no result
/// from this type alone may be used as residual structural strength, a fatigue
/// or service-life estimate, a geometry compensation factor, or safety/operation
/// authority.
pub struct MaterialAgingModel {
    neuron: HdcLtcUnifiedNeuron,
    encoder: MaterialHdcEncoder,
}

impl MaterialAgingModel {
    /// Create a new representation-drift model with a 1-day base timescale.
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
            // Compatibility quarantine: never derive a physical-looking
            // strength fraction from HDC representation similarity.
            remaining_strength: 1.0,
        }
    }

    /// Predict representation drift at all 5 standard horizons (1 day to 50 years).
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
        86_400.0 // 1 day (materials representation timescale)
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
    fn test_remaining_strength_is_neutral_compatibility_sentinel() {
        for p in MaterialAgingModel::new().predict_all_horizons(&MaterialProperty::steel_a36()) {
            assert_eq!(p.remaining_strength, 1.0);
            assert_eq!(p.claim_class(), MATERIAL_AGING_CLAIM_CLASS);
        }
    }

    #[test]
    fn test_research_signal_is_state_similarity_only() {
        for p in MaterialAgingModel::new().predict_all_horizons(&MaterialProperty::steel_a36()) {
            assert!(p.state_similarity.is_finite());
            assert_eq!(p.representation_similarity(), p.state_similarity);
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
    fn test_all_preset_materials_have_finite_research_signal() {
        let model = MaterialAgingModel::new();
        for mat in MaterialProperty::presets() {
            let preds = model.predict_all_horizons(&mat);
            for p in &preds {
                assert!(
                    p.state_similarity.is_finite(),
                    "NaN state_similarity for {}",
                    mat.name
                );
                assert_eq!(
                    p.remaining_strength, 1.0,
                    "compatibility sentinel drifted for {}",
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
