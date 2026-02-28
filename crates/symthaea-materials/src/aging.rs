//! Material aging prediction using O(1) CfC temporal jumps.

use crate::encoder::MaterialHdcEncoder;
use crate::properties::MaterialProperty;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

pub const AGING_HORIZONS: &[f32] = &[86_400.0, 2_592_000.0, 31_536_000.0, 315_360_000.0, 1_576_800_000.0];
pub const AGING_HORIZON_LABELS: &[&str] = &["1 day", "1 month", "1 year", "10 years", "50 years"];

#[derive(Debug, Clone)]
pub struct AgingPrediction {
    pub horizon_seconds: f32,
    pub horizon_label: String,
    pub predicted_state: ContinuousHV,
    pub state_similarity: f32,
    pub remaining_strength: f32,
}

pub struct MaterialAgingModel {
    neuron: HdcLtcUnifiedNeuron,
    encoder: MaterialHdcEncoder,
}

impl MaterialAgingModel {
    pub fn new() -> Self {
        let config = UnifiedConfig { tau_base: 86_400.0, backbone_tau: 0.1, dimension: HDC_DIMENSION, ..UnifiedConfig::default() };
        Self { neuron: HdcLtcUnifiedNeuron::new(config, 0xA61_0E00), encoder: MaterialHdcEncoder::new() }
    }

    pub fn predict_at_horizon(&self, material: &MaterialProperty, horizon_seconds: f32) -> AgingPrediction {
        let current_hv = self.encoder.encode(material);
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(horizon_seconds, &current_hv);
        let predicted = neuron_copy.state().clone();
        let state_similarity = current_hv.similarity(&predicted);
        let label = AGING_HORIZONS.iter().position(|&h| (h - horizon_seconds).abs() < 1.0)
            .map(|i| AGING_HORIZON_LABELS[i].to_string()).unwrap_or_else(|| format!("{:.0}s", horizon_seconds));
        AgingPrediction { horizon_seconds, horizon_label: label, predicted_state: predicted, state_similarity, remaining_strength: state_similarity.max(0.0) }
    }

    pub fn predict_all_horizons(&self, material: &MaterialProperty) -> Vec<AgingPrediction> {
        AGING_HORIZONS.iter().map(|&h| self.predict_at_horizon(material, h)).collect()
    }
}

impl Default for MaterialAgingModel { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_aging_horizons_ordered() { for i in 1..AGING_HORIZONS.len() { assert!(AGING_HORIZONS[i] > AGING_HORIZONS[i - 1]); } }
    #[test] fn test_aging_horizons_labels_match() { assert_eq!(AGING_HORIZONS.len(), AGING_HORIZON_LABELS.len()); }
    #[test] fn test_predict_dimension() { assert_eq!(MaterialAgingModel::new().predict_at_horizon(&MaterialProperty::steel_a36(), 86_400.0).predicted_state.dim(), HDC_DIMENSION); }
    #[test] fn test_predict_all_horizons_count() { assert_eq!(MaterialAgingModel::new().predict_all_horizons(&MaterialProperty::steel_a36()).len(), AGING_HORIZONS.len()); }

    #[test]
    fn test_o1_property_aging() {
        let m = MaterialAgingModel::new();
        let s = MaterialProperty::steel_a36();
        let t1 = std::time::Instant::now(); for _ in 0..100 { let _ = m.predict_at_horizon(&s, 86_400.0); } let d1 = t1.elapsed();
        let t2 = std::time::Instant::now(); for _ in 0..100 { let _ = m.predict_at_horizon(&s, 1_576_800_000.0); } let d2 = t2.elapsed();
        let ratio = d2.as_nanos() as f64 / d1.as_nanos().max(1) as f64;
        assert!(ratio < 5.0 && ratio > 0.2, "O(1): 1d={:?}, 50y={:?}, ratio={}", d1, d2, ratio);
    }

    #[test]
    fn test_remaining_strength_bounded() {
        for p in MaterialAgingModel::new().predict_all_horizons(&MaterialProperty::steel_a36()) {
            assert!(p.remaining_strength >= 0.0 && p.remaining_strength <= 1.0);
        }
    }
}
