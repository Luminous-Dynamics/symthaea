//! Isotope decay backdating using O(1) CfC temporal jumps with classical corrections.

use crate::encoder::IsotopicHdcEncoder;
use crate::isotope::IsotopicSignature;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Backdating horizons in seconds: 1 day, 1 month, 1 year, 10 years, 50 years.
pub const DECAY_HORIZONS: &[f32] = &[86_400.0, 2_592_000.0, 31_536_000.0, 315_360_000.0, 1_576_800_000.0];
pub const DECAY_HORIZON_LABELS: &[&str] = &["1 day", "1 month", "1 year", "10 years", "50 years"];

/// Half-lives in seconds for key isotopes.
const PU241_HALF_LIFE: f64 = 4.544e8;   // 14.4 years
const CS137_HALF_LIFE: f64 = 9.467e8;   // 30.0 years
const SR90_HALF_LIFE:  f64 = 9.119e8;   // 28.9 years

#[derive(Debug, Clone)]
pub struct BackdatedResult {
    pub horizon_seconds: f32,
    pub horizon_label: String,
    pub backdated_state: ContinuousHV,
    pub state_drift: f32,
}

#[derive(Debug, Clone)]
pub struct AgeEstimate {
    pub estimated_age_seconds: f32,
    pub confidence: f32,
}

pub struct IsotopeDecayModel {
    neuron: HdcLtcUnifiedNeuron,
    encoder: IsotopicHdcEncoder,
}

impl IsotopeDecayModel {
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 31_536_000.0, // 1 year base timescale
            backbone_tau: 0.1,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xDEC_A700),
            encoder: IsotopicHdcEncoder::new(),
        }
    }

    /// O(1) backdate: evolve CfC neuron backwards in time.
    ///
    /// # Panics
    ///
    /// Panics if `horizon_seconds` is not finite or is non-positive.
    pub fn backdate(&self, sig: &IsotopicSignature, horizon_seconds: f32) -> BackdatedResult {
        assert!(
            horizon_seconds.is_finite() && horizon_seconds > 0.0,
            "horizon_seconds must be finite and positive, got {}",
            horizon_seconds
        );
        let current_hv = self.encoder.encode(sig);
        let mut neuron_copy = self.neuron.clone();
        // Evolve with negative time for backdating
        neuron_copy.evolve_closed_form(-horizon_seconds, &current_hv);
        let backdated = neuron_copy.state().clone();
        let state_drift = 1.0 - current_hv.similarity(&backdated);
        let label = DECAY_HORIZONS.iter().position(|&h| (h - horizon_seconds).abs() < 1.0)
            .map(|i| DECAY_HORIZON_LABELS[i].to_string())
            .unwrap_or_else(|| format!("{:.0}s", horizon_seconds));
        BackdatedResult { horizon_seconds, horizon_label: label, backdated_state: backdated, state_drift }
    }

    /// Backdate at all standard horizons.
    pub fn backdate_all_horizons(&self, sig: &IsotopicSignature) -> Vec<BackdatedResult> {
        DECAY_HORIZONS.iter().map(|&h| self.backdate(sig, h)).collect()
    }

    /// Estimate age by classical decay of Pu-241, Cs-137, Sr-90.
    pub fn estimate_age(&self, sig: &IsotopicSignature) -> AgeEstimate {
        let mut estimates = Vec::new();

        // Pu-241 decay (if Pu present and ratio is valid for ln())
        if sig.pu241_pu239 > 0.01 {
            let ratio = sig.pu241_pu239 as f64;
            let ln_arg = ratio / 0.25;
            if ln_arg > 0.0 {
                let age = -PU241_HALF_LIFE * ln_arg.ln() / std::f64::consts::LN_2;
                if age.is_finite() && age > 0.0 && age < 1e10 {
                    estimates.push(age);
                }
            }
        }

        // Cs-137/Sr-90 ratio (if both present and ratio is valid for ln())
        if sig.cs137_activity > 0.01 && sig.sr90_activity > 0.01 {
            let ratio = (sig.cs137_activity / sig.sr90_activity) as f64;
            if ratio > 0.0 {
                let half_life_diff = 1.0 / CS137_HALF_LIFE - 1.0 / SR90_HALF_LIFE;
                if half_life_diff.abs() > 1e-20 {
                    let age = (ratio.ln() / half_life_diff / std::f64::consts::LN_2).abs();
                    if age.is_finite() && age < 1e10 {
                        estimates.push(age);
                    }
                }
            }
        }

        if estimates.is_empty() {
            AgeEstimate { estimated_age_seconds: 0.0, confidence: 0.0 }
        } else {
            let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
            let confidence = if estimates.len() > 1 {
                let variance = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / estimates.len() as f64;
                (1.0 / (1.0 + (variance.sqrt() / mean.max(1.0)))) as f32
            } else {
                0.5
            };
            AgeEstimate { estimated_age_seconds: mean as f32, confidence }
        }
    }
}

impl Default for IsotopeDecayModel { fn default() -> Self { Self::new() } }

impl symthaea_core::temporal::TemporalPredictor for IsotopeDecayModel {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(horizon_seconds, current_state);
        neuron_copy.state().clone()
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.neuron.evolve_closed_form(dt_seconds, state);
    }

    fn domain(&self) -> &'static str {
        "nuclear-forensics"
    }

    fn tau_base(&self) -> f32 {
        31_536_000.0 // 1 year (isotope decay timescale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_decay_horizons_ordered() { for i in 1..DECAY_HORIZONS.len() { assert!(DECAY_HORIZONS[i] > DECAY_HORIZONS[i - 1]); } }

    #[test] fn test_pu241_half_life() { assert!(PU241_HALF_LIFE > 4e8 && PU241_HALF_LIFE < 5e8); }
    #[test] fn test_cs137_half_life() { assert!(CS137_HALF_LIFE > 9e8 && CS137_HALF_LIFE < 1e9); }
    #[test] fn test_sr90_half_life() { assert!(SR90_HALF_LIFE > 8e8 && SR90_HALF_LIFE < 1e9); }

    #[test]
    fn test_backdate_dimension() {
        let r = IsotopeDecayModel::new().backdate(&IsotopicSignature::spent_fuel(), 86_400.0);
        assert_eq!(r.backdated_state.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_backdate_all_horizons_count() {
        assert_eq!(IsotopeDecayModel::new().backdate_all_horizons(&IsotopicSignature::spent_fuel()).len(), DECAY_HORIZONS.len());
    }

    #[test]
    fn test_estimate_age_returns_valid() {
        let e = IsotopeDecayModel::new().estimate_age(&IsotopicSignature::spent_fuel());
        assert!(e.estimated_age_seconds >= 0.0);
        assert!(e.confidence >= 0.0 && e.confidence <= 1.0);
    }

    #[test]
    fn test_o1_property_decay() {
        let m = IsotopeDecayModel::new();
        let s = IsotopicSignature::spent_fuel();
        let t1 = std::time::Instant::now();
        for _ in 0..100 { let _ = m.backdate(&s, 86_400.0); }
        let d1 = t1.elapsed();
        let t2 = std::time::Instant::now();
        for _ in 0..100 { let _ = m.backdate(&s, 1_576_800_000.0); }
        let d2 = t2.elapsed();
        let ratio = d2.as_nanos() as f64 / d1.as_nanos().max(1) as f64;
        assert!(ratio < 5.0 && ratio > 0.2, "O(1): 1d={:?}, 50y={:?}, ratio={}", d1, d2, ratio);
    }

    // ── Track B: failure-path tests ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_backdate_rejects_nan() {
        IsotopeDecayModel::new().backdate(&IsotopicSignature::spent_fuel(), f32::NAN);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_backdate_rejects_zero() {
        IsotopeDecayModel::new().backdate(&IsotopicSignature::spent_fuel(), 0.0);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_backdate_rejects_negative() {
        IsotopeDecayModel::new().backdate(&IsotopicSignature::spent_fuel(), -1.0);
    }

    #[test]
    fn test_estimate_age_zero_ratios() {
        // All isotope ratios near zero → should return age 0 with confidence 0
        let mut sig = IsotopicSignature::natural_uranium();
        sig.pu241_pu239 = 0.0;
        sig.cs137_activity = 0.0;
        sig.sr90_activity = 0.0;
        let age = IsotopeDecayModel::new().estimate_age(&sig);
        assert_eq!(age.estimated_age_seconds, 0.0);
        assert_eq!(age.confidence, 0.0);
    }

    #[test]
    fn test_estimate_age_all_references_finite() {
        let model = IsotopeDecayModel::new();
        for sig in &IsotopicSignature::references() {
            let age = model.estimate_age(sig);
            assert!(age.estimated_age_seconds.is_finite(), "NaN age for {}", sig.name);
            assert!(age.confidence.is_finite(), "NaN confidence for {}", sig.name);
            assert!(age.confidence >= 0.0 && age.confidence <= 1.0);
        }
    }
}
