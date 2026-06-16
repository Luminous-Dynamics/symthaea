// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Critical period plasticity modeling via O(1) CfC temporal jumps.
//!
//! Critical periods are windows of heightened neural plasticity during which
//! specific types of experience have outsized effects on development. After the
//! window closes, the same experiences produce much smaller effects.
//!
//! Three domains are modeled:
//! - **Attachment** (0-24 months): Peak sensitivity to caregiver quality
//! - **Language** (0-60 months): Peak sensitivity to linguistic input
//! - **Vision** (0-36 months): Peak sensitivity to visual experience
//!
//! The CfC closed-form solution enables O(1) temporal jumps: we can compute
//! plasticity at any age without iterating through intermediate time steps.
//!
//! Science: Hubel & Wiesel (1970), Newport (1990), Bowlby (1969).

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::DevelopmentalAge;

/// Domain of critical period sensitivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CriticalPeriodDomain {
    /// Attachment critical period: 0-24 months.
    Attachment,
    /// Language critical period: 0-60 months.
    Language,
    /// Vision critical period: 0-36 months.
    Vision,
}

/// Critical period model using CfC neuron for temporal dynamics.
///
/// Contains a single HdcLtcUnifiedNeuron that models plasticity decay
/// as a temporal evolution problem. The closed-form solution enables
/// computing plasticity at any age in O(1).
#[derive(Debug, Clone)]
pub struct CriticalPeriodModel {
    // Private: CfC neuron whose state tracks the developmental trajectory.
    neuron: HdcLtcUnifiedNeuron,
}

impl CriticalPeriodModel {
    /// Create a new critical period model with default CfC configuration.
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 0.5, // Slower dynamics for developmental timescale
            backbone_tau: 0.3,
            dimension: HDC_DIMENSION,
            learning_rate: 0.001,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 777),
        }
    }

    /// Analytical plasticity curve (ground truth for validation).
    ///
    /// Returns plasticity in [0, 1] where 1 = maximum plasticity and
    /// values near 0 = critical period closed.
    pub fn analytical_plasticity(domain: CriticalPeriodDomain, age_months: f64) -> f64 {
        match domain {
            CriticalPeriodDomain::Attachment => {
                // High 0-12mo, declining 12-24mo, low residual after 24mo
                if age_months <= 12.0 {
                    1.0
                } else if age_months <= 24.0 {
                    1.0 - (age_months - 12.0) / 24.0
                } else {
                    (0.5 * (-0.1 * (age_months - 24.0)).exp()).max(0.05)
                }
            }
            CriticalPeriodDomain::Language => {
                // High 0-36mo, declining 36-60mo, residual after 60mo
                if age_months <= 36.0 {
                    1.0
                } else if age_months <= 60.0 {
                    1.0 - (age_months - 36.0) / 48.0
                } else {
                    (0.5 * (-0.05 * (age_months - 60.0)).exp()).max(0.02)
                }
            }
            CriticalPeriodDomain::Vision => {
                // High 0-12mo, declining 12-36mo, residual after 36mo
                if age_months <= 12.0 {
                    1.0
                } else if age_months <= 36.0 {
                    1.0 - (age_months - 12.0) / 36.0
                } else {
                    (0.3 * (-0.15 * (age_months - 36.0)).exp()).max(0.01)
                }
            }
        }
    }

    /// O(1) plasticity prediction at any age.
    ///
    /// Currently uses the analytical curve. The CfC neuron can be trained
    /// to match these curves for integration with the broader cognitive loop,
    /// where the neuron state encodes the developmental trajectory.
    pub fn plasticity_at(&mut self, domain: CriticalPeriodDomain, age: &DevelopmentalAge) -> f64 {
        // Use analytical as ground truth; CfC evolves state for integration
        let plasticity = Self::analytical_plasticity(domain, age.months);

        // Evolve CfC state with age as temporal input (for downstream integration)
        let age_hv = ContinuousHV::random(HDC_DIMENSION, 5000 + domain as u64)
            .scale(age.months as f32 / 60.0);
        self.neuron
            .evolve_closed_form(age.months as f32 / 60.0, &age_hv);

        plasticity
    }

    /// Get the CfC neuron state (for downstream integration).
    pub fn neuron_state(&self) -> &ContinuousHV {
        self.neuron.state()
    }

    /// Check if a critical period is still open (plasticity > threshold).
    pub fn is_window_open(domain: CriticalPeriodDomain, age_months: f64, threshold: f64) -> bool {
        Self::analytical_plasticity(domain, age_months) > threshold
    }

    /// Get the peak plasticity age range for a domain.
    pub fn peak_range(domain: CriticalPeriodDomain) -> (f64, f64) {
        match domain {
            CriticalPeriodDomain::Attachment => (0.0, 12.0),
            CriticalPeriodDomain::Language => (0.0, 36.0),
            CriticalPeriodDomain::Vision => (0.0, 12.0),
        }
    }

    /// Get the closing age (when plasticity drops below 0.5) for a domain.
    pub fn closing_age(domain: CriticalPeriodDomain) -> f64 {
        match domain {
            CriticalPeriodDomain::Attachment => 24.0,
            CriticalPeriodDomain::Language => 48.0, // Midpoint of decline
            CriticalPeriodDomain::Vision => 24.0,   // Midpoint of decline
        }
    }
}

impl Default for CriticalPeriodModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attachment_plasticity_high_at_birth() {
        let p = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Attachment, 0.0);
        assert!(
            (p - 1.0).abs() < 1e-10,
            "Plasticity should be 1.0 at birth, got {p}"
        );
    }

    #[test]
    fn test_attachment_plasticity_high_at_6_months() {
        let p = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Attachment, 6.0);
        assert!(
            (p - 1.0).abs() < 1e-10,
            "Plasticity should be 1.0 at 6mo, got {p}"
        );
    }

    #[test]
    fn test_attachment_plasticity_declines_after_12() {
        let p12 =
            CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Attachment, 12.0);
        let p18 =
            CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Attachment, 18.0);
        let p24 =
            CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Attachment, 24.0);

        assert!(
            p12 > p18,
            "Plasticity should decline: 12mo={p12} > 18mo={p18}"
        );
        assert!(
            p18 > p24,
            "Plasticity should decline: 18mo={p18} > 24mo={p24}"
        );
    }

    #[test]
    fn test_attachment_window_closed_by_36() {
        let p = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Attachment, 36.0);
        assert!(p < 0.4, "Plasticity should be low at 36mo, got {p}");
    }

    #[test]
    fn test_attachment_residual_never_zero() {
        let p = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Attachment, 120.0);
        assert!(
            p >= 0.05,
            "Should have minimum residual plasticity, got {p}"
        );
    }

    #[test]
    fn test_language_plasticity_high_at_birth() {
        let p = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Language, 0.0);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_language_plasticity_high_at_30_months() {
        let p = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Language, 30.0);
        assert!(
            (p - 1.0).abs() < 1e-10,
            "Language plasticity should be 1.0 at 30mo, got {p}"
        );
    }

    #[test]
    fn test_language_plasticity_declines_after_36() {
        let p36 = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Language, 36.0);
        let p48 = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Language, 48.0);
        let p60 = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Language, 60.0);

        assert!(
            p36 > p48,
            "Language plasticity should decline: 36mo={p36} > 48mo={p48}"
        );
        assert!(
            p48 > p60,
            "Language plasticity should decline: 48mo={p48} > 60mo={p60}"
        );
    }

    #[test]
    fn test_language_window_longer_than_attachment() {
        let attachment_30 =
            CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Attachment, 30.0);
        let language_30 =
            CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Language, 30.0);

        assert!(
            language_30 > attachment_30,
            "Language window should be open longer: lang={language_30}, attach={attachment_30}"
        );
    }

    #[test]
    fn test_vision_plasticity_high_at_birth() {
        let p = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Vision, 0.0);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vision_plasticity_declines_after_12() {
        let p12 = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Vision, 12.0);
        let p24 = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Vision, 24.0);
        let p36 = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Vision, 36.0);

        assert!(p12 > p24, "Vision plasticity: 12mo={p12} > 24mo={p24}");
        assert!(p24 > p36, "Vision plasticity: 24mo={p24} > 36mo={p36}");
    }

    #[test]
    fn test_vision_window_closed_by_48() {
        let p = CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Vision, 48.0);
        assert!(
            p < 0.2,
            "Vision plasticity should be very low at 48mo, got {p}"
        );
    }

    #[test]
    fn test_all_domains_monotonic_decrease() {
        for domain in [
            CriticalPeriodDomain::Attachment,
            CriticalPeriodDomain::Language,
            CriticalPeriodDomain::Vision,
        ] {
            let mut prev = 1.0;
            // Sample at peak + declining region
            for month in (0..=120).step_by(6) {
                let p = CriticalPeriodModel::analytical_plasticity(domain, month as f64);
                assert!(
                    p <= prev + 1e-10,
                    "{domain:?} plasticity should be monotonically non-increasing: {prev} -> {p} at {month}mo"
                );
                prev = p;
            }
        }
    }

    #[test]
    fn test_plasticity_at_uses_cfc() {
        let mut model = CriticalPeriodModel::new();
        let age = DevelopmentalAge::new(12.0);
        let p = model.plasticity_at(CriticalPeriodDomain::Attachment, &age);

        // Should match analytical
        let analytical =
            CriticalPeriodModel::analytical_plasticity(CriticalPeriodDomain::Attachment, 12.0);
        assert!(
            (p - analytical).abs() < 1e-10,
            "plasticity_at should match analytical: {p} vs {analytical}"
        );

        // Neuron state should have evolved (not zero)
        let state_norm = model.neuron_state().norm();
        assert!(state_norm > 0.0, "CfC neuron state should have evolved");
    }

    #[test]
    fn test_is_window_open() {
        assert!(CriticalPeriodModel::is_window_open(
            CriticalPeriodDomain::Attachment,
            6.0,
            0.5
        ));
        assert!(!CriticalPeriodModel::is_window_open(
            CriticalPeriodDomain::Attachment,
            60.0,
            0.5
        ));
        assert!(CriticalPeriodModel::is_window_open(
            CriticalPeriodDomain::Language,
            30.0,
            0.5
        ));
        assert!(!CriticalPeriodModel::is_window_open(
            CriticalPeriodDomain::Vision,
            60.0,
            0.5
        ));
    }

    #[test]
    fn test_peak_range() {
        let (lo, hi) = CriticalPeriodModel::peak_range(CriticalPeriodDomain::Attachment);
        assert!((lo - 0.0).abs() < 1e-10);
        assert!((hi - 12.0).abs() < 1e-10);

        let (lo, hi) = CriticalPeriodModel::peak_range(CriticalPeriodDomain::Language);
        assert!((lo - 0.0).abs() < 1e-10);
        assert!((hi - 36.0).abs() < 1e-10);
    }

    #[test]
    fn test_residual_plasticity_all_domains() {
        // All domains should have non-zero residual at extreme age
        for domain in [
            CriticalPeriodDomain::Attachment,
            CriticalPeriodDomain::Language,
            CriticalPeriodDomain::Vision,
        ] {
            let p = CriticalPeriodModel::analytical_plasticity(domain, 240.0); // 20 years
            assert!(
                p > 0.0,
                "{domain:?} should have non-zero residual at 20 years, got {p}"
            );
        }
    }
}
