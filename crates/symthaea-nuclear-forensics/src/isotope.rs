//! Isotopic signature definitions with 5 reference materials.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NuclearSource {
    NaturalUranium,
    LowEnrichedUranium,
    HighlyEnrichedUranium,
    MixedOxide,
    SpentFuel,
}

/// Isotopic signature with 9 ratio dimensions characterizing nuclear material.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotopicSignature {
    pub name: String,
    pub source: NuclearSource,
    pub u235_u238: f32,
    pub u234_u238: f32,
    pub pu239_pu240: f32,
    pub pu241_pu239: f32,
    pub cs137_activity: f32,
    pub sr90_activity: f32,
    pub am241_activity: f32,
    pub np237_activity: f32,
    pub burnup_gwd_t: f32,
}

impl IsotopicSignature {
    pub fn natural_uranium() -> Self {
        Self {
            name: "Natural Uranium".into(),
            source: NuclearSource::NaturalUranium,
            u235_u238: 0.0072,
            u234_u238: 0.000055,
            pu239_pu240: 0.0,
            pu241_pu239: 0.0,
            cs137_activity: 0.0,
            sr90_activity: 0.0,
            am241_activity: 0.0,
            np237_activity: 0.0,
            burnup_gwd_t: 0.0,
        }
    }

    pub fn low_enriched_uranium() -> Self {
        Self {
            name: "LEU (3.5%)".into(),
            source: NuclearSource::LowEnrichedUranium,
            u235_u238: 0.035,
            u234_u238: 0.00035,
            pu239_pu240: 0.0,
            pu241_pu239: 0.0,
            cs137_activity: 0.0,
            sr90_activity: 0.0,
            am241_activity: 0.0,
            np237_activity: 0.0,
            burnup_gwd_t: 0.0,
        }
    }

    pub fn highly_enriched_uranium() -> Self {
        Self {
            name: "HEU (93%)".into(),
            source: NuclearSource::HighlyEnrichedUranium,
            u235_u238: 0.93,
            u234_u238: 0.01,
            pu239_pu240: 0.0,
            pu241_pu239: 0.0,
            cs137_activity: 0.0,
            sr90_activity: 0.0,
            am241_activity: 0.0,
            np237_activity: 0.0,
            burnup_gwd_t: 0.0,
        }
    }

    pub fn mixed_oxide() -> Self {
        Self {
            name: "MOX Fuel".into(),
            source: NuclearSource::MixedOxide,
            u235_u238: 0.0072,
            u234_u238: 0.000055,
            pu239_pu240: 1.8,
            pu241_pu239: 0.15,
            cs137_activity: 0.1,
            sr90_activity: 0.08,
            am241_activity: 0.05,
            np237_activity: 0.02,
            burnup_gwd_t: 5.0,
        }
    }

    pub fn spent_fuel() -> Self {
        Self {
            name: "Spent Fuel (45 GWd/t)".into(),
            source: NuclearSource::SpentFuel,
            u235_u238: 0.008,
            u234_u238: 0.00008,
            pu239_pu240: 1.5,
            pu241_pu239: 0.20,
            cs137_activity: 0.85,
            sr90_activity: 0.72,
            am241_activity: 0.15,
            np237_activity: 0.08,
            burnup_gwd_t: 45.0,
        }
    }

    pub fn references() -> Vec<Self> {
        vec![
            Self::natural_uranium(),
            Self::low_enriched_uranium(),
            Self::highly_enriched_uranium(),
            Self::mixed_oxide(),
            Self::spent_fuel(),
        ]
    }

    /// Create a perturbed version (simulating measurement noise).
    pub fn perturbed(&self, noise_fraction: f32, seed: u64) -> Self {
        let mut state = seed;
        let mut next = move || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = ((state >> 33) ^ state) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut perturb = |v: f32| -> f32 { (v * (1.0 + noise_fraction * next())).max(0.0) };
        Self {
            name: format!("{} (perturbed)", self.name),
            source: self.source,
            u235_u238: perturb(self.u235_u238),
            u234_u238: perturb(self.u234_u238),
            pu239_pu240: perturb(self.pu239_pu240),
            pu241_pu239: perturb(self.pu241_pu239),
            cs137_activity: perturb(self.cs137_activity),
            sr90_activity: perturb(self.sr90_activity),
            am241_activity: perturb(self.am241_activity),
            np237_activity: perturb(self.np237_activity),
            burnup_gwd_t: perturb(self.burnup_gwd_t),
        }
    }

    /// Return all 9 ratio values as an array for encoding.
    pub fn values(&self) -> [f32; 9] {
        [
            self.u235_u238,
            self.u234_u238,
            self.pu239_pu240,
            self.pu241_pu239,
            self.cs137_activity,
            self.sr90_activity,
            self.am241_activity,
            self.np237_activity,
            self.burnup_gwd_t,
        ]
    }

    /// Normalize values to [0, 1] using domain-appropriate ranges.
    pub fn normalized_values(&self) -> [f32; 9] {
        [
            self.u235_u238.min(1.0),
            (self.u234_u238 * 100.0).min(1.0),
            (self.pu239_pu240 / 2.0).min(1.0),
            (self.pu241_pu239 / 0.3).min(1.0),
            self.cs137_activity.min(1.0),
            self.sr90_activity.min(1.0),
            self.am241_activity.min(1.0),
            self.np237_activity.min(1.0),
            (self.burnup_gwd_t / 60.0).min(1.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_references_count() { assert_eq!(IsotopicSignature::references().len(), 5); }
    #[test] fn test_heu_high_enrichment() { assert!(IsotopicSignature::highly_enriched_uranium().u235_u238 > 0.9); }
    #[test] fn test_natural_no_fission_products() { let n = IsotopicSignature::natural_uranium(); assert_eq!(n.cs137_activity, 0.0); }
    #[test] fn test_spent_fuel_has_fission_products() { let s = IsotopicSignature::spent_fuel(); assert!(s.cs137_activity > 0.5); }

    #[test]
    fn test_perturbed_preserves_source() {
        let p = IsotopicSignature::highly_enriched_uranium().perturbed(0.05, 42);
        assert_eq!(p.source, NuclearSource::HighlyEnrichedUranium);
    }

    #[test]
    fn test_perturbed_small_deviation() {
        let orig = IsotopicSignature::highly_enriched_uranium();
        let p = orig.perturbed(0.05, 42);
        assert!((p.u235_u238 - orig.u235_u238).abs() < orig.u235_u238 * 0.1);
    }

    #[test]
    fn test_normalized_values_bounded() {
        for sig in IsotopicSignature::references() {
            for v in sig.normalized_values() {
                assert!(v >= 0.0 && v <= 1.0, "Normalized value {} out of range for {}", v, sig.name);
            }
        }
    }
}
