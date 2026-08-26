// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative olfactory-gustatory flavor binding.
//!
//! A flavor percept is created only when one trustworthy olfactory percept and
//! one trustworthy gustatory percept are close enough in time to plausibly
//! belong to the same sampling episode. The original evidence from both senses
//! remains attached to the derived flavor representation.

use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};

use crate::{ChemicalModality, ChemicalPercept};

#[derive(Debug, Clone, PartialEq)]
pub struct FlavorBindingConfig {
    /// Maximum allowed timestamp skew between smell and taste observations.
    pub max_skew_us: u64,
    /// Minimum confidence required from each contributing modality.
    pub min_confidence: f32,
}

impl Default for FlavorBindingConfig {
    fn default() -> Self {
        Self {
            max_skew_us: 2_000_000,
            min_confidence: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlavorConfigError {
    InvalidMinimumConfidence(f32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlavorBindingError {
    DuplicateModality(ChemicalModality),
    TemporalSkew {
        skew_us: u64,
        max_skew_us: u64,
    },
    LowConfidence {
        modality: ChemicalModality,
        confidence: f32,
        minimum: f32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlavorPercept {
    pub olfactory: ChemicalPercept,
    pub gustatory: ChemicalPercept,
    pub vector: ContinuousHV,
    /// Conservative joint confidence: the weaker modality limits the pair.
    pub confidence: f32,
    pub temporal_skew_us: u64,
}

#[derive(Debug, Clone)]
pub struct FlavorBinder {
    config: FlavorBindingConfig,
    olfactory_role: ContinuousHV,
    gustatory_role: ContinuousHV,
}

impl FlavorBinder {
    pub fn new(config: FlavorBindingConfig) -> Result<Self, FlavorConfigError> {
        if !config.min_confidence.is_finite() || !(0.0..=1.0).contains(&config.min_confidence) {
            return Err(FlavorConfigError::InvalidMinimumConfidence(
                config.min_confidence,
            ));
        }
        Ok(Self {
            config,
            olfactory_role: ContinuousHV::random(HDC_DIMENSION, 0xF1A0_0000_0000_0001),
            gustatory_role: ContinuousHV::random(HDC_DIMENSION, 0xF1A0_0000_0000_0002),
        })
    }

    pub fn config(&self) -> &FlavorBindingConfig {
        &self.config
    }

    /// Bind one olfactory and one gustatory percept into a flavor percept.
    ///
    /// Argument order is intentionally irrelevant. Validation completes before
    /// constructing the derived representation.
    pub fn bind(
        &self,
        first: &ChemicalPercept,
        second: &ChemicalPercept,
    ) -> Result<FlavorPercept, FlavorBindingError> {
        let (olfactory, gustatory) = match (first.evidence.modality, second.evidence.modality) {
            (ChemicalModality::Olfactory, ChemicalModality::Gustatory) => (first, second),
            (ChemicalModality::Gustatory, ChemicalModality::Olfactory) => (second, first),
            (ChemicalModality::Olfactory, ChemicalModality::Olfactory) => {
                return Err(FlavorBindingError::DuplicateModality(
                    ChemicalModality::Olfactory,
                ));
            }
            (ChemicalModality::Gustatory, ChemicalModality::Gustatory) => {
                return Err(FlavorBindingError::DuplicateModality(
                    ChemicalModality::Gustatory,
                ));
            }
        };

        for percept in [olfactory, gustatory] {
            if percept.confidence() < self.config.min_confidence {
                return Err(FlavorBindingError::LowConfidence {
                    modality: percept.evidence.modality,
                    confidence: percept.confidence(),
                    minimum: self.config.min_confidence,
                });
            }
        }

        let temporal_skew_us = olfactory.timestamp_us().abs_diff(gustatory.timestamp_us());
        if temporal_skew_us > self.config.max_skew_us {
            return Err(FlavorBindingError::TemporalSkew {
                skew_us: temporal_skew_us,
                max_skew_us: self.config.max_skew_us,
            });
        }

        let odor_component = self.olfactory_role.bind(&olfactory.fingerprint.vector);
        let taste_component = self.gustatory_role.bind(&gustatory.fingerprint.vector);
        let mut vector = ContinuousHV::bundle(&[&odor_component, &taste_component]);
        vector.l2_normalize();

        Ok(FlavorPercept {
            olfactory: olfactory.clone(),
            gustatory: gustatory.clone(),
            vector,
            confidence: olfactory.confidence().min(gustatory.confidence()),
            temporal_skew_us,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalFingerprint, ChemicalObservation, EnvironmentReading};

    fn percept(
        modality: ChemicalModality,
        timestamp_us: u64,
        seed: u64,
        confidence: f32,
    ) -> ChemicalPercept {
        ChemicalPercept {
            evidence: ChemicalObservation {
                timestamp_us,
                modality,
                source: format!("source-{seed}"),
                channels: vec![],
                environment: EnvironmentReading::default(),
            },
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, seed),
                confidence,
                used_channels: 1,
                ignored_channels: 0,
            },
        }
    }

    #[test]
    fn binding_is_argument_order_invariant() {
        let binder = FlavorBinder::new(FlavorBindingConfig::default()).unwrap();
        let odor = percept(ChemicalModality::Olfactory, 1_000_000, 1, 0.9);
        let taste = percept(ChemicalModality::Gustatory, 1_500_000, 2, 0.8);
        let forward = binder.bind(&odor, &taste).unwrap();
        let reverse = binder.bind(&taste, &odor).unwrap();
        assert_eq!(forward.vector, reverse.vector);
        assert_eq!(forward.olfactory.evidence, odor.evidence);
        assert_eq!(forward.gustatory.evidence, taste.evidence);
        assert!((forward.confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn same_modality_pair_is_rejected() {
        let binder = FlavorBinder::new(FlavorBindingConfig::default()).unwrap();
        let a = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let b = percept(ChemicalModality::Olfactory, 2, 2, 0.9);
        assert!(matches!(
            binder.bind(&a, &b),
            Err(FlavorBindingError::DuplicateModality(
                ChemicalModality::Olfactory
            ))
        ));
    }

    #[test]
    fn temporally_unrelated_samples_are_not_bound() {
        let binder = FlavorBinder::new(FlavorBindingConfig {
            max_skew_us: 100,
            min_confidence: 0.5,
        })
        .unwrap();
        let odor = percept(ChemicalModality::Olfactory, 0, 1, 0.9);
        let taste = percept(ChemicalModality::Gustatory, 1_000, 2, 0.9);
        assert!(matches!(
            binder.bind(&odor, &taste),
            Err(FlavorBindingError::TemporalSkew { .. })
        ));
    }

    #[test]
    fn weak_modality_blocks_flavor_binding() {
        let binder = FlavorBinder::new(FlavorBindingConfig::default()).unwrap();
        let odor = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let taste = percept(ChemicalModality::Gustatory, 2, 2, 0.2);
        assert!(matches!(
            binder.bind(&odor, &taste),
            Err(FlavorBindingError::LowConfidence {
                modality: ChemicalModality::Gustatory,
                ..
            })
        ));
    }

    #[test]
    fn changing_taste_changes_flavor_representation() {
        let binder = FlavorBinder::new(FlavorBindingConfig::default()).unwrap();
        let odor = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let taste_a = percept(ChemicalModality::Gustatory, 2, 2, 0.9);
        let taste_b = percept(ChemicalModality::Gustatory, 2, 3, 0.9);
        let flavor_a = binder.bind(&odor, &taste_a).unwrap();
        let flavor_b = binder.bind(&odor, &taste_b).unwrap();
        assert!(flavor_a.vector.similarity(&flavor_b.vector) < 0.99);
    }
}
