// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative olfactory-gustatory flavor binding.
//!
//! A flavor percept is created only when one trustworthy olfactory percept and
//! one trustworthy gustatory percept are close enough in time to plausibly
//! belong to the same sampling episode, share one declared clock domain, and
//! share one HDC coordinate system. The original evidence from both senses
//! remains attached to the derived flavor representation.

use blake3::Hasher;
use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};

use crate::{ChemicalClockDomainId, ChemicalEncodingSpaceId, ChemicalModality, ChemicalPercept};

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
    EncodingSpaceMismatch {
        olfactory: ChemicalEncodingSpaceId,
        gustatory: ChemicalEncodingSpaceId,
    },
    MissingClockDomain(ChemicalModality),
    ClockDomainMismatch {
        olfactory: ChemicalClockDomainId,
        gustatory: ChemicalClockDomainId,
    },
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
    /// Input chemical coordinate system from which both component percepts came.
    pub source_encoding_space_id: ChemicalEncodingSpaceId,
    /// Distinct identity of the derived flavor representation. This includes the
    /// source space plus the flavor binder's role vectors, so a flavor vector is
    /// never advertised as interchangeable with a raw chemical fingerprint.
    pub encoding_space_id: ChemicalEncodingSpaceId,
    /// Declared common timebase under which `temporal_skew_us` is meaningful.
    /// This is not an accuracy or synchronization-quality claim.
    pub clock_domain: ChemicalClockDomainId,
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
    /// constructing the derived representation. Timestamp skew is never
    /// interpreted until both inputs prove membership in one declared clock
    /// domain.
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

        let olfactory_space = olfactory.fingerprint.encoding_space_id;
        let gustatory_space = gustatory.fingerprint.encoding_space_id;
        if olfactory_space != gustatory_space {
            return Err(FlavorBindingError::EncodingSpaceMismatch {
                olfactory: olfactory_space,
                gustatory: gustatory_space,
            });
        }

        for percept in [olfactory, gustatory] {
            if percept.confidence() < self.config.min_confidence {
                return Err(FlavorBindingError::LowConfidence {
                    modality: percept.evidence.modality,
                    confidence: percept.confidence(),
                    minimum: self.config.min_confidence,
                });
            }
        }

        let olfactory_clock = olfactory
            .evidence
            .clock_domain
            .clone()
            .ok_or(FlavorBindingError::MissingClockDomain(
                ChemicalModality::Olfactory,
            ))?;
        let gustatory_clock = gustatory
            .evidence
            .clock_domain
            .clone()
            .ok_or(FlavorBindingError::MissingClockDomain(
                ChemicalModality::Gustatory,
            ))?;
        if olfactory_clock != gustatory_clock {
            return Err(FlavorBindingError::ClockDomainMismatch {
                olfactory: olfactory_clock,
                gustatory: gustatory_clock,
            });
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
        let encoding_space_id = flavor_space_id(
            olfactory_space,
            &self.olfactory_role,
            &self.gustatory_role,
        );

        Ok(FlavorPercept {
            olfactory: olfactory.clone(),
            gustatory: gustatory.clone(),
            vector,
            source_encoding_space_id: olfactory_space,
            encoding_space_id,
            clock_domain: olfactory_clock,
            confidence: olfactory.confidence().min(gustatory.confidence()),
            temporal_skew_us,
        })
    }
}

fn flavor_space_id(
    source_space: ChemicalEncodingSpaceId,
    olfactory_role: &ContinuousHV,
    gustatory_role: &ContinuousHV,
) -> ChemicalEncodingSpaceId {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-chemosensation-flavor-space-v1");
    hasher.update(source_space.as_bytes());
    hash_hv(&mut hasher, olfactory_role);
    hash_hv(&mut hasher, gustatory_role);
    ChemicalEncodingSpaceId::from_bytes(*hasher.finalize().as_bytes())
}

fn hash_hv(hasher: &mut Hasher, hv: &ContinuousHV) {
    hasher.update(&(hv.values.len() as u64).to_le_bytes());
    for value in &hv.values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalFingerprint, ChemicalObservation};

    fn test_clock() -> ChemicalClockDomainId {
        ChemicalClockDomainId::new("test-rig/monotonic").unwrap()
    }

    fn percept(
        modality: ChemicalModality,
        timestamp_us: u64,
        seed: u64,
        confidence: f32,
    ) -> ChemicalPercept {
        ChemicalPercept {
            evidence: ChemicalObservation::new(
                timestamp_us,
                modality,
                format!("source-{seed}"),
                vec![],
            )
            .with_clock_domain(test_clock()),
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, seed),
                confidence,
                used_channels: 1,
                ignored_channels: 0,
                encoding_space_id: ChemicalEncodingSpaceId::from_bytes([7; 32]),
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
        assert_eq!(forward.encoding_space_id, reverse.encoding_space_id);
        assert_eq!(forward.clock_domain, test_clock());
        assert_eq!(forward.olfactory.evidence, odor.evidence);
        assert_eq!(forward.gustatory.evidence, taste.evidence);
        assert_eq!(
            forward.source_encoding_space_id,
            ChemicalEncodingSpaceId::from_bytes([7; 32])
        );
        assert_ne!(forward.encoding_space_id, forward.source_encoding_space_id);
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
    fn different_encoding_spaces_are_not_bound_as_flavor() {
        let binder = FlavorBinder::new(FlavorBindingConfig::default()).unwrap();
        let odor = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let mut taste = percept(ChemicalModality::Gustatory, 2, 2, 0.9);
        taste.fingerprint.encoding_space_id = ChemicalEncodingSpaceId::from_bytes([8; 32]);
        assert!(matches!(
            binder.bind(&odor, &taste),
            Err(FlavorBindingError::EncodingSpaceMismatch { .. })
        ));
    }

    #[test]
    fn missing_clock_domain_blocks_temporal_binding() {
        let binder = FlavorBinder::new(FlavorBindingConfig::default()).unwrap();
        let odor = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let mut taste = percept(ChemicalModality::Gustatory, 2, 2, 0.9);
        taste.evidence.clock_domain = None;
        assert!(matches!(
            binder.bind(&odor, &taste),
            Err(FlavorBindingError::MissingClockDomain(
                ChemicalModality::Gustatory
            ))
        ));
    }

    #[test]
    fn mixed_clock_domains_are_not_interpreted_as_temporal_skew() {
        let binder = FlavorBinder::new(FlavorBindingConfig::default()).unwrap();
        let odor = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let mut taste = percept(ChemicalModality::Gustatory, 2, 2, 0.9);
        taste.evidence.clock_domain =
            Some(ChemicalClockDomainId::new("other-rig/monotonic").unwrap());
        assert!(matches!(
            binder.bind(&odor, &taste),
            Err(FlavorBindingError::ClockDomainMismatch { .. })
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
    fn changing_taste_changes_flavor_representation_not_flavor_space() {
        let binder = FlavorBinder::new(FlavorBindingConfig::default()).unwrap();
        let odor = percept(ChemicalModality::Olfactory, 1, 1, 0.9);
        let taste_a = percept(ChemicalModality::Gustatory, 2, 2, 0.9);
        let taste_b = percept(ChemicalModality::Gustatory, 2, 3, 0.9);
        let flavor_a = binder.bind(&odor, &taste_a).unwrap();
        let flavor_b = binder.bind(&odor, &taste_b).unwrap();
        assert!(flavor_a.vector.similarity(&flavor_b.vector) < 0.99);
        assert_eq!(flavor_a.encoding_space_id, flavor_b.encoding_space_id);
    }
}
