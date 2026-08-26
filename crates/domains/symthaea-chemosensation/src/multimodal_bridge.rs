// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Root-agnostic bridge from chemical percepts to one current-cycle modality input.
//!
//! The root `MultiModalIntegrator` should receive at most one fresh input per
//! modality per integration cycle. Chemical hardware may legitimately provide
//! several noses, electrode arrays, or independently calibrated devices at once.
//! Feeding those samples independently into one root cycle would make same-cycle
//! sensor multiplicity look like temporal evolution in the modality channel.
//!
//! This adapter therefore combines comparable, same-modality [`ChemicalPercept`]
//! values into one evidence-preserving representation. Sensor disagreement is
//! retained as an explicit agreement score and reduces the bridge confidence;
//! it is never averaged away into false certainty.
//!
//! Hypervector comparison is only meaningful when components were encoded in
//! the same HDC coordinate system. Comparability is proven from each fingerprint's
//! content-addressed [`ChemicalEncodingSpaceId`], not from a caller-supplied label.
//!
//! The numeric target IDs mirror the canonical root identity contract introduced
//! by PR #84 (`consciousness::integration::modality_identity`). This domain crate
//! intentionally does not depend on the root `symthaea` package, avoiding a
//! dependency cycle. The final root bridge must assert the mapping on its side too.

use crate::{ChemicalEncodingSpaceId, ChemicalModality, ChemicalPercept};
use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChemicalBridgeTarget {
    Olfactory,
    Gustatory,
}

impl ChemicalBridgeTarget {
    pub const fn stable_id(self) -> u16 {
        match self {
            Self::Olfactory => 13,
            Self::Gustatory => 14,
        }
    }

    pub const fn modality(self) -> ChemicalModality {
        match self {
            Self::Olfactory => ChemicalModality::Olfactory,
            Self::Gustatory => ChemicalModality::Gustatory,
        }
    }
}

impl From<ChemicalModality> for ChemicalBridgeTarget {
    fn from(value: ChemicalModality) -> Self {
        match value {
            ChemicalModality::Olfactory => Self::Olfactory,
            ChemicalModality::Gustatory => Self::Gustatory,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalModalBridgeConfig {
    /// Maximum timestamp spread among components treated as one current-cycle
    /// observation. This is a protocol choice, not a universal psychophysical
    /// constant.
    pub max_component_skew_us: u64,
}

impl Default for ChemicalModalBridgeConfig {
    fn default() -> Self {
        Self {
            max_component_skew_us: 100_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalModalBridgeError {
    EmptyInput,
    MixedEncodingSpaces {
        expected: ChemicalEncodingSpaceId,
        actual: ChemicalEncodingSpaceId,
    },
    MixedModalities {
        expected: ChemicalModality,
        actual: ChemicalModality,
    },
    InvalidConfidence,
    UntrustedComponent,
    NonFiniteVector,
    UnexpectedDimension {
        expected: usize,
        actual: usize,
    },
    ComponentSkew {
        skew_us: u64,
        max_skew_us: u64,
    },
}

/// One root-ready chemical modality representation plus all source percepts.
///
/// `components` remain attached so downstream code can inspect disagreement,
/// sensor identity, raw observations, calibration provenance, timestamps, and
/// their common encoding-space identity. The aggregate vector is a convenience
/// representation, not replacement evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalModalBridgeInput {
    pub target: ChemicalBridgeTarget,
    pub encoding_space_id: ChemicalEncodingSpaceId,
    pub vector: ContinuousHV,
    /// Conservative effective confidence after same-modality agreement is
    /// considered. For a single component this equals that percept's confidence.
    pub confidence: f32,
    /// Confidence-weighted pairwise cosine agreement in [0, 1]. A single source
    /// has agreement 1 by definition. Negative similarity is treated as maximal
    /// disagreement (0), not as evidence that cancels into confidence.
    pub agreement: f32,
    pub earliest_timestamp_us: u64,
    pub latest_timestamp_us: u64,
    pub components: Vec<ChemicalPercept>,
}

impl ChemicalModalBridgeInput {
    pub fn modality(&self) -> ChemicalModality {
        self.target.modality()
    }

    pub fn stable_target_id(&self) -> u16 {
        self.target.stable_id()
    }

    pub fn timestamp_us(&self) -> u64 {
        self.latest_timestamp_us
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }
}

#[derive(Debug, Clone)]
pub struct ChemicalModalBridge {
    config: ChemicalModalBridgeConfig,
}

impl ChemicalModalBridge {
    pub fn new(config: ChemicalModalBridgeConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &ChemicalModalBridgeConfig {
        &self.config
    }

    /// Aggregate comparable same-modality percepts into exactly one root-ready
    /// current-cycle input.
    ///
    /// Components are sorted deterministically before floating-point accumulation
    /// so caller iteration order cannot change the resulting vector. Validation
    /// completes before any derived representation is constructed.
    pub fn aggregate(
        &self,
        percepts: &[ChemicalPercept],
    ) -> Result<ChemicalModalBridgeInput, ChemicalModalBridgeError> {
        let first = percepts
            .first()
            .ok_or(ChemicalModalBridgeError::EmptyInput)?;
        let encoding_space_id = first.fingerprint.encoding_space_id;
        let modality = first.evidence.modality;

        for percept in percepts {
            if percept.fingerprint.encoding_space_id != encoding_space_id {
                return Err(ChemicalModalBridgeError::MixedEncodingSpaces {
                    expected: encoding_space_id,
                    actual: percept.fingerprint.encoding_space_id,
                });
            }
            if percept.evidence.modality != modality {
                return Err(ChemicalModalBridgeError::MixedModalities {
                    expected: modality,
                    actual: percept.evidence.modality,
                });
            }
            let confidence = percept.confidence();
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                return Err(ChemicalModalBridgeError::InvalidConfidence);
            }
            if confidence <= 0.0 {
                return Err(ChemicalModalBridgeError::UntrustedComponent);
            }
            let actual = percept.fingerprint.vector.dim();
            if actual != HDC_DIMENSION {
                return Err(ChemicalModalBridgeError::UnexpectedDimension {
                    expected: HDC_DIMENSION,
                    actual,
                });
            }
            if percept
                .fingerprint
                .vector
                .values
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(ChemicalModalBridgeError::NonFiniteVector);
            }
        }

        let earliest_timestamp_us = percepts
            .iter()
            .map(ChemicalPercept::timestamp_us)
            .min()
            .expect("non-empty input validated above");
        let latest_timestamp_us = percepts
            .iter()
            .map(ChemicalPercept::timestamp_us)
            .max()
            .expect("non-empty input validated above");
        let skew_us = latest_timestamp_us.saturating_sub(earliest_timestamp_us);
        if skew_us > self.config.max_component_skew_us {
            return Err(ChemicalModalBridgeError::ComponentSkew {
                skew_us,
                max_skew_us: self.config.max_component_skew_us,
            });
        }

        let mut components = percepts.to_vec();
        components.sort_by(|left, right| {
            left.timestamp_us()
                .cmp(&right.timestamp_us())
                .then_with(|| left.evidence.source.cmp(&right.evidence.source))
                .then_with(|| {
                    left.fingerprint
                        .vector
                        .values
                        .iter()
                        .map(|value| value.to_bits())
                        .cmp(
                            right
                                .fingerprint
                                .vector
                                .values
                                .iter()
                                .map(|value| value.to_bits()),
                        )
                })
                .then_with(|| {
                    left.confidence()
                        .to_bits()
                        .cmp(&right.confidence().to_bits())
                })
        });

        if components.len() == 1 {
            return Ok(ChemicalModalBridgeInput {
                target: modality.into(),
                encoding_space_id,
                vector: components[0].fingerprint.vector.clone(),
                confidence: components[0].confidence(),
                agreement: 1.0,
                earliest_timestamp_us,
                latest_timestamp_us,
                components,
            });
        }

        let agreement = pairwise_agreement(&components);
        let weakest_confidence = components
            .iter()
            .map(ChemicalPercept::confidence)
            .fold(1.0f32, f32::min);
        let confidence = (weakest_confidence * agreement).clamp(0.0, 1.0);

        let hvs: Vec<&ContinuousHV> = components
            .iter()
            .map(|percept| &percept.fingerprint.vector)
            .collect();
        let weights: Vec<f32> = components
            .iter()
            .map(ChemicalPercept::confidence)
            .collect();
        let mut vector = ContinuousHV::weighted_bundle(&hvs, &weights);
        vector.l2_normalize();

        Ok(ChemicalModalBridgeInput {
            target: modality.into(),
            encoding_space_id,
            vector,
            confidence,
            agreement,
            earliest_timestamp_us,
            latest_timestamp_us,
            components,
        })
    }
}

impl Default for ChemicalModalBridge {
    fn default() -> Self {
        Self::new(ChemicalModalBridgeConfig::default())
    }
}

fn pairwise_agreement(components: &[ChemicalPercept]) -> f32 {
    if components.len() < 2 {
        return 1.0;
    }

    let mut weighted_similarity = 0.0f32;
    let mut pair_weight_sum = 0.0f32;
    for left in 0..components.len() {
        for right in (left + 1)..components.len() {
            let pair_weight = components[left].confidence() * components[right].confidence();
            let similarity = components[left]
                .fingerprint
                .vector
                .similarity(&components[right].fingerprint.vector)
                .clamp(0.0, 1.0);
            weighted_similarity += similarity * pair_weight;
            pair_weight_sum += pair_weight;
        }
    }

    if pair_weight_sum <= f32::EPSILON {
        0.0
    } else {
        (weighted_similarity / pair_weight_sum).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalFingerprint, ChemicalObservation, EnvironmentReading};

    fn percept(
        modality: ChemicalModality,
        timestamp_us: u64,
        source: &str,
        vector: ContinuousHV,
        confidence: f32,
    ) -> ChemicalPercept {
        ChemicalPercept {
            evidence: ChemicalObservation {
                timestamp_us,
                modality,
                source: source.into(),
                channels: vec![],
                environment: EnvironmentReading::default(),
            },
            fingerprint: ChemicalFingerprint {
                vector,
                confidence,
                used_channels: 1,
                ignored_channels: 0,
                encoding_space_id: ChemicalEncodingSpaceId::from_bytes([7; 32]),
            },
        }
    }

    fn odor(timestamp_us: u64, source: &str, seed: u64, confidence: f32) -> ChemicalPercept {
        percept(
            ChemicalModality::Olfactory,
            timestamp_us,
            source,
            ContinuousHV::random(HDC_DIMENSION, seed),
            confidence,
        )
    }

    #[test]
    fn single_percept_round_trips_without_reencoding() {
        let bridge = ChemicalModalBridge::default();
        let input = odor(10, "nose-a", 1, 0.8);
        let output = bridge.aggregate(std::slice::from_ref(&input)).unwrap();

        assert_eq!(output.vector, input.fingerprint.vector);
        assert_eq!(output.confidence, 0.8);
        assert_eq!(output.agreement, 1.0);
        assert_eq!(
            output.encoding_space_id,
            ChemicalEncodingSpaceId::from_bytes([7; 32])
        );
        assert_eq!(output.component_count(), 1);
        assert_eq!(output.components[0], input);
        assert_eq!(output.stable_target_id(), 13);
    }

    #[test]
    fn same_modality_sources_collapse_to_one_input_and_preserve_components() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.9);
        let b = percept(
            ChemicalModality::Olfactory,
            20,
            "nose-b",
            a.fingerprint.vector.clone(),
            0.8,
        );

        let output = bridge.aggregate(&[b.clone(), a.clone()]).unwrap();
        assert_eq!(output.modality(), ChemicalModality::Olfactory);
        assert_eq!(output.component_count(), 2);
        assert_eq!(output.earliest_timestamp_us, 10);
        assert_eq!(output.latest_timestamp_us, 20);
        assert!((output.agreement - 1.0).abs() < 1e-5);
        assert!((output.confidence - 0.8).abs() < 1e-5);
        assert_eq!(output.components[0], a);
        assert_eq!(output.components[1], b);
    }

    #[test]
    fn disagreement_reduces_influence_without_erasing_sources() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.9);
        let b = odor(20, "nose-b", 2, 0.9);
        let output = bridge.aggregate(&[a, b]).unwrap();

        assert!(output.agreement < 0.25);
        assert!(output.confidence < 0.25);
        assert_eq!(output.component_count(), 2);
    }

    #[test]
    fn weakest_source_caps_joint_confidence_even_under_perfect_agreement() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.95);
        let b = percept(
            ChemicalModality::Olfactory,
            20,
            "nose-b",
            a.fingerprint.vector.clone(),
            0.4,
        );
        let output = bridge.aggregate(&[a, b]).unwrap();
        assert!((output.agreement - 1.0).abs() < 1e-5);
        assert!((output.confidence - 0.4).abs() < 1e-5);
    }

    #[test]
    fn mixed_smell_and_taste_are_not_collapsed_into_one_root_modality() {
        let bridge = ChemicalModalBridge::default();
        let odor = odor(10, "nose", 1, 0.9);
        let taste = percept(
            ChemicalModality::Gustatory,
            10,
            "tongue",
            ContinuousHV::random(HDC_DIMENSION, 2),
            0.9,
        );
        assert!(matches!(
            bridge.aggregate(&[odor, taste]),
            Err(ChemicalModalBridgeError::MixedModalities {
                expected: ChemicalModality::Olfactory,
                actual: ChemicalModality::Gustatory,
            })
        ));
    }

    #[test]
    fn different_hdc_spaces_are_not_compared_as_sensor_disagreement() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.9);
        let mut b = odor(20, "nose-b", 1, 0.9);
        b.fingerprint.encoding_space_id = ChemicalEncodingSpaceId::from_bytes([8; 32]);

        assert!(matches!(
            bridge.aggregate(&[a, b]),
            Err(ChemicalModalBridgeError::MixedEncodingSpaces { .. })
        ));
    }

    #[test]
    fn excessive_same_cycle_skew_is_rejected() {
        let bridge = ChemicalModalBridge::new(ChemicalModalBridgeConfig {
            max_component_skew_us: 10,
        });
        assert!(matches!(
            bridge.aggregate(&[
                odor(0, "nose-a", 1, 0.9),
                odor(11, "nose-b", 1, 0.9),
            ]),
            Err(ChemicalModalBridgeError::ComponentSkew {
                skew_us: 11,
                max_skew_us: 10,
            })
        ));
    }

    #[test]
    fn aggregation_is_order_invariant() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(20, "nose-b", 2, 0.8);
        let b = odor(10, "nose-a", 1, 0.9);
        let forward = bridge.aggregate(&[a.clone(), b.clone()]).unwrap();
        let reverse = bridge.aggregate(&[b, a]).unwrap();

        assert_eq!(forward.vector, reverse.vector);
        assert_eq!(forward.confidence, reverse.confidence);
        assert_eq!(forward.agreement, reverse.agreement);
        assert_eq!(forward.components, reverse.components);
    }

    #[test]
    fn target_ids_match_reserved_root_contract() {
        assert_eq!(ChemicalBridgeTarget::Olfactory.stable_id(), 13);
        assert_eq!(ChemicalBridgeTarget::Gustatory.stable_id(), 14);
    }

    #[test]
    fn flavor_is_not_a_third_root_input() {
        assert_eq!(
            ChemicalBridgeTarget::Olfactory.modality(),
            ChemicalModality::Olfactory
        );
        assert_eq!(
            ChemicalBridgeTarget::Gustatory.modality(),
            ChemicalModality::Gustatory
        );
    }
}
