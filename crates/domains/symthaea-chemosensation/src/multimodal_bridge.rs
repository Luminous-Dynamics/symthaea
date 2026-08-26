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
//! the same HDC coordinate system. Each [`ChemicalBridgeComponent`] therefore
//! carries an explicit `encoding_space` identifier and mixed spaces are rejected.
//!
//! The numeric target IDs mirror the canonical root identity contract introduced
//! by PR #84 (`consciousness::integration::modality_identity`). This domain crate
//! intentionally does not depend on the root `symthaea` package, avoiding a
//! dependency cycle. The final root bridge must assert the mapping on its side too.

use crate::{ChemicalModality, ChemicalPercept};
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

/// A chemical percept plus the HDC coordinate-system identity in which its
/// fingerprint was encoded.
///
/// The identifier is evidence about representation compatibility, not a semantic
/// odor/taste label. Callers should derive it from a versioned encoder/config
/// contract rather than from the observed sample identity.
#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalBridgeComponent {
    pub percept: ChemicalPercept,
    pub encoding_space: String,
}

impl ChemicalBridgeComponent {
    pub fn new(
        percept: ChemicalPercept,
        encoding_space: impl Into<String>,
    ) -> Result<Self, ChemicalModalBridgeError> {
        let encoding_space = encoding_space.into();
        if encoding_space.trim().is_empty() {
            return Err(ChemicalModalBridgeError::BlankEncodingSpace);
        }
        Ok(Self {
            percept,
            encoding_space,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalModalBridgeConfig {
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
    BlankEncodingSpace,
    MixedEncodingSpaces {
        expected: String,
        actual: String,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalModalBridgeInput {
    pub target: ChemicalBridgeTarget,
    pub encoding_space: String,
    pub vector: ContinuousHV,
    pub confidence: f32,
    pub agreement: f32,
    pub earliest_timestamp_us: u64,
    pub latest_timestamp_us: u64,
    pub components: Vec<ChemicalBridgeComponent>,
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

    pub fn aggregate(
        &self,
        components: &[ChemicalBridgeComponent],
    ) -> Result<ChemicalModalBridgeInput, ChemicalModalBridgeError> {
        let first = components
            .first()
            .ok_or(ChemicalModalBridgeError::EmptyInput)?;
        if first.encoding_space.trim().is_empty() {
            return Err(ChemicalModalBridgeError::BlankEncodingSpace);
        }
        let encoding_space = first.encoding_space.clone();
        let modality = first.percept.evidence.modality;

        for component in components {
            if component.encoding_space.trim().is_empty() {
                return Err(ChemicalModalBridgeError::BlankEncodingSpace);
            }
            if component.encoding_space != encoding_space {
                return Err(ChemicalModalBridgeError::MixedEncodingSpaces {
                    expected: encoding_space.clone(),
                    actual: component.encoding_space.clone(),
                });
            }
            let percept = &component.percept;
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

        let earliest_timestamp_us = components
            .iter()
            .map(|component| component.percept.timestamp_us())
            .min()
            .expect("non-empty input validated above");
        let latest_timestamp_us = components
            .iter()
            .map(|component| component.percept.timestamp_us())
            .max()
            .expect("non-empty input validated above");
        let skew_us = latest_timestamp_us.saturating_sub(earliest_timestamp_us);
        if skew_us > self.config.max_component_skew_us {
            return Err(ChemicalModalBridgeError::ComponentSkew {
                skew_us,
                max_skew_us: self.config.max_component_skew_us,
            });
        }

        let mut components = components.to_vec();
        components.sort_by(|left, right| {
            left.percept
                .timestamp_us()
                .cmp(&right.percept.timestamp_us())
                .then_with(|| left.percept.evidence.source.cmp(&right.percept.evidence.source))
                .then_with(|| {
                    left.percept
                        .fingerprint
                        .vector
                        .values
                        .iter()
                        .map(|value| value.to_bits())
                        .cmp(
                            right
                                .percept
                                .fingerprint
                                .vector
                                .values
                                .iter()
                                .map(|value| value.to_bits()),
                        )
                })
                .then_with(|| {
                    left.percept
                        .confidence()
                        .to_bits()
                        .cmp(&right.percept.confidence().to_bits())
                })
        });

        if components.len() == 1 {
            return Ok(ChemicalModalBridgeInput {
                target: modality.into(),
                encoding_space,
                vector: components[0].percept.fingerprint.vector.clone(),
                confidence: components[0].percept.confidence(),
                agreement: 1.0,
                earliest_timestamp_us,
                latest_timestamp_us,
                components,
            });
        }

        let agreement = pairwise_agreement(&components);
        let weakest_confidence = components
            .iter()
            .map(|component| component.percept.confidence())
            .fold(1.0f32, f32::min);
        let confidence = (weakest_confidence * agreement).clamp(0.0, 1.0);

        let hvs: Vec<&ContinuousHV> = components
            .iter()
            .map(|component| &component.percept.fingerprint.vector)
            .collect();
        let weights: Vec<f32> = components
            .iter()
            .map(|component| component.percept.confidence())
            .collect();
        let mut vector = ContinuousHV::weighted_bundle(&hvs, &weights);
        vector.l2_normalize();

        Ok(ChemicalModalBridgeInput {
            target: modality.into(),
            encoding_space,
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

fn pairwise_agreement(components: &[ChemicalBridgeComponent]) -> f32 {
    if components.len() < 2 {
        return 1.0;
    }

    let mut weighted_similarity = 0.0f32;
    let mut pair_weight_sum = 0.0f32;
    for left in 0..components.len() {
        for right in (left + 1)..components.len() {
            let left_percept = &components[left].percept;
            let right_percept = &components[right].percept;
            let pair_weight = left_percept.confidence() * right_percept.confidence();
            let similarity = left_percept
                .fingerprint
                .vector
                .similarity(&right_percept.fingerprint.vector)
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
            },
        }
    }

    fn component(percept: ChemicalPercept) -> ChemicalBridgeComponent {
        ChemicalBridgeComponent::new(percept, "chemical-fingerprint-v1").unwrap()
    }

    fn odor(timestamp_us: u64, source: &str, seed: u64, confidence: f32) -> ChemicalBridgeComponent {
        component(percept(
            ChemicalModality::Olfactory,
            timestamp_us,
            source,
            ContinuousHV::random(HDC_DIMENSION, seed),
            confidence,
        ))
    }

    #[test]
    fn single_percept_round_trips_without_reencoding() {
        let bridge = ChemicalModalBridge::default();
        let input = odor(10, "nose-a", 1, 0.8);
        let output = bridge.aggregate(std::slice::from_ref(&input)).unwrap();

        assert_eq!(output.vector, input.percept.fingerprint.vector);
        assert_eq!(output.confidence, 0.8);
        assert_eq!(output.agreement, 1.0);
        assert_eq!(output.encoding_space, "chemical-fingerprint-v1");
        assert_eq!(output.component_count(), 1);
        assert_eq!(output.components[0], input);
        assert_eq!(output.stable_target_id(), 13);
    }

    #[test]
    fn same_modality_sources_collapse_to_one_input_and_preserve_components() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.9);
        let b = component(percept(
            ChemicalModality::Olfactory,
            20,
            "nose-b",
            a.percept.fingerprint.vector.clone(),
            0.8,
        ));

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
        let b = component(percept(
            ChemicalModality::Olfactory,
            20,
            "nose-b",
            a.percept.fingerprint.vector.clone(),
            0.4,
        ));
        let output = bridge.aggregate(&[a, b]).unwrap();
        assert!((output.agreement - 1.0).abs() < 1e-5);
        assert!((output.confidence - 0.4).abs() < 1e-5);
    }

    #[test]
    fn mixed_smell_and_taste_are_not_collapsed_into_one_root_modality() {
        let bridge = ChemicalModalBridge::default();
        let odor = odor(10, "nose", 1, 0.9);
        let taste = component(percept(
            ChemicalModality::Gustatory,
            10,
            "tongue",
            ContinuousHV::random(HDC_DIMENSION, 2),
            0.9,
        ));
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
        let b = ChemicalBridgeComponent::new(
            percept(
                ChemicalModality::Olfactory,
                20,
                "nose-b",
                ContinuousHV::random(HDC_DIMENSION, 1),
                0.9,
            ),
            "different-role-seeds-v2",
        )
        .unwrap();

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
