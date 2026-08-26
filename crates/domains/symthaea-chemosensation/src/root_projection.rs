// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Assessment boundary for projecting chemical ContinuousHV percepts into the
//! BinaryHV representation consumed by Symthaea's root multimodal integrator.
//!
//! The current root `ModalInput` uses [`BinaryHV`], while chemical cognition
//! intentionally preserves a continuous HDC representation. Sign-thresholding is
//! a standard HDC conversion, but it is still lossy. This module therefore makes
//! that loss explicit and measurable rather than treating conversion as a free
//! cast.
//!
//! Projection quality is descriptive evidence, not a pass/fail claim. Thresholds
//! for acceptable neighborhood distortion should be preregistered from held-out
//! experiments before canonical root integration depends on them.
//!
//! Assessment geometry is deliberately full-resolution and independent of
//! Symthaea's adaptive global cognitive stride. An experiment receipt must not
//! change because an unrelated runtime throttle changed how many HDC dimensions
//! cognition samples.

use symthaea_core::hdc::{HDC_DIMENSION, binary_hv::BinaryHV};

use crate::projection_geometry::{exact_cosine, validate_non_degenerate};
use crate::{
    ChemicalBridgeTarget, ChemicalClockDomainId, ChemicalEncodingSpaceId,
    ChemicalEvidenceBundleId, ChemicalModalBridgeInput, ChemicalModality,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChemicalRootProjectionConfig {
    /// Threshold used to binarize each continuous HDC dimension.
    ///
    /// Zero matches Symthaea's existing ContinuousHV -> BinaryHV bridge policy.
    pub threshold: f32,
}

impl Default for ChemicalRootProjectionConfig {
    fn default() -> Self {
        Self { threshold: 0.0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalRootProjectionError {
    NonFiniteThreshold,
    EmptyComponents,
    InvalidConfidence,
    InvalidAgreement,
    UnexpectedDimension {
        expected: usize,
        actual: usize,
    },
    NonFiniteVector,
    DegenerateVector,
    EvidenceBundleMismatch {
        expected: ChemicalEvidenceBundleId,
        actual: ChemicalEvidenceBundleId,
    },
    EncodingSpaceMismatch {
        expected: ChemicalEncodingSpaceId,
        actual: ChemicalEncodingSpaceId,
    },
    ModalityMismatch {
        expected: ChemicalModality,
        actual: ChemicalModality,
    },
    MissingSharedClockDomain,
    MixedClockDomains {
        expected: ChemicalClockDomainId,
        actual: ChemicalClockDomainId,
    },
    ClockDomainMismatch {
        expected: Option<ChemicalClockDomainId>,
        actual: Option<ChemicalClockDomainId>,
    },
    TimestampEnvelopeMismatch {
        expected_earliest_us: u64,
        actual_earliest_us: u64,
        expected_latest_us: u64,
        actual_latest_us: u64,
    },
    PairTargetMismatch {
        left: ChemicalBridgeTarget,
        right: ChemicalBridgeTarget,
    },
    PairEncodingSpaceMismatch {
        left: ChemicalEncodingSpaceId,
        right: ChemicalEncodingSpaceId,
    },
}

/// Quantization diagnostics for one projected root input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChemicalProjectionQuality {
    /// Threshold actually used for sign quantization.
    pub threshold: f32,
    /// Full-resolution cosine similarity between the source ContinuousHV and
    /// the bipolar ContinuousHV obtained by expanding the projected BinaryHV.
    pub source_to_bipolar_similarity: f32,
    /// Fraction of BinaryHV dimensions set to 1 after projection.
    pub positive_fraction: f32,
}

/// Root-facing BinaryHV projection while retaining the identities and trust
/// metadata of the continuous chemical aggregate that produced it.
///
/// `binary_vector` is a derived transport/integration representation. The raw
/// evidence identity and continuous encoding-space identity remain authoritative
/// provenance for the source percept. `clock_domain` names the timebase of the
/// timestamp envelope when one is declared; it is not a synchronization-quality
/// or authenticity claim.
#[derive(Clone, PartialEq)]
pub struct ChemicalRootProjection {
    pub target: ChemicalBridgeTarget,
    pub evidence_bundle_id: ChemicalEvidenceBundleId,
    pub encoding_space_id: ChemicalEncodingSpaceId,
    pub clock_domain: Option<ChemicalClockDomainId>,
    pub binary_vector: BinaryHV,
    pub confidence: f32,
    pub agreement: f32,
    pub earliest_timestamp_us: u64,
    pub latest_timestamp_us: u64,
    pub component_count: usize,
    pub quality: ChemicalProjectionQuality,
}

/// Pairwise distortion introduced by projecting two comparable chemical inputs
/// into the BinaryHV root representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChemicalProjectionPairAssessment {
    pub continuous_similarity: f32,
    pub projected_bipolar_similarity: f32,
    pub absolute_similarity_distortion: f32,
}

#[derive(Debug, Clone)]
pub struct ChemicalRootProjector {
    config: ChemicalRootProjectionConfig,
}

impl ChemicalRootProjector {
    pub fn new(
        config: ChemicalRootProjectionConfig,
    ) -> Result<Self, ChemicalRootProjectionError> {
        if !config.threshold.is_finite() {
            return Err(ChemicalRootProjectionError::NonFiniteThreshold);
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> ChemicalRootProjectionConfig {
        self.config
    }

    /// Project one validated chemical bridge input into the BinaryHV root
    /// representation while emitting explicit quantization diagnostics.
    ///
    /// Because `ChemicalModalBridgeInput` is a public struct, this method does
    /// not blindly trust its receipt fields. It revalidates the evidence bundle,
    /// encoding-space consistency, modality consistency, clock-domain consistency,
    /// timestamp envelope, confidence bounds, vector dimensionality, finite
    /// numeric content, and non-degenerate geometry before quantization.
    pub fn project(
        &self,
        input: &ChemicalModalBridgeInput,
    ) -> Result<ChemicalRootProjection, ChemicalRootProjectionError> {
        validate_bridge_input(input)?;

        let binary_vector = input.vector.to_binary(self.config.threshold);
        let bipolar = binary_vector.to_continuous();
        let source_to_bipolar_similarity = exact_cosine(&input.vector, &bipolar)
            .map_err(|_| ChemicalRootProjectionError::DegenerateVector)?;
        let positive_bits: u32 = binary_vector.0.iter().map(|byte| byte.count_ones()).sum();
        let positive_fraction = positive_bits as f32 / BinaryHV::DIM as f32;

        Ok(ChemicalRootProjection {
            target: input.target,
            evidence_bundle_id: input.evidence_bundle_id,
            encoding_space_id: input.encoding_space_id,
            clock_domain: input.clock_domain.clone(),
            binary_vector,
            confidence: input.confidence,
            agreement: input.agreement,
            earliest_timestamp_us: input.earliest_timestamp_us,
            latest_timestamp_us: input.latest_timestamp_us,
            component_count: input.components.len(),
            quality: ChemicalProjectionQuality {
                threshold: self.config.threshold,
                source_to_bipolar_similarity,
                positive_fraction,
            },
        })
    }

    /// Measure pairwise semantic distortion caused by BinaryHV projection.
    ///
    /// Pair comparison is only meaningful for the same modality target and same
    /// continuous encoding coordinate system. Both similarities are fixed,
    /// full-resolution cosine measurements on the same [-1, 1] scale. This
    /// geometric comparison does not infer temporal simultaneity between the pair.
    pub fn assess_pair(
        &self,
        left: &ChemicalModalBridgeInput,
        right: &ChemicalModalBridgeInput,
    ) -> Result<ChemicalProjectionPairAssessment, ChemicalRootProjectionError> {
        validate_bridge_input(left)?;
        validate_bridge_input(right)?;

        if left.target != right.target {
            return Err(ChemicalRootProjectionError::PairTargetMismatch {
                left: left.target,
                right: right.target,
            });
        }
        if left.encoding_space_id != right.encoding_space_id {
            return Err(ChemicalRootProjectionError::PairEncodingSpaceMismatch {
                left: left.encoding_space_id,
                right: right.encoding_space_id,
            });
        }

        let continuous_similarity = exact_cosine(&left.vector, &right.vector)
            .map_err(|_| ChemicalRootProjectionError::DegenerateVector)?;
        let left_binary = left.vector.to_binary(self.config.threshold).to_continuous();
        let right_binary = right.vector.to_binary(self.config.threshold).to_continuous();
        let projected_bipolar_similarity = exact_cosine(&left_binary, &right_binary)
            .map_err(|_| ChemicalRootProjectionError::DegenerateVector)?;

        Ok(ChemicalProjectionPairAssessment {
            continuous_similarity,
            projected_bipolar_similarity,
            absolute_similarity_distortion: (continuous_similarity - projected_bipolar_similarity)
                .abs(),
        })
    }
}

impl Default for ChemicalRootProjector {
    fn default() -> Self {
        Self::new(ChemicalRootProjectionConfig::default())
            .expect("default root projection threshold is finite")
    }
}

fn validate_bridge_input(
    input: &ChemicalModalBridgeInput,
) -> Result<(), ChemicalRootProjectionError> {
    if input.components.is_empty() {
        return Err(ChemicalRootProjectionError::EmptyComponents);
    }
    if !input.confidence.is_finite() || !(0.0..=1.0).contains(&input.confidence) {
        return Err(ChemicalRootProjectionError::InvalidConfidence);
    }
    if !input.agreement.is_finite() || !(0.0..=1.0).contains(&input.agreement) {
        return Err(ChemicalRootProjectionError::InvalidAgreement);
    }
    let actual_dimension = input.vector.dim();
    if actual_dimension != HDC_DIMENSION {
        return Err(ChemicalRootProjectionError::UnexpectedDimension {
            expected: HDC_DIMENSION,
            actual: actual_dimension,
        });
    }
    if input.vector.values.iter().any(|value| !value.is_finite()) {
        return Err(ChemicalRootProjectionError::NonFiniteVector);
    }
    if validate_non_degenerate(&input.vector).is_err() {
        return Err(ChemicalRootProjectionError::DegenerateVector);
    }

    let expected_bundle = ChemicalEvidenceBundleId::from_percepts(&input.components);
    if expected_bundle != input.evidence_bundle_id {
        return Err(ChemicalRootProjectionError::EvidenceBundleMismatch {
            expected: expected_bundle,
            actual: input.evidence_bundle_id,
        });
    }

    let expected_modality = input.target.modality();
    for component in &input.components {
        if component.fingerprint.encoding_space_id != input.encoding_space_id {
            return Err(ChemicalRootProjectionError::EncodingSpaceMismatch {
                expected: input.encoding_space_id,
                actual: component.fingerprint.encoding_space_id,
            });
        }
        if component.evidence.modality != expected_modality {
            return Err(ChemicalRootProjectionError::ModalityMismatch {
                expected: expected_modality,
                actual: component.evidence.modality,
            });
        }
    }

    let expected_clock_domain = expected_clock_domain(&input.components)?;
    if expected_clock_domain != input.clock_domain {
        return Err(ChemicalRootProjectionError::ClockDomainMismatch {
            expected: expected_clock_domain,
            actual: input.clock_domain.clone(),
        });
    }

    let expected_earliest_us = input
        .components
        .iter()
        .map(|component| component.timestamp_us())
        .min()
        .expect("non-empty components validated above");
    let expected_latest_us = input
        .components
        .iter()
        .map(|component| component.timestamp_us())
        .max()
        .expect("non-empty components validated above");
    if expected_earliest_us != input.earliest_timestamp_us
        || expected_latest_us != input.latest_timestamp_us
    {
        return Err(ChemicalRootProjectionError::TimestampEnvelopeMismatch {
            expected_earliest_us,
            actual_earliest_us: input.earliest_timestamp_us,
            expected_latest_us,
            actual_latest_us: input.latest_timestamp_us,
        });
    }

    Ok(())
}

fn expected_clock_domain(
    components: &[crate::ChemicalPercept],
) -> Result<Option<ChemicalClockDomainId>, ChemicalRootProjectionError> {
    let first = components
        .first()
        .ok_or(ChemicalRootProjectionError::EmptyComponents)?;
    if components.len() == 1 {
        return Ok(first.evidence.clock_domain.clone());
    }

    let expected = first
        .evidence
        .clock_domain
        .clone()
        .ok_or(ChemicalRootProjectionError::MissingSharedClockDomain)?;
    for component in components.iter().skip(1) {
        let actual = component
            .evidence
            .clock_domain
            .clone()
            .ok_or(ChemicalRootProjectionError::MissingSharedClockDomain)?;
        if actual != expected {
            return Err(ChemicalRootProjectionError::MixedClockDomains {
                expected,
                actual,
            });
        }
    }
    Ok(Some(expected))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationState, ChannelEncodingSpec, ChemicalChannel, ChemicalClockDomainId,
        ChemicalFingerprintEncoder, ChemicalModalBridge, ChemicalObservation, ChemicalPercept,
        MeasurementUnit, SensorHealth,
    };
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    fn encoder() -> ChemicalFingerprintEncoder {
        ChemicalFingerprintEncoder::new(vec![ChannelEncodingSpec::new(
            "voc",
            MeasurementUnit::PartsPerMillion,
            0.0,
            100.0,
            11,
            11,
            101,
        )])
        .unwrap()
    }

    fn percept(
        encoder: &ChemicalFingerprintEncoder,
        timestamp_us: u64,
        value: f32,
        source: &str,
    ) -> ChemicalPercept {
        let observation = ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            source,
            vec![ChemicalChannel {
                name: "voc".into(),
                raw_value: value,
                unit: MeasurementUnit::PartsPerMillion,
                calibration: CalibrationState::identity("cal-v1"),
                health: SensorHealth::default(),
            }],
        );
        let fingerprint = encoder.encode(&observation).unwrap().unwrap();
        ChemicalPercept {
            evidence: observation,
            fingerprint,
        }
    }

    fn bridge_input(value: f32, timestamp_us: u64) -> ChemicalModalBridgeInput {
        let encoder = encoder();
        ChemicalModalBridge::default()
            .aggregate(&[percept(&encoder, timestamp_us, value, "nose-a")])
            .unwrap()
    }

    #[test]
    fn zero_threshold_projection_preserves_provenance_and_emits_quality() {
        let input = bridge_input(50.0, 10);
        let projected = ChemicalRootProjector::default().project(&input).unwrap();

        assert_eq!(projected.target, ChemicalBridgeTarget::Olfactory);
        assert_eq!(projected.evidence_bundle_id, input.evidence_bundle_id);
        assert_eq!(projected.encoding_space_id, input.encoding_space_id);
        assert_eq!(projected.clock_domain, input.clock_domain);
        assert_eq!(projected.component_count, 1);
        assert_eq!(projected.quality.threshold, 0.0);
        assert!(projected.quality.source_to_bipolar_similarity > 0.0);
        assert!((0.0..=1.0).contains(&projected.quality.positive_fraction));
    }

    #[test]
    fn projection_is_deterministic() {
        let input = bridge_input(37.5, 10);
        let projector = ChemicalRootProjector::default();
        let a = projector.project(&input).unwrap();
        let b = projector.project(&input).unwrap();
        assert_eq!(a.binary_vector, b.binary_vector);
        assert_eq!(a.quality, b.quality);
    }

    #[test]
    fn non_finite_threshold_is_rejected() {
        assert!(matches!(
            ChemicalRootProjector::new(ChemicalRootProjectionConfig {
                threshold: f32::NAN,
            }),
            Err(ChemicalRootProjectionError::NonFiniteThreshold)
        ));
    }

    #[test]
    fn degenerate_vector_is_rejected_before_quality_is_computed() {
        let mut input = bridge_input(50.0, 10);
        input.vector = ContinuousHV::zero(HDC_DIMENSION);
        assert!(matches!(
            ChemicalRootProjector::default().project(&input),
            Err(ChemicalRootProjectionError::DegenerateVector)
        ));
    }

    #[test]
    fn forged_evidence_bundle_is_rejected_before_projection() {
        let mut input = bridge_input(50.0, 10);
        input.evidence_bundle_id = ChemicalEvidenceBundleId::from_bytes([9; 32]);
        assert!(matches!(
            ChemicalRootProjector::default().project(&input),
            Err(ChemicalRootProjectionError::EvidenceBundleMismatch { .. })
        ));
    }

    #[test]
    fn forged_encoding_space_is_rejected_before_projection() {
        let mut input = bridge_input(50.0, 10);
        input.encoding_space_id = ChemicalEncodingSpaceId::from_bytes([9; 32]);
        assert!(matches!(
            ChemicalRootProjector::default().project(&input),
            Err(ChemicalRootProjectionError::EncodingSpaceMismatch { .. })
        ));
    }

    #[test]
    fn forged_clock_domain_is_rejected_before_projection() {
        let mut input = bridge_input(50.0, 10);
        input.clock_domain = Some(ChemicalClockDomainId::unix_epoch());
        assert!(matches!(
            ChemicalRootProjector::default().project(&input),
            Err(ChemicalRootProjectionError::ClockDomainMismatch { .. })
        ));
    }

    #[test]
    fn pair_assessment_reports_projection_distortion_without_declaring_a_gate() {
        let center = bridge_input(50.0, 10);
        let near = bridge_input(51.0, 20);
        let assessment = ChemicalRootProjector::default()
            .assess_pair(&center, &near)
            .unwrap();

        assert!((-1.0..=1.0).contains(&assessment.continuous_similarity));
        assert!((-1.0..=1.0).contains(&assessment.projected_bipolar_similarity));
        assert!(assessment.absolute_similarity_distortion >= 0.0);
    }

    #[test]
    fn sign_projection_preserves_basic_locality_for_current_scalar_fixture() {
        let center = bridge_input(50.0, 10);
        let near = bridge_input(51.0, 20);
        let far = bridge_input(90.0, 30);
        let projector = ChemicalRootProjector::default();

        let near_assessment = projector.assess_pair(&center, &near).unwrap();
        let far_assessment = projector.assess_pair(&center, &far).unwrap();

        assert!(near_assessment.continuous_similarity > far_assessment.continuous_similarity);
        assert!(
            near_assessment.projected_bipolar_similarity
                > far_assessment.projected_bipolar_similarity
        );
    }

    #[test]
    fn pair_assessment_refuses_different_encoding_spaces() {
        let left = bridge_input(50.0, 10);
        let mut right = bridge_input(51.0, 20);
        right.encoding_space_id = ChemicalEncodingSpaceId::from_bytes([8; 32]);
        for component in &mut right.components {
            component.fingerprint.encoding_space_id = right.encoding_space_id;
        }

        assert!(matches!(
            ChemicalRootProjector::default().assess_pair(&left, &right),
            Err(ChemicalRootProjectionError::PairEncodingSpaceMismatch { .. })
        ));
    }
}
