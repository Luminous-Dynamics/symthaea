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
//! The public [`ChemicalModalBridge::aggregate`] path retains the legacy raw-time
//! contract: multiple sources must share one explicit [`ChemicalClockDomainId`]
//! and fit inside the configured raw timestamp skew. New evidence-bearing timing
//! code may instead call the crate-private `aggregate_after_temporal_admission`
//! only after generic bounded temporal admission has succeeded. That path skips
//! the duplicate legacy temporal decision while reusing this exact validation,
//! ordering, evidence-bundle, conflict, confidence, and HDC bundling code.
//!
//! Raw chemical clock metadata is never rewritten by either path. On an
//! externally admitted aggregate, `clock_domain` remains raw-source provenance:
//! it is `Some` only when all raw components already share one declared chemical
//! clock. A mixed-clock admitted aggregate therefore has `clock_domain = None`
//! even though its separate generic timing evidence may prove comparability.
//!
//! The numeric target IDs mirror the canonical root identity contract introduced
//! by PR #84 (`consciousness::integration::modality_identity`). This domain crate
//! intentionally does not depend on the root `symthaea` package, avoiding a
//! dependency cycle. The final root bridge must assert the mapping on its side too.

use crate::{
    ChemicalClockDomainId, ChemicalEncodingSpaceId, ChemicalEvidenceBundleId, ChemicalModality,
    ChemicalPercept,
};
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
    /// Maximum raw timestamp spread among components treated as one current-cycle
    /// observation by the legacy bridge path. This is a protocol choice, not a
    /// universal psychophysical constant. The generic timed wrapper uses the same
    /// threshold against evidence-bearing comparison timestamps before invoking
    /// the admitted aggregation path.
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
    /// More than one source was supplied but at least one timestamp had no
    /// declared comparison domain on the legacy raw-time path.
    MissingSharedClockDomain,
    MixedClockDomains {
        expected: ChemicalClockDomainId,
        actual: ChemicalClockDomainId,
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
/// encoding-space identity. `evidence_bundle_id` gives the exact raw evidence
/// bundle a compact stable identity independent of the derived HDC representation.
///
/// `clock_domain`, `earliest_timestamp_us`, and `latest_timestamp_us` are raw
/// acquisition provenance. The public legacy aggregate requires them to form one
/// comparable raw-time envelope. An aggregate created inside a validated generic
/// temporal wrapper may instead contain mixed raw clocks; in that case
/// `clock_domain` is `None` and the generic wrapper is the authority for temporal
/// comparability. Such an input must not be detached from that wrapper and passed
/// to a legacy temporal validator as though `None` meant comparable raw time.
#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalModalBridgeInput {
    pub target: ChemicalBridgeTarget,
    /// Identity of the raw observations summarized by this aggregate.
    pub evidence_bundle_id: ChemicalEvidenceBundleId,
    /// Identity of the HDC coordinate system used to represent those observations.
    pub encoding_space_id: ChemicalEncodingSpaceId,
    /// Shared raw chemical timebase when all components already declare one.
    /// `None` means either a single unclocked source or no single raw timebase.
    pub clock_domain: Option<ChemicalClockDomainId>,
    pub vector: ContinuousHV,
    /// Effective confidence after component trust and cross-source conflict are
    /// combined. Multiple agreeing sensors do not automatically inflate this
    /// above the confidence scale of their components.
    pub confidence: f32,
    /// Conflict-aware same-modality agreement in [0, 1]. Strong contradictory
    /// components can drive this to zero; a very weak contradictory component
    /// contributes only in proportion to its ability to support that conflict.
    pub agreement: f32,
    /// Raw acquisition timestamp envelope. Generic normalized comparison time is
    /// retained separately by the timed wrapper and is never written here.
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
    /// current-cycle input using the legacy raw-time contract.
    ///
    /// Components are sorted deterministically before floating-point accumulation
    /// so caller iteration order cannot change the resulting vector. Validation
    /// completes before any derived representation is constructed. For multiple
    /// components, raw timestamp skew is evaluated only after every component
    /// proves membership in the same explicit chemical clock domain.
    pub fn aggregate(
        &self,
        percepts: &[ChemicalPercept],
    ) -> Result<ChemicalModalBridgeInput, ChemicalModalBridgeError> {
        let (encoding_space_id, modality) = validate_component_geometry(percepts)?;
        let clock_domain = shared_clock_domain(percepts)?;
        let (earliest_timestamp_us, latest_timestamp_us) = raw_timestamp_envelope(percepts)?;
        let skew_us = latest_timestamp_us.saturating_sub(earliest_timestamp_us);
        if skew_us > self.config.max_component_skew_us {
            return Err(ChemicalModalBridgeError::ComponentSkew {
                skew_us,
                max_skew_us: self.config.max_component_skew_us,
            });
        }

        Ok(build_aggregate(
            percepts,
            encoding_space_id,
            modality,
            clock_domain,
            earliest_timestamp_us,
            latest_timestamp_us,
        ))
    }

    /// Aggregate HDC/evidence geometry after a stronger temporal layer has
    /// already admitted the exact component set.
    ///
    /// This method is crate-private by design. It performs *no* timestamp
    /// comparability decision. Callers must retain and later revalidate the
    /// generic temporal admission evidence that authorized this operation.
    ///
    /// All non-temporal bridge invariants and the exact legacy aggregation math
    /// are shared with [`Self::aggregate`]. Raw timestamps/domains remain attached
    /// solely as acquisition provenance and are never normalized in place.
    pub(crate) fn aggregate_after_temporal_admission(
        &self,
        percepts: &[ChemicalPercept],
    ) -> Result<ChemicalModalBridgeInput, ChemicalModalBridgeError> {
        let (encoding_space_id, modality) = validate_component_geometry(percepts)?;
        let clock_domain = uniform_raw_clock_domain(percepts);
        let (earliest_timestamp_us, latest_timestamp_us) = raw_timestamp_envelope(percepts)?;

        Ok(build_aggregate(
            percepts,
            encoding_space_id,
            modality,
            clock_domain,
            earliest_timestamp_us,
            latest_timestamp_us,
        ))
    }
}

impl Default for ChemicalModalBridge {
    fn default() -> Self {
        Self::new(ChemicalModalBridgeConfig::default())
    }
}

fn validate_component_geometry(
    percepts: &[ChemicalPercept],
) -> Result<(ChemicalEncodingSpaceId, ChemicalModality), ChemicalModalBridgeError> {
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

    Ok((encoding_space_id, modality))
}

fn raw_timestamp_envelope(
    percepts: &[ChemicalPercept],
) -> Result<(u64, u64), ChemicalModalBridgeError> {
    let earliest = percepts
        .iter()
        .map(ChemicalPercept::timestamp_us)
        .min()
        .ok_or(ChemicalModalBridgeError::EmptyInput)?;
    let latest = percepts
        .iter()
        .map(ChemicalPercept::timestamp_us)
        .max()
        .ok_or(ChemicalModalBridgeError::EmptyInput)?;
    Ok((earliest, latest))
}

fn build_aggregate(
    percepts: &[ChemicalPercept],
    encoding_space_id: ChemicalEncodingSpaceId,
    modality: ChemicalModality,
    clock_domain: Option<ChemicalClockDomainId>,
    earliest_timestamp_us: u64,
    latest_timestamp_us: u64,
) -> ChemicalModalBridgeInput {
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
    let evidence_bundle_id = ChemicalEvidenceBundleId::from_percepts(&components);

    if components.len() == 1 {
        return ChemicalModalBridgeInput {
            target: modality.into(),
            evidence_bundle_id,
            encoding_space_id,
            clock_domain,
            vector: components[0].fingerprint.vector.clone(),
            confidence: components[0].confidence(),
            agreement: 1.0,
            earliest_timestamp_us,
            latest_timestamp_us,
            components,
        };
    }

    let agreement = conflict_aware_agreement(&components);
    let base_confidence = evidence_weighted_confidence(&components);
    let confidence = (base_confidence * agreement).clamp(0.0, 1.0);

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

    ChemicalModalBridgeInput {
        target: modality.into(),
        evidence_bundle_id,
        encoding_space_id,
        clock_domain,
        vector,
        confidence,
        agreement,
        earliest_timestamp_us,
        latest_timestamp_us,
        components,
    }
}

/// Determine whether timestamp comparison is admissible for the legacy raw-time
/// aggregate. A single observation does not require a clock domain because no
/// cross-source timestamp comparison is performed. Two or more observations must
/// all declare the exact same clock-domain identity.
fn shared_clock_domain(
    components: &[ChemicalPercept],
) -> Result<Option<ChemicalClockDomainId>, ChemicalModalBridgeError> {
    let first = components
        .first()
        .ok_or(ChemicalModalBridgeError::EmptyInput)?;
    if components.len() == 1 {
        return Ok(first.evidence.clock_domain.clone());
    }

    let expected = first
        .evidence
        .clock_domain
        .clone()
        .ok_or(ChemicalModalBridgeError::MissingSharedClockDomain)?;
    for component in components.iter().skip(1) {
        let actual = component
            .evidence
            .clock_domain
            .clone()
            .ok_or(ChemicalModalBridgeError::MissingSharedClockDomain)?;
        if actual != expected {
            return Err(ChemicalModalBridgeError::MixedClockDomains {
                expected,
                actual,
            });
        }
    }
    Ok(Some(expected))
}

/// Preserve one raw clock domain only when every component actually declares the
/// same one. This function never decides temporal comparability.
fn uniform_raw_clock_domain(components: &[ChemicalPercept]) -> Option<ChemicalClockDomainId> {
    let first = components.first()?.evidence.clock_domain.clone()?;
    if components
        .iter()
        .skip(1)
        .all(|component| component.evidence.clock_domain.as_ref() == Some(&first))
    {
        Some(first)
    } else {
        None
    }
}

/// Confidence scale of the evidence set without multiplying certainty simply
/// because more sensors exist. Weighting confidence by itself gives strong
/// evidence more influence while preserving the invariant that N identical
/// components at confidence C still have base confidence C.
fn evidence_weighted_confidence(components: &[ChemicalPercept]) -> f32 {
    let total_confidence: f32 = components.iter().map(ChemicalPercept::confidence).sum();
    if total_confidence <= f32::EPSILON {
        return 0.0;
    }

    components
        .iter()
        .map(|component| {
            let confidence = component.confidence();
            confidence * confidence
        })
        .sum::<f32>()
        / total_confidence
}

/// Convert pairwise geometric disagreement into a trust-aware conflict penalty.
///
/// For each pair, only the weaker component's confidence can support a conflict
/// claim. The normalization is chosen so mutually orthogonal, equally trusted
/// components can drive agreement to zero, while a very weak outlier cannot veto
/// a much stronger consensus. Negative cosine similarity is treated as maximal
/// conflict rather than as signed cancellation.
fn conflict_aware_agreement(components: &[ChemicalPercept]) -> f32 {
    if components.len() < 2 {
        return 1.0;
    }

    let total_confidence: f32 = components.iter().map(ChemicalPercept::confidence).sum();
    if total_confidence <= f32::EPSILON {
        return 0.0;
    }

    let mut conflict_mass = 0.0f32;
    for left in 0..components.len() {
        for right in (left + 1)..components.len() {
            let similarity = components[left]
                .fingerprint
                .vector
                .similarity(&components[right].fingerprint.vector)
                .clamp(0.0, 1.0);
            let conflict_support = components[left]
                .confidence()
                .min(components[right].confidence());
            conflict_mass += conflict_support * (1.0 - similarity);
        }
    }

    let max_conflict_mass =
        0.5 * (components.len().saturating_sub(1) as f32) * total_confidence;
    if max_conflict_mass <= f32::EPSILON {
        0.0
    } else {
        (1.0 - conflict_mass / max_conflict_mass).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChemicalFingerprint, ChemicalObservation, EnvironmentReading};

    fn test_clock() -> ChemicalClockDomainId {
        ChemicalClockDomainId::new("test-rig/monotonic").unwrap()
    }

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
                clock_domain: Some(test_clock()),
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
        assert_eq!(output.clock_domain.as_ref(), input.evidence.clock_domain.as_ref());
        assert_eq!(
            output.encoding_space_id,
            ChemicalEncodingSpaceId::from_bytes([7; 32])
        );
        assert_eq!(
            output.evidence_bundle_id,
            ChemicalEvidenceBundleId::from_percepts(std::slice::from_ref(&input))
        );
        assert_eq!(output.component_count(), 1);
        assert_eq!(output.components[0], input);
        assert_eq!(output.stable_target_id(), 13);
    }

    #[test]
    fn single_unspecified_clock_is_preserved_without_cross_source_comparison() {
        let bridge = ChemicalModalBridge::default();
        let mut input = odor(10, "nose-a", 1, 0.8);
        input.evidence.clock_domain = None;
        let output = bridge.aggregate(std::slice::from_ref(&input)).unwrap();
        assert!(output.clock_domain.is_none());
    }

    #[test]
    fn multiple_sources_require_an_explicit_shared_clock_domain() {
        let bridge = ChemicalModalBridge::default();
        let mut a = odor(10, "nose-a", 1, 0.9);
        let b = odor(20, "nose-b", 1, 0.9);
        a.evidence.clock_domain = None;

        assert!(matches!(
            bridge.aggregate(&[a, b]),
            Err(ChemicalModalBridgeError::MissingSharedClockDomain)
        ));
    }

    #[test]
    fn mixed_clock_domains_are_rejected_before_skew_is_interpreted() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.9);
        let mut b = odor(20, "nose-b", 1, 0.9);
        b.evidence.clock_domain = Some(ChemicalClockDomainId::new("other-rig/monotonic").unwrap());

        assert!(matches!(
            bridge.aggregate(&[a, b]),
            Err(ChemicalModalBridgeError::MixedClockDomains { .. })
        ));
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
        assert_eq!(output.clock_domain.as_ref(), Some(&test_clock()));
        assert_eq!(output.earliest_timestamp_us, 10);
        assert_eq!(output.latest_timestamp_us, 20);
        assert!((output.agreement - 1.0).abs() < 1e-5);
        assert!(output.confidence > 0.8 && output.confidence < 0.9);
        assert_eq!(output.components[0], a);
        assert_eq!(output.components[1], b);
    }

    #[test]
    fn admitted_geometry_matches_legacy_geometry_when_raw_time_is_already_valid() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.9);
        let b = odor(20, "nose-b", 2, 0.8);
        let legacy = bridge.aggregate(&[a.clone(), b.clone()]).unwrap();
        let admitted = bridge
            .aggregate_after_temporal_admission(&[a, b])
            .unwrap();
        assert_eq!(admitted, legacy);
    }

    #[test]
    fn admitted_geometry_preserves_mixed_raw_clocks_without_relabeling() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.9);
        let mut b = odor(20, "nose-b", 2, 0.8);
        b.evidence.clock_domain = Some(ChemicalClockDomainId::new("other-rig/monotonic").unwrap());

        let output = bridge
            .aggregate_after_temporal_admission(&[a.clone(), b.clone()])
            .unwrap();
        assert!(output.clock_domain.is_none());
        assert_eq!(output.component_count(), 2);
        assert!(output.components.contains(&a));
        assert!(output.components.contains(&b));
    }

    #[test]
    fn equal_confidence_redundancy_does_not_inflate_confidence() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.8);
        let b = percept(
            ChemicalModality::Olfactory,
            20,
            "nose-b",
            a.fingerprint.vector.clone(),
            0.8,
        );
        let output = bridge.aggregate(&[a, b]).unwrap();
        assert!((output.confidence - 0.8).abs() < 1e-5);
    }

    #[test]
    fn strong_disagreement_can_collapse_influence_without_erasing_sources() {
        let bridge = ChemicalModalBridge::default();
        let a = odor(10, "nose-a", 1, 0.9);
        let b = odor(20, "nose-b", 2, 0.9);
        let output = bridge.aggregate(&[a, b]).unwrap();

        assert!(output.agreement < 0.25);
        assert!(output.confidence < 0.25);
        assert_eq!(output.component_count(), 2);
    }

    #[test]
    fn weak_conflicting_source_cannot_veto_strong_evidence() {
        let bridge = ChemicalModalBridge::default();
        let strong = odor(10, "nose-a", 1, 0.95);
        let weak = odor(20, "nose-b", 2, 0.05);
        let output = bridge.aggregate(&[strong, weak]).unwrap();

        assert!(output.agreement > 0.85);
        assert!(output.confidence > 0.75);
        assert!(output.confidence < 0.95);
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
    fn excessive_same_cycle_skew_is_rejected_only_with_shared_clock() {
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
        assert_eq!(forward.clock_domain, reverse.clock_domain);
        assert_eq!(forward.evidence_bundle_id, reverse.evidence_bundle_id);
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
