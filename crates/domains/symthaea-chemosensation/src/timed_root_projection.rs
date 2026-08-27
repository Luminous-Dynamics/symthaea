// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Root projection for chemical aggregates whose temporal comparability is
//! carried by generic evidence rather than one shared raw chemical clock.
//!
//! A mixed-clock [`crate::ChemicalModalBridgeInput`] is intentionally not valid on
//! the legacy [`crate::ChemicalRootProjector::project`] path when detached from its
//! timing evidence. This module keeps the timing proof attached and revalidates
//! it before projection:
//!
//! 1. recompute generic temporal admission from the retained timed components;
//! 2. require the recomputed receipt to equal the stored admission exactly;
//! 3. require the admission to permit same-cycle aggregation;
//! 4. recompute the HDC aggregate from the exact retained raw percepts using the
//!    shared admitted bridge path;
//! 5. require the recomputed aggregate to equal the stored aggregate exactly;
//! 6. validate projection geometry and quantize to the root BinaryHV space.
//!
//! Thus an "already admitted" aggregate is never trusted merely because a caller
//! presents that label. Raw acquisition timestamps/domains remain inside the
//! chemical percepts while generic comparison time remains inside the timed
//! components and admission receipt.

use std::fmt;

use symthaea_core::hdc::{HDC_DIMENSION, binary_hv::BinaryHV};
use symthaea_time_integrity::{ClockDomainId, ClockEpochId};

use crate::projection_geometry::{exact_cosine, validate_non_degenerate};
use crate::{
    ChemicalModalBridge, ChemicalModalBridgeError, ChemicalProjectionQuality,
    ChemicalRootProjection, ChemicalRootProjectionError, ChemicalRootProjector,
    ChemicalTemporalAdmission, ChemicalTemporalAdmissionStatus, ChemicalTimeAlignmentError,
    TimedChemicalAggregation, TimedChemicalPercept, classify_chemical_temporal_admission,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimedChemicalRootProjectionError {
    /// The wrapper contains no aggregate because temporal evidence abstained.
    TemporalAdmissionNotPermitted {
        status: ChemicalTemporalAdmissionStatus,
    },
    /// Recomputing temporal admission from the retained timed components failed.
    Time(ChemicalTimeAlignmentError),
    /// The stored temporal receipt does not equal a fresh recomputation from the
    /// retained timed components and threshold.
    AdmissionMismatch,
    /// Recomputing the non-temporal HDC aggregate failed.
    Bridge(ChemicalModalBridgeError),
    /// The stored HDC aggregate does not equal a fresh recomputation from the
    /// exact retained raw percepts.
    AggregateMismatch,
    /// Root projection geometry is invalid.
    Projection(ChemicalRootProjectionError),
}

impl fmt::Display for TimedChemicalRootProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TemporalAdmissionNotPermitted { status } => write!(
                f,
                "chemical timed root projection requires an admitted aggregate, got {status:?}"
            ),
            Self::Time(error) => write!(f, "chemical temporal evidence cannot be revalidated: {error}"),
            Self::AdmissionMismatch => write!(
                f,
                "stored chemical temporal admission does not match retained timed components"
            ),
            Self::Bridge(error) => write!(f, "chemical HDC aggregate cannot be recomputed: {error:?}"),
            Self::AggregateMismatch => write!(
                f,
                "stored chemical HDC aggregate does not match retained timed components"
            ),
            Self::Projection(error) => write!(f, "chemical timed root projection failed: {error:?}"),
        }
    }
}

impl std::error::Error for TimedChemicalRootProjectionError {}

impl From<ChemicalTimeAlignmentError> for TimedChemicalRootProjectionError {
    fn from(value: ChemicalTimeAlignmentError) -> Self {
        Self::Time(value)
    }
}

impl From<ChemicalModalBridgeError> for TimedChemicalRootProjectionError {
    fn from(value: ChemicalModalBridgeError) -> Self {
        Self::Bridge(value)
    }
}

impl From<ChemicalRootProjectionError> for TimedChemicalRootProjectionError {
    fn from(value: ChemicalRootProjectionError) -> Self {
        Self::Projection(value)
    }
}

/// Root projection plus the exact generic timing evidence that authorized the
/// same-cycle chemical aggregate.
///
/// `projection` retains the unchanged raw chemical evidence identities and raw
/// source timestamp envelope. The comparison-time envelope below is separate and
/// comes only from the revalidated generic timed components.
#[derive(Clone, PartialEq)]
pub struct TimedChemicalRootProjection {
    projection: ChemicalRootProjection,
    admission: ChemicalTemporalAdmission,
    timed_components: Vec<TimedChemicalPercept>,
    earliest_comparison_timestamp_us: u64,
    latest_comparison_timestamp_us: u64,
}

impl TimedChemicalRootProjection {
    pub fn projection(&self) -> &ChemicalRootProjection {
        &self.projection
    }

    pub fn into_projection(self) -> ChemicalRootProjection {
        self.projection
    }

    pub fn admission(&self) -> &ChemicalTemporalAdmission {
        &self.admission
    }

    pub fn timed_components(&self) -> &[TimedChemicalPercept] {
        &self.timed_components
    }

    pub fn comparison_clock_domain(&self) -> &ClockDomainId {
        self.admission.clock_domain()
    }

    pub fn comparison_clock_epoch(&self) -> Option<&ClockEpochId> {
        self.admission.clock_epoch()
    }

    pub fn earliest_comparison_timestamp_us(&self) -> u64 {
        self.earliest_comparison_timestamp_us
    }

    pub fn latest_comparison_timestamp_us(&self) -> u64 {
        self.latest_comparison_timestamp_us
    }

    pub fn comparison_is_unix_epoch(&self) -> bool {
        self.comparison_clock_domain() == &ClockDomainId::unix_epoch()
    }
}

impl ChemicalRootProjector {
    /// Project a temporally admitted chemical aggregate while revalidating the
    /// complete timing + aggregation chain.
    ///
    /// This is the only projection path that may accept an aggregate whose raw
    /// chemical components do not share one legacy `ChemicalClockDomainId`.
    pub fn project_timed_aggregation(
        &self,
        aggregation: &TimedChemicalAggregation,
    ) -> Result<TimedChemicalRootProjection, TimedChemicalRootProjectionError> {
        let (admission, input, timed_components) = match aggregation {
            TimedChemicalAggregation::Aggregated {
                admission,
                input,
                timed_components,
            } => (admission, input, timed_components),
            TimedChemicalAggregation::Abstained { admission, .. } => {
                return Err(TimedChemicalRootProjectionError::TemporalAdmissionNotPermitted {
                    status: admission.status(),
                });
            }
        };

        let recomputed_admission = classify_chemical_temporal_admission(
            timed_components,
            admission.max_component_skew_us(),
        )?;
        if recomputed_admission != *admission {
            return Err(TimedChemicalRootProjectionError::AdmissionMismatch);
        }
        if !recomputed_admission.permits_same_cycle_aggregation() {
            return Err(TimedChemicalRootProjectionError::TemporalAdmissionNotPermitted {
                status: recomputed_admission.status(),
            });
        }

        let percepts: Vec<_> = timed_components
            .iter()
            .map(|component| component.percept().clone())
            .collect();
        let recomputed_input = ChemicalModalBridge::default()
            .aggregate_after_temporal_admission(&percepts)?;
        if recomputed_input != *input {
            return Err(TimedChemicalRootProjectionError::AggregateMismatch);
        }

        let projection = project_revalidated_geometry(self, &recomputed_input)?;
        let earliest_comparison_timestamp_us = timed_components
            .iter()
            .map(TimedChemicalPercept::comparison_timestamp_us)
            .min()
            .expect("aggregated timed input is non-empty");
        let latest_comparison_timestamp_us = timed_components
            .iter()
            .map(TimedChemicalPercept::comparison_timestamp_us)
            .max()
            .expect("aggregated timed input is non-empty");

        Ok(TimedChemicalRootProjection {
            projection,
            admission: recomputed_admission,
            timed_components: timed_components.clone(),
            earliest_comparison_timestamp_us,
            latest_comparison_timestamp_us,
        })
    }
}

/// Project geometry after the temporal wrapper has independently reconstructed
/// the exact aggregate. This mirrors the small quantization kernel used by the
/// legacy projector while deliberately omitting its raw-clock validator.
fn project_revalidated_geometry(
    projector: &ChemicalRootProjector,
    input: &crate::ChemicalModalBridgeInput,
) -> Result<ChemicalRootProjection, ChemicalRootProjectionError> {
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

    // The aggregate was just rebuilt from validated raw percepts. Recheck the
    // content identity explicitly so future refactors cannot accidentally weaken
    // that implication.
    let expected_bundle = crate::ChemicalEvidenceBundleId::from_percepts(&input.components);
    if expected_bundle != input.evidence_bundle_id {
        return Err(ChemicalRootProjectionError::EvidenceBundleMismatch {
            expected: expected_bundle,
            actual: input.evidence_bundle_id,
        });
    }

    let threshold = projector.config().threshold;
    if !threshold.is_finite() {
        return Err(ChemicalRootProjectionError::NonFiniteThreshold);
    }
    let binary_vector = input.vector.to_binary(threshold);
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
            threshold,
            source_to_bipolar_similarity,
            positive_fraction,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChemicalClockDomainId, ChemicalEncodingSpaceId, ChemicalFingerprint, ChemicalModalBridgeConfig,
        ChemicalModality, ChemicalObservation, TimedChemicalPercept, aggregate_timed_chemical_percepts,
    };
    use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};
    use symthaea_time_integrity::{
        ClockDomainId, ClockEpochId, ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty,
    };
    use symthaea_time_normalization::{ClockTransformReceipt, normalize_timestamp_us};

    fn percept_in_domain(
        timestamp_us: u64,
        source: &str,
        seed: u64,
        clock_domain: &str,
    ) -> crate::ChemicalPercept {
        let mut evidence = ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            source,
            vec![],
        );
        evidence.clock_domain = Some(ChemicalClockDomainId::new(clock_domain).unwrap());
        crate::ChemicalPercept {
            evidence,
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, seed),
                confidence: 0.9,
                used_channels: 1,
                ignored_channels: 0,
                encoding_space_id: ChemicalEncodingSpaceId::from_bytes([7; 32]),
            },
        }
    }

    fn normalized(
        raw_timestamp_us: u64,
        target_timestamp_us: u64,
        source: &str,
        seed: u64,
        raw_domain: &str,
        raw_epoch: &str,
        target_domain: ClockDomainId,
        target_epoch: ClockEpochId,
    ) -> TimedChemicalPercept {
        let source_domain = ClockDomainId::new(raw_domain).unwrap();
        let source_epoch = ClockEpochId::new(raw_epoch).unwrap();
        let source_receipt = TimeIntegrityReceipt::declared(source_domain.clone())
            .with_epoch(source_epoch.clone())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(5));
        let transform = ClockTransformReceipt::offset(
            source_domain,
            source_epoch,
            target_domain,
            target_epoch,
            raw_timestamp_us,
            target_timestamp_us,
            raw_timestamp_us.saturating_sub(100),
            raw_timestamp_us.saturating_add(100),
        )
        .unwrap()
        .with_mapping_continuity(ContinuityStatus::Continuous)
        .with_target_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(5));
        let normalized = normalize_timestamp_us(raw_timestamp_us, &source_receipt, &transform)
            .unwrap();
        TimedChemicalPercept::from_normalized(
            percept_in_domain(raw_timestamp_us, source, seed, raw_domain),
            normalized,
        )
        .unwrap()
    }

    fn normalized_pair(target_domain: ClockDomainId) -> TimedChemicalAggregation {
        let target_epoch = ClockEpochId::new("capture-session-1").unwrap();
        let bridge = ChemicalModalBridge::new(ChemicalModalBridgeConfig {
            max_component_skew_us: 100,
        });
        aggregate_timed_chemical_percepts(
            &bridge,
            vec![
                normalized(
                    1_000,
                    10_000,
                    "nose-a",
                    1,
                    "nose-a/monotonic",
                    "nose-a-boot-1",
                    target_domain.clone(),
                    target_epoch.clone(),
                ),
                normalized(
                    5_000,
                    10_050,
                    "nose-b",
                    2,
                    "nose-b/monotonic",
                    "nose-b-boot-9",
                    target_domain,
                    target_epoch,
                ),
            ],
        )
        .unwrap()
    }

    #[test]
    fn mixed_raw_clocks_project_only_with_attached_generic_timing_evidence() {
        let aggregation = normalized_pair(ClockDomainId::new("capture-host/monotonic").unwrap());
        let raw_input = aggregation.input().unwrap();
        assert!(raw_input.clock_domain.is_none());
        // Detaching the aggregate correctly fails the legacy raw-clock validator.
        assert!(matches!(
            ChemicalRootProjector::default().project(raw_input),
            Err(ChemicalRootProjectionError::MixedClockDomains { .. })
        ));

        let timed = ChemicalRootProjector::default()
            .project_timed_aggregation(&aggregation)
            .unwrap();
        assert_eq!(
            timed.admission().status(),
            ChemicalTemporalAdmissionStatus::DefinitelyWithin
        );
        assert_eq!(timed.earliest_comparison_timestamp_us(), 10_000);
        assert_eq!(timed.latest_comparison_timestamp_us(), 10_050);
        assert_eq!(
            timed.comparison_clock_domain().as_str(),
            "capture-host/monotonic"
        );
        assert_eq!(timed.projection().component_count, 2);
        assert!(timed.projection().clock_domain.is_none());
    }

    #[test]
    fn timed_projection_matches_legacy_projection_when_raw_time_is_already_valid() {
        let domain = ClockDomainId::new("shared/monotonic").unwrap();
        let epoch = ClockEpochId::new("shared-boot-1").unwrap();
        let receipt = |timestamp_us: u64| {
            let percept = percept_in_domain(timestamp_us, "nose", timestamp_us, "shared/monotonic");
            TimedChemicalPercept::new(
                percept,
                TimeIntegrityReceipt::declared(domain.clone())
                    .with_epoch(epoch.clone())
                    .with_continuity(ContinuityStatus::Continuous)
                    .with_uncertainty(TimeUncertainty::bounded(5)),
            )
            .unwrap()
        };
        let bridge = ChemicalModalBridge::new(ChemicalModalBridgeConfig {
            max_component_skew_us: 100,
        });
        let aggregation = aggregate_timed_chemical_percepts(
            &bridge,
            vec![receipt(1_000), receipt(1_050)],
        )
        .unwrap();
        let legacy = ChemicalRootProjector::default()
            .project(aggregation.input().unwrap())
            .unwrap();
        let timed = ChemicalRootProjector::default()
            .project_timed_aggregation(&aggregation)
            .unwrap();
        assert_eq!(timed.projection(), &legacy);
    }

    #[test]
    fn tampered_stored_aggregate_is_rejected_before_projection() {
        let mut aggregation = normalized_pair(ClockDomainId::new("capture-host/monotonic").unwrap());
        match &mut aggregation {
            TimedChemicalAggregation::Aggregated { input, .. } => {
                input.confidence *= 0.5;
            }
            TimedChemicalAggregation::Abstained { .. } => panic!("fixture must aggregate"),
        }
        assert!(matches!(
            ChemicalRootProjector::default().project_timed_aggregation(&aggregation),
            Err(TimedChemicalRootProjectionError::AggregateMismatch)
        ));
    }

    #[test]
    fn unix_normalized_comparison_is_explicit_without_rewriting_raw_source_time() {
        let aggregation = normalized_pair(ClockDomainId::unix_epoch());
        let timed = ChemicalRootProjector::default()
            .project_timed_aggregation(&aggregation)
            .unwrap();
        assert!(timed.comparison_is_unix_epoch());
        assert_eq!(timed.latest_comparison_timestamp_us(), 10_050);
        assert_eq!(timed.projection().latest_timestamp_us, 5_000);
        assert!(timed.projection().clock_domain.is_none());
    }

    #[test]
    fn abstained_timing_cannot_be_projected() {
        let domain = ClockDomainId::new("shared/monotonic").unwrap();
        let epoch = ClockEpochId::new("shared-boot-1").unwrap();
        let timed = |timestamp_us: u64, error_us: u64| {
            TimedChemicalPercept::new(
                percept_in_domain(timestamp_us, "nose", timestamp_us, "shared/monotonic"),
                TimeIntegrityReceipt::declared(domain.clone())
                    .with_epoch(epoch.clone())
                    .with_continuity(ContinuityStatus::Continuous)
                    .with_uncertainty(TimeUncertainty::bounded(error_us)),
            )
            .unwrap()
        };
        let bridge = ChemicalModalBridge::new(ChemicalModalBridgeConfig {
            max_component_skew_us: 100,
        });
        let aggregation = aggregate_timed_chemical_percepts(
            &bridge,
            vec![timed(1_000, 60), timed(1_100, 60)],
        )
        .unwrap();
        assert_eq!(aggregation.admission().status(), ChemicalTemporalAdmissionStatus::Ambiguous);
        assert!(matches!(
            ChemicalRootProjector::default().project_timed_aggregation(&aggregation),
            Err(TimedChemicalRootProjectionError::TemporalAdmissionNotPermitted {
                status: ChemicalTemporalAdmissionStatus::Ambiguous
            })
        ));
    }
}
