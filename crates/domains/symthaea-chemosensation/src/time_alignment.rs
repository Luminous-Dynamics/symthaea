// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic, evidence-bearing temporal admission for chemical sensor fusion.
//!
//! Raw [`crate::ChemicalObservation`] values retain their legacy
//! [`crate::ChemicalClockDomainId`] metadata for compatibility and stable content
//! identity. Physical acquisition can attach generic timing evidence to the
//! derived percept without mutating that raw evidence.
//!
//! A [`TimedChemicalPercept`] distinguishes the immutable raw acquisition
//! timestamp from the timestamp actually used for cross-source comparison. The
//! default constructor compares in the raw clock domain. Evidence-bound clock
//! normalization may instead attach a [`NormalizedTimePoint`], preserving the
//! source timestamp and transform while comparing in the target clock domain.
//!
//! Multi-source admission uses [`bounded_separation_window_us`] pairwise. A set
//! is admitted as definitely simultaneous only when every pair's *maximum*
//! possible separation is within the configured skew threshold. If no pair is
//! definitely outside but at least one interval crosses the threshold, the
//! result is `Ambiguous` rather than a fabricated boolean decision.

use std::fmt;

use symthaea_time_integrity::{
    ClockDomainId, ClockEpochId, SeparationWindowUs, TimeComparisonError,
    TimeIntegrityReceipt, bounded_separation_window_us,
};
use symthaea_time_normalization::NormalizedTimePoint;

use crate::{ChemicalClockDomainId, ChemicalObservationId, ChemicalPercept};

/// One chemical percept plus independent generic timing evidence.
///
/// `percept.evidence.timestamp_us` always remains the original acquisition
/// timestamp. `comparison_timestamp_us` is the timestamp whose domain/epoch and
/// uncertainty are described by `time`. When normalization is present, the
/// complete [`NormalizedTimePoint`] is retained so the target timestamp cannot
/// become detached from its source receipt and transform.
///
/// Attaching timing evidence does not mutate the underlying chemical observation
/// and therefore does not change its [`ChemicalObservationId`].
#[derive(Debug, Clone, PartialEq)]
pub struct TimedChemicalPercept {
    percept: ChemicalPercept,
    comparison_timestamp_us: u64,
    time: TimeIntegrityReceipt,
    normalization: Option<NormalizedTimePoint>,
}

impl TimedChemicalPercept {
    /// Attach timing evidence that describes the raw acquisition timestamp.
    pub fn new(
        percept: ChemicalPercept,
        time: TimeIntegrityReceipt,
    ) -> Result<Self, ChemicalTimeAlignmentError> {
        validate_legacy_source_domain(&percept, &time.clock_domain)?;
        let comparison_timestamp_us = percept.timestamp_us();
        Ok(Self {
            percept,
            comparison_timestamp_us,
            time,
            normalization: None,
        })
    }

    /// Attach an already normalized, auditable comparison timestamp.
    ///
    /// The normalization source timestamp must equal the immutable chemical
    /// acquisition timestamp exactly. Legacy chemical clock metadata, when
    /// present, is checked against the *source* receipt rather than the target
    /// receipt. The target timestamp/receipt then become the comparison pair.
    pub fn from_normalized(
        percept: ChemicalPercept,
        normalization: NormalizedTimePoint,
    ) -> Result<Self, ChemicalTimeAlignmentError> {
        if percept.timestamp_us() != normalization.source_timestamp_us() {
            return Err(ChemicalTimeAlignmentError::NormalizedSourceTimestampMismatch {
                observation_timestamp_us: percept.timestamp_us(),
                normalized_source_timestamp_us: normalization.source_timestamp_us(),
            });
        }
        validate_legacy_source_domain(
            &percept,
            &normalization.source_receipt().clock_domain,
        )?;

        let comparison_timestamp_us = normalization.target_timestamp_us();
        let time = normalization.target_receipt().clone();
        Ok(Self {
            percept,
            comparison_timestamp_us,
            time,
            normalization: Some(normalization),
        })
    }

    pub fn percept(&self) -> &ChemicalPercept {
        &self.percept
    }

    pub fn into_percept(self) -> ChemicalPercept {
        self.percept
    }

    /// Timestamp used by strict cross-source temporal comparison.
    pub fn comparison_timestamp_us(&self) -> u64 {
        self.comparison_timestamp_us
    }

    /// Timing receipt that describes [`Self::comparison_timestamp_us`].
    pub fn time(&self) -> &TimeIntegrityReceipt {
        &self.time
    }

    /// Full source->target normalization provenance, when comparison occurs in a
    /// normalized target timebase rather than the raw acquisition timebase.
    pub fn normalization(&self) -> Option<&NormalizedTimePoint> {
        self.normalization.as_ref()
    }

    pub fn observation_id(&self) -> ChemicalObservationId {
        self.percept.observation_id()
    }

    fn source_clock_domain(&self) -> &ClockDomainId {
        match self.normalization.as_ref() {
            Some(normalized) => &normalized.source_receipt().clock_domain,
            None => &self.time.clock_domain,
        }
    }
}

fn validate_legacy_source_domain(
    percept: &ChemicalPercept,
    generic_source_domain: &ClockDomainId,
) -> Result<(), ChemicalTimeAlignmentError> {
    if let Some(legacy) = percept.evidence.clock_domain.as_ref() {
        if legacy.as_str() != generic_source_domain.as_str() {
            return Err(ChemicalTimeAlignmentError::LegacyClockDomainMismatch {
                legacy: legacy.clone(),
                generic: generic_source_domain.clone(),
            });
        }
    }
    Ok(())
}

/// Result of asking whether one set of chemical percepts may be treated as one
/// same-cycle observation under a stated skew threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChemicalTemporalAdmissionStatus {
    /// One source needs no cross-source simultaneity claim.
    NoComparisonNeeded,
    /// Every pair is definitely within the requested skew bound.
    DefinitelyWithin,
    /// No pair is definitely outside, but at least one uncertainty interval
    /// crosses the requested skew bound.
    Ambiguous,
    /// At least one pair is definitely farther apart than the requested bound.
    DefinitelyOutside,
}

impl ChemicalTemporalAdmissionStatus {
    pub const fn permits_same_cycle_aggregation(self) -> bool {
        matches!(self, Self::NoComparisonNeeded | Self::DefinitelyWithin)
    }
}

/// Auditable pairwise temporal evidence retained by the admission result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChemicalPairwiseTimeWindow {
    pub left_index: usize,
    pub right_index: usize,
    pub separation: SeparationWindowUs,
}

/// Evidence-bearing temporal admission result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChemicalTemporalAdmission {
    status: ChemicalTemporalAdmissionStatus,
    max_component_skew_us: u64,
    clock_domain: ClockDomainId,
    clock_epoch: Option<ClockEpochId>,
    pairwise_windows: Vec<ChemicalPairwiseTimeWindow>,
}

impl ChemicalTemporalAdmission {
    pub fn status(&self) -> ChemicalTemporalAdmissionStatus {
        self.status
    }

    pub fn max_component_skew_us(&self) -> u64 {
        self.max_component_skew_us
    }

    pub fn clock_domain(&self) -> &ClockDomainId {
        &self.clock_domain
    }

    pub fn clock_epoch(&self) -> Option<&ClockEpochId> {
        self.clock_epoch.as_ref()
    }

    pub fn pairwise_windows(&self) -> &[ChemicalPairwiseTimeWindow] {
        &self.pairwise_windows
    }

    pub fn permits_same_cycle_aggregation(&self) -> bool {
        self.status.permits_same_cycle_aggregation()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalTimeAlignmentError {
    EmptyInput,
    LegacyClockDomainMismatch {
        legacy: ChemicalClockDomainId,
        generic: ClockDomainId,
    },
    NormalizedSourceTimestampMismatch {
        observation_timestamp_us: u64,
        normalized_source_timestamp_us: u64,
    },
    Comparison(TimeComparisonError),
}

impl fmt::Display for ChemicalTimeAlignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "chemical temporal admission requires at least one percept"),
            Self::LegacyClockDomainMismatch { legacy, generic } => write!(
                f,
                "legacy chemical clock domain {legacy} disagrees with generic source time domain {generic}"
            ),
            Self::NormalizedSourceTimestampMismatch {
                observation_timestamp_us,
                normalized_source_timestamp_us,
            } => write!(
                f,
                "normalized source timestamp {normalized_source_timestamp_us} does not match chemical acquisition timestamp {observation_timestamp_us}"
            ),
            Self::Comparison(error) => write!(f, "chemical timing evidence is insufficient: {error}"),
        }
    }
}

impl std::error::Error for ChemicalTimeAlignmentError {}

impl From<TimeComparisonError> for ChemicalTimeAlignmentError {
    fn from(value: TimeComparisonError) -> Self {
        Self::Comparison(value)
    }
}

/// Classify whether a set of chemical percepts can support one same-cycle
/// multi-source observation.
///
/// This function does not aggregate HDC vectors. It is a temporal evidence gate
/// that should run before [`crate::ChemicalModalBridge`] aggregation for physical
/// multi-source acquisition. `Ambiguous` preserves the evidence while refusing
/// to assert simultaneity.
pub fn classify_chemical_temporal_admission(
    components: &[TimedChemicalPercept],
    max_component_skew_us: u64,
) -> Result<ChemicalTemporalAdmission, ChemicalTimeAlignmentError> {
    let first = components
        .first()
        .ok_or(ChemicalTimeAlignmentError::EmptyInput)?;

    // Recheck source-domain compatibility even after normalization. The generic
    // comparison receipt may be in a target domain, while legacy chemical clock
    // metadata still describes the immutable raw acquisition timestamp.
    for component in components {
        validate_legacy_source_domain(&component.percept, component.source_clock_domain())?;
    }

    if components.len() == 1 {
        return Ok(ChemicalTemporalAdmission {
            status: ChemicalTemporalAdmissionStatus::NoComparisonNeeded,
            max_component_skew_us,
            clock_domain: first.time.clock_domain.clone(),
            clock_epoch: first.time.clock_epoch.clone(),
            pairwise_windows: Vec::new(),
        });
    }

    let mut pairwise_windows = Vec::new();
    for left_index in 0..components.len() {
        for right_index in (left_index + 1)..components.len() {
            let left = &components[left_index];
            let right = &components[right_index];
            let separation = bounded_separation_window_us(
                left.comparison_timestamp_us,
                &left.time,
                right.comparison_timestamp_us,
                &right.time,
            )?;
            pairwise_windows.push(ChemicalPairwiseTimeWindow {
                left_index,
                right_index,
                separation,
            });
        }
    }

    let any_definitely_outside = pairwise_windows
        .iter()
        .any(|pair| pair.separation.minimum_us > max_component_skew_us);
    let all_definitely_within = pairwise_windows
        .iter()
        .all(|pair| pair.separation.maximum_us <= max_component_skew_us);

    let status = if any_definitely_outside {
        ChemicalTemporalAdmissionStatus::DefinitelyOutside
    } else if all_definitely_within {
        ChemicalTemporalAdmissionStatus::DefinitelyWithin
    } else {
        ChemicalTemporalAdmissionStatus::Ambiguous
    };

    Ok(ChemicalTemporalAdmission {
        status,
        max_component_skew_us,
        clock_domain: first.time.clock_domain.clone(),
        clock_epoch: first.time.clock_epoch.clone(),
        pairwise_windows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChemicalEncodingSpaceId, ChemicalFingerprint, ChemicalModality, ChemicalObservation,
    };
    use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};
    use symthaea_time_integrity::{ContinuityStatus, TimeUncertainty};
    use symthaea_time_normalization::{ClockTransformReceipt, normalize_timestamp_us};

    fn generic_domain() -> ClockDomainId {
        ClockDomainId::new("rig-01/monotonic").unwrap()
    }

    fn epoch() -> ClockEpochId {
        ClockEpochId::new("rig-01-boot-7").unwrap()
    }

    fn target_domain() -> ClockDomainId {
        ClockDomainId::new("capture-host/monotonic").unwrap()
    }

    fn target_epoch() -> ClockEpochId {
        ClockEpochId::new("capture-host-boot-3").unwrap()
    }

    fn receipt(error_us: u64) -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(generic_domain())
            .with_epoch(epoch())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(error_us))
    }

    fn percept(timestamp_us: u64, source: &str) -> ChemicalPercept {
        percept_in_domain(timestamp_us, source, "rig-01/monotonic", timestamp_us + 1)
    }

    fn percept_in_domain(
        timestamp_us: u64,
        source: &str,
        clock_domain: &str,
        seed: u64,
    ) -> ChemicalPercept {
        let mut evidence = ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            source,
            vec![],
        );
        evidence.clock_domain = Some(ChemicalClockDomainId::new(clock_domain).unwrap());
        ChemicalPercept {
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

    fn timed(timestamp_us: u64, error_us: u64, source: &str) -> TimedChemicalPercept {
        TimedChemicalPercept::new(percept(timestamp_us, source), receipt(error_us)).unwrap()
    }

    fn transform(
        source_domain: ClockDomainId,
        source_epoch: ClockEpochId,
        source_anchor_us: u64,
        target_anchor_us: u64,
    ) -> ClockTransformReceipt {
        ClockTransformReceipt::offset(
            source_domain,
            source_epoch,
            target_domain(),
            target_epoch(),
            source_anchor_us,
            target_anchor_us,
            source_anchor_us.saturating_sub(100),
            source_anchor_us.saturating_add(100),
        )
        .unwrap()
        .with_mapping_continuity(ContinuityStatus::Continuous)
        .with_target_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(5))
    }

    #[test]
    fn timing_sidecar_does_not_change_raw_evidence_identity() {
        let percept = percept(1_000, "nose-a");
        let before = percept.observation_id();
        let timed = TimedChemicalPercept::new(percept, receipt(10)).unwrap();
        assert_eq!(timed.observation_id(), before);
        assert_eq!(timed.comparison_timestamp_us(), 1_000);
        assert!(timed.normalization().is_none());
    }

    #[test]
    fn legacy_clock_metadata_must_not_contradict_generic_receipt() {
        let percept = percept(1_000, "nose-a");
        let time = TimeIntegrityReceipt::declared(ClockDomainId::new("other-rig/monotonic").unwrap())
            .with_epoch(epoch())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(10));
        assert!(matches!(
            TimedChemicalPercept::new(percept, time),
            Err(ChemicalTimeAlignmentError::LegacyClockDomainMismatch { .. })
        ));
    }

    #[test]
    fn normalized_sidecar_preserves_raw_time_and_identity() {
        let percept = percept(1_000, "nose-a");
        let before = percept.observation_id();
        let normalized = normalize_timestamp_us(
            1_000,
            &receipt(10),
            &transform(generic_domain(), epoch(), 1_000, 5_000),
        )
        .unwrap();
        let timed = TimedChemicalPercept::from_normalized(percept, normalized).unwrap();

        assert_eq!(timed.percept().timestamp_us(), 1_000);
        assert_eq!(timed.comparison_timestamp_us(), 5_000);
        assert_eq!(timed.time().clock_domain, target_domain());
        assert_eq!(timed.observation_id(), before);
        assert_eq!(
            timed.normalization().unwrap().source_timestamp_us(),
            timed.percept().timestamp_us()
        );
    }

    #[test]
    fn normalization_must_belong_to_the_exact_raw_timestamp() {
        let percept = percept(1_001, "nose-a");
        let normalized = normalize_timestamp_us(
            1_000,
            &receipt(10),
            &transform(generic_domain(), epoch(), 1_000, 5_000),
        )
        .unwrap();
        assert!(matches!(
            TimedChemicalPercept::from_normalized(percept, normalized),
            Err(ChemicalTimeAlignmentError::NormalizedSourceTimestampMismatch { .. })
        ));
    }

    #[test]
    fn normalized_devices_compare_target_time_not_incomparable_raw_time() {
        let domain_a = ClockDomainId::new("nose-a/monotonic").unwrap();
        let epoch_a = ClockEpochId::new("nose-a-boot-1").unwrap();
        let domain_b = ClockDomainId::new("nose-b/monotonic").unwrap();
        let epoch_b = ClockEpochId::new("nose-b-boot-9").unwrap();

        let source_a = TimeIntegrityReceipt::declared(domain_a.clone())
            .with_epoch(epoch_a.clone())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(5));
        let source_b = TimeIntegrityReceipt::declared(domain_b.clone())
            .with_epoch(epoch_b.clone())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(5));

        let normalized_a = normalize_timestamp_us(
            1_000,
            &source_a,
            &transform(domain_a, epoch_a, 1_000, 10_000),
        )
        .unwrap();
        let normalized_b = normalize_timestamp_us(
            5_000,
            &source_b,
            &transform(domain_b, epoch_b, 5_000, 10_050),
        )
        .unwrap();

        let a = TimedChemicalPercept::from_normalized(
            percept_in_domain(1_000, "nose-a", "nose-a/monotonic", 1),
            normalized_a,
        )
        .unwrap();
        let b = TimedChemicalPercept::from_normalized(
            percept_in_domain(5_000, "nose-b", "nose-b/monotonic", 2),
            normalized_b,
        )
        .unwrap();

        let result = classify_chemical_temporal_admission(&[a, b], 100).unwrap();
        assert_eq!(result.status(), ChemicalTemporalAdmissionStatus::DefinitelyWithin);
        assert_eq!(result.pairwise_windows()[0].separation.nominal_us, 50);
        assert_eq!(result.clock_domain(), &target_domain());
    }

    #[test]
    fn single_source_needs_no_cross_source_time_claim() {
        let percept = percept(1_000, "nose-a");
        let weak_time = TimeIntegrityReceipt::declared(generic_domain());
        let timed = TimedChemicalPercept::new(percept, weak_time).unwrap();
        let result = classify_chemical_temporal_admission(&[timed], 100).unwrap();
        assert_eq!(
            result.status(),
            ChemicalTemporalAdmissionStatus::NoComparisonNeeded
        );
        assert!(result.permits_same_cycle_aggregation());
        assert!(result.pairwise_windows().is_empty());
    }

    #[test]
    fn bounded_pair_definitely_within_is_admissible() {
        let result = classify_chemical_temporal_admission(
            &[timed(1_000, 10, "nose-a"), timed(1_100, 10, "nose-b")],
            150,
        )
        .unwrap();
        assert_eq!(result.status(), ChemicalTemporalAdmissionStatus::DefinitelyWithin);
        assert_eq!(result.pairwise_windows()[0].separation.minimum_us, 80);
        assert_eq!(result.pairwise_windows()[0].separation.maximum_us, 120);
        assert!(result.permits_same_cycle_aggregation());
    }

    #[test]
    fn threshold_crossing_uncertainty_is_ambiguous_not_a_boolean_pass() {
        let result = classify_chemical_temporal_admission(
            &[timed(1_000, 40, "nose-a"), timed(1_100, 40, "nose-b")],
            150,
        )
        .unwrap();
        assert_eq!(result.status(), ChemicalTemporalAdmissionStatus::Ambiguous);
        assert_eq!(result.pairwise_windows()[0].separation.minimum_us, 20);
        assert_eq!(result.pairwise_windows()[0].separation.maximum_us, 180);
        assert!(!result.permits_same_cycle_aggregation());
    }

    #[test]
    fn definitely_outside_pair_is_not_same_cycle_admissible() {
        let result = classify_chemical_temporal_admission(
            &[timed(1_000, 10, "nose-a"), timed(1_300, 10, "nose-b")],
            150,
        )
        .unwrap();
        assert_eq!(
            result.status(),
            ChemicalTemporalAdmissionStatus::DefinitelyOutside
        );
        assert_eq!(result.pairwise_windows()[0].separation.minimum_us, 280);
        assert!(!result.permits_same_cycle_aggregation());
    }

    #[test]
    fn multi_source_bounded_comparison_requires_explicit_epoch() {
        let a = timed(1_000, 10, "nose-a");
        let mut weak = receipt(10);
        weak.clock_epoch = None;
        let b = TimedChemicalPercept::new(percept(1_100, "nose-b"), weak).unwrap();
        let error = classify_chemical_temporal_admission(&[a, b], 150).unwrap_err();
        assert!(matches!(
            error,
            ChemicalTimeAlignmentError::Comparison(TimeComparisonError::MissingClockEpoch { .. })
        ));
    }
}
