// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit clock-domain normalization with bounded error propagation.
//!
//! This crate does not synchronize clocks and does not authenticate timing
//! claims. It represents a validated, finite-window mapping from one declared
//! clock domain/continuity epoch into another and propagates the source and
//! transform uncertainty into a target-domain [`TimeIntegrityReceipt`].

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};
use symthaea_time_integrity::{
    ClockDomainId, ClockEpochId, ContinuityStatus, SeparationWindowUs, TimeIntegrityReceipt,
    TimeUncertainty,
};

/// Transform model supported by the v1 normalization contract.
///
/// `Offset` assumes equal clock rate within the receipt's validity interval.
/// Drift, calibration residual, and synchronization error must therefore be
/// covered by the transform's finite uncertainty bound. A future model may add
/// explicit rate/skew terms without changing the meaning of this variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockTransformModel {
    Offset {
        source_anchor_us: u64,
        target_anchor_us: u64,
    },
}

impl ClockTransformModel {
    fn source_anchor_us(&self) -> u64 {
        match self {
            Self::Offset {
                source_anchor_us, ..
            } => *source_anchor_us,
        }
    }

    fn map_timestamp_us(&self, source_timestamp_us: u64) -> Result<u64, ClockTransformError> {
        match self {
            Self::Offset {
                source_anchor_us,
                target_anchor_us,
            } => {
                if source_timestamp_us >= *source_anchor_us {
                    let delta = source_timestamp_us - *source_anchor_us;
                    target_anchor_us
                        .checked_add(delta)
                        .ok_or(ClockTransformError::TargetTimestampOverflow)
                } else {
                    let delta = *source_anchor_us - source_timestamp_us;
                    target_anchor_us
                        .checked_sub(delta)
                        .ok_or(ClockTransformError::TargetTimestampUnderflow)
                }
            }
        }
    }
}

/// Validated claim that timestamps in one clock domain can be normalized into
/// another over a finite source-time interval.
///
/// This is evidence metadata, not synchronization or authenticity proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClockTransformReceipt {
    source_domain: ClockDomainId,
    source_epoch: ClockEpochId,
    target_domain: ClockDomainId,
    target_epoch: ClockEpochId,
    model: ClockTransformModel,
    valid_source_start_us: u64,
    valid_source_end_us: u64,
    continuity: ContinuityStatus,
    uncertainty: TimeUncertainty,
    sequence: Option<u64>,
}

impl ClockTransformReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn offset(
        source_domain: ClockDomainId,
        source_epoch: ClockEpochId,
        target_domain: ClockDomainId,
        target_epoch: ClockEpochId,
        source_anchor_us: u64,
        target_anchor_us: u64,
        valid_source_start_us: u64,
        valid_source_end_us: u64,
    ) -> Result<Self, ClockTransformError> {
        let receipt = Self {
            source_domain,
            source_epoch,
            target_domain,
            target_epoch,
            model: ClockTransformModel::Offset {
                source_anchor_us,
                target_anchor_us,
            },
            valid_source_start_us,
            valid_source_end_us,
            continuity: ContinuityStatus::Unverified,
            uncertainty: TimeUncertainty::Unbounded,
            sequence: None,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn with_continuity(mut self, continuity: ContinuityStatus) -> Self {
        self.continuity = continuity;
        self
    }

    pub fn with_uncertainty(mut self, uncertainty: TimeUncertainty) -> Self {
        self.uncertainty = uncertainty;
        self
    }

    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = Some(sequence);
        self
    }

    pub fn source_domain(&self) -> &ClockDomainId {
        &self.source_domain
    }

    pub fn source_epoch(&self) -> &ClockEpochId {
        &self.source_epoch
    }

    pub fn target_domain(&self) -> &ClockDomainId {
        &self.target_domain
    }

    pub fn target_epoch(&self) -> &ClockEpochId {
        &self.target_epoch
    }

    pub fn model(&self) -> &ClockTransformModel {
        &self.model
    }

    pub fn valid_source_range_us(&self) -> (u64, u64) {
        (self.valid_source_start_us, self.valid_source_end_us)
    }

    pub fn continuity(&self) -> ContinuityStatus {
        self.continuity
    }

    pub fn uncertainty(&self) -> TimeUncertainty {
        self.uncertainty
    }

    pub fn sequence(&self) -> Option<u64> {
        self.sequence
    }

    pub fn validate(&self) -> Result<(), ClockTransformError> {
        if self.valid_source_start_us > self.valid_source_end_us {
            return Err(ClockTransformError::InvalidValidityRange {
                start_us: self.valid_source_start_us,
                end_us: self.valid_source_end_us,
            });
        }
        let anchor = self.model.source_anchor_us();
        if anchor < self.valid_source_start_us || anchor > self.valid_source_end_us {
            return Err(ClockTransformError::AnchorOutsideValidity {
                anchor_us: anchor,
                start_us: self.valid_source_start_us,
                end_us: self.valid_source_end_us,
            });
        }
        Ok(())
    }

    fn contains_source_timestamp(&self, timestamp_us: u64) -> bool {
        timestamp_us >= self.valid_source_start_us && timestamp_us <= self.valid_source_end_us
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawClockTransformReceipt {
    source_domain: ClockDomainId,
    source_epoch: ClockEpochId,
    target_domain: ClockDomainId,
    target_epoch: ClockEpochId,
    model: ClockTransformModel,
    valid_source_start_us: u64,
    valid_source_end_us: u64,
    continuity: ContinuityStatus,
    uncertainty: TimeUncertainty,
    #[serde(default)]
    sequence: Option<u64>,
}

impl<'de> Deserialize<'de> for ClockTransformReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawClockTransformReceipt::deserialize(deserializer)?;
        let receipt = Self {
            source_domain: raw.source_domain,
            source_epoch: raw.source_epoch,
            target_domain: raw.target_domain,
            target_epoch: raw.target_epoch,
            model: raw.model,
            valid_source_start_us: raw.valid_source_start_us,
            valid_source_end_us: raw.valid_source_end_us,
            continuity: raw.continuity,
            uncertainty: raw.uncertainty,
            sequence: raw.sequence,
        };
        receipt.validate().map_err(de::Error::custom)?;
        Ok(receipt)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformEvidenceSide {
    Source,
    Transform,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockTransformError {
    InvalidValidityRange {
        start_us: u64,
        end_us: u64,
    },
    AnchorOutsideValidity {
        anchor_us: u64,
        start_us: u64,
        end_us: u64,
    },
    SourceClockDomainMismatch {
        receipt: ClockDomainId,
        transform: ClockDomainId,
    },
    SourceClockEpochMissing,
    SourceClockEpochMismatch {
        receipt: ClockEpochId,
        transform: ClockEpochId,
    },
    SourceTimestampOutsideValidity {
        timestamp_us: u64,
        start_us: u64,
        end_us: u64,
    },
    ContinuityNotEstablished {
        side: TransformEvidenceSide,
        status: ContinuityStatus,
    },
    UnboundedUncertainty {
        side: TransformEvidenceSide,
    },
    TargetTimestampUnderflow,
    TargetTimestampOverflow,
}

impl fmt::Display for ClockTransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValidityRange { start_us, end_us } => write!(
                f,
                "clock transform validity range is inverted: {start_us}..={end_us}"
            ),
            Self::AnchorOutsideValidity {
                anchor_us,
                start_us,
                end_us,
            } => write!(
                f,
                "clock transform source anchor {anchor_us} lies outside validity range {start_us}..={end_us}"
            ),
            Self::SourceClockDomainMismatch { receipt, transform } => write!(
                f,
                "source clock domain does not match transform: {receipt} != {transform}"
            ),
            Self::SourceClockEpochMissing => {
                write!(f, "source timestamp has no explicit continuity epoch")
            }
            Self::SourceClockEpochMismatch { receipt, transform } => write!(
                f,
                "source clock epoch does not match transform: {receipt} != {transform}"
            ),
            Self::SourceTimestampOutsideValidity {
                timestamp_us,
                start_us,
                end_us,
            } => write!(
                f,
                "source timestamp {timestamp_us} lies outside transform validity range {start_us}..={end_us}"
            ),
            Self::ContinuityNotEstablished { side, status } => write!(
                f,
                "{side:?} continuity is not established: {status:?}"
            ),
            Self::UnboundedUncertainty { side } => {
                write!(f, "{side:?} timing evidence has no finite uncertainty bound")
            }
            Self::TargetTimestampUnderflow => {
                write!(f, "clock transform would produce a negative target timestamp")
            }
            Self::TargetTimestampOverflow => {
                write!(f, "clock transform would overflow the target timestamp")
            }
        }
    }
}

impl std::error::Error for ClockTransformError {}

/// A source timestamp plus its complete, auditable normalization result.
///
/// The original source receipt and transform remain attached. The derived
/// target receipt deliberately does not copy the source sequence number into a
/// different clock domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NormalizedTimePoint {
    source_timestamp_us: u64,
    source_receipt: TimeIntegrityReceipt,
    target_timestamp_us: u64,
    target_receipt: TimeIntegrityReceipt,
    transform: ClockTransformReceipt,
}

impl NormalizedTimePoint {
    pub fn source_timestamp_us(&self) -> u64 {
        self.source_timestamp_us
    }

    pub fn source_receipt(&self) -> &TimeIntegrityReceipt {
        &self.source_receipt
    }

    pub fn target_timestamp_us(&self) -> u64 {
        self.target_timestamp_us
    }

    pub fn target_receipt(&self) -> &TimeIntegrityReceipt {
        &self.target_receipt
    }

    pub fn transform(&self) -> &ClockTransformReceipt {
        &self.transform
    }
}

/// Normalize one bounded, continuous source timestamp into the transform's
/// target domain.
///
/// The operation fails closed unless:
///
/// - source domain and explicit source epoch match the transform;
/// - the timestamp lies within the transform's finite validity interval;
/// - source and transform continuity are both established;
/// - source and transform both provide finite uncertainty bounds;
/// - applying the transform cannot underflow or overflow `u64`.
///
/// The target uncertainty is the saturating sum of the two supplied error
/// bounds. The target sequence is left unset because a source-domain sequence
/// is not automatically meaningful in the target domain.
pub fn normalize_timestamp_us(
    source_timestamp_us: u64,
    source: &TimeIntegrityReceipt,
    transform: &ClockTransformReceipt,
) -> Result<NormalizedTimePoint, ClockTransformError> {
    transform.validate()?;

    if source.clock_domain != *transform.source_domain() {
        return Err(ClockTransformError::SourceClockDomainMismatch {
            receipt: source.clock_domain.clone(),
            transform: transform.source_domain().clone(),
        });
    }

    let source_epoch = source
        .clock_epoch
        .as_ref()
        .ok_or(ClockTransformError::SourceClockEpochMissing)?;
    if source_epoch != transform.source_epoch() {
        return Err(ClockTransformError::SourceClockEpochMismatch {
            receipt: source_epoch.clone(),
            transform: transform.source_epoch().clone(),
        });
    }

    if !transform.contains_source_timestamp(source_timestamp_us) {
        let (start_us, end_us) = transform.valid_source_range_us();
        return Err(ClockTransformError::SourceTimestampOutsideValidity {
            timestamp_us: source_timestamp_us,
            start_us,
            end_us,
        });
    }

    if source.continuity != ContinuityStatus::Continuous {
        return Err(ClockTransformError::ContinuityNotEstablished {
            side: TransformEvidenceSide::Source,
            status: source.continuity,
        });
    }
    if transform.continuity() != ContinuityStatus::Continuous {
        return Err(ClockTransformError::ContinuityNotEstablished {
            side: TransformEvidenceSide::Transform,
            status: transform.continuity(),
        });
    }

    let source_error_us = source
        .uncertainty
        .max_error_us()
        .ok_or(ClockTransformError::UnboundedUncertainty {
            side: TransformEvidenceSide::Source,
        })?;
    let transform_error_us = transform
        .uncertainty()
        .max_error_us()
        .ok_or(ClockTransformError::UnboundedUncertainty {
            side: TransformEvidenceSide::Transform,
        })?;

    let target_timestamp_us = transform.model().map_timestamp_us(source_timestamp_us)?;
    let target_receipt = TimeIntegrityReceipt::declared(transform.target_domain().clone())
        .with_epoch(transform.target_epoch().clone())
        .with_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(
            source_error_us.saturating_add(transform_error_us),
        ));

    Ok(NormalizedTimePoint {
        source_timestamp_us,
        source_receipt: source.clone(),
        target_timestamp_us,
        target_receipt,
        transform: transform.clone(),
    })
}

/// Result of asking whether an uncertainty interval lies within a temporal
/// threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalThresholdDecision {
    /// Even the worst-case separation is within the threshold.
    DefinitelyWithin,
    /// The uncertainty interval crosses the threshold; the evidence cannot
    /// determine which side is true.
    Ambiguous,
    /// Even the best-case separation exceeds the threshold.
    DefinitelyOutside,
}

/// Classify an already bounded separation window against an inclusive maximum
/// allowed separation.
///
/// This deliberately avoids a boolean `within()` helper, because a boolean can
/// collapse the scientifically important ambiguous case.
pub fn classify_separation_window(
    window: SeparationWindowUs,
    max_allowed_us: u64,
) -> TemporalThresholdDecision {
    if window.maximum_us <= max_allowed_us {
        TemporalThresholdDecision::DefinitelyWithin
    } else if window.minimum_us > max_allowed_us {
        TemporalThresholdDecision::DefinitelyOutside
    } else {
        TemporalThresholdDecision::Ambiguous
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_time_integrity::bounded_separation_window_us;

    fn source_domain() -> ClockDomainId {
        ClockDomainId::new("device-a/monotonic").unwrap()
    }

    fn source_epoch() -> ClockEpochId {
        ClockEpochId::new("device-a/boot-7").unwrap()
    }

    fn target_domain() -> ClockDomainId {
        ClockDomainId::unix_epoch()
    }

    fn target_epoch() -> ClockEpochId {
        ClockEpochId::new("sync-session-42").unwrap()
    }

    fn source_receipt(error_us: u64) -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(source_domain())
            .with_epoch(source_epoch())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(error_us))
            .with_sequence(99)
    }

    fn transform(error_us: u64) -> ClockTransformReceipt {
        ClockTransformReceipt::offset(
            source_domain(),
            source_epoch(),
            target_domain(),
            target_epoch(),
            1_000,
            10_000,
            900,
            1_100,
        )
        .unwrap()
        .with_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(error_us))
        .with_sequence(7)
    }

    #[test]
    fn transform_round_trips_with_validation() {
        let transform = transform(12);
        let json = serde_json::to_string(&transform).unwrap();
        let decoded: ClockTransformReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, transform);
    }

    #[test]
    fn transform_wire_rejects_unknown_fields() {
        let value = serde_json::to_value(transform(12)).unwrap();
        let mut object = value.as_object().unwrap().clone();
        object.insert("pretend_quality".into(), serde_json::json!(1.0));
        assert!(serde_json::from_value::<ClockTransformReceipt>(object.into()).is_err());
    }

    #[test]
    fn invalid_validity_range_is_rejected() {
        let error = ClockTransformReceipt::offset(
            source_domain(),
            source_epoch(),
            target_domain(),
            target_epoch(),
            1_000,
            10_000,
            1_100,
            900,
        )
        .unwrap_err();
        assert!(matches!(error, ClockTransformError::InvalidValidityRange { .. }));
    }

    #[test]
    fn source_anchor_must_lie_inside_validity() {
        let error = ClockTransformReceipt::offset(
            source_domain(),
            source_epoch(),
            target_domain(),
            target_epoch(),
            1_200,
            10_000,
            900,
            1_100,
        )
        .unwrap_err();
        assert!(matches!(error, ClockTransformError::AnchorOutsideValidity { .. }));
    }

    #[test]
    fn normalization_maps_positive_and_negative_anchor_deltas() {
        let transform = transform(10);
        let source = source_receipt(5);

        let before = normalize_timestamp_us(950, &source, &transform).unwrap();
        let after = normalize_timestamp_us(1_050, &source, &transform).unwrap();

        assert_eq!(before.target_timestamp_us(), 9_950);
        assert_eq!(after.target_timestamp_us(), 10_050);
    }

    #[test]
    fn normalization_combines_uncertainty_without_copying_sequence() {
        let normalized = normalize_timestamp_us(1_025, &source_receipt(7), &transform(11)).unwrap();
        assert_eq!(normalized.target_receipt().clock_domain, target_domain());
        assert_eq!(normalized.target_receipt().clock_epoch, Some(target_epoch()));
        assert_eq!(normalized.target_receipt().uncertainty, TimeUncertainty::bounded(18));
        assert_eq!(normalized.target_receipt().sequence, None);
        assert_eq!(normalized.source_receipt().sequence, Some(99));
        assert_eq!(normalized.transform().sequence(), Some(7));
    }

    #[test]
    fn wrong_source_domain_fails_before_mapping() {
        let source = TimeIntegrityReceipt::declared(ClockDomainId::new("device-b/monotonic").unwrap())
            .with_epoch(source_epoch())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(1));
        let error = normalize_timestamp_us(u64::MAX, &source, &transform(1)).unwrap_err();
        assert!(matches!(error, ClockTransformError::SourceClockDomainMismatch { .. }));
    }

    #[test]
    fn explicit_source_epoch_is_required() {
        let source = TimeIntegrityReceipt::declared(source_domain())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(1));
        let error = normalize_timestamp_us(1_000, &source, &transform(1)).unwrap_err();
        assert_eq!(error, ClockTransformError::SourceClockEpochMissing);
    }

    #[test]
    fn wrong_source_epoch_is_rejected() {
        let source = TimeIntegrityReceipt::declared(source_domain())
            .with_epoch(ClockEpochId::new("device-a/boot-8").unwrap())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(1));
        let error = normalize_timestamp_us(1_000, &source, &transform(1)).unwrap_err();
        assert!(matches!(error, ClockTransformError::SourceClockEpochMismatch { .. }));
    }

    #[test]
    fn source_timestamp_must_be_inside_calibrated_window() {
        let error = normalize_timestamp_us(2_000, &source_receipt(1), &transform(1)).unwrap_err();
        assert!(matches!(
            error,
            ClockTransformError::SourceTimestampOutsideValidity { .. }
        ));
    }

    #[test]
    fn source_and_transform_continuity_are_both_required() {
        let source = TimeIntegrityReceipt::declared(source_domain())
            .with_epoch(source_epoch())
            .with_uncertainty(TimeUncertainty::bounded(1));
        let error = normalize_timestamp_us(1_000, &source, &transform(1)).unwrap_err();
        assert_eq!(
            error,
            ClockTransformError::ContinuityNotEstablished {
                side: TransformEvidenceSide::Source,
                status: ContinuityStatus::Unverified,
            }
        );

        let unverified_transform = ClockTransformReceipt::offset(
            source_domain(),
            source_epoch(),
            target_domain(),
            target_epoch(),
            1_000,
            10_000,
            900,
            1_100,
        )
        .unwrap()
        .with_uncertainty(TimeUncertainty::bounded(1));
        let error = normalize_timestamp_us(1_000, &source_receipt(1), &unverified_transform)
            .unwrap_err();
        assert_eq!(
            error,
            ClockTransformError::ContinuityNotEstablished {
                side: TransformEvidenceSide::Transform,
                status: ContinuityStatus::Unverified,
            }
        );
    }

    #[test]
    fn finite_source_and_transform_uncertainty_are_both_required() {
        let source = TimeIntegrityReceipt::declared(source_domain())
            .with_epoch(source_epoch())
            .with_continuity(ContinuityStatus::Continuous);
        let error = normalize_timestamp_us(1_000, &source, &transform(1)).unwrap_err();
        assert_eq!(
            error,
            ClockTransformError::UnboundedUncertainty {
                side: TransformEvidenceSide::Source,
            }
        );

        let unbounded_transform = ClockTransformReceipt::offset(
            source_domain(),
            source_epoch(),
            target_domain(),
            target_epoch(),
            1_000,
            10_000,
            900,
            1_100,
        )
        .unwrap()
        .with_continuity(ContinuityStatus::Continuous);
        let error = normalize_timestamp_us(1_000, &source_receipt(1), &unbounded_transform)
            .unwrap_err();
        assert_eq!(
            error,
            ClockTransformError::UnboundedUncertainty {
                side: TransformEvidenceSide::Transform,
            }
        );
    }

    #[test]
    fn target_mapping_underflow_and_overflow_fail_closed() {
        let underflow = ClockTransformReceipt::offset(
            source_domain(),
            source_epoch(),
            target_domain(),
            target_epoch(),
            1_000,
            10,
            900,
            1_100,
        )
        .unwrap()
        .with_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(1));
        let error = normalize_timestamp_us(900, &source_receipt(1), &underflow).unwrap_err();
        assert_eq!(error, ClockTransformError::TargetTimestampUnderflow);

        let overflow = ClockTransformReceipt::offset(
            source_domain(),
            source_epoch(),
            target_domain(),
            target_epoch(),
            1_000,
            u64::MAX - 10,
            900,
            1_100,
        )
        .unwrap()
        .with_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(1));
        let error = normalize_timestamp_us(1_100, &source_receipt(1), &overflow).unwrap_err();
        assert_eq!(error, ClockTransformError::TargetTimestampOverflow);
    }

    #[test]
    fn threshold_classification_preserves_ambiguity() {
        assert_eq!(
            classify_separation_window(
                SeparationWindowUs {
                    nominal_us: 50,
                    minimum_us: 40,
                    maximum_us: 60,
                },
                70,
            ),
            TemporalThresholdDecision::DefinitelyWithin
        );
        assert_eq!(
            classify_separation_window(
                SeparationWindowUs {
                    nominal_us: 50,
                    minimum_us: 40,
                    maximum_us: 60,
                },
                50,
            ),
            TemporalThresholdDecision::Ambiguous
        );
        assert_eq!(
            classify_separation_window(
                SeparationWindowUs {
                    nominal_us: 80,
                    minimum_us: 70,
                    maximum_us: 90,
                },
                60,
            ),
            TemporalThresholdDecision::DefinitelyOutside
        );
    }

    #[test]
    fn normalized_points_can_be_compared_with_base_integrity_math() {
        let left = normalize_timestamp_us(1_000, &source_receipt(3), &transform(5)).unwrap();
        let right = normalize_timestamp_us(1_020, &source_receipt(4), &transform(6)).unwrap();
        let window = bounded_separation_window_us(
            left.target_timestamp_us(),
            left.target_receipt(),
            right.target_timestamp_us(),
            right.target_receipt(),
        )
        .unwrap();
        assert_eq!(window.nominal_us, 20);
        assert_eq!(window.minimum_us, 0);
        assert_eq!(window.maximum_us, 38);
    }
}
