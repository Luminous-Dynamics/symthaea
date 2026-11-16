//! Uncertainty-bounded mapping between source-local event clocks.
//!
//! A microsecond is a unit, not a shared clock. Two devices can report values
//! in microseconds while using different epochs, offsets, or oscillator rates.
//! This module maps source-local instants and exclusive watermarks into a
//! declared reference clock using an explicit, knowledge-time-qualified
//! calibration interval. Completeness uses the earliest possible mapped
//! watermark, never its midpoint or optimistic upper bound.
//!
//! This is a calibration contract, not source authentication. A caller still
//! needs an independently governed calibration/authority registry.

use sha2::{Digest, Sha256};
use std::fmt;

const DRIFT_DENOMINATOR: i128 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ClockDomainId(pub u32);

impl ClockDomainId {
    pub fn validate(self) -> Result<(), ClockCalibrationError> {
        if self.0 == 0 {
            Err(ClockCalibrationError::ZeroClockDomain)
        } else {
            Ok(())
        }
    }
}

/// An interval containing the possible reference-clock coordinate of one
/// source-local timestamp after applying a calibration known at the requested
/// knowledge-time cutoff.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappedClockTime {
    source_domain: ClockDomainId,
    reference_domain: ClockDomainId,
    source_time_us: i64,
    earliest_reference_us: i64,
    latest_reference_us: i64,
    calibration_id: u64,
    calibration_sha256: [u8; 32],
}

impl MappedClockTime {
    pub fn source_domain(&self) -> ClockDomainId {
        self.source_domain
    }

    pub fn reference_domain(&self) -> ClockDomainId {
        self.reference_domain
    }

    pub fn source_time_us(&self) -> i64 {
        self.source_time_us
    }

    pub fn earliest_reference_us(&self) -> i64 {
        self.earliest_reference_us
    }

    pub fn latest_reference_us(&self) -> i64 {
        self.latest_reference_us
    }

    pub fn calibration_id(&self) -> u64 {
        self.calibration_id
    }

    pub fn calibration_sha256(&self) -> [u8; 32] {
        self.calibration_sha256
    }
}

/// An exclusive source watermark mapped to a common reference clock.
///
/// The lower endpoint is the conservative bound suitable for deciding
/// whether a half-open interval is closed. The upper endpoint is diagnostic
/// only; it must not be used to make an absence claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappedExclusiveWatermark {
    source_domain: ClockDomainId,
    reference_domain: ClockDomainId,
    source_exclusive_watermark_us: i64,
    earliest_reference_bound_us: i64,
    latest_reference_bound_us: i64,
    calibration_id: u64,
    calibration_sha256: [u8; 32],
}

impl MappedExclusiveWatermark {
    pub fn source_domain(&self) -> ClockDomainId {
        self.source_domain
    }

    pub fn reference_domain(&self) -> ClockDomainId {
        self.reference_domain
    }

    pub fn source_exclusive_watermark_us(&self) -> i64 {
        self.source_exclusive_watermark_us
    }

    /// Conservative lower endpoint in the reference clock.
    pub fn earliest_reference_bound_us(&self) -> i64 {
        self.earliest_reference_bound_us
    }

    /// Optimistic upper endpoint for diagnostics only.
    pub fn latest_reference_bound_us(&self) -> i64 {
        self.latest_reference_bound_us
    }

    pub fn calibration_id(&self) -> u64 {
        self.calibration_id
    }

    pub fn calibration_sha256(&self) -> [u8; 32] {
        self.calibration_sha256
    }

    /// Whether this source alone conservatively closes [start, end).
    pub fn closes_half_open_interval(&self, end_exclusive_us: i64) -> bool {
        self.earliest_reference_bound_us >= end_exclusive_us
    }
}

/// Conservative meet of mapped exclusive watermarks from all required
/// sources, expressed in exactly one reference clock domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConservativeExclusiveWatermark {
    reference_domain: ClockDomainId,
    earliest_guaranteed_exclusive_us: i64,
    source_count: usize,
}

impl ConservativeExclusiveWatermark {
    pub fn reference_domain(&self) -> ClockDomainId {
        self.reference_domain
    }

    pub fn earliest_guaranteed_exclusive_us(&self) -> i64 {
        self.earliest_guaranteed_exclusive_us
    }

    pub fn source_count(&self) -> usize {
        self.source_count
    }

    pub fn closes_half_open_interval(&self, end_exclusive_us: i64) -> bool {
        self.earliest_guaranteed_exclusive_us >= end_exclusive_us
    }
}

/// A policy-bound affine clock calibration with explicit offset and drift
/// uncertainty. The valid source-time range is inclusive at both ends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClockCalibrationV1 {
    pub calibration_id: u64,
    pub source_domain: ClockDomainId,
    pub reference_domain: ClockDomainId,
    pub valid_from_source_us: i64,
    pub valid_through_source_us: i64,
    pub anchor_source_us: i64,
    pub anchor_reference_us: i64,
    /// Lower/upper offset error in microseconds at the calibration anchor.
    pub offset_lower_us: i64,
    pub offset_upper_us: i64,
    /// Maximum absolute relative clock-rate error, in parts per billion.
    pub max_abs_drift_ppb: u64,
    /// Knowledge-time coordinate at which this calibration became available.
    pub recorded_at_knowledge_us: i64,
    pub policy_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClockCalibrationError {
    ZeroClockDomain,
    ZeroCalibrationId,
    ZeroPolicyDigest,
    SameClockDomain,
    InvalidValidityRange,
    AnchorOutsideValidityRange,
    InvertedOffsetInterval,
    FutureCalibration {
        recorded_at_knowledge_us: i64,
        known_at_knowledge_us: i64,
    },
    SourceDomainMismatch {
        expected: ClockDomainId,
        actual: ClockDomainId,
    },
    OutsideCalibrationRange {
        source_time_us: i64,
        valid_from_source_us: i64,
        valid_through_source_us: i64,
    },
    EmptyWatermarkSet,
    ReferenceDomainMismatch {
        expected: ClockDomainId,
        actual: ClockDomainId,
    },
    ArithmeticOverflow,
}

impl fmt::Display for ClockCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroClockDomain => write!(f, "clock domain ID must be nonzero"),
            Self::ZeroCalibrationId => write!(f, "clock calibration ID must be nonzero"),
            Self::ZeroPolicyDigest => write!(f, "clock calibration policy digest must be nonzero"),
            Self::SameClockDomain => write!(f, "source and reference clock domains must differ"),
            Self::InvalidValidityRange => write!(f, "clock calibration validity range must be non-empty"),
            Self::AnchorOutsideValidityRange => write!(f, "clock calibration anchor must lie inside its validity range"),
            Self::InvertedOffsetInterval => write!(f, "clock calibration lower offset exceeds upper offset"),
            Self::FutureCalibration { recorded_at_knowledge_us, known_at_knowledge_us } => write!(
                f,
                "clock calibration recorded at {recorded_at_knowledge_us} is not known at cutoff {known_at_knowledge_us}"
            ),
            Self::SourceDomainMismatch { expected, actual } => write!(
                f,
                "clock calibration expects source domain {:?}, received {:?}",
                expected, actual
            ),
            Self::OutsideCalibrationRange {
                source_time_us,
                valid_from_source_us,
                valid_through_source_us,
            } => write!(
                f,
                "source time {source_time_us} lies outside calibration validity [{valid_from_source_us}, {valid_through_source_us}]"
            ),
            Self::EmptyWatermarkSet => write!(f, "at least one mapped source watermark is required"),
            Self::ReferenceDomainMismatch { expected, actual } => write!(
                f,
                "mapped watermark reference domain {:?} does not match {:?}",
                actual, expected
            ),
            Self::ArithmeticOverflow => write!(f, "clock conversion or uncertainty arithmetic overflowed"),
        }
    }
}

impl std::error::Error for ClockCalibrationError {}

impl ClockCalibrationV1 {
    pub fn validate(&self) -> Result<(), ClockCalibrationError> {
        self.source_domain.validate()?;
        self.reference_domain.validate()?;
        if self.calibration_id == 0 {
            return Err(ClockCalibrationError::ZeroCalibrationId);
        }
        if self.source_domain == self.reference_domain {
            return Err(ClockCalibrationError::SameClockDomain);
        }
        if self.valid_from_source_us > self.valid_through_source_us {
            return Err(ClockCalibrationError::InvalidValidityRange);
        }
        if self.anchor_source_us < self.valid_from_source_us
            || self.anchor_source_us > self.valid_through_source_us
        {
            return Err(ClockCalibrationError::AnchorOutsideValidityRange);
        }
        if self.offset_lower_us > self.offset_upper_us {
            return Err(ClockCalibrationError::InvertedOffsetInterval);
        }
        if self.policy_sha256 == [0; 32] {
            return Err(ClockCalibrationError::ZeroPolicyDigest);
        }
        Ok(())
    }

    /// Stable digest binding the calibration parameters and their policy ID.
    pub fn calibration_sha256(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"SYMTHAEA-CLOCK-CALIBRATION-V1\0");
        hasher.update(self.calibration_id.to_le_bytes());
        hasher.update(self.source_domain.0.to_le_bytes());
        hasher.update(self.reference_domain.0.to_le_bytes());
        hasher.update(self.valid_from_source_us.to_le_bytes());
        hasher.update(self.valid_through_source_us.to_le_bytes());
        hasher.update(self.anchor_source_us.to_le_bytes());
        hasher.update(self.anchor_reference_us.to_le_bytes());
        hasher.update(self.offset_lower_us.to_le_bytes());
        hasher.update(self.offset_upper_us.to_le_bytes());
        hasher.update(self.max_abs_drift_ppb.to_le_bytes());
        hasher.update(self.recorded_at_knowledge_us.to_le_bytes());
        hasher.update(self.policy_sha256);
        hasher.finalize().into()
    }

    /// Map one source-local timestamp to a conservative interval in the
    /// reference domain. Future knowledge, wrong domains, expired validity and
    /// arithmetic overflow all reject rather than guessing.
    pub fn map_time(
        &self,
        actual_source_domain: ClockDomainId,
        source_time_us: i64,
        known_at_knowledge_us: i64,
    ) -> Result<MappedClockTime, ClockCalibrationError> {
        self.validate()?;
        if actual_source_domain != self.source_domain {
            return Err(ClockCalibrationError::SourceDomainMismatch {
                expected: self.source_domain,
                actual: actual_source_domain,
            });
        }
        if self.recorded_at_knowledge_us > known_at_knowledge_us {
            return Err(ClockCalibrationError::FutureCalibration {
                recorded_at_knowledge_us: self.recorded_at_knowledge_us,
                known_at_knowledge_us,
            });
        }
        if source_time_us < self.valid_from_source_us
            || source_time_us > self.valid_through_source_us
        {
            return Err(ClockCalibrationError::OutsideCalibrationRange {
                source_time_us,
                valid_from_source_us: self.valid_from_source_us,
                valid_through_source_us: self.valid_through_source_us,
            });
        }

        let delta = i128::from(source_time_us)
            .checked_sub(i128::from(self.anchor_source_us))
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let drift_numerator = delta
            .unsigned_abs()
            .checked_mul(u128::from(self.max_abs_drift_ppb))
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let denominator = DRIFT_DENOMINATOR as u128;
        let drift_floor = drift_numerator / denominator;
        let round_up = if drift_numerator % denominator == 0 {
            0
        } else {
            1
        };
        let drift_ceil = drift_floor
            .checked_add(round_up)
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let drift_uncertainty = i128::try_from(drift_ceil)
            .map_err(|_| ClockCalibrationError::ArithmeticOverflow)?;

        let nominal_reference = i128::from(self.anchor_reference_us)
            .checked_add(delta)
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let earliest = nominal_reference
            .checked_add(i128::from(self.offset_lower_us))
            .and_then(|value| value.checked_sub(drift_uncertainty))
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;
        let latest = nominal_reference
            .checked_add(i128::from(self.offset_upper_us))
            .and_then(|value| value.checked_add(drift_uncertainty))
            .ok_or(ClockCalibrationError::ArithmeticOverflow)?;

        Ok(MappedClockTime {
            source_domain: self.source_domain,
            reference_domain: self.reference_domain,
            source_time_us,
            earliest_reference_us: i64::try_from(earliest)
                .map_err(|_| ClockCalibrationError::ArithmeticOverflow)?,
            latest_reference_us: i64::try_from(latest)
                .map_err(|_| ClockCalibrationError::ArithmeticOverflow)?,
            calibration_id: self.calibration_id,
            calibration_sha256: self.calibration_sha256(),
        })
    }

    /// Map a source-local exclusive event-time watermark. Only the earliest
    /// possible mapped bound is safe for interval-completeness qualification.
    pub fn map_exclusive_watermark(
        &self,
        source_exclusive_watermark_us: i64,
        known_at_knowledge_us: i64,
    ) -> Result<MappedExclusiveWatermark, ClockCalibrationError> {
        let mapped = self.map_time(self.source_domain, source_exclusive_watermark_us, known_at_knowledge_us)?;
        Ok(MappedExclusiveWatermark {
            source_domain: mapped.source_domain,
            reference_domain: mapped.reference_domain,
            source_exclusive_watermark_us,
            earliest_reference_bound_us: mapped.earliest_reference_us,
            latest_reference_bound_us: mapped.latest_reference_us,
            calibration_id: mapped.calibration_id,
            calibration_sha256: mapped.calibration_sha256,
        })
    }
}

/// Combine watermarks only after mapping every source into the same reference
/// clock. The minimum lower endpoint is the only conservative global bound.
pub fn conservative_min_exclusive_watermark(
    reference_domain: ClockDomainId,
    watermarks: &[MappedExclusiveWatermark],
) -> Result<ConservativeExclusiveWatermark, ClockCalibrationError> {
    reference_domain.validate()?;
    if watermarks.is_empty() {
        return Err(ClockCalibrationError::EmptyWatermarkSet);
    }
    let mut earliest = i64::MAX;
    for watermark in watermarks {
        if watermark.reference_domain != reference_domain {
            return Err(ClockCalibrationError::ReferenceDomainMismatch {
                expected: reference_domain,
                actual: watermark.reference_domain,
            });
        }
        earliest = earliest.min(watermark.earliest_reference_bound_us);
    }
    Ok(ConservativeExclusiveWatermark {
        reference_domain,
        earliest_guaranteed_exclusive_us: earliest,
        source_count: watermarks.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> ClockCalibrationV1 {
        ClockCalibrationV1 {
            calibration_id: 7,
            source_domain: ClockDomainId(1),
            reference_domain: ClockDomainId(2),
            valid_from_source_us: 0,
            valid_through_source_us: 10_000_000,
            anchor_source_us: 0,
            anchor_reference_us: 10_000,
            offset_lower_us: 2,
            offset_upper_us: 4,
            max_abs_drift_ppb: 0,
            recorded_at_knowledge_us: 100,
            policy_sha256: [0xA5; 32],
        }
    }

    #[test]
    fn exact_mapping_preserves_offset_uncertainty_interval() {
        let mapped = calibration()
            .map_time(ClockDomainId(1), 1_000, 100)
            .unwrap();

        assert_eq!(mapped.earliest_reference_us(), 11_002);
        assert_eq!(mapped.latest_reference_us(), 11_004);
        assert_eq!(mapped.calibration_id(), 7);
    }

    #[test]
    fn drift_uncertainty_expands_outward_and_rounds_up() {
        let mut clock = calibration();
        clock.anchor_reference_us = 0;
        clock.offset_lower_us = -1;
        clock.offset_upper_us = 1;
        clock.max_abs_drift_ppb = 1_000_000;

        let mapped = clock.map_time(ClockDomainId(1), 1_001, 100).unwrap();
        assert_eq!(mapped.earliest_reference_us(), 998);
        assert_eq!(mapped.latest_reference_us(), 1_004);
    }

    #[test]
    fn calibration_is_not_usable_before_its_knowledge_time() {
        let clock = calibration();
        assert_eq!(
            clock.map_time(ClockDomainId(1), 1_000, 99),
            Err(ClockCalibrationError::FutureCalibration {
                recorded_at_knowledge_us: 100,
                known_at_knowledge_us: 99,
            })
        );
    }

    #[test]
    fn wrong_clock_domain_and_expired_calibration_reject() {
        let clock = calibration();
        assert_eq!(
            clock.map_time(ClockDomainId(3), 1_000, 100),
            Err(ClockCalibrationError::SourceDomainMismatch {
                expected: ClockDomainId(1),
                actual: ClockDomainId(3),
            })
        );
        assert!(matches!(
            clock.map_time(ClockDomainId(1), 10_000_001, 100),
            Err(ClockCalibrationError::OutsideCalibrationRange { .. })
        ));
    }

    #[test]
    fn invalid_calibration_parameters_fail_closed() {
        let mut clock = calibration();
        clock.offset_lower_us = 5;
        clock.offset_upper_us = 4;
        assert_eq!(
            clock.validate(),
            Err(ClockCalibrationError::InvertedOffsetInterval)
        );

        let mut clock = calibration();
        clock.policy_sha256 = [0; 32];
        assert_eq!(clock.validate(), Err(ClockCalibrationError::ZeroPolicyDigest));
    }

    #[test]
    fn watermark_uses_the_earliest_not_latest_mapped_bound() {
        let clock = calibration();
        let mapped = clock.map_exclusive_watermark(1_000, 100).unwrap();
        assert_eq!(mapped.earliest_reference_bound_us(), 11_002);
        assert_eq!(mapped.latest_reference_bound_us(), 11_004);
        assert!(mapped.closes_half_open_interval(11_002));
        assert!(!mapped.closes_half_open_interval(11_003));
    }

    #[test]
    fn cross_source_meet_uses_the_least_conservative_bound() {
        let first = calibration().map_exclusive_watermark(1_000, 100).unwrap();
        let mut second_clock = calibration();
        second_clock.calibration_id = 8;
        second_clock.anchor_reference_us = 9_500;
        let second = second_clock.map_exclusive_watermark(1_000, 100).unwrap();

        let combined = conservative_min_exclusive_watermark(ClockDomainId(2), &[first, second])
            .unwrap();
        assert_eq!(combined.earliest_guaranteed_exclusive_us(), 10_502);
        assert_eq!(combined.source_count(), 2);
        assert!(combined.closes_half_open_interval(10_502));
        assert!(!combined.closes_half_open_interval(10_503));
    }

    #[test]
    fn unmapped_reference_domains_cannot_be_aggregated() {
        let first = calibration().map_exclusive_watermark(1_000, 100).unwrap();
        let mut second_clock = calibration();
        second_clock.calibration_id = 8;
        second_clock.reference_domain = ClockDomainId(3);
        let second = second_clock.map_exclusive_watermark(1_000, 100).unwrap();

        assert_eq!(
            conservative_min_exclusive_watermark(ClockDomainId(2), &[first, second]),
            Err(ClockCalibrationError::ReferenceDomainMismatch {
                expected: ClockDomainId(2),
                actual: ClockDomainId(3),
            })
        );
    }

    #[test]
    fn empty_watermark_set_cannot_claim_completeness() {
        assert_eq!(
            conservative_min_exclusive_watermark(ClockDomainId(2), &[]),
            Err(ClockCalibrationError::EmptyWatermarkSet)
        );
    }

    #[test]
    fn arithmetic_overflow_rejects_unrepresentable_clock_mapping() {
        let mut clock = calibration();
        clock.anchor_reference_us = i64::MAX;
        assert_eq!(
            clock.map_time(ClockDomainId(1), 1, 100),
            Err(ClockCalibrationError::ArithmeticOverflow)
        );
    }
}
