// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bound finite holdover for clock normalization.
//!
//! This crate composes an accepted, evidence-bound calibration decision with an
//! explicit holdover claim. It first transports each exchange-derived offset
//! interval to one deterministic source-time anchor under the claimed relative
//! drift bound, intersects those anchor-admissible intervals, and then grows
//! uncertainty again across the requested finite holdover window.
//!
//! The holdover claim remains a claim container; this crate does not measure
//! oscillator drift, prove continuity, or authenticate timing evidence.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};
use symthaea_time_calibration::{CalibrationError, ClockOffsetIntervalUs};
use symthaea_time_calibration_bundle::{CalibrationBundleError, CalibrationDecisionBundle};
use symthaea_time_integrity::{ContinuityStatus, TimeUncertainty};
use symthaea_time_normalization::{ClockTransformError, ClockTransformReceipt};

pub const PPB_PER_UNIT: u128 = 1_000_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoldoverError {
    InvalidValidityRange {
        start_us: u64,
        end_us: u64,
    },
    CalibrationEvidenceOutsideValidity {
        evidence_index: usize,
        timestamp_us: u64,
        start_us: u64,
        end_us: u64,
    },
    AnchorOutsideValidity {
        anchor_us: u64,
        start_us: u64,
        end_us: u64,
    },
    SourceContinuityNotEstablished {
        status: ContinuityStatus,
    },
    MappingContinuityNotEstablished {
        status: ContinuityStatus,
    },
    TargetContinuityNotEstablished {
        status: ContinuityStatus,
    },
    TargetAnchorUnderflow,
    TargetAnchorOverflow,
    UncertaintyOverflow,
    Calibration(CalibrationBundleError),
    CalibrationInterval(CalibrationError),
    Transform(ClockTransformError),
    TransformMismatch,
}

impl fmt::Display for HoldoverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValidityRange { start_us, end_us } => write!(
                f,
                "holdover validity start {start_us} exceeds end {end_us}"
            ),
            Self::CalibrationEvidenceOutsideValidity {
                evidence_index,
                timestamp_us,
                start_us,
                end_us,
            } => write!(
                f,
                "calibration evidence {evidence_index} source timestamp {timestamp_us} lies outside holdover validity [{start_us}, {end_us}]"
            ),
            Self::AnchorOutsideValidity {
                anchor_us,
                start_us,
                end_us,
            } => write!(
                f,
                "calibration anchor {anchor_us} is outside holdover validity [{start_us}, {end_us}]"
            ),
            Self::SourceContinuityNotEstablished { status } => write!(
                f,
                "source-clock continuity is not established for holdover: {status:?}"
            ),
            Self::MappingContinuityNotEstablished { status } => write!(
                f,
                "mapping continuity is not established for holdover: {status:?}"
            ),
            Self::TargetContinuityNotEstablished { status } => write!(
                f,
                "target-clock continuity is not established for holdover: {status:?}"
            ),
            Self::TargetAnchorUnderflow => {
                write!(f, "derived offset maps calibration anchor below target time zero")
            }
            Self::TargetAnchorOverflow => {
                write!(f, "derived offset maps calibration anchor above target u64 time")
            }
            Self::UncertaintyOverflow => write!(
                f,
                "holdover uncertainty cannot be represented safely"
            ),
            Self::Calibration(error) => write!(f, "calibration bundle invalid: {error}"),
            Self::CalibrationInterval(error) => {
                write!(f, "holdover calibration interval invalid: {error}")
            }
            Self::Transform(error) => write!(f, "derived clock transform invalid: {error}"),
            Self::TransformMismatch => write!(
                f,
                "stored clock transform does not match calibration + holdover derivation"
            ),
        }
    }
}

impl std::error::Error for HoldoverError {}

impl From<CalibrationBundleError> for HoldoverError {
    fn from(value: CalibrationBundleError) -> Self {
        Self::Calibration(value)
    }
}

impl From<CalibrationError> for HoldoverError {
    fn from(value: CalibrationError) -> Self {
        Self::CalibrationInterval(value)
    }
}

impl From<ClockTransformError> for HoldoverError {
    fn from(value: ClockTransformError) -> Self {
        Self::Transform(value)
    }
}

/// Claim that one accepted source->target calibration may be held over a finite
/// source-time interval with bounded relative offset drift.
///
/// `max_relative_drift_ppb` bounds source-vs-target *offset drift rate*. It is
/// not a scalar oscillator-quality score and does not describe either clock in
/// isolation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HoldoverClaim {
    valid_source_start_us: u64,
    valid_source_end_us: u64,
    max_relative_drift_ppb: u64,
    fixed_model_error_us: u64,
    source_continuity: ContinuityStatus,
    mapping_continuity: ContinuityStatus,
    target_continuity: ContinuityStatus,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HoldoverClaimWire {
    valid_source_start_us: u64,
    valid_source_end_us: u64,
    max_relative_drift_ppb: u64,
    fixed_model_error_us: u64,
    source_continuity: ContinuityStatus,
    mapping_continuity: ContinuityStatus,
    target_continuity: ContinuityStatus,
}

impl<'de> Deserialize<'de> for HoldoverClaim {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = HoldoverClaimWire::deserialize(deserializer)?;
        Self::new(
            wire.valid_source_start_us,
            wire.valid_source_end_us,
            wire.max_relative_drift_ppb,
            wire.fixed_model_error_us,
            wire.source_continuity,
            wire.mapping_continuity,
            wire.target_continuity,
        )
        .map_err(de::Error::custom)
    }
}

impl HoldoverClaim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        valid_source_start_us: u64,
        valid_source_end_us: u64,
        max_relative_drift_ppb: u64,
        fixed_model_error_us: u64,
        source_continuity: ContinuityStatus,
        mapping_continuity: ContinuityStatus,
        target_continuity: ContinuityStatus,
    ) -> Result<Self, HoldoverError> {
        if valid_source_start_us > valid_source_end_us {
            return Err(HoldoverError::InvalidValidityRange {
                start_us: valid_source_start_us,
                end_us: valid_source_end_us,
            });
        }
        Ok(Self {
            valid_source_start_us,
            valid_source_end_us,
            max_relative_drift_ppb,
            fixed_model_error_us,
            source_continuity,
            mapping_continuity,
            target_continuity,
        })
    }

    pub fn valid_source_range_us(&self) -> (u64, u64) {
        (self.valid_source_start_us, self.valid_source_end_us)
    }

    pub fn max_relative_drift_ppb(&self) -> u64 {
        self.max_relative_drift_ppb
    }

    pub fn fixed_model_error_us(&self) -> u64 {
        self.fixed_model_error_us
    }

    pub fn source_continuity(&self) -> ContinuityStatus {
        self.source_continuity
    }

    pub fn mapping_continuity(&self) -> ContinuityStatus {
        self.mapping_continuity
    }

    pub fn target_continuity(&self) -> ContinuityStatus {
        self.target_continuity
    }
}

fn calibration_source_envelope_us(bundle: &CalibrationDecisionBundle) -> (u64, u64) {
    let mut minimum = u64::MAX;
    let mut maximum = 0_u64;
    for item in bundle.evidence() {
        let exchange = item.exchange();
        minimum = minimum.min(exchange.source_send().timestamp_us);
        minimum = minimum.min(exchange.source_receive().timestamp_us);
        maximum = maximum.max(exchange.source_send().timestamp_us);
        maximum = maximum.max(exchange.source_receive().timestamp_us);
    }
    (minimum, maximum)
}

fn calibration_anchor_source_us(bundle: &CalibrationDecisionBundle) -> u64 {
    let (minimum, maximum) = calibration_source_envelope_us(bundle);
    minimum + (maximum - minimum) / 2
}

fn apply_signed_offset(source_timestamp_us: u64, offset_us: i128) -> Result<u64, HoldoverError> {
    if offset_us >= 0 {
        let offset = u128::try_from(offset_us).map_err(|_| HoldoverError::TargetAnchorOverflow)?;
        let target = u128::from(source_timestamp_us) + offset;
        u64::try_from(target).map_err(|_| HoldoverError::TargetAnchorOverflow)
    } else {
        let magnitude = offset_us.unsigned_abs();
        if magnitude > u128::from(source_timestamp_us) {
            return Err(HoldoverError::TargetAnchorUnderflow);
        }
        let magnitude_u64 =
            u64::try_from(magnitude).map_err(|_| HoldoverError::TargetAnchorUnderflow)?;
        Ok(source_timestamp_us - magnitude_u64)
    }
}

/// Conservative ceil(distance_us * ppb / 1e9).
fn drift_growth_us(distance_us: u64, max_relative_drift_ppb: u64) -> u128 {
    let product = u128::from(distance_us) * u128::from(max_relative_drift_ppb);
    let quotient = product / PPB_PER_UNIT;
    let remainder = product % PPB_PER_UNIT;
    quotient + u128::from(remainder != 0)
}

fn distance_to_closed_interval_us(point: u64, start: u64, end: u64) -> u64 {
    if point < start {
        start - point
    } else if point > end {
        point - end
    } else {
        0
    }
}

fn expand_offset_interval(
    interval: ClockOffsetIntervalUs,
    growth_us: u128,
) -> Result<ClockOffsetIntervalUs, HoldoverError> {
    let growth = i128::try_from(growth_us).map_err(|_| HoldoverError::UncertaintyOverflow)?;
    let lower = interval
        .lower_us()
        .checked_sub(growth)
        .ok_or(HoldoverError::UncertaintyOverflow)?;
    let upper = interval
        .upper_us()
        .checked_add(growth)
        .ok_or(HoldoverError::UncertaintyOverflow)?;
    Ok(ClockOffsetIntervalUs::new(lower, upper)?)
}

fn anchor_offset_interval(
    calibration: &CalibrationDecisionBundle,
    holdover: &HoldoverClaim,
    source_anchor_us: u64,
) -> Result<ClockOffsetIntervalUs, HoldoverError> {
    let mut combined: Option<ClockOffsetIntervalUs> = None;

    for (evidence_index, item) in calibration.evidence().iter().enumerate() {
        let exchange = item.exchange();
        let source_send = exchange.source_send().timestamp_us;
        let source_receive = exchange.source_receive().timestamp_us;

        for timestamp_us in [source_send, source_receive] {
            if timestamp_us < holdover.valid_source_start_us
                || timestamp_us > holdover.valid_source_end_us
            {
                return Err(HoldoverError::CalibrationEvidenceOutsideValidity {
                    evidence_index,
                    timestamp_us,
                    start_us: holdover.valid_source_start_us,
                    end_us: holdover.valid_source_end_us,
                });
            }
        }

        // Each four-timestamp interval applies over its source-side exchange
        // interval under the calibration model's approximately-constant-offset
        // assumption. Drift is needed only to transport that interval from the
        // exchange interval to the common anchor.
        let exchange_start = source_send.min(source_receive);
        let exchange_end = source_send.max(source_receive);
        let distance =
            distance_to_closed_interval_us(source_anchor_us, exchange_start, exchange_end);
        let growth = drift_growth_us(distance, holdover.max_relative_drift_ppb);
        let at_anchor = expand_offset_interval(item.offset_interval(), growth)?;

        combined = Some(match combined {
            None => at_anchor,
            Some(current) => current.intersect(at_anchor)?,
        });
    }

    // CalibrationDecisionBundle::verify_self() requires a non-empty evidence
    // set, so reaching this point without an interval would indicate an internal
    // invariant violation rather than user-provided timing evidence.
    combined.ok_or(HoldoverError::Calibration(
        CalibrationBundleError::EmptyEvidence,
    ))
}

fn derive_transform(
    calibration: &CalibrationDecisionBundle,
    holdover: &HoldoverClaim,
) -> Result<ClockTransformReceipt, HoldoverError> {
    calibration.verify_self()?;
    let accepted = calibration.accepted_estimate()?;

    if holdover.source_continuity != ContinuityStatus::Continuous {
        return Err(HoldoverError::SourceContinuityNotEstablished {
            status: holdover.source_continuity,
        });
    }
    if holdover.mapping_continuity != ContinuityStatus::Continuous {
        return Err(HoldoverError::MappingContinuityNotEstablished {
            status: holdover.mapping_continuity,
        });
    }
    if holdover.target_continuity != ContinuityStatus::Continuous {
        return Err(HoldoverError::TargetContinuityNotEstablished {
            status: holdover.target_continuity,
        });
    }

    let source_anchor_us = calibration_anchor_source_us(calibration);
    if source_anchor_us < holdover.valid_source_start_us
        || source_anchor_us > holdover.valid_source_end_us
    {
        return Err(HoldoverError::AnchorOutsideValidity {
            anchor_us: source_anchor_us,
            start_us: holdover.valid_source_start_us,
            end_us: holdover.valid_source_end_us,
        });
    }

    let anchor_interval = anchor_offset_interval(calibration, holdover, source_anchor_us)?;
    let anchor_offset_us = anchor_interval.midpoint_us();
    let anchor_radius_u128 = anchor_interval.symmetric_radius_us();
    let target_anchor_us = apply_signed_offset(source_anchor_us, anchor_offset_us)?;

    let left_distance = source_anchor_us - holdover.valid_source_start_us;
    let right_distance = holdover.valid_source_end_us - source_anchor_us;
    let worst_holdover_distance = left_distance.max(right_distance);
    let holdover_drift_error =
        drift_growth_us(worst_holdover_distance, holdover.max_relative_drift_ppb);

    let total_error = anchor_radius_u128
        .checked_add(holdover_drift_error)
        .and_then(|value| value.checked_add(u128::from(holdover.fixed_model_error_us)))
        .ok_or(HoldoverError::UncertaintyOverflow)?;
    let total_error_us =
        u64::try_from(total_error).map_err(|_| HoldoverError::UncertaintyOverflow)?;

    let transform = ClockTransformReceipt::offset(
        accepted.source_domain().clone(),
        accepted.source_epoch().clone(),
        accepted.target_domain().clone(),
        accepted.target_epoch().clone(),
        source_anchor_us,
        target_anchor_us,
        holdover.valid_source_start_us,
        holdover.valid_source_end_us,
    )?
    .with_mapping_continuity(holdover.mapping_continuity)
    .with_target_continuity(holdover.target_continuity)
    .with_uncertainty(TimeUncertainty::bounded(total_error_us));
    transform.validate()?;
    Ok(transform)
}

/// Self-verifying composition of exact calibration evidence, holdover claims,
/// and the finite-window clock transform derived from them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoundedHoldoverTransform {
    calibration: CalibrationDecisionBundle,
    holdover: HoldoverClaim,
    transform: ClockTransformReceipt,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundedHoldoverTransformWire {
    calibration: CalibrationDecisionBundle,
    holdover: HoldoverClaim,
    transform: ClockTransformReceipt,
}

impl<'de> Deserialize<'de> for BoundedHoldoverTransform {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = BoundedHoldoverTransformWire::deserialize(deserializer)?;
        let value = Self::new(wire.calibration, wire.holdover).map_err(de::Error::custom)?;
        if value.transform != wire.transform {
            return Err(de::Error::custom(HoldoverError::TransformMismatch));
        }
        Ok(value)
    }
}

impl BoundedHoldoverTransform {
    pub fn new(
        calibration: CalibrationDecisionBundle,
        holdover: HoldoverClaim,
    ) -> Result<Self, HoldoverError> {
        let transform = derive_transform(&calibration, &holdover)?;
        Ok(Self {
            calibration,
            holdover,
            transform,
        })
    }

    pub fn calibration(&self) -> &CalibrationDecisionBundle {
        &self.calibration
    }

    pub fn holdover(&self) -> &HoldoverClaim {
        &self.holdover
    }

    pub fn transform(&self) -> &ClockTransformReceipt {
        &self.transform
    }

    pub fn calibration_anchor_source_us(&self) -> u64 {
        calibration_anchor_source_us(&self.calibration)
    }

    pub fn anchor_offset_interval(&self) -> Result<ClockOffsetIntervalUs, HoldoverError> {
        anchor_offset_interval(
            &self.calibration,
            &self.holdover,
            self.calibration_anchor_source_us(),
        )
    }

    pub fn verify_self(&self) -> Result<(), HoldoverError> {
        let expected = derive_transform(&self.calibration, &self.holdover)?;
        if expected != self.transform {
            return Err(HoldoverError::TransformMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_time_calibration::{
        CalibrationConsensus, ClockCalibrationEvidence, FourTimestampExchange, TimestampEvidence,
    };
    use symthaea_time_calibration_policy::{CalibrationDecisionPolicy, CalibrationPolicyId};
    use symthaea_time_integrity::{ClockDomainId, ClockEpochId, TimeIntegrityReceipt};
    use symthaea_time_normalization::ClockTransformModel;

    fn source_domain() -> ClockDomainId {
        ClockDomainId::new("sensor-a/monotonic").unwrap()
    }

    fn target_domain() -> ClockDomainId {
        ClockDomainId::new("capture-host/monotonic").unwrap()
    }

    fn source_epoch() -> ClockEpochId {
        ClockEpochId::new("sensor-a-boot-7").unwrap()
    }

    fn target_epoch() -> ClockEpochId {
        ClockEpochId::new("capture-host-boot-3").unwrap()
    }

    fn receipt(domain: ClockDomainId, epoch: ClockEpochId) -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(domain)
            .with_epoch(epoch)
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(0))
    }

    fn stamp(timestamp_us: u64, domain: ClockDomainId, epoch: ClockEpochId) -> TimestampEvidence {
        TimestampEvidence::new(timestamp_us, receipt(domain, epoch))
    }

    fn calibration_evidence(source_start_us: u64, delay: u64) -> ClockCalibrationEvidence {
        let offset = 500;
        let processing = 10;
        let t1 = source_start_us;
        let t2 = t1 + offset + delay;
        let t3 = t2 + processing;
        let t4 = t1 + delay + processing + delay;
        let exchange = FourTimestampExchange::new(
            stamp(t1, source_domain(), source_epoch()),
            stamp(t2, target_domain(), target_epoch()),
            stamp(t3, target_domain(), target_epoch()),
            stamp(t4, source_domain(), source_epoch()),
        )
        .unwrap();
        ClockCalibrationEvidence::derive(exchange).unwrap()
    }

    fn accepted_bundle_from(evidence: Vec<ClockCalibrationEvidence>) -> CalibrationDecisionBundle {
        let consensus = CalibrationConsensus::from_evidence(&evidence).unwrap();
        let policy = CalibrationDecisionPolicy::new(
            CalibrationPolicyId::new("holdover-v1").unwrap(),
            1,
            20,
            Some(100),
        )
        .unwrap();
        let decision = policy.evaluate(&consensus);
        CalibrationDecisionBundle::new(decision, evidence).unwrap()
    }

    fn accepted_bundle() -> CalibrationDecisionBundle {
        accepted_bundle_from(vec![calibration_evidence(1_000, 10)])
    }

    fn continuous_claim() -> HoldoverClaim {
        // Calibration source envelope is [1000, 1030], anchor = 1015.
        HoldoverClaim::new(
            15,
            2_015,
            1_000,
            2,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
        )
        .unwrap()
    }

    #[test]
    fn uncertainty_grows_with_distance_from_calibration_anchor() {
        let value = BoundedHoldoverTransform::new(accepted_bundle(), continuous_claim()).unwrap();
        assert_eq!(value.calibration_anchor_source_us(), 1_015);
        assert_eq!(
            value.anchor_offset_interval().unwrap(),
            ClockOffsetIntervalUs::new(490, 510).unwrap()
        );
        assert_eq!(value.transform().valid_source_range_us(), (15, 2_015));
        assert_eq!(
            value.transform().uncertainty(),
            TimeUncertainty::Bounded { max_error_us: 13 }
        );
        value.verify_self().unwrap();
    }

    #[test]
    fn separated_calibration_exchanges_are_drift_transportable_to_anchor() {
        let bundle = accepted_bundle_from(vec![
            calibration_evidence(1_000, 10),
            calibration_evidence(3_000, 10),
        ]);
        let claim = HoldoverClaim::new(
            500,
            3_500,
            1_000_000,
            0,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
        )
        .unwrap();
        let value = BoundedHoldoverTransform::new(bundle, claim).unwrap();
        assert_eq!(value.calibration_anchor_source_us(), 2_015);
        // Each exchange admits [490, 510]. The anchor is 985 us outside each
        // source exchange; at 1,000,000 ppb that adds 1 us per interval.
        assert_eq!(
            value.anchor_offset_interval().unwrap(),
            ClockOffsetIntervalUs::new(489, 511).unwrap()
        );
    }

    #[test]
    fn exact_calibration_anchor_maps_using_derived_anchor_midpoint() {
        let value = BoundedHoldoverTransform::new(accepted_bundle(), continuous_claim()).unwrap();
        match value.transform().model() {
            ClockTransformModel::Offset {
                source_anchor_us,
                target_anchor_us,
            } => {
                assert_eq!(*source_anchor_us, 1_015);
                assert_eq!(*target_anchor_us, 1_515);
            }
        }
    }

    #[test]
    fn validity_must_cover_all_calibration_source_evidence() {
        let claim = HoldoverClaim::new(
            1_010,
            2_015,
            1_000,
            0,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
        )
        .unwrap();
        let error = BoundedHoldoverTransform::new(accepted_bundle(), claim).unwrap_err();
        assert!(matches!(
            error,
            HoldoverError::CalibrationEvidenceOutsideValidity {
                timestamp_us: 1_000,
                ..
            }
        ));
    }

    #[test]
    fn missing_mapping_continuity_fails_closed() {
        let claim = HoldoverClaim::new(
            15,
            2_015,
            1_000,
            2,
            ContinuityStatus::Continuous,
            ContinuityStatus::Unverified,
            ContinuityStatus::Continuous,
        )
        .unwrap();
        let error = BoundedHoldoverTransform::new(accepted_bundle(), claim).unwrap_err();
        assert_eq!(
            error,
            HoldoverError::MappingContinuityNotEstablished {
                status: ContinuityStatus::Unverified,
            }
        );
    }

    #[test]
    fn huge_holdover_uncertainty_fails_instead_of_saturating() {
        let claim = HoldoverClaim::new(
            0,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
        )
        .unwrap();
        let error = BoundedHoldoverTransform::new(accepted_bundle(), claim).unwrap_err();
        assert_eq!(error, HoldoverError::UncertaintyOverflow);
    }

    #[test]
    fn wire_roundtrip_rederives_transform() {
        let value = BoundedHoldoverTransform::new(accepted_bundle(), continuous_claim()).unwrap();
        let json = serde_json::to_string(&value).unwrap();
        let decoded: BoundedHoldoverTransform = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn tampered_transform_is_rejected_on_wire() {
        let value = BoundedHoldoverTransform::new(accepted_bundle(), continuous_claim()).unwrap();
        let mut json = serde_json::to_value(&value).unwrap();
        json["transform"]["valid_source_end_us"] = serde_json::json!(2_014);
        assert!(serde_json::from_value::<BoundedHoldoverTransform>(json).is_err());
    }
}
