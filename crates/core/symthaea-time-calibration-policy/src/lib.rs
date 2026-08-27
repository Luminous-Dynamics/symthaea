// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Asymmetric decision policy for clock-calibration evidence.
//!
//! This crate deliberately sits between calibration evidence and clock-transform
//! authority. A sufficiently tight offset interval may be *accepted* for a
//! stated policy, but acceptance produces only an offset estimate + error radius.
//! It does not manufacture transform validity, mapping continuity, target-clock
//! continuity, synchronization authenticity, or holdover evidence.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use symthaea_time_calibration::{CalibrationConsensus, ClockOffsetIntervalUs};
use symthaea_time_integrity::{ClockDomainId, ClockEpochId};

pub const MAX_CALIBRATION_POLICY_ID_LEN: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationPolicyIdError {
    Empty,
    TooLong { actual: usize, max: usize },
    NonCanonical,
}

impl fmt::Display for CalibrationPolicyIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "calibration policy ID must not be empty"),
            Self::TooLong { actual, max } => write!(
                f,
                "calibration policy ID length {actual} exceeds maximum {max}"
            ),
            Self::NonCanonical => write!(
                f,
                "calibration policy ID must use lowercase ASCII letters, digits, '.', '_', '-', '/', or ':'"
            ),
        }
    }
}

impl std::error::Error for CalibrationPolicyIdError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CalibrationPolicyId(String);

impl CalibrationPolicyId {
    pub fn new(value: impl Into<String>) -> Result<Self, CalibrationPolicyIdError> {
        let value = value.into();
        if value.is_empty() {
            return Err(CalibrationPolicyIdError::Empty);
        }
        if value.len() > MAX_CALIBRATION_POLICY_ID_LEN {
            return Err(CalibrationPolicyIdError::TooLong {
                actual: value.len(),
                max: MAX_CALIBRATION_POLICY_ID_LEN,
            });
        }
        if !value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b'-' | b'/' | b':')
        }) {
            return Err(CalibrationPolicyIdError::NonCanonical);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CalibrationPolicyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for CalibrationPolicyId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for CalibrationPolicyId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationDecision {
    /// The interval is tight enough for this policy's acceptance gate.
    Accepted,
    /// The interval crosses a separately configured practical-failure bound.
    Rejected,
    /// Neither acceptance nor practical failure is established.
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationPolicyError {
    ZeroVersion,
    InvalidThresholdOrder {
        acceptance_max_radius_us: u64,
        practical_failure_min_radius_us: u64,
    },
    DecisionMismatch {
        expected: CalibrationDecision,
        actual: CalibrationDecision,
    },
    NotAccepted {
        decision: CalibrationDecision,
    },
    AcceptedRadiusOverflow,
}

impl fmt::Display for CalibrationPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroVersion => write!(f, "calibration policy version must be non-zero"),
            Self::InvalidThresholdOrder {
                acceptance_max_radius_us,
                practical_failure_min_radius_us,
            } => write!(
                f,
                "practical-failure radius {practical_failure_min_radius_us} must be greater than acceptance radius {acceptance_max_radius_us}"
            ),
            Self::DecisionMismatch { expected, actual } => write!(
                f,
                "stored calibration decision {actual:?} does not match policy-derived decision {expected:?}"
            ),
            Self::NotAccepted { decision } => write!(
                f,
                "offset estimate is available only for Accepted decisions, got {decision:?}"
            ),
            Self::AcceptedRadiusOverflow => write!(
                f,
                "accepted calibration radius cannot be represented as u64 microseconds"
            ),
        }
    }
}

impl std::error::Error for CalibrationPolicyError {}

/// Frozen decision rule for one use of calibration evidence.
///
/// The two thresholds are deliberately asymmetric. Failing the acceptance gate
/// is not automatically a negative result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CalibrationDecisionPolicy {
    policy_id: CalibrationPolicyId,
    version: u32,
    acceptance_max_radius_us: u64,
    practical_failure_min_radius_us: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationDecisionPolicyWire {
    policy_id: CalibrationPolicyId,
    version: u32,
    acceptance_max_radius_us: u64,
    #[serde(default)]
    practical_failure_min_radius_us: Option<u64>,
}

impl<'de> Deserialize<'de> for CalibrationDecisionPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CalibrationDecisionPolicyWire::deserialize(deserializer)?;
        Self::new(
            wire.policy_id,
            wire.version,
            wire.acceptance_max_radius_us,
            wire.practical_failure_min_radius_us,
        )
        .map_err(de::Error::custom)
    }
}

impl CalibrationDecisionPolicy {
    pub fn new(
        policy_id: CalibrationPolicyId,
        version: u32,
        acceptance_max_radius_us: u64,
        practical_failure_min_radius_us: Option<u64>,
    ) -> Result<Self, CalibrationPolicyError> {
        if version == 0 {
            return Err(CalibrationPolicyError::ZeroVersion);
        }
        if let Some(failure) = practical_failure_min_radius_us {
            if failure <= acceptance_max_radius_us {
                return Err(CalibrationPolicyError::InvalidThresholdOrder {
                    acceptance_max_radius_us,
                    practical_failure_min_radius_us: failure,
                });
            }
        }
        Ok(Self {
            policy_id,
            version,
            acceptance_max_radius_us,
            practical_failure_min_radius_us,
        })
    }

    pub fn policy_id(&self) -> &CalibrationPolicyId {
        &self.policy_id
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn acceptance_max_radius_us(&self) -> u64 {
        self.acceptance_max_radius_us
    }

    pub fn practical_failure_min_radius_us(&self) -> Option<u64> {
        self.practical_failure_min_radius_us
    }

    pub fn decide_interval(&self, interval: ClockOffsetIntervalUs) -> CalibrationDecision {
        let radius = interval.symmetric_radius_us();
        if radius <= u128::from(self.acceptance_max_radius_us) {
            return CalibrationDecision::Accepted;
        }
        if let Some(failure) = self.practical_failure_min_radius_us {
            if radius >= u128::from(failure) {
                return CalibrationDecision::Rejected;
            }
        }
        CalibrationDecision::Inconclusive
    }

    pub fn evaluate(&self, consensus: &CalibrationConsensus) -> CalibrationDecisionReceipt {
        let offset_interval = consensus.offset_interval();
        CalibrationDecisionReceipt {
            policy: self.clone(),
            source_domain: consensus.source_domain().clone(),
            source_epoch: consensus.source_epoch().clone(),
            target_domain: consensus.target_domain().clone(),
            target_epoch: consensus.target_epoch().clone(),
            offset_interval,
            decision: self.decide_interval(offset_interval),
        }
    }
}

/// Self-verifying snapshot of one policy decision over one calibration interval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CalibrationDecisionReceipt {
    policy: CalibrationDecisionPolicy,
    source_domain: ClockDomainId,
    source_epoch: ClockEpochId,
    target_domain: ClockDomainId,
    target_epoch: ClockEpochId,
    offset_interval: ClockOffsetIntervalUs,
    decision: CalibrationDecision,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationDecisionReceiptWire {
    policy: CalibrationDecisionPolicy,
    source_domain: ClockDomainId,
    source_epoch: ClockEpochId,
    target_domain: ClockDomainId,
    target_epoch: ClockEpochId,
    offset_interval: ClockOffsetIntervalUs,
    decision: CalibrationDecision,
}

impl<'de> Deserialize<'de> for CalibrationDecisionReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CalibrationDecisionReceiptWire::deserialize(deserializer)?;
        let receipt = Self {
            policy: wire.policy,
            source_domain: wire.source_domain,
            source_epoch: wire.source_epoch,
            target_domain: wire.target_domain,
            target_epoch: wire.target_epoch,
            offset_interval: wire.offset_interval,
            decision: wire.decision,
        };
        receipt.verify_self().map_err(de::Error::custom)?;
        Ok(receipt)
    }
}

impl CalibrationDecisionReceipt {
    pub fn policy(&self) -> &CalibrationDecisionPolicy {
        &self.policy
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

    pub fn offset_interval(&self) -> ClockOffsetIntervalUs {
        self.offset_interval
    }

    pub fn decision(&self) -> CalibrationDecision {
        self.decision
    }

    pub fn verify_self(&self) -> Result<(), CalibrationPolicyError> {
        let expected = self.policy.decide_interval(self.offset_interval);
        if expected != self.decision {
            return Err(CalibrationPolicyError::DecisionMismatch {
                expected,
                actual: self.decision,
            });
        }
        Ok(())
    }

    /// Return the accepted nominal offset and conservative symmetric radius.
    ///
    /// The midpoint is a deterministic representation of the accepted interval,
    /// not a claim that the true offset equals the midpoint.
    pub fn accepted_estimate(&self) -> Result<AcceptedOffsetEstimate, CalibrationPolicyError> {
        self.verify_self()?;
        if self.decision != CalibrationDecision::Accepted {
            return Err(CalibrationPolicyError::NotAccepted {
                decision: self.decision,
            });
        }
        let radius_u128 = self.offset_interval.symmetric_radius_us();
        let max_error_radius_us = u64::try_from(radius_u128)
            .map_err(|_| CalibrationPolicyError::AcceptedRadiusOverflow)?;
        Ok(AcceptedOffsetEstimate {
            source_domain: self.source_domain.clone(),
            source_epoch: self.source_epoch.clone(),
            target_domain: self.target_domain.clone(),
            target_epoch: self.target_epoch.clone(),
            nominal_offset_us: self.offset_interval.midpoint_us(),
            max_error_radius_us,
        })
    }
}

/// Accepted calibration result that may inform a later transform-construction
/// policy. This is not itself a `ClockTransformReceipt`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedOffsetEstimate {
    source_domain: ClockDomainId,
    source_epoch: ClockEpochId,
    target_domain: ClockDomainId,
    target_epoch: ClockEpochId,
    nominal_offset_us: i128,
    max_error_radius_us: u64,
}

impl AcceptedOffsetEstimate {
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

    pub fn nominal_offset_us(&self) -> i128 {
        self.nominal_offset_us
    }

    pub fn max_error_radius_us(&self) -> u64 {
        self.max_error_radius_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_time_calibration::{
        ClockCalibrationEvidence, FourTimestampExchange, TimestampEvidence,
    };
    use symthaea_time_integrity::{ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty};

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

    fn consensus(one_way_delay_us: u64) -> CalibrationConsensus {
        let t1 = 1_000;
        let target_offset = 500;
        let target_processing = 10;
        let t2 = t1 + target_offset + one_way_delay_us;
        let t3 = t2 + target_processing;
        let t4 = t1 + one_way_delay_us + target_processing + one_way_delay_us;
        let exchange = FourTimestampExchange::new(
            stamp(t1, source_domain(), source_epoch()),
            stamp(t2, target_domain(), target_epoch()),
            stamp(t3, target_domain(), target_epoch()),
            stamp(t4, source_domain(), source_epoch()),
        )
        .unwrap();
        let evidence = ClockCalibrationEvidence::derive(exchange).unwrap();
        CalibrationConsensus::from_evidence(&[evidence]).unwrap()
    }

    fn policy() -> CalibrationDecisionPolicy {
        CalibrationDecisionPolicy::new(
            CalibrationPolicyId::new("physical-fusion-v1").unwrap(),
            1,
            20,
            Some(100),
        )
        .unwrap()
    }

    #[test]
    fn acceptance_failure_and_inconclusive_are_asymmetric() {
        assert_eq!(
            policy().evaluate(&consensus(10)).decision(),
            CalibrationDecision::Accepted
        );
        assert_eq!(
            policy().evaluate(&consensus(50)).decision(),
            CalibrationDecision::Inconclusive
        );
        assert_eq!(
            policy().evaluate(&consensus(200)).decision(),
            CalibrationDecision::Rejected
        );
    }

    #[test]
    fn missing_failure_threshold_never_turns_nonacceptance_into_rejection() {
        let policy = CalibrationDecisionPolicy::new(
            CalibrationPolicyId::new("acceptance-only-v1").unwrap(),
            1,
            20,
            None,
        )
        .unwrap();
        assert_eq!(
            policy.evaluate(&consensus(500)).decision(),
            CalibrationDecision::Inconclusive
        );
    }

    #[test]
    fn invalid_threshold_order_fails_closed() {
        let result = CalibrationDecisionPolicy::new(
            CalibrationPolicyId::new("bad-policy").unwrap(),
            1,
            100,
            Some(100),
        );
        assert!(matches!(
            result,
            Err(CalibrationPolicyError::InvalidThresholdOrder { .. })
        ));
    }

    #[test]
    fn accepted_estimate_preserves_interval_as_error_not_truth() {
        let receipt = policy().evaluate(&consensus(10));
        let estimate = receipt.accepted_estimate().unwrap();
        assert_eq!(estimate.nominal_offset_us(), 500);
        assert_eq!(estimate.max_error_radius_us(), 10);
        assert_eq!(estimate.source_domain(), &source_domain());
        assert_eq!(estimate.target_domain(), &target_domain());
    }

    #[test]
    fn nonaccepted_decision_cannot_produce_offset_estimate() {
        let receipt = policy().evaluate(&consensus(50));
        let error = receipt.accepted_estimate().unwrap_err();
        assert_eq!(
            error,
            CalibrationPolicyError::NotAccepted {
                decision: CalibrationDecision::Inconclusive,
            }
        );
    }

    #[test]
    fn decision_receipt_roundtrip_revalidates_policy_outcome() {
        let receipt = policy().evaluate(&consensus(10));
        let json = serde_json::to_string(&receipt).unwrap();
        let decoded: CalibrationDecisionReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, receipt);
    }

    #[test]
    fn tampered_decision_fails_wire_validation() {
        let receipt = policy().evaluate(&consensus(10));
        let mut json = serde_json::to_value(&receipt).unwrap();
        json["decision"] = serde_json::json!("rejected");
        assert!(serde_json::from_value::<CalibrationDecisionReceipt>(json).is_err());
    }

    #[test]
    fn invalid_policy_wire_fails_validation() {
        let policy = policy();
        let mut json = serde_json::to_value(&policy).unwrap();
        json["practical_failure_min_radius_us"] = serde_json::json!(10);
        assert!(serde_json::from_value::<CalibrationDecisionPolicy>(json).is_err());
    }
}
