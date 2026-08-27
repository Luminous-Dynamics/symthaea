// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Content identity for the exact calibration + holdover authority that permits
//! one chemical acquisition timestamp to be normalized into another timebase.
//!
//! The derived [`symthaea_time_normalization::ClockTransformReceipt`] is not by
//! itself enough to recover which calibration exchanges, frozen policy, and
//! holdover claims earned that transform. This module content-addresses the full
//! self-verifying [`BoundedHoldoverTransform`] semantically so downstream timing
//! authorization can retain that distinction without importing calibration
//! internals into root cognition.
//!
//! This is content identity only. It is not a signature, trusted timestamp,
//! synchronization authority, oscillator attestation, or proof that the producer
//! was authorized to make the underlying timing claims.

use std::fmt;

use blake3::Hasher;
use symthaea_evidence_plane::{ContentAddress32, ContentAddressError};
use symthaea_time_calibration::{
    ClockCalibrationEvidence, ClockOffsetIntervalUs, FourTimestampExchange, TimestampEvidence,
};
use symthaea_time_calibration_bundle::CalibrationDecisionBundle;
use symthaea_time_calibration_policy::{
    CalibrationDecision, CalibrationDecisionPolicy, CalibrationDecisionReceipt,
};
use symthaea_time_holdover::{BoundedHoldoverTransform, HoldoverClaim, HoldoverError};
use symthaea_time_integrity::{ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty};
use symthaea_time_normalization::{ClockTransformModel, ClockTransformReceipt};

pub const CHEMICAL_ACQUISITION_TIME_AUTHORIZATION_NAMESPACE: &str =
    "symthaea-chemosensation-acquisition-time-authorization-v1";
const BLAKE3_256: &str = "blake3-256";
const HASH_DOMAIN: &[u8] = b"symthaea-chemosensation-acquisition-time-authorization-v1";
const CALIBRATION_EVIDENCE_HASH_DOMAIN: &[u8] =
    b"symthaea-chemosensation-acquisition-calibration-evidence-v1";

/// Strong domain identity for one exact calibration-decision + holdover authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChemicalAcquisitionTimeAuthorizationId([u8; 32]);

impl ChemicalAcquisitionTimeAuthorizationId {
    /// Reverify and content-address the complete finite holdover authority.
    ///
    /// Calibration evidence records are treated as a set because calibration
    /// consensus is interval intersection and the bundle already rejects exact
    /// duplicates. Each record is hashed in its semantic exchange role first,
    /// then those record digests are sorted before entering the bundle identity.
    pub fn from_holdover(
        holdover: &BoundedHoldoverTransform,
    ) -> Result<Self, ChemicalAcquisitionTimeAuthorizationError> {
        holdover.verify_self()?;

        let mut hasher = Hasher::new();
        hasher.update(HASH_DOMAIN);
        hash_calibration_bundle(&mut hasher, holdover.calibration());
        hash_holdover_claim(&mut hasher, holdover.holdover());
        hash_transform(&mut hasher, holdover.transform());
        Ok(Self(*hasher.finalize().as_bytes()))
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn content_address(&self) -> Result<ContentAddress32, ContentAddressError> {
        ContentAddress32::new(
            BLAKE3_256,
            CHEMICAL_ACQUISITION_TIME_AUTHORIZATION_NAMESPACE,
            self.0,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalAcquisitionTimeAuthorizationError {
    Holdover(HoldoverError),
}

impl fmt::Display for ChemicalAcquisitionTimeAuthorizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Holdover(error) => write!(
                f,
                "acquisition-time authority failed self-verification: {error}"
            ),
        }
    }
}

impl std::error::Error for ChemicalAcquisitionTimeAuthorizationError {}

impl From<HoldoverError> for ChemicalAcquisitionTimeAuthorizationError {
    fn from(value: HoldoverError) -> Self {
        Self::Holdover(value)
    }
}

fn hash_calibration_bundle(hasher: &mut Hasher, bundle: &CalibrationDecisionBundle) {
    hash_decision_receipt(hasher, bundle.decision());

    let mut evidence_digests: Vec<[u8; 32]> = bundle
        .evidence()
        .iter()
        .map(calibration_evidence_digest)
        .collect();
    evidence_digests.sort_unstable();

    hash_u64(hasher, evidence_digests.len() as u64);
    for digest in evidence_digests {
        hasher.update(&digest);
    }
}

fn calibration_evidence_digest(evidence: &ClockCalibrationEvidence) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(CALIBRATION_EVIDENCE_HASH_DOMAIN);
    hash_exchange(&mut hasher, evidence.exchange());
    hash_offset_interval(&mut hasher, evidence.offset_interval());
    *hasher.finalize().as_bytes()
}

fn hash_exchange(hasher: &mut Hasher, exchange: &FourTimestampExchange) {
    // Positional tags make request/response roles identity-significant even if
    // two stamps happen to contain identical values.
    hash_tag(hasher, 0);
    hash_timestamp_evidence(hasher, exchange.source_send());
    hash_tag(hasher, 1);
    hash_timestamp_evidence(hasher, exchange.target_receive());
    hash_tag(hasher, 2);
    hash_timestamp_evidence(hasher, exchange.target_send());
    hash_tag(hasher, 3);
    hash_timestamp_evidence(hasher, exchange.source_receive());
}

fn hash_timestamp_evidence(hasher: &mut Hasher, evidence: &TimestampEvidence) {
    hash_u64(hasher, evidence.timestamp_us);
    hash_time_receipt(hasher, &evidence.receipt);
}

fn hash_decision_receipt(hasher: &mut Hasher, receipt: &CalibrationDecisionReceipt) {
    hash_policy(hasher, receipt.policy());
    hash_str(hasher, receipt.source_domain().as_str());
    hash_str(hasher, receipt.source_epoch().as_str());
    hash_str(hasher, receipt.target_domain().as_str());
    hash_str(hasher, receipt.target_epoch().as_str());
    hash_offset_interval(hasher, receipt.offset_interval());
    hash_tag(
        hasher,
        match receipt.decision() {
            CalibrationDecision::Accepted => 0,
            CalibrationDecision::Rejected => 1,
            CalibrationDecision::Inconclusive => 2,
        },
    );
}

fn hash_policy(hasher: &mut Hasher, policy: &CalibrationDecisionPolicy) {
    hash_str(hasher, policy.policy_id().as_str());
    hash_u32(hasher, policy.version());
    hash_u64(hasher, policy.acceptance_max_radius_us());
    hash_optional_u64(hasher, policy.practical_failure_min_radius_us());
}

fn hash_offset_interval(hasher: &mut Hasher, interval: ClockOffsetIntervalUs) {
    hash_i128(hasher, interval.lower_us());
    hash_i128(hasher, interval.upper_us());
}

fn hash_holdover_claim(hasher: &mut Hasher, holdover: &HoldoverClaim) {
    let (start_us, end_us) = holdover.valid_source_range_us();
    hash_u64(hasher, start_us);
    hash_u64(hasher, end_us);
    hash_u64(hasher, holdover.max_relative_drift_ppb());
    hash_u64(hasher, holdover.fixed_model_error_us());
    hash_continuity(hasher, holdover.source_continuity());
    hash_continuity(hasher, holdover.mapping_continuity());
    hash_continuity(hasher, holdover.target_continuity());
}

fn hash_transform(hasher: &mut Hasher, transform: &ClockTransformReceipt) {
    hash_str(hasher, transform.source_domain().as_str());
    hash_str(hasher, transform.source_epoch().as_str());
    hash_str(hasher, transform.target_domain().as_str());
    hash_str(hasher, transform.target_epoch().as_str());
    match transform.model() {
        ClockTransformModel::Offset {
            source_anchor_us,
            target_anchor_us,
        } => {
            hash_tag(hasher, 0);
            hash_u64(hasher, *source_anchor_us);
            hash_u64(hasher, *target_anchor_us);
        }
    }
    let (start_us, end_us) = transform.valid_source_range_us();
    hash_u64(hasher, start_us);
    hash_u64(hasher, end_us);
    hash_continuity(hasher, transform.mapping_continuity());
    hash_continuity(hasher, transform.target_continuity());
    hash_uncertainty(hasher, transform.uncertainty());
    hash_optional_u64(hasher, transform.sequence());
}

fn hash_time_receipt(hasher: &mut Hasher, receipt: &TimeIntegrityReceipt) {
    hash_str(hasher, receipt.clock_domain.as_str());
    match receipt.clock_epoch.as_ref() {
        None => hash_tag(hasher, 0),
        Some(epoch) => {
            hash_tag(hasher, 1);
            hash_str(hasher, epoch.as_str());
        }
    }
    hash_continuity(hasher, receipt.continuity);
    hash_uncertainty(hasher, receipt.uncertainty);
    hash_optional_u64(hasher, receipt.sequence);
}

fn hash_continuity(hasher: &mut Hasher, status: ContinuityStatus) {
    hash_tag(
        hasher,
        match status {
            ContinuityStatus::Unverified => 0,
            ContinuityStatus::Continuous => 1,
            ContinuityStatus::Broken => 2,
        },
    );
}

fn hash_uncertainty(hasher: &mut Hasher, uncertainty: TimeUncertainty) {
    match uncertainty {
        TimeUncertainty::Unbounded => hash_tag(hasher, 0),
        TimeUncertainty::Bounded { max_error_us } => {
            hash_tag(hasher, 1);
            hash_u64(hasher, max_error_us);
        }
    }
}

fn hash_optional_u64(hasher: &mut Hasher, value: Option<u64>) {
    match value {
        None => hash_tag(hasher, 0),
        Some(value) => {
            hash_tag(hasher, 1);
            hash_u64(hasher, value);
        }
    }
}

fn hash_tag(hasher: &mut Hasher, value: u8) {
    hasher.update(&[value]);
}

fn hash_u32(hasher: &mut Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn hash_u64(hasher: &mut Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn hash_i128(hasher: &mut Hasher, value: i128) {
    hasher.update(&value.to_le_bytes());
}

fn hash_str(hasher: &mut Hasher, value: &str) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_time_calibration::{
        CalibrationConsensus, FourTimestampExchange, TimestampEvidence,
    };
    use symthaea_time_calibration_bundle::CalibrationDecisionBundle;
    use symthaea_time_calibration_policy::{
        CalibrationDecisionPolicy, CalibrationPolicyId,
    };
    use symthaea_time_integrity::{
        ClockDomainId, ClockEpochId, ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty,
    };

    fn source_domain() -> ClockDomainId {
        ClockDomainId::new("sensor-a/monotonic").unwrap()
    }

    fn source_epoch() -> ClockEpochId {
        ClockEpochId::new("sensor-a-boot-7").unwrap()
    }

    fn target_domain() -> ClockDomainId {
        ClockDomainId::new("capture-host/monotonic").unwrap()
    }

    fn target_epoch() -> ClockEpochId {
        ClockEpochId::new("capture-host-boot-3").unwrap()
    }

    fn receipt(domain: ClockDomainId, epoch: ClockEpochId, sequence: u64) -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(domain)
            .with_epoch(epoch)
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(0))
            .with_sequence(sequence)
    }

    fn stamp(
        timestamp_us: u64,
        domain: ClockDomainId,
        epoch: ClockEpochId,
        sequence: u64,
    ) -> TimestampEvidence {
        TimestampEvidence::new(timestamp_us, receipt(domain, epoch, sequence))
    }

    fn evidence(source_start_us: u64, sequence_base: u64) -> ClockCalibrationEvidence {
        let offset = 500;
        let delay = 10;
        let processing = 10;
        let t1 = source_start_us;
        let t2 = t1 + offset + delay;
        let t3 = t2 + processing;
        let t4 = t1 + delay + processing + delay;
        ClockCalibrationEvidence::derive(
            FourTimestampExchange::new(
                stamp(t1, source_domain(), source_epoch(), sequence_base),
                stamp(t2, target_domain(), target_epoch(), sequence_base + 1),
                stamp(t3, target_domain(), target_epoch(), sequence_base + 2),
                stamp(t4, source_domain(), source_epoch(), sequence_base + 3),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn holdover_with(evidence: Vec<ClockCalibrationEvidence>) -> BoundedHoldoverTransform {
        let consensus = CalibrationConsensus::from_evidence(&evidence).unwrap();
        let policy = CalibrationDecisionPolicy::new(
            CalibrationPolicyId::new("chemical-acquisition-v1").unwrap(),
            1,
            20,
            Some(100),
        )
        .unwrap();
        let decision = policy.evaluate(&consensus);
        let bundle = CalibrationDecisionBundle::new(decision, evidence).unwrap();
        let holdover = HoldoverClaim::new(
            900,
            2_200,
            0,
            0,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
        )
        .unwrap();
        BoundedHoldoverTransform::new(bundle, holdover).unwrap()
    }

    #[test]
    fn same_authority_is_deterministic() {
        let holdover = holdover_with(vec![evidence(1_000, 10)]);
        assert_eq!(
            ChemicalAcquisitionTimeAuthorizationId::from_holdover(&holdover).unwrap(),
            ChemicalAcquisitionTimeAuthorizationId::from_holdover(&holdover).unwrap()
        );
    }

    #[test]
    fn evidence_order_does_not_change_authority_identity() {
        let first = evidence(1_000, 10);
        let second = evidence(2_000, 20);
        let left = holdover_with(vec![first.clone(), second.clone()]);
        let right = holdover_with(vec![second, first]);
        assert_eq!(left.transform(), right.transform());
        assert_eq!(
            ChemicalAcquisitionTimeAuthorizationId::from_holdover(&left).unwrap(),
            ChemicalAcquisitionTimeAuthorizationId::from_holdover(&right).unwrap()
        );
    }

    #[test]
    fn calibration_metadata_change_is_visible_even_when_transform_is_identical() {
        let left = holdover_with(vec![evidence(1_000, 10)]);
        let right = holdover_with(vec![evidence(1_000, 100)]);
        assert_eq!(left.transform(), right.transform());
        assert_ne!(
            ChemicalAcquisitionTimeAuthorizationId::from_holdover(&left).unwrap(),
            ChemicalAcquisitionTimeAuthorizationId::from_holdover(&right).unwrap()
        );
    }

    #[test]
    fn generic_address_preserves_authority_namespace() {
        let holdover = holdover_with(vec![evidence(1_000, 10)]);
        let id = ChemicalAcquisitionTimeAuthorizationId::from_holdover(&holdover).unwrap();
        let address = id.content_address().unwrap();
        assert_eq!(address.algorithm(), BLAKE3_256);
        assert_eq!(
            address.namespace(),
            CHEMICAL_ACQUISITION_TIME_AUTHORIZATION_NAMESPACE
        );
        assert_eq!(address.digest(), id.as_bytes());
    }
}
