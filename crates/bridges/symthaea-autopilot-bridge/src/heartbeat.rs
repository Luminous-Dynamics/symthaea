// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bound MAVLink heartbeat receipt and freshness assessment.
//!
//! MAVLink `HEARTBEAT` carries no source timestamp. Freshness is therefore a
//! proposition about when a specific heartbeat was **received** on a local clock
//! and how old that receipt is under one explicit max-age policy. The immutable
//! receipt itself is not sufficient evidence for the time-varying proposition
//! “heartbeat is fresh now”; [`HeartbeatFreshnessAssessmentV1`] binds receipt,
//! policy, and evaluation time together.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{EvidenceValidationError, TimestampV1};
use thiserror::Error;

use crate::readiness::{
    Px4ReadinessFactV1, Px4ReadinessRequirementV1, Px4ReadinessValidationError,
};

/// Schema version for heartbeat receipt/freshness evidence.
pub const MAVLINK_HEARTBEAT_FRESHNESS_SCHEMA_V1: u16 = 1;
const RECEIPT_DOMAIN_V1: &[u8] = b"symthaea.autopilot.mavlink.heartbeat-receipt.v1\0";
const PROFILE_DOMAIN_V1: &[u8] = b"symthaea.autopilot.mavlink.heartbeat-freshness-profile.v1\0";
const ASSESSMENT_DOMAIN_V1: &[u8] =
    b"symthaea.autopilot.mavlink.heartbeat-freshness-assessment.v1\0";

/// Content-addressed observation of one received MAVLink `HEARTBEAT`.
///
/// The MAVLink header/source identity and raw heartbeat fields are preserved
/// without assigning additional semantics to unknown enum values. `received_at`
/// is a local receive-clock timestamp; no vehicle/source timestamp is invented.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MavlinkHeartbeatReceiptV1 {
    /// Schema version. Must equal [`MAVLINK_HEARTBEAT_FRESHNESS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Adapter/producer profile that captured this receipt.
    pub producer_profile_id: String,
    /// Stable identity of the transport/link on which the heartbeat arrived.
    pub transport_link_id: String,
    /// Raw MAVLink source system ID from the received packet header.
    pub source_system_id: u8,
    /// Raw MAVLink source component ID from the received packet header.
    pub source_component_id: u8,
    /// Raw MAVLink packet sequence number when captured.
    pub message_sequence: u8,
    /// Raw HEARTBEAT `type` / MAV_TYPE value.
    pub mav_type: u8,
    /// Raw HEARTBEAT `autopilot` / MAV_AUTOPILOT value.
    pub autopilot: u8,
    /// Raw HEARTBEAT `base_mode` bitmap.
    pub base_mode: u8,
    /// Raw HEARTBEAT autopilot-specific `custom_mode` bitfield.
    pub custom_mode: u32,
    /// Raw HEARTBEAT `system_status` / MAV_STATE value.
    pub system_status: u8,
    /// Raw HEARTBEAT `mavlink_version` value.
    pub mavlink_version: u8,
    /// Local time at which this exact heartbeat was received.
    pub received_at: TimestampV1,
    /// Domain-separated content commitment over the complete receipt.
    pub receipt_digest_hex: String,
}

impl MavlinkHeartbeatReceiptV1 {
    /// Construct, content-bind, and validate a received heartbeat.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        transport_link_id: impl Into<String>,
        source_system_id: u8,
        source_component_id: u8,
        message_sequence: u8,
        mav_type: u8,
        autopilot: u8,
        base_mode: u8,
        custom_mode: u32,
        system_status: u8,
        mavlink_version: u8,
        received_at: TimestampV1,
    ) -> Result<Self, HeartbeatEvidenceError> {
        let mut value = Self {
            schema_version: MAVLINK_HEARTBEAT_FRESHNESS_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            transport_link_id: transport_link_id.into(),
            source_system_id,
            source_component_id,
            message_sequence,
            mav_type,
            autopilot,
            base_mode,
            custom_mode,
            system_status,
            mavlink_version,
            received_at,
            receipt_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.receipt_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate the receipt structure and content commitment.
    pub fn validate(&self) -> Result<(), HeartbeatEvidenceError> {
        self.validate_without_digest()?;
        if self.receipt_digest_hex != self.compute_digest_hex() {
            return Err(HeartbeatEvidenceError::DigestMismatch("heartbeat_receipt"));
        }
        Ok(())
    }

    /// Content-addressed identity for this exact heartbeat receipt.
    pub fn receipt_id(&self) -> Result<String, HeartbeatEvidenceError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.mavlink.heartbeat-receipt.v1:{}",
            self.receipt_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), HeartbeatEvidenceError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.transport_link_id, "transport_link_id")?;
        self.received_at
            .validate()
            .map_err(HeartbeatEvidenceError::Timestamp)
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(RECEIPT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.transport_link_id);
        hasher.update(&[
            self.source_system_id,
            self.source_component_id,
            self.message_sequence,
            self.mav_type,
            self.autopilot,
            self.base_mode,
        ]);
        hasher.update(&self.custom_mode.to_le_bytes());
        hasher.update(&[self.system_status, self.mavlink_version]);
        feed_timestamp(&mut hasher, &self.received_at);
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Content-addressed policy describing how old a received heartbeat may be
/// before the `HeartbeatFresh` readiness proposition becomes unsatisfied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatFreshnessProfileV1 {
    /// Schema version. Must equal [`MAVLINK_HEARTBEAT_FRESHNESS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Stable identity of the deployment/channel freshness policy.
    pub profile_id: String,
    /// Maximum allowed receipt age in nanoseconds. Must be non-zero.
    pub max_age_ns: u64,
    /// Domain-separated content commitment over the policy.
    pub profile_digest_hex: String,
}

impl HeartbeatFreshnessProfileV1 {
    /// Construct, content-bind, and validate a heartbeat freshness profile.
    pub fn new(
        profile_id: impl Into<String>,
        max_age_ns: u64,
    ) -> Result<Self, HeartbeatEvidenceError> {
        let mut value = Self {
            schema_version: MAVLINK_HEARTBEAT_FRESHNESS_SCHEMA_V1,
            profile_id: profile_id.into(),
            max_age_ns,
            profile_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.profile_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate the freshness policy and content commitment.
    pub fn validate(&self) -> Result<(), HeartbeatEvidenceError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(HeartbeatEvidenceError::DigestMismatch("freshness_profile"));
        }
        Ok(())
    }

    /// Content-addressed identity for this exact freshness policy.
    pub fn profile_identity(&self) -> Result<String, HeartbeatEvidenceError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.mavlink.heartbeat-freshness-profile.v1:{}",
            self.profile_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), HeartbeatEvidenceError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        if self.max_age_ns == 0 {
            return Err(HeartbeatEvidenceError::ZeroMaxAge);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile_id);
        hasher.update(&self.max_age_ns.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Content-addressed evaluation of one heartbeat receipt under one exact
/// freshness policy at one explicit local-clock instant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatFreshnessAssessmentV1 {
    /// Schema version. Must equal [`MAVLINK_HEARTBEAT_FRESHNESS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Exact heartbeat receipt being evaluated.
    pub receipt: MavlinkHeartbeatReceiptV1,
    /// Exact freshness policy used for the evaluation.
    pub profile: HeartbeatFreshnessProfileV1,
    /// Local time at which freshness was evaluated.
    pub evaluated_at: TimestampV1,
    /// Domain-separated content commitment over receipt, policy, and evaluation time.
    pub assessment_digest_hex: String,
}

impl HeartbeatFreshnessAssessmentV1 {
    /// Construct, content-bind, and validate one freshness assessment.
    pub fn new(
        receipt: MavlinkHeartbeatReceiptV1,
        profile: HeartbeatFreshnessProfileV1,
        evaluated_at: TimestampV1,
    ) -> Result<Self, HeartbeatEvidenceError> {
        let mut value = Self {
            schema_version: MAVLINK_HEARTBEAT_FRESHNESS_SCHEMA_V1,
            receipt,
            profile,
            evaluated_at,
            assessment_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.assessment_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate nested evidence, same-clock temporal relation, and commitment.
    pub fn validate(&self) -> Result<(), HeartbeatEvidenceError> {
        self.validate_without_digest()?;
        if self.assessment_digest_hex != self.compute_digest_hex() {
            return Err(HeartbeatEvidenceError::DigestMismatch(
                "freshness_assessment",
            ));
        }
        Ok(())
    }

    /// Content-addressed identity for this exact freshness evaluation.
    pub fn assessment_id(&self) -> Result<String, HeartbeatEvidenceError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.mavlink.heartbeat-freshness-assessment.v1:{}",
            self.assessment_digest_hex
        ))
    }

    /// Age of the heartbeat receipt at this exact evaluation instant.
    pub fn age_ns(&self) -> Result<u64, HeartbeatEvidenceError> {
        self.validate()?;
        self.evaluated_at
            .elapsed_since(&self.receipt.received_at)
            .map_err(HeartbeatEvidenceError::Timestamp)
    }

    /// Whether the heartbeat is fresh under this exact profile and evaluation.
    pub fn is_fresh(&self) -> Result<bool, HeartbeatEvidenceError> {
        Ok(self.age_ns()? <= self.profile.max_age_ns)
    }

    /// Derive the PX4 readiness `HeartbeatFresh` fact from this exact assessment.
    ///
    /// The fact's evidence ID is the freshness-assessment identity, not merely
    /// the immutable heartbeat receipt, because freshness depends on policy and
    /// evaluation time.
    pub fn readiness_fact(&self) -> Result<Px4ReadinessFactV1, HeartbeatEvidenceError> {
        let evidence_id = self.assessment_id()?;
        if self.is_fresh()? {
            Px4ReadinessFactV1::satisfied(
                Px4ReadinessRequirementV1::HeartbeatFresh,
                evidence_id,
            )
        } else {
            Px4ReadinessFactV1::unsatisfied(
                Px4ReadinessRequirementV1::HeartbeatFresh,
                evidence_id,
            )
        }
        .map_err(HeartbeatEvidenceError::Readiness)
    }

    fn validate_without_digest(&self) -> Result<(), HeartbeatEvidenceError> {
        validate_schema(self.schema_version)?;
        self.receipt.validate()?;
        self.profile.validate()?;
        self.evaluated_at
            .validate()
            .map_err(HeartbeatEvidenceError::Timestamp)?;
        self.evaluated_at
            .elapsed_since(&self.receipt.received_at)
            .map_err(HeartbeatEvidenceError::Timestamp)?;
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSESSMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.receipt.receipt_digest_hex);
        feed_str(&mut hasher, &self.profile.profile_digest_hex);
        feed_timestamp(&mut hasher, &self.evaluated_at);
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_schema(schema_version: u16) -> Result<(), HeartbeatEvidenceError> {
    if schema_version != MAVLINK_HEARTBEAT_FRESHNESS_SCHEMA_V1 {
        return Err(HeartbeatEvidenceError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), HeartbeatEvidenceError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(HeartbeatEvidenceError::InvalidIdentifier(field));
    }
    Ok(())
}

fn feed_timestamp(hasher: &mut blake3::Hasher, timestamp: &TimestampV1) {
    feed_str(hasher, timestamp.clock_domain.as_str());
    hasher.update(&timestamp.nanoseconds.to_le_bytes());
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Validation failure for MAVLink heartbeat receipt/freshness evidence.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum HeartbeatEvidenceError {
    /// Record uses an unsupported schema version.
    #[error("unsupported MAVLink heartbeat evidence schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Required identifier is empty, padded, too long, or contains controls.
    #[error("invalid MAVLink heartbeat evidence identifier field {0}")]
    InvalidIdentifier(&'static str),
    /// Freshness profile cannot define an empty time window.
    #[error("heartbeat freshness max age must be greater than zero")]
    ZeroMaxAge,
    /// Receipt/evaluation timestamp or same-clock temporal relation is invalid.
    #[error("invalid MAVLink heartbeat timestamp relation: {0}")]
    Timestamp(EvidenceValidationError),
    /// Stored commitment no longer matches the record fields.
    #[error("MAVLink heartbeat {0} content commitment mismatch")]
    DigestMismatch(&'static str),
    /// Derived PX4 readiness fact failed readiness-layer validation.
    #[error("invalid derived heartbeat readiness fact: {0}")]
    Readiness(Px4ReadinessValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::embodiment_evidence::ClockDomainId;

    fn ts(clock: &str, nanoseconds: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new(clock).unwrap(), nanoseconds)
    }

    fn receipt() -> MavlinkHeartbeatReceiptV1 {
        MavlinkHeartbeatReceiptV1::new(
            "mavlink.heartbeat.fixture.v1",
            "udp:127.0.0.1:14540",
            1,
            1,
            42,
            2,
            12,
            0x81,
            0x1234_5678,
            4,
            3,
            ts("companion.steady", 1_000),
        )
        .unwrap()
    }

    fn profile(max_age_ns: u64) -> HeartbeatFreshnessProfileV1 {
        HeartbeatFreshnessProfileV1::new("mavlink.heartbeat.fixture-timeout.v1", max_age_ns)
            .unwrap()
    }

    #[test]
    fn receipt_preserves_raw_header_and_heartbeat_fields() {
        let value = receipt();
        assert_eq!(value.transport_link_id, "udp:127.0.0.1:14540");
        assert_eq!(value.source_system_id, 1);
        assert_eq!(value.source_component_id, 1);
        assert_eq!(value.message_sequence, 42);
        assert_eq!(value.mav_type, 2);
        assert_eq!(value.autopilot, 12);
        assert_eq!(value.base_mode, 0x81);
        assert_eq!(value.custom_mode, 0x1234_5678);
        assert_eq!(value.system_status, 4);
        assert_eq!(value.mavlink_version, 3);
        assert_eq!(value.received_at.nanoseconds, 1_000);
    }

    #[test]
    fn exact_max_age_boundary_is_fresh() {
        let assessment = HeartbeatFreshnessAssessmentV1::new(
            receipt(),
            profile(500),
            ts("companion.steady", 1_500),
        )
        .unwrap();
        assert_eq!(assessment.age_ns().unwrap(), 500);
        assert!(assessment.is_fresh().unwrap());
        assert_eq!(
            assessment.readiness_fact().unwrap().state,
            crate::readiness::Px4ReadinessFactStateV1::Satisfied
        );
    }

    #[test]
    fn age_beyond_profile_is_unsatisfied() {
        let assessment = HeartbeatFreshnessAssessmentV1::new(
            receipt(),
            profile(499),
            ts("companion.steady", 1_500),
        )
        .unwrap();
        assert!(!assessment.is_fresh().unwrap());
        assert_eq!(
            assessment.readiness_fact().unwrap().state,
            crate::readiness::Px4ReadinessFactStateV1::Unsatisfied
        );
    }

    #[test]
    fn policy_changes_identity_and_can_change_outcome_for_same_receipt() {
        let heartbeat = receipt();
        let fresh = HeartbeatFreshnessAssessmentV1::new(
            heartbeat.clone(),
            profile(600),
            ts("companion.steady", 1_500),
        )
        .unwrap();
        let stale = HeartbeatFreshnessAssessmentV1::new(
            heartbeat,
            profile(400),
            ts("companion.steady", 1_500),
        )
        .unwrap();

        assert!(fresh.is_fresh().unwrap());
        assert!(!stale.is_fresh().unwrap());
        assert_ne!(fresh.assessment_id().unwrap(), stale.assessment_id().unwrap());
    }

    #[test]
    fn cross_clock_and_backward_evaluation_fail_closed() {
        assert!(matches!(
            HeartbeatFreshnessAssessmentV1::new(
                receipt(),
                profile(500),
                ts("px4.hrt", 1_500),
            ),
            Err(HeartbeatEvidenceError::Timestamp(
                EvidenceValidationError::ClockDomainMismatch
            ))
        ));

        assert!(matches!(
            HeartbeatFreshnessAssessmentV1::new(
                receipt(),
                profile(500),
                ts("companion.steady", 999),
            ),
            Err(HeartbeatEvidenceError::Timestamp(
                EvidenceValidationError::NonMonotonicTimestamp
            ))
        ));
    }

    #[test]
    fn zero_max_age_is_rejected() {
        assert_eq!(
            HeartbeatFreshnessProfileV1::new("profile:v1", 0),
            Err(HeartbeatEvidenceError::ZeroMaxAge)
        );
    }

    #[test]
    fn mutation_invalidates_receipt_profile_and_assessment() {
        let mut heartbeat = receipt();
        heartbeat.message_sequence = heartbeat.message_sequence.wrapping_add(1);
        assert_eq!(
            heartbeat.validate(),
            Err(HeartbeatEvidenceError::DigestMismatch("heartbeat_receipt"))
        );

        let mut freshness = profile(500);
        freshness.max_age_ns = 501;
        assert_eq!(
            freshness.validate(),
            Err(HeartbeatEvidenceError::DigestMismatch("freshness_profile"))
        );

        let mut assessment = HeartbeatFreshnessAssessmentV1::new(
            receipt(),
            profile(500),
            ts("companion.steady", 1_500),
        )
        .unwrap();
        assessment.evaluated_at.nanoseconds = 1_499;
        assert_eq!(
            assessment.validate(),
            Err(HeartbeatEvidenceError::DigestMismatch(
                "freshness_assessment"
            ))
        );
    }

    #[test]
    fn readiness_fact_cites_freshness_assessment_not_receipt() {
        let assessment = HeartbeatFreshnessAssessmentV1::new(
            receipt(),
            profile(500),
            ts("companion.steady", 1_500),
        )
        .unwrap();
        let assessment_id = assessment.assessment_id().unwrap();
        let receipt_id = assessment.receipt.receipt_id().unwrap();
        let fact = assessment.readiness_fact().unwrap();

        assert_eq!(fact.requirement, Px4ReadinessRequirementV1::HeartbeatFresh);
        assert_eq!(fact.evidence_id.as_deref(), Some(assessment_id.as_str()));
        assert_ne!(fact.evidence_id.as_deref(), Some(receipt_id.as_str()));
    }

    #[test]
    fn serde_round_trips_preserve_validated_identities() {
        let assessment = HeartbeatFreshnessAssessmentV1::new(
            receipt(),
            profile(500),
            ts("companion.steady", 1_500),
        )
        .unwrap();
        let receipt_id = assessment.receipt.receipt_id().unwrap();
        let profile_id = assessment.profile.profile_identity().unwrap();
        let assessment_id = assessment.assessment_id().unwrap();

        let bytes = serde_json::to_vec(&assessment).unwrap();
        let restored: HeartbeatFreshnessAssessmentV1 = serde_json::from_slice(&bytes).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.receipt.receipt_id().unwrap(), receipt_id);
        assert_eq!(restored.profile.profile_identity().unwrap(), profile_id);
        assert_eq!(restored.assessment_id().unwrap(), assessment_id);
    }
}
