// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Receiver-clock MAVLink HEARTBEAT evidence and PX4 freshness qualification.
//!
//! MAVLink HEARTBEAT has no sender timestamp. Freshness therefore belongs to the
//! receiving adapter's explicit clock domain. MAVLink also deliberately leaves
//! heartbeat timeout/frequency policy to the channel/application, so this module
//! evaluates a versioned Symthaea profile rather than embedding one global timeout.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{ClockDomainId, EvidenceValidationError, TimestampV1};
use thiserror::Error;

use crate::readiness::{
    Px4ReadinessFactV1, Px4ReadinessRequirementV1, Px4ReadinessValidationError,
};

/// MAVLink `MAV_AUTOPILOT_PX4` raw value.
pub const MAV_AUTOPILOT_PX4_RAW: u8 = 12;
/// Evidence schema version for heartbeat records and qualification.
pub const PX4_HEARTBEAT_EVIDENCE_SCHEMA_V1: u16 = 1;
const HEARTBEAT_DOMAIN_V1: &[u8] = b"symthaea.autopilot.mavlink.heartbeat-evidence.v1\0";
const PROFILE_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.heartbeat-freshness-profile.v1\0";
const QUALIFICATION_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.heartbeat-freshness.v1\0";

/// Content-addressed preservation of one decoded MAVLink HEARTBEAT reception.
///
/// `system_id` and `component_id` come from the MAVLink message header. The
/// remaining raw fields are preserved from HEARTBEAT itself. `received_at` is a
/// local receive timestamp; this record invents no remote/source timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MavlinkHeartbeatEvidenceV1 {
    /// Schema version.
    pub schema_version: u16,
    /// Decoder/transport producer profile that observed this heartbeat.
    pub producer_profile_id: String,
    /// Source MAVLink system ID from the message header.
    pub system_id: u8,
    /// Source MAVLink component ID from the message header.
    pub component_id: u8,
    /// Raw HEARTBEAT `type` / MAV_TYPE.
    pub vehicle_type: u8,
    /// Raw HEARTBEAT `autopilot` / MAV_AUTOPILOT.
    pub autopilot: u8,
    /// Raw HEARTBEAT `base_mode` bitmap.
    pub base_mode: u8,
    /// Raw HEARTBEAT `custom_mode`.
    pub custom_mode: u32,
    /// Raw HEARTBEAT `system_status` / MAV_STATE.
    pub system_status: u8,
    /// Raw HEARTBEAT `mavlink_version`.
    pub mavlink_version: u8,
    /// Receiver-local timestamp when this heartbeat was accepted by the adapter.
    pub received_at: TimestampV1,
    /// Domain-separated content commitment.
    pub heartbeat_digest_hex: String,
}

impl MavlinkHeartbeatEvidenceV1 {
    /// Construct, content-bind, and validate one received heartbeat.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        system_id: u8,
        component_id: u8,
        vehicle_type: u8,
        autopilot: u8,
        base_mode: u8,
        custom_mode: u32,
        system_status: u8,
        mavlink_version: u8,
        received_at: TimestampV1,
    ) -> Result<Self, Px4HeartbeatEvidenceError> {
        let mut value = Self {
            schema_version: PX4_HEARTBEAT_EVIDENCE_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            system_id,
            component_id,
            vehicle_type,
            autopilot,
            base_mode,
            custom_mode,
            system_status,
            mavlink_version,
            received_at,
            heartbeat_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.heartbeat_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate source identity, receive clock, schema, and content commitment.
    pub fn validate(&self) -> Result<(), Px4HeartbeatEvidenceError> {
        self.validate_without_digest()?;
        if self.heartbeat_digest_hex != self.compute_digest_hex() {
            return Err(Px4HeartbeatEvidenceError::DigestMismatch("heartbeat"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact received heartbeat.
    pub fn heartbeat_id(&self) -> Result<String, Px4HeartbeatEvidenceError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.mavlink.heartbeat-evidence.v1:{}",
            self.heartbeat_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4HeartbeatEvidenceError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        if self.system_id == 0 {
            return Err(Px4HeartbeatEvidenceError::InvalidSourceSystemId);
        }
        if self.component_id == 0 {
            return Err(Px4HeartbeatEvidenceError::InvalidSourceComponentId);
        }
        self.received_at
            .validate()
            .map_err(Px4HeartbeatEvidenceError::Timestamp)
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(HEARTBEAT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        hasher.update(&[self.system_id, self.component_id, self.vehicle_type, self.autopilot]);
        hasher.update(&[self.base_mode]);
        hasher.update(&self.custom_mode.to_le_bytes());
        hasher.update(&[self.system_status, self.mavlink_version]);
        feed_str(&mut hasher, self.received_at.clock_domain.as_str());
        hasher.update(&self.received_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Versioned liveness policy for one expected PX4 MAVLink component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4HeartbeatFreshnessProfileV1 {
    /// Schema version.
    pub schema_version: u16,
    /// Stable versioned policy identity.
    pub profile_id: String,
    /// Expected MAVLink system ID of the bound vehicle.
    pub expected_system_id: u8,
    /// Expected MAVLink component ID of the bound PX4 autopilot component.
    pub expected_component_id: u8,
    /// Receiver clock in which freshness and gaps are evaluated.
    pub receiver_clock_domain: ClockDomainId,
    /// Minimum repeated heartbeat count required for a liveness verdict.
    pub min_samples: u16,
    /// Maximum age of the newest heartbeat, in nanoseconds.
    pub max_latest_age_ns: u64,
    /// Optional maximum gap between adjacent received heartbeats, in nanoseconds.
    pub max_inter_heartbeat_gap_ns: Option<u64>,
    /// Domain-separated content commitment.
    pub profile_digest_hex: String,
}

impl Px4HeartbeatFreshnessProfileV1 {
    /// Construct, content-bind, and validate one heartbeat freshness profile.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_id: impl Into<String>,
        expected_system_id: u8,
        expected_component_id: u8,
        receiver_clock_domain: ClockDomainId,
        min_samples: u16,
        max_latest_age_ns: u64,
        max_inter_heartbeat_gap_ns: Option<u64>,
    ) -> Result<Self, Px4HeartbeatEvidenceError> {
        let mut value = Self {
            schema_version: PX4_HEARTBEAT_EVIDENCE_SCHEMA_V1,
            profile_id: profile_id.into(),
            expected_system_id,
            expected_component_id,
            receiver_clock_domain,
            min_samples,
            max_latest_age_ns,
            max_inter_heartbeat_gap_ns,
            profile_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.profile_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate profile identity, target identity, clock, thresholds, and commitment.
    pub fn validate(&self) -> Result<(), Px4HeartbeatEvidenceError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4HeartbeatEvidenceError::DigestMismatch("freshness_profile"));
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), Px4HeartbeatEvidenceError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        self.receiver_clock_domain
            .validate()
            .map_err(Px4HeartbeatEvidenceError::Timestamp)?;
        if self.expected_system_id == 0 {
            return Err(Px4HeartbeatEvidenceError::InvalidSourceSystemId);
        }
        if self.expected_component_id == 0 {
            return Err(Px4HeartbeatEvidenceError::InvalidSourceComponentId);
        }
        if self.min_samples < 2 {
            return Err(Px4HeartbeatEvidenceError::MinimumSamplesTooSmall);
        }
        if self.max_latest_age_ns == 0 {
            return Err(Px4HeartbeatEvidenceError::ZeroLatestAge);
        }
        if self.max_inter_heartbeat_gap_ns == Some(0) {
            return Err(Px4HeartbeatEvidenceError::ZeroInterHeartbeatGap);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile_id);
        hasher.update(&[self.expected_system_id, self.expected_component_id]);
        feed_str(&mut hasher, self.receiver_clock_domain.as_str());
        hasher.update(&self.min_samples.to_le_bytes());
        hasher.update(&self.max_latest_age_ns.to_le_bytes());
        match self.max_inter_heartbeat_gap_ns {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_le_bytes());
            }
            None => hasher.update(&[0]),
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Failed freshness condition after sufficient target-matched evidence exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Px4HeartbeatFreshnessFailureV1 {
    /// The newest heartbeat is older than the profile permits.
    LatestHeartbeatStale,
    /// An adjacent heartbeat reception gap exceeds the profile ceiling.
    InterHeartbeatGapExceeded,
}

/// Derived freshness outcome; this is not independent evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Px4HeartbeatFreshnessOutcomeV1 {
    /// Repeated target-matched heartbeat evidence is current under the exact profile.
    Fresh,
    /// Enough target-matched evidence exists, but freshness/continuity failed.
    NotFresh {
        /// Deterministically ordered failed conditions.
        failures: Vec<Px4HeartbeatFreshnessFailureV1>,
    },
    /// Too few repeated heartbeats exist to make the liveness claim.
    Incomplete {
        /// Number of target-matched samples retained.
        available_samples: usize,
        /// Number required by the profile.
        required_samples: u16,
    },
}

/// Content-addressed receiver-clock heartbeat freshness evidence for one PX4 target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4HeartbeatFreshnessV1 {
    /// Schema version.
    pub schema_version: u16,
    /// Exact decoder/transport producer shared by retained heartbeat records.
    pub producer_profile_id: String,
    /// Exact target/freshness policy.
    pub profile: Px4HeartbeatFreshnessProfileV1,
    /// Receiver-local instant at which freshness is assessed.
    pub assessed_at: TimestampV1,
    /// Canonically ordered heartbeat receptions.
    pub samples: Vec<MavlinkHeartbeatEvidenceV1>,
    /// Domain-separated content commitment.
    pub freshness_digest_hex: String,
}

impl Px4HeartbeatFreshnessV1 {
    /// Construct, canonicalize, content-bind, and validate one freshness assessment.
    pub fn new(
        producer_profile_id: impl Into<String>,
        profile: Px4HeartbeatFreshnessProfileV1,
        assessed_at: TimestampV1,
        mut samples: Vec<MavlinkHeartbeatEvidenceV1>,
    ) -> Result<Self, Px4HeartbeatEvidenceError> {
        samples.sort_by_key(|sample| sample.received_at.nanoseconds);
        let mut value = Self {
            schema_version: PX4_HEARTBEAT_EVIDENCE_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            profile,
            assessed_at,
            samples,
            freshness_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.freshness_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate target identity, clocks, ordered evidence, and content commitment.
    pub fn validate(&self) -> Result<(), Px4HeartbeatEvidenceError> {
        self.validate_without_digest()?;
        if self.freshness_digest_hex != self.compute_digest_hex() {
            return Err(Px4HeartbeatEvidenceError::DigestMismatch("freshness"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact freshness assessment.
    pub fn freshness_id(&self) -> Result<String, Px4HeartbeatEvidenceError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.heartbeat-freshness.v1:{}",
            self.freshness_digest_hex
        ))
    }

    /// Derive freshness/currentness under the exact stored profile.
    pub fn outcome(&self) -> Result<Px4HeartbeatFreshnessOutcomeV1, Px4HeartbeatEvidenceError> {
        self.validate()?;
        if self.samples.len() < usize::from(self.profile.min_samples) {
            return Ok(Px4HeartbeatFreshnessOutcomeV1::Incomplete {
                available_samples: self.samples.len(),
                required_samples: self.profile.min_samples,
            });
        }

        let mut failures = Vec::new();
        let latest = self.samples.last().ok_or(Px4HeartbeatEvidenceError::EmptySampleWindow)?;
        let age_ns = self
            .assessed_at
            .elapsed_since(&latest.received_at)
            .map_err(Px4HeartbeatEvidenceError::Timestamp)?;
        if age_ns > self.profile.max_latest_age_ns {
            failures.push(Px4HeartbeatFreshnessFailureV1::LatestHeartbeatStale);
        }
        if let Some(limit) = self.profile.max_inter_heartbeat_gap_ns {
            let max_gap = self
                .samples
                .windows(2)
                .map(|pair| pair[1].received_at.nanoseconds - pair[0].received_at.nanoseconds)
                .max()
                .unwrap_or(0);
            if max_gap > limit {
                failures.push(Px4HeartbeatFreshnessFailureV1::InterHeartbeatGapExceeded);
            }
        }

        if failures.is_empty() {
            Ok(Px4HeartbeatFreshnessOutcomeV1::Fresh)
        } else {
            Ok(Px4HeartbeatFreshnessOutcomeV1::NotFresh { failures })
        }
    }

    /// Lower freshness into the PX4 readiness vocabulary.
    pub fn readiness_fact(&self) -> Result<Px4ReadinessFactV1, Px4HeartbeatEvidenceError> {
        match self.outcome()? {
            Px4HeartbeatFreshnessOutcomeV1::Fresh => Px4ReadinessFactV1::satisfied(
                Px4ReadinessRequirementV1::HeartbeatFresh,
                self.freshness_id()?,
            )
            .map_err(Px4HeartbeatEvidenceError::Readiness),
            Px4HeartbeatFreshnessOutcomeV1::NotFresh { .. } => Px4ReadinessFactV1::unsatisfied(
                Px4ReadinessRequirementV1::HeartbeatFresh,
                self.freshness_id()?,
            )
            .map_err(Px4HeartbeatEvidenceError::Readiness),
            Px4HeartbeatFreshnessOutcomeV1::Incomplete { .. } => Ok(
                Px4ReadinessFactV1::unobserved(Px4ReadinessRequirementV1::HeartbeatFresh),
            ),
        }
    }

    fn validate_without_digest(&self) -> Result<(), Px4HeartbeatEvidenceError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        self.profile.validate()?;
        self.assessed_at
            .validate()
            .map_err(Px4HeartbeatEvidenceError::Timestamp)?;
        if self.assessed_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4HeartbeatEvidenceError::ReceiverClockMismatch);
        }

        let mut seen = HashSet::with_capacity(self.samples.len());
        let mut previous_receive_ns = None;
        for sample in &self.samples {
            sample.validate()?;
            if sample.producer_profile_id != self.producer_profile_id {
                return Err(Px4HeartbeatEvidenceError::MixedProducer);
            }
            if sample.system_id != self.profile.expected_system_id {
                return Err(Px4HeartbeatEvidenceError::UnexpectedSystemId {
                    found: sample.system_id,
                    expected: self.profile.expected_system_id,
                });
            }
            if sample.component_id != self.profile.expected_component_id {
                return Err(Px4HeartbeatEvidenceError::UnexpectedComponentId {
                    found: sample.component_id,
                    expected: self.profile.expected_component_id,
                });
            }
            if sample.autopilot != MAV_AUTOPILOT_PX4_RAW {
                return Err(Px4HeartbeatEvidenceError::UnexpectedAutopilotClass {
                    found: sample.autopilot,
                });
            }
            if sample.received_at.clock_domain != self.profile.receiver_clock_domain {
                return Err(Px4HeartbeatEvidenceError::ReceiverClockMismatch);
            }
            if !seen.insert(sample.heartbeat_digest_hex.clone()) {
                return Err(Px4HeartbeatEvidenceError::DuplicateSample);
            }
            if previous_receive_ns.is_some_and(|previous| sample.received_at.nanoseconds <= previous) {
                return Err(Px4HeartbeatEvidenceError::NonAdvancingReceiveTime);
            }
            previous_receive_ns = Some(sample.received_at.nanoseconds);
        }

        if let Some(latest) = self.samples.last() {
            self.assessed_at
                .elapsed_since(&latest.received_at)
                .map_err(|error| match error {
                    EvidenceValidationError::NonMonotonicTimestamp => {
                        Px4HeartbeatEvidenceError::AssessmentBeforeLatestHeartbeat
                    }
                    other => Px4HeartbeatEvidenceError::Timestamp(other),
                })?;
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(QUALIFICATION_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.profile.profile_digest_hex);
        feed_str(&mut hasher, self.assessed_at.clock_domain.as_str());
        hasher.update(&self.assessed_at.nanoseconds.to_le_bytes());
        hasher.update(&(self.samples.len() as u64).to_le_bytes());
        for sample in &self.samples {
            feed_str(&mut hasher, &sample.heartbeat_digest_hex);
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_schema(schema_version: u16) -> Result<(), Px4HeartbeatEvidenceError> {
    if schema_version != PX4_HEARTBEAT_EVIDENCE_SCHEMA_V1 {
        return Err(Px4HeartbeatEvidenceError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), Px4HeartbeatEvidenceError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4HeartbeatEvidenceError::InvalidIdentifier(field));
    }
    Ok(())
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

/// Validation or derivation failure for heartbeat liveness evidence.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4HeartbeatEvidenceError {
    /// Unsupported schema version.
    #[error("unsupported PX4 heartbeat evidence schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Required identifier is empty, padded, too long, or contains controls.
    #[error("invalid {0}")]
    InvalidIdentifier(&'static str),
    /// MAVLink source system ID zero is not accepted for a concrete heartbeat source.
    #[error("invalid MAVLink source system id")]
    InvalidSourceSystemId,
    /// MAVLink source component ID zero is not accepted for a concrete heartbeat source.
    #[error("invalid MAVLink source component id")]
    InvalidSourceComponentId,
    /// Repeated liveness qualification requires at least two samples.
    #[error("PX4 heartbeat freshness min_samples must be at least 2")]
    MinimumSamplesTooSmall,
    /// Latest-age threshold cannot be zero.
    #[error("PX4 heartbeat max latest age must be non-zero")]
    ZeroLatestAge,
    /// Configured heartbeat gap threshold cannot be zero.
    #[error("PX4 heartbeat max inter-heartbeat gap must be non-zero when configured")]
    ZeroInterHeartbeatGap,
    /// Source evidence comes from another producer.
    #[error("PX4 heartbeat freshness mixes producer identities")]
    MixedProducer,
    /// Heartbeat belongs to another MAVLink system.
    #[error("heartbeat system id {found} does not match expected {expected}")]
    UnexpectedSystemId {
        /// Source system ID found.
        found: u8,
        /// Expected system ID.
        expected: u8,
    },
    /// Heartbeat belongs to another MAVLink component.
    #[error("heartbeat component id {found} does not match expected {expected}")]
    UnexpectedComponentId {
        /// Source component ID found.
        found: u8,
        /// Expected component ID.
        expected: u8,
    },
    /// Heartbeat does not identify a PX4 autopilot component.
    #[error("heartbeat autopilot class {found} is not MAV_AUTOPILOT_PX4")]
    UnexpectedAutopilotClass {
        /// Raw MAV_AUTOPILOT value found.
        found: u8,
    },
    /// Evidence/assessment clocks do not match the profile's receiver clock.
    #[error("heartbeat receiver clock does not match freshness profile")]
    ReceiverClockMismatch,
    /// Same heartbeat evidence was included more than once.
    #[error("PX4 heartbeat freshness contains a duplicate sample")]
    DuplicateSample,
    /// Receive timestamps do not advance strictly.
    #[error("PX4 heartbeat receive time does not advance strictly")]
    NonAdvancingReceiveTime,
    /// Assessment precedes the latest heartbeat in the same receiver clock.
    #[error("PX4 heartbeat assessment precedes the latest heartbeat")]
    AssessmentBeforeLatestHeartbeat,
    /// Metrics were requested for an empty evidence window.
    #[error("PX4 heartbeat sample window is empty")]
    EmptySampleWindow,
    /// Timestamp/clock evidence is invalid.
    #[error("invalid heartbeat timestamp: {0}")]
    Timestamp(EvidenceValidationError),
    /// Lowering into PX4 readiness vocabulary failed validation.
    #[error("invalid PX4 readiness fact: {0}")]
    Readiness(Px4ReadinessValidationError),
    /// Stored record no longer matches its content commitment.
    #[error("PX4 heartbeat {0} content commitment mismatch")]
    DigestMismatch(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readiness::Px4ReadinessFactStateV1;

    fn clock(name: &str) -> ClockDomainId {
        ClockDomainId::new(name).unwrap()
    }

    fn ts(ms: u64) -> TimestampV1 {
        TimestampV1::new(clock("companion.monotonic"), ms * 1_000_000)
    }

    fn heartbeat(ms: u64, system_id: u8, component_id: u8, autopilot: u8) -> MavlinkHeartbeatEvidenceV1 {
        MavlinkHeartbeatEvidenceV1::new(
            "mavlink.decoder.fixture.v1",
            system_id,
            component_id,
            2,
            autopilot,
            0,
            0,
            3,
            3,
            ts(ms),
        )
        .unwrap()
    }

    fn profile() -> Px4HeartbeatFreshnessProfileV1 {
        Px4HeartbeatFreshnessProfileV1::new(
            "px4.heartbeat.sitl.fixture.v1",
            1,
            1,
            clock("companion.monotonic"),
            3,
            1_500_000_000,
            Some(1_500_000_000),
        )
        .unwrap()
    }

    fn good_samples() -> Vec<MavlinkHeartbeatEvidenceV1> {
        vec![
            heartbeat(1_000, 1, 1, MAV_AUTOPILOT_PX4_RAW),
            heartbeat(2_000, 1, 1, MAV_AUTOPILOT_PX4_RAW),
            heartbeat(3_000, 1, 1, MAV_AUTOPILOT_PX4_RAW),
        ]
    }

    fn assessment(samples: Vec<MavlinkHeartbeatEvidenceV1>, now_ms: u64) -> Px4HeartbeatFreshnessV1 {
        Px4HeartbeatFreshnessV1::new(
            "mavlink.decoder.fixture.v1",
            profile(),
            ts(now_ms),
            samples,
        )
        .unwrap()
    }

    #[test]
    fn repeated_current_px4_heartbeat_is_fresh_and_satisfied() {
        let value = assessment(good_samples(), 3_500);
        assert_eq!(value.outcome().unwrap(), Px4HeartbeatFreshnessOutcomeV1::Fresh);
        let fact = value.readiness_fact().unwrap();
        assert_eq!(fact.requirement, Px4ReadinessRequirementV1::HeartbeatFresh);
        assert_eq!(fact.state, Px4ReadinessFactStateV1::Satisfied);
        assert_eq!(fact.evidence_id, Some(value.freshness_id().unwrap()));
    }

    #[test]
    fn timeout_and_gap_policy_are_profile_specific() {
        assert!(matches!(
            assessment(good_samples(), 4_501).outcome().unwrap(),
            Px4HeartbeatFreshnessOutcomeV1::NotFresh { failures }
                if failures.contains(&Px4HeartbeatFreshnessFailureV1::LatestHeartbeatStale)
        ));

        let gapped = vec![
            heartbeat(1_000, 1, 1, MAV_AUTOPILOT_PX4_RAW),
            heartbeat(3_000, 1, 1, MAV_AUTOPILOT_PX4_RAW),
            heartbeat(4_000, 1, 1, MAV_AUTOPILOT_PX4_RAW),
        ];
        assert!(matches!(
            assessment(gapped, 4_100).outcome().unwrap(),
            Px4HeartbeatFreshnessOutcomeV1::NotFresh { failures }
                if failures.contains(&Px4HeartbeatFreshnessFailureV1::InterHeartbeatGapExceeded)
        ));
    }

    #[test]
    fn insufficient_including_zero_samples_remains_unobserved() {
        for samples in [vec![], vec![heartbeat(1_000, 1, 1, MAV_AUTOPILOT_PX4_RAW)]] {
            let value = assessment(samples, 1_100);
            assert!(matches!(value.outcome().unwrap(), Px4HeartbeatFreshnessOutcomeV1::Incomplete { .. }));
            let fact = value.readiness_fact().unwrap();
            assert_eq!(fact.state, Px4ReadinessFactStateV1::Unobserved);
            assert!(fact.evidence_id.is_none());
        }
    }

    #[test]
    fn wrong_target_or_autopilot_is_unrelated_not_negative_liveness() {
        for samples in [
            vec![heartbeat(1_000, 2, 1, MAV_AUTOPILOT_PX4_RAW)],
            vec![heartbeat(1_000, 1, 2, MAV_AUTOPILOT_PX4_RAW)],
            vec![heartbeat(1_000, 1, 1, 3)],
        ] {
            assert!(Px4HeartbeatFreshnessV1::new(
                "mavlink.decoder.fixture.v1",
                profile(),
                ts(1_100),
                samples,
            )
            .is_err());
        }
    }

    #[test]
    fn receiver_clock_mismatch_and_backward_assessment_fail_closed() {
        assert_eq!(
            Px4HeartbeatFreshnessV1::new(
                "mavlink.decoder.fixture.v1",
                profile(),
                TimestampV1::new(clock("other.clock"), 4_000_000_000),
                good_samples(),
            )
            .unwrap_err(),
            Px4HeartbeatEvidenceError::ReceiverClockMismatch
        );
        assert_eq!(
            Px4HeartbeatFreshnessV1::new(
                "mavlink.decoder.fixture.v1",
                profile(),
                ts(2_999),
                good_samples(),
            )
            .unwrap_err(),
            Px4HeartbeatEvidenceError::AssessmentBeforeLatestHeartbeat
        );
    }

    #[test]
    fn mutation_and_serde_boundaries_are_explicit() {
        let value = assessment(good_samples(), 3_500);
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut restored: Px4HeartbeatFreshnessV1 = serde_json::from_slice(&bytes).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.freshness_id().unwrap(), value.freshness_id().unwrap());
        restored.samples[0].base_mode ^= 1;
        assert!(restored.validate().is_err());
    }
}
