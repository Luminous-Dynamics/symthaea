// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PX4 Offboard proof-of-life *submission-cadence* evidence.
//!
//! This module observes locally timestamped submission attempts for either a
//! supported MAVLink setpoint or ROS 2 OffboardControlMode. A local transport
//! acceptance is not proof of wire delivery or PX4 receipt. The assessment does
//! not establish Offboard mode, whether a mode-entry event happened after this
//! window, estimator validity, authorization, failsafe behavior, or physical
//! response. It deliberately does not lower into readiness.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{ClockDomainId, EvidenceValidationError, TimestampV1};
use thiserror::Error;

/// Schema version for Offboard submission records, profiles, and assessments.
pub const PX4_OFFBOARD_SUBMISSION_SCHEMA_V1: u16 = 1;
const SUBMISSION_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.offboard-submission.v1\0";
const PROFILE_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.offboard-submission-profile.v1\0";
const ASSESSMENT_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.offboard-submission-assessment.v1\0";

/// Outbound transport contract whose local submissions are being observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4OffboardTransportV1 {
    /// MAVLink stream where an approved setpoint message carries proof of life.
    MavlinkSetpoint,
    /// ROS 2 stream where OffboardControlMode carries proof of life separately
    /// from the corresponding setpoint payload.
    Ros2OffboardControlMode,
}

/// Type of outbound payload submitted by the local companion-side producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Px4OffboardMessageKindV1 {
    /// One MAVLink setpoint message; allowlisting is supplied by the exact
    /// firmware/vehicle-specific profile rather than assumed universally here.
    MavlinkSetpoint {
        /// Raw MAVLink message ID of the serialized setpoint.
        message_id: u32,
    },
    /// One ROS 2 OffboardControlMode message.
    Ros2OffboardControlMode,
}

/// Result returned by the local publisher/transport submission boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4LocalSubmissionResultV1 {
    /// The local API accepted the submission; this does not prove wire delivery.
    AcceptedByLocalTransport,
    /// The local API rejected the submission.
    RejectedByLocalTransport,
}

/// Content-addressed evidence of one local Offboard proof-of-life submission attempt.
///
/// `submitted_at` is the companion-side timestamp associated with invoking the
/// transport/publisher boundary. It is not a PX4 receive timestamp. The payload
/// digest is supplied by the capturing adapter and is committed into this record;
/// this type does not have the payload bytes with which to independently recompute it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4OffboardSubmissionEvidenceV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// Identity of the local sender/adapter profile that emitted the event.
    pub producer_profile_id: String,
    /// Stable identity of the target instance bound by the upstream provider.
    pub target_instance_id: String,
    /// Sequence allocated for every submission attempt in this sender session.
    /// Assessments require a contiguous sequence to expose missing capture records.
    pub sequence_number: u64,
    /// Transport/message family and, for MAVLink, the concrete message ID.
    pub message_kind: Px4OffboardMessageKindV1,
    /// Lowercase BLAKE3 hex digest of the exact serialized payload, as declared by
    /// the capturing adapter. This record validates shape and binds the claim, but
    /// cannot recompute it because payload bytes are intentionally not retained.
    pub serialized_payload_digest_blake3_hex: String,
    /// What the local publishing API returned for this submission attempt.
    pub local_result: Px4LocalSubmissionResultV1,
    /// Companion-side submission timestamp in an explicit local clock domain.
    pub submitted_at: TimestampV1,
    /// Domain-separated commitment to this record.
    pub submission_digest_hex: String,
}

impl Px4OffboardSubmissionEvidenceV1 {
    /// Construct, bind, and validate one local submission event.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        sequence_number: u64,
        message_kind: Px4OffboardMessageKindV1,
        serialized_payload_digest_blake3_hex: impl Into<String>,
        local_result: Px4LocalSubmissionResultV1,
        submitted_at: TimestampV1,
    ) -> Result<Self, Px4OffboardSubmissionError> {
        let mut value = Self {
            schema_version: PX4_OFFBOARD_SUBMISSION_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            target_instance_id: target_instance_id.into(),
            sequence_number,
            message_kind,
            serialized_payload_digest_blake3_hex: serialized_payload_digest_blake3_hex.into(),
            local_result,
            submitted_at,
            submission_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.submission_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate required identities, digest shape, timestamp, and commitment.
    pub fn validate(&self) -> Result<(), Px4OffboardSubmissionError> {
        self.validate_without_digest()?;
        if self.submission_digest_hex != self.compute_digest_hex() {
            return Err(Px4OffboardSubmissionError::DigestMismatch("submission"));
        }
        Ok(())
    }

    /// Content-addressed identity of this submission record.
    pub fn submission_id(&self) -> Result<String, Px4OffboardSubmissionError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.offboard-submission.v1:{}",
            self.submission_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4OffboardSubmissionError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_hex_digest(&self.serialized_payload_digest_blake3_hex)?;
        self.submitted_at
            .validate()
            .map_err(Px4OffboardSubmissionError::Timestamp)?;
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SUBMISSION_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        hasher.update(&self.sequence_number.to_le_bytes());
        match self.message_kind {
            Px4OffboardMessageKindV1::MavlinkSetpoint { message_id } => {
                hasher.update(&[0]);
                hasher.update(&message_id.to_le_bytes());
            }
            Px4OffboardMessageKindV1::Ros2OffboardControlMode => {
                hasher.update(&[1]);
            }
        }
        feed_str(&mut hasher, &self.serialized_payload_digest_blake3_hex);
        hasher.update(&[match self.local_result {
            Px4LocalSubmissionResultV1::AcceptedByLocalTransport => 0,
            Px4LocalSubmissionResultV1::RejectedByLocalTransport => 1,
        }]);
        feed_str(&mut hasher, self.submitted_at.clock_domain.as_str());
        hasher.update(&self.submitted_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Versioned local submission-cadence policy for one bound PX4 instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4OffboardSubmissionProfileV1 {
    /// Profile schema version.
    pub schema_version: u16,
    /// Stable versioned identity for this exact submission-cadence policy.
    pub profile_id: String,
    /// Bound target instance identity; this profile does not authenticate that binding.
    pub target_instance_id: String,
    /// Exact controller firmware, vehicle class, and control-profile qualification ID.
    /// This is a declared binding, not authenticated firmware measurement.
    pub controller_profile_id: String,
    /// Local publisher/adapter identity expected to produce every record.
    pub producer_profile_id: String,
    /// The proof-of-life transport family this profile evaluates.
    pub transport: Px4OffboardTransportV1,
    /// Companion-side clock in which submission timestamps are comparable.
    pub submission_clock_domain: ClockDomainId,
    /// Number of consecutive captured submission attempts required.
    pub min_samples: u16,
    /// Minimum elapsed span from the first to last submission in this observed window.
    pub min_observation_window_ns: u64,
    /// Maximum permitted interval between adjacent submissions.
    pub max_inter_submission_gap_ns: u64,
    /// Maximum age of the newest local submission at assessment time.
    pub max_latest_submission_age_ns: u64,
    /// Explicit MAVLink setpoint message IDs accepted by this qualified profile.
    /// Must be non-empty only for MAVLink and is irrelevant to ROS 2.
    pub allowed_mavlink_setpoint_message_ids: Vec<u32>,
    /// Domain-separated commitment to all profile fields.
    pub profile_digest_hex: String,
}

impl Px4OffboardSubmissionProfileV1 {
    /// Construct a canonical, content-addressed submission-cadence profile.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        controller_profile_id: impl Into<String>,
        producer_profile_id: impl Into<String>,
        transport: Px4OffboardTransportV1,
        submission_clock_domain: ClockDomainId,
        min_samples: u16,
        min_observation_window_ns: u64,
        max_inter_submission_gap_ns: u64,
        max_latest_submission_age_ns: u64,
        mut allowed_mavlink_setpoint_message_ids: Vec<u32>,
    ) -> Result<Self, Px4OffboardSubmissionError> {
        allowed_mavlink_setpoint_message_ids.sort_unstable();
        let mut value = Self {
            schema_version: PX4_OFFBOARD_SUBMISSION_SCHEMA_V1,
            profile_id: profile_id.into(),
            target_instance_id: target_instance_id.into(),
            controller_profile_id: controller_profile_id.into(),
            producer_profile_id: producer_profile_id.into(),
            transport,
            submission_clock_domain,
            min_samples,
            min_observation_window_ns,
            max_inter_submission_gap_ns,
            max_latest_submission_age_ns,
            allowed_mavlink_setpoint_message_ids,
            profile_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.profile_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate identity, clocks, thresholds, allowlist canonicality, and commitment.
    pub fn validate(&self) -> Result<(), Px4OffboardSubmissionError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4OffboardSubmissionError::DigestMismatch("profile"));
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), Px4OffboardSubmissionError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.controller_profile_id, "controller_profile_id")?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        self.submission_clock_domain
            .validate()
            .map_err(Px4OffboardSubmissionError::Timestamp)?;
        if self.min_samples < 2 {
            return Err(Px4OffboardSubmissionError::MinimumSamplesTooSmall);
        }
        if self.min_observation_window_ns == 0 {
            return Err(Px4OffboardSubmissionError::ZeroObservationWindow);
        }
        if self.max_inter_submission_gap_ns == 0 {
            return Err(Px4OffboardSubmissionError::ZeroInterSubmissionGap);
        }
        if self.max_latest_submission_age_ns == 0 {
            return Err(Px4OffboardSubmissionError::ZeroLatestSubmissionAge);
        }
        let mut seen = HashSet::with_capacity(self.allowed_mavlink_setpoint_message_ids.len());
        let mut previous: Option<u32> = None;
        for id in &self.allowed_mavlink_setpoint_message_ids {
            if !seen.insert(*id) {
                return Err(Px4OffboardSubmissionError::DuplicateMavlinkMessageId);
            }
            if previous.is_some_and(|value| *id <= value) {
                return Err(Px4OffboardSubmissionError::NonCanonicalMavlinkAllowlist);
            }
            previous = Some(*id);
        }
        match self.transport {
            Px4OffboardTransportV1::MavlinkSetpoint
                if self.allowed_mavlink_setpoint_message_ids.is_empty() =>
            {
                return Err(Px4OffboardSubmissionError::EmptyMavlinkAllowlist);
            }
            Px4OffboardTransportV1::Ros2OffboardControlMode
                if !self.allowed_mavlink_setpoint_message_ids.is_empty() =>
            {
                return Err(Px4OffboardSubmissionError::MavlinkAllowlistForRos2);
            }
            _ => {}
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        feed_str(&mut hasher, &self.controller_profile_id);
        feed_str(&mut hasher, &self.producer_profile_id);
        hasher.update(&[match self.transport {
            Px4OffboardTransportV1::MavlinkSetpoint => 0,
            Px4OffboardTransportV1::Ros2OffboardControlMode => 1,
        }]);
        feed_str(&mut hasher, self.submission_clock_domain.as_str());
        hasher.update(&self.min_samples.to_le_bytes());
        hasher.update(&self.min_observation_window_ns.to_le_bytes());
        hasher.update(&self.max_inter_submission_gap_ns.to_le_bytes());
        hasher.update(&self.max_latest_submission_age_ns.to_le_bytes());
        hasher.update(&(self.allowed_mavlink_setpoint_message_ids.len() as u64).to_le_bytes());
        for id in &self.allowed_mavlink_setpoint_message_ids {
            hasher.update(&id.to_le_bytes());
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Failed condition in a sufficiently sampled local submission-cadence window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4OffboardSubmissionFailureV1 {
    /// At least one local publisher/transport attempt was rejected.
    LocalSubmissionRejected,
    /// The newest local submission is older than the configured age limit.
    LatestSubmissionStale,
    /// An adjacent local submission interval exceeds the profile limit.
    InterSubmissionGapExceeded,
    /// The observed stream span is shorter than the configured observation window.
    ObservationWindowTooShort,
}

/// Derived outcome for local submission cadence; it is not Offboard readiness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Px4OffboardSubmissionOutcomeV1 {
    /// Local submission cadence meets the supplied profile. This does not prove delivery.
    WithinProfile,
    /// At least the required sample count exists, but one or more cadence checks fail.
    OutsideProfile {
        /// Deterministically ordered failed conditions.
        failures: Vec<Px4OffboardSubmissionFailureV1>,
    },
    /// Fewer than the configured number of submissions are available.
    Incomplete {
        /// Number of submission records retained.
        available_samples: usize,
        /// Number required by the profile.
        required_samples: u16,
    },
}

/// Content-addressed assessment of one local outbound submission window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4OffboardSubmissionCadenceV1 {
    /// Assessment schema version.
    pub schema_version: u16,
    /// Exact submission-cadence profile.
    pub profile: Px4OffboardSubmissionProfileV1,
    /// Companion-side instant when the submission window was assessed.
    pub assessed_at: TimestampV1,
    /// Submission attempts in canonical chronological order.
    pub samples: Vec<Px4OffboardSubmissionEvidenceV1>,
    /// Domain-separated commitment to the profile, time, and ordered records.
    pub assessment_digest_hex: String,
}

impl Px4OffboardSubmissionCadenceV1 {
    /// Construct, canonicalize, content-bind, and validate a submission assessment.
    pub fn new(
        profile: Px4OffboardSubmissionProfileV1,
        assessed_at: TimestampV1,
        mut samples: Vec<Px4OffboardSubmissionEvidenceV1>,
    ) -> Result<Self, Px4OffboardSubmissionError> {
        samples.sort_by_key(|sample| sample.submitted_at.nanoseconds);
        let mut value = Self {
            schema_version: PX4_OFFBOARD_SUBMISSION_SCHEMA_V1,
            profile,
            assessed_at,
            samples,
            assessment_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.assessment_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate all nested commitments, exact profile binding, chronology, and identity.
    pub fn validate(&self) -> Result<(), Px4OffboardSubmissionError> {
        self.validate_without_digest()?;
        if self.assessment_digest_hex != self.compute_digest_hex() {
            return Err(Px4OffboardSubmissionError::DigestMismatch("assessment"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact submission-cadence assessment.
    pub fn assessment_id(&self) -> Result<String, Px4OffboardSubmissionError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.offboard-submission-assessment.v1:{}",
            self.assessment_digest_hex
        ))
    }

    /// Derive local cadence status under the exact profile.
    ///
    /// A positive outcome proves only that the retained local submission events
    /// fit these thresholds. It does not prove PX4 received them or entered Offboard.
    pub fn outcome(
        &self,
    ) -> Result<Px4OffboardSubmissionOutcomeV1, Px4OffboardSubmissionError> {
        self.validate()?;
        // Preserve an observed local rejection even when the history is too short
        // to assess cadence; absence of enough samples must not erase known failure.
        if self.samples.iter().any(|sample| {
            sample.local_result == Px4LocalSubmissionResultV1::RejectedByLocalTransport
        }) {
            return Ok(Px4OffboardSubmissionOutcomeV1::OutsideProfile {
                failures: vec![Px4OffboardSubmissionFailureV1::LocalSubmissionRejected],
            });
        }
        if self.samples.len() < usize::from(self.profile.min_samples) {
            return Ok(Px4OffboardSubmissionOutcomeV1::Incomplete {
                available_samples: self.samples.len(),
                required_samples: self.profile.min_samples,
            });
        }

        let first = self.samples.first().ok_or(Px4OffboardSubmissionError::EmptySampleWindow)?;
        let latest = self.samples.last().ok_or(Px4OffboardSubmissionError::EmptySampleWindow)?;
        let mut failures = Vec::new();
        let latest_age_ns = self
            .assessed_at
            .elapsed_since(&latest.submitted_at)
            .map_err(Px4OffboardSubmissionError::Timestamp)?;
        if latest_age_ns > self.profile.max_latest_submission_age_ns {
            failures.push(Px4OffboardSubmissionFailureV1::LatestSubmissionStale);
        }
        let max_gap = self
            .samples
            .windows(2)
            .map(|pair| pair[1].submitted_at.nanoseconds - pair[0].submitted_at.nanoseconds)
            .max()
            .unwrap_or(0);
        if max_gap > self.profile.max_inter_submission_gap_ns {
            failures.push(Px4OffboardSubmissionFailureV1::InterSubmissionGapExceeded);
        }
        let observed_window_ns = latest.submitted_at.nanoseconds - first.submitted_at.nanoseconds;
        if observed_window_ns < self.profile.min_observation_window_ns {
            failures.push(Px4OffboardSubmissionFailureV1::ObservationWindowTooShort);
        }

        if failures.is_empty() {
            Ok(Px4OffboardSubmissionOutcomeV1::WithinProfile)
        } else {
            Ok(Px4OffboardSubmissionOutcomeV1::OutsideProfile { failures })
        }
    }

    fn validate_without_digest(&self) -> Result<(), Px4OffboardSubmissionError> {
        validate_schema(self.schema_version)?;
        self.profile.validate()?;
        self.assessed_at
            .validate()
            .map_err(Px4OffboardSubmissionError::Timestamp)?;
        if self.assessed_at.clock_domain != self.profile.submission_clock_domain {
            return Err(Px4OffboardSubmissionError::AssessmentClockMismatch);
        }

        let mut seen_digest = HashSet::with_capacity(self.samples.len());
        let mut seen_sequence = HashSet::with_capacity(self.samples.len());
        let mut previous_timestamp: Option<u64> = None;
        let mut previous_sequence: Option<u64> = None;
        for sample in &self.samples {
            sample.validate()?;
            if sample.producer_profile_id != self.profile.producer_profile_id {
                return Err(Px4OffboardSubmissionError::ProducerMismatch);
            }
            if sample.target_instance_id != self.profile.target_instance_id {
                return Err(Px4OffboardSubmissionError::TargetMismatch);
            }
            if sample.submitted_at.clock_domain != self.profile.submission_clock_domain {
                return Err(Px4OffboardSubmissionError::SubmissionClockMismatch);
            }
            match (self.profile.transport, sample.message_kind) {
                (
                    Px4OffboardTransportV1::MavlinkSetpoint,
                    Px4OffboardMessageKindV1::MavlinkSetpoint { message_id },
                ) if self.profile.allowed_mavlink_setpoint_message_ids.contains(&message_id) => {}
                (
                    Px4OffboardTransportV1::Ros2OffboardControlMode,
                    Px4OffboardMessageKindV1::Ros2OffboardControlMode,
                ) => {}
                (
                    Px4OffboardTransportV1::MavlinkSetpoint,
                    Px4OffboardMessageKindV1::MavlinkSetpoint { message_id },
                ) => {
                    return Err(Px4OffboardSubmissionError::UnapprovedMavlinkMessageId(message_id));
                }
                _ => return Err(Px4OffboardSubmissionError::MessageKindTransportMismatch),
            }
            if !seen_digest.insert(sample.submission_digest_hex.clone()) {
                return Err(Px4OffboardSubmissionError::DuplicateSubmission);
            }
            if !seen_sequence.insert(sample.sequence_number) {
                return Err(Px4OffboardSubmissionError::DuplicateSequenceNumber);
            }
            if previous_timestamp
                .is_some_and(|previous| sample.submitted_at.nanoseconds <= previous)
            {
                return Err(Px4OffboardSubmissionError::NonAdvancingSubmissionTime);
            }
            if let Some(previous) = previous_sequence {
                let expected = previous
                    .checked_add(1)
                    .ok_or(Px4OffboardSubmissionError::SequenceOverflow)?;
                if sample.sequence_number != expected {
                    return Err(Px4OffboardSubmissionError::SubmissionSequenceGap {
                        previous,
                        found: sample.sequence_number,
                    });
                }
            }
            previous_timestamp = Some(sample.submitted_at.nanoseconds);
            previous_sequence = Some(sample.sequence_number);
        }

        if let Some(latest) = self.samples.last() {
            self.assessed_at
                .elapsed_since(&latest.submitted_at)
                .map_err(|error| match error {
                    EvidenceValidationError::NonMonotonicTimestamp => {
                        Px4OffboardSubmissionError::AssessmentBeforeLatestSubmission
                    }
                    other => Px4OffboardSubmissionError::Timestamp(other),
                })?;
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSESSMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile.profile_digest_hex);
        feed_str(&mut hasher, self.assessed_at.clock_domain.as_str());
        hasher.update(&self.assessed_at.nanoseconds.to_le_bytes());
        hasher.update(&(self.samples.len() as u64).to_le_bytes());
        for sample in &self.samples {
            feed_str(&mut hasher, &sample.submission_digest_hex);
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_schema(schema_version: u16) -> Result<(), Px4OffboardSubmissionError> {
    if schema_version != PX4_OFFBOARD_SUBMISSION_SCHEMA_V1 {
        return Err(Px4OffboardSubmissionError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), Px4OffboardSubmissionError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4OffboardSubmissionError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_hex_digest(value: &str) -> Result<(), Px4OffboardSubmissionError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Px4OffboardSubmissionError::InvalidPayloadDigest);
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

/// Validation or derivation failure for local Offboard submission evidence.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4OffboardSubmissionError {
    /// Unsupported evidence schema version.
    #[error("unsupported Offboard submission schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Identifier is blank, padded, too long, or contains control characters.
    #[error("invalid {0}")]
    InvalidIdentifier(&'static str),
    /// Payload BLAKE3 digest is not 64 lowercase hexadecimal characters.
    #[error("invalid serialized payload BLAKE3 digest")]
    InvalidPayloadDigest,
    /// Profile requires a concrete setpoint allowlist for MAVLink.
    #[error("MAVLink Offboard profile requires an explicit setpoint message allowlist")]
    EmptyMavlinkAllowlist,
    /// ROS 2 profile must not carry a MAVLink message allowlist.
    #[error("ROS 2 Offboard profile must not carry a MAVLink allowlist")]
    MavlinkAllowlistForRos2,
    /// The supplied MAVLink allowlist is not canonical and strictly increasing.
    #[error("MAVLink allowlist must be sorted and contain unique IDs")]
    NonCanonicalMavlinkAllowlist,
    /// The same raw MAVLink message ID appeared more than once in the profile.
    #[error("duplicate MAVLink message ID in profile")]
    DuplicateMavlinkMessageId,
    /// Profile requires at least two observations.
    #[error("Offboard submission min_samples must be at least 2")]
    MinimumSamplesTooSmall,
    /// Observation window must be non-zero and selected by a qualified profile.
    #[error("Offboard observation window must be non-zero")]
    ZeroObservationWindow,
    /// A zero interval cannot qualify a recurring stream.
    #[error("Offboard maximum inter-submission gap must be non-zero")]
    ZeroInterSubmissionGap,
    /// A zero age limit cannot be represented by a practical local polling loop.
    #[error("Offboard maximum submission age must be non-zero")]
    ZeroLatestSubmissionAge,
    /// Submission records from different producer profiles were mixed.
    #[error("Offboard submission producer does not match the profile")]
    ProducerMismatch,
    /// Submission evidence target differs from the profile's bound target identity.
    #[error("Offboard submission target does not match the profile")]
    TargetMismatch,
    /// Submission event's timestamp clock differs from the selected profile.
    #[error("Offboard submission timestamp clock does not match the profile")]
    SubmissionClockMismatch,
    /// Assessment time's clock differs from the selected profile.
    #[error("Offboard submission assessment clock does not match the profile")]
    AssessmentClockMismatch,
    /// Observed message kind is not compatible with the transport profile.
    #[error("Offboard message kind does not match the profile's transport")]
    MessageKindTransportMismatch,
    /// MAVLink message ID is not in the exact profile allowlist.
    #[error("MAVLink setpoint message ID {0} is not allowlisted by the profile")]
    UnapprovedMavlinkMessageId(u32),
    /// The same content-addressed submission was included more than once.
    #[error("duplicate Offboard submission record")]
    DuplicateSubmission,
    /// Two records reused one sequence value.
    #[error("duplicate Offboard submission sequence number")]
    DuplicateSequenceNumber,
    /// Submission timestamps do not advance strictly.
    #[error("Offboard submission timestamp does not advance strictly")]
    NonAdvancingSubmissionTime,
    /// Sequence counter advanced by more than one; at least one captured attempt is absent.
    #[error("Offboard submission sequence gap after {previous}: next captured sequence is {found}")]
    SubmissionSequenceGap {
        /// Last captured sequence number.
        previous: u64,
        /// Next captured sequence number.
        found: u64,
    },
    /// The sequence counter cannot advance past u64::MAX.
    #[error("Offboard submission sequence overflow")]
    SequenceOverflow,
    /// Assessment instant precedes the latest local submission.
    #[error("Offboard assessment precedes the latest submission")]
    AssessmentBeforeLatestSubmission,
    /// The evidence window is empty.
    #[error("Offboard submission window is empty")]
    EmptySampleWindow,
    /// Timestamp or clock evidence is invalid.
    #[error("invalid Offboard submission timestamp: {0}")]
    Timestamp(EvidenceValidationError),
    /// Stored content no longer matches its commitment.
    #[error("Offboard submission {0} content commitment mismatch")]
    DigestMismatch(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> ClockDomainId {
        ClockDomainId::new("companion.monotonic").unwrap()
    }

    fn ts(ns: u64) -> TimestampV1 {
        TimestampV1::new(clock(), ns)
    }

    fn profile() -> Px4OffboardSubmissionProfileV1 {
        Px4OffboardSubmissionProfileV1::new(
            "px4.offboard.publisher.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-control.fixture.v1",
            "companion.publisher.fixture.v1",
            Px4OffboardTransportV1::MavlinkSetpoint,
            clock(),
            3,
            1_000_000_000,
            500_000_000,
            250_000_000,
            vec![84, 82, 86],
        )
        .unwrap()
    }

    fn event(
        sequence: u64,
        ns: u64,
        result: Px4LocalSubmissionResultV1,
    ) -> Px4OffboardSubmissionEvidenceV1 {
        Px4OffboardSubmissionEvidenceV1::new(
            "companion.publisher.fixture.v1",
            "vehicle.slot-a",
            sequence,
            Px4OffboardMessageKindV1::MavlinkSetpoint { message_id: 84 },
            format!("{:064x}", sequence + 1),
            result,
            ts(ns),
        )
        .unwrap()
    }

    fn assessment(
        samples: Vec<Px4OffboardSubmissionEvidenceV1>,
        assessed_at_ns: u64,
    ) -> Px4OffboardSubmissionCadenceV1 {
        Px4OffboardSubmissionCadenceV1::new(profile(), ts(assessed_at_ns), samples).unwrap()
    }

    #[test]
    fn regular_local_submissions_meet_only_the_declared_cadence_profile() {
        let value = assessment(
            vec![
                event(10, 0, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(11, 500_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(12, 1_000_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            ],
            1_100_000_000,
        );
        assert_eq!(value.outcome().unwrap(), Px4OffboardSubmissionOutcomeV1::WithinProfile);
        assert!(value.assessment_id().unwrap().starts_with(
            "symthaea.autopilot.px4.offboard-submission-assessment.v1:"
        ));
    }

    #[test]
    fn timing_boundaries_are_inclusive_and_one_nanosecond_over_fails() {
        let boundary = assessment(
            vec![
                event(10, 0, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(11, 500_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(12, 1_000_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            ],
            1_250_000_000,
        );
        assert_eq!(boundary.outcome().unwrap(), Px4OffboardSubmissionOutcomeV1::WithinProfile);

        let stale = assessment(
            vec![
                event(10, 0, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(11, 500_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(12, 1_000_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            ],
            1_250_000_001,
        );
        assert!(matches!(
            stale.outcome().unwrap(),
            Px4OffboardSubmissionOutcomeV1::OutsideProfile { failures }
                if failures.contains(&Px4OffboardSubmissionFailureV1::LatestSubmissionStale)
        ));
    }

    #[test]
    fn large_gap_and_rejected_local_submission_never_qualify() {
        let gapped = assessment(
            vec![
                event(10, 0, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(11, 500_000_001, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(12, 1_000_000_002, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            ],
            1_100_000_000,
        );
        assert!(matches!(
            gapped.outcome().unwrap(),
            Px4OffboardSubmissionOutcomeV1::OutsideProfile { failures }
                if failures.contains(&Px4OffboardSubmissionFailureV1::InterSubmissionGapExceeded)
        ));

        let rejected = assessment(
            vec![
                event(10, 0, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(11, 500_000_000, Px4LocalSubmissionResultV1::RejectedByLocalTransport),
                event(12, 1_000_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            ],
            1_100_000_000,
        );
        assert!(matches!(
            rejected.outcome().unwrap(),
            Px4OffboardSubmissionOutcomeV1::OutsideProfile { failures }
                if failures.contains(&Px4OffboardSubmissionFailureV1::LocalSubmissionRejected)
        ));
    }

    #[test]
    fn insufficient_history_is_incomplete_not_a_positive_or_negative_verdict() {
        let value = assessment(
            vec![
                event(10, 0, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(11, 500_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            ],
            600_000_000,
        );
        assert_eq!(
            value.outcome().unwrap(),
            Px4OffboardSubmissionOutcomeV1::Incomplete {
                available_samples: 2,
                required_samples: 3,
            }
        );

        let known_rejection = assessment(
            vec![event(10, 0, Px4LocalSubmissionResultV1::RejectedByLocalTransport)],
            100,
        );
        assert_eq!(
            known_rejection.outcome().unwrap(),
            Px4OffboardSubmissionOutcomeV1::OutsideProfile {
                failures: vec![Px4OffboardSubmissionFailureV1::LocalSubmissionRejected],
            }
        );
    }

    #[test]
    fn observed_window_too_short_is_not_qualified() {
        let value = assessment(
            vec![
                event(10, 0, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(11, 100_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(12, 200_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            ],
            300_000_000,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4OffboardSubmissionOutcomeV1::OutsideProfile { failures }
                if failures.contains(&Px4OffboardSubmissionFailureV1::ObservationWindowTooShort)
        ));
    }

    #[test]
    fn sequence_gaps_are_not_hidden_by_canonical_timestamp_sorting() {
        let samples = vec![
            event(10, 0, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            event(12, 500_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            event(13, 1_000_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
        ];
        assert_eq!(
            Px4OffboardSubmissionCadenceV1::new(profile(), ts(1_100_000_000), samples)
                .unwrap_err(),
            Px4OffboardSubmissionError::SubmissionSequenceGap {
                previous: 10,
                found: 12,
            }
        );
    }

    #[test]
    fn wrong_target_clock_transport_and_unapproved_setpoint_fail_closed() {
        let wrong_target = Px4OffboardSubmissionEvidenceV1::new(
            "companion.publisher.fixture.v1",
            "vehicle.other",
            10,
            Px4OffboardMessageKindV1::MavlinkSetpoint { message_id: 84 },
            format!("{:064x}", 11),
            Px4LocalSubmissionResultV1::AcceptedByLocalTransport,
            ts(0),
        )
        .unwrap();
        assert_eq!(
            Px4OffboardSubmissionCadenceV1::new(
                profile(),
                ts(1),
                vec![wrong_target],
            )
            .unwrap_err(),
            Px4OffboardSubmissionError::TargetMismatch
        );

        let wrong_kind = Px4OffboardSubmissionEvidenceV1::new(
            "companion.publisher.fixture.v1",
            "vehicle.slot-a",
            10,
            Px4OffboardMessageKindV1::MavlinkSetpoint { message_id: 999 },
            format!("{:064x}", 11),
            Px4LocalSubmissionResultV1::AcceptedByLocalTransport,
            ts(0),
        )
        .unwrap();
        assert_eq!(
            Px4OffboardSubmissionCadenceV1::new(profile(), ts(1), vec![wrong_kind]).unwrap_err(),
            Px4OffboardSubmissionError::UnapprovedMavlinkMessageId(999)
        );

        let wrong_transport = Px4OffboardSubmissionEvidenceV1::new(
            "companion.publisher.fixture.v1",
            "vehicle.slot-a",
            10,
            Px4OffboardMessageKindV1::Ros2OffboardControlMode,
            format!("{:064x}", 11),
            Px4LocalSubmissionResultV1::AcceptedByLocalTransport,
            ts(0),
        )
        .unwrap();
        assert_eq!(
            Px4OffboardSubmissionCadenceV1::new(profile(), ts(1), vec![wrong_transport])
                .unwrap_err(),
            Px4OffboardSubmissionError::MessageKindTransportMismatch
        );

        let wrong_clock = Px4OffboardSubmissionEvidenceV1::new(
            "companion.publisher.fixture.v1",
            "vehicle.slot-a",
            10,
            Px4OffboardMessageKindV1::MavlinkSetpoint { message_id: 84 },
            format!("{:064x}", 11),
            Px4LocalSubmissionResultV1::AcceptedByLocalTransport,
            TimestampV1::new(ClockDomainId::new("sim.time").unwrap(), 0),
        )
        .unwrap();
        assert_eq!(
            Px4OffboardSubmissionCadenceV1::new(profile(), ts(1), vec![wrong_clock]).unwrap_err(),
            Px4OffboardSubmissionError::SubmissionClockMismatch
        );
    }

    #[test]
    fn ros2_offboard_control_mode_is_a_distinct_stream_kind() {
        let ros_profile = Px4OffboardSubmissionProfileV1::new(
            "px4.offboard.ros2.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-control.ros2-fixture.v1",
            "companion.ros2.fixture.v1",
            Px4OffboardTransportV1::Ros2OffboardControlMode,
            clock(),
            2,
            1,
            10,
            10,
            vec![],
        )
        .unwrap();
        let first = Px4OffboardSubmissionEvidenceV1::new(
            "companion.ros2.fixture.v1",
            "vehicle.slot-a",
            4,
            Px4OffboardMessageKindV1::Ros2OffboardControlMode,
            format!("{:064x}", 5),
            Px4LocalSubmissionResultV1::AcceptedByLocalTransport,
            ts(0),
        )
        .unwrap();
        let second = Px4OffboardSubmissionEvidenceV1::new(
            "companion.ros2.fixture.v1",
            "vehicle.slot-a",
            5,
            Px4OffboardMessageKindV1::Ros2OffboardControlMode,
            format!("{:064x}", 6),
            Px4LocalSubmissionResultV1::AcceptedByLocalTransport,
            ts(1),
        )
        .unwrap();
        let value = Px4OffboardSubmissionCadenceV1::new(
            ros_profile,
            ts(2),
            vec![first, second],
        )
        .unwrap();
        assert_eq!(value.outcome().unwrap(), Px4OffboardSubmissionOutcomeV1::WithinProfile);
    }

    #[test]
    fn profile_allowlist_is_canonical_and_bound_into_identity() {
        let unsorted = Px4OffboardSubmissionProfileV1::new(
            "px4.offboard.publisher.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-control.fixture.v1",
            "companion.publisher.fixture.v1",
            Px4OffboardTransportV1::MavlinkSetpoint,
            clock(),
            3,
            1_000_000_000,
            500_000_000,
            250_000_000,
            vec![86, 82, 84],
        )
        .unwrap();
        assert_eq!(unsorted.allowed_mavlink_setpoint_message_ids, vec![82, 84, 86]);
        let other_policy = Px4OffboardSubmissionProfileV1::new(
            "px4.offboard.publisher.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-control.fixture.v1",
            "companion.publisher.fixture.v1",
            Px4OffboardTransportV1::MavlinkSetpoint,
            clock(),
            3,
            1_000_000_000,
            500_000_000,
            250_000_000,
            vec![82, 84],
        )
        .unwrap();
        assert_ne!(unsorted.profile_digest_hex, other_policy.profile_digest_hex);
    }

    #[test]
    fn controller_profile_id_is_bound_into_profile_identity() {
        let original = profile();
        let revised_controller_profile = Px4OffboardSubmissionProfileV1::new(
            "px4.offboard.publisher.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-control.revision-2.fixture.v1",
            "companion.publisher.fixture.v1",
            Px4OffboardTransportV1::MavlinkSetpoint,
            clock(),
            3,
            1_000_000_000,
            500_000_000,
            250_000_000,
            vec![82, 84, 86],
        )
        .unwrap();
        assert_ne!(
            original.profile_digest_hex,
            revised_controller_profile.profile_digest_hex
        );
    }

    #[test]
    fn mutation_and_serde_round_trip_preserve_validation_boundary() {
        let value = assessment(
            vec![
                event(10, 0, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(11, 500_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
                event(12, 1_000_000_000, Px4LocalSubmissionResultV1::AcceptedByLocalTransport),
            ],
            1_100_000_000,
        );
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut restored: Px4OffboardSubmissionCadenceV1 = serde_json::from_slice(&bytes).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.assessment_id().unwrap(), value.assessment_id().unwrap());
        restored.samples[0].sequence_number += 1;
        assert!(restored.validate().is_err());
    }

    #[test]
    fn profile_thresholds_and_payload_digest_shape_fail_closed() {
        assert_eq!(
            Px4OffboardSubmissionProfileV1::new(
                "bad.profile",
                "vehicle.slot-a",
                "px4.firmware-vehicle-control.fixture.v1",
                "sender",
                Px4OffboardTransportV1::MavlinkSetpoint,
                clock(),
                1,
                1,
                1,
                1,
                vec![84],
            )
            .unwrap_err(),
            Px4OffboardSubmissionError::MinimumSamplesTooSmall
        );
        assert_eq!(
            Px4OffboardSubmissionEvidenceV1::new(
                "sender",
                "vehicle.slot-a",
                0,
                Px4OffboardMessageKindV1::MavlinkSetpoint { message_id: 84 },
                "not-a-digest",
                Px4LocalSubmissionResultV1::AcceptedByLocalTransport,
                ts(0),
            )
            .unwrap_err(),
            Px4OffboardSubmissionError::InvalidPayloadDigest
        );
    }
}
