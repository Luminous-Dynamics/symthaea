// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Content-addressed observations of PX4-reported mode and failsafe state.
//!
//! This module models captured PX4 VehicleStatus/VehicleStatusV3-like records.
//! PX4 boot-relative source timestamps are preserved separately from local
//! receiver-clock timestamps. The evidence can describe what a telemetry producer
//! reported; it does not authenticate that producer or establish physical truth,
//! mode-entry authority, readiness, a correct failsafe response, or safe flight.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{
    ClockDomainId, EvidenceValidationError, TimestampV1,
};
use thiserror::Error;

/// Schema version for PX4 mode/failsafe observations and assessments.
pub const PX4_MODE_FAILSAFE_OBSERVATION_SCHEMA_V1: u16 = 1;
const OBSERVATION_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.mode-failsafe-observation.v1\0";
const PROFILE_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.mode-failsafe-profile.v1\0";
const ASSESSMENT_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.mode-failsafe-assessment.v1\0";

/// Content-addressed capture of one decoded PX4 vehicle-status publication.
///
/// Numeric mode/state values remain raw and are interpreted only against the
/// explicitly bound controller profile. The source timestamp is microseconds in
/// the named PX4 boot session; `received_at` is a separate receiver-local clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ModeFailsafeObservationV1 {
    /// Evidence schema version.
    pub schema_version: u16,
    /// Stable producer/decoder profile identity.
    pub producer_profile_id: String,
    /// Upstream-bound vehicle/autopilot instance identity; not authenticated here.
    pub target_instance_id: String,
    /// Declared controller firmware, vehicle class, and message-profile identity.
    pub controller_profile_id: String,
    /// PX4 boot/session identity; evidence from different sessions cannot be mixed.
    pub source_boot_session_id: String,
    /// Adapter-capture sequence, allocated once for each captured status publication.
    pub capture_sequence: u64,
    /// Raw PX4 source timestamp in microseconds since this boot session started.
    pub source_timestamp_us: u64,
    /// Raw PX4 timestamp when the current navigation state was activated, in microseconds.
    pub nav_state_timestamp_us: u64,
    /// Raw PX4 VehicleStatus `nav_state` value.
    pub nav_state_raw: u8,
    /// Raw PX4 VehicleStatus `nav_state_user_intention` value.
    pub nav_state_user_intention_raw: u8,
    /// PX4-reported capability of the current mode to accept MAVLink offboard setpoints.
    pub accepts_offboard_setpoints: bool,
    /// PX4-reported system failsafe state.
    pub failsafe_active: bool,
    /// PX4-reported state indicating failsafe with user takeover.
    pub failsafe_and_user_took_over: bool,
    /// Raw PX4 deferred-failsafe state; interpreted only by the bound controller profile.
    pub failsafe_defer_state_raw: u8,
    /// Timestamp when the adapter accepted this observation, in an explicit local clock.
    pub received_at: TimestampV1,
    /// Domain-separated content commitment.
    pub observation_digest_hex: String,
}

impl Px4ModeFailsafeObservationV1 {
    /// Construct, bind, and validate one captured PX4 status observation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        controller_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        capture_sequence: u64,
        source_timestamp_us: u64,
        nav_state_timestamp_us: u64,
        nav_state_raw: u8,
        nav_state_user_intention_raw: u8,
        accepts_offboard_setpoints: bool,
        failsafe_active: bool,
        failsafe_and_user_took_over: bool,
        failsafe_defer_state_raw: u8,
        received_at: TimestampV1,
    ) -> Result<Self, Px4ModeFailsafeObservationError> {
        let mut value = Self {
            schema_version: PX4_MODE_FAILSAFE_OBSERVATION_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            target_instance_id: target_instance_id.into(),
            controller_profile_id: controller_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            capture_sequence,
            source_timestamp_us,
            nav_state_timestamp_us,
            nav_state_raw,
            nav_state_user_intention_raw,
            accepts_offboard_setpoints,
            failsafe_active,
            failsafe_and_user_took_over,
            failsafe_defer_state_raw,
            received_at,
            observation_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.observation_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate source/session identity, timestamps, and content commitment.
    pub fn validate(&self) -> Result<(), Px4ModeFailsafeObservationError> {
        self.validate_without_digest()?;
        if self.observation_digest_hex != self.compute_digest_hex() {
            return Err(Px4ModeFailsafeObservationError::DigestMismatch("observation"));
        }
        Ok(())
    }

    /// Content-addressed identity of this observation.
    pub fn observation_id(&self) -> Result<String, Px4ModeFailsafeObservationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.mode-failsafe-observation.v1:{}",
            self.observation_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4ModeFailsafeObservationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.controller_profile_id, "controller_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        self.received_at
            .validate()
            .map_err(Px4ModeFailsafeObservationError::Timestamp)?;
        if self.nav_state_timestamp_us > self.source_timestamp_us {
            return Err(Px4ModeFailsafeObservationError::NavStateTimestampAfterStatus);
        }
        if self.failsafe_and_user_took_over && !self.failsafe_active {
            return Err(Px4ModeFailsafeObservationError::TakeoverWithoutFailsafe);
        }
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(OBSERVATION_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.target_instance_id);
        feed_str(&mut hasher, &self.controller_profile_id);
        feed_str(&mut hasher, &self.source_boot_session_id);
        hasher.update(&self.capture_sequence.to_le_bytes());
        hasher.update(&self.source_timestamp_us.to_le_bytes());
        hasher.update(&self.nav_state_timestamp_us.to_le_bytes());
        hasher.update(&[
            self.nav_state_raw,
            self.nav_state_user_intention_raw,
            u8::from(self.accepts_offboard_setpoints),
            u8::from(self.failsafe_active),
            u8::from(self.failsafe_and_user_took_over),
            self.failsafe_defer_state_raw,
        ]);
        feed_str(&mut hasher, self.received_at.clock_domain.as_str());
        hasher.update(&self.received_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Exact profile used to interpret and freshness-qualify captured PX4 status.
///
/// The Offboard state numeric value is intentionally supplied by this profile:
/// raw navigation-state enums differ across firmware/schema versions and are not
/// inferred from the companion's requested mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ModeFailsafeObservationProfileV1 {
    /// Profile schema version.
    pub schema_version: u16,
    /// Stable versioned profile identity.
    pub profile_id: String,
    /// Exact target identity expected on every source record.
    pub target_instance_id: String,
    /// Exact controller firmware, vehicle, and status-schema profile.
    pub controller_profile_id: String,
    /// Exact decoder/telemetry producer.
    pub producer_profile_id: String,
    /// PX4 boot/session identity whose raw source timestamps are comparable.
    pub source_boot_session_id: String,
    /// Receiver-local clock domain for freshness calculations.
    pub receiver_clock_domain: ClockDomainId,
    /// Firmware/profile-specific raw navigation-state value for Offboard.
    pub offboard_nav_state_raw: u8,
    /// Minimum repeated status observations required for trend/transition inspection.
    pub min_samples: u16,
    /// Maximum permitted age of the newest receiver-side observation.
    pub max_latest_observation_age_ns: u64,
    /// Domain-separated commitment to the profile.
    pub profile_digest_hex: String,
}

impl Px4ModeFailsafeObservationProfileV1 {
    /// Construct, bind, and validate one mode/failsafe observation profile.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_id: impl Into<String>,
        target_instance_id: impl Into<String>,
        controller_profile_id: impl Into<String>,
        producer_profile_id: impl Into<String>,
        source_boot_session_id: impl Into<String>,
        receiver_clock_domain: ClockDomainId,
        offboard_nav_state_raw: u8,
        min_samples: u16,
        max_latest_observation_age_ns: u64,
    ) -> Result<Self, Px4ModeFailsafeObservationError> {
        let mut value = Self {
            schema_version: PX4_MODE_FAILSAFE_OBSERVATION_SCHEMA_V1,
            profile_id: profile_id.into(),
            target_instance_id: target_instance_id.into(),
            controller_profile_id: controller_profile_id.into(),
            producer_profile_id: producer_profile_id.into(),
            source_boot_session_id: source_boot_session_id.into(),
            receiver_clock_domain,
            offboard_nav_state_raw,
            min_samples,
            max_latest_observation_age_ns,
            profile_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.profile_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate profile identity, clocks, sample floor, and commitment.
    pub fn validate(&self) -> Result<(), Px4ModeFailsafeObservationError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4ModeFailsafeObservationError::DigestMismatch("profile"));
        }
        Ok(())
    }

    fn validate_without_digest(&self) -> Result<(), Px4ModeFailsafeObservationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        validate_identifier(&self.target_instance_id, "target_instance_id")?;
        validate_identifier(&self.controller_profile_id, "controller_profile_id")?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        validate_identifier(&self.source_boot_session_id, "source_boot_session_id")?;
        self.receiver_clock_domain
            .validate()
            .map_err(Px4ModeFailsafeObservationError::Timestamp)?;
        if self.min_samples < 2 {
            return Err(Px4ModeFailsafeObservationError::MinimumSamplesTooSmall);
        }
        if self.max_latest_observation_age_ns == 0 {
            return Err(Px4ModeFailsafeObservationError::ZeroLatestAge);
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
        feed_str(&mut hasher, &self.source_boot_session_id);
        feed_str(&mut hasher, self.receiver_clock_domain.as_str());
        hasher.update(&[self.offboard_nav_state_raw]);
        hasher.update(&self.min_samples.to_le_bytes());
        hasher.update(&self.max_latest_observation_age_ns.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Observed properties of the newest retained PX4 status record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ObservedModeFailsafeStateV1 {
    /// Raw current navigation state reported by PX4.
    pub nav_state_raw: u8,
    /// Whether the raw mode matches the profile's declared Offboard mode value.
    pub reports_offboard_mode: bool,
    /// PX4's own statement that the current mode accepts MAVLink Offboard setpoints.
    pub accepts_offboard_setpoints: bool,
    /// PX4-reported system failsafe indicator.
    pub failsafe_active: bool,
    /// PX4-reported failsafe/user-takeover indicator.
    pub failsafe_and_user_took_over: bool,
    /// Raw current mode's user-intention value.
    pub nav_state_user_intention_raw: u8,
    /// Raw deferred-failsafe state.
    pub failsafe_defer_state_raw: u8,
    /// Previous observed raw navigation state, when repeated evidence exists.
    pub previous_nav_state_raw: Option<u8>,
    /// Whether the raw navigation state changed across the last two observations.
    pub nav_state_transition_observed: bool,
    /// Previous observed failsafe indicator, when repeated evidence exists.
    pub previous_failsafe_active: Option<bool>,
    /// Whether the reported failsafe indicator changed across the last two observations.
    pub failsafe_transition_observed: bool,
    /// Receiver-clock age of the latest status observation.
    pub latest_observation_age_ns: u64,
}

/// Derived inspection result. It does not lower to PX4 readiness or authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Px4ModeFailsafeObservationOutcomeV1 {
    /// Not enough repeated source observations to inspect transitions.
    Incomplete {
        /// Number of source records retained.
        available_samples: usize,
        /// Number required by the bound profile.
        required_samples: u16,
    },
    /// Latest source record is stale under the exact receiver-clock profile.
    Stale {
        /// Measured age of the latest status record.
        age_ns: u64,
        /// Bound profile age ceiling.
        max_age_ns: u64,
    },
    /// Latest mode/failsafe fields and any observed state changes are preserved.
    FreshObservation {
        /// Current PX4-reported mode/failsafe snapshot.
        state: Px4ObservedModeFailsafeStateV1,
    },
}

/// Content-addressed assessment of repeated PX4 mode/failsafe status publications.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ModeFailsafeObservationAssessmentV1 {
    /// Assessment schema version.
    pub schema_version: u16,
    /// Exact mode/failsafe interpretation profile.
    pub profile: Px4ModeFailsafeObservationProfileV1,
    /// Receiver-local instant when the assessment was performed.
    pub assessed_at: TimestampV1,
    /// Canonically ordered captured source records.
    pub samples: Vec<Px4ModeFailsafeObservationV1>,
    /// Domain-separated assessment commitment.
    pub assessment_digest_hex: String,
}

impl Px4ModeFailsafeObservationAssessmentV1 {
    /// Construct, sort, content-bind, and validate one status-window assessment.
    pub fn new(
        profile: Px4ModeFailsafeObservationProfileV1,
        assessed_at: TimestampV1,
        mut samples: Vec<Px4ModeFailsafeObservationV1>,
    ) -> Result<Self, Px4ModeFailsafeObservationError> {
        samples.sort_by_key(|sample| sample.received_at.nanoseconds);
        let mut value = Self {
            schema_version: PX4_MODE_FAILSAFE_OBSERVATION_SCHEMA_V1,
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

    /// Validate exact target/session/profile binding, sample order, and commitments.
    pub fn validate(&self) -> Result<(), Px4ModeFailsafeObservationError> {
        self.validate_without_digest()?;
        if self.assessment_digest_hex != self.compute_digest_hex() {
            return Err(Px4ModeFailsafeObservationError::DigestMismatch("assessment"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact status-window assessment.
    pub fn assessment_id(&self) -> Result<String, Px4ModeFailsafeObservationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.mode-failsafe-assessment.v1:{}",
            self.assessment_digest_hex
        ))
    }

    /// Inspect the latest source fields after validating the complete retained window.
    ///
    /// A fresh observation only states what the configured telemetry profile reports.
    /// It is not authentication, flight safety, a failsafe-success verdict, or authority.
    pub fn outcome(
        &self,
    ) -> Result<Px4ModeFailsafeObservationOutcomeV1, Px4ModeFailsafeObservationError> {
        self.validate()?;
        if self.samples.len() < usize::from(self.profile.min_samples) {
            return Ok(Px4ModeFailsafeObservationOutcomeV1::Incomplete {
                available_samples: self.samples.len(),
                required_samples: self.profile.min_samples,
            });
        }
        let latest = self
            .samples
            .last()
            .ok_or(Px4ModeFailsafeObservationError::EmptySampleWindow)?;
        let age_ns = self
            .assessed_at
            .elapsed_since(&latest.received_at)
            .map_err(Px4ModeFailsafeObservationError::Timestamp)?;
        if age_ns > self.profile.max_latest_observation_age_ns {
            return Ok(Px4ModeFailsafeObservationOutcomeV1::Stale {
                age_ns,
                max_age_ns: self.profile.max_latest_observation_age_ns,
            });
        }

        let previous = self.samples.get(self.samples.len() - 2);
        Ok(Px4ModeFailsafeObservationOutcomeV1::FreshObservation {
            state: Px4ObservedModeFailsafeStateV1 {
                nav_state_raw: latest.nav_state_raw,
                reports_offboard_mode: latest.nav_state_raw == self.profile.offboard_nav_state_raw,
                accepts_offboard_setpoints: latest.accepts_offboard_setpoints,
                failsafe_active: latest.failsafe_active,
                failsafe_and_user_took_over: latest.failsafe_and_user_took_over,
                nav_state_user_intention_raw: latest.nav_state_user_intention_raw,
                failsafe_defer_state_raw: latest.failsafe_defer_state_raw,
                previous_nav_state_raw: previous.map(|sample| sample.nav_state_raw),
                nav_state_transition_observed: previous
                    .is_some_and(|sample| sample.nav_state_raw != latest.nav_state_raw),
                previous_failsafe_active: previous.map(|sample| sample.failsafe_active),
                failsafe_transition_observed: previous
                    .is_some_and(|sample| sample.failsafe_active != latest.failsafe_active),
                latest_observation_age_ns: age_ns,
            },
        })
    }

    fn validate_without_digest(&self) -> Result<(), Px4ModeFailsafeObservationError> {
        validate_schema(self.schema_version)?;
        self.profile.validate()?;
        self.assessed_at
            .validate()
            .map_err(Px4ModeFailsafeObservationError::Timestamp)?;
        if self.assessed_at.clock_domain != self.profile.receiver_clock_domain {
            return Err(Px4ModeFailsafeObservationError::ReceiverClockMismatch);
        }

        let mut seen_digest = HashSet::with_capacity(self.samples.len());
        let mut seen_sequence = HashSet::with_capacity(self.samples.len());
        let mut previous_received_ns: Option<u64> = None;
        let mut previous_source_us: Option<u64> = None;
        let mut previous_sequence: Option<u64> = None;
        let mut previous_nav_state: Option<u8> = None;
        let mut previous_nav_state_timestamp_us: Option<u64> = None;
        for sample in &self.samples {
            sample.validate()?;
            if sample.producer_profile_id != self.profile.producer_profile_id {
                return Err(Px4ModeFailsafeObservationError::ProducerMismatch);
            }
            if sample.target_instance_id != self.profile.target_instance_id {
                return Err(Px4ModeFailsafeObservationError::TargetMismatch);
            }
            if sample.controller_profile_id != self.profile.controller_profile_id {
                return Err(Px4ModeFailsafeObservationError::ControllerProfileMismatch);
            }
            if sample.source_boot_session_id != self.profile.source_boot_session_id {
                return Err(Px4ModeFailsafeObservationError::BootSessionMismatch);
            }
            if sample.received_at.clock_domain != self.profile.receiver_clock_domain {
                return Err(Px4ModeFailsafeObservationError::ReceiverClockMismatch);
            }
            if !seen_digest.insert(sample.observation_digest_hex.clone()) {
                return Err(Px4ModeFailsafeObservationError::DuplicateObservation);
            }
            if !seen_sequence.insert(sample.capture_sequence) {
                return Err(Px4ModeFailsafeObservationError::DuplicateSequence);
            }
            if previous_received_ns
                .is_some_and(|previous| sample.received_at.nanoseconds <= previous)
            {
                return Err(Px4ModeFailsafeObservationError::NonAdvancingReceiveTime);
            }
            if previous_source_us
                .is_some_and(|previous| sample.source_timestamp_us <= previous)
            {
                return Err(Px4ModeFailsafeObservationError::NonAdvancingSourceTime);
            }
            if let Some(previous) = previous_sequence {
                let expected = previous
                    .checked_add(1)
                    .ok_or(Px4ModeFailsafeObservationError::SequenceOverflow)?;
                if sample.capture_sequence != expected {
                    return Err(Px4ModeFailsafeObservationError::CaptureSequenceGap {
                        previous,
                        found: sample.capture_sequence,
                    });
                }
            }
            if let (Some(previous_state), Some(previous_state_timestamp)) =
                (previous_nav_state, previous_nav_state_timestamp_us)
            {
                if sample.nav_state_raw != previous_state
                    && sample.nav_state_timestamp_us <= previous_state_timestamp
                {
                    return Err(Px4ModeFailsafeObservationError::ModeTransitionTimestampDidNotAdvance);
                }
                if sample.nav_state_raw == previous_state
                    && sample.nav_state_timestamp_us != previous_state_timestamp
                {
                    return Err(Px4ModeFailsafeObservationError::UnexplainedModeTimestampChange);
                }
            }
            previous_received_ns = Some(sample.received_at.nanoseconds);
            previous_source_us = Some(sample.source_timestamp_us);
            previous_sequence = Some(sample.capture_sequence);
            previous_nav_state = Some(sample.nav_state_raw);
            previous_nav_state_timestamp_us = Some(sample.nav_state_timestamp_us);
        }

        if let Some(latest) = self.samples.last() {
            self.assessed_at
                .elapsed_since(&latest.received_at)
                .map_err(|error| match error {
                    EvidenceValidationError::NonMonotonicTimestamp => {
                        Px4ModeFailsafeObservationError::AssessmentBeforeLatestObservation
                    }
                    other => Px4ModeFailsafeObservationError::Timestamp(other),
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
            feed_str(&mut hasher, &sample.observation_digest_hex);
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_schema(schema_version: u16) -> Result<(), Px4ModeFailsafeObservationError> {
    if schema_version != PX4_MODE_FAILSAFE_OBSERVATION_SCHEMA_V1 {
        return Err(Px4ModeFailsafeObservationError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), Px4ModeFailsafeObservationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4ModeFailsafeObservationError::InvalidIdentifier(field));
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

/// Validation or interpretation error for PX4 mode/failsafe observations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4ModeFailsafeObservationError {
    /// Schema version is not supported by this v1 record.
    #[error("unsupported PX4 mode/failsafe schema version {found}")]
    UnsupportedSchemaVersion { /// Unsupported version encountered.
        found: u16,
    },
    /// Identifier is empty, padded, too long, or contains control characters.
    #[error("invalid {0}")]
    InvalidIdentifier(&'static str),
    /// Two records in the assessment share the same observation digest.
    #[error("duplicate PX4 mode/failsafe observation")]
    DuplicateObservation,
    /// Capture sequence was reused.
    #[error("duplicate PX4 mode/failsafe capture sequence")]
    DuplicateSequence,
    /// Capture sequence skipped at least one record.
    #[error("capture sequence gap after {previous}: next captured sequence is {found}")]
    CaptureSequenceGap { /// Last captured sequence.
        previous: u64,
        /// Next captured sequence.
        found: u64,
    },
    /// Sequence counter cannot advance past u64::MAX.
    #[error("PX4 mode/failsafe capture sequence overflow")]
    SequenceOverflow,
    /// Source record is for a different producer.
    #[error("PX4 mode/failsafe producer profile mismatch")]
    ProducerMismatch,
    /// Source record is for a different target vehicle instance.
    #[error("PX4 mode/failsafe target instance mismatch")]
    TargetMismatch,
    /// Source record uses a different controller/schema qualification.
    #[error("PX4 mode/failsafe controller profile mismatch")]
    ControllerProfileMismatch,
    /// Source record is from a different PX4 reboot/boot-relative clock session.
    #[error("PX4 mode/failsafe boot session mismatch")]
    BootSessionMismatch,
    /// Receiver-side clock differs from the exact profile.
    #[error("PX4 mode/failsafe receiver clock mismatch")]
    ReceiverClockMismatch,
    /// Local receive timestamp is not strictly advancing.
    #[error("PX4 mode/failsafe receive time does not advance strictly")]
    NonAdvancingReceiveTime,
    /// PX4 source timestamp is not strictly advancing within a boot session.
    #[error("PX4 mode/failsafe source time does not advance strictly")]
    NonAdvancingSourceTime,
    /// New navigation state lacks an advancing PX4 state-activation timestamp.
    #[error("PX4 nav-state transition timestamp did not advance")]
    ModeTransitionTimestampDidNotAdvance,
    /// Navigation state remained unchanged but its activation timestamp changed.
    #[error("PX4 nav-state activation timestamp changed without a nav-state transition")]
    UnexplainedModeTimestampChange,
    /// Navigation-state activation timestamp exceeds the source status timestamp.
    #[error("PX4 nav-state timestamp is later than the source status timestamp")]
    NavStateTimestampAfterStatus,
    /// Takeover cannot be asserted when the source's failsafe flag is false.
    #[error("PX4 failsafe/user-takeover flag set while failsafe is false")]
    TakeoverWithoutFailsafe,
    /// Repeated observation window was empty.
    #[error("PX4 mode/failsafe observation window is empty")]
    EmptySampleWindow,
    /// Latest receiver-side observation predates its source capture.
    #[error("assessment precedes latest PX4 observation")]
    AssessmentBeforeLatestObservation,
    /// Sample floor must be at least two for transition inspection.
    #[error("PX4 mode/failsafe assessment min_samples must be at least 2")]
    MinimumSamplesTooSmall,
    /// Latest-observation age limit cannot be zero.
    #[error("PX4 mode/failsafe latest-observation age limit must be non-zero")]
    ZeroLatestAge,
    /// Timestamp or clock evidence validation failed.
    #[error("invalid PX4 mode/failsafe timestamp: {0}")]
    Timestamp(EvidenceValidationError),
    /// Stored record, profile, or assessment no longer matches its commitment.
    #[error("PX4 mode/failsafe {0} commitment mismatch")]
    DigestMismatch(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(name: &str) -> ClockDomainId {
        ClockDomainId::new(name).unwrap()
    }

    fn ts(ns: u64) -> TimestampV1 {
        TimestampV1::new(clock("companion.monotonic"), ns)
    }

    fn profile() -> Px4ModeFailsafeObservationProfileV1 {
        Px4ModeFailsafeObservationProfileV1::new(
            "px4.vehicle-status.v3.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-control.fixture.v1",
            "px4.status.decoder.fixture.v1",
            "px4.boot.fixture.session-a",
            clock("companion.monotonic"),
            14,
            2,
            1_500_000_000,
        )
        .unwrap()
    }

    fn observation(
        sequence: u64,
        source_us: u64,
        nav_state_timestamp_us: u64,
        nav_state_raw: u8,
        failsafe_active: bool,
        accepts_offboard_setpoints: bool,
        received_ns: u64,
    ) -> Px4ModeFailsafeObservationV1 {
        Px4ModeFailsafeObservationV1::new(
            "px4.status.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-control.fixture.v1",
            "px4.boot.fixture.session-a",
            sequence,
            source_us,
            nav_state_timestamp_us,
            nav_state_raw,
            nav_state_raw,
            accepts_offboard_setpoints,
            failsafe_active,
            failsafe_active && sequence == 12,
            0,
            ts(received_ns),
        )
        .unwrap()
    }

    fn assessment(
        samples: Vec<Px4ModeFailsafeObservationV1>,
        assessed_at_ns: u64,
    ) -> Px4ModeFailsafeObservationAssessmentV1 {
        Px4ModeFailsafeObservationAssessmentV1::new(profile(), ts(assessed_at_ns), samples).unwrap()
    }

    #[test]
    fn fresh_offboard_mode_is_reported_without_becoming_readiness() {
        let value = assessment(
            vec![
                observation(10, 1_000, 900, 14, false, true, 1_000_000_000),
                observation(11, 2_000, 900, 14, false, true, 1_100_000_000),
            ],
            1_200_000_000,
        );
        let outcome = value.outcome().unwrap();
        assert!(matches!(
            outcome,
            Px4ModeFailsafeObservationOutcomeV1::FreshObservation { state }
                if state.reports_offboard_mode
                    && state.accepts_offboard_setpoints
                    && !state.failsafe_active
                    && !state.nav_state_transition_observed
                    && !state.failsafe_transition_observed
        ));
        assert!(value.assessment_id().unwrap().starts_with(
            "symthaea.autopilot.px4.mode-failsafe-assessment.v1:"
        ));
    }

    #[test]
    fn observed_mode_and_failsafe_changes_are_preserved_independently() {
        let value = assessment(
            vec![
                observation(10, 1_000, 900, 3, false, false, 1_000_000_000),
                observation(11, 2_000, 2_000, 14, true, true, 1_100_000_000),
            ],
            1_200_000_000,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4ModeFailsafeObservationOutcomeV1::FreshObservation { state }
                if state.reports_offboard_mode
                    && state.nav_state_transition_observed
                    && state.failsafe_transition_observed
                    && state.previous_nav_state_raw == Some(3)
                    && state.previous_failsafe_active == Some(false)
                    && state.failsafe_active
                    && state.failsafe_and_user_took_over
        ));
    }

    #[test]
    fn fresh_but_not_offboard_is_reported_as_observation_not_erased() {
        let value = assessment(
            vec![
                observation(10, 1_000, 900, 14, false, true, 1_000_000_000),
                observation(11, 2_000, 2_000, 3, false, false, 1_100_000_000),
            ],
            1_200_000_000,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4ModeFailsafeObservationOutcomeV1::FreshObservation { state }
                if !state.reports_offboard_mode
                    && !state.accepts_offboard_setpoints
                    && state.nav_state_transition_observed
        ));
    }

    #[test]
    fn stale_latest_status_is_not_reported_as_fresh() {
        let value = assessment(
            vec![
                observation(10, 1_000, 900, 14, false, true, 1_000_000_000),
                observation(11, 2_000, 900, 14, false, true, 1_100_000_000),
            ],
            3_000_000_001,
        );
        assert!(matches!(
            value.outcome().unwrap(),
            Px4ModeFailsafeObservationOutcomeV1::Stale { age_ns, max_age_ns }
                if age_ns == 1_900_000_001 && max_age_ns == 1_500_000_000
        ));
    }

    #[test]
    fn insufficient_history_stays_incomplete_even_if_last_state_is_offboard() {
        let value = Px4ModeFailsafeObservationAssessmentV1::new(
            profile(),
            ts(1_100_000_000),
            vec![observation(10, 1_000, 900, 14, false, true, 1_000_000_000)],
        )
        .unwrap();
        assert_eq!(
            value.outcome().unwrap(),
            Px4ModeFailsafeObservationOutcomeV1::Incomplete {
                available_samples: 1,
                required_samples: 2,
            }
        );
    }

    #[test]
    fn sequence_gap_and_source_timestamp_regression_fail_closed() {
        let gap = Px4ModeFailsafeObservationAssessmentV1::new(
            profile(),
            ts(1_200_000_000),
            vec![
                observation(10, 1_000, 900, 14, false, true, 1_000_000_000),
                observation(12, 2_000, 900, 14, false, true, 1_100_000_000),
            ],
        );
        assert_eq!(
            gap.unwrap_err(),
            Px4ModeFailsafeObservationError::CaptureSequenceGap {
                previous: 10,
                found: 12,
            }
        );

        let regressed = Px4ModeFailsafeObservationV1::new(
            "px4.status.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-control.fixture.v1",
            "px4.boot.fixture.session-a",
            11,
            999,
            900,
            14,
            14,
            true,
            false,
            false,
            0,
            ts(1_100_000_000),
        )
        .unwrap();
        assert_eq!(
            Px4ModeFailsafeObservationAssessmentV1::new(
                profile(),
                ts(1_200_000_000),
                vec![
                    observation(10, 1_000, 900, 14, false, true, 1_000_000_000),
                    regressed,
                ],
            )
            .unwrap_err(),
            Px4ModeFailsafeObservationError::NonAdvancingSourceTime
        );
    }

    #[test]
    fn a_rebooted_source_session_cannot_be_mixed_into_one_window() {
        let rebooted = Px4ModeFailsafeObservationV1::new(
            "px4.status.decoder.fixture.v1",
            "vehicle.slot-a",
            "px4.firmware-vehicle-control.fixture.v1",
            "px4.boot.fixture.session-b",
            11,
            1,
            1,
            14,
            14,
            true,
            false,
            false,
            0,
            ts(1_100_000_000),
        )
        .unwrap();
        assert_eq!(
            Px4ModeFailsafeObservationAssessmentV1::new(
                profile(),
                ts(1_200_000_000),
                vec![
                    observation(10, 1_000, 900, 14, false, true, 1_000_000_000),
                    rebooted,
                ],
            )
            .unwrap_err(),
            Px4ModeFailsafeObservationError::BootSessionMismatch
        );
    }

    #[test]
    fn contradictory_takeover_and_timestamp_evidence_is_rejected() {
        assert_eq!(
            Px4ModeFailsafeObservationV1::new(
                "px4.status.decoder.fixture.v1",
                "vehicle.slot-a",
                "px4.firmware-vehicle-control.fixture.v1",
                "px4.boot.fixture.session-a",
                1,
                100,
                101,
                14,
                14,
                true,
                false,
                true,
                0,
                ts(1_000),
            )
            .unwrap_err(),
            Px4ModeFailsafeObservationError::NavStateTimestampAfterStatus
        );
        assert_eq!(
            Px4ModeFailsafeObservationV1::new(
                "px4.status.decoder.fixture.v1",
                "vehicle.slot-a",
                "px4.firmware-vehicle-control.fixture.v1",
                "px4.boot.fixture.session-a",
                1,
                100,
                90,
                14,
                14,
                true,
                false,
                true,
                0,
                ts(1_000),
            )
            .unwrap_err(),
            Px4ModeFailsafeObservationError::TakeoverWithoutFailsafe
        );
    }

    #[test]
    fn serde_mutation_and_profile_change_are_bound_into_identity() {
        let value = assessment(
            vec![
                observation(10, 1_000, 900, 14, false, true, 1_000_000_000),
                observation(11, 2_000, 900, 14, false, true, 1_100_000_000),
            ],
            1_200_000_000,
        );
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut restored: Px4ModeFailsafeObservationAssessmentV1 =
            serde_json::from_slice(&bytes).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.assessment_id().unwrap(), value.assessment_id().unwrap());

        restored.samples[0].failsafe_active = true;
        assert!(restored.validate().is_err());

        let mut changed_profile = value.clone();
        changed_profile.profile.offboard_nav_state_raw = 15;
        assert!(changed_profile.validate().is_err());
    }
}
