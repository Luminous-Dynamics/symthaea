// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Repeated-evidence qualification for the PX4 `TimesyncQualified` readiness fact.
//!
//! PX4 `TimesyncStatus` preserves raw and estimated offset plus RTT, but does not
//! publish one universal `converged` boolean. This module evaluates an explicit
//! Symthaea qualification profile over repeated status evidence. It does not infer
//! PX4's private synchronizer state and exposes no arbitrary clock-conversion API.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{ClockDomainId, EvidenceValidationError, TimestampV1};
use thiserror::Error;

use crate::readiness::{
    Px4ReadinessFactV1, Px4ReadinessRequirementV1, Px4ReadinessValidationError,
};
use crate::{Px4TimesyncEvidenceError, Px4TimesyncStatusEvidenceV1};

/// Schema version for PX4 timesync qualification records.
pub const PX4_TIMESYNC_QUALIFICATION_SCHEMA_V1: u16 = 1;
const PROFILE_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.timesync-qualification-profile.v1\0";
const QUALIFICATION_DOMAIN_V1: &[u8] = b"symthaea.autopilot.px4.timesync-qualification.v1\0";

/// Known PX4 synchronization source required by one qualification profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Px4TimesyncRequiredSourceV1 {
    /// MAVLink TIMESYNC (`source_protocol == 1`).
    Mavlink,
    /// DDS/uXRCE-DDS synchronization (`source_protocol == 2`).
    Dds,
}

impl Px4TimesyncRequiredSourceV1 {
    /// Raw PX4 `source_protocol` value required by this profile.
    pub const fn raw_value(self) -> u8 {
        match self {
            Self::Mavlink => 1,
            Self::Dds => 2,
        }
    }

    /// Stable wire token used in content commitments.
    pub const fn wire_token(self) -> &'static str {
        match self {
            Self::Mavlink => "mavlink",
            Self::Dds => "dds",
        }
    }
}

/// Exact quality/currentness policy applied to repeated PX4 timesync status.
///
/// Absolute offset is deliberately absent. A companion and PX4 may use very
/// different clock origins while maintaining a stable, useful relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4TimesyncQualificationProfileV1 {
    /// Schema version. Must equal [`PX4_TIMESYNC_QUALIFICATION_SCHEMA_V1`].
    pub schema_version: u16,
    /// Stable versioned profile identity.
    pub profile_id: String,
    /// Required PX4 synchronization source for every retained status sample.
    pub required_source: Px4TimesyncRequiredSourceV1,
    /// Minimum number of distinct repeated samples required for a verdict.
    pub min_samples: u16,
    /// Maximum admitted RTT for every sample, in microseconds.
    pub max_round_trip_time_us: u32,
    /// Maximum admitted span of PX4's estimated offset across the window.
    pub max_estimated_offset_span_us: u64,
    /// Maximum age of the newest status at assessment, in microseconds.
    pub max_latest_status_age_us: u64,
    /// Optional maximum gap between adjacent status publications, in microseconds.
    pub max_inter_sample_gap_us: Option<u64>,
    /// Domain-separated content commitment over this exact policy.
    pub profile_digest_hex: String,
}

impl Px4TimesyncQualificationProfileV1 {
    /// Construct, content-bind, and validate one qualification profile.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile_id: impl Into<String>,
        required_source: Px4TimesyncRequiredSourceV1,
        min_samples: u16,
        max_round_trip_time_us: u32,
        max_estimated_offset_span_us: u64,
        max_latest_status_age_us: u64,
        max_inter_sample_gap_us: Option<u64>,
    ) -> Result<Self, Px4TimesyncQualificationError> {
        let mut value = Self {
            schema_version: PX4_TIMESYNC_QUALIFICATION_SCHEMA_V1,
            profile_id: profile_id.into(),
            required_source,
            min_samples,
            max_round_trip_time_us,
            max_estimated_offset_span_us,
            max_latest_status_age_us,
            max_inter_sample_gap_us,
            profile_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.profile_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate schema, threshold representability, identity, and commitment.
    pub fn validate(&self) -> Result<(), Px4TimesyncQualificationError> {
        self.validate_without_digest()?;
        if self.profile_digest_hex != self.compute_digest_hex() {
            return Err(Px4TimesyncQualificationError::DigestMismatch(
                "qualification_profile",
            ));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact qualification policy.
    pub fn profile_content_id(&self) -> Result<String, Px4TimesyncQualificationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.timesync-qualification-profile.v1:{}",
            self.profile_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), Px4TimesyncQualificationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.profile_id, "profile_id")?;
        if self.min_samples < 2 {
            return Err(Px4TimesyncQualificationError::MinimumSamplesTooSmall);
        }
        if self.max_inter_sample_gap_us == Some(0) {
            return Err(Px4TimesyncQualificationError::ZeroInterSampleGap);
        }
        micros_to_nanos(
            self.max_latest_status_age_us,
            "max_latest_status_age_us",
        )?;
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.profile_id);
        feed_str(&mut hasher, self.required_source.wire_token());
        hasher.update(&self.min_samples.to_le_bytes());
        hasher.update(&self.max_round_trip_time_us.to_le_bytes());
        hasher.update(&self.max_estimated_offset_span_us.to_le_bytes());
        hasher.update(&self.max_latest_status_age_us.to_le_bytes());
        match self.max_inter_sample_gap_us {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_le_bytes());
            }
            None => hasher.update(&[0]),
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

/// Window-level diagnostics recomputed from retained statuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Px4TimesyncQualificationMetricsV1 {
    /// Number of distinct samples in the window.
    pub sample_count: usize,
    /// Largest per-sample PX4 RTT, in microseconds.
    pub max_round_trip_time_us: u32,
    /// Minimum estimated offset observed in the window.
    pub min_estimated_offset_us: i64,
    /// Maximum estimated offset observed in the window.
    pub max_estimated_offset_us: i64,
    /// Difference between maximum and minimum estimated offset, in microseconds.
    pub estimated_offset_span_us: u64,
    /// Age of the newest PX4 status at assessment, in nanoseconds.
    pub newest_status_age_ns: u64,
    /// Largest adjacent PX4 publication gap, in microseconds.
    pub max_inter_sample_gap_us: u64,
}

/// Quality/currentness condition preventing a sufficient window from qualifying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Px4TimesyncQualificationFailureV1 {
    /// At least one sample exceeds the RTT ceiling.
    RoundTripTimeExceeded,
    /// Estimated offset varies beyond the admitted span.
    EstimatedOffsetSpanExceeded,
    /// The newest retained status is too old.
    LatestStatusStale,
    /// An adjacent publication gap exceeds the optional continuity ceiling.
    InterSampleGapExceeded,
}

/// Derived qualification result; this is a runtime view, not independent evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Px4TimesyncQualificationOutcomeV1 {
    /// Repeated evidence satisfies the selected Symthaea profile.
    Qualified,
    /// Enough evidence exists, but one or more quality/currentness thresholds fail.
    NotQualified {
        /// Deterministically ordered failed conditions.
        failures: Vec<Px4TimesyncQualificationFailureV1>,
    },
    /// Too few repeated samples exist to make the profile claim.
    Incomplete {
        /// Number of samples currently retained.
        available_samples: usize,
        /// Number required by the profile.
        required_samples: u16,
    },
}

/// Content-addressed repeated PX4 timesync evidence under one exact profile.
///
/// Samples are canonicalized by PX4 publication timestamp. `assessed_at` must use
/// the explicitly supplied PX4-local clock domain; no cross-clock arithmetic or
/// arbitrary clock transform is performed here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4TimesyncQualificationV1 {
    /// Schema version. Must equal [`PX4_TIMESYNC_QUALIFICATION_SCHEMA_V1`].
    pub schema_version: u16,
    /// Exact adapter/producer identity shared by all retained samples.
    pub producer_profile_id: String,
    /// Exact quality/currentness policy applied to the evidence window.
    pub profile: Px4TimesyncQualificationProfileV1,
    /// Explicit clock domain in which PX4 status publication time is interpreted.
    pub px4_clock_domain: ClockDomainId,
    /// PX4-local instant at which qualification is evaluated.
    pub assessed_at: TimestampV1,
    /// Canonically ordered repeated PX4 timesync statuses.
    pub samples: Vec<Px4TimesyncStatusEvidenceV1>,
    /// Domain-separated commitment over this complete evidence window.
    pub qualification_digest_hex: String,
}

impl Px4TimesyncQualificationV1 {
    /// Construct, canonicalize, content-bind, and validate one qualification window.
    pub fn new(
        producer_profile_id: impl Into<String>,
        profile: Px4TimesyncQualificationProfileV1,
        px4_clock_domain: ClockDomainId,
        assessed_at: TimestampV1,
        mut samples: Vec<Px4TimesyncStatusEvidenceV1>,
    ) -> Result<Self, Px4TimesyncQualificationError> {
        samples.sort_by_key(|sample| sample.timestamp_us);
        let mut value = Self {
            schema_version: PX4_TIMESYNC_QUALIFICATION_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            profile,
            px4_clock_domain,
            assessed_at,
            samples,
            qualification_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.qualification_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate the complete repeated-evidence window and content commitment.
    pub fn validate(&self) -> Result<(), Px4TimesyncQualificationError> {
        self.validate_without_digest()?;
        if self.qualification_digest_hex != self.compute_digest_hex() {
            return Err(Px4TimesyncQualificationError::DigestMismatch(
                "timesync_qualification",
            ));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact qualification window.
    pub fn qualification_id(&self) -> Result<String, Px4TimesyncQualificationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.timesync-qualification.v1:{}",
            self.qualification_digest_hex
        ))
    }

    /// Recompute diagnostics from retained status evidence.
    pub fn metrics(&self) -> Result<Px4TimesyncQualificationMetricsV1, Px4TimesyncQualificationError> {
        self.validate()?;
        self.metrics_after_structure()
    }

    /// Derive qualification status under the exact stored profile.
    pub fn outcome(&self) -> Result<Px4TimesyncQualificationOutcomeV1, Px4TimesyncQualificationError> {
        self.validate()?;
        if self.samples.len() < usize::from(self.profile.min_samples) {
            return Ok(Px4TimesyncQualificationOutcomeV1::Incomplete {
                available_samples: self.samples.len(),
                required_samples: self.profile.min_samples,
            });
        }

        let metrics = self.metrics_after_structure()?;
        let mut failures = Vec::new();
        if metrics.max_round_trip_time_us > self.profile.max_round_trip_time_us {
            failures.push(Px4TimesyncQualificationFailureV1::RoundTripTimeExceeded);
        }
        if metrics.estimated_offset_span_us > self.profile.max_estimated_offset_span_us {
            failures.push(Px4TimesyncQualificationFailureV1::EstimatedOffsetSpanExceeded);
        }
        let max_age_ns = self.profile.max_latest_status_age_us * 1_000;
        if metrics.newest_status_age_ns > max_age_ns {
            failures.push(Px4TimesyncQualificationFailureV1::LatestStatusStale);
        }
        if self
            .profile
            .max_inter_sample_gap_us
            .is_some_and(|limit| metrics.max_inter_sample_gap_us > limit)
        {
            failures.push(Px4TimesyncQualificationFailureV1::InterSampleGapExceeded);
        }

        if failures.is_empty() {
            Ok(Px4TimesyncQualificationOutcomeV1::Qualified)
        } else {
            Ok(Px4TimesyncQualificationOutcomeV1::NotQualified { failures })
        }
    }

    /// Lower qualification into the PX4 readiness vocabulary.
    ///
    /// Incomplete evidence remains `Unobserved`; it is not fabricated into a
    /// negative measurement merely because the sample window is too small.
    pub fn readiness_fact(&self) -> Result<Px4ReadinessFactV1, Px4TimesyncQualificationError> {
        match self.outcome()? {
            Px4TimesyncQualificationOutcomeV1::Qualified => Px4ReadinessFactV1::satisfied(
                Px4ReadinessRequirementV1::TimesyncQualified,
                self.qualification_id()?,
            )
            .map_err(Px4TimesyncQualificationError::Readiness),
            Px4TimesyncQualificationOutcomeV1::NotQualified { .. } => {
                Px4ReadinessFactV1::unsatisfied(
                    Px4ReadinessRequirementV1::TimesyncQualified,
                    self.qualification_id()?,
                )
                .map_err(Px4TimesyncQualificationError::Readiness)
            }
            Px4TimesyncQualificationOutcomeV1::Incomplete { .. } => Ok(
                Px4ReadinessFactV1::unobserved(Px4ReadinessRequirementV1::TimesyncQualified),
            ),
        }
    }

    fn validate_without_digest(&self) -> Result<(), Px4TimesyncQualificationError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")?;
        self.profile.validate()?;
        self.px4_clock_domain
            .validate()
            .map_err(Px4TimesyncQualificationError::Timestamp)?;
        self.assessed_at
            .validate()
            .map_err(Px4TimesyncQualificationError::Timestamp)?;
        if self.assessed_at.clock_domain != self.px4_clock_domain {
            return Err(Px4TimesyncQualificationError::AssessmentClockMismatch);
        }

        let required_source = self.profile.required_source.raw_value();
        let mut seen = HashSet::with_capacity(self.samples.len());
        let mut previous_publication = None;
        let mut previous_remote = None;
        for sample in &self.samples {
            sample.validate()?;
            if sample.producer_profile_id != self.producer_profile_id {
                return Err(Px4TimesyncQualificationError::MixedProducer);
            }
            if sample.source_protocol != required_source {
                return Err(Px4TimesyncQualificationError::UnexpectedSourceProtocol {
                    found: sample.source_protocol,
                    required: required_source,
                });
            }
            if !seen.insert(sample.status_digest_hex.clone()) {
                return Err(Px4TimesyncQualificationError::DuplicateSample);
            }
            if previous_publication.is_some_and(|previous| sample.timestamp_us <= previous) {
                return Err(Px4TimesyncQualificationError::NonAdvancingPublicationTime);
            }
            if previous_remote.is_some_and(|previous| sample.remote_timestamp_us <= previous) {
                return Err(Px4TimesyncQualificationError::NonAdvancingRemoteTime);
            }
            previous_publication = Some(sample.timestamp_us);
            previous_remote = Some(sample.remote_timestamp_us);
        }

        if let Some(latest) = self.samples.last() {
            let latest_at = latest.status_observed_at(self.px4_clock_domain.clone())?;
            self.assessed_at
                .elapsed_since(&latest_at)
                .map_err(|error| match error {
                    EvidenceValidationError::NonMonotonicTimestamp => {
                        Px4TimesyncQualificationError::AssessmentBeforeLatestStatus
                    }
                    other => Px4TimesyncQualificationError::Timestamp(other),
                })?;
        }
        Ok(())
    }

    fn metrics_after_structure(
        &self,
    ) -> Result<Px4TimesyncQualificationMetricsV1, Px4TimesyncQualificationError> {
        let first = self
            .samples
            .first()
            .ok_or(Px4TimesyncQualificationError::EmptySampleWindow)?;
        let mut min_offset = first.estimated_offset_us;
        let mut max_offset = first.estimated_offset_us;
        let mut max_rtt = first.round_trip_time_us;
        let mut max_gap = 0u64;

        for sample in self.samples.iter().skip(1) {
            min_offset = min_offset.min(sample.estimated_offset_us);
            max_offset = max_offset.max(sample.estimated_offset_us);
            max_rtt = max_rtt.max(sample.round_trip_time_us);
        }
        for pair in self.samples.windows(2) {
            let gap = pair[1]
                .timestamp_us
                .checked_sub(pair[0].timestamp_us)
                .ok_or(Px4TimesyncQualificationError::NonAdvancingPublicationTime)?;
            max_gap = max_gap.max(gap);
        }

        let estimated_offset_span_us = u64::try_from(
            i128::from(max_offset) - i128::from(min_offset),
        )
        .map_err(|_| Px4TimesyncQualificationError::OffsetSpanOutOfRange)?;
        let latest_at = self
            .samples
            .last()
            .ok_or(Px4TimesyncQualificationError::EmptySampleWindow)?
            .status_observed_at(self.px4_clock_domain.clone())?;
        let newest_status_age_ns = self
            .assessed_at
            .elapsed_since(&latest_at)
            .map_err(Px4TimesyncQualificationError::Timestamp)?;

        Ok(Px4TimesyncQualificationMetricsV1 {
            sample_count: self.samples.len(),
            max_round_trip_time_us: max_rtt,
            min_estimated_offset_us: min_offset,
            max_estimated_offset_us: max_offset,
            estimated_offset_span_us,
            newest_status_age_ns,
            max_inter_sample_gap_us: max_gap,
        })
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(QUALIFICATION_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        feed_str(&mut hasher, &self.profile.profile_digest_hex);
        feed_str(&mut hasher, self.px4_clock_domain.as_str());
        feed_str(&mut hasher, self.assessed_at.clock_domain.as_str());
        hasher.update(&self.assessed_at.nanoseconds.to_le_bytes());
        hasher.update(&(self.samples.len() as u64).to_le_bytes());
        for sample in &self.samples {
            feed_str(&mut hasher, &sample.status_digest_hex);
        }
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_schema(schema_version: u16) -> Result<(), Px4TimesyncQualificationError> {
    if schema_version != PX4_TIMESYNC_QUALIFICATION_SCHEMA_V1 {
        return Err(Px4TimesyncQualificationError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), Px4TimesyncQualificationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4TimesyncQualificationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn micros_to_nanos(value: u64, field: &'static str) -> Result<u64, Px4TimesyncQualificationError> {
    value
        .checked_mul(1_000)
        .ok_or(Px4TimesyncQualificationError::TimestampOverflow(field))
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

/// Validation or derivation failure for PX4 timesync qualification evidence.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4TimesyncQualificationError {
    /// Unsupported qualification schema version.
    #[error("unsupported PX4 timesync qualification schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Required identifier is empty, padded, too long, or contains controls.
    #[error("invalid {0}")]
    InvalidIdentifier(&'static str),
    /// Qualification profiles require at least two repeated samples.
    #[error("PX4 timesync qualification min_samples must be at least 2")]
    MinimumSamplesTooSmall,
    /// Configured inter-sample continuity bound cannot be zero.
    #[error("PX4 timesync max inter-sample gap must be non-zero when configured")]
    ZeroInterSampleGap,
    /// Metrics require at least one retained status sample.
    #[error("PX4 timesync qualification sample window is empty")]
    EmptySampleWindow,
    /// Retained statuses do not all come from the declared producer.
    #[error("PX4 timesync qualification mixes producer identities")]
    MixedProducer,
    /// A retained status source differs from the exact qualification profile.
    #[error("PX4 timesync source protocol {found} does not match required {required}")]
    UnexpectedSourceProtocol {
        /// Raw source found in evidence.
        found: u8,
        /// Raw source required by the profile.
        required: u8,
    },
    /// Same content-addressed status was included more than once.
    #[error("PX4 timesync qualification contains a duplicate status sample")]
    DuplicateSample,
    /// PX4 publication timestamps must advance strictly.
    #[error("PX4 timesync status publication time does not advance strictly")]
    NonAdvancingPublicationTime,
    /// Remote synchronization timestamps must advance strictly.
    #[error("PX4 timesync remote timestamp does not advance strictly")]
    NonAdvancingRemoteTime,
    /// Assessment time is labeled with a clock different from the declared PX4 clock.
    #[error("PX4 timesync assessment clock does not match declared PX4 clock")]
    AssessmentClockMismatch,
    /// Assessment instant precedes the newest PX4 status publication.
    #[error("PX4 timesync assessment time precedes the latest status")]
    AssessmentBeforeLatestStatus,
    /// Microsecond value cannot be represented in the nanosecond evidence clock.
    #[error("PX4 timesync qualification field {0} overflows nanosecond representation")]
    TimestampOverflow(&'static str),
    /// Estimated-offset extrema could not be represented as an unsigned span.
    #[error("PX4 timesync estimated-offset span is out of range")]
    OffsetSpanOutOfRange,
    /// Nested preserved PX4 timesync evidence is invalid.
    #[error("invalid PX4 timesync status evidence: {0}")]
    Status(#[from] Px4TimesyncEvidenceError),
    /// Assessment or PX4 clock evidence is invalid.
    #[error("invalid PX4 timesync timestamp: {0}")]
    Timestamp(EvidenceValidationError),
    /// Lowering into PX4 readiness vocabulary failed validation.
    #[error("invalid PX4 readiness fact: {0}")]
    Readiness(Px4ReadinessValidationError),
    /// Stored profile or qualification commitment no longer matches its contents.
    #[error("PX4 timesync {0} content commitment mismatch")]
    DigestMismatch(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readiness::Px4ReadinessFactStateV1;

    fn clock(name: &str) -> ClockDomainId {
        ClockDomainId::new(name).unwrap()
    }

    fn ts_us(clock_name: &str, value: u64) -> TimestampV1 {
        TimestampV1::new(clock(clock_name), value.checked_mul(1_000).unwrap())
    }

    fn profile() -> Px4TimesyncQualificationProfileV1 {
        Px4TimesyncQualificationProfileV1::new(
            "px4.timesync.readiness.fixture.v1",
            Px4TimesyncRequiredSourceV1::Mavlink,
            3,
            100,
            20,
            250,
            Some(200),
        )
        .unwrap()
    }

    fn sample(
        producer: &str,
        timestamp_us: u64,
        source_protocol: u8,
        remote_timestamp_us: u64,
        estimated_offset_us: i64,
        rtt_us: u32,
    ) -> Px4TimesyncStatusEvidenceV1 {
        Px4TimesyncStatusEvidenceV1::new(
            producer,
            timestamp_us,
            source_protocol,
            remote_timestamp_us,
            estimated_offset_us,
            estimated_offset_us,
            rtt_us,
        )
        .unwrap()
    }

    fn good_samples(offset_base: i64) -> Vec<Px4TimesyncStatusEvidenceV1> {
        vec![
            sample("px4.fixture", 1_000, 1, 10_000, offset_base, 40),
            sample("px4.fixture", 1_100, 1, 10_100, offset_base + 5, 50),
            sample("px4.fixture", 1_200, 1, 10_200, offset_base - 5, 45),
        ]
    }

    fn qualification(
        samples: Vec<Px4TimesyncStatusEvidenceV1>,
        assessed_at_us: u64,
    ) -> Px4TimesyncQualificationV1 {
        Px4TimesyncQualificationV1::new(
            "px4.fixture",
            profile(),
            clock("px4.hrt"),
            ts_us("px4.hrt", assessed_at_us),
            samples,
        )
        .unwrap()
    }

    #[test]
    fn repeated_stable_status_qualifies_and_emits_satisfied_fact() {
        let value = qualification(good_samples(500), 1_250);
        assert_eq!(
            value.outcome().unwrap(),
            Px4TimesyncQualificationOutcomeV1::Qualified
        );
        let fact = value.readiness_fact().unwrap();
        assert_eq!(fact.requirement, Px4ReadinessRequirementV1::TimesyncQualified);
        assert_eq!(fact.state, Px4ReadinessFactStateV1::Satisfied);
        assert_eq!(fact.evidence_id, Some(value.qualification_id().unwrap()));
    }

    #[test]
    fn absolute_clock_origin_offset_is_not_a_qualification_threshold() {
        let near = qualification(good_samples(500), 1_250);
        let far = qualification(good_samples(5_000_000_000), 1_250);
        assert_eq!(near.outcome().unwrap(), Px4TimesyncQualificationOutcomeV1::Qualified);
        assert_eq!(far.outcome().unwrap(), Px4TimesyncQualificationOutcomeV1::Qualified);
        assert_ne!(near.qualification_id().unwrap(), far.qualification_id().unwrap());
    }

    #[test]
    fn bad_rtt_and_offset_instability_are_not_qualified() {
        let high_rtt = vec![
            sample("px4.fixture", 1_000, 1, 10_000, 500, 40),
            sample("px4.fixture", 1_100, 1, 10_100, 505, 101),
            sample("px4.fixture", 1_200, 1, 10_200, 495, 45),
        ];
        assert!(matches!(
            qualification(high_rtt, 1_250).outcome().unwrap(),
            Px4TimesyncQualificationOutcomeV1::NotQualified { failures }
                if failures.contains(&Px4TimesyncQualificationFailureV1::RoundTripTimeExceeded)
        ));

        let unstable = vec![
            sample("px4.fixture", 1_000, 1, 10_000, 500, 40),
            sample("px4.fixture", 1_100, 1, 10_100, 550, 40),
            sample("px4.fixture", 1_200, 1, 10_200, 490, 40),
        ];
        assert!(matches!(
            qualification(unstable, 1_250).outcome().unwrap(),
            Px4TimesyncQualificationOutcomeV1::NotQualified { failures }
                if failures.contains(&Px4TimesyncQualificationFailureV1::EstimatedOffsetSpanExceeded)
        ));
    }

    #[test]
    fn insufficient_window_is_unobserved_not_negative_including_zero_samples() {
        for samples in [vec![], vec![sample("px4.fixture", 1_000, 1, 10_000, 500, 40)]] {
            let value = qualification(samples, 1_050);
            assert!(matches!(
                value.outcome().unwrap(),
                Px4TimesyncQualificationOutcomeV1::Incomplete { .. }
            ));
            let fact = value.readiness_fact().unwrap();
            assert_eq!(fact.state, Px4ReadinessFactStateV1::Unobserved);
            assert!(fact.evidence_id.is_none());
        }
    }

    #[test]
    fn stale_newest_status_is_not_qualified() {
        assert!(matches!(
            qualification(good_samples(500), 1_451).outcome().unwrap(),
            Px4TimesyncQualificationOutcomeV1::NotQualified { failures }
                if failures.contains(&Px4TimesyncQualificationFailureV1::LatestStatusStale)
        ));
    }

    #[test]
    fn mixed_producer_protocol_and_nonadvancing_remote_time_fail_structure() {
        let mut producers = good_samples(500);
        producers[2] = sample("other.fixture", 1_200, 1, 10_200, 500, 40);
        assert_eq!(
            Px4TimesyncQualificationV1::new(
                "px4.fixture",
                profile(),
                clock("px4.hrt"),
                ts_us("px4.hrt", 1_250),
                producers,
            )
            .unwrap_err(),
            Px4TimesyncQualificationError::MixedProducer
        );

        let mut protocols = good_samples(500);
        protocols[2] = sample("px4.fixture", 1_200, 2, 10_200, 500, 40);
        assert!(matches!(
            Px4TimesyncQualificationV1::new(
                "px4.fixture",
                profile(),
                clock("px4.hrt"),
                ts_us("px4.hrt", 1_250),
                protocols,
            ),
            Err(Px4TimesyncQualificationError::UnexpectedSourceProtocol { .. })
        ));

        let nonadvancing = vec![
            sample("px4.fixture", 1_000, 1, 10_000, 500, 40),
            sample("px4.fixture", 1_100, 1, 10_000, 505, 40),
        ];
        assert_eq!(
            Px4TimesyncQualificationV1::new(
                "px4.fixture",
                profile(),
                clock("px4.hrt"),
                ts_us("px4.hrt", 1_150),
                nonadvancing,
            )
            .unwrap_err(),
            Px4TimesyncQualificationError::NonAdvancingRemoteTime
        );
    }

    #[test]
    fn assessment_clock_is_explicit_and_profile_thresholds_are_representable() {
        assert_eq!(
            Px4TimesyncQualificationV1::new(
                "px4.fixture",
                profile(),
                clock("px4.hrt"),
                ts_us("host.monotonic", 1_250),
                good_samples(500),
            )
            .unwrap_err(),
            Px4TimesyncQualificationError::AssessmentClockMismatch
        );

        assert!(matches!(
            Px4TimesyncQualificationProfileV1::new(
                "overflow.fixture",
                Px4TimesyncRequiredSourceV1::Mavlink,
                2,
                100,
                20,
                u64::MAX,
                None,
            ),
            Err(Px4TimesyncQualificationError::TimestampOverflow(
                "max_latest_status_age_us"
            ))
        ));
    }

    #[test]
    fn mutation_and_serde_boundaries_are_explicit() {
        let value = qualification(good_samples(500), 1_250);
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut restored: Px4TimesyncQualificationV1 = serde_json::from_slice(&bytes).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.qualification_id().unwrap(), value.qualification_id().unwrap());

        restored.samples[0].estimated_offset_us += 1;
        assert!(restored.validate().is_err());
    }
}
