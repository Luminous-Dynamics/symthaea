// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transport-neutral autopilot evidence adapters for Symthaea Embodiment v2.
//!
//! The first tranche preserves PX4 `TimesyncStatus` semantics without opening a
//! network connection, sending a setpoint, changing mode, or arming a vehicle.
//! Raw product/protocol evidence stays at the edge; universal clock semantics are
//! expressed through `symthaea-core` only when a specific correspondence can be
//! justified without inventing unavailable synchronization state.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_clock_alignment::{
    ClockAlignmentSampleV1, ClockAlignmentValidationError,
};
use symthaea_core::embodiment_evidence::{ClockDomainId, TimestampV1};
use thiserror::Error;

/// Schema version for [`Px4TimesyncStatusEvidenceV1`].
pub const PX4_TIMESYNC_STATUS_SCHEMA_V1: u16 = 1;
const PX4_TIMESYNC_STATUS_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea.autopilot.px4.timesync-status.v1\0";

/// PX4's documented synchronization-source values, with unknown values preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Px4TimesyncSourceV1 {
    /// PX4 reports no known synchronization protocol (`source_protocol == 0`).
    Unknown,
    /// MAVLink TIMESYNC (`source_protocol == 1`).
    Mavlink,
    /// DDS/uXRCE-DDS time synchronization (`source_protocol == 2`).
    Dds,
    /// A future or unsupported raw source value.
    Other(u8),
}

impl Px4TimesyncSourceV1 {
    /// Interpret the raw PX4 `source_protocol` value without discarding unknowns.
    pub const fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Unknown,
            1 => Self::Mavlink,
            2 => Self::Dds,
            other => Self::Other(other),
        }
    }

    fn raw_alignment_profile(self) -> Option<&'static str> {
        match self {
            Self::Mavlink => Some("px4.timesync-status.mavlink.raw-observed-offset.rtt-bound.v1"),
            Self::Dds => Some("px4.timesync-status.dds.raw-observed-offset.rtt-bound.v1"),
            Self::Unknown | Self::Other(_) => None,
        }
    }
}

/// Content-addressed preservation of one PX4 `TimesyncStatus` publication.
///
/// Units and field meanings follow the PX4 message contract: timestamps and
/// offsets are microseconds, `timestamp_us` is PX4-local publication time,
/// `remote_timestamp_us` is the remote time associated with the synchronization
/// observation, and `observed_offset_us` / `estimated_offset_us` are kept
/// distinct. This type intentionally has no `converged` field because the PX4
/// `TimesyncStatus` message itself does not provide one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4TimesyncStatusEvidenceV1 {
    /// Schema version. Must equal [`PX4_TIMESYNC_STATUS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Adapter/producer identity that captured this status publication.
    pub producer_profile_id: String,
    /// PX4-local publication timestamp, microseconds since PX4 system start.
    pub timestamp_us: u64,
    /// Raw PX4 source-protocol value; currently 0 unknown, 1 MAVLink, 2 DDS.
    pub source_protocol: u8,
    /// Remote timestamp associated with the synchronization observation, microseconds.
    pub remote_timestamp_us: u64,
    /// Raw observed PX4-local minus remote offset, microseconds.
    pub observed_offset_us: i64,
    /// PX4's smoothed/estimated offset, microseconds.
    pub estimated_offset_us: i64,
    /// Round-trip time associated with the synchronization observation, microseconds.
    pub round_trip_time_us: u32,
    /// Domain-separated content commitment over the fields above.
    pub status_digest_hex: String,
}

impl Px4TimesyncStatusEvidenceV1 {
    /// Construct, content-bind, and validate one preserved PX4 timesync status.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        timestamp_us: u64,
        source_protocol: u8,
        remote_timestamp_us: u64,
        observed_offset_us: i64,
        estimated_offset_us: i64,
        round_trip_time_us: u32,
    ) -> Result<Self, Px4TimesyncEvidenceError> {
        let mut value = Self {
            schema_version: PX4_TIMESYNC_STATUS_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            timestamp_us,
            source_protocol,
            remote_timestamp_us,
            observed_offset_us,
            estimated_offset_us,
            round_trip_time_us,
            status_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.status_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate producer identity, schema, and content commitment.
    pub fn validate(&self) -> Result<(), Px4TimesyncEvidenceError> {
        self.validate_without_digest()?;
        if self.status_digest_hex != self.compute_digest_hex() {
            return Err(Px4TimesyncEvidenceError::DigestMismatch);
        }
        Ok(())
    }

    /// Content-addressed identity for this exact PX4 status publication.
    ///
    /// This detects mutation but is not a signature or authenticated PX4 identity.
    pub fn status_id(&self) -> Result<String, Px4TimesyncEvidenceError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.timesync-status.v1:{}",
            self.status_digest_hex
        ))
    }

    /// Interpret the raw PX4 synchronization-source field.
    pub const fn source(&self) -> Px4TimesyncSourceV1 {
        Px4TimesyncSourceV1::from_raw(self.source_protocol)
    }

    /// Represent when PX4 published this status in the caller-supplied PX4 clock domain.
    ///
    /// This is the status publication time. It is deliberately **not** used as the
    /// local timestamp corresponding to `remote_timestamp_us` in a TIMESYNC exchange.
    pub fn status_observed_at(
        &self,
        px4_clock_domain: ClockDomainId,
    ) -> Result<TimestampV1, Px4TimesyncEvidenceError> {
        self.validate()?;
        let nanoseconds = micros_to_nanos(self.timestamp_us, "timestamp_us")?;
        Ok(TimestampV1::new(px4_clock_domain, nanoseconds))
    }

    /// Derive one raw correspondence candidate from PX4's **observed** offset.
    ///
    /// PX4's synchronizer uses the convention `local = remote + offset`; this
    /// method therefore produces a directed `remote_clock -> px4_clock` sample.
    /// It does not use `estimated_offset_us` and does not claim convergence.
    ///
    /// The stated correspondence-error bound is deliberately conservative:
    /// full reported RTT plus one microsecond of field-quantization margin. This
    /// is a named adapter assumption, not a claim that one-way delay is RTT/2 or
    /// that the PX4 estimator is converged.
    pub fn raw_observed_alignment_sample(
        &self,
        remote_clock_domain: ClockDomainId,
        px4_clock_domain: ClockDomainId,
    ) -> Result<ClockAlignmentSampleV1, Px4TimesyncEvidenceError> {
        self.validate()?;
        let method_profile_id = self
            .source()
            .raw_alignment_profile()
            .ok_or(Px4TimesyncEvidenceError::UnsupportedSourceProtocol(
                self.source_protocol,
            ))?;

        let remote_ns = micros_to_nanos(self.remote_timestamp_us, "remote_timestamp_us")?;
        let offset_ns = i128::from(self.observed_offset_us) * 1_000;
        let local_ns_i128 = i128::from(remote_ns) + offset_ns;
        if !(0..=i128::from(u64::MAX)).contains(&local_ns_i128) {
            return Err(Px4TimesyncEvidenceError::DerivedLocalTimestampOutOfRange);
        }
        let local_ns = u64::try_from(local_ns_i128)
            .map_err(|_| Px4TimesyncEvidenceError::DerivedLocalTimestampOutOfRange)?;

        let error_bound_ns = (u64::from(self.round_trip_time_us) + 1)
            .checked_mul(1_000)
            .ok_or(Px4TimesyncEvidenceError::TimestampOverflow(
                "round_trip_time_us",
            ))?;

        ClockAlignmentSampleV1::new(
            TimestampV1::new(remote_clock_domain, remote_ns),
            TimestampV1::new(px4_clock_domain, local_ns),
            method_profile_id,
            error_bound_ns,
            self.status_id()?,
        )
        .map_err(Px4TimesyncEvidenceError::Alignment)
    }

    fn validate_without_digest(&self) -> Result<(), Px4TimesyncEvidenceError> {
        if self.schema_version != PX4_TIMESYNC_STATUS_SCHEMA_V1 {
            return Err(Px4TimesyncEvidenceError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        validate_identifier(&self.producer_profile_id)?;
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PX4_TIMESYNC_STATUS_COMMITMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        hasher.update(&self.timestamp_us.to_le_bytes());
        hasher.update(&[self.source_protocol]);
        hasher.update(&self.remote_timestamp_us.to_le_bytes());
        hasher.update(&self.observed_offset_us.to_le_bytes());
        hasher.update(&self.estimated_offset_us.to_le_bytes());
        hasher.update(&self.round_trip_time_us.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_identifier(value: &str) -> Result<(), Px4TimesyncEvidenceError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.chars().any(char::is_control)
    {
        return Err(Px4TimesyncEvidenceError::InvalidProducerProfileId);
    }
    Ok(())
}

fn micros_to_nanos(value: u64, field: &'static str) -> Result<u64, Px4TimesyncEvidenceError> {
    value
        .checked_mul(1_000)
        .ok_or(Px4TimesyncEvidenceError::TimestampOverflow(field))
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

/// Error produced while preserving or deriving evidence from PX4 timesync status.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4TimesyncEvidenceError {
    /// Unsupported evidence schema version.
    #[error("unsupported PX4 timesync evidence schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Producer/profile identity was empty, padded, or contained control characters.
    #[error("invalid PX4 timesync producer profile identifier")]
    InvalidProducerProfileId,
    /// Preserved status fields no longer match their content commitment.
    #[error("PX4 timesync status content commitment mismatch")]
    DigestMismatch,
    /// Raw source protocol does not have a qualified raw-alignment adapter profile.
    #[error("PX4 timesync source protocol {0} is unsupported for raw alignment")]
    UnsupportedSourceProtocol(u8),
    /// Microsecond timestamp could not be represented as nanoseconds in `u64`.
    #[error("PX4 timesync field {0} overflows nanosecond representation")]
    TimestampOverflow(&'static str),
    /// Raw observed offset would place the derived PX4-local correspondence outside `u64` time.
    #[error("PX4 raw observed offset derives a local timestamp outside the supported range")]
    DerivedLocalTimestampOutOfRange,
    /// Derived generic alignment sample failed core validation.
    #[error("invalid derived clock-alignment sample: {0}")]
    Alignment(ClockAlignmentValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(name: &str) -> ClockDomainId {
        ClockDomainId::new(name).unwrap()
    }

    fn status(
        source_protocol: u8,
        remote_timestamp_us: u64,
        observed_offset_us: i64,
        estimated_offset_us: i64,
        rtt_us: u32,
    ) -> Px4TimesyncStatusEvidenceV1 {
        Px4TimesyncStatusEvidenceV1::new(
            "px4.timesync-status.fixture.v1",
            99_000,
            source_protocol,
            remote_timestamp_us,
            observed_offset_us,
            estimated_offset_us,
            rtt_us,
        )
        .unwrap()
    }

    #[test]
    fn raw_alignment_uses_remote_plus_observed_offset_not_status_publication_time() {
        let value = status(1, 1_000, 500, 450, 40);
        let sample = value
            .raw_observed_alignment_sample(clock("companion.steady"), clock("px4.hrt"))
            .unwrap();

        assert_eq!(sample.source_at.nanoseconds, 1_000_000);
        assert_eq!(sample.target_at.nanoseconds, 1_500_000);
        assert_ne!(sample.target_at.nanoseconds, 99_000_000);
        assert_eq!(sample.max_alignment_error_ns, 41_000);
        assert_eq!(
            sample.method_profile_id,
            "px4.timesync-status.mavlink.raw-observed-offset.rtt-bound.v1"
        );
    }

    #[test]
    fn negative_observed_offset_is_supported() {
        let value = status(2, 5_000, -1_000, -900, 10);
        let sample = value
            .raw_observed_alignment_sample(clock("dds.remote"), clock("px4.hrt"))
            .unwrap();

        assert_eq!(sample.source_at.nanoseconds, 5_000_000);
        assert_eq!(sample.target_at.nanoseconds, 4_000_000);
        assert_eq!(sample.max_alignment_error_ns, 11_000);
        assert_eq!(
            sample.method_profile_id,
            "px4.timesync-status.dds.raw-observed-offset.rtt-bound.v1"
        );
    }

    #[test]
    fn unknown_or_future_source_protocol_is_preserved_but_not_promoted() {
        for raw in [0, 9] {
            let value = status(raw, 1_000, 5, 5, 10);
            assert_eq!(
                value.raw_observed_alignment_sample(clock("remote"), clock("px4.hrt")),
                Err(Px4TimesyncEvidenceError::UnsupportedSourceProtocol(raw))
            );
        }
    }

    #[test]
    fn timestamp_conversion_and_offset_underflow_fail_closed() {
        let overflowing = status(1, u64::MAX, 0, 0, 1);
        assert_eq!(
            overflowing.raw_observed_alignment_sample(clock("remote"), clock("px4.hrt")),
            Err(Px4TimesyncEvidenceError::TimestampOverflow(
                "remote_timestamp_us"
            ))
        );

        let underflow = status(1, 1, -2, -2, 1);
        assert_eq!(
            underflow.raw_observed_alignment_sample(clock("remote"), clock("px4.hrt")),
            Err(Px4TimesyncEvidenceError::DerivedLocalTimestampOutOfRange)
        );
    }

    #[test]
    fn estimated_offset_is_preserved_in_identity_but_not_used_for_raw_correspondence() {
        let first = status(1, 1_000, 100, 110, 20);
        let second = status(1, 1_000, 100, 9_999, 20);

        assert_ne!(first.status_id().unwrap(), second.status_id().unwrap());
        let first_sample = first
            .raw_observed_alignment_sample(clock("remote"), clock("px4.hrt"))
            .unwrap();
        let second_sample = second
            .raw_observed_alignment_sample(clock("remote"), clock("px4.hrt"))
            .unwrap();
        assert_eq!(first_sample.source_at, second_sample.source_at);
        assert_eq!(first_sample.target_at, second_sample.target_at);
        assert_ne!(first_sample.evidence_id, second_sample.evidence_id);
    }

    #[test]
    fn content_mutation_is_detected_before_derivation() {
        let mut value = status(1, 1_000, 100, 90, 20);
        value.observed_offset_us = 101;
        assert_eq!(value.validate(), Err(Px4TimesyncEvidenceError::DigestMismatch));
        assert_eq!(
            value.raw_observed_alignment_sample(clock("remote"), clock("px4.hrt")),
            Err(Px4TimesyncEvidenceError::DigestMismatch)
        );
    }

    #[test]
    fn publication_timestamp_has_its_own_px4_clock_observation() {
        let value = status(1, 1_000, 100, 90, 20);
        let observed = value.status_observed_at(clock("px4.hrt")).unwrap();
        assert_eq!(observed.nanoseconds, 99_000_000);
        assert_eq!(observed.clock_domain.as_str(), "px4.hrt");
    }

    #[test]
    fn serde_round_trip_preserves_status_identity() {
        let value = status(2, 1_000, 100, 90, 20);
        let bytes = serde_json::to_vec(&value).unwrap();
        let restored: Px4TimesyncStatusEvidenceV1 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, value);
        assert_eq!(restored.status_id().unwrap(), value.status_id().unwrap());
    }
}
