// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bound cross-clock correspondence samples for Embodiment Contract v2.
//!
//! A single synchronization observation can establish a bounded correspondence
//! between two clock readings. It cannot, by itself, establish a general clock
//! transform: clocks may drift, slew, jump, reset, or lose synchronization away
//! from the sampled point. This module therefore records one directed alignment
//! sample and deliberately exposes no arbitrary timestamp-mapping API.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::embodiment_evidence::{ClockDomainId, EvidenceValidationError, TimestampV1};

/// Schema version for [`ClockAlignmentSampleV1`].
pub const CLOCK_ALIGNMENT_SAMPLE_SCHEMA_V1: u16 = 1;
const CLOCK_ALIGNMENT_SAMPLE_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea.embodiment.clock-alignment-sample.v1\0";

/// One directed, bounded correspondence observation between two distinct clocks.
///
/// `source_at` and `target_at` are stated to correspond within
/// `max_alignment_error_ns` under `method_profile_id`, as supported by
/// `evidence_id`. The sample is content-addressed but not authenticated merely by
/// its digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockAlignmentSampleV1 {
    /// Schema version. Must equal [`CLOCK_ALIGNMENT_SAMPLE_SCHEMA_V1`].
    pub schema_version: u16,
    /// Source-clock reading participating in the correspondence.
    pub source_at: TimestampV1,
    /// Target-clock reading participating in the correspondence.
    pub target_at: TimestampV1,
    /// Profile describing how this correspondence was established.
    pub method_profile_id: String,
    /// Maximum stated correspondence/alignment error in nanoseconds.
    ///
    /// Zero is allowed for a separately justified exact relation, such as an
    /// instrumented simulator pair known to share an exact time transform.
    pub max_alignment_error_ns: u64,
    /// External evidence reference supporting the alignment observation.
    pub evidence_id: String,
    /// Domain-separated content commitment over every semantic field above.
    pub sample_digest_hex: String,
}

impl ClockAlignmentSampleV1 {
    /// Construct, commit, and validate one directed clock-alignment sample.
    pub fn new(
        source_at: TimestampV1,
        target_at: TimestampV1,
        method_profile_id: impl Into<String>,
        max_alignment_error_ns: u64,
        evidence_id: impl Into<String>,
    ) -> Result<Self, ClockAlignmentValidationError> {
        let mut sample = Self {
            schema_version: CLOCK_ALIGNMENT_SAMPLE_SCHEMA_V1,
            source_at,
            target_at,
            method_profile_id: method_profile_id.into(),
            max_alignment_error_ns,
            evidence_id: evidence_id.into(),
            sample_digest_hex: String::new(),
        };
        sample.validate_without_digest()?;
        sample.sample_digest_hex = sample.compute_digest_hex();
        sample.validate()?;
        Ok(sample)
    }

    /// Validate structure plus the content commitment.
    pub fn validate(&self) -> Result<(), ClockAlignmentValidationError> {
        self.validate_without_digest()?;
        if self.sample_digest_hex != self.compute_digest_hex() {
            return Err(ClockAlignmentValidationError::DigestMismatch);
        }
        Ok(())
    }

    /// Content-addressed identity for this exact alignment observation.
    ///
    /// This identity detects mutation but is not a signature/authentication claim.
    pub fn alignment_id(&self) -> Result<String, ClockAlignmentValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.clock-alignment-sample.v1:{}",
            self.sample_digest_hex
        ))
    }

    /// Source clock domain for this directed correspondence.
    pub fn source_clock_domain(&self) -> &ClockDomainId {
        &self.source_at.clock_domain
    }

    /// Target clock domain for this directed correspondence.
    pub fn target_clock_domain(&self) -> &ClockDomainId {
        &self.target_at.clock_domain
    }

    /// Whether another sample has the same directed source/target clock pair.
    ///
    /// This does not claim the two samples are mutually consistent or sufficient
    /// to fit a clock model; it only compares the directed clock identities.
    pub fn same_directed_clock_pair(&self, other: &Self) -> bool {
        self.source_at.clock_domain == other.source_at.clock_domain
            && self.target_at.clock_domain == other.target_at.clock_domain
    }

    fn validate_without_digest(&self) -> Result<(), ClockAlignmentValidationError> {
        if self.schema_version != CLOCK_ALIGNMENT_SAMPLE_SCHEMA_V1 {
            return Err(ClockAlignmentValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }

        self.source_at
            .validate()
            .map_err(ClockAlignmentValidationError::Timestamp)?;
        self.target_at
            .validate()
            .map_err(ClockAlignmentValidationError::Timestamp)?;

        if self.source_at.clock_domain == self.target_at.clock_domain {
            return Err(ClockAlignmentValidationError::SameClockDomain);
        }

        validate_identifier(&self.method_profile_id, "method_profile_id")?;
        validate_identifier(&self.evidence_id, "evidence_id")?;
        Ok(())
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(CLOCK_ALIGNMENT_SAMPLE_COMMITMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, self.source_at.clock_domain.as_str());
        hasher.update(&self.source_at.nanoseconds.to_le_bytes());
        feed_str(&mut hasher, self.target_at.clock_domain.as_str());
        hasher.update(&self.target_at.nanoseconds.to_le_bytes());
        feed_str(&mut hasher, &self.method_profile_id);
        hasher.update(&self.max_alignment_error_ns.to_le_bytes());
        feed_str(&mut hasher, &self.evidence_id);
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), ClockAlignmentValidationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.chars().any(char::is_control)
    {
        return Err(ClockAlignmentValidationError::InvalidIdentifier(field));
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

/// Validation failure for [`ClockAlignmentSampleV1`].
#[derive(Debug, Clone, PartialEq)]
pub enum ClockAlignmentValidationError {
    /// Sample uses an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Source or target timestamp/clock identity is invalid.
    Timestamp(EvidenceValidationError),
    /// Source and target use the same clock domain; no cross-clock relation exists.
    SameClockDomain,
    /// Required identifier is empty, padded, or contains control characters.
    InvalidIdentifier(&'static str),
    /// Content commitment does not match the sample fields.
    DigestMismatch,
}

impl std::fmt::Display for ClockAlignmentValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported clock-alignment sample schema version {found}")
            }
            Self::Timestamp(error) => write!(f, "invalid clock-alignment timestamp: {error}"),
            Self::SameClockDomain => write!(f, "clock-alignment sample requires two distinct clock domains"),
            Self::InvalidIdentifier(field) => write!(f, "invalid {field} identifier"),
            Self::DigestMismatch => write!(f, "clock-alignment sample content commitment mismatch"),
        }
    }
}

impl std::error::Error for ClockAlignmentValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(clock: &str, nanoseconds: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new(clock).unwrap(), nanoseconds)
    }

    #[test]
    fn valid_exact_cross_clock_sample_is_allowed() {
        let sample = ClockAlignmentSampleV1::new(
            ts("host.monotonic", 1_000),
            ts("sim.model", 5_000),
            "simulator.exact-clock-relation.v1",
            0,
            "evidence:sim-clock-binding:01",
        )
        .unwrap();

        assert_eq!(sample.max_alignment_error_ns, 0);
        assert!(sample
            .alignment_id()
            .unwrap()
            .starts_with("symthaea.embodiment.clock-alignment-sample.v1:"));
    }

    #[test]
    fn same_clock_domain_is_rejected() {
        let error = ClockAlignmentSampleV1::new(
            ts("host.monotonic", 1),
            ts("host.monotonic", 2),
            "test.method.v1",
            10,
            "evidence:1",
        )
        .unwrap_err();
        assert_eq!(error, ClockAlignmentValidationError::SameClockDomain);
    }

    #[test]
    fn malformed_method_or_evidence_id_is_rejected() {
        assert_eq!(
            ClockAlignmentSampleV1::new(
                ts("host", 1),
                ts("device", 2),
                "  ",
                1,
                "evidence:1",
            ),
            Err(ClockAlignmentValidationError::InvalidIdentifier(
                "method_profile_id"
            ))
        );
        assert_eq!(
            ClockAlignmentSampleV1::new(
                ts("host", 1),
                ts("device", 2),
                "test.method.v1",
                1,
                "bad\nvalue",
            ),
            Err(ClockAlignmentValidationError::InvalidIdentifier("evidence_id"))
        );
    }

    #[test]
    fn direction_participates_in_identity() {
        let forward = ClockAlignmentSampleV1::new(
            ts("host", 100),
            ts("device", 900),
            "test.method.v1",
            50,
            "evidence:pair:1",
        )
        .unwrap();
        let reverse = ClockAlignmentSampleV1::new(
            ts("device", 900),
            ts("host", 100),
            "test.method.v1",
            50,
            "evidence:pair:1",
        )
        .unwrap();

        assert_ne!(forward.sample_digest_hex, reverse.sample_digest_hex);
        assert!(!forward.same_directed_clock_pair(&reverse));
    }

    #[test]
    fn semantically_valid_mutation_breaks_commitment() {
        let mut sample = ClockAlignmentSampleV1::new(
            ts("host", 100),
            ts("device", 900),
            "mavlink.timesync.v1",
            100_000,
            "evidence:timesync:1",
        )
        .unwrap();

        sample.target_at.nanoseconds = 901;
        assert_eq!(sample.validate(), Err(ClockAlignmentValidationError::DigestMismatch));
        assert_eq!(
            sample.alignment_id(),
            Err(ClockAlignmentValidationError::DigestMismatch)
        );
    }

    #[test]
    fn serde_round_trip_preserves_identity() {
        let sample = ClockAlignmentSampleV1::new(
            ts("host", 10),
            ts("device", 20),
            "ptp.sample.v1",
            500,
            "evidence:ptp:1",
        )
        .unwrap();
        let bytes = serde_json::to_vec(&sample).unwrap();
        let restored: ClockAlignmentSampleV1 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, sample);
        restored.validate().unwrap();
    }

    #[test]
    fn same_directed_pair_does_not_claim_model_consistency() {
        let first = ClockAlignmentSampleV1::new(
            ts("host", 100),
            ts("device", 200),
            "test.method.v1",
            10,
            "evidence:1",
        )
        .unwrap();
        let second = ClockAlignmentSampleV1::new(
            ts("host", 1_000),
            ts("device", 9_999),
            "another.method.v1",
            1_000,
            "evidence:2",
        )
        .unwrap();

        assert!(first.same_directed_clock_pair(&second));
        assert_ne!(first.sample_digest_hex, second.sample_digest_hex);
    }
}
