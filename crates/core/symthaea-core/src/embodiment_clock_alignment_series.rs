// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-sample cross-clock alignment evidence for Embodiment Contract v2.
//!
//! A series groups multiple [`crate::embodiment_clock_alignment::ClockAlignmentSampleV1`]
//! observations that belong to one directed clock pair, one alignment method, and one
//! caller-declared continuity segment. The series is deliberately diagnostic only:
//! it does **not** expose an arbitrary timestamp conversion API or claim that an affine
//! clock model is qualified merely because multiple samples exist.

use std::collections::HashSet;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::embodiment_clock_alignment::{
    ClockAlignmentSampleV1, ClockAlignmentValidationError,
};

/// Schema version for [`ClockAlignmentSeriesV1`].
pub const CLOCK_ALIGNMENT_SERIES_SCHEMA_V1: u16 = 1;
const CLOCK_ALIGNMENT_SERIES_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea.embodiment.clock-alignment-series.v1\0";

/// Diagnostic summary of one validated multi-sample clock-alignment series.
///
/// These values describe the observed samples only. They are not a fitted clock
/// transform, drift estimate, or synchronization guarantee. The summary is a
/// runtime view and is intentionally not part of the serialized evidence schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClockAlignmentSeriesDiagnosticsV1 {
    /// Number of validated alignment samples in the series.
    pub sample_count: u64,
    /// Source-clock span between first and last sample, in nanoseconds.
    pub source_span_ns: u64,
    /// Target-clock span between first and last sample, in nanoseconds.
    pub target_span_ns: u64,
    /// Minimum observed `target - source` offset among samples, in nanoseconds.
    pub min_observed_offset_ns: i128,
    /// Maximum observed `target - source` offset among samples, in nanoseconds.
    pub max_observed_offset_ns: i128,
    /// Largest per-sample stated correspondence error bound.
    pub max_sample_alignment_error_ns: u64,
}

/// Content-addressed collection of alignment samples from one continuity segment.
///
/// `continuity_segment_id` is caller-supplied evidence metadata identifying a region
/// in which both clock readings are intended to advance continuously. Structural
/// validation additionally requires both source and target sample timestamps to
/// advance strictly. A clock pause, reset, or backward jump therefore requires a new
/// series/segment rather than being silently fit into the same relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockAlignmentSeriesV1 {
    /// Schema version. Must equal [`CLOCK_ALIGNMENT_SERIES_SCHEMA_V1`].
    pub schema_version: u16,
    /// Identity of the declared continuous clock interval represented by this series.
    pub continuity_segment_id: String,
    /// Ordered alignment samples. At least two samples are required.
    pub samples: Vec<ClockAlignmentSampleV1>,
    /// Domain-separated content commitment over the segment and ordered sample identities.
    pub series_digest_hex: String,
}

impl ClockAlignmentSeriesV1 {
    /// Construct, commit, and validate a multi-sample alignment series.
    pub fn new(
        continuity_segment_id: impl Into<String>,
        samples: Vec<ClockAlignmentSampleV1>,
    ) -> Result<Self, ClockAlignmentSeriesValidationError> {
        let mut series = Self {
            schema_version: CLOCK_ALIGNMENT_SERIES_SCHEMA_V1,
            continuity_segment_id: continuity_segment_id.into(),
            samples,
            series_digest_hex: String::new(),
        };
        series.validate_without_digest()?;
        series.series_digest_hex = series.compute_digest_hex()?;
        series.validate()?;
        Ok(series)
    }

    /// Validate structure and content commitment.
    pub fn validate(&self) -> Result<(), ClockAlignmentSeriesValidationError> {
        self.validate_without_digest()?;
        if self.series_digest_hex != self.compute_digest_hex()? {
            return Err(ClockAlignmentSeriesValidationError::DigestMismatch);
        }
        Ok(())
    }

    /// Content-addressed identity of this exact ordered sample series.
    ///
    /// This detects mutation but is not a signature or clock-synchronization claim.
    pub fn series_id(&self) -> Result<String, ClockAlignmentSeriesValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.clock-alignment-series.v1:{}",
            self.series_digest_hex
        ))
    }

    /// Return diagnostics over the observed samples without fitting a clock model.
    pub fn diagnostics(
        &self,
    ) -> Result<ClockAlignmentSeriesDiagnosticsV1, ClockAlignmentSeriesValidationError> {
        self.validate()?;
        let first = &self.samples[0];
        let last = &self.samples[self.samples.len() - 1];

        let source_span_ns = last
            .source_at
            .nanoseconds
            .checked_sub(first.source_at.nanoseconds)
            .ok_or(ClockAlignmentSeriesValidationError::NonMonotonicSourceTime)?;
        let target_span_ns = last
            .target_at
            .nanoseconds
            .checked_sub(first.target_at.nanoseconds)
            .ok_or(ClockAlignmentSeriesValidationError::NonMonotonicTargetTime)?;

        let mut min_observed_offset_ns = i128::MAX;
        let mut max_observed_offset_ns = i128::MIN;
        let mut max_sample_alignment_error_ns = 0u64;

        for sample in &self.samples {
            let offset = i128::from(sample.target_at.nanoseconds)
                - i128::from(sample.source_at.nanoseconds);
            min_observed_offset_ns = min_observed_offset_ns.min(offset);
            max_observed_offset_ns = max_observed_offset_ns.max(offset);
            max_sample_alignment_error_ns = max_sample_alignment_error_ns
                .max(sample.max_alignment_error_ns);
        }

        Ok(ClockAlignmentSeriesDiagnosticsV1 {
            sample_count: self.samples.len() as u64,
            source_span_ns,
            target_span_ns,
            min_observed_offset_ns,
            max_observed_offset_ns,
            max_sample_alignment_error_ns,
        })
    }

    fn validate_without_digest(&self) -> Result<(), ClockAlignmentSeriesValidationError> {
        if self.schema_version != CLOCK_ALIGNMENT_SERIES_SCHEMA_V1 {
            return Err(ClockAlignmentSeriesValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        validate_identifier(&self.continuity_segment_id)?;
        if self.samples.len() < 2 {
            return Err(ClockAlignmentSeriesValidationError::InsufficientSamples {
                found: self.samples.len() as u64,
            });
        }

        let first = &self.samples[0];
        first
            .validate()
            .map_err(ClockAlignmentSeriesValidationError::Sample)?;

        let mut seen = HashSet::with_capacity(self.samples.len());
        seen.insert(first.sample_digest_hex.as_str());

        let mut previous_source = first.source_at.nanoseconds;
        let mut previous_target = first.target_at.nanoseconds;

        for sample in self.samples.iter().skip(1) {
            sample
                .validate()
                .map_err(ClockAlignmentSeriesValidationError::Sample)?;
            if !first.same_directed_clock_pair(sample) {
                return Err(ClockAlignmentSeriesValidationError::ClockPairMismatch);
            }
            if sample.method_profile_id != first.method_profile_id {
                return Err(ClockAlignmentSeriesValidationError::MethodProfileMismatch);
            }
            if !seen.insert(sample.sample_digest_hex.as_str()) {
                return Err(ClockAlignmentSeriesValidationError::DuplicateSample);
            }
            if sample.source_at.nanoseconds <= previous_source {
                return Err(ClockAlignmentSeriesValidationError::NonMonotonicSourceTime);
            }
            if sample.target_at.nanoseconds <= previous_target {
                return Err(ClockAlignmentSeriesValidationError::NonMonotonicTargetTime);
            }
            previous_source = sample.source_at.nanoseconds;
            previous_target = sample.target_at.nanoseconds;
        }

        Ok(())
    }

    fn compute_digest_hex(&self) -> Result<String, ClockAlignmentSeriesValidationError> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(CLOCK_ALIGNMENT_SERIES_COMMITMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.continuity_segment_id);
        hasher.update(&(self.samples.len() as u64).to_le_bytes());
        for sample in &self.samples {
            sample
                .validate()
                .map_err(ClockAlignmentSeriesValidationError::Sample)?;
            feed_str(&mut hasher, &sample.sample_digest_hex);
        }
        Ok(digest_hex(hasher.finalize().as_bytes()))
    }
}

fn validate_identifier(value: &str) -> Result<(), ClockAlignmentSeriesValidationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.chars().any(char::is_control)
    {
        return Err(ClockAlignmentSeriesValidationError::InvalidContinuitySegmentId);
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

/// Validation failure for [`ClockAlignmentSeriesV1`].
#[derive(Debug, Clone, PartialEq)]
pub enum ClockAlignmentSeriesValidationError {
    /// Series uses an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Continuity-segment identifier is empty, padded, or contains controls.
    InvalidContinuitySegmentId,
    /// Fewer than two samples were supplied.
    InsufficientSamples {
        /// Number of samples supplied.
        found: u64,
    },
    /// A nested sample failed validation.
    Sample(ClockAlignmentValidationError),
    /// A sample changed the directed source/target clock pair.
    ClockPairMismatch,
    /// A sample changed the alignment method/profile within one series.
    MethodProfileMismatch,
    /// The same content-addressed alignment sample appears more than once.
    DuplicateSample,
    /// Source-clock sample time did not advance strictly.
    NonMonotonicSourceTime,
    /// Target-clock sample time did not advance strictly.
    NonMonotonicTargetTime,
    /// Content commitment does not match the series fields.
    DigestMismatch,
}

impl std::fmt::Display for ClockAlignmentSeriesValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported clock-alignment series schema version {found}")
            }
            Self::InvalidContinuitySegmentId => {
                write!(f, "invalid clock-alignment continuity-segment identifier")
            }
            Self::InsufficientSamples { found } => {
                write!(f, "clock-alignment series requires at least two samples, found {found}")
            }
            Self::Sample(error) => write!(f, "invalid clock-alignment sample: {error}"),
            Self::ClockPairMismatch => write!(f, "clock-alignment series changed directed clock pair"),
            Self::MethodProfileMismatch => write!(f, "clock-alignment series changed method profile"),
            Self::DuplicateSample => write!(f, "clock-alignment series contains a duplicate sample"),
            Self::NonMonotonicSourceTime => write!(f, "source clock did not advance strictly within continuity segment"),
            Self::NonMonotonicTargetTime => write!(f, "target clock did not advance strictly within continuity segment"),
            Self::DigestMismatch => write!(f, "clock-alignment series content commitment mismatch"),
        }
    }
}

impl std::error::Error for ClockAlignmentSeriesValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_clock_alignment::ClockAlignmentSampleV1;
    use crate::embodiment_evidence::{ClockDomainId, TimestampV1};

    fn ts(clock: &str, nanoseconds: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new(clock).unwrap(), nanoseconds)
    }

    fn sample(source: u64, target: u64, evidence: &str) -> ClockAlignmentSampleV1 {
        ClockAlignmentSampleV1::new(
            ts("host.steady", source),
            ts("autopilot.boot", target),
            "mavlink.timesync.v2",
            50_000,
            evidence,
        )
        .unwrap()
    }

    #[test]
    fn valid_series_reports_diagnostics_without_fitting_transform() {
        let series = ClockAlignmentSeriesV1::new(
            "session:flight-001:timesync-segment-1",
            vec![
                sample(1_000_000, 6_000_100, "timesync:1"),
                sample(2_000_000, 7_000_110, "timesync:2"),
                sample(4_000_000, 9_000_130, "timesync:3"),
            ],
        )
        .unwrap();

        let diagnostics = series.diagnostics().unwrap();
        assert_eq!(diagnostics.sample_count, 3);
        assert_eq!(diagnostics.source_span_ns, 3_000_000);
        assert_eq!(diagnostics.target_span_ns, 3_000_030);
        assert_eq!(diagnostics.min_observed_offset_ns, 5_000_100);
        assert_eq!(diagnostics.max_observed_offset_ns, 5_000_130);
        assert_eq!(diagnostics.max_sample_alignment_error_ns, 50_000);
        assert!(series
            .series_id()
            .unwrap()
            .starts_with("symthaea.embodiment.clock-alignment-series.v1:"));
    }

    #[test]
    fn requires_at_least_two_samples() {
        assert_eq!(
            ClockAlignmentSeriesV1::new("segment:1", vec![sample(1, 10, "evidence:1")]),
            Err(ClockAlignmentSeriesValidationError::InsufficientSamples { found: 1 })
        );
    }

    #[test]
    fn rejects_changed_clock_pair_or_method() {
        let first = sample(100, 1_000, "evidence:1");
        let changed_pair = ClockAlignmentSampleV1::new(
            ts("host.steady", 200),
            ts("camera.boot", 2_000),
            "mavlink.timesync.v2",
            50,
            "evidence:2",
        )
        .unwrap();
        assert_eq!(
            ClockAlignmentSeriesV1::new("segment:1", vec![first.clone(), changed_pair]),
            Err(ClockAlignmentSeriesValidationError::ClockPairMismatch)
        );

        let changed_method = ClockAlignmentSampleV1::new(
            ts("host.steady", 200),
            ts("autopilot.boot", 2_000),
            "dds.timesync.v1",
            50,
            "evidence:3",
        )
        .unwrap();
        assert_eq!(
            ClockAlignmentSeriesV1::new("segment:1", vec![first, changed_method]),
            Err(ClockAlignmentSeriesValidationError::MethodProfileMismatch)
        );
    }

    #[test]
    fn rejects_duplicate_or_non_advancing_samples() {
        let first = sample(100, 1_000, "evidence:1");
        assert_eq!(
            ClockAlignmentSeriesV1::new("segment:1", vec![first.clone(), first.clone()]),
            Err(ClockAlignmentSeriesValidationError::DuplicateSample)
        );

        let source_back = sample(99, 2_000, "evidence:2");
        assert_eq!(
            ClockAlignmentSeriesV1::new("segment:1", vec![first.clone(), source_back]),
            Err(ClockAlignmentSeriesValidationError::NonMonotonicSourceTime)
        );

        let target_back = sample(200, 999, "evidence:3");
        assert_eq!(
            ClockAlignmentSeriesV1::new("segment:1", vec![first, target_back]),
            Err(ClockAlignmentSeriesValidationError::NonMonotonicTargetTime)
        );
    }

    #[test]
    fn ros_time_jump_requires_a_new_series_segment() {
        let first = ClockAlignmentSampleV1::new(
            ts("host.steady", 100),
            ts("ros.time", 10_000),
            "ros.clock-observation.v1",
            100,
            "ros-clock:1",
        )
        .unwrap();
        let after_jump = ClockAlignmentSampleV1::new(
            ts("host.steady", 200),
            ts("ros.time", 5_000),
            "ros.clock-observation.v1",
            100,
            "ros-clock:2",
        )
        .unwrap();

        assert_eq!(
            ClockAlignmentSeriesV1::new("ros-segment:before-jump", vec![first, after_jump]),
            Err(ClockAlignmentSeriesValidationError::NonMonotonicTargetTime)
        );
    }

    #[test]
    fn semantically_valid_sample_mutation_breaks_series_commitment() {
        let mut series = ClockAlignmentSeriesV1::new(
            "segment:1",
            vec![sample(100, 1_000, "evidence:1"), sample(200, 1_100, "evidence:2")],
        )
        .unwrap();
        series.samples[1] = sample(200, 1_101, "evidence:replacement");
        assert_eq!(
            series.validate(),
            Err(ClockAlignmentSeriesValidationError::DigestMismatch)
        );
    }

    #[test]
    fn malformed_segment_id_is_rejected() {
        assert_eq!(
            ClockAlignmentSeriesV1::new(
                " bad ",
                vec![sample(100, 1_000, "evidence:1"), sample(200, 1_100, "evidence:2")],
            ),
            Err(ClockAlignmentSeriesValidationError::InvalidContinuitySegmentId)
        );
    }
}
