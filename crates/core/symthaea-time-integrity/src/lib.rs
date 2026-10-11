// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain-neutral temporal evidence contracts for Symthaea.
//!
//! This crate deliberately does **not** synchronize clocks. It represents what
//! an acquisition or synchronization layer claims about clock identity,
//! continuity, and timestamp uncertainty, then provides conservative comparison
//! helpers that fail closed when those claims are insufficient.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

pub const MAX_CLOCK_DOMAIN_LEN: usize = 128;
pub const MAX_CLOCK_EPOCH_LEN: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeIdentityError {
    Empty { kind: &'static str },
    TooLong {
        kind: &'static str,
        actual: usize,
        max: usize,
    },
    NonCanonical { kind: &'static str },
}

impl fmt::Display for TimeIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { kind } => write!(f, "{kind} must not be empty"),
            Self::TooLong { kind, actual, max } => {
                write!(f, "{kind} length {actual} exceeds maximum {max}")
            }
            Self::NonCanonical { kind } => write!(
                f,
                "{kind} must use lowercase ASCII letters, digits, '.', '_', '-', '/', or ':'"
            ),
        }
    }
}

impl std::error::Error for TimeIdentityError {}

fn validate_identifier(
    value: String,
    kind: &'static str,
    max_len: usize,
) -> Result<String, TimeIdentityError> {
    if value.is_empty() {
        return Err(TimeIdentityError::Empty { kind });
    }
    if value.len() > max_len {
        return Err(TimeIdentityError::TooLong {
            kind,
            actual: value.len(),
            max: max_len,
        });
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'.' | b'_' | b'-' | b'/' | b':')
    }) {
        return Err(TimeIdentityError::NonCanonical { kind });
    }
    Ok(value)
}

/// Opaque identifier for one timestamp-comparison domain.
///
/// Equal IDs mean the producer intends timestamps to be interpreted against the
/// same timebase. Equality does not prove synchronization accuracy, continuity,
/// authenticity, or agreement with wall-clock time.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClockDomainId(String);

impl ClockDomainId {
    pub fn new(value: impl Into<String>) -> Result<Self, TimeIdentityError> {
        validate_identifier(value.into(), "clock-domain ID", MAX_CLOCK_DOMAIN_LEN).map(Self)
    }

    /// Well-known domain for timestamps explicitly expressed as microseconds
    /// since the Unix epoch. This remains a producer assertion, not accuracy
    /// proof.
    pub fn unix_epoch() -> Self {
        Self("unix-epoch".into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ClockDomainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for ClockDomainId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ClockDomainId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Identity for one continuity epoch of a clock source.
///
/// Device reboot, oscillator reset, counter reinitialization, or other temporal
/// discontinuity should produce a new epoch ID. `None` means the producer did
/// not provide epoch provenance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClockEpochId(String);

impl ClockEpochId {
    pub fn new(value: impl Into<String>) -> Result<Self, TimeIdentityError> {
        validate_identifier(value.into(), "clock-epoch ID", MAX_CLOCK_EPOCH_LEN).map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ClockEpochId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for ClockEpochId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ClockEpochId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

/// Evidence about continuity of the declared clock within the attached epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContinuityStatus {
    /// No continuity monitor has established a usable claim.
    Unverified,
    /// The producer claims continuity checks passed for this sample.
    Continuous,
    /// A reset, replay, reorder, backwards jump, or other discontinuity was
    /// detected. Strict temporal comparison must fail closed.
    Broken,
}

/// Upper bound on timestamp error relative to the declared clock domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeUncertainty {
    /// No finite error bound is claimed.
    Unbounded,
    /// The producer claims the true timestamp lies within ±`max_error_us` of
    /// the reported timestamp in the declared clock domain.
    Bounded { max_error_us: u64 },
}

impl TimeUncertainty {
    pub fn bounded(max_error_us: u64) -> Self {
        Self::Bounded { max_error_us }
    }

    pub fn max_error_us(self) -> Option<u64> {
        match self {
            Self::Unbounded => None,
            Self::Bounded { max_error_us } => Some(max_error_us),
        }
    }
}

/// Evidence receipt attached to one timestamp-bearing observation.
///
/// This is a claim container, not a trust primitive. A caller must establish
/// where the receipt came from and whether the producer is authorized to make
/// the stated continuity/uncertainty claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeIntegrityReceipt {
    pub clock_domain: ClockDomainId,
    #[serde(default)]
    pub clock_epoch: Option<ClockEpochId>,
    pub continuity: ContinuityStatus,
    pub uncertainty: TimeUncertainty,
    #[serde(default)]
    pub sequence: Option<u64>,
}

impl TimeIntegrityReceipt {
    /// Construct the weakest useful receipt: declared clock identity only.
    pub fn declared(clock_domain: ClockDomainId) -> Self {
        Self {
            clock_domain,
            clock_epoch: None,
            continuity: ContinuityStatus::Unverified,
            uncertainty: TimeUncertainty::Unbounded,
            sequence: None,
        }
    }

    pub fn with_epoch(mut self, clock_epoch: ClockEpochId) -> Self {
        self.clock_epoch = Some(clock_epoch);
        self
    }

    pub fn with_continuity(mut self, continuity: ContinuityStatus) -> Self {
        self.continuity = continuity;
        self
    }

    pub fn with_uncertainty(mut self, uncertainty: TimeUncertainty) -> Self {
        self.uncertainty = uncertainty;
        self
    }

    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = Some(sequence);
        self
    }

    /// Whether this receipt can support a strict bounded-separation claim.
    pub fn supports_bounded_comparison(&self) -> bool {
        self.continuity == ContinuityStatus::Continuous
            && matches!(self.uncertainty, TimeUncertainty::Bounded { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonSide {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeComparisonError {
    ClockDomainMismatch {
        left: ClockDomainId,
        right: ClockDomainId,
    },
    ClockEpochMismatch {
        left: Option<ClockEpochId>,
        right: Option<ClockEpochId>,
    },
    ContinuityNotEstablished {
        side: ComparisonSide,
        status: ContinuityStatus,
    },
    UnboundedUncertainty {
        side: ComparisonSide,
    },
}

impl fmt::Display for TimeComparisonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClockDomainMismatch { left, right } => {
                write!(f, "clock domains are not comparable: {left} != {right}")
            }
            Self::ClockEpochMismatch { left, right } => write!(
                f,
                "clock continuity epochs are not comparable: {left:?} != {right:?}"
            ),
            Self::ContinuityNotEstablished { side, status } => write!(
                f,
                "{side:?} timestamp continuity is not established: {status:?}"
            ),
            Self::UnboundedUncertainty { side } => {
                write!(f, "{side:?} timestamp has no finite uncertainty bound")
            }
        }
    }
}

impl std::error::Error for TimeComparisonError {}

/// Conservative interval for the possible absolute separation between two
/// timestamps, after accounting for both timestamp error bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeparationWindowUs {
    pub nominal_us: u64,
    pub minimum_us: u64,
    pub maximum_us: u64,
}

fn ensure_declared_comparable(
    left: &TimeIntegrityReceipt,
    right: &TimeIntegrityReceipt,
) -> Result<(), TimeComparisonError> {
    if left.clock_domain != right.clock_domain {
        return Err(TimeComparisonError::ClockDomainMismatch {
            left: left.clock_domain.clone(),
            right: right.clock_domain.clone(),
        });
    }
    if left.clock_epoch != right.clock_epoch {
        return Err(TimeComparisonError::ClockEpochMismatch {
            left: left.clock_epoch.clone(),
            right: right.clock_epoch.clone(),
        });
    }
    Ok(())
}

/// Compare two timestamps using declared clock/epoch identity only.
///
/// This does not imply the clocks are accurate or synchronized. Use
/// [`bounded_separation_window_us`] for a claim that incorporates explicit
/// finite timestamp error bounds.
pub fn declared_separation_us(
    left_timestamp_us: u64,
    left: &TimeIntegrityReceipt,
    right_timestamp_us: u64,
    right: &TimeIntegrityReceipt,
) -> Result<u64, TimeComparisonError> {
    ensure_declared_comparable(left, right)?;
    Ok(left_timestamp_us.abs_diff(right_timestamp_us))
}

/// Compute a conservative absolute-separation interval.
///
/// Both receipts must refer to the same declared clock domain and continuity
/// epoch, claim continuous operation, and provide finite timestamp error bounds.
/// The returned interval is safe under the supplied claims:
///
/// `actual separation ∈ [minimum_us, maximum_us]`.
pub fn bounded_separation_window_us(
    left_timestamp_us: u64,
    left: &TimeIntegrityReceipt,
    right_timestamp_us: u64,
    right: &TimeIntegrityReceipt,
) -> Result<SeparationWindowUs, TimeComparisonError> {
    ensure_declared_comparable(left, right)?;

    if left.continuity != ContinuityStatus::Continuous {
        return Err(TimeComparisonError::ContinuityNotEstablished {
            side: ComparisonSide::Left,
            status: left.continuity,
        });
    }
    if right.continuity != ContinuityStatus::Continuous {
        return Err(TimeComparisonError::ContinuityNotEstablished {
            side: ComparisonSide::Right,
            status: right.continuity,
        });
    }

    let left_error = left
        .uncertainty
        .max_error_us()
        .ok_or(TimeComparisonError::UnboundedUncertainty {
            side: ComparisonSide::Left,
        })?;
    let right_error = right
        .uncertainty
        .max_error_us()
        .ok_or(TimeComparisonError::UnboundedUncertainty {
            side: ComparisonSide::Right,
        })?;

    let nominal_us = left_timestamp_us.abs_diff(right_timestamp_us);
    let combined_error_us = left_error.saturating_add(right_error);

    Ok(SeparationWindowUs {
        nominal_us,
        minimum_us: nominal_us.saturating_sub(combined_error_us),
        maximum_us: nominal_us.saturating_add(combined_error_us),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain() -> ClockDomainId {
        ClockDomainId::new("capture-rig-01/monotonic").unwrap()
    }

    fn epoch() -> ClockEpochId {
        ClockEpochId::new("boot-0007").unwrap()
    }

    fn bounded(max_error_us: u64) -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(domain())
            .with_epoch(epoch())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(max_error_us))
    }

    #[test]
    fn canonical_ids_round_trip() {
        let receipt = bounded(25).with_sequence(42);
        let json = serde_json::to_string(&receipt).unwrap();
        let decoded: TimeIntegrityReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, receipt);
    }

    #[test]
    fn invalid_wire_identity_fails_closed() {
        assert!(serde_json::from_str::<ClockDomainId>("\"Bad Clock\"").is_err());
        assert!(ClockDomainId::new("").is_err());
        assert!(ClockEpochId::new("Boot 7").is_err());
    }

    #[test]
    fn unix_epoch_is_explicit() {
        assert_eq!(ClockDomainId::unix_epoch().as_str(), "unix-epoch");
    }

    #[test]
    fn declared_comparison_rejects_mixed_domains_before_arithmetic() {
        let left = TimeIntegrityReceipt::declared(domain());
        let right = TimeIntegrityReceipt::declared(ClockDomainId::unix_epoch());
        let error = declared_separation_us(10, &left, u64::MAX, &right).unwrap_err();
        assert!(matches!(error, TimeComparisonError::ClockDomainMismatch { .. }));
    }

    #[test]
    fn declared_comparison_rejects_mixed_epochs() {
        let left = TimeIntegrityReceipt::declared(domain()).with_epoch(epoch());
        let right = TimeIntegrityReceipt::declared(domain())
            .with_epoch(ClockEpochId::new("boot-0008").unwrap());
        let error = declared_separation_us(10, &left, 20, &right).unwrap_err();
        assert!(matches!(error, TimeComparisonError::ClockEpochMismatch { .. }));
    }

    #[test]
    fn bounded_comparison_requires_continuity() {
        let left = TimeIntegrityReceipt::declared(domain())
            .with_epoch(epoch())
            .with_uncertainty(TimeUncertainty::bounded(5));
        let right = bounded(5);
        let error = bounded_separation_window_us(100, &left, 120, &right).unwrap_err();
        assert_eq!(
            error,
            TimeComparisonError::ContinuityNotEstablished {
                side: ComparisonSide::Left,
                status: ContinuityStatus::Unverified,
            }
        );
    }

    #[test]
    fn bounded_comparison_requires_finite_uncertainty() {
        let left = TimeIntegrityReceipt::declared(domain())
            .with_epoch(epoch())
            .with_continuity(ContinuityStatus::Continuous);
        let right = bounded(5);
        let error = bounded_separation_window_us(100, &left, 120, &right).unwrap_err();
        assert_eq!(
            error,
            TimeComparisonError::UnboundedUncertainty {
                side: ComparisonSide::Left,
            }
        );
    }

    #[test]
    fn bounded_window_combines_both_error_bounds() {
        let left = bounded(7);
        let right = bounded(11);
        let window = bounded_separation_window_us(1_000, &left, 1_050, &right).unwrap();
        assert_eq!(
            window,
            SeparationWindowUs {
                nominal_us: 50,
                minimum_us: 32,
                maximum_us: 68,
            }
        );
    }

    #[test]
    fn overlapping_uncertainty_can_reduce_minimum_to_zero() {
        let left = bounded(20);
        let right = bounded(20);
        let window = bounded_separation_window_us(1_000, &left, 1_010, &right).unwrap();
        assert_eq!(window.minimum_us, 0);
        assert_eq!(window.maximum_us, 50);
    }

    #[test]
    fn maximum_bound_saturates_instead_of_wrapping() {
        let left = bounded(u64::MAX);
        let right = bounded(u64::MAX);
        let window = bounded_separation_window_us(0, &left, u64::MAX, &right).unwrap();
        assert_eq!(window.maximum_us, u64::MAX);
    }
}
