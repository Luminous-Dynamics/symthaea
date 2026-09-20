// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit temporal semantics for governed Nixward evidence.
//!
//! The purpose of these types is to prevent naked `u64` timestamps with implicit
//! units from crossing realization/authorization boundaries, and to preserve the
//! difference between current, future-dated, expired, static, and unknown evidence.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnixMillisV1(u64);

impl UnixMillisV1 {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }

    pub fn try_into_exact_seconds(self) -> Result<UnixSecondsV1, NixTimeErrorV1> {
        if self.0 % 1_000 != 0 {
            return Err(NixTimeErrorV1::NotExactSeconds);
        }
        Ok(UnixSecondsV1(self.0 / 1_000))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnixSecondsV1(u64);

impl UnixSecondsV1 {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }

    pub fn try_into_millis(self) -> Result<UnixMillisV1, NixTimeErrorV1> {
        self.0
            .checked_mul(1_000)
            .map(UnixMillisV1)
            .ok_or(NixTimeErrorV1::ConversionOverflow)
    }
}

/// Inclusive validity window for observed evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvidenceWindowMillisV1 {
    observed_at: UnixMillisV1,
    valid_until: UnixMillisV1,
}

impl EvidenceWindowMillisV1 {
    pub fn new(
        observed_at: UnixMillisV1,
        valid_until: UnixMillisV1,
    ) -> Result<Self, NixTimeErrorV1> {
        if valid_until < observed_at {
            return Err(NixTimeErrorV1::InvalidWindow);
        }
        Ok(Self {
            observed_at,
            valid_until,
        })
    }

    pub const fn observed_at(self) -> UnixMillisV1 {
        self.observed_at
    }

    pub const fn valid_until(self) -> UnixMillisV1 {
        self.valid_until
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceCurrentnessV1 {
    Static,
    Observed(EvidenceWindowMillisV1),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceTemporalStatusV1 {
    Current,
    NotYetValid,
    Expired,
    Unknown,
}

/// Replayable result of evaluating one currentness declaration at one exact wall-clock time.
///
/// Historical `Current` means current *at `evaluated_at`*. It is not permission to
/// reinterpret the same evidence as current at a later time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvidenceTemporalEvaluationV1 {
    currentness: EvidenceCurrentnessV1,
    evaluated_at: UnixMillisV1,
    status: EvidenceTemporalStatusV1,
}

impl EvidenceTemporalEvaluationV1 {
    pub fn evaluate(currentness: EvidenceCurrentnessV1, evaluated_at: UnixMillisV1) -> Self {
        let status = match currentness {
            EvidenceCurrentnessV1::Static => EvidenceTemporalStatusV1::Current,
            EvidenceCurrentnessV1::Unknown => EvidenceTemporalStatusV1::Unknown,
            EvidenceCurrentnessV1::Observed(window) => {
                if evaluated_at < window.observed_at() {
                    EvidenceTemporalStatusV1::NotYetValid
                } else if evaluated_at > window.valid_until() {
                    EvidenceTemporalStatusV1::Expired
                } else {
                    EvidenceTemporalStatusV1::Current
                }
            }
        };
        Self {
            currentness,
            evaluated_at,
            status,
        }
    }

    pub const fn currentness(self) -> EvidenceCurrentnessV1 {
        self.currentness
    }

    pub const fn evaluated_at(self) -> UnixMillisV1 {
        self.evaluated_at
    }

    pub const fn status(self) -> EvidenceTemporalStatusV1 {
        self.status
    }

    pub const fn is_current(self) -> bool {
        matches!(self.status, EvidenceTemporalStatusV1::Current)
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NixTimeErrorV1 {
    #[error("evidence validity window ends before observation time")]
    InvalidWindow,
    #[error("seconds-to-milliseconds conversion overflow")]
    ConversionOverflow,
    #[error("millisecond timestamp is not an exact whole-second value")]
    NotExactSeconds,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed() -> EvidenceCurrentnessV1 {
        EvidenceCurrentnessV1::Observed(
            EvidenceWindowMillisV1::new(UnixMillisV1::new(100_000), UnixMillisV1::new(200_000))
                .unwrap(),
        )
    }

    #[test]
    fn observed_window_boundaries_are_inclusive() {
        for now in [100_000, 150_000, 200_000] {
            let result = EvidenceTemporalEvaluationV1::evaluate(observed(), UnixMillisV1::new(now));
            assert_eq!(result.status(), EvidenceTemporalStatusV1::Current);
            assert!(result.is_current());
            assert_eq!(result.evaluated_at(), UnixMillisV1::new(now));
        }
    }

    #[test]
    fn future_dated_and_expired_are_distinct() {
        let future = EvidenceTemporalEvaluationV1::evaluate(observed(), UnixMillisV1::new(99_999));
        assert_eq!(future.status(), EvidenceTemporalStatusV1::NotYetValid);
        assert!(!future.is_current());

        let expired = EvidenceTemporalEvaluationV1::evaluate(observed(), UnixMillisV1::new(200_001));
        assert_eq!(expired.status(), EvidenceTemporalStatusV1::Expired);
        assert!(!expired.is_current());
    }

    #[test]
    fn static_and_unknown_are_explicit() {
        let static_result = EvidenceTemporalEvaluationV1::evaluate(
            EvidenceCurrentnessV1::Static,
            UnixMillisV1::new(999_000),
        );
        assert_eq!(static_result.status(), EvidenceTemporalStatusV1::Current);

        let unknown = EvidenceTemporalEvaluationV1::evaluate(
            EvidenceCurrentnessV1::Unknown,
            UnixMillisV1::new(999_000),
        );
        assert_eq!(unknown.status(), EvidenceTemporalStatusV1::Unknown);
        assert!(!unknown.is_current());
    }

    #[test]
    fn invalid_window_is_rejected() {
        assert_eq!(
            EvidenceWindowMillisV1::new(UnixMillisV1::new(2), UnixMillisV1::new(1)).unwrap_err(),
            NixTimeErrorV1::InvalidWindow
        );
    }

    #[test]
    fn seconds_to_millis_is_checked_and_unit_explicit() {
        assert_eq!(
            UnixSecondsV1::new(1_000).try_into_millis().unwrap(),
            UnixMillisV1::new(1_000_000)
        );
        assert_eq!(
            UnixSecondsV1::new(u64::MAX).try_into_millis().unwrap_err(),
            NixTimeErrorV1::ConversionOverflow
        );
    }

    #[test]
    fn millis_to_seconds_requires_exact_boundary() {
        assert_eq!(
            UnixMillisV1::new(1_000_000)
                .try_into_exact_seconds()
                .unwrap(),
            UnixSecondsV1::new(1_000)
        );
        assert_eq!(
            UnixMillisV1::new(1_001)
                .try_into_exact_seconds()
                .unwrap_err(),
            NixTimeErrorV1::NotExactSeconds
        );
    }

    #[test]
    fn historical_current_result_retains_original_evaluation_time() {
        let result = EvidenceTemporalEvaluationV1::evaluate(observed(), UnixMillisV1::new(150_000));
        assert_eq!(result.status(), EvidenceTemporalStatusV1::Current);
        assert_eq!(result.evaluated_at(), UnixMillisV1::new(150_000));

        let later = EvidenceTemporalEvaluationV1::evaluate(result.currentness(), UnixMillisV1::new(250_000));
        assert_eq!(later.status(), EvidenceTemporalStatusV1::Expired);
        assert_eq!(result.status(), EvidenceTemporalStatusV1::Current);
    }
}
