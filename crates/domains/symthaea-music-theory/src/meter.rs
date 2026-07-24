// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full time signatures and beat grouping.
//!
//! The original score model stores only a beats-per-bar scalar. That is enough
//! for 4/4-style rendering but cannot distinguish 3/4 from 6/8, 7/8 grouped
//! 2+2+3 from 3+2+2, or a compound beat from a simple one. [`TimeSignature`]
//! adds those distinctions as an additive migration path.

use crate::rhythm::Duration;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Why a time-signature definition is invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeterError {
    ZeroNumerator,
    InvalidDenominator(u8),
    EmptyGrouping,
    ZeroGroup,
    GroupingSum { expected: u8, actual: u16 },
}

impl fmt::Display for MeterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroNumerator => f.write_str("time-signature numerator must be non-zero"),
            Self::InvalidDenominator(value) => write!(
                f,
                "time-signature denominator must be a non-zero power of two, got {value}"
            ),
            Self::EmptyGrouping => f.write_str("beat grouping must not be empty"),
            Self::ZeroGroup => f.write_str("beat groups must be non-zero"),
            Self::GroupingSum { expected, actual } => write!(
                f,
                "beat grouping sums to {actual}, but numerator is {expected}"
            ),
        }
    }
}

impl std::error::Error for MeterError {}

/// A written time signature with an explicit additive grouping.
///
/// `grouping` is expressed in denominator units. Examples:
///
/// - 4/4 defaults to `[1, 1, 1, 1]`
/// - 6/8 defaults to `[3, 3]`
/// - 7/8 may be authored as `[2, 2, 3]` or `[3, 2, 2]`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSignature {
    numerator: u8,
    denominator: u8,
    grouping: Vec<u8>,
}

impl TimeSignature {
    /// Construct a signature with an idiomatic default grouping.
    pub fn new(numerator: u8, denominator: u8) -> Result<Self, MeterError> {
        Self::with_grouping(
            numerator,
            denominator,
            Self::default_grouping(numerator, denominator),
        )
    }

    /// Construct a signature with an explicit additive grouping.
    pub fn with_grouping(
        numerator: u8,
        denominator: u8,
        grouping: Vec<u8>,
    ) -> Result<Self, MeterError> {
        if numerator == 0 {
            return Err(MeterError::ZeroNumerator);
        }
        if denominator == 0 || !denominator.is_power_of_two() {
            return Err(MeterError::InvalidDenominator(denominator));
        }
        if grouping.is_empty() {
            return Err(MeterError::EmptyGrouping);
        }
        if grouping.contains(&0) {
            return Err(MeterError::ZeroGroup);
        }
        let sum: u16 = grouping.iter().map(|&group| u16::from(group)).sum();
        if sum != u16::from(numerator) {
            return Err(MeterError::GroupingSum {
                expected: numerator,
                actual: sum,
            });
        }

        Ok(Self {
            numerator,
            denominator,
            grouping,
        })
    }

    /// Compatibility constructor for the original `Score::meter` convention:
    /// `beats_per_bar` quarter-note beats, equivalent to `beats_per_bar/4`.
    pub fn quarter_note_meter(beats_per_bar: u8) -> Self {
        Self::new(beats_per_bar, 4).expect("a non-zero quarter-note meter is valid")
    }

    pub const fn numerator(&self) -> u8 {
        self.numerator
    }

    pub const fn denominator(&self) -> u8 {
        self.denominator
    }

    pub fn grouping(&self) -> &[u8] {
        &self.grouping
    }

    /// Exact bar length measured in quarter-note beats.
    pub fn quarter_note_beats_per_bar(&self) -> Duration {
        Duration::new(i64::from(self.numerator) * 4, i64::from(self.denominator))
    }

    /// True for compound meters whose primary groups are dotted beats.
    pub fn is_compound(&self) -> bool {
        self.denominator >= 8
            && self.numerator >= 6
            && self.numerator % 3 == 0
            && self.grouping.iter().all(|&group| group == 3)
    }

    /// True when the written bar contains unequal additive groups.
    pub fn is_additive(&self) -> bool {
        self.grouping
            .first()
            .is_some_and(|first| self.grouping.iter().any(|group| group != first))
    }

    fn default_grouping(numerator: u8, denominator: u8) -> Vec<u8> {
        if denominator >= 8 && numerator >= 6 && numerator % 3 == 0 {
            vec![3; usize::from(numerator / 3)]
        } else {
            vec![1; usize::from(numerator)]
        }
    }
}

impl fmt::Display for TimeSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_and_compound_meters_have_distinct_grouping() {
        let three_four = TimeSignature::new(3, 4).unwrap();
        let six_eight = TimeSignature::new(6, 8).unwrap();

        assert_eq!(three_four.grouping(), &[1, 1, 1]);
        assert_eq!(six_eight.grouping(), &[3, 3]);
        assert!(!three_four.is_compound());
        assert!(six_eight.is_compound());
        assert_eq!(three_four.quarter_note_beats_per_bar(), Duration::new(3, 1));
        assert_eq!(six_eight.quarter_note_beats_per_bar(), Duration::new(3, 1));
    }

    #[test]
    fn additive_grouping_is_preserved() {
        let seven_eight = TimeSignature::with_grouping(7, 8, vec![2, 2, 3]).unwrap();
        assert_eq!(seven_eight.grouping(), &[2, 2, 3]);
        assert!(seven_eight.is_additive());
        assert_eq!(
            seven_eight.quarter_note_beats_per_bar(),
            Duration::new(7, 2)
        );
    }

    #[test]
    fn malformed_meters_fail_explicitly() {
        assert_eq!(
            TimeSignature::new(4, 3),
            Err(MeterError::InvalidDenominator(3))
        );
        assert_eq!(
            TimeSignature::with_grouping(7, 8, vec![2, 2, 2]),
            Err(MeterError::GroupingSum {
                expected: 7,
                actual: 6,
            })
        );
    }
}
