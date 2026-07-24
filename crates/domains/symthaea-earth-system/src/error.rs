// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation errors for checked climate-model APIs.

use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelError {
    NonFinite {
        parameter: &'static str,
        value: f64,
    },
    NonPositive {
        parameter: &'static str,
        value: f64,
    },
    OutOfRange {
        parameter: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    InvalidOrdering {
        lower: &'static str,
        lower_value: f64,
        upper: &'static str,
        upper_value: f64,
    },
    ZeroSteps,
    EmptySeries {
        series: &'static str,
    },
    SingularCalibration {
        reason: &'static str,
    },
    NonMonotonicTime {
        index: usize,
        previous: f64,
        current: f64,
    },
    EnsembleTooLarge {
        requested: usize,
        maximum: usize,
    },
    ScheduleTooLarge {
        requested: usize,
        maximum: usize,
    },
    TrajectoryTooLarge {
        requested: usize,
        maximum: usize,
    },
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        found: usize,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NonFinite { parameter, value } => {
                write!(f, "{parameter} must be finite, got {value}")
            }
            Self::NonPositive { parameter, value } => {
                write!(f, "{parameter} must be positive, got {value}")
            }
            Self::OutOfRange {
                parameter,
                value,
                min,
                max,
            } => write!(f, "{parameter} must be in [{min}, {max}], got {value}"),
            Self::InvalidOrdering {
                lower,
                lower_value,
                upper,
                upper_value,
            } => write!(
                f,
                "{lower} ({lower_value}) must be less than {upper} ({upper_value})"
            ),
            Self::ZeroSteps => write!(f, "integration steps must be greater than zero"),
            Self::EmptySeries { series } => write!(f, "{series} must not be empty"),
            Self::SingularCalibration { reason } => {
                write!(f, "calibration is singular: {reason}")
            }
            Self::NonMonotonicTime {
                index,
                previous,
                current,
            } => write!(
                f,
                "time must be strictly increasing at index {index}: previous={previous}, current={current}"
            ),
            Self::EnsembleTooLarge { requested, maximum } => write!(
                f,
                "ensemble requests {requested} members; maximum is {maximum}"
            ),
            Self::ScheduleTooLarge { requested, maximum } => write!(
                f,
                "integration schedule requests {requested} intervals; maximum is {maximum}"
            ),
            Self::TrajectoryTooLarge { requested, maximum } => write!(
                f,
                "trajectory requests {requested} steps; maximum is {maximum}"
            ),
            Self::DimensionMismatch {
                context,
                expected,
                found,
            } => write!(f, "{context} has length {found}; expected {expected}"),
        }
    }
}

impl std::error::Error for ModelError {}

pub(crate) fn require_finite(parameter: &'static str, value: f64) -> Result<(), ModelError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ModelError::NonFinite { parameter, value })
    }
}

pub(crate) fn require_positive(parameter: &'static str, value: f64) -> Result<(), ModelError> {
    require_finite(parameter, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ModelError::NonPositive { parameter, value })
    }
}

pub(crate) fn require_non_negative(parameter: &'static str, value: f64) -> Result<(), ModelError> {
    require_finite(parameter, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ModelError::OutOfRange {
            parameter,
            value,
            min: 0.0,
            max: f64::INFINITY,
        })
    }
}

pub(crate) fn require_fraction(parameter: &'static str, value: f64) -> Result<(), ModelError> {
    require_finite(parameter, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ModelError::OutOfRange {
            parameter,
            value,
            min: 0.0,
            max: 1.0,
        })
    }
}
