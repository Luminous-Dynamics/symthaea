// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation errors for checked ecology APIs.

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
    Negative {
        parameter: &'static str,
        value: f64,
    },
    OutOfRange {
        parameter: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    ZeroSteps,
    EmptySeries {
        series: &'static str,
    },
    IntegrationDomainViolation {
        step: usize,
        component: &'static str,
        value: f64,
    },
    InsufficientSamples {
        required: usize,
        found: usize,
    },
    SingularCalibration {
        reason: &'static str,
    },
    NonMonotonicTime {
        index: usize,
        previous: f64,
        current: f64,
    },
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        found: usize,
    },
    NoConvergence {
        context: &'static str,
        iterations: usize,
    },
    TrajectoryTooLarge {
        requested: usize,
        maximum: usize,
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
            Self::Negative { parameter, value } => {
                write!(f, "{parameter} must be non-negative, got {value}")
            }
            Self::OutOfRange {
                parameter,
                value,
                min,
                max,
            } => {
                write!(f, "{parameter} must be in [{min}, {max}], got {value}")
            }
            Self::ZeroSteps => write!(f, "integration steps must be greater than zero"),
            Self::EmptySeries { series } => write!(f, "{series} must not be empty"),
            Self::IntegrationDomainViolation {
                step,
                component,
                value,
            } => write!(
                f,
                "integration left the positive finite domain at step {step}: {component}={value}"
            ),
            Self::InsufficientSamples { required, found } => {
                write!(f, "at least {required} samples are required, found {found}")
            }
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
            Self::DimensionMismatch {
                context,
                expected,
                found,
            } => write!(f, "{context} has length {found}; expected {expected}"),
            Self::NoConvergence {
                context,
                iterations,
            } => write!(f, "{context} did not converge in {iterations} iterations"),
            Self::TrajectoryTooLarge { requested, maximum } => write!(
                f,
                "trajectory requests {requested} steps; maximum is {maximum}"
            ),
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
        Err(ModelError::Negative { parameter, value })
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
