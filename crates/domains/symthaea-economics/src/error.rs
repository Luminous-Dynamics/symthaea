// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared validation and numerical error types.

use core::fmt;

/// Errors returned when an economic calculation is undefined, invalid, or
/// numerically unsafe.
#[derive(Debug, Clone, PartialEq)]
pub enum EconomicsError {
    /// A required collection contained no observations.
    EmptyInput { context: &'static str },
    /// An input was `NaN` or infinite.
    NonFiniteInput { context: &'static str },
    /// A periodic rate was outside the domain `rate > -1`.
    InvalidRate { context: &'static str, rate: f64 },
    /// A model parameter violated a documented constraint.
    InvalidParameter { context: &'static str },
    /// The requested economic solution does not exist in the supported domain.
    NoSolution { context: &'static str },
    /// More than one economically relevant solution exists.
    AmbiguousSolution { context: &'static str },
    /// Floating-point evaluation overflowed or failed to converge.
    NumericalFailure { context: &'static str },
}

impl fmt::Display for EconomicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { context } => write!(f, "empty input: {context}"),
            Self::NonFiniteInput { context } => write!(f, "non-finite input: {context}"),
            Self::InvalidRate { context, rate } => {
                write!(f, "invalid periodic rate {rate}: {context}")
            }
            Self::InvalidParameter { context } => write!(f, "invalid parameter: {context}"),
            Self::NoSolution { context } => write!(f, "no solution: {context}"),
            Self::AmbiguousSolution { context } => write!(f, "ambiguous solution: {context}"),
            Self::NumericalFailure { context } => write!(f, "numerical failure: {context}"),
        }
    }
}

impl std::error::Error for EconomicsError {}

/// Result type used throughout this crate.
pub type Result<T> = core::result::Result<T, EconomicsError>;

pub(crate) fn ensure_finite(value: f64, context: &'static str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EconomicsError::NonFiniteInput { context })
    }
}

pub(crate) fn ensure_rate(rate: f64, context: &'static str) -> Result<()> {
    ensure_finite(rate, context)?;
    if rate > -1.0 {
        Ok(())
    } else {
        Err(EconomicsError::InvalidRate { context, rate })
    }
}

pub(crate) fn ensure_slice_finite(values: &[f64], context: &'static str) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(EconomicsError::NonFiniteInput { context })
    }
}
