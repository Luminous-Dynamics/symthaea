// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::fmt;

/// Errors that can occur during integration measurement.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OracleError {
    /// Observation dimension doesn't match encoder's expected dimension.
    DimensionMismatch { expected: usize, got: usize },
    /// Not enough observations have been accumulated to compute a result.
    InsufficientData { have: usize, need: usize },
    /// Covariance matrix has wrong size (expected n*n elements).
    InvalidCovariance { expected: usize, got: usize },
    /// Configuration validation failed.
    InvalidConfig(String),
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InsufficientData { have, need } => {
                write!(
                    f,
                    "insufficient data: have {have} observations, need {need}"
                )
            }
            Self::InvalidCovariance { expected, got } => {
                write!(
                    f,
                    "invalid covariance matrix: expected {expected} elements, got {got}"
                )
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for OracleError {}
