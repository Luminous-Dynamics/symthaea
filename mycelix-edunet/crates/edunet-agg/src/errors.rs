// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error types for aggregation operations

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AggregationError {
    #[error("Insufficient updates: got {got}, need at least {min}")]
    InsufficientUpdates { got: usize, min: usize },

    #[error("Invalid trim percentage: {0} (must be between 0.0 and 0.5)")]
    InvalidTrimPercent(f64),

    #[error("Empty update vector")]
    EmptyUpdates,

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid weight: {0} (must be positive)")]
    InvalidWeight(f64),

    #[error("Trimming removed all values; lower trim_percent or require more updates")]
    TrimmedAwayAllValues,

    #[error("Statistical error: {0}")]
    StatisticalError(String),
}

pub type Result<T> = std::result::Result<T, AggregationError>;
