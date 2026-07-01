// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use thiserror::Error;

/// Error types for the symthaea-field-dynamics crate.
#[derive(Debug, Error)]
pub enum DynamicsError {
    #[error("Field dynamics error: {0}")]
    FieldError(String),
    #[error("Wave equation error: {0}")]
    WaveError(String),
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Convenience result type for dynamics operations.
pub type DynamicsResult<T> = Result<T, DynamicsError>;
