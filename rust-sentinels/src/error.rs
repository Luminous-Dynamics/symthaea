// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error types for the Sentinels library

use thiserror::Error;

/// Errors that can occur during consciousness analysis
#[derive(Error, Debug)]
pub enum SentinelError {
    /// Not enough data samples for analysis
    #[error("Insufficient data: required {required} samples, got {provided}")]
    InsufficientData { required: usize, provided: usize },

    /// Invalid sample rate
    #[error("Invalid sample rate: {0} Hz")]
    InvalidSampleRate(f32),

    /// Signal processing error
    #[error("Signal processing error: {0}")]
    SignalProcessing(String),

    /// Invalid channel configuration
    #[error("Invalid channel configuration: {0}")]
    InvalidChannels(String),

    /// Numerical computation error
    #[error("Numerical error: {0}")]
    Numerical(String),
}

/// Result type alias for Sentinel operations
pub type SentinelResult<T> = Result<T, SentinelError>;