// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Error types for the mycelix-fl crate.

use thiserror::Error;

/// Errors that can occur during federated learning operations.
#[derive(Debug, Error)]
pub enum FlError {
    /// No gradients were provided for aggregation.
    #[error("no gradients provided for aggregation")]
    EmptyGradients,

    /// Not enough gradients to perform aggregation.
    #[error("insufficient gradients: got {got}, need {need}")]
    InsufficientGradients {
        /// Number of gradients provided.
        got: usize,
        /// Minimum number required.
        need: usize,
    },

    /// Gradient dimension mismatch between nodes.
    #[error("gradient dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension (from the first gradient).
        expected: usize,
        /// Actual dimension found.
        got: usize,
    },

    /// A gradient vector is empty.
    #[error("empty gradient vector")]
    EmptyGradient,

    /// Input validation failed.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// An algorithm parameter is invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Defense has not been calibrated before use.
    #[error("defense not calibrated: {0}")]
    NotCalibrated(String),

    /// Numerical computation error (e.g., division by zero, NaN).
    #[error("numerical error: {0}")]
    NumericalError(String),

    /// Node is not registered with the coordinator.
    #[error("node not registered: {0}")]
    NodeNotRegistered(String),

    /// Node is blacklisted.
    #[error("node blacklisted: {0}")]
    NodeBlacklisted(String),

    /// Maximum node capacity reached.
    #[error("max nodes reached: capacity {0}")]
    MaxNodesReached(usize),

    /// Rate limit exceeded for a node.
    #[error("rate limit exceeded for node {0}")]
    RateLimitExceeded(String),

    /// No active round to submit to.
    #[error("no active round")]
    NoActiveRound,

    /// Duplicate submission from a node in the same round.
    #[error("duplicate submission from node {0} in round {1}")]
    DuplicateSubmission(String, u64),

    /// Round is not in the expected state for the requested operation.
    #[error("invalid round state: expected {expected}, got {got}")]
    InvalidRoundState {
        /// Expected state.
        expected: String,
        /// Actual state.
        got: String,
    },

    /// Gradient contains non-finite values (NaN or infinity).
    #[error("gradient contains non-finite values from node {0}")]
    NonFiniteGradient(String),
}
