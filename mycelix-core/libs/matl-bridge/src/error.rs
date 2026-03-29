// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error types for MATL bridge

use thiserror::Error;

/// Errors that can occur in MATL bridge operations
#[derive(Error, Debug, Clone)]
pub enum MatlError {
    /// Agent not found in trust registry
    #[error("Agent not found: {agent_id}")]
    AgentNotFound { agent_id: String },

    /// Trust score computation failed
    #[error("Trust computation failed: {reason}")]
    TrustComputationFailed { reason: String },

    /// PoGQ validation failed
    #[error("PoGQ validation failed for {agent_id}: {reason}")]
    PoGQValidationFailed { agent_id: String, reason: String },

    /// TCDM update failed
    #[error("TCDM update failed for {agent_id}: {reason}")]
    TCDMUpdateFailed { agent_id: String, reason: String },

    /// Entropy calculation failed
    #[error("Entropy calculation failed: {reason}")]
    EntropyCalculationFailed { reason: String },

    /// Sync operation failed
    #[error("Sync failed: {reason}")]
    SyncFailed { reason: String },

    /// Invalid gradient data
    #[error("Invalid gradient: {reason}")]
    InvalidGradient { reason: String },

    /// Differential privacy violation
    #[error("DP violation: epsilon={epsilon:.2}, required={required:.2}")]
    DPViolation { epsilon: f32, required: f32 },

    /// Connection error
    #[error("Connection error: {reason}")]
    ConnectionError { reason: String },

    /// Timeout
    #[error("Operation timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for MATL operations
pub type MatlResult<T> = Result<T, MatlError>;
