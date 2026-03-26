// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Error types for the Holochain conductor bridge.

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Not connected to conductor")]
    NotConnected,
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Zome call failed: {0}")]
    ZomeCallFailed(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    #[error("Conductor timeout after {0}ms")]
    Timeout(u64),
    #[error("All conductors unhealthy: {0}")]
    AllConductorsUnhealthy(String),
}

pub type Result<T> = std::result::Result<T, BridgeError>;
