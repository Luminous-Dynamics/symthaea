// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Core types for the Holochain browser client.
//!
//! These types mirror the Holochain conductor wire protocol at a level
//! sufficient for zome calls without depending on `holochain_types` or
//! `holochain_conductor_api` (which pull in tokio and cannot compile to
//! `wasm32-unknown-unknown`).

use serde::{Deserialize, Serialize};

use crate::error::ClientError;

// ---------------------------------------------------------------------------
// Connection status
// ---------------------------------------------------------------------------

/// Current state of the WebSocket connection to the conductor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// No connection attempt has been made, or the connection was closed.
    Disconnected,
    /// A connection attempt is in progress.
    Connecting,
    /// Successfully connected and ready for zome calls.
    Connected,
    /// The connection is in an error state.
    Error(String),
}

// ---------------------------------------------------------------------------
// Zome call request/response (our internal representation)
// ---------------------------------------------------------------------------

/// A request to call a zome function on the conductor.
///
/// The `payload` field is already MessagePack-encoded (the zome function's
/// input type). The transport layer wraps this in the conductor's wire
/// protocol envelope before sending.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZomeCallRequest {
    /// Role name from the hApp manifest (e.g. "governance", "commons").
    pub role_name: String,
    /// Zome name within the DNA (e.g. "agora", "food-production").
    pub zome_name: String,
    /// Function name exported by the zome (e.g. "create_proposal").
    pub fn_name: String,
    /// MessagePack-encoded input payload for the zome function.
    pub payload: Vec<u8>,
}

/// A response from a zome call.
///
/// The `payload` field contains the MessagePack-encoded return value from
/// the zome function. Use [`decode`] to deserialize it into the expected type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZomeCallResponse {
    /// MessagePack-encoded output from the zome function.
    pub payload: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Holochain wire protocol types
// ---------------------------------------------------------------------------

/// Envelope for requests sent to the conductor over WebSocket.
///
/// The Holochain conductor expects MessagePack-encoded requests with a
/// numeric `id` field for correlating responses, a `type` field indicating
/// the request kind, and `data` containing the actual payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireRequest {
    /// Unique request identifier for response correlation.
    pub id: u64,
    /// Request type discriminator (e.g. "zome_call").
    #[serde(rename = "type")]
    pub request_type: String,
    /// The request payload, serialized as MessagePack bytes.
    pub data: Vec<u8>,
}

/// Envelope for responses received from the conductor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireResponse {
    /// The request ID this response corresponds to.
    pub id: u64,
    /// Response type discriminator.
    #[serde(rename = "type")]
    pub response_type: String,
    /// The response payload (MessagePack-encoded).
    #[serde(default)]
    pub data: Vec<u8>,
    /// Error message, if the call failed.
    #[serde(default)]
    pub error: Option<String>,
}

/// The inner data for a zome call request, matching the conductor's
/// expected `AppRequest::CallZome` structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ZomeCallWireData {
    /// Provenance (agent public key) — 32 bytes, all zeros = unsigned.
    pub provenance: Vec<u8>,
    /// The cell ID target, encoded as a role name.
    pub role_name: String,
    /// Zome name.
    pub zome_name: String,
    /// Function name.
    pub fn_name: String,
    /// MessagePack-encoded function input (ExternIO-compatible).
    pub payload: Vec<u8>,
    /// Capabilities token — empty for public calls.
    pub cap_secret: Option<Vec<u8>>,
    /// Nonce for replay protection.
    pub nonce: Vec<u8>,
    /// Expiration timestamp (microseconds since epoch).
    pub expires_at: u64,
}

// ---------------------------------------------------------------------------
// Convenience encode/decode
// ---------------------------------------------------------------------------

/// Encode a value as MessagePack bytes.
///
/// This matches what Holochain's `ExternIO::encode` does internally.
///
/// # Errors
///
/// Returns [`ClientError::SerializationError`] if serialization fails.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, ClientError> {
    rmp_serde::to_vec_named(value)
        .map_err(|e| ClientError::SerializationError(e.to_string()))
}

/// Decode MessagePack bytes into a typed value.
///
/// This matches what Holochain's `ExternIO::decode` does internally.
///
/// # Errors
///
/// Returns [`ClientError::SerializationError`] if deserialization fails.
pub fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, ClientError> {
    rmp_serde::from_slice(bytes)
        .map_err(|e| ClientError::SerializationError(e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_encode_decode() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct TestPayload {
            name: String,
            value: u64,
        }

        let original = TestPayload {
            name: "test".into(),
            value: 42,
        };
        let encoded = encode(&original).unwrap();
        let decoded: TestPayload = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn encode_unit_produces_bytes() {
        let encoded = encode(&()).unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn decode_bad_bytes_errors() {
        let result = decode::<String>(&[0xFF, 0xFF, 0xFF]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ClientError::SerializationError(_)));
    }

    #[test]
    fn connection_status_equality() {
        assert_eq!(ConnectionStatus::Connected, ConnectionStatus::Connected);
        assert_ne!(ConnectionStatus::Connected, ConnectionStatus::Disconnected);
        assert_eq!(
            ConnectionStatus::Error("x".into()),
            ConnectionStatus::Error("x".into())
        );
    }
}
