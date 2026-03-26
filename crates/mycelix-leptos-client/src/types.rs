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
// Connection configuration
// ---------------------------------------------------------------------------

/// Configuration for connecting to a Holochain conductor.
#[derive(Debug, Clone)]
pub struct ConnectConfig {
    /// WebSocket URL (e.g. "ws://localhost:8888").
    pub url: String,
    /// The installed app ID to discover cell mappings for.
    pub app_id: String,
    /// Optional authentication token. If `Some`, the transport sends an
    /// `authenticate` message immediately after the WebSocket opens.
    pub auth_token: Option<Vec<u8>>,
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
    /// Request type discriminator (e.g. "call_zome").
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

// ---------------------------------------------------------------------------
// AppRequest / AppResponse — conductor wire protocol enums
// ---------------------------------------------------------------------------

/// Requests the conductor App API understands.
///
/// Serialized as externally tagged: `{"type": "call_zome", "data": {...}}`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub(crate) enum AppRequest {
    /// Authenticate with the conductor using an issued token.
    #[serde(rename = "authenticate")]
    Authenticate { token: Vec<u8> },

    /// Request app info (installed cells, role→cell_id map).
    #[serde(rename = "app_info")]
    AppInfo { installed_app_id: String },

    /// Call a zome function.
    #[serde(rename = "call_zome")]
    CallZome(CallZomeRequestWire),
}

/// The inner data for a zome call request, matching the conductor's
/// expected `AppRequest::CallZome` structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CallZomeRequestWire {
    /// Cell ID as `(DnaHash, AgentPubKey)` — each is 39 bytes (32 hash + 3 prefix + 4 loc).
    pub cell_id: (Vec<u8>, Vec<u8>),
    /// Zome name.
    pub zome_name: String,
    /// Function name.
    pub fn_name: String,
    /// MessagePack-encoded function input (ExternIO-compatible).
    pub payload: Vec<u8>,
    /// Capabilities token — None for public/author calls.
    pub cap_secret: Option<Vec<u8>>,
    /// Agent public key of the caller (provenance).
    pub provenance: Vec<u8>,
    /// Signature over the call — zeroed for unsigned calls.
    pub signature: Vec<u8>,
    /// Nonce for replay protection.
    pub nonce: Vec<u8>,
    /// Expiration timestamp (microseconds since epoch).
    pub expires_at: u64,
}

/// Responses from the conductor App API.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "data")]
pub(crate) enum AppResponse {
    /// App info listing installed cells.
    #[serde(rename = "app_info")]
    AppInfo(AppInfoResponse),

    /// Successful zome call result (ExternIO bytes).
    #[serde(rename = "zome_called")]
    ZomeCalled(Vec<u8>),

    /// Error from the conductor.
    #[serde(rename = "error")]
    Error(AppError),
}

/// App info response from the conductor.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AppInfoResponse {
    /// The installed app ID.
    pub installed_app_id: String,
    /// Cell info grouped by role name.
    #[serde(default)]
    pub cell_info: Vec<CellInfoEntry>,
}

/// A single role→cell mapping from app_info.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CellInfoEntry {
    /// Role name from the hApp manifest.
    pub role_name: String,
    /// The cells assigned to this role.
    pub cells: Vec<CellInfoVariant>,
}

/// Cell info variant — we only care about Provisioned cells.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "data")]
pub(crate) enum CellInfoVariant {
    /// A provisioned (running) cell.
    #[serde(rename = "provisioned")]
    Provisioned(ProvisionedCell),
    /// Cloned cells and stem cells — we pass through but don't use them for lookup.
    #[serde(other)]
    Other,
}

/// A provisioned cell with its cell_id.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ProvisionedCell {
    /// `(DnaHash, AgentPubKey)` as raw bytes.
    pub cell_id: (Vec<u8>, Vec<u8>),
}

/// Error payload from the conductor.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AppError {
    #[serde(default)]
    pub message: String,
}

/// A `CellId` is `(DnaHash, AgentPubKey)` as raw byte vectors.
pub(crate) type CellId = (Vec<u8>, Vec<u8>);

// ---------------------------------------------------------------------------
// Legacy alias (kept for ZomeCallWireData references in tests)
// ---------------------------------------------------------------------------

/// Legacy inner data for a zome call. Kept for backward compatibility;
/// new code should use [`CallZomeRequestWire`] via [`AppRequest::CallZome`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ZomeCallWireData {
    pub provenance: Vec<u8>,
    pub role_name: String,
    pub zome_name: String,
    pub fn_name: String,
    pub payload: Vec<u8>,
    pub cap_secret: Option<Vec<u8>>,
    pub nonce: Vec<u8>,
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

    #[test]
    fn connect_config_creation() {
        let config = ConnectConfig {
            url: "ws://localhost:8888".into(),
            app_id: "mycelix-unified".into(),
            auth_token: Some(vec![1, 2, 3]),
        };
        assert_eq!(config.url, "ws://localhost:8888");
        assert_eq!(config.app_id, "mycelix-unified");
        assert!(config.auth_token.is_some());
    }

    #[test]
    fn app_request_serialization() {
        let req = AppRequest::CallZome(CallZomeRequestWire {
            cell_id: (vec![0u8; 39], vec![0u8; 39]),
            zome_name: "test".into(),
            fn_name: "hello".into(),
            payload: vec![],
            cap_secret: None,
            provenance: vec![0u8; 39],
            signature: vec![0u8; 64],
            nonce: vec![0u8; 32],
            expires_at: 1000000,
        });
        // Should serialize without panicking
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        assert!(!bytes.is_empty());
    }
}
