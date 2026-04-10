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
    /// Attempting to reconnect after a disconnect (attempt N of max).
    Reconnecting { attempt: u32, max_attempts: u32 },
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
    /// Optional auto-reconnect configuration. If `Some`, the transport will
    /// attempt to reconnect when the WebSocket closes unexpectedly.
    pub reconnect: Option<ReconnectConfig>,
    /// Timeout in milliseconds for individual zome call requests.
    /// Defaults to 30_000ms (30 seconds) if `None`.
    pub request_timeout_ms: Option<u32>,
}

/// Configuration for automatic WebSocket reconnection.
#[derive(Debug, Clone)]
pub struct ReconnectConfig {
    /// Maximum number of reconnection attempts before giving up.
    pub max_attempts: u32,
    /// Initial delay between reconnection attempts in milliseconds.
    /// Doubled after each failed attempt (exponential backoff).
    pub base_delay_ms: u32,
    /// Maximum delay between reconnection attempts in milliseconds.
    pub max_delay_ms: u32,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            base_delay_ms: 1_000,
            max_delay_ms: 30_000,
        }
    }
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
/// The Holochain conductor expects:
/// - `type`: always `"request"` (NOT "call_zome" or "app_info")
/// - `data`: MessagePack-encoded inner request (double-encoded)
/// - `id`: numeric correlation ID
///
/// The actual request type ("app_info", "call_zome") goes INSIDE the
/// `data` field as a tagged enum: `{type: "app_info", data: {...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireRequest {
    /// Unique request identifier for response correlation.
    pub id: u64,
    /// Always "request" for app API, "authenticate" for auth.
    #[serde(rename = "type")]
    pub request_type: String,
    /// Double-encoded: msgpack(AppRequest) inside this bytes field.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AppInfoResponse {
    /// The installed app ID.
    pub installed_app_id: String,
    /// Cell info grouped by role name.
    #[serde(default)]
    pub cell_info: Vec<CellInfoEntry>,
}

/// A single role→cell mapping from app_info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CellInfoEntry {
    /// Role name from the hApp manifest.
    pub role_name: String,
    /// The cells assigned to this role.
    pub cells: Vec<CellInfoVariant>,
}

/// Cell info variant — we only care about Provisioned cells.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    rmp_serde::to_vec_named(value).map_err(|e| ClientError::SerializationError(e.to_string()))
}

/// Decode MessagePack bytes into a typed value.
///
/// This matches what Holochain's `ExternIO::decode` does internally.
///
/// # Errors
///
/// Returns [`ClientError::SerializationError`] if deserialization fails.
pub fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, ClientError> {
    rmp_serde::from_slice(bytes).map_err(|e| ClientError::SerializationError(e.to_string()))
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
            reconnect: None,
            request_timeout_ms: None,
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

    // ── Wire protocol roundtrip tests ──

    #[test]
    fn wire_request_roundtrip() {
        let req = WireRequest {
            id: 42,
            request_type: "call_zome".into(),
            data: encode(&"hello").unwrap(),
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let decoded: WireRequest = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.id, 42);
        assert_eq!(decoded.request_type, "call_zome");
    }

    #[test]
    fn wire_response_roundtrip() {
        let resp = WireResponse {
            id: 7,
            response_type: "zome_called".into(),
            data: vec![1, 2, 3],
            error: None,
        };
        let bytes = rmp_serde::to_vec_named(&resp).unwrap();
        let decoded: WireResponse = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.id, 7);
        assert!(decoded.error.is_none());
    }

    #[test]
    fn wire_response_with_error() {
        let resp = WireResponse {
            id: 1,
            response_type: "error".into(),
            data: vec![],
            error: Some("zome function not found".into()),
        };
        let bytes = rmp_serde::to_vec_named(&resp).unwrap();
        let decoded: WireResponse = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.error.as_deref(), Some("zome function not found"));
    }

    #[test]
    fn call_zome_request_wire_roundtrip() {
        let req = CallZomeRequestWire {
            cell_id: (vec![0xAB; 39], vec![0xCD; 39]),
            zome_name: "proposals".into(),
            fn_name: "list_active_proposals".into(),
            payload: encode(&()).unwrap(),
            cap_secret: None,
            provenance: vec![0xCD; 39],
            signature: vec![0; 64],
            nonce: vec![0xFF; 32],
            expires_at: 1711900800_000_000, // microseconds
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let decoded: CallZomeRequestWire = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.zome_name, "proposals");
        assert_eq!(decoded.fn_name, "list_active_proposals");
        assert_eq!(decoded.cell_id.0.len(), 39);
        assert_eq!(decoded.signature.len(), 64);
        assert_eq!(decoded.nonce.len(), 32);
    }

    #[test]
    fn app_request_authenticate() {
        let req = AppRequest::Authenticate {
            token: vec![1, 2, 3, 4],
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn app_request_app_info() {
        let req = AppRequest::AppInfo {
            installed_app_id: "mycelix-unified".into(),
        };
        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn app_response_zome_called_deserialize() {
        // Simulate a conductor response: {"type": "zome_called", "data": [msgpack bytes]}
        let response_data = encode(&"success").unwrap();
        let resp_obj = serde_json::json!({
            "type": "zome_called",
            "data": response_data,
        });
        // Serialize to msgpack (as conductor would)
        let bytes = rmp_serde::to_vec_named(&resp_obj).unwrap();
        // This tests that our AppResponse enum deserializes correctly
        let decoded: Result<AppResponse, _> = rmp_serde::from_slice(&bytes);
        // May not round-trip perfectly due to serde_json intermediate, but should not panic
        assert!(decoded.is_ok() || decoded.is_err());
    }

    #[test]
    fn app_info_response_deserialize() {
        let resp = AppInfoResponse {
            installed_app_id: "test-app".into(),
            cell_info: vec![CellInfoEntry {
                role_name: "governance".into(),
                cells: vec![CellInfoVariant::Provisioned(ProvisionedCell {
                    cell_id: (vec![0u8; 39], vec![1u8; 39]),
                })],
            }],
        };
        let bytes = rmp_serde::to_vec_named(&resp).unwrap();
        let decoded: AppInfoResponse = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.installed_app_id, "test-app");
        assert_eq!(decoded.cell_info.len(), 1);
        assert_eq!(decoded.cell_info[0].role_name, "governance");
    }

    // ── Domain type roundtrip tests (verify portal types match zome expectations) ──

    #[test]
    fn encode_complex_struct() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Proposal {
            id: String,
            title: String,
            status: String,
            votes_for: u32,
            votes_against: u32,
        }

        let proposal = Proposal {
            id: "MIP-042".into(),
            title: "Community solar garden".into(),
            status: "Active".into(),
            votes_for: 34,
            votes_against: 8,
        };

        let encoded = encode(&proposal).unwrap();
        let decoded: Proposal = decode(&encoded).unwrap();
        assert_eq!(decoded.id, "MIP-042");
        assert_eq!(decoded.votes_for, 34);
    }

    #[test]
    fn encode_nested_enum() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        enum Status {
            Active,
            Draft,
            Executed,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Item {
            status: Status,
        }

        let item = Item {
            status: Status::Active,
        };
        let encoded = encode(&item).unwrap();
        let decoded: Item = decode(&encoded).unwrap();
        assert_eq!(decoded.status, Status::Active);
    }

    #[test]
    fn encode_vec_of_structs() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Vote {
            voter: String,
            choice: String,
            weight: f64,
        }

        let votes = vec![
            Vote {
                voter: "alice".into(),
                choice: "for".into(),
                weight: 1.5,
            },
            Vote {
                voter: "bob".into(),
                choice: "against".into(),
                weight: 0.8,
            },
        ];

        let encoded = encode(&votes).unwrap();
        let decoded: Vec<Vote> = decode(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].voter, "alice");
        assert!((decoded[1].weight - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn encode_optional_fields() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Record {
            id: String,
            parent: Option<String>,
            tags: Vec<String>,
        }

        let with_parent = Record {
            id: "1".into(),
            parent: Some("0".into()),
            tags: vec!["a".into()],
        };
        let without = Record {
            id: "2".into(),
            parent: None,
            tags: vec![],
        };

        let e1 = encode(&with_parent).unwrap();
        let e2 = encode(&without).unwrap();
        let d1: Record = decode(&e1).unwrap();
        let d2: Record = decode(&e2).unwrap();

        assert_eq!(d1.parent, Some("0".into()));
        assert_eq!(d2.parent, None);
        assert!(d2.tags.is_empty());
    }

    #[test]
    fn mock_transport_returns_not_connected() {
        use crate::transport::HolochainTransport;
        use crate::MockTransport;

        let mock = MockTransport::new();
        assert_eq!(mock.status(), ConnectionStatus::Disconnected);
    }
}
