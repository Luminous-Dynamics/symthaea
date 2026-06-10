// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! High-level Holochain client wrapping a transport.
//!
//! Provides typed zome call methods with automatic MessagePack
//! serialization and deserialization.

use serde::{de::DeserializeOwned, Serialize};

use crate::error::ClientError;
use crate::transport::HolochainTransport;
use crate::types::{decode, encode, ConnectConfig, ConnectionStatus};

/// A typed Holochain client bound to a specific hApp and default role.
///
/// Wraps a [`HolochainTransport`] and provides ergonomic methods for
/// calling zome functions with automatic serde.
///
/// # Type parameter
///
/// `T` — the transport implementation (e.g. [`BrowserWsTransport`](crate::BrowserWsTransport)).
///
/// # Example
///
/// ```rust,no_run
/// # use mycelix_leptos_client::*;
/// # async fn example(transport: impl HolochainTransport) {
/// let client = HolochainClient::new(transport, "mycelix-unified", "governance");
///
/// // Call with typed input/output
/// let result: Vec<u8> = client.call_zome("agora", "get_proposals", &()).await.unwrap();
///
/// // Call a different role
/// let data: serde_json::Value = client
///     .call_zome_on_role("commons", "food-production", "list_plots", &())
///     .await
///     .unwrap();
/// # }
/// ```
pub struct HolochainClient<T: HolochainTransport> {
    transport: T,
    /// hApp identifier (for logging/diagnostics).
    app_id: String,
    /// Default role name for `call_zome` calls.
    default_role: String,
}

impl<T: HolochainTransport> HolochainClient<T> {
    /// Create a new client wrapping the given transport.
    ///
    /// # Arguments
    /// * `transport` — An already-constructed (but not necessarily connected) transport.
    /// * `app_id` — The hApp identifier (e.g. "mycelix-unified"). Used for logging.
    /// * `default_role` — The default role name for `call_zome` (can be overridden
    ///   per-call with `call_zome_on_role`).
    pub fn new(transport: T, app_id: impl Into<String>, default_role: impl Into<String>) -> Self {
        Self {
            transport,
            app_id: app_id.into(),
            default_role: default_role.into(),
        }
    }

    /// Connect the underlying transport to a conductor.
    ///
    /// Uses the client's `app_id` and an optional auth token to build
    /// the [`ConnectConfig`]. After connecting, the transport will have
    /// the role->cell_id mapping populated.
    ///
    /// # Arguments
    /// * `url` — WebSocket URL (e.g. "ws://localhost:8888")
    /// * `auth_token` — Optional authentication token from the admin API.
    pub async fn connect(&self, url: &str, auth_token: Option<Vec<u8>>) -> Result<(), ClientError> {
        let config = ConnectConfig {
            url: url.to_string(),
            app_id: self.app_id.clone(),
            auth_token,
            reconnect: None,
            request_timeout_ms: None,
        };
        self.transport.connect(config).await
    }

    /// Disconnect from the conductor.
    pub fn disconnect(&self) {
        self.transport.disconnect();
    }

    /// Current connection status.
    pub fn status(&self) -> ConnectionStatus {
        self.transport.status()
    }

    /// The hApp identifier this client was created with.
    pub fn app_id(&self) -> &str {
        &self.app_id
    }

    /// The default role name used by `call_zome`.
    pub fn default_role(&self) -> &str {
        &self.default_role
    }

    /// Get a reference to the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Call a zome function on the default role with typed input/output.
    ///
    /// The input is MessagePack-encoded before sending, and the response
    /// is MessagePack-decoded into the output type.
    ///
    /// # Type parameters
    /// * `I` — Input type (must implement `Serialize`)
    /// * `O` — Output type (must implement `DeserializeOwned`)
    ///
    /// # Arguments
    /// * `zome_name` — Zome within the DNA
    /// * `fn_name` — Exported function name
    /// * `input` — The function input (will be MessagePack-encoded)
    pub async fn call_zome<I: Serialize, O: DeserializeOwned>(
        &self,
        zome_name: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<O, ClientError> {
        self.call_zome_on_role(&self.default_role, zome_name, fn_name, input)
            .await
    }

    /// Call a zome function on a specific role with typed input/output.
    ///
    /// Like [`call_zome`](Self::call_zome) but targets a specific role
    /// instead of the default.
    pub async fn call_zome_on_role<I: Serialize, O: DeserializeOwned>(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<O, ClientError> {
        let payload = encode(input)?;
        let response_bytes = self
            .transport
            .call_zome(role_name, zome_name, fn_name, payload)
            .await?;
        decode(&response_bytes)
    }

    /// Call a zome function with raw bytes (no automatic serialization).
    ///
    /// Useful when you already have MessagePack-encoded bytes, or when
    /// working with opaque payloads.
    pub async fn call_zome_raw(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>, ClientError> {
        self.transport
            .call_zome(role_name, zome_name, fn_name, payload)
            .await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;

    /// Mock transport for testing the client layer.
    struct MockTransport {
        response: Vec<u8>,
    }

    impl MockTransport {
        fn responding_with<T: Serialize>(value: &T) -> Self {
            Self {
                response: encode(value).unwrap(),
            }
        }
    }

    impl HolochainTransport for MockTransport {
        fn call_zome(
            &self,
            _role_name: &str,
            _zome_name: &str,
            _fn_name: &str,
            _payload: Vec<u8>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ClientError>>>> {
            let resp = self.response.clone();
            Box::pin(async move { Ok(resp) })
        }

        fn status(&self) -> ConnectionStatus {
            ConnectionStatus::Connected
        }

        fn connect(
            &self,
            _config: ConnectConfig,
        ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>>>> {
            Box::pin(async { Ok(()) })
        }

        fn disconnect(&self) {}
    }

    #[test]
    fn client_creation() {
        let transport = MockTransport::responding_with(&"ok");
        let client = HolochainClient::new(transport, "test-app", "test-role");
        assert_eq!(client.app_id(), "test-app");
        assert_eq!(client.default_role(), "test-role");
        assert_eq!(client.status(), ConnectionStatus::Connected);
    }
}
