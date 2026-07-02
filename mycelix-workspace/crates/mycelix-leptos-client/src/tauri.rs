// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tauri IPC transport for Holochain conductor communication.
//!
//! Uses Tauri's `invoke` mechanism (via `wasm-bindgen` FFI) to call a backend
//! command that proxies zome calls to a Holochain conductor over the native
//! `holochain_client` crate with real tokio networking.
//!
//! # Architecture
//!
//! ```text
//! Leptos WASM  ──invoke("call_zome", args)──>  Tauri Rust backend
//!              <──JSON response──────────────   (holochain_client)
//! ```
//!
//! The Tauri backend is expected to expose a command:
//!
//! ```rust,ignore
//! #[tauri::command]
//! async fn call_zome(
//!     role: String,
//!     zome: String,
//!     fn_name: String,
//!     payload: Vec<u8>,
//! ) -> Result<Vec<u8>, String>;
//! ```
//!
//! # Feature gate
//!
//! This module is only available with the `tauri` feature enabled.

use crate::error::ClientError;
use crate::transport::HolochainTransport;
use crate::types::{ConnectConfig, ConnectionStatus};

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Tauri JS interop
// ---------------------------------------------------------------------------

#[wasm_bindgen]
extern "C" {
    /// Invoke a Tauri command via the `__TAURI__.core.invoke` global.
    ///
    /// The Tauri IPC bridge serializes `args` as JSON and routes to the
    /// Rust backend command identified by `cmd`.
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], js_name = invoke, catch)]
    async fn tauri_invoke(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;
}

// ---------------------------------------------------------------------------
// IPC payload types (JSON serialized for Tauri)
// ---------------------------------------------------------------------------

/// Arguments sent to the `call_zome` Tauri command.
#[derive(Serialize)]
struct CallZomeArgs {
    role: String,
    zome: String,
    fn_name: String,
    /// MessagePack-encoded payload, base64-encoded for JSON transport.
    payload: Vec<u8>,
}

/// Response from the `call_zome` Tauri command.
#[derive(Deserialize)]
struct CallZomeResponse {
    /// MessagePack-encoded zome output bytes.
    data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// TauriIpcTransport
// ---------------------------------------------------------------------------

/// Transport that communicates with the Holochain conductor through Tauri's
/// IPC bridge.
///
/// Unlike [`BrowserWsTransport`](crate::browser::BrowserWsTransport), this
/// transport does not maintain a WebSocket connection. The Tauri Rust backend
/// owns the `holochain_client` connection with native tokio async. Each
/// `call_zome` invocation is a standalone IPC round-trip.
///
/// # Connection model
///
/// `connect()` verifies that the Tauri IPC bridge is available by invoking
/// a `ping` command. `disconnect()` is a no-op since the backend manages
/// the actual conductor connection.
pub struct TauriIpcTransport {
    status: Rc<RefCell<ConnectionStatus>>,
}

impl TauriIpcTransport {
    /// Create a new Tauri IPC transport.
    ///
    /// Does not establish any connection. Call [`connect`](Self::connect) to
    /// verify the Tauri bridge is available.
    pub fn new() -> Self {
        Self {
            status: Rc::new(RefCell::new(ConnectionStatus::Disconnected)),
        }
    }
}

impl Default for TauriIpcTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl HolochainTransport for TauriIpcTransport {
    fn connect(
        &self,
        config: ConnectConfig,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>>>> {
        let status = Rc::clone(&self.status);

        Box::pin(async move {
            *status.borrow_mut() = ConnectionStatus::Connecting;

            // For Tauri, the backend manages conductor connection, authentication,
            // and cell mapping internally. We just verify the IPC bridge is alive.
            // The config.url and config.auth_token are passed to the backend's
            // `connect` command so it can establish the conductor connection.
            let connect_args = serde_wasm_bindgen::to_value(&serde_json::json!({
                "url": config.url,
                "app_id": config.app_id,
                "auth_token": config.auth_token,
            }))
            .map_err(|e| ClientError::SerializationError(e.to_string()))?;

            match tauri_invoke("holochain_connect", connect_args).await {
                Ok(_) => {
                    *status.borrow_mut() = ConnectionStatus::Connected;
                    Ok(())
                }
                Err(e) => {
                    let msg = format!("{:?}", e);
                    *status.borrow_mut() = ConnectionStatus::Error(msg.clone());
                    Err(ClientError::ConnectionFailed(msg))
                }
            }
        })
    }

    fn call_zome(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        payload: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ClientError>>>> {
        let status = Rc::clone(&self.status);
        let role = role_name.to_string();
        let zome = zome_name.to_string();
        let fn_name = fn_name.to_string();

        Box::pin(async move {
            // Check that we believe we are connected
            if *status.borrow() != ConnectionStatus::Connected {
                return Err(ClientError::NotConnected);
            }

            // Serialize the arguments as JSON for Tauri IPC
            let args = CallZomeArgs {
                role,
                zome,
                fn_name,
                payload,
            };

            let args_js = serde_wasm_bindgen::to_value(&args)
                .map_err(|e| ClientError::SerializationError(e.to_string()))?;

            // Invoke the Tauri backend command
            let result = tauri_invoke("call_zome", args_js).await.map_err(|e| {
                let msg = js_value_to_string(&e);
                *status.borrow_mut() = ConnectionStatus::Error(msg.clone());
                ClientError::ZomeCallFailed(msg)
            })?;

            // Deserialize the response
            let response: CallZomeResponse = serde_wasm_bindgen::from_value(result)
                .map_err(|e| ClientError::InvalidResponse(e.to_string()))?;

            Ok(response.data)
        })
    }

    fn status(&self) -> ConnectionStatus {
        self.status.borrow().clone()
    }

    fn disconnect(&self) {
        *self.status.borrow_mut() = ConnectionStatus::Disconnected;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a JsValue to a displayable string for error messages.
fn js_value_to_string(val: &JsValue) -> String {
    if let Some(s) = val.as_string() {
        s
    } else {
        format!("{:?}", val)
    }
}
