// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transport trait abstracting the communication mechanism with a Holochain conductor.
//!
//! Implementations:
//! - [`BrowserWsTransport`](crate::browser::BrowserWsTransport) — browser `WebSocket`
//! - [`TauriIpcTransport`](crate::tauri::TauriIpcTransport) — Tauri IPC bridge

use crate::error::ClientError;
use crate::types::{ConnectConfig, ConnectionStatus};
use std::future::Future;
use std::pin::Pin;

/// Abstraction over the communication channel to a Holochain conductor.
///
/// This trait uses `Pin<Box<dyn Future>>` instead of `async fn` in trait
/// because `async_fn_in_traits` is not yet stable, and we need object safety
/// is not required (generic `HolochainClient<T: HolochainTransport>`).
///
/// The `'static` bound is needed so futures can be spawned in WASM.
pub trait HolochainTransport: 'static {
    /// Call a zome function on the conductor.
    ///
    /// # Arguments
    /// * `role_name` — Role from the hApp manifest (e.g. "governance")
    /// * `zome_name` — Zome within the DNA (e.g. "agora")
    /// * `fn_name` — Exported function name (e.g. "create_proposal")
    /// * `payload` — MessagePack-encoded input bytes
    ///
    /// # Returns
    /// MessagePack-encoded output bytes from the zome function.
    fn call_zome(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        payload: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ClientError>>>>;

    /// Current connection status.
    fn status(&self) -> ConnectionStatus;

    /// Establish a connection to the conductor.
    ///
    /// Performs WebSocket open, optional authentication, and app_info
    /// discovery to build the role→cell_id mapping.
    ///
    /// # Arguments
    /// * `config` — Connection configuration including URL, app ID, and
    ///   optional auth token.
    fn connect(
        &self,
        config: ConnectConfig,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>>>>;

    /// Disconnect from the conductor, dropping the underlying connection.
    fn disconnect(&self);
}
