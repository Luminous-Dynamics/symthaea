// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Browser-compatible Holochain client for Leptos frontends.
//!
//! Replaces `@holochain/client` (JavaScript) for Rust WASM frontends.
//! Uses `web-sys::WebSocket` + `rmp-serde` (MessagePack) to communicate
//! with a Holochain conductor over binary WebSocket frames.
//!
//! # Architecture
//!
//! The crate is built around the [`HolochainTransport`] trait, which abstracts
//! the underlying communication mechanism. Two implementations are provided:
//!
//! - [`BrowserWsTransport`] — Uses `web-sys::WebSocket` for browser WASM targets
//! - `TauriIpcTransport` — (future) Calls Tauri backend via `wasm-bindgen` `invoke()`
//!
//! The [`HolochainClient`] type wraps any transport and provides a typed,
//! ergonomic API for calling zome functions with automatic MessagePack
//! serialization/deserialization.
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_leptos_client::{HolochainClient, BrowserWsTransport};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize)]
//! struct CreateProposal { title: String, body: String }
//!
//! #[derive(Deserialize)]
//! struct ProposalHash { hash: Vec<u8> }
//!
//! async fn example() {
//!     let transport = BrowserWsTransport::new();
//!     let client = HolochainClient::new(transport, "mycelix-unified", "governance");
//!     // Connect with optional auth token (None = no authentication)
//!     client.connect("ws://localhost:8888", None).await.unwrap();
//!
//!     let result: ProposalHash = client.call_zome(
//!         "agora",
//!         "create_proposal",
//!         &CreateProposal { title: "Test".into(), body: "Body".into() },
//!     ).await.unwrap();
//! }
//! ```

#[cfg(feature = "browser")]
pub mod browser;
pub mod client;
pub mod error;
pub mod mock;
#[cfg(feature = "native")]
pub mod native;
#[cfg(feature = "tauri")]
pub mod tauri;
pub mod transport;
pub mod types;

// Re-exports for convenience
pub use client::HolochainClient;
pub use error::ClientError;
pub use mock::MockTransport;
#[cfg(feature = "native")]
pub use native::NativeWsTransport;
pub use transport::HolochainTransport;
pub use types::{
    decode, encode, ConnectConfig, ConnectionStatus, ReconnectConfig, ZomeCallRequest,
    ZomeCallResponse,
};

#[cfg(feature = "browser")]
pub use browser::BrowserWsTransport;

#[cfg(feature = "tauri")]
pub use tauri::TauriIpcTransport;
