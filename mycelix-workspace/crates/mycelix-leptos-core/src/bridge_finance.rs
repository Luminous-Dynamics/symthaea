// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Type-safe bridge for the Finance cluster.
//!
//! (Manually implemented to validate the Bridge Codegen pattern)

use leptos::prelude::*;
use mycelix_leptos_client::{ClientError, use_holochain};
pub use personal_leptos_types::ActionHash;
pub use personal_leptos_types::Record;

// Import wire types
pub use personal_leptos_types::SapBalanceResponse; // Assuming shared or mirroring
// For now we'll use placeholders for types not in personal-leptos-types
// In a real rollout, these come from finance-wire-types

pub type BridgeResult<T> = Result<T, ClientError>;

/// Initialize SAP balance for a member.
pub fn initialize_sap_balance() -> Action<String, BridgeResult<Record>> {
    create_server_action(|member_did: String| async move {
        let client = use_holochain();
        client
            .call_zome("payments", "initialize_sap_balance", member_did)
            .await
    })
}

/// Send a payment.
pub fn send_payment() -> Action<serde_json::Value, BridgeResult<Record>> {
    create_server_action(|input: serde_json::Value| async move {
        let client = use_holochain();
        client.call_zome("payments", "send_payment", input).await
    })
}

/// Get runtime discovery data (DIDs, Treasury IDs, etc).
pub fn runtime_discovery() -> Action<(), BridgeResult<serde_json::Value>> {
    create_server_action(|_: ()| async move {
        let client = use_holochain();
        client.call_zome("bridge", "runtime_discovery", ()).await
    })
}

/// Health check for the finance cluster.
pub fn health_check() -> Action<(), BridgeResult<serde_json::Value>> {
    create_server_action(|_: ()| async move {
        let client = use_holochain();
        client.call_zome("bridge", "health_check", ()).await
    })
}
