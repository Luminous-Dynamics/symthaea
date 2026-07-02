// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Holochain conductor context for Praxis Leptos CSR.
//!
//! Wraps [`mycelix_leptos_core::holochain_provider`] with Praxis-specific
//! defaults (app_id, log prefix, default role). All pages use `use_holochain()`
//! to call zome functions.
//!
//! Gains from using the shared provider:
//! - Auto-reconnect with exponential backoff
//! - Cross-role dispatch (call Craft zomes via "craft" role)
//! - Request timeout (30s default)
//! - 8D Sovereign Profile integration

use leptos::prelude::*;

pub use mycelix_leptos_core::holochain_provider::{
    use_holochain, HolochainCtx, HolochainProviderConfig, ConnectStrategy,
    ConnectionStatus, ConnectionBadge,
};

/// Wraps children with Praxis-configured HolochainProvider.
///
/// Connects to the shared ecosystem conductor with app_id "praxis",
/// auto-reconnect enabled (5 attempts, exponential backoff), and
/// 30-second request timeout.
#[component]
pub fn HolochainProvider(children: Children) -> impl IntoView {
    let config = HolochainProviderConfig {
        app_id: "praxis".to_string(),
        default_role: Some("praxis".to_string()),
        log_prefix: "[Praxis]",
        connect_strategy: ConnectStrategy::WebSocket,
        status_labels: None,
    };

    view! {
        <mycelix_leptos_core::holochain_provider::HolochainProviderAuto config=config>
            {children()}
        </mycelix_leptos_core::holochain_provider::HolochainProviderAuto>
    }
}
