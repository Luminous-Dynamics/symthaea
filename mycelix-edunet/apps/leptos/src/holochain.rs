// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Holochain conductor context for Leptos CSR.
//!
//! Provides a [`HolochainCtx`] via Leptos context that all pages can use
//! to call zome functions. Initially uses mock data, with the real
//! [`BrowserWsTransport`](mycelix_leptos_client::BrowserWsTransport) available
//! when a conductor is running.

use leptos::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

// ---------------------------------------------------------------------------
// Connection status (UI-facing, simpler than the transport-level enum)
// ---------------------------------------------------------------------------

/// Connection status for the UI status indicator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// No connection attempt has been made.
    Disconnected,
    /// Currently trying to connect to the conductor.
    Connecting,
    /// Successfully connected — zome calls go to a real conductor.
    Connected,
    /// Running with mock data (no conductor available).
    Mock,
}

impl ConnectionStatus {
    /// CSS class name for the status badge.
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Disconnected => "status-disconnected",
            Self::Connecting => "status-connecting",
            Self::Connected => "status-connected",
            Self::Mock => "status-mock",
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Disconnected => "Disconnected",
            Self::Connecting => "Connecting...",
            Self::Connected => "Connected",
            Self::Mock => "Mock",
        }
    }
}

// ---------------------------------------------------------------------------
// Holochain context
// ---------------------------------------------------------------------------

/// The Holochain client context shared across the app via Leptos context.
///
/// All pages access this through [`use_holochain()`]. Zome calls return
/// mock data when no conductor is available, allowing the UI to be
/// developed and tested independently.
#[derive(Clone)]
pub struct HolochainCtx {
    /// Reactive connection status signal (read half).
    pub status: ReadSignal<ConnectionStatus>,
    /// Write half — used internally and by the connect/disconnect methods.
    set_status: WriteSignal<ConnectionStatus>,
}

impl HolochainCtx {
    /// Call a zome function and decode the result.
    ///
    /// When connected to a real conductor this will serialize `input` as
    /// MessagePack, send it over WebSocket, and decode the response.
    ///
    /// In mock mode this returns `Err` so callers can fall back to mock data.
    pub async fn call_zome<I: Serialize, O: DeserializeOwned>(
        &self,
        zome: &str,
        fn_name: &str,
        _input: &I,
    ) -> Result<O, String> {
        // TODO: Wire to real conductor via BrowserWsTransport
        //
        // When ready:
        // 1. Store an Rc<HolochainClient<BrowserWsTransport>> in this struct
        // 2. Call self.client.call_zome(zome, fn_name, input).await
        // 3. Map ClientError to String
        //
        // For now, every call returns an error so the UI falls back to mock data.
        Err(format!("Mock mode: {}.{} — no conductor connected", zome, fn_name))
    }

    /// Whether the context is in mock mode (no conductor).
    pub fn is_mock(&self) -> bool {
        self.status.get_untracked() == ConnectionStatus::Mock
    }
}

// ---------------------------------------------------------------------------
// Provider component
// ---------------------------------------------------------------------------

/// Wraps children with a [`HolochainCtx`] in Leptos context.
///
/// Place this around the `<Router>` in `App` so every page can call
/// [`use_holochain()`].
#[component]
pub fn HolochainProvider(children: Children) -> impl IntoView {
    let (status, set_status) = signal(ConnectionStatus::Mock);

    let ctx = HolochainCtx {
        status,
        set_status,
    };

    provide_context(ctx);
    children()
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/// Retrieve the [`HolochainCtx`] from the nearest ancestor `HolochainProvider`.
///
/// # Panics
///
/// Panics if called outside a `HolochainProvider` subtree.
pub fn use_holochain() -> HolochainCtx {
    expect_context::<HolochainCtx>()
}

// ---------------------------------------------------------------------------
// Connection status badge (reusable component)
// ---------------------------------------------------------------------------

/// Small badge showing the current conductor connection status.
///
/// Renders a colored dot + label. Intended for the navbar.
#[component]
pub fn ConnectionBadge() -> impl IntoView {
    let ctx = use_holochain();

    view! {
        <span class=move || {
            format!("connection-badge {}", ctx.status.get().css_class())
        }>
            <span class="status-dot"></span>
            {move || ctx.status.get().label()}
        </span>
    }
}
