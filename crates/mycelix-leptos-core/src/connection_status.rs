// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Connection status indicator component.
//!
//! Displays the current Holochain conductor connection state as a small
//! colored dot with label text, suitable for embedding in a navbar.

use leptos::prelude::*;

/// Describes the connection state for the indicator.
///
/// This mirrors [`mycelix_leptos_client::ConnectionStatus`] but is decoupled
/// so the component can be used with any status source (e.g., mock mode).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    /// Successfully connected to the conductor.
    Connected,
    /// A connection attempt is in progress.
    Connecting,
    /// Not connected.
    Disconnected,
    /// Running against a mock transport (development/testing).
    MockMode,
}

impl std::fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connected => write!(f, "Connected"),
            Self::Connecting => write!(f, "Connecting..."),
            Self::Disconnected => write!(f, "Disconnected"),
            Self::MockMode => write!(f, "Mock Mode"),
        }
    }
}

/// A small connection status indicator showing a colored dot and label.
///
/// Colors:
/// - Connected: green
/// - Connecting: amber/yellow
/// - Disconnected: red
/// - Mock Mode: purple
///
/// # Props
///
/// * `state` — A reactive signal providing the current [`ConnectionState`].
#[component]
pub fn ConnectionStatusIndicator(
    #[prop(optional)] state: Option<ReadSignal<ConnectionState>>,
) -> impl IntoView {
    let state = state.unwrap_or_else(|| {
        // Default to MockMode when no state signal provided
        signal(ConnectionState::MockMode).0
    });
    let dot_color = move || match state.get() {
        ConnectionState::Connected => "#22c55e",    // green-500
        ConnectionState::Connecting => "#eab308",   // yellow-500
        ConnectionState::Disconnected => "#ef4444", // red-500
        ConnectionState::MockMode => "#a855f7",     // purple-500
    };

    let label_text = move || state.get().to_string();

    let pulse_animation = move || match state.get() {
        ConnectionState::Connecting => "animation: mycelix-pulse 1.5s ease-in-out infinite;",
        _ => "",
    };

    view! {
        <span
            style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.8rem; font-family: system-ui, sans-serif;"
        >
            <span
                style=move || format!(
                    "display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {}; {}",
                    dot_color(),
                    pulse_animation()
                )
            ></span>
            <span style="opacity: 0.8;">{label_text}</span>
        </span>
    }
}
