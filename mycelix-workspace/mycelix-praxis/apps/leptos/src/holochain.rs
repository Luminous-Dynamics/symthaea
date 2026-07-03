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
    ConnectStrategy, ConnectionBadge, ConnectionStatus, HolochainCtx, HolochainProviderConfig,
    use_holochain,
};

use crate::persistence;

/// localStorage key for a user-set conductor URL override.
const CONDUCTOR_URL_OVERRIDE_KEY: &str = "praxis_conductor_url_override";

/// Default conductor App WebSocket URL when no override or `index.html`
/// injection is present.
const DEFAULT_CONDUCTOR_URL: &str = "ws://localhost:8888";

/// Apply a saved conductor URL override (if any) to `window.__HC_CONDUCTOR_URL`.
///
/// Must run **before** [`HolochainProvider`] mounts — `HolochainProviderAuto`
/// reads `window.__HC_CONDUCTOR_URL` exactly once, at the start of its
/// connect task. Call this at the top of `main()`, before `mount_to_body`.
///
/// Without this, the only way to point Praxis at a non-default conductor was
/// editing `window.__HC_CONDUCTOR_URL` / `window.__HC_AUTH_TOKEN` directly in
/// `index.html` — there was no in-app way to set or even see it.
pub fn apply_conductor_url_override() {
    let Some(url) = persistence::load::<String>(CONDUCTOR_URL_OVERRIDE_KEY) else {
        return;
    };
    if url.trim().is_empty() {
        return;
    }
    if let Some(window) = web_sys::window() {
        let _ = js_sys::Reflect::set(
            &window,
            &wasm_bindgen::JsValue::from_str("__HC_CONDUCTOR_URL"),
            &wasm_bindgen::JsValue::from_str(&url),
        );
    }
}

fn active_conductor_url() -> String {
    web_sys::window()
        .and_then(|w| {
            js_sys::Reflect::get(&w, &wasm_bindgen::JsValue::from_str("__HC_CONDUCTOR_URL")).ok()
        })
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| DEFAULT_CONDUCTOR_URL.to_string())
}

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

/// In-app conductor connection settings.
///
/// Shows the live connection status (via [`use_holochain`]) alongside the
/// conductor WebSocket URL currently in effect, and lets the learner
/// override it — persisted to localStorage, applied on next load via
/// [`apply_conductor_url_override`]. This is the first in-app UI for
/// confirming/setting the conductor connection; previously this was only
/// configurable by editing `window.__HC_CONDUCTOR_URL` in `index.html`.
#[component]
pub fn ConductorSettingsPanel() -> impl IntoView {
    let hc = use_holochain();
    let hc_label = hc.clone();

    let (draft_url, set_draft_url) = signal(
        persistence::load::<String>(CONDUCTOR_URL_OVERRIDE_KEY)
            .unwrap_or_else(active_conductor_url),
    );
    let (saved, set_saved) = signal(false);

    let on_save = move |_| {
        let url = draft_url.get();
        if url.trim().is_empty() {
            return;
        }
        persistence::save(CONDUCTOR_URL_OVERRIDE_KEY, &url);
        set_saved.set(true);
    };

    let on_reset = move |_| {
        persistence::remove(CONDUCTOR_URL_OVERRIDE_KEY);
        set_draft_url.set(DEFAULT_CONDUCTOR_URL.to_string());
        set_saved.set(true);
    };

    view! {
        <div
            class="conductor-settings"
            style="margin-top: 0.75rem; padding: 0.75rem; background: var(--surface-low); border: 1px solid var(--border); border-radius: 8px; font-size: 0.85rem"
        >
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem">
                <strong>"Conductor Connection"</strong>
                <span>{move || hc_label.status_label()}</span>
            </div>
            <div style="display: flex; gap: 0.5rem">
                <input
                    type="text"
                    style="flex: 1; padding: 0.4rem; border: 1px solid var(--border); border-radius: 4px; background: var(--surface); color: var(--text)"
                    prop:value=move || draft_url.get()
                    on:input=move |ev| set_draft_url.set(event_target_value(&ev))
                    placeholder=DEFAULT_CONDUCTOR_URL
                />
                <button class="btn-sm btn-primary" on:click=on_save>"Save"</button>
                <button class="btn-sm btn-outline" on:click=on_reset>"Reset"</button>
            </div>
            {move || if saved.get() {
                view! {
                    <p style="margin: 0.5rem 0 0; color: var(--text-secondary)">
                        "Saved. Reload the app to connect using the new conductor URL."
                    </p>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}
        </div>
    }
}
