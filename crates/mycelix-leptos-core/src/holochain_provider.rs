// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Auto-connecting Holochain provider with mock fallback.
//!
//! Replaces per-app `holochain.rs` files. Configurable via
//! [`HolochainProviderConfig`] for single-role, multi-role, and
//! JS-status-only connection strategies.

use leptos::prelude::*;
use send_wrapper::SendWrapper;
use serde::{de::DeserializeOwned, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::spawn_local;

use mycelix_leptos_client::{
    decode, encode, BrowserWsTransport, ConnectConfig, HolochainTransport, ReconnectConfig,
};

// ---------------------------------------------------------------------------
// Connection status
// ---------------------------------------------------------------------------

/// Connection status for UI display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Mock,
}

impl ConnectionStatus {
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Disconnected => "status-disconnected",
            Self::Connecting => "status-connecting",
            Self::Connected => "status-connected",
            Self::Reconnecting => "status-reconnecting",
            Self::Mock => "status-mock",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Disconnected => "Disconnected",
            Self::Connecting => "Connecting\u{2026}",
            Self::Connected => "Connected",
            Self::Reconnecting => "Reconnecting\u{2026}",
            Self::Mock => "Mock",
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How the provider connects to the conductor.
#[derive(Clone, Debug)]
pub enum ConnectStrategy {
    /// Auto-connect via BrowserWsTransport, fall back to mock.
    WebSocket,
    /// Read `__HC_STATUS` from JS window, no real transport.
    JsStatusOnly,
    /// Always mock, no connection attempt.
    MockOnly,
}

/// Configuration for the auto-connecting Holochain provider.
#[derive(Clone, Debug)]
pub struct HolochainProviderConfig {
    /// hApp identifier (e.g. "praxis", "hearth", "mycelix-unified").
    pub app_id: String,
    /// Default role for single-role apps (e.g. Some("hearth")).
    /// Multi-role apps (governance) set this to None and pass role explicitly.
    pub default_role: Option<String>,
    /// Log prefix for console messages (e.g. "[Hearth]").
    pub log_prefix: &'static str,
    /// Connection strategy.
    pub connect_strategy: ConnectStrategy,
    /// Custom status labels (e.g. Health uses "Local Demo" instead of "Mock").
    pub status_labels: Option<StatusLabels>,
}

/// Custom labels for connection status display.
#[derive(Clone, Debug)]
pub struct StatusLabels {
    pub disconnected: &'static str,
    pub connecting: &'static str,
    pub connected: &'static str,
    pub mock: &'static str,
}

// ---------------------------------------------------------------------------
// Holochain context
// ---------------------------------------------------------------------------

type TransportCell = SendWrapper<Rc<RefCell<Option<BrowserWsTransport>>>>;

/// The Holochain client context shared across the app via Leptos context.
#[derive(Clone)]
pub struct HolochainCtx {
    pub status: ReadSignal<ConnectionStatus>,
    set_status: WriteSignal<ConnectionStatus>,
    transport: TransportCell,
    default_role: Option<String>,
    status_labels: Option<StatusLabels>,
}

impl HolochainCtx {
    /// Call a zome function on the specified role.
    pub async fn call_zome<I: Serialize, O: DeserializeOwned>(
        &self,
        role: &str,
        zome: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<O, String> {
        let transport = self.transport.borrow();
        let transport = match transport.as_ref() {
            Some(t) => t.clone(),
            None => {
                return Err(format!(
                    "Mock mode: {role}.{zome}.{fn_name} \u{2014} no conductor connected"
                ));
            }
        };
        drop(self.transport.borrow());

        let payload = encode(input).map_err(|e| format!("Encode error: {e}"))?;

        let response_bytes = transport
            .call_zome(role, zome, fn_name, payload)
            .await
            .map_err(|e| format!("Zome call {role}.{zome}.{fn_name} failed: {e}"))?;

        decode(&response_bytes).map_err(|e| format!("Decode error: {e}"))
    }

    /// Call a zome function using the default role.
    ///
    /// # Panics
    /// Panics if no `default_role` was configured.
    pub async fn call_zome_default<I: Serialize, O: DeserializeOwned>(
        &self,
        zome: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<O, String> {
        let role = self
            .default_role
            .as_deref()
            .expect("call_zome_default requires a default_role in HolochainProviderConfig");
        self.call_zome(role, zome, fn_name, input).await
    }

    pub fn is_mock(&self) -> bool {
        self.status.get_untracked() == ConnectionStatus::Mock
    }

    /// Return the connected agent pubkey in Holochain's `u...` display form.
    pub fn connected_agent_pub_key_b64(&self) -> Option<String> {
        self.transport
            .borrow()
            .as_ref()
            .and_then(|transport| transport.connected_agent_pub_key_b64())
    }

    /// Return the connected agent DID derived from the conductor cell.
    pub fn connected_agent_did(&self) -> Option<String> {
        self.transport
            .borrow()
            .as_ref()
            .and_then(|transport| transport.connected_agent_did())
    }

    /// Get the display label for the current status.
    pub fn status_label(&self) -> &'static str {
        if let Some(ref labels) = self.status_labels {
            match self.status.get() {
                ConnectionStatus::Disconnected => labels.disconnected,
                ConnectionStatus::Connecting => labels.connecting,
                ConnectionStatus::Connected => labels.connected,
                ConnectionStatus::Reconnecting => labels.connecting,
                ConnectionStatus::Mock => labels.mock,
            }
        } else {
            self.status.get().label()
        }
    }
}

// ---------------------------------------------------------------------------
// Provider component
// ---------------------------------------------------------------------------

const DEFAULT_CONDUCTOR_URL: &str = "ws://localhost:8888";

fn conductor_url() -> String {
    web_sys::window()
        .and_then(|w| {
            js_sys::Reflect::get(&w, &JsValue::from_str("__HC_CONDUCTOR_URL"))
                .ok()
                .and_then(|v| v.as_string())
        })
        .unwrap_or_else(|| DEFAULT_CONDUCTOR_URL.to_string())
}

fn auth_token() -> Option<String> {
    web_sys::window().and_then(|w| {
        js_sys::Reflect::get(&w, &JsValue::from_str("__HC_AUTH_TOKEN"))
            .ok()
            .and_then(|v| v.as_string())
    })
}

fn read_js_conductor_status() -> ConnectionStatus {
    let Some(window) = web_sys::window() else {
        return ConnectionStatus::Mock;
    };
    let val = js_sys::Reflect::get(&window, &JsValue::from_str("__HC_STATUS"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "mock".to_string());
    match val.as_str() {
        "connected" => ConnectionStatus::Connected,
        "connecting" => ConnectionStatus::Connecting,
        "disconnected" => ConnectionStatus::Disconnected,
        _ => ConnectionStatus::Mock,
    }
}

/// Auto-connecting Holochain provider.
///
/// Provides [`HolochainCtx`] via Leptos context. On mount, connects to the
/// conductor using the configured strategy, falling back to mock mode.
#[component]
pub fn HolochainProviderAuto(config: HolochainProviderConfig, children: Children) -> impl IntoView {
    let initial_status = match &config.connect_strategy {
        ConnectStrategy::WebSocket => ConnectionStatus::Connecting,
        ConnectStrategy::JsStatusOnly => read_js_conductor_status(),
        ConnectStrategy::MockOnly => ConnectionStatus::Mock,
    };

    let (status, set_status) = signal(initial_status);
    let transport: TransportCell = SendWrapper::new(Rc::new(RefCell::new(None)));

    let ctx = HolochainCtx {
        status,
        set_status,
        transport: transport.clone(),
        default_role: config.default_role.clone(),
        status_labels: config.status_labels.clone(),
    };

    provide_context(ctx);

    match config.connect_strategy {
        ConnectStrategy::WebSocket => {
            let transport_for_connect = transport.clone();
            let log_prefix = config.log_prefix;
            let app_id = config.app_id.clone();

            spawn_local(async move {
                let url = conductor_url();
                let token = auth_token();
                web_sys::console::log_1(
                    &format!("{log_prefix} Connecting to conductor at {url}\u{2026}").into(),
                );

                let ws_transport = BrowserWsTransport::new();
                let connect_config = ConnectConfig {
                    url,
                    app_id,
                    auth_token: token.map(|s| s.into_bytes()),
                    reconnect: Some(ReconnectConfig::default()),
                    request_timeout_ms: Some(30_000),
                };

                match ws_transport.connect(connect_config).await {
                    Ok(()) => {
                        web_sys::console::log_1(
                            &format!("{log_prefix} Connected to conductor!").into(),
                        );
                        *transport_for_connect.borrow_mut() = Some(ws_transport);
                        set_status.set(ConnectionStatus::Connected);
                    }
                    Err(e) => {
                        web_sys::console::log_1(
                            &format!("{log_prefix} Could not connect: {e}. Running in mock mode.")
                                .into(),
                        );
                        set_status.set(ConnectionStatus::Mock);
                    }
                }
            });
        }
        ConnectStrategy::JsStatusOnly => {
            // For JsStatusOnly with Connecting status, poll after a delay
            if initial_status == ConnectionStatus::Connecting {
                let log_prefix = config.log_prefix;
                leptos::prelude::set_timeout(
                    move || {
                        let resolved = read_js_conductor_status();
                        set_status.set(resolved);
                        web_sys::console::log_1(
                            &format!("{log_prefix} Conductor status: {resolved:?}").into(),
                        );
                    },
                    std::time::Duration::from_millis(3500),
                );
            }
        }
        ConnectStrategy::MockOnly => {
            // Nothing to do
        }
    }

    children()
}

/// Retrieve the [`HolochainCtx`] from the nearest [`HolochainProviderAuto`].
pub fn use_holochain() -> HolochainCtx {
    expect_context::<HolochainCtx>()
}

/// Connection status badge component (CSS-class-based).
#[component]
pub fn ConnectionBadge() -> impl IntoView {
    let ctx = use_holochain();
    let ctx2 = ctx.clone();

    view! {
        <span class=move || format!("connection-badge {}", ctx.status.get().css_class())>
            <span class="status-dot"></span>
            {move || ctx2.status_label()}
        </span>
    }
}
