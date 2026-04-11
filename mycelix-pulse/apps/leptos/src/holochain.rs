// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Holochain conductor connection for Mycelix Pulse.
//!
//! Connects via WebSocket, falls back to mock mode.
//! Runtime identifiers still target the legacy `mycelix_mail` role and app id
//! until a compatibility migration is designed and executed.

use leptos::prelude::*;
use serde::{de::DeserializeOwned, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use send_wrapper::SendWrapper;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::spawn_local;

use mycelix_leptos_client::{
    BrowserWsTransport, ConnectConfig, HolochainTransport,
    encode, decode,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Mock,
}

type TransportCell = SendWrapper<Rc<RefCell<Option<BrowserWsTransport>>>>;

#[derive(Clone)]
pub struct HolochainCtx {
    pub status: ReadSignal<ConnectionStatus>,
    set_status: WriteSignal<ConnectionStatus>,
    transport: TransportCell,
}

impl HolochainCtx {
    pub fn is_mock(&self) -> bool {
        self.status.get_untracked() == ConnectionStatus::Mock
    }

    /// Register a callback for real-time signals from the conductor.
    pub fn set_signal_handler<F: Fn(Vec<u8>) + 'static>(&self, handler: F) {
        let transport_ref = self.transport.borrow();
        if let Some(transport) = transport_ref.as_ref() {
            transport.set_signal_handler(handler);
        }
    }

    pub async fn call_zome<I: Serialize, O: DeserializeOwned>(
        &self,
        zome: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<O, String> {
        self.call_zome_on_role("main", zome, fn_name, input).await
    }

    /// Call a zome function on a specific role (e.g. "identity" for DID operations).
    pub async fn call_zome_on_role<I: Serialize, O: DeserializeOwned>(
        &self,
        role: &str,
        zome: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<O, String> {
        // Use BrowserWsTransport directly (Rust-native, like Prism)
        let transport_ref = self.transport.borrow();
        let transport = transport_ref
            .as_ref()
            .ok_or_else(|| "Not connected to conductor".to_string())?;

        let payload = encode(input).map_err(|e| format!("Encode error: {e}"))?;
        let response = transport
            .call_zome(role, zome, fn_name, payload)
            .await
            .map_err(|e| format!("Zome call {zome}.{fn_name} failed: {e}"))?;
        decode(&response).map_err(|e| format!("Decode error for {zome}.{fn_name}: {e}"))
    }

}

const LOCAL_CONDUCTOR_URL: &str = "ws://localhost:8888";
const TUNNEL_CONDUCTOR_URL: &str = "wss://mail-conductor.luminousdynamics.io";

/// Fetch auth token from the SPA server's /api/token endpoint.
/// Works on both local (localhost:8117) and remote (mail.mycelix.net) deployments.
async fn fetch_auth_token() -> Option<Vec<u8>> {
    let resp = wasm_bindgen_futures::JsFuture::from(
        web_sys::window()?.fetch_with_str("/api/token")
    ).await.ok()?;
    let resp: web_sys::Response = resp.dyn_into().ok()?;
    if !resp.ok() { return None; }
    let json = wasm_bindgen_futures::JsFuture::from(resp.json().ok()?).await.ok()?;
    let token_b64 = js_sys::Reflect::get(&json, &JsValue::from_str("token")).ok()?.as_string()?;

    // Decode base64 token to bytes
    let decoded = web_sys::window()?.atob(&token_b64).ok()?;
    let bytes: Vec<u8> = decoded.chars().map(|c| c as u8).collect();
    web_sys::console::log_1(&format!("[Mail] Auth token: {} bytes from /api/token", bytes.len()).into());
    Some(bytes)
}

fn conductor_url() -> String {
    // 1. Explicit override via window.__HC_CONDUCTOR_URL
    if let Some(url) = web_sys::window().and_then(|w| {
        js_sys::Reflect::get(&w, &JsValue::from_str("__HC_CONDUCTOR_URL"))
            .ok()
            .and_then(|v| v.as_string())
    }) {
        return url;
    }

    // 2. Auto-detect: if we're on a remote origin, use the tunnel proxy
    let is_remote = web_sys::window()
        .and_then(|w| w.location().hostname().ok())
        .map(|h| !h.is_empty() && h != "localhost" && h != "127.0.0.1" && !h.starts_with("10.") && !h.starts_with("192.168."))
        .unwrap_or(false);

    if is_remote {
        TUNNEL_CONDUCTOR_URL.to_string()
    } else {
        LOCAL_CONDUCTOR_URL.to_string()
    }
}

/// Wait for the dynamic token promise (from /api/token), then read the value.
#[component]
pub fn HolochainProvider(children: Children) -> impl IntoView {
    let (status, set_status) = signal(ConnectionStatus::Connecting);
    let transport: TransportCell = SendWrapper::new(Rc::new(RefCell::new(None)));

    let ctx = HolochainCtx {
        status,
        set_status,
        transport: transport.clone(),
    };

    provide_context(ctx);

    let transport_for_connect = transport.clone();
    spawn_local(async move {
        // Get auth token from SPA server (/api/token)
        let token_bytes = fetch_auth_token().await;

        // Connect via BrowserWsTransport (pure Rust, like Prism)
        let url = conductor_url();
        web_sys::console::log_1(
            &format!("[Mail] Connecting to conductor at {url}...").into(),
        );

        let ws_transport = BrowserWsTransport::new();
        let config = ConnectConfig {
            url: url.clone(),
            app_id: "mycelix_mail".to_string(),
            auth_token: token_bytes,
            reconnect: None,
            request_timeout_ms: Some(30_000),
        };

        match ws_transport.connect(config).await {
            Ok(()) => {
                web_sys::console::log_1(&"[Mail] Connected to conductor!".into());
                *transport_for_connect.borrow_mut() = Some(ws_transport);
                set_status.set(ConnectionStatus::Connected);
            }
            Err(e) => {
                web_sys::console::log_1(
                    &format!("[Mail] Could not connect: {e:?}. Running in mock mode.").into(),
                );
                set_status.set(ConnectionStatus::Mock);
            }
        }
    });

    children()
}

pub fn use_holochain() -> HolochainCtx {
    expect_context::<HolochainCtx>()
}
