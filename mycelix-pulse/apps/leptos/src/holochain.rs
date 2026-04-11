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
use wasm_bindgen::JsValue;
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
        // Try JS bridge first (has proper signing credentials via @holochain/client)
        if let Some(result) = self.call_zome_via_js_role(role, zome, fn_name, input).await? {
            return Ok(result);
        }

        // Fall back to BrowserWsTransport (may fail without signing)
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

    async fn call_zome_via_js<I: Serialize, O: DeserializeOwned>(
        &self,
        zome: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<Option<O>, String> {
        self.call_zome_via_js_role("main", zome, fn_name, input).await
    }

    /// Call a zome function via the JS bridge (__HC_CALL_ZOME) if available.
    /// Returns Ok(Some(result)) if the JS bridge handled it, Ok(None) if not available.
    async fn call_zome_via_js_role<I: Serialize, O: DeserializeOwned>(
        &self,
        role: &str,
        zome: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<Option<O>, String> {
        let has_bridge = web_sys::window()
            .and_then(|w| js_sys::Reflect::get(&w, &JsValue::from_str("__HC_CALL_ZOME")).ok())
            .map(|v| v.is_function())
            .unwrap_or(false);

        if !has_bridge {
            return Ok(None);
        }

        let payload_json = serde_json::to_string(input)
            .map_err(|e| format!("Serialize error: {e}"))?;

        // Call window.__HC_CALL_ZOME(roleName, zomeName, fnName, payloadJson) -> Promise<string>
        let js_code = format!(
            "window.__HC_CALL_ZOME('{}', '{}', '{}', '{}')",
            role.replace('\'', "\\'"),
            zome.replace('\'', "\\'"),
            fn_name.replace('\'', "\\'"),
            payload_json.replace('\\', "\\\\").replace('\'', "\\'"),
        );

        let promise = js_sys::eval(&js_code)
            .map_err(|e| format!("JS eval error: {e:?}"))?;

        let result = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise))
            .await
            .map_err(|e| {
                let msg = js_sys::JSON::stringify(&e)
                    .map(|s| String::from(s))
                    .unwrap_or_else(|_| format!("{e:?}"));
                format!("Zome call {zome}.{fn_name} failed: {msg}")
            })?;

        let result_str = result.as_string()
            .ok_or_else(|| "JS bridge returned non-string".to_string())?;

        let parsed: O = serde_json::from_str(&result_str)
            .map_err(|e| format!("Decode JS result for {zome}.{fn_name}: {e}"))?;

        Ok(Some(parsed))
    }
}

const LOCAL_CONDUCTOR_URL: &str = "ws://localhost:8888";
const TUNNEL_CONDUCTOR_URL: &str = "wss://mail-conductor.luminousdynamics.io";

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
async fn auth_token() -> Option<String> {
    // Wait for the token fetch promise if it exists
    if let Some(promise) = web_sys::window().and_then(|w| {
        js_sys::Reflect::get(&w, &JsValue::from_str("__HC_TOKEN_PROMISE"))
            .ok()
            .and_then(|v| {
                if v.is_undefined() || v.is_null() { None }
                else { Some(js_sys::Promise::from(v)) }
            })
    }) {
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }
    // Now read the token value
    web_sys::window().and_then(|w| {
        js_sys::Reflect::get(&w, &JsValue::from_str("__HC_AUTH_TOKEN"))
            .ok()
            .and_then(|v| v.as_string())
    })
}

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
        // Wait for JS bridge to complete its connection attempt
        let token = auth_token().await;

        // Check if the JS bridge (@holochain/client) already connected successfully
        let js_connected = web_sys::window()
            .and_then(|w| js_sys::Reflect::get(&w, &JsValue::from_str("__HC_CALL_ZOME")).ok())
            .map(|v| v.is_function())
            .unwrap_or(false);

        if js_connected {
            web_sys::console::log_1(&"[Mail] Using JS bridge (@holochain/client) for zome calls".into());
            set_status.set(ConnectionStatus::Connected);
            // No BrowserWsTransport needed — all calls go through JS bridge
            return;
        }

        // Fallback: try BrowserWsTransport (won't have signing, but shows connection status)
        let url = conductor_url();
        web_sys::console::log_1(
            &format!("[Mail] JS bridge not available, trying BrowserWsTransport at {url}...").into(),
        );

        let has_token = token.is_some();
        let token_bytes = token.and_then(|s| {
            web_sys::window()
                .and_then(|w| w.atob(&s).ok())
                .map(|decoded| decoded.chars().map(|c| c as u8).collect::<Vec<u8>>())
        });
        let token_len = token_bytes.as_ref().map(|b| b.len()).unwrap_or(0);
        web_sys::console::log_1(
            &format!("[Mail] Auth token: has_raw={has_token}, decoded_len={token_len}").into(),
        );

        let ws_transport = BrowserWsTransport::new();
        // Keep the installed app id stable until the conductor migration path is ready.
        let config = ConnectConfig {
            url: url.clone(),
            app_id: "mycelix_mail".to_string(),
            auth_token: token_bytes,
            reconnect: None,
            request_timeout_ms: Some(30_000),
        };

        web_sys::console::log_1(&format!("[Mail] Calling connect({url})...").into());
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
