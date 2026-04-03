// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Holochain conductor context for the Hearth Leptos app.
//!
//! Replicates the EduNet pattern: [`HolochainCtx`] via Leptos context,
//! real WebSocket connection with mock fallback.

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

// ---------------------------------------------------------------------------
// Connection status
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Mock,
}

impl ConnectionStatus {
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Disconnected => "status-disconnected",
            Self::Connecting => "status-connecting",
            Self::Connected => "status-connected",
            Self::Mock => "status-mock",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Disconnected => "Disconnected",
            Self::Connecting => "Connecting…",
            Self::Connected => "Connected",
            Self::Mock => "Mock",
        }
    }
}

// ---------------------------------------------------------------------------
// Holochain context
// ---------------------------------------------------------------------------

type TransportCell = SendWrapper<Rc<RefCell<Option<BrowserWsTransport>>>>;

#[derive(Clone)]
#[allow(dead_code)]
pub struct HolochainCtx {
    pub status: ReadSignal<ConnectionStatus>,
    set_status: WriteSignal<ConnectionStatus>,
    transport: TransportCell,
}

impl HolochainCtx {
    /// Call a zome function on the hearth DNA.
    pub async fn call_zome<I: Serialize, O: DeserializeOwned>(
        &self,
        zome: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<O, String> {
        let transport = self.transport.borrow();
        let transport = match transport.as_ref() {
            Some(t) => t.clone(),
            None => {
                return Err(format!(
                    "Mock mode: {zome}.{fn_name} — no conductor connected"
                ));
            }
        };
        drop(self.transport.borrow());

        let payload = encode(input).map_err(|e| format!("Encode error: {e}"))?;

        let response_bytes = transport
            .call_zome("hearth", zome, fn_name, payload)
            .await
            .map_err(|e| format!("Zome call {zome}.{fn_name} failed: {e}"))?;

        decode(&response_bytes).map_err(|e| format!("Decode error: {e}"))
    }

    pub fn is_mock(&self) -> bool {
        self.status.get_untracked() == ConnectionStatus::Mock
    }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

const CONDUCTOR_URL: &str = "ws://localhost:8888";

fn conductor_url() -> String {
    web_sys::window()
        .and_then(|w| {
            js_sys::Reflect::get(&w, &JsValue::from_str("__HC_CONDUCTOR_URL"))
                .ok()
                .and_then(|v| v.as_string())
        })
        .unwrap_or_else(|| CONDUCTOR_URL.to_string())
}

fn auth_token() -> Option<String> {
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
        let url = conductor_url();
        let token = auth_token();
        web_sys::console::log_1(
            &format!("[Hearth] Connecting to conductor at {url}…").into(),
        );

        let ws_transport = BrowserWsTransport::new();
        let config = ConnectConfig {
            url,
            app_id: "mycelix-unified".to_string(),
            auth_token: token.map(|s| s.into_bytes()),
        };

        match ws_transport.connect(config).await {
            Ok(()) => {
                web_sys::console::log_1(&"[Hearth] Connected to conductor!".into());
                *transport_for_connect.borrow_mut() = Some(ws_transport);
                set_status.set(ConnectionStatus::Connected);
            }
            Err(e) => {
                web_sys::console::log_1(
                    &format!("[Hearth] Could not connect: {e}. Running in mock mode.").into(),
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

#[component]
pub fn ConnectionBadge() -> impl IntoView {
    let ctx = use_holochain();

    view! {
        <span class=move || format!("connection-badge {}", ctx.status.get().css_class())>
            <span class="status-dot"></span>
            {move || ctx.status.get().label()}
        </span>
    }
}
