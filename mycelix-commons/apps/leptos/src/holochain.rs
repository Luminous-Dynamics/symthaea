// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Holochain conductor context for the Commons Leptos app.
//! Role-aware: targets "commons_land" and "commons_care" roles.

use leptos::prelude::*;
use serde::{de::DeserializeOwned, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use send_wrapper::SendWrapper;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::spawn_local;
use mycelix_leptos_client::{BrowserWsTransport, ConnectConfig, HolochainTransport, encode, decode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ConnectionStatus { Disconnected, Connecting, Connected, Mock }

impl ConnectionStatus {
    pub fn css_class(&self) -> &'static str {
        match self { Self::Disconnected => "status-disconnected", Self::Connecting => "status-connecting", Self::Connected => "status-connected", Self::Mock => "status-mock" }
    }
    pub fn label(&self) -> &'static str {
        match self { Self::Disconnected => "Disconnected", Self::Connecting => "Connecting…", Self::Connected => "Connected", Self::Mock => "Mock" }
    }
}

type TransportCell = SendWrapper<Rc<RefCell<Option<BrowserWsTransport>>>>;

#[derive(Clone)]
#[allow(dead_code)]
pub struct HolochainCtx {
    pub status: ReadSignal<ConnectionStatus>,
    set_status: WriteSignal<ConnectionStatus>,
    transport: TransportCell,
}

impl HolochainCtx {
    pub async fn call_zome<I: Serialize, O: DeserializeOwned>(
        &self, role: &str, zome: &str, fn_name: &str, input: &I,
    ) -> Result<O, String> {
        let transport = self.transport.borrow();
        let transport = match transport.as_ref() {
            Some(t) => t.clone(),
            None => return Err(format!("Mock mode: {role}.{zome}.{fn_name}")),
        };
        drop(self.transport.borrow());
        let payload = encode(input).map_err(|e| format!("Encode: {e}"))?;
        let response = transport.call_zome(role, zome, fn_name, payload).await
            .map_err(|e| format!("{role}.{zome}.{fn_name} failed: {e}"))?;
        decode(&response).map_err(|e| format!("Decode: {e}"))
    }
    pub fn is_mock(&self) -> bool { self.status.get_untracked() == ConnectionStatus::Mock }
}

#[component]
pub fn HolochainProvider(children: Children) -> impl IntoView {
    let (status, set_status) = signal(ConnectionStatus::Connecting);
    let transport: TransportCell = SendWrapper::new(Rc::new(RefCell::new(None)));
    let ctx = HolochainCtx { status, set_status, transport: transport.clone() };
    provide_context(ctx);
    let tc = transport.clone();
    spawn_local(async move {
        let url = web_sys::window()
            .and_then(|w| js_sys::Reflect::get(&w, &JsValue::from_str("__HC_CONDUCTOR_URL")).ok().and_then(|v| v.as_string()))
            .unwrap_or_else(|| "ws://localhost:8888".into());
        let token = web_sys::window()
            .and_then(|w| js_sys::Reflect::get(&w, &JsValue::from_str("__HC_AUTH_TOKEN")).ok().and_then(|v| v.as_string()));
        let ws = BrowserWsTransport::new();
        match ws.connect(ConnectConfig { url, app_id: "mycelix-unified".into(), auth_token: token.map(|s| s.into_bytes()) }).await {
            Ok(()) => { *tc.borrow_mut() = Some(ws); set_status.set(ConnectionStatus::Connected); }
            Err(_) => { set_status.set(ConnectionStatus::Mock); }
        }
    });
    children()
}

pub fn use_holochain() -> HolochainCtx { expect_context::<HolochainCtx>() }

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
