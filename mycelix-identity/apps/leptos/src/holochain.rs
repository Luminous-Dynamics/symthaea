// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Holochain conductor connection — identical pattern to Hearth.
//!
//! Connects to conductor via WebSocket, falls back to mock mode.
//! All zome calls target the `identity` role within `mycelix-unified`.

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

    pub async fn call_zome<I: Serialize, O: DeserializeOwned>(
        &self,
        zome: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<O, String> {
        let transport_ref = self.transport.borrow();
        let transport = transport_ref
            .as_ref()
            .ok_or_else(|| "Not connected to conductor".to_string())?;

        let payload = encode(input).map_err(|e| format!("Encode error: {e}"))?;
        let response = transport
            .call_zome("identity", zome, fn_name, payload)
            .await
            .map_err(|e| format!("Zome call {zome}.{fn_name} failed: {e}"))?;
        decode(&response).map_err(|e| format!("Decode error for {zome}.{fn_name}: {e}"))
    }
}

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
            &format!("[Identity] Connecting to conductor at {url}...").into(),
        );

        let ws_transport = BrowserWsTransport::new();
        let config = ConnectConfig {
            url,
            app_id: "mycelix-unified".to_string(),
            auth_token: token.map(|s| s.into_bytes()),
        };

        match ws_transport.connect(config).await {
            Ok(()) => {
                web_sys::console::log_1(&"[Identity] Connected to conductor!".into());
                *transport_for_connect.borrow_mut() = Some(ws_transport);
                set_status.set(ConnectionStatus::Connected);
            }
            Err(e) => {
                web_sys::console::log_1(
                    &format!("[Identity] Could not connect: {e}. Running in mock mode.").into(),
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
