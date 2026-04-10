// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Browser WebSocket transport for Holochain conductor communication.
//!
//! Uses `web-sys::WebSocket` to send binary (MessagePack) frames to the
//! Holochain conductor. Request/response correlation is handled via a
//! monotonic request ID and an in-memory pending-request map.
//!
//! # Wire protocol
//!
//! The conductor App API expects:
//! 1. **Authentication** (optional) — send `AppRequest::Authenticate` with
//!    the token issued by the admin API.
//! 2. **App info discovery** — send `AppRequest::AppInfo` to get the
//!    `role_name → CellId` mapping.
//! 3. **Zome calls** — send `AppRequest::CallZome` with the resolved `CellId`,
//!    nonce, expiry, and unsigned provenance/signature.

use crate::error::ClientError;
use crate::transport::HolochainTransport;
use crate::types::{
    AppRequest, AppResponse, CallZomeRequestWire, CellId, CellInfoVariant, ConnectConfig,
    ConnectionStatus, ReconnectConfig, WireRequest, WireResponse,
};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;

use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use js_sys::{ArrayBuffer, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{MessageEvent, WebSocket};

// ---------------------------------------------------------------------------
// Pending request tracking
// ---------------------------------------------------------------------------

type PendingMap = HashMap<u64, PendingRequest>;

struct PendingRequest {
    /// Waker to resolve the future when the response arrives.
    resolve: Box<dyn FnOnce(Result<Vec<u8>, ClientError>)>,
}

// ---------------------------------------------------------------------------
// Internal shared state
// ---------------------------------------------------------------------------

struct Inner {
    ws: Option<WebSocket>,
    status: ConnectionStatus,
    next_id: u64,
    pending: PendingMap,
    /// Role name → CellId mapping, populated by app_info after connect.
    cell_map: HashMap<String, CellId>,
    /// Agent public key from the first cell, used as provenance for unsigned calls.
    agent_pub_key: Option<Vec<u8>>,
    /// Closures that must be kept alive for the WebSocket callbacks.
    /// Stored as JsValue to avoid type-parameter complexity.
    _callbacks: Vec<JsValue>,
    /// Signal handler — called when the conductor sends a signal (not a response).
    signal_handler: Option<Box<dyn Fn(Vec<u8>)>>,
    /// Stored connect config for reconnection attempts.
    connect_config: Option<ConnectConfig>,
    /// Current reconnect attempt counter (0 = not reconnecting).
    reconnect_attempt: u32,
}

impl Default for Inner {
    fn default() -> Self {
        Self {
            ws: None,
            status: ConnectionStatus::Disconnected,
            next_id: 1,
            pending: HashMap::new(),
            cell_map: HashMap::new(),
            agent_pub_key: None,
            _callbacks: Vec::new(),
            signal_handler: None,
            connect_config: None,
            reconnect_attempt: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// BrowserWsTransport
// ---------------------------------------------------------------------------

/// WebSocket-based transport for browser WASM targets.
///
/// This transport opens a binary WebSocket to the Holochain conductor,
/// authenticates (if a token is provided), discovers the role→cell_id
/// mapping via `app_info`, and then sends properly-formed zome call
/// requests with the correct `CellId`.
///
/// # Thread safety
///
/// Browser WASM is single-threaded, so this uses `Rc<RefCell<_>>` instead
/// of `Arc<Mutex<_>>`. This type is `!Send` and `!Sync`, which is correct
/// for the browser environment.
///
/// # Reconnection
///
/// If `ConnectConfig.reconnect` is set to `Some(ReconnectConfig)`, the transport
/// will automatically attempt to reconnect when the WebSocket closes. It uses
/// exponential backoff (base_delay_ms doubled each attempt, capped at max_delay_ms).
/// Pending requests at the time of disconnect are failed immediately.
/// If reconnect is `None`, the transport enters `Disconnected` and subsequent
/// `call_zome` calls return [`ClientError::NotConnected`].
#[derive(Clone)]
pub struct BrowserWsTransport {
    inner: Rc<RefCell<Inner>>,
}

impl BrowserWsTransport {
    /// Create a new transport instance. Does not connect immediately.
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(Inner::default())),
        }
    }

    /// Register a callback for Holochain signals (app signals broadcast by the conductor).
    ///
    /// Signals are fire-and-forget messages from the conductor that don't correspond
    /// to any request. They carry MessagePack-encoded payloads from zome signal handlers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// transport.set_signal_handler(|payload: Vec<u8>| {
    ///     // Decode and route signal to the appropriate domain
    ///     web_sys::console::log_1(&format!("Signal: {} bytes", payload.len()).into());
    /// });
    /// ```
    pub fn set_signal_handler<F: Fn(Vec<u8>) + 'static>(&self, handler: F) {
        self.inner.borrow_mut().signal_handler = Some(Box::new(handler));
    }

    /// Return the connected agent pubkey bytes discovered from app info.
    pub fn connected_agent_pub_key(&self) -> Option<Vec<u8>> {
        self.inner.borrow().agent_pub_key.clone()
    }

    /// Return the connected agent pubkey in the same `u...` base64url form
    /// used by Holochain hash display formatting.
    pub fn connected_agent_pub_key_b64(&self) -> Option<String> {
        self.connected_agent_pub_key()
            .map(|bytes| format!("u{}", URL_SAFE_NO_PAD.encode(bytes)))
    }

    /// Return the connected Mycelix DID derived from the conductor agent key.
    pub fn connected_agent_did(&self) -> Option<String> {
        self.connected_agent_pub_key_b64()
            .map(|key| format!("did:mycelix:{key}"))
    }

    /// Allocate the next request ID.
    fn next_id(inner: &mut Inner) -> u64 {
        let id = inner.next_id;
        inner.next_id = inner.next_id.wrapping_add(1);
        id
    }

    /// Send a request envelope over the WebSocket and await the response bytes.
    ///
    /// This is the core request/response primitive used by authenticate,
    /// app_info, and call_zome. If a `request_timeout_ms` is configured, the
    /// request will be failed with `ClientError::Timeout` after that duration.
    fn send_request(
        inner_rc: &Rc<RefCell<Inner>>,
        request_type: &str,
        data: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ClientError>>>> {
        let inner = Rc::clone(inner_rc);
        let request_type = request_type.to_string();

        Box::pin(async move {
            let (id, wire_bytes, timeout_ms) = {
                let mut state = inner.borrow_mut();

                let ws = state.ws.as_ref().ok_or(ClientError::NotConnected)?;
                if ws.ready_state() != WebSocket::OPEN {
                    return Err(ClientError::NotConnected);
                }

                let id = Self::next_id(&mut state);
                let envelope = WireRequest {
                    id,
                    request_type,
                    data,
                };

                let wire_bytes = rmp_serde::to_vec_named(&envelope)
                    .map_err(|e| ClientError::SerializationError(e.to_string()))?;

                let timeout_ms = state.connect_config.as_ref()
                    .and_then(|c| c.request_timeout_ms)
                    .unwrap_or(30_000);

                (id, wire_bytes, timeout_ms)
            };

            // Create a oneshot channel for the response
            let (tx, rx) = futures::channel::oneshot::channel();

            {
                let mut state = inner.borrow_mut();
                state.pending.insert(
                    id,
                    PendingRequest {
                        resolve: Box::new(move |result| {
                            let _ = tx.send(result);
                        }),
                    },
                );

                let ws = state.ws.as_ref().ok_or(ClientError::NotConnected)?;
                ws.send_with_u8_array(&wire_bytes)
                    .map_err(|e| ClientError::WebSocketError(format!("{e:?}")))?;
            }

            // Schedule a timeout that removes the pending request
            let inner_for_timeout = Rc::clone(&inner);
            let timeout_closure = Closure::once(move || {
                let mut state = inner_for_timeout.borrow_mut();
                if let Some(pending) = state.pending.remove(&id) {
                    (pending.resolve)(Err(ClientError::Timeout(timeout_ms)));
                }
            });
            if let Some(window) = web_sys::window() {
                let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                    timeout_closure.as_ref().unchecked_ref(),
                    timeout_ms as i32,
                );
                timeout_closure.forget();
            }

            rx.await
                .map_err(|_| ClientError::WebSocketError("Response channel dropped".to_string()))?
        })
    }

    /// Set up WebSocket event callbacks (onmessage, onerror, onclose).
    fn attach_callbacks(inner_rc: &Rc<RefCell<Inner>>, ws: &WebSocket) {
        let mut callbacks = Vec::new();

        // -- onmessage: decode response and resolve pending future --
        {
            let inner = Rc::clone(inner_rc);
            let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                let data = event.data();

                // Binary frame -> ArrayBuffer
                let bytes = if let Ok(buf) = data.dyn_into::<ArrayBuffer>() {
                    let arr = Uint8Array::new(&buf);
                    let mut vec = vec![0u8; arr.length() as usize];
                    arr.copy_to(&mut vec);
                    vec
                } else {
                    // Not a binary frame -- ignore (could be text heartbeat)
                    return;
                };

                // Decode the wire response envelope
                let response: WireResponse = match rmp_serde::from_slice(&bytes) {
                    Ok(r) => r,
                    Err(e) => {
                        web_sys::console::warn_1(
                            &format!("Failed to decode conductor response: {e}").into(),
                        );
                        return;
                    }
                };

                // Check if this is a signal (fire-and-forget from conductor)
                if response.response_type == "signal" {
                    let state = inner.borrow();
                    if let Some(ref handler) = state.signal_handler {
                        handler(response.data);
                    }
                    return;
                }

                // Resolve the pending request
                let mut state = inner.borrow_mut();
                if let Some(pending) = state.pending.remove(&response.id) {
                    if let Some(err_msg) = response.error {
                        (pending.resolve)(Err(ClientError::ZomeCallFailed(err_msg)));
                    } else {
                        (pending.resolve)(Ok(response.data));
                    }
                } else {
                    web_sys::console::warn_1(
                        &format!("Response for unknown request ID: {}", response.id).into(),
                    );
                }
            });
            ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
            callbacks.push(onmessage.into_js_value());
        }

        // -- onerror --
        {
            let inner = Rc::clone(inner_rc);
            let onerror = Closure::<dyn FnMut(web_sys::ErrorEvent)>::new(
                move |event: web_sys::ErrorEvent| {
                    let msg = event.message();
                    let mut state = inner.borrow_mut();
                    state.status = ConnectionStatus::Error(msg.clone());

                    // Fail all pending requests
                    let pending: Vec<_> = state.pending.drain().collect();
                    drop(state);
                    for (_, req) in pending {
                        (req.resolve)(Err(ClientError::WebSocketError(msg.clone())));
                    }
                },
            );
            ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            callbacks.push(onerror.into_js_value());
        }

        // -- onclose --
        {
            let inner = Rc::clone(inner_rc);
            let onclose = Closure::<dyn FnMut(web_sys::CloseEvent)>::new(
                move |event: web_sys::CloseEvent| {
                    let reason = if event.reason().is_empty() {
                        format!("WebSocket closed (code {})", event.code())
                    } else {
                        event.reason()
                    };

                    let mut state = inner.borrow_mut();
                    state.ws = None;

                    // Check if auto-reconnect is configured
                    let reconnect_config = state.connect_config.as_ref()
                        .and_then(|c| c.reconnect.clone());

                    if let Some(ref rc) = reconnect_config {
                        let attempt = state.reconnect_attempt + 1;
                        if attempt <= rc.max_attempts {
                            state.reconnect_attempt = attempt;
                            state.status = ConnectionStatus::Reconnecting {
                                attempt,
                                max_attempts: rc.max_attempts,
                            };
                            let delay = (rc.base_delay_ms * 2u32.saturating_pow(attempt - 1))
                                .min(rc.max_delay_ms);
                            let config = state.connect_config.clone().unwrap();
                            let inner_for_reconnect = Rc::clone(&inner);

                            // Fail pending requests (they won't survive the reconnect gap)
                            let pending: Vec<_> = state.pending.drain().collect();
                            drop(state);
                            for (_, req) in pending {
                                (req.resolve)(Err(ClientError::ConnectionFailed(reason.clone())));
                            }

                            web_sys::console::log_1(&format!(
                                "[HC] Reconnect attempt {attempt}/{} in {delay}ms",
                                rc.max_attempts
                            ).into());

                            // Schedule reconnect after delay
                            let closure = Closure::once(move || {
                                let transport = BrowserWsTransport { inner: inner_for_reconnect };
                                wasm_bindgen_futures::spawn_local(async move {
                                    match transport.connect(config).await {
                                        Ok(()) => {
                                            web_sys::console::log_1(
                                                &"[HC] Reconnected successfully".into()
                                            );
                                            transport.inner.borrow_mut().reconnect_attempt = 0;
                                        }
                                        Err(e) => {
                                            web_sys::console::warn_1(
                                                &format!("[HC] Reconnect failed: {e}").into()
                                            );
                                            // on_close will fire again → next attempt
                                        }
                                    }
                                });
                            });
                            let _ = web_sys::window().map(|w: web_sys::Window| {
                                let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(
                                    closure.as_ref().unchecked_ref(),
                                    delay as i32,
                                );
                                closure.forget();
                            });

                            return;
                        }
                    }

                    // No reconnect or exhausted attempts: go to Disconnected
                    state.status = ConnectionStatus::Disconnected;
                    let pending: Vec<_> = state.pending.drain().collect();
                    drop(state);
                    for (_, req) in pending {
                        (req.resolve)(Err(ClientError::ConnectionFailed(reason.clone())));
                    }
                },
            );
            ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
            callbacks.push(onclose.into_js_value());
        }

        // Store callbacks to prevent GC
        inner_rc.borrow_mut()._callbacks = callbacks;
    }
}

impl Default for BrowserWsTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl HolochainTransport for BrowserWsTransport {
    fn connect(
        &self,
        config: ConnectConfig,
    ) -> Pin<Box<dyn Future<Output = Result<(), ClientError>>>> {
        let inner = Rc::clone(&self.inner);

        Box::pin(async move {
            // If already connected, no-op
            {
                let state = inner.borrow();
                if state.ws.is_some() && state.status == ConnectionStatus::Connected {
                    return Ok(());
                }
            }

            // Create WebSocket
            let ws = WebSocket::new(&config.url)
                .map_err(|e| ClientError::ConnectionFailed(format!("{e:?}")))?;

            // Set binary type to arraybuffer for MessagePack
            ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

            // Set up callbacks before waiting for open
            Self::attach_callbacks(&inner, &ws);

            // Wait for the WebSocket to open
            let open_promise = js_sys::Promise::new(&mut |resolve, reject| {
                let onopen = Closure::once(move |_: JsValue| {
                    resolve.call0(&JsValue::NULL).unwrap_or(JsValue::UNDEFINED);
                });
                ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
                // Prevent GC -- the closure is consumed after one call
                onopen.forget();

                // Also wire up a rejection on the pre-open error case
                let onerror_reject = Closure::once(move |_: web_sys::ErrorEvent| {
                    reject
                        .call1(&JsValue::NULL, &"Connection failed".into())
                        .unwrap_or(JsValue::UNDEFINED);
                });
                // Note: this overwrites the onerror we set in attach_callbacks,
                // but only until onopen fires. After connection, attach_callbacks
                // re-sets it. We accept this brief window.
                ws.set_onerror(Some(onerror_reject.as_ref().unchecked_ref()));
                onerror_reject.forget();
            });

            JsFuture::from(open_promise)
                .await
                .map_err(|e| ClientError::ConnectionFailed(format!("{e:?}")))?;

            // Re-attach callbacks after the open event (onerror was overwritten)
            Self::attach_callbacks(&inner, &ws);

            // Store the connected WebSocket and config (for reconnect)
            {
                let mut state = inner.borrow_mut();
                state.ws = Some(ws);
                state.status = ConnectionStatus::Connected;
                state.connect_config = Some(config.clone());
            }

            // --- Fix 1: Authentication ---
            if let Some(token) = config.auth_token {
                let auth_payload = AppRequest::Authenticate { token };
                let auth_bytes = rmp_serde::to_vec_named(&auth_payload)
                    .map_err(|e| ClientError::SerializationError(e.to_string()))?;

                let response_bytes = Self::send_request(&inner, "authenticate", auth_bytes).await?;

                // The conductor responds with an empty success or an error.
                // If we got here without error, authentication succeeded.
                // Check for an explicit error in the response.
                if let Ok(resp) = rmp_serde::from_slice::<AppResponse>(&response_bytes) {
                    if let AppResponse::Error(e) = resp {
                        let mut state = inner.borrow_mut();
                        state.status = ConnectionStatus::Disconnected;
                        if let Some(ws) = state.ws.take() {
                            let _ = ws.close();
                        }
                        return Err(ClientError::AuthenticationFailed(e.message));
                    }
                }
            }

            // --- Fix 2: AppInfo discovery ---
            {
                let app_info_payload = AppRequest::AppInfo {
                    installed_app_id: config.app_id.clone(),
                };
                let app_info_bytes = rmp_serde::to_vec_named(&app_info_payload)
                    .map_err(|e| ClientError::SerializationError(e.to_string()))?;

                let response_bytes = Self::send_request(&inner, "request", app_info_bytes).await?;

                // Parse the response to build role→cell_id mapping
                let app_info: AppResponse =
                    rmp_serde::from_slice(&response_bytes).map_err(|e| {
                        ClientError::InvalidResponse(format!(
                            "Failed to parse app_info response: {e}"
                        ))
                    })?;

                match app_info {
                    AppResponse::AppInfo(info) => {
                        let mut state = inner.borrow_mut();
                        let mut first_agent: Option<Vec<u8>> = None;

                        for entry in &info.cell_info {
                            for cell_variant in &entry.cells {
                                if let CellInfoVariant::Provisioned(cell) = cell_variant {
                                    state
                                        .cell_map
                                        .insert(entry.role_name.clone(), cell.cell_id.clone());
                                    // Capture agent pub key from the first cell
                                    if first_agent.is_none() {
                                        first_agent = Some(cell.cell_id.1.clone());
                                    }
                                    break; // Use first provisioned cell per role
                                }
                            }
                        }

                        state.agent_pub_key = first_agent;

                        if state.cell_map.is_empty() {
                            web_sys::console::warn_1(
                                &format!(
                                    "app_info for '{}' returned no provisioned cells",
                                    config.app_id
                                )
                                .into(),
                            );
                        }
                    }
                    AppResponse::Error(e) => {
                        return Err(ClientError::ConnectionFailed(format!(
                            "app_info failed: {}",
                            e.message
                        )));
                    }
                    _ => {
                        return Err(ClientError::InvalidResponse(
                            "Unexpected response type for app_info".into(),
                        ));
                    }
                }
            }

            Ok(())
        })
    }

    fn call_zome(
        &self,
        role_name: &str,
        zome_name: &str,
        fn_name: &str,
        payload: Vec<u8>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ClientError>>>> {
        let inner = Rc::clone(&self.inner);
        let role_name = role_name.to_string();
        let zome_name = zome_name.to_string();
        let fn_name = fn_name.to_string();

        Box::pin(async move {
            // --- Fix 3: Look up cell_id from stored mapping ---
            let (cell_id, agent_pub_key) = {
                let state = inner.borrow();

                let cell_id = state
                    .cell_map
                    .get(&role_name)
                    .ok_or_else(|| ClientError::UnknownRole(role_name.clone()))?
                    .clone();

                let agent_pub_key = state
                    .agent_pub_key
                    .clone()
                    .unwrap_or_else(|| cell_id.1.clone());

                (cell_id, agent_pub_key)
            };

            // --- Fix 4: Proper AppRequest enum serialization ---
            // --- Fix 6: Unsigned calls with zeroed signature, proper nonce ---
            let call_zome_data = CallZomeRequestWire {
                cell_id,
                zome_name,
                fn_name,
                payload,
                cap_secret: None,
                provenance: agent_pub_key,
                signature: vec![0u8; 64], // Unsigned — zeroed signature
                nonce: generate_nonce(),
                expires_at: now_micros() + 5_000_000, // 5 second expiry
            };

            let call_request = AppRequest::CallZome(call_zome_data);
            let call_bytes = rmp_serde::to_vec_named(&call_request)
                .map_err(|e| ClientError::SerializationError(e.to_string()))?;

            let response_bytes = Self::send_request(&inner, "request", call_bytes).await?;

            // Try to decode as AppResponse for proper error handling
            match rmp_serde::from_slice::<AppResponse>(&response_bytes) {
                Ok(AppResponse::ZomeCalled(data)) => Ok(data),
                Ok(AppResponse::Error(e)) => Err(ClientError::ZomeCallFailed(e.message)),
                Ok(other) => Err(ClientError::InvalidResponse(format!(
                    "Unexpected response type for call_zome: {other:?}"
                ))),
                // If we can't parse as AppResponse, return the raw bytes
                // (the WireResponse layer already handled errors)
                Err(_) => Ok(response_bytes),
            }
        })
    }

    fn status(&self) -> ConnectionStatus {
        self.inner.borrow().status.clone()
    }

    fn disconnect(&self) {
        let mut state = self.inner.borrow_mut();
        if let Some(ws) = state.ws.take() {
            let _ = ws.close();
        }
        state.status = ConnectionStatus::Disconnected;
        state.cell_map.clear();
        state.agent_pub_key = None;
        state._callbacks.clear();

        // Fail all pending requests
        let pending: Vec<_> = state.pending.drain().collect();
        drop(state);
        for (_, req) in pending {
            (req.resolve)(Err(ClientError::NotConnected));
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a random 32-byte nonce using the browser's crypto API.
fn generate_nonce() -> Vec<u8> {
    let mut nonce = vec![0u8; 32];

    // Try crypto.getRandomValues first (available in all modern browsers)
    let crypto = js_sys::Reflect::get(&js_sys::global(), &"crypto".into()).ok();
    if let Some(crypto) = crypto {
        if !crypto.is_undefined() {
            let arr = js_sys::Uint8Array::new_with_length(32);
            let _ = js_sys::Reflect::get(&crypto, &"getRandomValues".into())
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .map(|f| f.call1(&crypto, &arr));
            arr.copy_to(&mut nonce);
            return nonce;
        }
    }

    // Fallback: Math.random (NOT cryptographically secure, but functional)
    for byte in &mut nonce {
        *byte = (js_sys::Math::random() * 256.0) as u8;
    }
    nonce
}

/// Current time in microseconds since epoch, using `Date.now()`.
fn now_micros() -> u64 {
    (js_sys::Date::now() * 1000.0) as u64
}
