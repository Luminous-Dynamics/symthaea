// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Browser WebSocket transport for Holochain conductor communication.
//!
//! Uses `web-sys::WebSocket` to send binary (MessagePack) frames to the
//! Holochain conductor. Request/response correlation is handled via a
//! monotonic request ID and an in-memory pending-request map.

use crate::error::ClientError;
use crate::transport::HolochainTransport;
use crate::types::{
    ConnectionStatus, WireRequest, WireResponse, ZomeCallWireData,
};

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
    /// Closures that must be kept alive for the WebSocket callbacks.
    /// Stored as JsValue to avoid type-parameter complexity.
    _callbacks: Vec<JsValue>,
}

impl Default for Inner {
    fn default() -> Self {
        Self {
            ws: None,
            status: ConnectionStatus::Disconnected,
            next_id: 1,
            pending: HashMap::new(),
            _callbacks: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// BrowserWsTransport
// ---------------------------------------------------------------------------

/// WebSocket-based transport for browser WASM targets.
///
/// This transport opens a binary WebSocket to the Holochain conductor,
/// sends MessagePack-encoded zome call requests, and correlates responses
/// by request ID.
///
/// # Thread safety
///
/// Browser WASM is single-threaded, so this uses `Rc<RefCell<_>>` instead
/// of `Arc<Mutex<_>>`. This type is `!Send` and `!Sync`, which is correct
/// for the browser environment.
///
/// # Reconnection
///
/// The current implementation does not auto-reconnect. If the WebSocket
/// closes, subsequent `call_zome` calls will return [`ClientError::NotConnected`].
/// Call [`connect`](BrowserWsTransport::connect) again to re-establish.
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

    /// Allocate the next request ID.
    fn next_id(inner: &mut Inner) -> u64 {
        let id = inner.next_id;
        inner.next_id = inner.next_id.wrapping_add(1);
        id
    }

    /// Set up WebSocket event callbacks (onmessage, onerror, onclose).
    fn attach_callbacks(inner_rc: &Rc<RefCell<Inner>>, ws: &WebSocket) {
        let mut callbacks = Vec::new();

        // -- onmessage: decode response and resolve pending future --
        {
            let inner = Rc::clone(inner_rc);
            let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                let data = event.data();

                // Binary frame → ArrayBuffer
                let bytes = if let Ok(buf) = data.dyn_into::<ArrayBuffer>() {
                    let arr = Uint8Array::new(&buf);
                    let mut vec = vec![0u8; arr.length() as usize];
                    arr.copy_to(&mut vec);
                    vec
                } else {
                    // Not a binary frame — ignore (could be text heartbeat)
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
                    state.status = ConnectionStatus::Disconnected;
                    state.ws = None;

                    // Fail all pending requests
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
    fn connect(&self, url: &str) -> Pin<Box<dyn Future<Output = Result<(), ClientError>>>> {
        let inner = Rc::clone(&self.inner);
        let url = url.to_string();

        Box::pin(async move {
            // If already connected, no-op
            {
                let state = inner.borrow();
                if state.ws.is_some() && state.status == ConnectionStatus::Connected {
                    return Ok(());
                }
            }

            // Create WebSocket
            let ws = WebSocket::new(&url)
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
                // Prevent GC — the closure is consumed after one call
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

            // Store the connected WebSocket
            let mut state = inner.borrow_mut();
            state.ws = Some(ws);
            state.status = ConnectionStatus::Connected;

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
            // Build the wire-protocol zome call data
            let call_data = ZomeCallWireData {
                provenance: vec![0u8; 32], // Unsigned — signing handled by conductor
                role_name: role_name.clone(),
                zome_name: zome_name.clone(),
                fn_name: fn_name.clone(),
                payload,
                cap_secret: None,
                nonce: generate_nonce(),
                expires_at: now_micros() + 5_000_000, // 5 second expiry
            };

            let call_bytes = rmp_serde::to_vec_named(&call_data)
                .map_err(|e| ClientError::SerializationError(e.to_string()))?;

            // Allocate request ID and build envelope
            let (id, wire_bytes) = {
                let mut state = inner.borrow_mut();

                let ws = state.ws.as_ref().ok_or(ClientError::NotConnected)?;
                if ws.ready_state() != WebSocket::OPEN {
                    return Err(ClientError::NotConnected);
                }

                let id = Self::next_id(&mut state);
                let envelope = WireRequest {
                    id,
                    request_type: "zome_call".to_string(),
                    data: call_bytes,
                };

                let wire_bytes = rmp_serde::to_vec_named(&envelope)
                    .map_err(|e| ClientError::SerializationError(e.to_string()))?;

                (id, wire_bytes)
            };

            // Create a future that will be resolved when the response arrives
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

                // Send the binary frame
                let ws = state.ws.as_ref().ok_or(ClientError::NotConnected)?;
                ws.send_with_u8_array(&wire_bytes)
                    .map_err(|e| ClientError::WebSocketError(format!("{e:?}")))?;
            }

            // Await the response
            rx.await.map_err(|_| {
                ClientError::WebSocketError("Response channel dropped".to_string())
            })?
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
