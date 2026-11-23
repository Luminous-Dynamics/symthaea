// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Client for the `symthaea-service` HTTP gateway
//! (SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md Phases 1+2).
//!
//! Payloads are handled as opaque `serde_json::Value` rather than the
//! actual Rust wire types (`Response`, `CycleMetadata`) — those live in
//! the `symthaea` crate, which is far too heavy to pull into a WASM
//! frontend build, and the plan doc explicitly deferred deciding whether
//! a shared `symthaea-view-types` crate is worth creating until this real
//! constraint was known. It wasn't needed for a v0 that only reads a
//! handful of fields out of each payload.

use futures::StreamExt;
use gloo_net::http::Request;
use gloo_net::websocket::{Message, State};
use gloo_net::websocket::futures::WebSocket;
use serde_json::Value;

/// Send one `{"type":"query","content":...}` request to `POST /v1/service`
/// and return the parsed JSON response (a `Response::QueryResponse` or
/// `Response::Error` per the wire protocol).
pub async fn send_query(gateway: &str, content: &str) -> Result<Value, String> {
    let url = format!("{}/v1/service", gateway.trim_end_matches('/'));
    let body = serde_json::json!({ "type": "query", "content": content });
    let resp = Request::post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .map_err(|e| format!("failed to encode request: {e}"))?
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

/// One request/response round-trip for status/introspect/etc — same shape
/// as `send_query` but for the request types that take no `content`.
pub async fn send_simple(gateway: &str, request_type: &str) -> Result<Value, String> {
    let url = format!("{}/v1/service", gateway.trim_end_matches('/'));
    let body = serde_json::json!({ "type": request_type });
    let resp = Request::post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .map_err(|e| format!("failed to encode request: {e}"))?
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    resp.json::<Value>()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))
}

/// Open the live telemetry WebSocket and invoke `on_message` for each
/// `CycleMetadata` JSON payload received. `on_connection_change(true)` is
/// called only after `WebSocket::open` succeeds; `false` is emitted when the
/// stream ends. Callers own retry/backoff policy and should spawn this via
/// `wasm_bindgen_futures::spawn_local`.
pub async fn stream_telemetry(
    gateway: &str,
    mut on_message: impl FnMut(Value),
    mut on_connection_change: impl FnMut(bool),
) {
    let ws_url = telemetry_ws_url(gateway);
    let ws = match WebSocket::open(&ws_url) {
        Ok(ws) => ws,
        Err(e) => {
            leptos::logging::error!("telemetry websocket open failed: {e}");
            return;
        }
    };

    // WebSocket::open constructs the browser socket but it may still be in
    // CONNECTING state. Wait for the actual OPEN ready state before exposing
    // a connected indicator. Bound the handshake wait so a broken endpoint
    // returns to the caller's retry policy.
    let mut opened = false;
    for _ in 0..40 {
        match ws.state() {
            State::Open => {
                opened = true;
                break;
            }
            State::Closed => break,
            State::Connecting | State::Closing => {
                gloo_timers::future::TimeoutFuture::new(250).await;
            }
        }
    }
    if !opened {
        leptos::logging::warn!("telemetry websocket did not reach OPEN within 10 seconds");
        return;
    }

    on_connection_change(true);
    let (_write, mut read) = ws.split();
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => match serde_json::from_str::<Value>(&text) {
                Ok(v) => on_message(v),
                Err(e) => leptos::logging::warn!("telemetry payload was not JSON: {e}"),
            },
            Ok(Message::Bytes(_)) => {}
            Err(e) => {
                leptos::logging::warn!("telemetry websocket error: {e}");
                break;
            }
        }
    }
    on_connection_change(false);
}

fn telemetry_ws_url(gateway: &str) -> String {
    format!(
        "{}/v1/ws/live",
        gateway
            .trim_end_matches('/')
            .replacen("http://", "ws://", 1)
            .replacen("https://", "wss://", 1)
    )
}

#[cfg(test)]
mod tests {
    use super::telemetry_ws_url;

    #[test]
    fn telemetry_endpoint_maps_http_schemes_to_websocket_schemes() {
        assert_eq!(
            telemetry_ws_url("http://127.0.0.1:8090/"),
            "ws://127.0.0.1:8090/v1/ws/live"
        );
        assert_eq!(
            telemetry_ws_url("https://gateway.example.test"),
            "wss://gateway.example.test/v1/ws/live"
        );
    }

    #[test]
    fn telemetry_endpoint_trims_multiple_trailing_slashes() {
        assert_eq!(
            telemetry_ws_url("http://localhost:8090///"),
            "ws://localhost:8090/v1/ws/live"
        );
    }
}
