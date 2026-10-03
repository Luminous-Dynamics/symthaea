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
use gloo_net::websocket::Message;
use gloo_net::websocket::futures::WebSocket;
use serde_json::Value;

/// Bound one inbound telemetry text frame before handing it to serde_json.
///
/// The live mental-movie representation is already bounded to at most 32 MiB
/// of decoded RGBA storage; 40 MiB leaves room for its base64 encoding and
/// surrounding JSON while ensuring an oversized websocket frame is rejected
/// before a second full JSON object is materialized in the WASM heap.
const MAX_TELEMETRY_TEXT_BYTES: usize = 40 * 1024 * 1024;

fn validate_telemetry_text_len(byte_len: usize) -> Result<(), String> {
    if byte_len > MAX_TELEMETRY_TEXT_BYTES {
        return Err(format!(
            "telemetry websocket frame exceeds {} byte bound",
            MAX_TELEMETRY_TEXT_BYTES
        ));
    }
    Ok(())
}

fn parse_telemetry_text(text: &str) -> Result<Value, String> {
    validate_telemetry_text_len(text.len())?;
    serde_json::from_str::<Value>(text).map_err(|error| format!("telemetry payload was not JSON: {error}"))
}

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
/// `CycleMetadata` JSON payload received. Runs until the socket closes;
/// callers spawn this via `wasm_bindgen_futures::spawn_local`.
pub async fn stream_telemetry(gateway: &str, mut on_message: impl FnMut(Value)) {
    let ws_url = format!(
        "{}/v1/ws/live",
        gateway
            .trim_end_matches('/')
            .replacen("http://", "ws://", 1)
            .replacen("https://", "wss://", 1)
    );
    let ws = match WebSocket::open(&ws_url) {
        Ok(ws) => ws,
        Err(e) => {
            leptos::logging::error!("telemetry websocket open failed: {e}");
            return;
        }
    };
    let (_write, mut read) = ws.split();
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => match parse_telemetry_text(&text) {
                Ok(v) => on_message(v),
                Err(e) => leptos::logging::warn!("{e}"),
            },
            Ok(Message::Bytes(_)) => {}
            Err(e) => {
                leptos::logging::warn!("telemetry websocket error: {e}");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_text_budget_accepts_boundary() {
        assert!(validate_telemetry_text_len(MAX_TELEMETRY_TEXT_BYTES).is_ok());
        assert!(parse_telemetry_text("{}").is_ok());
    }

    #[test]
    fn telemetry_text_budget_rejects_oversize_before_parsing() {
        let error = validate_telemetry_text_len(MAX_TELEMETRY_TEXT_BYTES + 1)
            .expect_err("oversized telemetry must be rejected");
        assert!(error.contains("exceeds"));
    }

    #[test]
    fn telemetry_text_parser_reports_invalid_json() {
        let error = parse_telemetry_text("not-json").expect_err("invalid JSON must be rejected");
        assert!(error.contains("not JSON"));
    }
}
