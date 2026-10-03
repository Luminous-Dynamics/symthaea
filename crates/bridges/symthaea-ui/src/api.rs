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
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

/// Bound one inbound telemetry text frame before handing it to serde_json.
///
/// The live mental-movie representation is already bounded to at most 32 MiB
/// of decoded RGBA storage; 40 MiB leaves room for its base64 encoding and
/// surrounding JSON while ensuring an oversized websocket frame is rejected
/// before a second full JSON object is materialized in the WASM heap.
const MAX_TELEMETRY_TEXT_BYTES: usize = 40 * 1024 * 1024;

/// Bound one /v1/service HTTP response before JSON parsing.
const MAX_SERVICE_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

fn validate_service_response_len(byte_len: usize) -> Result<(), String> {
    if byte_len > MAX_SERVICE_RESPONSE_BYTES {
        return Err(format!(
            "service response exceeds {} byte bound",
            MAX_SERVICE_RESPONSE_BYTES
        ));
    }
    Ok(())
}

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

async fn parse_service_response(resp: gloo_net::http::Response) -> Result<Value, String> {
    if let Some(length) = resp.headers().get("content-length") {
        if let Ok(length) = length.parse::<usize>() {
            validate_service_response_len(length)?;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let stream = resp
            .body()
            .ok_or_else(|| "service response has no readable body".to_string())?;
        let reader = web_sys::ReadableStreamDefaultReader::new(&stream)
            .map_err(|error| format!("failed to create response reader: {error:?}"))?;
        let mut bytes = Vec::new();

        loop {
            let result = JsFuture::from(reader.read())
                .await
                .map_err(|error| format!("failed to read response chunk: {error:?}"))?;
            let done = web_sys::js_sys::Reflect::get(&result, &JsValue::from_str("done"))
                .map_err(|error| format!("failed to inspect response chunk: {error:?}"))?
                .as_bool()
                .unwrap_or(false);
            if done {
                break;
            }

            let value = web_sys::js_sys::Reflect::get(&result, &JsValue::from_str("value"))
                .map_err(|error| format!("failed to inspect response chunk value: {error:?}"))?;
            let chunk = web_sys::js_sys::Uint8Array::new(&value);
            let chunk_len = chunk.length() as usize;
            let next_len = bytes
                .len()
                .checked_add(chunk_len)
                .ok_or_else(|| "service response size overflow".to_string())?;
            if let Err(error) = validate_service_response_len(next_len) {
                let _ = JsFuture::from(reader.cancel()).await;
                reader.release_lock();
                return Err(error);
            }

            let old_len = bytes.len();
            bytes.resize(next_len, 0);
            chunk.copy_to(&mut bytes[old_len..]);
        }

        reader.release_lock();
        let text = String::from_utf8(bytes)
            .map_err(|error| format!("service response was not valid UTF-8: {error}"))?;
        return serde_json::from_str::<Value>(&text)
            .map_err(|error| format!("failed to parse response: {error}"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let text = resp
            .text()
            .await
            .map_err(|e| format!("failed to read response: {e}"))?;
        validate_service_response_len(text.len())?;
        serde_json::from_str::<Value>(&text)
            .map_err(|error| format!("failed to parse response: {error}"))
    }
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
    parse_service_response(resp).await
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
    parse_service_response(resp).await
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
    fn service_response_budget_enforces_boundary() {
        assert!(validate_service_response_len(MAX_SERVICE_RESPONSE_BYTES).is_ok());
        let error = validate_service_response_len(MAX_SERVICE_RESPONSE_BYTES + 1)
            .expect_err("oversized service response must be rejected");
        assert!(error.contains("exceeds"));
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
