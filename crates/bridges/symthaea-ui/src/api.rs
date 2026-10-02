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
use web_sys::Url;

/// HTTP responses are bounded before serde_json parsing. This is intentionally
/// separate from the WebSocket budget because service queries/status should
/// remain small even though telemetry may carry a bounded mental movie.
const MAX_HTTP_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
/// Outbound chat/query content is bounded independently of response limits so
/// an oversized local input cannot create an unbounded JSON request body.
const MAX_QUERY_CONTENT_BYTES: usize = 256 * 1024;
/// Request discriminators are protocol tokens, not arbitrary user content.
const MAX_REQUEST_TYPE_BYTES: usize = 128;
/// The telemetry budget covers the maximum 64 MiB RGBA movie projection plus
/// base64 expansion and JSON framing, while remaining finite before parsing.
const MAX_WS_TEXT_BYTES: usize = 72 * 1024 * 1024;

fn parse_json_with_limit(text: &str, max_bytes: usize) -> Result<Value, String> {
    if text.len() > max_bytes {
        return Err(format!("JSON payload exceeds {max_bytes} byte limit"));
    }
    serde_json::from_str::<Value>(text).map_err(|e| format!("failed to parse response: {e}"))
}

/// Read an HTTP response body with a cheap header-level rejection when the
/// server provides a numeric Content-Length. The body-size check remains
/// mandatory because chunked responses and browser/CORS conditions may omit
/// or hide the header.
async fn read_json_response(
    resp: gloo_net::http::Response,
    max_bytes: usize,
) -> Result<Value, String> {
    if let Some(content_length) = resp.headers().get("content-length") {
        if let Ok(length) = content_length.trim().parse::<u64>() {
            if length > max_bytes as u64 {
                return Err(format!(
                    "HTTP response Content-Length exceeds {max_bytes} byte limit"
                ));
            }
        }
    }
    let text = resp
        .text()
        .await
        .map_err(|e| format!("failed to read response: {e}"))?;
    parse_json_with_limit(&text, max_bytes)
}

/// Parse and canonicalize the user-configured service gateway before it is
/// used for either HTTP or WebSocket traffic.
///
/// The gateway is a trust boundary: the UI is allowed to talk to a configured
/// service, but arbitrary URL syntax must not silently change the destination
/// or acquire credentials/query/fragment semantics when endpoint paths are
/// appended. WHATWG URL parsing is used so normalization follows the browser's
/// own URL model rather than a second ad-hoc parser.
fn gateway_base(gateway: &str) -> Result<Url, String> {
    let gateway = gateway.trim();
    let url = Url::new(gateway).map_err(|_| "gateway must be a valid absolute URL".to_string())?;
    match url.protocol().as_str() {
        "http:" | "https:" => {}
        _ => return Err("gateway must use http or https".to_string()),
    }
    if url.hostname().is_empty() {
        return Err("gateway must include a hostname".to_string());
    }
    if !url.username().is_empty() || !url.password().is_empty() {
        return Err("gateway credentials are not permitted".to_string());
    }
    if !url.search().is_empty() || !url.hash().is_empty() {
        return Err("gateway query and fragment are not permitted".to_string());
    }
    Ok(url)
}

/// Build one endpoint from the same canonical gateway representation used by
/// HTTP and WebSocket callers. This prevents the two transports from drifting
/// apart when a gateway contains a path prefix or non-default port.
fn gateway_endpoint(gateway: &str, path: &str) -> Result<String, String> {
    let url = gateway_base(gateway)?;
    let mut base = url.to_string();
    while base.ends_with('/') {
        base.pop();
    }
    Ok(format!("{base}/{path}"))
}

/// Send one `{"type":"query","content":...}` request to `POST /v1/service`
/// and return the parsed JSON response (a `Response::QueryResponse` or
/// `Response::Error` per the wire protocol).
pub async fn send_query(gateway: &str, content: &str) -> Result<Value, String> {
    if content.len() > MAX_QUERY_CONTENT_BYTES {
        return Err(format!("query content exceeds {MAX_QUERY_CONTENT_BYTES} byte limit"));
    }
    let url = gateway_endpoint(gateway, "v1/service")?;
    let body = serde_json::json!({ "type": "query", "content": content });
    let resp = Request::post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .map_err(|e| format!("failed to encode request: {e}"))?
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    read_json_response(resp, MAX_HTTP_RESPONSE_BYTES).await
}

/// One request/response round-trip for status/introspect/etc — same shape
/// as `send_query` but for the request types that take no `content`.
pub async fn send_simple(gateway: &str, request_type: &str) -> Result<Value, String> {
    if request_type.len() > MAX_REQUEST_TYPE_BYTES {
        return Err(format!("request type exceeds {MAX_REQUEST_TYPE_BYTES} byte limit"));
    }
    let url = gateway_endpoint(gateway, "v1/service")?;
    let body = serde_json::json!({ "type": request_type });
    let resp = Request::post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .map_err(|e| format!("failed to encode request: {e}"))?
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    read_json_response(resp, MAX_HTTP_RESPONSE_BYTES).await
}

/// Open the live telemetry WebSocket and invoke `on_message` for each
/// `CycleMetadata` JSON payload received. Runs until the socket closes;
/// callers should scope this future to the component owner with
/// `leptos::task::spawn_local_scoped_with_cancellation` so the WebSocket
/// cannot outlive the UI owner. The presentation layer should additionally
/// gate callbacks by telemetry-session generation when multiple streams can
/// overlap during reconnects or gateway changes.
pub async fn stream_telemetry(
    gateway: &str,
    mut on_message: impl FnMut(Value),
    on_connected: impl FnOnce(),
) -> bool {
    let mut ws_url = match gateway_endpoint(gateway, "v1/ws/live") {
        Ok(url) => url,
        Err(e) => {
            leptos::logging::error!("telemetry gateway rejected: {e}");
            return false;
        }
    };
    if ws_url.starts_with("http://") {
        ws_url.replace_range(..7, "ws://");
    } else if ws_url.starts_with("https://") {
        ws_url.replace_range(..8, "wss://");
    } else {
        leptos::logging::error!("gateway protocol could not be mapped to WebSocket");
        return false;
    }
    let ws = match WebSocket::open(&ws_url) {
        Ok(ws) => ws,
        Err(e) => {
            leptos::logging::error!("telemetry websocket open failed: {e}");
            return false;
        }
    };
    on_connected();
    let (_write, mut read) = ws.split();
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if text.len() > MAX_WS_TEXT_BYTES {
                    leptos::logging::warn!(
                        "telemetry websocket message exceeded {} byte limit; closing stream",
                        MAX_WS_TEXT_BYTES
                    );
                    break;
                }
                match serde_json::from_str::<Value>(&text) {
                    Ok(v) => on_message(v),
                    Err(e) => leptos::logging::warn!("telemetry payload was not JSON: {e}"),
                }
            },
            Ok(Message::Bytes(_)) => {}
            Err(e) => {
                leptos::logging::warn!("telemetry websocket error: {e}");
                break;
            }
        }
    }
    true
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_limit_accepts_boundary_and_rejects_overflow() {
        let text = "{\"ok\":true}";
        assert!(parse_json_with_limit(text, text.len()).is_ok());
        assert!(parse_json_with_limit(text, text.len() - 1).is_err());
    }

    #[test]
    fn json_limit_rejects_malformed_payload() {
        assert!(parse_json_with_limit("{", MAX_HTTP_RESPONSE_BYTES).is_err());
    }

    #[test]
    fn outbound_query_and_request_type_limits_are_explicit() {
        assert!("x".repeat(MAX_QUERY_CONTENT_BYTES).len() <= MAX_QUERY_CONTENT_BYTES);
        assert!("x".repeat(MAX_QUERY_CONTENT_BYTES + 1).len() > MAX_QUERY_CONTENT_BYTES);
        assert!("x".repeat(MAX_REQUEST_TYPE_BYTES + 1).len() > MAX_REQUEST_TYPE_BYTES);
    }
}
