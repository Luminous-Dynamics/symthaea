// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HTTP API for Option 3 — gateway-mediated zome calls.
//!
//! `PULSE_REAL_MODE_PLAN.md` describes three paths to get remote visitors
//! out of mock mode. Option 3 extends this binary with an HTTP surface
//! that proxies zome calls, so the tunnel in front of the gateway only
//! needs to carry HTTPS — not raw Holochain WebSocket. The benefits laid
//! out in the plan:
//!
//!   - HTTP/JSON is easier to secure than raw WS (per-DID JWT, rate-limit,
//!     audit log) and aligns with the Phase 11 "federated gateway mesh"
//!     endgame, where every user runs their own gateway.
//!   - Reuses the gateway binary's systemd hardening + config + DID
//!     signing.
//!   - Doesn't expose the conductor directly — the gateway is the airlock.
//!
//! This module is the *scaffold*. It stands up:
//!
//!   - `POST /api/zome/{zome_name}/{fn_name}`  — generic zome dispatch.
//!   - `GET  /healthz`                         — liveness probe.
//!
//! What it deliberately does NOT do yet (deferred to Phase 5B, separately
//! tracked in `PULSE_REAL_MODE_PLAN.md` §"Option 3"):
//!
//!   - Per-DID JWT authentication (gateway signs with its Dilithium key).
//!   - TLS termination (expected to be handled by the tunnel or nginx).
//!   - Per-DID rate-limiting / quota enforcement.
//!   - Zome-allowlist (which zomes the HTTP API may reach).
//!   - Streaming signals (would need a WS endpoint here too).
//!   - Request-timeout + graceful-shutdown of in-flight calls.
//!
//! Until those land, `HttpApiConfig::enabled` defaults to `false` in the
//! TOML config and the route returns raw stub bytes.

use crate::zome::GenericZomeDispatch;
use crate::{GatewayError, GatewayResult};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;

/// State shared across handlers.
#[derive(Clone)]
pub struct HttpApiState {
    pub bridge: Arc<dyn GenericZomeDispatch>,
    pub max_body_bytes: usize,
}

/// Request body for `POST /api/zome/{zome}/{fn}`.
///
/// `payload_b64` is an opaque blob — the client decides the encoding
/// (msgpack, JSON, bincode) based on the target zome's externs. The
/// gateway does not inspect it.
#[derive(Debug, Serialize, Deserialize)]
pub struct ZomeCallRequest {
    pub payload_b64: String,
}

/// Response body on success.
#[derive(Debug, Serialize, Deserialize)]
pub struct ZomeCallResponse {
    pub result_b64: String,
}

/// Response body on error. Deliberately flat — no stack trace, no
/// internal type details. Operators consult the journal; clients get
/// enough to retry or give up.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Build the axum router. Separated from the server bind so tests can
/// call handlers directly through a `tower::ServiceExt`.
pub fn router(state: HttpApiState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/api/zome/{zome_name}/{fn_name}", post(zome_call))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

/// POST /api/zome/{zome_name}/{fn_name}
///
/// Body: `{"payload_b64": "..."}`.
/// Response 200: `{"result_b64": "..."}`.
/// Response 4xx: `{"error": "..."}`.
async fn zome_call(
    State(state): State<HttpApiState>,
    Path((zome_name, fn_name)): Path<(String, String)>,
    Json(req): Json<ZomeCallRequest>,
) -> Response {
    // Decode the opaque payload. On failure, 400 — we don't want the
    // zome to see malformed bytes.
    let payload = match B64.decode(&req.payload_b64) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("invalid base64: {e}"),
                }),
            )
                .into_response();
        }
    };

    if payload.len() > state.max_body_bytes {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                error: format!("payload {} > max {}", payload.len(), state.max_body_bytes),
            }),
        )
            .into_response();
    }

    match state
        .bridge
        .call_raw_zome(&zome_name, &fn_name, &payload)
        .await
    {
        Ok(result) => (
            StatusCode::OK,
            Json(ZomeCallResponse {
                result_b64: B64.encode(result),
            }),
        )
            .into_response(),
        Err(e) => {
            tracing::warn!(
                %zome_name, %fn_name, error = %e,
                "zome call failed"
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response()
        }
    }
}

/// Bind the router to the configured socket and serve until shutdown.
///
/// The caller is expected to `tokio::spawn` this and race it with the
/// gateway's ctrl-c handler via `tokio::select!`, so that a ctrl-c
/// drops the HTTP server at the same time it drops the SMTP server.
pub async fn serve(
    bridge: Arc<dyn GenericZomeDispatch>,
    cfg: &crate::config::HttpApiConfig,
) -> GatewayResult<()> {
    let addr = SocketAddr::new(cfg.bind, cfg.port);
    let state = HttpApiState {
        bridge,
        max_body_bytes: cfg.max_body_bytes,
    };
    let app = router(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| GatewayError::Config(format!("http_api bind {addr}: {e}")))?;

    tracing::info!(bind = %addr, "http API listening");

    axum::serve(listener, app)
        .await
        .map_err(|e| GatewayError::Config(format!("http_api serve: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zome::StubZomeBridge;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn stub_state() -> (HttpApiState, Arc<StubZomeBridge>) {
        let stub = Arc::new(StubZomeBridge::new());
        let state = HttpApiState {
            bridge: stub.clone() as Arc<dyn GenericZomeDispatch>,
            max_body_bytes: 1024,
        };
        (state, stub)
    }

    #[tokio::test]
    async fn healthz_returns_ok() {
        let (state, _) = stub_state();
        let app = router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64).await.unwrap();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn zome_call_echoes_stub_response() {
        let (state, stub) = stub_state();
        stub.program_response("mycelix_mail", "get_folders", b"programmed".to_vec())
            .await;
        let app = router(state);

        let body = serde_json::to_vec(&ZomeCallRequest {
            payload_b64: B64.encode(b"{}"),
        })
        .unwrap();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/zome/mycelix_mail/get_folders")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        let parsed: ZomeCallResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(B64.decode(&parsed.result_b64).unwrap(), b"programmed");

        let calls = stub.generic_calls.lock().await;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].zome_name, "mycelix_mail");
        assert_eq!(calls[0].fn_name, "get_folders");
        assert_eq!(&calls[0].payload, b"{}");
    }

    #[tokio::test]
    async fn zome_call_rejects_bad_base64() {
        let (state, _) = stub_state();
        let app = router(state);
        let body = br#"{"payload_b64":"!!!not-valid-b64!!!"}"#.to_vec();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/zome/anything/anything")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn zome_call_rejects_oversized_payload() {
        let (mut state, _) = stub_state();
        state.max_body_bytes = 4;
        let app = router(state);
        let body = serde_json::to_vec(&ZomeCallRequest {
            payload_b64: B64.encode(b"12345"),
        })
        .unwrap();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/zome/z/f")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
