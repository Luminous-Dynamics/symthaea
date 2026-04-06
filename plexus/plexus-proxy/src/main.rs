// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CORS proxy for Symthaea Prism.
//!
//! Fetches external URLs on behalf of the WASM app and returns
//! responses with permissive CORS headers.
//!
//! Usage: cargo run -p plexus-proxy
//! Listens on http://127.0.0.1:8131
//!
//! WASM app requests: GET /proxy?url=https://example.com

use axum::{extract::Query, http::StatusCode, response::IntoResponse, routing::get, Router};
use tower_http::cors::CorsLayer;

const MAX_RESPONSE_SIZE: usize = 10 * 1024 * 1024; // 10MB

#[derive(serde::Deserialize)]
struct ProxyParams {
    url: String,
}

async fn proxy_handler(Query(params): Query<ProxyParams>) -> impl IntoResponse {
    let url = &params.url;
    log::info!("Proxying: {}", url);

    let client = reqwest::Client::builder()
        .user_agent("SymthaePrism/0.1 (proxy; +https://luminousdynamics.org)")
        .redirect(reqwest::redirect::Policy::limited(5))
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap();

    match client.get(url).send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("text/html")
                .to_string();

            match resp.bytes().await {
                Ok(body) => {
                    if body.len() > MAX_RESPONSE_SIZE {
                        return (
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "Response too large".to_string(),
                        )
                            .into_response();
                    }
                    let response = axum::response::Response::builder()
                        .status(status)
                        .header("content-type", content_type)
                        .header("x-prism-proxied", "true")
                        .body(axum::body::Body::from(body))
                        .unwrap();
                    response.into_response()
                }
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    format!("Failed to read response body: {}", e),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("Fetch failed: {}", e),
        )
            .into_response(),
    }
}

/// DuckDuckGo Instant Answer proxy — bypasses CORS.
#[derive(serde::Deserialize)]
struct DdgParams {
    q: String,
}

async fn ddg_handler(Query(params): Query<DdgParams>) -> impl IntoResponse {
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        params.q.replace(' ', "+")
    );
    log::info!("DDG proxy: {}", params.q);

    let client = reqwest::Client::builder()
        .user_agent("SymthaePrism/0.2 (ddg-proxy)")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();

    match client.get(&url).send().await {
        Ok(resp) => {
            let body = resp.bytes().await.unwrap_or_default();
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .unwrap()
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("DDG fetch failed: {}", e),
        ).into_response(),
    }
}

/// Brave Search proxy — forwards API key from query param as auth header.
#[derive(serde::Deserialize)]
struct BraveParams {
    q: String,
    key: String,
}

async fn brave_handler(Query(params): Query<BraveParams>) -> impl IntoResponse {
    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}",
        params.q.replace(' ', "+")
    );
    log::info!("Brave proxy: {}", params.q);

    let client = reqwest::Client::builder()
        .user_agent("SymthaePrism/0.2 (brave-proxy)")
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap();

    match client
        .get(&url)
        .header("X-Subscription-Token", &params.key)
        .header("Accept", "application/json")
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.bytes().await.unwrap_or_default();
            axum::response::Response::builder()
                .status(status)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .unwrap()
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("Brave fetch failed: {}", e),
        ).into_response(),
    }
}

/// Perplexity proxy — reads API key from custom header, forwards as Bearer token.
async fn perplexity_handler(
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let api_key = headers
        .get("X-Perplexity-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if api_key.is_empty() {
        return (StatusCode::UNAUTHORIZED, "Missing X-Perplexity-Key header").into_response();
    }

    log::info!("Perplexity proxy request");

    let client = reqwest::Client::builder()
        .user_agent("SymthaePrism/0.2 (perplexity-proxy)")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap();

    match client
        .post("https://api.perplexity.ai/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .body(body.to_vec())
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.bytes().await.unwrap_or_default();
            axum::response::Response::builder()
                .status(status)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .unwrap()
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("Perplexity fetch failed: {}", e),
        ).into_response(),
    }
}

async fn health() -> &'static str {
    "Symthaea Prism proxy OK"
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let cors = CorsLayer::permissive();

    let app = Router::new()
        .route("/proxy", get(proxy_handler))
        .route("/api/ddg", get(ddg_handler))
        .route("/api/brave", get(brave_handler))
        .route("/api/perplexity", axum::routing::post(perplexity_handler))
        .route("/health", get(health))
        .layer(cors);

    let addr = "127.0.0.1:8131";
    log::info!("Prism proxy listening on http://{}", addr);
    println!("Symthaea Prism CORS proxy on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
