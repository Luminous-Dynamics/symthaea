// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CORS proxy for Prism.
//!
//! Fetches external URLs on behalf of the WASM app and returns
//! responses with permissive CORS headers.
//!
//! Usage: cargo run -p prism-proxy
//! Listens on http://127.0.0.1:8131
//!
//! WASM app requests: GET /proxy?url=https://example.com

use axum::{Router, extract::Query, http::StatusCode, response::IntoResponse, routing::get};
use prism_common::ssrf::validate_proxy_url;
use tower_http::cors::{AllowOrigin, CorsLayer};

const MAX_RESPONSE_SIZE: usize = 10 * 1024 * 1024; // 10MB

fn build_client(user_agent: &str, timeout_secs: u64) -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(user_agent)
        .redirect(reqwest::redirect::Policy::limited(5))
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .expect("HTTP client TLS init failed")
}

fn internal_error_response() -> axum::response::Response<axum::body::Body> {
    let mut resp = axum::response::Response::new(axum::body::Body::from("Internal error"));
    *resp.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
    resp
}

fn json_response(
    status: u16,
    body: axum::body::Bytes,
) -> axum::response::Response<axum::body::Body> {
    axum::response::Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body))
        .unwrap_or_else(|_| internal_error_response())
}

#[derive(serde::Deserialize)]
struct ProxyParams {
    url: String,
}

async fn proxy_handler(Query(params): Query<ProxyParams>) -> impl IntoResponse {
    let validated = match validate_proxy_url(&params.url) {
        Ok(u) => u,
        Err(reason) => {
            log::warn!("Blocked proxy request to {}: {}", params.url, reason);
            return (StatusCode::FORBIDDEN, reason.to_string()).into_response();
        }
    };

    log::info!("Proxying: {}", validated);

    let client = build_client("Prism/0.1 (proxy; +https://luminousdynamics.org)", 15);

    match client.get(validated.as_str()).send().await {
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
                    axum::response::Response::builder()
                        .status(status)
                        .header("content-type", content_type)
                        .header("x-prism-proxied", "true")
                        .body(axum::body::Body::from(body))
                        .unwrap_or_else(|_| internal_error_response())
                        .into_response()
                }
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    format!("Failed to read response body: {}", e),
                )
                    .into_response(),
            }
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("Fetch failed: {}", e)).into_response(),
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

    let client = build_client("Prism/0.2 (ddg-proxy)", 10);

    match client.get(&url).send().await {
        Ok(resp) => {
            let body = resp.bytes().await.unwrap_or_default();
            json_response(200, body).into_response()
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("DDG fetch failed: {}", e)).into_response(),
    }
}

/// Brave Search proxy — reads API key from X-Brave-Key header, forwards as auth.
#[derive(serde::Deserialize)]
struct BraveParams {
    q: String,
}

async fn brave_handler(
    headers: axum::http::HeaderMap,
    Query(params): Query<BraveParams>,
) -> impl IntoResponse {
    let api_key = headers
        .get("X-Brave-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if api_key.is_empty() {
        return (StatusCode::UNAUTHORIZED, "Missing X-Brave-Key header").into_response();
    }

    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}",
        params.q.replace(' ', "+")
    );
    log::info!("Brave proxy: {}", params.q);

    let client = build_client("Prism/0.2 (brave-proxy)", 15);

    match client
        .get(&url)
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json")
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.bytes().await.unwrap_or_default();
            json_response(status, body).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("Brave fetch failed: {}", e),
        )
            .into_response(),
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

    let client = build_client("Prism/0.2 (perplexity-proxy)", 30);

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
            json_response(status, body).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("Perplexity fetch failed: {}", e),
        )
            .into_response(),
    }
}

async fn health() -> &'static str {
    "Prism proxy OK"
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let allowed_origins = AllowOrigin::list([
        "http://localhost:8130".parse().unwrap(),
        "https://prism.mycelix.net".parse().unwrap(),
        "https://prism.luminousdynamics.io".parse().unwrap(),
        "https://app.mycelix.net".parse().unwrap(),
    ]);

    let cors = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        .route("/proxy", get(proxy_handler))
        .route("/api/ddg", get(ddg_handler))
        .route("/api/brave", get(brave_handler))
        .route("/api/perplexity", axum::routing::post(perplexity_handler))
        .route("/health", get(health))
        .layer(cors)
        .layer(tower::limit::ConcurrencyLimitLayer::new(50));

    let addr = "127.0.0.1:8131";
    log::info!("Prism proxy listening on http://{}", addr);
    println!("Prism CORS proxy on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind port 8131");
    axum::serve(listener, app).await.expect("Server error");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_localhost() {
        assert!(validate_proxy_url("http://localhost:8080").is_err());
        assert!(validate_proxy_url("http://127.0.0.1:5432").is_err());
        assert!(validate_proxy_url("http://[::1]:80").is_err());
        assert!(validate_proxy_url("http://0.0.0.0").is_err());
    }

    #[test]
    fn rejects_private_ips() {
        assert!(validate_proxy_url("http://10.0.0.1").is_err());
        assert!(validate_proxy_url("http://172.16.0.1").is_err());
        assert!(validate_proxy_url("http://192.168.1.1").is_err());
    }

    #[test]
    fn rejects_cloud_metadata() {
        assert!(validate_proxy_url("http://169.254.169.254/latest/meta-data/").is_err());
        assert!(validate_proxy_url("http://metadata.google.internal").is_err());
    }

    #[test]
    fn rejects_non_http_schemes() {
        assert!(validate_proxy_url("file:///etc/passwd").is_err());
        assert!(validate_proxy_url("ftp://example.com").is_err());
        assert!(validate_proxy_url("gopher://evil.com").is_err());
    }

    #[test]
    fn rejects_ipv4_mapped_ipv6() {
        // ::ffff:127.0.0.1 — loopback disguised as IPv6
        assert!(validate_proxy_url("http://[::ffff:127.0.0.1]").is_err());
        // ::ffff:192.168.1.1 — private IP disguised as IPv6
        assert!(validate_proxy_url("http://[::ffff:192.168.1.1]").is_err());
        // ::ffff:169.254.169.254 — cloud metadata disguised as IPv6
        assert!(validate_proxy_url("http://[::ffff:169.254.169.254]").is_err());
        // ::ffff:10.0.0.1 — private range disguised as IPv6
        assert!(validate_proxy_url("http://[::ffff:10.0.0.1]").is_err());
    }

    #[test]
    fn rejects_ipv6_link_local_and_ula() {
        // fe80:: — link-local
        assert!(validate_proxy_url("http://[fe80::1]").is_err());
        // fc00:: — unique-local address
        assert!(validate_proxy_url("http://[fc00::1]").is_err());
        // fd00:: — also unique-local
        assert!(validate_proxy_url("http://[fd12::1]").is_err());
    }

    #[test]
    fn allows_valid_external_urls() {
        assert!(validate_proxy_url("https://example.com").is_ok());
        assert!(validate_proxy_url("https://api.duckduckgo.com/?q=test").is_ok());
        assert!(validate_proxy_url("http://en.wikipedia.org/wiki/Rust").is_ok());
    }
}
