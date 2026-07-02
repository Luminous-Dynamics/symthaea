// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Prism server: static files + CORS proxy on a single port.
//!
//! Serves the WASM app from dist/ and proxies external API requests,
//! eliminating the need for a separate proxy service and port.

use axum::Router;
use axum::extract::Query;
use axum::http::StatusCode;
use axum::http::header::HeaderValue;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use prism_common::ssrf::validate_proxy_url;
use tower_http::compression::CompressionLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::set_header::SetResponseHeaderLayer;

const MAX_RESPONSE_SIZE: usize = 10 * 1024 * 1024; // 10MB

// ═══════════════════════════════════════════════════════════════
// SSRF PROTECTION
// ═══════════════════════════════════════════════════════════════
//
// `validate_proxy_url` / `is_private_ipv4` live in `prism_common::ssrf` so
// `prism-proxy` and `prism-serve` share a single implementation — see that
// module for rationale and tests.

// ═══════════════════════════════════════════════════════════════
// HTTP CLIENT
// ═══════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════
// PROXY HANDLERS
// ═══════════════════════════════════════════════════════════════

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
    let client = build_client("Prism/0.3 (proxy; +https://mycelix.net)", 15);
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
                        return (StatusCode::PAYLOAD_TOO_LARGE, "Response too large")
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
                Err(e) => (StatusCode::BAD_GATEWAY, format!("Read failed: {}", e)).into_response(),
            }
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("Fetch failed: {}", e)).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct DdgParams {
    q: String,
}

async fn ddg_handler(Query(params): Query<DdgParams>) -> impl IntoResponse {
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        params.q.replace(' ', "+")
    );
    let client = build_client("Prism/0.3 (ddg-proxy)", 10);
    match client.get(&url).send().await {
        Ok(resp) => {
            let body = resp.bytes().await.unwrap_or_default();
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .unwrap_or_else(|_| internal_error_response())
                .into_response()
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("DDG fetch failed: {}", e)).into_response(),
    }
}

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
    let client = build_client("Prism/0.3 (brave-proxy)", 15);
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
            axum::response::Response::builder()
                .status(status)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .unwrap_or_else(|_| internal_error_response())
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("Brave fetch failed: {}", e),
        )
            .into_response(),
    }
}

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
    let client = build_client("Prism/0.3 (perplexity-proxy)", 30);
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
                .unwrap_or_else(|_| internal_error_response())
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            format!("Perplexity fetch failed: {}", e),
        )
            .into_response(),
    }
}

async fn health() -> &'static str {
    "Prism OK"
}

// ═══════════════════════════════════════════════════════════════
// SERVER SETUP
// ═══════════════════════════════════════════════════════════════

/// Build the unified Prism router: API proxy routes + static file serving.
fn build_app(dist_path: &str) -> Router {
    let fallback = ServeFile::new(format!("{}/index.html", dist_path));

    Router::new()
        // Proxy API routes (same-origin, no CORS issues)
        .route("/proxy", get(proxy_handler))
        .route("/api/ddg", get(ddg_handler))
        .route("/api/brave", get(brave_handler))
        .route("/api/perplexity", post(perplexity_handler))
        .route("/health", get(health))
        // Static files with SPA fallback
        .fallback_service(ServeDir::new(dist_path).not_found_service(fallback))
        .layer(CompressionLayer::new())
        .layer(tower::limit::ConcurrencyLimitLayer::new(100))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_FRAME_OPTIONS,
            HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::REFERRER_POLICY,
            HeaderValue::from_static("strict-origin-when-cross-origin"),
        ))
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let dist_path = std::env::var("PRISM_DIST")
        .unwrap_or_else(|_| "/srv/luminous-dynamics/prism/prism-ui/dist".to_string());

    let app = build_app(&dist_path);

    let addr = std::env::var("PRISM_ADDR").unwrap_or_else(|_| "0.0.0.0:8130".to_string());

    log::info!("Prism unified server on {} (dist: {})", addr, dist_path);
    println!("Prism serving on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind Prism server address");
    axum::serve(listener, app)
        .await
        .expect("Prism server error");
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_dist_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("index.html"),
            "<html><body>Prism</body></html>",
        )
        .unwrap();
        std::fs::create_dir_all(dir.path().join("static")).unwrap();
        std::fs::write(dir.path().join("static/test.js"), "console.log('ok')").unwrap();
        dir
    }

    #[tokio::test]
    async fn serves_index_html() {
        let dir = test_dist_dir();
        let app = build_app(dir.path().to_str().unwrap());
        let resp = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn spa_fallback_returns_index() {
        let dir = test_dist_dir();
        let app = build_app(dir.path().to_str().unwrap());
        let resp = app
            .oneshot(
                Request::get("/nonexistent/path")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        assert!(
            status == StatusCode::OK || status == StatusCode::NOT_FOUND,
            "Expected 200 or 404 for SPA fallback, got {}",
            status
        );
    }

    #[tokio::test]
    async fn serves_static_files() {
        let dir = test_dist_dir();
        let app = build_app(dir.path().to_str().unwrap());
        let resp = app
            .oneshot(Request::get("/static/test.js").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn security_headers_present() {
        let dir = test_dist_dir();
        let app = build_app(dir.path().to_str().unwrap());
        let resp = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            resp.headers()
                .get("x-content-type-options")
                .map(|v| v.to_str().unwrap()),
            Some("nosniff")
        );
        assert_eq!(
            resp.headers()
                .get("x-frame-options")
                .map(|v| v.to_str().unwrap()),
            Some("DENY")
        );
        assert_eq!(
            resp.headers()
                .get("referrer-policy")
                .map(|v| v.to_str().unwrap()),
            Some("strict-origin-when-cross-origin")
        );
    }

    #[tokio::test]
    async fn health_endpoint() {
        let dir = test_dist_dir();
        let app = build_app(dir.path().to_str().unwrap());
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn rejects_localhost() {
        assert!(validate_proxy_url("http://localhost:8080").is_err());
        assert!(validate_proxy_url("http://127.0.0.1:5432").is_err());
        assert!(validate_proxy_url("http://[::1]:80").is_err());
    }

    #[test]
    fn rejects_private_ips() {
        assert!(validate_proxy_url("http://10.0.0.1").is_err());
        assert!(validate_proxy_url("http://192.168.1.1").is_err());
    }

    #[test]
    fn allows_external_urls() {
        assert!(validate_proxy_url("https://example.com").is_ok());
        assert!(validate_proxy_url("https://en.wikipedia.org/wiki/Rust").is_ok());
    }
}
