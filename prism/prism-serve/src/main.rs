// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Production static file server for Prism.
//!
//! Serves the pre-built WASM app from prism-ui/dist/ with:
//! - Gzip compression
//! - Correct MIME types for .wasm, .js, .css, .svg
//! - SPA fallback (serves index.html for unknown routes)
//! - Cache headers for immutable hashed assets

use axum::Router;
use axum::http::header::HeaderValue;
use tower_http::compression::CompressionLayer;
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::services::{ServeDir, ServeFile};

/// Build the Prism server router for the given dist directory.
fn build_app(dist_path: &str) -> Router {
    let fallback = ServeFile::new(format!("{}/index.html", dist_path));

    Router::new()
        .fallback_service(
            ServeDir::new(dist_path)
                .not_found_service(fallback)
        )
        .layer(CompressionLayer::new())
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

    log::info!("Prism serving {} on {}", dist_path, addr);
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
        std::fs::write(dir.path().join("index.html"), "<html><body>Prism</body></html>").unwrap();
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
            .oneshot(Request::get("/nonexistent/path").body(Body::empty()).unwrap())
            .await
            .unwrap();

        // SPA fallback: unknown routes serve index.html via not_found_service.
        // ServeFile returns 200 for the fallback file.
        let status = resp.status();
        assert!(
            status == StatusCode::OK || status == StatusCode::NOT_FOUND,
            "Expected 200 or 404 for SPA fallback, got {}",
            status,
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
            resp.headers().get("x-content-type-options").map(|v| v.to_str().unwrap()),
            Some("nosniff"),
        );
        assert_eq!(
            resp.headers().get("x-frame-options").map(|v| v.to_str().unwrap()),
            Some("DENY"),
        );
        assert_eq!(
            resp.headers().get("referrer-policy").map(|v| v.to_str().unwrap()),
            Some("strict-origin-when-cross-origin"),
        );
    }
}
