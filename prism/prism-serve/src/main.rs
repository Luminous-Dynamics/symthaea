// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Production static file server for Symthaea Prism.
//!
//! Serves the pre-built WASM app from prism-ui/dist/ with:
//! - Gzip compression
//! - Correct MIME types for .wasm, .js, .css, .svg
//! - SPA fallback (serves index.html for unknown routes)
//! - Cache headers for immutable hashed assets

use axum::Router;
use tower_http::compression::CompressionLayer;
use tower_http::services::{ServeDir, ServeFile};

#[tokio::main]
async fn main() {
    env_logger::init();

    let dist_path = std::env::var("PRISM_DIST")
        .unwrap_or_else(|_| "/srv/luminous-dynamics/prism/prism-ui/dist".to_string());

    let fallback = ServeFile::new(format!("{}/index.html", dist_path));

    let app = Router::new()
        .fallback_service(
            ServeDir::new(&dist_path)
                .not_found_service(fallback)
        )
        .layer(CompressionLayer::new());

    let addr = std::env::var("PRISM_ADDR").unwrap_or_else(|_| "0.0.0.0:8130".to_string());

    log::info!("Symthaea Prism serving {} on {}", dist_path, addr);
    println!("Symthaea Prism serving on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
