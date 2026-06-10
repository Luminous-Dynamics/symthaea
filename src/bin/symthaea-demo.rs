// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Live Demo Server
//!
//! Launches a browser-accessible visualization of the consciousness pipeline.
//!
//! ```bash
//! cargo run --features api_module --bin symthaea-demo
//! ```
//!
//! Then open http://localhost:8080 in your browser.

use std::env;
use std::net::SocketAddr;

fn addr_is_loopback(addr: &str) -> bool {
    if addr.starts_with("localhost:") {
        return true;
    }
    if let Ok(sa) = addr.parse::<SocketAddr>() {
        return sa.ip().is_loopback();
    }
    false
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Secure-by-default: localhost only unless explicitly configured otherwise.
    let addr = env::var("SYMTHAEA_DEMO_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    let bearer_token = env::var("SYMTHAEA_DEMO_BEARER_TOKEN")
        .ok()
        .and_then(|t| if t.trim().is_empty() { None } else { Some(t) });
    let insecure_allow_unauth = env_truthy("SYMTHAEA_DEMO_INSECURE_ALLOW_UNAUTH");

    if !addr_is_loopback(&addr) && bearer_token.is_none() && !insecure_allow_unauth {
        eprintln!("Refusing to bind Symthaea Demo to non-loopback address without auth.");
        eprintln!("  addr: {}", addr);
        eprintln!();
        eprintln!("Set one of:");
        eprintln!("  - SYMTHAEA_DEMO_BEARER_TOKEN=...   (recommended)");
        eprintln!("  - SYMTHAEA_DEMO_INSECURE_ALLOW_UNAUTH=1   (NOT recommended)");
        std::process::exit(2);
    }

    if !addr_is_loopback(&addr) && bearer_token.is_none() && insecure_allow_unauth {
        eprintln!("WARNING: Symthaea Demo is binding publicly without auth (insecure).");
        eprintln!("  addr: {}", addr);
    }

    let mut config = symthaea::api::ApiConfig::default();
    config.bearer_token = bearer_token;

    if let Ok(origins) = env::var("SYMTHAEA_DEMO_ALLOWED_ORIGINS") {
        config.allowed_origins = origins
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }

    // Run server with graceful shutdown on SIGTERM/Ctrl-C.
    // This ensures P2P peers get a clean Disconnected event instead of TCP RST.
    tokio::select! {
        result = symthaea::api::serve_demo_with_config(&addr, config) => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            eprintln!("\nShutting down gracefully...");
        }
    }

    Ok(())
}
