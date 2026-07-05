// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea API server entrypoint.

use std::env;
use std::net::SocketAddr;
use tracing::{info, warn};

fn addr_is_loopback(addr: &str) -> bool {
    if addr.starts_with("localhost:") {
        return true;
    }

    if let Ok(sa) = addr.parse::<SocketAddr>() {
        return sa.ip().is_loopback();
    }

    // If we can't parse it, treat it as non-loopback to avoid accidental exposure.
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
    let addr = env::var("SYMTHAEA_API_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    let bearer_token = env::var("SYMTHAEA_API_BEARER_TOKEN")
        .ok()
        .and_then(|t| if t.trim().is_empty() { None } else { Some(t) });

    // If the operator intentionally wants to run unauthenticated on a non-loopback bind, require
    // an explicit "insecure" opt-in so this can't happen by accident.
    let insecure_allow_unauth = env_truthy("SYMTHAEA_API_INSECURE_ALLOW_UNAUTH");

    if !addr_is_loopback(&addr) && bearer_token.is_none() && !insecure_allow_unauth {
        eprintln!("Refusing to bind Symthaea API to non-loopback address without auth.");
        eprintln!("  addr: {}", addr);
        eprintln!();
        eprintln!("Set one of:");
        eprintln!("  - SYMTHAEA_API_BEARER_TOKEN=...   (recommended)");
        eprintln!("  - SYMTHAEA_API_INSECURE_ALLOW_UNAUTH=1   (NOT recommended)");
        std::process::exit(2);
    }

    if !addr_is_loopback(&addr) && bearer_token.is_none() && insecure_allow_unauth {
        eprintln!("WARNING: Symthaea API is binding publicly without auth (insecure).");
        eprintln!("  addr: {}", addr);
    }

    let mut config = symthaea::api::ApiConfig {
        bearer_token,
        audit_log_path: env::var("SYMTHAEA_API_AUDIT_LOG_PATH").ok().and_then(|p| {
            if p.trim().is_empty() {
                None
            } else {
                Some(std::path::PathBuf::from(p))
            }
        }),
        ..Default::default()
    };

    if let Some(path) = config.audit_log_path.as_ref() {
        info!("API audit log persistence enabled at {}", path.display());
    } else {
        warn!(
            "API audit events are retained in memory only; set SYMTHAEA_API_AUDIT_LOG_PATH for JSONL persistence"
        );
    }

    if let Ok(origins) = env::var("SYMTHAEA_API_ALLOWED_ORIGINS") {
        config.allowed_origins = origins
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }

    symthaea::api::serve_with_config(&addr, config).await
}
