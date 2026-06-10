// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Nix Evaluation API
//!
//! HTTP service for evaluating NixOS flakes from the browser portal.
//!
//! # Usage
//! ```bash
//! cargo run --bin eval-api --features server -- --port 8090
//! ```
//!
//! # Endpoints
//! - POST /api/v1/eval — Evaluate a NixOS flake configuration
//! - GET /api/v1/health — Health check

use axum::{
    Router,
    extract::{ConnectInfo, State},
    http::{HeaderValue, Method, StatusCode, header},
    response::Json,
    routing::{get, post},
};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tower_http::cors::{AllowOrigin, CorsLayer};

/// Nix evaluation request from the browser portal.
#[derive(serde::Deserialize)]
struct NixEvalRequest {
    flake_ref: String,
    hardware_config: String,
    hostname: String,
    disko_config: Option<String>,
    platform: Option<String>,
    include_holochain: Option<bool>,
    include_broca_weights: Option<bool>,
    substrate_type: Option<String>,
    lanzaboote_enabled: Option<bool>,
}

/// Nix evaluation response.
#[derive(serde::Serialize)]
struct NixEvalResponse {
    success: bool,
    derivation_count: u64,
    closure_size_bytes: u64,
    closure_size_human: String,
    closure_hash: String,
    store_paths: Vec<String>,
    system_config_preview: String,
    error: Option<String>,
    eval_time_ms: u64,
}

/// Health check response.
#[derive(serde::Serialize)]
struct HealthResponse {
    ok: bool,
    nix_available: bool,
    version: &'static str,
}

/// Rate limiter state.
struct RateLimiter {
    requests: HashMap<String, Vec<Instant>>,
    max_per_minute: usize,
}

impl RateLimiter {
    fn new(max_per_minute: usize) -> Self {
        Self {
            requests: HashMap::new(),
            max_per_minute,
        }
    }

    fn check(&mut self, ip: &str) -> bool {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(60);
        let entries = self.requests.entry(ip.to_string()).or_default();
        entries.retain(|t| now.duration_since(*t) < window);
        if entries.len() >= self.max_per_minute {
            return false;
        }
        entries.push(now);
        true
    }
}

type SharedState = Arc<Mutex<RateLimiter>>;

fn evaluate_flake(req: &NixEvalRequest) -> NixEvalResponse {
    let start = Instant::now();
    let platform = req.platform.as_deref().unwrap_or("x86_64-linux");

    // Write temporary flake to /tmp for evaluation
    let eval_dir = format!(
        "/tmp/symthaea-eval-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );
    let _ = std::fs::create_dir_all(&eval_dir);

    let hw_path = format!("{}/hardware-configuration.nix", eval_dir);
    let _ = std::fs::write(&hw_path, &req.hardware_config);

    // Write disko config if provided
    if let Some(ref disko) = req.disko_config {
        let disko_path = format!("{}/disko-config.nix", eval_dir);
        let _ = std::fs::write(&disko_path, disko);
    }

    let flake_content = format!(
        r#"{{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = {{ nixpkgs, ... }}: {{
    nixosConfigurations.{hostname} = nixpkgs.lib.nixosSystem {{
      system = "{platform}";
      modules = [
        ./hardware-configuration.nix
        ({{ ... }}: {{
          networking.hostName = "{hostname}";
          boot.loader.systemd-boot.enable = true;
          system.stateVersion = "24.11";
        }})
      ];
    }};
  }};
}}"#,
        hostname = req.hostname,
        platform = platform
    );

    let flake_path = format!("{}/flake.nix", eval_dir);
    if std::fs::write(&flake_path, &flake_content).is_err() {
        return error_response("Failed to write evaluation files", &start);
    }

    // Run nix eval to get the system derivation
    let eval_output = Command::new("nix")
        .args([
            "eval",
            "--json",
            &format!(
                "path:{}#nixosConfigurations.{}.config.system.build.toplevel.drvPath",
                eval_dir, req.hostname
            ),
        ])
        .output();

    let drv_path = match eval_output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .trim()
            .trim_matches('"')
            .to_string(),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = std::fs::remove_dir_all(&eval_dir);
            return error_response(
                &format!(
                    "Nix eval failed: {}",
                    stderr.chars().take(500).collect::<String>()
                ),
                &start,
            );
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&eval_dir);
            return error_response(&format!("Failed to execute nix: {}", e), &start);
        }
    };

    // Get closure size
    let closure_output = Command::new("nix")
        .args(["path-info", "--closure-size", "--json", &drv_path])
        .output();

    let (closure_size, store_paths) = match closure_output {
        Ok(output) if output.status.success() => {
            let json_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(paths) = serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                let size: u64 = paths
                    .iter()
                    .filter_map(|p| p.get("closureSize").and_then(|s| s.as_u64()))
                    .max()
                    .unwrap_or(0);
                let store: Vec<String> = paths
                    .iter()
                    .filter_map(|p| p.get("path").and_then(|s| s.as_str()).map(String::from))
                    .collect();
                (size, store)
            } else {
                (0, Vec::new())
            }
        }
        _ => (0, Vec::new()),
    };

    let hash = blake3::hash(
        format!("{}{}{}", req.flake_ref, req.hostname, req.hardware_config).as_bytes(),
    );

    let _ = std::fs::remove_dir_all(&eval_dir);

    let closure_size_human = if closure_size > 1_000_000_000 {
        format!("{:.1} GB", closure_size as f64 / 1_000_000_000.0)
    } else if closure_size > 1_000_000 {
        format!("{:.0} MB", closure_size as f64 / 1_000_000.0)
    } else {
        format!("{} bytes", closure_size)
    };

    NixEvalResponse {
        success: true,
        derivation_count: store_paths.len() as u64,
        closure_size_bytes: closure_size,
        closure_size_human,
        closure_hash: hash.to_hex().to_string(),
        store_paths: store_paths.into_iter().take(20).collect(),
        system_config_preview: flake_content
            .lines()
            .take(50)
            .collect::<Vec<_>>()
            .join("\n"),
        error: None,
        eval_time_ms: start.elapsed().as_millis() as u64,
    }
}

fn error_response(msg: &str, start: &Instant) -> NixEvalResponse {
    NixEvalResponse {
        success: false,
        derivation_count: 0,
        closure_size_bytes: 0,
        closure_size_human: String::new(),
        closure_hash: String::new(),
        store_paths: Vec::new(),
        system_config_preview: String::new(),
        error: Some(msg.to_string()),
        eval_time_ms: start.elapsed().as_millis() as u64,
    }
}

async fn eval_handler(
    State(state): State<SharedState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<NixEvalRequest>,
) -> Result<Json<NixEvalResponse>, StatusCode> {
    // Rate limit by the actual socket peer address (do not trust X-Forwarded-For by default).
    let ip = addr.ip().to_string();

    {
        let mut limiter = state.lock().unwrap();
        if !limiter.check(&ip) {
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
    }

    // Run evaluation (blocking — Nix is a subprocess)
    let response = tokio::task::spawn_blocking(move || evaluate_flake(&req))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(response))
}

async fn health_handler() -> Json<HealthResponse> {
    let nix_available = Command::new("nix")
        .args(["--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    Json(HealthResponse {
        ok: true,
        nix_available,
        version: "0.1.0",
    })
}

#[tokio::main]
async fn main() {
    let mut port: u16 = 8090;
    let mut bind: String = "127.0.0.1".into();
    let mut allowed_origins: Vec<String> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                if let Some(p) = args.next().and_then(|p| p.parse::<u16>().ok()) {
                    port = p;
                }
            }
            "--bind" => {
                if let Some(b) = args.next() {
                    bind = b;
                }
            }
            "--allow-origin" => {
                if let Some(o) = args.next() {
                    allowed_origins.push(o);
                }
            }
            "--help" | "-h" => {
                eprintln!("Symthaea Eval API (locked down)");
                eprintln!("Usage:");
                eprintln!(
                    "  eval-api [--bind <ip|localhost>] [--port <port>] [--allow-origin <origin> ...]"
                );
                eprintln!();
                eprintln!("Security defaults:");
                eprintln!("  - Binds to 127.0.0.1 only");
                eprintln!("  - CORS restricted to explicit localhost origins");
                eprintln!();
                eprintln!("Examples:");
                eprintln!(
                    "  eval-api --port 8090 --allow-origin http://localhost:7777 --allow-origin http://127.0.0.1:7777"
                );
                return;
            }
            _ => {}
        }
    }

    // Default allowlist: local Symthaea dashboard + common local test servers.
    if allowed_origins.is_empty() {
        allowed_origins = vec![
            "http://localhost:7777".into(),
            "http://127.0.0.1:7777".into(),
            "http://localhost:8093".into(),
            "http://127.0.0.1:8093".into(),
            "http://localhost:8094".into(),
            "http://127.0.0.1:8094".into(),
        ];
    }

    let origin_values: Vec<HeaderValue> = allowed_origins
        .iter()
        .filter_map(|o| o.parse::<HeaderValue>().ok())
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(origin_values))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    let state: SharedState = Arc::new(Mutex::new(RateLimiter::new(10)));

    let app = Router::new()
        .route("/api/v1/eval", post(eval_handler))
        .route("/api/v1/health", get(health_handler))
        .layer(cors)
        .with_state(state);

    let ip: IpAddr = match bind.as_str() {
        "localhost" => IpAddr::V4(Ipv4Addr::LOCALHOST),
        other => other
            .parse()
            .unwrap_or_else(|_| IpAddr::V4(Ipv4Addr::LOCALHOST)),
    };
    let addr = SocketAddr::new(ip, port);
    eprintln!("Symthaea Eval API listening on http://{}", addr);
    eprintln!("  POST /api/v1/eval  — Evaluate NixOS flake");
    eprintln!("  GET  /api/v1/health — Health check");
    eprintln!("  CORS allowlist:");
    for o in &allowed_origins {
        eprintln!("    - {}", o);
    }

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap();
}
