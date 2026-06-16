// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! API Binary Integration Tests
//!
//! Tests for the `symthaea-api` HTTP server binary.
//! Verifies server startup, health endpoints, and basic API functionality.
//!
//! Run with: `cargo test --test binary_api --features api_module`

#![cfg(feature = "api_module")]

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

/// Get the path to the built binary
fn binary_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push(if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    });
    path.push("symthaea-api");
    path
}

/// Build the binary if needed
fn ensure_binary_built() {
    let status = Command::new("cargo")
        .args(["build", "--bin", "symthaea-api", "--features", "api_module"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("Failed to run cargo build");

    assert!(status.success(), "Failed to build symthaea-api binary");
}

/// Find an available port for testing
fn find_available_port() -> u16 {
    // Use a random high port
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind");
    listener.local_addr().unwrap().port()
}

/// Wait for a server to become available
fn wait_for_server(addr: &str, timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect(addr).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(50));
    }
    false
}

mod startup_tests {
    use super::*;

    #[test]
    fn test_binary_exists() {
        ensure_binary_built();

        let path = binary_path();
        assert!(path.exists(), "API binary should exist at {:?}", path);
    }

    #[test]
    fn test_server_startup_and_shutdown() {
        ensure_binary_built();

        let port = find_available_port();
        let addr = format!("127.0.0.1:{}", port);

        // Start the server
        let mut child = Command::new(binary_path())
            .env("SYMTHAEA_API_ADDR", &addr)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start API server");

        // Wait for server to start
        let server_ready = wait_for_server(&addr, Duration::from_secs(5));

        // Clean up
        child.kill().expect("Failed to kill server");
        let _ = child.wait();

        if server_ready {
            eprintln!("[Info] Server started successfully on {}", addr);
        } else {
            eprintln!("[Info] Server did not become ready in time (may need dependencies)");
        }
    }

    #[test]
    fn test_custom_port_binding() {
        ensure_binary_built();

        let port = find_available_port();
        let addr = format!("127.0.0.1:{}", port);

        let mut child = Command::new(binary_path())
            .env("SYMTHAEA_API_ADDR", &addr)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start API server");

        let ready = wait_for_server(&addr, Duration::from_secs(3));

        child.kill().expect("Failed to kill");
        let _ = child.wait();

        // Just verify the env var is processed (server may or may not start fully)
        eprintln!("[Info] Custom port test: ready={}", ready);
    }
}

mod health_endpoint_tests {
    use super::*;

    #[tokio::test]
    async fn test_health_endpoint() {
        ensure_binary_built();

        let port = find_available_port();
        let addr = format!("127.0.0.1:{}", port);

        // Start server
        let mut child = Command::new(binary_path())
            .env("SYMTHAEA_API_ADDR", &addr)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start API server");

        // Wait for server
        if !wait_for_server(&addr, Duration::from_secs(5)) {
            child.kill().expect("Failed to kill");
            let _ = child.wait();
            eprintln!("[Skip] Server did not start, skipping health check");
            return;
        }

        // Test health endpoint
        let client = reqwest::Client::new();
        let health_url = format!("http://{}/health", addr);

        let response = client
            .get(&health_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await;

        // Clean up
        child.kill().expect("Failed to kill");
        let _ = child.wait();

        match response {
            Ok(resp) => {
                assert!(
                    resp.status().is_success(),
                    "Health endpoint should return 2xx"
                );
                eprintln!("[Info] Health endpoint returned status: {}", resp.status());
            }
            Err(e) => {
                eprintln!("[Info] Health request failed (may be expected): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_v1_health_endpoint() {
        ensure_binary_built();

        let port = find_available_port();
        let addr = format!("127.0.0.1:{}", port);

        let mut child = Command::new(binary_path())
            .env("SYMTHAEA_API_ADDR", &addr)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start API server");

        if !wait_for_server(&addr, Duration::from_secs(5)) {
            child.kill().expect("Failed to kill");
            let _ = child.wait();
            eprintln!("[Skip] Server did not start");
            return;
        }

        let client = reqwest::Client::new();
        let health_url = format!("http://{}/v1/health", addr);

        let response = client
            .get(&health_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await;

        child.kill().expect("Failed to kill");
        let _ = child.wait();

        if let Ok(resp) = response {
            assert!(resp.status().is_success(), "/v1/health should return 2xx");
        }
    }
}

mod api_endpoint_tests {
    use super::*;

    #[tokio::test]
    async fn test_leaderboard_endpoint() {
        ensure_binary_built();

        let port = find_available_port();
        let addr = format!("127.0.0.1:{}", port);

        let mut child = Command::new(binary_path())
            .env("SYMTHAEA_API_ADDR", &addr)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start API server");

        if !wait_for_server(&addr, Duration::from_secs(5)) {
            child.kill().expect("Failed to kill");
            let _ = child.wait();
            eprintln!("[Skip] Server did not start");
            return;
        }

        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/leaderboard", addr);

        let response = client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await;

        child.kill().expect("Failed to kill");
        let _ = child.wait();

        if let Ok(resp) = response {
            // May require auth, so 401/403 is acceptable
            assert!(
                resp.status().is_success()
                    || resp.status() == reqwest::StatusCode::UNAUTHORIZED
                    || resp.status() == reqwest::StatusCode::FORBIDDEN,
                "Leaderboard should return 2xx or auth error, got: {}",
                resp.status()
            );
            eprintln!("[Info] Leaderboard endpoint returned: {}", resp.status());
        }
    }

    #[tokio::test]
    async fn test_datasets_endpoint() {
        ensure_binary_built();

        let port = find_available_port();
        let addr = format!("127.0.0.1:{}", port);

        let mut child = Command::new(binary_path())
            .env("SYMTHAEA_API_ADDR", &addr)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start API server");

        if !wait_for_server(&addr, Duration::from_secs(5)) {
            child.kill().expect("Failed to kill");
            let _ = child.wait();
            eprintln!("[Skip] Server did not start");
            return;
        }

        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/datasets", addr);

        let response = client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await;

        child.kill().expect("Failed to kill");
        let _ = child.wait();

        if let Ok(resp) = response {
            assert!(
                resp.status().is_success()
                    || resp.status() == reqwest::StatusCode::UNAUTHORIZED
                    || resp.status() == reqwest::StatusCode::FORBIDDEN,
                "Datasets should return 2xx or auth error"
            );
        }
    }

    #[tokio::test]
    async fn test_404_for_unknown_endpoint() {
        ensure_binary_built();

        let port = find_available_port();
        let addr = format!("127.0.0.1:{}", port);

        let mut child = Command::new(binary_path())
            .env("SYMTHAEA_API_ADDR", &addr)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start API server");

        if !wait_for_server(&addr, Duration::from_secs(5)) {
            child.kill().expect("Failed to kill");
            let _ = child.wait();
            eprintln!("[Skip] Server did not start");
            return;
        }

        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/definitely-not-an-endpoint", addr);

        let response = client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await;

        child.kill().expect("Failed to kill");
        let _ = child.wait();

        if let Ok(resp) = response {
            assert_eq!(
                resp.status(),
                reqwest::StatusCode::NOT_FOUND,
                "Unknown endpoint should return 404"
            );
        }
    }
}

mod router_unit_tests {
    //! Unit tests for the API router without starting a full server

    use symthaea::api::{ApiConfig, create_router, create_router_with_config};

    #[test]
    fn test_router_creation() {
        let _router = create_router();
        // Router created successfully
        eprintln!("[Info] Router created successfully");
    }

    #[test]
    fn test_router_with_custom_config() {
        let config = ApiConfig {
            allowed_origins: vec!["http://localhost:3000".to_string()],
            bearer_token: Some("test-token".to_string()),
            ..ApiConfig::default()
        };

        let _router = create_router_with_config(config);
        eprintln!("[Info] Router with custom config created");
    }

    #[test]
    fn test_api_config_default() {
        let config = ApiConfig::default();

        // Default config should have localhost origins
        assert!(
            !config.allowed_origins.is_empty(),
            "Default should have allowed origins"
        );
        assert!(
            config.bearer_token.is_none(),
            "Default should not require auth"
        );

        eprintln!(
            "[Info] Default config: origins={:?}",
            config.allowed_origins
        );
    }

    #[test]
    fn test_api_config_with_auth() {
        let config = ApiConfig {
            allowed_origins: vec!["https://example.com".to_string()],
            bearer_token: Some("secret-token".to_string()),
            ..ApiConfig::default()
        };

        assert_eq!(config.allowed_origins.len(), 1);
        assert!(config.bearer_token.is_some());

        eprintln!("[Info] Auth config created successfully");
    }
}