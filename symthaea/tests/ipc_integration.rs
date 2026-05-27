// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IPC Integration Tests
//!
//! Tests for the symthaea-service IPC protocol:
//! - Request/Response round-trips
//! - MessagePack serialization
//! - Request batching
//! - Heartbeat mechanism
//! - Metrics streaming

#![cfg(feature = "shell")]

use serde_json;
use std::path::PathBuf;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;

use symthaea::shell::ipc_client::{
    IpcClientConfig, Request, RequestEnvelope, Response, ResponseEnvelope, ShellContextData,
    ShellIpcClient, WireProtocol,
};

/// Generate unique test socket path using PID, thread ID and timestamp
fn test_socket_path() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    let counter = COUNTER.fetch_add(1, Ordering::SeqCst);
    PathBuf::from(format!("/tmp/symthaea-test-{}-{}.sock", pid, counter))
}

/// Clean up test socket
fn cleanup_socket(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
}

/// Mock server that echoes requests
async fn mock_server(socket_path: PathBuf, responses: Vec<Response>) {
    cleanup_socket(&socket_path);

    let listener = UnixListener::bind(&socket_path).unwrap();
    let (stream, _) = listener.accept().await.unwrap();
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    let mut response_iter = responses.into_iter();

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                // Parse request
                if let Ok(envelope) = serde_json::from_str::<RequestEnvelope>(&line) {
                    // Get next response or Pong
                    let response = response_iter.next().unwrap_or(Response::Pong {
                        timestamp_ms: 12345,
                    });

                    let response_envelope = ResponseEnvelope {
                        id: envelope.id,
                        latency_ms: Some(1),
                        response,
                    };

                    let json = serde_json::to_string(&response_envelope).unwrap() + "\n";
                    write_half.write_all(json.as_bytes()).await.unwrap();
                }
            }
            Err(_) => break,
        }
    }

    cleanup_socket(&socket_path);
}

#[tokio::test]
async fn test_ping_pong() {
    let socket_path = test_socket_path();

    // Start mock server
    let server_path = socket_path.clone();
    let server = tokio::spawn(async move {
        mock_server(
            server_path,
            vec![Response::Pong {
                timestamp_ms: 12345,
            }],
        )
        .await;
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Create client
    let mut client = ShellIpcClient::with_socket_path(&socket_path);
    assert!(client.connect().await.is_ok());

    // Send ping, get pong
    let response = client.ping().await;
    assert!(response.is_ok());

    // Clean up
    drop(client);
    server.abort();
    cleanup_socket(&socket_path);
}

#[tokio::test]
async fn test_get_status() {
    let socket_path = test_socket_path();

    let server_path = socket_path.clone();
    let server = tokio::spawn(async move {
        mock_server(
            server_path,
            vec![Response::Status {
                phi: 0.85,
                coherence: 0.92,
                consciousness_level: 0.88,
                is_conscious: true,
                uptime_secs: 3600,
            }],
        )
        .await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = ShellIpcClient::with_socket_path(&socket_path);
    assert!(client.connect().await.is_ok());

    let status = client.get_status().await;
    assert!(status.is_ok());

    let (phi, coherence, is_conscious) = status.unwrap();
    assert!((phi - 0.85).abs() < 0.01);
    assert!((coherence - 0.92).abs() < 0.01);
    assert!(is_conscious);

    drop(client);
    server.abort();
    cleanup_socket(&socket_path);
}

#[tokio::test]
async fn test_request_with_id() {
    let socket_path = test_socket_path();

    let server_path = socket_path.clone();
    let server = tokio::spawn(async move {
        mock_server(
            server_path,
            vec![Response::Pong {
                timestamp_ms: 12345,
            }],
        )
        .await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = ShellIpcClient::with_socket_path(&socket_path);
    assert!(client.connect().await.is_ok());

    // Send request with correlation ID
    let response = client
        .send_request_with_id(Request::Ping, Some("test-123".to_string()))
        .await;

    assert!(response.is_ok());
    let envelope = response.unwrap();
    assert_eq!(envelope.id, Some("test-123".to_string()));

    drop(client);
    server.abort();
    cleanup_socket(&socket_path);
}

#[tokio::test]
async fn test_intellisense() {
    let socket_path = test_socket_path();

    let server_path = socket_path.clone();
    let server = tokio::spawn(async move {
        mock_server(
            server_path,
            vec![Response::IntelliSenseResult {
                completions: vec![],
                command_preview: None,
                phi: 0.85,
                confidence: 0.9,
            }],
        )
        .await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = ShellIpcClient::with_socket_path(&socket_path);
    assert!(client.connect().await.is_ok());

    let context = ShellContextData {
        cwd: "/home/user".to_string(),
        history: vec!["ls".to_string(), "pwd".to_string()],
    };

    let result = client.get_intellisense("install ngi", 11, &context).await;
    assert!(result.is_ok());

    drop(client);
    server.abort();
    cleanup_socket(&socket_path);
}

#[tokio::test]
async fn test_validate_command() {
    let socket_path = test_socket_path();

    let server_path = socket_path.clone();
    let server = tokio::spawn(async move {
        mock_server(
            server_path,
            vec![Response::ValidationResult {
                valid: true,
                safety_level: symthaea::shell::ipc_client::SafetyLevelData {
                    level: "GREEN".to_string(),
                    color: "green".to_string(),
                    description: "Safe".to_string(),
                },
                phi_required: 0.5,
                warnings: vec![],
            }],
        )
        .await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = ShellIpcClient::with_socket_path(&socket_path);
    assert!(client.connect().await.is_ok());

    let result = client.validate_command("nix search firefox", false).await;
    assert!(result.is_ok());

    let (valid, _safety, _phi, warnings) = result.unwrap();
    assert!(valid);
    assert!(warnings.is_empty());

    drop(client);
    server.abort();
    cleanup_socket(&socket_path);
}

#[tokio::test]
async fn test_execute_gated() {
    let socket_path = test_socket_path();

    let server_path = socket_path.clone();
    let server = tokio::spawn(async move {
        mock_server(
            server_path,
            vec![Response::ExecutionResult {
                executed: true,
                output: "Success".to_string(),
                phi_at_execution: 0.85,
                gate_reason: None,
            }],
        )
        .await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = ShellIpcClient::with_socket_path(&socket_path);
    assert!(client.connect().await.is_ok());

    let result = client.execute_gated("install firefox", 0.5, false).await;
    assert!(result.is_ok());

    let (executed, output, phi, gate_reason) = result.unwrap();
    assert!(executed);
    assert_eq!(output, "Success");
    assert!((phi - 0.85).abs() < 0.01);
    assert!(gate_reason.is_none());

    drop(client);
    server.abort();
    cleanup_socket(&socket_path);
}

#[tokio::test]
async fn test_connection_retry() {
    let socket_path = test_socket_path();

    // Don't start server yet - test retry behavior
    let mut client = ShellIpcClient::with_socket_path(&socket_path);

    // Should fail to connect (no server)
    let result = client.connect().await;
    assert!(result.is_err());

    // Now start server
    let server_path = socket_path.clone();
    let server = tokio::spawn(async move {
        mock_server(
            server_path,
            vec![Response::Pong {
                timestamp_ms: 12345,
            }],
        )
        .await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Retry should succeed
    let result = client.connect().await;
    assert!(result.is_ok());

    drop(client);
    server.abort();
    cleanup_socket(&socket_path);
}

#[tokio::test]
async fn test_heartbeat() {
    let socket_path = test_socket_path();

    let server_path = socket_path.clone();
    let server = tokio::spawn(async move {
        mock_server(
            server_path,
            vec![
                Response::Pong {
                    timestamp_ms: 12345,
                },
                Response::Pong {
                    timestamp_ms: 12346,
                },
                Response::Pong {
                    timestamp_ms: 12347,
                },
            ],
        )
        .await;
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = ShellIpcClient::with_socket_path(&socket_path);
    assert!(client.connect().await.is_ok());

    // Send multiple heartbeats
    for _ in 0..3 {
        let result = client.heartbeat().await;
        assert!(result.is_ok());
    }

    // Check health
    assert!(client.is_healthy());

    drop(client);
    server.abort();
    cleanup_socket(&socket_path);
}

#[test]
fn test_request_serialization() {
    // Test JSON serialization
    let request = Request::Ping;
    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("Ping"));

    let parsed: Request = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, Request::Ping));
}

#[test]
fn test_response_serialization() {
    let response = Response::Status {
        phi: 0.85,
        coherence: 0.92,
        consciousness_level: 0.88,
        is_conscious: true,
        uptime_secs: 3600,
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("0.85"));
    assert!(json.contains("Status"));

    let parsed: Response = serde_json::from_str(&json).unwrap();
    if let Response::Status { phi, .. } = parsed {
        assert!((phi - 0.85).abs() < 0.01);
    } else {
        panic!("Expected Status response");
    }
}

#[test]
fn test_envelope_with_id() {
    let envelope = RequestEnvelope {
        id: Some("test-123".to_string()),
        request: Request::Ping,
    };

    let json = serde_json::to_string(&envelope).unwrap();
    assert!(json.contains("test-123"));

    let parsed: RequestEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, Some("test-123".to_string()));
}

#[test]
fn test_batch_request_creation() {
    let mut client = ShellIpcClient::new();

    // Add multiple requests to batch
    client.batch_add(Request::Ping);
    client.batch_add(Request::GetStatus);
    client.batch_add(Request::Ping);

    assert_eq!(client.batch_size(), 3);
}

#[cfg(feature = "rmp-serde")]
#[test]
fn test_messagepack_serialization() {
    use symthaea::shell::ipc_client::WireProtocol;

    let request = Request::Ping;

    // Serialize to MessagePack
    let bytes = rmp_serde::to_vec(&request).unwrap();
    assert!(!bytes.is_empty());

    // Deserialize
    let parsed: Request = rmp_serde::from_slice(&bytes).unwrap();
    assert!(matches!(parsed, Request::Ping));
}