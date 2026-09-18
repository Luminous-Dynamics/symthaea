// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thin integration seam between the shell-facing request vocabulary and the
//! compatibility-safe duplex IPC transport.
//!
//! This module deliberately owns no connection state, cognitive telemetry, or UI
//! policy. It only composes the pure protocol adapter with
//! [`super::ipc_connection::DuplexShellConnection`]. That keeps the eventual
//! `ShellIpcClient` migration mechanical: its public domain methods can continue
//! producing [`Request`] values while socket ownership moves underneath them.

use anyhow::Result;

use super::ipc_adapter::{ResponseMapContext, map_request, map_response};
use super::ipc_client::{Request, Response};
use super::ipc_connection::DuplexShellConnection;

/// Send one shell-facing request through the negotiated duplex connection.
///
/// Translation is pure on both sides of the transport. The caller supplies the
/// small amount of response context that historically lived inside
/// `ShellIpcClient` (`latest_phi` and the Pong timestamp).
pub async fn send_shell_request(
    connection: &DuplexShellConnection,
    request: &Request,
    context: ResponseMapContext,
) -> Result<Response> {
    let wire_request = map_request(request);
    let wire_response = connection.request(wire_request).await?;
    Ok(map_response(request, wire_response, context))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shell::ipc_client::{IpcRequest, IpcResponse, MetricsSnapshot};
    use crate::shell::ipc_connection::LEGACY_IPC_PROTOCOL_VERSION;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{UnixListener, UnixStream};

    async fn read_request(stream: &mut UnixStream) -> IpcRequest {
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await.unwrap();
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload).await.unwrap();
        rmp_serde::from_slice(&payload).unwrap()
    }

    async fn write_response(stream: &mut UnixStream, response: &IpcResponse) {
        let payload = rmp_serde::to_vec(response).unwrap();
        stream
            .write_all(&(payload.len() as u32).to_le_bytes())
            .await
            .unwrap();
        stream.write_all(&payload).await.unwrap();
        stream.flush().await.unwrap();
    }

    async fn accept_and_negotiate(listener: UnixListener) -> UnixStream {
        let (mut stream, _) = listener.accept().await.unwrap();
        assert!(matches!(
            read_request(&mut stream).await,
            IpcRequest::Hello {
                version: LEGACY_IPC_PROTOCOL_VERSION
            }
        ));
        write_response(
            &mut stream,
            &IpcResponse::HelloAck {
                server_version: LEGACY_IPC_PROTOCOL_VERSION,
            },
        )
        .await;
        stream
    }

    fn observed_metrics(phi: f64) -> MetricsSnapshot {
        MetricsSnapshot {
            phi,
            coherence: 0.84,
            is_conscious: true,
            consciousness_level: 0.81,
            uptime_secs: 77,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn get_status_crosses_real_duplex_transport_and_updates_latest_metrics() {
        let temp = tempfile::tempdir().unwrap();
        let socket_path = temp.path().join("shell-bridge-status.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();

        let server = tokio::spawn(async move {
            let mut stream = accept_and_negotiate(listener).await;
            assert!(matches!(
                read_request(&mut stream).await,
                IpcRequest::GetMetrics
            ));
            write_response(
                &mut stream,
                &IpcResponse::Metrics(observed_metrics(0.78)),
            )
            .await;
        });

        let connection = DuplexShellConnection::connect_path(
            &socket_path,
            std::time::Duration::from_secs(1),
        )
        .await
        .unwrap();
        let mut metrics_rx = connection.metrics_receiver();
        assert!(metrics_rx.borrow().is_none());

        let response = send_shell_request(
            &connection,
            &Request::GetStatus,
            ResponseMapContext::new(0.0, 10),
        )
        .await
        .unwrap();

        assert!(matches!(
            response,
            Response::Status {
                phi,
                coherence,
                consciousness_level,
                is_conscious: true,
                uptime_secs: 77,
            } if (phi - 0.78).abs() < f64::EPSILON
                && (coherence - 0.84).abs() < f64::EPSILON
                && (consciousness_level - 0.81).abs() < f64::EPSILON
        ));

        metrics_rx.changed().await.unwrap();
        assert_eq!(metrics_rx.borrow().as_ref().map(|m| m.phi), Some(0.78));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn dry_run_maps_to_validation_without_execution_claim() {
        let temp = tempfile::tempdir().unwrap();
        let socket_path = temp.path().join("shell-bridge-dry-run.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();

        let server = tokio::spawn(async move {
            let mut stream = accept_and_negotiate(listener).await;
            assert!(matches!(
                read_request(&mut stream).await,
                IpcRequest::Validate { ref command } if command == "danger"
            ));
            write_response(
                &mut stream,
                &IpcResponse::ValidationResult {
                    valid: true,
                    safety_level: "YELLOW".into(),
                    preview: Some("would execute danger".into()),
                    warnings: vec![],
                },
            )
            .await;
        });

        let connection = DuplexShellConnection::connect_path(
            &socket_path,
            std::time::Duration::from_secs(1),
        )
        .await
        .unwrap();

        let response = send_shell_request(
            &connection,
            &Request::Execute {
                command: "danger".into(),
                phi_required: 0.9,
                dry_run: true,
            },
            ResponseMapContext::new(0.71, 20),
        )
        .await
        .unwrap();

        assert!(matches!(
            response,
            Response::ExecutionResult {
                executed: false,
                output,
                gate_reason: Some(reason),
                ..
            } if output == "would execute danger" && reason == "dry-run"
        ));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn pong_timestamp_is_caller_owned_across_real_transport() {
        let temp = tempfile::tempdir().unwrap();
        let socket_path = temp.path().join("shell-bridge-ping.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();

        let server = tokio::spawn(async move {
            let mut stream = accept_and_negotiate(listener).await;
            assert!(matches!(read_request(&mut stream).await, IpcRequest::Ping));
            write_response(&mut stream, &IpcResponse::Pong).await;
        });

        let connection = DuplexShellConnection::connect_path(
            &socket_path,
            std::time::Duration::from_secs(1),
        )
        .await
        .unwrap();

        let response = send_shell_request(
            &connection,
            &Request::Ping,
            ResponseMapContext::new(0.71, 123_456),
        )
        .await
        .unwrap();

        assert!(matches!(
            response,
            Response::Pong {
                timestamp_ms: 123_456
            }
        ));
        server.await.unwrap();
    }
}
