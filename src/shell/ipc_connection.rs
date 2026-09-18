// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Connected transport wrapper for the compatibility-safe duplex IPC pump.
//!
//! This module deliberately stops at the transport boundary. It owns socket
//! discovery/connect, protocol-v1 negotiation, and autonomous metrics subscription,
//! but it leaves request-to-domain mapping in `ShellIpcClient` for the migration
//! tranche. That avoids creating a second shell application client while giving the
//! legacy client a clean one-reader connection object to adopt.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::net::UnixStream;
use tokio::sync::watch;

use super::ipc_client::{
    IpcClientConfig, IpcRequest, IpcResponse, MetricsSnapshot, discover_socket,
};
use super::ipc_duplex::LegacyDuplexPump;

/// Legacy shell/service protocol negotiated by the current IPC server.
pub const LEGACY_IPC_PROTOCOL_VERSION: u32 = 1;

/// A negotiated Unix-socket connection with one authoritative reader task.
pub struct DuplexShellConnection {
    pump: LegacyDuplexPump,
    socket_path: PathBuf,
}

impl DuplexShellConnection {
    /// Discover/connect using the existing shell IPC configuration, then perform
    /// the protocol-v1 `Hello`/`HelloAck` exchange before returning.
    pub async fn connect(config: &IpcClientConfig) -> Result<Self> {
        let socket_path = config
            .socket_path
            .clone()
            .or_else(discover_socket)
            .context("No symthaea socket found. Is the service running?")?;

        Self::connect_path(&socket_path, config.request_timeout).await
    }

    /// Connect to one explicit socket path and negotiate protocol v1.
    pub async fn connect_path(path: &Path, request_timeout: Duration) -> Result<Self> {
        let stream = UnixStream::connect(path)
            .await
            .with_context(|| format!("Failed to connect to {}", path.display()))?;
        let pump = LegacyDuplexPump::from_stream(stream, request_timeout);
        pump.negotiate(LEGACY_IPC_PROTOCOL_VERSION)
            .await
            .context("IPC handshake failed")?;

        Ok(Self {
            pump,
            socket_path: path.to_path_buf(),
        })
    }

    /// Send one raw protocol-v1 request through the serialized foreground lane.
    pub async fn request(&self, request: IpcRequest) -> Result<IpcResponse> {
        self.pump.request(request).await
    }

    /// Perform the real service-side metrics subscription exchange.
    pub async fn subscribe_metrics(&self) -> Result<()> {
        self.pump.subscribe_metrics().await
    }

    /// Perform the real service-side metrics unsubscription exchange.
    pub async fn unsubscribe_metrics(&self) -> Result<()> {
        self.pump.unsubscribe_metrics().await
    }

    /// Latest observed service metrics. `None` means no real metrics frame has
    /// crossed this connection yet.
    pub fn metrics_receiver(&self) -> watch::Receiver<Option<MetricsSnapshot>> {
        self.pump.metrics_receiver()
    }

    /// Whether foreground response ordering is unsafe to reuse without reconnect.
    pub fn is_poisoned(&self) -> bool {
        self.pump.is_poisoned()
    }

    /// Socket path backing this connection.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;

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

    fn observed_metrics(phi: f64) -> MetricsSnapshot {
        MetricsSnapshot {
            phi,
            coherence: 0.82,
            is_conscious: true,
            consciousness_level: phi,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn connect_negotiates_then_subscription_receives_autonomous_metrics() {
        let temp = tempfile::tempdir().unwrap();
        let socket_path = temp.path().join("symthaea-test.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();

        let server = tokio::spawn(async move {
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

            assert!(matches!(
                read_request(&mut stream).await,
                IpcRequest::SubscribeMetrics
            ));
            write_response(&mut stream, &IpcResponse::Subscribed).await;
            write_response(&mut stream, &IpcResponse::Metrics(observed_metrics(0.77))).await;
        });

        let connection = DuplexShellConnection::connect_path(&socket_path, Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!(connection.socket_path(), socket_path.as_path());
        assert!(!connection.is_poisoned());

        let mut metrics = connection.metrics_receiver();
        assert!(metrics.borrow().is_none());
        connection.subscribe_metrics().await.unwrap();
        metrics.changed().await.unwrap();
        assert_eq!(metrics.borrow().as_ref().map(|m| m.phi), Some(0.77));

        server.await.unwrap();
    }

    #[tokio::test]
    async fn connect_fails_closed_on_protocol_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        let socket_path = temp.path().join("symthaea-mismatch.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();

        let server = tokio::spawn(async move {
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
                    server_version: LEGACY_IPC_PROTOCOL_VERSION + 1,
                },
            )
            .await;
        });

        let result = DuplexShellConnection::connect_path(&socket_path, Duration::from_secs(1)).await;
        assert!(result.is_err());
        server.await.unwrap();
    }
}
