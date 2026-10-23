// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compatibility-safe duplex pump for the legacy shell IPC protocol.
//!
//! Protocol v1 multiplexes pushed [`IpcResponse::Metrics`] values and foreground
//! request replies on one Unix socket, but it does not carry correlation IDs on
//! the actual wire messages. A correct client therefore needs two properties:
//!
//! 1. exactly one task owns the socket read half; and
//! 2. at most one foreground request is in flight at a time.
//!
//! The reader continuously diverts pushed metrics into a latest-wins `watch`
//! channel while delivering the next non-metrics response to the single pending
//! foreground waiter. `GetMetrics` is the one intentional overlap: a metrics
//! frame both updates the state watch and may satisfy that foreground request.
//!
//! A foreground timeout poisons further request use. Without response IDs, a late
//! response after timeout cannot be distinguished from the reply to a newer
//! request, so continuing would risk silent response misattribution.

use anyhow::{Context, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::{Mutex, oneshot, watch};

use super::ipc_client::{IpcRequest, IpcResponse, MetricsSnapshot};

const LEGACY_IPC_MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;

type PendingReply = oneshot::Sender<std::result::Result<IpcResponse, String>>;

struct PendingForeground {
    accepts_metrics: bool,
    reply: PendingReply,
}

/// One-reader compatibility pump for the legacy protocol-v1 Unix socket.
///
/// Metrics are latest-wins state. Foreground responses are lossless but strictly
/// serialized because protocol v1 has no correlation identifier on its live wire
/// messages.
pub struct LegacyDuplexPump {
    writer: Mutex<BufWriter<OwnedWriteHalf>>,
    request_gate: Mutex<()>,
    pending: Arc<Mutex<Option<PendingForeground>>>,
    metrics_tx: watch::Sender<Option<MetricsSnapshot>>,
    poisoned: Arc<AtomicBool>,
    request_timeout: Duration,
    reader_task: tokio::task::JoinHandle<()>,
}

impl LegacyDuplexPump {
    /// Create a duplex pump from a connected Unix stream.
    pub fn from_stream(stream: UnixStream, request_timeout: Duration) -> Self {
        let (reader, writer) = stream.into_split();
        Self::from_halves(reader, writer, request_timeout)
    }

    /// Create a duplex pump from already-owned socket halves.
    ///
    /// This is the migration seam for `ShellIpcClient`, which already splits its
    /// socket during connection establishment.
    pub fn from_halves(
        reader: OwnedReadHalf,
        writer: OwnedWriteHalf,
        request_timeout: Duration,
    ) -> Self {
        let pending: Arc<Mutex<Option<PendingForeground>>> = Arc::new(Mutex::new(None));
        let (metrics_tx, _) = watch::channel(None);
        let poisoned = Arc::new(AtomicBool::new(false));

        let reader_pending = Arc::clone(&pending);
        let reader_metrics = metrics_tx.clone();
        let reader_poisoned = Arc::clone(&poisoned);

        let reader_task = tokio::spawn(async move {
            Self::reader_loop(
                BufReader::new(reader),
                reader_pending,
                reader_metrics,
                reader_poisoned,
            )
            .await;
        });

        Self {
            writer: Mutex::new(BufWriter::new(writer)),
            request_gate: Mutex::new(()),
            pending,
            metrics_tx,
            poisoned,
            request_timeout,
            reader_task,
        }
    }

    /// Subscribe to latest observed service metrics.
    ///
    /// `None` means no real service metric has been observed by this pump yet. The
    /// channel deliberately has no fabricated numeric default.
    pub fn metrics_receiver(&self) -> watch::Receiver<Option<MetricsSnapshot>> {
        self.metrics_tx.subscribe()
    }

    /// Whether foreground request routing is no longer safe to reuse.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Negotiate the legacy protocol version before normal requests are sent.
    pub async fn negotiate(&self, version: u32) -> Result<()> {
        match self.request(IpcRequest::Hello { version }).await? {
            IpcResponse::HelloAck { server_version } if server_version == version => Ok(()),
            IpcResponse::HelloAck { server_version } => anyhow::bail!(
                "legacy IPC protocol mismatch: requested {version}, server acknowledged {server_version}"
            ),
            IpcResponse::Error(message) => anyhow::bail!("legacy IPC handshake rejected: {message}"),
            other => anyhow::bail!("unexpected legacy IPC handshake response: {other:?}"),
        }
    }

    /// Send one foreground request and wait for its reply.
    ///
    /// Foreground calls serialize behind `request_gate`; pushed metrics continue
    /// to flow independently through the reader task while a request is pending.
    /// A `GetMetrics` request may be satisfied by the first metrics frame observed
    /// while it is pending; that same frame is also published to the state watch.
    pub async fn request(&self, request: IpcRequest) -> Result<IpcResponse> {
        let _gate = self.request_gate.lock().await;
        self.ensure_usable()?;

        let accepts_metrics = matches!(&request, IpcRequest::GetMetrics);
        let (reply_tx, reply_rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().await;
            if pending.is_some() {
                self.poisoned.store(true, Ordering::Release);
                anyhow::bail!("legacy IPC invariant violated: multiple pending foreground replies");
            }
            *pending = Some(PendingForeground {
                accepts_metrics,
                reply: reply_tx,
            });
        }

        if let Err(error) = self.write_request(&request).await {
            self.poisoned.store(true, Ordering::Release);
            self.pending.lock().await.take();
            return Err(error);
        }

        match tokio::time::timeout(self.request_timeout, reply_rx).await {
            Ok(Ok(Ok(response))) => Ok(response),
            Ok(Ok(Err(message))) => {
                self.poisoned.store(true, Ordering::Release);
                anyhow::bail!("legacy IPC reader failed: {message}")
            }
            Ok(Err(_closed)) => {
                self.poisoned.store(true, Ordering::Release);
                anyhow::bail!("legacy IPC reader dropped foreground response channel")
            }
            Err(_elapsed) => {
                // Protocol v1 has no response IDs. Once a request times out, a
                // future late reply cannot safely be associated with any newer
                // request. Fail closed instead of risking response confusion.
                self.poisoned.store(true, Ordering::Release);
                self.pending.lock().await.take();
                anyhow::bail!(
                    "legacy IPC foreground request timed out; connection is response-desynchronized"
                )
            }
        }
    }

    /// Ask the service to begin autonomous metrics pushes.
    pub async fn subscribe_metrics(&self) -> Result<()> {
        match self.request(IpcRequest::SubscribeMetrics).await? {
            IpcResponse::Subscribed => Ok(()),
            IpcResponse::Error(message) => anyhow::bail!("metrics subscription rejected: {message}"),
            other => anyhow::bail!("unexpected metrics subscription response: {other:?}"),
        }
    }

    /// Ask the service to stop autonomous metrics pushes.
    pub async fn unsubscribe_metrics(&self) -> Result<()> {
        match self.request(IpcRequest::UnsubscribeMetrics).await? {
            IpcResponse::Subscribed => Ok(()),
            IpcResponse::Error(message) => anyhow::bail!("metrics unsubscribe rejected: {message}"),
            other => anyhow::bail!("unexpected metrics unsubscribe response: {other:?}"),
        }
    }

    fn ensure_usable(&self) -> Result<()> {
        if self.is_poisoned() {
            anyhow::bail!("legacy IPC foreground routing is poisoned; reconnect before reuse")
        }
        Ok(())
    }

    async fn write_request(&self, request: &IpcRequest) -> Result<()> {
        let data = rmp_serde::to_vec(request).context("Failed to encode legacy IPC request")?;
        if data.is_empty() || data.len() > LEGACY_IPC_MAX_FRAME_BYTES {
            anyhow::bail!("invalid legacy IPC request frame size: {}", data.len());
        }

        let len = (data.len() as u32).to_le_bytes();
        let mut writer = self.writer.lock().await;
        writer
            .write_all(&len)
            .await
            .context("Failed to write legacy IPC request length")?;
        writer
            .write_all(&data)
            .await
            .context("Failed to write legacy IPC request payload")?;
        writer
            .flush()
            .await
            .context("Failed to flush legacy IPC request")?;
        Ok(())
    }

    async fn reader_loop(
        mut reader: BufReader<OwnedReadHalf>,
        pending: Arc<Mutex<Option<PendingForeground>>>,
        metrics_tx: watch::Sender<Option<MetricsSnapshot>>,
        poisoned: Arc<AtomicBool>,
    ) {
        loop {
            let response = match Self::read_response(&mut reader).await {
                Ok(response) => response,
                Err(error) => {
                    poisoned.store(true, Ordering::Release);
                    if let Some(waiter) = pending.lock().await.take() {
                        let _ = waiter.reply.send(Err(error.to_string()));
                    }
                    break;
                }
            };

            match response {
                IpcResponse::Metrics(metrics) => {
                    metrics_tx.send_replace(Some(metrics.clone()));

                    let waiter = {
                        let mut pending = pending.lock().await;
                        if pending
                            .as_ref()
                            .map(|waiter| waiter.accepts_metrics)
                            .unwrap_or(false)
                        {
                            pending.take()
                        } else {
                            None
                        }
                    };
                    if let Some(waiter) = waiter {
                        let _ = waiter.reply.send(Ok(IpcResponse::Metrics(metrics)));
                    }
                }
                foreground => {
                    let waiter = pending.lock().await.take();
                    match waiter {
                        Some(waiter) => {
                            let _ = waiter.reply.send(Ok(foreground));
                        }
                        None => {
                            // A non-metrics response with no waiter means protocol
                            // v1 ordering is no longer trustworthy (commonly a late
                            // reply after timeout). Keep state streaming possible,
                            // but forbid any newer foreground request on this socket.
                            poisoned.store(true, Ordering::Release);
                            tracing::warn!(
                                "legacy IPC received an uncorrelated foreground response; reconnect required"
                            );
                        }
                    }
                }
            }
        }
    }

    async fn read_response(reader: &mut BufReader<OwnedReadHalf>) -> Result<IpcResponse> {
        let mut len_buf = [0u8; 4];
        reader
            .read_exact(&mut len_buf)
            .await
            .context("Failed to read legacy IPC response length")?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len == 0 || len > LEGACY_IPC_MAX_FRAME_BYTES {
            anyhow::bail!("invalid legacy IPC response frame size: {len}");
        }

        let mut payload = vec![0u8; len];
        reader
            .read_exact(&mut payload)
            .await
            .context("Failed to read legacy IPC response payload")?;
        rmp_serde::from_slice(&payload).context("Failed to decode legacy IPC response")
    }
}

impl Drop for LegacyDuplexPump {
    fn drop(&mut self) {
        self.reader_task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            coherence: 0.8,
            is_conscious: true,
            consciousness_level: phi,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn negotiation_is_an_explicit_foreground_exchange() {
        let (client, mut server) = UnixStream::pair().unwrap();
        let pump = LegacyDuplexPump::from_stream(client, Duration::from_secs(1));

        let server_task = tokio::spawn(async move {
            assert!(matches!(
                read_request(&mut server).await,
                IpcRequest::Hello { version: 1 }
            ));
            write_response(
                &mut server,
                &IpcResponse::HelloAck { server_version: 1 },
            )
            .await;
        });

        pump.negotiate(1).await.unwrap();
        assert!(!pump.is_poisoned());
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn metrics_are_routed_without_consuming_foreground_reply() {
        let (client, mut server) = UnixStream::pair().unwrap();
        let pump = LegacyDuplexPump::from_stream(client, Duration::from_secs(1));
        let mut metrics = pump.metrics_receiver();
        assert!(metrics.borrow().is_none());

        let server_task = tokio::spawn(async move {
            assert!(matches!(read_request(&mut server).await, IpcRequest::Ping));
            write_response(&mut server, &IpcResponse::Metrics(observed_metrics(0.73))).await;
            write_response(&mut server, &IpcResponse::Pong).await;
        });

        let response = pump.request(IpcRequest::Ping).await.unwrap();
        assert!(matches!(response, IpcResponse::Pong));
        metrics.changed().await.unwrap();
        assert!((metrics.borrow().as_ref().unwrap().phi - 0.73).abs() < f64::EPSILON);
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn get_metrics_response_updates_state_and_completes_foreground_request() {
        let (client, mut server) = UnixStream::pair().unwrap();
        let pump = LegacyDuplexPump::from_stream(client, Duration::from_secs(1));
        let mut metrics = pump.metrics_receiver();

        let server_task = tokio::spawn(async move {
            assert!(matches!(
                read_request(&mut server).await,
                IpcRequest::GetMetrics
            ));
            write_response(&mut server, &IpcResponse::Metrics(observed_metrics(0.66))).await;
        });

        let response = pump.request(IpcRequest::GetMetrics).await.unwrap();
        match response {
            IpcResponse::Metrics(snapshot) => {
                assert!((snapshot.phi - 0.66).abs() < f64::EPSILON);
            }
            other => panic!("unexpected GetMetrics response: {other:?}"),
        }
        metrics.changed().await.unwrap();
        assert_eq!(metrics.borrow().as_ref().map(|m| m.phi), Some(0.66));
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn subscription_really_sends_request_then_receives_autonomous_metrics() {
        let (client, mut server) = UnixStream::pair().unwrap();
        let pump = LegacyDuplexPump::from_stream(client, Duration::from_secs(1));
        let mut metrics = pump.metrics_receiver();

        let server_task = tokio::spawn(async move {
            assert!(matches!(
                read_request(&mut server).await,
                IpcRequest::SubscribeMetrics
            ));
            write_response(&mut server, &IpcResponse::Subscribed).await;
            write_response(&mut server, &IpcResponse::Metrics(observed_metrics(0.91))).await;
        });

        pump.subscribe_metrics().await.unwrap();
        metrics.changed().await.unwrap();
        assert!((metrics.borrow().as_ref().unwrap().phi - 0.91).abs() < f64::EPSILON);
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn latest_metrics_replace_older_state() {
        let (client, mut server) = UnixStream::pair().unwrap();
        let pump = LegacyDuplexPump::from_stream(client, Duration::from_secs(1));
        let mut metrics = pump.metrics_receiver();

        write_response(&mut server, &IpcResponse::Metrics(observed_metrics(0.1))).await;
        write_response(&mut server, &IpcResponse::Metrics(observed_metrics(0.2))).await;
        write_response(&mut server, &IpcResponse::Metrics(observed_metrics(0.3))).await;

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if metrics.borrow().as_ref().map(|m| m.phi) == Some(0.3) {
                    break;
                }
                metrics.changed().await.unwrap();
            }
        })
        .await
        .expect("reader should converge to newest metrics");

        assert_eq!(metrics.borrow().as_ref().map(|m| m.phi), Some(0.3));
    }

    #[tokio::test]
    async fn foreground_timeout_poisoning_prevents_late_reply_confusion() {
        let (client, mut server) = UnixStream::pair().unwrap();
        let pump = LegacyDuplexPump::from_stream(client, Duration::from_millis(10));

        let server_task = tokio::spawn(async move {
            assert!(matches!(read_request(&mut server).await, IpcRequest::Ping));
            tokio::time::sleep(Duration::from_millis(50)).await;
            write_response(&mut server, &IpcResponse::Pong).await;
        });

        assert!(pump.request(IpcRequest::Ping).await.is_err());
        assert!(pump.is_poisoned());
        assert!(pump.request(IpcRequest::Ping).await.is_err());
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn uncorrelated_foreground_response_poisoning_is_fail_closed() {
        let (client, mut server) = UnixStream::pair().unwrap();
        let pump = LegacyDuplexPump::from_stream(client, Duration::from_secs(1));

        write_response(&mut server, &IpcResponse::Pong).await;
        tokio::time::timeout(Duration::from_secs(1), async {
            while !pump.is_poisoned() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("uncorrelated response should poison foreground routing");

        assert!(pump.is_poisoned());
    }
}
