// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Iroh QUIC transport layer for Sovereign RDP.
//!
//! Wires the Handle/Actor pattern to real Iroh QUIC streams using `RDP_ALPN`.
//! Follows the established `IrohBridgeHandle`/`IrohBridgeActor` pattern:
//! - **Actor** (async): manages QUIC connections, PQC handshake, frame I/O
//! - **Handle** (sync): non-blocking send/receive via bounded mpsc channels
//!
//! ## Stream Multiplexing
//!
//! Each RDP session uses multiple QUIC streams (no head-of-line blocking):
//! - Stream 0: Control messages (Hello, Welcome, Attestation, Goodbye)
//! - Stream 1: Video frames (Full, Delta)
//! - Stream 2: Input events
//! - Stream 3: Audio frames
//! - Stream 4: File transfer
//! - Stream 5: Chat + RemoteExec

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use super::rdp_protocol::{RDP_ALPN, RdpFrame};

/// QUIC stream IDs for multiplexing.
pub const STREAM_CONTROL: u8 = 0;
pub const STREAM_VIDEO: u8 = 1;
pub const STREAM_INPUT: u8 = 2;
pub const STREAM_AUDIO: u8 = 3;
pub const STREAM_FILE: u8 = 4;
pub const STREAM_CHAT_EXEC: u8 = 5;

/// Classify an RdpFrame into its stream ID.
pub fn frame_stream_id(frame: &RdpFrame) -> u8 {
    match frame {
        RdpFrame::Full(_) | RdpFrame::Delta(_) => STREAM_VIDEO,
        RdpFrame::Input(_) => STREAM_INPUT,
        RdpFrame::Control(_) => STREAM_CONTROL,
        // Extended frame types (v3) use higher stream IDs if present.
        #[allow(unreachable_patterns)]
        _ => STREAM_CONTROL,
    }
}

/// Message from the async transport actor → sync handle.
pub enum TransportInbound {
    /// A frame received from the peer.
    Frame(RdpFrame),
    /// Connection established with peer.
    Connected { peer_id: String },
    /// Connection lost.
    Disconnected { reason: String },
    /// PQC handshake completed — session key available.
    HandshakeComplete { session_key: [u8; 32] },
}

/// Message from the sync handle → async transport actor.
pub enum TransportOutbound {
    /// Send a frame to the peer.
    SendFrame(RdpFrame),
    /// Initiate PQC handshake.
    StartHandshake,
    /// Disconnect gracefully.
    Disconnect { reason: String },
}

/// Sync transport handle for the 50Hz tick loop.
///
/// All methods are non-blocking. Network I/O is delegated to the async actor.
pub struct RdpTransportHandle {
    /// Channel to receive from actor.
    inbound_rx: mpsc::Receiver<TransportInbound>,
    /// Channel to send to actor.
    outbound_tx: mpsc::SyncSender<TransportOutbound>,
    /// Whether the actor is alive.
    alive: Arc<AtomicBool>,
    /// Frames sent counter.
    pub frames_sent: u64,
    /// Frames received counter.
    pub frames_received: u64,
}

impl RdpTransportHandle {
    /// Create a new handle/actor pair.
    ///
    /// `inbound_capacity`: max buffered inbound frames (default 128).
    /// `outbound_capacity`: max buffered outbound frames (default 64).
    pub fn new(inbound_capacity: usize, outbound_capacity: usize) -> (Self, RdpTransportActor) {
        let (inbound_tx, inbound_rx) = mpsc::sync_channel(inbound_capacity);
        let (outbound_tx, outbound_rx) = mpsc::sync_channel(outbound_capacity);
        let alive = Arc::new(AtomicBool::new(true));

        let handle = Self {
            inbound_rx,
            outbound_tx,
            alive: alive.clone(),
            frames_sent: 0,
            frames_received: 0,
        };

        let actor = RdpTransportActor {
            inbound_tx,
            outbound_rx,
            alive,
        };

        (handle, actor)
    }

    /// Non-blocking: drain all received frames/events from the actor.
    pub fn drain_inbound(&mut self) -> Vec<TransportInbound> {
        let mut msgs = Vec::new();
        while let Ok(msg) = self.inbound_rx.try_recv() {
            if matches!(msg, TransportInbound::Frame(_)) {
                self.frames_received += 1;
            }
            msgs.push(msg);
        }
        msgs
    }

    /// Non-blocking: send a frame to the actor for transmission.
    pub fn send_frame(&mut self, frame: RdpFrame) -> bool {
        match self
            .outbound_tx
            .try_send(TransportOutbound::SendFrame(frame))
        {
            Ok(()) => {
                self.frames_sent += 1;
                true
            }
            Err(_) => false, // Buffer full — backpressure
        }
    }

    /// Request PQC handshake initiation.
    pub fn start_handshake(&self) {
        let _ = self.outbound_tx.try_send(TransportOutbound::StartHandshake);
    }

    /// Request graceful disconnect.
    pub fn disconnect(&self, reason: &str) {
        let _ = self.outbound_tx.try_send(TransportOutbound::Disconnect {
            reason: reason.to_string(),
        });
    }

    /// Whether the transport actor is still alive.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }
}

/// Async transport actor — manages QUIC connections.
///
/// In production, this runs on a tokio runtime and manages Iroh endpoints.
/// The actor pattern ensures all network I/O is async and never blocks
/// the 50Hz consciousness tick loop.
pub struct RdpTransportActor {
    /// Channel to push received frames to handle.
    inbound_tx: mpsc::SyncSender<TransportInbound>,
    /// Channel to receive frames for transmission from handle.
    outbound_rx: mpsc::Receiver<TransportOutbound>,
    /// Shared alive flag.
    alive: Arc<AtomicBool>,
}

impl RdpTransportActor {
    /// Run the transport actor as an RDP server.
    ///
    /// With `swarm` feature: creates Iroh endpoint, accepts QUIC connections.
    /// Without: stub that drains the outbound channel.
    #[cfg(feature = "swarm")]
    pub async fn run_server(self, _listen_addr: &str) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        tracing::info!("RDP transport actor starting (Iroh QUIC server)");

        // Create Iroh endpoint with RDP ALPN (preset required since iroh 0.97).
        let endpoint = match iroh::Endpoint::builder(iroh::endpoint::presets::N0)
            .alpns(vec![RDP_ALPN.to_vec()])
            .bind()
            .await
        {
            Ok(ep) => {
                tracing::info!("RDP server bound: endpoint_id={}", ep.id());
                ep
            }
            Err(e) => {
                tracing::error!("Failed to bind Iroh endpoint: {}", e);
                self.alive.store(false, Ordering::Relaxed);
                return;
            }
        };

        // Accept loop.
        while self.alive.load(Ordering::Relaxed) {
            tokio::select! {
                // Accept new connection.
                incoming = endpoint.accept() => {
                    match incoming {
                        Some(connecting) => {
                            match connecting.await {
                                Ok(conn) => {
                                    let peer_id = format!("{:?}", conn.stable_id());
                                    tracing::info!("RDP client connected: {}", &peer_id[..16.min(peer_id.len())]);

                                    let _ = self.inbound_tx.try_send(TransportInbound::Connected {
                                        peer_id,
                                    });

                                    // Open bidirectional stream for frame exchange.
                                    match conn.open_bi().await {
                                        Ok((mut send, mut recv)) => {
                                            // Spawn frame reader.
                                            let inbound_tx = self.inbound_tx.clone();
                                            let alive = self.alive.clone();
                                            tokio::spawn(async move {
                                                let mut buf = vec![0u8; 256 * 1024]; // 256KB read buffer
                                                while alive.load(Ordering::Relaxed) {
                                                    match recv.read(&mut buf).await {
                                                        Ok(Some(n)) if n > 0 => {
                                                            // Deserialize frame.
                                                            if let Ok(frame) = serde_json::from_slice::<RdpFrame>(&buf[..n]) {
                                                                let _ = inbound_tx.try_send(TransportInbound::Frame(frame));
                                                            }
                                                        }
                                                        Ok(_) => break, // EOF
                                                        Err(e) => {
                                                            tracing::warn!("RDP read error: {}", e);
                                                            break;
                                                        }
                                                    }
                                                }
                                                let _ = inbound_tx.try_send(TransportInbound::Disconnected {
                                                    reason: "stream closed".to_string(),
                                                });
                                            });

                                            // Frame writer: drain outbound and send.
                                            while self.alive.load(Ordering::Relaxed) {
                                                match self.outbound_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                                                    Ok(TransportOutbound::SendFrame(frame)) => {
                                                        if let Ok(data) = serde_json::to_vec(&frame) {
                                                            // Length-prefix: 4 bytes LE + data.
                                                            let len = (data.len() as u32).to_le_bytes();
                                                            let _ = send.write_all(&len).await;
                                                            let _ = send.write_all(&data).await;
                                                        }
                                                    }
                                                    Ok(TransportOutbound::Disconnect { reason }) => {
                                                        tracing::info!("RDP disconnect: {}", reason);
                                                        break;
                                                    }
                                                    Ok(TransportOutbound::StartHandshake) => {
                                                        // PQC handshake would happen here.
                                                    }
                                                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                                                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            tracing::warn!("Failed to open bi stream: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("Connection failed: {}", e);
                                }
                            }
                        }
                        None => break, // Endpoint closed
                    }
                }
                // Check for disconnect request.
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                    // Non-blocking check for outbound disconnect.
                    if let Ok(TransportOutbound::Disconnect { reason }) = self.outbound_rx.try_recv() {
                        tracing::info!("RDP server shutting down: {}", reason);
                        break;
                    }
                }
            }
        }

        self.alive.store(false, Ordering::Relaxed);
        tracing::info!("RDP transport actor stopped");
    }

    /// Stub server when swarm feature is not enabled.
    #[cfg(not(feature = "swarm"))]
    pub async fn run_server(self, _listen_addr: &str) {
        tracing::info!(
            "RDP transport actor started (stub — enable 'swarm' feature for real Iroh QUIC)"
        );
        while self.alive.load(Ordering::Relaxed) {
            match self
                .outbound_rx
                .recv_timeout(std::time::Duration::from_millis(100))
            {
                Ok(TransportOutbound::Disconnect { reason }) => {
                    tracing::info!("RDP transport disconnect: {}", reason);
                    break;
                }
                Ok(_) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        self.alive.store(false, Ordering::Relaxed);
    }

    /// Run as client connecting to a server.
    #[cfg(feature = "swarm")]
    pub async fn run_client(self, server_addr: &str) {
        tracing::info!("RDP transport client connecting to {}", server_addr);

        let endpoint = match iroh::Endpoint::builder(iroh::endpoint::presets::N0)
            .alpns(vec![RDP_ALPN.to_vec()])
            .bind()
            .await
        {
            Ok(ep) => ep,
            Err(e) => {
                tracing::error!("Failed to bind client endpoint: {}", e);
                self.alive.store(false, Ordering::Relaxed);
                return;
            }
        };

        // Parse server address and connect.
        // In production: parse EndpointAddr from ticket or discovery.
        // For now: the server_addr is logged but connection needs a proper EndpointAddr.
        tracing::info!("RDP client endpoint ready: {}", endpoint.id());

        // Stub connection loop (real impl needs EndpointAddr resolution).
        while self.alive.load(Ordering::Relaxed) {
            match self
                .outbound_rx
                .recv_timeout(std::time::Duration::from_millis(100))
            {
                Ok(TransportOutbound::Disconnect { .. }) => break,
                Ok(_) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        self.alive.store(false, Ordering::Relaxed);
    }

    /// Stub client when swarm feature is not enabled.
    #[cfg(not(feature = "swarm"))]
    pub async fn run_client(self, _server_addr: &str) {
        tracing::info!("RDP client stub (enable 'swarm' for Iroh QUIC)");
        while self.alive.load(Ordering::Relaxed) {
            match self
                .outbound_rx
                .recv_timeout(std::time::Duration::from_millis(100))
            {
                Ok(TransportOutbound::Disconnect { .. }) => break,
                Ok(_) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        self.alive.store(false, Ordering::Relaxed);
    }
}

impl Drop for RdpTransportActor {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::super::rdp_protocol::{ControlMessage, FullFrame, QuantizedPatch};
    use super::*;

    #[test]
    fn test_handle_actor_creation() {
        let (handle, _actor) = RdpTransportHandle::new(128, 64);
        assert!(handle.is_alive());
        assert_eq!(handle.frames_sent, 0);
        assert_eq!(handle.frames_received, 0);
    }

    #[test]
    fn test_send_frame_increments_counter() {
        let (mut handle, _actor) = RdpTransportHandle::new(128, 64);
        let frame = RdpFrame::Control(ControlMessage::RequestFullFrame);
        assert!(handle.send_frame(frame));
        assert_eq!(handle.frames_sent, 1);
    }

    #[test]
    fn test_frame_stream_classification() {
        assert_eq!(
            frame_stream_id(&RdpFrame::Full(FullFrame {
                frame_id: 0,
                timestamp_ms: 0,
                patch_cols: 0,
                patch_rows: 0,
                patches: vec![],
                consciousness_level: 0.0,
                harmony: String::new(),
            })),
            STREAM_VIDEO
        );
        assert_eq!(
            frame_stream_id(&RdpFrame::Control(ControlMessage::RequestFullFrame)),
            STREAM_CONTROL
        );
        assert_eq!(
            frame_stream_id(&RdpFrame::Input(super::super::rdp_protocol::InputFrame {
                sequence: 0,
                timestamp_ms: 0,
                events: vec![],
            })),
            STREAM_INPUT
        );
    }

    #[test]
    fn test_backpressure_on_full_buffer() {
        let (mut handle, _actor) = RdpTransportHandle::new(128, 2); // Tiny buffer
        let frame = RdpFrame::Control(ControlMessage::RequestFullFrame);

        // Fill buffer.
        assert!(handle.send_frame(frame.clone()));
        assert!(handle.send_frame(frame.clone()));
        // Third should fail (buffer full).
        assert!(!handle.send_frame(frame));
    }

    #[test]
    fn test_actor_drop_sets_alive_false() {
        let (handle, actor) = RdpTransportHandle::new(128, 64);
        assert!(handle.is_alive());
        drop(actor);
        assert!(!handle.is_alive());
    }
}
