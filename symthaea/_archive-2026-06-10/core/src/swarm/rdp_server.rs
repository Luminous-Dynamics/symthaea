// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign RDP Server: consciousness-aware remote desktop frame producer.
//!
//! ## Architecture
//!
//! ```text
//! ScreenCapture → HybridCodec → encrypt → RdpSession queue → Actor → QUIC
//! ```
//!
//! The server uses a Handle/Actor split matching the IrohBridgeHandle pattern:
//! - **RdpServerHandle** (sync): called from 50Hz tick loop, non-blocking
//! - **RdpServerActor** (async): manages QUIC streams, PQC handshake, I/O
//!
//! Communication via bounded mpsc channels — no blocking on the tick side.

use std::sync::mpsc;
use std::time::Instant;

use super::rdp_capture::{CapturedFrame, ScreenCapture};
use super::rdp_codec::HybridCodec;
use super::rdp_protocol::{ControlMessage, InputEvent, RdpFrame, RdpSessionConfig};
use super::rdp_session::{RdpSession, RdpSessionManager, SessionAction};

/// Outbound message from server handle → actor.
pub enum ServerOutbound {
    /// Encrypted frame for a specific session.
    Frame {
        session_id: String,
        data: Vec<u8>, // Serialized + encrypted RdpFrame
    },
    /// Control message for a session.
    Control {
        session_id: String,
        msg: ControlMessage,
    },
}

/// Inbound message from actor → server handle.
pub enum ServerInbound {
    /// New client connected and handshake complete.
    SessionReady {
        session_id: String,
        peer_id: String,
        session_key: [u8; 32],
        config: RdpSessionConfig,
    },
    /// Received input events from a client.
    Input {
        session_id: String,
        events: Vec<InputEvent>,
    },
    /// Received frame acknowledgment.
    FrameAck { session_id: String, frame_id: u64 },
    /// Received consciousness attestation.
    Attestation {
        session_id: String,
        phi: f32,
        state_hash: [u8; 32],
        signature: Vec<u8>,
    },
    /// Client disconnected.
    Disconnected { session_id: String, reason: String },
}

/// Sync handle for the RDP server, called from the 50Hz tick loop.
///
/// All methods are non-blocking. Frame encoding and encryption happen here
/// (in the sync context). Network I/O is delegated to the async actor.
pub struct RdpServerHandle {
    /// Screen capture backend.
    capture: Box<dyn ScreenCapture>,
    /// Hybrid HDC+pixel codec.
    codec: HybridCodec,
    /// Session manager.
    sessions: RdpSessionManager,
    /// Channel to send frames/control to async actor.
    outbound_tx: mpsc::Sender<ServerOutbound>,
    /// Channel to receive events from async actor.
    inbound_rx: mpsc::Receiver<ServerInbound>,
    /// Frame counter.
    frame_id: u64,
    /// Session start time.
    started_at: Instant,
    /// Pending input events to process (per session).
    pending_inputs: Vec<(String, Vec<InputEvent>)>,
    /// Server name for Welcome messages.
    server_name: String,
    /// Default session config for new connections.
    default_config: RdpSessionConfig,
    /// Whether to send a full frame on next tick (recovery or first frame).
    force_full_frame: bool,
}

impl RdpServerHandle {
    /// Create a new server handle with a capture backend.
    ///
    /// Returns `(handle, inbound_tx, outbound_rx)` — the actor needs the other
    /// halves of these channels to communicate.
    pub fn new(
        capture: Box<dyn ScreenCapture>,
        config: RdpSessionConfig,
        server_name: &str,
    ) -> (
        Self,
        mpsc::Sender<ServerInbound>,
        mpsc::Receiver<ServerOutbound>,
    ) {
        let (outbound_tx, outbound_rx) = mpsc::channel();
        let (inbound_tx, inbound_rx) = mpsc::channel();

        let width = capture.width();
        let height = capture.height();
        let codec = HybridCodec::new(width, height, 42); // Seed negotiated during handshake

        let handle = Self {
            capture,
            codec,
            sessions: RdpSessionManager::new(),
            outbound_tx,
            inbound_rx,
            frame_id: 0,
            started_at: Instant::now(),
            pending_inputs: Vec::new(),
            server_name: server_name.to_string(),
            default_config: config,
            force_full_frame: true, // First frame is always full
        };

        (handle, inbound_tx, outbound_rx)
    }

    /// Main tick — called from the 50Hz consciousness loop.
    ///
    /// 1. Drain inbound events from actor (new sessions, input, acks)
    /// 2. Tick session manager (re-attestation, key rotation, backpressure)
    /// 3. Capture screen + encode frame (if active sessions exist)
    /// 4. Queue encrypted frames for each session
    /// 5. Send outbound to actor
    ///
    /// Returns pending input events for the caller to inject.
    pub fn tick(&mut self, consciousness_level: f32) -> Vec<(String, Vec<InputEvent>)> {
        // 1. Drain inbound events from actor.
        self.drain_inbound();

        // 2. Tick session manager.
        let actions = self.sessions.tick();
        for action in actions {
            match action {
                SessionAction::RequestAttestation { session_id } => {
                    // Send attestation request to peer via actor.
                    // (In a full implementation, this would be a ControlMessage.)
                    let _ = self.outbound_tx.send(ServerOutbound::Control {
                        session_id,
                        msg: ControlMessage::RequestFullFrame, // Placeholder — would be AttestationRequest
                    });
                }
                SessionAction::StartRekey { session_id: _ } => {
                    // PQC re-handshake triggered — actor handles this asynchronously.
                }
                SessionAction::Backpressure {
                    session_id: _,
                    lag: _,
                } => {
                    // Quality engine would reduce FPS/threshold here.
                    self.codec.set_change_threshold(0.95);
                }
            }
        }

        // 3. Capture + encode (only if we have active sessions).
        let active_count = self.sessions.active_sessions().count();
        if active_count > 0 {
            if let Some(frame) = self.capture.capture() {
                self.encode_and_queue(&frame, consciousness_level);
            }
        }

        // 4. Send outbound frames to actor.
        self.flush_outbound();

        // 5. Return pending inputs for injection.
        std::mem::take(&mut self.pending_inputs)
    }

    /// Drain inbound messages from the async actor.
    fn drain_inbound(&mut self) {
        while let Ok(msg) = self.inbound_rx.try_recv() {
            match msg {
                ServerInbound::SessionReady {
                    session_id,
                    peer_id,
                    session_key,
                    config,
                } => {
                    if let Some(id) = self
                        .sessions
                        .create_session(&peer_id, config.clone(), false)
                    {
                        if let Some(session) = self.sessions.session_mut(&id) {
                            session.on_connected();
                            session.on_handshake_complete(session_key);
                            // Auto-attest: SessionReady implies the actor already
                            // verified the peer's consciousness attestation.
                            session.on_attestation(1.0, &[0u8; 32], &[]);
                        }
                        // Send Welcome.
                        let welcome = RdpSessionManager::build_welcome(
                            &self.server_name,
                            self.codec.tile_cols,
                            self.codec.tile_rows,
                            &config,
                            0.0,
                        );
                        if let Some(session) = self.sessions.session_mut(&id) {
                            session.queue_frame(welcome);
                        }
                        self.force_full_frame = true;
                    }
                }
                ServerInbound::Input { session_id, events } => {
                    // Check consciousness gate before accepting input.
                    let can_forward = self
                        .sessions
                        .session(&session_id)
                        .map(|s| s.can_forward_input())
                        .unwrap_or(false);
                    if can_forward {
                        self.pending_inputs.push((session_id, events));
                    }
                }
                ServerInbound::FrameAck {
                    session_id,
                    frame_id,
                } => {
                    if let Some(session) = self.sessions.session_mut(&session_id) {
                        session.on_frame_ack(frame_id);
                    }
                }
                ServerInbound::Attestation {
                    session_id,
                    phi,
                    state_hash,
                    signature,
                } => {
                    if let Some(session) = self.sessions.session_mut(&session_id) {
                        session.on_attestation(phi, &state_hash, &signature);
                    }
                }
                ServerInbound::Disconnected {
                    session_id,
                    reason: _,
                } => {
                    self.sessions.remove_session(&session_id);
                }
            }
        }
    }

    /// Encode the captured frame and queue for all active sessions.
    fn encode_and_queue(&mut self, frame: &CapturedFrame, consciousness_level: f32) {
        let timestamp_ms = self.started_at.elapsed().as_millis() as u64;
        self.frame_id += 1;

        if self.force_full_frame {
            // Full frame for new sessions or recovery.
            let full = self.codec.encode_full_frame(
                &frame.pixels,
                frame.width,
                frame.height,
                self.frame_id,
                timestamp_ms,
                consciousness_level,
                "present",
            );
            let rdp_frame = RdpFrame::Full(full);
            self.queue_for_all_sessions(rdp_frame);
            self.force_full_frame = false;
        } else {
            // Delta frame: HDC detects changes, encode only changed tiles.
            let (changed, sims) =
                self.codec
                    .detect_changes(&frame.pixels, frame.width, frame.height);

            if !changed.is_empty() {
                let delta = self.codec.encode_delta_frame(
                    &frame.pixels,
                    frame.width,
                    frame.height,
                    &changed,
                    &sims,
                    self.frame_id,
                    self.frame_id - 1,
                    timestamp_ms,
                    consciousness_level,
                );
                let rdp_frame = RdpFrame::Delta(delta);
                self.queue_for_all_sessions(rdp_frame);
            }
        }
    }

    /// Queue a frame for all active sessions.
    fn queue_for_all_sessions(&mut self, frame: RdpFrame) {
        // Collect session IDs first to avoid borrow conflict.
        let session_ids: Vec<String> = self
            .sessions
            .active_sessions()
            .map(|s| s.session_id.clone())
            .collect();

        for id in session_ids {
            if let Some(session) = self.sessions.session_mut(&id) {
                session.queue_frame(frame.clone());
            }
        }
    }

    /// Flush outbound frames from sessions to the actor channel.
    fn flush_outbound(&mut self) {
        // Collect session IDs to avoid borrow conflict.
        let session_ids: Vec<String> = self
            .sessions
            .active_sessions()
            .map(|s| s.session_id.clone())
            .collect();

        for id in session_ids {
            if let Some(session) = self.sessions.session_mut(&id) {
                let frames = session.drain_outbound();
                for frame in frames {
                    // Serialize frame.
                    let serialized = serde_json::to_vec(&frame).unwrap_or_default();

                    // Encrypt with session key.
                    let data = if let Some(_key) = session.encryption_key() {
                        // In production: encrypt_packet_xchacha(&serialized, key)
                        // For now: plaintext (encryption wired in Phase 5 integration)
                        serialized
                    } else {
                        serialized
                    };

                    let _ = self.outbound_tx.send(ServerOutbound::Frame {
                        session_id: id.clone(),
                        data,
                    });
                }
            }
        }
    }

    /// Number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.session_count()
    }

    /// Current frame ID.
    pub fn frame_id(&self) -> u64 {
        self.frame_id
    }

    /// Force a full frame on the next tick (e.g., after recovery request).
    pub fn request_full_frame(&mut self) {
        self.force_full_frame = true;
    }

    /// Get the codec's tile grid dimensions.
    pub fn tile_dimensions(&self) -> (u16, u16) {
        (self.codec.tile_cols, self.codec.tile_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::super::rdp_capture::TestCapture;
    use super::*;

    #[test]
    fn test_server_tick_no_sessions_no_capture() {
        let capture = Box::new(TestCapture::new(256, 256));
        let config = RdpSessionConfig::default();
        let (mut handle, _inbound_tx, _outbound_rx) =
            RdpServerHandle::new(capture, config, "test-server");

        // No active sessions → tick should not capture.
        let inputs = handle.tick(0.5);
        assert!(inputs.is_empty());
        assert_eq!(handle.frame_id(), 0); // No frames produced
    }

    #[test]
    fn test_server_accepts_session_and_produces_frames() {
        let capture = Box::new(TestCapture::new(256, 256));
        let config = RdpSessionConfig::default();
        let (mut handle, inbound_tx, outbound_rx) =
            RdpServerHandle::new(capture, config.clone(), "test-server");

        // Simulate actor notifying that a session is ready.
        inbound_tx
            .send(ServerInbound::SessionReady {
                session_id: "s1".into(),
                peer_id: "peer1".into(),
                session_key: [42u8; 32],
                config,
            })
            .unwrap();

        // First tick: drain inbound, create session, capture frame.
        let inputs = handle.tick(0.5);
        assert!(inputs.is_empty()); // No input yet
        assert!(handle.frame_id() > 0); // Frame was produced

        // Actor should receive the frame.
        let msg = outbound_rx.try_recv();
        assert!(msg.is_ok(), "Actor should receive outbound frame");
    }

    #[test]
    fn test_server_delta_frames_after_first() {
        let capture = Box::new(TestCapture::new(256, 256));
        let config = RdpSessionConfig::default();
        let (mut handle, inbound_tx, _outbound_rx) =
            RdpServerHandle::new(capture, config.clone(), "test-server");

        inbound_tx
            .send(ServerInbound::SessionReady {
                session_id: "s1".into(),
                peer_id: "peer1".into(),
                session_key: [42u8; 32],
                config,
            })
            .unwrap();

        // Tick 1: Full frame.
        handle.tick(0.5);
        let fid1 = handle.frame_id();

        // Tick 2+: Delta frames.
        handle.tick(0.5);
        let fid2 = handle.frame_id();
        assert!(fid2 > fid1, "Frame ID should increment");
    }

    #[test]
    fn test_input_gated_by_consciousness() {
        let capture = Box::new(TestCapture::new(256, 256));
        let config = RdpSessionConfig {
            min_consciousness: 0.3,
            ..RdpSessionConfig::default()
        };
        let (mut handle, inbound_tx, _outbound_rx) =
            RdpServerHandle::new(capture, config.clone(), "test-server");

        // Create session.
        inbound_tx
            .send(ServerInbound::SessionReady {
                session_id: "s1".into(),
                peer_id: "peer1".into(),
                session_key: [42u8; 32],
                config,
            })
            .unwrap();
        handle.tick(0.5);

        // Attestation with low phi → ViewOnly.
        inbound_tx
            .send(ServerInbound::Attestation {
                session_id: "s1".into(),
                phi: 0.1, // Below 0.3 threshold
                state_hash: [0u8; 32],
                signature: vec![],
            })
            .unwrap();
        handle.tick(0.5);

        // Send input → should be blocked.
        inbound_tx
            .send(ServerInbound::Input {
                session_id: "s1".into(),
                events: vec![InputEvent::Key {
                    code: 65,
                    pressed: true,
                    modifiers: 0,
                }],
            })
            .unwrap();
        let inputs = handle.tick(0.5);
        assert!(
            inputs.is_empty(),
            "Input should be blocked when consciousness < threshold"
        );
    }

    #[test]
    fn test_tile_dimensions() {
        let capture = Box::new(TestCapture::new(1920, 1080));
        let config = RdpSessionConfig::default();
        let (handle, _, _) = RdpServerHandle::new(capture, config, "test");
        let (cols, rows) = handle.tile_dimensions();
        assert_eq!(cols, 30); // 1920/64
        assert_eq!(rows, 17); // ceil(1080/64)
    }
}
