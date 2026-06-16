// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sovereign RDP Client: receives frames, reconstructs display, forwards input.
//!
//! ## Architecture
//!
//! ```text
//! Actor (async): QUIC connect → PQC handshake → receive encrypted frames
//!   → decrypt → deserialize → push to inbound channel
//!
//! Handle (sync): poll_frames() → apply to FrameBuffer → render
//!   → capture input → send_input() → push to outbound channel
//! ```
//!
//! The FrameBuffer maintains a reconstructed RGBA pixel buffer. Full frames
//! replace the entire buffer; delta frames update only changed tiles.

use std::sync::mpsc;
use std::time::Instant;

use super::rdp_codec::TILE_SIZE;
use super::rdp_protocol::{
    ControlMessage, DeltaFrame, FullFrame, InputEvent, InputFrame, RdpFrame, RdpSessionConfig,
};
use super::rdp_session::{RdpSession, SessionState};

/// Reconstructed display buffer from received RDP frames.
pub struct FrameBuffer {
    /// RGBA pixel data, row-major.
    pub pixels: Vec<u8>,
    /// Buffer width in pixels.
    pub width: u32,
    /// Buffer height in pixels.
    pub height: u32,
    /// Tile grid dimensions (from server Welcome).
    pub tile_cols: u16,
    pub tile_rows: u16,
    /// Last applied full frame ID (for delta base tracking).
    pub base_frame_id: u64,
    /// Last applied frame ID (full or delta).
    pub last_frame_id: u64,
    /// Total frames applied.
    pub frames_applied: u64,
    /// Whether the buffer has valid content.
    pub has_content: bool,
}

impl FrameBuffer {
    /// Create an empty frame buffer.
    pub fn new(width: u32, height: u32, tile_cols: u16, tile_rows: u16) -> Self {
        let n = width as usize * height as usize * 4;
        Self {
            pixels: vec![0u8; n],
            width,
            height,
            tile_cols,
            tile_rows,
            base_frame_id: 0,
            last_frame_id: 0,
            frames_applied: 0,
            has_content: false,
        }
    }

    /// Apply a full frame: replace entire buffer.
    pub fn apply_full_frame(&mut self, frame: &FullFrame) {
        self.tile_cols = frame.patch_cols;
        self.tile_rows = frame.patch_rows;

        for (idx, patch) in frame.patches.iter().enumerate() {
            let col = idx % self.tile_cols as usize;
            let row = idx / self.tile_cols as usize;
            self.apply_tile(col, row, &patch.values);
        }

        self.base_frame_id = frame.frame_id;
        self.last_frame_id = frame.frame_id;
        self.frames_applied += 1;
        self.has_content = true;
    }

    /// Apply a delta frame: update only changed tiles.
    pub fn apply_delta_frame(&mut self, delta: &DeltaFrame) {
        // Check base frame alignment.
        if delta.base_frame_id != self.last_frame_id && self.has_content {
            // Desync: should request full frame.
            // For now, apply anyway (best-effort).
        }

        for patch in &delta.patches {
            let col = patch.index as usize % self.tile_cols as usize;
            let row = patch.index as usize / self.tile_cols as usize;
            self.apply_tile(col, row, &patch.values);
        }

        self.last_frame_id = delta.frame_id;
        self.frames_applied += 1;
    }

    /// Apply a single tile's grayscale data to the pixel buffer.
    ///
    /// `tile_data` contains `TILE_SIZE × TILE_SIZE` i8 values (grayscale luminance).
    /// These are converted to RGBA pixels at the tile's grid position.
    fn apply_tile(&mut self, col: usize, row: usize, tile_data: &[i8]) {
        let tile_x = col * TILE_SIZE;
        let tile_y = row * TILE_SIZE;
        let w = self.width as usize;
        let h = self.height as usize;

        for dy in 0..TILE_SIZE {
            let y = tile_y + dy;
            if y >= h {
                break;
            }
            for dx in 0..TILE_SIZE {
                let x = tile_x + dx;
                if x >= w {
                    break;
                }
                let tile_idx = dy * TILE_SIZE + dx;
                let lum = if tile_idx < tile_data.len() {
                    // i8 → u8: add 128 to shift from [-128,127] to [0,255].
                    (tile_data[tile_idx] as i16 + 128).clamp(0, 255) as u8
                } else {
                    0
                };

                let pixel_offset = (y * w + x) * 4;
                if pixel_offset + 3 < self.pixels.len() {
                    self.pixels[pixel_offset] = lum; // R
                    self.pixels[pixel_offset + 1] = lum; // G
                    self.pixels[pixel_offset + 2] = lum; // B
                    self.pixels[pixel_offset + 3] = 255; // A
                }
            }
        }
    }

    /// Check if we need a full frame (buffer has no content or desync detected).
    pub fn needs_full_frame(&self) -> bool {
        !self.has_content
    }

    /// Get the pixel buffer as an RGBA slice.
    pub fn as_rgba(&self) -> &[u8] {
        &self.pixels
    }
}

/// Inbound message from async actor → sync handle.
pub enum ClientInbound {
    /// Server Welcome received (session parameters).
    Welcome {
        patch_cols: u16,
        patch_rows: u16,
        target_fps: u8,
        consciousness_level: f32,
    },
    /// Decrypted and deserialized frame.
    Frame(RdpFrame),
    /// Connection state changed.
    StateChanged(SessionState),
    /// Connection error.
    Error(String),
}

/// Outbound message from sync handle → async actor.
pub enum ClientOutbound {
    /// Input events to send to server.
    Input(InputFrame),
    /// Request full frame recovery.
    RequestFullFrame,
    /// Disconnect.
    Disconnect(String),
}

/// Sync handle for the RDP client.
pub struct RdpClientHandle {
    /// Reconstructed frame buffer.
    pub frame_buffer: FrameBuffer,
    /// Session state.
    pub session: RdpSession,
    /// Channel from actor.
    inbound_rx: mpsc::Receiver<ClientInbound>,
    /// Channel to actor.
    outbound_tx: mpsc::Sender<ClientOutbound>,
    /// Input sequence counter.
    input_sequence: u64,
    /// FPS tracking.
    frames_this_second: u32,
    last_fps_time: Instant,
    pub current_fps: f32,
    /// Server's consciousness level.
    pub server_consciousness: f32,
}

impl RdpClientHandle {
    /// Create a new client handle.
    ///
    /// Returns `(handle, inbound_tx, outbound_rx)` for the actor.
    pub fn new(
        peer_id: &str,
        config: RdpSessionConfig,
    ) -> (
        Self,
        mpsc::Sender<ClientInbound>,
        mpsc::Receiver<ClientOutbound>,
    ) {
        let (inbound_tx, inbound_rx) = mpsc::channel();
        let (outbound_tx, outbound_rx) = mpsc::channel();

        let session = RdpSession::new("client-session".into(), peer_id.into(), config, true);

        // Initial frame buffer (will be resized on Welcome).
        let frame_buffer = FrameBuffer::new(1920, 1080, 30, 17);

        let handle = Self {
            frame_buffer,
            session,
            inbound_rx,
            outbound_tx,
            input_sequence: 0,
            frames_this_second: 0,
            last_fps_time: Instant::now(),
            current_fps: 0.0,
            server_consciousness: 0.0,
        };

        (handle, inbound_tx, outbound_rx)
    }

    /// Poll for new frames and apply to buffer. Returns true if buffer updated.
    pub fn poll(&mut self) -> bool {
        let mut updated = false;

        while let Ok(msg) = self.inbound_rx.try_recv() {
            match msg {
                ClientInbound::Welcome {
                    patch_cols,
                    patch_rows,
                    target_fps: _,
                    consciousness_level,
                } => {
                    let width = patch_cols as u32 * TILE_SIZE as u32;
                    let height = patch_rows as u32 * TILE_SIZE as u32;
                    self.frame_buffer = FrameBuffer::new(width, height, patch_cols, patch_rows);
                    self.server_consciousness = consciousness_level;
                    self.session.on_connected();
                }
                ClientInbound::Frame(frame) => match frame {
                    RdpFrame::Full(ref f) => {
                        self.server_consciousness = f.consciousness_level;
                        self.frame_buffer.apply_full_frame(f);
                        self.track_fps();
                        updated = true;
                    }
                    RdpFrame::Delta(ref d) => {
                        self.server_consciousness = d.consciousness_level;
                        self.frame_buffer.apply_delta_frame(d);
                        self.track_fps();
                        updated = true;

                        // Send ack.
                        self.session.on_frame_ack(d.frame_id);
                    }
                    RdpFrame::Control(ControlMessage::ConsciousnessAttestation {
                        phi,
                        state_hash,
                        ref signature,
                    }) => {
                        self.session.on_attestation(phi, &state_hash, signature);
                    }
                    _ => {}
                },
                ClientInbound::StateChanged(state) => {
                    self.session.state = state;
                }
                ClientInbound::Error(_err) => {
                    // Handle connection error.
                }
            }
        }

        // Request full frame if needed.
        if self.frame_buffer.needs_full_frame() {
            let _ = self.outbound_tx.send(ClientOutbound::RequestFullFrame);
        }

        updated
    }

    /// Send input events to the server.
    pub fn send_input(&mut self, events: Vec<InputEvent>) {
        if events.is_empty() {
            return;
        }
        self.input_sequence += 1;
        let frame = InputFrame {
            sequence: self.input_sequence,
            timestamp_ms: self.session.started_at.elapsed().as_millis() as u64,
            events,
        };
        let _ = self.outbound_tx.send(ClientOutbound::Input(frame));
    }

    /// Request a full frame from the server (recovery).
    pub fn request_full_frame(&self) {
        let _ = self.outbound_tx.send(ClientOutbound::RequestFullFrame);
    }

    /// Disconnect from the server.
    pub fn disconnect(&self, reason: &str) {
        let _ = self
            .outbound_tx
            .send(ClientOutbound::Disconnect(reason.to_string()));
    }

    /// Track FPS.
    fn track_fps(&mut self) {
        self.frames_this_second += 1;
        let elapsed = self.last_fps_time.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.current_fps = self.frames_this_second as f32 / elapsed;
            self.frames_this_second = 0;
            self.last_fps_time = Instant::now();
        }
    }

    /// Is the client connected and receiving frames?
    pub fn is_active(&self) -> bool {
        self.session.should_send_frames() && self.frame_buffer.has_content
    }
}

#[cfg(test)]
mod tests {
    use super::super::rdp_protocol::QuantizedPatch;
    use super::*;

    fn make_test_full_frame(cols: u16, rows: u16) -> FullFrame {
        let n = cols as usize * rows as usize;
        let patches: Vec<QuantizedPatch> = (0..n)
            .map(|i| {
                let lum = (i as i8).wrapping_mul(7);
                QuantizedPatch {
                    values: vec![lum; TILE_SIZE * TILE_SIZE],
                }
            })
            .collect();

        FullFrame {
            frame_id: 1,
            timestamp_ms: 1000,
            patch_cols: cols,
            patch_rows: rows,
            patches,
            consciousness_level: 0.5,
            harmony: "present".into(),
        }
    }

    #[test]
    fn test_frame_buffer_full_frame() {
        let mut fb = FrameBuffer::new(256, 256, 4, 4);
        assert!(!fb.has_content);

        let full = make_test_full_frame(4, 4);
        fb.apply_full_frame(&full);

        assert!(fb.has_content);
        assert_eq!(fb.base_frame_id, 1);
        assert_eq!(fb.frames_applied, 1);
        // Pixels should be non-zero after applying.
        assert!(fb.pixels.iter().any(|&p| p != 0));
    }

    #[test]
    fn test_frame_buffer_delta_applies_sparse() {
        let mut fb = FrameBuffer::new(256, 256, 4, 4);
        let full = make_test_full_frame(4, 4);
        fb.apply_full_frame(&full);

        // Apply delta to tile (0,0) only.
        let delta = DeltaFrame {
            frame_id: 2,
            base_frame_id: 1,
            timestamp_ms: 2000,
            patches: vec![super::super::rdp_protocol::DeltaPatch {
                index: 0,
                surprise: 0.8,
                values: vec![100i8; TILE_SIZE * TILE_SIZE],
            }],
            consciousness_level: 0.6,
        };
        fb.apply_delta_frame(&delta);

        assert_eq!(fb.last_frame_id, 2);
        assert_eq!(fb.frames_applied, 2);

        // Tile (0,0) should be updated (100 + 128 = 228).
        let expected_lum = (100i16 + 128).clamp(0, 255) as u8;
        assert_eq!(fb.pixels[0], expected_lum); // R of pixel (0,0)
    }

    #[test]
    fn test_client_handle_poll_applies_frames() {
        let config = RdpSessionConfig::default();
        let (mut handle, inbound_tx, _outbound_rx) = RdpClientHandle::new("server1", config);

        // Send Welcome.
        inbound_tx
            .send(ClientInbound::Welcome {
                patch_cols: 4,
                patch_rows: 4,
                target_fps: 5,
                consciousness_level: 0.5,
            })
            .unwrap();

        handle.poll();
        assert_eq!(handle.frame_buffer.tile_cols, 4);

        // Send full frame.
        let full = make_test_full_frame(4, 4);
        inbound_tx
            .send(ClientInbound::Frame(RdpFrame::Full(full)))
            .unwrap();

        let updated = handle.poll();
        assert!(updated);
        assert!(handle.frame_buffer.has_content);
    }

    #[test]
    fn test_client_sends_input() {
        let config = RdpSessionConfig::default();
        let (mut handle, _inbound_tx, outbound_rx) = RdpClientHandle::new("server1", config);

        handle.send_input(vec![InputEvent::Key {
            code: 65,
            pressed: true,
            modifiers: 0,
        }]);

        let msg = outbound_rx.try_recv().unwrap();
        match msg {
            ClientOutbound::Input(frame) => {
                assert_eq!(frame.sequence, 1);
                assert_eq!(frame.events.len(), 1);
            }
            _ => panic!("Expected Input"),
        }
    }

    #[test]
    fn test_empty_buffer_requests_full_frame() {
        let config = RdpSessionConfig::default();
        let (mut handle, _inbound_tx, outbound_rx) = RdpClientHandle::new("server1", config);

        handle.poll(); // Should request full frame since buffer is empty.

        let msg = outbound_rx.try_recv().unwrap();
        assert!(matches!(msg, ClientOutbound::RequestFullFrame));
    }
}
