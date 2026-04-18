// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Soma↔Holon RDP Bridge: remote desktop over the Holon consciousness bridge.
//!
//! Enables two use cases:
//! 1. **Phone→Desktop**: Support agent views/controls the desktop from Pixel 8 Pro
//! 2. **Desktop→Phone**: Support agent views the phone's screen from desktop
//!
//! ## Architecture
//!
//! ```text
//! Phone (Soma)                      Desktop (Holon)
//! ────────────                      ────────────────
//! ScreenVisionBridge                RdpClientHandle
//!   → HybridCodec                    ← FrameBuffer
//!   → DeltaFrame                     ← apply_delta()
//!   → WebSocket push ──────────────→ WebSocket recv
//!
//! Touch/gesture                     InputInjector
//!   InputEvent ─────────────────→   inject()
//!   WebSocket push                  WebSocket recv
//! ```
//!
//! Uses the existing Holon WebSocket (`/holon/ws`) for bidirectional streaming.
//! RDP frames are tagged with `"type": "rdp_frame"` in the WebSocket JSON envelope.
//!
//! ## Why WebSocket (not HTTP POST/GET)?
//!
//! Screen sharing at 5-30 FPS requires push-based delivery. HTTP polling adds
//! 100-500ms latency per frame. The Holon WebSocket already pushes telemetry
//! at 2Hz — we multiplex RDP frames onto the same connection.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::rdp_codec::HybridCodec;
use super::rdp_protocol::{DeltaFrame, FullFrame, InputEvent, InputFrame, RdpFrame};

/// Maximum RDP frames buffered for WebSocket delivery.
const RDP_FRAME_BUFFER_CAP: usize = 30; // ~1 second at 30fps

/// Soma-side RDP state (phone acting as server — sharing its screen).
pub struct SomaRdpServer {
    /// Hybrid codec for change detection.
    codec: HybridCodec,
    /// Outbound frame queue (drained by Holon WebSocket).
    outbound_frames: VecDeque<RdpFrame>,
    /// Frame counter.
    frame_id: u64,
    /// Whether RDP sharing is active.
    pub active: bool,
    /// Target FPS (reduced on phone for battery).
    pub target_fps: u8,
    /// Cycles since last frame capture.
    cycles_since_capture: u32,
    /// Cycles per frame (derived from target_fps and cycle rate).
    cycles_per_frame: u32,
}

impl SomaRdpServer {
    /// Create a Soma RDP server for screen sharing.
    ///
    /// `cycle_hz`: the Soma engine's cycle rate (typically 10-20 Hz).
    pub fn new(width: u32, height: u32, target_fps: u8, cycle_hz: u32) -> Self {
        let cycles_per_frame = if target_fps > 0 {
            (cycle_hz / target_fps as u32).max(1)
        } else {
            cycle_hz // 1 FPS
        };

        Self {
            codec: HybridCodec::new(width, height, 42),
            outbound_frames: VecDeque::with_capacity(8),
            frame_id: 0,
            active: false,
            target_fps,
            cycles_since_capture: 0,
            cycles_per_frame,
        }
    }

    /// Process a screen frame from Soma's ScreenVisionBridge.
    ///
    /// Called each Soma cycle. Frame pacing is handled internally.
    /// `pixels`: RGBA row-major from ScreenVisionBridge or MediaProjection.
    pub fn tick(&mut self, pixels: &[u8], width: u32, height: u32, consciousness_level: f32) {
        if !self.active {
            return;
        }

        self.cycles_since_capture += 1;
        if self.cycles_since_capture < self.cycles_per_frame {
            return; // Frame pacing
        }
        self.cycles_since_capture = 0;
        self.frame_id += 1;

        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if self.frame_id == 1 {
            // First frame: full.
            let full = self.codec.encode_full_frame(
                pixels,
                width,
                height,
                self.frame_id,
                timestamp_ms,
                consciousness_level,
                "soma",
            );
            self.push_frame(RdpFrame::Full(full));
        } else {
            // Subsequent: delta.
            let (changed, sims) = self.codec.detect_changes(pixels, width, height);
            if !changed.is_empty() {
                let delta = self.codec.encode_delta_frame(
                    pixels,
                    width,
                    height,
                    &changed,
                    &sims,
                    self.frame_id,
                    self.frame_id - 1,
                    timestamp_ms,
                    consciousness_level,
                );
                self.push_frame(RdpFrame::Delta(delta));
            }
        }
    }

    /// Drain outbound frames for WebSocket delivery.
    pub fn drain_frames(&mut self) -> Vec<RdpFrame> {
        self.outbound_frames.drain(..).collect()
    }

    /// Start screen sharing.
    pub fn start(&mut self) {
        self.active = true;
        self.frame_id = 0;
        self.cycles_since_capture = self.cycles_per_frame; // Capture immediately on next tick
    }

    /// Stop screen sharing.
    pub fn stop(&mut self) {
        self.active = false;
        self.outbound_frames.clear();
    }

    fn push_frame(&mut self, frame: RdpFrame) {
        if self.outbound_frames.len() >= RDP_FRAME_BUFFER_CAP {
            self.outbound_frames.pop_front(); // Drop oldest
        }
        self.outbound_frames.push_back(frame);
    }
}

/// Desktop-side RDP state for viewing a Soma device's screen.
pub struct HolonRdpViewer {
    /// Frame buffer for reconstructing the phone's screen.
    pub frame_buffer: super::rdp_client::FrameBuffer,
    /// Input events to send to the phone.
    outbound_input: Vec<InputEvent>,
    /// Input sequence counter.
    input_seq: u64,
    /// Whether viewing is active.
    pub active: bool,
    /// Frames received counter.
    pub frames_received: u64,
}

impl HolonRdpViewer {
    /// Create a viewer for a Soma device's screen.
    pub fn new(width: u32, height: u32, tile_cols: u16, tile_rows: u16) -> Self {
        Self {
            frame_buffer: super::rdp_client::FrameBuffer::new(width, height, tile_cols, tile_rows),
            outbound_input: Vec::new(),
            input_seq: 0,
            active: false,
            frames_received: 0,
        }
    }

    /// Apply a received RDP frame from the Soma device.
    pub fn apply_frame(&mut self, frame: &RdpFrame) -> bool {
        match frame {
            RdpFrame::Full(f) => {
                self.frame_buffer.apply_full_frame(f);
                self.frames_received += 1;
                true
            }
            RdpFrame::Delta(d) => {
                self.frame_buffer.apply_delta_frame(d);
                self.frames_received += 1;
                true
            }
            _ => false,
        }
    }

    /// Queue an input event to send to the phone.
    pub fn send_input(&mut self, event: InputEvent) {
        self.outbound_input.push(event);
    }

    /// Drain input events for WebSocket delivery to phone.
    pub fn drain_input(&mut self) -> Option<InputFrame> {
        if self.outbound_input.is_empty() {
            return None;
        }
        self.input_seq += 1;
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Some(InputFrame {
            sequence: self.input_seq,
            timestamp_ms,
            events: std::mem::take(&mut self.outbound_input),
        })
    }

    /// Start viewing.
    pub fn start(&mut self) {
        self.active = true;
    }

    /// Stop viewing.
    pub fn stop(&mut self) {
        self.active = false;
        self.outbound_input.clear();
    }
}

/// WebSocket message envelope for RDP over Holon.
///
/// Two parallel paths are supported:
///
/// - **JSON variants** (`Frame`, `Input`, `Control`) — serialized as text via
///   `serde_json` over `Message::Text`. Used by the dev-inspect path and the
///   initial `phone_rdp_share` MVP measurement proxy. These are not encrypted;
///   they are for local debug only and must not be used on a network.
/// - **Binary variants** (`FrameBin`, `InputBin`) — carry a `Vec<u8>` produced
///   by `rdp_wire::seal_frame`/`seal_input`, which applies bincode + ChaCha20-
///   Poly1305 via the session key. These are the real wire path and travel
///   over `Message::Binary`. **Everything on an actual network uses these.**
///
/// The dispatch side of `holon_ws_handler` routes `Message::Binary` to
/// `FrameBin`/`InputBin` and `Message::Text` to the JSON variants (plus the
/// existing `SomaMessage` telemetry path).
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HolonRdpMessage {
    /// RDP frame (screen data) — JSON debug path.
    #[serde(rename = "rdp_frame")]
    Frame {
        /// JSON-serialized RdpFrame.
        data: String,
    },
    /// Input events (touch/keyboard from viewer to phone) — JSON debug path.
    #[serde(rename = "rdp_input")]
    Input {
        /// JSON-serialized InputFrame.
        data: String,
    },
    /// Control: start/stop sharing.
    #[serde(rename = "rdp_control")]
    Control {
        action: String, // "start", "stop", "request_full"
    },
    /// Sealed binary RDP frame — the real wire path. Opaque bytes produced by
    /// `rdp_wire::seal_frame` and consumed by `rdp_wire::open_frame`. This
    /// variant is intended for out-of-band binary WebSocket framing, not JSON
    /// text framing — but `Serialize`/`Deserialize` are derived for symmetry
    /// with the rest of the enum (base64-encoded when forced into JSON).
    #[serde(rename = "rdp_frame_bin")]
    FrameBin {
        /// Sealed bytes: `[nonce (12) | bincode(RdpFrame) + poly1305 tag (16)]`.
        data: Vec<u8>,
    },
    /// Sealed binary input frame (reverse path from viewer to server).
    #[serde(rename = "rdp_input_bin")]
    InputBin {
        /// Sealed bytes: `[nonce (12) | bincode(InputFrame) + poly1305 tag (16)]`.
        data: Vec<u8>,
    },
}

#[cfg(test)]
mod tests {
    use super::super::rdp_codec::TILE_SIZE;
    use super::*;

    fn make_test_pixels(w: u32, h: u32, seed: u8) -> Vec<u8> {
        let n = w as usize * h as usize;
        let mut pixels = vec![0u8; n * 4];
        for i in 0..n {
            pixels[i * 4] = seed.wrapping_add(i as u8);
            pixels[i * 4 + 1] = seed;
            pixels[i * 4 + 2] = 128;
            pixels[i * 4 + 3] = 255;
        }
        pixels
    }

    #[test]
    fn test_soma_server_produces_frames() {
        let mut server = SomaRdpServer::new(256, 256, 5, 20);
        server.start();

        let pixels = make_test_pixels(256, 256, 0);

        // First tick: should produce full frame (cycles_per_frame=4, but we set it to capture immediately).
        server.tick(&pixels, 256, 256, 0.5);
        let frames = server.drain_frames();
        assert!(!frames.is_empty(), "First tick should produce a frame");
        assert!(matches!(frames[0], RdpFrame::Full(_)));
    }

    #[test]
    fn test_soma_server_frame_pacing() {
        let mut server = SomaRdpServer::new(256, 256, 5, 20); // 20Hz cycle, 5fps → every 4 cycles
        server.start();

        let pixels = make_test_pixels(256, 256, 0);

        // Tick 1: capture.
        server.tick(&pixels, 256, 256, 0.5);
        assert!(!server.drain_frames().is_empty());

        // Ticks 2-4: pacing (no capture).
        for _ in 0..3 {
            server.tick(&pixels, 256, 256, 0.5);
        }
        // Frames should still be empty (pacing).
        // Tick 5: next capture.
        server.tick(&pixels, 256, 256, 0.5);
        // May or may not have a frame (delta might be empty if pixels unchanged).
    }

    #[test]
    fn test_holon_viewer_applies_frames() {
        let mut viewer = HolonRdpViewer::new(256, 256, 4, 4);
        viewer.start();

        // Create a synthetic full frame.
        let patches: Vec<super::super::rdp_protocol::QuantizedPatch> = (0..16)
            .map(|i| super::super::rdp_protocol::QuantizedPatch {
                values: vec![(i * 10) as i8; TILE_SIZE * TILE_SIZE],
            })
            .collect();
        let full = RdpFrame::Full(FullFrame {
            frame_id: 1,
            timestamp_ms: 0,
            patch_cols: 4,
            patch_rows: 4,
            patches,
            consciousness_level: 0.5,
            harmony: "soma".into(),
        });

        let applied = viewer.apply_frame(&full);
        assert!(applied);
        assert!(viewer.frame_buffer.has_content);
        assert_eq!(viewer.frames_received, 1);
    }

    #[test]
    fn test_viewer_input_forwarding() {
        let mut viewer = HolonRdpViewer::new(256, 256, 4, 4);
        viewer.start();

        viewer.send_input(InputEvent::Pointer {
            x: 0.5,
            y: 0.5,
            button: 1,
            pressed: true,
        });

        let input_frame = viewer.drain_input().unwrap();
        assert_eq!(input_frame.events.len(), 1);
        assert_eq!(input_frame.sequence, 1);
    }

    #[test]
    fn test_inactive_server_no_frames() {
        let mut server = SomaRdpServer::new(256, 256, 5, 20);
        // Don't call start().

        let pixels = make_test_pixels(256, 256, 0);
        server.tick(&pixels, 256, 256, 0.5);
        assert!(server.drain_frames().is_empty());
    }

    #[test]
    fn test_holon_rdp_message_serialization() {
        let msg = HolonRdpMessage::Control {
            action: "start".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("rdp_control"));
        assert!(json.contains("start"));

        let decoded: HolonRdpMessage = serde_json::from_str(&json).unwrap();
        match decoded {
            HolonRdpMessage::Control { action } => assert_eq!(action, "start"),
            _ => panic!("Expected Control"),
        }
    }
}
