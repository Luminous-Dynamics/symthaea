// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PQC-Secured Remote Display Protocol (RDP) for Symthaea.
//!
//! Enables screen sharing and remote control between Symthaea instances
//! using consciousness-native HDC frame encoding and post-quantum encryption.
//!
//! Architecture:
//! ```text
//! Server: Screen capture → VisionManifold HDC encode → delta compress
//!       → PQC encrypt (ChaCha20-Poly1305 with ML-KEM session key)
//!       → Iroh QUIC stream → Client
//!
//! Client: Receive → decrypt → decompress → reconstruct HDC frame
//!       → render to display
//!       → capture input → encrypt → send to server
//! ```
//!
//! Frame format uses sparse delta encoding: only screen patches that
//! changed (surprise > threshold) are transmitted, typically 20% of
//! patches → ~6KB/frame at 5Hz = 30KB/sec.

use serde::{Deserialize, Serialize};

/// Protocol version identifier.
pub const RDP_PROTOCOL_VERSION: u8 = 1;

/// ALPN for Iroh QUIC negotiation.
pub const RDP_ALPN: &[u8] = b"symthaea/rdp/1";

/// Maximum patches per delta frame.
pub const MAX_DELTA_PATCHES: usize = 1024;

/// A remote display frame: either a full snapshot or a delta update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RdpFrame {
    /// Full frame snapshot (used for initial frame and recovery).
    Full(FullFrame),
    /// Delta update: only changed patches since last acknowledged frame.
    Delta(DeltaFrame),
    /// Input event from remote client.
    Input(InputFrame),
    /// Session control message.
    Control(ControlMessage),
}

/// Complete frame snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullFrame {
    /// Monotonically increasing frame identifier.
    pub frame_id: u64,
    /// Timestamp (milliseconds since session start).
    pub timestamp_ms: u64,
    /// Frame width in patches (not pixels).
    pub patch_cols: u16,
    /// Frame height in patches.
    pub patch_rows: u16,
    /// Quantized HDC vectors for each patch (row-major order).
    /// Each patch is quantized to int8 for compression.
    pub patches: Vec<QuantizedPatch>,
    /// Server's consciousness level at capture time.
    pub consciousness_level: f32,
    /// Dominant harmony name.
    pub harmony: String,
}

/// Delta frame: sparse update of changed patches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaFrame {
    /// Current frame identifier.
    pub frame_id: u64,
    /// Frame this delta is relative to.
    pub base_frame_id: u64,
    /// Timestamp (milliseconds since session start).
    pub timestamp_ms: u64,
    /// Changed patches with their grid positions.
    pub patches: Vec<DeltaPatch>,
    /// Server's consciousness level.
    pub consciousness_level: f32,
}

/// A quantized HDC patch (for full frames).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedPatch {
    /// Quantized HDC values (int8: -128 to 127).
    /// Dimension depends on vision manifold config (typically 64 or 128).
    pub values: Vec<i8>,
}

/// A changed patch in a delta frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaPatch {
    /// Patch grid index (row * patch_cols + col).
    pub index: u16,
    /// Surprise level that triggered this patch's inclusion.
    pub surprise: f32,
    /// Quantized HDC delta values.
    pub values: Vec<i8>,
}

/// Remote input event from client to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFrame {
    /// Sequence number for ordering.
    pub sequence: u64,
    /// Timestamp (client-local, milliseconds).
    pub timestamp_ms: u64,
    /// Batch of input events.
    pub events: Vec<InputEvent>,
}

/// A single input event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputEvent {
    /// Mouse/touch position and button state.
    Pointer {
        /// Normalized X position (0.0-1.0).
        x: f32,
        /// Normalized Y position (0.0-1.0).
        y: f32,
        /// Button state (0=none, 1=primary, 2=secondary, 3=middle).
        button: u8,
        /// true=pressed, false=released.
        pressed: bool,
    },
    /// Keyboard event.
    Key {
        /// Platform-independent key code.
        code: u32,
        /// true=pressed, false=released.
        pressed: bool,
        /// Modifier flags (shift=1, ctrl=2, alt=4, meta=8).
        modifiers: u8,
    },
    /// Touch event (multi-touch).
    Touch {
        /// Touch point index.
        index: u8,
        /// Normalized X (0.0-1.0).
        x: f32,
        /// Normalized Y (0.0-1.0).
        y: f32,
        /// Touch phase (0=begin, 1=move, 2=end, 3=cancel).
        phase: u8,
        /// Touch pressure (0.0-1.0).
        pressure: f32,
    },
}

/// Session control messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Request session parameters.
    Hello {
        protocol_version: u8,
        client_name: String,
    },
    /// Server response with session parameters.
    Welcome {
        protocol_version: u8,
        server_name: String,
        patch_cols: u16,
        patch_rows: u16,
        target_fps: u8,
        consciousness_level: f32,
    },
    /// Request a full frame (after connection loss or desync).
    RequestFullFrame,
    /// Acknowledge receipt of frame (for flow control).
    FrameAck { frame_id: u64 },
    /// Graceful session termination.
    Goodbye { reason: String },
    /// Consciousness attestation (proves server is a genuine Symthaea).
    ConsciousnessAttestation {
        /// Phi level at attestation time.
        phi: f32,
        /// BLAKE3 hash of recent consciousness state.
        state_hash: [u8; 32],
        /// Ed25519 signature over (phi || state_hash).
        signature: Vec<u8>,
    },
    /// Clipboard content update (bidirectional sync).
    ClipboardUpdate {
        /// Clipboard content as UTF-8 text.
        content: String,
        /// Sequence number for ordering.
        sequence: u64,
    },
}

/// Session configuration negotiated during Hello/Welcome.
#[derive(Debug, Clone)]
pub struct RdpSessionConfig {
    /// Target frames per second (5 typical, 30 for interactive).
    pub target_fps: u8,
    /// Surprise threshold for delta patch inclusion (0.0-1.0).
    pub surprise_threshold: f32,
    /// HDC dimension per patch (smaller = faster, less detail).
    pub patch_hdc_dim: usize,
    /// Whether to use LZ4 compression on wire.
    pub use_compression: bool,
    /// Whether PQC encryption is required.
    pub require_pqc: bool,
    /// Minimum consciousness level for session (consciousness-gated access).
    pub min_consciousness: f32,
}

impl Default for RdpSessionConfig {
    fn default() -> Self {
        Self {
            target_fps: 5,
            surprise_threshold: 0.3,
            patch_hdc_dim: 64,
            use_compression: true,
            require_pqc: true,
            min_consciousness: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_serialization_roundtrip() {
        let frame = RdpFrame::Delta(DeltaFrame {
            frame_id: 42,
            base_frame_id: 41,
            timestamp_ms: 1000,
            patches: vec![DeltaPatch {
                index: 5,
                surprise: 0.7,
                values: vec![1, -3, 5, 0, -1],
            }],
            consciousness_level: 0.28,
        });

        let json = serde_json::to_string(&frame).unwrap();
        let decoded: RdpFrame = serde_json::from_str(&json).unwrap();

        match decoded {
            RdpFrame::Delta(d) => {
                assert_eq!(d.frame_id, 42);
                assert_eq!(d.patches.len(), 1);
                assert_eq!(d.patches[0].index, 5);
            }
            _ => panic!("Expected Delta frame"),
        }
    }

    #[test]
    fn test_control_hello_welcome() {
        let hello = ControlMessage::Hello {
            protocol_version: RDP_PROTOCOL_VERSION,
            client_name: "soma-pixel8".into(),
        };
        let json = serde_json::to_string(&hello).unwrap();
        assert!(json.contains("soma-pixel8"));
    }

    #[test]
    fn test_input_event_serialization() {
        let input = InputFrame {
            sequence: 1,
            timestamp_ms: 500,
            events: vec![
                InputEvent::Pointer {
                    x: 0.5,
                    y: 0.3,
                    button: 1,
                    pressed: true,
                },
                InputEvent::Key {
                    code: 65,
                    pressed: true,
                    modifiers: 0,
                },
            ],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: InputFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.events.len(), 2);
    }
}
