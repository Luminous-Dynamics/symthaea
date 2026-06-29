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

use bincode::Options;
use serde::{Deserialize, Serialize};

/// Protocol version identifier.
pub const RDP_PROTOCOL_VERSION: u8 = 2;

/// ALPN for Iroh QUIC negotiation.
pub const RDP_ALPN: &[u8] = b"symthaea/rdp/2";

/// Maximum patches per delta frame.
pub const MAX_DELTA_PATCHES: usize = 1024;

/// Maximum patches in a full frame snapshot.
pub const MAX_FULL_FRAME_PATCHES: usize = 16_384;

/// Maximum quantized values per visual patch.
pub const MAX_PATCH_VALUES: usize = 4_096;

/// Maximum input events in one reverse-path frame.
pub const MAX_INPUT_EVENTS: usize = 256;

/// Maximum audio payload per frame (16 KB).
pub const RDP_AUDIO_MAX_CHUNK_BYTES: usize = 16384;

/// Maximum encoded RdpFrame plaintext before AEAD sealing.
pub const RDP_MAX_FRAME_BIN_BYTES: u64 = 4 * 1024 * 1024;

/// Maximum encoded InputFrame plaintext before AEAD sealing.
pub const RDP_MAX_INPUT_BIN_BYTES: u64 = 64 * 1024;

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
    /// Audio frame: streaming consciousness-encoded music.
    Audio(AudioFrame),
}

/// A compressed audio chunk with consciousness metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFrame {
    pub audio_seq: u64,
    pub timestamp_ms: u64,
    pub sample_rate: u32,
    pub num_samples: u16,
    pub data: Vec<u8>,
    pub encoding: AudioEncoding,
    pub consciousness_level: f32,
    pub allostatic_load: f32,
    pub dominant_harmony: u8,
}

/// Audio encoding format for RDP transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEncoding {
    /// Raw PCM f32 stereo (~352 kbps at 44.1kHz).
    RawF32Stereo,
    /// Raw PCM i16 stereo (~176 kbps).
    RawI16Stereo,
    /// Mu-law compressed mono (~88 kbps).
    MuLawMono,
}

impl AudioEncoding {
    /// Select encoding based on peer consciousness level.
    pub fn from_phi(phi: f32) -> Self {
        if phi > 0.7 {
            Self::RawF32Stereo
        } else if phi > 0.4 {
            Self::RawI16Stereo
        } else {
            Self::MuLawMono
        }
    }
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
    /// Clipboard content shared between peers.
    ClipboardUpdate {
        format: super::rdp_protocol_ext::ClipboardFormat,
        data: Vec<u8>,
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

// ═══════════════════════════════════════════════════════════════════════════════
// Binary encoding (bincode) — compact wire format for PQC-sealed transport
// ═══════════════════════════════════════════════════════════════════════════════

/// Error type for binary frame codec operations.
#[derive(Debug)]
pub enum FrameCodecError {
    /// bincode serialization failure (should not happen with valid input).
    Encode(bincode::Error),
    /// bincode deserialization failure (malformed input).
    Decode(bincode::Error),
    /// Binary payload exceeded the codec limit before decode.
    FrameTooLarge { len: usize, max: u64 },
    /// Decoded frame failed structural validation.
    InvalidFrame(String),
}

impl std::fmt::Display for FrameCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(e) => write!(f, "RdpFrame encode failed: {e}"),
            Self::Decode(e) => write!(f, "RdpFrame decode failed: {e}"),
            Self::FrameTooLarge { len, max } => {
                write!(
                    f,
                    "RdpFrame decode rejected oversized payload: {len} > {max}"
                )
            }
            Self::InvalidFrame(reason) => write!(f, "RdpFrame validation failed: {reason}"),
        }
    }
}

impl std::error::Error for FrameCodecError {}

impl RdpFrame {
    fn validate(&self) -> Result<(), FrameCodecError> {
        match self {
            Self::Full(frame) => {
                if frame.patches.len() > MAX_FULL_FRAME_PATCHES {
                    return Err(FrameCodecError::InvalidFrame(format!(
                        "full frame has {} patches, max {MAX_FULL_FRAME_PATCHES}",
                        frame.patches.len()
                    )));
                }
                let grid_cells = frame.patch_cols as usize * frame.patch_rows as usize;
                if frame.patches.len() > grid_cells {
                    return Err(FrameCodecError::InvalidFrame(format!(
                        "full frame has {} patches for {grid_cells} grid cells",
                        frame.patches.len()
                    )));
                }
                if let Some((idx, patch)) = frame
                    .patches
                    .iter()
                    .enumerate()
                    .find(|(_, patch)| patch.values.len() > MAX_PATCH_VALUES)
                {
                    return Err(FrameCodecError::InvalidFrame(format!(
                        "full frame patch {idx} has {} values, max {MAX_PATCH_VALUES}",
                        patch.values.len()
                    )));
                }
                if frame.harmony.len() > 256 {
                    return Err(FrameCodecError::InvalidFrame(
                        "harmony label exceeds 256 bytes".to_string(),
                    ));
                }
            }
            Self::Delta(frame) => {
                if frame.patches.len() > MAX_DELTA_PATCHES {
                    return Err(FrameCodecError::InvalidFrame(format!(
                        "delta frame has {} patches, max {MAX_DELTA_PATCHES}",
                        frame.patches.len()
                    )));
                }
                if let Some((idx, patch)) = frame
                    .patches
                    .iter()
                    .enumerate()
                    .find(|(_, patch)| patch.values.len() > MAX_PATCH_VALUES)
                {
                    return Err(FrameCodecError::InvalidFrame(format!(
                        "delta patch {idx} has {} values, max {MAX_PATCH_VALUES}",
                        patch.values.len()
                    )));
                }
            }
            Self::Input(input) => input.validate()?,
            Self::Control(control) => control.validate()?,
            Self::Audio(audio) => {
                if audio.data.len() > RDP_AUDIO_MAX_CHUNK_BYTES {
                    return Err(FrameCodecError::InvalidFrame(format!(
                        "audio frame has {} bytes, max {RDP_AUDIO_MAX_CHUNK_BYTES}",
                        audio.data.len()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Serialize this frame to a compact binary envelope using bincode.
    ///
    /// The binary path is ~3-5× smaller than `serde_json::to_vec(&self)`
    /// and is the wire format consumed by `rdp_wire::seal_frame()` before
    /// ChaCha20-Poly1305 sealing. Tests + dev-inspect paths still use JSON.
    pub fn to_bin(&self) -> Result<Vec<u8>, FrameCodecError> {
        self.validate()?;
        bincode::options()
            .with_limit(RDP_MAX_FRAME_BIN_BYTES)
            .serialize(self)
            .map_err(FrameCodecError::Encode)
    }

    /// Deserialize a binary envelope back into an `RdpFrame`.
    ///
    /// Fails on truncation, byte corruption, or version skew between peers.
    pub fn from_bin(bytes: &[u8]) -> Result<Self, FrameCodecError> {
        if bytes.len() as u64 > RDP_MAX_FRAME_BIN_BYTES {
            return Err(FrameCodecError::FrameTooLarge {
                len: bytes.len(),
                max: RDP_MAX_FRAME_BIN_BYTES,
            });
        }
        let frame: Self = bincode::options()
            .with_limit(RDP_MAX_FRAME_BIN_BYTES)
            .deserialize(bytes)
            .map_err(FrameCodecError::Decode)?;
        frame.validate()?;
        Ok(frame)
    }
}

impl InputFrame {
    fn validate(&self) -> Result<(), FrameCodecError> {
        if self.events.len() > MAX_INPUT_EVENTS {
            return Err(FrameCodecError::InvalidFrame(format!(
                "input frame has {} events, max {MAX_INPUT_EVENTS}",
                self.events.len()
            )));
        }
        Ok(())
    }

    /// Serialize an InputFrame (viewer → server reverse path) to binary.
    pub fn to_bin(&self) -> Result<Vec<u8>, FrameCodecError> {
        self.validate()?;
        bincode::options()
            .with_limit(RDP_MAX_INPUT_BIN_BYTES)
            .serialize(self)
            .map_err(FrameCodecError::Encode)
    }

    /// Deserialize an InputFrame from a binary envelope.
    pub fn from_bin(bytes: &[u8]) -> Result<Self, FrameCodecError> {
        if bytes.len() as u64 > RDP_MAX_INPUT_BIN_BYTES {
            return Err(FrameCodecError::FrameTooLarge {
                len: bytes.len(),
                max: RDP_MAX_INPUT_BIN_BYTES,
            });
        }
        let input: Self = bincode::options()
            .with_limit(RDP_MAX_INPUT_BIN_BYTES)
            .deserialize(bytes)
            .map_err(FrameCodecError::Decode)?;
        input.validate()?;
        Ok(input)
    }
}

impl ControlMessage {
    fn validate(&self) -> Result<(), FrameCodecError> {
        match self {
            Self::Hello { client_name, .. } if client_name.len() > 256 => Err(
                FrameCodecError::InvalidFrame("client_name exceeds 256 bytes".to_string()),
            ),
            Self::Welcome { server_name, .. } if server_name.len() > 256 => Err(
                FrameCodecError::InvalidFrame("server_name exceeds 256 bytes".to_string()),
            ),
            Self::Goodbye { reason } if reason.len() > 1024 => Err(FrameCodecError::InvalidFrame(
                "goodbye reason exceeds 1024 bytes".to_string(),
            )),
            Self::ConsciousnessAttestation { signature, .. } if signature.len() > 256 => {
                Err(FrameCodecError::InvalidFrame(
                    "attestation signature exceeds 256 bytes".to_string(),
                ))
            }
            Self::ClipboardUpdate { data, .. } if data.len() > 1024 * 1024 => Err(
                FrameCodecError::InvalidFrame("clipboard payload exceeds 1MiB".to_string()),
            ),
            _ => Ok(()),
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
    fn test_frame_binary_roundtrip_and_size() {
        // Realistic delta frame: 32 patches of 64 i8 values each.
        let patches: Vec<DeltaPatch> = (0..32)
            .map(|i| DeltaPatch {
                index: i as u16,
                surprise: (i as f32) / 32.0,
                values: (0..64).map(|j| ((i + j) as i8).wrapping_mul(3)).collect(),
            })
            .collect();
        let frame = RdpFrame::Delta(DeltaFrame {
            frame_id: 42,
            base_frame_id: 41,
            timestamp_ms: 1_700_000_000_000,
            patches,
            consciousness_level: 0.65,
        });

        let bin = frame.to_bin().expect("bincode encode");
        let json = serde_json::to_vec(&frame).expect("json encode");

        // Binary must beat JSON by a meaningful margin. Observed ratio on
        // dense i8 patch arrays is ~3.0× (see `integration_rdp_wire` for the
        // sealed-envelope measurement with AEAD overhead included).
        let ratio = json.len() as f64 / bin.len().max(1) as f64;
        assert!(
            ratio >= 2.5,
            "bincode should be ≥2.5× smaller than JSON: bin={} json={} ratio={:.3}",
            bin.len(),
            json.len(),
            ratio,
        );

        // Round-trip equality on the decoded frame.
        let decoded = RdpFrame::from_bin(&bin).expect("bincode decode");
        match decoded {
            RdpFrame::Delta(d) => {
                assert_eq!(d.frame_id, 42);
                assert_eq!(d.base_frame_id, 41);
                assert_eq!(d.patches.len(), 32);
                assert_eq!(d.patches[5].index, 5);
                assert_eq!(d.patches[5].values.len(), 64);
                assert!((d.consciousness_level - 0.65).abs() < 1e-6);
            }
            _ => panic!("Expected Delta frame after binary round-trip"),
        }
    }

    #[test]
    fn test_input_frame_binary_roundtrip() {
        let input = InputFrame {
            sequence: 7,
            timestamp_ms: 1_700_000_000_500,
            events: vec![
                InputEvent::Pointer {
                    x: 0.5,
                    y: 0.5,
                    button: 1,
                    pressed: true,
                },
                InputEvent::Touch {
                    index: 0,
                    x: 0.3,
                    y: 0.7,
                    phase: 0,
                    pressure: 0.8,
                },
            ],
        };
        let bin = input.to_bin().expect("bincode encode");
        let decoded = InputFrame::from_bin(&bin).expect("bincode decode");
        assert_eq!(decoded.sequence, 7);
        assert_eq!(decoded.events.len(), 2);
    }

    #[test]
    fn test_from_bin_rejects_garbage() {
        let bad = [0xFFu8; 8];
        assert!(RdpFrame::from_bin(&bad).is_err());
    }

    #[test]
    fn oversized_binary_frame_is_rejected_before_decode() {
        let bytes = vec![0u8; RDP_MAX_FRAME_BIN_BYTES as usize + 1];
        let err = RdpFrame::from_bin(&bytes).unwrap_err();
        assert!(matches!(err, FrameCodecError::FrameTooLarge { .. }));
    }

    #[test]
    fn oversized_delta_patch_is_rejected() {
        let frame = RdpFrame::Delta(DeltaFrame {
            frame_id: 1,
            base_frame_id: 0,
            timestamp_ms: 1,
            patches: vec![DeltaPatch {
                index: 0,
                surprise: 1.0,
                values: vec![0; MAX_PATCH_VALUES + 1],
            }],
            consciousness_level: 0.5,
        });

        let err = frame.to_bin().unwrap_err();
        assert!(matches!(err, FrameCodecError::InvalidFrame(_)));
    }

    #[test]
    fn oversized_input_batch_is_rejected() {
        let input = InputFrame {
            sequence: 1,
            timestamp_ms: 1,
            events: (0..=MAX_INPUT_EVENTS)
                .map(|_| InputEvent::Pointer {
                    x: 0.5,
                    y: 0.5,
                    button: 0,
                    pressed: false,
                })
                .collect(),
        };

        let err = input.to_bin().unwrap_err();
        assert!(matches!(err, FrameCodecError::InvalidFrame(_)));
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
