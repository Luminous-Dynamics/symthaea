// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binary wire envelope for Soma↔Holon RDP frames.
//!
//! Composition:
//!
//! ```text
//! RdpFrame (serde)
//!   → bincode::serialize        (RdpFrame::to_bin)
//!   → RdpSession::seal          (ChaCha20-Poly1305, per-session epoch + type-separated nonce)
//!   → WebSocket Binary message  (push through holon_ws_handler)
//!     ─────────────────────────────────────────────────────────────
//!   → WebSocket Binary message  (received by peer)
//!   → RdpSession::open          (AEAD verify, key rotation grace fallback)
//!   → bincode::deserialize      (RdpFrame::from_bin)
//!   → RdpFrame                  (consumed by HolonRdpViewer)
//! ```
//!
//! This module is a thin composer: serialization via `bincode`, and AEAD
//! sealing via `RdpSession::seal`/`open` which call ChaCha20-Poly1305
//! directly (inlined in `rdp_session.rs`). The nonce layout mirrors
//! `swarm::mesh::packet_crypto::build_nonce` for forward-compatibility with
//! the mesh AEAD primitives should the mesh module path be re-wired later.
//!
//! ## Payload type assignments
//!
//! The `packet_crypto::build_nonce` layout includes a `payload_type` byte that
//! prevents cross-stream nonce collisions even when the same session key is
//! reused across message types. These are the RDP-layer assignments:
//!
//! | Value | Meaning                               |
//! |-------|---------------------------------------|
//! | `0x10`| `RdpFrame` (Full / Delta / Control / Audio) |
//! | `0x11`| `InputFrame` (viewer → server reverse path) |
//!
//! Values `0x00..=0x0F` are reserved for mesh wisdom/heartbeat/affective/gradient
//! streams (see `packet_crypto::build_nonce` docs). Do not reuse them here.

#![cfg(feature = "mesh-encryption")]

use super::rdp_protocol::{FrameCodecError, InputFrame, RdpFrame};
use super::rdp_session::RdpSession;

/// Payload-type byte used when sealing an `RdpFrame` over the wire.
///
/// See module-level docs for the full assignment table.
pub const PAYLOAD_TYPE_RDP_FRAME: u8 = 0x10;

/// Payload-type byte used when sealing an `InputFrame` (reverse path).
pub const PAYLOAD_TYPE_RDP_INPUT: u8 = 0x11;

/// Errors returned by the wire codec.
#[derive(Debug)]
pub enum WireError {
    /// bincode encode/decode failure.
    Codec(FrameCodecError),
    /// Session has no established key (handshake not complete or session closed).
    NoSessionKey,
    /// AEAD seal failed (invalid key or AEAD implementation error — should not
    /// happen in practice).
    SealFailed,
    /// AEAD open failed (wrong key, tampered ciphertext, or truncated envelope).
    OpenFailed,
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Codec(e) => write!(f, "rdp_wire codec: {e}"),
            Self::NoSessionKey => write!(f, "rdp_wire: session key not established"),
            Self::SealFailed => write!(f, "rdp_wire: AEAD seal failed"),
            Self::OpenFailed => write!(f, "rdp_wire: AEAD open failed"),
        }
    }
}

impl std::error::Error for WireError {}

impl From<FrameCodecError> for WireError {
    fn from(e: FrameCodecError) -> Self {
        Self::Codec(e)
    }
}

/// Serialize an `RdpFrame` to binary and seal it under the session key.
///
/// Produces a byte string suitable for sending as a WebSocket `Message::Binary`.
/// The recipient decodes it with [`open_frame`].
pub fn seal_frame(frame: &RdpFrame, session: &mut RdpSession) -> Result<Vec<u8>, WireError> {
    let plaintext = frame.to_bin()?;
    session
        .seal(&plaintext, PAYLOAD_TYPE_RDP_FRAME)
        .ok_or(WireError::SealFailed)
        .and_then(|sealed| {
            if session.encryption_key().is_none() {
                Err(WireError::NoSessionKey)
            } else {
                Ok(sealed)
            }
        })
}

/// Open a sealed envelope produced by [`seal_frame`] and decode the `RdpFrame`.
///
/// Mutates `session` to advance the replay window on accepted messages.
/// Returns `Err(WireError::OpenFailed)` for AEAD failure, replay duplicate,
/// or out-of-window sequence.
pub fn open_frame(bytes: &[u8], session: &mut RdpSession) -> Result<RdpFrame, WireError> {
    let plaintext = session.open(bytes).ok_or(WireError::OpenFailed)?;
    let frame = RdpFrame::from_bin(&plaintext)?;
    Ok(frame)
}

/// Serialize an `InputFrame` (reverse path) to binary and seal it.
pub fn seal_input(input: &InputFrame, session: &mut RdpSession) -> Result<Vec<u8>, WireError> {
    let plaintext = input.to_bin()?;
    session
        .seal(&plaintext, PAYLOAD_TYPE_RDP_INPUT)
        .ok_or(WireError::SealFailed)
}

/// Open a sealed input envelope produced by [`seal_input`].
///
/// Mutates `session` to advance the replay window. See [`open_frame`] for
/// the full set of failure conditions.
pub fn open_input(bytes: &[u8], session: &mut RdpSession) -> Result<InputFrame, WireError> {
    let plaintext = session.open(bytes).ok_or(WireError::OpenFailed)?;
    let input = InputFrame::from_bin(&plaintext)?;
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::rdp_protocol::{DeltaFrame, DeltaPatch, InputEvent};
    use crate::swarm::rdp_protocol::RdpSessionConfig;

    /// Build a session with a deterministic key already installed (handshake shortcut
    /// for unit tests — production uses PQC handshake via `on_handshake_complete`).
    fn test_session(key_byte: u8) -> RdpSession {
        let cfg = RdpSessionConfig::default();
        let mut s = RdpSession::new("test-session".into(), "peer".into(), cfg, true);
        s.on_connected();
        s.on_handshake_complete([key_byte; 32]);
        s
    }

    fn sample_delta_frame() -> RdpFrame {
        RdpFrame::Delta(DeltaFrame {
            frame_id: 17,
            base_frame_id: 16,
            timestamp_ms: 1_700_000_000_000,
            patches: (0..16)
                .map(|i| DeltaPatch {
                    index: i as u16,
                    surprise: (i as f32) / 16.0,
                    values: (0..64).map(|j| ((i * 3 + j) as i8)).collect(),
                })
                .collect(),
            consciousness_level: 0.65,
        })
    }

    fn sample_input_frame() -> InputFrame {
        InputFrame {
            sequence: 3,
            timestamp_ms: 1_700_000_000_050,
            events: vec![InputEvent::Pointer {
                x: 0.42,
                y: 0.58,
                button: 1,
                pressed: true,
            }],
        }
    }

    #[test]
    fn seal_open_frame_roundtrip() {
        let mut sender = test_session(0xAB);
        let mut receiver = test_session(0xAB);
        // Sender and receiver share the same key but have independent
        // source_id/epoch — receiver's replay window will accept any first
        // sequence from the sender's stream.

        let frame = sample_delta_frame();
        let sealed = seal_frame(&frame, &mut sender).expect("seal");
        let opened = open_frame(&sealed, &mut receiver).expect("open");

        match opened {
            RdpFrame::Delta(d) => {
                assert_eq!(d.frame_id, 17);
                assert_eq!(d.patches.len(), 16);
                assert_eq!(d.patches[5].values.len(), 64);
            }
            _ => panic!("expected Delta"),
        }
    }

    #[test]
    fn seal_open_input_roundtrip() {
        let mut sender = test_session(0xCD);
        let mut receiver = test_session(0xCD);

        let input = sample_input_frame();
        let sealed = seal_input(&input, &mut sender).expect("seal");
        let opened = open_input(&sealed, &mut receiver).expect("open");
        assert_eq!(opened.sequence, 3);
        assert_eq!(opened.events.len(), 1);
    }

    #[test]
    fn wrong_key_fails_to_open() {
        let mut sender = test_session(0x11);
        let mut wrong = test_session(0x22);

        let frame = sample_delta_frame();
        let sealed = seal_frame(&frame, &mut sender).expect("seal");
        let opened = open_frame(&sealed, &mut wrong);
        assert!(opened.is_err(), "open with wrong key must fail");
    }

    #[test]
    fn truncated_ciphertext_fails_gracefully() {
        let mut sender = test_session(0x77);
        let mut receiver = test_session(0x77);

        let frame = sample_delta_frame();
        let mut sealed = seal_frame(&frame, &mut sender).expect("seal");
        sealed.truncate(5); // Below nonce + tag size.
        let opened = open_frame(&sealed, &mut receiver);
        assert!(opened.is_err());
    }

    #[test]
    fn replay_attack_rejected() {
        // Phase I.A.5 Track 2.2 acceptance test: same envelope opened twice
        // succeeds the first time, fails the second time with the replay
        // window primitive intercepting at step 3 of open().
        let mut sender = test_session(0x88);
        let mut receiver = test_session(0x88);

        let frame = sample_delta_frame();
        let sealed = seal_frame(&frame, &mut sender).expect("seal");

        // First open succeeds.
        let first = open_frame(&sealed, &mut receiver);
        assert!(first.is_ok(), "first open should succeed");

        // Replay (same envelope, same nonce, same sequence) must fail.
        let replayed = open_frame(&sealed, &mut receiver);
        assert!(
            replayed.is_err(),
            "replayed envelope must be rejected by the sliding window"
        );
    }

    #[test]
    fn bandwidth_beats_json_materially() {
        // bincode-sealed frames must be materially smaller than JSON-sealed
        // frames. On delta payloads dominated by dense i8 patch arrays the
        // real ratio lands around 3.0× (JSON encodes each i8 value as ~4 ASCII
        // bytes, bincode as 1 byte). 2.5× is the conservative floor.
        let frame = sample_delta_frame();
        let bin = frame.to_bin().expect("bin encode");
        let json = serde_json::to_vec(&frame).expect("json encode");
        let ratio = json.len() as f64 / bin.len().max(1) as f64;
        assert!(
            ratio >= 2.5,
            "expected bincode ≥2.5× smaller than JSON: bin={} json={} ratio={:.3}",
            bin.len(),
            json.len(),
            ratio,
        );
    }

    #[test]
    fn seal_fails_without_key() {
        // Construct a session that never completed the handshake.
        let cfg = RdpSessionConfig::default();
        let mut s = RdpSession::new("no-key".into(), "peer".into(), cfg, true);
        let frame = sample_delta_frame();
        assert!(matches!(
            seal_frame(&frame, &mut s),
            Err(WireError::SealFailed) | Err(WireError::NoSessionKey)
        ));
    }
}
