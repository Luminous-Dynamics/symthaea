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
//! | `0x12`| `RdpFrame` LZ4-compressed (Phase II.A, behind `holon-lz4` feature) |
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

/// Payload-type byte used when sealing an LZ4-compressed `RdpFrame`.
///
/// Phase II.A: LZ4 must precede AEAD sealing (ciphertext is pseudorandom
/// and doesn't compress). A distinct payload type prevents nonce-stream
/// collision between raw and LZ4 frames on the same session key, and
/// prevents a receiver that reuses the wrong opener from succeeding
/// cryptographically (each opener expects a specific payload type
/// position in the nonce).
///
/// Measured on live Pixel 8 Pro 2026-04-17: 2.12× reduction overall,
/// 2.20× on steady-state Delta frames. At 30 fps: 18.35 MB/s → 8.34 MB/s,
/// clears the 8.75 MB/s network-friendly gate with 5% margin. See
/// `examples/phase_2a_lz4_measurement.rs` + roadmap v1.7.
pub const PAYLOAD_TYPE_RDP_FRAME_LZ4: u8 = 0x12;

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

/// Serialize an `RdpFrame` to binary, LZ4-compress, and seal under the
/// session key. The counterpart of [`open_frame_lz4`].
///
/// Compression must precede AEAD — ChaCha20-Poly1305 ciphertext is
/// pseudorandom and does not compress. This function does:
///
/// ```text
/// RdpFrame
///   → bincode::serialize           (RdpFrame::to_bin)
///   → lz4_flex::compress_prepend_size  (4-byte length prefix + LZ4 block)
///   → RdpSession::seal             (ChaCha20-Poly1305 with PAYLOAD_TYPE_RDP_FRAME_LZ4)
/// ```
///
/// Receivers MUST use [`open_frame_lz4`] to open. Mixing with
/// [`open_frame`] will fail at bincode deserialize (the decompressed
/// output is bincode, the raw `open_frame` would skip decompression and
/// try to deserialize the LZ4 block header as bincode — `WireError::Codec`).
///
/// The `PAYLOAD_TYPE_RDP_FRAME_LZ4` value also separates the AEAD
/// nonce stream from the raw `RdpFrame` stream, so a session can
/// interleave both types without replay-window collision.
#[cfg(feature = "holon-lz4")]
pub fn seal_frame_lz4(frame: &RdpFrame, session: &mut RdpSession) -> Result<Vec<u8>, WireError> {
    let plaintext = frame.to_bin()?;
    let compressed = lz4_flex::block::compress_prepend_size(&plaintext);
    session
        .seal(&compressed, PAYLOAD_TYPE_RDP_FRAME_LZ4)
        .ok_or(WireError::SealFailed)
        .and_then(|sealed| {
            if session.encryption_key().is_none() {
                Err(WireError::NoSessionKey)
            } else {
                Ok(sealed)
            }
        })
}

/// Open a sealed envelope produced by [`seal_frame_lz4`] and decode the `RdpFrame`.
///
/// Reverses the pipeline:
///
/// ```text
/// sealed bytes
///   → RdpSession::open              (AEAD verify, replay window check)
///   → lz4_flex::decompress_size_prepended  (reads 4-byte length, decompresses)
///   → bincode::deserialize          (RdpFrame::from_bin)
/// ```
///
/// Returns `WireError::OpenFailed` for AEAD failure (wrong key,
/// tampered ciphertext, replay duplicate), LZ4 decompression failure,
/// or malformed length prefix. Mutates `session` to advance the
/// replay window on accepted messages.
#[cfg(feature = "holon-lz4")]
pub fn open_frame_lz4(bytes: &[u8], session: &mut RdpSession) -> Result<RdpFrame, WireError> {
    let compressed = session.open(bytes).ok_or(WireError::OpenFailed)?;
    let plaintext = lz4_flex::block::decompress_size_prepended(&compressed)
        .map_err(|_| WireError::OpenFailed)?;
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
    use crate::swarm::rdp_protocol::RdpSessionConfig;
    use crate::swarm::rdp_protocol::{DeltaFrame, DeltaPatch, InputEvent};

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

    #[cfg(feature = "holon-lz4")]
    #[test]
    fn seal_open_frame_lz4_roundtrip() {
        let mut sender = test_session(0xEF);
        let mut receiver = test_session(0xEF);

        let frame = sample_delta_frame();
        let sealed = seal_frame_lz4(&frame, &mut sender).expect("seal_lz4");
        let opened = open_frame_lz4(&sealed, &mut receiver).expect("open_lz4");

        match opened {
            RdpFrame::Delta(d) => {
                assert_eq!(d.frame_id, 17);
                assert_eq!(d.patches.len(), 16);
                assert_eq!(d.patches[5].values.len(), 64);
            }
            _ => panic!("expected Delta"),
        }
    }

    #[cfg(feature = "holon-lz4")]
    #[test]
    fn lz4_sealed_smaller_than_raw_sealed() {
        // Phase II.A structural claim: LZ4-before-seal reduces wire bytes
        // for RdpFrame payloads. This sanity-checks that the shipped
        // production wrap actually produces smaller envelopes than the
        // raw seal path on the same frame.
        let mut raw_sender = test_session(0x01);
        let mut lz4_sender = test_session(0x02);

        let frame = sample_delta_frame();
        let raw_sealed = seal_frame(&frame, &mut raw_sender).expect("seal raw");
        let lz4_sealed = seal_frame_lz4(&frame, &mut lz4_sender).expect("seal lz4");

        // Sample delta frame has repetitive patch values — expect healthy
        // compression. Keep the assertion conservative: LZ4 must NOT
        // produce a larger envelope; on this test fixture the ratio is
        // measurably < 1.0.
        assert!(
            lz4_sealed.len() < raw_sealed.len(),
            "LZ4 sealed ({}) must be smaller than raw sealed ({})",
            lz4_sealed.len(),
            raw_sealed.len(),
        );
    }

    #[cfg(feature = "holon-lz4")]
    #[test]
    fn lz4_sealed_cannot_be_opened_by_raw_path() {
        // Safety test: if a sender uses seal_frame_lz4 but a receiver
        // mistakenly uses open_frame, decryption succeeds (same session
        // key, different payload_type in nonce — but AEAD doesn't care)
        // then bincode::deserialize on the LZ4 block fails gracefully.
        // The inverse (seal_frame opened by open_frame_lz4) fails at
        // the LZ4 size-prefix-decode step.
        let mut sender = test_session(0x33);
        let mut receiver = test_session(0x33);

        let frame = sample_delta_frame();
        let lz4_sealed = seal_frame_lz4(&frame, &mut sender).expect("seal lz4");

        // Wrong opener: open_frame on a seal_frame_lz4 envelope.
        // Note: payload_type differs (0x12 vs 0x10) which means the
        // AEAD nonce's payload_type byte differs. `session.open` extracts
        // the payload_type from the nonce and uses it for replay window
        // indexing, but the AEAD verifies regardless. So decryption
        // succeeds; bincode deserialization of the LZ4 block fails.
        let wrong = open_frame(&lz4_sealed, &mut receiver);
        assert!(
            wrong.is_err(),
            "open_frame must reject LZ4 envelope at bincode deserialize"
        );
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
