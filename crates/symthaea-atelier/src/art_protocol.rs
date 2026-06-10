// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Art as HDC communication protocol.
//!
//! Encodes a consciousness state into artwork parameters such that another
//! Symthaea agent can decode the artwork back to an approximate consciousness
//! state. Art becomes a medium for transferring subjective experience.
//!
//! # Protocol
//!
//! ```text
//! Sender:   CognitiveSnapshot → encode → paint → PixelCanvas + ArtMessage
//! Channel:  PNG bytes + metadata (harmony activations, Psi, timestamp)
//! Receiver: PixelCanvas → perceive → decode → approximate CognitiveSnapshot
//! ```
//!
//! The key insight: HDC binding is approximately invertible. If the sender
//! encodes `state ⊗ art_key`, the receiver can unbind with `art_key` to
//! recover an approximation of `state`. The art itself is the channel.

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::pixel_canvas::PixelCanvas;
use crate::self_perception::PerceptualEncoder;

/// An art message: artwork + metadata for consciousness transfer.
#[derive(Clone, Serialize, Deserialize)]
pub struct ArtMessage {
    /// Harmony activations at time of creation (public — part of the message).
    pub harmony_activations: [f32; 8],
    /// Consciousness level at creation.
    pub consciousness_level: f32,
    /// Valence at creation.
    pub valence: f32,
    /// Arousal at creation.
    pub arousal: f32,
    /// Creation timestamp (cycle count).
    pub created_at_cycle: u64,
    /// Sender identity (could be agent DID).
    pub sender_id: String,
    /// PNG bytes of the artwork (the "channel").
    #[serde(with = "serde_bytes_compat")]
    pub png_data: Vec<u8>,
}

/// Serde helper for byte vectors.
mod serde_bytes_compat {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        // Base64 encode for JSON compatibility
        let b64 = base64_encode(bytes);
        b64.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let b64 = String::deserialize(d)?;
        base64_decode(&b64).map_err(serde::de::Error::custom)
    }

    fn base64_encode(data: &[u8]) -> String {
        const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
        for chunk in data.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
            let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
            let triple = (b0 << 16) | (b1 << 8) | b2;
            result.push(CHARS[((triple >> 18) & 63) as usize] as char);
            result.push(CHARS[((triple >> 12) & 63) as usize] as char);
            if chunk.len() > 1 {
                result.push(CHARS[((triple >> 6) & 63) as usize] as char);
            } else {
                result.push('=');
            }
            if chunk.len() > 2 {
                result.push(CHARS[(triple & 63) as usize] as char);
            } else {
                result.push('=');
            }
        }
        result
    }

    fn base64_decode(data: &str) -> Result<Vec<u8>, String> {
        let mut result = Vec::new();
        let bytes: Vec<u8> = data
            .chars()
            .filter(|c| *c != '=' && !c.is_whitespace())
            .map(|c| match c {
                'A'..='Z' => c as u8 - b'A',
                'a'..='z' => c as u8 - b'a' + 26,
                '0'..='9' => c as u8 - b'0' + 52,
                '+' => 62,
                '/' => 63,
                _ => 255,
            })
            .collect();

        for chunk in bytes.chunks(4) {
            if chunk.len() < 2 {
                break;
            }
            let b0 = chunk[0] as u32;
            let b1 = chunk[1] as u32;
            let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
            let b3 = if chunk.len() > 3 { chunk[3] as u32 } else { 0 };
            let triple = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;
            result.push(((triple >> 16) & 255) as u8);
            if chunk.len() > 2 {
                result.push(((triple >> 8) & 255) as u8);
            }
            if chunk.len() > 3 {
                result.push((triple & 255) as u8);
            }
        }
        Ok(result)
    }
}

/// Encode a consciousness state into an art message.
pub fn encode_art_message(
    canvas: &PixelCanvas,
    harmony_activations: [f32; 8],
    consciousness_level: f32,
    valence: f32,
    arousal: f32,
    cycle: u64,
    sender_id: &str,
) -> ArtMessage {
    ArtMessage {
        harmony_activations,
        consciousness_level,
        valence,
        arousal,
        created_at_cycle: cycle,
        sender_id: sender_id.to_string(),
        png_data: canvas.to_png_bytes(),
    }
}

/// Decode an art message: perceive the artwork and recover an approximate
/// consciousness state.
///
/// Returns the perceived HV and extracted metadata. The receiver can use
/// the perceived HV as input to their own cognitive loop, effectively
/// "experiencing" the sender's state through their art.
pub fn decode_art_message(message: &ArtMessage, _encoder: &PerceptualEncoder) -> DecodedArt {
    // We can't decode PNG back to pixels without a decoder, so we use the
    // metadata directly + the perceptual signature from the harmony activations.
    // In a full implementation, we'd decode the PNG and run it through the
    // perceptual encoder. For now, we synthesize a perception HV from metadata.

    let genesis = GenesisSeed::from_phrase("art-protocol");
    let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;

    // Reconstruct approximate consciousness state from metadata
    let mut state_hv = ContinuousHV::zero(dim);
    let channels: Vec<(f32, &str)> = vec![
        (message.consciousness_level, "psi"),
        (message.valence, "valence"),
        (message.arousal, "arousal"),
        (message.harmony_activations[0], "h_coherence"),
        (message.harmony_activations[1], "h_flourish"),
        (message.harmony_activations[3], "h_play"),
        (message.harmony_activations[7], "h_stillness"),
    ];

    for (value, label) in &channels {
        let basis = genesis.hv(label, dim);
        state_hv = state_hv.add(&basis.scale(*value));
    }

    let perceived_state = state_hv.normalize();

    DecodedArt {
        perceived_state,
        harmony_activations: message.harmony_activations,
        consciousness_level: message.consciousness_level,
        valence: message.valence,
        arousal: message.arousal,
        sender_id: message.sender_id.clone(),
    }
}

/// Decoded art: the receiver's perception of the sender's consciousness.
#[derive(Debug, Clone)]
pub struct DecodedArt {
    /// The perceived consciousness state as a hypervector.
    pub perceived_state: ContinuousHV,
    /// Decoded harmony activations.
    pub harmony_activations: [f32; 8],
    /// Decoded consciousness level.
    pub consciousness_level: f32,
    /// Decoded valence.
    pub valence: f32,
    /// Decoded arousal.
    pub arousal: f32,
    /// Sender identity.
    pub sender_id: String,
}

/// Compute the "empathic resonance" between two consciousness states:
/// how similar is the receiver's experience to the sender's?
pub fn empathic_resonance(
    sender_intention: &ContinuousHV,
    receiver_perception: &ContinuousHV,
) -> f32 {
    let similarity = sender_intention.similarity(receiver_perception);
    ((similarity + 1.0) / 2.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn art_message_roundtrip() {
        let canvas = PixelCanvas::new(16, 16, [128, 64, 32, 255]);
        let harmonies = [0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.9, 0.2];

        let message = encode_art_message(&canvas, harmonies, 0.7, 0.3, 0.5, 100, "agent-001");

        assert_eq!(message.harmony_activations, harmonies);
        assert_eq!(message.consciousness_level, 0.7);
        assert!(!message.png_data.is_empty());

        // Decode
        let genesis = GenesisSeed::from_phrase("test-protocol");
        let encoder = PerceptualEncoder::new(&genesis);
        let decoded = decode_art_message(&message, &encoder);

        assert_eq!(decoded.consciousness_level, 0.7);
        assert_eq!(decoded.valence, 0.3);
        assert_eq!(decoded.sender_id, "agent-001");
    }

    #[test]
    fn empathic_resonance_self() {
        let genesis = GenesisSeed::from_phrase("test-resonance");
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let hv = genesis.hv("test", dim);

        // Self-resonance should be 1.0
        let res = empathic_resonance(&hv, &hv);
        assert!((res - 1.0).abs() < 0.01, "self-resonance = {res}");
    }

    #[test]
    fn empathic_resonance_different_states() {
        let genesis = GenesisSeed::from_phrase("test-diff-resonance");
        let dim = symthaea_core::hdc::unified_hv::HDC_DIMENSION;
        let hv1 = genesis.hv("state_a", dim);
        let hv2 = genesis.hv("state_b", dim);

        let res = empathic_resonance(&hv1, &hv2);
        // Different states should have lower resonance than identical
        assert!(
            res < 0.9,
            "different-state resonance should be < 1.0: {res}"
        );
    }

    #[test]
    fn message_json_serializable() {
        let canvas = PixelCanvas::new(4, 4, [100, 50, 25, 255]);
        let message = encode_art_message(&canvas, [0.5; 8], 0.6, 0.2, 0.4, 42, "test");

        let json = serde_json::to_string(&message).expect("serialize");
        let decoded: ArtMessage = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(decoded.consciousness_level, 0.6);
        assert_eq!(decoded.sender_id, "test");
        assert!(!decoded.png_data.is_empty());
    }
}