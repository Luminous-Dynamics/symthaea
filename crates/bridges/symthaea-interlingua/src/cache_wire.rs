// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical binary wire frame for SCIP semantic-cache feedback.
//!
//! JSON remains useful for diagnostics, but transport/transcript binding should
//! not depend on object-key ordering or serializer behavior. This module defines
//! a fixed-length v1 frame that independent implementations can reproduce.

use crate::{
    InterchangeError, SemanticCacheAck, SemanticCacheFeedback, SemanticCacheMiss,
    SemanticCacheMissKind, SemanticCacheRevoke,
};

/// ASCII frame-family marker. The version is carried in the following byte.
pub const CACHE_FEEDBACK_WIRE_MAGIC: [u8; 3] = *b"SCF";
/// Current cache-feedback binary frame version.
pub const CACHE_FEEDBACK_WIRE_VERSION: u8 = 1;
/// `SCF` + version + kind + requirement + raw 32-byte content digest.
pub const CACHE_FEEDBACK_WIRE_LEN: usize = 38;

const KIND_ACK: u8 = 1;
const KIND_MISS: u8 = 2;
const KIND_REVOKE: u8 = 3;

const REQUIREMENT_NONE: u8 = 0;
const REQUIREMENT_SEMANTIC_REFERENCE_TARGET: u8 = 1;
const REQUIREMENT_GRAPH_DELTA_BASE: u8 = 2;

impl SemanticCacheFeedback {
    /// Encode one cache-feedback event into the canonical fixed-length v1 frame.
    ///
    /// The semantic hash is carried as its raw 32-byte digest rather than ASCII
    /// hex. This removes textual case ambiguity and cuts the content-address
    /// portion of the frame in half. Legacy uppercase hash strings therefore
    /// normalize to the same binary frame as their lowercase form.
    pub fn wire_bytes(&self) -> Result<[u8; CACHE_FEEDBACK_WIRE_LEN], InterchangeError> {
        self.validate()?;
        let (kind, requirement, semantic_hash) = match self {
            Self::Ack(ack) => (KIND_ACK, REQUIREMENT_NONE, ack.semantic_hash.as_str()),
            Self::Miss(miss) => (
                KIND_MISS,
                match miss.requirement {
                    SemanticCacheMissKind::SemanticReferenceTarget => {
                        REQUIREMENT_SEMANTIC_REFERENCE_TARGET
                    }
                    SemanticCacheMissKind::GraphDeltaBase => REQUIREMENT_GRAPH_DELTA_BASE,
                },
                miss.semantic_hash.as_str(),
            ),
            Self::Revoke(revoke) => (KIND_REVOKE, REQUIREMENT_NONE, revoke.semantic_hash.as_str()),
        };
        let digest = decode_semantic_hash(semantic_hash)?;

        let mut frame = [0_u8; CACHE_FEEDBACK_WIRE_LEN];
        frame[0..3].copy_from_slice(&CACHE_FEEDBACK_WIRE_MAGIC);
        frame[3] = CACHE_FEEDBACK_WIRE_VERSION;
        frame[4] = kind;
        frame[5] = requirement;
        frame[6..].copy_from_slice(&digest);
        Ok(frame)
    }

    /// Decode and validate one canonical v1 cache-feedback frame.
    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, InterchangeError> {
        if bytes.len() != CACHE_FEEDBACK_WIRE_LEN {
            return Err(wire_error(format!(
                "frame length {} does not equal {CACHE_FEEDBACK_WIRE_LEN}",
                bytes.len()
            )));
        }
        if bytes[0..3] != CACHE_FEEDBACK_WIRE_MAGIC {
            return Err(wire_error("invalid magic"));
        }
        if bytes[3] != CACHE_FEEDBACK_WIRE_VERSION {
            return Err(wire_error(format!("unsupported version {}", bytes[3])));
        }

        let semantic_hash = encode_semantic_hash(&bytes[6..]);
        let feedback = match (bytes[4], bytes[5]) {
            (KIND_ACK, REQUIREMENT_NONE) => Self::Ack(SemanticCacheAck::new(semantic_hash)?),
            (KIND_REVOKE, REQUIREMENT_NONE) => {
                Self::Revoke(SemanticCacheRevoke::new(semantic_hash)?)
            }
            (KIND_MISS, REQUIREMENT_SEMANTIC_REFERENCE_TARGET) => {
                Self::Miss(SemanticCacheMiss::new(
                    semantic_hash,
                    SemanticCacheMissKind::SemanticReferenceTarget,
                )?)
            }
            (KIND_MISS, REQUIREMENT_GRAPH_DELTA_BASE) => Self::Miss(SemanticCacheMiss::new(
                semantic_hash,
                SemanticCacheMissKind::GraphDeltaBase,
            )?),
            (KIND_ACK | KIND_REVOKE, requirement) => {
                return Err(wire_error(format!(
                    "kind {} requires requirement byte {REQUIREMENT_NONE}, got {requirement}",
                    bytes[4]
                )));
            }
            (KIND_MISS, requirement) => {
                return Err(wire_error(format!(
                    "invalid miss requirement byte {requirement}"
                )));
            }
            (kind, _) => return Err(wire_error(format!("unknown feedback kind {kind}"))),
        };
        feedback.validate()?;
        Ok(feedback)
    }
}

fn decode_semantic_hash(value: &str) -> Result<[u8; 32], InterchangeError> {
    if value.len() != 64 {
        return Err(wire_error(
            "semantic hash must be exactly 64 hexadecimal characters",
        ));
    }
    let source = value.as_bytes();
    let mut digest = [0_u8; 32];
    for (index, output) in digest.iter_mut().enumerate() {
        let high = hex_nibble(source[index * 2])
            .ok_or_else(|| wire_error("semantic hash contains a non-hexadecimal character"))?;
        let low = hex_nibble(source[index * 2 + 1])
            .ok_or_else(|| wire_error("semantic hash contains a non-hexadecimal character"))?;
        *output = (high << 4) | low;
    }
    Ok(digest)
}

fn encode_semantic_hash(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn wire_error(message: impl Into<String>) -> InterchangeError {
    InterchangeError::Serialization(format!(
        "invalid cache feedback wire frame: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn cases() -> Vec<SemanticCacheFeedback> {
        vec![
            SemanticCacheFeedback::Ack(SemanticCacheAck::new(hash('a')).unwrap()),
            SemanticCacheFeedback::Miss(
                SemanticCacheMiss::new(hash('b'), SemanticCacheMissKind::SemanticReferenceTarget)
                    .unwrap(),
            ),
            SemanticCacheFeedback::Miss(
                SemanticCacheMiss::new(hash('c'), SemanticCacheMissKind::GraphDeltaBase).unwrap(),
            ),
            SemanticCacheFeedback::Revoke(SemanticCacheRevoke::new(hash('d')).unwrap()),
        ]
    }

    #[test]
    fn wire_round_trip_is_exact_for_every_feedback_form() {
        for feedback in cases() {
            let bytes = feedback.wire_bytes().unwrap();
            assert_eq!(bytes.len(), CACHE_FEEDBACK_WIRE_LEN);
            assert_eq!(
                SemanticCacheFeedback::from_wire_bytes(&bytes).unwrap(),
                feedback
            );
        }
    }

    #[test]
    fn wire_rejects_length_magic_version_kind_and_requirement_corruption() {
        let mut bytes = cases()[0].wire_bytes().unwrap();
        assert!(SemanticCacheFeedback::from_wire_bytes(&bytes[..37]).is_err());

        bytes[0] = b'X';
        assert!(SemanticCacheFeedback::from_wire_bytes(&bytes).is_err());
        bytes = cases()[0].wire_bytes().unwrap();

        bytes[3] = 2;
        assert!(SemanticCacheFeedback::from_wire_bytes(&bytes).is_err());
        bytes = cases()[0].wire_bytes().unwrap();

        bytes[4] = 99;
        assert!(SemanticCacheFeedback::from_wire_bytes(&bytes).is_err());
        bytes = cases()[0].wire_bytes().unwrap();

        bytes[5] = REQUIREMENT_GRAPH_DELTA_BASE;
        assert!(SemanticCacheFeedback::from_wire_bytes(&bytes).is_err());

        let mut miss = cases()[1].wire_bytes().unwrap();
        miss[5] = REQUIREMENT_NONE;
        assert!(SemanticCacheFeedback::from_wire_bytes(&miss).is_err());
    }

    #[test]
    fn wire_normalizes_legacy_uppercase_hash_text() {
        let uppercase = SemanticCacheFeedback::Ack(SemanticCacheAck::new(hash('A')).unwrap());
        let lowercase = SemanticCacheFeedback::Ack(SemanticCacheAck::new(hash('a')).unwrap());
        assert_eq!(
            uppercase.wire_bytes().unwrap(),
            lowercase.wire_bytes().unwrap()
        );
        assert_eq!(
            SemanticCacheFeedback::from_wire_bytes(&uppercase.wire_bytes().unwrap()).unwrap(),
            lowercase
        );
    }

    #[test]
    fn golden_frame_layout_is_stable() {
        let feedback = cases()[1].clone();
        let bytes = feedback.wire_bytes().unwrap();
        assert_eq!(&bytes[0..3], b"SCF");
        assert_eq!(bytes[3], 1);
        assert_eq!(bytes[4], KIND_MISS);
        assert_eq!(bytes[5], REQUIREMENT_SEMANTIC_REFERENCE_TARGET);
        assert_eq!(&bytes[6..], &[0xbb; 32]);
    }
}
