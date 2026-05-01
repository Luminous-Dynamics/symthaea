// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IPC bridge between the Prism renderer and the Symthaea Spore kernel.
//!
//! Uses Unix domain sockets with MessagePack encoding, matching the pattern
//! from `symthaea/src/shell/ipc_client.rs` and `symthaea-soma/src/holon_bridge.rs`.
//!
//! The bridge is async (tokio) with bounded channels to prevent backpressure.

use prism_common::{ContentZone, QueryId, SafetyLevel, SearchResult, TabId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum IPC frame size (2MB, matching existing Symthaea convention).
pub const MAX_FRAME_SIZE: usize = 2 * 1024 * 1024;

/// Default socket path for the Spore process.
pub const DEFAULT_SOCKET_PATH: &str = "/tmp/prism-spore.sock";

#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("Connection failed: {0}")]
    Connection(String),
    #[error("Serialization error: {0}")]
    Serialize(#[from] rmp_serde::encode::Error),
    #[error("Deserialization error: {0}")]
    Deserialize(#[from] rmp_serde::decode::Error),
    #[error("Frame too large: {size} bytes (max {max})")]
    FrameTooLarge { size: usize, max: usize },
    #[error("Channel closed")]
    ChannelClosed,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ═══════════════════════════════════════════════════════════════════════
// MESSAGES: Renderer → Spore
// ═══════════════════════════════════════════════════════════════════════

/// Messages sent from the Prism renderer to the Spore consciousness kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RendererToSpore {
    /// Content escalated from reflex arc for deep immune assessment.
    EscalatedContent {
        tab_id: TabId,
        url: String,
        content_hash: [u8; 32],
        snippet: String,
    },
    /// Page content for HDC encoding (only Public/Local zone pages).
    PageForEncoding {
        tab_id: TabId,
        url: String,
        text_content: String,
        zone: ContentZone,
    },
    /// Epistemic search query.
    SearchQuery { query_id: QueryId, text: String },
    /// Ping/keepalive.
    Ping(u64),
}

// ═══════════════════════════════════════════════════════════════════════
// MESSAGES: Spore → Renderer
// ═══════════════════════════════════════════════════════════════════════

/// Messages sent from the Spore kernel back to the renderer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SporeToRenderer {
    /// Deep threat assessment result (in response to EscalatedContent).
    ThreatVerdict {
        tab_id: TabId,
        safety_level: SafetyLevel,
        reason: String,
    },
    /// Epistemic search results.
    SearchResults {
        query_id: QueryId,
        results: Vec<SearchResult>,
    },
    /// Periodic consciousness heartbeat.
    Heartbeat {
        phi: f32,
        safety_level: SafetyLevel,
        cycle: u64,
    },
    /// Pong response.
    Pong(u64),
}

// ═══════════════════════════════════════════════════════════════════════
// FRAME ENCODING
// ═══════════════════════════════════════════════════════════════════════

/// Encode a message to a length-prefixed MessagePack frame.
pub fn encode_frame<T: Serialize>(msg: &T) -> Result<Vec<u8>, BridgeError> {
    let payload = rmp_serde::to_vec(msg)?;
    if payload.len() > MAX_FRAME_SIZE {
        return Err(BridgeError::FrameTooLarge {
            size: payload.len(),
            max: MAX_FRAME_SIZE,
        });
    }
    let len = (payload.len() as u32).to_be_bytes();
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&len);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decode a MessagePack payload (without length prefix).
pub fn decode_payload<'a, T: Deserialize<'a>>(data: &'a [u8]) -> Result<T, BridgeError> {
    Ok(rmp_serde::from_slice(data)?)
}

/// Compute BLAKE3 hash of content (for deduplication and escalation).
pub fn content_hash(content: &str) -> [u8; 32] {
    blake3::hash(content.as_bytes()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_renderer_to_spore() {
        let msg = RendererToSpore::SearchQuery {
            query_id: 42,
            text: "What causes ocean acidification?".to_string(),
        };
        let frame = encode_frame(&msg).unwrap();
        assert!(frame.len() > 4);

        // Decode (skip 4-byte length prefix)
        let decoded: RendererToSpore = decode_payload(&frame[4..]).unwrap();
        match decoded {
            RendererToSpore::SearchQuery { query_id, text } => {
                assert_eq!(query_id, 42);
                assert_eq!(text, "What causes ocean acidification?");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn roundtrip_spore_to_renderer() {
        let msg = SporeToRenderer::Heartbeat {
            phi: 0.72,
            safety_level: SafetyLevel::Green,
            cycle: 1000,
        };
        let frame = encode_frame(&msg).unwrap();
        let decoded: SporeToRenderer = decode_payload(&frame[4..]).unwrap();
        match decoded {
            SporeToRenderer::Heartbeat {
                phi,
                safety_level,
                cycle,
            } => {
                assert!((phi - 0.72).abs() < 0.001);
                assert_eq!(safety_level, SafetyLevel::Green);
                assert_eq!(cycle, 1000);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn roundtrip_escalated_content() {
        let hash = content_hash("suspicious content");
        let msg = RendererToSpore::EscalatedContent {
            tab_id: 1,
            url: "https://example.com".to_string(),
            content_hash: hash,
            snippet: "suspicious content".to_string(),
        };
        let frame = encode_frame(&msg).unwrap();
        let decoded: RendererToSpore = decode_payload(&frame[4..]).unwrap();
        match decoded {
            RendererToSpore::EscalatedContent {
                content_hash: h, ..
            } => {
                assert_eq!(h, hash);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn frame_too_large_rejected() {
        let big = "x".repeat(MAX_FRAME_SIZE + 1);
        let msg = RendererToSpore::PageForEncoding {
            tab_id: 1,
            url: "https://example.com".to_string(),
            text_content: big,
            zone: ContentZone::Local,
        };
        assert!(matches!(
            encode_frame(&msg),
            Err(BridgeError::FrameTooLarge { .. })
        ));
    }

    #[test]
    fn content_hash_deterministic() {
        let h1 = content_hash("test content");
        let h2 = content_hash("test content");
        let h3 = content_hash("different content");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn roundtrip_search_results() {
        use prism_common::{EmpiricalLevel, MaterialityLevel, NormativeLevel};

        let msg = SporeToRenderer::SearchResults {
            query_id: 99,
            results: vec![SearchResult {
                content: "Ocean acidification is caused by CO2 absorption.".to_string(),
                sources: vec!["https://noaa.gov".to_string()],
                empirical_level: EmpiricalLevel::E4,
                normative_level: NormativeLevel::N2,
                materiality_level: MaterialityLevel::M2,
                query_similarity: 0.92,
                author_reputation: 0.95,
                age_days: 30,
                tags: vec!["climate".to_string(), "ocean".to_string()],
            }],
        };
        let frame = encode_frame(&msg).unwrap();
        let decoded: SporeToRenderer = decode_payload(&frame[4..]).unwrap();
        match decoded {
            SporeToRenderer::SearchResults { query_id, results } => {
                assert_eq!(query_id, 99);
                assert_eq!(results.len(), 1);
                assert!(results[0].rank_score() > 0.5);
            }
            _ => panic!("Wrong variant"),
        }
    }
}
