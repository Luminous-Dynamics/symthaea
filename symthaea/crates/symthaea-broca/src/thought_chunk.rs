// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thought Chunk abstractions for semantic autoregressive generation.
//!
//! A ThoughtChunk represents a semantic unit of meaning (roughly 8–20 tokens).
//! The goal is to move from token-level to chunk-level prediction.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

/// Semantic output domain for a thought chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThoughtChunkKind {
    Text,
    Code,
    Action,
    StructuredData,
}

/// A single semantic unit for high-psi thought prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtChunk {
    pub id: String,
    pub kind: ThoughtChunkKind,
    pub thought_hv: ContinuousHV,
    pub token_span: Option<(usize, usize)>,
    pub psi: f32,
    pub confidence: f32,
    pub target: Option<String>,
}

impl ThoughtChunk {
    pub fn new(
        id: impl Into<String>,
        kind: ThoughtChunkKind,
        thought_hv: ContinuousHV,
        psi: f32,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            thought_hv,
            token_span: None,
            psi: psi.clamp(0.0, 1.0),
            confidence: 0.0,
            target: None,
        }
    }

    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    pub fn with_token_span(mut self, start: usize, end: usize) -> Self {
        self.token_span = Some((start, end.max(start)));
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn is_high_psi(&self) -> bool {
        self.psi > 0.7
    }

    pub fn summary(&self) -> String {
        format!(
            "[Chunk {} | {:?} | ψ={:.2} | conf={:.2}]",
            self.id, self.kind, self.psi, self.confidence
        )
    }
}

/// Sequence of semantic chunks (primary output of chunk-aware generation).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThoughtChunkSequence {
    pub source_id: String,
    pub chunks: Vec<ThoughtChunk>,
}

impl ThoughtChunkSequence {
    pub fn new(source_id: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            chunks: Vec::new(),
        }
    }

    pub fn push(&mut self, chunk: ThoughtChunk) {
        self.chunks.push(chunk);
    }

    pub fn mean_psi(&self) -> f32 {
        if self.chunks.is_empty() {
            0.0
        } else {
            self.chunks.iter().map(|c| c.psi).sum::<f32>() / self.chunks.len() as f32
        }
    }

    pub fn total_confidence(&self) -> f32 {
        if self.chunks.is_empty() {
            0.0
        } else {
            self.chunks.iter().map(|c| c.confidence).sum::<f32>() / self.chunks.len() as f32
        }
    }

    pub fn to_text(&self) -> String {
        self.chunks
            .iter()
            .filter_map(|c| c.target.as_deref())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Convert semantic monologue into Geodesic ProgramNode sequence.
    /// This is the primary bridge from Broca → Geodesic.
    pub fn to_program_nodes(&self) -> Vec<ProgramNode> {
        self.chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| ProgramNode::from_chunk(chunk, i))
            .collect()
    }

    /// Convert to a topological skeleton (for Geodesic skeleton synthesis).
    pub fn to_topological_skeleton(&self) -> Vec<(String, Vec<String>)> {
        self.chunks
            .iter()
            .enumerate()
            .map(|(i, _chunk)| {
                let id = format!("chunk_{}", i);
                let deps = if i > 0 {
                    vec![format!("chunk_{}", i - 1)]
                } else {
                    vec![]
                };
                (id, deps)
            })
            .collect()
    }
}

/// Node kind for Geodesic program synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Text,
    Code,
    Action,
    StructuredData,
    Hypothesis,
    PlanStep,
}

/// A node in the Geodesic program graph.
/// Produced from ThoughtChunks for topological synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramNode {
    pub id: String,
    pub kind: NodeKind,
    pub semantic_hv: ContinuousHV,
    pub content: String,
    pub psi: f32,
    pub confidence: f32,
    pub parent_id: Option<String>,
}

impl ProgramNode {
    pub fn from_chunk(chunk: &ThoughtChunk, index: usize) -> Self {
        Self {
            id: format!("node_{}", index),
            kind: match chunk.kind {
                ThoughtChunkKind::Text => NodeKind::Text,
                ThoughtChunkKind::Code => NodeKind::Code,
                ThoughtChunkKind::Action => NodeKind::Action,
                ThoughtChunkKind::StructuredData => NodeKind::StructuredData,
            },
            semantic_hv: chunk.thought_hv.clone(),
            content: chunk.target.clone().unwrap_or_default(),
            psi: chunk.psi,
            confidence: chunk.confidence,
            parent_id: if index > 0 {
                Some(format!("node_{}", index - 1))
            } else {
                None
            },
        }
    }
}

/// Dynamic chunker for hierarchical thought chunks.
pub struct DynamicChunker {
    pub coherence_velocity_threshold: f32,
    pub min_chunk_size: usize,
    pub current_chunk_tokens: Vec<u32>,
    pub current_chunk_hvs: Vec<ContinuousHV>,
    pub last_coherence: f32,
}

impl DynamicChunker {
    pub fn new(velocity_threshold: f32, min_chunk_size: usize) -> Self {
        Self {
            coherence_velocity_threshold: velocity_threshold,
            min_chunk_size,
            current_chunk_tokens: Vec::new(),
            current_chunk_hvs: Vec::new(),
            last_coherence: 1.0,
        }
    }

    pub fn process_token(&mut self, token: u32, hv: ContinuousHV, current_coherence: f32) -> bool {
        self.current_chunk_tokens.push(token);
        self.current_chunk_hvs.push(hv);

        let velocity = current_coherence - self.last_coherence;
        self.last_coherence = current_coherence;

        let should_boundary = self.current_chunk_tokens.len() >= self.min_chunk_size
            && velocity < -self.coherence_velocity_threshold;

        should_boundary
    }

    pub fn reset(&mut self) {
        self.current_chunk_tokens.clear();
        self.current_chunk_hvs.clear();
        self.last_coherence = 1.0;
    }
}

/// Contract for turning semantic chunks into external representations.
pub trait ThoughtChunkDecoder {
    type Error;

    fn decode_chunk(&self, chunk: &ThoughtChunk) -> Result<String, Self::Error>;
}

/// Simple decoder that prefers the `target` field, with fallback.
pub struct SimpleThoughtChunkDecoder;

impl ThoughtChunkDecoder for SimpleThoughtChunkDecoder {
    type Error = anyhow::Error;

    fn decode_chunk(&self, chunk: &ThoughtChunk) -> Result<String, Self::Error> {
        if let Some(target) = &chunk.target {
            return Ok(target.clone());
        }
        Ok(format!("[chunk:{}]", chunk.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::HDC_DIMENSION;

    #[test]
    fn thought_chunk_helpers_work() {
        let hv = ContinuousHV::zero(HDC_DIMENSION);
        let chunk = ThoughtChunk::new("c0", ThoughtChunkKind::Text, hv, 0.85)
            .with_confidence(0.92)
            .with_target("Hello world");

        assert!(chunk.is_high_psi());
        assert_eq!(chunk.confidence, 0.92);
        assert_eq!(chunk.target.as_deref(), Some("Hello world"));
    }

    #[test]
    fn thought_chunk_sequence_metrics() {
        let hv = ContinuousHV::zero(HDC_DIMENSION);
        let mut seq = ThoughtChunkSequence::new("test");
        seq.push(
            ThoughtChunk::new("a", ThoughtChunkKind::Text, hv.clone(), 0.6).with_confidence(0.8),
        );
        seq.push(ThoughtChunk::new("b", ThoughtChunkKind::Action, hv, 0.9).with_confidence(0.7));

        assert_eq!(seq.mean_psi(), 0.75);
        assert_eq!(seq.total_confidence(), 0.75);
    }
}
