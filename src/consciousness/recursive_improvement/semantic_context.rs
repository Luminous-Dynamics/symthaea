// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic context retrieval for dream/replay reasoning.
//!
//! Exact context hashes remain the identity/provenance key. This module adds a
//! separate HDC similarity key for approximate retrieval only. Semantic matches
//! carry no evidence authority and cannot by themselves promote confidence.

use super::dream_feedback::hash_context;
use symthaea_core::hdc::BinaryHV;

pub const DEFAULT_SEMANTIC_LEVELS: u16 = 64;
pub const DEFAULT_CONTEXT_CLAMP_ABS: f32 = 1.0;

const ROLE_SALT: u64 = 0x5345_4d41_4e54_4943;
const LEVEL_LOW_SEED: u64 = 0x4c45_5645_4c5f_4c4f;
const LEVEL_HIGH_SEED: u64 = 0x4c45_5645_4c5f_4849;
const LEVEL_RANK_SALT: u64 = 0x5448_4552_4d4f_4d45;
const LENGTH_SALT: u64 = 0x434f_4e54_4558_544c;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticContextError {
    EmptyContext,
    NonFiniteValue { index: usize },
    InvalidLevelCount,
    InvalidClamp,
    InvalidSimilarityThreshold,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticContextMatch {
    /// Exact identity/provenance key for the indexed context.
    pub exact_context_hash: u64,
    /// HDC similarity only. This is not an evidence or validation score.
    pub similarity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticContextEncoder {
    levels: u16,
    clamp_abs: f32,
}

impl Default for SemanticContextEncoder {
    fn default() -> Self {
        Self {
            levels: DEFAULT_SEMANTIC_LEVELS,
            clamp_abs: DEFAULT_CONTEXT_CLAMP_ABS,
        }
    }
}

impl SemanticContextEncoder {
    pub fn new(levels: u16, clamp_abs: f32) -> Result<Self, SemanticContextError> {
        if levels < 2 {
            return Err(SemanticContextError::InvalidLevelCount);
        }
        if !clamp_abs.is_finite() || clamp_abs <= 0.0 {
            return Err(SemanticContextError::InvalidClamp);
        }
        Ok(Self { levels, clamp_abs })
    }

    pub fn levels(&self) -> u16 {
        self.levels
    }

    pub fn clamp_abs(&self) -> f32 {
        self.clamp_abs
    }

    /// Encode a numeric context into a deterministic locality-preserving HDC key.
    ///
    /// Each scalar uses monotone level encoding between deterministic endpoint
    /// hypervectors. Nearby values therefore differ by only a small subset of bits.
    /// Dimension-role binding prevents equal scalar values in different positions
    /// from collapsing to the same representation; bundling forms the context key.
    pub fn encode(&self, context: &[f32]) -> Result<BinaryHV, SemanticContextError> {
        if context.is_empty() {
            return Err(SemanticContextError::EmptyContext);
        }

        let mut components = Vec::with_capacity(context.len() + 1);
        components.push(BinaryHV::random(mix64(
            LENGTH_SALT ^ context.len() as u64,
        )));

        for (index, &value) in context.iter().enumerate() {
            if !value.is_finite() {
                return Err(SemanticContextError::NonFiniteValue { index });
            }
            let role = BinaryHV::random(mix64(ROLE_SALT ^ index as u64));
            let level = self.level_hv(value);
            components.push(role.bind(&level));
        }

        Ok(BinaryHV::bundle(&components))
    }

    fn level_hv(&self, value: f32) -> BinaryHV {
        let low = BinaryHV::random(LEVEL_LOW_SEED);
        let high = BinaryHV::random(LEVEL_HIGH_SEED);
        let normalized = ((value.clamp(-self.clamp_abs, self.clamp_abs) + self.clamp_abs)
            / (2.0 * self.clamp_abs))
            .clamp(0.0, 1.0);
        let steps = u64::from(self.levels - 1);
        let level = (normalized * steps as f32).round() as u64;

        let mut bytes = low.0;
        for (byte_index, out) in bytes.iter_mut().enumerate() {
            let low_byte = low.0[byte_index];
            let high_byte = high.0[byte_index];
            let differing = low_byte ^ high_byte;
            if differing == 0 {
                continue;
            }

            let mut next = low_byte;
            for bit in 0..8_u64 {
                let mask = 1_u8 << bit;
                if differing & mask == 0 {
                    continue;
                }
                let bit_index = byte_index as u64 * 8 + bit;
                let rank = mix64(LEVEL_RANK_SALT ^ bit_index) % steps;
                if rank < level {
                    if high_byte & mask != 0 {
                        next |= mask;
                    } else {
                        next &= !mask;
                    }
                }
            }
            *out = next;
        }

        BinaryHV(bytes)
    }
}

#[derive(Clone)]
struct SemanticContextEntry {
    exact_context_hash: u64,
    semantic_key: BinaryHV,
}

/// Runtime-only semantic index over exact context identities.
///
/// The index intentionally stores no evidence kind, validation token, confidence
/// authority, or generated outcome. Retrieval produces candidate identities only.
#[derive(Clone, Default)]
pub struct SemanticContextIndex {
    encoder: SemanticContextEncoder,
    entries: Vec<SemanticContextEntry>,
}

impl SemanticContextIndex {
    pub fn new(encoder: SemanticContextEncoder) -> Self {
        Self {
            encoder,
            entries: Vec::new(),
        }
    }

    pub fn encoder(&self) -> SemanticContextEncoder {
        self.encoder
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert or refresh a context. The returned key is the pre-existing exact
    /// FNV identity used by the dream-feedback bridge, not the HDC similarity key.
    pub fn insert(&mut self, context: &[f32]) -> Result<u64, SemanticContextError> {
        let exact_context_hash = hash_context(context);
        let semantic_key = self.encoder.encode(context)?;
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.exact_context_hash == exact_context_hash)
        {
            entry.semantic_key = semantic_key;
        } else {
            self.entries.push(SemanticContextEntry {
                exact_context_hash,
                semantic_key,
            });
        }
        Ok(exact_context_hash)
    }

    pub fn similarity(
        &self,
        left: &[f32],
        right: &[f32],
    ) -> Result<f32, SemanticContextError> {
        let left = self.encoder.encode(left)?;
        let right = self.encoder.encode(right)?;
        Ok(left.similarity(&right))
    }

    /// Return the nearest exact context identities by semantic similarity.
    ///
    /// No returned match is empirical evidence. Callers must resolve the exact
    /// hash back through their own provenance/evidence store before using it for
    /// anything stronger than retrieval or hypothesis generation.
    pub fn nearest(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<SemanticContextMatch>, SemanticContextError> {
        if !min_similarity.is_finite() || !(-1.0..=1.0).contains(&min_similarity) {
            return Err(SemanticContextError::InvalidSimilarityThreshold);
        }
        if limit == 0 {
            return Ok(Vec::new());
        }

        let query_key = self.encoder.encode(query)?;
        let mut matches: Vec<_> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let similarity = query_key.similarity(&entry.semantic_key);
                (similarity >= min_similarity).then_some(SemanticContextMatch {
                    exact_context_hash: entry.exact_context_hash,
                    similarity,
                })
            })
            .collect();
        matches.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.exact_context_hash.cmp(&right.exact_context_hash))
        });
        matches.truncate(limit);
        Ok(matches)
    }
}

#[inline]
fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_encoding_is_deterministic() {
        let encoder = SemanticContextEncoder::default();
        let context = [0.2, -0.4, 0.8];
        assert_eq!(encoder.encode(&context).unwrap(), encoder.encode(&context).unwrap());
    }

    #[test]
    fn tiny_float_change_keeps_semantic_similarity_while_exact_identity_changes() {
        let encoder = SemanticContextEncoder::default();
        let left = [0.20, -0.40, 0.80];
        let right = [0.21, -0.40, 0.80];
        assert_ne!(hash_context(&left), hash_context(&right));
        let similarity = encoder
            .encode(&left)
            .unwrap()
            .similarity(&encoder.encode(&right).unwrap());
        assert!(similarity > 0.90, "semantic similarity={similarity}");
    }

    #[test]
    fn nearby_contexts_are_more_similar_than_distant_contexts() {
        let index = SemanticContextIndex::default();
        let anchor = [0.0, 0.0, 0.0, 0.0];
        let nearby = [0.02, 0.0, -0.01, 0.0];
        let distant = [1.0, -1.0, 1.0, -1.0];
        let near_similarity = index.similarity(&anchor, &nearby).unwrap();
        let far_similarity = index.similarity(&anchor, &distant).unwrap();
        assert!(near_similarity > far_similarity);
    }

    #[test]
    fn nearest_returns_exact_identity_not_generated_evidence() {
        let mut index = SemanticContextIndex::default();
        let exact = index.insert(&[0.1, 0.2, 0.3]).unwrap();
        index.insert(&[-0.9, 0.8, -0.7]).unwrap();

        let matches = index.nearest(&[0.11, 0.2, 0.3], -1.0, 1).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].exact_context_hash, exact);
        assert!(matches[0].similarity.is_finite());
    }

    #[test]
    fn invalid_context_values_fail_closed() {
        let encoder = SemanticContextEncoder::default();
        assert_eq!(
            encoder.encode(&[0.0, f32::NAN]),
            Err(SemanticContextError::NonFiniteValue { index: 1 })
        );
    }
}
