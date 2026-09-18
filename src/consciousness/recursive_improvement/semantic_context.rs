// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic context retrieval for dream/replay reasoning.
//!
//! Context memory deliberately uses three different representations for three
//! different jobs:
//!
//! - a fast 64-bit hash for legacy lookup/bucketing;
//! - a collision-resistant BLAKE3 digest for exact identity/provenance;
//! - a deterministic HDC key for approximate semantic retrieval.
//!
//! None of these representations is evidence authority. Semantic matches may
//! retrieve candidate prior contexts, but callers must resolve exact provenance
//! through the evidence store before making empirical or confidence claims.

use super::dream_feedback::hash_context;
use symthaea_core::hdc::BinaryHV;

pub const DEFAULT_SEMANTIC_LEVELS: u16 = 64;
pub const DEFAULT_CONTEXT_CLAMP_ABS: f32 = 1.0;

/// Expected Hamming similarity for unrelated binary hypervectors.
///
/// This is used only to normalize retrieval support. It is not a probability,
/// evidence score, confidence value, or validation threshold.
pub const HDC_CHANCE_SIMILARITY: f32 = 0.5;

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

/// Exact context identity.
///
/// `fast_hash` preserves the historical 64-bit FNV-derived lookup key. It is not
/// sufficient for durable identity on its own. `exact_digest` is the canonical
/// BLAKE3 digest over little-endian `f32` bytes and is the collision-resistant
/// identity used for deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextIdentity {
    pub fast_hash: u64,
    pub exact_digest: [u8; 32],
}

impl ContextIdentity {
    pub fn from_context(context: &[f32]) -> Result<Self, SemanticContextError> {
        validate_context(context)?;
        Ok(Self {
            fast_hash: hash_context(context),
            exact_digest: exact_context_digest(context),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticContextMatch {
    /// Collision-resistant exact identity of the indexed context.
    pub identity: ContextIdentity,
    /// HDC Hamming similarity in [0, 1]. This is not evidence confidence.
    pub similarity: f32,
    /// Semantic distance = 1 - similarity.
    pub semantic_distance: f32,
    /// Chance-corrected retrieval weight in [0, 1].
    ///
    /// Unrelated binary hypervectors average around 0.5 similarity, so chance
    /// similarity maps to zero retrieval support. This value is appropriate for
    /// ranking or attenuating candidate priors only; it grants no epistemic authority.
    pub retrieval_weight: f32,
}

impl SemanticContextMatch {
    fn new(identity: ContextIdentity, similarity: f32) -> Self {
        let similarity = similarity.clamp(0.0, 1.0);
        Self {
            identity,
            similarity,
            semantic_distance: 1.0 - similarity,
            retrieval_weight: retrieval_weight(similarity),
        }
    }
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
        validate_context(context)?;

        let mut components = Vec::with_capacity(context.len() + 1);
        components.push(BinaryHV::random(mix64(
            LENGTH_SALT ^ context.len() as u64,
        )));

        for (index, &value) in context.iter().enumerate() {
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
    identity: ContextIdentity,
    semantic_key: BinaryHV,
}

/// Runtime-only semantic index over exact context identities.
///
/// The index intentionally stores no evidence kind, validation token, confidence
/// authority, or generated outcome. Retrieval produces candidate identities only.
/// Exact deduplication uses the 256-bit digest, never the legacy 64-bit hash.
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

    /// Insert or refresh a context.
    ///
    /// Deduplication is based on BLAKE3 exact identity. The 64-bit hash remains a
    /// lookup hint only, so two contexts that collide in the fast hash cannot
    /// overwrite each other unless they also collide in the 256-bit digest.
    pub fn insert(&mut self, context: &[f32]) -> Result<ContextIdentity, SemanticContextError> {
        let identity = ContextIdentity::from_context(context)?;
        let semantic_key = self.encoder.encode(context)?;
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.identity.exact_digest == identity.exact_digest)
        {
            entry.identity = identity;
            entry.semantic_key = semantic_key;
        } else {
            self.entries.push(SemanticContextEntry {
                identity,
                semantic_key,
            });
        }
        Ok(identity)
    }

    /// Return whether this exact collision-resistant context identity is indexed.
    pub fn contains_exact(&self, identity: ContextIdentity) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.identity == identity)
    }

    /// Return exact identities sharing a fast-hash bucket.
    ///
    /// Multiple results are valid and must remain distinct; callers must compare
    /// `exact_digest` before treating any candidate as an exact match.
    pub fn identities_for_fast_hash(&self, fast_hash: u64) -> Vec<ContextIdentity> {
        self.entries
            .iter()
            .filter(|entry| entry.identity.fast_hash == fast_hash)
            .map(|entry| entry.identity)
            .collect()
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
    /// `min_similarity` must be in the actual BinaryHV Hamming-similarity range
    /// [0, 1]. No returned match is empirical evidence. Callers must resolve the
    /// exact identity through their provenance/evidence store before using it for
    /// anything stronger than retrieval or hypothesis generation.
    pub fn nearest(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<SemanticContextMatch>, SemanticContextError> {
        if !min_similarity.is_finite() || !(0.0..=1.0).contains(&min_similarity) {
            return Err(SemanticContextError::InvalidSimilarityThreshold);
        }
        self.nearest_impl(query, Some(min_similarity), limit)
    }

    /// Return nearest contexts without applying a similarity cutoff.
    ///
    /// This explicit API replaces the old convention of passing a negative
    /// threshold, which was outside BinaryHV's real similarity range.
    pub fn nearest_unfiltered(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<SemanticContextMatch>, SemanticContextError> {
        self.nearest_impl(query, None, limit)
    }

    fn nearest_impl(
        &self,
        query: &[f32],
        min_similarity: Option<f32>,
        limit: usize,
    ) -> Result<Vec<SemanticContextMatch>, SemanticContextError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let query_key = self.encoder.encode(query)?;
        let mut matches: Vec<_> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let similarity = query_key.similarity(&entry.semantic_key);
                if min_similarity.is_some_and(|minimum| similarity < minimum) {
                    return None;
                }
                Some(SemanticContextMatch::new(entry.identity, similarity))
            })
            .collect();
        matches.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.identity.exact_digest.cmp(&right.identity.exact_digest))
        });
        matches.truncate(limit);
        Ok(matches)
    }
}

fn validate_context(context: &[f32]) -> Result<(), SemanticContextError> {
    if context.is_empty() {
        return Err(SemanticContextError::EmptyContext);
    }
    for (index, value) in context.iter().enumerate() {
        if !value.is_finite() {
            return Err(SemanticContextError::NonFiniteValue { index });
        }
    }
    Ok(())
}

/// Canonical collision-resistant digest for an exact numeric context.
///
/// This intentionally mirrors Symthaea's existing integrity convention: each
/// `f32` contributes its IEEE-754 little-endian bytes to BLAKE3. The helper is
/// local because recursive-improvement is available without the optional
/// `integrity` feature.
fn exact_context_digest(context: &[f32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for value in context {
        hasher.update(&value.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[inline]
fn retrieval_weight(similarity: f32) -> f32 {
    ((similarity - HDC_CHANCE_SIMILARITY) / (1.0 - HDC_CHANCE_SIMILARITY)).clamp(0.0, 1.0)
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
    fn exact_identity_uses_digest_in_addition_to_fast_hash() {
        let left = ContextIdentity::from_context(&[0.20, -0.40, 0.80]).unwrap();
        let right = ContextIdentity::from_context(&[0.21, -0.40, 0.80]).unwrap();
        assert_ne!(left, right);
        assert_ne!(left.exact_digest, right.exact_digest);
    }

    #[test]
    fn tiny_float_change_keeps_semantic_similarity_while_exact_identity_changes() {
        let encoder = SemanticContextEncoder::default();
        let left = [0.20, -0.40, 0.80];
        let right = [0.21, -0.40, 0.80];
        assert_ne!(
            ContextIdentity::from_context(&left).unwrap(),
            ContextIdentity::from_context(&right).unwrap()
        );
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
    fn nearest_returns_collision_resistant_identity_not_generated_evidence() {
        let mut index = SemanticContextIndex::default();
        let exact = index.insert(&[0.1, 0.2, 0.3]).unwrap();
        index.insert(&[-0.9, 0.8, -0.7]).unwrap();

        let matches = index.nearest_unfiltered(&[0.11, 0.2, 0.3], 1).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].identity, exact);
        assert!(matches[0].similarity.is_finite());
        assert!(matches[0].semantic_distance.is_finite());
        assert!(matches[0].retrieval_weight.is_finite());
    }

    #[test]
    fn exact_reinsert_deduplicates_by_digest() {
        let mut index = SemanticContextIndex::default();
        let identity = index.insert(&[0.3, 0.4]).unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(index.insert(&[0.3, 0.4]).unwrap(), identity);
        assert_eq!(index.len(), 1);
        assert!(index.contains_exact(identity));
    }

    #[test]
    fn chance_similarity_has_zero_retrieval_weight() {
        assert_eq!(retrieval_weight(HDC_CHANCE_SIMILARITY), 0.0);
        assert_eq!(retrieval_weight(1.0), 1.0);
        assert_eq!(retrieval_weight(0.0), 0.0);
    }

    #[test]
    fn negative_similarity_threshold_is_rejected() {
        let index = SemanticContextIndex::default();
        assert_eq!(
            index.nearest(&[0.1], -0.01, 1),
            Err(SemanticContextError::InvalidSimilarityThreshold)
        );
    }

    #[test]
    fn invalid_context_values_fail_closed() {
        let encoder = SemanticContextEncoder::default();
        assert_eq!(
            encoder.encode(&[0.0, f32::NAN]),
            Err(SemanticContextError::NonFiniteValue { index: 1 })
        );
        assert_eq!(
            ContextIdentity::from_context(&[f32::INFINITY]),
            Err(SemanticContextError::NonFiniteValue { index: 0 })
        );
    }
}
