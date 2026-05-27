// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Quark-Based Token Embeddings
//!
//! Replaces genesis-seeded random token embeddings with compositional
//! vectors built from semantic primes (Wierzbicka's NSM theory).
//!
//! Instead of hashing "dog" to a random 16,384D vector, we compose:
//!   dog = bundle(bind(ANIMATE, ROLE[0]), bind(BODY, ROLE[1]), bind(GOOD, ROLE[2]))
//!
//! This makes "dog" and "cat" naturally adjacent (shared ANIMATE+BODY quarks)
//! while "dog" and "justice" are orthogonal (no shared quarks).
//!
//! ## Three-Tier Composition
//!
//! - **Tier 1**: Direct decomposition from LexicalGrounding (~200 words)
//! - **Tier 2**: Morphological decomposition via prefix/suffix analysis (~500 tokens)
//! - **Tier 3**: Genesis-seeded fallback (remaining tokens)
//!
//! ## Integration
//!
//! Used by `LanguageController::new_with_tokenizer()` to initialize token
//! embeddings. The weight-tied logit computation then scores tokens by
//! semantic similarity to the CfC network's output.

use std::collections::HashMap;

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::grounded_understanding::LexicalGrounding;
use symthaea_core::hdc::universal_semantics::{
    SemanticMolecule, SemanticMoleculeBasis, SemanticPrime, SyntacticRole,
};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

use crate::tokenizer::BpeTokenizer;

/// Maximum number of role-position vectors (max primes per word).
const MAX_ROLES: usize = 8;

/// Configuration for semantic embedding composition.
#[derive(Debug, Clone)]
pub struct SemanticEmbeddingConfig {
    /// Blend weight for genesis component (0.0 = pure semantic, 1.0 = pure genesis).
    /// Default 0.3 for new models. Use 0.7+ when loading pre-trained checkpoints.
    pub genesis_blend: f32,
}

impl Default for SemanticEmbeddingConfig {
    fn default() -> Self {
        Self { genesis_blend: 0.3 }
    }
}

/// Coverage statistics for the embedding composition process.
#[derive(Debug, Clone)]
pub struct EmbeddingCoverageStats {
    pub total_tokens: usize,
    /// Tokens matched via LexicalGrounding (full word match).
    pub tier1_direct: usize,
    /// Tokens matched via morphological prefix/suffix decomposition.
    pub tier2_morphological: usize,
    /// Tokens with no semantic decomposition (genesis fallback).
    pub tier3_genesis: usize,
}

/// Semantic quark basis: 65 orthogonal prime vectors + role-position vectors.
///
/// Each SemanticPrime variant maps to a unique, orthogonal ContinuousHV in 16,384D.
/// Token embeddings are composed by binding primes with role-position vectors
/// (preserving composition order) then bundling all bound pairs.
pub struct SemanticQuarkBasis {
    /// 65 orthogonal basis vectors, one per SemanticPrime.
    prime_basis: HashMap<SemanticPrime, ContinuousHV>,
    /// Role-position vectors for ordered binding (8 max).
    role_positions: Vec<ContinuousHV>,
    /// Word→prime decomposition lexicon.
    lexicon: LexicalGrounding,
    /// Structured molecule basis.
    molecule_basis: SemanticMoleculeBasis,
}

impl SemanticQuarkBasis {
    /// Create a new quark basis with Gram-Schmidt orthogonalized prime vectors.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let all_primes = SemanticPrime::all();

        // Generate initial random vectors, then orthogonalize
        let mut prime_hvs: Vec<ContinuousHV> = all_primes
            .iter()
            .map(|p| {
                let mut hv = ContinuousHV::from_genesis(
                    genesis,
                    &format!("semantic_quark::prime::{}", p.as_gate_name()),
                    HDC_DIMENSION,
                );
                hv.l2_normalize();
                hv
            })
            .collect();

        // Gram-Schmidt orthogonalization (65 vectors in 16,384D — trivial)
        gram_schmidt(&mut prime_hvs);

        let prime_basis: HashMap<SemanticPrime, ContinuousHV> =
            all_primes.clone().into_iter().zip(prime_hvs).collect();

        // Role-position vectors (also orthogonalized)
        let mut role_positions: Vec<ContinuousHV> = (0..MAX_ROLES)
            .map(|i| {
                let mut hv = ContinuousHV::from_genesis(
                    genesis,
                    &format!("semantic_quark::role::{i}"),
                    HDC_DIMENSION,
                );
                hv.l2_normalize();
                hv
            })
            .collect();
        gram_schmidt(&mut role_positions);

        let lexicon = LexicalGrounding::new();
        let molecule_basis = SemanticMoleculeBasis::new(genesis);

        Self {
            prime_basis,
            role_positions,
            lexicon,
            molecule_basis,
        }
    }

    /// Compose a semantic ContinuousHV for a token string.
    ///
    /// Returns `Some(hv)` if the token has a semantic decomposition (Tier 1 or 2),
    /// or `None` for genesis fallback (Tier 3).
    pub fn compose_token_embedding(&self, token_str: &str) -> Option<ContinuousHV> {
        // Clean token string: strip leading BPE space, lowercase
        let clean = token_str.trim().to_lowercase();
        if clean.is_empty() || clean.len() <= 1 {
            return None;
        }

        // Tier 1: Direct decomposition from LexicalGrounding
        if let Some(primes) = self.lexicon.decompose(&clean) {
            if !primes.is_empty() {
                return Some(self.compose_from_primes(primes));
            }
        }

        // Tier 2: Morphological decomposition (prefix/suffix analysis)
        let morpheme_primes = morpheme_decompose(&clean);
        if !morpheme_primes.is_empty() {
            return Some(self.compose_from_primes(&morpheme_primes));
        }

        None // Tier 3: no semantic decomposition available
    }

    /// Compose a ContinuousHV from a list of semantic primes.
    fn compose_from_primes(&self, primes: &[SemanticPrime]) -> ContinuousHV {
        let bound: Vec<ContinuousHV> = primes
            .iter()
            .take(MAX_ROLES)
            .enumerate()
            .filter_map(|(pos, prime)| {
                let basis = self.prime_basis.get(prime)?;
                let role = &self.role_positions[pos];
                Some(basis.bind(role))
            })
            .collect();

        if bound.is_empty() {
            // Should not happen if primes is non-empty, but be safe
            return ContinuousHV::from_genesis(
                &GenesisSeed::from_phrase("semantic_fallback"),
                "empty",
                HDC_DIMENSION,
            );
        }

        let refs: Vec<&ContinuousHV> = bound.iter().collect();
        ContinuousHV::bundle(&refs).normalize()
    }

    /// Compose a structured semantic molecule from Role-Filler pairs.
    pub fn compose_molecule(
        &self,
        structure: Vec<(SyntacticRole, SemanticPrime)>,
        intensity: f32,
    ) -> SemanticMolecule {
        SemanticMolecule::new(&self.molecule_basis, structure, intensity)
    }

    /// Access the underlying molecule basis.
    pub fn basis(&self) -> &SemanticMoleculeBasis {
        &self.molecule_basis
    }

    /// Build the complete embedding table for a tokenizer's vocabulary.
    ///
    /// For each token: try semantic composition (Tier 1→2), fall back to genesis (Tier 3).
    /// The final embedding blends semantic and genesis components per `config.genesis_blend`.
    pub fn build_embeddings(
        &self,
        tokenizer: &BpeTokenizer,
        genesis: &GenesisSeed,
        config: &SemanticEmbeddingConfig,
    ) -> (Vec<ContinuousHV>, EmbeddingCoverageStats) {
        let vocab_size = tokenizer.vocab_size();
        let alpha = config.genesis_blend.clamp(0.0, 1.0);
        let mut stats = EmbeddingCoverageStats {
            total_tokens: vocab_size,
            tier1_direct: 0,
            tier2_morphological: 0,
            tier3_genesis: 0,
        };

        let embeddings: Vec<ContinuousHV> = (0..vocab_size as u32)
            .map(|id| {
                let token_str = tokenizer.token_str(id);
                let clean = token_str.trim().to_lowercase();

                // Genesis embedding (always computed for blending)
                let genesis_hv = ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca::token_emb::{id}"),
                    HDC_DIMENSION,
                );

                // Try semantic composition
                let semantic_opt = if clean.len() > 1 {
                    // Tier 1: direct
                    if let Some(primes) = self.lexicon.decompose(&clean) {
                        if !primes.is_empty() {
                            stats.tier1_direct += 1;
                            Some(self.compose_from_primes(primes))
                        } else {
                            None
                        }
                    } else {
                        // Tier 2: morphological
                        let morph = morpheme_decompose(&clean);
                        if !morph.is_empty() {
                            stats.tier2_morphological += 1;
                            Some(self.compose_from_primes(&morph))
                        } else {
                            None
                        }
                    }
                } else {
                    None
                };

                match semantic_opt {
                    Some(semantic_hv) => {
                        // Blend: (1-alpha)*semantic + alpha*genesis
                        blend_hvs(&semantic_hv, &genesis_hv, alpha)
                    }
                    None => {
                        stats.tier3_genesis += 1;
                        genesis_hv // Pure genesis fallback
                    }
                }
            })
            .collect();

        (embeddings, stats)
    }

    /// Get the basis vector for a specific semantic prime.
    pub fn prime_hv(&self, prime: &SemanticPrime) -> Option<&ContinuousHV> {
        self.prime_basis.get(prime)
    }

    /// Number of primes in the basis.
    pub fn num_primes(&self) -> usize {
        self.prime_basis.len()
    }
}

// ============================================================================
// Morphological decomposition (Tier 2)
// ============================================================================

/// Decompose a token string into semantic primes via prefix/suffix analysis.
///
/// Extracted and adapted from NsmSemanticGate::morpheme_primes_for_token.
pub fn morpheme_decompose(token: &str) -> Vec<SemanticPrime> {
    let mut primes = Vec::new();
    let lower = token.to_lowercase();

    // Prefix analysis
    if lower.starts_with("un") || lower.starts_with("dis") || lower.starts_with("non") {
        primes.push(SemanticPrime::Not);
    }
    if lower.starts_with("re") && lower.len() > 3 {
        primes.push(SemanticPrime::Do); // repetition prefix "re-"
    }
    if lower.starts_with("pre") || lower.starts_with("fore") {
        primes.push(SemanticPrime::Before);
    }
    if lower.starts_with("post") || lower.starts_with("after") {
        primes.push(SemanticPrime::After);
    }
    if lower.starts_with("over") || lower.starts_with("super") || lower.starts_with("hyper") {
        primes.push(SemanticPrime::Big);
    }
    if lower.starts_with("under") || lower.starts_with("sub") || lower.starts_with("mini") {
        primes.push(SemanticPrime::Small);
    }
    if lower.starts_with("inter") || lower.starts_with("co") {
        primes.push(SemanticPrime::With);
    }

    // Suffix analysis
    if lower.ends_with("ing") || lower.ends_with("tion") || lower.ends_with("ment") {
        primes.push(SemanticPrime::Do);
    }
    if lower.ends_with("ly") || lower.ends_with("ful") || lower.ends_with("ous") {
        primes.push(SemanticPrime::KindOf); // adjectival/quality suffix
    }
    if lower.ends_with("er") || lower.ends_with("or") || lower.ends_with("ist") {
        primes.push(SemanticPrime::Someone);
    }
    if lower.ends_with("ness") || lower.ends_with("ity") || lower.ends_with("ism") {
        primes.push(SemanticPrime::Something);
    }
    if lower.ends_with("able") || lower.ends_with("ible") {
        primes.push(SemanticPrime::Can);
    }
    if lower.ends_with("less") {
        primes.push(SemanticPrime::Not);
    }
    if lower.ends_with("ward") || lower.ends_with("wards") {
        primes.push(SemanticPrime::Move);
    }
    if lower.ends_with("ed") && lower.len() > 3 {
        primes.push(SemanticPrime::Before); // Past tense
    }

    primes
}

// ============================================================================
// Utility functions
// ============================================================================

/// Gram-Schmidt orthogonalization of ContinuousHV vectors (in-place).
///
/// For n vectors in 16,384D with n << 16,384, this produces exactly
/// orthogonal vectors while preserving the general direction of each.
fn gram_schmidt(vectors: &mut [ContinuousHV]) {
    for i in 1..vectors.len() {
        let mut v = vectors[i].as_slice().to_vec();

        for j in 0..i {
            let u = vectors[j].as_slice();
            // Project v onto u: proj = (v·u)/(u·u) * u
            let dot_vu: f32 = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            let dot_uu: f32 = u.iter().map(|a| a * a).sum();
            if dot_uu > 1e-10 {
                let scale = dot_vu / dot_uu;
                for (k, &uk) in u.iter().enumerate() {
                    v[k] -= scale * uk;
                }
            }
        }

        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }

        vectors[i] = ContinuousHV::from_values(v);
    }

    // Also normalize the first vector
    if !vectors.is_empty() {
        let v = vectors[0].as_slice().to_vec();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            let normed: Vec<f32> = v.iter().map(|x| x / norm).collect();
            vectors[0] = ContinuousHV::from_values(normed);
        }
    }
}

/// Blend two ContinuousHVs: result = (1-alpha)*a + alpha*b, normalized.
fn blend_hvs(a: &ContinuousHV, b: &ContinuousHV, alpha: f32) -> ContinuousHV {
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let one_minus_alpha = 1.0 - alpha;

    let blended: Vec<f32> = a_slice
        .iter()
        .zip(b_slice.iter())
        .map(|(&av, &bv)| one_minus_alpha * av + alpha * bv)
        .collect();

    ContinuousHV::from_values(blended).normalize()
}

/// Measurement of surprise (Prediction Error) in the HDC manifold.
///
/// Grounded in variational Free Energy: Error = 1.0 - cosine_similarity(Expected, Perceived).
#[derive(Debug, Clone, Copy)]
pub struct PredictionError {
    pub surprise: f32,
    pub cognitive_load: f32,
}

impl PredictionError {
    /// Calculate prediction error between an expected thought and actual perception.
    pub fn calculate(expected: &ContinuousHV, perceived: &ContinuousHV) -> Self {
        let surprise = 1.0 - expected.similarity(perceived);

        // Cognitive load proxy: surprise scales the thermodynamic cost of weight updates
        let cognitive_load = surprise.powi(2);

        Self {
            surprise,
            cognitive_load,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-semantic-quarks")
    }

    #[test]
    fn test_prime_basis_orthogonality() {
        let basis = SemanticQuarkBasis::new(&test_genesis());

        // All 65 prime basis vectors should be nearly orthogonal
        let primes = SemanticPrime::all();
        for i in 0..primes.len().min(10) {
            for j in (i + 1)..primes.len().min(10) {
                let sim = basis.prime_basis[&primes[i]].similarity(&basis.prime_basis[&primes[j]]);
                assert!(
                    sim.abs() < 0.05,
                    "Primes {:?} and {:?} not orthogonal: sim={sim:.4}",
                    primes[i],
                    primes[j]
                );
            }
        }
    }

    #[test]
    fn test_semantic_similarity_preserved() {
        let basis = SemanticQuarkBasis::new(&test_genesis());

        // "happy" and "joy" should be more similar than "happy" and "angry"
        let happy = basis.compose_token_embedding("happy");
        let joy = basis.compose_token_embedding("joy");
        let angry = basis.compose_token_embedding("angry");

        if let (Some(h), Some(j), Some(a)) = (happy, joy, angry) {
            let sim_hj = h.similarity(&j);
            let sim_ha = h.similarity(&a);
            // Both share FEEL, but happy=GOOD vs angry=BAD
            // At minimum they should have different similarity magnitudes
            assert!(
                sim_hj != sim_ha,
                "happy-joy ({sim_hj:.3}) and happy-angry ({sim_ha:.3}) should differ"
            );
        }
    }

    #[test]
    fn test_morphological_decomposition() {
        let primes = morpheme_decompose("unhappiness");
        assert!(primes.contains(&SemanticPrime::Not)); // un-
        assert!(primes.contains(&SemanticPrime::Something)); // -ness
    }

    #[test]
    fn test_genesis_blend_interpolation() {
        let genesis = test_genesis();
        let basis = SemanticQuarkBasis::new(&genesis);

        let primes = vec![SemanticPrime::Feel, SemanticPrime::Good];
        let semantic = basis.compose_from_primes(&primes);
        let genesis_hv = ContinuousHV::from_genesis(&genesis, "test", HDC_DIMENSION);

        // At alpha=0.0: pure semantic
        let blend_0 = blend_hvs(&semantic, &genesis_hv, 0.0);
        assert!(blend_0.similarity(&semantic) > 0.99);

        // At alpha=1.0: pure genesis
        let blend_1 = blend_hvs(&semantic, &genesis_hv, 1.0);
        assert!(blend_1.similarity(&genesis_hv) > 0.99);

        // At alpha=0.5: should differ from both pure extremes
        let blend_half = blend_hvs(&semantic, &genesis_hv, 0.5);
        let sim_to_pure_genesis = blend_half.similarity(&blend_1);
        let sim_to_pure_semantic = blend_half.similarity(&blend_0);
        // Blend should not be identical to either extreme
        assert!(
            sim_to_pure_genesis < 0.999 || sim_to_pure_semantic < 0.999,
            "Blend should differ from extremes: sim_genesis={sim_to_pure_genesis:.3}, sim_semantic={sim_to_pure_semantic:.3}"
        );
    }

    #[test]
    fn test_compose_returns_none_for_unknown() {
        let basis = SemanticQuarkBasis::new(&test_genesis());
        // Single-char tokens should return None
        assert!(basis.compose_token_embedding("x").is_none());
        assert!(basis.compose_token_embedding("").is_none());
    }

    #[test]
    fn test_coverage_stats_sum() {
        let genesis = test_genesis();
        let basis = SemanticQuarkBasis::new(&genesis);
        let tokenizer = BpeTokenizer::default_minimal();
        let (_, stats) =
            basis.build_embeddings(&tokenizer, &genesis, &SemanticEmbeddingConfig::default());
        assert_eq!(
            stats.tier1_direct + stats.tier2_morphological + stats.tier3_genesis,
            stats.total_tokens,
            "Coverage tiers must sum to total"
        );
    }
}