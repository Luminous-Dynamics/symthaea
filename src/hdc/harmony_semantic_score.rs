// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic (HDC-projection) Eight Harmonies scoring.
//!
//! Phase 3.3 of the 2026-07-06 art/culture review
//! (`ART_CULTURE_REVIEW_AND_PLAN_2026-07-06.md`, "replacing/augmenting
//! `symthaea-harmonies::evaluate_harmony`'s keyword-matching scorer with a
//! semantic version backed by the existing HDC harmony basis, so alignment
//! survives paraphrase"). The review flagged that
//! `symthaea_harmonies::EightHarmonies::evaluate` (and its private per-harmony
//! `evaluate_harmony` keyword matcher) is a wordlist scorer — it cannot
//! recognize a paraphrase that doesn't reuse the exact keyword vocabulary
//! ("we act as one" scores zero on `ResonantCoherence` because "one" isn't a
//! listed keyword, even though "unify"/"coherent" are). This module adds a
//! semantic counterpart that projects an already-encoded HDC hypervector onto
//! the [`HarmonyBasis`] built in `harmony_basis.rs`, so alignment scoring can
//! survive paraphrase — semantic similarity in HDC space rather than literal
//! keyword overlap.
//!
//! ## Why this lives in the root crate, not `symthaea-harmonies`
//!
//! `symthaea-harmonies` depends only on `symthaea-core` (for the
//! `ContinuousHV` *type*) and `symthaea-types` (for `Harmony`/`N_HARMONIES`)
//! — it does not, and must not, depend on the root `symthaea` crate, because
//! the root crate already depends on `symthaea-harmonies`
//! (`src/consciousness/values/eight_harmonies.rs` re-exports it, and the
//! root `Cargo.toml` lists it as a normal path dependency). `HarmonyBasis`
//! and its `TextHdcEncoder`-based construction live in the root crate's
//! `src/hdc/harmony_basis.rs`, not in `symthaea-core`. Adding a
//! `HarmonyBasis`-typed parameter to a function inside `symthaea-harmonies`
//! would therefore require `symthaea-harmonies -> symthaea (root) ->
//! symthaea-harmonies`, a workspace dependency cycle. Keeping the bridge
//! here (the root crate already depends on both `HarmonyBasis` locally and
//! `symthaea_harmonies` as a normal dependency) avoids that cycle entirely,
//! at the cost of the semantic scorer not being directly callable from
//! other domain crates that depend on `symthaea-harmonies` alone without
//! also depending on the root crate. If that becomes a real need, the fix
//! is to move `HarmonyBasis`/`TextHdcEncoder` down into `symthaea-core` (or
//! a new small `symthaea-harmony-basis` crate) rather than pulling the root
//! crate down into `symthaea-harmonies`.
//!
//! ## Keyword vs. semantic tradeoff
//!
//! - **Keyword** (`symthaea_harmonies::EightHarmonies::evaluate` /
//!   `evaluate_harmony`): works standalone on raw text with no HDC
//!   dependency; fast; fully explainable (evidence = matched words);
//!   brittle to paraphrase and vocabulary drift.
//! - **Semantic** (this module): robust to paraphrase and vocabulary drift
//!   because scoring is cosine similarity in HDC space; but it is only as
//!   good as (a) the `HarmonyBasis` vectors — which, by default
//!   (`HarmonyBasis::new`), are *also* built from a fixed keyword list
//!   (`HARMONY_KEYWORDS`) via the lexical `TextHdcEncoder`, so they inherit
//!   some of the same vocabulary bias unless `HarmonyBasis::with_dense_vectors`
//!   is used with a real contextual embedder — and (b) whatever text->HV
//!   encoder produced `text_hv`. A bad or narrow encoder can silently
//!   produce plausible-looking but wrong scores, with no keyword evidence
//!   trail to sanity-check against (unlike the keyword scorer's `evidence`
//!   field).

use symthaea_core::hdc::ContinuousHV;
use symthaea_harmonies::{AlignmentResult, Harmony, HarmonyAlignment};

use super::harmony_basis::HarmonyBasis;

/// Project a single already-encoded HDC hypervector onto the Eight
/// Harmonies basis and package the result in the same [`AlignmentResult`]
/// shape that `EightHarmonies::evaluate` produces, so callers can swap
/// scorers without touching downstream code (courage-override, veto
/// gating, etc. all key off `AlignmentResult`/`HarmonyAlignment`).
///
/// Confidence is a simple heuristic — `|cosine similarity|` — under the
/// assumption that a projection far from zero (strongly positive OR
/// strongly negative) is a more confident read than a near-zero projection
/// (ambiguous / off-topic). This is not calibrated against ground truth;
/// treat it as a coarse proxy, not a probability.
pub fn evaluate_harmony_semantic(text_hv: &ContinuousHV, basis: &HarmonyBasis) -> AlignmentResult {
    let coords = basis.project(text_hv);
    let alignments: Vec<HarmonyAlignment> = Harmony::all()
        .into_iter()
        .zip(coords.iter())
        .map(|(harmony, &score)| {
            let confidence = score.abs().clamp(0.0, 1.0);
            HarmonyAlignment::new(harmony, score, confidence).with_explanation(format!(
                "semantic projection: cosine similarity {score:.3} to {harmony:?} basis vector"
            ))
        })
        .collect();
    AlignmentResult::from_alignments(alignments)
}

/// Convenience wrapper: encode `text` via any text->HV closure, then run
/// [`evaluate_harmony_semantic`]. Returns `None` (never panics, never
/// fabricates a score) if the encoder fails to produce a vector — e.g. a
/// remote embedding-model call that times out, or a model that isn't
/// loaded.
///
/// Decouples the harmony math from any one specific text encoder,
/// mirroring the `WordEmbedder` trait abstraction from Phase 2 of this
/// review.
///
/// Real encoders to plug in, cheapest/most-consistent first:
/// - [`super::moral_text_encoder::TextHdcEncoder::encode`] — deterministic,
///   offline, no model download; this is the *same* encoder used to build
///   `HARMONY_KEYWORDS` into basis vectors, so scenario and basis embeddings
///   live in the identical space (no domain mismatch). Lightweight, but
///   still substantially keyword/character-trigram-flavored rather than
///   truly distributional — see the module-level tradeoff note.
/// - `symthaea_embeddings::HdcBridge::project_continuous` (fed by a real
///   dense embedder such as Qwen3/BGE-M3) — captures genuine contextual
///   paraphrase invariance but requires a loaded/downloaded embedding
///   model upstream to produce the dense vector in the first place; pair
///   with `HarmonyBasis::with_dense_vectors` so basis and scenario vectors
///   share the same dense embedding space.
///
/// ```ignore
/// use symthaea_core::hdc::ContinuousHV;
/// use symthaea::hdc::harmony_basis::HarmonyBasis;
/// use symthaea::hdc::moral_text_encoder::TextHdcEncoder;
/// use symthaea::hdc::harmony_semantic_score::evaluate_harmony_semantic_from_text;
///
/// let dim = 2048;
/// let encoder = TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2);
/// let basis = HarmonyBasis::new(dim);
/// let mut text_to_hv = |t: &str| -> Option<ContinuousHV> { Some(encoder.encode(t)) };
/// let result = evaluate_harmony_semantic_from_text(
///     "we are one, together", &mut text_to_hv, &basis,
/// );
/// assert!(result.is_some());
/// ```
pub fn evaluate_harmony_semantic_from_text(
    text: &str,
    encoder: &mut impl FnMut(&str) -> Option<ContinuousHV>,
    basis: &HarmonyBasis,
) -> Option<AlignmentResult> {
    let hv = encoder(text)?;
    Some(evaluate_harmony_semantic(&hv, basis))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::moral_text_encoder::TextHdcEncoder;

    /// (a) Projection math correctness: a synthetic HV built to be *exactly*
    /// the ResonantCoherence basis vector should score maximally (cosine
    /// similarity 1.0, within float tolerance) on ResonantCoherence and
    /// strictly lower on every other harmony. This validates
    /// `evaluate_harmony_semantic`'s use of `HarmonyBasis::project`, not
    /// real-world semantics (no text encoder involved).
    #[test]
    fn test_synthetic_hv_scores_highest_on_its_own_basis_axis() {
        let dim = 512;
        let basis = HarmonyBasis::new(dim);
        let resonant_idx = 0; // ResonantCoherence is index 0 in Harmony::all()
        assert_eq!(Harmony::all()[resonant_idx], Harmony::ResonantCoherence);

        let synthetic_hv = basis.vectors[resonant_idx].clone();
        let result = evaluate_harmony_semantic(&synthetic_hv, &basis);

        let resonant_alignment = result
            .get(Harmony::ResonantCoherence)
            .expect("ResonantCoherence must be present in a full 8-harmony result");

        // Self-similarity must be (approximately) 1.0.
        assert!(
            (resonant_alignment.score - 1.0).abs() < 1e-4,
            "expected ~1.0 self-similarity, got {}",
            resonant_alignment.score
        );

        // ResonantCoherence must be the single highest-scoring harmony.
        let (top_harmony, top_alignment) = result
            .most_aligned()
            .expect("8 alignments were just inserted");
        assert_eq!(
            *top_harmony,
            Harmony::ResonantCoherence,
            "synthetic HV built from the ResonantCoherence basis vector should score \
             highest on ResonantCoherence, but {top_harmony:?} scored higher \
             ({} vs {})",
            top_alignment.score,
            resonant_alignment.score
        );

        // Every other harmony must score strictly lower than the self-similarity.
        for alignment in result.harmonies() {
            if alignment.harmony != Harmony::ResonantCoherence {
                assert!(
                    alignment.score < resonant_alignment.score,
                    "{:?} scored {} which is not lower than ResonantCoherence's {}",
                    alignment.harmony,
                    alignment.score,
                    resonant_alignment.score
                );
            }
        }

        // All 8 harmonies must be present with finite, in-range scores.
        assert_eq!(result.alignments.len(), 8);
        for alignment in result.harmonies() {
            assert!(alignment.score.is_finite());
            assert!((-1.0..=1.0).contains(&alignment.score));
            assert!((0.0..=1.0).contains(&alignment.confidence));
        }
    }

    /// (b) Closure-based API: encoder success threads through to `Some`,
    /// and its content matches calling `evaluate_harmony_semantic` directly
    /// on the same HV (no silent divergence between the two entry points).
    #[test]
    fn test_from_text_threads_encoder_success() {
        let dim = 256;
        let basis = HarmonyBasis::new(dim);
        let hv = ContinuousHV::random(dim, 12345);
        let hv_for_closure = hv.clone();

        let mut encoder =
            move |_text: &str| -> Option<ContinuousHV> { Some(hv_for_closure.clone()) };

        let direct = evaluate_harmony_semantic(&hv, &basis);
        let via_closure = evaluate_harmony_semantic_from_text("anything", &mut encoder, &basis)
            .expect("encoder returned Some, so this must be Some");

        assert_eq!(direct.overall_score, via_closure.overall_score);
        for harmony in Harmony::all() {
            let a = direct.get(harmony).unwrap();
            let b = via_closure.get(harmony).unwrap();
            assert!((a.score - b.score).abs() < 1e-9);
        }
    }

    /// (b) Closure-based API: encoder failure (e.g. a remote embedder that
    /// timed out, or a model that isn't loaded) must propagate as `None`,
    /// never a panic and never a fabricated/default score.
    #[test]
    fn test_from_text_threads_encoder_failure_as_none() {
        let dim = 256;
        let basis = HarmonyBasis::new(dim);

        let mut always_fails = |_text: &str| -> Option<ContinuousHV> { None };

        let result = evaluate_harmony_semantic_from_text("anything", &mut always_fails, &basis);
        assert!(
            result.is_none(),
            "encoder failure must propagate as None, not a fabricated AlignmentResult"
        );
    }

    /// (c) Paraphrase invariance demonstration — the actual point of Phase
    /// 3.3. IGNORED by default: the only encoder available in this crate
    /// without a network call / model download is `TextHdcEncoder`, which
    /// is a character-trigram + word-hash lexical encoder (see
    /// `moral_text_encoder.rs`), not a real distributional/contextual
    /// embedder. It captures morphological variation ("integrate" vs
    /// "integration") but NOT vocabulary-disjoint paraphrase ("we are one,
    /// together" sharing no words or trigrams with "unify"/"coherent"), so
    /// this test is not a reliable CI gate with today's encoder — it
    /// documents the *intended* validation for Phase 3.3 once a real dense
    /// embedder is wired via `HarmonyBasis::with_dense_vectors` +
    /// `symthaea_embeddings::HdcBridge::project_continuous` (which itself
    /// requires an upstream embedding model call — network/model download,
    /// unavailable in this offline environment). Run manually with
    /// `cargo test -- --ignored` to inspect actual coordinates once that
    /// wiring lands.
    #[test]
    #[ignore = "requires a real distributional text encoder (network/model download); \
                TextHdcEncoder is lexical and not expected to generalize across \
                vocabulary-disjoint paraphrases — see doc comment"]
    fn test_paraphrase_invariance_resonant_coherence() {
        let dim = 2048;
        let basis = HarmonyBasis::new(dim);
        let encoder = TextHdcEncoder::with_sentiment(dim, 3, 0.5, 0.2);
        let mut text_to_hv = |t: &str| -> Option<ContinuousHV> { Some(encoder.encode(t)) };

        let a =
            evaluate_harmony_semantic_from_text("we are one, together", &mut text_to_hv, &basis)
                .expect("encoder always succeeds here");
        let b = evaluate_harmony_semantic_from_text(
            "unity and collective wholeness",
            &mut text_to_hv,
            &basis,
        )
        .expect("encoder always succeeds here");

        let a_score = a.get(Harmony::ResonantCoherence).unwrap().score;
        let b_score = b.get(Harmony::ResonantCoherence).unwrap().score;

        assert!(
            a_score > 0.0 && b_score > 0.0,
            "both paraphrases of 'unity' should score positively on ResonantCoherence \
             (got a={a_score}, b={b_score})"
        );
    }
}
