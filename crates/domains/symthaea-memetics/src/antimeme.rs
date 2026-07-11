// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Anti-memetics — self-concealing information (plan Phase 4).
//!
//! **This is a research model, not a capability.** An anti-meme in the SCP sense
//! is information that resists its own retention, recall, and detection. You
//! cannot make a hypervector physically invisible to cosine similarity — the
//! representation is what it is — so this module does **not** claim to hide real
//! information. Instead it models an *agent* whose retention and recall are
//! biased against a target pattern: the "perception gap" is a modeled parameter,
//! a simulation of the SCP concept for studying blind spots, not a physical
//! property of the data.
//!
//! Two modeled effects of an [`AntiMeme`] keyed to a `target` pattern:
//!
//! 1. **Accelerated forgetting** — memes resonant with the target decay out of
//!    retention faster (the negative-salience analog of the dream-consolidation
//!    boost real threat memories get).
//! 2. **Perception gap** — when the agent tries to *recall* the target, the
//!    similarity it perceives is depressed, so it "can't find" what it once
//!    knew. Again: the stored vector is unchanged; only the agent's perceived
//!    recall is suppressed.

use crate::propagation::resonance_gain;
use symthaea_core::hdc::binary_hv::BinaryHV;

/// A self-concealing pattern: it suppresses retention and recall of memes that
/// resonate with its `target`.
#[derive(Debug, Clone)]
pub struct AntiMeme {
    /// The pattern this anti-meme conceals.
    pub target: BinaryHV,
    /// Concealment strength in `[0, 1]` — how strongly it suppresses
    /// retention/recall of target-resonant patterns.
    pub strength: f32,
}

impl AntiMeme {
    pub fn new(target: BinaryHV, strength: f32) -> Self {
        Self {
            target,
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// How strongly this anti-meme suppresses a pattern `p` (0..1): its strength
    /// scaled by how well `p` resonates with the concealment target. A pattern
    /// unrelated to the target is untouched.
    fn suppression_of(&self, p: &BinaryHV) -> f32 {
        (self.strength * resonance_gain(p.similarity(&self.target))).clamp(0.0, 1.0)
    }
}

/// A field of active anti-memes affecting one agent's memory.
#[derive(Debug, Clone, Default)]
pub struct AntiMemeField {
    anti_memes: Vec<AntiMeme>,
}

impl AntiMemeField {
    pub fn new() -> Self {
        Self::default()
    }

    /// Expose the agent to an anti-meme.
    pub fn add(&mut self, anti_meme: AntiMeme) {
        self.anti_memes.push(anti_meme);
    }

    pub fn len(&self) -> usize {
        self.anti_memes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.anti_memes.is_empty()
    }

    /// Aggregate suppression on a pattern (0..1): the strongest concealment any
    /// active anti-meme applies to it.
    fn suppression(&self, p: &BinaryHV) -> f32 {
        self.anti_memes
            .iter()
            .map(|a| a.suppression_of(p))
            .fold(0.0f32, f32::max)
    }

    /// Effective per-tick retention decay for a stored meme with payload `p`.
    ///
    /// `base_decay` is the ordinary forgetting rate; anti-memes ADD to it for
    /// target-resonant memes, so those fade faster. Result clamped to `[0, 1]`.
    pub fn effective_decay(&self, p: &BinaryHV, base_decay: f32) -> f32 {
        (base_decay + self.suppression(p)).clamp(0.0, 1.0)
    }

    /// Perceived recall similarity between a `query` and a `stored` pattern.
    ///
    /// Returns the true Hamming similarity **depressed** by suppression keyed to
    /// the *query* — i.e. when the agent reaches for a concealed pattern, what it
    /// perceives is weakened, so it fails to recognize what it holds. With no
    /// anti-memes (or an unrelated query) this equals the true similarity.
    ///
    /// The stored vector is never modified; this is the modeled perception gap.
    pub fn perceived_recall(&self, query: &BinaryHV, stored: &BinaryHV) -> f32 {
        let truth = query.similarity(stored).clamp(0.0, 1.0);
        truth * (1.0 - self.suppression(query))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_anti_memes_is_transparent() {
        let field = AntiMemeField::new();
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        // Recall unchanged, decay unchanged.
        assert_eq!(
            field.perceived_recall(&a, &a),
            a.similarity(&a).clamp(0.0, 1.0)
        );
        assert_eq!(field.effective_decay(&a, 0.1), 0.1);
        assert!(field.perceived_recall(&a, &b) >= 0.0);
    }

    #[test]
    fn anti_meme_accelerates_target_forgetting() {
        let target = BinaryHV::random(10);
        let mut field = AntiMemeField::new();
        field.add(AntiMeme::new(target.clone(), 0.4));

        // A target-resonant meme decays faster than the baseline...
        let resonant = target.add_noise(0.08, 5); // ~high similarity to target
        let unrelated = BinaryHV::random(99); // ~chance similarity to target

        let base = 0.1f32;
        let decay_resonant = field.effective_decay(&resonant, base);
        let decay_unrelated = field.effective_decay(&unrelated, base);

        assert!(
            decay_resonant > decay_unrelated,
            "target-resonant meme should fade faster: {decay_resonant} > {decay_unrelated}"
        );
        assert!(
            (decay_unrelated - base).abs() < 0.05,
            "unrelated meme decay should be near baseline, got {decay_unrelated}"
        );
    }

    #[test]
    fn anti_meme_creates_perception_gap() {
        let target = BinaryHV::random(7);
        let mut field = AntiMemeField::new();
        field.add(AntiMeme::new(target.clone(), 0.8));

        // Reaching FOR the target: perceived recall is depressed vs the truth.
        let truth = target.similarity(&target).clamp(0.0, 1.0); // 1.0
        let perceived = field.perceived_recall(&target, &target);
        assert!(
            perceived < truth * 0.7,
            "the concealed target should be hard to recall: perceived {perceived} vs truth {truth}"
        );

        // Reaching for an UNRELATED pattern is unaffected (no blanket amnesia).
        let other = BinaryHV::random(123);
        let p_other = field.perceived_recall(&other, &other);
        assert!(
            (p_other - other.similarity(&other).clamp(0.0, 1.0)).abs() < 0.05,
            "unrelated recall must be intact, got {p_other}"
        );
    }

    #[test]
    fn stronger_anti_meme_conceals_more() {
        let target = BinaryHV::random(3);
        let weak = {
            let mut f = AntiMemeField::new();
            f.add(AntiMeme::new(target.clone(), 0.2));
            f.perceived_recall(&target, &target)
        };
        let strong = {
            let mut f = AntiMemeField::new();
            f.add(AntiMeme::new(target.clone(), 0.9));
            f.perceived_recall(&target, &target)
        };
        assert!(
            strong < weak,
            "a stronger anti-meme should conceal more: strong {strong} < weak {weak}"
        );
    }

    #[test]
    fn strength_is_clamped() {
        let a = AntiMeme::new(BinaryHV::random(1), 5.0);
        assert_eq!(a.strength, 1.0);
    }
}
