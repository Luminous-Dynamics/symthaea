// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Theory of Mind — perspective-taking and empathy for the Spore WASM kernel.
//!
//! Port of the desktop SocialManager / HumanPartnerModel subsystem into a
//! lightweight, WASM-compatible module. Tracks mental models of interaction
//! partners, predicts their responses, and modulates the Spore's own
//! neuromodulators based on empathic resonance.
//!
//! Fully WASM-compatible: no std::time, no filesystem, no tokio.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of partner models (Dunbar-like cognitive limit).
const DEFAULT_MAX_PARTNERS: usize = 8;

/// Trust EMA blending factor for incremental updates.
const TRUST_EMA_ALPHA: f32 = 0.15;

/// Prediction accuracy EMA blending factor.
const ACCURACY_EMA_ALPHA: f32 = 0.2;

/// Empathy neuromod scaling (oxytocin boost per unit of partner distress).
const EMPATHY_OXYTOCIN_SCALE: f32 = 0.04;

/// Empathy neuromod scaling (NE boost per unit of partner arousal).
const EMPATHY_NE_SCALE: f32 = 0.02;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A model of another agent's mental state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartnerModel {
    /// Stable identifier (screen name, handle, etc.).
    pub name: String,
    /// What we think their consciousness level is (0-1).
    pub believed_consciousness: f32,
    /// What we think they feel: -1 (negative) to +1 (positive).
    pub believed_valence: f32,
    /// What we think they want.
    pub believed_intent: String,
    /// How good our model is (0-1).
    pub prediction_accuracy: f32,
    /// Total interaction count.
    pub interaction_count: u32,
    /// Trust in this partner (0-1).
    pub trust: f32,
    /// Arousal level we attribute to them (0-1).
    pub believed_arousal: f32,
}

impl PartnerModel {
    /// Create a new partner model with conservative defaults.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            believed_consciousness: 0.5,
            believed_valence: 0.0,
            believed_intent: "unknown".into(),
            prediction_accuracy: 0.5,
            interaction_count: 0,
            trust: 0.3, // start cautiously
            believed_arousal: 0.3,
        }
    }
}

/// Summary of social state for WASM/JSON export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialMindState {
    pub partner_count: usize,
    pub empathy_level: f32,
    pub social_coherence: f32,
    pub partners: Vec<PartnerSummary>,
}

/// Lightweight partner info for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartnerSummary {
    pub name: String,
    pub trust: f32,
    pub believed_valence: f32,
    pub believed_consciousness: f32,
    pub prediction_accuracy: f32,
    pub interaction_count: u32,
}

/// Empathic modulation signals to apply to the Spore's neuromodulators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpathicModulation {
    /// Oxytocin delta (social bonding).
    pub oxytocin_delta: f32,
    /// Norepinephrine delta (shared arousal).
    pub norepinephrine_delta: f32,
    /// Serotonin delta (shared contentment).
    pub serotonin_delta: f32,
    /// Dopamine delta (shared excitement).
    pub dopamine_delta: f32,
}

// ---------------------------------------------------------------------------
// SocialMind
// ---------------------------------------------------------------------------

/// Social Theory of Mind — perspective-taking and empathy.
///
/// Tracks mental models of interaction partners, predicts their responses,
/// and computes empathic modulation of the Spore's own neuromodulators.
pub struct SocialMind {
    partners: Vec<PartnerModel>,
    max_partners: usize,
    /// How much others' states affect us (0-1).
    empathy_level: f32,
    /// Overall social alignment across all partners (0-1).
    social_coherence: f32,
}

impl SocialMind {
    /// Create a new SocialMind with default settings.
    pub fn new() -> Self {
        Self {
            partners: Vec::with_capacity(DEFAULT_MAX_PARTNERS),
            max_partners: DEFAULT_MAX_PARTNERS,
            empathy_level: 0.5,
            social_coherence: 0.0,
        }
    }

    /// Create a SocialMind with custom capacity and empathy.
    pub fn with_config(max_partners: usize, empathy_level: f32) -> Self {
        Self {
            partners: Vec::with_capacity(max_partners),
            max_partners: max_partners.max(1),
            empathy_level: empathy_level.clamp(0.0, 1.0),
            social_coherence: 0.0,
        }
    }

    /// Observe a partner's input and update their mental model.
    ///
    /// Infers emotional valence and arousal from simple text heuristics
    /// (keyword matching). In the full desktop CLS this would use HDC encoding;
    /// here we keep it lightweight for WASM.
    pub fn observe_partner(&mut self, name: &str, input_text: &str) {
        let partner = self.find_or_create(name);
        partner.interaction_count = partner.interaction_count.saturating_add(1);

        // Simple sentiment heuristic (WASM-compatible, no NLP crate needed)
        let lower = input_text.to_lowercase();
        let positive_words = [
            "happy",
            "great",
            "love",
            "good",
            "excellent",
            "wonderful",
            "yes",
            "thank",
        ];
        let negative_words = [
            "sad", "bad", "angry", "hate", "terrible", "no", "wrong", "fear",
        ];
        let arousal_words = ["urgent", "excited", "panic", "amazing", "critical", "hurry"];

        let pos_count = positive_words.iter().filter(|w| lower.contains(*w)).count() as f32;
        let neg_count = negative_words.iter().filter(|w| lower.contains(*w)).count() as f32;
        let arousal_count = arousal_words.iter().filter(|w| lower.contains(*w)).count() as f32;

        let total = (pos_count + neg_count).max(1.0);
        let raw_valence = (pos_count - neg_count) / total;

        // EMA update of believed valence
        partner.believed_valence = partner.believed_valence * 0.7 + raw_valence * 0.3;
        partner.believed_valence = partner.believed_valence.clamp(-1.0, 1.0);

        // Arousal update
        let raw_arousal = (arousal_count * 0.3).clamp(0.0, 1.0);
        partner.believed_arousal = partner.believed_arousal * 0.7 + raw_arousal * 0.3;

        // Trust grows slowly with interaction
        partner.trust = (partner.trust + TRUST_EMA_ALPHA * 0.1).clamp(0.0, 1.0);

        // Infer intent from content length and keywords
        if lower.contains("help") || lower.contains("question") || lower.contains("?") {
            partner.believed_intent = "seeking_help".into();
        } else if lower.contains("tell") || lower.contains("explain") || lower.contains("teach") {
            partner.believed_intent = "sharing_knowledge".into();
        } else if lower.contains("feel") || lower.contains("emotion") {
            partner.believed_intent = "emotional_expression".into();
        } else {
            partner.believed_intent = "conversing".into();
        }

        // Recalculate social coherence
        self.recompute_coherence();
    }

    /// Predict what a partner would say/feel given their current model.
    pub fn predict_response(&self, name: &str) -> Option<String> {
        let partner = self.partners.iter().find(|p| p.name == name)?;

        let mood = if partner.believed_valence > 0.3 {
            "positive"
        } else if partner.believed_valence < -0.3 {
            "negative"
        } else {
            "neutral"
        };

        let engagement = if partner.believed_arousal > 0.5 {
            "highly engaged"
        } else {
            "calm"
        };

        Some(format!(
            "Partner '{}' likely feels {} and is {}. Intent: {}. Trust: {:.2}",
            partner.name, mood, engagement, partner.believed_intent, partner.trust
        ))
    }

    /// Compute how partners' states should modulate our neuromodulators.
    ///
    /// Empathic resonance: we partially mirror our partners' emotional states.
    /// High empathy_level amplifies this effect.
    pub fn empathic_modulation(&self) -> EmpathicModulation {
        if self.partners.is_empty() {
            return EmpathicModulation {
                oxytocin_delta: 0.0,
                norepinephrine_delta: 0.0,
                serotonin_delta: 0.0,
                dopamine_delta: 0.0,
            };
        }

        let n = self.partners.len() as f32;
        let avg_valence: f32 = self
            .partners
            .iter()
            .map(|p| p.believed_valence)
            .sum::<f32>()
            / n;
        let avg_arousal: f32 = self
            .partners
            .iter()
            .map(|p| p.believed_arousal)
            .sum::<f32>()
            / n;
        let avg_trust: f32 = self.partners.iter().map(|p| p.trust).sum::<f32>() / n;

        let empathy = self.empathy_level;

        EmpathicModulation {
            // Social bonding increases with trust and positive valence
            oxytocin_delta: EMPATHY_OXYTOCIN_SCALE
                * empathy
                * avg_trust
                * (0.5 + avg_valence * 0.5),
            // Shared arousal from partners
            norepinephrine_delta: EMPATHY_NE_SCALE * empathy * avg_arousal,
            // Contentment from positive social environment
            serotonin_delta: 0.02 * empathy * avg_valence.max(0.0),
            // Excitement from high-arousal, positive interactions
            dopamine_delta: 0.02 * empathy * (avg_arousal * avg_valence.max(0.0)),
        }
    }

    /// Aggregate social alignment across all partners (0-1).
    pub fn social_coherence(&self) -> f32 {
        self.social_coherence
    }

    /// Current empathy level (0-1).
    pub fn empathy_level(&self) -> f32 {
        self.empathy_level
    }

    /// Number of tracked partners.
    pub fn partner_count(&self) -> usize {
        self.partners.len()
    }

    /// Simulate seeing from a partner's viewpoint.
    ///
    /// Returns a perspective-taking score (0-1) based on how well we can
    /// model their state. Requires sufficient interactions for accuracy.
    pub fn perspective_take(&self, name: &str) -> Option<f32> {
        let partner = self.partners.iter().find(|p| p.name == name)?;

        // Perspective-taking quality depends on:
        // 1. Number of interactions (familiarity)
        // 2. Prediction accuracy (model quality)
        // 3. Trust (openness to understanding)
        let familiarity = (partner.interaction_count as f32 / 20.0).clamp(0.0, 1.0);
        let quality = familiarity * 0.3 + partner.prediction_accuracy * 0.4 + partner.trust * 0.3;

        Some(quality.clamp(0.0, 1.0))
    }

    /// Update prediction accuracy for a partner (call when we can verify
    /// our prediction against their actual response).
    pub fn update_accuracy(&mut self, name: &str, was_correct: bool) {
        if let Some(partner) = self.partners.iter_mut().find(|p| p.name == name) {
            let signal = if was_correct { 1.0 } else { 0.0 };
            partner.prediction_accuracy = partner.prediction_accuracy * (1.0 - ACCURACY_EMA_ALPHA)
                + signal * ACCURACY_EMA_ALPHA;
        }
    }

    /// Export full state for WASM/JSON serialization.
    pub fn state(&self) -> SocialMindState {
        SocialMindState {
            partner_count: self.partners.len(),
            empathy_level: self.empathy_level,
            social_coherence: self.social_coherence,
            partners: self
                .partners
                .iter()
                .map(|p| PartnerSummary {
                    name: p.name.clone(),
                    trust: p.trust,
                    believed_valence: p.believed_valence,
                    believed_consciousness: p.believed_consciousness,
                    prediction_accuracy: p.prediction_accuracy,
                    interaction_count: p.interaction_count,
                })
                .collect(),
        }
    }

    // ── Internal ──────────────────────────────────────────────────────

    /// Find an existing partner or create a new one.
    /// If at capacity, evicts the least-interacted partner.
    fn find_or_create(&mut self, name: &str) -> &mut PartnerModel {
        // Check if partner already exists
        if let Some(idx) = self.partners.iter().position(|p| p.name == name) {
            return &mut self.partners[idx];
        }

        // Evict if at capacity (remove least-interacted)
        if self.partners.len() >= self.max_partners {
            let min_idx = self
                .partners
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| p.interaction_count)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.partners.remove(min_idx);
        }

        self.partners.push(PartnerModel::new(name));
        self.partners.last_mut().unwrap()
    }

    /// Recompute social coherence as the average trust weighted by interaction count.
    fn recompute_coherence(&mut self) {
        if self.partners.is_empty() {
            self.social_coherence = 0.0;
            return;
        }

        let total_weight: f32 = self
            .partners
            .iter()
            .map(|p| (p.interaction_count as f32).max(1.0))
            .sum();
        let weighted_trust: f32 = self
            .partners
            .iter()
            .map(|p| p.trust * (p.interaction_count as f32).max(1.0))
            .sum();

        self.social_coherence = (weighted_trust / total_weight).clamp(0.0, 1.0);
    }
}

impl Default for SocialMind {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observe_and_predict() {
        let mut mind = SocialMind::new();
        mind.observe_partner("alice", "I feel happy and great today!");
        mind.observe_partner("alice", "This is wonderful, thank you!");

        let prediction = mind.predict_response("alice");
        assert!(prediction.is_some());
        let text = prediction.unwrap();
        assert!(text.contains("positive"));

        let alice = &mind.partners[0];
        assert_eq!(alice.interaction_count, 2);
        assert!(alice.believed_valence > 0.0);
    }

    #[test]
    fn test_empathic_modulation() {
        let mut mind = SocialMind::with_config(4, 0.8);
        mind.observe_partner("bob", "I am so happy and excited!");
        mind.observe_partner("carol", "Everything is great and wonderful!");

        let mods = mind.empathic_modulation();
        assert!(
            mods.oxytocin_delta > 0.0,
            "positive partners should boost oxytocin"
        );
        assert!(
            mods.serotonin_delta >= 0.0,
            "positive valence should not decrease serotonin"
        );
    }

    #[test]
    fn test_social_coherence_and_perspective() {
        let mut mind = SocialMind::new();
        // Multiple interactions to build familiarity
        for _ in 0..10 {
            mind.observe_partner("dave", "good conversation, thank you");
        }

        assert!(mind.social_coherence() > 0.0);

        let pt = mind.perspective_take("dave");
        assert!(pt.is_some());
        assert!(pt.unwrap() > 0.0);

        // Unknown partner
        assert!(mind.perspective_take("unknown").is_none());
    }

    #[test]
    fn test_partner_eviction() {
        let mut mind = SocialMind::with_config(2, 0.5);
        mind.observe_partner("a", "hello");
        mind.observe_partner("b", "hello");
        // Interact more with 'b' so 'a' gets evicted
        mind.observe_partner("b", "hello again");
        mind.observe_partner("b", "still here");

        // Adding a third should evict the least-interacted (a)
        mind.observe_partner("c", "new partner");
        assert_eq!(mind.partner_count(), 2);
        assert!(mind.partners.iter().any(|p| p.name == "b"));
        assert!(mind.partners.iter().any(|p| p.name == "c"));
    }

    #[test]
    fn test_accuracy_update() {
        let mut mind = SocialMind::new();
        mind.observe_partner("eve", "testing");
        mind.update_accuracy("eve", true);
        mind.update_accuracy("eve", true);
        let acc = mind.partners[0].prediction_accuracy;
        assert!(acc > 0.5, "correct predictions should raise accuracy");
    }
}
