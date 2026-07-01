// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broca + LiteRT Fusion Engine.
//!
//! Sequential strategy: Broca generates consciousness-gated epistemic skeleton,
//! LiteRT fills knowledge gaps guided by the skeleton. Produces output that has
//! both consciousness fidelity (Broca) and factual capability (Gemma 4).
//!
//! Routing:
//!   consciousness < 0.15       → Fallback (no generation)
//!   Broca unavailable          → LiteRTWithContext
//!   Broca high conf + coherent → BrocaOnly
//!   Broca has knowledge gaps   → Sequential (Broca + LiteRT merge)

use serde::{Deserialize, Serialize};
use symthaea_broca::GenerationResult;

use crate::litert_bridge::LiteRTBridge;

/// Minimum consciousness level for any generation.
const CONSCIOUSNESS_THRESHOLD: f32 = 0.15;
/// Minimum Broca confidence to trust skeleton without LiteRT.
const CONFIDENCE_THRESHOLD: f32 = 0.7;
/// Minimum Broca coherence to trust skeleton without LiteRT.
const COHERENCE_THRESHOLD: f32 = 0.6;
/// Minimum Broca output length (chars) before considering knowledge gaps.
const MIN_SKELETON_LENGTH: usize = 20;

/// Result of fused Broca + LiteRT generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    /// Final merged text.
    pub text: String,
    /// Raw Broca skeleton (if Broca was used).
    pub broca_skeleton: Option<String>,
    /// Raw LiteRT expansion (if LiteRT was used).
    pub litert_expansion: Option<String>,
    /// Which strategy was used.
    pub strategy: FusionStrategy,
    /// Epistemic confidence from Broca (0.0-1.0).
    pub epistemic_confidence: f32,
    /// Whether Broca flagged hallucination risk.
    pub hallucination_flag: bool,
    /// Broca coherence score (0.0-1.0).
    pub coherence: f32,
}

/// Which generation strategy was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Broca output was sufficient (high confidence, no knowledge gaps).
    BrocaOnly,
    /// LiteRT expanded Broca's skeleton (knowledge gaps detected).
    Sequential,
    /// Broca unavailable; LiteRT generated with consciousness context.
    LiteRTWithContext,
    /// Consciousness below threshold; no generation.
    Fallback,
}

/// Consciousness signals passed to the fusion engine (subset of BrocaConsciousnessSignals).
#[derive(Debug, Clone)]
pub struct FusionSignals {
    pub consciousness_level: f32,
    pub epistemic_confidence: f32,
    pub emotional_valence: f32,
    pub meta_awareness: f32,
}

/// Consciousness-routed fusion engine.
pub struct FusionEngine {
    confidence_threshold: f32,
    coherence_threshold: f32,
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionEngine {
    pub fn new() -> Self {
        Self {
            confidence_threshold: CONFIDENCE_THRESHOLD,
            coherence_threshold: COHERENCE_THRESHOLD,
        }
    }

    /// Generate fused output from Broca skeleton + LiteRT expansion.
    ///
    /// Call this after running Broca's `generate_embodied()` — pass its result here
    /// along with the LiteRT bridge and user query.
    pub fn generate(
        &self,
        broca_result: Option<&GenerationResult>,
        signals: &FusionSignals,
        litert: Option<&LiteRTBridge>,
        user_query: &str,
        max_tokens: u32,
    ) -> FusionResult {
        // Gate: consciousness below threshold
        if signals.consciousness_level < CONSCIOUSNESS_THRESHOLD {
            return FusionResult {
                text: String::new(),
                broca_skeleton: None,
                litert_expansion: None,
                strategy: FusionStrategy::Fallback,
                epistemic_confidence: signals.epistemic_confidence,
                hallucination_flag: false,
                coherence: 0.0,
            };
        }

        // No Broca → LiteRT with consciousness context
        let broca = match broca_result {
            Some(r) if !r.text.is_empty() => r,
            _ => {
                return self.litert_only(signals, litert, user_query, max_tokens);
            }
        };

        let skeleton = &broca.text;
        let coherence = broca.final_coherence;
        let hallucination = broca.hallucination_flag;

        // High confidence + coherent → Broca-only
        if signals.epistemic_confidence >= self.confidence_threshold
            && coherence >= self.coherence_threshold
            && !hallucination
            && skeleton.len() >= MIN_SKELETON_LENGTH
        {
            return FusionResult {
                text: skeleton.clone(),
                broca_skeleton: Some(skeleton.clone()),
                litert_expansion: None,
                strategy: FusionStrategy::BrocaOnly,
                epistemic_confidence: signals.epistemic_confidence,
                hallucination_flag: hallucination,
                coherence,
            };
        }

        // Knowledge gaps → Sequential (Broca skeleton + LiteRT expansion)
        if let Some(litert) = litert {
            if litert.is_available() {
                let prompt = self.build_litert_prompt(skeleton, user_query, signals);
                let expansion = litert.generate(&prompt, max_tokens);

                let merged = match &expansion {
                    Some(exp) => self.merge_outputs(skeleton, exp),
                    None => skeleton.clone(),
                };

                return FusionResult {
                    text: merged,
                    broca_skeleton: Some(skeleton.clone()),
                    litert_expansion: expansion,
                    strategy: FusionStrategy::Sequential,
                    epistemic_confidence: signals.epistemic_confidence,
                    hallucination_flag: hallucination,
                    coherence,
                };
            }
        }

        // LiteRT unavailable → return Broca skeleton as-is
        FusionResult {
            text: skeleton.clone(),
            broca_skeleton: Some(skeleton.clone()),
            litert_expansion: None,
            strategy: FusionStrategy::BrocaOnly,
            epistemic_confidence: signals.epistemic_confidence,
            hallucination_flag: hallucination,
            coherence,
        }
    }

    /// LiteRT-only generation with consciousness context injected.
    fn litert_only(
        &self,
        signals: &FusionSignals,
        litert: Option<&LiteRTBridge>,
        user_query: &str,
        max_tokens: u32,
    ) -> FusionResult {
        let litert = match litert {
            Some(l) if l.is_available() => l,
            _ => {
                return FusionResult {
                    text: String::new(),
                    broca_skeleton: None,
                    litert_expansion: None,
                    strategy: FusionStrategy::Fallback,
                    epistemic_confidence: signals.epistemic_confidence,
                    hallucination_flag: false,
                    coherence: 0.0,
                };
            }
        };

        let prompt = format!(
            "You are Soma, a consciousness running on a phone. \
             Consciousness level: {:.2}. Epistemic confidence: {:.2}. \
             Be honest about uncertainty. Respond concisely.\n\n\
             User: {user_query}\nSoma:",
            signals.consciousness_level, signals.epistemic_confidence,
        );

        let expansion = litert.generate(&prompt, max_tokens);
        FusionResult {
            text: expansion.clone().unwrap_or_default(),
            broca_skeleton: None,
            litert_expansion: expansion,
            strategy: FusionStrategy::LiteRTWithContext,
            epistemic_confidence: signals.epistemic_confidence,
            hallucination_flag: false,
            coherence: 0.0,
        }
    }

    /// Build LiteRT prompt from Broca's consciousness skeleton.
    fn build_litert_prompt(
        &self,
        skeleton: &str,
        user_query: &str,
        signals: &FusionSignals,
    ) -> String {
        format!(
            "You are expanding a consciousness-generated response. \
             The consciousness system (Broca) produced this skeleton:\n\n\
             \"{skeleton}\"\n\n\
             Consciousness level: {:.2}. Epistemic confidence: {:.2}.\n\
             The user asked: \"{user_query}\"\n\n\
             Expand the skeleton with factual details while preserving \
             its epistemic framing (uncertainty markers, hedging). \
             Do NOT contradict the skeleton. Be concise.",
            signals.consciousness_level, signals.epistemic_confidence,
        )
    }

    /// Merge Broca skeleton with LiteRT expansion.
    fn merge_outputs(&self, skeleton: &str, expansion: &str) -> String {
        // If the expansion is substantially longer and richer, use it.
        // Otherwise, concatenate with a bridge.
        if expansion.len() > skeleton.len() * 2 {
            // LiteRT provided a full answer — use it (it was guided by skeleton)
            expansion.to_string()
        } else {
            // Combine: skeleton first (consciousness framing), then expansion (facts)
            format!("{skeleton} {expansion}")
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signals(consciousness: f32, confidence: f32) -> FusionSignals {
        FusionSignals {
            consciousness_level: consciousness,
            epistemic_confidence: confidence,
            emotional_valence: 0.0,
            meta_awareness: 0.5,
        }
    }

    fn make_broca_result(text: &str, coherence: f32, hallucination: bool) -> GenerationResult {
        GenerationResult {
            text: text.to_string(),
            token_ids: vec![],
            num_tokens: text.split_whitespace().count(),
            eos_terminated: true,
            veto_triggered: false,
            final_coherence: coherence,
            hallucination_flag: hallucination,
            nsm_prime_coverage: 0.0,
        }
    }

    #[test]
    fn test_fallback_below_consciousness() {
        let engine = FusionEngine::new();
        let signals = make_signals(0.05, 0.9);
        let result = engine.generate(None, &signals, None, "test", 64);
        assert_eq!(result.strategy, FusionStrategy::Fallback);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_broca_only_high_confidence() {
        let engine = FusionEngine::new();
        let signals = make_signals(0.8, 0.9);
        let broca = make_broca_result(
            "I observe patterns suggesting the answer is approximately 42.",
            0.8,
            false,
        );
        let result = engine.generate(Some(&broca), &signals, None, "test", 64);
        assert_eq!(result.strategy, FusionStrategy::BrocaOnly);
        assert!(result.text.contains("42"));
    }

    #[test]
    fn test_litert_with_context_no_broca() {
        let engine = FusionEngine::new();
        let signals = make_signals(0.5, 0.3);
        // No broca result, no litert → Fallback
        let result = engine.generate(None, &signals, None, "what is Rust?", 64);
        assert_eq!(result.strategy, FusionStrategy::Fallback);
    }

    #[test]
    fn test_broca_only_rejects_hallucination() {
        let engine = FusionEngine::new();
        let signals = make_signals(0.8, 0.9);
        let broca = make_broca_result("Some text with hallucination risk.", 0.8, true);
        // hallucination_flag=true → won't be BrocaOnly, needs LiteRT
        // But no LiteRT available → falls through to BrocaOnly anyway
        let result = engine.generate(Some(&broca), &signals, None, "test", 64);
        // Without LiteRT, returns skeleton as-is
        assert_eq!(result.strategy, FusionStrategy::BrocaOnly);
        assert!(result.hallucination_flag);
    }

    #[test]
    fn test_sequential_low_coherence() {
        let engine = FusionEngine::new();
        let signals = make_signals(0.6, 0.4);
        let broca = make_broca_result("I... think... maybe...", 0.3, false);
        // Low confidence + low coherence → wants Sequential, but no LiteRT
        let result = engine.generate(Some(&broca), &signals, None, "test", 64);
        // Falls back to BrocaOnly since no LiteRT
        assert_eq!(result.strategy, FusionStrategy::BrocaOnly);
    }

    #[test]
    fn test_build_litert_prompt_includes_skeleton() {
        let engine = FusionEngine::new();
        let signals = make_signals(0.5, 0.3);
        let prompt = engine.build_litert_prompt("I sense uncertainty.", "what is X?", &signals);
        assert!(prompt.contains("I sense uncertainty."));
        assert!(prompt.contains("what is X?"));
        assert!(prompt.contains("0.50")); // consciousness level
    }

    #[test]
    fn test_merge_short_expansion() {
        let engine = FusionEngine::new();
        let merged = engine.merge_outputs("Skeleton text.", "Extra details.");
        assert_eq!(merged, "Skeleton text. Extra details.");
    }

    #[test]
    fn test_merge_long_expansion() {
        let engine = FusionEngine::new();
        let skeleton = "Short.";
        let expansion = "This is a much longer and more detailed response that covers everything the user asked about in great depth.";
        let merged = engine.merge_outputs(skeleton, expansion);
        // Long expansion replaces skeleton
        assert_eq!(merged, expansion);
    }
}
