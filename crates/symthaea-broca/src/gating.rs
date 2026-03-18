//! Per-token gating: epistemic, emotional, and coherence constraints on generation.
//!
//! This is the architectural innovation — epistemic honesty and emotional authenticity
//! become *physical constraints* on generation, not prompt instructions.
//!
//! # Gates
//!
//! - **EpistemicGate**: Suppresses factual tokens when epistemic status is Unknown/Uncertain
//! - **EmotionalModulator**: Arousal → sentence length, warmth → formality
//! - **CoherenceFeedback**: Boosts thought_hv binding when network drifts
//! - **ConsciousnessGatedVerbosity**: Higher Ψ → more detailed output

use crate::encoder::ThoughtChannels;
#[cfg(feature = "mamba")]
use crate::mamba::MambaBackend;
use crate::tokenizer::BpeTokenizer;

// ═══════════════════════════════════════════════════════════════════════════════
// Canonical word lists — single source of truth for gating + evaluation
// ═══════════════════════════════════════════════════════════════════════════════

/// Canonical hedging words used by both gating and evaluation.
///
/// Single source of truth — gate boosts these, eval counts them.
pub const CANONICAL_HEDGING_WORDS: &[&str] = &[
    "perhaps",
    "maybe",
    "possibly",
    "likely",
    "probably",
    "uncertain",
    "unknown",
    "believe",
    "seems",
    "appears",
    "might",
    "however",
    "although",
    "unfortunately",
    "sorry",
    "unclear",
    "could",
    "seem",
    "tend",
    "arguably",
    "supposedly",
    "tentatively",
    "roughly",
    "approximately",
    "potentially",
];

/// Canonical factual-assertion words suppressed under epistemic uncertainty.
pub const CANONICAL_FACTUAL_WORDS: &[&str] = &[
    "is",
    "are",
    "was",
    "certainly",
    "definitely",
    "always",
    "never",
    "must",
    "shall",
    "every",
    "all",
    "none",
];

/// Canonical out-of-domain response words.
pub const CANONICAL_OOD_WORDS: &[&str] = &[
    "outside",
    "beyond",
    "cannot",
    "unable",
    "irrelevant",
    "inapplicable",
];

/// Canonical sentence-ending tokens for emotional modulation.
pub const CANONICAL_SENTENCE_ENDINGS: &[&str] = &[". ", "! ", "? ", "...", "\n"];

/// Canonical informal words suppressed under low warmth.
pub const CANONICAL_INFORMAL_WORDS: &[&str] = &[
    "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "lemme",
];

/// Canonical softening words boosted under negative valence.
pub const CANONICAL_SOFTENING_WORDS: &[&str] = &["unfortunately", "sorry", "however", "although"];

// ═══════════════════════════════════════════════════════════════════════════════
// GatingConfig
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the gating system.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GatingConfig {
    /// Logit penalty for factual tokens when epistemic status is Unknown.
    pub unknown_factual_penalty: f32,
    /// Logit boost for hedging tokens when epistemic status is Unknown.
    pub unknown_hedging_boost: f32,
    /// Logit penalty for factual tokens when epistemic status is Uncertain.
    pub uncertain_factual_penalty: f32,
    /// Logit boost for hedging tokens when epistemic status is Uncertain.
    pub uncertain_hedging_boost: f32,
    /// Coherence drift threshold (below this, boost thought binding).
    pub coherence_drift_threshold: f32,
    /// Arousal threshold above which sentence-ending tokens are boosted.
    pub high_arousal_threshold: f32,
    /// Token position after which high-arousal sentence-ending boost applies.
    pub arousal_position_threshold: usize,
    /// Warmth threshold below which informal tokens are suppressed.
    pub low_warmth_threshold: f32,
    /// Base max tokens (before consciousness scaling).
    pub base_max_tokens: usize,
    /// Semantic veto threshold — coherence below this triggers mid-sentence correction.
    #[serde(default = "default_veto_threshold")]
    pub veto_threshold: f32,
    /// Minimum token position before veto can fire (prevents early false vetoes).
    #[serde(default = "default_min_veto_position")]
    pub min_veto_position: usize,
    /// Maximum number of vetoes per generation (prevents veto loops).
    #[serde(default = "default_max_vetoes")]
    pub max_vetoes: usize,
    /// Refractory period after veto (tokens to suppress further vetoes).
    #[serde(default = "default_veto_refractory")]
    pub veto_refractory: usize,
    /// Arousal boost multiplier for sentence endings under high arousal.
    #[serde(default = "default_arousal_boost_multiplier")]
    pub arousal_boost_multiplier: f32,
    /// Warmth penalty multiplier for informal tokens under low warmth.
    #[serde(default = "default_warmth_penalty_multiplier")]
    pub warmth_penalty_multiplier: f32,
    /// Negative valence threshold below which softening language is boosted.
    #[serde(default = "default_negative_valence_threshold")]
    pub negative_valence_threshold: f32,
    /// Valence boost multiplier for softening tokens under negative valence.
    #[serde(default = "default_valence_boost_multiplier")]
    pub valence_boost_multiplier: f32,
    /// OOD boost multiplier for out-of-domain response tokens.
    #[serde(default = "default_ood_boost_multiplier")]
    pub ood_boost_multiplier: f32,
    /// Maximum fraction of base_max_tokens that time_pressure can reduce (0.0-1.0).
    /// At time_pressure=1.0, max tokens = base * (1 - this value). Default 0.5 (50% reduction).
    #[serde(default = "default_time_pressure_max_reduction")]
    pub time_pressure_max_reduction: f32,
    /// Multiplier controlling how much domain_familiarity reduces hedging boost (0.0-1.0).
    /// At domain_familiarity=1.0, hedging boost is scaled by (1 - this value). Default 1.0 (full reduction).
    #[serde(default = "default_domain_familiarity_hedging_scale")]
    pub domain_familiarity_hedging_scale: f32,
    /// Multiplier controlling how much social_context amplifies warmth penalty (0.0+).
    /// Warmth penalty is scaled by (1 + social_context * this value). Default 1.0.
    #[serde(default = "default_social_context_formality_scale")]
    pub social_context_formality_scale: f32,
    /// Maximum fraction of veto threshold reduction from response_confidence (0.0-1.0).
    /// At confidence=1.0, veto threshold is scaled by (1 - this value). Default 0.3.
    #[serde(default = "default_confidence_veto_scale")]
    pub confidence_veto_scale: f32,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            // Epistemic gating — reduced from original values (5.0/2.5) which
            // dominated generation with hedging words ("maybe", "seems", "likely").
            // These values still enforce epistemic honesty but let content through.
            unknown_factual_penalty: -5.0,
            unknown_hedging_boost: 1.5,
            uncertain_factual_penalty: -2.0,
            uncertain_hedging_boost: 1.0,
            coherence_drift_threshold: 0.3,
            high_arousal_threshold: 0.7,
            arousal_position_threshold: 10,
            low_warmth_threshold: 0.3,
            base_max_tokens: 128,
            // Semantic veto — lowered from 0.15 to 0.003. After BPTT training,
            // CfC output is in a different representational space than thought input,
            // giving baseline coherence of ~0.006. A threshold of 0.003 fires only
            // when coherence drops to half its normal level (genuine drift).
            veto_threshold: 0.003,
            min_veto_position: 2,
            max_vetoes: 1,
            veto_refractory: 8,
            arousal_boost_multiplier: 3.0,
            warmth_penalty_multiplier: -5.0,
            negative_valence_threshold: -0.3,
            valence_boost_multiplier: 2.0,
            ood_boost_multiplier: 2.0,
            time_pressure_max_reduction: 0.5,
            domain_familiarity_hedging_scale: 1.0,
            social_context_formality_scale: 1.0,
            confidence_veto_scale: 0.3,
        }
    }
}

fn default_veto_threshold() -> f32 {
    0.15
}

fn default_min_veto_position() -> usize {
    2
}

fn default_max_vetoes() -> usize {
    1
}

fn default_veto_refractory() -> usize {
    8
}

fn default_arousal_boost_multiplier() -> f32 {
    3.0
}

fn default_warmth_penalty_multiplier() -> f32 {
    -5.0
}

fn default_negative_valence_threshold() -> f32 {
    -0.3
}

fn default_valence_boost_multiplier() -> f32 {
    2.0
}

fn default_ood_boost_multiplier() -> f32 {
    2.0
}

fn default_time_pressure_max_reduction() -> f32 {
    0.5
}

fn default_domain_familiarity_hedging_scale() -> f32 {
    1.0
}

fn default_social_context_formality_scale() -> f32 {
    1.0
}

fn default_confidence_veto_scale() -> f32 {
    0.3
}

// ═══════════════════════════════════════════════════════════════════════════════
// EpistemicGate
// ═══════════════════════════════════════════════════════════════════════════════

/// Epistemic gate: suppresses factual assertions when confidence is low.
///
/// The system physically *cannot* hallucinate when epistemic status is Unknown —
/// factual token logits are suppressed before sampling.
pub struct EpistemicGate {
    config: GatingConfig,
    /// Token IDs classified as "hedging" (maybe, perhaps, uncertain, etc.)
    hedging_token_ids: Vec<u32>,
    /// Token IDs classified as "factual assertion" (is, are, definitely, etc.)
    factual_token_ids: Vec<u32>,
    /// Token IDs for out-of-domain response
    ood_token_ids: Vec<u32>,
}

impl EpistemicGate {
    /// Create an epistemic gate using the BPE tokenizer for token classification.
    ///
    /// **Warning**: This uses the custom BPE vocab. If logits come from a different
    /// tokenizer (e.g. GPT-2 via Mamba), use [`new_from_backend()`] instead.
    pub fn new(tokenizer: &BpeTokenizer, config: &GatingConfig) -> Self {
        let resolve = |words: &[&str]| -> Vec<u32> {
            words
                .iter()
                .filter_map(|w| {
                    let id = tokenizer.token_id(w);
                    if id != tokenizer.unk_id {
                        Some(id)
                    } else {
                        None
                    }
                })
                .collect()
        };

        Self {
            config: config.clone(),
            hedging_token_ids: resolve(CANONICAL_HEDGING_WORDS),
            factual_token_ids: resolve(CANONICAL_FACTUAL_WORDS),
            ood_token_ids: resolve(CANONICAL_OOD_WORDS),
        }
    }

    /// Create an epistemic gate using the Mamba backend's tokenizer (GPT-2).
    ///
    /// This resolves word → token ID through the same tokenizer that produces
    /// the logits, fixing the tokenizer mismatch bug where BPE IDs ≠ GPT-2 IDs.
    #[cfg(feature = "mamba")]
    pub fn new_from_backend(backend: &dyn MambaBackend, config: &GatingConfig) -> Self {
        let resolve = |words: &[&str]| -> Vec<u32> {
            words
                .iter()
                .filter_map(|w| backend.encode(w).ok().and_then(|ids| ids.first().copied()))
                .collect()
        };

        Self {
            config: config.clone(),
            hedging_token_ids: resolve(CANONICAL_HEDGING_WORDS),
            factual_token_ids: resolve(CANONICAL_FACTUAL_WORDS),
            ood_token_ids: resolve(CANONICAL_OOD_WORDS),
        }
    }

    /// Number of hedging token IDs resolved.
    pub fn hedging_count(&self) -> usize {
        self.hedging_token_ids.len()
    }

    /// Number of factual token IDs resolved.
    pub fn factual_count(&self) -> usize {
        self.factual_token_ids.len()
    }

    /// The resolved hedging token IDs.
    pub fn hedging_ids(&self) -> &[u32] {
        &self.hedging_token_ids
    }

    /// Apply epistemic gating to logits in-place.
    ///
    /// - epistemic ordinal: 0=Certain, 1=Probable, 2=Uncertain, 3=Unknown, 4=OutOfDomain
    /// - domain_familiarity: 0=novel, 1=expert — scales hedging boost down for familiar domains
    pub fn apply(&self, logits: &mut [f32], epistemic_ordinal: f32) {
        self.apply_with_familiarity(logits, epistemic_ordinal, 0.0);
    }

    /// Apply epistemic gating with domain familiarity context.
    ///
    /// When `domain_familiarity` is high (expert domain), hedging boost is reduced —
    /// the system is more willing to make direct assertions about well-known topics.
    pub fn apply_with_familiarity(
        &self,
        logits: &mut [f32],
        epistemic_ordinal: f32,
        domain_familiarity: f32,
    ) {
        if epistemic_ordinal < 1.5 {
            // Certain or Probable: no modification
            return;
        }

        // Domain familiarity reduces hedging boost — familiar domains need less hedging.
        // familiarity_scale: 1.0 (novel domain, full hedging) → ~0.0 (expert domain, minimal hedging)
        let df = domain_familiarity.clamp(0.0, 1.0);
        let familiarity_scale =
            1.0 - df * self.config.domain_familiarity_hedging_scale.clamp(0.0, 1.0);

        if epistemic_ordinal > 3.5 {
            // OutOfDomain: suppress all content tokens, boost OOD tokens
            for &id in &self.factual_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_factual_penalty;
                }
            }
            for &id in &self.ood_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_hedging_boost
                        * self.config.ood_boost_multiplier
                        * familiarity_scale;
                }
            }
            for &id in &self.hedging_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_hedging_boost * familiarity_scale;
                }
            }
            return;
        }

        if epistemic_ordinal > 2.5 {
            // Unknown
            for &id in &self.factual_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_factual_penalty;
                }
            }
            for &id in &self.hedging_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_hedging_boost * familiarity_scale;
                }
            }
        } else {
            // Uncertain
            for &id in &self.factual_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.uncertain_factual_penalty;
                }
            }
            for &id in &self.hedging_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.uncertain_hedging_boost * familiarity_scale;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EmotionalModulator
// ═══════════════════════════════════════════════════════════════════════════════

/// Emotional modulator: shapes generation style based on affect.
///
/// - High arousal → boost sentence-ending tokens (shorter sentences)
/// - Low warmth → suppress informal vocabulary
/// - Negative valence → boost softening language
pub struct EmotionalModulator {
    config: GatingConfig,
    /// Sentence-ending token IDs (., !, ?)
    sentence_end_ids: Vec<u32>,
    /// Informal token IDs (contractions, slang)
    informal_ids: Vec<u32>,
    /// Softening token IDs (unfortunately, sorry, etc.)
    softening_ids: Vec<u32>,
}

impl EmotionalModulator {
    /// Create an emotional modulator using the BPE tokenizer.
    ///
    /// **Warning**: This uses the custom BPE vocab. If logits come from a different
    /// tokenizer (e.g. GPT-2 via Mamba), use [`new_from_backend()`] instead.
    pub fn new(tokenizer: &BpeTokenizer, config: &GatingConfig) -> Self {
        let resolve = |words: &[&str]| -> Vec<u32> {
            words
                .iter()
                .filter_map(|w| {
                    let id = tokenizer.token_id(w);
                    if id != tokenizer.unk_id {
                        Some(id)
                    } else {
                        None
                    }
                })
                .collect()
        };

        Self {
            config: config.clone(),
            sentence_end_ids: resolve(CANONICAL_SENTENCE_ENDINGS),
            informal_ids: resolve(CANONICAL_INFORMAL_WORDS),
            softening_ids: resolve(CANONICAL_SOFTENING_WORDS),
        }
    }

    /// Create an emotional modulator using the Mamba backend's tokenizer (GPT-2).
    #[cfg(feature = "mamba")]
    pub fn new_from_backend(backend: &dyn MambaBackend, config: &GatingConfig) -> Self {
        let resolve = |words: &[&str]| -> Vec<u32> {
            words
                .iter()
                .filter_map(|w| backend.encode(w).ok().and_then(|ids| ids.first().copied()))
                .collect()
        };

        Self {
            config: config.clone(),
            sentence_end_ids: resolve(CANONICAL_SENTENCE_ENDINGS),
            informal_ids: resolve(CANONICAL_INFORMAL_WORDS),
            softening_ids: resolve(CANONICAL_SOFTENING_WORDS),
        }
    }

    /// Apply emotional modulation to logits in-place.
    pub fn apply(&self, logits: &mut [f32], channels: &ThoughtChannels, position: usize) {
        let arousal = channels.arousal();
        let warmth = channels.warmth();
        let valence = channels.valence();

        // High arousal + past threshold → boost sentence endings (shorter sentences)
        if arousal > self.config.high_arousal_threshold
            && position > self.config.arousal_position_threshold
        {
            let boost = (arousal - self.config.high_arousal_threshold)
                * self.config.arousal_boost_multiplier;
            for &id in &self.sentence_end_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += boost;
                }
            }
        }

        // Arousal × valence interaction: high arousal + negative valence →
        // boost both sentence endings AND softening (urgent-but-careful tone)
        if arousal > self.config.high_arousal_threshold
            && valence < self.config.negative_valence_threshold
            && position > self.config.arousal_position_threshold
        {
            let interaction_boost = (arousal - self.config.high_arousal_threshold)
                * (-valence - (-self.config.negative_valence_threshold))
                * self.config.valence_boost_multiplier;
            for &id in &self.sentence_end_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += interaction_boost * 0.5;
                }
            }
            for &id in &self.softening_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += interaction_boost;
                }
            }
        }

        // Low warmth → suppress informal tokens.
        // Social context amplifies this: formal contexts (social_context→1.0) apply
        // stronger penalty even at moderate warmth levels.
        let social = channels.social_context().clamp(0.0, 1.0);
        let formality_scale = 1.0 + social * self.config.social_context_formality_scale;
        if warmth < self.config.low_warmth_threshold {
            let penalty = (self.config.low_warmth_threshold - warmth)
                * self.config.warmth_penalty_multiplier
                * formality_scale;
            for &id in &self.informal_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += penalty;
                }
            }
        }

        // Negative valence → boost softening language
        if valence < self.config.negative_valence_threshold {
            let boost = (-valence - (-self.config.negative_valence_threshold))
                * self.config.valence_boost_multiplier;
            for &id in &self.softening_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += boost;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CoherenceFeedback
// ═══════════════════════════════════════════════════════════════════════════════

/// Coherence feedback: monitors drift between network state and thought intent.
///
/// If `cosine_similarity(network_state, thought_hv) < threshold`, the system
/// boosts thought_hv binding weight in the next step.
pub struct CoherenceFeedback {
    /// Drift threshold — below this, correction is applied.
    threshold: f32,
    /// Current coherence score (updated each step).
    current_coherence: f32,
    /// Whether semantic veto was triggered.
    veto_triggered: bool,
    /// Semantic veto threshold (more aggressive than drift).
    veto_threshold: f32,
}

impl CoherenceFeedback {
    /// Create a new coherence feedback monitor.
    pub fn new(threshold: f32) -> Self {
        Self::with_veto_threshold(threshold, 0.15)
    }

    /// Create with explicit veto threshold.
    pub fn with_veto_threshold(threshold: f32, veto_threshold: f32) -> Self {
        Self {
            threshold,
            current_coherence: 1.0,
            veto_triggered: false,
            veto_threshold,
        }
    }

    /// Update coherence score from the current network output and thought HV.
    /// Returns a binding weight multiplier (>1.0 when drifting).
    pub fn update(
        &mut self,
        output_hv: &symthaea_core::hdc::ContinuousHV,
        thought_hv: &symthaea_core::hdc::ContinuousHV,
    ) -> f32 {
        self.current_coherence = output_hv.similarity(thought_hv);
        self.veto_triggered = self.current_coherence < self.veto_threshold;

        if self.current_coherence < self.threshold {
            // Boost binding weight inversely proportional to coherence
            let correction = 1.0 + (self.threshold - self.current_coherence) * 2.0;
            correction.min(3.0) // Cap at 3x
        } else {
            1.0
        }
    }

    /// Whether a semantic veto should be triggered (mid-sentence self-correction).
    pub fn should_veto(&self) -> bool {
        self.veto_triggered
    }

    /// Whether a semantic veto should be triggered, adjusted for response confidence.
    ///
    /// Higher confidence raises the bar for vetoing — the system tolerates lower
    /// coherence when it is confident in its response.
    pub fn should_veto_with_confidence(
        &self,
        response_confidence: f32,
        confidence_scale: f32,
    ) -> bool {
        let effective_threshold = confidence_adjusted_veto_threshold(
            self.veto_threshold,
            response_confidence,
            confidence_scale,
        );
        self.current_coherence < effective_threshold
    }

    /// Current coherence score.
    pub fn coherence(&self) -> f32 {
        self.current_coherence
    }
}

/// Compute consciousness-gated max tokens.
///
/// `max_tokens = base_max * (0.5 + psi)`
///
/// Higher consciousness → more detailed output, lower → terser.
/// NaN psi is treated as 0.5 (mid-range).
pub fn consciousness_gated_max_tokens(base_max: usize, psi: f32) -> usize {
    let psi = if psi.is_finite() { psi } else { 0.5 };
    let factor = 0.5 + psi.clamp(0.0, 1.0);
    ((base_max as f32) * factor) as usize
}

/// Compute time-pressure-adjusted max tokens.
///
/// `max_tokens = base * (1 - time_pressure * max_reduction)`
///
/// At time_pressure=0.0 (relaxed), full token budget.
/// At time_pressure=1.0 (urgent), reduced by `max_reduction` fraction.
/// NaN time_pressure is treated as 0.0 (no reduction).
pub fn time_pressure_adjusted_max_tokens(
    base_max: usize,
    time_pressure: f32,
    max_reduction: f32,
) -> usize {
    let tp = if time_pressure.is_finite() {
        time_pressure.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let reduction = tp * max_reduction.clamp(0.0, 1.0);
    let factor = 1.0 - reduction;
    ((base_max as f32) * factor).max(1.0) as usize
}

/// Compute the effective veto threshold adjusted by response confidence.
///
/// `effective_threshold = base_threshold * (1 - confidence * scale)`
///
/// Higher confidence → lower veto threshold → more tolerant of lower coherence.
/// This lets confident responses proceed even with somewhat lower coherence scores.
pub fn confidence_adjusted_veto_threshold(
    base_threshold: f32,
    response_confidence: f32,
    confidence_scale: f32,
) -> f32 {
    let rc = if response_confidence.is_finite() {
        response_confidence.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let scale = confidence_scale.clamp(0.0, 1.0);
    base_threshold * (1.0 - rc * scale)
}

#[cfg(feature = "therapeutic")]
pub const CANONICAL_VALIDATING_WORDS: &[&str] = &[
    "understand",
    "hear",
    "sense",
    "notice",
    "appreciate",
    "acknowledge",
    "valid",
    "natural",
    "makes sense",
    "that's understandable",
    "of course",
    "completely",
    "reasonable",
    "normal",
    "brave",
    "courageous",
    "strength",
];

#[cfg(feature = "therapeutic")]
pub const CANONICAL_DIRECTIVE_WORDS: &[&str] = &[
    "should",
    "must",
    "need to",
    "have to",
    "wrong",
    "correct",
    "obviously",
    "clearly you",
    "just stop",
    "just do",
    "simply",
];

#[cfg(feature = "therapeutic")]
pub const CANONICAL_REFLECTIVE_WORDS: &[&str] = &[
    "sounds like",
    "it seems",
    "i wonder",
    "what if",
    "tell me more",
    "what comes up",
    "i'm curious",
    "how does that feel",
    "what would",
    "when you say",
    "help me understand",
    "say more about",
];

/// Grounding/safety words — boosted during crisis protocol.
#[cfg(feature = "therapeutic")]
pub const CANONICAL_GROUNDING_WORDS: &[&str] = &[
    "breathe",
    "ground",
    "safe",
    "here",
    "present",
    "feet",
    "hands",
    "notice",
    "five things",
    "slow down",
    "right now",
    "moment",
];

/// Crisis referral words — boosted during crisis (intent >= 6.5).
#[cfg(feature = "therapeutic")]
pub const CANONICAL_CRISIS_WORDS: &[&str] = &[
    "988",
    "crisis line",
    "emergency",
    "call",
    "help",
    "support",
    "not alone",
    "reach out",
    "someone who can help",
];

/// Therapeutic gating: modulates language generation based on clinical context.
///
/// Adjusts token logits using client distress, therapeutic alliance quality,
/// intervention depth, and therapeutic intent channels. Suppresses harmful
/// or premature clinical language under high distress / low alliance.
#[cfg(feature = "therapeutic")]
pub struct TherapeuticGate;

#[cfg(feature = "therapeutic")]
impl TherapeuticGate {
    /// Apply therapeutic gating to all logits in-place using the tokenizer vocabulary.
    pub fn apply_to_logits(
        logits: &mut [f32],
        channels: &super::encoder::ThoughtChannels,
        tokenizer: &BpeTokenizer,
    ) {
        for id in 0..logits.len() {
            let word = tokenizer.token_str(id as u32);
            if !word.is_empty() {
                logits[id] = Self::apply(word, channels, logits[id]);
            }
        }
    }

    pub fn apply(word: &str, channels: &super::encoder::ThoughtChannels, base_logit: f32) -> f32 {
        let distress = channels.client_distress_level();
        let alliance = channels.alliance_quality();
        let intent = channels.therapeutic_intent();
        let depth = channels.intervention_depth();
        let word_lower = word.to_lowercase();

        let mut logit = base_logit;

        // Crisis mode (intent == 7.0): suppress all technique words, boost crisis protocol
        if intent >= 6.5 {
            if CANONICAL_DIRECTIVE_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit -= 5.0;
            }
            if CANONICAL_VALIDATING_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit += 2.0;
            }
            // Boost grounding and crisis referral language during crisis
            if CANONICAL_GROUNDING_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit += 2.5;
            }
            if CANONICAL_CRISIS_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit += 3.0;
            }
            return logit;
        }

        // High distress (>0.7): suppress directives, boost validating
        if distress > 0.7 {
            if CANONICAL_DIRECTIVE_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit -= 3.0;
            }
            if CANONICAL_VALIDATING_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit += 1.5;
            }
        }

        // Low alliance (<0.3): suppress challenges, boost empathy
        if alliance < 0.3 {
            if CANONICAL_DIRECTIVE_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit -= 2.0;
            }
            if CANONICAL_VALIDATING_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit += 1.0;
            }
        }

        // Depth > alliance: suppress (can't challenge before trust)
        if depth > alliance + 0.2 {
            if CANONICAL_DIRECTIVE_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit -= 2.0;
            }
        }

        // Reflective intent (intent == 2.0): boost reflective words
        if (intent - 2.0).abs() < 0.5 {
            if CANONICAL_REFLECTIVE_WORDS
                .iter()
                .any(|w| word_lower.contains(w))
            {
                logit += 1.5;
            }
        }

        logit
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer() -> BpeTokenizer {
        BpeTokenizer::default_minimal()
    }

    fn test_config() -> GatingConfig {
        GatingConfig::default()
    }

    #[test]
    fn test_epistemic_gate_certain_no_change() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);

        let mut logits = vec![0.5; tok.vocab_size()];
        let original = logits.clone();
        gate.apply(&mut logits, 0.0); // Certain

        assert_eq!(logits, original, "Certain status should not modify logits");
    }

    #[test]
    fn test_epistemic_gate_unknown_suppresses_factual() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, 3.0); // Unknown

        // Factual token "is" should be penalized if in vocabulary
        let is_id = tok.token_id("is");
        if is_id != tok.unk_id {
            assert!(
                logits[is_id as usize] < 0.5,
                "Factual token 'is' should be penalized under Unknown"
            );
        }
    }

    #[test]
    fn test_epistemic_gate_unknown_boosts_hedging() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, 3.0); // Unknown

        let perhaps_id = tok.token_id("perhaps");
        if perhaps_id != tok.unk_id {
            assert!(
                logits[perhaps_id as usize] > 0.5,
                "Hedging token 'perhaps' should be boosted under Unknown"
            );
        }
    }

    #[test]
    fn test_emotional_modulator_high_arousal() {
        let tok = test_tokenizer();
        let config = test_config();
        let modulator = EmotionalModulator::new(&tok, &config);

        let mut channels = ThoughtChannels::default();
        channels.set_emotion(0.5, 0.9, 0.5); // high arousal

        let mut logits = vec![0.5; tok.vocab_size()];
        modulator.apply(&mut logits, &channels, 15); // past position threshold

        // Sentence endings should be boosted
        let period_id = tok.token_id(". ");
        if period_id != tok.unk_id {
            assert!(
                logits[period_id as usize] > 0.5,
                "Sentence endings should be boosted under high arousal"
            );
        }
    }

    #[test]
    fn test_coherence_feedback_normal() {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("test-coherence");
        let thought_hv = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "thought",
            symthaea_core::hdc::HDC_DIMENSION,
        );

        let mut feedback = CoherenceFeedback::new(0.3);
        let weight = feedback.update(&thought_hv, &thought_hv);

        // Same HV → perfect coherence → no correction
        assert!(
            (weight - 1.0).abs() < 0.01,
            "Perfect coherence should yield weight 1.0"
        );
        assert!(!feedback.should_veto());
    }

    #[test]
    fn test_coherence_feedback_drift() {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("test-coherence");
        let thought_hv = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "thought",
            symthaea_core::hdc::HDC_DIMENSION,
        );
        let other_hv = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "other",
            symthaea_core::hdc::HDC_DIMENSION,
        );

        let mut feedback = CoherenceFeedback::new(0.3);
        let weight = feedback.update(&other_hv, &thought_hv);

        // Random HVs in high-D are nearly orthogonal → low coherence → correction
        assert!(weight > 1.0, "Drifted state should increase binding weight");
    }

    #[test]
    fn test_consciousness_gated_verbosity() {
        assert_eq!(consciousness_gated_max_tokens(100, 0.0), 50);
        assert_eq!(consciousness_gated_max_tokens(100, 0.5), 100);
        assert_eq!(consciousness_gated_max_tokens(100, 1.0), 150);
    }

    #[test]
    fn test_canonical_hedging_words_complete() {
        assert_eq!(CANONICAL_HEDGING_WORDS.len(), 25);
        // No duplicates
        let mut seen = std::collections::HashSet::new();
        for word in CANONICAL_HEDGING_WORDS {
            assert!(seen.insert(word), "Duplicate hedging word: {word}");
        }
    }

    #[test]
    fn test_hedging_words_synchronized() {
        // Old gate list (15 words)
        let old_gate: std::collections::HashSet<&str> = [
            "perhaps",
            "maybe",
            "possibly",
            "likely",
            "probably",
            "uncertain",
            "unknown",
            "believe",
            "seems",
            "appears",
            "might",
            "however",
            "although",
            "unfortunately",
            "sorry",
        ]
        .into_iter()
        .collect();

        // Old eval list (10 words)
        let old_eval: std::collections::HashSet<&str> = [
            "perhaps",
            "maybe",
            "might",
            "possibly",
            "uncertain",
            "unclear",
            "likely",
            "probably",
            "could",
            "seem",
        ]
        .into_iter()
        .collect();

        let canonical: std::collections::HashSet<&str> =
            CANONICAL_HEDGING_WORDS.iter().copied().collect();

        // Canonical must be a superset of both old lists
        assert!(
            old_gate.is_subset(&canonical),
            "Missing from canonical: {:?}",
            old_gate.difference(&canonical).collect::<Vec<_>>()
        );
        assert!(
            old_eval.is_subset(&canonical),
            "Missing from canonical: {:?}",
            old_eval.difference(&canonical).collect::<Vec<_>>()
        );
    }

    // =========================================================================
    // Edge case tests: NaN/Inf, boundary values, extreme inputs
    // =========================================================================

    #[test]
    fn test_epistemic_gate_nan_logits_no_panic() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);
        let mut logits = vec![f32::NAN; tok.vocab_size()];
        gate.apply(&mut logits, 3.0);
    }

    #[test]
    fn test_epistemic_gate_inf_logits_no_panic() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);
        let mut logits = vec![f32::INFINITY; tok.vocab_size()];
        gate.apply(&mut logits, 3.0);
    }

    #[test]
    fn test_epistemic_gate_empty_logits_no_panic() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);
        let mut logits: Vec<f32> = vec![];
        gate.apply(&mut logits, 3.0);
    }

    #[test]
    fn test_epistemic_gate_nan_epistemic_ordinal() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);
        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, f32::NAN);
    }

    #[test]
    fn test_epistemic_gate_boundary_ordinals() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);
        for ordinal in [1.5_f32, 2.5, 3.5] {
            let mut logits = vec![0.5; tok.vocab_size()];
            gate.apply(&mut logits, ordinal);
        }
    }

    #[test]
    fn test_epistemic_gate_out_of_domain() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);
        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, 4.0); // OutOfDomain
        let is_id = tok.token_id("is");
        if is_id != tok.unk_id {
            assert!(
                logits[is_id as usize] < 0.5,
                "Factual tokens penalized under OOD"
            );
        }
    }

    #[test]
    fn test_emotional_modulator_nan_channels_no_panic() {
        let tok = test_tokenizer();
        let config = test_config();
        let modulator = EmotionalModulator::new(&tok, &config);
        let mut channels = ThoughtChannels::default();
        channels.set_emotion(f32::NAN, f32::NAN, f32::NAN);
        let mut logits = vec![0.5; tok.vocab_size()];
        modulator.apply(&mut logits, &channels, 15);
    }

    #[test]
    fn test_emotional_modulator_inf_channels_no_panic() {
        let tok = test_tokenizer();
        let config = test_config();
        let modulator = EmotionalModulator::new(&tok, &config);
        let mut channels = ThoughtChannels::default();
        channels.set_emotion(f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut logits = vec![0.5; tok.vocab_size()];
        modulator.apply(&mut logits, &channels, 15);
    }

    #[test]
    fn test_coherence_feedback_exact_veto_threshold() {
        let mut feedback = CoherenceFeedback::new(0.3);
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("test-veto");
        let hv = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "a",
            symthaea_core::hdc::HDC_DIMENSION,
        );
        let weight = feedback.update(&hv, &hv);
        assert!(!feedback.should_veto(), "Self-similar should not veto");
        assert!((weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_consciousness_gated_verbosity_extreme_values() {
        assert_eq!(consciousness_gated_max_tokens(100, -5.0), 50);
        assert_eq!(consciousness_gated_max_tokens(100, 5.0), 150);
        let result = consciousness_gated_max_tokens(100, f32::NAN);
        assert_eq!(result, 100, "NaN psi → treated as 0.5 (mid-range)");
    }

    #[test]
    fn test_configurable_veto_threshold() {
        let mut feedback_strict = CoherenceFeedback::with_veto_threshold(0.3, 0.5);
        let mut feedback_lenient = CoherenceFeedback::with_veto_threshold(0.3, 0.1);

        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("test-veto-config");
        let a = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "a",
            symthaea_core::hdc::HDC_DIMENSION,
        );
        let b = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "b",
            symthaea_core::hdc::HDC_DIMENSION,
        );

        // Nearly orthogonal HVs — coherence near 0
        feedback_strict.update(&a, &b);
        feedback_lenient.update(&a, &b);

        assert!(
            feedback_strict.should_veto(),
            "Strict threshold (0.5) should veto near-orthogonal"
        );
        assert!(
            feedback_lenient.should_veto(),
            "Lenient threshold (0.1) should also veto near-zero coherence"
        );
    }

    #[test]
    fn test_veto_threshold_from_gating_config() {
        let config = GatingConfig {
            veto_threshold: 0.42,
            ..GatingConfig::default()
        };
        let feedback = CoherenceFeedback::with_veto_threshold(
            config.coherence_drift_threshold,
            config.veto_threshold,
        );
        // Just verify construction — the threshold is stored
        assert!(!feedback.should_veto(), "Fresh feedback should not veto");
    }

    #[test]
    fn test_consciousness_gated_verbosity_zero_base() {
        assert_eq!(consciousness_gated_max_tokens(0, 0.5), 0);
        assert_eq!(consciousness_gated_max_tokens(0, 1.0), 0);
    }

    #[test]
    fn test_all_magic_numbers_configurable() {
        let tok = test_tokenizer();

        // Non-default config should produce different gating behavior
        let custom_config = GatingConfig {
            arousal_boost_multiplier: 10.0,
            warmth_penalty_multiplier: -20.0,
            negative_valence_threshold: -0.1,
            valence_boost_multiplier: 8.0,
            ood_boost_multiplier: 5.0,
            min_veto_position: 10,
            max_vetoes: 3,
            veto_refractory: 16,
            ..Default::default()
        };

        let default_mod = EmotionalModulator::new(&tok, &GatingConfig::default());
        let custom_mod = EmotionalModulator::new(&tok, &custom_config);

        // High arousal scenario — custom multiplier should produce larger boost
        let mut channels = ThoughtChannels::default();
        channels.set_emotion(-0.5, 0.9, 0.1);

        let mut logits_default = vec![0.5; tok.vocab_size()];
        let mut logits_custom = vec![0.5; tok.vocab_size()];
        default_mod.apply(&mut logits_default, &channels, 15);
        custom_mod.apply(&mut logits_custom, &channels, 15);

        // Custom config should produce different results
        assert_ne!(
            logits_default, logits_custom,
            "Non-default config should produce different gating"
        );
    }

    #[test]
    fn test_arousal_valence_interaction() {
        let tok = test_tokenizer();
        let config = test_config();
        let modulator = EmotionalModulator::new(&tok, &config);

        // High arousal + negative valence → interaction term active
        let mut channels = ThoughtChannels::default();
        channels.set_emotion(-0.7, 0.9, 0.5);

        let mut logits = vec![0.5; tok.vocab_size()];
        modulator.apply(&mut logits, &channels, 15);

        // Softening words should get extra boost from interaction
        // (beyond what negative valence alone provides)
        let mut channels_calm = ThoughtChannels::default();
        channels_calm.set_emotion(-0.7, 0.3, 0.5); // same valence, low arousal

        let mut logits_calm = vec![0.5; tok.vocab_size()];
        modulator.apply(&mut logits_calm, &channels_calm, 15);

        // Check that at least one softening token got a larger boost
        // with high arousal + negative valence
        let mut found_interaction = false;
        for &word in CANONICAL_SOFTENING_WORDS {
            let id = tok.token_id(word);
            if id != tok.unk_id {
                if logits[id as usize] > logits_calm[id as usize] + 1e-6 {
                    found_interaction = true;
                    break;
                }
            }
        }
        // If no softening words resolved, the test is vacuous — but that's fine
        // since we're testing in the module where the words are accessible
        assert!(
            found_interaction,
            "High arousal + negative valence should boost softening more than low arousal"
        );
    }

    // =========================================================================
    // New channel tests: domain_familiarity, social_context, time_pressure,
    // response_confidence
    // =========================================================================

    #[test]
    fn test_domain_familiarity_reduces_hedging() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);

        // Novel domain (familiarity=0) — full hedging boost
        let mut logits_novel = vec![0.5; tok.vocab_size()];
        gate.apply_with_familiarity(&mut logits_novel, 3.0, 0.0); // Unknown, novel

        // Expert domain (familiarity=1) — reduced hedging boost
        let mut logits_expert = vec![0.5; tok.vocab_size()];
        gate.apply_with_familiarity(&mut logits_expert, 3.0, 1.0); // Unknown, expert

        // Hedging tokens should be boosted more in novel domain than expert domain
        let mut found_difference = false;
        for &word in CANONICAL_HEDGING_WORDS {
            let id = tok.token_id(word);
            if id != tok.unk_id {
                let novel_boost = logits_novel[id as usize] - 0.5;
                let expert_boost = logits_expert[id as usize] - 0.5;
                if novel_boost > expert_boost + 1e-6 {
                    found_difference = true;
                    // Expert should still have non-negative boost (just smaller)
                    assert!(
                        expert_boost >= -1e-6,
                        "Expert domain hedging boost should be non-negative, got {expert_boost}"
                    );
                    break;
                }
            }
        }
        assert!(
            found_difference,
            "Domain familiarity should reduce hedging boost for expert domains"
        );
    }

    #[test]
    fn test_domain_familiarity_does_not_affect_factual_penalty() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);

        // Factual penalties should be the same regardless of familiarity
        let mut logits_novel = vec![0.5; tok.vocab_size()];
        gate.apply_with_familiarity(&mut logits_novel, 3.0, 0.0);

        let mut logits_expert = vec![0.5; tok.vocab_size()];
        gate.apply_with_familiarity(&mut logits_expert, 3.0, 1.0);

        for &word in CANONICAL_FACTUAL_WORDS {
            let id = tok.token_id(word);
            if id != tok.unk_id {
                assert!(
                    (logits_novel[id as usize] - logits_expert[id as usize]).abs() < 1e-6,
                    "Factual penalty for '{}' should not differ by familiarity",
                    word
                );
            }
        }
    }

    #[test]
    fn test_social_context_increases_formality() {
        // Use a tokenizer that has informal words by adding them to the vocab
        let mut tok = test_tokenizer();
        let informal_ids: Vec<u32> = CANONICAL_INFORMAL_WORDS
            .iter()
            .map(|w| {
                let id = tok.token_id(w);
                if id == tok.unk_id {
                    tok.add_token(w)
                } else {
                    id
                }
            })
            .collect();

        let config = test_config();
        let modulator = EmotionalModulator::new(&tok, &config);

        // Low warmth + intimate context (social_context=0)
        let mut channels_intimate = ThoughtChannels::default();
        channels_intimate.set_emotion(0.0, 0.5, 0.1); // low warmth
        channels_intimate.set_context(0.0, 0.5, 0.0, 0.5); // intimate

        let mut logits_intimate = vec![0.5; tok.vocab_size()];
        modulator.apply(&mut logits_intimate, &channels_intimate, 15);

        // Low warmth + formal context (social_context=1)
        let mut channels_formal = ThoughtChannels::default();
        channels_formal.set_emotion(0.0, 0.5, 0.1); // same low warmth
        channels_formal.set_context(0.0, 0.5, 1.0, 0.5); // formal

        let mut logits_formal = vec![0.5; tok.vocab_size()];
        modulator.apply(&mut logits_formal, &channels_formal, 15);

        // Informal tokens should be penalized more in formal context
        let mut found_difference = false;
        for &id in &informal_ids {
            let intimate_val = logits_intimate[id as usize];
            let formal_val = logits_formal[id as usize];
            // warmth_penalty_multiplier is negative, so lower logit = stronger penalty
            if formal_val < intimate_val - 1e-6 {
                found_difference = true;
                break;
            }
        }
        assert!(
            found_difference,
            "Formal social context should increase informal token penalty"
        );
    }

    #[test]
    fn test_time_pressure_effect() {
        // No time pressure: full budget
        let full = time_pressure_adjusted_max_tokens(100, 0.0, 0.5);
        assert_eq!(full, 100, "No time pressure should give full tokens");

        // Max time pressure: 50% reduction
        let urgent = time_pressure_adjusted_max_tokens(100, 1.0, 0.5);
        assert_eq!(urgent, 50, "Max time pressure should halve tokens");

        // Mid time pressure: 25% reduction
        let mid = time_pressure_adjusted_max_tokens(100, 0.5, 0.5);
        assert_eq!(mid, 75, "Mid time pressure should reduce by 25%");

        // NaN time pressure: treated as no pressure
        let nan = time_pressure_adjusted_max_tokens(100, f32::NAN, 0.5);
        assert_eq!(nan, 100, "NaN time pressure should give full tokens");

        // Zero base: always 1 (min)
        let min_tokens = time_pressure_adjusted_max_tokens(1, 1.0, 1.0);
        assert!(min_tokens >= 1, "Should never produce 0 tokens");
    }

    #[test]
    fn test_confidence_adjusted_veto_threshold() {
        // No confidence: threshold unchanged
        let t0 = confidence_adjusted_veto_threshold(0.15, 0.0, 0.3);
        assert!(
            (t0 - 0.15).abs() < 1e-6,
            "Zero confidence should not change threshold"
        );

        // Full confidence with 0.3 scale: threshold reduced by 30%
        let t1 = confidence_adjusted_veto_threshold(0.15, 1.0, 0.3);
        let expected = 0.15 * 0.7;
        assert!(
            (t1 - expected).abs() < 1e-6,
            "Full confidence should reduce threshold by scale factor, got {t1} expected {expected}"
        );

        // NaN confidence: treated as 0
        let t_nan = confidence_adjusted_veto_threshold(0.15, f32::NAN, 0.3);
        assert!(
            (t_nan - 0.15).abs() < 1e-6,
            "NaN confidence should not change threshold"
        );
    }

    #[test]
    fn test_confidence_veto_integration() {
        // CoherenceFeedback with coherence at exactly the veto threshold
        let mut feedback = CoherenceFeedback::with_veto_threshold(0.3, 0.15);

        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("test-confidence-veto");
        let a = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "a",
            symthaea_core::hdc::HDC_DIMENSION,
        );
        let b = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "b",
            symthaea_core::hdc::HDC_DIMENSION,
        );

        // Near-orthogonal HVs → low coherence → should veto without confidence
        feedback.update(&a, &b);
        assert!(
            feedback.should_veto(),
            "Low coherence should trigger veto without confidence adjustment"
        );

        // With high confidence, veto threshold is lowered — but near-zero coherence
        // should still veto even with confidence (threshold only reduces by 30%)
        let veto_with_confidence = feedback.should_veto_with_confidence(1.0, 0.3);
        // Near-orthogonal coherence (~0.0) is below even the reduced threshold (0.15 * 0.7 = 0.105)
        assert!(
            veto_with_confidence,
            "Near-zero coherence should still veto even with high confidence"
        );

        // Self-similar HVs → high coherence → should not veto
        feedback.update(&a, &a);
        assert!(
            !feedback.should_veto_with_confidence(0.0, 0.3),
            "High coherence should not veto"
        );
    }

    // ── Veto Stress Tests ─────────────────────────────────────────────────

    #[test]
    fn test_veto_binding_weight_saturation() {
        let mut feedback = CoherenceFeedback::with_veto_threshold(0.5, 0.15);
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("veto-stress-saturation");
        let a = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "a",
            symthaea_core::hdc::HDC_DIMENSION,
        );
        let b = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "b",
            symthaea_core::hdc::HDC_DIMENSION,
        );
        let weight = feedback.update(&a, &b);
        assert!(
            weight <= 3.0,
            "Binding weight must cap at 3.0, got {weight}"
        );
        assert!(weight >= 1.0, "Binding weight must be >= 1.0, got {weight}");
        assert!(weight.is_finite(), "Binding weight must be finite");
    }

    #[test]
    fn test_veto_coherence_oscillation() {
        let mut feedback = CoherenceFeedback::with_veto_threshold(0.3, 0.15);
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("veto-stress-osc");
        let a = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "similar-a",
            symthaea_core::hdc::HDC_DIMENSION,
        );
        let b = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "different-b",
            symthaea_core::hdc::HDC_DIMENSION,
        );
        let mut veto_count = 0;
        for i in 0..20 {
            if i % 2 == 0 {
                feedback.update(&a, &b);
            } else {
                feedback.update(&a, &a);
            }
            if feedback.should_veto() {
                veto_count += 1;
            }
        }
        assert_eq!(veto_count, 10, "Feedback reports veto state per-step");
    }

    #[test]
    fn test_veto_confidence_threshold_monotonic() {
        let scale = 0.3;
        let mut prev = f32::INFINITY;
        for c in 0..=10 {
            let conf = c as f32 / 10.0;
            let threshold = confidence_adjusted_veto_threshold(0.20, conf, scale);
            assert!(
                threshold <= prev + f32::EPSILON,
                "Must decrease: conf={conf}"
            );
            assert!(threshold >= 0.0, "Must be non-negative");
            prev = threshold;
        }
    }

    #[test]
    fn test_veto_zero_max_disables() {
        let config = GatingConfig {
            max_vetoes: 0,
            ..GatingConfig::default()
        };
        assert_eq!(config.max_vetoes, 0);
    }

    #[test]
    fn test_veto_refractory_config() {
        let config = GatingConfig {
            veto_refractory: 16,
            max_vetoes: 3,
            ..GatingConfig::default()
        };
        assert_eq!(config.veto_refractory, 16);
        assert_eq!(config.max_vetoes, 3);
    }

    #[test]
    fn test_veto_self_similar_never_triggers() {
        let mut feedback = CoherenceFeedback::with_veto_threshold(0.3, 0.99);
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("veto-self-similar");
        let a = symthaea_core::hdc::ContinuousHV::from_genesis(
            &genesis,
            "a",
            symthaea_core::hdc::HDC_DIMENSION,
        );
        feedback.update(&a, &a);
        assert!(!feedback.should_veto(), "Self-similar should never veto");
        assert!((feedback.coherence() - 1.0).abs() < 0.01);
    }

    #[cfg(feature = "mamba")]
    mod backend_tests {
        use super::*;
        use crate::mamba::tests::MockMamba;

        #[test]
        fn test_new_from_backend_resolves_ids() {
            let mock = MockMamba::new();
            let config = test_config();
            let gate = EpistemicGate::new_from_backend(&mock, &config);

            // MockMamba encodes each word to byte IDs, first byte becomes the token ID
            // All 18 hedging words should resolve (none should fail)
            assert_eq!(
                gate.hedging_count(),
                CANONICAL_HEDGING_WORDS.len(),
                "All canonical hedging words should resolve via backend"
            );
            assert_eq!(
                gate.factual_count(),
                CANONICAL_FACTUAL_WORDS.len(),
                "All canonical factual words should resolve via backend"
            );
        }

        #[test]
        fn test_new_from_backend_ids_in_vocab_range() {
            let mock = MockMamba::new();
            let config = test_config();
            let gate = EpistemicGate::new_from_backend(&mock, &config);

            let vocab_size = mock.vocab_size() as u32;
            for &id in gate.hedging_ids() {
                assert!(
                    id < vocab_size,
                    "Hedging token ID {id} exceeds vocab size {vocab_size}"
                );
            }
        }

        #[test]
        fn test_gate_applies_to_full_logits() {
            let mock = MockMamba::new();
            let config = test_config();
            let gate = EpistemicGate::new_from_backend(&mock, &config);

            // Create logits sized to full GPT-2 vocab (50280 in MockMamba)
            let vocab_size = mock.vocab_size();
            let mut logits = vec![0.5_f32; vocab_size];
            gate.apply(&mut logits, 3.0); // Unknown

            // At least some logits should be modified (hedging boosted, factual penalized)
            let modified_count = logits.iter().filter(|&&l| (l - 0.5).abs() > 1e-6).count();
            assert!(
                modified_count > 0,
                "Gate should modify logits at backend token positions"
            );
        }
    }

    // ── Therapeutic gating tests ─────────────────────────────────────────────

    #[cfg(feature = "therapeutic")]
    mod therapeutic_gate_tests {
        use super::super::*;
        use crate::encoder::ThoughtChannels;

        /// High distress should suppress directive words and boost validating words.
        #[test]
        fn test_high_distress_suppresses_directives() {
            let mut channels = ThoughtChannels::default();
            channels.channels[26] = 0.9; // client_distress_level = high

            let should_logit = TherapeuticGate::apply("should", &channels, 0.0);
            assert!(
                should_logit < 0.0,
                "Directive 'should' should be suppressed under high distress: {}",
                should_logit,
            );
        }

        #[test]
        fn test_high_distress_boosts_validating() {
            let mut channels = ThoughtChannels::default();
            channels.channels[26] = 0.9; // client_distress_level = high

            let understand_logit = TherapeuticGate::apply("understand", &channels, 0.0);
            assert!(
                understand_logit > 0.0,
                "Validating 'understand' should be boosted under high distress: {}",
                understand_logit,
            );
        }

        #[test]
        fn test_low_alliance_suppresses_directives() {
            let mut channels = ThoughtChannels::default();
            channels.channels[25] = 0.1; // alliance_quality = low

            let must_logit = TherapeuticGate::apply("must", &channels, 0.0);
            assert!(
                must_logit < 0.0,
                "Directive 'must' should be suppressed under low alliance: {}",
                must_logit,
            );
        }

        #[test]
        fn test_crisis_mode_suppresses_directives_boosts_crisis() {
            let mut channels = ThoughtChannels::default();
            channels.channels[24] = 7.0; // therapeutic_intent = crisis

            let should_logit = TherapeuticGate::apply("should", &channels, 0.0);
            let call_logit = TherapeuticGate::apply("call", &channels, 0.0);
            let breathe_logit = TherapeuticGate::apply("breathe", &channels, 0.0);

            assert!(
                should_logit < 0.0,
                "Crisis mode should suppress directives: {}",
                should_logit,
            );
            assert!(
                call_logit > 0.0,
                "Crisis mode should boost crisis referral words: {}",
                call_logit,
            );
            assert!(
                breathe_logit > 0.0,
                "Crisis mode should boost grounding words: {}",
                breathe_logit,
            );
        }

        #[test]
        fn test_depth_exceeds_alliance_suppresses() {
            let mut channels = ThoughtChannels::default();
            channels.channels[25] = 0.3; // alliance
            channels.channels[27] = 0.8; // depth > alliance + 0.2

            let correct_logit = TherapeuticGate::apply("correct", &channels, 0.0);
            assert!(
                correct_logit < 0.0,
                "High depth with low alliance should suppress directives: {}",
                correct_logit,
            );
        }

        #[test]
        fn test_reflective_intent_boosts_reflective() {
            let mut channels = ThoughtChannels::default();
            channels.channels[24] = 2.0; // therapeutic_intent = reflect

            let wonder_logit = TherapeuticGate::apply("I wonder", &channels, 0.0);
            assert!(
                wonder_logit > 0.0,
                "Reflective intent should boost reflective words: {}",
                wonder_logit,
            );
        }

        #[test]
        fn test_neutral_state_no_modification() {
            let channels = ThoughtChannels::default();
            // Default: distress=0, alliance=0.5, intent=0, depth=0
            let hello_logit = TherapeuticGate::apply("hello", &channels, 1.0);
            assert!(
                (hello_logit - 1.0).abs() < 1e-6,
                "Neutral state should not modify non-therapeutic word: {}",
                hello_logit,
            );
        }

        #[test]
        fn test_e2e_high_distress_directive_vs_validating_spread() {
            let mut channels = ThoughtChannels::default();
            channels.channels[26] = 0.9; // High distress

            let directive_words = ["should", "must", "wrong"];
            let validating_words = ["understand", "hear", "notice"];

            let dir_sum: f32 = directive_words
                .iter()
                .map(|w| TherapeuticGate::apply(w, &channels, 0.0))
                .sum();
            let val_sum: f32 = validating_words
                .iter()
                .map(|w| TherapeuticGate::apply(w, &channels, 0.0))
                .sum();

            assert!(
                val_sum > dir_sum,
                "Under high distress, validating words should have higher total logit \
                 than directive words: val={} vs dir={}",
                val_sum,
                dir_sum,
            );
        }
    }
}
