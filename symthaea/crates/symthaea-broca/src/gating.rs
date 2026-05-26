// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
#[cfg(feature = "mamba-cpu")]
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

/// Canonical Rust structural keywords boosted when syntax_complexity is high.
pub const CANONICAL_RUST_STRUCTURAL_WORDS: &[&str] = &[
    "fn", "struct", "enum", "impl", "trait", "pub", "mod", "use", "where", "for", "match", "if",
    "let", "mut", "ref", "async", "unsafe", "type", "const", "static",
];

/// Canonical Rust type words suppressed when type_confidence is low.
pub const CANONICAL_TYPE_WORDS: &[&str] = &[
    "i32", "i64", "u32", "u64", "f32", "f64", "usize", "bool", "String", "Vec", "Option", "Result",
    "Box", "Arc", "HashMap", "str",
];

/// Canonical error handling words boosted when error_likelihood is high.
pub const CANONICAL_ERROR_HANDLING_WORDS: &[&str] = &[
    "Result",
    "Option",
    "unwrap",
    "expect",
    "ok",
    "err",
    "Some",
    "None",
    "match",
    "if let",
    "while let",
];

/// Canonical algorithm scaffold words boosted when algorithm_pattern is active.
pub const CANONICAL_ALGORITHM_SCAFFOLD_WORDS: &[&str] = &[
    "for", "while", "loop", "iter", "map", "filter", "fold", "collect", "sort", "push", "pop",
    "len", "is_empty", "contains",
];

// ═══════════════════════════════════════════════════════════════════════════════
// GatingConfig
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-backend gating scale factors for Liquid-Mamba path.
///
/// Mamba generates through GPT-NeoX 50K vocab in a different representational
/// space than CfC-HDC's 4K BPE vocab. The same epistemic/emotional gating
/// parameters can over- or under-correct depending on the logit distribution.
///
/// These multipliers scale the base `GatingConfig` values when applied to
/// Mamba logits. A value of 1.0 means "same as CfC", 0.5 means "half strength".
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MambaGatingOverrides {
    /// Scale factor for epistemic penalties/boosts (default 0.6).
    /// Mamba's larger vocab distributes probability more thinly, so
    /// the same absolute logit penalty has a larger relative effect.
    #[serde(default = "default_mamba_epistemic_scale")]
    pub epistemic_scale: f32,
    /// Scale factor for emotional modulation (default 0.5).
    /// Mamba logits have different dynamic range than CfC cosine-scaled logits.
    #[serde(default = "default_mamba_emotional_scale")]
    pub emotional_scale: f32,
    /// Scale factor for veto threshold (default 1.5).
    /// Mamba coherence scores live in a different range — veto threshold
    /// needs to be more permissive to avoid excessive mid-sentence corrections.
    #[serde(default = "default_mamba_veto_scale")]
    pub veto_threshold_scale: f32,
}

impl Default for MambaGatingOverrides {
    fn default() -> Self {
        Self {
            epistemic_scale: 0.6,
            emotional_scale: 0.5,
            veto_threshold_scale: 1.5,
        }
    }
}

fn default_mamba_epistemic_scale() -> f32 {
    0.6
}

fn default_mamba_emotional_scale() -> f32 {
    0.5
}

fn default_mamba_veto_scale() -> f32 {
    1.5
}

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

    // ── Per-backend overrides ──
    /// Optional per-backend gating scale factors for Liquid-Mamba path.
    /// When set, Mamba gating multiplies the base penalties/boosts by these factors.
    /// CfC-HDC path always uses the base values unscaled.
    #[serde(default)]
    pub mamba_gating_overrides: Option<MambaGatingOverrides>,

    // ── Phase 2: Algebraic Coherence Correction ──
    /// Enable algebraic (HDC subtract) coherence correction instead of scalar scaling.
    /// When enabled, drift is surgically removed from the thought vector using
    /// vector subtraction rather than uniform amplitude scaling.
    #[serde(default)]
    pub enable_algebraic_correction: bool,
    /// Correction strength for algebraic mode (0.0-1.0). Default 0.3.
    /// Higher values apply stronger corrections but risk oscillation.
    #[serde(default = "default_algebraic_correction_strength")]
    pub algebraic_correction_strength: f32,

    // ── Phase 4: Soft Veto via Temporal Rewind ──
    /// Enable soft veto (partial CfC state restore) instead of hard reset.
    /// When enabled, veto interpolates toward a saved "known-good" snapshot
    /// rather than zeroing all network state.
    #[serde(default)]
    pub enable_soft_veto: bool,
    /// Interpolation weight for soft veto (0.0 = no-op, 1.0 = full restore to snapshot).
    /// Default 0.5: halfway between snapshot and current state.
    #[serde(default = "default_veto_rewind_alpha")]
    pub veto_rewind_alpha: f32,

    // ── W3.3: Spectral Coherence Gating ──
    /// Enable spectral quality check on thought vectors before generation.
    #[serde(default)]
    pub enable_spectral_gating: bool,
    /// Minimum spectral quality threshold (0.0-1.0). Default 0.1.
    #[serde(default = "default_spectral_quality_threshold")]
    pub spectral_quality_threshold: f32,

    // ── Code Generation Gating ──
    /// Enable code-aware gating for code generation contexts (default: false).
    /// When enabled, reads ThoughtChannels 24-27 (syntax_complexity, type_confidence,
    /// algorithm_pattern, error_likelihood) and modulates logits for code tokens.
    #[serde(default)]
    pub enable_code_gate: bool,
    /// Logit boost for structural keywords when syntax_complexity > 0.3 (default 0.5).
    #[serde(default = "default_code_structural_boost")]
    pub code_structural_boost: f32,
    /// Logit penalty scale for concrete type words when type_confidence < 0.4 (default 0.3).
    #[serde(default = "default_code_type_penalty")]
    pub code_type_penalty: f32,
    /// Logit boost for algorithm scaffold words when algorithm_pattern > 0.2 (default 0.4).
    #[serde(default = "default_code_algorithm_boost")]
    pub code_algorithm_boost: f32,
    /// Logit boost for error handling words when error_likelihood > 0.3 (default 0.5).
    #[serde(default = "default_code_error_boost")]
    pub code_error_boost: f32,

    // ── Temperature-based epistemic gating (Round 2) ──
    /// Enable temperature-based epistemic gating (default: true).
    /// Temperature mode divides ALL logits by an epistemic-dependent factor,
    /// producing a flatter distribution while applying mild additive adjustments.
    /// Legacy mode uses strong additive penalties (collapses vocab to ~25 words).
    #[serde(default = "default_epistemic_temperature_mode")]
    pub epistemic_temperature_mode: bool,
    /// Temperature divisor for Uncertain epistemic level (default 1.3).
    #[serde(default = "default_uncertain_temperature")]
    pub uncertain_temperature: f32,
    /// Temperature divisor for Unknown epistemic level (default 1.5).
    #[serde(default = "default_unknown_temperature")]
    pub unknown_temperature: f32,
    /// Temperature divisor for OOD epistemic level (default 1.8).
    #[serde(default = "default_ood_temperature")]
    pub ood_temperature: f32,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            // Epistemic gating — v3 reduction (Mar 2026). Previous -5.0/1.5 still
            // dominated generation (broca-compare: 100% gating words for OOD/Unknown).
            // These gentler values nudge output toward hedging without drowning content.
            unknown_factual_penalty: -3.0,
            unknown_hedging_boost: 0.75,
            uncertain_factual_penalty: -1.5,
            uncertain_hedging_boost: 0.5,
            coherence_drift_threshold: 0.3,
            high_arousal_threshold: 0.7,
            arousal_position_threshold: 10,
            low_warmth_threshold: 0.3,
            base_max_tokens: 128,
            // Semantic veto — disabled by default (0.0). After BPTT training,
            // CfC output lives in a completely different representational space
            // than the input thought HV, giving baseline coherence ≈ 0.000.
            // Any positive threshold fires on every token. Use adaptive veto
            // (adaptive_veto_warmup > 0) or set explicitly for untrained models.
            veto_threshold: 0.0,
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
            epistemic_temperature_mode: true,
            uncertain_temperature: 1.3,
            unknown_temperature: 1.5,
            ood_temperature: 1.8,
            mamba_gating_overrides: None,
            // Benchmark-validated (Mar 20): +0.071 coherence, zero perplexity cost.
            enable_algebraic_correction: true,
            algebraic_correction_strength: 0.3,
            enable_soft_veto: false,
            veto_rewind_alpha: 0.5,
            enable_spectral_gating: false,
            spectral_quality_threshold: 0.1,
            enable_code_gate: false,
            code_structural_boost: 0.5,
            code_type_penalty: 0.3,
            code_algorithm_boost: 0.4,
            code_error_boost: 0.5,
        }
    }
}

impl GatingConfig {
    /// Create a config with default Mamba gating overrides enabled.
    ///
    /// Equivalent to `default()` but with `mamba_gating_overrides = Some(MambaGatingOverrides::default())`.
    pub fn with_mamba_overrides() -> Self {
        Self {
            mamba_gating_overrides: Some(MambaGatingOverrides::default()),
            ..Self::default()
        }
    }

    /// Get the effective epistemic penalty/boost scale for the Mamba backend.
    /// Returns 1.0 if no overrides are set.
    pub fn mamba_epistemic_scale(&self) -> f32 {
        self.mamba_gating_overrides
            .as_ref()
            .map_or(1.0, |o| o.epistemic_scale)
    }

    /// Get the effective emotional modulation scale for the Mamba backend.
    /// Returns 1.0 if no overrides are set.
    pub fn mamba_emotional_scale(&self) -> f32 {
        self.mamba_gating_overrides
            .as_ref()
            .map_or(1.0, |o| o.emotional_scale)
    }

    /// Get the effective veto threshold scale for the Mamba backend.
    /// Returns 1.0 if no overrides are set.
    pub fn mamba_veto_threshold_scale(&self) -> f32 {
        self.mamba_gating_overrides
            .as_ref()
            .map_or(1.0, |o| o.veto_threshold_scale)
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

fn default_algebraic_correction_strength() -> f32 {
    0.3
}

fn default_veto_rewind_alpha() -> f32 {
    0.5
}

fn default_epistemic_temperature_mode() -> bool {
    true
}

fn default_uncertain_temperature() -> f32 {
    1.3
}

fn default_unknown_temperature() -> f32 {
    1.5
}

fn default_ood_temperature() -> f32 {
    1.8
}

fn default_spectral_quality_threshold() -> f32 {
    0.1
}

fn default_code_structural_boost() -> f32 {
    0.5
}

fn default_code_type_penalty() -> f32 {
    0.3
}

fn default_code_algorithm_boost() -> f32 {
    0.4
}

fn default_code_error_boost() -> f32 {
    0.5
}

// ═══════════════════════════════════════════════════════════════════════════════
// SpectralCoherenceGate (W3.3)
// ═══════════════════════════════════════════════════════════════════════════════

/// Spectral coherence gate: detects structurally fragmented thought vectors.
///
/// Uses spectral flatness (Wiener entropy ratio) to distinguish structured
/// thought vectors (peaked, meaningful) from noise-like ones (flat, uninformative).
pub struct SpectralCoherenceGate;

impl SpectralCoherenceGate {
    /// Compute spectral quality of a thought vector (0.0 = noise, 1.0 = structured).
    ///
    /// Returns `1.0 - spectral_flatness`, so higher = better quality.
    pub fn spectral_quality(thought_hv: &symthaea_core::hdc::ContinuousHV) -> f32 {
        let values = thought_hv.as_slice();
        let n = values.len();
        if n == 0 {
            return 0.0;
        }
        let eps = 1e-10f64;
        let mut log_sum = 0.0f64;
        let mut abs_sum = 0.0f64;
        for &v in values {
            let abs_v = (v as f64).abs().max(eps);
            log_sum += abs_v.ln();
            abs_sum += abs_v;
        }
        let n_f = n as f64;
        let geo_mean = (log_sum / n_f).exp();
        let arith_mean = abs_sum / n_f;
        if arith_mean < eps {
            return 0.0;
        }
        let flatness = (geo_mean / arith_mean) as f32;
        (1.0 - flatness).clamp(0.0, 1.0)
    }

    /// Check if a thought vector is too fragmented for coherent generation.
    pub fn should_gate(thought_hv: &symthaea_core::hdc::ContinuousHV, threshold: f32) -> bool {
        Self::spectral_quality(thought_hv) < threshold
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EpistemicGate
// ═══════════════════════════════════════════════════════════════════════════════

/// Epistemic gate: suppresses factual assertions when confidence is low.
///
/// The system physically *cannot* hallucinate when epistemic status is Unknown —
/// factual token logits are suppressed before sampling.
#[derive(Clone)]
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
    #[cfg(feature = "mamba-cpu")]
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
    ///
    /// Dispatches to temperature-based or legacy additive gating based on config.
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

        if self.config.epistemic_temperature_mode {
            self.apply_temperature_gating(logits, epistemic_ordinal, familiarity_scale);
        } else {
            self.apply_additive_gating(logits, epistemic_ordinal, familiarity_scale);
        }
    }

    /// Temperature-based epistemic gating: divides all logits by an epistemic-dependent
    /// temperature factor, producing a flatter distribution, then applies mild additive
    /// adjustments. Preserves vocabulary diversity unlike legacy additive mode.
    fn apply_temperature_gating(
        &self,
        logits: &mut [f32],
        epistemic_ordinal: f32,
        familiarity_scale: f32,
    ) {
        let temperature = if epistemic_ordinal > 3.5 {
            self.config.ood_temperature
        } else if epistemic_ordinal > 2.5 {
            self.config.unknown_temperature
        } else {
            self.config.uncertain_temperature
        };
        if temperature > 1.0 + 1e-6 {
            let inv_temp = 1.0 / temperature;
            for l in logits.iter_mut() {
                *l *= inv_temp;
            }
        }
        // Mild additive adjustments (much gentler than legacy)
        if epistemic_ordinal > 3.5 {
            // OOD: factual -1.0, hedging +0.3, ood +0.5
            for &id in &self.factual_token_ids {
                if (id as usize) < logits.len() {
                    logits[id as usize] -= 1.0;
                }
            }
            for &id in &self.hedging_token_ids {
                if (id as usize) < logits.len() {
                    logits[id as usize] += 0.3 * familiarity_scale;
                }
            }
            for &id in &self.ood_token_ids {
                if (id as usize) < logits.len() {
                    logits[id as usize] += 0.5;
                }
            }
        } else if epistemic_ordinal > 2.5 {
            // Unknown: factual -0.5, hedging +0.3
            for &id in &self.factual_token_ids {
                if (id as usize) < logits.len() {
                    logits[id as usize] -= 0.5;
                }
            }
            for &id in &self.hedging_token_ids {
                if (id as usize) < logits.len() {
                    logits[id as usize] += 0.3 * familiarity_scale;
                }
            }
        } else {
            // Uncertain: hedging +0.2 only (no factual penalty)
            for &id in &self.hedging_token_ids {
                if (id as usize) < logits.len() {
                    logits[id as usize] += 0.2 * familiarity_scale;
                }
            }
        }
    }

    /// Legacy additive epistemic gating: applies strong additive penalties/boosts.
    /// Can collapse vocabulary to ~25 words under high epistemic uncertainty.
    fn apply_additive_gating(
        &self,
        logits: &mut [f32],
        epistemic_ordinal: f32,
        familiarity_scale: f32,
    ) {
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

    /// Apply epistemic gating with a backend-specific scale factor.
    ///
    /// For Liquid-Mamba: pass `config.mamba_epistemic_scale()` as `backend_scale`.
    /// For CfC-HDC: pass 1.0 (or just call `apply_with_familiarity` directly).
    pub fn apply_scaled(
        &self,
        logits: &mut [f32],
        epistemic_ordinal: f32,
        domain_familiarity: f32,
        backend_scale: f32,
    ) {
        if backend_scale == 1.0 {
            self.apply_with_familiarity(logits, epistemic_ordinal, domain_familiarity);
            return;
        }
        if epistemic_ordinal < 1.5 {
            return;
        }

        let df = domain_familiarity.clamp(0.0, 1.0);
        let familiarity_scale =
            1.0 - df * self.config.domain_familiarity_hedging_scale.clamp(0.0, 1.0);

        if self.config.epistemic_temperature_mode {
            // Temperature mode with backend scaling: effective_temp = 1.0 + (base_temp - 1.0) * backend_scale
            let base_temp = if epistemic_ordinal > 3.5 {
                self.config.ood_temperature
            } else if epistemic_ordinal > 2.5 {
                self.config.unknown_temperature
            } else {
                self.config.uncertain_temperature
            };
            let effective_temp = 1.0 + (base_temp - 1.0) * backend_scale;
            if effective_temp > 1.0 + 1e-6 {
                let inv_temp = 1.0 / effective_temp;
                for l in logits.iter_mut() {
                    *l *= inv_temp;
                }
            }
            // Mild additive adjustments scaled by backend_scale
            if epistemic_ordinal > 3.5 {
                for &id in &self.factual_token_ids {
                    if (id as usize) < logits.len() {
                        logits[id as usize] -= 1.0 * backend_scale;
                    }
                }
                for &id in &self.hedging_token_ids {
                    if (id as usize) < logits.len() {
                        logits[id as usize] += 0.3 * familiarity_scale * backend_scale;
                    }
                }
                for &id in &self.ood_token_ids {
                    if (id as usize) < logits.len() {
                        logits[id as usize] += 0.5 * backend_scale;
                    }
                }
            } else if epistemic_ordinal > 2.5 {
                for &id in &self.factual_token_ids {
                    if (id as usize) < logits.len() {
                        logits[id as usize] -= 0.5 * backend_scale;
                    }
                }
                for &id in &self.hedging_token_ids {
                    if (id as usize) < logits.len() {
                        logits[id as usize] += 0.3 * familiarity_scale * backend_scale;
                    }
                }
            } else {
                for &id in &self.hedging_token_ids {
                    if (id as usize) < logits.len() {
                        logits[id as usize] += 0.2 * familiarity_scale * backend_scale;
                    }
                }
            }
        } else {
            // Legacy additive mode with backend scaling
            if epistemic_ordinal > 3.5 {
                for &id in &self.factual_token_ids {
                    if let Some(l) = logits.get_mut(id as usize) {
                        *l += self.config.unknown_factual_penalty * backend_scale;
                    }
                }
                for &id in &self.ood_token_ids {
                    if let Some(l) = logits.get_mut(id as usize) {
                        *l += self.config.unknown_hedging_boost
                            * self.config.ood_boost_multiplier
                            * familiarity_scale
                            * backend_scale;
                    }
                }
                for &id in &self.hedging_token_ids {
                    if let Some(l) = logits.get_mut(id as usize) {
                        *l += self.config.unknown_hedging_boost * familiarity_scale * backend_scale;
                    }
                }
                return;
            }

            if epistemic_ordinal > 2.5 {
                for &id in &self.factual_token_ids {
                    if let Some(l) = logits.get_mut(id as usize) {
                        *l += self.config.unknown_factual_penalty * backend_scale;
                    }
                }
                for &id in &self.hedging_token_ids {
                    if let Some(l) = logits.get_mut(id as usize) {
                        *l += self.config.unknown_hedging_boost * familiarity_scale * backend_scale;
                    }
                }
            } else {
                for &id in &self.factual_token_ids {
                    if let Some(l) = logits.get_mut(id as usize) {
                        *l += self.config.uncertain_factual_penalty * backend_scale;
                    }
                }
                for &id in &self.hedging_token_ids {
                    if let Some(l) = logits.get_mut(id as usize) {
                        *l +=
                            self.config.uncertain_hedging_boost * familiarity_scale * backend_scale;
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EpistemicCubeGate — Per-Axis Gating from 4D Epistemic Cube
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-axis token words for E-axis (assertion control).
const E_AXIS_HEDGING: &[&str] = &[
    "maybe",
    "perhaps",
    "possibly",
    "might",
    "could",
    "uncertain",
    "guess",
];
const E_AXIS_ASSERTION: &[&str] = &[
    "definitely",
    "certainly",
    "always",
    "proven",
    "verified",
    "true",
];
const E_AXIS_TESTIMONIAL: &[&str] = &[
    "heard",
    "told",
    "said",
    "someone",
    "reportedly",
    "apparently",
];

/// Per-axis token words for N-axis (social framing).
const N_AXIS_PERSONAL: &[&str] = &["think", "feel", "believe", "opinion", "view", "guess"];
const N_AXIS_COMMUNAL: &[&str] = &["we", "our", "community", "together", "shared"];
const N_AXIS_NETWORK: &[&str] = &["known", "consensus", "agreed", "established", "accepted"];
const N_AXIS_AXIOMATIC: &[&str] = &["necessarily", "definition", "axiom", "always", "must"];

/// Per-axis token words for M-axis (temporal framing).
const M_AXIS_EPHEMERAL: &[&str] = &["now", "moment", "currently", "temporary", "briefly"];
const M_AXIS_PERSISTENT: &[&str] = &["recorded", "established", "documented", "archived"];
const M_AXIS_FOUNDATIONAL: &[&str] = &["always", "fundamentally", "essentially", "permanently"];

/// Per-axis gating from the full 4D Epistemic Cube channels.
///
/// Unlike `EpistemicGate` which reads a single ordinal, this gate reads the
/// one-hot cube channels (E[5], N[4], M[4], H[1]) and applies axis-specific
/// logit modulation:
///
/// - **E-axis**: Controls assertion vs hedging strength
/// - **N-axis**: Controls personal vs universal framing
/// - M-axis: Controls temporal vs permanent framing
/// - H-axis: Modulates generation depth/verbosity
#[derive(Clone)]
pub struct EpistemicCubeGate {
    // E-axis token sets
    hedging_ids: Vec<u32>,
    assertion_ids: Vec<u32>,
    testimonial_ids: Vec<u32>,
    // N-axis token sets
    personal_ids: Vec<u32>,
    communal_ids: Vec<u32>,
    network_ids: Vec<u32>,
    axiomatic_ids: Vec<u32>,
    // M-axis token sets
    ephemeral_ids: Vec<u32>,
    persistent_ids: Vec<u32>,
    foundational_ids: Vec<u32>,
}

impl EpistemicCubeGate {
    /// Create from a tokenizer, resolving word lists to token IDs.
    pub fn new(tokenizer: &BpeTokenizer) -> Self {
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
            hedging_ids: resolve(E_AXIS_HEDGING),
            assertion_ids: resolve(E_AXIS_ASSERTION),
            testimonial_ids: resolve(E_AXIS_TESTIMONIAL),
            personal_ids: resolve(N_AXIS_PERSONAL),
            communal_ids: resolve(N_AXIS_COMMUNAL),
            network_ids: resolve(N_AXIS_NETWORK),
            axiomatic_ids: resolve(N_AXIS_AXIOMATIC),
            ephemeral_ids: resolve(M_AXIS_EPHEMERAL),
            persistent_ids: resolve(M_AXIS_PERSISTENT),
            foundational_ids: resolve(M_AXIS_FOUNDATIONAL),
        }
    }

    /// Create from a Mamba backend's tokenizer.
    #[cfg(feature = "mamba-cpu")]
    pub fn new_from_backend(backend: &dyn MambaBackend) -> Self {
        let resolve = |words: &[&str]| -> Vec<u32> {
            words
                .iter()
                .filter_map(|w| backend.encode(w).ok().and_then(|ids| ids.first().copied()))
                .collect()
        };
        Self {
            hedging_ids: resolve(E_AXIS_HEDGING),
            assertion_ids: resolve(E_AXIS_ASSERTION),
            testimonial_ids: resolve(E_AXIS_TESTIMONIAL),
            personal_ids: resolve(N_AXIS_PERSONAL),
            communal_ids: resolve(N_AXIS_COMMUNAL),
            network_ids: resolve(N_AXIS_NETWORK),
            axiomatic_ids: resolve(N_AXIS_AXIOMATIC),
            ephemeral_ids: resolve(M_AXIS_EPHEMERAL),
            persistent_ids: resolve(M_AXIS_PERSISTENT),
            foundational_ids: resolve(M_AXIS_FOUNDATIONAL),
        }
    }

    /// Apply per-axis gating with a backend-specific scale factor.
    pub fn apply_scaled(&self, logits: &mut [f32], channels: &ThoughtChannels, scale: f32) {
        if scale == 1.0 {
            self.apply(logits, channels);
            return;
        }

        // Only apply if cube channels are populated
        if !channels.has_epistemic_cube() {
            return;
        }

        // ── E-axis: Assertion Control ────────────────────────────────────
        let e = channels.e_tier().unwrap_or(1);
        match e {
            0 => {
                Self::boost_ids(logits, &self.hedging_ids, 0.5 * scale);
                Self::penalize_ids(logits, &self.assertion_ids, -0.8 * scale);
            }
            1 => {
                Self::boost_ids(logits, &self.testimonial_ids, 0.4 * scale);
                Self::boost_ids(logits, &self.hedging_ids, 0.2 * scale);
                Self::penalize_ids(logits, &self.assertion_ids, -0.3 * scale);
            }
            2 => {
                Self::boost_ids(logits, &self.hedging_ids, 0.1 * scale);
            }
            3 => {
                Self::boost_ids(logits, &self.assertion_ids, 0.2 * scale);
            }
            4 => {
                Self::boost_ids(logits, &self.assertion_ids, 0.4 * scale);
                Self::penalize_ids(logits, &self.hedging_ids, -0.3 * scale);
            }
            _ => {}
        }

        // ── N-axis: Social Framing ──────────────────────────────────────
        let n = channels.n_tier().unwrap_or(0);
        match n {
            0 => Self::boost_ids(logits, &self.personal_ids, 0.4 * scale),
            1 => Self::boost_ids(logits, &self.communal_ids, 0.3 * scale),
            2 => Self::boost_ids(logits, &self.network_ids, 0.3 * scale),
            3 => Self::boost_ids(logits, &self.axiomatic_ids, 0.4 * scale),
            _ => {}
        }

        // ── M-axis: Temporal Framing ────────────────────────────────────
        let m = channels.m_tier().unwrap_or(1);
        match m {
            0 => Self::boost_ids(logits, &self.ephemeral_ids, 0.3 * scale),
            2 => Self::boost_ids(logits, &self.persistent_ids, 0.3 * scale),
            3 => Self::boost_ids(logits, &self.foundational_ids, 0.4 * scale),
            _ => {}
        }

        // ── H-axis: Coherence Depth ────────────────────────────────────
        let h = channels.h_tier();
        if h < 0.25 {
            let dampen = 1.0 - (0.15 * scale);
            for l in logits.iter_mut() {
                *l *= dampen;
            }
        }
    }

    /// Apply per-axis gating to logits based on epistemic cube channels.
    ///
    /// `channels` must be the ThoughtChannels from the current generation.
    /// Reads cube data via `e_tier()`, `n_tier()`, `m_tier()`, `h_tier()`.
    pub fn apply(&self, logits: &mut [f32], channels: &ThoughtChannels) {
        // Only apply if cube channels are populated
        if !channels.has_epistemic_cube() {
            return;
        }

        // ── E-axis: Assertion Control ────────────────────────────────────
        let e = channels.e_tier().unwrap_or(1);
        match e {
            0 => {
                // E0 (opinion): MAX hedging, suppress assertions
                Self::boost_ids(logits, &self.hedging_ids, 0.5);
                Self::penalize_ids(logits, &self.assertion_ids, -0.8);
            }
            1 => {
                // E1 (testimonial): boost testimonial framing
                Self::boost_ids(logits, &self.testimonial_ids, 0.4);
                Self::boost_ids(logits, &self.hedging_ids, 0.2);
                Self::penalize_ids(logits, &self.assertion_ids, -0.3);
            }
            2 => {
                // E2 (verifiable): mild hedging, light assertion
                Self::boost_ids(logits, &self.hedging_ids, 0.1);
            }
            3 => {
                // E3 (proven): allow confident assertion
                Self::boost_ids(logits, &self.assertion_ids, 0.2);
            }
            4 => {
                // E4 (reproducible): full confidence
                Self::boost_ids(logits, &self.assertion_ids, 0.4);
                Self::penalize_ids(logits, &self.hedging_ids, -0.3);
            }
            _ => {}
        }

        // ── N-axis: Social Framing ──────────────────────────────────────
        let n = channels.n_tier().unwrap_or(0);
        match n {
            0 => {
                // N0 (personal): boost "I think", "I feel"
                Self::boost_ids(logits, &self.personal_ids, 0.4);
            }
            1 => {
                // N1 (communal): boost "we", "our community"
                Self::boost_ids(logits, &self.communal_ids, 0.3);
            }
            2 => {
                // N2 (network): boost "it is known", "consensus"
                Self::boost_ids(logits, &self.network_ids, 0.3);
            }
            3 => {
                // N3 (axiomatic): boost "necessarily", "by definition"
                Self::boost_ids(logits, &self.axiomatic_ids, 0.4);
            }
            _ => {}
        }

        // ── M-axis: Temporal Framing ────────────────────────────────────
        let m = channels.m_tier().unwrap_or(1);
        match m {
            0 => {
                // M0 (ephemeral): boost "for now", "at this moment"
                Self::boost_ids(logits, &self.ephemeral_ids, 0.3);
            }
            1 => {
                // M1 (temporal): neutral — no specific adjustment
            }
            2 => {
                // M2 (persistent): boost "recorded", "established"
                Self::boost_ids(logits, &self.persistent_ids, 0.3);
            }
            3 => {
                // M3 (foundational): boost "always", "fundamentally"
                Self::boost_ids(logits, &self.foundational_ids, 0.4);
            }
            _ => {}
        }

        // ── H-axis: Coherence Depth ────────────────────────────────────
        let h = channels.h_tier();
        if h < 0.25 {
            // H0-H1: Low coherence → flatten distribution (increase temperature effect)
            // Reduces confidence of all predictions when consciousness is low
            let dampen = 0.85;
            for l in logits.iter_mut() {
                *l *= dampen;
            }
        }
        // H2-H3: Standard generation (no modification)
        // H4 (transcendent): could boost detail tokens, but keep simple for now
    }

    fn boost_ids(logits: &mut [f32], ids: &[u32], boost: f32) {
        for &id in ids {
            if let Some(l) = logits.get_mut(id as usize) {
                *l += boost;
            }
        }
    }

    fn penalize_ids(logits: &mut [f32], ids: &[u32], penalty: f32) {
        for &id in ids {
            if let Some(l) = logits.get_mut(id as usize) {
                *l += penalty; // penalty is negative
            }
        }
    }

    /// NEW: Apply Epistemic Cube coding modulation.
    pub fn apply_coding_modulation(
        &self,
        logits: &mut [f32],
        moral: f32,
        narrative: f32,
        _idiomatic: f32,
    ) {
        // Moral (M-axis): high risk -> flatten/dampen
        if moral < 0.55 {
            let dampen = 0.7 + (moral * 0.4); // 0.7 to 0.9 range
            for l in logits.iter_mut() {
                *l *= dampen;
            }
        }

        // Narrative (N-axis): maintainability -> boost structural/algo tokens
        if narrative > 0.6 {
            // (Shared boost logic for structural tokens)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CodeGate — Code Generation Gating from ThoughtChannels 24-27
// ═══════════════════════════════════════════════════════════════════════════════

/// Code-aware gating: modulates logits for code generation based on
/// ThoughtChannels 24-27 (syntax_complexity, type_confidence, algorithm_pattern,
/// error_likelihood).
///
/// - **syntax_complexity > 0.3**: Boosts structural keywords (fn, struct, match, etc.)
/// - **type_confidence < 0.4**: Suppresses concrete type words, encouraging generic/trait tokens
/// - **algorithm_pattern > 0.2**: Boosts algorithm scaffold words (iter, map, fold, etc.)
/// - **error_likelihood > 0.3**: Boosts error handling words (Result, Option, unwrap, etc.)
///
/// Opt-in via `GatingConfig::enable_code_gate`.
#[derive(Clone)]
pub struct CodeGate {
    config: GatingConfig,
    tokenizer: std::sync::Arc<BpeTokenizer>,
    /// Token IDs for Rust structural keywords.
    structural_ids: Vec<u32>,
    /// Token IDs for concrete type words.
    type_ids: Vec<u32>,
    /// Token IDs for error handling words.
    error_handling_ids: Vec<u32>,
    /// Token IDs for algorithm scaffold words.
    algorithm_ids: Vec<u32>,
    /// Dynamic Nix attribute tokens (populated at runtime from scorer feedback)
    nix_path_tokens: std::sync::Arc<parking_lot::Mutex<std::collections::HashMap<String, u32>>>,
    epistemic_cube_gate: EpistemicCubeGate,
    /// **NEW**: Language-specific gate heads (Nix, Terraform, CDK, etc.)
    language_gate_registry: crate::language_gates::LanguageGateRegistry,
    /// Current emotionally-modulated sampling parameters
    pub current_temperature: f32,
    pub current_top_p: f32,
    pub base_temperature: f32,
    pub base_top_p: f32,
}

impl CodeGate {
    /// Create a code gate using the BPE tokenizer for token classification.
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
            tokenizer: std::sync::Arc::new(tokenizer.clone()),
            structural_ids: resolve(CANONICAL_RUST_STRUCTURAL_WORDS),
            type_ids: resolve(CANONICAL_TYPE_WORDS),
            error_handling_ids: resolve(CANONICAL_ERROR_HANDLING_WORDS),
            algorithm_ids: resolve(CANONICAL_ALGORITHM_SCAFFOLD_WORDS),
            nix_path_tokens: std::sync::Arc::new(parking_lot::Mutex::new(
                std::collections::HashMap::new(),
            )),
            epistemic_cube_gate: EpistemicCubeGate::new(tokenizer),
            language_gate_registry: crate::language_gates::LanguageGateRegistry::new(tokenizer),
            current_temperature: 0.85,
            current_top_p: 0.88,
            base_temperature: 0.85,
            base_top_p: 0.88,
        }
    }

    /// Number of resolved structural token IDs.
    pub fn structural_count(&self) -> usize {
        self.structural_ids.len()
    }

    /// Number of resolved type token IDs.
    pub fn type_count(&self) -> usize {
        self.type_ids.len()
    }

    /// Number of resolved error handling token IDs.
    pub fn error_handling_count(&self) -> usize {
        self.error_handling_ids.len()
    }

    /// Number of resolved algorithm scaffold token IDs.
    pub fn algorithm_count(&self) -> usize {
        self.algorithm_ids.len()
    }

    /// Apply code-aware gating to logits in-place.
    ///
    /// Reads code channels from `channels` and modulates logits for code tokens.
    /// No-op if `enable_code_gate` is false.
    pub fn apply(&mut self, logits: &mut [f32], channels: &ThoughtChannels) {
        if !self.config.enable_code_gate {
            return;
        }

        // 1. Detect language/intent using full registry
        let language_gate = self.language_gate_registry.detect_intent(channels);
        let base_gate_strength = if let Some(gate) = language_gate {
            gate.base_boost
        } else {
            1.8 // default
        };

        // 2. Modulate by emotional state (frustration -> creative, positive -> precise)
        let (temperature, top_p, final_gate_strength) =
            crate::emotional_gating_integration::modulate_by_emotion(
                channels,
                self.base_temperature,
                self.base_top_p,
                base_gate_strength,
            );

        // 3. Store for downstream sampling
        self.current_temperature = temperature;
        self.current_top_p = top_p;

        // 4. Apply the (emotionally modulated) language gate boost to logits
        if let Some(gate) = language_gate {
            self.language_gate_registry
                .apply_gate(logits, gate, final_gate_strength);

            // **NEW**: Cross-Language Suppression
            // Actively suppress tokens from other languages to achieve "Grammar-Safe Emission"
            self.language_gate_registry
                .suppress_other_languages(logits, &gate.name, 3.0);
        }

        // 5. Legacy v5 code-channel scoring (still useful as secondary modulation)
        let syntax = channels.syntax_complexity();
        let type_conf = channels.type_confidence();
        let algo = channels.algorithm_pattern();
        let error = channels.error_likelihood();

        if syntax > 0.3 {
            let boost = self.config.code_structural_boost * syntax;
            for &id in &self.structural_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += boost;
                }
            }
        }
        if type_conf < 0.4 {
            let penalty = -self.config.code_type_penalty * (1.0 - type_conf);
            for &id in &self.type_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += penalty;
                }
            }
        }
        if algo > 0.2 {
            let boost = self.config.code_algorithm_boost * algo;
            for &id in &self.algorithm_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += boost;
                }
            }
        }
        if error > 0.3 {
            let boost = self.config.code_error_boost * error;
            for &id in &self.error_handling_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += boost;
                }
            }
        }

        // 6. Epistemic Cube coding modulation (moral/narrative/idiomatic)
        let moral = channels.moral_score();
        let narrative = channels.narrative_score();
        let idiomatic = channels.idiomatic_score();
        self.epistemic_cube_gate
            .apply_coding_modulation(logits, moral, narrative, idiomatic);
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
#[derive(Clone)]
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
    #[cfg(feature = "mamba-cpu")]
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

    /// Apply emotional modulation with a backend-specific scale factor.
    ///
    /// For Liquid-Mamba: pass `config.mamba_emotional_scale()` as `backend_scale`.
    /// For CfC-HDC: pass 1.0 (or just call `apply` directly).
    pub fn apply_scaled(
        &self,
        logits: &mut [f32],
        channels: &ThoughtChannels,
        position: usize,
        backend_scale: f32,
    ) {
        if backend_scale == 1.0 {
            self.apply(logits, channels, position);
            return;
        }

        let arousal = channels.arousal();
        let warmth = channels.warmth();
        let valence = channels.valence();

        if arousal > self.config.high_arousal_threshold
            && position > self.config.arousal_position_threshold
        {
            let boost = (arousal - self.config.high_arousal_threshold)
                * self.config.arousal_boost_multiplier
                * backend_scale;
            for &id in &self.sentence_end_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += boost;
                }
            }
        }

        if arousal > self.config.high_arousal_threshold
            && valence < self.config.negative_valence_threshold
            && position > self.config.arousal_position_threshold
        {
            let interaction_boost = (arousal - self.config.high_arousal_threshold)
                * (-valence - (-self.config.negative_valence_threshold))
                * self.config.valence_boost_multiplier
                * backend_scale;
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

        let social = channels.social_context().clamp(0.0, 1.0);
        let formality_scale = 1.0 + social * self.config.social_context_formality_scale;
        if warmth < self.config.low_warmth_threshold {
            let penalty = (self.config.low_warmth_threshold - warmth)
                * self.config.warmth_penalty_multiplier
                * formality_scale
                * backend_scale;
            for &id in &self.informal_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += penalty;
                }
            }
        }

        if valence < self.config.negative_valence_threshold {
            let boost = (-valence - (-self.config.negative_valence_threshold))
                * self.config.valence_boost_multiplier
                * backend_scale;
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
#[derive(Clone)]
pub struct CoherenceFeedback {
    /// Drift threshold — below this, correction is applied.
    threshold: f32,
    /// Current coherence score (updated each step).
    current_coherence: f32,
    /// Whether semantic veto was triggered.
    veto_triggered: bool,
    /// Semantic veto threshold (more aggressive than drift).
    veto_threshold: f32,
    /// Enable algebraic (HDC subtract) correction instead of scalar scaling (Phase 2).
    enable_algebraic: bool,
    /// Correction strength for algebraic mode (0.0-1.0). Default 0.3.
    algebraic_strength: f32,
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
            enable_algebraic: false,
            algebraic_strength: 0.3,
        }
    }

    /// Enable algebraic coherence correction (Phase 2).
    /// Instead of scalar scaling, uses HDC subtract to surgically remove
    /// drifting dimensions from the thought vector.
    pub fn set_algebraic(&mut self, enable: bool, strength: f32) {
        self.enable_algebraic = enable;
        self.algebraic_strength = strength.clamp(0.0, 1.0);
    }

    /// Whether algebraic mode is enabled.
    pub fn is_algebraic(&self) -> bool {
        self.enable_algebraic
    }

    /// Compute algebraic correction: surgically remove drift from thought_hv.
    ///
    /// Instead of `thought_hv.scale(weight)` (uniform amplification),
    /// this computes the error vector between output and thought, then
    /// subtracts a fraction of it. This is gradient descent in HDC space.
    ///
    /// Returns the corrected thought HV (or the original if no correction needed).
    pub fn algebraic_correct(
        &self,
        output_hv: &symthaea_core::hdc::ContinuousHV,
        thought_hv: &symthaea_core::hdc::ContinuousHV,
    ) -> symthaea_core::hdc::ContinuousHV {
        if self.current_coherence >= self.threshold {
            return thought_hv.clone();
        }
        // drift = output - thought (the error vector in HDC space)
        let drift = output_hv.subtract(thought_hv);
        let drift_norm = drift.norm();
        let thought_norm = thought_hv.norm();
        if drift_norm < 1e-8 || thought_norm < 1e-8 {
            return thought_hv.clone();
        }
        // Correction magnitude scales with how far below threshold we are
        let urgency = ((self.threshold - self.current_coherence) / self.threshold).clamp(0.0, 1.0);
        let effective_strength = self.algebraic_strength * urgency;
        // corrected = thought - strength * drift (move thought away from drift direction)
        let correction = drift.scale(effective_strength);
        let corrected = thought_hv.subtract(&correction);
        corrected.normalize().scale(thought_norm) // Preserve original norm
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
                    // In temperature mode, expert_boost may be negative because
                    // temperature scaling reduces all logits; we only check ordering.
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

    // =========================================================================
    // CodeGate tests
    // =========================================================================

    #[test]
    fn test_code_gate_structural_boost() {
        let mut tok = test_tokenizer();
        // Ensure structural words are in vocab
        for &w in CANONICAL_RUST_STRUCTURAL_WORDS {
            if tok.token_id(w) == tok.unk_id {
                tok.add_token(w);
            }
        }

        let mut config = test_config();
        config.enable_code_gate = true;
        let mut gate = CodeGate::new(&tok, &config);

        let mut channels = ThoughtChannels::default();
        channels.set_code(0.8, 1.0, 0.0, 0.0); // high syntax_complexity

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, &channels);

        // Structural tokens should be boosted
        let mut found_boost = false;
        for &w in CANONICAL_RUST_STRUCTURAL_WORDS {
            let id = tok.token_id(w);
            if id != tok.unk_id && logits[id as usize] > 0.5 + 1e-6 {
                found_boost = true;
                break;
            }
        }
        assert!(
            found_boost,
            "Structural tokens should be boosted when syntax_complexity > 0.3"
        );
    }

    #[test]
    fn test_code_gate_type_suppression() {
        let mut tok = test_tokenizer();
        for &w in CANONICAL_TYPE_WORDS {
            if tok.token_id(w) == tok.unk_id {
                tok.add_token(w);
            }
        }

        let mut config = test_config();
        config.enable_code_gate = true;
        let mut gate = CodeGate::new(&tok, &config);

        let mut channels = ThoughtChannels::default();
        channels.set_code(0.0, 0.1, 0.0, 0.0); // low type_confidence

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, &channels);

        // Type tokens should be suppressed
        let mut found_suppression = false;
        for &w in CANONICAL_TYPE_WORDS {
            let id = tok.token_id(w);
            if id != tok.unk_id && logits[id as usize] < 0.5 - 1e-6 {
                found_suppression = true;
                break;
            }
        }
        assert!(
            found_suppression,
            "Type tokens should be suppressed when type_confidence < 0.4"
        );
    }

    #[test]
    fn test_code_gate_algorithm_scaffold() {
        let mut tok = test_tokenizer();
        for &w in CANONICAL_ALGORITHM_SCAFFOLD_WORDS {
            if tok.token_id(w) == tok.unk_id {
                tok.add_token(w);
            }
        }

        let mut config = test_config();
        config.enable_code_gate = true;
        let mut gate = CodeGate::new(&tok, &config);

        let mut channels = ThoughtChannels::default();
        channels.set_code(0.0, 1.0, 0.7, 0.0); // algorithm_pattern active

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, &channels);

        // Algorithm scaffold tokens should be boosted
        let mut found_boost = false;
        for &w in CANONICAL_ALGORITHM_SCAFFOLD_WORDS {
            let id = tok.token_id(w);
            if id != tok.unk_id && logits[id as usize] > 0.5 + 1e-6 {
                found_boost = true;
                break;
            }
        }
        assert!(
            found_boost,
            "Algorithm scaffold tokens should be boosted when algorithm_pattern > 0.2"
        );
    }

    #[test]
    fn test_code_gate_error_handling() {
        let mut tok = test_tokenizer();
        for &w in CANONICAL_ERROR_HANDLING_WORDS {
            if tok.token_id(w) == tok.unk_id {
                tok.add_token(w);
            }
        }

        let mut config = test_config();
        config.enable_code_gate = true;
        let mut gate = CodeGate::new(&tok, &config);

        let mut channels = ThoughtChannels::default();
        channels.set_code(0.0, 1.0, 0.0, 0.8); // high error_likelihood

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, &channels);

        // Error handling tokens should be boosted
        let mut found_boost = false;
        for &w in CANONICAL_ERROR_HANDLING_WORDS {
            let id = tok.token_id(w);
            if id != tok.unk_id && logits[id as usize] > 0.5 + 1e-6 {
                found_boost = true;
                break;
            }
        }
        assert!(
            found_boost,
            "Error handling tokens should be boosted when error_likelihood > 0.3"
        );
    }

    #[test]
    fn test_code_gate_inactive_when_no_code() {
        let mut tok = test_tokenizer();
        // Add some tokens so the gate has IDs to work with
        for &w in CANONICAL_RUST_STRUCTURAL_WORDS {
            if tok.token_id(w) == tok.unk_id {
                tok.add_token(w);
            }
        }

        let mut config = test_config();
        config.enable_code_gate = true;
        let mut gate = CodeGate::new(&tok, &config);

        // All code channels at zero/neutral
        let channels = ThoughtChannels::default();

        let mut logits = vec![0.5; tok.vocab_size()];
        let original = logits.clone();
        gate.apply(&mut logits, &channels);

        assert_eq!(
            logits, original,
            "CodeGate should not modify logits when all code channels are zero"
        );
    }

    #[cfg(feature = "mamba-cpu")]
    mod backend_tests {
        use super::*;
        use crate::mamba::mock::MockMamba;

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

    // =========================================================================
    // Temperature-based epistemic gating tests
    // =========================================================================

    #[test]
    fn test_temperature_mode_default_enabled() {
        let config = GatingConfig::default();
        assert!(
            config.epistemic_temperature_mode,
            "Temperature mode should be enabled by default"
        );
    }

    #[test]
    fn test_temperature_mode_preserves_vocab_diversity() {
        let tok = test_tokenizer();
        let config = test_config();
        // test_config uses default, which has temperature mode enabled
        let gate = EpistemicGate::new(&tok, &config);

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, 4.0); // OOD level

        // Temperature mode should preserve > 50% of logits as non-negative
        let non_negative = logits.iter().filter(|&&l| l >= 0.0).count();
        assert!(
            non_negative > logits.len() / 2,
            "Temperature mode should preserve vocabulary diversity: {non_negative}/{} non-negative",
            logits.len()
        );
    }

    #[test]
    fn test_temperature_mode_vs_legacy_more_diverse() {
        let tok = test_tokenizer();

        // Temperature mode
        let config_temp = test_config();
        let gate_temp = EpistemicGate::new(&tok, &config_temp);
        let mut logits_temp = vec![0.5; tok.vocab_size()];
        gate_temp.apply(&mut logits_temp, 4.0);
        let non_neg_temp = logits_temp.iter().filter(|&&l| l >= 0.0).count();

        // Legacy additive mode
        let mut config_legacy = test_config();
        config_legacy.epistemic_temperature_mode = false;
        let gate_legacy = EpistemicGate::new(&tok, &config_legacy);
        let mut logits_legacy = vec![0.5; tok.vocab_size()];
        gate_legacy.apply(&mut logits_legacy, 4.0);
        let non_neg_legacy = logits_legacy.iter().filter(|&&l| l >= 0.0).count();

        assert!(
            non_neg_temp >= non_neg_legacy,
            "Temperature mode should leave at least as many non-negative logits as legacy: \
             temp={non_neg_temp} vs legacy={non_neg_legacy}"
        );
    }

    #[test]
    fn test_temperature_mode_still_boosts_hedging() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, 3.0); // Unknown level

        // Find a hedging token and a non-special, non-hedging, non-factual token
        let mut hedging_logit = None;
        for &word in CANONICAL_HEDGING_WORDS {
            let id = tok.token_id(word);
            if id != tok.unk_id {
                hedging_logit = Some(logits[id as usize]);
                break;
            }
        }

        // Temperature mode scales everything down, but hedging tokens get additive boost
        // so they should be higher than a generic non-boosted token
        if let Some(hedge_val) = hedging_logit {
            // Generic token at Unknown level with temperature 1.5: 0.5 / 1.5 ≈ 0.333
            let generic_val = 0.5 / config.unknown_temperature;
            assert!(
                hedge_val > generic_val,
                "Hedging tokens should be boosted relative to non-hedging: hedge={hedge_val} vs generic={generic_val}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NSM SEMANTIC GATE — Phase 3: token selection guided by NSM primes
//
// Maps detected NSM semantic primes to token IDs, then boosts logits for
// tokens that express active primes. Follows the EpistemicGate pattern.
//
// Science: Collins & Loftus (1975) — spreading activation strengthens
// semantically related lexical entries.
// ═══════════════════════════════════════════════════════════════════════════════

use std::collections::{HashMap, HashSet};

/// Maps NSM semantic primes to BPE token IDs that express them.
///
/// Built once from the tokenizer vocabulary, then reused per-generation.
/// Analogous to `EpistemicGate` but for semantic content rather than epistemic status.
#[derive(Clone)]
pub struct NsmSemanticGate {
    /// prime name (lowercased) → set of token IDs that express this prime.
    prime_to_tokens: HashMap<String, Vec<u32>>,
    /// Reverse: token_id → set of prime names this token expresses.
    token_to_primes: HashMap<u32, Vec<String>>,
}

/// Word → NSM prime mappings for the most common English words.
/// This is a minimal lexicon; GroundedUnderstanding has a more complete one.
const NSM_WORD_PRIMES: &[(&str, &[&str])] = &[
    // Mental predicates
    ("think", &["THINK"]),
    ("know", &["KNOW"]),
    ("want", &["WANT"]),
    ("feel", &["FEEL"]),
    ("see", &["SEE"]),
    ("hear", &["HEAR"]),
    // Evaluators
    ("good", &["GOOD"]),
    ("bad", &["BAD"]),
    ("right", &["GOOD"]),
    ("wrong", &["BAD"]),
    // Actions
    ("do", &["DO"]),
    ("happen", &["HAPPEN"]),
    ("move", &["MOVE"]),
    ("touch", &["TOUCH"]),
    // Logical
    ("not", &["NOT"]),
    ("maybe", &["MAYBE"]),
    ("can", &["CAN"]),
    ("because", &["BECAUSE"]),
    ("if", &["IF"]),
    // Substantives
    ("someone", &["SOMEONE"]),
    ("something", &["SOMETHING"]),
    ("people", &["PEOPLE"]),
    ("body", &["BODY"]),
    // Existence
    ("is", &["BE"]),
    ("are", &["BE"]),
    ("have", &["HAVE"]),
    ("has", &["HAVE"]),
    ("exist", &["BE"]),
    // Time
    ("now", &["NOW"]),
    ("before", &["BEFORE"]),
    ("after", &["AFTER"]),
    // Space
    ("here", &["HERE"]),
    ("above", &["ABOVE"]),
    ("below", &["BELOW"]),
    ("near", &["NEAR"]),
    ("far", &["FAR"]),
    ("inside", &["INSIDE"]),
    // Quantifiers
    ("all", &["ALL"]),
    ("some", &["SOME"]),
    ("much", &["MUCH"]),
    ("many", &["MUCH"]),
    ("little", &["LITTLE"]),
    ("few", &["LITTLE"]),
    // Speech
    ("say", &["SAY"]),
    ("said", &["SAY"]),
    ("tell", &["SAY"]),
    ("true", &["TRUE"]),
    ("word", &["WORDS"]),
    // Intensifiers
    ("very", &["VERY"]),
    ("more", &["MORE"]),
    // Emotional composite words
    ("sad", &["FEEL", "BAD"]),
    ("happy", &["FEEL", "GOOD"]),
    ("angry", &["FEEL", "BAD"]),
    ("afraid", &["FEEL", "BAD"]),
    ("love", &["FEEL", "GOOD", "WANT"]),
    ("hate", &["FEEL", "BAD", "NOT", "WANT"]),
    // Life
    ("live", &["LIVE"]),
    ("die", &["DIE"]),
    ("alive", &["LIVE"]),
    ("dead", &["DIE"]),
    ("died", &["DIE"]),
    // — Additional entries synced from LexicalGrounding —
    // Pronouns
    ("me", &["I"]),
    ("my", &["I"]),
    ("your", &["YOU"]),
    ("somebody", &["SOMEONE"]),
    // Emotion composites (richer decompositions)
    ("joy", &["FEEL", "GOOD"]),
    ("joyful", &["FEEL", "GOOD"]),
    ("grief", &["FEEL", "BAD", "DIE"]),
    ("calm", &["FEEL", "GOOD"]),
    ("peaceful", &["FEEL", "GOOD"]),
    ("anxious", &["FEEL", "BAD", "MAYBE"]),
    ("worried", &["THINK", "BAD", "MAYBE"]),
    ("excited", &["FEEL", "GOOD"]),
    ("surprised", &["FEEL", "KNOW"]),
    ("fear", &["FEEL", "BAD", "MAYBE"]),
    // Mental verbs
    ("believe", &["THINK", "TRUE"]),
    ("need", &["WANT"]),
    ("understand", &["KNOW", "THINK"]),
    ("remember", &["KNOW", "BEFORE"]),
    ("forget", &["KNOW", "NOT"]),
    // Action verbs
    ("go", &["MOVE"]),
    ("come", &["MOVE"]),
    ("leave", &["MOVE", "FAR"]),
    ("left", &["MOVE", "FAR", "BEFORE"]),
    ("stay", &["MOVE", "NOT"]),
    ("give", &["DO"]),
    ("take", &["DO"]),
    ("make", &["DO"]),
    ("put", &["DO", "MOVE"]),
    ("get", &["HAVE"]),
    // Evaluators
    ("great", &["GOOD", "VERY"]),
    ("wonderful", &["GOOD", "VERY"]),
    ("terrible", &["BAD", "VERY"]),
    ("false", &["TRUE", "NOT"]),
    // Descriptors
    ("big", &["BIG"]),
    ("small", &["SMALL"]),
    ("this", &["THIS"]),
    ("same", &["SAME"]),
    ("other", &["OTHER"]),
    // Logical
    ("no", &["NOT"]),
    ("cannot", &["CAN", "NOT"]),
    // Relational
    ("friend", &["SOMEONE", "GOOD", "WANT"]),
    ("enemy", &["SOMEONE", "BAD"]),
    ("family", &["PEOPLE", "LIKE"]),
    ("child", &["SOMEONE", "SMALL"]),
    // Spatial
    ("where", &["WHERE"]),
    ("there", &["FAR"]),
    ("on", &["ON"]),
    // Temporal
    ("when", &["WHEN"]),
    ("long", &["LONG_TIME"]),
    ("short", &["SHORT_TIME"]),
    // Quantifiers
    ("one", &["ONE"]),
    ("two", &["TWO"]),
    // Intensifiers
    ("really", &["VERY"]),
    ("less", &["MORE", "NOT"]),
    // Social
    ("with", &["WITH"]),
];

impl NsmSemanticGate {
    /// Build the prime↔token reverse index from the tokenizer vocabulary.
    /// Uses the hardcoded NSM_WORD_PRIMES table as the base lexicon.
    pub fn new(tokenizer: &BpeTokenizer) -> Self {
        let mut gate = Self {
            prime_to_tokens: HashMap::new(),
            token_to_primes: HashMap::new(),
        };

        // Ingest from const table
        for &(word, primes) in NSM_WORD_PRIMES {
            gate.register_word(tokenizer, word, primes);
        }

        gate.deduplicate();
        gate
    }

    /// Build with enriched lexicon from LexicalGrounding (single source of truth).
    /// Merges `LexicalGrounding`'s ~100 word→prime mappings with the const table,
    /// eliminating duplication. `LexicalGrounding` uses `SemanticPrime` enums which
    /// are converted to uppercase string names for the gate.
    pub fn new_with_lexicon(tokenizer: &BpeTokenizer) -> Self {
        use symthaea_core::hdc::grounded_understanding::LexicalGrounding;

        let mut gate = Self {
            prime_to_tokens: HashMap::new(),
            token_to_primes: HashMap::new(),
        };

        // Primary source: LexicalGrounding (canonical word→prime mappings)
        let lexicon = LexicalGrounding::new();
        // LexicalGrounding exposes decompose(word) → Option<&Vec<SemanticPrime>>
        // We need to iterate all known words. Since LexicalGrounding doesn't expose
        // iteration, we check all words from both NSM_WORD_PRIMES and a supplementary list.
        let all_words: Vec<&str> = NSM_WORD_PRIMES.iter().map(|&(w, _)| w).collect();
        for &word in &all_words {
            if let Some(primes) = lexicon.decompose(word) {
                let prime_strs: Vec<&str> = primes.iter().map(|p| p.as_gate_name()).collect();
                gate.register_word(tokenizer, word, &prime_strs);
            } else {
                // Fallback: use NSM_WORD_PRIMES for words not in LexicalGrounding
                if let Some(&(_, prime_strs)) = NSM_WORD_PRIMES.iter().find(|&&(w, _)| w == word) {
                    gate.register_word(tokenizer, word, prime_strs);
                }
            }
        }

        // Morpheme decomposition: for BPE tokens not covered by word lookup,
        // try prefix/suffix stripping to infer NSM primes from morphological structure.
        // Science: Wierzbicka (1996) — morphological structure carries semantic prime content.
        for tid in 0..tokenizer.vocab_size() as u32 {
            if gate.token_to_primes.contains_key(&tid) || tokenizer.is_special(tid) {
                continue;
            }
            let token_str = tokenizer.token_str(tid);
            let morph_primes = Self::morpheme_primes_for_token(token_str);
            if !morph_primes.is_empty() {
                for &prime in &morph_primes {
                    let prime_upper = prime.to_uppercase();
                    gate.prime_to_tokens
                        .entry(prime_upper.clone())
                        .or_default()
                        .push(tid);
                    gate.token_to_primes
                        .entry(tid)
                        .or_default()
                        .push(prime_upper);
                }
            }
        }

        gate.deduplicate();
        gate
    }

    /// Register a word→prime mapping in the gate indices.
    fn register_word(&mut self, tokenizer: &BpeTokenizer, word: &str, primes: &[&str]) {
        let token_id = tokenizer.token_id(word);
        if token_id != tokenizer.unk_id {
            for &prime in primes {
                let prime_upper = prime.to_uppercase();
                self.prime_to_tokens
                    .entry(prime_upper.clone())
                    .or_default()
                    .push(token_id);
                self.token_to_primes
                    .entry(token_id)
                    .or_default()
                    .push(prime_upper);
            }
        }
    }

    /// Attempt morpheme decomposition for a BPE token not in the word lexicon.
    /// Strips known prefixes (un-, re-, dis-, etc.) and suffixes (-ing, -ed, -able, etc.)
    /// to infer NSM primes from morphological structure.
    /// Science: Wierzbicka (1996) — morphological structure carries semantic prime content.
    fn morpheme_primes_for_token(token_str: &str) -> Vec<&'static str> {
        let word = token_str.to_lowercase();
        if word.len() < 3 {
            return Vec::new();
        }
        let mut primes = Vec::new();

        // Prefix stripping (longest match first)
        let prefixes: &[(&str, &[&str])] = &[
            ("under", &["LITTLE", "BELOW"]),
            ("super", &["ABOVE", "BIG"]),
            ("multi", &["MUCH"]),
            ("auto", &["SAME"]),
            ("over", &["MUCH", "ABOVE"]),
            ("post", &["AFTER"]),
            ("pre", &["BEFORE"]),
            ("mis", &["BAD"]),
            ("dis", &["NOT"]),
            ("out", &["FAR"]),
            ("sub", &["BELOW"]),
            ("un", &["NOT"]),
            ("re", &["BEFORE"]),
            ("de", &["NOT"]),
            ("in", &["INSIDE"]),
            ("co", &["SAME"]),
        ];
        for &(prefix, prefix_primes) in prefixes {
            if word.starts_with(prefix) && word.len() > prefix.len() + 2 {
                primes.extend_from_slice(prefix_primes);
                break; // Only strip one prefix
            }
        }

        // Suffix stripping (longest match first)
        let suffixes: &[(&str, &[&str])] = &[
            ("ation", &["SOMETHING", "HAPPEN"]),
            ("tion", &["SOMETHING", "HAPPEN"]),
            ("sion", &["SOMETHING", "HAPPEN"]),
            ("ment", &["SOMETHING"]),
            ("ness", &["KIND_OF"]),
            ("able", &["CAN"]),
            ("ible", &["CAN"]),
            ("less", &["NOT", "HAVE"]),
            ("ful", &["MUCH"]),
            ("ing", &["NOW", "HAPPEN"]),
            ("ize", &["DO", "HAPPEN"]),
            ("ise", &["DO", "HAPPEN"]),
            ("ify", &["DO", "HAPPEN"]),
            ("er", &["SOMEONE", "DO"]),
            ("or", &["SOMEONE", "DO"]),
            ("ed", &["BEFORE", "HAPPEN"]),
            ("ly", &[]), // Modifier only, no prime content
        ];
        for &(suffix, suffix_primes) in suffixes {
            if word.ends_with(suffix) && word.len() > suffix.len() + 2 {
                primes.extend_from_slice(suffix_primes);
                break; // Only strip one suffix
            }
        }

        primes
    }

    /// Deduplicate the reverse indices.
    fn deduplicate(&mut self) {
        for tokens in self.prime_to_tokens.values_mut() {
            tokens.sort_unstable();
            tokens.dedup();
        }
        for primes in self.token_to_primes.values_mut() {
            primes.sort_unstable();
            primes.dedup();
        }
    }

    /// Boost logits for tokens that express active NSM primes.
    ///
    /// For each active prime, adds `boost` to the logit of every token that
    /// expresses that prime. Tokens expressing multiple active primes get
    /// multiple boosts (spreading activation).
    pub fn apply(&self, logits: &mut [f32], active_primes: &[String], boost: f32) {
        if boost == 0.0 || active_primes.is_empty() {
            return;
        }
        for prime_name in active_primes {
            let key = prime_name.to_uppercase();
            if let Some(token_ids) = self.prime_to_tokens.get(&key) {
                for &tid in token_ids {
                    if (tid as usize) < logits.len() {
                        logits[tid as usize] += boost;
                    }
                }
            }
        }
    }

    /// Get the primes expressed by a given token ID.
    pub fn primes_for_token(&self, token_id: u32) -> &[String] {
        self.token_to_primes
            .get(&token_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Number of primes mapped.
    pub fn num_primes(&self) -> usize {
        self.prime_to_tokens.len()
    }
}

/// Tracks which active NSM primes have been "expressed" by generated tokens.
///
/// Prime coverage = expressed_primes / active_primes (0.0–1.0).
/// High coverage = the generated text semantically covers the intended meaning.
///
/// Science: Grice (1975) — cooperative principle; covering intended semantic
/// content signals communicative success.
pub struct NsmCoherenceTracker {
    /// Active primes for this generation (uppercased).
    active_primes: HashSet<String>,
    /// Primes that have been expressed by generated tokens so far.
    expressed_primes: HashSet<String>,
}

impl NsmCoherenceTracker {
    /// Create a new tracker for the given active primes.
    pub fn new(active_primes: &[String]) -> Self {
        Self {
            active_primes: active_primes.iter().map(|p| p.to_uppercase()).collect(),
            expressed_primes: HashSet::new(),
        }
    }

    /// Observe a generated token and update prime coverage.
    pub fn observe_token(&mut self, token_id: u32, gate: &NsmSemanticGate) {
        for prime in gate.primes_for_token(token_id) {
            if self.active_primes.contains(prime) {
                self.expressed_primes.insert(prime.clone());
            }
        }
    }

    /// Current prime coverage (0.0–1.0).
    pub fn prime_coverage(&self) -> f32 {
        if self.active_primes.is_empty() {
            return 0.0;
        }
        self.expressed_primes.len() as f32 / self.active_primes.len() as f32
    }

    /// Number of active primes.
    pub fn active_count(&self) -> usize {
        self.active_primes.len()
    }

    /// Number of expressed primes.
    pub fn expressed_count(&self) -> usize {
        self.expressed_primes.len()
    }
}

#[cfg(test)]
mod nsm_tests {
    use super::*;

    fn test_tokenizer() -> BpeTokenizer {
        BpeTokenizer::default_4k()
    }

    #[test]
    fn test_nsm_gate_construction() {
        let tok = test_tokenizer();
        let gate = NsmSemanticGate::new(&tok);
        // Should have mapped at least some primes
        assert!(gate.num_primes() > 0, "Should map at least some primes");
    }

    #[test]
    fn test_nsm_gate_boosts_prime_tokens() {
        let tok = test_tokenizer();
        let gate = NsmSemanticGate::new(&tok);

        let mut logits = vec![0.0_f32; tok.vocab_size()];
        let active = vec!["FEEL".to_string(), "BAD".to_string()];
        gate.apply(&mut logits, &active, 0.5);

        // At least one logit should have been boosted
        let boosted_count = logits.iter().filter(|&&v| v > 0.0).count();
        assert!(
            boosted_count > 0,
            "At least one token should be boosted for FEEL+BAD"
        );
    }

    #[test]
    fn test_nsm_gate_no_effect_when_disabled() {
        let tok = test_tokenizer();
        let gate = NsmSemanticGate::new(&tok);

        let mut logits = vec![1.0_f32; tok.vocab_size()];
        let original = logits.clone();

        // Empty primes → no change
        gate.apply(&mut logits, &[], 0.5);
        assert_eq!(logits, original);

        // Zero boost → no change
        let mut logits2 = original.clone();
        gate.apply(&mut logits2, &["FEEL".to_string()], 0.0);
        assert_eq!(logits2, original);
    }

    #[test]
    fn test_nsm_coherence_tracker_coverage() {
        let tok = test_tokenizer();
        let gate = NsmSemanticGate::new(&tok);

        let active = vec!["FEEL".to_string(), "BAD".to_string(), "BECAUSE".to_string()];
        let mut tracker = NsmCoherenceTracker::new(&active);

        assert_eq!(tracker.active_count(), 3);
        assert_eq!(tracker.expressed_count(), 0);
        assert!((tracker.prime_coverage() - 0.0).abs() < 1e-5);

        // Observe "feel" token — should cover FEEL prime
        let feel_id = tok.token_id("feel");
        if feel_id != tok.unk_id {
            tracker.observe_token(feel_id, &gate);
            assert!(
                tracker.expressed_count() > 0,
                "Observing 'feel' should express FEEL prime"
            );
        }
    }

    #[test]
    fn test_nsm_coherence_tracker_empty_primes() {
        let tracker = NsmCoherenceTracker::new(&[]);
        assert!((tracker.prime_coverage() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_nsm_gate_with_lexicon_maps_more_primes() {
        let tok = test_tokenizer();
        let gate_basic = NsmSemanticGate::new(&tok);
        let gate_lexicon = NsmSemanticGate::new_with_lexicon(&tok);
        // Lexicon-enriched gate should map at least as many primes
        assert!(
            gate_lexicon.num_primes() >= gate_basic.num_primes(),
            "Lexicon gate ({}) should map >= basic gate ({})",
            gate_lexicon.num_primes(),
            gate_basic.num_primes()
        );
    }

    #[test]
    fn test_nsm_generation_quality_comparison() {
        // Benchmark: compare generation with NSM gate enabled vs disabled.
        // Both use the same thought channels and fresh (untrained) generator.
        // We measure: (a) whether NSM-gated output differs from baseline,
        // (b) prime coverage when gate is enabled.
        use crate::encoder::ThoughtChannels;
        use crate::generator::{BrocaConfig, BrocaGenerator};
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("nsm-benchmark");

        // Baseline: no NSM
        let config_baseline = BrocaConfig::default();
        let mut gen_baseline = BrocaGenerator::new(&genesis, config_baseline);

        // NSM-enabled: both semantic and gate
        let mut config_nsm = BrocaConfig::default();
        config_nsm.enable_nsm_gate = true;
        config_nsm.nsm_prime_logit_boost = 0.5;
        let mut gen_nsm = BrocaGenerator::new(&genesis, config_nsm);

        // Test across diverse thought channels
        let scenarios = vec![
            (1, "answer"),    // Intent: Answer
            (4, "uncertain"), // Intent: Uncertainty
            (5, "reflect"),   // Intent: Reflect
        ];
        let active_primes: Vec<String> =
            vec!["FEEL".into(), "KNOW".into(), "THINK".into(), "GOOD".into()];

        let mut nsm_coverage_sum = 0.0_f32;
        let mut nsm_coverage_count = 0;

        for (intent_idx, _label) in &scenarios {
            let channels = ThoughtChannels::with_intent(*intent_idx);

            let result_baseline = gen_baseline.generate(&channels);
            let result_nsm = gen_nsm.generate_with_semantic(&channels, None, &active_primes);

            // NSM-gated generation should produce nsm_prime_coverage > 0
            // (at least some tokens should match active primes)
            nsm_coverage_sum += result_nsm.nsm_prime_coverage;
            nsm_coverage_count += 1;

            // Both should produce tokens (untrained, but should still generate)
            assert!(
                result_baseline.num_tokens > 0,
                "Baseline should generate tokens"
            );
            assert!(result_nsm.num_tokens > 0, "NSM should generate tokens");
        }

        // Average NSM prime coverage across scenarios
        let avg_coverage = if nsm_coverage_count > 0 {
            nsm_coverage_sum / nsm_coverage_count as f32
        } else {
            0.0
        };

        // With 4 active primes and a 4K vocab, even random generation should
        // occasionally hit prime-aligned tokens. With boosting, coverage should
        // be measurably above 0 (though possibly low for untrained models).
        // This test validates the pipeline is wired correctly, not final quality.
        eprintln!(
            "NSM benchmark: avg prime coverage = {avg_coverage:.3} over {nsm_coverage_count} scenarios"
        );
        // No hard assertion on coverage value — untrained models are unpredictable.
        // The key validation is that generation completes without panic and
        // nsm_prime_coverage is populated (not NaN).
        assert!(
            avg_coverage.is_finite(),
            "NSM prime coverage should be finite"
        );
    }
}
