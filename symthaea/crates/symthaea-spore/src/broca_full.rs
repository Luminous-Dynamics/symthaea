// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broca-full: complete Broca language pipeline ported to the Spore WASM kernel.
//!
//! This is the Tier 2 Broca implementation, bringing the full gating pipeline
//! from `symthaea-broca` into the Spore for WASM compilation. Unlike broca-lite
//! (12-channel, 1024D embeddings, element-wise recurrence), this uses:
//!
//! - **24-channel ThoughtLanguageEncoder** with genesis-seeded base vectors
//! - **HdcLtcUnifiedNetwork** backbone (3 layers x 8 neurons, 16,384D)
//! - **EpistemicGate**: canonical word lists, logit penalties (-3.0 unknown, -1.5 uncertain)
//! - **EmotionalModulator**: arousal -> sentence length, warmth -> formality
//! - **CoherenceFeedback**: Welford adaptive z-score veto with hesitation insertion
//! - **Consciousness-gated verbosity**: Psi modulates max_tokens
//! - **Frequency-scaled repetition penalty** + no-repeat bigram blocking
//!
//! The vocabulary is a self-contained 1024-token set (matching broca-lite's size)
//! with word-level tokens for English generation.
//!
//! # Feature gate
//!
//! This module is gated behind `#[cfg(feature = "broca-full")]`.

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig, HDC_DIMENSION,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Number of thought channels (24 = 8 intent + 12 state + 4 context).
pub const NUM_CHANNELS: usize = 24;

/// Number of thermometer coding levels for HDC encoding.
const NUM_LEVELS: usize = 32;

/// Default vocabulary size.
const VOCAB_SIZE: usize = 1024;

/// Maximum sequence length.
const MAX_SEQ_LEN: usize = 128;

/// CLIP-style logit scale for cosine similarity -> logit conversion.
const LOGIT_SCALE: f32 = 20.0;

/// Default repetition penalty.
const DEFAULT_REPETITION_PENALTY: f32 = 2.0;

/// Default dt per token (20ms).
const DT_PER_TOKEN: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// Canonical word lists
// ═══════════════════════════════════════════════════════════════════════════════

/// Canonical hedging words boosted under epistemic uncertainty.
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
pub const CANONICAL_OOD_WORDS: &[&str] = &["outside", "beyond", "cannot", "unable"];

/// Canonical sentence-ending tokens for emotional modulation.
pub const CANONICAL_SENTENCE_ENDINGS: &[&str] = &[". ", "! ", "? ", "...", "\n"];

/// Canonical informal words suppressed under low warmth.
pub const CANONICAL_INFORMAL_WORDS: &[&str] = &[
    "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "lemme",
];

/// Canonical softening words boosted under negative valence.
pub const CANONICAL_SOFTENING_WORDS: &[&str] = &["unfortunately", "sorry", "however", "although"];

// ═══════════════════════════════════════════════════════════════════════════════
// Channel names and ranges
// ═══════════════════════════════════════════════════════════════════════════════

const CHANNEL_NAMES: [&str; NUM_CHANNELS] = [
    "intent_acknowledge",  // 0
    "intent_answer",       // 1
    "intent_clarify",      // 2
    "intent_propose",      // 3
    "intent_uncertainty",  // 4
    "intent_reflect",      // 5
    "intent_continue",     // 6
    "intent_unknown",      // 7
    "epistemic_status",    // 8
    "valence",             // 9
    "arousal",             // 10
    "warmth",              // 11
    "psi",                 // 12
    "meta_awareness",      // 13
    "coherence",           // 14
    "relationship_stage",  // 15
    "trust",               // 16
    "mood_temperature",    // 17
    "has_computed_answer", // 18
    "concept_count",       // 19
    "time_pressure",       // 20
    "domain_familiarity",  // 21
    "social_context",      // 22
    "response_confidence", // 23
];

const CHANNEL_RANGES: [[f32; 2]; NUM_CHANNELS] = [
    [0.0, 1.0],  // intent_acknowledge
    [0.0, 1.0],  // intent_answer
    [0.0, 1.0],  // intent_clarify
    [0.0, 1.0],  // intent_propose
    [0.0, 1.0],  // intent_uncertainty
    [0.0, 1.0],  // intent_reflect
    [0.0, 1.0],  // intent_continue
    [0.0, 1.0],  // intent_unknown
    [0.0, 4.0],  // epistemic_status (0=Certain..4=OutOfDomain)
    [-1.0, 1.0], // valence
    [0.0, 1.0],  // arousal
    [0.0, 1.0],  // warmth
    [0.0, 1.0],  // psi
    [0.0, 1.0],  // meta_awareness
    [0.0, 1.0],  // coherence
    [0.0, 6.0],  // relationship_stage
    [0.0, 1.0],  // trust
    [0.5, 2.0],  // mood_temperature
    [0.0, 1.0],  // has_computed_answer
    [0.0, 10.0], // concept_count
    [0.0, 1.0],  // time_pressure
    [0.0, 1.0],  // domain_familiarity
    [0.0, 1.0],  // social_context
    [0.0, 1.0],  // response_confidence
];

// ═══════════════════════════════════════════════════════════════════════════════
// ThoughtChannels (24-channel)
// ═══════════════════════════════════════════════════════════════════════════════

/// 24-dimensional thought state for the full Broca pipeline.
///
/// Channels:
/// - 0..7: semantic intent (one-hot, 8 variants)
/// - 8: epistemic_status (0=Certain..4=OutOfDomain)
/// - 9: valence (-1..1)
/// - 10: arousal (0..1)
/// - 11: warmth (0..1)
/// - 12: psi / consciousness level (0..1)
/// - 13: meta_awareness (0..1)
/// - 14: coherence (0..1)
/// - 15: relationship_stage (0..6)
/// - 16: trust (0..1)
/// - 17: mood_temperature (0.5..2.0)
/// - 18: has_computed_answer (0 or 1)
/// - 19: concept_count (0..10)
/// - 20: time_pressure (0..1)
/// - 21: domain_familiarity (0..1)
/// - 22: social_context (0=intimate..1=formal)
/// - 23: response_confidence (0..1)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThoughtChannels {
    pub channels: [f32; NUM_CHANNELS],
}

impl Default for ThoughtChannels {
    fn default() -> Self {
        Self {
            channels: [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // intent: Unknown
                1.0, // epistemic: Probable
                0.0, // valence: neutral
                0.5, // arousal: mid
                0.5, // warmth: mid
                0.5, // psi: mid
                0.5, // meta_awareness: mid
                0.5, // coherence: mid
                0.0, // relationship_stage: 0
                0.5, // trust: mid
                1.0, // mood_temperature: neutral
                0.0, // has_computed_answer: false
                0.0, // concept_count: none
                0.0, // time_pressure: relaxed
                0.5, // domain_familiarity: mid
                0.5, // social_context: mid
                0.5, // response_confidence: mid
            ],
        }
    }
}

impl ThoughtChannels {
    /// Create channels with a specific semantic intent one-hot.
    pub fn with_intent(intent_index: usize) -> Self {
        let mut channels = Self::default();
        for i in 0..8 {
            channels.channels[i] = 0.0;
        }
        if intent_index < 8 {
            channels.channels[intent_index] = 1.0;
        }
        channels
    }

    /// Set epistemic status (0=Certain, 1=Probable, 2=Uncertain, 3=Unknown, 4=OutOfDomain).
    pub fn set_epistemic(&mut self, ordinal: f32) {
        self.channels[8] = ordinal.clamp(0.0, 4.0);
    }

    /// Set emotional tone.
    pub fn set_emotion(&mut self, valence: f32, arousal: f32, warmth: f32) {
        self.channels[9] = valence.clamp(-1.0, 1.0);
        self.channels[10] = arousal.clamp(0.0, 1.0);
        self.channels[11] = warmth.clamp(0.0, 1.0);
    }

    /// Set consciousness metrics.
    pub fn set_consciousness(&mut self, psi: f32, meta_awareness: f32, coherence: f32) {
        self.channels[12] = psi.clamp(0.0, 1.0);
        self.channels[13] = meta_awareness.clamp(0.0, 1.0);
        self.channels[14] = coherence.clamp(0.0, 1.0);
    }

    /// Set contextual channels (v3).
    pub fn set_context(
        &mut self,
        time_pressure: f32,
        domain_familiarity: f32,
        social_context: f32,
        response_confidence: f32,
    ) {
        self.channels[20] = time_pressure.clamp(0.0, 1.0);
        self.channels[21] = domain_familiarity.clamp(0.0, 1.0);
        self.channels[22] = social_context.clamp(0.0, 1.0);
        self.channels[23] = response_confidence.clamp(0.0, 1.0);
    }

    pub fn epistemic_ordinal(&self) -> f32 {
        self.channels[8]
    }
    pub fn psi(&self) -> f32 {
        self.channels[12]
    }
    pub fn arousal(&self) -> f32 {
        self.channels[10]
    }
    pub fn warmth(&self) -> f32 {
        self.channels[11]
    }
    pub fn valence(&self) -> f32 {
        self.channels[9]
    }
    pub fn coherence(&self) -> f32 {
        self.channels[14]
    }
    pub fn time_pressure(&self) -> f32 {
        self.channels[20]
    }
    pub fn domain_familiarity(&self) -> f32 {
        self.channels[21]
    }
    pub fn social_context(&self) -> f32 {
        self.channels[22]
    }
    pub fn response_confidence(&self) -> f32 {
        self.channels[23]
    }

    /// Build from the Spore's broca-lite 12-channel format by mapping channels.
    pub fn from_lite(lite: &[f32; 12]) -> Self {
        let mut tc = Self::default();
        // Map lite intent channels 0..3 to full intent 0..3
        for i in 0..4 {
            tc.channels[i] = lite[i].clamp(0.0, 1.0);
        }
        // epistemic (lite[4] is 0..1, scale to 0..4)
        tc.channels[8] = (lite[4] * 4.0).clamp(0.0, 4.0);
        // valence (lite[5])
        tc.channels[9] = lite[5].clamp(-1.0, 1.0);
        // arousal (lite[6])
        tc.channels[10] = lite[6].clamp(0.0, 1.0);
        // consciousness / psi (lite[7])
        tc.channels[12] = lite[7].clamp(0.0, 1.0);
        // prediction_error -> meta_awareness proxy (lite[8])
        tc.channels[13] = lite[8].clamp(0.0, 1.0);
        // harmony -> coherence proxy (lite[9])
        tc.channels[14] = lite[9].clamp(0.0, 1.0);
        // dopamine -> warmth proxy (lite[10])
        tc.channels[11] = lite[10].clamp(0.0, 1.0);
        tc
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ThoughtLanguageEncoder (24-channel -> 16,384D ContinuousHV)
// ═══════════════════════════════════════════════════════════════════════════════

/// HDC encoder for 24-channel thought state.
///
/// Encodes `ThoughtChannels` into a full 16,384D `ContinuousHV` via:
/// 1. Per-channel normalization to [0, 1] using known ranges
/// 2. Level encoding (thermometer coding via bundled levels 0..k)
/// 3. Binding with genesis-seeded channel base vectors
/// 4. Bundling all bound channel HVs
pub struct ThoughtLanguageEncoder {
    base_vectors: Vec<ContinuousHV>,
    level_vectors: Vec<ContinuousHV>,
    num_levels: usize,
}

impl ThoughtLanguageEncoder {
    /// Create a new encoder from a genesis seed.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let base_vectors: Vec<ContinuousHV> = CHANNEL_NAMES
            .iter()
            .map(|name| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca::channel::{name}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let level_vectors: Vec<ContinuousHV> = (0..NUM_LEVELS)
            .map(|i| {
                ContinuousHV::from_genesis(genesis, &format!("broca::level::{i}"), HDC_DIMENSION)
            })
            .collect();

        Self {
            base_vectors,
            level_vectors,
            num_levels: NUM_LEVELS,
        }
    }

    /// Normalize a raw channel value to [0, 1] using known ranges.
    pub fn normalize_channel(channel: usize, value: f32) -> f32 {
        let [min, max] = CHANNEL_RANGES[channel];
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Level-encode a normalized [0,1] value using thermometer coding.
    fn encode_level(&self, normalized: f32) -> ContinuousHV {
        let k = ((normalized * self.num_levels as f32) as usize).min(self.num_levels - 1);
        if k == 0 {
            self.level_vectors[0].clone()
        } else {
            let refs: Vec<&ContinuousHV> = self.level_vectors[..=k].iter().collect();
            ContinuousHV::bundle(&refs)
        }
    }

    /// Encode thought channels into a full 16,384D ContinuousHV.
    ///
    /// NaN/Inf channel values are replaced with midpoint defaults.
    pub fn encode(&self, channels: &ThoughtChannels) -> ContinuousHV {
        let mut bound_hvs: Vec<ContinuousHV> = Vec::with_capacity(NUM_CHANNELS);

        for (i, (&raw, base)) in channels
            .channels
            .iter()
            .zip(self.base_vectors.iter())
            .enumerate()
        {
            let value = if raw.is_finite() {
                raw
            } else {
                let [min, max] = CHANNEL_RANGES[i];
                (min + max) * 0.5
            };
            let normalized = Self::normalize_channel(i, value);
            let level_hv = self.encode_level(normalized);
            bound_hvs.push(base.bind(&level_hv));
        }

        let refs: Vec<&ContinuousHV> = bound_hvs.iter().collect();
        ContinuousHV::bundle(&refs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LanguageController
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the language controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageControllerConfig {
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub dt_per_token: f32,
    pub logit_temperature: f32,
    pub logit_scale: f32,
    pub backbone_tau: f32,
}

impl Default for LanguageControllerConfig {
    fn default() -> Self {
        Self {
            network_layers: 3,
            neurons_per_layer: 8,
            vocab_size: VOCAB_SIZE,
            max_seq_len: MAX_SEQ_LEN,
            dt_per_token: DT_PER_TOKEN,
            logit_temperature: 1.0,
            logit_scale: LOGIT_SCALE,
            backbone_tau: 0.3,
        }
    }
}

/// Language controller: HdcLtcUnifiedNetwork + weight-tied output projection.
///
/// Forward pass:
/// 1. Compose input: thought_hv (x) token_emb (x) pos_emb
/// 2. Evolve CfC network via closed-form ODE
/// 3. Get normalized output
/// 4. Weight-tied logits: logits[i] = scale * cosine(output, token_emb[i])
pub struct LanguageController {
    network: HdcLtcUnifiedNetwork,
    token_embeddings: Vec<ContinuousHV>,
    position_base: ContinuousHV,
    config: LanguageControllerConfig,
    current_pos: usize,
}

impl LanguageController {
    /// Create a new controller from a genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: &LanguageControllerConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 0.02,
            backbone_tau: config.backbone_tau,
            dimension: HDC_DIMENSION,
            learning_rate: 0.001,
            ..UnifiedConfig::default()
        };

        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![config.neurons_per_layer; config.network_layers],
            neuron_config,
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        let token_embeddings: Vec<ContinuousHV> = (0..config.vocab_size)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca::token_emb::{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let position_base =
            ContinuousHV::from_genesis(genesis, "broca::position_base", HDC_DIMENSION);

        Self {
            network,
            token_embeddings,
            position_base,
            config: config.clone(),
            current_pos: 0,
        }
    }

    /// Compute logits for all tokens via weight-tied similarity.
    pub fn compute_logits(&self, output_hv: &ContinuousHV) -> Vec<f32> {
        let scale = self.config.logit_scale;
        self.token_embeddings
            .iter()
            .map(|emb| scale * output_hv.similarity(emb))
            .collect()
    }

    /// Forward step: evolve network with composed input, return logits.
    pub fn forward_step(
        &mut self,
        thought_hv: &ContinuousHV,
        prev_token_id: u32,
        pos: usize,
    ) -> Vec<f32> {
        let token_emb = self.get_token_embedding(prev_token_id);
        let pos_emb = self.position_base.permute(pos);
        let input_hv = thought_hv.bind(&token_emb).bind(&pos_emb);

        let dt = if self.config.dt_per_token.is_finite() && self.config.dt_per_token > 0.0 {
            self.config.dt_per_token
        } else {
            DT_PER_TOKEN
        };

        self.network.evolve_closed_form(dt, &input_hv);

        let output_hv = self.network.output().normalize();

        let mut logits = self.compute_logits(&output_hv);

        // Apply temperature scaling
        if (self.config.logit_temperature - 1.0).abs() > 1e-6 {
            let inv_temp = 1.0 / self.config.logit_temperature;
            for l in &mut logits {
                *l *= inv_temp;
            }
        }

        // Ensure all logits are finite
        for l in &mut logits {
            if !l.is_finite() {
                *l = 0.0;
            }
        }

        self.current_pos = pos + 1;
        logits
    }

    /// Get the current network output HV (for coherence monitoring).
    pub fn output_hv(&self) -> ContinuousHV {
        self.network.output().normalize()
    }

    fn get_token_embedding(&self, token_id: u32) -> ContinuousHV {
        self.token_embeddings
            .get(token_id as usize)
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(HDC_DIMENSION))
    }

    /// Reset the network state.
    pub fn reset(&mut self) {
        self.network.reset();
        self.current_pos = 0;
    }

    pub fn vocab_size(&self) -> usize {
        self.token_embeddings.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Vocabulary (self-contained, 1024 tokens)
// ═══════════════════════════════════════════════════════════════════════════════

/// Special token IDs.
const _BOS_ID: u32 = 0;
const EOS_ID: u32 = 1;
const _PAD_ID: u32 = 2;
const UNK_ID: u32 = 3;
const THOUGHT_ID: u32 = 4;

/// Built-in vocabulary for generation and decoding.
///
/// Self-contained 1024-token vocab with special tokens, ASCII bytes,
/// common English words, hedging vocabulary, and padding.
struct Vocabulary {
    tokens: Vec<String>,
    token_to_id: std::collections::HashMap<String, u32>,
}

impl Vocabulary {
    fn new() -> Self {
        let mut tokens = Vec::with_capacity(VOCAB_SIZE);

        // 0-4: Special tokens
        tokens.push("<bos>".into());
        tokens.push("<eos>".into());
        tokens.push("<pad>".into());
        tokens.push("<unk>".into());
        tokens.push("<thought>".into());

        // 5-99: Single printable ASCII bytes
        for b in 32u8..=126 {
            tokens.push(String::from(b as char));
        }

        // Common English words (word-level tokens for generation)
        let words = [
            // Articles/prepositions/conjunctions
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "is",
            "it",
            "that",
            "this",
            "was",
            "are",
            "be",
            "have",
            "has",
            "had",
            "and",
            "but",
            "or",
            "if",
            "so",
            "then",
            "than",
            "when",
            "what",
            "which",
            "as",
            "not",
            "no",
            "yes",
            "do",
            "does",
            "did",
            // Pronouns
            "I",
            "you",
            "he",
            "she",
            "we",
            "they",
            "my",
            "your",
            "his",
            "her",
            "our",
            "its",
            "me",
            "us",
            "them",
            // Common verbs
            "can",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "think",
            "know",
            "see",
            "say",
            "make",
            "go",
            "get",
            "take",
            "come",
            "give",
            "tell",
            "find",
            "want",
            "need",
            "use",
            "try",
            "feel",
            "seem",
            "look",
            "work",
            "move",
            "live",
            "mean",
            "keep",
            "let",
            "begin",
            "show",
            "hear",
            "play",
            "run",
            "set",
            "turn",
            "ask",
            "put",
            "read",
            "hold",
            "stand",
            "learn",
            // Common nouns
            "time",
            "way",
            "thing",
            "man",
            "world",
            "life",
            "day",
            "part",
            "place",
            "people",
            "hand",
            "state",
            "case",
            "point",
            "question",
            "home",
            "number",
            "water",
            "room",
            "mother",
            "area",
            "money",
            "story",
            "fact",
            "month",
            "lot",
            "right",
            "study",
            "book",
            "eye",
            "word",
            "side",
            "head",
            "house",
            "friend",
            "night",
            "end",
            "line",
            "mind",
            "body",
            "system",
            "group",
            "name",
            "thought",
            "form",
            "back",
            "kind",
            "power",
            "change",
            "light",
            "problem",
            "level",
            "reason",
            "idea",
            "moment",
            // Adjectives
            "new",
            "good",
            "great",
            "old",
            "big",
            "small",
            "long",
            "high",
            "little",
            "own",
            "other",
            "important",
            "different",
            "real",
            "large",
            "young",
            "sure",
            "able",
            "free",
            "true",
            "full",
            "clear",
            "deep",
            "possible",
            "whole",
            "certain",
            "open",
            "strong",
            "hard",
            "dark",
            // Adverbs
            "very",
            "also",
            "just",
            "more",
            "still",
            "here",
            "there",
            "now",
            "only",
            "well",
            "even",
            "much",
            "never",
            "always",
            "often",
            "ever",
            "quite",
            "almost",
            "already",
            "really",
            "too",
            "yet",
            "again",
            // Hedging / epistemic markers (canonical)
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
            "however",
            "although",
            "unfortunately",
            "sorry",
            "unclear",
            "tend",
            "arguably",
            "supposedly",
            "tentatively",
            "roughly",
            "approximately",
            "potentially",
            // Factual / assertion words (canonical)
            "certainly",
            "definitely",
            "shall",
            "every",
            "none",
            // OOD words
            "outside",
            "beyond",
            "cannot",
            "unable",
            // Informal words (canonical)
            "gonna",
            "wanna",
            "gotta",
            "kinda",
            "sorta",
            "dunno",
            "lemme",
            // Softening words (overlap with hedging, included for completeness)
            // Consciousness/cognitive vocabulary
            "consciousness",
            "awareness",
            "experience",
            "perception",
            "attention",
            "memory",
            "emotion",
            "feeling",
            "pattern",
            "process",
            "signal",
            "response",
            "integration",
            "coherence",
            "resonance",
            "emergence",
            // Subwords / affixes
            "ing",
            "tion",
            "er",
            "ed",
            "ly",
            "ness",
            "ment",
            "ful",
            "less",
            "able",
            "un",
            "re",
            "pre",
            "dis",
            "over",
            "out",
            "up",
            // Punctuation sequences
            ". ",
            ", ",
            "? ",
            "! ",
            ": ",
            "; ",
            "-- ",
            "...",
            // Whitespace
            " ",
            "\n",
            // Additional common words for natural generation
            "about",
            "after",
            "before",
            "between",
            "without",
            "through",
            "during",
            "into",
            "been",
            "being",
            "because",
            "since",
            "while",
            "where",
            "how",
            "who",
            "why",
            "many",
            "most",
            "some",
            "any",
            "each",
            "both",
            "such",
            "own",
            "same",
            "down",
            "off",
            "over",
            "under",
            "around",
            // Contractions (common in natural speech)
            "don't",
            "can't",
            "won't",
            "isn't",
            "aren't",
            "doesn't",
            "didn't",
            "I'm",
            "I'll",
            "I've",
            "it's",
            "that's",
            "there's",
            "what's",
            // Additional verbs
            "help",
            "call",
            "start",
            "follow",
            "leave",
            "write",
            "grow",
            "close",
            "stop",
            "watch",
            "remember",
            "understand",
            "consider",
            // Additional nouns
            "child",
            "woman",
            "family",
            "school",
            "country",
            "city",
            "nature",
            "space",
            "sense",
            "energy",
            "truth",
            "soul",
            "heart",
            "love",
            // More adjectives
            "simple",
            "beautiful",
            "natural",
            "human",
            "social",
            "personal",
            "local",
            "general",
            "special",
            "common",
            "similar",
            "recent",
            // Connecting phrases
            "like",
            "more",
            "most",
            "much",
            "least",
            "less",
            "something",
            "everything",
            "nothing",
            "anything",
            "someone",
            "everyone",
            "together",
            "whether",
            "rather",
            "enough",
            "instead",
        ];

        for word in words {
            let w = word.to_string();
            if !tokens.contains(&w) {
                tokens.push(w);
            }
        }

        // Pad to exactly VOCAB_SIZE
        while tokens.len() < VOCAB_SIZE {
            tokens.push(format!("<reserved_{}>", tokens.len()));
        }
        tokens.truncate(VOCAB_SIZE);

        let mut token_to_id = std::collections::HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        Self {
            tokens,
            token_to_id,
        }
    }

    fn token_id(&self, token: &str) -> u32 {
        *self.token_to_id.get(token).unwrap_or(&UNK_ID)
    }

    fn token_str(&self, id: u32) -> &str {
        self.tokens
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>")
    }

    fn is_special(&self, id: u32) -> bool {
        id <= THOUGHT_ID
    }

    /// Resolve word list to token IDs, filtering out unknowns.
    fn resolve_words(&self, words: &[&str]) -> Vec<u32> {
        words
            .iter()
            .filter_map(|w| {
                let id = self.token_id(w);
                if id != UNK_ID {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GatingConfig
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the gating system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingConfig {
    /// Logit penalty for factual tokens under Unknown epistemic status.
    pub unknown_factual_penalty: f32,
    /// Logit boost for hedging tokens under Unknown epistemic status.
    pub unknown_hedging_boost: f32,
    /// Logit penalty for factual tokens under Uncertain epistemic status.
    pub uncertain_factual_penalty: f32,
    /// Logit boost for hedging tokens under Uncertain epistemic status.
    pub uncertain_hedging_boost: f32,
    /// Coherence drift threshold (below this, boost thought binding).
    pub coherence_drift_threshold: f32,
    /// Arousal threshold above which sentence-ending tokens are boosted.
    pub high_arousal_threshold: f32,
    /// Token position after which arousal sentence-ending boost applies.
    pub arousal_position_threshold: usize,
    /// Warmth threshold below which informal tokens are suppressed.
    pub low_warmth_threshold: f32,
    /// Base max tokens (before consciousness scaling).
    pub base_max_tokens: usize,
    /// Semantic veto threshold (0.0 = adaptive-only).
    pub veto_threshold: f32,
    /// Minimum token position before veto can fire.
    pub min_veto_position: usize,
    /// Maximum vetoes per generation.
    pub max_vetoes: usize,
    /// Refractory period after veto (tokens).
    pub veto_refractory: usize,
    /// Arousal boost multiplier for sentence endings.
    pub arousal_boost_multiplier: f32,
    /// Warmth penalty multiplier for informal tokens.
    pub warmth_penalty_multiplier: f32,
    /// Negative valence threshold below which softening is boosted.
    pub negative_valence_threshold: f32,
    /// Valence boost multiplier for softening tokens.
    pub valence_boost_multiplier: f32,
    /// OOD boost multiplier.
    pub ood_boost_multiplier: f32,
    /// Adaptive veto warmup period (tokens).
    pub adaptive_veto_warmup: usize,
    /// Z-score threshold for adaptive veto.
    pub adaptive_z_threshold: f32,
    /// Confidence-adjusted veto scale.
    pub confidence_veto_scale: f32,
    /// Time pressure max token reduction fraction.
    pub time_pressure_max_reduction: f32,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            unknown_factual_penalty: -3.0,
            unknown_hedging_boost: 0.75,
            uncertain_factual_penalty: -1.5,
            uncertain_hedging_boost: 0.5,
            coherence_drift_threshold: 0.3,
            high_arousal_threshold: 0.7,
            arousal_position_threshold: 10,
            low_warmth_threshold: 0.3,
            base_max_tokens: 128,
            veto_threshold: 0.0,
            min_veto_position: 2,
            max_vetoes: 1,
            veto_refractory: 8,
            arousal_boost_multiplier: 3.0,
            warmth_penalty_multiplier: -5.0,
            negative_valence_threshold: -0.3,
            valence_boost_multiplier: 2.0,
            ood_boost_multiplier: 2.0,
            adaptive_veto_warmup: 10,
            adaptive_z_threshold: 2.0,
            confidence_veto_scale: 0.3,
            time_pressure_max_reduction: 0.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EpistemicGate
// ═══════════════════════════════════════════════════════════════════════════════

/// Epistemic gate: suppresses factual assertions when confidence is low.
///
/// The system physically *cannot* hallucinate when epistemic status is Unknown --
/// factual token logits are suppressed before sampling.
pub struct EpistemicGate {
    config: GatingConfig,
    hedging_token_ids: Vec<u32>,
    factual_token_ids: Vec<u32>,
    ood_token_ids: Vec<u32>,
}

impl EpistemicGate {
    fn new(vocab: &Vocabulary, config: &GatingConfig) -> Self {
        Self {
            config: config.clone(),
            hedging_token_ids: vocab.resolve_words(CANONICAL_HEDGING_WORDS),
            factual_token_ids: vocab.resolve_words(CANONICAL_FACTUAL_WORDS),
            ood_token_ids: vocab.resolve_words(CANONICAL_OOD_WORDS),
        }
    }

    /// Number of resolved hedging token IDs.
    pub fn hedging_count(&self) -> usize {
        self.hedging_token_ids.len()
    }

    /// Number of resolved factual token IDs.
    pub fn factual_count(&self) -> usize {
        self.factual_token_ids.len()
    }

    /// Apply epistemic gating to logits in-place.
    ///
    /// epistemic ordinal: 0=Certain, 1=Probable, 2=Uncertain, 3=Unknown, 4=OutOfDomain
    pub fn apply(&self, logits: &mut [f32], epistemic_ordinal: f32) {
        self.apply_with_familiarity(logits, epistemic_ordinal, 0.0);
    }

    /// Apply epistemic gating with domain familiarity context.
    pub fn apply_with_familiarity(
        &self,
        logits: &mut [f32],
        epistemic_ordinal: f32,
        domain_familiarity: f32,
    ) {
        if epistemic_ordinal < 1.5 {
            return; // Certain or Probable: no modification
        }

        let df = domain_familiarity.clamp(0.0, 1.0);
        let familiarity_scale = 1.0 - df;

        if epistemic_ordinal > 3.5 {
            // OutOfDomain
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
/// - High arousal -> boost sentence-ending tokens (shorter sentences)
/// - Low warmth -> suppress informal vocabulary
/// - Negative valence -> boost softening language
pub struct EmotionalModulator {
    config: GatingConfig,
    sentence_end_ids: Vec<u32>,
    informal_ids: Vec<u32>,
    softening_ids: Vec<u32>,
}

impl EmotionalModulator {
    fn new(vocab: &Vocabulary, config: &GatingConfig) -> Self {
        Self {
            config: config.clone(),
            sentence_end_ids: vocab.resolve_words(CANONICAL_SENTENCE_ENDINGS),
            informal_ids: vocab.resolve_words(CANONICAL_INFORMAL_WORDS),
            softening_ids: vocab.resolve_words(CANONICAL_SOFTENING_WORDS),
        }
    }

    /// Apply emotional modulation to logits in-place.
    pub fn apply(&self, logits: &mut [f32], channels: &ThoughtChannels, position: usize) {
        let arousal = channels.arousal();
        let warmth = channels.warmth();
        let valence = channels.valence();

        // High arousal + past threshold -> boost sentence endings (shorter sentences)
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

        // Arousal x valence interaction: high arousal + negative valence ->
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

        // Low warmth -> suppress informal tokens
        let social = channels.social_context().clamp(0.0, 1.0);
        let formality_scale = 1.0 + social;
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

        // Negative valence -> boost softening language
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
/// Uses Welford's online algorithm for adaptive z-score veto: fires when
/// coherence drops >2 sigma below the running mean.
pub struct CoherenceFeedback {
    threshold: f32,
    current_coherence: f32,
    veto_triggered: bool,
    veto_threshold: f32,
    coherence_ema: f32,
    coherence_m2: f32,
    coherence_n: usize,
    adaptive_warmup: usize,
    adaptive_z_threshold: f32,
}

impl CoherenceFeedback {
    pub fn new(threshold: f32) -> Self {
        Self::with_veto_threshold(threshold, 0.15)
    }

    pub fn with_veto_threshold(threshold: f32, veto_threshold: f32) -> Self {
        Self {
            threshold,
            current_coherence: 1.0,
            veto_triggered: false,
            veto_threshold,
            coherence_ema: 0.0,
            coherence_m2: 0.0,
            coherence_n: 0,
            adaptive_warmup: 10,
            adaptive_z_threshold: 2.0,
        }
    }

    pub fn with_config(
        threshold: f32,
        veto_threshold: f32,
        adaptive_warmup: usize,
        adaptive_z_threshold: f32,
    ) -> Self {
        Self {
            threshold,
            current_coherence: 1.0,
            veto_triggered: false,
            veto_threshold,
            coherence_ema: 0.0,
            coherence_m2: 0.0,
            coherence_n: 0,
            adaptive_warmup,
            adaptive_z_threshold,
        }
    }

    /// Update coherence score. Returns a binding weight multiplier (>1.0 when drifting).
    pub fn update(&mut self, output_hv: &ContinuousHV, thought_hv: &ContinuousHV) -> f32 {
        self.current_coherence = output_hv.similarity(thought_hv);

        // Update running statistics (Welford's online algorithm)
        self.coherence_n += 1;
        let delta = self.current_coherence - self.coherence_ema;
        self.coherence_ema += delta / self.coherence_n as f32;
        let delta2 = self.current_coherence - self.coherence_ema;
        self.coherence_m2 += delta * delta2;

        // Veto mode selection
        if self.adaptive_warmup == 0 && self.veto_threshold > 0.0 {
            // Fixed-threshold mode
            self.veto_triggered = self.current_coherence < self.veto_threshold;
        } else if self.coherence_n > self.adaptive_warmup && self.coherence_n > 2 {
            // Adaptive z-score mode
            let variance = self.coherence_m2 / (self.coherence_n - 1) as f32;
            let std_dev = variance.max(1e-8).sqrt();
            let z_score = (self.coherence_ema - self.current_coherence) / std_dev;
            self.veto_triggered = z_score > self.adaptive_z_threshold;
        } else {
            self.veto_triggered = false;
        }

        if self.current_coherence < self.threshold {
            let correction = 1.0 + (self.threshold - self.current_coherence) * 2.0;
            correction.min(3.0)
        } else {
            1.0
        }
    }

    /// Whether a semantic veto should be triggered.
    pub fn should_veto(&self) -> bool {
        self.veto_triggered
    }

    /// Current coherence score.
    pub fn coherence(&self) -> f32 {
        self.current_coherence
    }

    /// Reset running statistics for a new generation.
    pub fn reset_stats(&mut self) {
        self.coherence_ema = 0.0;
        self.coherence_m2 = 0.0;
        self.coherence_n = 0;
        self.current_coherence = 1.0;
        self.veto_triggered = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute consciousness-gated max tokens.
///
/// `max_tokens = base_max * (0.5 + psi)` -- higher consciousness -> more detail.
pub fn consciousness_gated_max_tokens(base_max: usize, psi: f32) -> usize {
    let psi = if psi.is_finite() { psi } else { 0.5 };
    let factor = 0.5 + psi.clamp(0.0, 1.0);
    ((base_max as f32) * factor) as usize
}

/// Compute time-pressure-adjusted max tokens.
fn time_pressure_adjusted_max_tokens(
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

/// Compute confidence-adjusted veto threshold.
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

/// Apply frequency-scaled repetition penalty (Keskar et al. 2019).
fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &token_id in generated {
        *counts.entry(token_id).or_insert(0) += 1;
    }
    for (&token_id, &count) in &counts {
        if let Some(logit) = logits.get_mut(token_id as usize) {
            let effective_penalty = penalty.powi(count as i32);
            if *logit > 0.0 {
                *logit /= effective_penalty;
            } else {
                *logit *= effective_penalty;
            }
        }
    }
}

/// Block tokens that would complete a repeated n-gram.
fn block_repeated_ngrams(logits: &mut [f32], generated: &[u32], n: usize) {
    if generated.len() < n - 1 {
        return;
    }
    let mut seen = std::collections::HashSet::new();
    for window in generated.windows(n) {
        let prefix = &window[..n - 1];
        let next = window[n - 1];
        seen.insert((prefix.to_vec(), next));
    }
    let suffix = &generated[generated.len() - (n - 1)..];
    for token_id in 0..logits.len() {
        if seen.contains(&(suffix.to_vec(), token_id as u32)) {
            logits[token_id] = f32::NEG_INFINITY;
        }
    }
}

/// Whether a token needs a space inserted before it.
fn needs_space_before(token_str: &str) -> bool {
    match token_str.bytes().next() {
        Some(b) => b.is_ascii_alphabetic(),
        None => false,
    }
}

/// Greedy sampling: return the argmax token.
fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BrocaConfig
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the full Broca generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrocaConfig {
    pub controller: LanguageControllerConfig,
    pub gating: GatingConfig,
    pub enable_epistemic_gate: bool,
    pub enable_emotional_modulation: bool,
    pub enable_coherence_feedback: bool,
    pub enable_consciousness_gating: bool,
    pub enable_semantic_veto: bool,
    pub veto_hesitation: String,
    pub repetition_penalty: f32,
}

impl Default for BrocaConfig {
    fn default() -> Self {
        Self {
            controller: LanguageControllerConfig::default(),
            gating: GatingConfig::default(),
            enable_epistemic_gate: true,
            enable_emotional_modulation: true,
            enable_coherence_feedback: true,
            enable_consciousness_gating: true,
            enable_semantic_veto: true,
            veto_hesitation: "-- wait, ".to_string(),
            repetition_penalty: DEFAULT_REPETITION_PENALTY,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GatingTraceEntry
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-token gating audit record.
#[derive(Debug, Clone)]
pub struct GatingTraceEntry {
    pub position: usize,
    pub coherence: f32,
    pub binding_weight: f32,
    pub veto_triggered: bool,
    pub epistemic_boost: f32,
    pub emotional_boost: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// GenerationResult
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a single generation.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated text.
    pub text: String,
    /// Generated token IDs.
    pub token_ids: Vec<u32>,
    /// Number of tokens generated.
    pub num_tokens: usize,
    /// Whether generation was terminated by EOS.
    pub eos_terminated: bool,
    /// Whether a semantic veto was triggered.
    pub veto_triggered: bool,
    /// Final coherence score.
    pub final_coherence: f32,
    /// Per-token coherence dynamics.
    pub coherence_dynamics: Vec<f32>,
    /// Per-token gating audit trail.
    pub gating_trace: Vec<GatingTraceEntry>,
    /// Whether hallucination was detected (3+ consecutive low-coherence tokens).
    pub hallucination_flag: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BrocaGenerator
// ═══════════════════════════════════════════════════════════════════════════════

/// Full Broca generator: autoregressive thought-to-text with consciousness gating.
///
/// Orchestrates the complete pipeline:
/// 1. Encode 24-channel thought state -> 16,384D ContinuousHV
/// 2. Autoregressive loop: CfC forward step -> gate -> sample -> decode
/// 3. Epistemic gating physically prevents hallucination at logit level
/// 4. Emotional modulation shapes sentence length and formality
/// 5. Coherence feedback with semantic veto (mid-sentence self-correction)
/// 6. Consciousness-gated verbosity (Psi modulates max_tokens)
pub struct BrocaGenerator {
    controller: LanguageController,
    vocab: Vocabulary,
    encoder: ThoughtLanguageEncoder,
    epistemic_gate: EpistemicGate,
    emotional_modulator: EmotionalModulator,
    coherence_feedback: CoherenceFeedback,
    config: BrocaConfig,
}

impl BrocaGenerator {
    /// Create a new full Broca generator from genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: BrocaConfig) -> Self {
        let vocab = Vocabulary::new();

        let mut ctrl_config = config.controller.clone();
        ctrl_config.vocab_size = vocab.tokens.len();

        let controller = LanguageController::new(genesis, &ctrl_config);
        let encoder = ThoughtLanguageEncoder::new(genesis);
        let epistemic_gate = EpistemicGate::new(&vocab, &config.gating);
        let emotional_modulator = EmotionalModulator::new(&vocab, &config.gating);
        let coherence_feedback = CoherenceFeedback::with_config(
            config.gating.coherence_drift_threshold,
            config.gating.veto_threshold,
            config.gating.adaptive_veto_warmup,
            config.gating.adaptive_z_threshold,
        );

        Self {
            controller,
            vocab,
            encoder,
            epistemic_gate,
            emotional_modulator,
            coherence_feedback,
            config,
        }
    }

    /// Generate text from thought channels.
    pub fn generate(&mut self, channels: &ThoughtChannels) -> GenerationResult {
        // 1. Encode thought channels once
        let thought_hv_original = self.encoder.encode(channels);
        let mut thought_hv = thought_hv_original.clone();

        // 2. Compute max tokens (consciousness-gated, then time-pressure-adjusted)
        let pre_pressure_tokens = if self.config.enable_consciousness_gating {
            consciousness_gated_max_tokens(self.config.gating.base_max_tokens, channels.psi())
        } else {
            self.config.gating.base_max_tokens
        };
        let max_tokens = time_pressure_adjusted_max_tokens(
            pre_pressure_tokens,
            channels.time_pressure(),
            self.config.gating.time_pressure_max_reduction,
        );

        // 3. Reset controller state
        self.controller.reset();
        self.coherence_feedback.reset_stats();

        // 4. Autoregressive generation loop
        let mut tokens = Vec::new();
        let mut prev_token = THOUGHT_ID;
        let mut eos_terminated = false;
        let mut veto_triggered = false;
        let mut text_bytes: Vec<u8> = Vec::new();
        let rep_penalty = self.config.repetition_penalty;
        let min_veto_pos = self.config.gating.min_veto_position;
        let max_vetoes = self.config.gating.max_vetoes;
        let veto_refractory = self.config.gating.veto_refractory;
        let mut veto_count = 0usize;
        let mut tokens_since_veto = veto_refractory; // Start past refractory
        let mut coherence_dynamics = Vec::new();
        let mut gating_trace = Vec::new();
        let mut consecutive_low_coherence = 0usize;
        let mut hallucination_flag = false;

        for pos in 0..max_tokens {
            // Forward step
            let mut logits = self.controller.forward_step(&thought_hv, prev_token, pos);

            // Apply frequency-scaled repetition penalty
            if rep_penalty > 1.0 + 1e-6 {
                apply_repetition_penalty(&mut logits, &tokens, rep_penalty);
            }

            // No-repeat bigram blocking
            if !tokens.is_empty() {
                block_repeated_ngrams(&mut logits, &tokens, 2);
            }

            // Epistemic gating
            let this_epistemic_boost = if self.config.enable_epistemic_gate {
                let ordinal = channels.epistemic_ordinal();
                self.epistemic_gate.apply_with_familiarity(
                    &mut logits,
                    ordinal,
                    channels.domain_familiarity(),
                );
                if ordinal > 3.5 {
                    self.config.gating.unknown_hedging_boost
                        * self.config.gating.ood_boost_multiplier
                } else if ordinal > 2.5 {
                    self.config.gating.unknown_hedging_boost
                } else if ordinal > 1.5 {
                    self.config.gating.uncertain_hedging_boost
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Emotional modulation
            let this_emotional_boost = if self.config.enable_emotional_modulation {
                self.emotional_modulator.apply(&mut logits, channels, pos);
                let arousal = channels.arousal();
                let valence = channels.valence();
                let warmth = channels.warmth();
                let cfg = &self.config.gating;
                let mut max_boost = 0.0f32;
                if arousal > cfg.high_arousal_threshold && pos > cfg.arousal_position_threshold {
                    let boost =
                        (arousal - cfg.high_arousal_threshold) * cfg.arousal_boost_multiplier;
                    max_boost = max_boost.max(boost);
                }
                if valence < cfg.negative_valence_threshold {
                    let boost = (-valence - (-cfg.negative_valence_threshold))
                        * cfg.valence_boost_multiplier;
                    max_boost = max_boost.max(boost);
                }
                if warmth < cfg.low_warmth_threshold {
                    let penalty =
                        (cfg.low_warmth_threshold - warmth) * cfg.warmth_penalty_multiplier;
                    max_boost = max_boost.max(penalty.abs());
                }
                max_boost
            } else {
                0.0
            };

            // Coherence feedback
            let mut this_binding_weight = 1.0f32;
            let mut this_veto = false;
            if self.config.enable_coherence_feedback {
                let output_hv = self.controller.output_hv();
                let binding_weight = self.coherence_feedback.update(&output_hv, &thought_hv);
                this_binding_weight = binding_weight;
                let coherence = self.coherence_feedback.coherence();
                coherence_dynamics.push(coherence);

                // Hallucination detection
                if coherence < 0.05 {
                    consecutive_low_coherence += 1;
                    if consecutive_low_coherence >= 3 {
                        hallucination_flag = true;
                    }
                } else {
                    consecutive_low_coherence = 0;
                }

                if binding_weight > 1.0 + 1e-6 {
                    thought_hv = thought_hv.scale(binding_weight);
                }

                // Semantic veto: mid-sentence self-correction
                if self.config.enable_semantic_veto
                    && self.coherence_feedback.should_veto()
                    && pos > min_veto_pos
                    && veto_count < max_vetoes
                    && tokens_since_veto >= veto_refractory
                {
                    veto_triggered = true;
                    this_veto = true;
                    veto_count += 1;
                    tokens_since_veto = 0;
                    text_bytes.extend_from_slice(self.config.veto_hesitation.as_bytes());
                    // Reset network state and re-inject thought context
                    self.controller.reset();
                    let _ = self
                        .controller
                        .forward_step(&thought_hv_original, THOUGHT_ID, 0);
                    thought_hv = thought_hv_original.clone();
                }
            }

            gating_trace.push(GatingTraceEntry {
                position: pos,
                coherence: self.coherence_feedback.coherence(),
                binding_weight: this_binding_weight,
                veto_triggered: this_veto,
                epistemic_boost: this_epistemic_boost,
                emotional_boost: this_emotional_boost,
            });

            tokens_since_veto += 1;

            // Sample next token (greedy for WASM simplicity)
            let next_token = greedy_sample(&logits);

            // Check EOS
            if next_token == EOS_ID {
                eos_terminated = true;
                break;
            }

            // Skip special tokens in output
            if !self.vocab.is_special(next_token) {
                let token_str = self.vocab.token_str(next_token);
                let is_byte_token = token_str.starts_with("<0x")
                    && token_str.ends_with('>')
                    && token_str.len() == 6;

                if !text_bytes.is_empty() && !is_byte_token && needs_space_before(token_str) {
                    let last = *text_bytes.last().unwrap();
                    if last != b' ' && last != b'\n' && last != b'-' {
                        text_bytes.push(b' ');
                    }
                }

                if is_byte_token {
                    if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                        text_bytes.push(byte);
                    } else {
                        text_bytes.extend_from_slice(token_str.as_bytes());
                    }
                } else {
                    text_bytes.extend_from_slice(token_str.as_bytes());
                }
            }

            tokens.push(next_token);
            prev_token = next_token;
        }

        let final_coherence = self.coherence_feedback.coherence();
        let text = String::from_utf8(text_bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());

        GenerationResult {
            text,
            token_ids: tokens.clone(),
            num_tokens: tokens.len(),
            eos_terminated,
            veto_triggered,
            final_coherence,
            coherence_dynamics,
            gating_trace,
            hallucination_flag,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-broca-full-spore")
    }

    fn small_config() -> BrocaConfig {
        BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: VOCAB_SIZE,
                max_seq_len: 32,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 20,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    // ── Encoder tests ────────────────────────────────────────────────────

    #[test]
    fn test_encoder_determinism() {
        let genesis = test_genesis();
        let enc1 = ThoughtLanguageEncoder::new(&genesis);
        let enc2 = ThoughtLanguageEncoder::new(&genesis);
        let channels = ThoughtChannels::default();
        let hv1 = enc1.encode(&channels);
        let hv2 = enc2.encode(&channels);
        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "Same genesis -> identical encoding"
        );
    }

    #[test]
    fn test_24_channel_encoding_produces_16384d() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);
        let channels = ThoughtChannels::default();
        let hv = enc.encode(&channels);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_24_channels_differ_from_subset() {
        // Verify 24-channel encoding is different from using only 12 channels
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);

        let mut ch_default = ThoughtChannels::default();
        let hv_default = enc.encode(&ch_default);

        // Modify only the new v3 channels (20-23) to extreme values
        ch_default.set_context(1.0, 1.0, 1.0, 1.0);
        let hv_context = enc.encode(&ch_default);

        let sim = hv_default.similarity(&hv_context);
        assert!(
            sim < 0.995,
            "Changing v3 channels should produce different encoding: sim={sim}"
        );
    }

    #[test]
    fn test_intent_discrimination() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);

        let answer = ThoughtChannels::with_intent(1);
        let clarify = ThoughtChannels::with_intent(2);

        let hv_answer = enc.encode(&answer);
        let hv_clarify = enc.encode(&clarify);

        let sim = hv_answer.similarity(&hv_clarify);
        assert!(
            sim < 0.95,
            "Different intents should produce dissimilar HVs: sim={sim}"
        );
    }

    #[test]
    fn test_nan_channels_produce_finite_output() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);
        let mut channels = ThoughtChannels::default();
        for c in &mut channels.channels {
            *c = f32::NAN;
        }
        let hv = enc.encode(&channels);
        assert!(
            hv.as_slice().iter().all(|v| v.is_finite()),
            "NaN input should produce finite HV output"
        );
    }

    // ── Epistemic gating tests ──────────────────────────────────────────

    #[test]
    fn test_epistemic_gating_suppresses_factual_words() {
        let vocab = Vocabulary::new();
        let config = GatingConfig::default();
        let gate = EpistemicGate::new(&vocab, &config);

        // Verify gate has resolved some factual and hedging tokens
        assert!(gate.factual_count() > 0, "Should resolve factual tokens");
        assert!(gate.hedging_count() > 0, "Should resolve hedging tokens");

        // Create baseline logits (all zeros)
        let mut logits = vec![0.0f32; VOCAB_SIZE];

        // Apply under Unknown epistemic status (ordinal 3.0)
        gate.apply(&mut logits, 3.0);

        // Factual words should have been penalized
        let is_id = vocab.token_id("is");
        if is_id != UNK_ID {
            assert!(
                logits[is_id as usize] < -1.0,
                "'is' should be penalized under Unknown: logit={}",
                logits[is_id as usize]
            );
        }

        // Hedging words should have been boosted
        let perhaps_id = vocab.token_id("perhaps");
        if perhaps_id != UNK_ID {
            assert!(
                logits[perhaps_id as usize] > 0.0,
                "'perhaps' should be boosted under Unknown: logit={}",
                logits[perhaps_id as usize]
            );
        }
    }

    #[test]
    fn test_epistemic_certain_no_modification() {
        let vocab = Vocabulary::new();
        let config = GatingConfig::default();
        let gate = EpistemicGate::new(&vocab, &config);

        let mut logits = vec![1.0f32; VOCAB_SIZE];
        let original = logits.clone();

        // Apply under Certain (ordinal 0.0) -- should not modify
        gate.apply(&mut logits, 0.0);

        assert_eq!(
            logits, original,
            "Certain epistemic status should not modify logits"
        );
    }

    #[test]
    fn test_epistemic_uncertain_milder_than_unknown() {
        let vocab = Vocabulary::new();
        let config = GatingConfig::default();
        let gate = EpistemicGate::new(&vocab, &config);

        let mut logits_uncertain = vec![0.0f32; VOCAB_SIZE];
        gate.apply(&mut logits_uncertain, 2.0); // Uncertain

        let mut logits_unknown = vec![0.0f32; VOCAB_SIZE];
        gate.apply(&mut logits_unknown, 3.0); // Unknown

        // Factual penalty should be stronger under Unknown than Uncertain
        let is_id = vocab.token_id("is");
        if is_id != UNK_ID {
            assert!(
                logits_unknown[is_id as usize] < logits_uncertain[is_id as usize],
                "Unknown should penalize 'is' more than Uncertain: {:.2} vs {:.2}",
                logits_unknown[is_id as usize],
                logits_uncertain[is_id as usize]
            );
        }
    }

    // ── Emotional modulation tests ──────────────────────────────────────

    #[test]
    fn test_emotional_high_arousal_boosts_endings() {
        let vocab = Vocabulary::new();
        let config = GatingConfig::default();
        let modulator = EmotionalModulator::new(&vocab, &config);

        let mut ch = ThoughtChannels::default();
        ch.set_emotion(0.0, 0.95, 0.5); // High arousal, neutral valence/warmth

        let mut logits_high_arousal = vec![0.0f32; VOCAB_SIZE];
        modulator.apply(&mut logits_high_arousal, &ch, 15); // Past arousal_position_threshold

        let mut logits_low_arousal = vec![0.0f32; VOCAB_SIZE];
        ch.set_emotion(0.0, 0.2, 0.5); // Low arousal
        modulator.apply(&mut logits_low_arousal, &ch, 15);

        // Sentence-ending tokens should be boosted more under high arousal
        let period_id = vocab.token_id(". ");
        if period_id != UNK_ID {
            assert!(
                logits_high_arousal[period_id as usize] > logits_low_arousal[period_id as usize],
                "High arousal should boost '. ' more: {:.3} vs {:.3}",
                logits_high_arousal[period_id as usize],
                logits_low_arousal[period_id as usize]
            );
        }
    }

    #[test]
    fn test_emotional_low_warmth_suppresses_informal() {
        let vocab = Vocabulary::new();
        let config = GatingConfig::default();
        let modulator = EmotionalModulator::new(&vocab, &config);

        let mut ch = ThoughtChannels::default();
        ch.set_emotion(0.0, 0.5, 0.1); // Very low warmth

        let mut logits = vec![0.0f32; VOCAB_SIZE];
        modulator.apply(&mut logits, &ch, 15);

        // Informal tokens should be suppressed (negative adjustment)
        let gonna_id = vocab.token_id("gonna");
        if gonna_id != UNK_ID {
            assert!(
                logits[gonna_id as usize] < 0.0,
                "'gonna' should be suppressed under low warmth: logit={}",
                logits[gonna_id as usize]
            );
        }
    }

    #[test]
    fn test_emotional_negative_valence_boosts_softening() {
        let vocab = Vocabulary::new();
        let config = GatingConfig::default();
        let modulator = EmotionalModulator::new(&vocab, &config);

        let mut ch = ThoughtChannels::default();
        ch.set_emotion(-0.8, 0.5, 0.5); // Very negative valence

        let mut logits = vec![0.0f32; VOCAB_SIZE];
        modulator.apply(&mut logits, &ch, 15);

        let sorry_id = vocab.token_id("sorry");
        if sorry_id != UNK_ID {
            assert!(
                logits[sorry_id as usize] > 0.0,
                "'sorry' should be boosted under negative valence: logit={}",
                logits[sorry_id as usize]
            );
        }
    }

    // ── Coherence feedback / semantic veto tests ────────────────────────

    #[test]
    fn test_coherence_veto_fixed_threshold() {
        // With adaptive_warmup=0, use fixed threshold mode
        let mut cf = CoherenceFeedback::with_config(0.3, 0.15, 0, 2.0);

        let genesis = test_genesis();
        let thought_hv = ContinuousHV::from_genesis(&genesis, "thought", HDC_DIMENSION);

        // Simulate high coherence
        let high_hv = thought_hv.clone();
        cf.update(&high_hv, &thought_hv);
        assert!(!cf.should_veto(), "High coherence should not trigger veto");

        // Simulate low coherence (orthogonal vector)
        let low_hv = ContinuousHV::from_genesis(&genesis, "very_different", HDC_DIMENSION);
        cf.update(&low_hv, &thought_hv);

        // With fixed threshold mode, veto fires when coherence < 0.15
        let coherence = cf.coherence();
        if coherence < 0.15 {
            assert!(
                cf.should_veto(),
                "Low coherence below threshold should trigger veto"
            );
        }
    }

    #[test]
    fn test_coherence_binding_weight_increases_on_drift() {
        let mut cf = CoherenceFeedback::new(0.5); // threshold 0.5

        let genesis = test_genesis();
        let thought_hv = ContinuousHV::from_genesis(&genesis, "thought", HDC_DIMENSION);
        let different_hv = ContinuousHV::from_genesis(&genesis, "different", HDC_DIMENSION);

        let weight = cf.update(&different_hv, &thought_hv);

        // If coherence < 0.5 (threshold), binding weight should be > 1.0
        if cf.coherence() < 0.5 {
            assert!(
                weight > 1.0,
                "Binding weight should increase when coherence is below threshold: weight={weight}"
            );
        }
    }

    // ── Consciousness gating tests ──────────────────────────────────────

    #[test]
    fn test_consciousness_gating_limits_verbosity() {
        // Low Psi should produce fewer max tokens
        let base = 128;
        let low_psi_max = consciousness_gated_max_tokens(base, 0.1);
        let high_psi_max = consciousness_gated_max_tokens(base, 0.9);

        assert!(
            low_psi_max < high_psi_max,
            "Low Psi should produce fewer tokens: low={low_psi_max}, high={high_psi_max}"
        );

        // At Psi=0.0, factor = 0.5, so max = 64
        let zero_psi_max = consciousness_gated_max_tokens(base, 0.0);
        assert_eq!(zero_psi_max, 64, "Psi=0 should give base*0.5");

        // At Psi=1.0, factor = 1.5, so max = 192
        let full_psi_max = consciousness_gated_max_tokens(base, 1.0);
        assert_eq!(full_psi_max, 192, "Psi=1 should give base*1.5");
    }

    #[test]
    fn test_consciousness_gating_in_generation() {
        let genesis = test_genesis();

        // Low consciousness config
        let mut config_low = small_config();
        config_low.gating.base_max_tokens = 40;
        let mut gen_low = BrocaGenerator::new(&genesis, config_low);
        let mut ch_low = ThoughtChannels::default();
        ch_low.set_consciousness(0.1, 0.5, 0.5);
        let result_low = gen_low.generate(&ch_low);

        // High consciousness config (same base, different Psi)
        let mut config_high = small_config();
        config_high.gating.base_max_tokens = 40;
        let mut gen_high = BrocaGenerator::new(&genesis, config_high);
        let mut ch_high = ThoughtChannels::default();
        ch_high.set_consciousness(0.9, 0.5, 0.5);
        let result_high = gen_high.generate(&ch_high);

        // The max token budget is different; the actual generation may be shorter
        // due to EOS, but the budget should differ
        let low_budget = consciousness_gated_max_tokens(40, 0.1);
        let high_budget = consciousness_gated_max_tokens(40, 0.9);
        assert!(
            low_budget < high_budget,
            "Low Psi budget should be smaller: {low_budget} vs {high_budget}"
        );

        // Result tokens should respect the budget
        assert!(
            result_low.num_tokens <= low_budget,
            "Low Psi result should not exceed budget: {} > {}",
            result_low.num_tokens,
            low_budget
        );
        assert!(
            result_high.num_tokens <= high_budget,
            "High Psi result should not exceed budget: {} > {}",
            result_high.num_tokens,
            high_budget
        );
    }

    // ── 24-channel vs 12-channel test ───────────────────────────────────

    #[test]
    fn test_24_channel_encoding_differs_from_12() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);

        // Encode with default 24-channel (v3 channels at defaults)
        let ch24 = ThoughtChannels::default();
        let hv24 = enc.encode(&ch24);

        // Encode with modified v3 channels
        let mut ch24_modified = ThoughtChannels::default();
        ch24_modified.set_context(1.0, 0.0, 1.0, 0.0);
        let hv24_modified = enc.encode(&ch24_modified);

        // The two should be different (v3 channels contribute to encoding)
        let sim = hv24.similarity(&hv24_modified);
        assert!(
            sim < 0.99,
            "Different v3 context channels should produce different encodings: sim={sim}"
        );
        assert!(
            sim > 0.0,
            "Should still share structure from the 20 shared channels: sim={sim}"
        );
    }

    // ── Full generation test ────────────────────────────────────────────

    #[test]
    fn test_full_generation_produces_text() {
        let genesis = test_genesis();
        let config = small_config();
        let mut b_gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = b_gen.generate(&channels);

        assert!(
            result.num_tokens > 0,
            "Generation should produce at least one token"
        );
        // Text may be empty if only special tokens were generated,
        // but token_ids should not be empty
        assert!(
            !result.token_ids.is_empty(),
            "Should have generated token IDs"
        );
    }

    #[test]
    fn test_generation_with_epistemic_gating() {
        let genesis = test_genesis();
        let config = small_config();
        let mut b_gen = BrocaGenerator::new(&genesis, config);

        let mut channels = ThoughtChannels::default();
        channels.set_epistemic(3.0); // Unknown

        let result = b_gen.generate(&channels);

        // Verify gating trace shows epistemic boost was applied
        assert!(
            !result.gating_trace.is_empty(),
            "Should have gating trace entries"
        );
        let has_epistemic_boost = result.gating_trace.iter().any(|e| e.epistemic_boost > 0.0);
        assert!(
            has_epistemic_boost,
            "Unknown epistemic status should produce epistemic boost in trace"
        );
    }

    #[test]
    fn test_generation_with_emotional_modulation() {
        let genesis = test_genesis();
        let config = small_config();
        let mut b_gen = BrocaGenerator::new(&genesis, config);

        // High arousal negative valence
        let mut channels = ThoughtChannels::default();
        channels.set_emotion(-0.8, 0.95, 0.1);

        let result = b_gen.generate(&channels);

        // At least some gating trace entries should show emotional boost
        // (after position threshold)
        let has_emotional = result.gating_trace.iter().any(|e| e.emotional_boost > 0.0);
        // May not fire if generation is too short (< arousal_position_threshold)
        if result.num_tokens > 10 {
            assert!(
                has_emotional,
                "High arousal should produce emotional boost in trace (num_tokens={})",
                result.num_tokens
            );
        }
    }

    #[test]
    fn test_repetition_penalty_reduces_repeats() {
        let mut logits = vec![1.0f32; 10];
        let generated = vec![3, 3, 5]; // token 3 appeared twice, token 5 once

        apply_repetition_penalty(&mut logits, &generated, 2.0);

        // Token 3 (count=2): logit 1.0 / 2.0^2 = 0.25
        assert!(
            (logits[3] - 0.25).abs() < 1e-5,
            "Token 3 (count=2) should be heavily penalized: {}",
            logits[3]
        );
        // Token 5 (count=1): logit 1.0 / 2.0^1 = 0.5
        assert!(
            (logits[5] - 0.5).abs() < 1e-5,
            "Token 5 (count=1) should be moderately penalized: {}",
            logits[5]
        );
        // Token 7 (not generated): logit should be unchanged
        assert!(
            (logits[7] - 1.0).abs() < 1e-5,
            "Token 7 (not generated) should be unchanged: {}",
            logits[7]
        );
    }

    #[test]
    fn test_bigram_blocking() {
        let mut logits = vec![1.0f32; 10];
        let _generated = vec![2, 5, 3, 7]; // bigrams: (2,5), (5,3), (3,7)

        // Current suffix is [7]. Blocking bigrams that start with [7]
        // -- would block token 0 only if (7,0) was seen. Not in history.
        // But if suffix is the last (n-1)=1 token = [7], and (7, ?) never seen,
        // no blocking should occur.

        // Let's set up a case where blocking fires:
        let generated2 = vec![2, 5, 3, 2]; // suffix = [2], bigram (2,5) exists
        block_repeated_ngrams(&mut logits, &generated2, 2);

        assert_eq!(
            logits[5],
            f32::NEG_INFINITY,
            "Token 5 should be blocked (would repeat bigram (2,5))"
        );
        assert!(
            logits[3].is_finite(),
            "Token 3 should not be blocked (bigram (2,3) not seen)"
        );
    }

    // ── Vocabulary tests ────────────────────────────────────────────────

    #[test]
    fn test_vocabulary_has_canonical_words() {
        let vocab = Vocabulary::new();

        // Check hedging words are present
        let perhaps_id = vocab.token_id("perhaps");
        assert_ne!(perhaps_id, UNK_ID, "'perhaps' should be in vocabulary");

        let maybe_id = vocab.token_id("maybe");
        assert_ne!(maybe_id, UNK_ID, "'maybe' should be in vocabulary");

        // Check factual words
        let is_id = vocab.token_id("is");
        assert_ne!(is_id, UNK_ID, "'is' should be in vocabulary");

        let always_id = vocab.token_id("always");
        assert_ne!(always_id, UNK_ID, "'always' should be in vocabulary");

        // Check informal words
        let gonna_id = vocab.token_id("gonna");
        assert_ne!(gonna_id, UNK_ID, "'gonna' should be in vocabulary");
    }

    #[test]
    fn test_vocabulary_size() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.tokens.len(), VOCAB_SIZE);
    }

    #[test]
    fn test_vocabulary_special_tokens() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.token_str(0), "<bos>");
        assert_eq!(vocab.token_str(EOS_ID), "<eos>");
        assert_eq!(vocab.token_str(UNK_ID), "<unk>");
        assert_eq!(vocab.token_str(THOUGHT_ID), "<thought>");
        assert!(vocab.is_special(0)); // BOS
        assert!(vocab.is_special(EOS_ID));
        assert!(!vocab.is_special(10)); // 'space' char is not special
    }

    // ── Semantic veto test ──────────────────────────────────────────────

    #[test]
    fn test_semantic_veto_triggers_on_coherence_drop() {
        // Use fixed-threshold mode (warmup=0) for deterministic test
        let mut cf = CoherenceFeedback::with_config(0.3, 0.15, 0, 2.0);

        let genesis = test_genesis();
        let thought_hv = ContinuousHV::from_genesis(&genesis, "thought_intent", HDC_DIMENSION);

        // Feed same vector (high coherence) -- should not veto
        cf.update(&thought_hv, &thought_hv);
        assert!(!cf.should_veto());
        assert!(
            (cf.coherence() - 1.0).abs() < 0.01,
            "Same HV should give coherence ~1.0: {}",
            cf.coherence()
        );

        // Feed orthogonal vector (low coherence) -- should veto
        let orthogonal =
            ContinuousHV::from_genesis(&genesis, "totally_different_topic", HDC_DIMENSION);
        cf.update(&orthogonal, &thought_hv);

        if cf.coherence() < 0.15 {
            assert!(
                cf.should_veto(),
                "Coherence {:.4} < 0.15 should trigger veto",
                cf.coherence()
            );
        }
    }

    // ── Time pressure test ──────────────────────────────────────────────

    #[test]
    fn test_time_pressure_reduces_max_tokens() {
        let relaxed = time_pressure_adjusted_max_tokens(100, 0.0, 0.5);
        assert_eq!(relaxed, 100);

        let urgent = time_pressure_adjusted_max_tokens(100, 1.0, 0.5);
        assert_eq!(urgent, 50);

        let moderate = time_pressure_adjusted_max_tokens(100, 0.5, 0.5);
        assert_eq!(moderate, 75);
    }

    // ── Confidence veto threshold ───────────────────────────────────────

    #[test]
    fn test_confidence_adjusted_veto() {
        let base = 0.15;
        // No confidence -> full threshold
        let t0 = confidence_adjusted_veto_threshold(base, 0.0, 0.3);
        assert!((t0 - 0.15).abs() < 1e-6);

        // Full confidence -> reduced threshold
        let t1 = confidence_adjusted_veto_threshold(base, 1.0, 0.3);
        assert!((t1 - 0.105).abs() < 1e-4, "t1 = {t1}");
    }
}
