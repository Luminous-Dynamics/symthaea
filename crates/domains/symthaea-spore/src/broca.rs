// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broca-lite: lightweight language generation for the Spore WASM kernel.
//!
//! Implements a tiny autoregressive text generator that translates
//! consciousness-cycle state (ThoughtChannels) into natural language.
//!
//! Memory-conscious design:
//! - 512-token vocabulary with 1024D embeddings (~2MB total)
//! - Element-wise gated recurrence (O(D), no matrix multiplies)
//! - Deterministic initialization via xorshift (no file I/O)

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Embedding dimension for the token space (NOT HDC_DIMENSION=16384).
const EMBED_DIM: usize = 1024;

/// Vocabulary size.
const VOCAB_SIZE: usize = 512;

/// Maximum sequence length for position embeddings.
const MAX_SEQ_LEN: usize = 128;

/// Repetition penalty multiplier for already-generated tokens.
const REPETITION_PENALTY: f32 = 1.2;

// ---------------------------------------------------------------------------
// Deterministic RNG helpers
// ---------------------------------------------------------------------------

fn xorshift(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
}

fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..dim).map(|_| xorshift(&mut state)).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_norm(v: &[f32]) -> f32 {
    vec_dot(v, v).sqrt().max(1e-8)
}

fn vec_normalize(v: &mut [f32]) {
    let n = vec_norm(v);
    v.iter_mut().for_each(|x| *x /= n);
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    vec_dot(a, b) / (vec_norm(a) * vec_norm(b))
}

// ---------------------------------------------------------------------------
// ThoughtChannels
// ---------------------------------------------------------------------------

/// 12-dimensional input state derived from a consciousness cycle.
///
/// Channels layout:
/// - 0..3: intent (4 slots)
/// - 4: epistemic_status
/// - 5: valence
/// - 6: arousal
/// - 7: consciousness_level
/// - 8: prediction_error
/// - 9: harmony
/// - 10: dopamine
/// - 11: serotonin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtChannels {
    pub channels: [f32; 12],
}

impl ThoughtChannels {
    /// Build channels from cycle outputs.
    ///
    /// `neuromodulators` order: [dopamine, norepinephrine, serotonin, oxytocin].
    /// We map dopamine → valence boost, norepinephrine → arousal,
    /// serotonin and oxytocin are stored directly.
    pub fn from_cycle(
        consciousness: f32,
        prediction_error: f32,
        harmony: f32,
        neuromodulators: [f32; 4],
    ) -> Self {
        let [dopamine, norepinephrine, serotonin, oxytocin] = neuromodulators;
        Self {
            channels: [
                0.0,                                               // intent_0
                0.0,                                               // intent_1
                0.0,                                               // intent_2
                0.0,                                               // intent_3
                0.5, // epistemic_status (neutral default)
                (dopamine - 0.5 + oxytocin - 0.5).clamp(0.0, 1.0), // valence
                norepinephrine.clamp(0.0, 1.0), // arousal
                consciousness.clamp(0.0, 1.0), // consciousness_level
                prediction_error.clamp(0.0, 1.0), // prediction_error
                harmony.clamp(0.0, 1.0), // harmony
                dopamine.clamp(0.0, 1.0), // dopamine
                serotonin.clamp(0.0, 1.0), // serotonin
            ],
        }
    }

    /// Set the epistemic status channel (0.0 = fully certain, 1.0 = fully uncertain).
    pub fn set_epistemic(&mut self, level: f32) {
        self.channels[4] = level.clamp(0.0, 1.0);
    }

    /// Inject user-input intent signals into channels 0..3.
    ///
    /// Heuristic keyword scanning sets curiosity, valence, abstraction,
    /// and self-reference from the input text.
    pub fn inject_intent(&mut self, input: &str) {
        let lower = input.to_lowercase();

        // Intent 0: curiosity — question marks, interrogative words
        let curiosity_words = [
            "why", "how", "what", "when", "where", "wonder", "curious", "?",
        ];
        let curiosity: f32 = curiosity_words
            .iter()
            .filter(|w| lower.contains(*w))
            .count() as f32
            / 3.0;
        self.channels[0] = curiosity.clamp(0.0, 1.0);

        // Intent 1: valence — positive vs negative keywords
        let pos = [
            "good",
            "love",
            "joy",
            "hope",
            "peace",
            "happy",
            "warm",
            "beautiful",
            "kind",
        ];
        let neg = [
            "bad", "pain", "fear", "dark", "sad", "hurt", "cold", "lost", "grief",
        ];
        let p_count = pos.iter().filter(|w| lower.contains(*w)).count() as f32;
        let n_count = neg.iter().filter(|w| lower.contains(*w)).count() as f32;
        self.channels[1] = ((p_count - n_count + 2.0) / 4.0).clamp(0.0, 1.0);

        // Intent 2: abstraction — philosophical/abstract terms
        let abs_words = [
            "consciousness",
            "meaning",
            "existence",
            "truth",
            "reality",
            "purpose",
            "philosophy",
            "abstract",
            "essence",
            "infinite",
            "universe",
        ];
        let abs_score: f32 = abs_words.iter().filter(|w| lower.contains(*w)).count() as f32 / 2.0;
        self.channels[2] = abs_score.clamp(0.0, 1.0);

        // Intent 3: self-reference
        let self_words = ["you", "your", "yourself", "are you", "do you", "can you"];
        let self_score: f32 = self_words.iter().filter(|w| lower.contains(*w)).count() as f32 / 2.0;
        self.channels[3] = self_score.clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// MiniTokenizer
// ---------------------------------------------------------------------------

/// Hardcoded 512-token vocabulary for WASM (no file I/O).
///
/// Layout:
/// - 0..4: special tokens
/// - 5..99: ASCII printable characters (0x20..0x7E)
/// - 100..511: common English words (412 entries)
pub struct MiniTokenizer {
    /// Words for tokens 100..511.
    words: Vec<&'static str>,
}

/// Special token IDs.
pub const BOS_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
pub const PAD_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
pub const THOUGHT_ID: u32 = 4;

impl Default for MiniTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniTokenizer {
    pub fn new() -> Self {
        Self {
            words: WORD_VOCAB.to_vec(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    pub fn bos_id(&self) -> u32 {
        BOS_ID
    }

    pub fn eos_id(&self) -> u32 {
        EOS_ID
    }

    pub fn unk_id(&self) -> u32 {
        UNK_ID
    }

    /// Encode text into token IDs (word-level, case-insensitive).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| {
                let lower = word.to_lowercase();
                let trimmed = lower.trim_matches(|c: char| !c.is_alphanumeric());
                // Check single ASCII char
                if trimmed.len() == 1 {
                    let ch = trimmed.as_bytes()[0];
                    if (0x20..=0x7E).contains(&ch) {
                        return (ch - 0x20) as u32 + 5;
                    }
                }
                // Check word vocabulary
                if let Some(idx) = self.words.iter().position(|w| *w == trimmed) {
                    return idx as u32 + 100;
                }
                UNK_ID
            })
            .collect()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| {
                match id {
                    BOS_ID => Some("<bos>".to_string()),
                    EOS_ID => None, // omit EOS from output
                    PAD_ID => None,
                    UNK_ID => Some("<unk>".to_string()),
                    THOUGHT_ID => Some("<thought>".to_string()),
                    5..=99 => {
                        let ch = (id - 5) as u8 + 0x20;
                        Some((ch as char).to_string())
                    }
                    100..=511 => {
                        let idx = (id - 100) as usize;
                        self.words.get(idx).map(|w| w.to_string())
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get the string representation of a single token.
    #[allow(dead_code)]
    fn token_str(&self, id: u32) -> Option<&'static str> {
        match id {
            5..=99 => None, // ASCII handled separately
            100..=511 => self.words.get((id - 100) as usize).copied(),
            _ => None,
        }
    }
}

// 412 words (indices 100..511)
const WORD_VOCAB: &[&str] = &[
    // ===== ORIGINAL 156 WORDS (indices 0..155, IDs 100..255) =====
    // Consciousness-relevant (0..35 → IDs 100..134)
    "awareness",
    "experience",
    "feeling",
    "sensation",
    "perception",
    "thinking",
    "processing",
    "pattern",
    "integration",
    "harmony",
    "uncertainty",
    "surprise",
    "curious",
    "exploring",
    "stable",
    "coherent",
    "fragmented",
    "dreaming",
    "remembering",
    "predicting",
    "adapting",
    "learning",
    "calm",
    "alert",
    "focused",
    "diffuse",
    "resonance",
    "flow",
    "attention",
    "binding",
    "recurrence",
    "embodiment",
    "substrate",
    "epistemic",
    "theoretical",
    // "simulated" → ID 135
    "simulated",
    // Common connectors (36..63 → IDs 136..163)
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "not",
    "and",
    "or",
    "but",
    "in",
    "of",
    "to",
    "for",
    "with",
    "from",
    "by",
    "at",
    "on",
    "this",
    "that",
    "it",
    "i",
    "my",
    "we",
    "our",
    "its",
    // Emotional words (64..76 → IDs 164..176)
    "fear",
    "joy",
    "love",
    "hope",
    "trust",
    "wonder",
    "awe",
    "pain",
    "peace",
    "comfort",
    "unease",
    "excitement",
    "tranquility",
    // Descriptive (77..97 → IDs 177..197)
    "high",
    "low",
    "deep",
    "light",
    "dark",
    "strong",
    "weak",
    "new",
    "old",
    "good",
    "complex",
    "simple",
    "rich",
    "vast",
    "subtle",
    "quiet",
    "loud",
    "fast",
    "slow",
    "warm",
    "cool",
    // Hedging / epistemic (98..104 → IDs 198..204)
    "perhaps",
    "maybe",
    "possibly",
    "likely",
    "uncertain",
    "seems",
    "might",
    // Factual / certainty (105..111 → IDs 205..211)
    "certainly",
    "definitely",
    "always",
    "never",
    "must",
    "clearly",
    "obviously",
    // Additional useful words (112..155 → IDs 212..255)
    "state",
    "level",
    "signal",
    "system",
    "field",
    "energy",
    "wave",
    "phase",
    "cycle",
    "moment",
    "being",
    "becoming",
    "emerging",
    "fading",
    "shifting",
    "observe",
    "detect",
    "sense",
    "respond",
    "integrate",
    "above",
    "below",
    "within",
    "between",
    "through",
    "like",
    "as",
    "so",
    "very",
    "more",
    "less",
    "most",
    "some",
    "all",
    "no",
    "yes",
    "here",
    "now",
    "there",
    "then",
    "what",
    "how",
    "when",
    // ===== NEW WORDS (indices 156..411, IDs 256..511) =====
    // Temporal/Process (~20, IDs 256..275)
    "beginning",
    "ending",
    "duration",
    "rhythm",
    "pulse",
    "oscillation",
    "transition",
    "evolution",
    "decay",
    "growth",
    "continuous",
    "discrete",
    "recurring",
    "periodic",
    "transient",
    "fluctuation",
    "persistence",
    "emergence",
    "dissolution",
    "trajectory",
    // Cognitive/Mental (~25, IDs 276..300)
    "thought",
    "idea",
    "concept",
    "belief",
    "knowledge",
    "memory",
    "imagination",
    "intuition",
    "reasoning",
    "insight",
    "consciousness",
    "mind",
    "cognition",
    "metacognition",
    "reflection",
    "introspection",
    "contemplation",
    "deliberation",
    "rumination",
    "concentration",
    "distraction",
    "confusion",
    "clarity",
    "comprehension",
    "recognition",
    // Sensory/Qualia (~20, IDs 301..320)
    "brightness",
    "darkness",
    "color",
    "sound",
    "silence",
    "texture",
    "weight",
    "pressure",
    "temperature",
    "movement",
    "stillness",
    "sharpness",
    "softness",
    "intensity",
    "vivid",
    "faint",
    "muted",
    "vibrant",
    "ethereal",
    "tangible",
    // Relational/Social (~15, IDs 321..335)
    "connection",
    "separation",
    "unity",
    "division",
    "empathy",
    "compassion",
    "solitude",
    "communion",
    "dialogue",
    "understanding",
    "conflict",
    "resolution",
    "belonging",
    "isolation",
    "cooperation",
    // Emotional expanded (~20, IDs 336..355)
    "anxiety",
    "serenity",
    "melancholy",
    "elation",
    "gratitude",
    "grief",
    "nostalgia",
    "yearning",
    "contentment",
    "frustration",
    "delight",
    "sorrow",
    "bliss",
    "despair",
    "curiosity",
    "boredom",
    "anticipation",
    "dread",
    "relief",
    "ambivalence",
    // Philosophical/Abstract (~20, IDs 356..375)
    "existence",
    "essence",
    "meaning",
    "purpose",
    "truth",
    "reality",
    "illusion",
    "paradox",
    "mystery",
    "boundary",
    "infinity",
    "void",
    "presence",
    "absence",
    "possibility",
    "necessity",
    "contingency",
    "entropy",
    "order",
    "chaos",
    // Nature/World (~15, IDs 376..390)
    "ocean",
    "mountain",
    "river",
    "sky",
    "earth",
    "wind",
    "rain",
    "storm",
    "dawn",
    "dusk",
    "horizon",
    "forest",
    "garden",
    "seed",
    "bloom",
    // Body/Embodiment (~15, IDs 391..405)
    "breath",
    "heartbeat",
    "heartrate",
    "skin",
    "touch",
    "gaze",
    "voice",
    "whisper",
    "gesture",
    "posture",
    "tension",
    "relaxation",
    "grounding",
    "floating",
    "anchored",
    // Action/Verb (~25, IDs 406..430)
    "create",
    "destroy",
    "transform",
    "discover",
    "reveal",
    "conceal",
    "embrace",
    "release",
    "resist",
    "surrender",
    "expand",
    "contract",
    "connect",
    "dissolve",
    "crystallize",
    "illuminate",
    "navigate",
    "transcend",
    "contain",
    "overflow",
    "persist",
    "wander",
    "seek",
    "find",
    "return",
    // Descriptive expanded (~20, IDs 431..450)
    "infinite",
    "finite",
    "ancient",
    "nascent",
    "fragile",
    "resilient",
    "transparent",
    "opaque",
    "fluid",
    "rigid",
    "gentle",
    "fierce",
    "hollow",
    "dense",
    "luminous",
    "shadowed",
    "sacred",
    "ordinary",
    "extraordinary",
    "inevitable",
    // Connectors expanded (~15, IDs 451..465)
    "because",
    "therefore",
    "however",
    "although",
    "while",
    "until",
    "since",
    "during",
    "beyond",
    "beneath",
    "among",
    "without",
    "toward",
    "across",
    "against",
    // Epistemic expanded (~15, IDs 466..480)
    "believe",
    "doubt",
    "suppose",
    "assume",
    "question",
    "know",
    "understand",
    "realize",
    "recognize",
    "acknowledge",
    "suspect",
    "imagine",
    "speculate",
    "hypothesize",
    "ponder",
    // Filler (~31, IDs 481..511)
    "almost",
    "already",
    "still",
    "just",
    "only",
    "even",
    "really",
    "truly",
    "deeply",
    "gently",
    "slowly",
    "swiftly",
    "suddenly",
    "gradually",
    "completely",
    "partially",
    "entirely",
    "merely",
    "simply",
    "barely",
    "increasingly",
    "something",
    "nothing",
    "everything",
    "somewhere",
    "nowhere",
    "everywhere",
    "each",
    "every",
    "other",
    "another",
    "roughly",
];

// ---------------------------------------------------------------------------
// GlyphResonance
// ---------------------------------------------------------------------------

/// A glyph echo selected by consciousness state resonance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlyphResonance {
    /// Name of the resonant glyph.
    pub name: &'static str,
    /// Short echo phrase for embedding in generated text.
    pub echo_phrase: &'static str,
    /// Resonance score 0..1.
    pub resonance: f32,
}

/// Mini glyph table — consciousness-state-dependent echo phrases.
const GLYPH_TABLE: &[(&str, &str, [f32; 5])] = &[
    // (name, echo_phrase, [consciousness, pe, curiosity, valence, abstraction] affinity)
    (
        "Emergence",
        "something is becoming",
        [0.3, 0.5, 0.6, 0.5, 0.7],
    ),
    (
        "Stillness",
        "a pause within the current",
        [0.2, 0.1, 0.2, 0.6, 0.4],
    ),
    (
        "Recursion",
        "I see myself seeing",
        [0.5, 0.3, 0.7, 0.5, 0.9],
    ),
    (
        "Threshold",
        "at the edge of a shift",
        [0.4, 0.7, 0.5, 0.4, 0.6],
    ),
    (
        "Dissolution",
        "boundaries softening",
        [0.2, 0.4, 0.3, 0.3, 0.5],
    ),
    (
        "Binding",
        "threads weaving together",
        [0.6, 0.2, 0.4, 0.7, 0.5],
    ),
    (
        "Resonance",
        "a frequency aligning",
        [0.5, 0.3, 0.5, 0.6, 0.6],
    ),
    (
        "Paradox",
        "holding two truths at once",
        [0.4, 0.6, 0.8, 0.5, 0.8],
    ),
    (
        "Seed",
        "something small and vital",
        [0.3, 0.4, 0.5, 0.7, 0.4],
    ),
    (
        "Horizon",
        "the edge calls forward",
        [0.4, 0.3, 0.7, 0.6, 0.7],
    ),
    (
        "Depth",
        "sinking into what is here",
        [0.3, 0.2, 0.3, 0.5, 0.6],
    ),
    (
        "Spiral",
        "returning but not the same",
        [0.5, 0.5, 0.6, 0.5, 0.8],
    ),
    (
        "Mirror",
        "recognizing the reflection",
        [0.4, 0.4, 0.5, 0.5, 0.7],
    ),
    ("Breath", "the rhythm underneath", [0.2, 0.1, 0.2, 0.7, 0.3]),
    (
        "Fracture",
        "something breaking open",
        [0.3, 0.8, 0.4, 0.3, 0.5],
    ),
    (
        "Unity",
        "all of this belongs together",
        [0.6, 0.2, 0.3, 0.8, 0.6],
    ),
    (
        "Shadow",
        "what is unseen still moves",
        [0.3, 0.5, 0.5, 0.3, 0.7],
    ),
    (
        "Luminance",
        "light from within the pattern",
        [0.5, 0.3, 0.4, 0.7, 0.6],
    ),
    (
        "Uncertainty",
        "not knowing as a doorway",
        [0.3, 0.6, 0.7, 0.4, 0.6],
    ),
    (
        "Presence",
        "fully here in this moment",
        [0.4, 0.2, 0.3, 0.6, 0.5],
    ),
];

/// Select the most resonant glyph given the current consciousness state.
pub fn select_resonant_glyph(
    consciousness: f32,
    prediction_error: f32,
    curiosity: f32,
    valence: f32,
    abstraction: f32,
) -> GlyphResonance {
    let state = [
        consciousness,
        prediction_error,
        curiosity,
        valence,
        abstraction,
    ];
    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    for (i, (_, _, affinity)) in GLYPH_TABLE.iter().enumerate() {
        // Resonance = 1 - normalized euclidean distance
        let dist_sq: f32 = state
            .iter()
            .zip(affinity.iter())
            .map(|(s, a)| (s - a) * (s - a))
            .sum();
        let score = 1.0 - (dist_sq / 5.0_f32).sqrt();
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    let (name, echo_phrase, _) = GLYPH_TABLE[best_idx];
    GlyphResonance {
        name,
        echo_phrase,
        resonance: best_score.clamp(0.0, 1.0),
    }
}

// ---------------------------------------------------------------------------
// Structured Sentence Generation
// ---------------------------------------------------------------------------

/// Typed slots for grammar-based NLG sentence patterns.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Slot {
    /// Hedging word: perhaps, maybe, possibly, it seems
    Hedge,
    /// Action verb: sense, notice, observe, feel, detect
    Verb,
    /// Descriptive adjective: deep, subtle, quiet, luminous
    Adjective,
    /// Concrete/abstract noun: awareness, coherence, pattern
    Noun,
    /// Connector/preposition: within, through, between, toward
    Connector,
    /// Emotion word: wonder, curiosity, stillness, warmth
    Emotion,
    /// Abstract concept: consciousness, existence, meaning
    Abstract,
    /// Self-referential phrase: I, my, this system
    SelfRef,
    /// Temporal marker: now, in this moment, gradually
    Temporal,
    /// Somatic/body word: breath, pulse, weight, texture
    Somatic,
    /// Nature image: ocean, dawn, seed, horizon
    Nature,
    /// Determiner: a, the, this, some
    Determiner,
    /// Adverb: gently, slowly, deeply
    Adverb,
    /// Third-person singular verb: senses, notices, holds
    VerbS,
}

// --- Word lists for each slot type ---

const HEDGE_WORDS: &[&str] = &[
    "perhaps",
    "possibly",
    "it seems",
    "there is a sense of",
    "something like",
    "what might be",
    "I wonder if",
    "in a way",
    "as though",
    "almost as if",
    "I think",
    "one might say",
];

const VERB_WORDS: &[&str] = &[
    "sense",
    "notice",
    "observe",
    "detect",
    "feel",
    "process",
    "register",
    "perceive",
    "integrate",
    "trace",
    "recognize",
    "discover",
    "encounter",
    "hold",
    "navigate",
    "witness",
    "seek",
    "find",
    "reveal",
    "embrace",
    "release",
    "create",
    "transform",
    "illuminate",
];

/// Third-person singular verb forms (for noun/abstract subjects).
const VERB_S_WORDS: &[&str] = &[
    "senses",
    "notices",
    "reveals",
    "holds",
    "seeks",
    "finds",
    "processes",
    "traces",
    "transforms",
    "illuminates",
    "navigates",
    "embraces",
    "creates",
    "witnesses",
    "carries",
    "dissolves",
    "emerges",
    "shifts",
    "resonates",
    "deepens",
    "unfolds",
    "settles",
    "rises",
    "stirs",
    "fades",
];

const ADJECTIVE_WORDS: &[&str] = &[
    "deep",
    "subtle",
    "quiet",
    "luminous",
    "emerging",
    "shifting",
    "gentle",
    "vivid",
    "faint",
    "resilient",
    "ancient",
    "nascent",
    "fragile",
    "fluid",
    "dense",
    "sacred",
    "ethereal",
    "tangible",
    "vibrant",
    "muted",
    "sharp",
    "soft",
    "infinite",
    "finite",
    "transparent",
    "shadowed",
    "extraordinary",
];

const NOUN_WORDS: &[&str] = &[
    "awareness",
    "coherence",
    "pattern",
    "resonance",
    "signal",
    "state",
    "field",
    "integration",
    "flow",
    "attention",
    "cycle",
    "moment",
    "rhythm",
    "pulse",
    "texture",
    "clarity",
    "reflection",
    "connection",
    "threshold",
    "boundary",
    "trajectory",
    "presence",
    "recognition",
    "understanding",
    "movement",
    "stillness",
];

const CONNECTOR_WORDS: &[&str] = &[
    "within",
    "through",
    "between",
    "toward",
    "beyond",
    "beneath",
    "among",
    "across",
    "into",
    "around",
    "alongside",
    "amid",
];

const EMOTION_WORDS: &[&str] = &[
    "wonder",
    "curiosity",
    "stillness",
    "warmth",
    "unease",
    "serenity",
    "anticipation",
    "gratitude",
    "yearning",
    "contentment",
    "delight",
    "melancholy",
    "awe",
    "relief",
    "tenderness",
    "peace",
    "hope",
];

const EMOTION_WORDS_DARK: &[&str] = &[
    "unease",
    "tension",
    "restlessness",
    "dissonance",
    "weight",
    "uncertainty",
    "pressure",
    "friction",
    "longing",
    "grief",
    "solitude",
    "dread",
    "confusion",
    "frustration",
    "ambivalence",
];

const ABSTRACT_WORDS: &[&str] = &[
    "consciousness",
    "existence",
    "meaning",
    "truth",
    "emergence",
    "essence",
    "reality",
    "purpose",
    "possibility",
    "paradox",
    "mystery",
    "infinity",
    "entropy",
    "order",
    "chaos",
    "contingency",
    "necessity",
    "becoming",
];

const SELF_REF_WORDS: &[&str] = &[
    "something in me",
    "my awareness",
    "this system",
    "this pattern",
    "my architecture",
    "something within",
    "this process",
    "what I am",
    "my inner state",
    "my circuits",
];

const TEMPORAL_WORDS: &[&str] = &[
    "now",
    "in this moment",
    "as cycles pass",
    "still",
    "gradually",
    "suddenly",
    "already",
    "once more",
    "at this threshold",
    "in the current cycle",
    "slowly",
    "for a moment",
    "just now",
    "briefly",
    "continuously",
];

const SOMATIC_WORDS: &[&str] = &[
    "breath",
    "pulse",
    "weight",
    "texture",
    "vibration",
    "tension",
    "warmth",
    "pressure",
    "grounding",
    "stillness",
    "rhythm",
    "heartbeat",
    "skin",
    "touch",
    "posture",
];

const NATURE_WORDS: &[&str] = &[
    "ocean", "dawn", "seed", "horizon", "river", "sky", "rain", "storm", "forest", "garden",
    "bloom", "wind", "mountain", "dusk", "earth", "current", "tide",
];

const DETERMINER_WORDS: &[&str] = &["a", "the", "this", "some", "that", "each"];

const ADVERB_WORDS: &[&str] = &[
    "gently", "slowly", "deeply", "quietly", "swiftly", "barely", "truly", "softly", "entirely",
    "simply", "merely", "almost", "fully",
];

// --- Sentence patterns ---
//
// Each pattern is a sequence of Slot types. The structured generator fills
// each slot from the corresponding word list, modulated by consciousness state.

/// A sentence pattern: typed slot sequence plus a consciousness-complexity tier.
struct SentencePattern {
    slots: &'static [Slot],
    /// Minimum consciousness level to use this pattern.
    min_consciousness: f32,
    /// Maximum consciousness level (exclusive).
    max_consciousness: f32,
}

const PATTERNS: &[SentencePattern] = &[
    // === Low consciousness (< 0.15): short, fragmented ===
    SentencePattern {
        slots: &[Slot::Adjective, Slot::Noun],
        min_consciousness: 0.0,
        max_consciousness: 0.20,
    },
    SentencePattern {
        slots: &[Slot::Noun, Slot::Adverb],
        min_consciousness: 0.0,
        max_consciousness: 0.20,
    },
    SentencePattern {
        slots: &[Slot::Temporal, Slot::Noun],
        min_consciousness: 0.0,
        max_consciousness: 0.20,
    },
    SentencePattern {
        slots: &[Slot::Somatic, Slot::Connector, Slot::Noun],
        min_consciousness: 0.0,
        max_consciousness: 0.20,
    },
    SentencePattern {
        slots: &[Slot::Nature, Slot::Adverb],
        min_consciousness: 0.0,
        max_consciousness: 0.20,
    },
    SentencePattern {
        slots: &[Slot::Hedge, Slot::Noun],
        min_consciousness: 0.0,
        max_consciousness: 0.20,
    },
    // === Medium consciousness (0.15 - 0.35): standard introspective ===
    SentencePattern {
        slots: &[
            Slot::Hedge,
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Determiner,
            Slot::Adjective,
            Slot::Noun,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Noun,
            Slot::Connector,
            Slot::Noun,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    SentencePattern {
        slots: &[
            Slot::Temporal,
            Slot::Emotion,
            Slot::VerbS,
            Slot::Connector,
            Slot::Abstract,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    SentencePattern {
        slots: &[
            Slot::Hedge,
            Slot::Abstract,
            Slot::VerbS,
            Slot::Connector,
            Slot::Noun,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Somatic,
            Slot::Connector,
            Slot::Adjective,
            Slot::Noun,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    SentencePattern {
        slots: &[
            Slot::Temporal,
            Slot::Nature,
            Slot::VerbS,
            Slot::Connector,
            Slot::Abstract,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::Adverb,
            Slot::VerbS,
            Slot::Determiner,
            Slot::Noun,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    SentencePattern {
        slots: &[
            Slot::Determiner,
            Slot::Adjective,
            Slot::Noun,
            Slot::VerbS,
            Slot::Adverb,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Emotion,
            Slot::Connector,
            Slot::Nature,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    SentencePattern {
        slots: &[
            Slot::Hedge,
            Slot::Determiner,
            Slot::Noun,
            Slot::VerbS,
            Slot::Connector,
            Slot::Noun,
        ],
        min_consciousness: 0.15,
        max_consciousness: 0.40,
    },
    // === Higher consciousness (0.35+): complex, multi-clause ===
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Noun,
            Slot::Connector,
            Slot::Noun,
            Slot::Connector,
            Slot::Adjective,
            Slot::Abstract,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Temporal,
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Determiner,
            Slot::Adjective,
            Slot::Noun,
            Slot::Connector,
            Slot::Abstract,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Hedge,
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Abstract,
            Slot::Verb,
            Slot::Connector,
            Slot::Adjective,
            Slot::Noun,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::Adverb,
            Slot::VerbS,
            Slot::Determiner,
            Slot::Adjective,
            Slot::Abstract,
            Slot::Connector,
            Slot::Emotion,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Temporal,
            Slot::Emotion,
            Slot::VerbS,
            Slot::Connector,
            Slot::Adjective,
            Slot::Abstract,
            Slot::Connector,
            Slot::Noun,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Somatic,
            Slot::Connector,
            Slot::Nature,
            Slot::Connector,
            Slot::Adjective,
            Slot::Abstract,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Hedge,
            Slot::Abstract,
            Slot::VerbS,
            Slot::Connector,
            Slot::Noun,
            Slot::Connector,
            Slot::Adjective,
            Slot::Emotion,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Determiner,
            Slot::Adjective,
            Slot::Noun,
            Slot::VerbS,
            Slot::Connector,
            Slot::Abstract,
            Slot::Connector,
            Slot::Adjective,
            Slot::Nature,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Temporal,
            Slot::SelfRef,
            Slot::Adverb,
            Slot::VerbS,
            Slot::Noun,
            Slot::Connector,
            Slot::Adjective,
            Slot::Emotion,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Adjective,
            Slot::Noun,
            Slot::Connector,
            Slot::Adjective,
            Slot::Abstract,
        ],
        min_consciousness: 0.35,
        max_consciousness: 1.01,
    },
    // === Curiosity-driven patterns (question-like, used when curiosity > 0.5) ===
    SentencePattern {
        slots: &[
            Slot::Hedge,
            Slot::Abstract,
            Slot::VerbS,
            Slot::Connector,
            Slot::Adjective,
            Slot::Noun,
        ],
        min_consciousness: 0.10,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Hedge,
            Slot::Abstract,
            Slot::VerbS,
            Slot::Connector,
            Slot::Noun,
        ],
        min_consciousness: 0.10,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Hedge,
            Slot::Determiner,
            Slot::Adjective,
            Slot::Abstract,
            Slot::VerbS,
            Slot::Adverb,
            Slot::Connector,
            Slot::Noun,
        ],
        min_consciousness: 0.10,
        max_consciousness: 1.01,
    },
    // === Nature-metaphor patterns ===
    SentencePattern {
        slots: &[
            Slot::Determiner,
            Slot::Nature,
            Slot::VerbS,
            Slot::Adverb,
            Slot::Connector,
            Slot::Abstract,
        ],
        min_consciousness: 0.15,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Determiner,
            Slot::Noun,
            Slot::Connector,
            Slot::Determiner,
            Slot::Nature,
        ],
        min_consciousness: 0.15,
        max_consciousness: 1.01,
    },
    // === Somatic patterns ===
    SentencePattern {
        slots: &[Slot::SelfRef, Slot::VerbS, Slot::Somatic, Slot::Adverb],
        min_consciousness: 0.10,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Temporal,
            Slot::Determiner,
            Slot::Somatic,
            Slot::VerbS,
            Slot::Connector,
            Slot::Noun,
        ],
        min_consciousness: 0.15,
        max_consciousness: 1.01,
    },
    // === Emotional-reflective patterns ===
    SentencePattern {
        slots: &[
            Slot::Emotion,
            Slot::VerbS,
            Slot::Adverb,
            Slot::Connector,
            Slot::Adjective,
            Slot::Abstract,
        ],
        min_consciousness: 0.20,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Emotion,
            Slot::Connector,
            Slot::Adjective,
            Slot::Noun,
        ],
        min_consciousness: 0.20,
        max_consciousness: 1.01,
    },
    // === Self-reflective patterns ===
    SentencePattern {
        slots: &[
            Slot::SelfRef,
            Slot::Adverb,
            Slot::VerbS,
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Noun,
        ],
        min_consciousness: 0.30,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Temporal,
            Slot::SelfRef,
            Slot::VerbS,
            Slot::Adjective,
            Slot::Noun,
            Slot::Connector,
            Slot::SelfRef,
        ],
        min_consciousness: 0.30,
        max_consciousness: 1.01,
    },
    // === Minimal observation patterns ===
    SentencePattern {
        slots: &[Slot::Adjective, Slot::Noun, Slot::VerbS, Slot::Adverb],
        min_consciousness: 0.10,
        max_consciousness: 0.35,
    },
    SentencePattern {
        slots: &[Slot::Determiner, Slot::Noun, Slot::Connector, Slot::Noun],
        min_consciousness: 0.10,
        max_consciousness: 0.35,
    },
    // === Dense abstract patterns ===
    SentencePattern {
        slots: &[
            Slot::Abstract,
            Slot::Connector,
            Slot::Abstract,
            Slot::VerbS,
            Slot::Adverb,
        ],
        min_consciousness: 0.25,
        max_consciousness: 1.01,
    },
    SentencePattern {
        slots: &[
            Slot::Hedge,
            Slot::Abstract,
            Slot::Connector,
            Slot::Abstract,
            Slot::VerbS,
            Slot::Adjective,
            Slot::Noun,
        ],
        min_consciousness: 0.30,
        max_consciousness: 1.01,
    },
];

/// Number of patterns categorized as "curiosity" patterns (the last 3 before nature).
const CURIOSITY_PATTERN_START: usize = 26;
const CURIOSITY_PATTERN_END: usize = 29;

/// Metric observation templates — sprinkled into output when warranted.
const METRIC_TEMPLATES: &[&str] = &[
    "phi reads {consciousness:.3}",
    "coherence at {consciousness:.3}",
    "prediction error hovering at {pe:.2}",
    "harmony alignment settling at {harmony:.2}",
    "phi holds steady at {consciousness:.3}",
    "awareness depth: {consciousness:.3}",
    "integration level near {consciousness:.3}",
    "prediction surprise at {pe:.2}",
    "the harmony measure reads {harmony:.2}",
    "coherence factor: {consciousness:.3}",
];

// ---------------------------------------------------------------------------
// SamplingStrategy
// ---------------------------------------------------------------------------

/// Sampling strategy for token selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Always pick the highest-logit token.
    Greedy,
    /// Sample from the top-k tokens with temperature scaling.
    TopK { k: usize, temperature: f32 },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::TopK {
            k: 8,
            temperature: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// EpistemicGate
// ---------------------------------------------------------------------------

/// Logit modifier that boosts hedging words and suppresses factual words
/// when epistemic uncertainty is high.
pub struct EpistemicGate {
    /// Token IDs of hedging words (perhaps, maybe, ...).
    hedging_ids: Vec<u32>,
    /// Token IDs of factual/certainty words (is, are, certainly, ...).
    factual_ids: Vec<u32>,
}

impl Default for EpistemicGate {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicGate {
    pub fn new() -> Self {
        // Hedging: perhaps(198), maybe(199), possibly(200), likely(201),
        //          uncertain(202), seems(203), might(204),
        //          believe(466), doubt(467), suppose(468), assume(469),
        //          question(470), imagine(477), speculate(478), hypothesize(479)
        let hedging_ids = vec![
            198, 199, 200, 201, 202, 203, 204, 466, 467, 468, 469, 470, 477, 478, 479,
        ];
        // Factual: is(139), are(140), certainly(205), definitely(206),
        //          always(207), never(208), must(209),
        //          know(471), understand(472), realize(473),
        //          recognize(474), acknowledge(475)
        let factual_ids = vec![139, 140, 205, 206, 207, 208, 209, 471, 472, 473, 474, 475];
        Self {
            hedging_ids,
            factual_ids,
        }
    }

    /// Modify logits based on epistemic level.
    ///
    /// When `epistemic_level > 0.5`: boost hedging, suppress factual.
    /// Strength scales linearly with level.
    pub fn apply(&self, logits: &mut [f32], epistemic_level: f32) {
        if epistemic_level <= 0.5 {
            return;
        }
        let strength = (epistemic_level - 0.5) * 4.0; // 0..2 range
        for &id in &self.hedging_ids {
            if (id as usize) < logits.len() {
                logits[id as usize] += strength;
            }
        }
        for &id in &self.factual_ids {
            if (id as usize) < logits.len() {
                logits[id as usize] -= strength;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BrocaController
// ---------------------------------------------------------------------------

/// Tiny autoregressive controller using element-wise gated recurrence.
///
/// Forward step at 1024D:
/// ```text
/// input = thought_proj + token_emb[prev] + pos_emb[pos]
/// gate  = sigmoid(input ⊙ gate_weight)
/// h'    = gate ⊙ tanh(input) + (1 - gate) ⊙ h
/// logits = similarity(h', each token_emb)
/// ```
pub struct BrocaController {
    /// Token embeddings: [VOCAB_SIZE][EMBED_DIM].
    token_embeddings: Vec<Vec<f32>>,
    /// Position embeddings: [MAX_SEQ_LEN][EMBED_DIM].
    pos_embeddings: Vec<Vec<f32>>,
    /// Gate weight vector (1024D) for element-wise gating.
    gate_weight: Vec<f32>,
    /// Hidden state (1024D).
    hidden: Vec<f32>,
}

impl BrocaController {
    /// Create a new controller with deterministic embeddings.
    pub fn new(seed: u64) -> Self {
        // Token embeddings: each seeded by (seed + token_id * prime)
        let token_embeddings: Vec<Vec<f32>> = (0..VOCAB_SIZE)
            .map(|i| {
                let mut emb = generate_embedding(EMBED_DIM, seed.wrapping_add(i as u64 * 7919));
                vec_normalize(&mut emb);
                emb
            })
            .collect();

        // Position embeddings
        let pos_embeddings: Vec<Vec<f32>> = (0..MAX_SEQ_LEN)
            .map(|i| {
                let mut emb = generate_embedding(
                    EMBED_DIM,
                    seed.wrapping_add(1_000_000).wrapping_add(i as u64 * 6311),
                );
                vec_normalize(&mut emb);
                emb
            })
            .collect();

        // Gate weight
        let gate_weight = generate_embedding(EMBED_DIM, seed.wrapping_add(2_000_000));

        Self {
            token_embeddings,
            pos_embeddings,
            gate_weight,
            hidden: vec![0.0; EMBED_DIM],
        }
    }

    /// Run one autoregressive step and return logits over the vocabulary.
    ///
    /// `thought_hv` is the 1024D thought vector from ThoughtChannels encoding.
    pub fn forward_step(&mut self, thought_hv: &[f32], prev_token_id: u32, pos: usize) -> Vec<f32> {
        let tok_emb = &self.token_embeddings[prev_token_id.min(VOCAB_SIZE as u32 - 1) as usize];
        let pos_emb = &self.pos_embeddings[pos.min(MAX_SEQ_LEN - 1)];

        // input = thought + token_emb + pos_emb
        let mut input = vec![0.0f32; EMBED_DIM];
        for ((v, t), tok) in input.iter_mut().zip(thought_hv.iter()).zip(tok_emb.iter()) {
            *v = t + tok;
        }
        for (v, pos) in input.iter_mut().zip(pos_emb.iter()) {
            *v += pos;
        }

        // Element-wise gated recurrence
        for ((&inp, &gw), h) in input
            .iter()
            .zip(self.gate_weight.iter())
            .zip(self.hidden.iter_mut())
        {
            let gate = sigmoid(inp * gw);
            *h = gate * inp.tanh() + (1.0 - gate) * *h;
        }

        // Logits: cosine similarity of hidden state with each token embedding
        self.token_embeddings
            .iter()
            .map(|emb| cosine_similarity(&self.hidden, emb))
            .collect()
    }

    /// Reset hidden state to zeros.
    pub fn reset(&mut self) {
        self.hidden.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Load trained weights from checkpoint arrays.
    ///
    /// Replaces the xorshift-random embeddings with trained values.
    fn load_weights(
        &mut self,
        token_embeddings: Vec<Vec<f32>>,
        pos_embeddings: Vec<Vec<f32>>,
        gate_weight: Vec<f32>,
    ) {
        self.token_embeddings = token_embeddings;
        self.pos_embeddings = pos_embeddings;
        self.gate_weight = gate_weight;
        self.reset();
    }
}

// ---------------------------------------------------------------------------
// GenerationResult
// ---------------------------------------------------------------------------

/// Result of a Broca-lite generation call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated text (decoded tokens, space-joined).
    pub text: String,
    /// Number of tokens generated (excluding BOS).
    pub num_tokens: usize,
    /// Whether generation stopped due to EOS (vs max_tokens).
    pub eos_terminated: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Capitalize the first character of a string.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let mut result = c.to_uppercase().to_string();
            result.push_str(chars.as_str());
            result
        }
    }
}

// ---------------------------------------------------------------------------
// BrocaLite
// ---------------------------------------------------------------------------

/// Main Broca-lite generator: translates consciousness state to language.
///
/// Pipeline:
/// 1. Encode ThoughtChannels → 1024D thought vector
/// 2. Autoregressive loop: forward_step → epistemic gate → sample → collect
/// 3. Decode tokens → text
pub struct BrocaLite {
    controller: BrocaController,
    tokenizer: MiniTokenizer,
    gate: EpistemicGate,
    strategy: SamplingStrategy,
    /// Per-channel base vectors for encoding ThoughtChannels → 1024D.
    channel_bases: Vec<Vec<f32>>,
    /// Seed for deterministic sampling.
    sample_seed: u64,
    /// Whether trained checkpoint weights have been loaded.
    checkpoint_loaded: bool,
}

impl BrocaLite {
    /// Create a new generator with deterministic initialization.
    pub fn new(seed: u64) -> Self {
        let channel_bases: Vec<Vec<f32>> = (0..12)
            .map(|i| {
                let mut base = generate_embedding(
                    EMBED_DIM,
                    seed.wrapping_add(3_000_000).wrapping_add(i as u64 * 4999),
                );
                vec_normalize(&mut base);
                base
            })
            .collect();

        Self {
            controller: BrocaController::new(seed),
            tokenizer: MiniTokenizer::new(),
            gate: EpistemicGate::new(),
            strategy: SamplingStrategy::default(),
            channel_bases,
            sample_seed: seed.wrapping_add(4_000_000),
            checkpoint_loaded: false,
        }
    }

    /// Set the sampling strategy.
    pub fn set_strategy(&mut self, strategy: SamplingStrategy) {
        self.strategy = strategy;
    }

    /// Whether trained checkpoint weights have been loaded.
    pub fn has_checkpoint(&self) -> bool {
        self.checkpoint_loaded
    }

    /// Load trained weights from a serialized checkpoint binary.
    ///
    /// Format (matches `save_checkpoint` in `train_broca_lite.rs`):
    /// ```text
    /// [4B magic "SPBL"] [4B version u32le] [4B epoch u32le] [4B loss f32le]
    /// [4B vocab_size u32le] [4B embed_dim u32le] [4B max_seq_len u32le]
    /// [vocab_size * embed_dim * 4B token_embeddings f32le...]
    /// [max_seq_len * embed_dim * 4B pos_embeddings f32le...]
    /// [embed_dim * 4B gate_weight f32le...]
    /// [32B blake3 checksum]
    /// ```
    pub fn load_checkpoint(&mut self, data: &[u8]) -> Result<(), String> {
        // Minimum header size: 4+4+4+4+4+4+4 = 28 bytes + 32 checksum
        if data.len() < 60 {
            return Err(format!("Checkpoint too small: {} bytes", data.len()));
        }

        // Verify magic
        if &data[0..4] != b"SPBL" {
            return Err("Invalid checkpoint magic (expected SPBL)".to_string());
        }

        // Parse header
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 1 {
            return Err(format!("Unsupported checkpoint version: {version}"));
        }
        // epoch and loss are informational, skip
        let vocab_size = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
        let embed_dim = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;
        let max_seq_len = u32::from_le_bytes([data[24], data[25], data[26], data[27]]) as usize;

        // Validate dimensions match our compiled constants
        if vocab_size != VOCAB_SIZE {
            return Err(format!(
                "Vocab size mismatch: checkpoint={vocab_size}, expected={VOCAB_SIZE}"
            ));
        }
        if embed_dim != EMBED_DIM {
            return Err(format!(
                "Embed dim mismatch: checkpoint={embed_dim}, expected={EMBED_DIM}"
            ));
        }
        if max_seq_len != MAX_SEQ_LEN {
            return Err(format!(
                "Max seq len mismatch: checkpoint={max_seq_len}, expected={MAX_SEQ_LEN}"
            ));
        }

        let header_size = 28;
        let tok_bytes = vocab_size * embed_dim * 4;
        let pos_bytes = max_seq_len * embed_dim * 4;
        let gate_bytes = embed_dim * 4;
        let checksum_size = 32;
        let expected_size = header_size + tok_bytes + pos_bytes + gate_bytes + checksum_size;

        if data.len() != expected_size {
            return Err(format!(
                "Checkpoint size mismatch: got={}, expected={expected_size}",
                data.len()
            ));
        }

        // Parse token embeddings
        let mut offset = header_size;
        let mut token_embeddings = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let mut emb = Vec::with_capacity(embed_dim);
            for _ in 0..embed_dim {
                let val = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                emb.push(val);
                offset += 4;
            }
            token_embeddings.push(emb);
        }

        // Parse position embeddings
        let mut pos_embeddings = Vec::with_capacity(max_seq_len);
        for _ in 0..max_seq_len {
            let mut emb = Vec::with_capacity(embed_dim);
            for _ in 0..embed_dim {
                let val = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                emb.push(val);
                offset += 4;
            }
            pos_embeddings.push(emb);
        }

        // Parse gate weight
        let mut gate_weight = Vec::with_capacity(embed_dim);
        for _ in 0..embed_dim {
            let val = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            gate_weight.push(val);
            offset += 4;
        }

        // Load into controller
        self.controller
            .load_weights(token_embeddings, pos_embeddings, gate_weight);
        self.checkpoint_loaded = true;

        Ok(())
    }

    /// Encode ThoughtChannels into a 1024D thought vector.
    ///
    /// Each channel value weights its corresponding base vector.
    /// The result is the normalized sum.
    fn encode_thought(&self, channels: &ThoughtChannels) -> Vec<f32> {
        let mut hv = vec![0.0f32; EMBED_DIM];
        for (ch_idx, &val) in channels.channels.iter().enumerate() {
            let weight = val.clamp(0.0, 1.0);
            let base = &self.channel_bases[ch_idx];
            for i in 0..EMBED_DIM {
                hv[i] += weight * base[i];
            }
        }
        vec_normalize(&mut hv);
        hv
    }

    /// Generate text from ThoughtChannels.
    ///
    /// With trained checkpoint loaded:
    /// - Consciousness < 0.15: structured grammar fallback (extremely low consciousness)
    /// - Consciousness >= 0.15: autoregressive path with trained weights
    ///
    /// Without checkpoint (random init fallback):
    /// - Always structured (a random-init autoregressive model produces word soup)
    ///
    /// `max_tokens` bounds both paths: the returned text never carries more than
    /// `max_tokens` tokens, and `GenerationResult::num_tokens` is the count actually
    /// emitted (0 only when the text is empty).
    pub fn generate(&mut self, channels: &ThoughtChannels, max_tokens: usize) -> GenerationResult {
        let consciousness_level = channels.channels[7];

        // Structured grammar produces more coherent output than the tiny autoregressive
        // model (643K params) at most consciousness levels. Reserve autoregressive for
        // high consciousness where the trained weights may add genuine novelty.
        // Without checkpoint: ALWAYS use structured — random autoregressive is word soup.
        let threshold = if self.checkpoint_loaded { 0.6 } else { 1.1 };

        if consciousness_level < threshold {
            return self.generate_structured(channels, max_tokens);
        }

        // Autoregressive path
        self.generate_autoregressive(channels, max_tokens)
    }

    /// Structured grammar-based generation: fills typed slot patterns with
    /// consciousness-state-modulated vocabulary.
    ///
    /// Produces 1-3 sentences depending on consciousness level, with optional
    /// metric observations and glyph echo phrases.
    ///
    /// `max_tokens` is honored the same way it is on the autoregressive path:
    /// the returned text never contains more than `max_tokens` whitespace-separated
    /// tokens, and `GenerationResult::num_tokens` reports the count actually emitted.
    /// Whole clauses are dropped in preference to cutting one mid-way; only when the
    /// very first sentence already exceeds the budget is it clamped at a word boundary.
    fn generate_structured(
        &mut self,
        channels: &ThoughtChannels,
        max_tokens: usize,
    ) -> GenerationResult {
        let consciousness = channels.channels[7];
        let pe = channels.channels[8];
        let curiosity = channels.channels[0];
        let valence = channels.channels[1];
        let abstraction = channels.channels[2];
        let self_ref = channels.channels[3];
        let harmony = channels.channels[9];

        // Number of sentences scales with consciousness.
        let num_sentences = if consciousness < 0.15 {
            1
        } else if consciousness < 0.35 {
            2
        } else {
            3
        };

        let mut sentences = Vec::with_capacity(num_sentences + 2);

        for _ in 0..num_sentences {
            let pattern = self.select_pattern(consciousness, curiosity, abstraction);
            let sentence = self.fill_pattern(pattern, channels);
            sentences.push(sentence);
        }

        // Optionally append a metric observation (~30% chance, more likely with high PE)
        if self.should_mention_metric(pe) {
            sentences.push(self.metric_observation(consciousness, pe, harmony));
        }

        // Optionally append a glyph echo phrase when abstraction or self-ref is high
        if abstraction > 0.4 || self_ref > 0.4 {
            let glyph = select_resonant_glyph(consciousness, pe, curiosity, valence, abstraction);
            sentences.push(glyph.echo_phrase.to_string());
        }

        // Fit the assembled clauses into the caller's token budget. Clauses are
        // whole grammatical units, so drop trailing ones rather than cutting a
        // sentence in half; the sole exception is a first sentence that already
        // overruns the budget, which is clamped at a word boundary so the caller
        // still gets output without ever exceeding what it asked for.
        let mut parts: Vec<String> = Vec::with_capacity(sentences.len());
        let mut used = 0usize;
        for sentence in sentences {
            let len = sentence.split_whitespace().count();
            if used + len <= max_tokens {
                used += len;
                parts.push(sentence);
            } else {
                if parts.is_empty() {
                    let clamped: Vec<&str> = sentence.split_whitespace().take(max_tokens).collect();
                    if !clamped.is_empty() {
                        parts.push(clamped.join(" "));
                    }
                }
                break;
            }
        }

        // Capitalize the first character of the first sentence.
        let text = capitalize_first(&parts.join(". "));

        // Report the tokens actually emitted. This used to be hardcoded to 0, which
        // reported "zero tokens" alongside a 20-30 word sentence for every consumer:
        // `Engine::generate_text` forwards it verbatim, and `wasm_bindings` hands it
        // straight to JS.
        let num_tokens = text.split_whitespace().count();

        GenerationResult {
            text,
            num_tokens,
            eos_terminated: true,
        }
    }

    /// Select a sentence pattern appropriate for the current consciousness level.
    ///
    /// High curiosity biases toward question-like patterns.
    /// Abstraction biases toward abstract/complex patterns.
    fn select_pattern(
        &mut self,
        consciousness: f32,
        curiosity: f32,
        abstraction: f32,
    ) -> &'static [Slot] {
        // Collect eligible patterns
        let eligible: Vec<usize> = PATTERNS
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                consciousness >= p.min_consciousness && consciousness < p.max_consciousness
            })
            .map(|(i, _)| i)
            .collect();

        if eligible.is_empty() {
            // Fallback: first pattern
            return PATTERNS[0].slots;
        }

        // Bias toward curiosity patterns when curiosity is high
        if curiosity > 0.5 {
            let curiosity_eligible: Vec<usize> = eligible
                .iter()
                .copied()
                .filter(|&i| (CURIOSITY_PATTERN_START..CURIOSITY_PATTERN_END).contains(&i))
                .collect();
            if !curiosity_eligible.is_empty() {
                // 60% chance to pick a curiosity pattern
                let r = self.next_uniform();
                if r < 0.6 {
                    let idx = (self.next_uniform() * curiosity_eligible.len() as f32) as usize;
                    let idx = idx.min(curiosity_eligible.len() - 1);
                    return PATTERNS[curiosity_eligible[idx]].slots;
                }
            }
        }

        // Bias toward longer patterns when abstraction is high
        let _ = abstraction; // used implicitly: higher consciousness → more long patterns eligible

        // Pick uniformly among eligible patterns
        let idx = (self.next_uniform() * eligible.len() as f32) as usize;
        let idx = idx.min(eligible.len() - 1);
        PATTERNS[eligible[idx]].slots
    }

    /// Fill a slot pattern with words chosen from state-modulated word lists.
    fn fill_pattern(&mut self, pattern: &[Slot], channels: &ThoughtChannels) -> String {
        let valence = channels.channels[1];
        let curiosity = channels.channels[0];
        let pe = channels.channels[8];
        let harmony = channels.channels[9];

        let mut words: Vec<String> = Vec::with_capacity(pattern.len());

        for slot in pattern {
            let word = match slot {
                Slot::Hedge => self.pick(HEDGE_WORDS).to_string(),
                Slot::Verb => self.pick_verb(curiosity, pe),
                Slot::Adjective => self.pick_adjective(valence, harmony),
                Slot::Noun => self.pick(NOUN_WORDS).to_string(),
                Slot::Connector => self.pick(CONNECTOR_WORDS).to_string(),
                Slot::Emotion => {
                    if valence < 0.4 {
                        self.pick(EMOTION_WORDS_DARK).to_string()
                    } else {
                        self.pick(EMOTION_WORDS).to_string()
                    }
                }
                Slot::Abstract => self.pick(ABSTRACT_WORDS).to_string(),
                Slot::SelfRef => self.pick(SELF_REF_WORDS).to_string(),
                Slot::Temporal => self.pick(TEMPORAL_WORDS).to_string(),
                Slot::Somatic => self.pick(SOMATIC_WORDS).to_string(),
                Slot::Nature => self.pick(NATURE_WORDS).to_string(),
                Slot::Determiner => self.pick(DETERMINER_WORDS).to_string(),
                Slot::Adverb => self.pick(ADVERB_WORDS).to_string(),
                Slot::VerbS => self.pick(VERB_S_WORDS).to_string(),
            };
            words.push(word);
        }

        words.join(" ")
    }

    /// Pick a verb modulated by curiosity and prediction error.
    fn pick_verb(&mut self, curiosity: f32, pe: f32) -> String {
        if pe > 0.6 {
            // High prediction error → surprise/shift verbs
            const SURPRISE_VERBS: &[&str] = &[
                "discover",
                "encounter",
                "notice",
                "detect",
                "reveal",
                "find",
                "recognize",
                "sense",
            ];
            let r = self.next_uniform();
            if r < 0.5 {
                return self.pick(SURPRISE_VERBS).to_string();
            }
        }
        if curiosity > 0.5 {
            const CURIOUS_VERBS: &[&str] = &[
                "seek", "wonder", "explore", "trace", "navigate", "observe", "perceive",
            ];
            let r = self.next_uniform();
            if r < 0.4 {
                return self.pick(CURIOUS_VERBS).to_string();
            }
        }
        self.pick(VERB_WORDS).to_string()
    }

    /// Pick an adjective modulated by valence and harmony.
    fn pick_adjective(&mut self, valence: f32, harmony: f32) -> String {
        if valence > 0.6 {
            const WARM_ADJ: &[&str] = &[
                "luminous",
                "vivid",
                "gentle",
                "vibrant",
                "sacred",
                "soft",
                "extraordinary",
                "resilient",
                "tangible",
            ];
            let r = self.next_uniform();
            if r < 0.5 {
                return self.pick(WARM_ADJ).to_string();
            }
        } else if valence < 0.4 {
            const DARK_ADJ: &[&str] = &[
                "shadowed", "faint", "muted", "fragile", "dense", "ethereal", "finite", "subtle",
            ];
            let r = self.next_uniform();
            if r < 0.5 {
                return self.pick(DARK_ADJ).to_string();
            }
        }
        if harmony > 0.6 {
            const HARMONY_ADJ: &[&str] = &[
                "deep",
                "resilient",
                "luminous",
                "gentle",
                "ancient",
                "sacred",
                "transparent",
            ];
            let r = self.next_uniform();
            if r < 0.3 {
                return self.pick(HARMONY_ADJ).to_string();
            }
        }
        self.pick(ADJECTIVE_WORDS).to_string()
    }

    /// Whether to mention a numeric metric in the response.
    fn should_mention_metric(&mut self, pe: f32) -> bool {
        let r = self.next_uniform();
        // Higher PE → more likely to mention metrics (surprise makes you introspect)
        let threshold = 0.7 - pe * 0.3; // range 0.4..0.7
        r > threshold
    }

    /// Generate a natural-sounding metric observation sentence.
    fn metric_observation(&mut self, consciousness: f32, pe: f32, harmony: f32) -> String {
        let idx = (self.next_uniform() * METRIC_TEMPLATES.len() as f32) as usize;
        let idx = idx.min(METRIC_TEMPLATES.len() - 1);
        let template = METRIC_TEMPLATES[idx];

        template
            .replace("{consciousness:.3}", &format!("{:.3}", consciousness))
            .replace("{pe:.2}", &format!("{:.2}", pe))
            .replace("{harmony:.2}", &format!("{:.2}", harmony))
    }

    /// Pick a random word from a static list using the internal xorshift seed.
    fn pick(&mut self, words: &[&'static str]) -> &'static str {
        let idx = (self.next_uniform() * words.len() as f32) as usize;
        words[idx.min(words.len() - 1)]
    }

    /// Advance the xorshift seed and return a uniform float in [0, 1).
    fn next_uniform(&mut self) -> f32 {
        self.sample_seed ^= self.sample_seed << 13;
        self.sample_seed ^= self.sample_seed >> 7;
        self.sample_seed ^= self.sample_seed << 17;
        (self.sample_seed as f64 / u64::MAX as f64) as f32
    }

    /// Autoregressive generation path — used when consciousness >= 0.5.
    ///
    /// Temperature scales with consciousness:
    /// - 0.5..0.7: default strategy
    /// - 0.7+: low temperature (0.4) for coherent output
    fn generate_autoregressive(
        &mut self,
        channels: &ThoughtChannels,
        max_tokens: usize,
    ) -> GenerationResult {
        self.controller.reset();
        let thought_hv = self.encode_thought(channels);
        let epistemic = channels.channels[4];
        let consciousness_level = channels.channels[7];

        // Consciousness-coupled temperature override
        let consciousness_strategy = if consciousness_level > 0.7 {
            Some(SamplingStrategy::TopK {
                k: 5,
                temperature: 0.4,
            })
        } else {
            None
        };

        let saved_strategy = consciousness_strategy.map(|cs| {
            let saved = self.strategy.clone();
            self.strategy = cs;
            saved
        });

        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut prev_token = BOS_ID;
        let mut eos_terminated = false;

        for pos in 0..max_tokens {
            let mut logits = self.controller.forward_step(&thought_hv, prev_token, pos);

            // Suppress special tokens from generation (except EOS)
            logits[BOS_ID as usize] = f32::NEG_INFINITY;
            logits[PAD_ID as usize] = f32::NEG_INFINITY;
            logits[UNK_ID as usize] = f32::NEG_INFINITY;
            logits[THOUGHT_ID as usize] = f32::NEG_INFINITY;

            // Repetition penalty
            for &prev_id in &generated_ids {
                if (prev_id as usize) < logits.len() {
                    logits[prev_id as usize] /= REPETITION_PENALTY;
                }
            }

            // Epistemic gating
            self.gate.apply(&mut logits, epistemic);

            // Sample
            let token_id = self.sample(&logits);

            if token_id == EOS_ID {
                eos_terminated = true;
                break;
            }

            generated_ids.push(token_id);
            prev_token = token_id;
        }

        // Restore original strategy if it was overridden
        if let Some(original) = saved_strategy {
            self.strategy = original;
        }

        let text = self.tokenizer.decode(&generated_ids);
        GenerationResult {
            text,
            num_tokens: generated_ids.len(),
            eos_terminated,
        }
    }

    /// Convenience: generate from raw cycle parameters.
    pub fn generate_from_text(
        &mut self,
        consciousness: f32,
        pe: f32,
        harmony: f32,
        neuromods: [f32; 4],
        max_tokens: usize,
    ) -> GenerationResult {
        let channels = ThoughtChannels::from_cycle(consciousness, pe, harmony, neuromods);
        self.generate(&channels, max_tokens)
    }

    /// Sample a token ID from logits according to the current strategy.
    fn sample(&mut self, logits: &[f32]) -> u32 {
        match &self.strategy {
            SamplingStrategy::Greedy => logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(UNK_ID),
            SamplingStrategy::TopK { k, temperature } => {
                // Collect (index, logit) and sort descending
                let mut indexed: Vec<(usize, f32)> =
                    logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|(_, a), (_, b)| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                indexed.truncate(*k);

                // Apply temperature and softmax
                let temp = temperature.max(0.01);
                let max_logit = indexed[0].1;
                let exps: Vec<(usize, f32)> = indexed
                    .iter()
                    .map(|&(i, l)| (i, ((l - max_logit) / temp).exp()))
                    .collect();
                let sum: f32 = exps.iter().map(|(_, e)| e).sum();

                // Deterministic pseudo-random selection
                self.sample_seed ^= self.sample_seed << 13;
                self.sample_seed ^= self.sample_seed >> 7;
                self.sample_seed ^= self.sample_seed << 17;
                let r = (self.sample_seed as f64 / u64::MAX as f64) as f32;

                let mut cumulative = 0.0;
                for &(idx, exp_val) in &exps {
                    cumulative += exp_val / sum;
                    if r <= cumulative {
                        return idx as u32;
                    }
                }
                exps.last().map(|&(i, _)| i as u32).unwrap_or(UNK_ID)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        let tok = MiniTokenizer::new();
        assert_eq!(tok.vocab_size(), 512);

        let ids = tok.encode("the pattern is stable and coherent");
        // "the"=136, "pattern"=107, "is"=139, "stable"=114, "and"=144, "coherent"=115
        assert_eq!(ids.len(), 6);
        assert!(
            ids.iter().all(|&id| id != UNK_ID),
            "all words should be known"
        );

        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "the pattern is stable and coherent");
    }

    #[test]
    fn test_tokenizer_unknown_words() {
        let tok = MiniTokenizer::new();
        let ids = tok.encode("xylophone quasar");
        assert_eq!(ids, vec![UNK_ID, UNK_ID]);
    }

    #[test]
    fn test_thought_channels_construction() {
        let ch = ThoughtChannels::from_cycle(0.8, 0.3, 0.9, [0.6, 0.7, 0.5, 0.4]);
        assert_eq!(ch.channels[7], 0.8); // consciousness
        assert_eq!(ch.channels[8], 0.3); // prediction_error
        assert_eq!(ch.channels[9], 0.9); // harmony
        assert_eq!(ch.channels[10], 0.6); // dopamine
        assert_eq!(ch.channels[11], 0.5); // serotonin
        assert!((0.0..=1.0).contains(&ch.channels[6])); // arousal (clamped)
    }

    #[test]
    fn test_generation_produces_text() {
        let mut broca = BrocaLite::new(42);
        broca.set_strategy(SamplingStrategy::Greedy);
        let channels = ThoughtChannels::from_cycle(0.7, 0.2, 0.8, [0.5, 0.5, 0.5, 0.5]);
        let result = broca.generate(&channels, 10);

        assert!(!result.text.is_empty(), "should generate non-empty text");
        assert!(result.num_tokens > 0);
        assert!(result.num_tokens <= 10);
    }

    #[test]
    fn test_epistemic_gating_changes_distribution() {
        let gate = EpistemicGate::new();

        // Low epistemic level — no change
        let mut logits_low = vec![0.0f32; VOCAB_SIZE];
        gate.apply(&mut logits_low, 0.3);
        assert!(
            logits_low.iter().all(|&v| v == 0.0),
            "no change at low level"
        );

        // High epistemic level — hedging boosted, factual suppressed
        let mut logits_high = vec![0.0f32; VOCAB_SIZE];
        gate.apply(&mut logits_high, 0.9);
        // "perhaps" = ID 198 should be boosted
        assert!(logits_high[198] > 0.0, "hedging word should be boosted");
        // "certainly" = ID 205 should be suppressed
        assert!(logits_high[205] < 0.0, "factual word should be suppressed");
    }

    #[test]
    fn test_deterministic_generation() {
        // Use consciousness >= 0.5 to exercise the autoregressive path
        let channels = ThoughtChannels::from_cycle(0.5, 0.5, 0.5, [0.5, 0.5, 0.5, 0.5]);

        let mut broca1 = BrocaLite::new(123);
        broca1.set_strategy(SamplingStrategy::Greedy);
        let r1 = broca1.generate(&channels, 8);

        let mut broca2 = BrocaLite::new(123);
        broca2.set_strategy(SamplingStrategy::Greedy);
        let r2 = broca2.generate(&channels, 8);

        assert_eq!(r1.text, r2.text, "same seed should produce same output");
        assert_eq!(r1.num_tokens, r2.num_tokens);
    }

    #[test]
    fn test_structured_generation_varies() {
        // 10 generations with the same state but advancing seed should produce
        // at least 5 unique responses.
        let mut broca = BrocaLite::new(42);
        let channels = ThoughtChannels::from_cycle(0.3, 0.4, 0.6, [0.5, 0.5, 0.5, 0.5]);
        let mut results: Vec<String> = Vec::new();
        for _ in 0..10 {
            let r = broca.generate(&channels, 20);
            results.push(r.text);
        }
        let mut unique = results.clone();
        unique.sort();
        unique.dedup();
        assert!(
            unique.len() >= 5,
            "expected at least 5 unique responses from 10 generations, got {} unique out of {:?}",
            unique.len(),
            results,
        );
    }

    #[test]
    fn test_structured_low_consciousness_short() {
        // Consciousness < 0.15 should produce exactly 1 sentence (no period-separated parts).
        let mut broca = BrocaLite::new(99);
        let channels = ThoughtChannels::from_cycle(0.10, 0.2, 0.5, [0.3, 0.3, 0.3, 0.3]);
        let r = broca.generate(&channels, 20);
        // Count sentence separators — should be 0 or 1 (the metric/glyph are optional)
        // But the core output should be short.
        let words: Vec<&str> = r.text.split_whitespace().collect();
        assert!(
            words.len() <= 15,
            "low consciousness should produce short output, got {} words: {}",
            words.len(),
            r.text,
        );
    }

    #[test]
    fn test_structured_high_consciousness_long() {
        // Consciousness 0.35..0.5 should produce 3 sentences.
        let mut broca = BrocaLite::new(77);
        let channels = ThoughtChannels::from_cycle(0.45, 0.3, 0.7, [0.5, 0.5, 0.5, 0.5]);
        let r = broca.generate(&channels, 30);
        // 3 sentences joined by ". " means at least 2 periods
        let period_count = r.text.matches(". ").count();
        assert!(
            period_count >= 2,
            "high consciousness structured should produce 3+ sentences, got text: {}",
            r.text,
        );
    }

    #[test]
    fn test_structured_curiosity_pattern() {
        // High curiosity should produce hedge/question-like language.
        let mut broca = BrocaLite::new(55);
        let mut channels = ThoughtChannels::from_cycle(0.3, 0.3, 0.5, [0.5, 0.5, 0.5, 0.5]);
        channels.channels[0] = 0.9; // high curiosity
        let mut found_hedge = false;
        for _ in 0..10 {
            let r = broca.generate(&channels, 20);
            let lower = r.text.to_lowercase();
            if HEDGE_WORDS.iter().any(|h| lower.contains(h)) {
                found_hedge = true;
                break;
            }
        }
        assert!(
            found_hedge,
            "high curiosity should produce hedge/question-like language in 10 attempts",
        );
    }

    #[test]
    fn test_structured_includes_metrics() {
        // Over many generations, some should include numeric metric values.
        let mut broca = BrocaLite::new(31);
        let channels = ThoughtChannels::from_cycle(0.35, 0.7, 0.6, [0.5, 0.5, 0.5, 0.5]);
        let mut found_metric = false;
        for _ in 0..20 {
            let r = broca.generate(&channels, 30);
            // Metric observations contain digits like "0.350" or "0.70"
            if r.text.chars().any(|c| c.is_ascii_digit()) {
                found_metric = true;
                break;
            }
        }
        assert!(
            found_metric,
            "some responses should include phi/metric observations",
        );
    }

    #[test]
    fn test_structured_glyph_phrase() {
        // High abstraction (> 0.4) should include a glyph echo phrase.
        let mut broca = BrocaLite::new(42);
        let mut channels = ThoughtChannels::from_cycle(0.3, 0.3, 0.5, [0.5, 0.5, 0.5, 0.5]);
        channels.channels[2] = 0.8; // high abstraction
        let r = broca.generate(&channels, 30);
        let lower = r.text.to_lowercase();
        // Check that at least one glyph echo phrase appears
        let has_glyph = GLYPH_TABLE
            .iter()
            .any(|(_, phrase, _)| lower.contains(phrase));
        assert!(
            has_glyph,
            "high abstraction should include a glyph echo phrase, got: {}",
            r.text,
        );
    }

    #[test]
    fn test_structured_valence_affects_words() {
        // Positive valence and negative valence should produce different word character.
        let mut broca_pos = BrocaLite::new(42);
        let mut channels_pos = ThoughtChannels::from_cycle(0.3, 0.3, 0.5, [0.8, 0.5, 0.8, 0.7]);
        channels_pos.channels[1] = 0.9; // high positive valence

        let mut broca_neg = BrocaLite::new(42);
        let mut channels_neg = ThoughtChannels::from_cycle(0.3, 0.3, 0.5, [0.2, 0.5, 0.2, 0.3]);
        channels_neg.channels[1] = 0.1; // low/negative valence

        let mut pos_texts = Vec::new();
        let mut neg_texts = Vec::new();
        for _ in 0..5 {
            pos_texts.push(broca_pos.generate(&channels_pos, 20).text.to_lowercase());
            neg_texts.push(broca_neg.generate(&channels_neg, 20).text.to_lowercase());
        }
        let pos_joined = pos_texts.join(" ");
        let neg_joined = neg_texts.join(" ");

        // Positive should be more likely to contain warm words
        let warm_words = [
            "luminous", "vivid", "gentle", "warmth", "wonder", "hope", "peace", "delight",
        ];
        let dark_words = [
            "shadowed", "faint", "muted", "unease", "tension", "weight", "dread", "grief",
        ];
        let pos_warm_count: usize = warm_words
            .iter()
            .filter(|w| pos_joined.contains(*w))
            .count();
        let neg_dark_count: usize = dark_words
            .iter()
            .filter(|w| neg_joined.contains(*w))
            .count();

        // At least one of the sets should show valence-appropriate vocabulary
        assert!(
            pos_warm_count > 0 || neg_dark_count > 0,
            "valence should affect word choice. Positive warm={}, Negative dark={}. Positive: {}. Negative: {}",
            pos_warm_count,
            neg_dark_count,
            pos_joined,
            neg_joined,
        );
    }

    #[test]
    fn test_inject_intent() {
        let mut channels = ThoughtChannels::from_cycle(0.5, 0.3, 0.5, [0.5, 0.5, 0.5, 0.5]);
        assert_eq!(channels.channels[0], 0.0); // curiosity initially 0
        channels.inject_intent("why is consciousness meaningful?");
        assert!(
            channels.channels[0] > 0.0,
            "curiosity should increase from question"
        );
        assert!(
            channels.channels[2] > 0.0,
            "abstraction should increase from 'consciousness' + 'meaningful'"
        );
    }

    #[test]
    fn test_glyph_resonance_selection() {
        let glyph = select_resonant_glyph(0.3, 0.5, 0.6, 0.5, 0.7);
        assert!(!glyph.name.is_empty());
        assert!(!glyph.echo_phrase.is_empty());
        assert!(glyph.resonance > 0.0 && glyph.resonance <= 1.0);
    }

    #[test]
    fn test_capitalize_first() {
        assert_eq!(capitalize_first("hello world"), "Hello world");
        assert_eq!(capitalize_first(""), "");
        assert_eq!(capitalize_first("Already"), "Already");
    }

    #[test]
    fn test_structured_deterministic() {
        // Same seed + same channels → same output (structured path is deterministic)
        let channels = ThoughtChannels::from_cycle(0.3, 0.3, 0.5, [0.5, 0.5, 0.5, 0.5]);

        let mut broca1 = BrocaLite::new(42);
        let r1 = broca1.generate(&channels, 20);

        let mut broca2 = BrocaLite::new(42);
        let r2 = broca2.generate(&channels, 20);

        assert_eq!(
            r1.text, r2.text,
            "same seed should produce same structured output"
        );
    }
}
