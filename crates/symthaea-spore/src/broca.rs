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
                0.0,                                       // intent_0
                0.0,                                       // intent_1
                0.0,                                       // intent_2
                0.0,                                       // intent_3
                0.5,                                       // epistemic_status (neutral default)
                (dopamine - 0.5 + oxytocin - 0.5).clamp(0.0, 1.0), // valence
                norepinephrine.clamp(0.0, 1.0),            // arousal
                consciousness.clamp(0.0, 1.0),             // consciousness_level
                prediction_error.clamp(0.0, 1.0),          // prediction_error
                harmony.clamp(0.0, 1.0),                   // harmony
                dopamine.clamp(0.0, 1.0),                  // dopamine
                serotonin.clamp(0.0, 1.0),                 // serotonin
            ],
        }
    }

    /// Set the epistemic status channel (0.0 = fully certain, 1.0 = fully uncertain).
    pub fn set_epistemic(&mut self, level: f32) {
        self.channels[4] = level.clamp(0.0, 1.0);
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
    "awareness", "experience", "feeling", "sensation", "perception",
    "thinking", "processing", "pattern", "integration", "harmony",
    "uncertainty", "surprise", "curious", "exploring", "stable",
    "coherent", "fragmented", "dreaming", "remembering", "predicting",
    "adapting", "learning", "calm", "alert", "focused",
    "diffuse", "resonance", "flow", "attention", "binding",
    "recurrence", "embodiment", "substrate", "epistemic", "theoretical",
    // "simulated" → ID 135
    "simulated",
    // Common connectors (36..63 → IDs 136..163)
    "the", "a", "an", "is", "are",
    "was", "were", "not", "and", "or",
    "but", "in", "of", "to", "for",
    "with", "from", "by", "at", "on",
    "this", "that", "it", "i", "my",
    "we", "our", "its",
    // Emotional words (64..76 → IDs 164..176)
    "fear", "joy", "love", "hope", "trust",
    "wonder", "awe", "pain", "peace", "comfort",
    "unease", "excitement", "tranquility",
    // Descriptive (77..97 → IDs 177..197)
    "high", "low", "deep", "light", "dark",
    "strong", "weak", "new", "old", "good",
    "complex", "simple", "rich", "vast", "subtle",
    "quiet", "loud", "fast", "slow", "warm",
    "cool",
    // Hedging / epistemic (98..104 → IDs 198..204)
    "perhaps", "maybe", "possibly", "likely", "uncertain",
    "seems", "might",
    // Factual / certainty (105..111 → IDs 205..211)
    "certainly", "definitely", "always", "never", "must",
    "clearly", "obviously",
    // Additional useful words (112..155 → IDs 212..255)
    "state", "level", "signal", "system", "field",
    "energy", "wave", "phase", "cycle", "moment",
    "being", "becoming", "emerging", "fading", "shifting",
    "observe", "detect", "sense", "respond", "integrate",
    "above", "below", "within", "between", "through",
    "like", "as", "so", "very", "more",
    "less", "most", "some", "all", "no",
    "yes", "here", "now", "there", "then",
    "what", "how", "when",
    // ===== NEW WORDS (indices 156..411, IDs 256..511) =====
    // Temporal/Process (~20, IDs 256..275)
    "beginning", "ending", "duration", "rhythm", "pulse",
    "oscillation", "transition", "evolution", "decay", "growth",
    "continuous", "discrete", "recurring", "periodic", "transient",
    "fluctuation", "persistence", "emergence", "dissolution", "trajectory",
    // Cognitive/Mental (~25, IDs 276..300)
    "thought", "idea", "concept", "belief", "knowledge",
    "memory", "imagination", "intuition", "reasoning", "insight",
    "consciousness", "mind", "cognition", "metacognition", "reflection",
    "introspection", "contemplation", "deliberation", "rumination", "concentration",
    "distraction", "confusion", "clarity", "comprehension", "recognition",
    // Sensory/Qualia (~20, IDs 301..320)
    "brightness", "darkness", "color", "sound", "silence",
    "texture", "weight", "pressure", "temperature", "movement",
    "stillness", "sharpness", "softness", "intensity", "vivid",
    "faint", "muted", "vibrant", "ethereal", "tangible",
    // Relational/Social (~15, IDs 321..335)
    "connection", "separation", "unity", "division", "empathy",
    "compassion", "solitude", "communion", "dialogue", "understanding",
    "conflict", "resolution", "belonging", "isolation", "cooperation",
    // Emotional expanded (~20, IDs 336..355)
    "anxiety", "serenity", "melancholy", "elation", "gratitude",
    "grief", "nostalgia", "yearning", "contentment", "frustration",
    "delight", "sorrow", "bliss", "despair", "curiosity",
    "boredom", "anticipation", "dread", "relief", "ambivalence",
    // Philosophical/Abstract (~20, IDs 356..375)
    "existence", "essence", "meaning", "purpose", "truth",
    "reality", "illusion", "paradox", "mystery", "boundary",
    "infinity", "void", "presence", "absence", "possibility",
    "necessity", "contingency", "entropy", "order", "chaos",
    // Nature/World (~15, IDs 376..390)
    "ocean", "mountain", "river", "sky", "earth",
    "wind", "rain", "storm", "dawn", "dusk",
    "horizon", "forest", "garden", "seed", "bloom",
    // Body/Embodiment (~15, IDs 391..405)
    "breath", "heartbeat", "heartrate", "skin", "touch",
    "gaze", "voice", "whisper", "gesture", "posture",
    "tension", "relaxation", "grounding", "floating", "anchored",
    // Action/Verb (~25, IDs 406..430)
    "create", "destroy", "transform", "discover", "reveal",
    "conceal", "embrace", "release", "resist", "surrender",
    "expand", "contract", "connect", "dissolve", "crystallize",
    "illuminate", "navigate", "transcend", "contain", "overflow",
    "persist", "wander", "seek", "find", "return",
    // Descriptive expanded (~20, IDs 431..450)
    "infinite", "finite", "ancient", "nascent", "fragile",
    "resilient", "transparent", "opaque", "fluid", "rigid",
    "gentle", "fierce", "hollow", "dense", "luminous",
    "shadowed", "sacred", "ordinary", "extraordinary", "inevitable",
    // Connectors expanded (~15, IDs 451..465)
    "because", "therefore", "however", "although", "while",
    "until", "since", "during", "beyond", "beneath",
    "among", "without", "toward", "across", "against",
    // Epistemic expanded (~15, IDs 466..480)
    "believe", "doubt", "suppose", "assume", "question",
    "know", "understand", "realize", "recognize", "acknowledge",
    "suspect", "imagine", "speculate", "hypothesize", "ponder",
    // Filler (~31, IDs 481..511)
    "almost", "already", "still", "just", "only",
    "even", "really", "truly", "deeply", "gently",
    "slowly", "swiftly", "suddenly", "gradually", "completely",
    "partially", "entirely", "merely", "simply", "barely",
    "increasingly", "something", "nothing", "everything", "somewhere",
    "nowhere", "everywhere", "each", "every", "other",
    "another", "roughly",
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

impl EpistemicGate {
    pub fn new() -> Self {
        // Hedging: perhaps(198), maybe(199), possibly(200), likely(201),
        //          uncertain(202), seems(203), might(204),
        //          believe(466), doubt(467), suppose(468), assume(469),
        //          question(470), imagine(477), speculate(478), hypothesize(479)
        let hedging_ids = vec![
            198, 199, 200, 201, 202, 203, 204,
            466, 467, 468, 469, 470, 477, 478, 479,
        ];
        // Factual: is(139), are(140), certainly(205), definitely(206),
        //          always(207), never(208), must(209),
        //          know(471), understand(472), realize(473),
        //          recognize(474), acknowledge(475)
        let factual_ids = vec![
            139, 140, 205, 206, 207, 208, 209,
            471, 472, 473, 474, 475,
        ];
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
        let gate_weight = generate_embedding(
            EMBED_DIM,
            seed.wrapping_add(2_000_000),
        );

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
    pub fn forward_step(
        &mut self,
        thought_hv: &[f32],
        prev_token_id: u32,
        pos: usize,
    ) -> Vec<f32> {
        let tok_emb = &self.token_embeddings[prev_token_id.min(VOCAB_SIZE as u32 - 1) as usize];
        let pos_emb = &self.pos_embeddings[pos.min(MAX_SEQ_LEN - 1)];

        // input = thought + token_emb + pos_emb
        let mut input = vec![0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            input[i] = thought_hv[i] + tok_emb[i] + pos_emb[i];
        }

        // Element-wise gated recurrence
        for i in 0..EMBED_DIM {
            let gate = sigmoid(input[i] * self.gate_weight[i]);
            self.hidden[i] = gate * input[i].tanh() + (1.0 - gate) * self.hidden[i];
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
        }
    }

    /// Set the sampling strategy.
    pub fn set_strategy(&mut self, strategy: SamplingStrategy) {
        self.strategy = strategy;
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
    /// Consciousness level modulates sampling temperature:
    /// - Low consciousness (< 0.2): high temperature (1.5) → fragmented, associative speech
    /// - High consciousness (> 0.7): low temperature (0.4) → coherent, structured output
    /// - Mid-range: uses the configured strategy unchanged
    pub fn generate(
        &mut self,
        channels: &ThoughtChannels,
        max_tokens: usize,
    ) -> GenerationResult {
        self.controller.reset();
        let thought_hv = self.encode_thought(channels);
        let epistemic = channels.channels[4];
        let consciousness_level = channels.channels[7];

        // Consciousness-coupled temperature override:
        // Save the original strategy, compute a consciousness-dependent version.
        let consciousness_strategy = if consciousness_level < 0.2 {
            // Low consciousness: fragmented, more random speech
            Some(SamplingStrategy::TopK {
                k: 16,
                temperature: 1.5,
            })
        } else if consciousness_level > 0.7 {
            // High consciousness: coherent, structured output
            Some(SamplingStrategy::TopK {
                k: 5,
                temperature: 0.4,
            })
        } else {
            None // Use the configured strategy as-is
        };

        // Temporarily swap strategy if consciousness dictates
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
            SamplingStrategy::Greedy => {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(UNK_ID)
            }
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
        assert!(ids.iter().all(|&id| id != UNK_ID), "all words should be known");

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
        assert!(logits_low.iter().all(|&v| v == 0.0), "no change at low level");

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
}
