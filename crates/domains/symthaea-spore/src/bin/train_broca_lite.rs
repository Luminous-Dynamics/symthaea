// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Train the Spore's BrocaLite model on the Broca training data.
//!
//! Usage:
//!   cargo run --release --bin train-broca-lite -- \
//!     --data ../symthaea-broca/data/train-v2-cleaned.jsonl \
//!     --eval ../symthaea-broca/data/eval-v2-cleaned.jsonl \
//!     --epochs 50 \
//!     --lr 0.001 \
//!     --output data/broca-spore-v1.bin
//!
//! The training data JSONL format (from Broca training pipeline):
//!   {"channels": [f32; 20], "target_text": "...", "target_ids": [...]}
//!
//! We use only `channels` (mapped to 12D Spore format) and `target_text`
//! (re-tokenized with MiniTokenizer's 512-word vocabulary).

use std::io::{Read, Write};
use std::path::Path;
use std::process;

// ============================================================================
// Constants (must match broca.rs exactly)
// ============================================================================

const EMBED_DIM: usize = 1024;
const VOCAB_SIZE: usize = 512;
const MAX_SEQ_LEN: usize = 128;

// ============================================================================
// Deterministic RNG (same as broca.rs)
// ============================================================================

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

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8)
}

fn vec_normalize(v: &mut [f32]) {
    let n = vec_norm(v);
    v.iter_mut().for_each(|x| *x /= n);
}

fn vec_normalize_to(v: &mut [f32], target: f32) {
    let n = vec_norm(v);
    let scale = target / n;
    v.iter_mut().for_each(|x| *x *= scale);
}

// ============================================================================
// Training data loading
// ============================================================================

#[derive(serde::Deserialize)]
struct RawTrainingPair {
    channels: Vec<f32>,
    target_text: String,
    #[serde(default)]
    target_ids: Vec<u32>,
}

/// A training pair mapped to Spore format.
struct SporeTrainingPair {
    /// 12-channel input (mapped from 20-channel Broca data).
    channels: [f32; 12],
    /// Token IDs from MiniTokenizer.
    token_ids: Vec<u32>,
}

/// Map 20-channel Broca format to 12-channel Spore format.
///
/// Broca 20-ch layout: [intent_0..7, phi, valence, arousal, consciousness,
///   prediction_error, harmony, dopamine, serotonin, ...]
/// Spore 12-ch layout: [intent_0..3, epistemic, valence, arousal,
///   consciousness, prediction_error, harmony, dopamine, serotonin]
fn map_channels(broca: &[f32]) -> [f32; 12] {
    let mut out = [0.0f32; 12];
    // Intent slots 0..3 from broca[0..4]
    for i in 0..4 {
        out[i] = if i < broca.len() {
            broca[i].clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
    // epistemic_status: not directly in training data, default 0.5
    out[4] = 0.5;
    // valence: broca[9] (if exists)
    out[5] = broca.get(9).copied().unwrap_or(0.5).clamp(0.0, 1.0);
    // arousal: broca[10]
    out[6] = broca.get(10).copied().unwrap_or(0.5).clamp(0.0, 1.0);
    // consciousness: broca[11] — but in training data ch[0..7] are intent
    // Actually the training data has ch[8]=phi, ch[9..]=continuous values
    // Let's use broca[8] as a proxy for consciousness level
    out[7] = broca.get(8).copied().unwrap_or(0.5).clamp(0.0, 1.0);
    // prediction_error: broca[12] or approximate from broca[8]
    out[8] = broca.get(12).copied().unwrap_or(0.3).clamp(0.0, 1.0);
    // harmony: broca[13]
    out[9] = broca.get(13).copied().unwrap_or(0.5).clamp(0.0, 1.0);
    // dopamine: broca[14]
    out[10] = broca.get(14).copied().unwrap_or(0.5).clamp(0.0, 1.0);
    // serotonin: broca[15]
    out[11] = broca.get(15).copied().unwrap_or(0.5).clamp(0.0, 1.0);
    out
}

fn load_dataset(path: &str, tokenizer: &MiniTokenizer) -> Result<Vec<SporeTrainingPair>, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;

    let mut pairs = Vec::new();
    for (i, line) in data.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw: RawTrainingPair =
            serde_json::from_str(line).map_err(|e| format!("parsing line {i}: {e}"))?;

        let channels = map_channels(&raw.channels);
        let token_ids = tokenizer.encode(&raw.target_text);

        // Skip pairs that are all UNK (no usable signal)
        let non_unk = token_ids.iter().filter(|&&id| id != UNK_ID).count();
        if non_unk < 2 {
            continue;
        }

        // Truncate to MAX_SEQ_LEN
        let token_ids = if token_ids.len() > MAX_SEQ_LEN - 1 {
            token_ids[..MAX_SEQ_LEN - 1].to_vec()
        } else {
            token_ids
        };

        pairs.push(SporeTrainingPair {
            channels,
            token_ids,
        });
    }

    Ok(pairs)
}

// ============================================================================
// MiniTokenizer (duplicated from broca.rs — needed for the binary)
// ============================================================================

const BOS_ID: u32 = 0;
const EOS_ID: u32 = 1;
#[allow(dead_code)]
const PAD_ID: u32 = 2;
const UNK_ID: u32 = 3;

struct MiniTokenizer {
    words: Vec<&'static str>,
}

impl MiniTokenizer {
    fn new() -> Self {
        Self {
            words: WORD_VOCAB.to_vec(),
        }
    }

    fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| {
                let lower = word.to_lowercase();
                let trimmed = lower.trim_matches(|c: char| !c.is_alphanumeric());
                if trimmed.len() == 1 {
                    let ch = trimmed.as_bytes()[0];
                    if (0x20..=0x7E).contains(&ch) {
                        return (ch - 0x20) as u32 + 5;
                    }
                }
                if let Some(idx) = self.words.iter().position(|w| *w == trimmed) {
                    return idx as u32 + 100;
                }
                UNK_ID
            })
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| match id {
                BOS_ID => Some("<bos>".to_string()),
                EOS_ID => None,
                PAD_ID => None,
                UNK_ID => Some("<unk>".to_string()),
                4 => Some("<thought>".to_string()),
                5..=99 => {
                    let ch = (id - 5) as u8 + 0x20;
                    Some((ch as char).to_string())
                }
                100..=511 => {
                    let idx = (id - 100) as usize;
                    self.words.get(idx).map(|w| w.to_string())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// The 412-word vocabulary (same as broca.rs)
const WORD_VOCAB: &[&str] = &[
    // Consciousness-relevant (0..35)
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
    "simulated",
    // Common connectors (36..63)
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
    // Emotional words (64..76)
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
    // Descriptive (77..97)
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
    // Hedging / epistemic (98..104)
    "perhaps",
    "maybe",
    "possibly",
    "likely",
    "uncertain",
    "seems",
    "might",
    // Factual / certainty (105..111)
    "certainly",
    "definitely",
    "always",
    "never",
    "must",
    "clearly",
    "obviously",
    // Additional useful words (112..155)
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
    // ===== NEW WORDS (indices 156..411) =====
    // Temporal/Process (~20)
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
    // Cognitive/Mental (~25)
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
    // Sensory/Qualia (~20)
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
    // Relational/Social (~15)
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
    // Emotional expanded (~20)
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
    // Philosophical/Abstract (~20)
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
    // Nature/World (~15)
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
    // Body/Embodiment (~15)
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
    // Action/Verb (~25)
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
    // Descriptive expanded (~20)
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
    // Connectors expanded (~15)
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
    // Epistemic expanded (~15)
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
    // Filler (~31)
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

// ============================================================================
// Model parameters (learnable)
// ============================================================================

struct BrocaLiteModel {
    /// Token embeddings: [VOCAB_SIZE][EMBED_DIM].
    token_embeddings: Vec<Vec<f32>>,
    /// Position embeddings: [MAX_SEQ_LEN][EMBED_DIM].
    pos_embeddings: Vec<Vec<f32>>,
    /// Gate weight vector (EMBED_DIM).
    gate_weight: Vec<f32>,
    /// Channel basis vectors (12 x EMBED_DIM) — frozen after init.
    channel_bases: Vec<Vec<f32>>,
}

impl BrocaLiteModel {
    fn new(seed: u64) -> Self {
        let token_embeddings: Vec<Vec<f32>> = (0..VOCAB_SIZE)
            .map(|i| {
                let mut emb = generate_embedding(EMBED_DIM, seed.wrapping_add(i as u64 * 7919));
                vec_normalize(&mut emb);
                emb
            })
            .collect();

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

        let gate_weight = generate_embedding(EMBED_DIM, seed.wrapping_add(2_000_000));

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
            token_embeddings,
            pos_embeddings,
            gate_weight,
            channel_bases,
        }
    }

    /// Encode 12-channel input to 1024D thought vector.
    fn encode_thought(&self, channels: &[f32; 12]) -> Vec<f32> {
        let mut hv = vec![0.0f32; EMBED_DIM];
        for (ch_idx, &val) in channels.iter().enumerate() {
            let weight = val.clamp(0.0, 1.0);
            let base = &self.channel_bases[ch_idx];
            for i in 0..EMBED_DIM {
                hv[i] += weight * base[i];
            }
        }
        vec_normalize(&mut hv);
        hv
    }

    /// Count total learnable parameters.
    fn param_count(&self) -> usize {
        VOCAB_SIZE * EMBED_DIM + MAX_SEQ_LEN * EMBED_DIM + EMBED_DIM
    }
}

// ============================================================================
// Forward pass with gradient tracking
// ============================================================================

/// Per-position forward state for backprop.
struct StepState {
    /// input = thought + tok_emb + pos_emb
    input: Vec<f32>,
    /// gate = sigmoid(input[i] * gate_weight[i])
    gate: Vec<f32>,
    /// tanh(input[i])
    tanh_input: Vec<f32>,
    /// hidden state BEFORE this step
    h_prev: Vec<f32>,
    /// hidden state AFTER this step
    h_next: Vec<f32>,
    /// Previous token ID (for embedding gradient routing).
    prev_token_id: u32,
    /// Position index.
    pos: usize,
    /// Logits (VOCAB_SIZE).
    logits: Vec<f32>,
    /// Softmax probabilities.
    probs: Vec<f32>,
    /// Target token ID.
    target_id: u32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Run one forward step, recording all intermediate state for backprop.
fn forward_step(
    model: &BrocaLiteModel,
    thought_hv: &[f32],
    prev_token_id: u32,
    pos: usize,
    h_prev: &[f32],
    target_id: u32,
) -> StepState {
    let tok_emb = &model.token_embeddings[prev_token_id.min(VOCAB_SIZE as u32 - 1) as usize];
    let pos_emb = &model.pos_embeddings[pos.min(MAX_SEQ_LEN - 1)];

    // input = thought + token_emb + pos_emb
    let mut input = vec![0.0f32; EMBED_DIM];
    for i in 0..EMBED_DIM {
        input[i] = thought_hv[i] + tok_emb[i] + pos_emb[i];
    }

    // Gate and hidden update
    let mut gate = vec![0.0f32; EMBED_DIM];
    let mut tanh_input = vec![0.0f32; EMBED_DIM];
    let mut h_next = vec![0.0f32; EMBED_DIM];

    for i in 0..EMBED_DIM {
        gate[i] = sigmoid(input[i] * model.gate_weight[i]);
        tanh_input[i] = input[i].tanh();
        h_next[i] = gate[i] * tanh_input[i] + (1.0 - gate[i]) * h_prev[i];
    }

    // Logits: dot(h_next, token_emb[j]) / (||h_next|| * ||token_emb[j]||)
    // = cosine similarity
    let h_norm = vec_norm(&h_next);
    let mut logits = vec![0.0f32; VOCAB_SIZE];
    for j in 0..VOCAB_SIZE {
        let emb_norm = vec_norm(&model.token_embeddings[j]);
        let dot: f32 = h_next
            .iter()
            .zip(model.token_embeddings[j].iter())
            .map(|(a, b)| a * b)
            .sum();
        logits[j] = dot / (h_norm * emb_norm);
    }

    // Scale logits for sharper softmax (cosine sim is in [-1,1], too flat for CE)
    let logit_scale = 10.0f32;
    for l in logits.iter_mut() {
        *l *= logit_scale;
    }

    // Softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; VOCAB_SIZE];
    let mut sum = 0.0f32;
    for j in 0..VOCAB_SIZE {
        probs[j] = (logits[j] - max_logit).exp();
        sum += probs[j];
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    StepState {
        input,
        gate,
        tanh_input,
        h_prev: h_prev.to_vec(),
        h_next,
        prev_token_id,
        pos,
        logits,
        probs,
        target_id,
    }
}

/// Compute cross-entropy loss for a sequence of steps.
fn compute_loss(steps: &[StepState]) -> f32 {
    let mut loss = 0.0f32;
    for step in steps {
        let p = step.probs[step.target_id as usize].max(1e-10);
        loss -= p.ln();
    }
    loss / steps.len().max(1) as f32
}

// ============================================================================
// Backward pass — manual BPTT through gated recurrence
// ============================================================================

struct Gradients {
    /// dL/d(token_embeddings): accumulated per-token.
    d_tok_emb: Vec<Vec<f32>>,
    /// dL/d(pos_embeddings): accumulated per-position.
    d_pos_emb: Vec<Vec<f32>>,
    /// dL/d(gate_weight).
    d_gate_weight: Vec<f32>,
}

impl Gradients {
    fn new() -> Self {
        Self {
            d_tok_emb: vec![vec![0.0; EMBED_DIM]; VOCAB_SIZE],
            d_pos_emb: vec![vec![0.0; EMBED_DIM]; MAX_SEQ_LEN],
            d_gate_weight: vec![0.0; EMBED_DIM],
        }
    }

    fn clip(&mut self, max_norm: f32) {
        clip_vec(&mut self.d_gate_weight, max_norm);
        for emb in &mut self.d_tok_emb {
            clip_vec(emb, max_norm);
        }
        for emb in &mut self.d_pos_emb {
            clip_vec(emb, max_norm);
        }
    }
}

fn clip_vec(v: &mut [f32], max_norm: f32) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        v.iter_mut().for_each(|x| *x *= scale);
    }
}

fn backward(model: &BrocaLiteModel, steps: &[StepState], bptt_window: usize) -> Gradients {
    let mut grads = Gradients::new();
    let n = steps.len();
    if n == 0 {
        return grads;
    }

    let logit_scale = 10.0f32;
    let inv_n = 1.0 / n as f32;

    // dL/dh accumulated for BPTT
    let mut dh = vec![0.0f32; EMBED_DIM];

    // Process steps in reverse
    let start = if n > bptt_window { n - bptt_window } else { 0 };

    for t in (start..n).rev() {
        let step = &steps[t];

        // dL/d(logits) = (probs - one_hot(target)) * inv_n
        // But logits = cosine * logit_scale, so we need dL/d(scaled_logits)
        let mut d_logits = vec![0.0f32; VOCAB_SIZE];
        for j in 0..VOCAB_SIZE {
            d_logits[j] = step.probs[j] * inv_n;
        }
        d_logits[step.target_id as usize] -= inv_n;

        // dL/d(h) from logits:
        // logit[j] = scale * dot(h, emb[j]) / (||h|| * ||emb[j]||)
        // Approximate gradient: treat ||h||, ||emb|| as constants for stability
        // dL/dh += sum_j d_logits[j] * scale * emb[j] / (||h|| * ||emb[j]||)
        let h_norm = vec_norm(&step.h_next);
        let mut dh_from_logits = vec![0.0f32; EMBED_DIM];
        for j in 0..VOCAB_SIZE {
            if d_logits[j].abs() < 1e-12 {
                continue;
            }
            let emb_norm = vec_norm(&model.token_embeddings[j]);
            let coeff = d_logits[j] * logit_scale / (h_norm * emb_norm);
            for i in 0..EMBED_DIM {
                dh_from_logits[i] += coeff * model.token_embeddings[j][i];
            }
        }

        // Also accumulate gradient on token embeddings from logits
        // dL/d(emb[j]) += d_logits[j] * scale * h / (||h|| * ||emb[j]||)
        for j in 0..VOCAB_SIZE {
            if d_logits[j].abs() < 1e-12 {
                continue;
            }
            let emb_norm = vec_norm(&model.token_embeddings[j]);
            let coeff = d_logits[j] * logit_scale / (h_norm * emb_norm);
            for i in 0..EMBED_DIM {
                grads.d_tok_emb[j][i] += coeff * step.h_next[i];
            }
        }

        // Total dh = dh from logits + dh carried from future step
        for i in 0..EMBED_DIM {
            dh[i] += dh_from_logits[i];
        }

        // h[i] = gate[i] * tanh(input[i]) + (1 - gate[i]) * h_prev[i]
        // dL/d(gate[i]) = dh[i] * (tanh(input[i]) - h_prev[i])
        // dL/d(tanh(input[i])) = dh[i] * gate[i]
        // dL/d(h_prev[i]) = dh[i] * (1 - gate[i])  -- carried backward

        let mut d_input = vec![0.0f32; EMBED_DIM];
        let mut dh_prev = vec![0.0f32; EMBED_DIM];

        for i in 0..EMBED_DIM {
            let g = step.gate[i];
            let ti = step.tanh_input[i];
            let hp = step.h_prev[i];

            // dL/d(gate) before sigmoid
            let d_gate = dh[i] * (ti - hp);
            // gate = sigmoid(input[i] * gate_weight[i])
            // d(sigmoid)/d(z) = g * (1 - g)
            let d_z = d_gate * g * (1.0 - g);

            // dL/d(gate_weight[i]) += d_z * input[i]
            grads.d_gate_weight[i] += d_z * step.input[i];

            // dL/d(input[i]) from gate path
            d_input[i] += d_z * model.gate_weight[i];

            // dL/d(input[i]) from tanh path
            // d(tanh)/d(input) = 1 - tanh^2
            let dtanh = 1.0 - ti * ti;
            d_input[i] += dh[i] * g * dtanh;

            // dL/d(h_prev)
            dh_prev[i] = dh[i] * (1.0 - g);
        }

        // input = thought + tok_emb + pos_emb
        // dL/d(tok_emb[prev_token]) += d_input
        // dL/d(pos_emb[pos]) += d_input
        let tok_idx = step.prev_token_id.min(VOCAB_SIZE as u32 - 1) as usize;
        let pos_idx = step.pos.min(MAX_SEQ_LEN - 1);
        for i in 0..EMBED_DIM {
            grads.d_tok_emb[tok_idx][i] += d_input[i];
            grads.d_pos_emb[pos_idx][i] += d_input[i];
        }

        // Carry dh backward
        dh = dh_prev;
    }

    grads
}

// ============================================================================
// Adam optimizer
// ============================================================================

struct AdamOptimizer {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    // Moments for token embeddings
    m_tok: Vec<Vec<f32>>,
    v_tok: Vec<Vec<f32>>,
    // Moments for pos embeddings
    m_pos: Vec<Vec<f32>>,
    v_pos: Vec<Vec<f32>>,
    // Moments for gate weight
    m_gate: Vec<f32>,
    v_gate: Vec<f32>,
}

impl AdamOptimizer {
    fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m_tok: vec![vec![0.0; EMBED_DIM]; VOCAB_SIZE],
            v_tok: vec![vec![0.0; EMBED_DIM]; VOCAB_SIZE],
            m_pos: vec![vec![0.0; EMBED_DIM]; MAX_SEQ_LEN],
            v_pos: vec![vec![0.0; EMBED_DIM]; MAX_SEQ_LEN],
            m_gate: vec![0.0; EMBED_DIM],
            v_gate: vec![0.0; EMBED_DIM],
        }
    }

    fn step(&mut self, model: &mut BrocaLiteModel, grads: &Gradients) {
        self.t += 1;
        let t = self.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;

        // Update token embeddings
        for j in 0..VOCAB_SIZE {
            adam_update_vec(
                &mut model.token_embeddings[j],
                &grads.d_tok_emb[j],
                &mut self.m_tok[j],
                &mut self.v_tok[j],
                bc1,
                bc2,
                lr,
                beta1,
                beta2,
                eps,
            );
        }

        // Update pos embeddings
        for j in 0..MAX_SEQ_LEN {
            adam_update_vec(
                &mut model.pos_embeddings[j],
                &grads.d_pos_emb[j],
                &mut self.m_pos[j],
                &mut self.v_pos[j],
                bc1,
                bc2,
                lr,
                beta1,
                beta2,
                eps,
            );
        }

        // Update gate weight
        adam_update_vec(
            &mut model.gate_weight,
            &grads.d_gate_weight,
            &mut self.m_gate,
            &mut self.v_gate,
            bc1,
            bc2,
            lr,
            beta1,
            beta2,
            eps,
        );
    }
}

fn adam_update_vec(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    bc1: f32,
    bc2: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) {
    for i in 0..param.len() {
        let g = grad[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        param[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

// ============================================================================
// Checkpoint save/load
// ============================================================================

fn save_checkpoint(
    model: &BrocaLiteModel,
    path: &str,
    epoch: usize,
    loss: f32,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // Header: magic + version + epoch + loss
    file.write_all(b"SPBL")?; // magic: SPore Broca Lite
    file.write_all(&1u32.to_le_bytes())?; // version
    file.write_all(&(epoch as u32).to_le_bytes())?;
    file.write_all(&loss.to_le_bytes())?;

    // Dimensions
    file.write_all(&(VOCAB_SIZE as u32).to_le_bytes())?;
    file.write_all(&(EMBED_DIM as u32).to_le_bytes())?;
    file.write_all(&(MAX_SEQ_LEN as u32).to_le_bytes())?;

    // Token embeddings
    for emb in &model.token_embeddings {
        let bytes: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;
    }

    // Position embeddings
    for emb in &model.pos_embeddings {
        let bytes: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;
    }

    // Gate weight
    let bytes: Vec<u8> = model
        .gate_weight
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    file.write_all(&bytes)?;

    // Blake3 checksum of all written data
    let data = std::fs::read(path)?;
    let hash = blake3::hash(&data);
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    file.write_all(hash.as_bytes())?;

    file.sync_all()?;
    Ok(())
}

fn checkpoint_size(model: &BrocaLiteModel) -> usize {
    let header = 4 + 4 + 4 + 4 + 4 + 4 + 4; // magic + version + epoch + loss + 3 dims
    let tok = model.token_embeddings.len() * EMBED_DIM * 4;
    let pos = model.pos_embeddings.len() * EMBED_DIM * 4;
    let gate = EMBED_DIM * 4;
    let checksum = 32;
    header + tok + pos + gate + checksum
}

// ============================================================================
// Training loop
// ============================================================================

fn train_epoch(
    model: &mut BrocaLiteModel,
    optimizer: &mut AdamOptimizer,
    data: &[SporeTrainingPair],
    bptt_window: usize,
    report_interval: usize,
    embedding_target_norm: f32,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut total_steps = 0usize;

    for (pair_idx, pair) in data.iter().enumerate() {
        if pair.token_ids.is_empty() {
            continue;
        }

        let thought_hv = model.encode_thought(&pair.channels);

        // Teacher-forced forward pass
        let mut h = vec![0.0f32; EMBED_DIM];
        let mut steps = Vec::with_capacity(pair.token_ids.len());

        // First step: BOS → first token
        let first_target = pair.token_ids[0];
        let state = forward_step(model, &thought_hv, BOS_ID, 0, &h, first_target);
        h = state.h_next.clone();
        steps.push(state);

        // Subsequent steps: token[t-1] → token[t]
        for t in 1..pair.token_ids.len() {
            let prev = pair.token_ids[t - 1];
            let target = pair.token_ids[t];
            let state = forward_step(model, &thought_hv, prev, t, &h, target);
            h = state.h_next.clone();
            steps.push(state);
        }

        let loss = compute_loss(&steps);
        total_loss += loss;
        total_steps += 1;

        // Backward
        let mut grads = backward(model, &steps, bptt_window);
        grads.clip(5.0);

        // Adam update
        optimizer.step(model, &grads);

        // Re-normalize embeddings to prevent norm explosion
        if embedding_target_norm > 0.0 {
            for emb in &mut model.token_embeddings {
                vec_normalize_to(emb, embedding_target_norm);
            }
        }

        if report_interval > 0 && (pair_idx + 1) % report_interval == 0 {
            let avg = total_loss / total_steps as f32;
            eprintln!(
                "  step {}/{}: avg_loss={:.4}",
                pair_idx + 1,
                data.len(),
                avg
            );
        }
    }

    total_loss / total_steps.max(1) as f32
}

fn eval_epoch(model: &BrocaLiteModel, data: &[SporeTrainingPair]) -> (f32, f32) {
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let mut total_pairs = 0usize;

    for pair in data {
        if pair.token_ids.is_empty() {
            continue;
        }

        let thought_hv = model.encode_thought(&pair.channels);
        let mut h = vec![0.0f32; EMBED_DIM];
        let mut steps = Vec::with_capacity(pair.token_ids.len());

        let first_target = pair.token_ids[0];
        let state = forward_step(model, &thought_hv, BOS_ID, 0, &h, first_target);
        h = state.h_next.clone();
        steps.push(state);

        for t in 1..pair.token_ids.len() {
            let prev = pair.token_ids[t - 1];
            let target = pair.token_ids[t];
            let state = forward_step(model, &thought_hv, prev, t, &h, target);
            h = state.h_next.clone();
            steps.push(state);
        }

        let loss = compute_loss(&steps);
        total_loss += loss * steps.len() as f32;
        total_tokens += steps.len();
        total_pairs += 1;
    }

    let avg_loss = total_loss / total_tokens.max(1) as f32;
    let perplexity = avg_loss.exp();
    (avg_loss, perplexity)
}

/// Generate a sample from the model (greedy decoding).
fn generate_sample(
    model: &BrocaLiteModel,
    channels: &[f32; 12],
    tokenizer: &MiniTokenizer,
    max_tokens: usize,
) -> String {
    let thought_hv = model.encode_thought(channels);
    let mut h = vec![0.0f32; EMBED_DIM];
    let mut prev_token = BOS_ID;
    let mut ids = Vec::new();

    for pos in 0..max_tokens {
        let state = forward_step(model, &thought_hv, prev_token, pos, &h, 0);
        h = state.h_next;

        // Greedy: pick argmax (skip special tokens)
        let mut best_id = UNK_ID;
        let mut best_logit = f32::NEG_INFINITY;
        for (j, &l) in state.logits.iter().enumerate() {
            if j <= 4 {
                continue; // skip BOS, EOS, PAD, UNK, THOUGHT
            }
            if l > best_logit {
                best_logit = l;
                best_id = j as u32;
            }
        }

        if best_id == EOS_ID {
            break;
        }
        ids.push(best_id);
        prev_token = best_id;
    }

    tokenizer.decode(&ids)
}

// ============================================================================
// CLI argument parsing
// ============================================================================

struct Opts {
    data_path: String,
    eval_path: Option<String>,
    epochs: usize,
    lr: f32,
    output_path: String,
    seed: u64,
    bptt_window: usize,
    patience: usize,
    report_interval: usize,
    embedding_target_norm: f32,
}

fn parse_args() -> Result<Opts, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut opts = Opts {
        data_path: String::new(),
        eval_path: None,
        epochs: 50,
        lr: 0.001,
        output_path: String::from("broca-spore-v1.bin"),
        seed: 42,
        bptt_window: 32,
        patience: 5,
        report_interval: 500,
        embedding_target_norm: 1.0,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                opts.data_path = args.get(i).cloned().ok_or("--data requires a path")?;
            }
            "--eval" => {
                i += 1;
                opts.eval_path = Some(args.get(i).cloned().ok_or("--eval requires a path")?);
            }
            "--epochs" => {
                i += 1;
                opts.epochs = args
                    .get(i)
                    .ok_or("--epochs requires a number")?
                    .parse()
                    .map_err(|_| "invalid epochs")?;
            }
            "--lr" => {
                i += 1;
                opts.lr = args
                    .get(i)
                    .ok_or("--lr requires a number")?
                    .parse()
                    .map_err(|_| "invalid lr")?;
            }
            "--output" => {
                i += 1;
                opts.output_path = args.get(i).cloned().ok_or("--output requires a path")?;
            }
            "--seed" => {
                i += 1;
                opts.seed = args
                    .get(i)
                    .ok_or("--seed requires a number")?
                    .parse()
                    .map_err(|_| "invalid seed")?;
            }
            "--bptt-window" => {
                i += 1;
                opts.bptt_window = args
                    .get(i)
                    .ok_or("--bptt-window requires a number")?
                    .parse()
                    .map_err(|_| "invalid bptt_window")?;
            }
            "--patience" => {
                i += 1;
                opts.patience = args
                    .get(i)
                    .ok_or("--patience requires a number")?
                    .parse()
                    .map_err(|_| "invalid patience")?;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    if opts.data_path.is_empty() {
        return Err("--data is required".to_string());
    }

    Ok(opts)
}

fn print_usage() {
    eprintln!(
        "Usage: train-broca-lite --data <path.jsonl> [options]

Options:
  --data <path>           Training data (JSONL)
  --eval <path>           Validation data (JSONL)
  --epochs <N>            Number of epochs (default: 50)
  --lr <float>            Learning rate (default: 0.001)
  --output <path>         Output checkpoint (default: broca-spore-v1.bin)
  --seed <N>              Random seed (default: 42)
  --bptt-window <N>       BPTT truncation window (default: 32)
  --patience <N>          Early stopping patience (default: 5)
  --help                  Show this help"
    );
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Error: {e}");
            print_usage();
            process::exit(1);
        }
    };

    let tokenizer = MiniTokenizer::new();

    // Load datasets
    eprintln!("Loading training data from: {}", opts.data_path);
    let train_data = match load_dataset(&opts.data_path, &tokenizer) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed: {e}");
            process::exit(1);
        }
    };
    eprintln!("  {} training pairs loaded", train_data.len());

    // Analyze UNK rate
    let total_tokens: usize = train_data.iter().map(|p| p.token_ids.len()).sum();
    let unk_tokens: usize = train_data
        .iter()
        .flat_map(|p| p.token_ids.iter())
        .filter(|&&id| id == UNK_ID)
        .count();
    eprintln!(
        "  UNK rate: {:.1}% ({} / {} tokens)",
        100.0 * unk_tokens as f64 / total_tokens.max(1) as f64,
        unk_tokens,
        total_tokens,
    );

    let eval_data = opts.eval_path.as_ref().map(|path| {
        eprintln!("Loading eval data from: {}", path);
        match load_dataset(path, &tokenizer) {
            Ok(d) => {
                eprintln!("  {} eval pairs loaded", d.len());
                d
            }
            Err(e) => {
                eprintln!("Warning: failed to load eval data: {e}");
                Vec::new()
            }
        }
    });

    // Initialize model
    eprintln!("Initializing BrocaLite model (seed={})", opts.seed);
    let mut model = BrocaLiteModel::new(opts.seed);
    eprintln!(
        "  Parameters: {} ({:.2} MB)",
        model.param_count(),
        model.param_count() as f64 * 4.0 / 1024.0 / 1024.0
    );
    eprintln!(
        "  Checkpoint size: {:.2} MB",
        checkpoint_size(&model) as f64 / 1024.0 / 1024.0
    );

    let mut optimizer = AdamOptimizer::new(opts.lr);

    // Training loop
    eprintln!(
        "\nTraining for {} epochs (lr={}, bptt_window={}, patience={})\n",
        opts.epochs, opts.lr, opts.bptt_window, opts.patience
    );

    let mut best_eval_loss = f32::MAX;
    let mut patience_counter = 0usize;
    let mut best_epoch = 0usize;

    for epoch in 1..=opts.epochs {
        let start = std::time::Instant::now();
        let train_loss = train_epoch(
            &mut model,
            &mut optimizer,
            &train_data,
            opts.bptt_window,
            opts.report_interval,
            opts.embedding_target_norm,
        );
        let elapsed = start.elapsed();

        // Eval
        let eval_str = if let Some(ref eval) = eval_data {
            if !eval.is_empty() {
                let (eval_loss, perplexity) = eval_epoch(&model, eval);

                // Early stopping
                if eval_loss < best_eval_loss {
                    best_eval_loss = eval_loss;
                    best_epoch = epoch;
                    patience_counter = 0;

                    // Save best checkpoint
                    if let Err(e) = save_checkpoint(&model, &opts.output_path, epoch, eval_loss) {
                        eprintln!("Warning: failed to save checkpoint: {e}");
                    }
                } else {
                    patience_counter += 1;
                }

                format!(
                    " | eval_loss={:.4} ppl={:.2} best={:.4}@e{}",
                    eval_loss, perplexity, best_eval_loss, best_epoch
                )
            } else {
                String::new()
            }
        } else {
            // No eval data — save every epoch
            if train_loss < best_eval_loss {
                best_eval_loss = train_loss;
                best_epoch = epoch;
                if let Err(e) = save_checkpoint(&model, &opts.output_path, epoch, train_loss) {
                    eprintln!("Warning: failed to save checkpoint: {e}");
                }
            }
            String::new()
        };

        // Sample generation every 5 epochs
        let sample_str = if epoch % 5 == 0 || epoch == 1 {
            // High-consciousness sample
            let channels = [0.5, 0.5, 0.3, 0.2, 0.5, 0.6, 0.5, 0.7, 0.3, 0.6, 0.5, 0.5];
            let text = generate_sample(&model, &channels, &tokenizer, 16);
            format!("\n    sample: \"{}\"", text)
        } else {
            String::new()
        };

        eprintln!(
            "epoch {:3}/{}: train_loss={:.4} ({:.1}s){}{}",
            epoch,
            opts.epochs,
            train_loss,
            elapsed.as_secs_f32(),
            eval_str,
            sample_str,
        );

        // Early stopping check
        if opts.patience > 0 && patience_counter >= opts.patience {
            eprintln!(
                "\nEarly stopping: no improvement for {} epochs (best={:.4}@e{})",
                opts.patience, best_eval_loss, best_epoch
            );
            break;
        }
    }

    // Final sample generation
    eprintln!("\n--- Final generation samples ---");
    let test_states: &[(&str, [f32; 12])] = &[
        (
            "high consciousness",
            [0.5, 0.5, 0.3, 0.2, 0.5, 0.7, 0.5, 0.8, 0.2, 0.7, 0.6, 0.5],
        ),
        (
            "curious + uncertain",
            [0.8, 0.3, 0.6, 0.1, 0.8, 0.4, 0.6, 0.5, 0.6, 0.4, 0.5, 0.4],
        ),
        (
            "calm + peaceful",
            [0.1, 0.7, 0.1, 0.0, 0.3, 0.8, 0.2, 0.4, 0.1, 0.9, 0.4, 0.7],
        ),
        (
            "high prediction error",
            [0.3, 0.3, 0.2, 0.1, 0.5, 0.4, 0.7, 0.6, 0.9, 0.3, 0.7, 0.3],
        ),
    ];

    for (label, channels) in test_states {
        let text = generate_sample(&model, channels, &tokenizer, 20);
        eprintln!("  [{}]: {}", label, text);
    }

    eprintln!(
        "\nCheckpoint saved to: {} ({:.2} MB)",
        opts.output_path,
        checkpoint_size(&model) as f64 / 1024.0 / 1024.0
    );
    eprintln!("Training complete.");
}
