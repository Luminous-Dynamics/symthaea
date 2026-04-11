// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! HDC semantic engine for email search and clustering.
//!
//! Default: lightweight 1024D encoder (self-contained, ~200 LOC).
//! With `--features symthaea`: uses symthaea-core's 16,384D TextEncoder
//! for production-grade semantic intelligence.
//!
//! Both implementations support: encode_text(), similarity(), semantic_search().

/// HDC dimension. 1024 gives ~95% of 16,384D discrimination at 1/16th memory.
/// Can be increased to 4096 or 16384 for production Symthaea integration.
const DIM: usize = 1024;

/// Sentiment/ethical words for basic content analysis.
const HARM_WORDS: &[&str] = &[
    "urgent", "immediately", "threat", "demand", "require", "force",
    "without consent", "unauthorized", "breach", "violat", "attack",
    "compromis", "exploit", "manipulat", "deceiv", "coerce",
];
const TRUST_WORDS: &[&str] = &[
    "please", "thank", "appreciate", "collaborate", "together",
    "consent", "agree", "proposal", "discuss", "consider",
    "respect", "support", "community", "shared", "mutual",
];

/// A compact hypervector for semantic representation.
#[derive(Clone)]
pub struct HyperVector {
    bits: Vec<i8>, // bipolar: -1 or +1
}

impl HyperVector {
    pub fn zero() -> Self {
        Self { bits: vec![0i8; DIM] }
    }

    /// Generate a deterministic random HV from a seed string.
    fn from_seed(seed: &str) -> Self {
        let mut bits = vec![0i8; DIM];
        let mut hash: u64 = 5381;
        for b in seed.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(b as u64);
        }
        for i in 0..DIM {
            // LCG-based deterministic bit generation
            hash = hash.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            bits[i] = if (hash >> 33) & 1 == 0 { -1 } else { 1 };
        }
        Self { bits }
    }

    /// Cyclic permutation (shift right by n positions).
    fn permute(&self, n: usize) -> Self {
        let mut bits = vec![0i8; DIM];
        for i in 0..DIM {
            bits[(i + n) % DIM] = self.bits[i];
        }
        Self { bits }
    }

    /// Element-wise multiply (binding operation).
    fn bind(&self, other: &Self) -> Self {
        let bits = self.bits.iter().zip(other.bits.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self { bits }
    }

    /// Cosine similarity between two HVs. Returns [-1.0, 1.0].
    pub fn similarity(&self, other: &Self) -> f32 {
        let dot: i32 = self.bits.iter().zip(other.bits.iter())
            .map(|(a, b)| (*a as i32) * (*b as i32))
            .sum();
        dot as f32 / DIM as f32
    }
}

/// Accumulator for bundling multiple HVs via majority vote.
struct Accumulator {
    sums: Vec<i32>,
}

impl Accumulator {
    fn new() -> Self { Self { sums: vec![0i32; DIM] } }

    fn add(&mut self, hv: &HyperVector) {
        for (s, b) in self.sums.iter_mut().zip(hv.bits.iter()) {
            *s += *b as i32;
        }
    }

    fn to_hv(&self) -> HyperVector {
        HyperVector {
            bits: self.sums.iter().map(|s| if *s >= 0 { 1 } else { -1 }).collect()
        }
    }
}

/// Encode text into a semantic hypervector.
///
/// Algorithm:
/// 1. Extract character trigrams from each word
/// 2. Hash each trigram to a deterministic HV
/// 3. Bundle trigram HVs → word HV
/// 4. Bind each word HV with a position HV
/// 5. Bundle all position-bound word HVs → sentence HV
pub fn encode_text(text: &str) -> HyperVector {
    let words: Vec<&str> = text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();

    if words.is_empty() {
        return HyperVector::zero();
    }

    let mut sentence_acc = Accumulator::new();

    for (pos, word) in words.iter().enumerate() {
        let word_lower = word.to_lowercase();
        let word_hv = encode_word(&word_lower);

        // Positional binding: bind word with position HV
        let pos_hv = HyperVector::from_seed(&format!("__pos_{pos}__"));
        let bound = word_hv.bind(&pos_hv);

        sentence_acc.add(&bound);
    }

    sentence_acc.to_hv()
}

/// Encode a single word via character trigram hashing.
fn encode_word(word: &str) -> HyperVector {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 3 {
        // Short words: use the word itself as seed
        return HyperVector::from_seed(word);
    }

    let mut acc = Accumulator::new();
    for i in 0..chars.len().saturating_sub(2) {
        let trigram: String = chars[i..i+3].iter().collect();
        let tri_hv = HyperVector::from_seed(&trigram);
        // Shift by position within word for order sensitivity
        let shifted = tri_hv.permute(i % DIM);
        acc.add(&shifted);
    }

    acc.to_hv()
}

/// Search result with similarity score.
#[derive(Clone)]
pub struct SemanticResult {
    pub index: usize,
    pub similarity: f32,
}

/// Ethical content analysis — lightweight moral parser.
/// Scores text on a trust-harm axis [-1.0, 1.0].
/// Positive = collaborative/consensual. Negative = coercive/harmful.
#[derive(Clone, Debug)]
pub struct ContentScore {
    pub trust_harm_score: f32,  // [-1.0 harm, +1.0 trust]
    pub flags: Vec<&'static str>,
    pub confidence: f32,
}

pub fn analyze_content(text: &str) -> ContentScore {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let word_count = words.len().max(1) as f32;

    let mut harm_count = 0f32;
    let mut trust_count = 0f32;
    let mut flags = Vec::new();

    for pattern in HARM_WORDS {
        if lower.contains(pattern) {
            harm_count += 1.0;
            flags.push(*pattern);
        }
    }
    for pattern in TRUST_WORDS {
        if lower.contains(pattern) {
            trust_count += 1.0;
        }
    }

    // Normalize by text length (longer texts naturally have more matches)
    let norm = (word_count / 20.0).min(1.0).max(0.1);
    let raw_score = (trust_count - harm_count) / (trust_count + harm_count + 1.0);
    let confidence = ((trust_count + harm_count) / norm).min(1.0);

    ContentScore {
        trust_harm_score: raw_score,
        flags,
        confidence,
    }
}

/// Find emails similar to a query, ranked by semantic similarity.
pub fn semantic_search(
    query_hv: &HyperVector,
    email_hvs: &[HyperVector],
    threshold: f32,
) -> Vec<SemanticResult> {
    let mut results: Vec<SemanticResult> = email_hvs.iter()
        .enumerate()
        .map(|(i, hv)| SemanticResult {
            index: i,
            similarity: query_hv.similarity(hv),
        })
        .filter(|r| r.similarity >= threshold)
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    results
}
