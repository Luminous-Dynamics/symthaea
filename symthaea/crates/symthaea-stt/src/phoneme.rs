// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phoneme recognition using Modern Hopfield Networks
//!
//! This module implements:
//! - A complete inventory of 44 English phonemes (IPA)
//! - Articulatory feature decomposition
//! - Modern Hopfield resonator for pattern matching
//! - Coarticulation model for sequence constraints

use crate::hdc::{BundleAccumulator, HV16, bundle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Place of articulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Place {
    Bilabial,
    Labiodental,
    Dental,
    Alveolar,
    Postalveolar,
    Palatal,
    Velar,
    Glottal,
    // Vowel positions
    Front,
    Central,
    Back,
}

/// Manner of articulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Manner {
    Stop,
    Fricative,
    Affricate,
    Nasal,
    Approximant,
    Lateral,
    // Vowel heights
    High,
    Mid,
    Low,
}

/// Voicing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Voicing {
    Voiced,
    Voiceless,
}

/// Complete phoneme specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phoneme {
    /// IPA symbol
    pub ipa: String,
    /// ARPAbet equivalent
    pub arpabet: String,
    /// Place of articulation
    pub place: Place,
    /// Manner of articulation
    pub manner: Manner,
    /// Voicing
    pub voicing: Voicing,
    /// Is this a vowel?
    pub is_vowel: bool,
    /// Prototype hypervector (learned)
    pub prototype: Option<HV16>,
}

/// Complete English phoneme inventory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeInventory {
    phonemes: Vec<Phoneme>,
    ipa_to_idx: HashMap<String, usize>,
    arpabet_to_idx: HashMap<String, usize>,
    /// Feature hypervectors
    place_hvs: HashMap<Place, HV16>,
    manner_hvs: HashMap<Manner, HV16>,
    voicing_hvs: HashMap<Voicing, HV16>,
}

impl Default for PhonemeInventory {
    fn default() -> Self {
        Self::new()
    }
}

impl PhonemeInventory {
    /// Create the standard 44 English phoneme inventory
    pub fn new() -> Self {
        use Manner::*;
        use Place::*;
        use Voicing::*;

        let phonemes = vec![
            // === CONSONANTS ===
            // Stops
            Phoneme::consonant("p", "P", Bilabial, Stop, Voiceless),
            Phoneme::consonant("b", "B", Bilabial, Stop, Voiced),
            Phoneme::consonant("t", "T", Alveolar, Stop, Voiceless),
            Phoneme::consonant("d", "D", Alveolar, Stop, Voiced),
            Phoneme::consonant("k", "K", Velar, Stop, Voiceless),
            Phoneme::consonant("ɡ", "G", Velar, Stop, Voiced),
            // Fricatives
            Phoneme::consonant("f", "F", Labiodental, Fricative, Voiceless),
            Phoneme::consonant("v", "V", Labiodental, Fricative, Voiced),
            Phoneme::consonant("θ", "TH", Dental, Fricative, Voiceless),
            Phoneme::consonant("ð", "DH", Dental, Fricative, Voiced),
            Phoneme::consonant("s", "S", Alveolar, Fricative, Voiceless),
            Phoneme::consonant("z", "Z", Alveolar, Fricative, Voiced),
            Phoneme::consonant("ʃ", "SH", Postalveolar, Fricative, Voiceless),
            Phoneme::consonant("ʒ", "ZH", Postalveolar, Fricative, Voiced),
            Phoneme::consonant("h", "HH", Glottal, Fricative, Voiceless),
            // Affricates
            Phoneme::consonant("tʃ", "CH", Postalveolar, Affricate, Voiceless),
            Phoneme::consonant("dʒ", "JH", Postalveolar, Affricate, Voiced),
            // Nasals
            Phoneme::consonant("m", "M", Bilabial, Nasal, Voiced),
            Phoneme::consonant("n", "N", Alveolar, Nasal, Voiced),
            Phoneme::consonant("ŋ", "NG", Velar, Nasal, Voiced),
            // Approximants
            Phoneme::consonant("l", "L", Alveolar, Lateral, Voiced),
            Phoneme::consonant("ɹ", "R", Alveolar, Approximant, Voiced),
            Phoneme::consonant("w", "W", Bilabial, Approximant, Voiced),
            Phoneme::consonant("j", "Y", Palatal, Approximant, Voiced),
            // === VOWELS ===
            // High vowels
            Phoneme::vowel("i", "IY", Front, High),
            Phoneme::vowel("ɪ", "IH", Front, High),
            Phoneme::vowel("u", "UW", Back, High),
            Phoneme::vowel("ʊ", "UH", Back, High),
            // Mid vowels
            Phoneme::vowel("e", "EY", Front, Mid),
            Phoneme::vowel("ɛ", "EH", Front, Mid),
            Phoneme::vowel("ə", "AH0", Central, Mid),
            Phoneme::vowel("ʌ", "AH", Central, Mid),
            Phoneme::vowel("o", "OW", Back, Mid),
            Phoneme::vowel("ɔ", "AO", Back, Mid),
            // Low vowels
            Phoneme::vowel("æ", "AE", Front, Low),
            Phoneme::vowel("ɑ", "AA", Back, Low),
            // Diphthongs (treated as mid vowels with place indicating start)
            Phoneme::vowel("aɪ", "AY", Front, Low),
            Phoneme::vowel("aʊ", "AW", Back, Low),
            Phoneme::vowel("ɔɪ", "OY", Back, Mid),
            // R-colored vowels
            Phoneme::vowel("ɝ", "ER", Central, Mid),
            Phoneme::vowel("ɚ", "AXR", Central, Mid),
            // === SPECIAL ===
            Phoneme::special("sil", "SIL"), // Silence
            Phoneme::special("spn", "SPN"), // Spoken noise
            Phoneme::special("nsn", "NSN"), // Non-speech noise
        ];

        // Build indices
        let mut ipa_to_idx = HashMap::new();
        let mut arpabet_to_idx = HashMap::new();
        for (i, p) in phonemes.iter().enumerate() {
            ipa_to_idx.insert(p.ipa.clone(), i);
            arpabet_to_idx.insert(p.arpabet.clone(), i);
        }

        // Generate feature hypervectors
        let mut place_hvs = HashMap::new();
        for place in [
            Bilabial,
            Labiodental,
            Dental,
            Alveolar,
            Postalveolar,
            Palatal,
            Velar,
            Glottal,
            Front,
            Central,
            Back,
        ] {
            place_hvs.insert(place, HV16::random(&format!("place:{place:?}")));
        }

        let mut manner_hvs = HashMap::new();
        for manner in [
            Stop,
            Fricative,
            Affricate,
            Nasal,
            Approximant,
            Lateral,
            High,
            Mid,
            Low,
        ] {
            manner_hvs.insert(manner, HV16::random(&format!("manner:{manner:?}")));
        }

        let mut voicing_hvs = HashMap::new();
        voicing_hvs.insert(Voiced, HV16::random("voicing:voiced"));
        voicing_hvs.insert(Voiceless, HV16::random("voicing:voiceless"));

        Self {
            phonemes,
            ipa_to_idx,
            arpabet_to_idx,
            place_hvs,
            manner_hvs,
            voicing_hvs,
        }
    }

    /// Get phoneme by IPA symbol
    pub fn get_ipa(&self, ipa: &str) -> Option<&Phoneme> {
        self.ipa_to_idx.get(ipa).map(|&i| &self.phonemes[i])
    }

    /// Get phoneme by ARPAbet symbol
    pub fn get_arpabet(&self, arpabet: &str) -> Option<&Phoneme> {
        // Strip stress markers (0, 1, 2)
        let clean = arpabet.trim_end_matches(|c: char| c.is_ascii_digit());
        self.arpabet_to_idx.get(clean).map(|&i| &self.phonemes[i])
    }

    /// Get all phonemes
    pub fn all(&self) -> &[Phoneme] {
        &self.phonemes
    }

    /// Number of phonemes
    pub fn len(&self) -> usize {
        self.phonemes.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.phonemes.is_empty()
    }

    /// Encode articulatory features as hypervector
    pub fn encode_features(&self, phoneme: &Phoneme) -> HV16 {
        let place_hv = self
            .place_hvs
            .get(&phoneme.place)
            .cloned()
            .unwrap_or_else(HV16::zero);
        let manner_hv = self
            .manner_hvs
            .get(&phoneme.manner)
            .cloned()
            .unwrap_or_else(HV16::zero);
        let voicing_hv = self
            .voicing_hvs
            .get(&phoneme.voicing)
            .cloned()
            .unwrap_or_else(HV16::zero);

        bundle(&[place_hv, manner_hv, voicing_hv])
    }

    /// Get mutable phoneme by index
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Phoneme> {
        self.phonemes.get_mut(idx)
    }

    /// Get index by IPA
    pub fn ipa_to_index(&self, ipa: &str) -> Option<usize> {
        self.ipa_to_idx.get(ipa).copied()
    }
}

impl Phoneme {
    fn consonant(ipa: &str, arpabet: &str, place: Place, manner: Manner, voicing: Voicing) -> Self {
        Self {
            ipa: ipa.to_string(),
            arpabet: arpabet.to_string(),
            place,
            manner,
            voicing,
            is_vowel: false,
            prototype: None,
        }
    }

    fn vowel(ipa: &str, arpabet: &str, place: Place, manner: Manner) -> Self {
        Self {
            ipa: ipa.to_string(),
            arpabet: arpabet.to_string(),
            place,
            manner,
            voicing: Voicing::Voiced, // Vowels are always voiced
            is_vowel: true,
            prototype: None,
        }
    }

    fn special(ipa: &str, arpabet: &str) -> Self {
        Self {
            ipa: ipa.to_string(),
            arpabet: arpabet.to_string(),
            place: Place::Glottal,
            manner: Manner::Stop,
            voicing: Voicing::Voiceless,
            is_vowel: false,
            prototype: None,
        }
    }
}

/// Modern Hopfield Network for phoneme retrieval
///
/// Uses exponential attention for O(N) storage capacity
#[derive(Debug, Clone)]
pub struct PhonemeResonator {
    /// Stored pattern prototypes
    prototypes: Vec<(String, HV16)>,
    /// Temperature parameter (higher = sharper retrieval)
    beta: f32,
}

impl Default for PhonemeResonator {
    fn default() -> Self {
        Self::new()
    }
}

impl PhonemeResonator {
    /// Create a new resonator
    pub fn new() -> Self {
        Self {
            prototypes: Vec::new(),
            beta: 8.0, // Default temperature
        }
    }

    /// Create with custom temperature
    pub fn with_beta(beta: f32) -> Self {
        Self {
            prototypes: Vec::new(),
            beta,
        }
    }

    /// Store a prototype pattern
    pub fn store(&mut self, label: &str, pattern: HV16) {
        self.prototypes.push((label.to_string(), pattern));
    }

    /// Query the resonator with a pattern
    ///
    /// Returns (label, similarity) sorted by similarity descending
    pub fn query(&self, pattern: &HV16, top_k: usize) -> Vec<(String, f32)> {
        if self.prototypes.is_empty() {
            return Vec::new();
        }

        // Compute similarities
        let mut results: Vec<(String, f32)> = self
            .prototypes
            .iter()
            .map(|(label, proto)| (label.clone(), pattern.similarity(proto)))
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results.truncate(top_k);
        results
    }

    /// Modern Hopfield retrieval with softmax attention
    ///
    /// Returns a weighted bundle of the closest prototypes
    pub fn retrieve(&self, query: &HV16) -> HV16 {
        if self.prototypes.is_empty() {
            return HV16::zero();
        }

        // Compute attention scores: exp(β * similarity)
        let scores: Vec<f32> = self
            .prototypes
            .iter()
            .map(|(_, proto)| (self.beta * query.similarity(proto)).exp())
            .collect();

        // Softmax normalization
        let sum: f32 = scores.iter().sum();
        if sum < 1e-10 {
            return HV16::zero();
        }

        // Weighted bundle
        let mut acc = BundleAccumulator::new();
        for ((_, proto), score) in self.prototypes.iter().zip(scores.iter()) {
            acc.add_weighted(proto, score / sum);
        }

        acc.finalize()
    }

    /// Number of stored prototypes
    pub fn len(&self) -> usize {
        self.prototypes.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.prototypes.is_empty()
    }

    /// Clear all prototypes
    pub fn clear(&mut self) {
        self.prototypes.clear();
    }
}

/// Phoneme decoder combining resonator with coarticulation constraints
#[derive(Debug)]
pub struct PhonemeDecoder {
    /// Phoneme inventory
    inventory: PhonemeInventory,
    /// Pattern resonator
    resonator: PhonemeResonator,
    /// Previous phoneme (for coarticulation)
    prev_phoneme: Option<String>,
    /// Transition probabilities (bigram model)
    transitions: HashMap<(String, String), f32>,
    /// Minimum confidence to accept a phoneme (below this = SILENCE)
    min_confidence: f32,
    /// Recent phoneme history for grammar rescoring
    recent_history: Vec<String>,
    /// Max history length for grammar context
    history_len: usize,
}

impl PhonemeDecoder {
    /// Create a new decoder
    pub fn new() -> Self {
        Self {
            inventory: PhonemeInventory::new(),
            resonator: PhonemeResonator::new(),
            prev_phoneme: None,
            transitions: HashMap::new(),
            min_confidence: -0.05, // Confidence gate - allow weak matches but block noise
            recent_history: Vec::new(),
            history_len: 10, // Longer history to detect oscillation patterns
        }
    }

    /// Create decoder with custom minimum confidence threshold
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set the minimum confidence threshold
    pub fn set_min_confidence(&mut self, min_confidence: f32) {
        self.min_confidence = min_confidence;
    }

    /// Initialize resonator from trained prototypes
    pub fn load_prototypes(&mut self, prototypes: &[(String, HV16)]) {
        self.resonator.clear();
        for (label, hv) in prototypes {
            self.resonator.store(label, *hv);
        }
    }

    /// Decode an acoustic hypervector to phoneme
    ///
    /// Applies:
    /// 1. Minimum confidence threshold (below = SILENCE)
    /// 2. Coarticulation/transition penalties
    /// 3. Repetition penalty to prevent mode collapse
    pub fn decode(&mut self, acoustic_hv: &HV16) -> Option<(String, f32)> {
        let candidates = self.resonator.query(acoustic_hv, 5);
        if candidates.is_empty() {
            return None;
        }

        // CONFIDENCE GATE: If best similarity is below threshold, output SILENCE
        // This prevents weak prototype matches from dominating
        let best_raw_sim = candidates.first().map(|(_, s)| *s).unwrap_or(0.0);
        if best_raw_sim < self.min_confidence {
            self.update_history("SIL".to_string());
            return Some(("SIL".to_string(), best_raw_sim));
        }

        // Apply coarticulation penalty and repetition penalty
        let scored: Vec<(String, f32)> = candidates
            .iter()
            .map(|(label, sim)| {
                // Transition score from bigram model
                let transition_score = self
                    .prev_phoneme
                    .as_ref()
                    .and_then(|prev| self.transitions.get(&(prev.clone(), label.clone())))
                    .copied()
                    .unwrap_or(1.0);

                // REPETITION PENALTY: Penalize if this phoneme appears too often in recent history
                // This helps break mode collapse (e.g., "W AY1 W AY1 W AY1...")
                let repetition_penalty = self.compute_repetition_penalty(label);

                (label.clone(), sim * transition_score * repetition_penalty)
            })
            .collect();

        // Find best after reranking
        let best = scored
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        self.update_history(best.0.clone());
        self.prev_phoneme = Some(best.0.clone());
        Some(best.clone())
    }

    /// Compute repetition penalty based on recent history
    /// Returns 1.0 for no penalty, <1.0 to penalize repetition
    /// Detects both simple repetition (A A A) and oscillating patterns (A B A B)
    fn compute_repetition_penalty(&self, phoneme: &str) -> f32 {
        if self.recent_history.is_empty() {
            return 1.0;
        }

        let len = self.recent_history.len();

        // Check 1: Simple repetition - count occurrences in recent history
        let simple_count = self
            .recent_history
            .iter()
            .filter(|p| p.as_str() == phoneme)
            .count();

        // Check 2: Oscillating pattern detection (A B A B)
        // If the last phoneme was different but the one before was the same, we're oscillating
        let oscillation_penalty = if len >= 2 {
            let prev = &self.recent_history[len - 1];
            let prev_prev = &self.recent_history[len - 2];

            // Pattern: prev_prev == phoneme AND prev != phoneme means oscillation
            if prev_prev.as_str() == phoneme && prev.as_str() != phoneme {
                // Check if this pattern repeats further back
                let mut oscillation_count = 1;
                let mut i = len.saturating_sub(4);
                while i + 1 < len - 2 {
                    if self.recent_history[i].as_str() == phoneme
                        && self.recent_history[i + 1].as_str() == prev.as_str()
                    {
                        oscillation_count += 1;
                    }
                    if i >= 2 {
                        i -= 2;
                    } else {
                        break;
                    }
                }
                // Heavy penalty for oscillation: 0.3 for 1 cycle, 0.1 for 2+ cycles
                if oscillation_count >= 2 { 0.1 } else { 0.3 }
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Combine penalties: take the minimum (harshest) penalty
        let simple_penalty: f32 = match simple_count {
            0 => 1.0,
            1 => 0.85,
            2 => 0.6,
            _ => 0.3,
        };

        simple_penalty.min(oscillation_penalty)
    }

    /// Update the recent phoneme history
    fn update_history(&mut self, phoneme: String) {
        self.recent_history.push(phoneme);
        if self.recent_history.len() > self.history_len {
            self.recent_history.remove(0);
        }
    }

    /// Get all similarity scores for an acoustic HV (for temporal decoding)
    ///
    /// Returns all phoneme similarities sorted by score (highest first)
    pub fn decode_all_scores(&self, acoustic_hv: &HV16) -> Vec<(String, f32)> {
        self.resonator.query(acoustic_hv, self.resonator.len())
    }

    /// Set transition probability
    pub fn set_transition(&mut self, from: &str, to: &str, prob: f32) {
        self.transitions
            .insert((from.to_string(), to.to_string()), prob);
    }

    /// Reset decoder state
    pub fn reset(&mut self) {
        self.prev_phoneme = None;
        self.recent_history.clear();
    }

    /// Get inventory reference
    pub fn inventory(&self) -> &PhonemeInventory {
        &self.inventory
    }

    /// Get mutable inventory reference
    pub fn inventory_mut(&mut self) -> &mut PhonemeInventory {
        &mut self.inventory
    }
}

impl Default for PhonemeDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TEMPORAL DECODER (Schmitt Trigger + Run-Length Encoding)
// ============================================================================

use std::collections::VecDeque;

/// Configuration for temporal decoding
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Sliding window size for smoothing (frames)
    pub window_size: usize,
    /// Minimum confidence to accept a phoneme
    pub confidence_threshold: f32,
    /// Minimum duration to emit a phoneme (frames)
    pub min_duration_frames: usize,
    /// Number of consecutive frames needed to switch phoneme
    pub stability_frames: usize,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            window_size: 5,            // 50ms smoothing window
            confidence_threshold: 0.3, // Minimum similarity to accept
            min_duration_frames: 3,    // 30ms minimum phoneme duration
            stability_frames: 2,       // 20ms to confirm transition
        }
    }
}

/// Temporal decoder with hysteresis for stable phoneme output
///
/// Implements a Schmitt Trigger approach to prevent flickering between
/// phoneme states, followed by run-length encoding to collapse repeated
/// phonemes into single emissions.
pub struct TemporalDecoder {
    config: TemporalConfig,

    /// Sliding window of similarity scores for smoothing
    window: VecDeque<Vec<(String, f32)>>,

    /// Current "locked" phoneme state
    current_phoneme: Option<String>,
    /// Duration of current phoneme (frames)
    current_duration: usize,
    /// Confidence of current phoneme
    current_confidence: f32,

    /// Candidate for transition (hysteresis)
    candidate_phoneme: Option<String>,
    /// Consecutive frames candidate has been leading
    candidate_streak: usize,

    /// Output buffer of emitted phonemes
    emitted: Vec<(String, f32, usize)>, // (phoneme, confidence, duration)
}

impl TemporalDecoder {
    /// Create a new temporal decoder with default config
    pub fn new() -> Self {
        Self::with_config(TemporalConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: TemporalConfig) -> Self {
        Self {
            config,
            window: VecDeque::new(),
            current_phoneme: None,
            current_duration: 0,
            current_confidence: 0.0,
            candidate_phoneme: None,
            candidate_streak: 0,
            emitted: Vec::new(),
        }
    }

    /// Reset decoder state for new utterance
    pub fn reset(&mut self) {
        self.window.clear();
        self.current_phoneme = None;
        self.current_duration = 0;
        self.current_confidence = 0.0;
        self.candidate_phoneme = None;
        self.candidate_streak = 0;
        self.emitted.clear();
    }

    /// Process a single frame of similarity scores
    ///
    /// # Arguments
    /// * `similarities` - Sorted list of (phoneme, similarity) pairs, highest first
    pub fn process_frame(&mut self, similarities: &[(String, f32)]) {
        // 1. Add to sliding window
        self.window.push_back(similarities.to_vec());
        if self.window.len() > self.config.window_size {
            self.window.pop_front();
        }

        // 2. Compute smoothed scores across window
        let smoothed = self.compute_smoothed_scores();

        // 3. Find winner with sufficient confidence
        let winner = smoothed
            .into_iter()
            .find(|(_, score)| *score >= self.config.confidence_threshold);

        if let Some((winner_label, winner_score)) = winner {
            self.update_state(&winner_label, winner_score);
        } else {
            // No confident match - continue current state
            if self.current_phoneme.is_some() {
                self.current_duration += 1;
            }
        }
    }

    /// Compute smoothed scores by averaging across the window
    fn compute_smoothed_scores(&self) -> Vec<(String, f32)> {
        if self.window.is_empty() {
            return Vec::new();
        }

        // Accumulate scores for each phoneme
        let mut score_sums: HashMap<String, (f32, usize)> = HashMap::new();

        for frame_scores in &self.window {
            for (label, score) in frame_scores {
                let entry = score_sums.entry(label.clone()).or_insert((0.0, 0));
                entry.0 += score;
                entry.1 += 1;
            }
        }

        // Compute averages and sort
        let mut averaged: Vec<(String, f32)> = score_sums
            .into_iter()
            .map(|(label, (sum, count))| (label, sum / count as f32))
            .collect();

        averaged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        averaged
    }

    /// Update internal state with hysteresis logic
    fn update_state(&mut self, new_label: &str, new_score: f32) {
        match &self.current_phoneme {
            None => {
                // First phoneme - start tracking
                self.current_phoneme = Some(new_label.to_string());
                self.current_duration = 1;
                self.current_confidence = new_score;
                self.candidate_phoneme = None;
                self.candidate_streak = 0;
            }
            Some(current) if current == new_label => {
                // Same phoneme continuing - reinforce
                self.current_duration += 1;
                // Running average of confidence
                self.current_confidence =
                    (self.current_confidence * (self.current_duration - 1) as f32 + new_score)
                        / self.current_duration as f32;
                // Reset any candidate
                self.candidate_phoneme = None;
                self.candidate_streak = 0;
            }
            Some(_) => {
                // Different phoneme - check hysteresis
                if self.candidate_phoneme.as_deref() == Some(new_label) {
                    // Same candidate as before
                    self.candidate_streak += 1;

                    if self.candidate_streak >= self.config.stability_frames {
                        // Transition confirmed!
                        self.finalize_segment();
                        self.current_phoneme = Some(new_label.to_string());
                        self.current_duration = self.candidate_streak;
                        self.current_confidence = new_score;
                        self.candidate_phoneme = None;
                        self.candidate_streak = 0;
                    }
                } else {
                    // New candidate
                    self.candidate_phoneme = Some(new_label.to_string());
                    self.candidate_streak = 1;
                }
                // Still count duration of current
                self.current_duration += 1;
            }
        }
    }

    /// Finalize current segment and emit if valid
    fn finalize_segment(&mut self) {
        if let Some(phoneme) = &self.current_phoneme {
            // Only emit if meets minimum duration
            if self.current_duration >= self.config.min_duration_frames {
                self.emitted.push((
                    phoneme.clone(),
                    self.current_confidence,
                    self.current_duration,
                ));
            }
        }
    }

    /// Flush any remaining segment at end of utterance
    pub fn flush(&mut self) {
        self.finalize_segment();
    }

    /// Get emitted phonemes
    pub fn get_phonemes(&self) -> Vec<String> {
        self.emitted.iter().map(|(p, _, _)| p.clone()).collect()
    }

    /// Get emitted phonemes with confidence and duration
    pub fn get_phonemes_detailed(&self) -> &[(String, f32, usize)] {
        &self.emitted
    }

    /// Process an entire sequence of frames
    pub fn decode_sequence(&mut self, all_similarities: &[Vec<(String, f32)>]) -> Vec<String> {
        self.reset();

        for frame_sims in all_similarities {
            self.process_frame(frame_sims);
        }

        self.flush();
        self.get_phonemes()
    }
}

impl Default for TemporalDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_size() {
        let inv = PhonemeInventory::new();
        assert_eq!(inv.len(), 44);
    }

    #[test]
    fn test_arpabet_lookup() {
        let inv = PhonemeInventory::new();
        let p = inv.get_arpabet("P").unwrap();
        assert_eq!(p.ipa, "p");
        assert_eq!(p.place, Place::Bilabial);

        // With stress marker
        let vowel = inv.get_arpabet("IY1").unwrap();
        assert_eq!(vowel.ipa, "i");
    }

    #[test]
    fn test_resonator_storage() {
        let mut res = PhonemeResonator::new();

        let p_hv = HV16::random("phoneme:p");
        let b_hv = HV16::random("phoneme:b");

        res.store("p", p_hv);
        res.store("b", b_hv);

        // Query should find the closest
        let results = res.query(&p_hv, 2);
        assert_eq!(results[0].0, "p");
        assert!(results[0].1 > 0.9);
    }

    #[test]
    fn test_feature_encoding() {
        let inv = PhonemeInventory::new();
        let p = inv.get_ipa("p").unwrap();
        let b = inv.get_ipa("b").unwrap();

        let p_features = inv.encode_features(p);
        let b_features = inv.encode_features(b);

        // p and b differ only in voicing, should be similar
        let sim = p_features.similarity(&b_features);
        assert!(sim > 0.3, "p-b similarity = {}", sim);
    }

    #[test]
    fn test_inventory_has_consonants_and_vowels() {
        let inv = PhonemeInventory::new();

        let vowels: Vec<_> = inv.all().iter().filter(|p| p.is_vowel).collect();
        let consonants: Vec<_> = inv.all().iter().filter(|p| !p.is_vowel).collect();

        assert!(!vowels.is_empty(), "should have vowels");
        assert!(!consonants.is_empty(), "should have consonants");
        assert_eq!(vowels.len() + consonants.len(), inv.len());
    }

    #[test]
    fn test_inventory_ipa_lookup() {
        let inv = PhonemeInventory::new();

        let s = inv.get_ipa("s").unwrap();
        assert_eq!(s.arpabet, "S");
        assert_eq!(s.place, Place::Alveolar);
        assert_eq!(s.manner, Manner::Fricative);
        assert_eq!(s.voicing, Voicing::Voiceless);
        assert!(!s.is_vowel);
    }

    #[test]
    fn test_inventory_vowel_always_voiced() {
        let inv = PhonemeInventory::new();
        for p in inv.all() {
            if p.is_vowel {
                assert_eq!(
                    p.voicing,
                    Voicing::Voiced,
                    "vowel {} should be voiced",
                    p.ipa
                );
            }
        }
    }

    #[test]
    fn test_inventory_unknown_ipa() {
        let inv = PhonemeInventory::new();
        assert!(inv.get_ipa("xyz").is_none());
    }

    #[test]
    fn test_arpabet_stress_stripping() {
        let inv = PhonemeInventory::new();

        // Stress marker 0 — strips digit, looks up "AH" which maps to "ʌ"
        let ah0 = inv.get_arpabet("AH0").unwrap();
        assert_eq!(ah0.ipa, "ʌ");

        // Stress marker 2
        let iy2 = inv.get_arpabet("IY2").unwrap();
        assert_eq!(iy2.ipa, "i");
    }

    #[test]
    fn test_resonator_empty_query() {
        let res = PhonemeResonator::new();
        let results = res.query(&HV16::random("test"), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_resonator_with_beta() {
        let res = PhonemeResonator::with_beta(16.0);
        assert_eq!(res.len(), 0);
        assert!(res.is_empty());
    }

    #[test]
    fn test_resonator_retrieve_empty() {
        let res = PhonemeResonator::new();
        let hv = res.retrieve(&HV16::random("query"));
        // Should return zero HV
        let popcount: u32 = hv.words.iter().map(|w| w.count_ones()).sum();
        assert_eq!(popcount, 0);
    }

    #[test]
    fn test_resonator_clear() {
        let mut res = PhonemeResonator::new();
        res.store("A", HV16::random("A"));
        res.store("B", HV16::random("B"));
        assert_eq!(res.len(), 2);

        res.clear();
        assert_eq!(res.len(), 0);
        assert!(res.is_empty());
    }

    #[test]
    fn test_decoder_default_and_reset() {
        let mut decoder = PhonemeDecoder::new();
        assert!(decoder.prev_phoneme.is_none());

        // Load some prototypes and decode
        let protos = vec![
            ("AA".to_string(), HV16::random("AA")),
            ("IY".to_string(), HV16::random("IY")),
        ];
        decoder.load_prototypes(&protos);

        // Decode should produce a result
        let result = decoder.decode(&HV16::random("AA"));
        assert!(result.is_some());

        // Reset should clear state
        decoder.reset();
        assert!(decoder.prev_phoneme.is_none());
    }

    #[test]
    fn test_decoder_min_confidence() {
        let decoder = PhonemeDecoder::new().with_min_confidence(0.5);
        assert!((decoder.min_confidence - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_no_history() {
        let decoder = PhonemeDecoder::new();
        let penalty = decoder.compute_repetition_penalty("AA");
        assert!(
            (penalty - 1.0).abs() < 1e-6,
            "no history should give 1.0 penalty"
        );
    }

    #[test]
    fn test_repetition_penalty_single_occurrence() {
        let mut decoder = PhonemeDecoder::new();
        decoder.update_history("AA".to_string());

        let penalty = decoder.compute_repetition_penalty("AA");
        assert!(
            (penalty - 0.85).abs() < 1e-6,
            "single occurrence should give 0.85, got {penalty}"
        );
    }

    #[test]
    fn test_repetition_penalty_multiple_occurrences() {
        let mut decoder = PhonemeDecoder::new();
        decoder.update_history("AA".to_string());
        decoder.update_history("AA".to_string());

        let penalty = decoder.compute_repetition_penalty("AA");
        assert!(
            (penalty - 0.6).abs() < 1e-6,
            "two occurrences should give 0.6, got {penalty}"
        );
    }

    #[test]
    fn test_repetition_penalty_oscillation() {
        let mut decoder = PhonemeDecoder::new();
        // Create oscillating pattern: AA BB AA
        decoder.update_history("AA".to_string());
        decoder.update_history("BB".to_string());
        decoder.update_history("AA".to_string());
        decoder.update_history("BB".to_string());

        // Next "AA" should trigger oscillation penalty
        let penalty = decoder.compute_repetition_penalty("AA");
        assert!(
            penalty < 0.85,
            "oscillation should give harsh penalty, got {penalty}"
        );
    }

    #[test]
    fn test_temporal_config_default() {
        let config = TemporalConfig::default();
        assert_eq!(config.window_size, 5);
        assert_eq!(config.min_duration_frames, 3);
        assert_eq!(config.stability_frames, 2);
    }

    #[test]
    fn test_temporal_decoder_empty_input() {
        let mut decoder = TemporalDecoder::new();
        let result = decoder.decode_sequence(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_temporal_decoder_stable_sequence() {
        let mut decoder = TemporalDecoder::with_config(TemporalConfig {
            window_size: 1,
            confidence_threshold: 0.1,
            min_duration_frames: 2,
            stability_frames: 2,
        });

        // Feed 10 frames of "AA" then 10 frames of "IY"
        let mut sims = Vec::new();
        for _ in 0..10 {
            sims.push(vec![("AA".to_string(), 0.8), ("IY".to_string(), 0.2)]);
        }
        for _ in 0..10 {
            sims.push(vec![("IY".to_string(), 0.8), ("AA".to_string(), 0.2)]);
        }

        let phonemes = decoder.decode_sequence(&sims);

        // Should emit at least AA and IY
        assert!(
            phonemes.contains(&"AA".to_string()),
            "should contain AA: {:?}",
            phonemes
        );
        assert!(
            phonemes.contains(&"IY".to_string()),
            "should contain IY: {:?}",
            phonemes
        );
    }

    #[test]
    fn test_temporal_decoder_reset() {
        let mut decoder = TemporalDecoder::new();

        // Process some frames
        decoder.process_frame(&[("AA".to_string(), 0.8)]);
        decoder.process_frame(&[("AA".to_string(), 0.8)]);

        decoder.reset();
        assert!(decoder.current_phoneme.is_none());
        assert_eq!(decoder.current_duration, 0);
        assert!(decoder.get_phonemes().is_empty());
    }

    #[test]
    fn test_temporal_decoder_below_confidence() {
        let mut decoder = TemporalDecoder::with_config(TemporalConfig {
            confidence_threshold: 0.9,
            min_duration_frames: 1,
            ..Default::default()
        });

        // All scores below threshold
        let sims: Vec<Vec<(String, f32)>> = (0..5).map(|_| vec![("AA".to_string(), 0.1)]).collect();

        let phonemes = decoder.decode_sequence(&sims);
        // No phoneme should be emitted since confidence is below threshold
        assert!(
            phonemes.is_empty(),
            "should emit nothing below confidence threshold: {:?}",
            phonemes
        );
    }

    #[test]
    fn test_temporal_decoder_min_duration_filter() {
        let mut decoder = TemporalDecoder::with_config(TemporalConfig {
            window_size: 1,
            confidence_threshold: 0.1,
            min_duration_frames: 5,
            stability_frames: 1,
        });

        // Only 2 frames of "AA" - too short
        let sims = vec![
            vec![("AA".to_string(), 0.8)],
            vec![("AA".to_string(), 0.8)],
            vec![("IY".to_string(), 0.8)],
            vec![("IY".to_string(), 0.8)],
            vec![("IY".to_string(), 0.8)],
            vec![("IY".to_string(), 0.8)],
            vec![("IY".to_string(), 0.8)],
            vec![("IY".to_string(), 0.8)],
        ];

        let phonemes = decoder.decode_sequence(&sims);
        // AA segment is too short (2 < 5), should be filtered
        assert!(
            !phonemes.contains(&"AA".to_string()),
            "short AA segment should be filtered: {:?}",
            phonemes
        );
    }
}
