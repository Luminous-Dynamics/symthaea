// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bootstrap training pipeline
//!
//! Self-supervised training using:
//! - LibriSpeech for audio-transcript pairs
//! - Salience-based forced alignment (no external tools needed)
//! - Prototype accumulation via HDC bundling

use crate::hdc::{BundleAccumulator, HV16};
use crate::lexicon::{CmuDictionary, TextToPhonemes};
use crate::ltc::{LtcCell, LtcConfig};
use crate::phoneme::PhonemeInventory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Bootstrap configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Sample rate (Hz)
    pub sample_rate: u32,
    /// Frame duration (seconds)
    pub frame_duration: f32,
    /// LTC configuration
    pub ltc_config: LtcConfig,
    /// Minimum instances per phoneme
    pub min_instances: usize,
    /// Salience threshold for boundary detection
    pub salience_threshold: f32,
    /// Whether to use silence tokens
    pub use_silence: bool,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_duration: 0.010,
            ltc_config: LtcConfig::default(),
            min_instances: 10,
            salience_threshold: 0.5,
            use_silence: true,
        }
    }
}

/// Trained phoneme prototypes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedPrototypes {
    /// Phoneme label -> prototype hypervector
    pub prototypes: HashMap<String, HV16>,
    /// Per-phoneme instance counts
    pub counts: HashMap<String, usize>,
    /// Training configuration used
    pub config: BootstrapConfig,
    /// Version for compatibility
    pub version: String,
}

impl TrainedPrototypes {
    /// Create empty prototypes
    pub fn new(config: BootstrapConfig) -> Self {
        Self {
            prototypes: HashMap::new(),
            counts: HashMap::new(),
            config,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Save to binary file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Load from file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let proto = serde_json::from_reader(reader)?;
        Ok(proto)
    }

    /// Get prototype for phoneme
    pub fn get(&self, phoneme: &str) -> Option<&HV16> {
        self.prototypes.get(phoneme)
    }

    /// Number of trained phonemes
    pub fn len(&self) -> usize {
        self.prototypes.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.prototypes.is_empty()
    }

    /// Get all prototypes as pairs
    pub fn as_pairs(&self) -> Vec<(String, HV16)> {
        self.prototypes
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Enforce minimum separation between ALL prototype pairs
    ///
    /// This should be called AFTER expanding variants into separate prototypes.
    /// Iterates until all pairs are below the threshold.
    pub fn enforce_separation(&mut self, min_separation: f32) {
        const MAX_ITERATIONS: usize = 100;

        for iteration in 0..MAX_ITERATIONS {
            let labels: Vec<String> = self.prototypes.keys().cloned().collect();
            let mut max_sim_seen: f32 = 0.0;
            let mut pairs_fixed = 0;

            for i in 0..labels.len() {
                for j in (i + 1)..labels.len() {
                    let label_i = &labels[i];
                    let label_j = &labels[j];

                    let proto_i = match self.prototypes.get(label_i) {
                        Some(hv) => *hv,
                        None => continue,
                    };
                    let proto_j = match self.prototypes.get(label_j) {
                        Some(hv) => *hv,
                        None => continue,
                    };

                    let similarity = proto_i.similarity(&proto_j);
                    max_sim_seen = max_sim_seen.max(similarity);

                    // If too similar, push apart
                    if similarity > min_separation {
                        pairs_fixed += 1;
                        // Use STRONGER repulsion to converge faster
                        let repulsion_strength = (similarity - min_separation) * 0.8;

                        // Use label hashes PLUS iteration number as seeds
                        // This ensures patterns change each iteration (no permanent collisions)
                        use std::hash::{Hash, Hasher};
                        let mut hasher_i = std::collections::hash_map::DefaultHasher::new();
                        label_i.hash(&mut hasher_i);
                        iteration.hash(&mut hasher_i);
                        let seed_i = hasher_i.finish();

                        let mut hasher_j = std::collections::hash_map::DefaultHasher::new();
                        label_j.hash(&mut hasher_j);
                        iteration.hash(&mut hasher_j);
                        let seed_j = hasher_j.finish();

                        // Apply repulsion with different seeds
                        let new_i = contrastive_repel_hv_with_seed(
                            &proto_i,
                            &proto_j,
                            repulsion_strength,
                            seed_i,
                        );
                        let new_j = contrastive_repel_hv_with_seed(
                            &proto_j,
                            &proto_i,
                            repulsion_strength,
                            seed_j,
                        );

                        self.prototypes.insert(label_i.clone(), new_i);
                        self.prototypes.insert(label_j.clone(), new_j);
                    }
                }
            }

            if iteration == 0 || iteration % 10 == 0 || pairs_fixed == 0 {
                eprintln!(
                    "  Separation iter {iteration}: {pairs_fixed} pairs above {min_separation:.2}, max_sim={max_sim_seen:.4}"
                );
            }

            // Debug: if still identical pairs, print which ones
            if iteration == 0 && max_sim_seen > 0.99 {
                let labels: Vec<String> = self.prototypes.keys().cloned().collect();
                for i in 0..labels.len() {
                    for j in (i + 1)..labels.len() {
                        let Some(proto_i) = self.prototypes.get(&labels[i]) else {
                            continue;
                        };
                        let Some(proto_j) = self.prototypes.get(&labels[j]) else {
                            continue;
                        };
                        let sim = proto_i.similarity(proto_j);
                        if sim > 0.99 {
                            eprintln!(
                                "    IDENTICAL: {} vs {} (sim={:.4})",
                                labels[i], labels[j], sim
                            );
                        }
                    }
                }
            }

            if max_sim_seen <= min_separation {
                eprintln!("  Separation complete after {} iterations", iteration + 1);
                break;
            }
        }
    }
}

/// Contrastive repulsion for HV16 directly (for TrainedPrototypes)
///
/// Uses a seed to ensure different prototypes get different bits flipped,
/// even when they are identical. This prevents the "both stay identical" problem.
fn contrastive_repel_hv_with_seed(proto: &HV16, negative: &HV16, strength: f32, seed: u64) -> HV16 {
    use crate::hdc::HDC_WORDS;

    let mut result_words = proto.words;
    let bits_to_flip = (2048.0 * strength * 0.75) as usize;
    let mut flipped = 0;

    // Use blake3 hash of seed to generate a pseudo-random mask
    // This ensures even similar seeds produce very different patterns
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    let hash_result = hasher.finalize();
    let hash_bytes = hash_result.as_bytes();

    for word_idx in 0..HDC_WORDS {
        let proto_word = proto.words[word_idx];
        let neg_word = negative.words[word_idx];
        let agreement = !(proto_word ^ neg_word);

        // Use hash byte to determine which agreeing bits to flip
        let hash_byte = hash_bytes[word_idx % 32] as usize;

        for bit in 0usize..128 {
            if flipped >= bits_to_flip {
                break;
            }
            let mask = 1u128 << bit;
            if (agreement & mask) != 0 {
                // Use hash to select bits (more random than simple modulo)
                if (bit.wrapping_add(word_idx).wrapping_add(hash_byte)) % 3 == 0 {
                    result_words[word_idx] ^= mask;
                    flipped += 1;
                }
            }
        }
        if flipped >= bits_to_flip {
            break;
        }
    }

    HV16 {
        words: result_words,
    }
}

/// Adaptive prototype with allophone clustering
///
/// Prevents "Grey Goo" collapse by maintaining distinct variants
/// instead of naively bundling all instances together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePrototype {
    /// Phoneme label (e.g., "T")
    pub label: String,
    /// Distinct allophonic variants
    pub variants: Vec<HV16>,
    /// Instance count per variant
    pub variant_counts: Vec<usize>,
    /// Similarity threshold for merging (0.0 to 1.0)
    pub threshold: f32,
    /// Maximum number of variants to maintain
    pub max_variants: usize,
}

impl AdaptivePrototype {
    /// Create a new adaptive prototype
    pub fn new(label: &str, threshold: f32, max_variants: usize) -> Self {
        Self {
            label: label.to_string(),
            variants: Vec::new(),
            variant_counts: Vec::new(),
            threshold,
            max_variants,
        }
    }

    /// Create with default parameters (threshold=0.75, max_variants=5)
    pub fn with_defaults(label: &str) -> Self {
        Self::new(label, 0.75, 5)
    }

    /// Add a new observation, either merging or spawning a variant
    ///
    /// Uses CRYSTALLIZATION to prevent prototype collapse:
    /// - Prototypes "freeze" after enough observations
    /// - This prevents drift toward global centroid
    pub fn train(&mut self, observation: &HV16) {
        // CRYSTALLIZATION THRESHOLD: Stop updating after this many observations
        // This prevents "grey goo" collapse where all prototypes drift to the mean
        const CRYSTALLIZE_AFTER: usize = 50;

        // 1. Find the best matching variant
        let mut best_match_idx = None;
        let mut best_sim = -2.0f32; // Similarity range is [-1, 1]

        for (i, variant) in self.variants.iter().enumerate() {
            let sim = variant.similarity(observation);
            if sim > best_sim {
                best_sim = sim;
                best_match_idx = Some(i);
            }
        }

        // 2. Decision: Merge or Spawn
        if let Some(idx) = best_match_idx
            && best_sim > self.threshold
        {
            let count = self.variant_counts[idx];

            // CRYSTALLIZATION: Stop updating after enough samples
            // The prototype has "learned enough" and should not drift further
            if count >= CRYSTALLIZE_AFTER {
                self.variant_counts[idx] += 1; // Still count for statistics
                return; // But don't modify the prototype!
            }

            // BOUNDED WEIGHT: Cap weight to prevent early samples from
            // being completely overwritten
            let weight = (1.0 / (count + 1) as f32).max(0.05);

            // Weighted combination: new_variant = (1-w)*old + w*new
            self.variants[idx] = weighted_blend(&self.variants[idx], observation, weight);
            self.variant_counts[idx] += 1;
            return;
        }

        // NO MATCH: Attempt to spawn new variant
        if self.variants.len() < self.max_variants {
            // Spawn new allophone variant
            self.variants.push(*observation);
            self.variant_counts.push(1);
        } else {
            // At capacity: merge into closest variant (but respect crystallization)
            if let Some(idx) = best_match_idx {
                let count = self.variant_counts[idx];

                if count >= CRYSTALLIZE_AFTER {
                    self.variant_counts[idx] += 1;
                    return;
                }

                let weight = (1.0 / (count + 1) as f32).max(0.05);
                self.variants[idx] = weighted_blend(&self.variants[idx], observation, weight);
                self.variant_counts[idx] += 1;
            }
        }
    }

    /// Get the primary prototype (most common variant)
    pub fn get_primary(&self) -> Option<HV16> {
        self.variant_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map(|(idx, _)| self.variants[idx])
    }

    /// Get centroid of all variants (for backward compatibility)
    pub fn get_centroid(&self) -> HV16 {
        if self.variants.is_empty() {
            return HV16::zero();
        }
        crate::hdc::bundle(&self.variants)
    }

    /// Get all variants as (label_N, HV16) pairs
    pub fn as_variant_pairs(&self) -> Vec<(String, HV16, usize)> {
        self.variants
            .iter()
            .zip(self.variant_counts.iter())
            .enumerate()
            .map(|(i, (hv, count))| (format!("{}_{}", self.label, i), *hv, *count))
            .collect()
    }

    /// Total instance count across all variants
    pub fn total_count(&self) -> usize {
        self.variant_counts.iter().sum()
    }

    /// Number of variants discovered
    pub fn num_variants(&self) -> usize {
        self.variants.len()
    }
}

/// Weighted blend of two hypervectors
///
/// Uses probabilistic bit selection: each bit is set based on
/// weighted probability from both vectors.
fn weighted_blend(base: &HV16, new: &HV16, new_weight: f32) -> HV16 {
    use crate::hdc::HDC_WORDS;

    let mut result_words = [0u128; HDC_WORDS];
    let base_weight = 1.0 - new_weight;

    // Use deterministic blending based on bit counts
    for i in 0..HDC_WORDS {
        let base_word = base.words[i];
        let new_word = new.words[i];

        // For each bit position, decide based on weighted vote
        let mut result_word = 0u128;
        for bit in 0..128 {
            let base_bit = ((base_word >> bit) & 1) as f32;
            let new_bit = ((new_word >> bit) & 1) as f32;

            // Weighted vote
            let vote = base_weight * base_bit + new_weight * new_bit;

            // Threshold at 0.5 with balanced tiebreaker
            // When vote == 0.5 exactly, use bit position parity to break ties
            // This ensures 50% of ties go to 1 and 50% go to 0
            let tie_goes_to_one = (bit % 2) == 0;
            if vote > 0.5 || (vote == 0.5 && tie_goes_to_one) {
                result_word |= 1u128 << bit;
            }
        }
        result_words[i] = result_word;
    }

    HV16 {
        words: result_words,
    }
}

/// Contrastive repulsion: push prototype away from negative example
///
/// Flips bits where `proto` and `negative` AGREE, proportional to strength.
/// This increases the Hamming distance between them.
fn contrastive_repel(proto: &HV16, negative: &HV16, strength: f32) -> HV16 {
    use crate::hdc::HDC_WORDS;

    let mut result_words = proto.words;

    // How many bits to flip - use STRONGER repulsion (up to 75% of bits)
    let bits_to_flip = (2048.0 * strength * 0.75) as usize;
    let mut flipped = 0;

    // Use a deterministic pattern based on the proto content
    for word_idx in 0..HDC_WORDS {
        let proto_word = proto.words[word_idx];
        let neg_word = negative.words[word_idx];

        // Find bits where they agree
        let agreement = !(proto_word ^ neg_word); // 1 where both same

        // Flip some agreeing bits in proto to increase distance
        // Use EVERY OTHER agreeing bit (more aggressive than every 7th)
        for bit in 0..128 {
            if flipped >= bits_to_flip {
                break;
            }
            let mask = 1u128 << bit;
            if (agreement & mask) != 0 {
                // This bit agrees - flip it in proto to push apart
                // Use a deterministic selection pattern (every 3rd agreeing bit)
                if (bit + word_idx * 3) % 3 == 0 {
                    result_words[word_idx] ^= mask;
                    flipped += 1;
                }
            }
        }
        if flipped >= bits_to_flip {
            break;
        }
    }

    HV16 {
        words: result_words,
    }
}

/// Collection of adaptive prototypes for all phonemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePrototypeSet {
    /// Phoneme label -> AdaptivePrototype
    pub prototypes: HashMap<String, AdaptivePrototype>,
    /// Configuration
    pub threshold: f32,
    pub max_variants: usize,
}

impl AdaptivePrototypeSet {
    /// Create a new set with given parameters
    pub fn new(threshold: f32, max_variants: usize) -> Self {
        Self {
            prototypes: HashMap::new(),
            threshold,
            max_variants,
        }
    }

    /// Create with default parameters
    pub fn with_defaults() -> Self {
        Self::new(0.75, 5)
    }

    /// Train on a phoneme observation with CONTRASTIVE updates
    ///
    /// 1. Pull the target phoneme toward the observation (attraction)
    /// 2. Push similar phonemes away from the observation (repulsion)
    ///
    /// This prevents "grey goo" collapse where all prototypes drift together.
    pub fn train(&mut self, phoneme: &str, observation: &HV16) {
        // First, train the positive prototype (attraction)
        let proto = self
            .prototypes
            .entry(phoneme.to_string())
            .or_insert_with(|| AdaptivePrototype::new(phoneme, self.threshold, self.max_variants));
        proto.train(observation);
    }

    /// Train with contrastive repulsion - call after training multiple phonemes
    ///
    /// For each prototype pair that's too similar, push them apart.
    /// This maintains separation and prevents collapse.
    pub fn enforce_separation(&mut self, min_separation: f32) {
        // ITERATE until all pairs are below threshold (up to max iterations)
        const MAX_ITERATIONS: usize = 20;

        for iteration in 0..MAX_ITERATIONS {
            let labels: Vec<String> = self.prototypes.keys().cloned().collect();
            let mut max_sim_seen: f32 = 0.0;
            let mut pairs_fixed = 0;

            for i in 0..labels.len() {
                for j in (i + 1)..labels.len() {
                    let label_i = &labels[i];
                    let label_j = &labels[j];

                    // Get primary prototypes
                    let proto_i = match self.prototypes.get(label_i) {
                        Some(p) => match p.get_primary() {
                            Some(hv) => hv,
                            None => continue,
                        },
                        None => continue,
                    };
                    let proto_j = match self.prototypes.get(label_j) {
                        Some(p) => match p.get_primary() {
                            Some(hv) => hv,
                            None => continue,
                        },
                        None => continue,
                    };

                    let similarity = proto_i.similarity(&proto_j);
                    max_sim_seen = max_sim_seen.max(similarity);

                    // If too similar, push apart by flipping disagreeing bits
                    if similarity > min_separation {
                        pairs_fixed += 1;
                        // Use STRONGER repulsion based on how far above threshold
                        // No cap - push as hard as needed
                        let repulsion_strength = (similarity - min_separation) * 0.6;

                        if let Some(p) = self.prototypes.get_mut(label_i)
                            && let Some(primary) = p.variants.first_mut()
                        {
                            *primary = contrastive_repel(primary, &proto_j, repulsion_strength);
                        }
                        if let Some(p) = self.prototypes.get_mut(label_j)
                            && let Some(primary) = p.variants.first_mut()
                        {
                            *primary = contrastive_repel(primary, &proto_i, repulsion_strength);
                        }
                    }
                }
            }

            // Print progress
            if iteration == 0 || pairs_fixed > 0 {
                eprintln!(
                    "  Separation iter {iteration}: {pairs_fixed} pairs above {min_separation:.2}, max_sim={max_sim_seen:.4}"
                );
            }

            // Stop if all pairs are below threshold
            if max_sim_seen <= min_separation {
                eprintln!("  Separation complete after {} iterations", iteration + 1);
                break;
            }
        }
    }

    /// Get all phonemes
    pub fn phonemes(&self) -> Vec<&str> {
        self.prototypes.keys().map(|s| s.as_str()).collect()
    }

    /// Convert to TrainedPrototypes (using PRIMARY variant, not centroid!)
    ///
    /// CRITICAL FIX: Using get_primary() instead of get_centroid() prevents
    /// the "double averaging" problem where:
    /// 1. Training averages observations into variants (weighted_blend)
    /// 2. Finalization was averaging all variants together (bundle)
    ///
    /// This second averaging step collapsed the carefully-separated variants
    /// back into grey goo. By using the PRIMARY (most common) variant,
    /// we preserve the distinct character of the most representative allophone.
    pub fn to_trained_prototypes(&self, config: BootstrapConfig) -> TrainedPrototypes {
        let mut prototypes = HashMap::new();
        let mut counts = HashMap::new();

        for (label, adaptive) in &self.prototypes {
            if adaptive.total_count() >= config.min_instances {
                // Use PRIMARY variant (most instances) instead of bundled centroid
                // This preserves the distinct character learned during training
                if let Some(primary) = adaptive.get_primary() {
                    prototypes.insert(label.clone(), primary);
                    counts.insert(label.clone(), adaptive.total_count());
                }
            }
        }

        TrainedPrototypes {
            prototypes,
            counts,
            config,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Convert to expanded prototypes (each variant becomes a prototype)
    pub fn to_expanded_prototypes(&self, config: BootstrapConfig) -> TrainedPrototypes {
        let mut prototypes = HashMap::new();
        let mut counts = HashMap::new();

        for adaptive in self.prototypes.values() {
            for (variant_label, hv, count) in adaptive.as_variant_pairs() {
                if count >= config.min_instances {
                    prototypes.insert(variant_label.clone(), hv);
                    counts.insert(variant_label, count);
                }
            }
        }

        TrainedPrototypes {
            prototypes,
            counts,
            config,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> AdaptiveStats {
        let mut total_phonemes = 0;
        let mut total_variants = 0;
        let mut total_instances = 0;
        let mut max_variants_seen = 0;

        for proto in self.prototypes.values() {
            total_phonemes += 1;
            total_variants += proto.num_variants();
            total_instances += proto.total_count();
            max_variants_seen = max_variants_seen.max(proto.num_variants());
        }

        AdaptiveStats {
            total_phonemes,
            total_variants,
            total_instances,
            avg_variants_per_phoneme: if total_phonemes > 0 {
                total_variants as f32 / total_phonemes as f32
            } else {
                0.0
            },
            max_variants_seen,
        }
    }
}

/// Statistics for adaptive prototype training
#[derive(Debug, Clone)]
pub struct AdaptiveStats {
    /// Number of unique phoneme labels
    pub total_phonemes: usize,
    /// Total allophone variants discovered
    pub total_variants: usize,
    /// Total training instances
    pub total_instances: usize,
    /// Average variants per phoneme
    pub avg_variants_per_phoneme: f32,
    /// Maximum variants for any single phoneme
    pub max_variants_seen: usize,
}

/// Prototype accumulator during training
#[derive(Debug)]
struct PrototypeAccumulator {
    accumulators: HashMap<String, BundleAccumulator>,
    counts: HashMap<String, usize>,
}

impl PrototypeAccumulator {
    fn new() -> Self {
        Self {
            accumulators: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    fn add(&mut self, phoneme: &str, hv: &HV16) {
        self.accumulators
            .entry(phoneme.to_string())
            .or_default()
            .add(hv);

        *self.counts.entry(phoneme.to_string()).or_insert(0) += 1;
    }

    fn finalize(self, min_instances: usize) -> HashMap<String, HV16> {
        self.accumulators
            .into_iter()
            .filter(|(label, _)| self.counts.get(label).copied().unwrap_or(0) >= min_instances)
            .map(|(label, acc)| (label, acc.finalize()))
            .collect()
    }

    fn counts(&self) -> HashMap<String, usize> {
        self.counts.clone()
    }
}

/// Aligned phoneme segment
#[derive(Debug, Clone)]
pub struct AlignedSegment {
    /// Phoneme label
    pub phoneme: String,
    /// Start frame index
    pub start_frame: usize,
    /// End frame index
    pub end_frame: usize,
    /// Confidence score
    pub confidence: f32,
}

/// Salience-based forced aligner
#[derive(Debug)]
pub struct SalienceAligner {
    /// Salience threshold
    threshold: f32,
}

impl SalienceAligner {
    /// Create a new aligner
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Detect salience peaks (potential phoneme boundaries)
    pub fn detect_boundaries(&self, saliences: &[f32]) -> Vec<usize> {
        let mut boundaries = vec![0]; // Start at frame 0

        for i in 1..saliences.len().saturating_sub(1) {
            // Local maximum above threshold
            if saliences[i] > self.threshold
                && saliences[i] > saliences[i - 1]
                && saliences[i] >= saliences[i + 1]
            {
                boundaries.push(i);
            }
        }

        boundaries.push(saliences.len()); // End
        boundaries
    }

    /// Align phoneme sequence to boundaries using DTW
    pub fn align(&self, phonemes: &[String], saliences: &[f32]) -> Vec<AlignedSegment> {
        if phonemes.is_empty() || saliences.is_empty() {
            return Vec::new();
        }

        let boundaries = self.detect_boundaries(saliences);

        if boundaries.len() < 2 {
            return Vec::new();
        }

        // Simple uniform distribution if not enough boundaries
        if boundaries.len() - 1 < phonemes.len() {
            let frames_per_phoneme = saliences.len() / phonemes.len().max(1);
            return phonemes
                .iter()
                .enumerate()
                .map(|(i, p)| AlignedSegment {
                    phoneme: p.clone(),
                    start_frame: i * frames_per_phoneme,
                    end_frame: (i + 1) * frames_per_phoneme,
                    confidence: 0.5,
                })
                .collect();
        }

        // Distribute phonemes across segments
        let num_segments = boundaries.len() - 1;
        let phonemes_per_segment = (phonemes.len() as f32 / num_segments as f32).ceil() as usize;

        let mut result = Vec::new();
        let mut phoneme_idx = 0;

        for seg_idx in 0..num_segments {
            let start = boundaries[seg_idx];
            let end = boundaries[seg_idx + 1];

            // How many phonemes in this segment
            let count = phonemes_per_segment.min(phonemes.len() - phoneme_idx);
            if count == 0 {
                break;
            }

            let frames_per = (end - start) / count.max(1);

            for j in 0..count {
                if phoneme_idx + j >= phonemes.len() {
                    break;
                }

                result.push(AlignedSegment {
                    phoneme: phonemes[phoneme_idx + j].clone(),
                    start_frame: start + j * frames_per,
                    end_frame: start + (j + 1) * frames_per,
                    confidence: 0.7,
                });
            }

            phoneme_idx += count;
        }

        result
    }
}

/// Bootstrap training pipeline
pub struct BootstrapPipeline {
    config: BootstrapConfig,
    ltc: LtcCell,
    text_to_phonemes: TextToPhonemes,
    aligner: SalienceAligner,
    accumulator: PrototypeAccumulator,
    inventory: PhonemeInventory,
}

impl BootstrapPipeline {
    /// Create a new bootstrap pipeline
    pub fn new(config: BootstrapConfig) -> Self {
        let ltc = LtcCell::new(40, config.ltc_config.clone()); // 40 mel bins
        let text_to_phonemes = TextToPhonemes::empty().with_silence(config.use_silence);
        let aligner = SalienceAligner::new(config.salience_threshold);

        Self {
            config,
            ltc,
            text_to_phonemes,
            aligner,
            accumulator: PrototypeAccumulator::new(),
            inventory: PhonemeInventory::new(),
        }
    }

    /// Load CMU dictionary
    pub fn load_dictionary<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        let dict = CmuDictionary::load(path)?;
        self.text_to_phonemes = TextToPhonemes::new(dict).with_silence(self.config.use_silence);
        Ok(())
    }

    /// Set dictionary directly
    pub fn set_dictionary(&mut self, dict: CmuDictionary) {
        self.text_to_phonemes = TextToPhonemes::new(dict).with_silence(self.config.use_silence);
    }

    /// Process a single utterance
    ///
    /// # Arguments
    /// * `audio` - Audio samples (normalized to [-1, 1])
    /// * `transcript` - Text transcript
    ///
    /// # Returns
    /// Number of phoneme instances added
    pub fn process_utterance(&mut self, audio: &[f32], transcript: &str) -> usize {
        // Convert transcript to phonemes
        let phonemes = self.text_to_phonemes.convert(transcript);
        if phonemes.is_empty() {
            return 0;
        }

        // Extract features and compute salience
        let (features, saliences) = self.extract_features(audio);

        if features.is_empty() {
            return 0;
        }

        // Align phonemes to audio
        let aligned = self.aligner.align(&phonemes, &saliences);

        // Accumulate prototypes
        let mut count = 0;
        for segment in aligned {
            if segment.start_frame >= features.len() {
                continue;
            }

            let end = segment.end_frame.min(features.len());
            let segment_features = &features[segment.start_frame..end];

            if !segment_features.is_empty() {
                // Bundle segment into single HV
                let hv = self.features_to_hv(segment_features);
                self.accumulator.add(&segment.phoneme, &hv);
                count += 1;
            }
        }

        count
    }

    /// Extract features from audio
    fn extract_features(&mut self, audio: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let frame_samples = (self.config.sample_rate as f32 * self.config.frame_duration) as usize;
        let hop_samples = frame_samples / 2;

        let mut features = Vec::new();
        let mut saliences = Vec::new();
        let mut prev_state = vec![0.0; self.ltc.hidden_size()];

        self.ltc.reset();

        let mut pos = 0;
        while pos + frame_samples <= audio.len() {
            let frame = &audio[pos..pos + frame_samples];

            // Simple energy-based features (placeholder for mel spectrogram)
            let energy = frame.iter().map(|x| x * x).sum::<f32>().sqrt();
            let zcr = frame
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count() as f32
                / frame_samples as f32;

            // Create feature vector (would be mel spectrogram in production)
            let mut feature = vec![0.0; 40];
            feature[0] = energy;
            feature[1] = zcr;

            // Process through LTC
            let state = self
                .ltc
                .forward(&feature, self.config.frame_duration)
                .to_vec();
            let salience = self.ltc.compute_salience(&prev_state);

            features.push(state.clone());
            saliences.push(salience);
            prev_state = state;

            pos += hop_samples;
        }

        (features, saliences)
    }

    /// Convert feature sequence to hypervector
    fn features_to_hv(&self, features: &[Vec<f32>]) -> HV16 {
        // Simple approach: hash the concatenated features
        let mut data = Vec::new();
        for f in features {
            for &v in f {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }

        // Use BLAKE3 to hash into HV
        let mut hasher = blake3::Hasher::new();
        hasher.update(&data);
        let mut output = [0u8; 256];
        hasher.finalize_xof().fill(&mut output);

        HV16::from_bytes(&output)
    }

    /// Finalize training and return prototypes
    pub fn finalize(self) -> TrainedPrototypes {
        let counts = self.accumulator.counts();
        let prototypes = self.accumulator.finalize(self.config.min_instances);

        TrainedPrototypes {
            prototypes,
            counts,
            config: self.config,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Get current instance counts
    pub fn instance_counts(&self) -> &HashMap<String, usize> {
        &self.accumulator.counts
    }

    /// Get phoneme inventory
    pub fn inventory(&self) -> &PhonemeInventory {
        &self.inventory
    }
}

/// LibriSpeech dataset parser
pub struct LibriSpeechParser {
    root: std::path::PathBuf,
}

impl LibriSpeechParser {
    /// Create parser for LibriSpeech dataset
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// Iterate over utterances
    ///
    /// Yields (audio_path, transcript) pairs
    pub fn utterances(&self) -> impl Iterator<Item = (std::path::PathBuf, String)> + '_ {
        self.find_trans_files()
            .flat_map(|trans_path| self.parse_trans_file(&trans_path))
    }

    fn find_trans_files(&self) -> impl Iterator<Item = std::path::PathBuf> + '_ {
        walkdir::WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "trans.txt")
                    .unwrap_or(false)
            })
            .map(|e| e.path().to_path_buf())
    }

    fn parse_trans_file(&self, path: &Path) -> Vec<(std::path::PathBuf, String)> {
        let dir = path.parent().unwrap_or(Path::new("."));

        std::fs::read_to_string(path)
            .map(|content| {
                content
                    .lines()
                    .filter_map(|line| {
                        let parts: Vec<&str> = line.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            let audio_path = dir.join(format!("{}.flac", parts[0]));
                            Some((audio_path, parts[1].to_string()))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

// Note: walkdir is not a dependency, this is a placeholder for the actual implementation
mod walkdir {
    use std::fs;
    use std::path::{Path, PathBuf};

    pub struct WalkDir {
        root: PathBuf,
    }

    impl WalkDir {
        pub fn new<P: AsRef<Path>>(root: P) -> Self {
            Self {
                root: root.as_ref().to_path_buf(),
            }
        }

        pub fn into_iter(self) -> impl Iterator<Item = Result<DirEntry, std::io::Error>> {
            WalkDirIter::new(self.root)
        }
    }

    pub struct DirEntry {
        path: PathBuf,
    }

    impl DirEntry {
        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    struct WalkDirIter {
        stack: Vec<PathBuf>,
    }

    impl WalkDirIter {
        fn new(root: PathBuf) -> Self {
            Self { stack: vec![root] }
        }
    }

    impl Iterator for WalkDirIter {
        type Item = Result<DirEntry, std::io::Error>;

        fn next(&mut self) -> Option<Self::Item> {
            while let Some(path) = self.stack.pop() {
                if path.is_dir() {
                    if let Ok(entries) = fs::read_dir(&path) {
                        for entry in entries.filter_map(|e| e.ok()) {
                            self.stack.push(entry.path());
                        }
                    }
                    continue;
                }
                return Some(Ok(DirEntry { path }));
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_salience_aligner() {
        let aligner = SalienceAligner::new(0.5);

        // Create test salience with clear peaks
        let saliences = vec![0.1, 0.2, 0.8, 0.2, 0.1, 0.9, 0.3, 0.2];
        let boundaries = aligner.detect_boundaries(&saliences);

        assert!(boundaries.contains(&0));
        assert!(boundaries.contains(&2)); // Peak at 0.8
        assert!(boundaries.contains(&5)); // Peak at 0.9
    }

    #[test]
    fn test_prototype_accumulator() {
        let mut acc = PrototypeAccumulator::new();

        let hv1 = HV16::random("test1");
        let hv2 = HV16::random("test2");

        acc.add("p", &hv1);
        acc.add("p", &hv2);
        acc.add("b", &hv1);

        assert_eq!(acc.counts["p"], 2);
        assert_eq!(acc.counts["b"], 1);

        let protos = acc.finalize(2);
        assert!(protos.contains_key("p"));
        assert!(!protos.contains_key("b")); // Below min_instances
    }

    #[test]
    fn test_trained_prototypes() {
        let config = BootstrapConfig::default();
        let mut protos = TrainedPrototypes::new(config);

        protos.prototypes.insert("p".to_string(), HV16::random("p"));
        protos.counts.insert("p".to_string(), 100);

        assert_eq!(protos.len(), 1);
        assert!(protos.get("p").is_some());
    }
}
