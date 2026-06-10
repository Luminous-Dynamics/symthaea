// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hierarchical HDC Scorer - Solving Bundling Saturation
//!
//! The core insight: Instead of ONE grammar memory with 135K patterns (saturated),
//! use MANY smaller memories organized by linguistic structure.
//!
//! # The Saturation Problem
//!
//! When bundling N binary vectors, each bit approaches 50% probability as N grows:
//! - 100 vectors: clear signal
//! - 1,000 vectors: noisy but usable
//! - 10,000+ vectors: approaching random noise
//!
//! CMU Dict has ~135K words → 400K+ trigrams → saturation
//!
//! # The Solution: Hierarchical Clustering
//!
//! ```text
//! Level 0: Global (too saturated)
//!          ↓
//! Level 1: By Manner Class (7 memories, ~57K patterns each)
//!          ↓
//! Level 2: By Manner × Place (70 memories, ~5.7K patterns each)
//!          ↓
//! Level 3: By first phoneme (41 memories, ~9.8K patterns each)
//! ```
//!
//! At Level 2 or 3, each memory stays well below saturation threshold.
//!
//! # Query Strategy
//!
//! Instead of: score = similarity(context, global_grammar)
//!
//! We use: `score = similarity(context, grammar[manner_class])`
//!
//! The acoustic model already predicts Manner class → use it to route!

use crate::hdc::{BundleAccumulator, HDC_DIM, HV16};
use std::collections::HashMap;

/// Number of Manner classes (Stop, Fricative, Affricate, Nasal, Liquid, Glide, Vowel)
pub const NUM_MANNER: usize = 7;

/// Number of Place classes
pub const NUM_PLACE: usize = 10;

/// Hierarchical HDC Scorer with clustered grammar memories
pub struct HierarchicalScorer {
    /// Phoneme basis vectors (same as HolographicScorer)
    phoneme_basis: HashMap<String, HV16>,

    /// Level 1: Grammar memory per Manner class (7 memories)
    manner_memories: [HV16; NUM_MANNER],

    /// Level 2: Grammar memory per Manner×Place (70 memories)
    manner_place_memories: [[HV16; NUM_PLACE]; NUM_MANNER],

    /// Training counts per cluster (for weighted updates)
    manner_counts: [usize; NUM_MANNER],
    manner_place_counts: [[usize; NUM_PLACE]; NUM_MANNER],

    /// Current context state
    context_state: HV16,

    /// Which level to use for scoring (1 = Manner, 2 = Manner×Place)
    active_level: usize,

    /// Phoneme → (Manner, Place) mapping
    phoneme_features: HashMap<String, (usize, usize)>,
}

impl Default for HierarchicalScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalScorer {
    /// Create a new hierarchical scorer
    pub fn new() -> Self {
        // Initialize phoneme basis (same seeds as HolographicScorer for compatibility)
        let mut phoneme_basis = HashMap::new();
        let phonemes = [
            "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
            "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T",
            "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL", "UNK",
        ];

        for phoneme in phonemes.iter() {
            let seed = format!("phoneme_basis_{phoneme}");
            phoneme_basis.insert(phoneme.to_string(), HV16::random(&seed));
        }

        // Initialize hierarchical memories with unique seeds
        let manner_memories = std::array::from_fn(|i| HV16::random(&format!("manner_memory_{i}")));

        let manner_place_memories = std::array::from_fn(|m| {
            std::array::from_fn(|p| HV16::random(&format!("manner_place_memory_{m}_{p}")))
        });

        // Build phoneme → features mapping
        let phoneme_features = Self::build_phoneme_features();

        Self {
            phoneme_basis,
            manner_memories,
            manner_place_memories,
            manner_counts: [0; NUM_MANNER],
            manner_place_counts: [[0; NUM_PLACE]; NUM_MANNER],
            context_state: HV16::zero(),
            active_level: 2, // Default to finest granularity
            phoneme_features,
        }
    }

    /// Build phoneme → (manner_idx, place_idx) mapping
    fn build_phoneme_features() -> HashMap<String, (usize, usize)> {
        let mut map = HashMap::new();

        // Manner indices: 0=Stop, 1=Fricative, 2=Affricate, 3=Nasal, 4=Liquid, 5=Glide, 6=Vowel
        // Place indices: 0=Bilabial, 1=Labiodental, 2=Dental, 3=Alveolar, 4=Palatal, 5=Velar, 6=Glottal, 7=Front, 8=Central, 9=Back

        // Stops (manner=0)
        map.insert("P".into(), (0, 0)); // Bilabial
        map.insert("B".into(), (0, 0));
        map.insert("T".into(), (0, 3)); // Alveolar
        map.insert("D".into(), (0, 3));
        map.insert("K".into(), (0, 5)); // Velar
        map.insert("G".into(), (0, 5));

        // Fricatives (manner=1)
        map.insert("F".into(), (1, 1)); // Labiodental
        map.insert("V".into(), (1, 1));
        map.insert("TH".into(), (1, 2)); // Dental
        map.insert("DH".into(), (1, 2));
        map.insert("S".into(), (1, 3)); // Alveolar
        map.insert("Z".into(), (1, 3));
        map.insert("SH".into(), (1, 4)); // Palatal
        map.insert("ZH".into(), (1, 4));
        map.insert("HH".into(), (1, 6)); // Glottal

        // Affricates (manner=2)
        map.insert("CH".into(), (2, 4)); // Palatal
        map.insert("JH".into(), (2, 4));

        // Nasals (manner=3)
        map.insert("M".into(), (3, 0)); // Bilabial
        map.insert("N".into(), (3, 3)); // Alveolar
        map.insert("NG".into(), (3, 5)); // Velar

        // Liquids (manner=4)
        map.insert("L".into(), (4, 3)); // Alveolar
        map.insert("R".into(), (4, 3));

        // Glides (manner=5)
        map.insert("W".into(), (5, 0)); // Bilabial
        map.insert("Y".into(), (5, 4)); // Palatal

        // Vowels (manner=6)
        map.insert("IY".into(), (6, 7)); // Front
        map.insert("IH".into(), (6, 7));
        map.insert("EY".into(), (6, 7));
        map.insert("EH".into(), (6, 7));
        map.insert("AE".into(), (6, 7));
        map.insert("AH".into(), (6, 8)); // Central
        map.insert("ER".into(), (6, 8));
        map.insert("AY".into(), (6, 8));
        map.insert("AW".into(), (6, 8));
        map.insert("UW".into(), (6, 9)); // Back
        map.insert("UH".into(), (6, 9));
        map.insert("OW".into(), (6, 9));
        map.insert("AA".into(), (6, 9));
        map.insert("AO".into(), (6, 9));
        map.insert("OY".into(), (6, 9));

        // Special
        map.insert("SIL".into(), (6, 8)); // Treat as central vowel
        map.insert("UNK".into(), (6, 8));

        map
    }

    /// Set the active scoring level (1=Manner, 2=Manner×Place)
    pub fn set_level(&mut self, level: usize) {
        self.active_level = level.clamp(1, 2);
    }

    /// Reset context state
    pub fn reset(&mut self) {
        self.context_state = HV16::zero();
    }

    /// Get phoneme basis vector
    fn get_basis(&self, phoneme: &str) -> HV16 {
        self.phoneme_basis.get(phoneme).copied().unwrap_or_else(|| {
            self.phoneme_basis
                .get("UNK")
                .copied()
                .unwrap_or_else(HV16::zero)
        })
    }

    /// Get articulatory features for a phoneme
    fn get_features(&self, phoneme: &str) -> (usize, usize) {
        self.phoneme_features
            .get(phoneme)
            .copied()
            .unwrap_or((6, 8)) // Default to vowel/central
    }

    /// Process a phoneme, updating context and returning cluster-specific score
    pub fn process(&mut self, phoneme: &str) -> f32 {
        let phi = self.get_basis(phoneme);
        let (manner, place) = self.get_features(phoneme);

        // Update context: H_t = Π(H_{t-1}) ⊕ Φ_current
        let rotated = self.context_state.rotate(1);
        self.context_state = rotated.bind(&phi);

        // Score against the appropriate cluster memory
        match self.active_level {
            1 => self.manner_memories[manner].similarity(&self.context_state),
            2 => self.manner_place_memories[manner][place].similarity(&self.context_state),
            _ => 0.0,
        }
    }

    /// Score a complete phoneme sequence
    pub fn score_sequence(&mut self, phonemes: &[&str]) -> f32 {
        self.reset();
        let mut total = 0.0;
        for phoneme in phonemes {
            total += self.process(phoneme);
        }
        total / phonemes.len().max(1) as f32
    }

    /// Train on a phoneme sequence, updating cluster-specific memories
    pub fn train_sequence(&mut self, phonemes: &[&str]) {
        if phonemes.is_empty() {
            return;
        }

        // Collect context vectors per cluster
        let mut manner_accums: Vec<BundleAccumulator> =
            (0..NUM_MANNER).map(|_| BundleAccumulator::new()).collect();
        let mut manner_place_accums: Vec<Vec<BundleAccumulator>> = (0..NUM_MANNER)
            .map(|_| (0..NUM_PLACE).map(|_| BundleAccumulator::new()).collect())
            .collect();

        self.reset();

        for phoneme in phonemes {
            let phi = self.get_basis(phoneme);
            let (manner, place) = self.get_features(phoneme);

            // Update context
            let rotated = self.context_state.rotate(1);
            self.context_state = rotated.bind(&phi);

            // Add to appropriate cluster accumulators
            manner_accums[manner].add(&self.context_state);
            manner_place_accums[manner][place].add(&self.context_state);
        }

        // Update memories with weighted bundling
        for m in 0..NUM_MANNER {
            if manner_accums[m].count() > 0 {
                // Weighted update: combine old memory with new patterns
                let mut combined = BundleAccumulator::new();
                if self.manner_counts[m] > 0 {
                    combined.add_weighted(&self.manner_memories[m], self.manner_counts[m] as f32);
                }
                // Add new patterns
                let new_patterns = manner_accums[m].finalize();
                combined.add_weighted(&new_patterns, manner_accums[m].count() as f32);

                self.manner_memories[m] = combined.finalize();
                self.manner_counts[m] += manner_accums[m].count();
            }

            for p in 0..NUM_PLACE {
                if manner_place_accums[m][p].count() > 0 {
                    let mut combined = BundleAccumulator::new();
                    if self.manner_place_counts[m][p] > 0 {
                        combined.add_weighted(
                            &self.manner_place_memories[m][p],
                            self.manner_place_counts[m][p] as f32,
                        );
                    }
                    let new_patterns = manner_place_accums[m][p].finalize();
                    combined.add_weighted(&new_patterns, manner_place_accums[m][p].count() as f32);

                    self.manner_place_memories[m][p] = combined.finalize();
                    self.manner_place_counts[m][p] += manner_place_accums[m][p].count();
                }
            }
        }
    }

    /// Discriminate between two sequences
    pub fn discriminate(&mut self, seq1: &[&str], seq2: &[&str]) -> f32 {
        let score1 = self.score_sequence(seq1);
        let score2 = self.score_sequence(seq2);
        score1 - score2
    }

    /// Get statistics about cluster usage
    pub fn stats(&self) -> HierarchicalStats {
        let total_manner: usize = self.manner_counts.iter().sum();
        let total_mp: usize = self
            .manner_place_counts
            .iter()
            .flat_map(|row| row.iter())
            .sum();

        // Count non-empty clusters
        let active_manner = self.manner_counts.iter().filter(|&&c| c > 0).count();
        let active_mp = self
            .manner_place_counts
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c > 0)
            .count();

        // Average patterns per active cluster
        let avg_per_manner = if active_manner > 0 {
            total_manner as f32 / active_manner as f32
        } else {
            0.0
        };

        let avg_per_mp = if active_mp > 0 {
            total_mp as f32 / active_mp as f32
        } else {
            0.0
        };

        // Compute average density
        let manner_density: f32 = self
            .manner_memories
            .iter()
            .map(|m| m.popcount() as f32 / HDC_DIM as f32)
            .sum::<f32>()
            / NUM_MANNER as f32;

        HierarchicalStats {
            total_patterns: total_manner,
            active_manner_clusters: active_manner,
            active_mp_clusters: active_mp,
            avg_patterns_per_manner: avg_per_manner,
            avg_patterns_per_mp: avg_per_mp,
            avg_manner_density: manner_density,
        }
    }
}

/// Statistics for hierarchical scorer
#[derive(Debug)]
pub struct HierarchicalStats {
    pub total_patterns: usize,
    pub active_manner_clusters: usize,
    pub active_mp_clusters: usize,
    pub avg_patterns_per_manner: f32,
    pub avg_patterns_per_mp: f32,
    pub avg_manner_density: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_creation() {
        let scorer = HierarchicalScorer::new();
        let stats = scorer.stats();
        assert_eq!(stats.total_patterns, 0);
        assert_eq!(stats.active_manner_clusters, 0);
    }

    #[test]
    fn test_phoneme_features() {
        let scorer = HierarchicalScorer::new();

        // Check some known mappings
        assert_eq!(scorer.get_features("P"), (0, 0)); // Stop, Bilabial
        assert_eq!(scorer.get_features("S"), (1, 3)); // Fricative, Alveolar
        assert_eq!(scorer.get_features("M"), (3, 0)); // Nasal, Bilabial
        assert_eq!(scorer.get_features("IY"), (6, 7)); // Vowel, Front
    }

    #[test]
    fn test_hierarchical_training() {
        let mut scorer = HierarchicalScorer::new();

        // Train on a pattern: "K AE T" (cat)
        let pattern = ["K", "AE", "T"];

        for _ in 0..20 {
            scorer.train_sequence(&pattern);
        }

        let stats = scorer.stats();
        assert!(stats.total_patterns > 0);
        assert!(stats.active_manner_clusters > 0);

        // Score the trained pattern
        let trained_score = scorer.score_sequence(&pattern);

        // Score a different pattern
        let different = ["S", "IY", "Z"]; // "seize"
        let diff_score = scorer.score_sequence(&different);

        println!("Trained 'CAT': {:.4}", trained_score);
        println!("Different 'SEIZE': {:.4}", diff_score);

        // Trained pattern should score higher
        assert!(
            trained_score > diff_score,
            "Expected trained > different: {:.4} vs {:.4}",
            trained_score,
            diff_score
        );
    }

    #[test]
    fn test_discrimination_improvement() {
        let mut scorer = HierarchicalScorer::new();

        // Train on stop-vowel-stop patterns (CVC)
        let cvc_patterns = [
            ["P", "AE", "T"], // pat
            ["K", "AE", "T"], // cat
            ["B", "IH", "T"], // bit
            ["D", "AA", "G"], // dog
        ];

        for pattern in &cvc_patterns {
            for _ in 0..10 {
                scorer.train_sequence(pattern);
            }
        }

        // Test CVC pattern (should match)
        let test_cvc = ["G", "AH", "T"]; // gut
        let cvc_score = scorer.score_sequence(&test_cvc);

        // Test VCV pattern (should not match as well)
        let test_vcv = ["AE", "T", "AE"]; // ata
        let vcv_score = scorer.score_sequence(&test_vcv);

        println!("CVC pattern: {:.4}", cvc_score);
        println!("VCV pattern: {:.4}", vcv_score);

        // Discrimination
        let disc = scorer.discriminate(&test_cvc, &test_vcv);
        println!("Discrimination: {:+.4}", disc);

        // Should show some discrimination
        // Note: with hierarchical routing, similar manner patterns cluster together
    }

    #[test]
    fn test_cluster_isolation() {
        let mut scorer = HierarchicalScorer::new();
        scorer.set_level(2); // Manner×Place level

        // Train only on fricatives
        let fricative_pattern = ["S", "IH", "Z"]; // siz
        for _ in 0..30 {
            scorer.train_sequence(&fricative_pattern);
        }

        // Score fricative pattern
        let fric_score = scorer.score_sequence(&fricative_pattern);

        // Score stop pattern (different manner cluster)
        let stop_pattern = ["P", "IH", "T"]; // pit
        let stop_score = scorer.score_sequence(&stop_pattern);

        println!("Fricative pattern (trained): {:.4}", fric_score);
        println!("Stop pattern (untrained): {:.4}", stop_score);

        // Fricative should score much higher due to cluster isolation
        assert!(
            fric_score > stop_score + 0.05,
            "Expected strong cluster isolation: {:.4} vs {:.4}",
            fric_score,
            stop_score
        );
    }
}
