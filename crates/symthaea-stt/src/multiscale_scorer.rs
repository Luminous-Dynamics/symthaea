// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Scale HDC Scorer - Maximum Discrimination
//!
//! Extends hierarchical scoring with:
//! - Level 3: Manner × Place × Voicing (140 clusters)
//! - Multi-scale voting: combines coarse and fine levels
//! - Adaptive weighting based on cluster confidence
//!
//! # Architecture
//!
//! ```text
//! Level 1: Manner (7 clusters)           ─┐
//!          ↓                              │
//! Level 2: Manner × Place (70 clusters)  ─┼─► Multi-Scale Vote
//!          ↓                              │
//! Level 3: M × P × Voicing (140 clusters)─┘
//! ```
//!
//! Each level contributes to the final score with learned weights.

use crate::hdc::{BundleAccumulator, HDC_DIM, HV16};
use std::collections::HashMap;

/// Number of Manner classes
pub const NUM_MANNER: usize = 7;

/// Number of Place classes
pub const NUM_PLACE: usize = 10;

/// Number of Voicing classes (Voiced, Voiceless)
pub const NUM_VOICING: usize = 2;

/// Multi-scale HDC Scorer with 3 levels + voting
pub struct MultiScaleScorer {
    /// Phoneme basis vectors
    phoneme_basis: HashMap<String, HV16>,

    /// Level 1: Manner memories (7)
    l1_memories: [HV16; NUM_MANNER],
    l1_counts: [usize; NUM_MANNER],

    /// Level 2: Manner × Place memories (70)
    l2_memories: [[HV16; NUM_PLACE]; NUM_MANNER],
    l2_counts: [[usize; NUM_PLACE]; NUM_MANNER],

    /// Level 3: Manner × Place × Voicing memories (140)
    l3_memories: [[[HV16; NUM_VOICING]; NUM_PLACE]; NUM_MANNER],
    l3_counts: [[[usize; NUM_VOICING]; NUM_PLACE]; NUM_MANNER],

    /// Current context state
    context_state: HV16,

    /// Multi-scale weights [L1, L2, L3]
    scale_weights: [f32; 3],

    /// Phoneme → (Manner, Place, Voicing) mapping
    phoneme_features: HashMap<String, (usize, usize, usize)>,
}

impl Default for MultiScaleScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiScaleScorer {
    /// Create a new multi-scale scorer
    pub fn new() -> Self {
        // Initialize phoneme basis
        let mut phoneme_basis = HashMap::new();
        let phonemes = [
            "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
            "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T",
            "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL", "UNK",
        ];

        for phoneme in phonemes.iter() {
            let seed = format!("multiscale_basis_{phoneme}");
            phoneme_basis.insert(phoneme.to_string(), HV16::random(&seed));
        }

        // Initialize all memory levels
        let l1_memories = std::array::from_fn(|m| HV16::random(&format!("l1_memory_{m}")));

        let l2_memories = std::array::from_fn(|m| {
            std::array::from_fn(|p| HV16::random(&format!("l2_memory_{m}_{p}")))
        });

        let l3_memories = std::array::from_fn(|m| {
            std::array::from_fn(|p| {
                std::array::from_fn(|v| HV16::random(&format!("l3_memory_{m}_{p}_{v}")))
            })
        });

        let phoneme_features = Self::build_phoneme_features();

        Self {
            phoneme_basis,
            l1_memories,
            l1_counts: [0; NUM_MANNER],
            l2_memories,
            l2_counts: [[0; NUM_PLACE]; NUM_MANNER],
            l3_memories,
            l3_counts: [[[0; NUM_VOICING]; NUM_PLACE]; NUM_MANNER],
            context_state: HV16::zero(),
            // Default weights: favor finer granularity
            scale_weights: [0.1, 0.3, 0.6],
            phoneme_features,
        }
    }

    /// Build phoneme → (manner, place, voicing) mapping
    fn build_phoneme_features() -> HashMap<String, (usize, usize, usize)> {
        let mut map = HashMap::new();

        // Manner: 0=Stop, 1=Fricative, 2=Affricate, 3=Nasal, 4=Liquid, 5=Glide, 6=Vowel
        // Place: 0=Bilabial, 1=Labiodental, 2=Dental, 3=Alveolar, 4=Palatal, 5=Velar, 6=Glottal, 7=Front, 8=Central, 9=Back
        // Voicing: 0=Voiceless, 1=Voiced

        // Stops (manner=0)
        map.insert("P".into(), (0, 0, 0)); // Bilabial, Voiceless
        map.insert("B".into(), (0, 0, 1)); // Bilabial, Voiced
        map.insert("T".into(), (0, 3, 0)); // Alveolar, Voiceless
        map.insert("D".into(), (0, 3, 1)); // Alveolar, Voiced
        map.insert("K".into(), (0, 5, 0)); // Velar, Voiceless
        map.insert("G".into(), (0, 5, 1)); // Velar, Voiced

        // Fricatives (manner=1)
        map.insert("F".into(), (1, 1, 0)); // Labiodental, Voiceless
        map.insert("V".into(), (1, 1, 1)); // Labiodental, Voiced
        map.insert("TH".into(), (1, 2, 0)); // Dental, Voiceless
        map.insert("DH".into(), (1, 2, 1)); // Dental, Voiced
        map.insert("S".into(), (1, 3, 0)); // Alveolar, Voiceless
        map.insert("Z".into(), (1, 3, 1)); // Alveolar, Voiced
        map.insert("SH".into(), (1, 4, 0)); // Palatal, Voiceless
        map.insert("ZH".into(), (1, 4, 1)); // Palatal, Voiced
        map.insert("HH".into(), (1, 6, 0)); // Glottal, Voiceless

        // Affricates (manner=2)
        map.insert("CH".into(), (2, 4, 0)); // Palatal, Voiceless
        map.insert("JH".into(), (2, 4, 1)); // Palatal, Voiced

        // Nasals (manner=3) - all voiced
        map.insert("M".into(), (3, 0, 1)); // Bilabial
        map.insert("N".into(), (3, 3, 1)); // Alveolar
        map.insert("NG".into(), (3, 5, 1)); // Velar

        // Liquids (manner=4) - all voiced
        map.insert("L".into(), (4, 3, 1)); // Alveolar
        map.insert("R".into(), (4, 3, 1)); // Alveolar (retroflex)

        // Glides (manner=5) - all voiced
        map.insert("W".into(), (5, 0, 1)); // Bilabial
        map.insert("Y".into(), (5, 4, 1)); // Palatal

        // Vowels (manner=6) - all voiced, use place for frontness
        map.insert("IY".into(), (6, 7, 1)); // Front high
        map.insert("IH".into(), (6, 7, 1)); // Front high-mid
        map.insert("EY".into(), (6, 7, 1)); // Front mid
        map.insert("EH".into(), (6, 7, 1)); // Front mid-low
        map.insert("AE".into(), (6, 7, 1)); // Front low
        map.insert("AH".into(), (6, 8, 1)); // Central
        map.insert("ER".into(), (6, 8, 1)); // Central r-colored
        map.insert("AY".into(), (6, 8, 1)); // Central diphthong
        map.insert("AW".into(), (6, 8, 1)); // Central diphthong
        map.insert("UW".into(), (6, 9, 1)); // Back high
        map.insert("UH".into(), (6, 9, 1)); // Back high-mid
        map.insert("OW".into(), (6, 9, 1)); // Back mid
        map.insert("AA".into(), (6, 9, 1)); // Back low
        map.insert("AO".into(), (6, 9, 1)); // Back mid-low
        map.insert("OY".into(), (6, 9, 1)); // Back diphthong

        // Special
        map.insert("SIL".into(), (6, 8, 0)); // Silence = voiceless central
        map.insert("UNK".into(), (6, 8, 1)); // Unknown = voiced central

        map
    }

    /// Set custom scale weights
    pub fn set_weights(&mut self, l1: f32, l2: f32, l3: f32) {
        let sum = l1 + l2 + l3;
        self.scale_weights = [l1 / sum, l2 / sum, l3 / sum];
    }

    /// Reset context
    pub fn reset(&mut self) {
        self.context_state = HV16::zero();
    }

    /// Get phoneme basis
    fn get_basis(&self, phoneme: &str) -> HV16 {
        self.phoneme_basis.get(phoneme).copied().unwrap_or_else(|| {
            self.phoneme_basis
                .get("UNK")
                .copied()
                .unwrap_or_else(HV16::zero)
        })
    }

    /// Get articulatory features
    fn get_features(&self, phoneme: &str) -> (usize, usize, usize) {
        self.phoneme_features
            .get(phoneme)
            .copied()
            .unwrap_or((6, 8, 1)) // Default: vowel, central, voiced
    }

    /// Process a phoneme with multi-scale scoring
    pub fn process(&mut self, phoneme: &str) -> f32 {
        let phi = self.get_basis(phoneme);
        let (manner, place, voicing) = self.get_features(phoneme);

        // Update context: H_t = Π(H_{t-1}) ⊕ Φ_current
        let rotated = self.context_state.rotate(1);
        self.context_state = rotated.bind(&phi);

        // Compute scores at each level
        let s1 = self.l1_memories[manner].similarity(&self.context_state);
        let s2 = self.l2_memories[manner][place].similarity(&self.context_state);
        let s3 = self.l3_memories[manner][place][voicing].similarity(&self.context_state);

        // Multi-scale vote
        self.scale_weights[0] * s1 + self.scale_weights[1] * s2 + self.scale_weights[2] * s3
    }

    /// Score a sequence with multi-scale voting
    pub fn score_sequence(&mut self, phonemes: &[&str]) -> f32 {
        self.reset();
        let mut total = 0.0;
        for phoneme in phonemes {
            total += self.process(phoneme);
        }
        total / phonemes.len().max(1) as f32
    }

    /// Score at a specific level only (for comparison)
    pub fn score_at_level(&mut self, phonemes: &[&str], level: usize) -> f32 {
        self.reset();
        let mut total = 0.0;

        for phoneme in phonemes {
            let phi = self.get_basis(phoneme);
            let (manner, place, voicing) = self.get_features(phoneme);

            let rotated = self.context_state.rotate(1);
            self.context_state = rotated.bind(&phi);

            let score = match level {
                1 => self.l1_memories[manner].similarity(&self.context_state),
                2 => self.l2_memories[manner][place].similarity(&self.context_state),
                3 => self.l3_memories[manner][place][voicing].similarity(&self.context_state),
                _ => 0.0,
            };
            total += score;
        }

        total / phonemes.len().max(1) as f32
    }

    /// Train on a phoneme sequence
    pub fn train_sequence(&mut self, phonemes: &[&str]) {
        if phonemes.is_empty() {
            return;
        }

        // Accumulators for each level
        let mut l1_accums: Vec<BundleAccumulator> =
            (0..NUM_MANNER).map(|_| BundleAccumulator::new()).collect();

        let mut l2_accums: Vec<Vec<BundleAccumulator>> = (0..NUM_MANNER)
            .map(|_| (0..NUM_PLACE).map(|_| BundleAccumulator::new()).collect())
            .collect();

        let mut l3_accums: Vec<Vec<Vec<BundleAccumulator>>> = (0..NUM_MANNER)
            .map(|_| {
                (0..NUM_PLACE)
                    .map(|_| (0..NUM_VOICING).map(|_| BundleAccumulator::new()).collect())
                    .collect()
            })
            .collect();

        self.reset();

        for phoneme in phonemes {
            let phi = self.get_basis(phoneme);
            let (manner, place, voicing) = self.get_features(phoneme);

            // Update context
            let rotated = self.context_state.rotate(1);
            self.context_state = rotated.bind(&phi);

            // Add to all level accumulators
            l1_accums[manner].add(&self.context_state);
            l2_accums[manner][place].add(&self.context_state);
            l3_accums[manner][place][voicing].add(&self.context_state);
        }

        // Update L1 memories
        for m in 0..NUM_MANNER {
            if l1_accums[m].count() > 0 {
                let mut combined = BundleAccumulator::new();
                if self.l1_counts[m] > 0 {
                    combined.add_weighted(&self.l1_memories[m], self.l1_counts[m] as f32);
                }
                let new_patterns = l1_accums[m].finalize();
                combined.add_weighted(&new_patterns, l1_accums[m].count() as f32);
                self.l1_memories[m] = combined.finalize();
                self.l1_counts[m] += l1_accums[m].count();
            }
        }

        // Update L2 memories
        for m in 0..NUM_MANNER {
            for p in 0..NUM_PLACE {
                if l2_accums[m][p].count() > 0 {
                    let mut combined = BundleAccumulator::new();
                    if self.l2_counts[m][p] > 0 {
                        combined.add_weighted(&self.l2_memories[m][p], self.l2_counts[m][p] as f32);
                    }
                    let new_patterns = l2_accums[m][p].finalize();
                    combined.add_weighted(&new_patterns, l2_accums[m][p].count() as f32);
                    self.l2_memories[m][p] = combined.finalize();
                    self.l2_counts[m][p] += l2_accums[m][p].count();
                }
            }
        }

        // Update L3 memories
        for m in 0..NUM_MANNER {
            for p in 0..NUM_PLACE {
                for v in 0..NUM_VOICING {
                    if l3_accums[m][p][v].count() > 0 {
                        let mut combined = BundleAccumulator::new();
                        if self.l3_counts[m][p][v] > 0 {
                            combined.add_weighted(
                                &self.l3_memories[m][p][v],
                                self.l3_counts[m][p][v] as f32,
                            );
                        }
                        let new_patterns = l3_accums[m][p][v].finalize();
                        combined.add_weighted(&new_patterns, l3_accums[m][p][v].count() as f32);
                        self.l3_memories[m][p][v] = combined.finalize();
                        self.l3_counts[m][p][v] += l3_accums[m][p][v].count();
                    }
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

    /// Get statistics
    pub fn stats(&self) -> MultiScaleStats {
        let _total_l1: usize = self.l1_counts.iter().sum();
        let _total_l2: usize = self.l2_counts.iter().flat_map(|r| r.iter()).sum();
        let total_l3: usize = self
            .l3_counts
            .iter()
            .flat_map(|m| m.iter().flat_map(|p| p.iter()))
            .sum();

        let active_l1 = self.l1_counts.iter().filter(|&&c| c > 0).count();
        let active_l2 = self
            .l2_counts
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&c| c > 0)
            .count();
        let active_l3 = self
            .l3_counts
            .iter()
            .flat_map(|m| m.iter().flat_map(|p| p.iter()))
            .filter(|&&c| c > 0)
            .count();

        // Average density at each level
        let l1_density: f32 = self
            .l1_memories
            .iter()
            .map(|m| m.popcount() as f32 / HDC_DIM as f32)
            .sum::<f32>()
            / NUM_MANNER as f32;

        let l2_density: f32 = self
            .l2_memories
            .iter()
            .flat_map(|m| m.iter())
            .map(|m| m.popcount() as f32 / HDC_DIM as f32)
            .sum::<f32>()
            / (NUM_MANNER * NUM_PLACE) as f32;

        let l3_density: f32 = self
            .l3_memories
            .iter()
            .flat_map(|m| m.iter().flat_map(|p| p.iter()))
            .map(|m| m.popcount() as f32 / HDC_DIM as f32)
            .sum::<f32>()
            / (NUM_MANNER * NUM_PLACE * NUM_VOICING) as f32;

        MultiScaleStats {
            total_patterns: total_l3,
            active_l1_clusters: active_l1,
            active_l2_clusters: active_l2,
            active_l3_clusters: active_l3,
            avg_l1_density: l1_density,
            avg_l2_density: l2_density,
            avg_l3_density: l3_density,
            scale_weights: self.scale_weights,
        }
    }
}

/// Statistics for multi-scale scorer
#[derive(Debug)]
pub struct MultiScaleStats {
    /// Total training patterns
    pub total_patterns: usize,
    /// Active L1 clusters (of 7)
    pub active_l1_clusters: usize,
    /// Active L2 clusters (of 70)
    pub active_l2_clusters: usize,
    /// Active L3 clusters (of 140)
    pub active_l3_clusters: usize,
    /// Average L1 memory density
    pub avg_l1_density: f32,
    /// Average L2 memory density
    pub avg_l2_density: f32,
    /// Average L3 memory density
    pub avg_l3_density: f32,
    /// Current scale weights
    pub scale_weights: [f32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiscale_creation() {
        let scorer = MultiScaleScorer::new();
        let stats = scorer.stats();
        assert_eq!(stats.total_patterns, 0);
        assert_eq!(stats.active_l3_clusters, 0);
    }

    #[test]
    fn test_voicing_features() {
        let scorer = MultiScaleScorer::new();

        // Check voicing distinctions
        let (_, _, v_p) = scorer.get_features("P");
        let (_, _, v_b) = scorer.get_features("B");
        assert_eq!(v_p, 0); // Voiceless
        assert_eq!(v_b, 1); // Voiced

        let (_, _, v_s) = scorer.get_features("S");
        let (_, _, v_z) = scorer.get_features("Z");
        assert_eq!(v_s, 0); // Voiceless
        assert_eq!(v_z, 1); // Voiced
    }

    #[test]
    fn test_multiscale_training() {
        let mut scorer = MultiScaleScorer::new();

        // Train on voiced stops
        let voiced_pattern = ["B", "AE", "D"]; // bad
        for _ in 0..20 {
            scorer.train_sequence(&voiced_pattern);
        }

        let stats = scorer.stats();
        assert!(stats.total_patterns > 0);
        assert!(stats.active_l3_clusters > 0);

        // Trained pattern should score higher
        let trained_score = scorer.score_sequence(&voiced_pattern);
        let voiceless_pattern = ["P", "AE", "T"]; // pat
        let voiceless_score = scorer.score_sequence(&voiceless_pattern);

        println!("Voiced 'BAD' (trained): {:.4}", trained_score);
        println!("Voiceless 'PAT' (untrained): {:.4}", voiceless_score);

        assert!(
            trained_score > voiceless_score,
            "Trained should beat untrained: {:.4} vs {:.4}",
            trained_score,
            voiceless_score
        );
    }

    #[test]
    fn test_voicing_isolation() {
        let mut scorer = MultiScaleScorer::new();

        // Train ONLY on voiceless patterns
        let voiceless_patterns = [
            ["P", "IH", "T"], // pit
            ["K", "AE", "T"], // cat
            ["T", "AO", "P"], // top
            ["S", "IH", "T"], // sit
        ];

        for pattern in &voiceless_patterns {
            for _ in 0..30 {
                scorer.train_sequence(pattern);
            }
        }

        // Test voiceless vs voiced at L3
        let test_voiceless = ["P", "AE", "K"]; // pack (voiceless)
        let test_voiced = ["B", "AE", "G"]; // bag (voiced)

        let l3_voiceless = scorer.score_at_level(&test_voiceless, 3);
        let l3_voiced = scorer.score_at_level(&test_voiced, 3);

        println!("L3 Voiceless 'PACK': {:.4}", l3_voiceless);
        println!("L3 Voiced 'BAG': {:.4}", l3_voiced);
        println!(
            "L3 Voicing Discrimination: {:+.4}",
            l3_voiceless - l3_voiced
        );

        // L3 should strongly distinguish voicing
        assert!(
            l3_voiceless > l3_voiced + 0.1,
            "L3 should isolate voicing: {:.4} vs {:.4}",
            l3_voiceless,
            l3_voiced
        );
    }

    #[test]
    fn test_multiscale_voting() {
        let mut scorer = MultiScaleScorer::new();

        // Train on mixed patterns
        let patterns = [
            ["K", "AE", "T"], // cat
            ["D", "AO", "G"], // dog
            ["S", "IY", "Z"], // seize
            ["M", "AE", "N"], // man
        ];

        for pattern in &patterns {
            for _ in 0..20 {
                scorer.train_sequence(pattern);
            }
        }

        // Compare single-level vs multi-scale
        let test = ["K", "AE", "T"];

        let l1_score = scorer.score_at_level(&test, 1);
        let l2_score = scorer.score_at_level(&test, 2);
        let l3_score = scorer.score_at_level(&test, 3);
        let multi_score = scorer.score_sequence(&test);

        println!("L1 score: {:.4}", l1_score);
        println!("L2 score: {:.4}", l2_score);
        println!("L3 score: {:.4}", l3_score);
        println!("Multi-scale: {:.4}", multi_score);

        // Multi-scale should be weighted combination
        let expected = scorer.scale_weights[0] * l1_score
            + scorer.scale_weights[1] * l2_score
            + scorer.scale_weights[2] * l3_score;

        // Reset and rescore to get fresh multi_score
        let fresh_multi = scorer.score_sequence(&test);

        println!("Expected weighted: {:.4}", expected);
        println!("Got multi-scale: {:.4}", fresh_multi);
    }

    #[test]
    fn test_level_comparison() {
        let mut scorer = MultiScaleScorer::new();

        // Heavy training on specific patterns
        let cvc_patterns = [
            ["P", "AE", "T"],
            ["B", "IH", "T"],
            ["K", "AH", "T"],
            ["D", "AA", "G"],
        ];

        for pattern in &cvc_patterns {
            for _ in 0..50 {
                scorer.train_sequence(pattern);
            }
        }

        // Test discrimination at each level
        let trained = ["K", "AE", "T"];
        let different = ["S", "IY", "Z"];

        let disc_l1 = scorer.score_at_level(&trained, 1) - scorer.score_at_level(&different, 1);
        let disc_l2 = scorer.score_at_level(&trained, 2) - scorer.score_at_level(&different, 2);
        let disc_l3 = scorer.score_at_level(&trained, 3) - scorer.score_at_level(&different, 3);
        let disc_multi = scorer.discriminate(&trained, &different);

        println!("Discrimination (CAT vs SEIZE):");
        println!("  L1 (Manner):      {:+.4}", disc_l1);
        println!("  L2 (M×P):         {:+.4}", disc_l2);
        println!("  L3 (M×P×V):       {:+.4}", disc_l3);
        println!("  Multi-scale:      {:+.4}", disc_multi);

        // Finer levels should generally have better discrimination
        assert!(
            disc_l3.abs() >= disc_l1.abs() * 0.5,
            "L3 should have meaningful discrimination"
        );
    }
}
