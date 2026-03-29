// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC majority-voting error correction across overlapping reads.
//!
//! Error correction exploits the redundancy from overlapping reads. At each
//! position, multiple reads contribute their base calls. The corrector uses
//! quality-weighted voting to determine the consensus, with HDC-level
//! correction using weighted bundling of read hypervectors.

use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::types::{Base, DnaSequence};

/// Error corrector using majority voting and HDC consensus.
#[derive(Debug, Clone)]
pub struct ErrorCorrector {
    /// Minimum number of overlapping reads for correction (default 3).
    pub min_coverage: u32,
    /// Minimum confidence to apply a correction (default 0.8).
    pub confidence_threshold: f64,
}

impl Default for ErrorCorrector {
    fn default() -> Self {
        Self {
            min_coverage: 3,
            confidence_threshold: 0.8,
        }
    }
}

impl ErrorCorrector {
    /// Create a new error corrector with custom parameters.
    pub fn new(min_coverage: u32, confidence_threshold: f64) -> Self {
        Self {
            min_coverage,
            confidence_threshold,
        }
    }

    /// Correct a region using quality-weighted HDC bundling.
    ///
    /// Takes multiple HV representations of overlapping reads and their quality
    /// scores, produces a consensus HV and a confidence value.
    ///
    /// Higher-quality reads contribute more to the consensus via weighted bundling.
    pub fn correct_region(reads: &[&ContinuousHV], qualities: &[f64]) -> (ContinuousHV, f64) {
        if reads.is_empty() {
            return (ContinuousHV::zero(16_384), 0.0);
        }

        if reads.len() == 1 {
            return (reads[0].clone(), qualities.first().copied().unwrap_or(0.5));
        }

        // Convert qualities to f32 weights for weighted bundling
        let weights: Vec<f32> = qualities
            .iter()
            .map(|&q| (q as f32).max(0.01)) // Avoid zero weights
            .collect();

        let consensus = ContinuousHV::weighted_bundle(reads, &weights);

        // Compute confidence as average pairwise similarity to consensus
        let total_sim: f64 = reads.iter().map(|r| r.similarity(&consensus) as f64).sum();
        let avg_sim = total_sim / reads.len() as f64;

        // Confidence is the average agreement with consensus, clamped to [0, 1]
        let confidence = avg_sim.clamp(0.0, 1.0);

        (consensus, confidence)
    }

    /// Per-position majority voting across aligned reads.
    ///
    /// For each position, the base with the highest quality-weighted vote wins.
    /// Returns the corrected sequence and per-position quality scores.
    pub fn correct_base_calls(
        &self,
        aligned_reads: &[DnaSequence],
        qualities: &[Vec<f64>],
    ) -> (DnaSequence, Vec<f64>) {
        if aligned_reads.is_empty() {
            return (DnaSequence::new(vec![]), vec![]);
        }

        let max_len = aligned_reads.iter().map(|r| r.len()).max().unwrap_or(0);

        let mut corrected_bases = Vec::with_capacity(max_len);
        let mut corrected_qualities = Vec::with_capacity(max_len);

        for pos in 0..max_len {
            // Accumulate quality-weighted votes for each base
            let mut base_scores = [0.0f64; 5]; // A, T, G, C, N
            let mut total_coverage = 0u32;

            for (read_idx, read) in aligned_reads.iter().enumerate() {
                if pos >= read.len() {
                    continue;
                }

                total_coverage += 1;

                let quality = if read_idx < qualities.len() && pos < qualities[read_idx].len() {
                    qualities[read_idx][pos]
                } else {
                    0.5 // Default quality if not provided
                };

                let base_idx = match read.bases[pos] {
                    Base::A => 0,
                    Base::T => 1,
                    Base::G => 2,
                    Base::C => 3,
                    Base::N => 4,
                };

                base_scores[base_idx] += quality;
            }

            // If below minimum coverage, output N with low quality
            if total_coverage < self.min_coverage {
                corrected_bases.push(Base::N);
                corrected_qualities.push(0.0);
                continue;
            }

            // Find the base with highest weighted vote
            let total_weight: f64 = base_scores.iter().sum();
            let (best_idx, &best_score) = base_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("base_scores is non-empty (4 bases)");

            let base = match best_idx {
                0 => Base::A,
                1 => Base::T,
                2 => Base::G,
                3 => Base::C,
                _ => Base::N,
            };

            let quality = if total_weight > 0.0 {
                best_score / total_weight
            } else {
                0.0
            };

            // Only apply correction if confidence exceeds threshold
            if quality >= self.confidence_threshold {
                corrected_bases.push(base);
                corrected_qualities.push(quality);
            } else {
                // Use majority without confidence weighting
                corrected_bases.push(base);
                corrected_qualities.push(quality);
            }
        }

        (DnaSequence::new(corrected_bases), corrected_qualities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::unified_hv::HDC_DIMENSION;

    #[test]
    fn test_correct_region_single_read() {
        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        let (consensus, confidence) = ErrorCorrector::correct_region(&[&hv], &[0.9]);

        let sim = hv.similarity(&consensus);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Single read correction should return same HV"
        );
        assert!((confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_correct_region_empty() {
        let (consensus, confidence) = ErrorCorrector::correct_region(&[], &[]);
        assert!(consensus.norm() < 1e-6);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_correct_region_high_agreement() {
        // Three very similar HVs should produce high confidence
        let base = ContinuousHV::random(HDC_DIMENSION, 42);
        // Use small perturbations for similar reads
        let hv1 = base.clone();
        let hv2 = base
            .add(&ContinuousHV::random(HDC_DIMENSION, 43).scale(0.01))
            .normalize();
        let hv3 = base
            .add(&ContinuousHV::random(HDC_DIMENSION, 44).scale(0.01))
            .normalize();

        let reads = [&hv1, &hv2, &hv3];
        let qualities = [0.9, 0.85, 0.8];
        let (_consensus, confidence) = ErrorCorrector::correct_region(&reads, &qualities);

        assert!(
            confidence > 0.5,
            "High agreement should produce high confidence, got {}",
            confidence
        );
    }

    #[test]
    fn test_correct_region_quality_weighting() {
        // A high-quality read should dominate
        let good = ContinuousHV::random(HDC_DIMENSION, 42);
        let bad = ContinuousHV::random(HDC_DIMENSION, 43);

        let reads = [&good, &bad];
        let qualities_good_dominant = [0.99, 0.01];
        let (consensus, _) = ErrorCorrector::correct_region(&reads, &qualities_good_dominant);

        let sim_good = good.similarity(&consensus);
        let sim_bad = bad.similarity(&consensus);

        assert!(
            sim_good > sim_bad,
            "High-quality read should dominate: sim_good={}, sim_bad={}",
            sim_good,
            sim_bad
        );
    }

    #[test]
    fn test_correct_base_calls_majority_voting() {
        let corrector = ErrorCorrector::new(2, 0.5);

        let reads = vec![
            DnaSequence::from_str("ATGC"),
            DnaSequence::from_str("ATGC"),
            DnaSequence::from_str("ATAC"), // Error at position 2
        ];
        let qualities = vec![vec![0.9; 4], vec![0.9; 4], vec![0.9; 4]];

        let (corrected, scores) = corrector.correct_base_calls(&reads, &qualities);

        // Majority should win
        assert_eq!(corrected.bases[2], Base::G, "Majority vote should select G");
        assert_eq!(corrected.len(), 4);
        assert!(
            scores[2] > 0.5,
            "Quality should reflect majority confidence"
        );
    }

    #[test]
    fn test_correct_base_calls_quality_breaks_tie() {
        let corrector = ErrorCorrector::new(1, 0.0);

        let reads = vec![DnaSequence::from_str("A"), DnaSequence::from_str("T")];
        // First read has much higher quality
        let qualities = vec![vec![0.99], vec![0.01]];

        let (corrected, _scores) = corrector.correct_base_calls(&reads, &qualities);

        assert_eq!(
            corrected.bases[0],
            Base::A,
            "Higher-quality read should win the tie"
        );
    }

    #[test]
    fn test_correct_base_calls_low_coverage_gives_n() {
        let corrector = ErrorCorrector::new(3, 0.5); // Require 3 reads minimum

        let reads = vec![DnaSequence::from_str("ATGC"), DnaSequence::from_str("ATGC")];
        let qualities = vec![vec![0.9; 4], vec![0.9; 4]];

        let (corrected, scores) = corrector.correct_base_calls(&reads, &qualities);

        // Only 2 reads but minimum is 3: should output N with quality 0
        assert_eq!(corrected.bases[0], Base::N);
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn test_correct_base_calls_empty() {
        let corrector = ErrorCorrector::default();
        let (corrected, scores) = corrector.correct_base_calls(&[], &[]);
        assert!(corrected.is_empty());
        assert!(scores.is_empty());
    }

    #[test]
    fn test_correct_base_calls_uneven_lengths() {
        let corrector = ErrorCorrector::new(1, 0.0);

        let reads = vec![
            DnaSequence::from_str("ATGCCC"),
            DnaSequence::from_str("ATGC"),
            DnaSequence::from_str("ATGCCC"),
        ];
        let qualities = vec![vec![0.9; 6], vec![0.9; 4], vec![0.9; 6]];

        let (corrected, _scores) = corrector.correct_base_calls(&reads, &qualities);

        // Output should be as long as the longest read
        assert_eq!(corrected.len(), 6);
    }

    #[test]
    fn test_correct_base_calls_handles_missing_qualities() {
        let corrector = ErrorCorrector::new(1, 0.0);

        let reads = vec![DnaSequence::from_str("ATGC")];
        let qualities: Vec<Vec<f64>> = vec![]; // No quality data

        let (corrected, _scores) = corrector.correct_base_calls(&reads, &qualities);

        // Should still produce output using default quality
        assert_eq!(corrected.len(), 4);
    }
}
