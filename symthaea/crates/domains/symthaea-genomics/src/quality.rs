// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Assembly quality assessment.
//!
//! Computes standard genomics quality metrics: N50, total length, coverage
//! statistics, completeness estimates, and Phred quality scores.

use crate::types::{AssemblyContig, GenomeQuality};

/// Quality assessment for genome assemblies.
///
/// Computes N50, coverage statistics, completeness, and Phred error estimates from assembled contigs.
pub struct QualityAssessor;

impl QualityAssessor {
    /// Assess the quality of an assembly from its contigs.
    ///
    /// Computes N50, total length, mean coverage, estimated completeness,
    /// and estimated error rate from the per-base quality scores.
    pub fn assess_assembly(contigs: &[AssemblyContig]) -> GenomeQuality {
        if contigs.is_empty() {
            return GenomeQuality {
                n50: 0,
                total_length: 0,
                num_contigs: 0,
                mean_coverage: 0.0,
                completeness: 0.0,
                error_rate: 1.0,
            };
        }

        let total_length: usize = contigs.iter().map(|c| c.sequence.len()).sum();
        let num_contigs = contigs.len();

        // Compute N50
        let n50 = Self::compute_n50(contigs, total_length);

        // Compute mean coverage across all positions
        let total_coverage: u64 = contigs
            .iter()
            .flat_map(|c| c.coverage.iter())
            .map(|&c| c as u64)
            .sum();
        let total_positions: u64 = contigs.iter().map(|c| c.coverage.len() as u64).sum();
        let mean_coverage = if total_positions > 0 {
            total_coverage as f64 / total_positions as f64
        } else {
            0.0
        };

        // Estimate completeness from coverage uniformity
        let completeness = Self::estimate_completeness(contigs);

        // Estimate error rate from quality scores
        let error_rate = Self::estimate_error_rate(contigs);

        GenomeQuality {
            n50,
            total_length,
            num_contigs,
            mean_coverage,
            completeness,
            error_rate,
        }
    }

    /// Compute N50: the contig length such that 50% of the total assembly
    /// length is in contigs of this length or longer.
    fn compute_n50(contigs: &[AssemblyContig], total_length: usize) -> usize {
        if contigs.is_empty() || total_length == 0 {
            return 0;
        }

        let mut lengths: Vec<usize> = contigs.iter().map(|c| c.sequence.len()).collect();
        lengths.sort_unstable_by(|a, b| b.cmp(a)); // Sort descending

        let half = total_length / 2;
        let mut cumulative = 0;

        for &len in &lengths {
            cumulative += len;
            if cumulative >= half {
                return len;
            }
        }

        // Should not reach here, but return smallest contig as fallback
        *lengths.last().unwrap_or(&0)
    }

    /// Compute Phred quality score from error probability.
    ///
    /// Q = -10 * log10(P_error)
    ///
    /// Standard Phred encoding used in FASTQ files.
    pub fn phred_score(error_probability: f64) -> f64 {
        if error_probability <= 0.0 {
            return 60.0; // Cap at Q60 for perfect calls
        }
        if error_probability >= 1.0 {
            return 0.0;
        }
        -10.0 * error_probability.log10()
    }

    /// Compute coverage uniformity as 1 - CV (coefficient of variation).
    ///
    /// Returns a value in [0, 1] where 1.0 = perfectly uniform coverage.
    pub fn coverage_uniformity(coverage: &[u32]) -> f64 {
        if coverage.is_empty() {
            return 0.0;
        }

        let n = coverage.len() as f64;
        let mean = coverage.iter().map(|&c| c as f64).sum::<f64>() / n;

        if mean < 1e-10 {
            return 0.0;
        }

        let variance = coverage
            .iter()
            .map(|&c| {
                let diff = c as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;

        let cv = variance.sqrt() / mean;

        // Invert so higher = more uniform, clamp to [0, 1]
        (1.0 - cv).clamp(0.0, 1.0)
    }

    /// Estimate completeness based on coverage.
    ///
    /// Positions with coverage >= 1 are considered "covered."
    /// Completeness = fraction of positions that are covered.
    fn estimate_completeness(contigs: &[AssemblyContig]) -> f64 {
        let total_positions: usize = contigs.iter().map(|c| c.coverage.len()).sum();
        if total_positions == 0 {
            return 0.0;
        }

        let covered: usize = contigs
            .iter()
            .flat_map(|c| c.coverage.iter())
            .filter(|&&c| c >= 1)
            .count();

        covered as f64 / total_positions as f64
    }

    /// Estimate error rate from quality scores.
    ///
    /// Error rate = 1 - mean(quality_scores).
    fn estimate_error_rate(contigs: &[AssemblyContig]) -> f64 {
        let all_qualities: Vec<f64> = contigs
            .iter()
            .flat_map(|c| c.quality_scores.iter().copied())
            .collect();

        if all_qualities.is_empty() {
            return 1.0;
        }

        let mean_quality = all_qualities.iter().sum::<f64>() / all_qualities.len() as f64;
        (1.0 - mean_quality).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DnaSequence;
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

    fn make_contig(len: usize, coverage: u32, quality: f64) -> AssemblyContig {
        AssemblyContig {
            sequence: DnaSequence::from_str(&"A".repeat(len)),
            quality_scores: vec![quality; len],
            coverage: vec![coverage; len],
            hv: ContinuousHV::zero(HDC_DIMENSION),
        }
    }

    #[test]
    fn test_assess_empty() {
        let quality = QualityAssessor::assess_assembly(&[]);
        assert_eq!(quality.n50, 0);
        assert_eq!(quality.total_length, 0);
        assert_eq!(quality.num_contigs, 0);
        assert_eq!(quality.error_rate, 1.0);
    }

    #[test]
    fn test_n50_single_contig() {
        let contigs = vec![make_contig(1000, 10, 0.99)];
        let quality = QualityAssessor::assess_assembly(&contigs);

        assert_eq!(quality.n50, 1000, "Single contig N50 should be its length");
        assert_eq!(quality.total_length, 1000);
        assert_eq!(quality.num_contigs, 1);
    }

    #[test]
    fn test_n50_multiple_contigs() {
        // Contigs of lengths 500, 300, 200
        let contigs = vec![
            make_contig(500, 10, 0.99),
            make_contig(300, 10, 0.99),
            make_contig(200, 10, 0.99),
        ];
        let quality = QualityAssessor::assess_assembly(&contigs);

        // Total = 1000. 50% = 500. Starting from largest:
        // 500 >= 500 => N50 = 500
        assert_eq!(quality.n50, 500);
        assert_eq!(quality.total_length, 1000);
        assert_eq!(quality.num_contigs, 3);
    }

    #[test]
    fn test_n50_requires_two_contigs() {
        // Contigs of lengths 400, 350, 250
        let contigs = vec![
            make_contig(400, 10, 0.99),
            make_contig(350, 10, 0.99),
            make_contig(250, 10, 0.99),
        ];
        let quality = QualityAssessor::assess_assembly(&contigs);

        // Total = 1000. 50% = 500.
        // 400 < 500. 400+350 = 750 >= 500. N50 = 350.
        assert_eq!(quality.n50, 350);
    }

    #[test]
    fn test_mean_coverage() {
        let contigs = vec![make_contig(100, 20, 0.99)];
        let quality = QualityAssessor::assess_assembly(&contigs);
        assert!((quality.mean_coverage - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_completeness_all_covered() {
        let contigs = vec![make_contig(100, 5, 0.99)];
        let quality = QualityAssessor::assess_assembly(&contigs);
        assert!((quality.completeness - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_completeness_partial_coverage() {
        let mut contig = make_contig(10, 5, 0.99);
        // Zero out half the coverage
        for i in 0..5 {
            contig.coverage[i] = 0;
        }
        let quality = QualityAssessor::assess_assembly(&[contig]);
        assert!(
            (quality.completeness - 0.5).abs() < 0.01,
            "Half-covered should have ~0.5 completeness, got {}",
            quality.completeness
        );
    }

    #[test]
    fn test_error_rate_high_quality() {
        let contigs = vec![make_contig(100, 10, 0.99)];
        let quality = QualityAssessor::assess_assembly(&contigs);
        assert!(
            quality.error_rate < 0.05,
            "High quality should have low error rate: {}",
            quality.error_rate
        );
    }

    #[test]
    fn test_error_rate_low_quality() {
        let contigs = vec![make_contig(100, 10, 0.5)];
        let quality = QualityAssessor::assess_assembly(&contigs);
        assert!(
            quality.error_rate > 0.4,
            "Low quality should have high error rate: {}",
            quality.error_rate
        );
    }

    #[test]
    fn test_phred_score_math() {
        // P=0.1 -> Q=10
        let q = QualityAssessor::phred_score(0.1);
        assert!(
            (q - 10.0).abs() < 0.01,
            "Phred(0.1) should be 10, got {}",
            q
        );

        // P=0.01 -> Q=20
        let q = QualityAssessor::phred_score(0.01);
        assert!(
            (q - 20.0).abs() < 0.01,
            "Phred(0.01) should be 20, got {}",
            q
        );

        // P=0.001 -> Q=30
        let q = QualityAssessor::phred_score(0.001);
        assert!(
            (q - 30.0).abs() < 0.01,
            "Phred(0.001) should be 30, got {}",
            q
        );
    }

    #[test]
    fn test_phred_score_edge_cases() {
        assert_eq!(QualityAssessor::phred_score(0.0), 60.0);
        assert_eq!(QualityAssessor::phred_score(1.0), 0.0);
        assert_eq!(QualityAssessor::phred_score(2.0), 0.0);
        assert_eq!(QualityAssessor::phred_score(-0.5), 60.0);
    }

    #[test]
    fn test_coverage_uniformity_perfect() {
        let coverage = vec![10, 10, 10, 10, 10];
        let uniformity = QualityAssessor::coverage_uniformity(&coverage);
        assert!(
            (uniformity - 1.0).abs() < 0.01,
            "Uniform coverage should give 1.0, got {}",
            uniformity
        );
    }

    #[test]
    fn test_coverage_uniformity_variable() {
        let coverage = vec![1, 100, 1, 100, 1];
        let uniformity = QualityAssessor::coverage_uniformity(&coverage);
        assert!(
            uniformity < 0.5,
            "Highly variable coverage should give low uniformity, got {}",
            uniformity
        );
    }

    #[test]
    fn test_coverage_uniformity_empty() {
        let uniformity = QualityAssessor::coverage_uniformity(&[]);
        assert_eq!(uniformity, 0.0);
    }

    #[test]
    fn test_coverage_uniformity_zero() {
        let coverage = vec![0, 0, 0];
        let uniformity = QualityAssessor::coverage_uniformity(&coverage);
        assert_eq!(uniformity, 0.0);
    }
}
