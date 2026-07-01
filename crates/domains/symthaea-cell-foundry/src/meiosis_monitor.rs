// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meiosis checkpoint verification for gamete quality control.
//!
//! Verifies that critical meiotic events (synapsis, crossover, segregation)
//! have occurred correctly based on cell state indicators.

use crate::types::{CellState, MeiosisCheckpoint};

/// Stateless monitor for meiosis quality checkpoints.
pub struct MeiosisMonitor;

impl MeiosisMonitor {
    /// Verify synapsis (homolog pairing).
    ///
    /// Returns (passed, quality_score). Synapsis quality depends on
    /// viability and pluripotency score (cells transitioning from iPSC
    /// should have shed most pluripotency markers).
    pub fn verify_synapsis(cell: &CellState) -> (bool, f64) {
        // Good synapsis requires adequate viability and partial loss of pluripotency
        let viability_score = cell.viability.clamp(0.0, 1.0);
        let differentiation_score = (1.0 - cell.pluripotency_score).clamp(0.0, 1.0);

        // Gene expression diversity contributes to synapsis quality
        let expr_diversity = if cell.gene_expression.is_empty() {
            0.5
        } else {
            let mean: f64 =
                cell.gene_expression.iter().sum::<f64>() / cell.gene_expression.len() as f64;
            let variance: f64 = cell
                .gene_expression
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / cell.gene_expression.len() as f64;
            (variance.sqrt()).clamp(0.0, 1.0)
        };

        let quality = 0.4 * viability_score + 0.4 * differentiation_score + 0.2 * expr_diversity;
        let passed = quality > 0.4;
        (passed, quality)
    }

    /// Verify crossover (recombination) events.
    ///
    /// Returns (passed, quality_score). Adequate crossovers are needed for
    /// proper chromosome segregation. `min_crossovers` is the minimum number
    /// expected (typically 1 per chromosome pair).
    pub fn verify_crossover(cell: &CellState, min_crossovers: u32) -> (bool, f64) {
        // Estimate crossover count from gene expression heterogeneity
        let heterogeneity = if cell.gene_expression.len() < 2 {
            0.0
        } else {
            let mut diffs = 0.0;
            for i in 1..cell.gene_expression.len() {
                diffs += (cell.gene_expression[i] - cell.gene_expression[i - 1]).abs();
            }
            diffs / (cell.gene_expression.len() - 1) as f64
        };

        // Estimated crossover count (higher heterogeneity = more crossovers)
        let estimated_crossovers = (heterogeneity * 10.0) as u32;
        let crossover_quality = if estimated_crossovers >= min_crossovers {
            1.0
        } else {
            estimated_crossovers as f64 / min_crossovers as f64
        };

        let viability_factor = cell.viability.clamp(0.0, 1.0);
        let quality = crossover_quality * viability_factor;
        let passed = quality > 0.4 && estimated_crossovers >= min_crossovers;

        (passed, quality)
    }

    /// Verify segregation (chromosome distribution after division).
    ///
    /// Compares pre-division cell with post-division daughter cells.
    /// Returns (passed, quality_score). Good segregation means daughter
    /// cells have complementary gene expression profiles.
    pub fn verify_segregation(pre: &CellState, post: &[CellState]) -> (bool, f64) {
        if post.is_empty() {
            return (false, 0.0);
        }

        // Check that daughter cells are viable
        let viable_count = post.iter().filter(|c| c.viability > 0.3).count();
        let viability_ratio = viable_count as f64 / post.len() as f64;

        // Check for complementary gene expression (daughters should differ from each other)
        let expression_quality = if post.len() >= 2 && !pre.gene_expression.is_empty() {
            let mut total_diff = 0.0;
            let mut comparisons = 0;
            for i in 0..post.len() {
                for j in (i + 1)..post.len() {
                    let min_len = post[i]
                        .gene_expression
                        .len()
                        .min(post[j].gene_expression.len());
                    if min_len > 0 {
                        let diff: f64 = (0..min_len)
                            .map(|k| {
                                (post[i].gene_expression[k] - post[j].gene_expression[k]).abs()
                            })
                            .sum::<f64>()
                            / min_len as f64;
                        total_diff += diff;
                        comparisons += 1;
                    }
                }
            }
            if comparisons > 0 {
                (total_diff / comparisons as f64).clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        let quality = 0.5 * viability_ratio + 0.5 * expression_quality;
        let passed = quality > 0.3 && viability_ratio > 0.5;
        (passed, quality)
    }

    /// Compute overall meiosis quality from checkpoint results.
    ///
    /// Returns a weighted average: synapsis and crossover are weighted more heavily
    /// because errors at those stages are more consequential.
    pub fn overall_quality(checkpoints: &[(MeiosisCheckpoint, bool, f64)]) -> f64 {
        if checkpoints.is_empty() {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for (checkpoint, passed, score) in checkpoints {
            let weight = match checkpoint {
                MeiosisCheckpoint::Synapsis => 2.0,
                MeiosisCheckpoint::Crossover => 2.0,
                MeiosisCheckpoint::SegregationI => 1.5,
                MeiosisCheckpoint::SegregationII => 1.0,
            };

            let effective_score = if *passed { *score } else { score * 0.5 };
            weighted_sum += effective_score * weight;
            weight_total += weight;
        }

        if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DEFAULT_GENE_EXPR_LEN;

    fn make_cell_for_synapsis() -> CellState {
        let mut cell = CellState::new_somatic();
        cell.viability = 0.9;
        cell.pluripotency_score = 0.2; // Low pluripotency = differentiated toward germ
        // Add some gene expression diversity
        cell.gene_expression = (0..DEFAULT_GENE_EXPR_LEN)
            .map(|i| i as f64 / DEFAULT_GENE_EXPR_LEN as f64)
            .collect();
        cell
    }

    #[test]
    fn test_synapsis_pass() {
        let cell = make_cell_for_synapsis();
        let (passed, score) = MeiosisMonitor::verify_synapsis(&cell);
        assert!(
            passed,
            "Healthy differentiated cell should pass synapsis, score={}",
            score
        );
        assert!(score > 0.4);
    }

    #[test]
    fn test_synapsis_fail_low_viability() {
        let mut cell = make_cell_for_synapsis();
        cell.viability = 0.05;
        cell.pluripotency_score = 0.5;
        let (passed, score) = MeiosisMonitor::verify_synapsis(&cell);
        assert!(
            !passed,
            "Low viability should fail synapsis, score={}",
            score
        );
    }

    #[test]
    fn test_crossover_pass() {
        let mut cell = CellState::new_somatic();
        cell.viability = 0.9;
        // High heterogeneity in gene expression
        cell.gene_expression = (0..DEFAULT_GENE_EXPR_LEN)
            .map(|i| if i % 2 == 0 { 0.9 } else { 0.1 })
            .collect();
        let (passed, score) = MeiosisMonitor::verify_crossover(&cell, 1);
        assert!(
            passed,
            "Heterogeneous expression should pass crossover, score={}",
            score
        );
    }

    #[test]
    fn test_crossover_fail_homogeneous() {
        let mut cell = CellState::new_somatic();
        cell.viability = 0.9;
        cell.gene_expression = vec![0.5; DEFAULT_GENE_EXPR_LEN]; // No heterogeneity
        let (passed, _score) = MeiosisMonitor::verify_crossover(&cell, 1);
        assert!(!passed, "Homogeneous expression should fail crossover");
    }

    #[test]
    fn test_segregation_pass() {
        let pre = CellState::new_somatic();
        let mut daughter1 = CellState::new_somatic();
        let mut daughter2 = CellState::new_somatic();
        daughter1.gene_expression = vec![0.9; DEFAULT_GENE_EXPR_LEN];
        daughter2.gene_expression = vec![0.1; DEFAULT_GENE_EXPR_LEN];
        let (passed, score) = MeiosisMonitor::verify_segregation(&pre, &[daughter1, daughter2]);
        assert!(
            passed,
            "Complementary daughters should pass segregation, score={}",
            score
        );
    }

    #[test]
    fn test_segregation_fail_empty() {
        let pre = CellState::new_somatic();
        let (passed, score) = MeiosisMonitor::verify_segregation(&pre, &[]);
        assert!(!passed);
        assert!((score).abs() < 1e-9);
    }

    #[test]
    fn test_overall_quality_empty() {
        assert!((MeiosisMonitor::overall_quality(&[])).abs() < 1e-9);
    }

    #[test]
    fn test_overall_quality_all_pass() {
        let checkpoints = vec![
            (MeiosisCheckpoint::Synapsis, true, 0.9),
            (MeiosisCheckpoint::Crossover, true, 0.8),
            (MeiosisCheckpoint::SegregationI, true, 0.85),
            (MeiosisCheckpoint::SegregationII, true, 0.7),
        ];
        let quality = MeiosisMonitor::overall_quality(&checkpoints);
        assert!(
            quality > 0.7,
            "All pass with high scores should give high quality: {}",
            quality
        );
    }

    #[test]
    fn test_overall_quality_partial_fail() {
        let checkpoints = vec![
            (MeiosisCheckpoint::Synapsis, true, 0.9),
            (MeiosisCheckpoint::Crossover, false, 0.3),
        ];
        let quality = MeiosisMonitor::overall_quality(&checkpoints);
        // Failed crossover should drag quality down
        assert!(
            quality < 0.8,
            "Partial failure should reduce quality: {}",
            quality
        );
    }

    #[test]
    fn test_synapsis_score_bounded() {
        let cell = make_cell_for_synapsis();
        let (_, score) = MeiosisMonitor::verify_synapsis(&cell);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
