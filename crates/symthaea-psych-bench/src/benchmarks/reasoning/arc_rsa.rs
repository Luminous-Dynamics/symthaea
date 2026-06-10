// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Representational Similarity Analysis (RSA) for HDC grid encodings
//! (SYNTHETIC — tests encoding structure, not reasoning).
//!
//! **Honesty note:** This benchmark explicitly measures encoding geometry
//! (whether HDC cosine similarity preserves grid structural similarity),
//! not reasoning. The z-scores reflect how faithfully the HDC space
//! represents task-relevant structure — an encoding quality metric.
//!
//! Builds two similarity matrices:
//! 1. **Encoding similarity**: pairwise cosine between HDC grid encodings
//! 2. **Structural similarity**: ground-truth overlap between grids
//!    (fraction of matching cells)
//!
//! Then computes Spearman rank correlation between the two matrices
//! (second-order isomorphism). High RSA correlation means the HDC space
//! preserves task-relevant structure.
//!
//! Also measures within-transform-type vs between-type encoding similarity
//! (analogous to neural "representational distance" in fMRI studies).
//!
//! References:
//! - Kriegeskorte et al. (2008) Representational similarity analysis
//! - Kanerva (2009) Hyperdimensional computing

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// Representational Similarity Analysis benchmark for HDC grid encodings.
pub struct ArcRsaBenchmark;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

fn gen_grid(rng: &mut u64, size: usize, num_colors: u8) -> Vec<Vec<u8>> {
    let mut grid = vec![vec![0u8; size]; size];
    for row in grid.iter_mut() {
        for cell in row.iter_mut() {
            xor_shift(rng);
            *cell = (*rng % num_colors as u64) as u8;
        }
    }
    grid
}

/// Compute structural similarity between two grids: fraction of matching cells.
fn structural_similarity(a: &[Vec<u8>], b: &[Vec<u8>]) -> f64 {
    let mut matches = 0usize;
    let mut total = 0usize;
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (&ca, &cb) in row_a.iter().zip(row_b.iter()) {
            total += 1;
            if ca == cb {
                matches += 1;
            }
        }
    }
    if total > 0 {
        matches as f64 / total as f64
    } else {
        0.0
    }
}

/// Spearman rank correlation between two vectors.
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len();

    let rank = |vals: &[f64]| -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n - 1 && (indexed[j + 1].1 - indexed[j].1).abs() < 1e-12 {
                j += 1;
            }
            let avg_rank = (i + j) as f64 / 2.0 + 1.0;
            for k in i..=j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j + 1;
        }
        ranks
    };

    let rx = rank(x);
    let ry = rank(y);

    // Pearson on ranks
    let mean_rx: f64 = rx.iter().sum::<f64>() / n as f64;
    let mean_ry: f64 = ry.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..n {
        let dxi = rx[i] - mean_rx;
        let dyi = ry[i] - mean_ry;
        num += dxi * dyi;
        dx2 += dxi * dxi;
        dy2 += dyi * dyi;
    }
    let den = (dx2 * dy2).sqrt();
    if den > 1e-12 { num / den } else { 0.0 }
}

#[derive(Debug, Clone, Copy)]
enum TransformType {
    ColorFill,
    Translation,
    ColorReplacement,
    Reflection,
}

const TRANSFORMS: [TransformType; 4] = [
    TransformType::ColorFill,
    TransformType::Translation,
    TransformType::ColorReplacement,
    TransformType::Reflection,
];

fn apply_transform(
    grid: &[Vec<u8>],
    tt: TransformType,
    param: u64,
    num_colors: u8,
) -> Vec<Vec<u8>> {
    match tt {
        TransformType::ColorFill => {
            let color = (param % num_colors as u64) as u8;
            let top = (param / 7 % 3) as usize;
            let left = (param / 11 % 3) as usize;
            GridEncoder::fill_region(grid, top, left, top + 1, left + 1, color)
        }
        TransformType::Translation => {
            let dx = ((param % 3) as i32) - 1;
            let dy = ((param / 3 % 3) as i32) - 1;
            let fill = (param / 9 % num_colors as u64) as u8;
            GridEncoder::translate_grid(grid, dx, dy, fill)
        }
        TransformType::ColorReplacement => {
            let from = (param % num_colors as u64) as u8;
            let to = ((param / 7 + 1) % num_colors as u64) as u8;
            GridEncoder::color_replace(grid, from, to)
        }
        TransformType::Reflection => {
            if param % 2 == 0 {
                GridEncoder::reflect_x(grid)
            } else {
                GridEncoder::reflect_y(grid)
            }
        }
    }
}

struct TrialResult {
    /// Spearman correlation between encoding and structural similarity matrices
    rsa_correlation: f64,
    /// Mean within-transform-type encoding similarity
    within_type_similarity: f64,
    /// Mean between-transform-type encoding similarity
    between_type_similarity: f64,
    /// Representational discriminability: within - between
    discriminability: f64,
    rt_ticks: f64,
}

impl ArcRsaBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _dim = config.dimension;
        let seed = config.trial_seed("reasoning", "arc_rsa", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let grid_size = 5;
        let num_colors: u8 = 6;
        let grids_per_type = 4; // 4 output grids per transform type

        let pressure = config.time_pressure;
        let tick_scale = 1.0 - pressure * 0.4;

        let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors as usize, seed);

        // Generate grids: for each transform type, apply to random inputs
        // Collect (grid, encoding, type_index) tuples
        let mut grids: Vec<Vec<Vec<u8>>> = Vec::new();
        let mut encodings: Vec<BinaryHV> = Vec::new();
        let mut type_labels: Vec<usize> = Vec::new();

        for (type_idx, &tt) in TRANSFORMS.iter().enumerate() {
            xor_shift(&mut rng);
            let param = rng;
            for _g in 0..grids_per_type {
                xor_shift(&mut rng);
                let input = gen_grid(&mut rng, grid_size, num_colors);
                let output = apply_transform(&input, tt, param, num_colors);
                let hv = encoder.encode_grid(&output);
                grids.push(output);
                encodings.push(hv);
                type_labels.push(type_idx);
            }
        }

        let n = grids.len();

        // Build upper-triangle similarity vectors
        let num_pairs = n * (n - 1) / 2;
        let mut encoding_sims = Vec::with_capacity(num_pairs);
        let mut structural_sims = Vec::with_capacity(num_pairs);
        let mut within_sims = Vec::new();
        let mut between_sims = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let enc_sim = encodings[i].similarity(&encodings[j]) as f64;
                let str_sim = structural_similarity(&grids[i], &grids[j]);

                encoding_sims.push(enc_sim);
                structural_sims.push(str_sim);

                if type_labels[i] == type_labels[j] {
                    within_sims.push(enc_sim);
                } else {
                    between_sims.push(enc_sim);
                }
            }
        }

        let rsa_correlation = spearman(&encoding_sims, &structural_sims);

        let within_type_similarity = if within_sims.is_empty() {
            0.0
        } else {
            within_sims.iter().sum::<f64>() / within_sims.len() as f64
        };
        let between_type_similarity = if between_sims.is_empty() {
            0.0
        } else {
            between_sims.iter().sum::<f64>() / between_sims.len() as f64
        };
        let discriminability = within_type_similarity - between_type_similarity;

        xor_shift(&mut rng);
        let rt_ticks = (5.0 + (rng % 5) as f64) * tick_scale;

        TrialResult {
            rsa_correlation,
            within_type_similarity,
            between_type_similarity,
            discriminability,
            rt_ticks,
        }
    }
}

impl PsychBenchmark for ArcRsaBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcRSA"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Representational Similarity Analysis",
            citation: "Kriegeskorte et al. (2008); Kanerva (2009)",
            year: 2008,
            doi: Some("10.3389/neuro.06.004.2008"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut rsa_corrs = Vec::new();
        let mut within_sims = Vec::new();
        let mut between_sims = Vec::new();
        let mut discrims = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            rsa_corrs.push(r.rsa_correlation);
            within_sims.push(r.within_type_similarity);
            between_sims.push(r.between_type_similarity);
            discrims.push(r.discriminability);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "arc_rsa".to_string(),
                    correct: r.rsa_correlation > 0.0,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("rsa_correlation", MetricValue::from_samples(&rsa_corrs));
        result.insert(
            "within_type_similarity",
            MetricValue::from_samples(&within_sims),
        );
        result.insert(
            "between_type_similarity",
            MetricValue::from_samples(&between_sims),
        );
        result.insert("discriminability", MetricValue::from_samples(&discrims));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 4; // 4 transform types
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.notes.push(
            "SYNTHETIC: Measures HDC encoding geometry (whether cosine similarity \
             preserves grid structural similarity), not reasoning. Z-scores reflect \
             representational fidelity of the HDC space."
                .to_string(),
        );
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_rsa_runs_with_metrics() {
        let result = ArcRsaBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("rsa_correlation"));
        assert!(result.metrics.contains_key("within_type_similarity"));
        assert!(result.metrics.contains_key("between_type_similarity"));
        assert!(result.metrics.contains_key("discriminability"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcRsaBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_rsa_correlation_bounded() {
        let result = ArcRsaBenchmark.run(&test_config());
        let rsa = result.metrics["rsa_correlation"].mean;
        assert!(
            rsa >= -1.0 && rsa <= 1.0,
            "RSA should be in [-1,1], got {}",
            rsa
        );
    }

    #[test]
    fn test_structural_similarity_self() {
        let grid = vec![vec![1, 2, 3], vec![4, 5, 6]];
        assert!((structural_similarity(&grid, &grid) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_structural_similarity_different() {
        let a = vec![vec![0, 0], vec![0, 0]];
        let b = vec![vec![1, 1], vec![1, 1]];
        assert!((structural_similarity(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = spearman(&x, &y);
        assert!((r - 1.0).abs() < 1e-6, "Perfect monotonic: r={}", r);
    }

    #[test]
    fn test_provenance() {
        let prov = ArcRsaBenchmark.provenance().unwrap();
        assert!(prov.citation.contains("Kriegeskorte"));
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 2,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcRsaBenchmark.run(&config);
        let r2 = ArcRsaBenchmark.run(&config);
        assert_eq!(
            r1.metrics["rsa_correlation"].mean,
            r2.metrics["rsa_correlation"].mean
        );
    }
}
