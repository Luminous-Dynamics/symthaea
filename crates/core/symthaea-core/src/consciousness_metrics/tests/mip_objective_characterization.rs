// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Characterization witnesses for PHI-MIP-001.
//!
//! These tests deliberately do **not** change production Phi/Sigma behavior.
//! They freeze an independent Gaussian partition-loss oracle and demonstrate
//! the objective currently used by `SynergisticIntegration::exhaustive_mip`.

use super::*;
use nalgebra::DMatrix;

#[derive(Debug, Clone)]
struct PartitionScore {
    part_a: Vec<usize>,
    part_b: Vec<usize>,
    retained: f64,
    loss: f64,
}

fn submatrix(cov: &[f64], n: usize, indices: &[usize]) -> Vec<f64> {
    let k = indices.len();
    let mut out = vec![0.0; k * k];
    for (ri, &i) in indices.iter().enumerate() {
        for (cj, &j) in indices.iter().enumerate() {
            out[ri * k + cj] = cov[i * n + j];
        }
    }
    out
}

/// Gaussian total correlation:
///
/// TC(X) = 0.5 * (sum_i ln(var_i) - ln(det(cov(X))))
fn gaussian_total_correlation(cov: &[f64], n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }

    assert_eq!(cov.len(), n * n);
    let matrix = DMatrix::from_row_slice(n, n, cov);
    let det = matrix.determinant();
    assert!(det > 0.0 && det.is_finite(), "fixture covariance must be positive definite");

    let sum_ln_var: f64 = (0..n)
        .map(|i| {
            let var = cov[i * n + i];
            assert!(var > 0.0 && var.is_finite());
            var.ln()
        })
        .sum();

    0.5 * (sum_ln_var - det.ln())
}

fn score_partitions(cov: &[f64], n: usize) -> Vec<PartitionScore> {
    assert!(n >= 2 && n < usize::BITS as usize);
    let total = gaussian_total_correlation(cov, n);
    let mut scores = Vec::new();

    // Fix variable 0 in A to remove A|B / B|A duplicates.
    for mask in 0usize..(1usize << (n - 1)) {
        let mut part_a = vec![0usize];
        let mut part_b = Vec::new();
        for bit in 0..(n - 1) {
            let idx = bit + 1;
            if (mask >> bit) & 1 == 1 {
                part_a.push(idx);
            } else {
                part_b.push(idx);
            }
        }
        if part_b.is_empty() {
            continue;
        }

        let tc_a = gaussian_total_correlation(&submatrix(cov, n, &part_a), part_a.len());
        let tc_b = gaussian_total_correlation(&submatrix(cov, n, &part_b), part_b.len());
        let retained = tc_a + tc_b;
        let loss = total - retained;
        scores.push(PartitionScore {
            part_a,
            part_b,
            retained,
            loss,
        });
    }

    scores
}

fn same_partition(score: &PartitionScore, expected_a: &[usize], expected_b: &[usize]) -> bool {
    (score.part_a == expected_a && score.part_b == expected_b)
        || (score.part_a == expected_b && score.part_b == expected_a)
}

#[test]
fn independent_subsystem_fixture_has_zero_loss_at_weakest_cut() {
    // A and B are strongly correlated; C is independent.
    // det([[1,.8],[.8,1]]) = 0.36, so TC(A,B,C) = -0.5 ln(0.36).
    let cov = vec![
        1.0, 0.8, 0.0, // A
        0.8, 1.0, 0.0, // B
        0.0, 0.0, 1.0, // C
    ];

    let total = gaussian_total_correlation(&cov, 3);
    let expected_total = -0.5 * 0.36_f64.ln();
    assert!((total - expected_total).abs() < 1e-12);

    let scores = score_partitions(&cov, 3);
    assert_eq!(scores.len(), 3);

    for score in &scores {
        // This identity is the proposition the later production repair must preserve.
        assert!((score.retained + score.loss - total).abs() < 1e-12);
    }

    let minimum_loss = scores
        .iter()
        .min_by(|a, b| a.loss.total_cmp(&b.loss))
        .expect("non-empty partition set");

    assert!(same_partition(minimum_loss, &[0, 1], &[2]));
    assert!(minimum_loss.loss.abs() < 1e-12);
    assert!((minimum_loss.retained - total).abs() < 1e-12);

    // The alternative cuts split the A-B dependence and therefore lose all
    // of the system's Gaussian total correlation in this fixture.
    for score in scores
        .iter()
        .filter(|score| !same_partition(score, &[0, 1], &[2]))
    {
        assert!(score.retained.abs() < 1e-12);
        assert!((score.loss - total).abs() < 1e-12);
    }
}

#[test]
fn legacy_exhaustive_sigma_search_minimizes_retained_information() {
    let cov = vec![
        1.0, 0.8, 0.0, // A
        0.8, 1.0, 0.0, // B
        0.0, 0.0, 1.0, // C
    ];

    let scores = score_partitions(&cov, 3);
    let minimum_retained = scores
        .iter()
        .map(|score| score.retained)
        .min_by(f64::total_cmp)
        .unwrap();
    let maximum_retained = scores
        .iter()
        .map(|score| score.retained)
        .max_by(f64::total_cmp)
        .unwrap();

    let legacy = SynergisticIntegration::new(SynergisticConfig {
        num_components: 3,
        window_size: 8,
        min_samples: 2,
        regularization: 0.0,
    });
    let legacy_partition_score = legacy.exhaustive_mip(&cov, 3);

    // Characterization only: current production code selects min retained MI.
    assert!((legacy_partition_score - minimum_retained).abs() < 1e-12);

    // But minimum partition loss is equivalent to maximum retained information
    // under this fixed-total Gaussian functional.
    assert!((maximum_retained - minimum_retained).abs() > 0.5);
    assert!((legacy_partition_score - maximum_retained).abs() > 0.5);
}
