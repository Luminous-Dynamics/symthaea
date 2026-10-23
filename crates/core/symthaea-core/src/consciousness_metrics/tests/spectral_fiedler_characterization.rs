// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Characterization witnesses for PHI-SPECTRAL-001.
//!
//! These tests deliberately freeze two properties of the historical
//! `TruePhiCalculator` spectral path without changing production behavior:
//!
//! 1. the helper named `power_iteration_fiedler` converges to the largest
//!    non-constant Laplacian eigenmode on the P4 path graph, not the Fiedler
//!    (second-smallest) eigenmode;
//! 2. `spectral_partition` changes when only the reporting diagonal of the
//!    pairwise-information matrix changes, even though a graph adjacency with
//!    zero self-edges would be unchanged.
//!
//! A later repair is expected to replace these legacy properties with qualified
//! graph-Laplacian / Fiedler invariants rather than preserving them.

use super::*;

fn rayleigh_quotient(matrix: &[Vec<f64>], vector: &[f64]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &vi) in vector.iter().enumerate() {
        denominator += vi * vi;
        let row_dot: f64 = matrix[i]
            .iter()
            .zip(vector.iter())
            .map(|(&mij, &vj)| mij * vj)
            .sum();
        numerator += vi * row_dot;
    }

    numerator / denominator
}

fn canonical_partition(partition: &TruePartition) -> (Vec<usize>, Vec<usize>) {
    let mut a = partition.part_a.clone();
    let mut b = partition.part_b.clone();
    a.sort_unstable();
    b.sort_unstable();

    if a <= b { (a, b) } else { (b, a) }
}

fn path4_information_matrix(diagonal: [f64; 4]) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; 4]; 4];

    for i in 0..4 {
        matrix[i][i] = diagonal[i];
    }

    // Hold the actual graph edges fixed: 0--1--2--3, unit weight.
    for (i, j) in [(0usize, 1usize), (1, 2), (2, 3)] {
        matrix[i][j] = 1.0;
        matrix[j][i] = 1.0;
    }

    matrix
}

#[test]
fn legacy_power_iteration_fiedler_targets_largest_nonconstant_path4_mode() {
    let calc = TruePhiCalculator::new();
    let laplacian = vec![
        vec![1.0, -1.0, 0.0, 0.0],
        vec![-1.0, 2.0, -1.0, 0.0],
        vec![0.0, -1.0, 2.0, -1.0],
        vec![0.0, 0.0, -1.0, 1.0],
    ];

    let vector = calc.power_iteration_fiedler(&laplacian, 4, 100);
    let quotient = rayleigh_quotient(&laplacian, &vector);

    // P4 Laplacian spectrum:
    //   0,
    //   2 - sqrt(2) ~= 0.585786 (Fiedler),
    //   2,
    //   2 + sqrt(2) ~= 3.414214 (largest mode).
    let fiedler_eigenvalue = 2.0 - 2.0_f64.sqrt();
    let largest_eigenvalue = 2.0 + 2.0_f64.sqrt();

    assert!(
        (quotient - largest_eigenvalue).abs() < 1e-6,
        "legacy power iteration should characterize the largest nonconstant mode: \
         Rayleigh quotient={quotient}, expected={largest_eigenvalue}"
    );
    assert!(
        (quotient - fiedler_eigenvalue).abs() > 1.0,
        "legacy result must remain discriminating from the true Fiedler eigenvalue: \
         quotient={quotient}, fiedler={fiedler_eigenvalue}"
    );
}

#[test]
fn legacy_spectral_partition_depends_on_reporting_diagonal() {
    let calc = TruePhiCalculator::new();

    // Both matrices have exactly the same off-diagonal pairwise-MI graph.
    // Only the reporting diagonal changes. A graph adjacency projection should
    // zero self-edges before forming D-W, making this change irrelevant to the
    // spectral graph partition.
    let uniform_reporting_diagonal = path4_information_matrix([1.0, 1.0, 1.0, 1.0]);
    let skewed_reporting_diagonal = path4_information_matrix([100.0, 1.0, 1.0, 1.0]);

    let uniform = calc
        .spectral_partition(&uniform_reporting_diagonal, 4)
        .expect("legacy spectral partition should produce a partition");
    let skewed = calc
        .spectral_partition(&skewed_reporting_diagonal, 4)
        .expect("legacy spectral partition should produce a partition");

    assert_ne!(
        canonical_partition(&uniform),
        canonical_partition(&skewed),
        "historical spectral path is expected to change when only the reporting \
         diagonal changes; once adjacency and reporting matrices are separated, \
         replace this characterization with diagonal-invariance"
    );
}
