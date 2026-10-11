// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Numerical qualification for the MI-graph Fiedler candidate path.
//!
//! These fixtures qualify only the small dense spectral candidate generator.
//! They do not claim global MIP optimality or validate an IIT interpretation.

use super::*;

fn mi_graph(n: usize, edges: &[(usize, usize, f64)], diagonal: &[f64]) -> Vec<Vec<f64>> {
    assert_eq!(diagonal.len(), n);
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        matrix[i][i] = diagonal[i];
    }
    for &(i, j, weight) in edges {
        matrix[i][j] = weight;
        matrix[j][i] = weight;
    }
    matrix
}

fn path4(diagonal: &[f64]) -> Vec<Vec<f64>> {
    mi_graph(
        4,
        &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
        diagonal,
    )
}

fn canonical_partition(partition: &TruePartition) -> (Vec<usize>, Vec<usize>) {
    let mut a = partition.part_a.clone();
    let mut b = partition.part_b.clone();
    a.sort_unstable();
    b.sort_unstable();
    if b < a {
        std::mem::swap(&mut a, &mut b);
    }
    (a, b)
}

#[test]
fn checked_fiedler_pair_matches_analytic_p4_eigenvalue_and_residual() {
    let matrix = path4(&[0.0; 4]);
    let laplacian = checked_mi_graph_laplacian(&matrix, 4).expect("valid connected P4 graph");
    let (lambda, vector) =
        checked_fiedler_eigenpair(&laplacian).expect("P4 has a unique Fiedler mode");

    let expected = 2.0 - 2.0_f64.sqrt();
    assert!(
        (lambda - expected).abs() < 1e-8,
        "expected λ₂={expected}, got {lambda}"
    );

    let residual_norm = (0..4)
        .map(|i| {
            let lv: f64 = (0..4).map(|j| laplacian[(i, j)] * vector[j]).sum();
            let residual = lv - lambda * vector[i];
            residual * residual
        })
        .sum::<f64>()
        .sqrt();
    assert!(residual_norm < 1e-8, "eigenpair residual={residual_norm}");
    assert!(vector.iter().sum::<f64>().abs() < 1e-8);
}

#[test]
fn checked_fiedler_abstains_when_solver_budget_is_exhausted() {
    let matrix = path4(&[0.0; 4]);
    let laplacian = checked_mi_graph_laplacian(&matrix, 4).expect("valid connected P4 graph");

    assert!(
        checked_fiedler_eigenpair_with_budget(&laplacian, 1).is_none(),
        "an exhausted eigen-solver must not produce a qualified spectral candidate"
    );
}

#[test]
fn spectral_partition_ignores_entropy_reporting_diagonal() {
    let calc = TruePhiCalculator::new();
    let uniform = path4(&[1.0, 1.0, 1.0, 1.0]);
    let skewed = path4(&[100.0, 0.25, 9.0, 3.0]);

    let a = calc.spectral_partition(&uniform, 4).expect("P4 is connected");
    let b = calc.spectral_partition(&skewed, 4).expect("P4 is connected");

    assert_eq!(
        canonical_partition(&a),
        canonical_partition(&b),
        "marginal-entropy reporting values must not become graph self-edge weights"
    );
    assert_eq!(canonical_partition(&a), (vec![0, 1], vec![2, 3]));
}

#[test]
fn spectral_partition_separates_dense_clusters_across_weak_bridge() {
    let calc = TruePhiCalculator::new();
    let mut edges = Vec::new();
    // Two internally dense three-node clusters.
    for start in [0usize, 3usize] {
        for i in start..start + 3 {
            for j in (i + 1)..start + 3 {
                edges.push((i, j, 1.0));
            }
        }
    }
    // One weak bridge between the clusters.
    edges.push((2, 3, 0.01));
    let matrix = mi_graph(6, &edges, &[0.0; 6]);

    let partition = calc
        .spectral_partition(&matrix, 6)
        .expect("the connected graph has a separated Fiedler mode");
    let (a, b) = canonical_partition(&partition);
    let cluster_a = vec![0, 1, 2];
    let cluster_b = vec![3, 4, 5];
    assert!(
        (a == cluster_a && b == cluster_b) || (a == cluster_b && b == cluster_a),
        "weak bridge should separate the two dense clusters, got {a:?} | {b:?}"
    );
}

#[test]
fn spectral_partition_abstains_for_disconnected_graph() {
    let calc = TruePhiCalculator::new();
    let matrix = mi_graph(4, &[(0, 1, 1.0), (2, 3, 1.0)], &[0.0; 4]);

    assert!(
        calc.spectral_partition(&matrix, 4).is_none(),
        "multiple zero Laplacian eigenvalues do not define a unique Fiedler cut"
    );
}

#[test]
fn spectral_partition_abstains_when_fiedler_eigenspace_is_degenerate() {
    let calc = TruePhiCalculator::new();
    let mut edges = Vec::new();
    for i in 0..4 {
        for j in (i + 1)..4 {
            edges.push((i, j, 1.0));
        }
    }
    let complete = mi_graph(4, &edges, &[0.0; 4]);

    assert!(
        calc.spectral_partition(&complete, 4).is_none(),
        "a complete graph has a repeated nonzero spectrum and no unique Fiedler direction"
    );
}

#[test]
fn spectral_partition_abstains_on_malformed_asymmetric_mi_graph() {
    let calc = TruePhiCalculator::new();
    let mut matrix = path4(&[0.0; 4]);
    matrix[0][1] = 1.0;
    matrix[1][0] = 0.5;

    assert!(calc.spectral_partition(&matrix, 4).is_none());
}

#[test]
fn spectral_partition_is_equivariant_under_node_permutation() {
    let calc = TruePhiCalculator::new();
    let original = path4(&[2.0, 5.0, 1.0, 7.0]);
    let base = calc
        .spectral_partition(&original, 4)
        .expect("P4 should have a unique Fiedler mode");

    // perm[new_index] = old_index
    let perm = [2usize, 0, 3, 1];
    let mut permuted = vec![vec![0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            permuted[i][j] = original[perm[i]][perm[j]];
        }
    }

    let changed = calc
        .spectral_partition(&permuted, 4)
        .expect("permutation preserves connectedness and eigengap");
    let mapped_back = TruePartition {
        part_a: changed.part_a.iter().map(|&i| perm[i]).collect(),
        part_b: changed.part_b.iter().map(|&i| perm[i]).collect(),
    };

    assert_eq!(canonical_partition(&base), canonical_partition(&mapped_back));
}
