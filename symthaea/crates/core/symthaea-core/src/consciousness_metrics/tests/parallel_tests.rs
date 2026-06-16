// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parallel entropy computation tests.

use super::*;

#[test]
fn test_parallel_entropy_batch() {
    let calc = ParallelEntropyCalculator::new();
    let serial = ContinuousEntropyEstimator::fast();

    let vectors: Vec<ContinuousHV> = (0..8)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    // Compute in parallel
    let parallel_results = calc.entropy_batch(&vectors);

    // Verify against serial computation
    for (i, hv) in vectors.iter().enumerate() {
        let serial_h = serial.entropy(hv);
        let parallel_h = parallel_results[i];

        let diff = (serial_h - parallel_h).abs();
        assert!(
            diff < 1e-10,
            "Parallel entropy should match serial: {:.6} vs {:.6}",
            parallel_h,
            serial_h
        );
    }

    println!(
        "Parallel entropy batch: {} vectors processed",
        vectors.len()
    );
}

#[test]
fn test_parallel_mi_matrix() {
    let calc = ParallelEntropyCalculator::new();
    let serial = ContinuousEntropyEstimator::fast();

    let vectors: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    let matrix = calc.mutual_information_matrix(&vectors);

    // Verify symmetry and diagonal
    assert_eq!(matrix.len(), 4);
    for i in 0..4 {
        for j in 0..4 {
            if i == j {
                // Diagonal should be entropy
                let expected = serial.entropy(&vectors[i]);
                let diff = (matrix[i][j] - expected).abs();
                assert!(
                    diff < 1e-10,
                    "Diagonal should be entropy: {:.6} vs {:.6}",
                    matrix[i][j],
                    expected
                );
            } else {
                // Off-diagonal should be symmetric
                let diff = (matrix[i][j] - matrix[j][i]).abs();
                assert!(
                    diff < 1e-10,
                    "MI matrix should be symmetric: [{},{}]={:.6}, [{},{}]={:.6}",
                    i,
                    j,
                    matrix[i][j],
                    j,
                    i,
                    matrix[j][i]
                );
            }
        }
    }

    println!("Parallel MI matrix: 4x4 computed");
}

#[test]
fn test_parallel_effective_information() {
    let calc = ParallelEntropyCalculator::new();
    let serial = TruePhiCalculator::new();

    let vectors: Vec<ContinuousHV> = (0..5)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    let parallel_ei = calc.effective_information(&vectors);
    let serial_ei = serial.effective_information(&vectors);

    let diff = (parallel_ei - serial_ei).abs();
    assert!(
        diff < 1e-6,
        "Parallel EI should match serial: {:.6} vs {:.6}",
        parallel_ei,
        serial_ei
    );

    println!("Parallel EI: {:.6}", parallel_ei);
}

#[test]
fn test_parallel_true_phi() {
    let calc = ParallelEntropyCalculator::new();
    let serial = TruePhiCalculator::new();

    let vectors: Vec<ContinuousHV> = (0..4)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
        .collect();

    let parallel_result = calc.compute_true_phi_parallel(&vectors);
    let serial_result = serial.compute_true_phi(&vectors);

    // Should produce consistent results
    assert!(parallel_result.phi >= 0.0);
    assert_eq!(parallel_result.component_entropies.len(), 4);
    assert_eq!(parallel_result.mutual_information_matrix.len(), 4);

    println!(
        "Parallel \u{03a6}: {:.6}, Serial \u{03a6}: {:.6}",
        parallel_result.phi, serial_result.phi
    );
}
