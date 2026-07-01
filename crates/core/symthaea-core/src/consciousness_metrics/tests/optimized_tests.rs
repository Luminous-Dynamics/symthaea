// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tests for optimized entropy estimation methods (fast k-NN, fast KDE).

use super::*;

#[test]
fn test_knn_fast_produces_reasonable_entropy() {
    let est = ContinuousEntropyEstimator::knn_fast(3);

    let uniform = ContinuousHV::random(HDC_DIMENSION, 42);
    let h = est.entropy(&uniform);

    assert!(
        h >= 0.0,
        "Fast k-NN entropy should be non-negative: {:.4}",
        h
    );
    assert!(h < 10.0, "Fast k-NN entropy should be reasonable: {:.4}", h);
}

#[test]
fn test_knn_fast_matches_slow_approximately() {
    let est_slow = ContinuousEntropyEstimator::knn(3);
    let est_fast = ContinuousEntropyEstimator::knn_fast(3);

    let hv = ContinuousHV::random(512, 123); // Smaller dimension for speed

    let h_slow = est_slow.entropy(&hv);
    let h_fast = est_fast.entropy(&hv);

    // They should be within 50% of each other (different algorithms, similar results)
    let ratio = if h_slow > 0.0 { h_fast / h_slow } else { 1.0 };
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "Fast and slow k-NN should give similar results: slow={:.4}, fast={:.4}",
        h_slow,
        h_fast
    );
}

#[test]
fn test_kde_fast_produces_reasonable_entropy() {
    let est = ContinuousEntropyEstimator::kde_fast();

    let hv = ContinuousHV::random(HDC_DIMENSION, 42);
    let h = est.entropy(&hv);

    assert!(
        h >= 0.0,
        "Fast KDE entropy should be non-negative: {:.4}",
        h
    );
    assert!(h < 10.0, "Fast KDE entropy should be reasonable: {:.4}", h);
}

#[test]
fn test_kde_fast_matches_slow_approximately() {
    let est_slow = ContinuousEntropyEstimator::kde();
    let est_fast = ContinuousEntropyEstimator::kde_fast();

    let hv = ContinuousHV::random(512, 456); // Smaller dimension for speed

    let h_slow = est_slow.entropy(&hv);
    let h_fast = est_fast.entropy(&hv);

    // They should be within 50% of each other
    let diff = (h_fast - h_slow).abs();
    let max_h = h_slow.max(h_fast);
    let relative_diff = if max_h > 0.0 { diff / max_h } else { 0.0 };

    assert!(
        relative_diff < 0.5,
        "Fast and slow KDE should give similar results: slow={:.4}, fast={:.4}, diff={:.1}%",
        h_slow,
        h_fast,
        relative_diff * 100.0
    );
}

#[test]
fn test_mutual_information_fast() {
    let est = ContinuousEntropyEstimator::default();

    // Correlated vectors
    let base = ContinuousHV::random(HDC_DIMENSION, 100);
    let hv1 = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 101)],
        &[0.8, 0.2],
    );
    let hv2 = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 102)],
        &[0.8, 0.2],
    );

    let mi = est.mutual_information_fast(&hv1, &hv2);
    assert!(mi >= 0.0, "Fast MI should be non-negative: {:.4}", mi);
}

#[test]
fn test_fast_mi_detects_correlation() {
    let est = ContinuousEntropyEstimator::default();

    // Independent vectors
    let ind1 = ContinuousHV::random(HDC_DIMENSION, 1);
    let ind2 = ContinuousHV::random(HDC_DIMENSION, 2);
    let mi_ind = est.mutual_information_fast(&ind1, &ind2);

    // Correlated vectors
    let base = ContinuousHV::random(HDC_DIMENSION, 100);
    let cor1 = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 101)],
        &[0.8, 0.2],
    );
    let cor2 = ContinuousHV::weighted_bundle(
        &[&base, &ContinuousHV::random(HDC_DIMENSION, 102)],
        &[0.8, 0.2],
    );
    let mi_cor = est.mutual_information_fast(&cor1, &cor2);

    assert!(
        mi_cor > mi_ind,
        "Fast MI should detect correlation: correlated={:.4} > independent={:.4}",
        mi_cor,
        mi_ind
    );
}

#[test]
fn test_fast_vs_accurate_constructors() {
    let fast = ContinuousEntropyEstimator::fast();
    let accurate = ContinuousEntropyEstimator::accurate();

    let hv = ContinuousHV::random(HDC_DIMENSION, 789);

    let h_fast = fast.entropy(&hv);
    let h_accurate = accurate.entropy(&hv);

    // Both should produce reasonable entropy values
    assert!(
        h_fast > 0.0 && h_fast < 10.0,
        "Fast entropy should be reasonable: {:.4}",
        h_fast
    );
    assert!(
        h_accurate > 0.0 && h_accurate < 10.0,
        "Accurate entropy should be reasonable: {:.4}",
        h_accurate
    );

    // They should be similar (same underlying data)
    let ratio = h_fast / h_accurate;
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "Fast and accurate should give similar results: fast={:.4}, accurate={:.4}",
        h_fast,
        h_accurate
    );
}
