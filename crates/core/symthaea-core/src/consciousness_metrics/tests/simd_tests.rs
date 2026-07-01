// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SIMD histogram tests.

use super::*;

#[test]
fn test_simd_histogram_basic() {
    let binner = SimdHistogramBinner::new(16);

    let values: Vec<f32> = (-8..8).map(|i| i as f32 / 8.0).collect();
    let counts = binner.compute_histogram(&values);

    // Should have 16 values distributed across bins
    let total: usize = counts.iter().sum();
    assert_eq!(total, 16, "Should have 16 values in histogram");

    println!("SIMD histogram: {:?}", counts);
}

#[test]
fn test_simd_histogram_entropy() {
    let binner = SimdHistogramBinner::new(16);
    let serial = ContinuousEntropyEstimator::fast();

    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    let simd_h = binner.entropy(&hv.values, true);
    let serial_h = serial.entropy(&hv);

    // Should be identical (same algorithm)
    let diff = (simd_h - serial_h).abs();
    assert!(
        diff < 1e-10,
        "SIMD entropy should match serial: {:.6} vs {:.6}",
        simd_h,
        serial_h
    );

    println!("SIMD entropy: {:.6}", simd_h);
}

#[test]
fn test_simd_joint_histogram() {
    let binner = SimdHistogramBinner::new(16);

    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);

    let joint = binner.compute_joint_histogram(&a.values, &b.values);

    assert_eq!(joint.len(), 16);
    assert_eq!(joint[0].len(), 16);

    let total: usize = joint.iter().flat_map(|row| row.iter()).sum();
    assert_eq!(
        total, HDC_DIMENSION,
        "Joint histogram should have all values"
    );

    println!("SIMD joint histogram: 16x16 computed");
}

#[test]
fn test_simd_mutual_information() {
    let binner = SimdHistogramBinner::new(16);

    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);

    let marginal_a = binner.compute_histogram(&a.values);
    let marginal_b = binner.compute_histogram(&b.values);
    let joint = binner.compute_joint_histogram(&a.values, &b.values);

    let mi = binner.mutual_information_from_histograms(&joint, &marginal_a, &marginal_b, true);

    assert!(mi >= 0.0, "MI should be non-negative: {:.6}", mi);

    // Compare with estimator
    let est = ContinuousEntropyEstimator::fast();
    let est_mi = est.mutual_information_fast(&a, &b);

    let diff = (mi - est_mi).abs();
    assert!(
        diff < 1e-6,
        "SIMD MI should match estimator: {:.6} vs {:.6}",
        mi,
        est_mi
    );

    println!("SIMD MI: {:.6}", mi);
}
