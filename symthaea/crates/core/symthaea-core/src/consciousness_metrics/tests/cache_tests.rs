// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Caching tests for entropy and mutual information computations.

use super::*;

#[test]
fn test_cached_entropy_consistent() {
    let calc = CachedEntropyCalculator::new();

    let hv = ContinuousHV::random(HDC_DIMENSION, 42);

    // First call computes
    let h1 = calc.entropy(&hv);
    // Second call uses cache
    let h2 = calc.entropy(&hv);

    assert!(
        (h1 - h2).abs() < 1e-10,
        "Cached entropy should be consistent: {:.6} vs {:.6}",
        h1,
        h2
    );

    println!("Cached entropy: {:.6} (consistent)", h1);
}

#[test]
fn test_cached_mi_consistent() {
    let calc = CachedEntropyCalculator::new();

    let a = ContinuousHV::random(HDC_DIMENSION, 1);
    let b = ContinuousHV::random(HDC_DIMENSION, 2);

    // First call computes
    let mi1 = calc.mutual_information(&a, &b);
    // Second call uses cache
    let mi2 = calc.mutual_information(&a, &b);
    // Reversed order should also use cache (symmetric key)
    let mi3 = calc.mutual_information(&b, &a);

    assert!((mi1 - mi2).abs() < 1e-10);
    assert!((mi1 - mi3).abs() < 1e-10);

    println!("Cached MI: {:.6} (consistent)", mi1);
}

#[test]
fn test_cache_statistics() {
    ParallelEntropyCalculator::clear_cache();

    let calc = CachedEntropyCalculator::new();

    let vectors: Vec<ContinuousHV> = (0..5)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, 100 + i as u64))
        .collect();

    // Compute entropies
    for hv in &vectors {
        calc.entropy(hv);
    }

    let (entropy_size, mi_size) = ParallelEntropyCalculator::cache_stats();
    assert!(
        entropy_size >= 5,
        "Cache should have at least 5 entries: {}",
        entropy_size
    );

    println!("Cache stats: entropy={}, mi={}", entropy_size, mi_size);
}
