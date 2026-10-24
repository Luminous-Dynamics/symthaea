// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Structural novelty metrics derived from deterministic CSG fingerprints.

use symthaea_passive_design_search::CsgFingerprint;

/// Fraction of digest bits that differ between two structural fingerprints.
pub fn fingerprint_hamming_distance(a: &CsgFingerprint, b: &CsgFingerprint) -> f64 {
    let differing = a
        .digest
        .iter()
        .zip(b.digest.iter())
        .map(|(left, right)| (left ^ right).count_ones() as usize)
        .sum::<usize>();
    differing as f64 / 256.0
}

/// Novelty of a candidate against a reference population.
pub fn novelty_against(candidate: &CsgFingerprint, population: &[CsgFingerprint]) -> f64 {
    if population.is_empty() {
        return 1.0;
    }
    population
        .iter()
        .map(|other| fingerprint_hamming_distance(candidate, other))
        .sum::<f64>()
        / population.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::csg::CSGNode;

    #[test]
    fn identical_fingerprints_have_zero_distance() {
        let a = CsgFingerprint::from_csg(&CSGNode::cube());
        assert_eq!(fingerprint_hamming_distance(&a, &a), 0.0);
    }

    #[test]
    fn different_structures_can_have_nonzero_distance() {
        let a = CsgFingerprint::from_csg(&CSGNode::cube());
        let b = CsgFingerprint::from_csg(&CSGNode::sphere());
        assert!(fingerprint_hamming_distance(&a, &b) > 0.0);
    }

    #[test]
    fn empty_population_is_maximally_novel() {
        let a = CsgFingerprint::from_csg(&CSGNode::cube());
        assert_eq!(novelty_against(&a, &[]), 1.0);
    }
}