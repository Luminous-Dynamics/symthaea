// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Skill affinity — Jaccard similarity between skill sets.
//!
//! Lightweight client-side alternative to full HDC hypervector proximity.
//! Computes |A∩B| / |A∪B| from credential/endorsement skill tags.

use std::collections::HashSet;

/// Jaccard similarity between two skill sets.
/// Returns 0.0 (no overlap) to 1.0 (identical sets).
pub fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { intersection / union }
}

/// Compute affinity as a percentage (0-100).
pub fn skill_affinity_pct(a: &HashSet<String>, b: &HashSet<String>) -> u8 {
    (jaccard_similarity(a, b) * 100.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn identical_sets() {
        let a = set(&["rust", "holochain"]);
        assert!((jaccard_similarity(&a, &a) - 1.0).abs() < 0.001);
    }

    #[test]
    fn disjoint_sets() {
        let a = set(&["rust", "holochain"]);
        let b = set(&["python", "django"]);
        assert!((jaccard_similarity(&a, &b)).abs() < 0.001);
    }

    #[test]
    fn partial_overlap() {
        let a = set(&["rust", "holochain", "wasm"]);
        let b = set(&["rust", "python", "wasm"]);
        // intersection: {rust, wasm} = 2, union: {rust, holochain, wasm, python} = 4
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 0.001);
    }

    #[test]
    fn empty_sets() {
        let empty: HashSet<String> = HashSet::new();
        assert_eq!(jaccard_similarity(&empty, &empty), 0.0);
    }

    #[test]
    fn affinity_percentage() {
        let a = set(&["rust", "holochain", "wasm"]);
        let b = set(&["rust", "python", "wasm"]);
        assert_eq!(skill_affinity_pct(&a, &b), 50);
    }
}
