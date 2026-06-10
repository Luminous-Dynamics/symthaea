// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Inbreeding coefficient (F) and kinship calculations.
//!
//! Provides both pedigree-based kinship (exact) and HDC-based kinship estimates (fast).
//! Pedigree path counting is capped at 10 generations back for performance.

use crate::types::{Individual, Pedigree, Population};
use std::collections::HashMap;

/// Inbreeding coefficient from heterozygosity deficit.
///
/// F = 1 - Ho/He
/// Where Ho = observed heterozygosity, He = expected heterozygosity.
/// Returns 0.0 if He is zero (monomorphic locus).
pub fn inbreeding_coefficient_from_heterozygosity(he: f64, ho: f64) -> f64 {
    if he == 0.0 { 0.0 } else { 1.0 - ho / he }
}

/// Kinship coefficient between two individuals from pedigree data.
///
/// Uses path counting algorithm (Wright 1922):
/// Kinship = Sum over common ancestors: (1/2)^(La + Lb + 1) * (1 + F_ancestor)
///
/// Capped at 10 generations back for performance.
/// Returns 0.0 if no common ancestors found within the search depth.
pub fn kinship_from_pedigree(pedigree: &Pedigree, id_a: u64, id_b: u64) -> f64 {
    const MAX_DEPTH: u32 = 10;

    // Self-kinship = 0.5 * (1 + F)
    if id_a == id_b {
        return 0.5;
    }

    // Build ancestor maps with depths
    let ancestors_a = find_ancestors(pedigree, id_a, MAX_DEPTH);
    let ancestors_b = find_ancestors(pedigree, id_b, MAX_DEPTH);

    // Find common ancestors
    let mut kinship = 0.0;
    for (ancestor_id, depth_a) in &ancestors_a {
        if let Some(depth_b) = ancestors_b.get(ancestor_id) {
            let path_length = depth_a + depth_b + 1;
            // (1/2)^(La + Lb + 1) — simplified: no F_ancestor correction for founders
            kinship += 0.5_f64.powi(path_length as i32);
        }
    }

    kinship
}

/// Find all ancestors of an individual up to max_depth generations back.
///
/// Returns a map of ancestor_id -> depth (number of generations).
fn find_ancestors(pedigree: &Pedigree, id: u64, max_depth: u32) -> HashMap<u64, u32> {
    let mut ancestors = HashMap::new();
    let mut queue: Vec<(u64, u32)> = vec![(id, 0)];

    while let Some((current_id, depth)) = queue.pop() {
        if depth > max_depth {
            continue;
        }
        ancestors.entry(current_id).or_insert(depth);

        if let Some(entry) = pedigree.find(current_id) {
            if let Some(parent_a) = entry.parent_a_id {
                if depth < max_depth {
                    queue.push((parent_a, depth + 1));
                }
            }
            if let Some(parent_b) = entry.parent_b_id {
                if depth < max_depth {
                    queue.push((parent_b, depth + 1));
                }
            }
        }
    }

    ancestors
}

/// Compute the full N x N kinship matrix for a population.
///
/// Uses pedigree-based kinship. Matrix is symmetric.
pub fn kinship_matrix(population: &Population, pedigree: &Pedigree) -> Vec<Vec<f64>> {
    let n = population.individuals.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        let id_i = population.individuals[i].id;
        matrix[i][i] = kinship_from_pedigree(pedigree, id_i, id_i);
        for j in (i + 1)..n {
            let id_j = population.individuals[j].id;
            let k = kinship_from_pedigree(pedigree, id_i, id_j);
            matrix[i][j] = k;
            matrix[j][i] = k;
        }
    }

    matrix
}

/// Fast HDC-based kinship estimate.
///
/// Higher genome_hv similarity approximates higher kinship.
/// Clamped to [0, 1] range. This is an estimate, not exact kinship.
pub fn hdc_kinship_estimate(a: &Individual, b: &Individual) -> f64 {
    (a.genome_hv.similarity(&b.genome_hv).max(0.0) as f64).min(1.0)
}

/// Check if a kinship value is below the safe threshold.
///
/// Common thresholds:
/// - 0.0625 = first cousin threshold
/// - 0.125 = half-sibling threshold
/// - 0.25 = full-sibling threshold
pub fn is_inbreeding_safe(kinship: f64, threshold: f64) -> bool {
    kinship < threshold
}

/// Mean kinship of an individual with the rest of the population.
///
/// Used by minimum kinship breeding strategy (Ballou & Lacy 1995).
pub fn mean_kinship(individual_idx: usize, kinship_mat: &[Vec<f64>]) -> f64 {
    let n = kinship_mat.len();
    if n <= 1 {
        return 0.0;
    }
    let sum: f64 = kinship_mat[individual_idx].iter().sum();
    // Subtract self-kinship and divide by (n-1)
    (sum - kinship_mat[individual_idx][individual_idx]) / (n - 1) as f64
}

/// Population mean kinship: average of all pairwise kinship coefficients.
pub fn population_mean_kinship(kinship_mat: &[Vec<f64>]) -> f64 {
    let n = kinship_mat.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            total += kinship_mat[i][j];
            count += 1;
        }
    }
    total / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BiologicalSex, PedigreeEntry};
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

    fn make_test_individual(id: u64, sex: BiologicalSex, seed: u64) -> Individual {
        Individual {
            id,
            sex,
            genotypes: vec![],
            genome_hv: ContinuousHV::random(HDC_DIMENSION, seed),
            generation: 0,
            parent_ids: (None, None),
            fitness: 1.0,
        }
    }

    fn make_simple_pedigree() -> Pedigree {
        // Founders: 1 (F), 2 (M), 3 (F), 4 (M)
        // Gen 1: 5 = 1x2, 6 = 3x4
        // Gen 2: 7 = 5x6 (common grandparents: none -> kinship=0)
        let mut ped = Pedigree::new();
        ped.add_entry(PedigreeEntry {
            individual_id: 1,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 2,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 3,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 4,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 5,
            parent_a_id: Some(1),
            parent_b_id: Some(2),
            generation: 1,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 6,
            parent_a_id: Some(3),
            parent_b_id: Some(4),
            generation: 1,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 7,
            parent_a_id: Some(5),
            parent_b_id: Some(6),
            generation: 2,
        });
        ped
    }

    fn make_inbred_pedigree() -> Pedigree {
        // Founders: 1 (F), 2 (M)
        // Gen 1: 3 = 1x2, 4 = 1x2 (full siblings)
        // Gen 2: 5 = 3x4 (offspring of siblings)
        let mut ped = Pedigree::new();
        ped.add_entry(PedigreeEntry {
            individual_id: 1,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 2,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 3,
            parent_a_id: Some(1),
            parent_b_id: Some(2),
            generation: 1,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 4,
            parent_a_id: Some(1),
            parent_b_id: Some(2),
            generation: 1,
        });
        ped.add_entry(PedigreeEntry {
            individual_id: 5,
            parent_a_id: Some(3),
            parent_b_id: Some(4),
            generation: 2,
        });
        ped
    }

    #[test]
    fn test_inbreeding_coefficient_no_deficit() {
        let f = inbreeding_coefficient_from_heterozygosity(0.5, 0.5);
        assert!(f.abs() < 1e-10, "no deficit: F=0, got {f}");
    }

    #[test]
    fn test_inbreeding_coefficient_complete() {
        let f = inbreeding_coefficient_from_heterozygosity(0.5, 0.0);
        assert!((f - 1.0).abs() < 1e-10, "complete inbreeding: F=1, got {f}");
    }

    #[test]
    fn test_inbreeding_coefficient_partial() {
        let f = inbreeding_coefficient_from_heterozygosity(0.5, 0.25);
        assert!((f - 0.5).abs() < 1e-10, "partial: F=0.5, got {f}");
    }

    #[test]
    fn test_inbreeding_coefficient_monomorphic() {
        let f = inbreeding_coefficient_from_heterozygosity(0.0, 0.0);
        assert_eq!(f, 0.0, "monomorphic should return 0");
    }

    #[test]
    fn test_inbreeding_coefficient_in_range() {
        for he in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for ho_frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let ho = he * ho_frac;
                let f = inbreeding_coefficient_from_heterozygosity(he, ho);
                assert!(
                    f >= -0.01 && f <= 1.01,
                    "F should be in [0,1], got {f} for He={he}, Ho={ho}"
                );
            }
        }
    }

    #[test]
    fn test_self_kinship() {
        let ped = make_simple_pedigree();
        let k = kinship_from_pedigree(&ped, 1, 1);
        assert!((k - 0.5).abs() < 1e-10, "self-kinship = 0.5, got {k}");
    }

    #[test]
    fn test_parent_offspring_kinship() {
        let ped = make_simple_pedigree();
        let k = kinship_from_pedigree(&ped, 1, 5);
        // Parent-offspring: (1/2)^(0+1+1) = 0.25
        assert!(
            (k - 0.25).abs() < 1e-10,
            "parent-offspring kinship = 0.25, got {k}"
        );
    }

    #[test]
    fn test_unrelated_kinship() {
        let ped = make_simple_pedigree();
        let k = kinship_from_pedigree(&ped, 1, 3);
        assert!(k.abs() < 1e-10, "unrelated founders kinship = 0, got {k}");
    }

    #[test]
    fn test_sibling_kinship() {
        let ped = make_inbred_pedigree();
        let k = kinship_from_pedigree(&ped, 3, 4);
        // Full siblings share both parents (1,2):
        // Path through parent 1: depth_a=1, depth_b=1 => (1/2)^3 = 0.125
        // Path through parent 2: depth_a=1, depth_b=1 => (1/2)^3 = 0.125
        // Total = 0.25
        assert!(
            (k - 0.25).abs() < 1e-10,
            "full sibling kinship = 0.25, got {k}"
        );
    }

    #[test]
    fn test_kinship_symmetry() {
        let ped = make_simple_pedigree();
        let k_ab = kinship_from_pedigree(&ped, 1, 5);
        let k_ba = kinship_from_pedigree(&ped, 5, 1);
        assert!(
            (k_ab - k_ba).abs() < 1e-10,
            "kinship should be symmetric: {k_ab} vs {k_ba}"
        );
    }

    #[test]
    fn test_kinship_matrix_symmetric() {
        let ped = make_simple_pedigree();
        let pop = Population {
            individuals: vec![
                make_test_individual(1, BiologicalSex::Female, 100),
                make_test_individual(2, BiologicalSex::Male, 200),
                make_test_individual(5, BiologicalSex::Female, 500),
            ],
            generation: 0,
            founding_size: 3,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let km = kinship_matrix(&pop, &ped);
        assert_eq!(km.len(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (km[i][j] - km[j][i]).abs() < 1e-10,
                    "kinship matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_kinship_matrix_diagonal() {
        let ped = make_simple_pedigree();
        let pop = Population {
            individuals: vec![
                make_test_individual(1, BiologicalSex::Female, 100),
                make_test_individual(2, BiologicalSex::Male, 200),
            ],
            generation: 0,
            founding_size: 2,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let km = kinship_matrix(&pop, &ped);
        assert!((km[0][0] - 0.5).abs() < 1e-10, "self-kinship = 0.5");
        assert!((km[1][1] - 0.5).abs() < 1e-10, "self-kinship = 0.5");
    }

    #[test]
    fn test_hdc_kinship_estimate_self() {
        let ind = make_test_individual(1, BiologicalSex::Female, 42);
        let k = hdc_kinship_estimate(&ind, &ind);
        assert!(
            (k - 1.0).abs() < 0.01,
            "self HDC kinship should be ~1.0, got {k}"
        );
    }

    #[test]
    fn test_hdc_kinship_estimate_unrelated() {
        let a = make_test_individual(1, BiologicalSex::Female, 42);
        let b = make_test_individual(2, BiologicalSex::Male, 99999);
        let k = hdc_kinship_estimate(&a, &b);
        assert!(k < 0.2, "unrelated HDC kinship should be low, got {k}");
    }

    #[test]
    fn test_is_inbreeding_safe() {
        assert!(is_inbreeding_safe(0.05, 0.0625));
        assert!(!is_inbreeding_safe(0.07, 0.0625));
        assert!(is_inbreeding_safe(0.0, 0.0625));
    }

    #[test]
    fn test_mean_kinship() {
        let km = vec![
            vec![0.5, 0.1, 0.2],
            vec![0.1, 0.5, 0.15],
            vec![0.2, 0.15, 0.5],
        ];
        let mk = mean_kinship(0, &km);
        // (0.1 + 0.2) / 2 = 0.15
        assert!(
            (mk - 0.15).abs() < 1e-10,
            "mean kinship of 0 = 0.15, got {mk}"
        );
    }

    #[test]
    fn test_population_mean_kinship() {
        let km = vec![
            vec![0.5, 0.1, 0.2],
            vec![0.1, 0.5, 0.15],
            vec![0.2, 0.15, 0.5],
        ];
        let pmk = population_mean_kinship(&km);
        // (0.1 + 0.2 + 0.15) / 3 = 0.15
        assert!(
            (pmk - 0.15).abs() < 1e-10,
            "pop mean kinship = 0.15, got {pmk}"
        );
    }

    #[test]
    fn test_population_mean_kinship_singleton() {
        let km = vec![vec![0.5]];
        assert_eq!(population_mean_kinship(&km), 0.0);
    }
}
