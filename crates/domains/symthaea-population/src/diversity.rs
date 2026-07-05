// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genetic diversity metrics.
//!
//! Provides expected and observed heterozygosity, allelic richness,
//! HDC-based diversity, and lineage counting.

use crate::types::{BiologicalSex, Genotype, Population};

/// Expected heterozygosity from allele frequencies.
///
/// He = 1 - Sum(pi^2) where pi is the frequency of allele i.
///
/// For a biallelic locus with 50/50 frequencies, He = 0.5.
/// All frequencies should sum to 1.0.
pub fn expected_heterozygosity(allele_freqs: &[f64]) -> f64 {
    1.0 - allele_freqs.iter().map(|p| p * p).sum::<f64>()
}

/// Observed heterozygosity from a sample of genotypes.
///
/// Ho = (number of heterozygous individuals) / (total individuals).
/// Returns 0.0 for empty input.
pub fn observed_heterozygosity(genotypes: &[Genotype]) -> f64 {
    if genotypes.is_empty() {
        return 0.0;
    }
    let het_count = genotypes
        .iter()
        .filter(|g| g.allele_a.id != g.allele_b.id)
        .count();
    het_count as f64 / genotypes.len() as f64
}

/// Predict heterozygosity after t generations of drift.
///
/// H(t) = H(0) * (1 - 1/(2*Ne))^t
///
/// Assumes no mutation, selection, or migration.
pub fn heterozygosity_after_generations(h0: f64, ne: f64, generations: u32) -> f64 {
    if ne <= 0.0 {
        return 0.0;
    }
    h0 * (1.0 - 1.0 / (2.0 * ne)).powi(generations as i32)
}

/// Allelic richness: mean number of unique alleles per locus.
///
/// Input: vector of allele ID lists, one per locus.
/// Returns 0.0 for empty input.
pub fn allelic_richness(alleles_per_locus: &[Vec<u64>]) -> f64 {
    if alleles_per_locus.is_empty() {
        return 0.0;
    }
    let total_unique: usize = alleles_per_locus
        .iter()
        .map(|locus_alleles| {
            let mut unique = locus_alleles.clone();
            unique.sort_unstable();
            unique.dedup();
            unique.len()
        })
        .sum();
    total_unique as f64 / alleles_per_locus.len() as f64
}

/// HDC-based diversity: mean pairwise HDC distance across all individuals.
///
/// This is equivalent to `population_diversity` from hdc_genetics but
/// operates directly on the Population struct.
pub fn hdc_diversity(population: &Population) -> f64 {
    let n = population.individuals.len();
    if n < 2 {
        return 0.0;
    }

    let mut total_distance = 0.0;
    let mut pair_count = 0u64;

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = population.individuals[i]
                .genome_hv
                .similarity(&population.individuals[j].genome_hv) as f64;
            total_distance += 1.0 - sim;
            pair_count += 1;
        }
    }

    total_distance / pair_count as f64
}

/// Count unique maternal lineages.
///
/// Traces maternal parent chains to founders and counts distinct founder females.
/// In a simplified model, each founder female defines a lineage.
pub fn mt_lineage_count(population: &Population) -> usize {
    // Count unique founder females (no maternal parent = founder)
    population
        .individuals
        .iter()
        .filter(|i| i.sex == BiologicalSex::Female && i.parent_ids.0.is_none())
        .count()
        .max(
            // Also count females whose maternal parent is not in the population
            {
                let ids: std::collections::HashSet<u64> =
                    population.individuals.iter().map(|i| i.id).collect();
                population
                    .individuals
                    .iter()
                    .filter(|i| {
                        i.sex == BiologicalSex::Female
                            && i.parent_ids.0.is_none_or(|pid| !ids.contains(&pid))
                    })
                    .count()
            },
        )
}

/// Count unique paternal lineages.
///
/// Traces paternal parent chains to founders and counts distinct founder males.
pub fn y_lineage_count(population: &Population) -> usize {
    population
        .individuals
        .iter()
        .filter(|i| i.sex == BiologicalSex::Male && i.parent_ids.1.is_none())
        .count()
        .max({
            let ids: std::collections::HashSet<u64> =
                population.individuals.iter().map(|i| i.id).collect();
            population
                .individuals
                .iter()
                .filter(|i| {
                    i.sex == BiologicalSex::Male
                        && i.parent_ids.1.is_none_or(|pid| !ids.contains(&pid))
                })
                .count()
        })
}

/// Fixation index: proportion of total genetic variance found between sub-populations.
///
/// Fst = (Ht - Hs) / Ht
/// Where Ht = total heterozygosity, Hs = mean subpopulation heterozygosity.
pub fn fixation_index(total_heterozygosity: f64, mean_subpop_heterozygosity: f64) -> f64 {
    if total_heterozygosity <= 0.0 {
        return 0.0;
    }
    (total_heterozygosity - mean_subpop_heterozygosity) / total_heterozygosity
}

/// Nucleotide diversity (pi): average pairwise differences per site.
///
/// pi = (n / (n-1)) * Sum(xi * xj * dij)
/// Simplified here as mean pairwise HDC distance scaled by number of loci.
pub fn nucleotide_diversity(population: &Population, n_loci: usize) -> f64 {
    if n_loci == 0 {
        return 0.0;
    }
    hdc_diversity(population) / n_loci as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Allele, BiologicalSex, Genotype, Individual};
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

    fn make_test_individual(id: u64, sex: BiologicalSex) -> Individual {
        Individual {
            id,
            sex,
            genotypes: vec![],
            genome_hv: ContinuousHV::random(HDC_DIMENSION, id * 1000),
            generation: 0,
            parent_ids: (None, None),
            fitness: 1.0,
        }
    }

    #[test]
    fn test_expected_heterozygosity_biallelic_equal() {
        let he = expected_heterozygosity(&[0.5, 0.5]);
        assert!(
            (he - 0.5).abs() < 1e-10,
            "biallelic 50/50: He=0.5, got {he}"
        );
    }

    #[test]
    fn test_expected_heterozygosity_monomorphic() {
        let he = expected_heterozygosity(&[1.0]);
        assert!(he.abs() < 1e-10, "monomorphic: He=0, got {he}");
    }

    #[test]
    fn test_expected_heterozygosity_triallelic() {
        // Three alleles at equal frequency: He = 1 - 3*(1/3)^2 = 2/3
        let he = expected_heterozygosity(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert!(
            (he - 2.0 / 3.0).abs() < 1e-10,
            "triallelic equal: He=2/3, got {he}"
        );
    }

    #[test]
    fn test_expected_heterozygosity_skewed() {
        let he = expected_heterozygosity(&[0.9, 0.1]);
        // He = 1 - (0.81 + 0.01) = 0.18
        assert!((he - 0.18).abs() < 1e-10, "skewed: He=0.18, got {he}");
    }

    #[test]
    fn test_observed_heterozygosity_all_het() {
        let genotypes = vec![
            Genotype {
                locus_id: 0,
                allele_a: Allele::neutral(1),
                allele_b: Allele::neutral(2),
            },
            Genotype {
                locus_id: 0,
                allele_a: Allele::neutral(3),
                allele_b: Allele::neutral(4),
            },
        ];
        let ho = observed_heterozygosity(&genotypes);
        assert!(
            (ho - 1.0).abs() < 1e-10,
            "all heterozygous: Ho=1.0, got {ho}"
        );
    }

    #[test]
    fn test_observed_heterozygosity_none_het() {
        let genotypes = vec![
            Genotype {
                locus_id: 0,
                allele_a: Allele::neutral(1),
                allele_b: Allele::neutral(1),
            },
            Genotype {
                locus_id: 0,
                allele_a: Allele::neutral(2),
                allele_b: Allele::neutral(2),
            },
        ];
        let ho = observed_heterozygosity(&genotypes);
        assert!(ho.abs() < 1e-10, "all homozygous: Ho=0, got {ho}");
    }

    #[test]
    fn test_observed_heterozygosity_mixed() {
        let genotypes = vec![
            Genotype {
                locus_id: 0,
                allele_a: Allele::neutral(1),
                allele_b: Allele::neutral(2),
            },
            Genotype {
                locus_id: 0,
                allele_a: Allele::neutral(1),
                allele_b: Allele::neutral(1),
            },
        ];
        let ho = observed_heterozygosity(&genotypes);
        assert!((ho - 0.5).abs() < 1e-10, "half het: Ho=0.5, got {ho}");
    }

    #[test]
    fn test_observed_heterozygosity_empty() {
        assert_eq!(observed_heterozygosity(&[]), 0.0);
    }

    #[test]
    fn test_heterozygosity_decay() {
        let h0 = 0.5;
        let ne = 100.0;
        let h_10 = heterozygosity_after_generations(h0, ne, 10);
        let h_100 = heterozygosity_after_generations(h0, ne, 100);
        let h_0 = heterozygosity_after_generations(h0, ne, 0);

        assert!((h_0 - h0).abs() < 1e-10, "t=0 should equal h0");
        assert!(h_10 < h0, "heterozygosity should decay");
        assert!(h_100 < h_10, "more generations = more decay");
        assert!(h_100 > 0.0, "shouldn't be exactly zero");
    }

    #[test]
    fn test_heterozygosity_decay_matches_analytical() {
        let h0 = 0.5;
        let ne = 50.0;
        let t = 20;
        let predicted = heterozygosity_after_generations(h0, ne, t);
        // Direct: H(t) = 0.5 * (1 - 1/100)^20 = 0.5 * 0.99^20
        let expected = 0.5 * 0.99_f64.powi(20);
        assert!(
            (predicted - expected).abs() < 1e-10,
            "should match: predicted={predicted}, expected={expected}"
        );
    }

    #[test]
    fn test_heterozygosity_zero_ne() {
        assert_eq!(heterozygosity_after_generations(0.5, 0.0, 10), 0.0);
    }

    #[test]
    fn test_allelic_richness_basic() {
        let alleles = vec![
            vec![1, 2, 3, 1, 2], // 3 unique
            vec![4, 4, 4, 4],    // 1 unique
            vec![5, 6],          // 2 unique
        ];
        let ar = allelic_richness(&alleles);
        assert!(
            (ar - 2.0).abs() < 1e-10,
            "mean richness = 6/3 = 2.0, got {ar}"
        );
    }

    #[test]
    fn test_allelic_richness_empty() {
        assert_eq!(allelic_richness(&[]), 0.0);
    }

    #[test]
    fn test_allelic_richness_single_locus() {
        let alleles = vec![vec![1, 2, 3, 4, 5]];
        let ar = allelic_richness(&alleles);
        assert!((ar - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_hdc_diversity_empty() {
        let pop = Population {
            individuals: vec![],
            generation: 0,
            founding_size: 0,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(hdc_diversity(&pop), 0.0);
    }

    #[test]
    fn test_hdc_diversity_single() {
        let pop = Population {
            individuals: vec![make_test_individual(1, BiologicalSex::Female)],
            generation: 0,
            founding_size: 1,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(hdc_diversity(&pop), 0.0);
    }

    #[test]
    fn test_hdc_diversity_positive() {
        let pop = Population {
            individuals: vec![
                make_test_individual(1, BiologicalSex::Female),
                make_test_individual(2, BiologicalSex::Male),
                make_test_individual(3, BiologicalSex::Female),
            ],
            generation: 0,
            founding_size: 3,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let d = hdc_diversity(&pop);
        assert!(
            d > 0.0,
            "diverse population should have positive diversity, got {d}"
        );
    }

    #[test]
    fn test_mt_lineage_count() {
        let pop = Population {
            individuals: vec![
                make_test_individual(1, BiologicalSex::Female),
                make_test_individual(2, BiologicalSex::Female),
                make_test_individual(3, BiologicalSex::Male),
            ],
            generation: 0,
            founding_size: 3,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(
            mt_lineage_count(&pop),
            2,
            "2 founder females = 2 mt lineages"
        );
    }

    #[test]
    fn test_y_lineage_count() {
        let pop = Population {
            individuals: vec![
                make_test_individual(1, BiologicalSex::Male),
                make_test_individual(2, BiologicalSex::Male),
                make_test_individual(3, BiologicalSex::Male),
                make_test_individual(4, BiologicalSex::Female),
            ],
            generation: 0,
            founding_size: 4,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(y_lineage_count(&pop), 3, "3 founder males = 3 Y lineages");
    }

    #[test]
    fn test_fixation_index() {
        // No differentiation
        assert!((fixation_index(0.5, 0.5)).abs() < 1e-10);
        // Complete differentiation
        assert!((fixation_index(0.5, 0.0) - 1.0).abs() < 1e-10);
        // Partial
        let fst = fixation_index(0.5, 0.3);
        assert!((fst - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_fixation_index_zero_total() {
        assert_eq!(fixation_index(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_nucleotide_diversity() {
        let pop = Population {
            individuals: vec![
                make_test_individual(1, BiologicalSex::Female),
                make_test_individual(2, BiologicalSex::Male),
            ],
            generation: 0,
            founding_size: 2,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let pi = nucleotide_diversity(&pop, 10);
        assert!(pi > 0.0, "nucleotide diversity should be positive");
        assert!(pi < 1.0, "per-site diversity should be < 1");
    }

    #[test]
    fn test_nucleotide_diversity_zero_loci() {
        let pop = Population {
            individuals: vec![
                make_test_individual(1, BiologicalSex::Female),
                make_test_individual(2, BiologicalSex::Male),
            ],
            generation: 0,
            founding_size: 2,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(nucleotide_diversity(&pop, 0), 0.0);
    }
}
