// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deleterious allele tracking and genetic rescue planning.
//!
//! Tracks the frequency and distribution of deleterious alleles in a population,
//! computes genetic load, purging potential, and plans genetic rescue interventions.

use crate::types::{GeneticRescuePlan, Individual, Population};

/// Count loci where at least one allele is deleterious for an individual.
pub fn count_deleterious(individual: &Individual) -> usize {
    individual
        .genotypes
        .iter()
        .filter(|g| g.allele_a.is_deleterious || g.allele_b.is_deleterious)
        .count()
}

/// Count loci where both alleles are deleterious (homozygous deleterious).
pub fn count_homozygous_deleterious(individual: &Individual) -> usize {
    individual
        .genotypes
        .iter()
        .filter(|g| g.allele_a.is_deleterious && g.allele_b.is_deleterious)
        .count()
}

/// Population genetic load: mean fraction of loci carrying deleterious alleles.
///
/// Load = mean(deleterious_loci / total_loci) across all individuals.
/// Returns 0.0 for empty populations or individuals with no loci.
pub fn population_genetic_load(population: &Population) -> f64 {
    if population.individuals.is_empty() {
        return 0.0;
    }

    let total_load: f64 = population
        .individuals
        .iter()
        .map(|ind| {
            if ind.genotypes.is_empty() {
                0.0
            } else {
                count_deleterious(ind) as f64 / ind.genotypes.len() as f64
            }
        })
        .sum();

    total_load / population.individuals.len() as f64
}

/// Purging potential: fraction of deleterious alleles that are homozygous.
///
/// Homozygous deleterious alleles are exposed to selection and can be purged.
/// Higher purging potential means selection is more effective at removing
/// deleterious alleles (though it also means more inbreeding depression).
///
/// Returns 0.0 if there are no deleterious alleles.
pub fn purging_potential(population: &Population) -> f64 {
    let mut total_deleterious_loci = 0usize;
    let mut homozygous_deleterious_loci = 0usize;

    for ind in &population.individuals {
        for g in &ind.genotypes {
            if g.has_deleterious() {
                total_deleterious_loci += 1;
                if g.is_homozygous_deleterious() {
                    homozygous_deleterious_loci += 1;
                }
            }
        }
    }

    if total_deleterious_loci == 0 {
        return 0.0;
    }

    homozygous_deleterious_loci as f64 / total_deleterious_loci as f64
}

/// Mean selection coefficient of deleterious alleles in the population.
///
/// Returns 0.0 if no deleterious alleles exist.
pub fn mean_deleterious_effect(population: &Population) -> f64 {
    let mut total_s = 0.0;
    let mut count = 0usize;

    for ind in &population.individuals {
        for g in &ind.genotypes {
            if g.allele_a.is_deleterious {
                total_s += g.allele_a.selection_coefficient.abs();
                count += 1;
            }
            if g.allele_b.is_deleterious {
                total_s += g.allele_b.selection_coefficient.abs();
                count += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        total_s / count as f64
    }
}

/// Compute individual fitness from genotype selection coefficients.
///
/// Multiplicative fitness model:
/// w = Product over loci: 1 + s_het (heterozygous) or 1 + s_hom (homozygous deleterious)
///
/// Where:
/// - s_het = h * s (heterozygous effect, h = dominance, s = selection coefficient)
/// - s_hom = s (homozygous deleterious effect)
pub fn compute_fitness(individual: &Individual) -> f64 {
    let mut fitness = 1.0;

    for g in &individual.genotypes {
        if g.is_homozygous_deleterious() {
            // Both alleles deleterious: full effect
            fitness *= 1.0 + g.allele_a.selection_coefficient;
        } else if g.allele_a.is_deleterious {
            // Only allele_a deleterious: heterozygous effect
            fitness *= 1.0 + g.allele_a.dominance * g.allele_a.selection_coefficient;
        } else if g.allele_b.is_deleterious {
            // Only allele_b deleterious: heterozygous effect
            fitness *= 1.0 + g.allele_b.dominance * g.allele_b.selection_coefficient;
        }
    }

    fitness.max(0.0)
}

/// Plan genetic rescue to restore diversity via migration.
///
/// Estimates the number of migrants needed based on the one-migrant-per-generation
/// rule (Mills & Allendorf 1996) and target heterozygosity.
///
/// The formula for heterozygosity gain from m migrants per generation:
/// delta_H ~ 2m * (H_source - H_local) / N
pub fn plan_genetic_rescue(
    population: &Population,
    target_heterozygosity: f64,
    current_heterozygosity: f64,
) -> GeneticRescuePlan {
    let deficit = target_heterozygosity - current_heterozygosity;
    let n = population.size() as f64;

    // Estimate migrants needed: m = deficit * N / (2 * (H_source - H_local))
    // Assume source population has high diversity (H_source = 0.5)
    let h_source = 0.5_f64.max(target_heterozygosity);
    let h_diff = h_source - current_heterozygosity;

    let num_migrants = if h_diff <= 0.0 || deficit <= 0.0 {
        1 // Minimum one migrant as genetic rescue
    } else {
        ((deficit * n) / (2.0 * h_diff)).ceil() as usize
    }
    .max(1);

    GeneticRescuePlan {
        source_population: "external_rescue_population".to_string(),
        num_migrants,
        target_generation: population.generation + 1,
        expected_heterozygosity_gain: deficit.max(0.0),
    }
}

/// Number of generations until deleterious alleles are expected to fix.
///
/// For a neutral allele in a population of Ne: ~4*Ne generations.
/// For a deleterious allele with selection: faster if purging is effective.
pub fn generations_to_fixation(ne: f64, selection_coefficient: f64) -> f64 {
    if selection_coefficient.abs() < 1e-10 {
        // Neutral: fixation time = ~4*Ne
        4.0 * ne
    } else {
        // With selection: approximate as 2*Ne*|ln(2*Ne*|s|)| / |s|
        let s_abs = selection_coefficient.abs();
        let nne_s = 2.0 * ne * s_abs;
        if nne_s < 1.0 {
            // Effectively neutral
            4.0 * ne
        } else {
            (2.0 * ne * nne_s.ln()) / s_abs
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Allele, BiologicalSex, Genotype};
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

    fn make_individual_with_genotypes(id: u64, genotypes: Vec<Genotype>) -> Individual {
        Individual {
            id,
            sex: BiologicalSex::Female,
            genotypes,
            genome_hv: ContinuousHV::zero(HDC_DIMENSION),
            generation: 0,
            parent_ids: (None, None),
            fitness: 1.0,
        }
    }

    #[test]
    fn test_count_deleterious_none() {
        let ind = make_individual_with_genotypes(
            1,
            vec![
                Genotype {
                    locus_id: 0,
                    allele_a: Allele::neutral(1),
                    allele_b: Allele::neutral(2),
                },
                Genotype {
                    locus_id: 1,
                    allele_a: Allele::neutral(3),
                    allele_b: Allele::neutral(4),
                },
            ],
        );
        assert_eq!(count_deleterious(&ind), 0);
    }

    #[test]
    fn test_count_deleterious_some() {
        let ind = make_individual_with_genotypes(
            1,
            vec![
                Genotype {
                    locus_id: 0,
                    allele_a: Allele::neutral(1),
                    allele_b: Allele::deleterious(2, -0.1, 0.5),
                },
                Genotype {
                    locus_id: 1,
                    allele_a: Allele::neutral(3),
                    allele_b: Allele::neutral(4),
                },
                Genotype {
                    locus_id: 2,
                    allele_a: Allele::deleterious(5, -0.05, 0.3),
                    allele_b: Allele::deleterious(6, -0.05, 0.3),
                },
            ],
        );
        assert_eq!(count_deleterious(&ind), 2);
    }

    #[test]
    fn test_count_homozygous_deleterious() {
        let ind = make_individual_with_genotypes(
            1,
            vec![
                Genotype {
                    locus_id: 0,
                    allele_a: Allele::deleterious(1, -0.1, 0.5),
                    allele_b: Allele::deleterious(2, -0.1, 0.5),
                },
                Genotype {
                    locus_id: 1,
                    allele_a: Allele::neutral(3),
                    allele_b: Allele::deleterious(4, -0.05, 0.3),
                },
            ],
        );
        assert_eq!(count_homozygous_deleterious(&ind), 1);
    }

    #[test]
    fn test_population_genetic_load_zero() {
        let pop = Population {
            individuals: vec![make_individual_with_genotypes(
                1,
                vec![Genotype {
                    locus_id: 0,
                    allele_a: Allele::neutral(1),
                    allele_b: Allele::neutral(2),
                }],
            )],
            generation: 0,
            founding_size: 1,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(population_genetic_load(&pop), 0.0);
    }

    #[test]
    fn test_population_genetic_load_full() {
        let pop = Population {
            individuals: vec![make_individual_with_genotypes(
                1,
                vec![Genotype {
                    locus_id: 0,
                    allele_a: Allele::deleterious(1, -0.1, 0.5),
                    allele_b: Allele::neutral(2),
                }],
            )],
            generation: 0,
            founding_size: 1,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert!((population_genetic_load(&pop) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_population_genetic_load_empty() {
        let pop = Population {
            individuals: vec![],
            generation: 0,
            founding_size: 0,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(population_genetic_load(&pop), 0.0);
    }

    #[test]
    fn test_population_genetic_load_increases_with_frequency() {
        let make_pop = |n_del: usize, n_total: usize| {
            let genotypes: Vec<Genotype> = (0..n_total)
                .map(|i| Genotype {
                    locus_id: i as u64,
                    allele_a: if i < n_del {
                        Allele::deleterious(i as u64 * 2, -0.1, 0.5)
                    } else {
                        Allele::neutral(i as u64 * 2)
                    },
                    allele_b: Allele::neutral(i as u64 * 2 + 1),
                })
                .collect();
            Population {
                individuals: vec![make_individual_with_genotypes(1, genotypes)],
                generation: 0,
                founding_size: 1,
                diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
            }
        };

        let load_low = population_genetic_load(&make_pop(1, 10));
        let load_high = population_genetic_load(&make_pop(5, 10));
        assert!(
            load_high > load_low,
            "more deleterious alleles = higher load"
        );
    }

    #[test]
    fn test_purging_potential_no_deleterious() {
        let pop = Population {
            individuals: vec![make_individual_with_genotypes(
                1,
                vec![Genotype {
                    locus_id: 0,
                    allele_a: Allele::neutral(1),
                    allele_b: Allele::neutral(2),
                }],
            )],
            generation: 0,
            founding_size: 1,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(purging_potential(&pop), 0.0);
    }

    #[test]
    fn test_purging_potential_all_homozygous() {
        let pop = Population {
            individuals: vec![make_individual_with_genotypes(
                1,
                vec![Genotype {
                    locus_id: 0,
                    allele_a: Allele::deleterious(1, -0.1, 0.5),
                    allele_b: Allele::deleterious(2, -0.1, 0.5),
                }],
            )],
            generation: 0,
            founding_size: 1,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert!(
            (purging_potential(&pop) - 1.0).abs() < 1e-10,
            "all homozygous deleterious = purging potential 1.0"
        );
    }

    #[test]
    fn test_purging_potential_heterozygous() {
        let pop = Population {
            individuals: vec![make_individual_with_genotypes(
                1,
                vec![Genotype {
                    locus_id: 0,
                    allele_a: Allele::neutral(1),
                    allele_b: Allele::deleterious(2, -0.1, 0.5),
                }],
            )],
            generation: 0,
            founding_size: 1,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        assert_eq!(
            purging_potential(&pop),
            0.0,
            "heterozygous deleterious = purging potential 0"
        );
    }

    #[test]
    fn test_compute_fitness_no_deleterious() {
        let ind = make_individual_with_genotypes(
            1,
            vec![
                Genotype {
                    locus_id: 0,
                    allele_a: Allele::neutral(1),
                    allele_b: Allele::neutral(2),
                },
                Genotype {
                    locus_id: 1,
                    allele_a: Allele::neutral(3),
                    allele_b: Allele::neutral(4),
                },
            ],
        );
        assert!((compute_fitness(&ind) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_fitness_heterozygous_deleterious() {
        let ind = make_individual_with_genotypes(
            1,
            vec![Genotype {
                locus_id: 0,
                allele_a: Allele::deleterious(1, -0.1, 0.5),
                allele_b: Allele::neutral(2),
            }],
        );
        let fitness = compute_fitness(&ind);
        // Expected: 1 + h*s = 1 + 0.5*(-0.1) = 0.95
        assert!(
            (fitness - 0.95).abs() < 1e-10,
            "het fitness = 0.95, got {fitness}"
        );
    }

    #[test]
    fn test_compute_fitness_homozygous_deleterious() {
        let ind = make_individual_with_genotypes(
            1,
            vec![Genotype {
                locus_id: 0,
                allele_a: Allele::deleterious(1, -0.1, 0.5),
                allele_b: Allele::deleterious(2, -0.1, 0.5),
            }],
        );
        let fitness = compute_fitness(&ind);
        // Expected: 1 + s = 1 + (-0.1) = 0.9
        assert!(
            (fitness - 0.9).abs() < 1e-10,
            "hom fitness = 0.9, got {fitness}"
        );
    }

    #[test]
    fn test_compute_fitness_non_negative() {
        let ind = make_individual_with_genotypes(
            1,
            vec![Genotype {
                locus_id: 0,
                allele_a: Allele::deleterious(1, -2.0, 1.0),
                allele_b: Allele::deleterious(2, -2.0, 1.0),
            }],
        );
        let fitness = compute_fitness(&ind);
        assert!(
            fitness >= 0.0,
            "fitness should be non-negative, got {fitness}"
        );
    }

    #[test]
    fn test_mean_deleterious_effect() {
        let pop = Population {
            individuals: vec![make_individual_with_genotypes(
                1,
                vec![Genotype {
                    locus_id: 0,
                    allele_a: Allele::deleterious(1, -0.1, 0.5),
                    allele_b: Allele::deleterious(2, -0.3, 0.5),
                }],
            )],
            generation: 0,
            founding_size: 1,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let mean_s = mean_deleterious_effect(&pop);
        assert!(
            (mean_s - 0.2).abs() < 1e-10,
            "mean |s| = (0.1+0.3)/2 = 0.2, got {mean_s}"
        );
    }

    #[test]
    fn test_plan_genetic_rescue() {
        let pop = Population {
            individuals: (0..50)
                .map(|i| make_individual_with_genotypes(i, vec![]))
                .collect(),
            generation: 5,
            founding_size: 50,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let plan = plan_genetic_rescue(&pop, 0.5, 0.3);
        assert!(
            plan.num_migrants >= 1,
            "should recommend at least 1 migrant"
        );
        assert_eq!(plan.target_generation, 6, "should target next generation");
        assert!(
            plan.expected_heterozygosity_gain > 0.0,
            "should expect positive gain"
        );
    }

    #[test]
    fn test_plan_genetic_rescue_no_deficit() {
        let pop = Population {
            individuals: vec![make_individual_with_genotypes(1, vec![])],
            generation: 0,
            founding_size: 1,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let plan = plan_genetic_rescue(&pop, 0.3, 0.5);
        assert_eq!(plan.num_migrants, 1, "min 1 migrant even with no deficit");
        assert_eq!(plan.expected_heterozygosity_gain, 0.0, "no gain expected");
    }

    #[test]
    fn test_generations_to_fixation_neutral() {
        let fix_time = generations_to_fixation(100.0, 0.0);
        assert!(
            (fix_time - 400.0).abs() < 1e-10,
            "neutral: 4*Ne = 400, got {fix_time}"
        );
    }

    #[test]
    fn test_generations_to_fixation_selected() {
        let fix_neutral = generations_to_fixation(100.0, 0.0);
        let fix_selected = generations_to_fixation(100.0, -0.1);
        // Selected alleles should have different fixation time
        assert!(
            (fix_neutral - fix_selected).abs() > 1.0,
            "selected and neutral should differ"
        );
    }
}
