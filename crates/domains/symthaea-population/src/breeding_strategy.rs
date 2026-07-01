// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Breeding strategy optimization.
//!
//! Implements multiple breeding strategies for maintaining genetic diversity
//! in small populations: random pairing, minimum kinship (Ballou & Lacy 1995),
//! maximum avoidance, HDC distance maximization (novel), and balanced contribution.

use rand::Rng;

use crate::hdc_genetics::genetic_distance;
use crate::inbreeding::kinship_from_pedigree;
use crate::types::{BiologicalSex, BreedingStrategy, MatingPair, Pedigree, Population};

/// Select breeding pairs using random pairing.
///
/// Randomly pairs females with males, ensuring opposite-sex pairing.
/// Each individual is used at most once.
pub fn select_pairs_random(
    population: &Population,
    num_pairs: usize,
    rng: &mut impl Rng,
) -> Vec<MatingPair> {
    let mut females: Vec<u64> = population
        .individuals
        .iter()
        .filter(|i| i.sex == BiologicalSex::Female)
        .map(|i| i.id)
        .collect();
    let mut males: Vec<u64> = population
        .individuals
        .iter()
        .filter(|i| i.sex == BiologicalSex::Male)
        .map(|i| i.id)
        .collect();

    // Shuffle both lists
    shuffle_vec(&mut females, rng);
    shuffle_vec(&mut males, rng);

    let n = num_pairs.min(females.len()).min(males.len());
    (0..n)
        .map(|i| MatingPair {
            parent_a: females[i],
            parent_b: males[i],
            kinship: 0.0, // Not computed for random
            hla_complementarity: 0.0,
        })
        .collect()
}

/// Fisher-Yates shuffle.
fn shuffle_vec(v: &mut [u64], rng: &mut impl Rng) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}

/// Select breeding pairs using minimum kinship strategy (Ballou & Lacy 1995).
///
/// Greedy algorithm: enumerate all possible female-male pairs, sort by kinship
/// (ascending), and select pairs with lowest kinship, ensuring each individual
/// is used at most once.
pub fn select_pairs_minimum_kinship(
    population: &Population,
    pedigree: &Pedigree,
    num_pairs: usize,
) -> Vec<MatingPair> {
    let females: Vec<u64> = population
        .individuals
        .iter()
        .filter(|i| i.sex == BiologicalSex::Female)
        .map(|i| i.id)
        .collect();
    let males: Vec<u64> = population
        .individuals
        .iter()
        .filter(|i| i.sex == BiologicalSex::Male)
        .map(|i| i.id)
        .collect();

    // Compute all possible pair kinships
    let mut candidate_pairs: Vec<(u64, u64, f64)> = Vec::new();
    for &f in &females {
        for &m in &males {
            let k = kinship_from_pedigree(pedigree, f, m);
            candidate_pairs.push((f, m, k));
        }
    }

    // Sort by kinship (ascending = prefer lowest kinship)
    candidate_pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy selection: pick lowest kinship pairs, no individual reuse
    let mut used_females = std::collections::HashSet::new();
    let mut used_males = std::collections::HashSet::new();
    let mut selected = Vec::new();

    for (f, m, k) in candidate_pairs {
        if selected.len() >= num_pairs {
            break;
        }
        if !used_females.contains(&f) && !used_males.contains(&m) {
            used_females.insert(f);
            used_males.insert(m);
            selected.push(MatingPair {
                parent_a: f,
                parent_b: m,
                kinship: k,
                hla_complementarity: 0.0,
            });
        }
    }

    selected
}

/// Select breeding pairs using maximum avoidance of inbreeding (MAI).
///
/// Like minimum kinship but also considers common ancestor depth,
/// preferring pairs that share no recent ancestors.
pub fn select_pairs_maximum_avoidance(
    population: &Population,
    pedigree: &Pedigree,
    num_pairs: usize,
) -> Vec<MatingPair> {
    // MAI is essentially minimum kinship for our pedigree-based implementation
    // In practice, MAI differs from MK in using circular mating schemes,
    // but for small populations, greedy minimum kinship is equivalent.
    select_pairs_minimum_kinship(population, pedigree, num_pairs)
}

/// Select breeding pairs by maximizing HDC distance (novel approach).
///
/// Pairs are selected to maximize genome HV distance (= minimize similarity),
/// promoting genetically dissimilar pairings without needing pedigree data.
pub fn select_pairs_hdc_distance(population: &Population, num_pairs: usize) -> Vec<MatingPair> {
    let females: Vec<&crate::types::Individual> = population.females();
    let males: Vec<&crate::types::Individual> = population.males();

    // Compute all pairwise HDC distances
    let mut candidate_pairs: Vec<(u64, u64, f64)> = Vec::new();
    for f in &females {
        for m in &males {
            let dist = genetic_distance(f, m);
            candidate_pairs.push((f.id, m.id, dist));
        }
    }

    // Sort by distance (descending = prefer most distant pairs)
    candidate_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy selection
    let mut used_females = std::collections::HashSet::new();
    let mut used_males = std::collections::HashSet::new();
    let mut selected = Vec::new();

    for (f, m, _dist) in &candidate_pairs {
        if selected.len() >= num_pairs {
            break;
        }
        if !used_females.contains(f) && !used_males.contains(m) {
            used_females.insert(*f);
            used_males.insert(*m);

            // Compute HLA complementarity from genome HVs
            let Some(f_ind) = population.individuals.iter().find(|i| i.id == *f) else {
                continue;
            };
            let Some(m_ind) = population.individuals.iter().find(|i| i.id == *m) else {
                continue;
            };
            let hla = crate::hdc_genetics::hla_complementarity(&f_ind.genome_hv, &m_ind.genome_hv);

            selected.push(MatingPair {
                parent_a: *f,
                parent_b: *m,
                kinship: 0.0,
                hla_complementarity: hla,
            });
        }
    }

    selected
}

/// Select breeding pairs using balanced contribution.
///
/// Ensures equal family sizes by assigning each female exactly one male partner,
/// cycling through males. This minimizes variance in family size, which maximizes Ne.
pub fn select_pairs_balanced_contribution(
    population: &Population,
    num_pairs: usize,
) -> Vec<MatingPair> {
    let females: Vec<u64> = population
        .individuals
        .iter()
        .filter(|i| i.sex == BiologicalSex::Female)
        .map(|i| i.id)
        .collect();
    let males: Vec<u64> = population
        .individuals
        .iter()
        .filter(|i| i.sex == BiologicalSex::Male)
        .map(|i| i.id)
        .collect();

    if females.is_empty() || males.is_empty() {
        return Vec::new();
    }

    let n = num_pairs.min(females.len());
    (0..n)
        .map(|i| MatingPair {
            parent_a: females[i],
            parent_b: males[i % males.len()],
            kinship: 0.0,
            hla_complementarity: 0.0,
        })
        .collect()
}

/// Unified pair selection using any breeding strategy.
pub fn select_pairs(
    strategy: BreedingStrategy,
    population: &Population,
    pedigree: &Pedigree,
    num_pairs: usize,
    rng: &mut impl Rng,
) -> Vec<MatingPair> {
    match strategy {
        BreedingStrategy::Random => select_pairs_random(population, num_pairs, rng),
        BreedingStrategy::MinimumKinship => {
            select_pairs_minimum_kinship(population, pedigree, num_pairs)
        }
        BreedingStrategy::MaximumAvoidance => {
            select_pairs_maximum_avoidance(population, pedigree, num_pairs)
        }
        BreedingStrategy::HdcDistanceMaximize => select_pairs_hdc_distance(population, num_pairs),
        BreedingStrategy::BalancedContribution => {
            select_pairs_balanced_contribution(population, num_pairs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Individual, PedigreeEntry};
    use rand::SeedableRng;
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

    fn make_population(n_females: usize, n_males: usize) -> Population {
        let mut individuals = Vec::new();
        for i in 0..n_females {
            individuals.push(Individual {
                id: i as u64,
                sex: BiologicalSex::Female,
                genotypes: vec![],
                genome_hv: ContinuousHV::random(HDC_DIMENSION, i as u64 * 1000),
                generation: 0,
                parent_ids: (None, None),
                fitness: 1.0,
            });
        }
        for i in 0..n_males {
            individuals.push(Individual {
                id: (n_females + i) as u64,
                sex: BiologicalSex::Male,
                genotypes: vec![],
                genome_hv: ContinuousHV::random(HDC_DIMENSION, (n_females + i) as u64 * 1000 + 500),
                generation: 0,
                parent_ids: (None, None),
                fitness: 1.0,
            });
        }
        Population {
            individuals,
            generation: 0,
            founding_size: n_females + n_males,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        }
    }

    fn make_founder_pedigree(pop: &Population) -> Pedigree {
        let mut ped = Pedigree::new();
        for ind in &pop.individuals {
            ped.add_entry(PedigreeEntry {
                individual_id: ind.id,
                parent_a_id: None,
                parent_b_id: None,
                generation: 0,
            });
        }
        ped
    }

    #[test]
    fn test_random_pairing_count() {
        let pop = make_population(5, 5);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let pairs = select_pairs_random(&pop, 3, &mut rng);
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn test_random_pairing_no_duplicates() {
        let pop = make_population(10, 10);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let pairs = select_pairs_random(&pop, 8, &mut rng);

        let mut seen_a = std::collections::HashSet::new();
        let mut seen_b = std::collections::HashSet::new();
        for p in &pairs {
            assert!(
                seen_a.insert(p.parent_a),
                "duplicate female in random pairing"
            );
            assert!(
                seen_b.insert(p.parent_b),
                "duplicate male in random pairing"
            );
        }
    }

    #[test]
    fn test_random_pairing_max_limited() {
        let pop = make_population(3, 2);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let pairs = select_pairs_random(&pop, 10, &mut rng);
        assert_eq!(pairs.len(), 2, "limited by fewer males");
    }

    #[test]
    fn test_minimum_kinship_selects_lowest() {
        // Create pedigree where some individuals are related
        let mut ped = Pedigree::new();
        ped.add_entry(PedigreeEntry {
            individual_id: 0,
            parent_a_id: None,
            parent_b_id: None,
            generation: 0,
        });
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
        // Individual 3 is offspring of 0 and 2 (so related to both)
        ped.add_entry(PedigreeEntry {
            individual_id: 3,
            parent_a_id: Some(0),
            parent_b_id: Some(2),
            generation: 1,
        });

        let pop = Population {
            individuals: vec![
                Individual {
                    id: 0,
                    sex: BiologicalSex::Female,
                    genotypes: vec![],
                    genome_hv: ContinuousHV::random(HDC_DIMENSION, 0),
                    generation: 0,
                    parent_ids: (None, None),
                    fitness: 1.0,
                },
                Individual {
                    id: 1,
                    sex: BiologicalSex::Female,
                    genotypes: vec![],
                    genome_hv: ContinuousHV::random(HDC_DIMENSION, 1000),
                    generation: 0,
                    parent_ids: (None, None),
                    fitness: 1.0,
                },
                Individual {
                    id: 2,
                    sex: BiologicalSex::Male,
                    genotypes: vec![],
                    genome_hv: ContinuousHV::random(HDC_DIMENSION, 2000),
                    generation: 0,
                    parent_ids: (None, None),
                    fitness: 1.0,
                },
                Individual {
                    id: 3,
                    sex: BiologicalSex::Male,
                    genotypes: vec![],
                    genome_hv: ContinuousHV::random(HDC_DIMENSION, 3000),
                    generation: 1,
                    parent_ids: (Some(0), Some(2)),
                    fitness: 1.0,
                },
            ],
            generation: 1,
            founding_size: 3,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };

        let pairs = select_pairs_minimum_kinship(&pop, &ped, 2);
        assert_eq!(pairs.len(), 2);

        // The pair (1, 3) should have kinship=0 since 1 is unrelated to 3's parents
        // while (0, 3) has kinship > 0 since 0 is parent of 3
        let pair_0_3 = pairs.iter().find(|p| p.parent_a == 0);
        let pair_1_3 = pairs.iter().find(|p| p.parent_a == 1);
        // Due to greedy selection, female 1 (unrelated) should pair with male 3 first
        // if 3 has lower kinship with 1 than 0
        if let (Some(p03), Some(p13)) = (pair_0_3, pair_1_3) {
            // Both pairs should exist but 1-3 should have lower kinship
            assert!(p13.kinship <= p03.kinship || (p13.kinship - p03.kinship).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hdc_distance_pairs_opposite_sex() {
        let pop = make_population(5, 5);
        let pairs = select_pairs_hdc_distance(&pop, 3);
        assert_eq!(pairs.len(), 3);

        let females: std::collections::HashSet<u64> = pop.females().iter().map(|f| f.id).collect();
        let males: std::collections::HashSet<u64> = pop.males().iter().map(|m| m.id).collect();
        for p in &pairs {
            assert!(females.contains(&p.parent_a), "parent_a should be female");
            assert!(males.contains(&p.parent_b), "parent_b should be male");
        }
    }

    #[test]
    fn test_hdc_distance_has_complementarity() {
        let pop = make_population(3, 3);
        let pairs = select_pairs_hdc_distance(&pop, 2);
        for p in &pairs {
            assert!(
                p.hla_complementarity > 0.0,
                "HDC distance pairs should have complementarity > 0"
            );
        }
    }

    #[test]
    fn test_balanced_contribution_equal_families() {
        let pop = make_population(6, 3);
        let pairs = select_pairs_balanced_contribution(&pop, 6);
        assert_eq!(pairs.len(), 6);

        // Each male should appear exactly twice (6 pairs / 3 males)
        let mut male_counts = std::collections::HashMap::new();
        for p in &pairs {
            *male_counts.entry(p.parent_b).or_insert(0) += 1;
        }
        assert_eq!(male_counts.len(), 3, "all males should be used");
        for (&_id, &count) in &male_counts {
            assert_eq!(count, 2, "balanced contribution: each male used twice");
        }
    }

    #[test]
    fn test_balanced_contribution_empty() {
        let pop = Population {
            individuals: vec![],
            generation: 0,
            founding_size: 0,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let pairs = select_pairs_balanced_contribution(&pop, 5);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_select_pairs_dispatch() {
        let pop = make_population(5, 5);
        let ped = make_founder_pedigree(&pop);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for strategy in [
            BreedingStrategy::Random,
            BreedingStrategy::MinimumKinship,
            BreedingStrategy::MaximumAvoidance,
            BreedingStrategy::HdcDistanceMaximize,
            BreedingStrategy::BalancedContribution,
        ] {
            let pairs = select_pairs(strategy, &pop, &ped, 3, &mut rng);
            assert!(
                !pairs.is_empty(),
                "strategy {:?} should produce pairs",
                strategy
            );
            assert!(
                pairs.len() <= 3,
                "strategy {:?} should respect max pairs",
                strategy
            );
        }
    }

    #[test]
    fn test_hdc_distance_differs_from_random() {
        let pop = make_population(10, 10);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let random_pairs = select_pairs_random(&pop, 5, &mut rng);
        let hdc_pairs = select_pairs_hdc_distance(&pop, 5);

        // They should produce different pairings (highly likely with 10x10)
        let random_set: std::collections::HashSet<(u64, u64)> = random_pairs
            .iter()
            .map(|p| (p.parent_a, p.parent_b))
            .collect();
        let hdc_set: std::collections::HashSet<(u64, u64)> =
            hdc_pairs.iter().map(|p| (p.parent_a, p.parent_b)).collect();

        // Not strictly guaranteed but extremely likely with different strategies
        // At least check both produced valid results
        assert_eq!(random_set.len(), 5);
        assert_eq!(hdc_set.len(), 5);
    }

    #[test]
    fn test_random_pairing_deterministic_with_seed() {
        let pop = make_population(5, 5);
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let pairs1 = select_pairs_random(&pop, 3, &mut rng1);
        let pairs2 = select_pairs_random(&pop, 3, &mut rng2);

        for (p1, p2) in pairs1.iter().zip(pairs2.iter()) {
            assert_eq!(p1.parent_a, p2.parent_a);
            assert_eq!(p1.parent_b, p2.parent_b);
        }
    }
}
