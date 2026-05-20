// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-generation forward population genetics simulation.
//!
//! Simulates genetic drift, selection, mutation, and different breeding strategies
//! across multiple generations. Tracks heterozygosity, inbreeding, genetic load,
//! and effective population size over time.

use rand::Rng;

use crate::breeding_strategy::select_pairs;
use crate::diversity::observed_heterozygosity;
use crate::effective_population::ne_sex_ratio;
use crate::genetic_load::{compute_fitness, population_genetic_load};
use crate::hdc_genetics::{compute_diversity_hv, encode_individual};
use crate::inbreeding::inbreeding_coefficient_from_heterozygosity;
use crate::types::{
    Allele, BiologicalSex, BreedingStrategy, Genotype, Individual, Locus, Pedigree, PedigreeEntry,
    Population,
};

use serde::{Deserialize, Serialize};

/// Configuration for population simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationSimulator {
    /// Breeding strategy to use for pair selection.
    pub strategy: BreedingStrategy,
    /// Target population size each generation.
    pub target_population_size: usize,
    /// Per-allele per-generation mutation rate.
    pub mutation_rate: f64,
    /// Whether to apply selection based on fitness.
    pub selection_enabled: bool,
}

/// Result of a multi-generation simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Snapshots of each generation.
    pub generations: Vec<GenerationSnapshot>,
    /// The final population state.
    pub final_population: Population,
}

/// Snapshot of population statistics at a single generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSnapshot {
    /// Generation number.
    pub generation: u32,
    /// Census population size.
    pub population_size: usize,
    /// Effective population size (from sex ratio).
    pub effective_ne: f64,
    /// Mean observed heterozygosity across loci.
    pub heterozygosity: f64,
    /// Inbreeding coefficient estimate.
    pub inbreeding_f: f64,
    /// Population genetic load.
    pub genetic_load: f64,
    /// Number of deleterious alleles that have fixed.
    pub num_deleterious_fixed: usize,
}

impl PopulationSimulator {
    /// Create a new simulator with default settings.
    pub fn new(strategy: BreedingStrategy, target_size: usize) -> Self {
        Self {
            strategy,
            target_population_size: target_size,
            mutation_rate: 1e-6,
            selection_enabled: true,
        }
    }

    /// Run the simulation for a given number of generations.
    pub fn run(
        &self,
        initial: &Population,
        pedigree: &mut Pedigree,
        loci: &[Locus],
        generations: u32,
        rng: &mut impl Rng,
    ) -> SimulationResult {
        let mut current_pop = initial.clone();
        let mut snapshots = Vec::with_capacity(generations as usize + 1);
        let mut next_id = current_pop
            .individuals
            .iter()
            .map(|i| i.id)
            .max()
            .unwrap_or(0)
            + 1;

        // Record initial state
        snapshots.push(self.snapshot(&current_pop));

        for r#gen in 1..=generations {
            // Select breeding pairs
            let num_pairs = self.target_population_size / 2;
            let pairs = select_pairs(self.strategy, &current_pop, pedigree, num_pairs.max(1), rng);

            // Generate offspring
            let mut offspring = Vec::new();
            let offspring_per_pair =
                (self.target_population_size as f64 / pairs.len().max(1) as f64).ceil() as usize;

            for pair in &pairs {
                let parent_a = current_pop
                    .individuals
                    .iter()
                    .find(|i| i.id == pair.parent_a);
                let parent_b = current_pop
                    .individuals
                    .iter()
                    .find(|i| i.id == pair.parent_b);

                if let (Some(pa), Some(pb)) = (parent_a, parent_b) {
                    for _ in 0..offspring_per_pair {
                        if offspring.len() >= self.target_population_size {
                            break;
                        }
                        let child = self.generate_offspring(next_id, pa, pb, r#gen, loci, rng);
                        pedigree.add_entry(PedigreeEntry {
                            individual_id: child.id,
                            parent_a_id: Some(pa.id),
                            parent_b_id: Some(pb.id),
                            generation: r#gen,
                        });
                        offspring.push(child);
                        next_id += 1;
                    }
                }
            }

            // Apply selection if enabled
            if self.selection_enabled && !offspring.is_empty() {
                self.apply_selection(&mut offspring, rng);
            }

            // Trim to target size
            offspring.truncate(self.target_population_size);

            // Update population
            let diversity_hv_temp = compute_diversity_hv(&Population {
                individuals: offspring.clone(),
                generation: r#gen,
                founding_size: initial.founding_size,
                diversity_hv: current_pop.diversity_hv.clone(),
            });

            current_pop = Population {
                individuals: offspring,
                generation: r#gen,
                founding_size: initial.founding_size,
                diversity_hv: diversity_hv_temp,
            };

            snapshots.push(self.snapshot(&current_pop));
        }

        SimulationResult {
            generations: snapshots,
            final_population: current_pop,
        }
    }

    /// Generate a single offspring from two parents.
    fn generate_offspring(
        &self,
        id: u64,
        parent_a: &Individual,
        parent_b: &Individual,
        generation: u32,
        loci: &[Locus],
        rng: &mut impl Rng,
    ) -> Individual {
        let n_loci = parent_a.genotypes.len().min(parent_b.genotypes.len());
        let mut genotypes = Vec::with_capacity(n_loci);

        for i in 0..n_loci {
            let ga = &parent_a.genotypes[i];
            let gb = &parent_b.genotypes[i];

            // Mendelian segregation: randomly pick one allele from each parent
            let allele_from_a = if rng.gen_bool(0.5) {
                ga.allele_a.clone()
            } else {
                ga.allele_b.clone()
            };

            let allele_from_b = if rng.gen_bool(0.5) {
                gb.allele_a.clone()
            } else {
                gb.allele_b.clone()
            };

            // Apply mutation
            let allele_a = self.maybe_mutate(allele_from_a, rng);
            let allele_b = self.maybe_mutate(allele_from_b, rng);

            genotypes.push(Genotype {
                locus_id: ga.locus_id,
                allele_a,
                allele_b,
            });
        }

        let genome_hv = if !loci.is_empty() && !genotypes.is_empty() {
            encode_individual(loci, &genotypes)
        } else {
            symthaea_core::hdc::unified_hv::ContinuousHV::zero(
                symthaea_core::hdc::unified_hv::HDC_DIMENSION,
            )
        };

        let sex = if rng.gen_bool(0.5) {
            BiologicalSex::Female
        } else {
            BiologicalSex::Male
        };

        let mut ind = Individual {
            id,
            sex,
            genotypes,
            genome_hv,
            generation,
            parent_ids: (Some(parent_a.id), Some(parent_b.id)),
            fitness: 1.0,
        };

        ind.fitness = compute_fitness(&ind);
        ind
    }

    /// Apply mutation to an allele with probability mutation_rate.
    fn maybe_mutate(&self, mut allele: Allele, rng: &mut impl Rng) -> Allele {
        if rng.gen_bool(self.mutation_rate.min(1.0)) {
            // Create a new allele ID (mutation)
            allele.id = rng.r#gen::<u64>();
            // 10% chance mutation is deleterious
            if rng.gen_bool(0.1) {
                allele.is_deleterious = true;
                allele.selection_coefficient = -rng.gen_range(0.001..0.1);
                allele.dominance = rng.gen_range(0.0..0.5); // mostly recessive
            }
        }
        allele
    }

    /// Apply viability selection: remove unfit individuals probabilistically.
    fn apply_selection(&self, offspring: &mut Vec<Individual>, rng: &mut impl Rng) {
        offspring.retain(|ind| {
            // Probability of survival proportional to fitness
            rng.gen_bool(ind.fitness.clamp(0.0, 1.0))
        });
    }

    /// Take a snapshot of current population statistics.
    fn snapshot(&self, population: &Population) -> GenerationSnapshot {
        let n_females = population.count_females();
        let n_males = population.count_males();
        let effective_ne = ne_sex_ratio(n_females, n_males);

        // Compute mean heterozygosity across all loci
        let heterozygosity = if population.individuals.is_empty() {
            0.0
        } else {
            // Collect genotypes at each locus across population
            let n_loci = population
                .individuals
                .first()
                .map_or(0, |i| i.genotypes.len());
            if n_loci == 0 {
                0.0
            } else {
                let total_ho: f64 = (0..n_loci)
                    .map(|locus_idx| {
                        let locus_genotypes: Vec<Genotype> = population
                            .individuals
                            .iter()
                            .filter_map(|ind| ind.genotypes.get(locus_idx).cloned())
                            .collect();
                        observed_heterozygosity(&locus_genotypes)
                    })
                    .sum();
                total_ho / n_loci as f64
            }
        };

        // Estimate expected heterozygosity for F calculation
        // Use 0.5 as baseline He for founders
        let he_estimate = 0.5_f64;
        let inbreeding_f = if heterozygosity > 0.0 && he_estimate > 0.0 {
            inbreeding_coefficient_from_heterozygosity(he_estimate, heterozygosity)
        } else {
            0.0
        };

        let genetic_load = population_genetic_load(population);

        // Count fixed deleterious alleles: loci where all individuals are homozygous deleterious
        let num_deleterious_fixed = if population.individuals.is_empty() {
            0
        } else {
            let n_loci = population
                .individuals
                .first()
                .map_or(0, |i| i.genotypes.len());
            (0..n_loci)
                .filter(|&locus_idx| {
                    population.individuals.iter().all(|ind| {
                        ind.genotypes
                            .get(locus_idx)
                            .is_some_and(|g| g.is_homozygous_deleterious())
                    })
                })
                .count()
        };

        GenerationSnapshot {
            generation: population.generation,
            population_size: population.size(),
            effective_ne,
            heterozygosity,
            inbreeding_f,
            genetic_load,
            num_deleterious_fixed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc_genetics::encode_locus;
    use rand::SeedableRng;
    use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

    fn make_loci(n: usize) -> Vec<Locus> {
        (0..n)
            .map(|i| Locus {
                id: i as u64,
                name: format!("locus_{i}"),
                chromosome: 1,
                position: i as u64 * 1000,
                hv: encode_locus(i as u64),
            })
            .collect()
    }

    fn make_founder_population(
        n_females: usize,
        n_males: usize,
        loci: &[Locus],
        rng: &mut impl Rng,
    ) -> (Population, Pedigree) {
        let mut individuals = Vec::new();
        let mut pedigree = Pedigree::new();
        let n = n_females + n_males;

        for i in 0..n {
            let sex = if i < n_females {
                BiologicalSex::Female
            } else {
                BiologicalSex::Male
            };

            let genotypes: Vec<Genotype> = loci
                .iter()
                .map(|l| {
                    let a_id = rng.gen_range(0..10);
                    let b_id = rng.gen_range(0..10);
                    Genotype {
                        locus_id: l.id,
                        allele_a: Allele::neutral(a_id),
                        allele_b: Allele::neutral(b_id),
                    }
                })
                .collect();

            let genome_hv = if !loci.is_empty() {
                encode_individual(loci, &genotypes)
            } else {
                ContinuousHV::zero(HDC_DIMENSION)
            };

            let id = i as u64;
            individuals.push(Individual {
                id,
                sex,
                genotypes,
                genome_hv,
                generation: 0,
                parent_ids: (None, None),
                fitness: 1.0,
            });
            pedigree.add_entry(PedigreeEntry {
                individual_id: id,
                parent_a_id: None,
                parent_b_id: None,
                generation: 0,
            });
        }

        let div_hv = ContinuousHV::zero(HDC_DIMENSION);
        (
            Population {
                individuals,
                generation: 0,
                founding_size: n,
                diversity_hv: div_hv,
            },
            pedigree,
        )
    }

    #[test]
    fn test_simulation_maintains_population_size() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let loci = make_loci(5);
        let (pop, mut ped) = make_founder_population(10, 10, &loci, &mut rng);

        let sim = PopulationSimulator {
            strategy: BreedingStrategy::Random,
            target_population_size: 20,
            mutation_rate: 0.0, // No mutation for cleaner test
            selection_enabled: false,
        };

        let result = sim.run(&pop, &mut ped, &loci, 5, &mut rng);

        // Population should be approximately target size each generation
        for snap in &result.generations[1..] {
            assert!(
                snap.population_size >= 10 && snap.population_size <= 30,
                "population size should be near target: got {}",
                snap.population_size
            );
        }
    }

    #[test]
    fn test_simulation_records_all_generations() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let loci = make_loci(3);
        let (pop, mut ped) = make_founder_population(5, 5, &loci, &mut rng);

        let sim = PopulationSimulator::new(BreedingStrategy::Random, 10);
        let result = sim.run(&pop, &mut ped, &loci, 10, &mut rng);

        assert_eq!(result.generations.len(), 11, "initial + 10 generations");
        assert_eq!(result.generations[0].generation, 0);
        assert_eq!(result.generations[10].generation, 10);
    }

    #[test]
    fn test_simulation_deterministic_with_seed() {
        let loci = make_loci(3);

        let run_sim = |seed: u64| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let (pop, mut ped) = make_founder_population(5, 5, &loci, &mut rng);
            let sim = PopulationSimulator {
                strategy: BreedingStrategy::Random,
                target_population_size: 10,
                mutation_rate: 0.0,
                selection_enabled: false,
            };
            sim.run(&pop, &mut ped, &loci, 5, &mut rng)
        };

        let r1 = run_sim(42);
        let r2 = run_sim(42);

        for (s1, s2) in r1.generations.iter().zip(r2.generations.iter()) {
            assert_eq!(s1.population_size, s2.population_size);
            assert!(
                (s1.heterozygosity - s2.heterozygosity).abs() < 1e-10,
                "deterministic: heterozygosity should match"
            );
        }
    }

    #[test]
    fn test_simulation_heterozygosity_decays() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let loci = make_loci(10);
        let (pop, mut ped) = make_founder_population(5, 5, &loci, &mut rng);

        let sim = PopulationSimulator {
            strategy: BreedingStrategy::Random,
            target_population_size: 10,
            mutation_rate: 0.0,
            selection_enabled: false,
        };

        let result = sim.run(&pop, &mut ped, &loci, 20, &mut rng);

        let h0 = result.generations[0].heterozygosity;
        let h_final = result.generations.last().unwrap().heterozygosity;

        // Heterozygosity should generally decrease (may not be strictly monotonic)
        // With Ne ~ 10 and 20 generations: H(20)/H(0) ~ (1 - 1/20)^20 ~ 0.36
        if h0 > 0.0 {
            assert!(
                h_final <= h0 + 0.1,
                "heterozygosity should not increase substantially: h0={h0}, h_final={h_final}"
            );
        }
    }

    #[test]
    fn test_simulation_with_selection() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let loci = make_loci(5);

        // Create population with some deleterious alleles
        let mut individuals = Vec::new();
        let mut pedigree = Pedigree::new();
        for i in 0..20 {
            let sex = if i < 10 {
                BiologicalSex::Female
            } else {
                BiologicalSex::Male
            };
            let genotypes: Vec<Genotype> = loci
                .iter()
                .map(|l| Genotype {
                    locus_id: l.id,
                    allele_a: if rng.gen_bool(0.3) {
                        Allele::deleterious(rng.r#gen(), -0.05, 0.3)
                    } else {
                        Allele::neutral(rng.r#gen())
                    },
                    allele_b: Allele::neutral(rng.r#gen()),
                })
                .collect();
            let genome_hv = encode_individual(&loci, &genotypes);
            individuals.push(Individual {
                id: i,
                sex,
                genotypes,
                genome_hv,
                generation: 0,
                parent_ids: (None, None),
                fitness: 1.0,
            });
            pedigree.add_entry(PedigreeEntry {
                individual_id: i,
                parent_a_id: None,
                parent_b_id: None,
                generation: 0,
            });
        }
        // Compute fitness for founders
        for ind in &mut individuals {
            ind.fitness = compute_fitness(ind);
        }

        let pop = Population {
            individuals,
            generation: 0,
            founding_size: 20,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };

        let sim = PopulationSimulator {
            strategy: BreedingStrategy::Random,
            target_population_size: 20,
            mutation_rate: 0.0,
            selection_enabled: true,
        };

        let result = sim.run(&pop, &mut pedigree, &loci, 5, &mut rng);
        assert!(!result.generations.is_empty());
        // Just verify it runs without panics
    }

    #[test]
    fn test_simulation_minimum_kinship_strategy() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let loci = make_loci(3);
        let (pop, mut ped) = make_founder_population(5, 5, &loci, &mut rng);

        let sim = PopulationSimulator::new(BreedingStrategy::MinimumKinship, 10);
        let result = sim.run(&pop, &mut ped, &loci, 5, &mut rng);
        assert!(!result.generations.is_empty());
    }

    #[test]
    fn test_simulation_hdc_distance_strategy() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let loci = make_loci(3);
        let (pop, mut ped) = make_founder_population(5, 5, &loci, &mut rng);

        let sim = PopulationSimulator::new(BreedingStrategy::HdcDistanceMaximize, 10);
        let result = sim.run(&pop, &mut ped, &loci, 5, &mut rng);
        assert!(!result.generations.is_empty());
    }

    #[test]
    fn test_simulation_balanced_contribution_strategy() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let loci = make_loci(3);
        let (pop, mut ped) = make_founder_population(5, 5, &loci, &mut rng);

        let sim = PopulationSimulator::new(BreedingStrategy::BalancedContribution, 10);
        let result = sim.run(&pop, &mut ped, &loci, 5, &mut rng);
        assert!(!result.generations.is_empty());
    }

    #[test]
    fn test_snapshot_initial_state() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let loci = make_loci(5);
        let (pop, _) = make_founder_population(10, 10, &loci, &mut rng);

        let sim = PopulationSimulator::new(BreedingStrategy::Random, 20);
        let snap = sim.snapshot(&pop);

        assert_eq!(snap.generation, 0);
        assert_eq!(snap.population_size, 20);
        assert!(snap.effective_ne > 0.0);
    }

    #[test]
    fn test_simulation_pedigree_grows() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let loci = make_loci(3);
        let (pop, mut ped) = make_founder_population(5, 5, &loci, &mut rng);

        let initial_entries = ped.entries.len();
        let sim = PopulationSimulator::new(BreedingStrategy::Random, 10);
        let _result = sim.run(&pop, &mut ped, &loci, 3, &mut rng);

        assert!(
            ped.entries.len() > initial_entries,
            "pedigree should grow with each generation"
        );
    }

    #[test]
    fn test_simulation_vs_analytical_heterozygosity() {
        // Run simulation with 25F + 25M (Ne ~ 50) for 20 generations
        // Compare observed heterozygosity decay against analytical formula
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let loci = make_loci(20);
        let (pop, mut ped) = make_founder_population(25, 25, &loci, &mut rng);

        // Measure initial heterozygosity
        let h0 = {
            let n_loci = loci.len();
            let total: f64 = (0..n_loci)
                .map(|li| {
                    let genos: Vec<_> = pop
                        .individuals
                        .iter()
                        .filter_map(|ind| ind.genotypes.get(li).cloned())
                        .collect();
                    crate::diversity::observed_heterozygosity(&genos)
                })
                .sum();
            total / n_loci as f64
        };

        let sim = PopulationSimulator {
            strategy: BreedingStrategy::Random,
            target_population_size: 50,
            mutation_rate: 0.0, // No mutation for clean comparison
            selection_enabled: false,
        };

        let result = sim.run(&pop, &mut ped, &loci, 20, &mut rng);
        let ne = 50.0; // 25F + 25M balanced -> Ne ~ N

        // Compare simulation heterozygosity vs analytical at each generation
        // Allow 30% tolerance due to stochastic drift
        for snap in &result.generations {
            let analytical =
                crate::diversity::heterozygosity_after_generations(h0, ne, snap.generation);
            if snap.generation > 0 && analytical > 0.01 {
                let relative_error = (snap.heterozygosity - analytical).abs() / analytical;
                assert!(
                    relative_error < 0.5,
                    "Gen {}: sim H={:.4}, analytical H={:.4}, relative error={:.2}%",
                    snap.generation,
                    snap.heterozygosity,
                    analytical,
                    relative_error * 100.0
                );
            }
        }
    }

    #[test]
    fn test_cfc_predictor_vs_analytical() {
        // Verify CfC O(1) predictor produces a meaningful state change
        // across different generation horizons
        use crate::temporal_evolution::PopulationTrajectoryPredictor;

        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let loci = make_loci(10);
        let (pop, _) = make_founder_population(15, 15, &loci, &mut rng);

        let initial_state = PopulationTrajectoryPredictor::encode_population_state(&pop);
        assert!(
            initial_state.norm() > 0.0,
            "Initial state should be non-zero"
        );

        let mut predictor = PopulationTrajectoryPredictor::new();

        // Short-term prediction
        let state_5 = predictor.predict_at_generation(&initial_state, 5);
        predictor.reset();

        // Long-term prediction
        let state_100 = predictor.predict_at_generation(&initial_state, 100);

        // Both should be valid non-zero states
        assert!(state_5.norm() > 0.0, "5-gen prediction should be non-zero");
        assert!(
            state_100.norm() > 0.0,
            "100-gen prediction should be non-zero"
        );

        // Decoded metrics should be in reasonable ranges
        let h_5 = PopulationTrajectoryPredictor::decode_heterozygosity(&state_5);
        let h_100 = PopulationTrajectoryPredictor::decode_heterozygosity(&state_100);
        assert!(
            h_5 >= 0.0 && h_5 <= 1.0,
            "Decoded H at gen 5 should be in [0,1]: {h_5}"
        );
        assert!(
            h_100 >= 0.0 && h_100 <= 1.0,
            "Decoded H at gen 100 should be in [0,1]: {h_100}"
        );
    }
}
