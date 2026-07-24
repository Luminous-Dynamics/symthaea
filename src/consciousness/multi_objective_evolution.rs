// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Objective Consciousness-Guided Evolution
//!
//! **Revolutionary Improvement #45: Multi-Dimensional Consciousness Optimization**
//!
//! Extends single-objective evolution (Φ only) to multi-objective optimization
//! across all consciousness dimensions:
//! - Φ (Integrated Information)
//! - ∇Φ (Gradient Flow)
//! - Entropy (Diversity)
//! - Complexity (Sophistication)
//! - Coherence (Stability)
//!
//! Uses NSGA-II-inspired algorithm to find **Pareto frontier** of optimal
//! trade-offs between dimensions.
//!
//! ## The Breakthrough
//!
//! Single-objective evolution finds ONE optimal primitive.
//! Multi-objective evolution finds a FRONTIER of optimal primitives,
//! each excelling in different consciousness dimensions!
//!
//! Example discovered primitives:
//! - High-Φ, Low-Entropy: Highly integrated but simple
//! - Low-Φ, High-Entropy: Less integrated but richer
//! - Balanced: Good across all dimensions

use crate::consciousness::{
    consciousness_profile::{ConsciousnessProfile, ParetoFrontier},
    primitive_evolution::{CandidatePrimitive, EvolutionConfig, PrimitiveEvolution},
};
use crate::hdc::primitive_system::PrimitiveTier;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Multi-objective evolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveResult {
    /// Pareto frontier - non-dominated solutions
    pub pareto_frontier: Vec<PrimitiveWithProfile>,

    /// All evolved primitives with their profiles
    pub all_primitives: Vec<PrimitiveWithProfile>,

    /// Evolution statistics
    pub generations_run: usize,
    pub converged: bool,
    pub total_time_ms: u64,

    /// Frontier statistics
    pub frontier_size: usize,
    pub frontier_spread: f64, // How diverse is the frontier?

    /// Best according to different criteria
    pub highest_phi: PrimitiveWithProfile,
    pub highest_entropy: PrimitiveWithProfile,
    pub highest_complexity: PrimitiveWithProfile,
    pub highest_composite: PrimitiveWithProfile,
}

/// Primitive paired with its full consciousness profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveWithProfile {
    pub primitive: CandidatePrimitive,
    pub profile: ConsciousnessProfile,
}

impl PrimitiveWithProfile {
    pub fn new(primitive: CandidatePrimitive, profile: ConsciousnessProfile) -> Self {
        Self { primitive, profile }
    }

    /// Create from primitive (computes profile)
    pub fn from_primitive(primitive: CandidatePrimitive) -> Self {
        // For now, use encoding as component for profile
        let components = vec![primitive.encoding];
        let profile = ConsciousnessProfile::from_components(&components);

        Self { primitive, profile }
    }
}

/// Multi-objective evolution engine
pub struct MultiObjectiveEvolution {
    _tier: PrimitiveTier,
    config: EvolutionConfig,
    population: Vec<PrimitiveWithProfile>,
    generation: usize,
}

impl MultiObjectiveEvolution {
    pub fn new(config: EvolutionConfig) -> Result<Self> {
        Ok(Self {
            _tier: config.tier,
            config,
            population: Vec::new(),
            generation: 0,
        })
    }

    /// Run multi-objective evolution
    pub fn evolve(&mut self) -> Result<MultiObjectiveResult> {
        use std::time::Instant;
        let start = Instant::now();

        // Initialize population using single-objective evolution
        println!("🌟 Initializing multi-objective population...");
        let mut initial_evolution = PrimitiveEvolution::new(self.config.clone())?;
        let initial_result = initial_evolution.evolve()?;

        // Convert to profiles
        self.population = initial_result
            .final_primitives
            .into_iter()
            .map(PrimitiveWithProfile::from_primitive)
            .collect();

        println!(
            "   Initial population: {} primitives",
            self.population.len()
        );

        // Multi-objective evolution loop
        let mut converged = false;
        let mut prev_frontier_size = 0;

        for r#gen in 0..self.config.num_generations {
            self.generation = r#gen;

            println!(
                "\n🧬 Multi-Objective Generation {}/{}...",
                r#gen + 1,
                self.config.num_generations
            );

            // Compute Pareto frontier
            let profiles: Vec<ConsciousnessProfile> =
                self.population.iter().map(|p| p.profile.clone()).collect();

            let frontier = ParetoFrontier::from_population(profiles);

            println!("   Pareto frontier size: {}", frontier.profiles.len());

            // Check convergence
            if r#gen > 0 && frontier.profiles.len() == prev_frontier_size {
                println!("   ✅ Frontier stabilized - converged!");
                converged = true;
                break;
            }

            prev_frontier_size = frontier.profiles.len();

            // Evolve population toward Pareto optimality
            self.evolve_generation(&frontier)?;

            // Display frontier statistics
            self.print_frontier_stats(&frontier);
        }

        let elapsed = start.elapsed();

        // Final Pareto frontier
        let final_profiles: Vec<ConsciousnessProfile> =
            self.population.iter().map(|p| p.profile.clone()).collect();

        let pareto_frontier_obj = ParetoFrontier::from_population(final_profiles.clone());

        // Extract frontier primitives
        let pareto_frontier: Vec<PrimitiveWithProfile> = self
            .population
            .iter()
            .filter(|p| p.profile.is_pareto_optimal(&final_profiles))
            .cloned()
            .collect();

        // Find best in each dimension — requires non-empty population
        let Some(highest_phi) = self.find_highest_phi() else {
            anyhow::bail!("Multi-objective evolution produced empty population");
        };
        let highest_entropy = self
            .find_highest_entropy()
            .unwrap_or_else(|| highest_phi.clone());
        let highest_complexity = self
            .find_highest_complexity()
            .unwrap_or_else(|| highest_phi.clone());
        let highest_composite = pareto_frontier_obj
            .highest_composite()
            .and_then(|prof| {
                self.population
                    .iter()
                    .find(|p| (p.profile.composite - prof.composite).abs() < 0.0001)
                    .cloned()
            })
            .unwrap_or_else(|| highest_phi.clone());

        // Compute frontier spread (diversity)
        let frontier_spread = self.compute_frontier_spread(&pareto_frontier);

        Ok(MultiObjectiveResult {
            pareto_frontier: pareto_frontier.clone(),
            all_primitives: self.population.clone(),
            generations_run: self.generation + 1,
            converged,
            total_time_ms: elapsed.as_millis() as u64,
            frontier_size: pareto_frontier.len(),
            frontier_spread,
            highest_phi,
            highest_entropy,
            highest_complexity,
            highest_composite,
        })
    }

    /// Evolve generation toward Pareto optimality
    fn evolve_generation(&mut self, frontier: &ParetoFrontier) -> Result<()> {
        let mut next_generation = Vec::new();

        // Elitism: keep all frontier members
        let frontier_members: Vec<PrimitiveWithProfile> = self
            .population
            .iter()
            .filter(|p| p.profile.is_pareto_optimal(&frontier.profiles))
            .cloned()
            .collect();

        next_generation.extend(frontier_members.clone());

        // Fill rest with offspring.
        //
        // `select_from_frontier` matches a tournament-picked frontier profile
        // back to a population member via a float tolerance check on
        // `composite` (`(a - b).abs() < 0.0001`). If `composite` is NaN for
        // every frontier member — which happens because `dominates()` uses
        // `>=`/`>` comparisons that are always false against NaN, so a
        // NaN-scored candidate can never be dominated and always survives
        // onto the frontier — that match fails unconditionally (NaN - NaN =
        // NaN, and NaN < 0.0001 is false). `select_from_frontier` then
        // returns `None` on every call, neither branch below ever pushes to
        // `next_generation`, and this loop spun forever (root cause of a 24h+
        // 97%-CPU hang found 2026-07-08 via `exp_loop_ablation.rs`'s "empty
        // input" regime — a degenerate/near-zero HDC encoding for `""` input
        // is the most likely NaN source upstream in profile computation).
        //
        // `stall_limit` guarantees forward progress regardless of whether the
        // NaN ever gets fixed at the source: after this many consecutive
        // non-productive iterations, clone an existing member directly
        // instead of routing through the frontier-match, which cannot fail.
        let stall_limit = self.config.population_size.max(8) * 4;
        let mut stalled = 0usize;
        while next_generation.len() < self.config.population_size {
            let pushed = if rand::random::<f64>() < self.config.crossover_rate {
                // Crossover from frontier members
                let parent1 = self.select_from_frontier(frontier);
                let parent2 = self.select_from_frontier(frontier);

                if let (Some(p1), Some(p2)) = (parent1, parent2) {
                    let child_primitive = CandidatePrimitive::recombine(
                        &p1.primitive,
                        &p2.primitive,
                        self.generation + 1,
                    );
                    let child = PrimitiveWithProfile::from_primitive(child_primitive);
                    next_generation.push(child);
                    true
                } else {
                    false
                }
            } else {
                // Mutation
                if let Some(parent) = self.select_from_frontier(frontier) {
                    let child_primitive = parent
                        .primitive
                        .mutate(self.config.mutation_rate, self.generation + 1);
                    let child = PrimitiveWithProfile::from_primitive(child_primitive);
                    next_generation.push(child);
                    true
                } else {
                    false
                }
            };

            if pushed {
                stalled = 0;
                continue;
            }
            stalled += 1;
            if stalled < stall_limit {
                continue;
            }

            // Forced-progress fallback: clone directly, bypassing the
            // NaN-fragile frontier match.
            if let Some(fallback) = frontier_members.first().or_else(|| self.population.first()) {
                next_generation.push(fallback.clone());
                stalled = 0;
            } else {
                // No members anywhere to fall back to — nothing more this
                // generation can produce. `evolve()` already handles an
                // under-sized/empty population via `find_highest_phi()`'s
                // `None` branch (`anyhow::bail!`), so returning early here
                // with a short `next_generation` is a safe degradation, not
                // a new failure mode.
                break;
            }
        }

        self.population = next_generation;
        Ok(())
    }

    /// Select parent from frontier (tournament)
    fn select_from_frontier(&self, frontier: &ParetoFrontier) -> Option<&PrimitiveWithProfile> {
        if frontier.profiles.is_empty() {
            return None;
        }

        // Tournament selection from frontier
        let mut best: Option<&PrimitiveWithProfile> = None;
        let mut best_composite = f64::NEG_INFINITY;

        for _ in 0..3 {
            let idx = rand::random::<usize>() % frontier.profiles.len();
            let profile = &frontier.profiles[idx];

            // Find corresponding primitive
            if let Some(prim) = self
                .population
                .iter()
                .find(|p| (p.profile.composite - profile.composite).abs() < 0.0001)
            {
                if profile.composite > best_composite {
                    best_composite = profile.composite;
                    best = Some(prim);
                }
            }
        }

        best
    }

    /// Find primitive with highest Φ
    fn find_highest_phi(&self) -> Option<PrimitiveWithProfile> {
        self.population
            .iter()
            .max_by(|a, b| a.profile.phi.total_cmp(&b.profile.phi))
            .cloned()
    }

    /// Find primitive with highest entropy
    fn find_highest_entropy(&self) -> Option<PrimitiveWithProfile> {
        self.population
            .iter()
            .max_by(|a, b| a.profile.entropy.total_cmp(&b.profile.entropy))
            .cloned()
    }

    /// Find primitive with highest complexity
    fn find_highest_complexity(&self) -> Option<PrimitiveWithProfile> {
        self.population
            .iter()
            .max_by(|a, b| a.profile.complexity.total_cmp(&b.profile.complexity))
            .cloned()
    }

    /// Compute frontier spread (diversity metric)
    fn compute_frontier_spread(&self, frontier: &[PrimitiveWithProfile]) -> f64 {
        if frontier.len() < 2 {
            return 0.0;
        }

        // Average pairwise distance in profile space
        let mut distances = Vec::new();

        for i in 0..frontier.len() {
            for j in (i + 1)..frontier.len() {
                let dist = frontier[i].profile.distance_to(&frontier[j].profile);
                distances.push(dist);
            }
        }

        if distances.is_empty() {
            return 0.0;
        }

        distances.iter().sum::<f64>() / distances.len() as f64
    }

    /// Print frontier statistics
    fn print_frontier_stats(&self, frontier: &ParetoFrontier) {
        if frontier.profiles.is_empty() {
            return;
        }

        let mean_phi =
            frontier.profiles.iter().map(|p| p.phi).sum::<f64>() / frontier.profiles.len() as f64;
        let mean_entropy = frontier.profiles.iter().map(|p| p.entropy).sum::<f64>()
            / frontier.profiles.len() as f64;
        let mean_complexity = frontier.profiles.iter().map(|p| p.complexity).sum::<f64>()
            / frontier.profiles.len() as f64;

        println!("   Frontier stats:");
        println!("      Mean Φ: {mean_phi:.4}");
        println!("      Mean Entropy: {mean_entropy:.4}");
        println!("      Mean Complexity: {mean_complexity:.4}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_objective_evolution_structure() {
        let config = EvolutionConfig {
            tier: PrimitiveTier::Physical,
            population_size: 10,
            num_generations: 3,
            mutation_rate: 0.2,
            crossover_rate: 0.5,
            elitism_count: 2,
            fitness_tasks: vec![],
            convergence_threshold: 0.01,
            phi_weight: 0.4,
            harmonic_weight: 0.3,
            epistemic_weight: 0.3,
        };

        let evolution = MultiObjectiveEvolution::new(config);
        assert!(evolution.is_ok());
    }

    /// Regression test for the 2026-07-08 24h+ hang (`""`-input regime of
    /// `exp_loop_ablation.rs`): an all-NaN-composite population must not
    /// spin forever in `evolve_generation`. Before the fix, this test would
    /// never return.
    #[test]
    fn test_evolve_generation_terminates_on_nan_profiles() {
        use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
        use crate::consciousness::primitives::primitive_evolution::CandidatePrimitive;
        use symthaea_core::hdc::binary_hv::BinaryHV;

        let config = EvolutionConfig {
            tier: PrimitiveTier::Physical,
            population_size: 10,
            num_generations: 1,
            mutation_rate: 0.2,
            crossover_rate: 0.5,
            elitism_count: 2,
            fitness_tasks: vec![],
            convergence_threshold: 0.01,
            phi_weight: 0.4,
            harmonic_weight: 0.3,
            epistemic_weight: 0.3,
        };
        let mut evolution = MultiObjectiveEvolution::new(config).unwrap();

        // Every profile field is NaN — reproduces the degenerate-encoding
        // scenario that made `dominates()` (all `>=`/`>` comparisons, always
        // false against NaN) put every candidate on the Pareto frontier while
        // `select_from_frontier`'s float-tolerance match against those NaN
        // composites can never succeed.
        let nan_profile = ConsciousnessProfile {
            phi: f64::NAN,
            gradient_magnitude: f64::NAN,
            entropy: f64::NAN,
            complexity: f64::NAN,
            coherence: f64::NAN,
            composite: f64::NAN,
        };
        evolution.population = (0..10)
            .map(|i| {
                let primitive = CandidatePrimitive {
                    name: format!("nan_{i}"),
                    tier: PrimitiveTier::Physical,
                    definition: String::new(),
                    fitness: f64::NAN,
                    encoding: BinaryHV::random(i as u64),
                    epistemic_coordinate: EpistemicCoordinate::default(),
                    harmonic_alignment: 0.0,
                };
                PrimitiveWithProfile::new(primitive, nan_profile.clone())
            })
            .collect();

        let profiles: Vec<ConsciousnessProfile> = evolution
            .population
            .iter()
            .map(|p| p.profile.clone())
            .collect();
        let frontier = ParetoFrontier::from_population(profiles);
        assert!(
            !frontier.profiles.is_empty(),
            "NaN profiles are expected to be un-dominated and survive onto the frontier"
        );

        // Bounded by the test harness's own timeout, not by the function
        // under test — before the fix this call never returns.
        let result = evolution.evolve_generation(&frontier);
        assert!(result.is_ok());
        assert_eq!(evolution.population.len(), 10);
    }
}
