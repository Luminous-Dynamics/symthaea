// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic pre-offspring evolutionary birth planning.
//!
//! This module composes the two already-separated evolutionary decisions needed by an
//! authoritative `Population` birth transition:
//!
//! 1. select the pre-mutation genome source;
//! 2. mutate that genome using only the mutation RNG stream.
//!
//! It does not change parent energy, allocate an AgentId, construct an [`crate::Organism`], or
//! mutate the population. Those remain authoritative `Population` responsibilities. The returned
//! snapshots are sufficient for both offspring construction and first-class lifecycle ancestry.

use crate::{
    AgentId, EvolutionRngStreamsV1, Genome, GenomeSourceSelectionErrorV1, InheritanceMode,
    Organism, select_genome_source_v1,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvolutionBirthPlanV1 {
    pub reproductive_parent_index: usize,
    pub reproductive_parent_id: AgentId,
    pub reproductive_parent_genome: Genome,
    pub genome_source_index: usize,
    pub genome_source_id: AgentId,
    pub genome_source_genome: Genome,
    pub offspring_genome: Genome,
}

pub type EvolutionBirthPlanErrorV1 = GenomeSourceSelectionErrorV1;

/// Prepare the complete genome-level plan for one birth without mutating the population.
///
/// `RandomPeer` source selection consumes only the inheritance-source RNG. `Genome::mutate`
/// consumes only the mutation RNG. Consequently, when two conditions experience the same number
/// of birth-plan calls and the selected source genomes are equal, their mutation draws remain
/// exactly aligned even if one condition performs RandomPeer source selection.
pub fn prepare_evolution_birth_v1(
    organisms: &[Organism],
    reproducer_index: usize,
    inheritance: InheritanceMode,
    mutation_rate: f64,
    mutation_std: f64,
    rng: &mut EvolutionRngStreamsV1,
) -> Result<EvolutionBirthPlanV1, EvolutionBirthPlanErrorV1> {
    let source = select_genome_source_v1(organisms, reproducer_index, inheritance, rng)?;
    let reproductive_parent = &organisms[reproducer_index];
    let reproductive_parent_genome = Genome::from_config(&reproductive_parent.cfg);
    let offspring_genome = source
        .genome
        .mutate(rng.mutation_state_mut(), mutation_rate, mutation_std);

    Ok(EvolutionBirthPlanV1 {
        reproductive_parent_index: reproducer_index,
        reproductive_parent_id: reproductive_parent.id,
        reproductive_parent_genome,
        genome_source_index: source.source_index,
        genome_source_id: source.source_id,
        genome_source_genome: source.genome,
        offspring_genome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentIdAllocator, LEGACY_MUTATION_SEED_OFFSET_V1, OrganismConfig,
    };

    fn homogeneous_population(count: usize) -> Vec<Organism> {
        let mut ids = AgentIdAllocator::new();
        (0..count)
            .map(|i| {
                Organism::new(OrganismConfig::default(), i as u64 + 11)
                    .with_id(ids.allocate())
            })
            .collect()
    }

    fn heterogeneous_population(count: usize) -> Vec<Organism> {
        let mut ids = AgentIdAllocator::new();
        (0..count)
            .map(|i| {
                let mut cfg = OrganismConfig::default();
                cfg.forage_efficiency += i as f64 * 0.07;
                cfg.set_point -= i as f64 * 0.02;
                Organism::new(cfg, i as u64 + 31).with_id(ids.allocate())
            })
            .collect()
    }

    #[test]
    fn from_parent_birth_plan_matches_historical_manual_mutation_sequence() {
        let seed = 4242u64;
        let organisms = homogeneous_population(3);
        let parent_genome = Genome::from_config(&organisms[1].cfg);
        let mut legacy_state = seed
            .wrapping_add(LEGACY_MUTATION_SEED_OFFSET_V1)
            .max(1);
        let mut split = EvolutionRngStreamsV1::new(seed);

        for birth in 0..64 {
            let expected = parent_genome.mutate(&mut legacy_state, 0.37, 0.05);
            let plan = prepare_evolution_birth_v1(
                &organisms,
                1,
                InheritanceMode::FromParent,
                0.37,
                0.05,
                &mut split,
            )
            .expect("valid parent");
            assert_eq!(plan.genome_source_index, 1);
            assert_eq!(plan.reproductive_parent_id, organisms[1].id);
            assert_eq!(plan.genome_source_id, organisms[1].id);
            assert_eq!(plan.offspring_genome, expected, "birth {birth}");
            assert_eq!(split.snapshot().mutation_state, legacy_state, "birth {birth}");
        }
    }

    #[test]
    fn random_peer_source_draws_do_not_shift_common_mutation_draws_for_equal_genomes() {
        let organisms = homogeneous_population(8);
        let mut selected = EvolutionRngStreamsV1::new(9001);
        let mut random_peer = EvolutionRngStreamsV1::new(9001);

        for birth in 0..64 {
            let selected_plan = prepare_evolution_birth_v1(
                &organisms,
                3,
                InheritanceMode::FromParent,
                0.41,
                0.05,
                &mut selected,
            )
            .expect("selected plan");
            let random_plan = prepare_evolution_birth_v1(
                &organisms,
                3,
                InheritanceMode::RandomPeer,
                0.41,
                0.05,
                &mut random_peer,
            )
            .expect("random-peer plan");

            assert_eq!(
                selected_plan.offspring_genome, random_plan.offspring_genome,
                "source-selection RNG shifted mutation outcome at birth {birth}"
            );
            assert_eq!(
                selected.snapshot().mutation_state,
                random_peer.snapshot().mutation_state,
                "mutation state diverged at birth {birth}"
            );
        }

        assert_ne!(
            selected.snapshot().inheritance_source_state,
            random_peer.snapshot().inheritance_source_state,
            "RandomPeer should consume only its own source-selection stream"
        );
    }

    #[test]
    fn plan_preserves_both_reproductive_and_genetic_ancestry_snapshots() {
        let organisms = heterogeneous_population(6);
        let mut rng = EvolutionRngStreamsV1::new(123);

        for _ in 0..32 {
            let plan = prepare_evolution_birth_v1(
                &organisms,
                4,
                InheritanceMode::RandomPeer,
                0.0,
                0.05,
                &mut rng,
            )
            .expect("birth plan");

            assert_eq!(plan.reproductive_parent_index, 4);
            assert_eq!(plan.reproductive_parent_id, organisms[4].id);
            assert_eq!(
                plan.reproductive_parent_genome,
                Genome::from_config(&organisms[4].cfg)
            );
            assert_eq!(plan.genome_source_id, organisms[plan.genome_source_index].id);
            assert_eq!(
                plan.genome_source_genome,
                Genome::from_config(&organisms[plan.genome_source_index].cfg)
            );
            assert_eq!(
                plan.offspring_genome, plan.genome_source_genome,
                "zero mutation rate must preserve selected source genome"
            );
        }
    }

    #[test]
    fn invalid_reproducer_is_rejected_before_any_mutation_draw() {
        let organisms = homogeneous_population(2);
        let mut rng = EvolutionRngStreamsV1::new(77);
        let before = rng.snapshot();
        assert_eq!(
            prepare_evolution_birth_v1(
                &organisms,
                2,
                InheritanceMode::RandomPeer,
                1.0,
                0.05,
                &mut rng,
            ),
            Err(GenomeSourceSelectionErrorV1::ReproducerOutOfBounds {
                reproducer_index: 2,
                population_len: 2,
            })
        );
        assert_eq!(rng.snapshot(), before);
    }
}
