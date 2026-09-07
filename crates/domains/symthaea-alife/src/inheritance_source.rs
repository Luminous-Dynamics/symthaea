// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic genome-source selection for ALife reproduction controls.
//!
//! This module extracts the `InheritanceMode` decision from `Population` without performing
//! mutation or constructing offspring. Its only random operation is RandomPeer source selection,
//! and that operation consumes only [`crate::EvolutionRngStreamsV1`]'s inheritance-source stream.
//!
//! Keeping source selection separate from mutation makes the later production wiring auditable:
//!
//! ```text
//! source selection -> inheritance_source_state
//! Genome::mutate   -> mutation_state
//! ```
//!
//! No production `Population` path calls this helper yet.

use crate::{AgentId, EvolutionRngStreamsV1, Genome, InheritanceMode, Organism};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenomeSourceSelectionV1 {
    pub source_index: usize,
    pub source_id: AgentId,
    pub genome: Genome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenomeSourceSelectionErrorV1 {
    EmptyPopulation,
    ReproducerOutOfBounds {
        reproducer_index: usize,
        population_len: usize,
    },
}

/// Select the pre-mutation genome source for one authoritative reproduction event.
///
/// `FromParent` returns the physical reproducer and consumes no RNG. `RandomPeer` samples one
/// uniformly-indexed current population member using only the inheritance-source RNG stream.
/// The reproducer remains eligible under RandomPeer, matching the historical control semantics.
pub fn select_genome_source_v1(
    organisms: &[Organism],
    reproducer_index: usize,
    inheritance: InheritanceMode,
    rng: &mut EvolutionRngStreamsV1,
) -> Result<GenomeSourceSelectionV1, GenomeSourceSelectionErrorV1> {
    if organisms.is_empty() {
        return Err(GenomeSourceSelectionErrorV1::EmptyPopulation);
    }
    if reproducer_index >= organisms.len() {
        return Err(GenomeSourceSelectionErrorV1::ReproducerOutOfBounds {
            reproducer_index,
            population_len: organisms.len(),
        });
    }

    let source_index = match inheritance {
        InheritanceMode::FromParent => reproducer_index,
        InheritanceMode::RandomPeer => {
            let r = rng.next_inheritance_source_unit();
            ((r * organisms.len() as f64) as usize).min(organisms.len() - 1)
        }
    };
    let source = &organisms[source_index];
    Ok(GenomeSourceSelectionV1 {
        source_index,
        source_id: source.id,
        genome: Genome::from_config(&source.cfg),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentIdAllocator, OrganismConfig};

    fn population(count: usize) -> Vec<Organism> {
        let mut ids = AgentIdAllocator::new();
        (0..count)
            .map(|i| {
                let mut cfg = OrganismConfig::default();
                cfg.forage_efficiency += i as f64 * 0.1;
                Organism::new(cfg, (i as u64 + 1) * 17).with_id(ids.allocate())
            })
            .collect()
    }

    #[test]
    fn from_parent_selects_exact_reproducer_without_consuming_rng() {
        let organisms = population(4);
        let mut rng = EvolutionRngStreamsV1::new(9);
        let before = rng.snapshot();
        let selected = select_genome_source_v1(
            &organisms,
            2,
            InheritanceMode::FromParent,
            &mut rng,
        )
        .expect("valid reproducer");

        assert_eq!(selected.source_index, 2);
        assert_eq!(selected.source_id, organisms[2].id);
        assert_eq!(selected.genome, Genome::from_config(&organisms[2].cfg));
        assert_eq!(rng.snapshot(), before, "FromParent must consume no evolution RNG");
    }

    #[test]
    fn random_peer_source_draw_never_advances_mutation_state() {
        let organisms = population(5);
        let mut rng = EvolutionRngStreamsV1::new(42);
        let mutation_before = rng.snapshot().mutation_state;
        for _ in 0..100 {
            let selected = select_genome_source_v1(
                &organisms,
                3,
                InheritanceMode::RandomPeer,
                &mut rng,
            )
            .expect("non-empty population");
            assert!(selected.source_index < organisms.len());
        }
        assert_eq!(rng.snapshot().mutation_state, mutation_before);
    }

    #[test]
    fn random_peer_selection_is_deterministic_for_same_seed_and_population() {
        let organisms = population(6);
        let mut a = EvolutionRngStreamsV1::new(12345);
        let mut b = EvolutionRngStreamsV1::new(12345);

        for _ in 0..64 {
            let left = select_genome_source_v1(
                &organisms,
                1,
                InheritanceMode::RandomPeer,
                &mut a,
            )
            .expect("selection");
            let right = select_genome_source_v1(
                &organisms,
                1,
                InheritanceMode::RandomPeer,
                &mut b,
            )
            .expect("selection");
            assert_eq!(left, right);
        }
        assert_eq!(a.snapshot(), b.snapshot());
    }

    #[test]
    fn selected_source_genome_is_snapshot_of_current_member() {
        let organisms = population(3);
        let mut rng = EvolutionRngStreamsV1::new(77);
        for _ in 0..32 {
            let selected = select_genome_source_v1(
                &organisms,
                0,
                InheritanceMode::RandomPeer,
                &mut rng,
            )
            .expect("selection");
            assert_eq!(
                selected.genome,
                Genome::from_config(&organisms[selected.source_index].cfg)
            );
            assert_eq!(selected.source_id, organisms[selected.source_index].id);
        }
    }

    #[test]
    fn invalid_population_inputs_fail_closed() {
        let mut rng = EvolutionRngStreamsV1::new(1);
        assert_eq!(
            select_genome_source_v1(&[], 0, InheritanceMode::FromParent, &mut rng),
            Err(GenomeSourceSelectionErrorV1::EmptyPopulation)
        );

        let organisms = population(2);
        assert_eq!(
            select_genome_source_v1(
                &organisms,
                2,
                InheritanceMode::FromParent,
                &mut rng,
            ),
            Err(GenomeSourceSelectionErrorV1::ReproducerOutOfBounds {
                reproducer_index: 2,
                population_len: 2,
            })
        );
    }
}
