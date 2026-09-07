// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic evolutionary-history analysis over validated lifecycle evidence.
//!
//! This module does not infer ancestry from behavioral observations. It consumes only a
//! [`crate::LifecycleLedgerV1`] that has already satisfied the complete, contiguous lifecycle
//! contract. The resulting report can therefore distinguish reproductive ancestry from genetic
//! ancestry, measure exact genome-field changes at birth, and propagate exact extinction evidence.

use std::collections::BTreeMap;

use crate::{
    AgentId, ExtinctionEvidenceV1, GenomeEvidenceV1, LifecycleLedgerV1,
};

/// Exact field-level difference between a genome source and one offspring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenomeDeltaV1 {
    pub set_point_changed: bool,
    pub forage_efficiency_changed: bool,
    pub action_temperature_changed: bool,
    pub perceptual_grain_changed: bool,
    pub changed_fields: u8,
}

impl GenomeDeltaV1 {
    pub fn between(source: GenomeEvidenceV1, offspring: GenomeEvidenceV1) -> Self {
        let set_point_changed = source.set_point_bits != offspring.set_point_bits;
        let forage_efficiency_changed =
            source.forage_efficiency_bits != offspring.forage_efficiency_bits;
        let action_temperature_changed =
            source.action_temperature_bits != offspring.action_temperature_bits;
        let perceptual_grain_changed =
            source.perceptual_grain_bits != offspring.perceptual_grain_bits;
        let changed_fields = u8::from(set_point_changed)
            + u8::from(forage_efficiency_changed)
            + u8::from(action_temperature_changed)
            + u8::from(perceptual_grain_changed);
        Self {
            set_point_changed,
            forage_efficiency_changed,
            action_temperature_changed,
            perceptual_grain_changed,
            changed_fields,
        }
    }

    pub fn is_mutation(self) -> bool {
        self.changed_fields > 0
    }
}

/// One offspring's exact genetic transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutationAncestryV1 {
    pub offspring_id: AgentId,
    pub reproductive_parent_id: AgentId,
    pub genome_source_id: AgentId,
    pub reproductive_lineage_id: AgentId,
    pub genome_source_lineage_id: AgentId,
    pub generation: u32,
    pub source_genome: GenomeEvidenceV1,
    pub offspring_genome: GenomeEvidenceV1,
    pub delta: GenomeDeltaV1,
    /// True when `InheritanceMode::RandomPeer`-like evidence imported a genome from a different
    /// lineage than the organism that physically reproduced.
    pub cross_lineage_genetic_import: bool,
}

/// Aggregate exact-history measurements for one founder lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageHistoryV1 {
    pub lineage_id: AgentId,
    pub members_total: usize,
    pub living_members: usize,
    pub max_generation: u32,
    pub first_birth_tick: u64,
    pub last_birth_tick: u64,
    pub deaths: usize,
    pub last_death_tick: Option<u64>,
    pub mutated_births: usize,
    pub total_changed_genome_fields: u64,
    pub max_changed_fields_in_one_birth: u8,
    pub cross_lineage_genetic_imports: usize,
}

/// Deterministic evolutionary history reconstructed from a validated lifecycle ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvolutionaryHistoryReportV1 {
    pub total_agents: usize,
    pub founder_count: usize,
    pub birth_count: usize,
    pub death_count: usize,
    pub living_agents: usize,
    pub exact_extinction: Option<ExtinctionEvidenceV1>,
    pub lineages: BTreeMap<AgentId, LineageHistoryV1>,
    pub mutation_ancestry: BTreeMap<AgentId, MutationAncestryV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolutionaryHistoryError {
    MissingReproductiveParent {
        offspring_id: AgentId,
        parent_id: AgentId,
    },
    MissingGenomeSource {
        offspring_id: AgentId,
        genome_source_id: AgentId,
    },
}

/// Analyze exact lineage and mutation ancestry from a validated lifecycle ledger.
pub fn analyze_evolutionary_history(
    ledger: &LifecycleLedgerV1,
) -> Result<EvolutionaryHistoryReportV1, EvolutionaryHistoryError> {
    let mut lineages = BTreeMap::<AgentId, LineageHistoryV1>::new();
    let mut mutation_ancestry = BTreeMap::<AgentId, MutationAncestryV1>::new();

    for record in ledger.records().values() {
        let lineage = lineages.entry(record.lineage_id).or_insert(LineageHistoryV1 {
            lineage_id: record.lineage_id,
            members_total: 0,
            living_members: 0,
            max_generation: record.generation,
            first_birth_tick: record.born_tick,
            last_birth_tick: record.born_tick,
            deaths: 0,
            last_death_tick: None,
            mutated_births: 0,
            total_changed_genome_fields: 0,
            max_changed_fields_in_one_birth: 0,
            cross_lineage_genetic_imports: 0,
        });
        lineage.members_total += 1;
        lineage.living_members += usize::from(record.is_alive());
        lineage.max_generation = lineage.max_generation.max(record.generation);
        lineage.first_birth_tick = lineage.first_birth_tick.min(record.born_tick);
        lineage.last_birth_tick = lineage.last_birth_tick.max(record.born_tick);
        if let Some(died_tick) = record.died_tick {
            lineage.deaths += 1;
            lineage.last_death_tick = Some(
                lineage
                    .last_death_tick
                    .map_or(died_tick, |previous| previous.max(died_tick)),
            );
        }

        let (Some(parent_id), Some(genome_source_id)) =
            (record.reproductive_parent_id, record.genome_source_id)
        else {
            continue;
        };

        let parent = ledger.records().get(&parent_id).ok_or(
            EvolutionaryHistoryError::MissingReproductiveParent {
                offspring_id: record.agent_id,
                parent_id,
            },
        )?;
        let genome_source = ledger.records().get(&genome_source_id).ok_or(
            EvolutionaryHistoryError::MissingGenomeSource {
                offspring_id: record.agent_id,
                genome_source_id,
            },
        )?;
        let delta = GenomeDeltaV1::between(genome_source.genome, record.genome);
        let cross_lineage_genetic_import = genome_source.lineage_id != record.lineage_id;

        if delta.is_mutation() {
            lineage.mutated_births += 1;
        }
        lineage.total_changed_genome_fields += u64::from(delta.changed_fields);
        lineage.max_changed_fields_in_one_birth =
            lineage.max_changed_fields_in_one_birth.max(delta.changed_fields);
        lineage.cross_lineage_genetic_imports += usize::from(cross_lineage_genetic_import);

        mutation_ancestry.insert(
            record.agent_id,
            MutationAncestryV1 {
                offspring_id: record.agent_id,
                reproductive_parent_id: parent_id,
                genome_source_id,
                reproductive_lineage_id: parent.lineage_id,
                genome_source_lineage_id: genome_source.lineage_id,
                generation: record.generation,
                source_genome: genome_source.genome,
                offspring_genome: record.genome,
                delta,
                cross_lineage_genetic_import,
            },
        );
    }

    Ok(EvolutionaryHistoryReportV1 {
        total_agents: ledger.records().len(),
        founder_count: ledger.founder_count(),
        birth_count: ledger.birth_count(),
        death_count: ledger.death_count(),
        living_agents: ledger.alive().len(),
        exact_extinction: ledger.extinction(),
        lineages,
        mutation_ancestry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentIdAllocator, Genome, LifecycleDeathCauseV1, LifecycleEventV1,
        LifecycleTransitionV1, OrganismConfig, analyze_lifecycle_events,
    };

    fn genome() -> GenomeEvidenceV1 {
        GenomeEvidenceV1::from_genome(Genome::from_config(&OrganismConfig::default()))
    }

    fn founder(sequence: u64, id: AgentId) -> LifecycleEventV1 {
        LifecycleEventV1 {
            sequence,
            tick: 0,
            transition: LifecycleTransitionV1::Founder {
                agent_id: id,
                lineage_id: id,
                generation: 0,
                genome: genome(),
                initial_energy_bits: 0.5f64.to_bits(),
            },
        }
    }

    #[test]
    fn reports_exact_mutation_and_cross_lineage_genetic_import() {
        let mut ids = AgentIdAllocator::new();
        let parent = ids.allocate();
        let donor = ids.allocate();
        let child = ids.allocate();
        let child_genome = GenomeEvidenceV1::from_genome(Genome {
            forage_efficiency: 0.31,
            ..genome().to_genome()
        });
        let lifecycle = analyze_lifecycle_events(&[
            founder(0, parent),
            founder(1, donor),
            LifecycleEventV1 {
                sequence: 2,
                tick: 5,
                transition: LifecycleTransitionV1::Birth {
                    reproductive_parent_id: parent,
                    genome_source_id: donor,
                    offspring_id: child,
                    lineage_id: parent,
                    generation: 1,
                    reproductive_parent_genome: genome(),
                    genome_source_genome: genome(),
                    offspring_genome: child_genome,
                    initial_energy_bits: 0.4f64.to_bits(),
                },
            },
        ])
        .expect("valid lifecycle");

        let history = analyze_evolutionary_history(&lifecycle).expect("valid history");
        let ancestry = history
            .mutation_ancestry
            .get(&child)
            .expect("child mutation ancestry");
        assert_eq!(ancestry.delta.changed_fields, 1);
        assert!(ancestry.delta.forage_efficiency_changed);
        assert!(ancestry.cross_lineage_genetic_import);
        assert_eq!(ancestry.reproductive_lineage_id, parent);
        assert_eq!(ancestry.genome_source_lineage_id, donor);

        let lineage = history.lineages.get(&parent).expect("parent lineage");
        assert_eq!(lineage.members_total, 2);
        assert_eq!(lineage.mutated_births, 1);
        assert_eq!(lineage.cross_lineage_genetic_imports, 1);
    }

    #[test]
    fn exact_extinction_propagates_from_lifecycle_evidence() {
        let mut ids = AgentIdAllocator::new();
        let founder_id = ids.allocate();
        let lifecycle = analyze_lifecycle_events(&[
            founder(0, founder_id),
            LifecycleEventV1 {
                sequence: 1,
                tick: 9,
                transition: LifecycleTransitionV1::Death {
                    agent_id: founder_id,
                    cause: LifecycleDeathCauseV1::PopulationEnergyThreshold,
                    energy_bits: 0.0f64.to_bits(),
                },
            },
        ])
        .expect("valid lifecycle");
        let history = analyze_evolutionary_history(&lifecycle).expect("valid history");
        assert_eq!(history.living_agents, 0);
        assert_eq!(
            history.exact_extinction,
            Some(ExtinctionEvidenceV1 {
                tick: 9,
                sequence: 1,
            })
        );
    }

    #[test]
    fn unchanged_inheritance_is_not_called_a_mutation() {
        let mut ids = AgentIdAllocator::new();
        let parent = ids.allocate();
        let child = ids.allocate();
        let lifecycle = analyze_lifecycle_events(&[
            founder(0, parent),
            LifecycleEventV1 {
                sequence: 1,
                tick: 3,
                transition: LifecycleTransitionV1::Birth {
                    reproductive_parent_id: parent,
                    genome_source_id: parent,
                    offspring_id: child,
                    lineage_id: parent,
                    generation: 1,
                    reproductive_parent_genome: genome(),
                    genome_source_genome: genome(),
                    offspring_genome: genome(),
                    initial_energy_bits: 0.4f64.to_bits(),
                },
            },
        ])
        .expect("valid lifecycle");
        let history = analyze_evolutionary_history(&lifecycle).expect("valid history");
        let ancestry = history.mutation_ancestry.get(&child).expect("ancestry");
        assert_eq!(ancestry.delta.changed_fields, 0);
        assert!(!ancestry.delta.is_mutation());
        assert!(!ancestry.cross_lineage_genetic_import);
        assert_eq!(history.lineages.get(&parent).unwrap().mutated_births, 0);
    }
}
