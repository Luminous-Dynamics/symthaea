// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use symthaea_alife::{
    Genome, GenomeEvidenceV1, InheritanceMode, LifecycleDeathCauseV1, LifecycleError,
    LifecycleTransitionV1, OrganismConfig, Population, PopulationConfig, analyze_lifecycle_events,
};

fn base_cfg() -> PopulationConfig {
    PopulationConfig {
        death_energy_threshold: 0.05,
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.4,
        organism_cfg: OrganismConfig::default(),
        ..Default::default()
    }
}

#[test]
fn population_construction_emits_exact_founder_lifecycle() {
    let pop = Population::new(base_cfg(), 3, 42);

    assert_eq!(pop.lifecycle_epoch(), 0);
    assert_eq!(pop.lifecycle_events().len(), 3);
    assert!(pop.lifecycle_events().iter().all(|event| event.tick == 0));
    assert!(pop.lifecycle_events().iter().all(|event| matches!(
        event.transition,
        LifecycleTransitionV1::Founder { .. }
    )));

    let ledger = analyze_lifecycle_events(pop.lifecycle_events()).expect("complete founders");
    let expected_alive: BTreeSet<_> = pop.organisms.iter().map(|organism| organism.id).collect();
    assert_eq!(ledger.founder_count(), 3);
    assert_eq!(ledger.birth_count(), 0);
    assert_eq!(ledger.death_count(), 0);
    assert_eq!(ledger.alive(), &expected_alive);
}

#[test]
fn cull_weakest_records_exact_deaths_before_removal_without_advancing_epoch() {
    let mut pop = Population::new(base_cfg(), 3, 7);
    let removed = pop.cull_weakest(2);

    assert_eq!(removed, 2);
    assert_eq!(pop.len(), 1);
    assert_eq!(pop.lifecycle_epoch(), 0);

    let deaths = &pop.lifecycle_events()[3..];
    assert_eq!(deaths.len(), 2);
    assert!(deaths.iter().all(|event| {
        event.tick == 0
            && matches!(
                event.transition,
                LifecycleTransitionV1::Death {
                    cause: LifecycleDeathCauseV1::CullWeakest,
                    ..
                }
            )
    }));

    let ledger = analyze_lifecycle_events(pop.lifecycle_events()).expect("complete cull lifecycle");
    let expected_alive: BTreeSet<_> = pop.organisms.iter().map(|organism| organism.id).collect();
    assert_eq!(ledger.death_count(), 2);
    assert_eq!(ledger.alive(), &expected_alive);
}

#[test]
fn threshold_extinction_is_exact_lifecycle_evidence_not_behavioral_absence() {
    let cfg = PopulationConfig {
        death_energy_threshold: 2.0,
        reproduction_energy_threshold: 3.0,
        ..base_cfg()
    };
    let mut pop = Population::new(cfg, 3, 99);

    let summary = pop.step(|_| 0.0);
    assert_eq!(summary.deaths_this_tick, 3);
    assert_eq!(summary.births_this_tick, 0);
    assert!(pop.is_empty());
    assert_eq!(pop.lifecycle_epoch(), 1);

    let ledger = analyze_lifecycle_events(pop.lifecycle_events()).expect("complete extinction");
    assert_eq!(ledger.death_count(), 3);
    assert!(ledger.alive().is_empty());
    let extinction = ledger.extinction().expect("exact extinction transition");
    assert_eq!(extinction.tick, 0);

    let deaths = &pop.lifecycle_events()[3..];
    assert!(deaths.iter().all(|event| matches!(
        event.transition,
        LifecycleTransitionV1::Death {
            cause: LifecycleDeathCauseV1::PopulationEnergyThreshold,
            ..
        }
    )));
}

#[test]
fn random_peer_births_emit_plan_bound_genetic_ancestry_and_actual_offspring_genomes() {
    let initial_count = 4usize;
    let cfg = PopulationConfig {
        death_energy_threshold: -1.0,
        reproduction_energy_threshold: 0.0,
        reproduction_energy_cost: 0.4,
        mutation_rate: 1.0,
        mutation_std: 0.05,
        inheritance: InheritanceMode::RandomPeer,
        organism_cfg: OrganismConfig::default(),
    };
    let mut pop = Population::new(cfg, initial_count, 0xa11fe);

    let summary = pop.step(|_| 1.0);
    assert_eq!(summary.births_this_tick, initial_count as u64);
    assert_eq!(summary.deaths_this_tick, 0);
    assert_eq!(pop.len(), initial_count * 2);
    assert_eq!(pop.lifecycle_epoch(), 1);

    let ledger = analyze_lifecycle_events(pop.lifecycle_events()).expect("complete birth lifecycle");
    assert_eq!(ledger.founder_count(), initial_count);
    assert_eq!(ledger.birth_count(), initial_count);
    assert_eq!(ledger.death_count(), 0);

    for offspring in &pop.organisms[initial_count..] {
        let record = ledger.records().get(&offspring.id).expect("offspring record");
        assert_eq!(record.generation, 1);
        assert!(record.reproductive_parent_id.is_some());
        assert!(record.genome_source_id.is_some());
        assert_eq!(
            record.genome,
            GenomeEvidenceV1::from_genome(Genome::from_config(&offspring.cfg))
        );
    }

    for event in &pop.lifecycle_events()[initial_count..] {
        if let LifecycleTransitionV1::Birth {
            reproductive_parent_id,
            genome_source_id,
            offspring_id,
            reproductive_parent_genome,
            genome_source_genome,
            offspring_genome,
            ..
        } = event.transition
        {
            let parent = ledger
                .records()
                .get(&reproductive_parent_id)
                .expect("reproductive parent");
            let source = ledger
                .records()
                .get(&genome_source_id)
                .expect("genome source");
            let child = ledger.records().get(&offspring_id).expect("offspring");
            assert_eq!(reproductive_parent_genome, parent.genome);
            assert_eq!(genome_source_genome, source.genome);
            assert_eq!(offspring_genome, child.genome);
        }
    }
}

#[test]
fn draining_lifecycle_chunks_preserves_global_sequence_and_does_not_fake_completeness() {
    let mut pop = Population::new(base_cfg(), 3, 123);
    let founders = pop.drain_lifecycle_events();
    assert_eq!(founders.len(), 3);
    assert_eq!(founders[0].sequence, 0);
    assert_eq!(founders[2].sequence, 2);

    pop.cull_weakest(1);
    let later = pop.drain_lifecycle_events();
    assert_eq!(later.len(), 1);
    assert_eq!(later[0].sequence, 3);
    assert_eq!(later[0].tick, 0);
    assert!(matches!(
        analyze_lifecycle_events(&later),
        Err(LifecycleError::NonContiguousSequence {
            expected: 0,
            observed: 3,
        })
    ));
}
