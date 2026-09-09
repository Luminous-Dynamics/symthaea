// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact semantic-key coverage for natural Genesis stochastic evidence.
//!
//! A structurally valid [`crate::ObservedStochasticTapeV1`] can still omit a natural stochastic
//! decision. This module derives the stochastic addresses required by the qualified Earth-forced
//! Genesis social profile from independent authorities: complete behavioral batches, the exact
//! lifecycle extension, population policy, and scheduler state at the fork.
//!
//! Coverage and origin are deliberately separate claims. Successful validation proves that every
//! stochastic *address* required by the current profile is represented exactly once and that every
//! scheduler order is behaviorally consistent. It does **not** prove that the recorded scalar value
//! was emitted by the corresponding live RNG owner. Live recorder wiring / stochastic-origin
//! receipts remain a later tranche.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    AgentId, GenesisEvent, GenomeEvidenceV1, GenomeTraitV1, InheritanceMode,
    LifecycleTransitionV1, ObservedSchedulerOrderKindV1, ObservedStochasticKeyV1, PairingMode,
    ValidatedGenesisEarthExecutionCapsuleV1, ValidatedObservedStochasticTapeV1,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ObservedStochasticCoverageCountsV1 {
    pub action_samples: usize,
    pub inheritance_source_samples: usize,
    pub mutation_occurrence_samples: usize,
    pub mutation_perturbation_samples: usize,
    pub offspring_construction_seeds: usize,
    pub scheduler_orders: usize,
    pub births: usize,
    pub behavior_ticks: usize,
}

impl ObservedStochasticCoverageCountsV1 {
    pub fn expected_scalar_total(self) -> usize {
        self.action_samples
            + self.inheritance_source_samples
            + self.mutation_occurrence_samples
            + self.mutation_perturbation_samples
            + self.offspring_construction_seeds
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedObservedStochasticCoverageV1 {
    counts: ObservedStochasticCoverageCountsV1,
}

impl ValidatedObservedStochasticCoverageV1 {
    pub fn counts(&self) -> ObservedStochasticCoverageCountsV1 {
        self.counts
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservedStochasticCoverageErrorV1 {
    BehaviorStartMismatch { expected: u64, observed: u64 },
    TapeStartBehaviorMismatch { expected: u64, observed: u64 },
    TapeStartLifecycleMismatch { expected: u64, observed: u64 },
    TapeEndBehaviorMismatch { expected: u64, observed: u64 },
    PopulationPolicyChanged,
    SchedulerModeChanged,
    ForkLifecycleReconstruction,
    NaturalLifecycleReconstruction,
    LifecyclePrefixMismatch,
    BirthOutsideBehaviorInterval { sequence: u64, tick: u64 },
    BehaviorBatchInvalid,
    DuplicateExpectedScalar { key: ObservedStochasticKeyV1 },
    MissingOccurrenceSample { key: ObservedStochasticKeyV1 },
    MissingScalar { key: ObservedStochasticKeyV1 },
    ExtraScalar { key: ObservedStochasticKeyV1 },
    MissingSchedulerOrder { tick: u64, kind: ObservedSchedulerOrderKindV1 },
    ExtraSchedulerOrder { tick: u64, kind: ObservedSchedulerOrderKindV1 },
    SchedulerInputOrderMismatch { tick: u64, kind: ObservedSchedulerOrderKindV1 },
    SchedulerPairingMismatch { tick: u64 },
    SchedulerFinalMapMismatch,
    CountOverflow { field: &'static str },
}

/// Prove exact stochastic-address coverage for one already-validated natural execution interval.
pub fn validate_observed_stochastic_coverage_v1(
    fork: &ValidatedGenesisEarthExecutionCapsuleV1,
    natural_end: &ValidatedGenesisEarthExecutionCapsuleV1,
    tape: &ValidatedObservedStochasticTapeV1,
) -> Result<ValidatedObservedStochasticCoverageV1, ObservedStochasticCoverageErrorV1> {
    let fork_tick = fork.evidence().behavior_next_tick();
    let natural_behavior = natural_end.evidence();
    if natural_behavior.behavior_start_tick() != fork_tick {
        return Err(ObservedStochasticCoverageErrorV1::BehaviorStartMismatch {
            expected: fork_tick,
            observed: natural_behavior.behavior_start_tick(),
        });
    }
    let end_tick = natural_behavior.behavior_next_tick();

    let subject = tape.subject();
    if subject.start_behavior_next_tick != fork_tick {
        return Err(ObservedStochasticCoverageErrorV1::TapeStartBehaviorMismatch {
            expected: fork_tick,
            observed: subject.start_behavior_next_tick,
        });
    }
    let fork_sequence = fork.evidence().lifecycle().next_sequence();
    if subject.start_lifecycle_next_sequence != fork_sequence {
        return Err(ObservedStochasticCoverageErrorV1::TapeStartLifecycleMismatch {
            expected: fork_sequence,
            observed: subject.start_lifecycle_next_sequence,
        });
    }
    if subject.end_behavior_next_tick != end_tick {
        return Err(ObservedStochasticCoverageErrorV1::TapeEndBehaviorMismatch {
            expected: end_tick,
            observed: subject.end_behavior_next_tick,
        });
    }

    if fork.population().config().as_snapshot() != natural_end.population().config().as_snapshot() {
        return Err(ObservedStochasticCoverageErrorV1::PopulationPolicyChanged);
    }
    if fork.scheduler().mode() != natural_end.scheduler().mode() {
        return Err(ObservedStochasticCoverageErrorV1::SchedulerModeChanged);
    }
    let cfg = fork.population().config().config();

    validate_lifecycle_extension(fork, natural_end)?;

    let mut expected_scalars = BTreeSet::new();
    let mut counts = ObservedStochasticCoverageCountsV1::default();
    for raw_batch in natural_behavior.behavior_batches() {
        let batch = raw_batch
            .clone()
            .validate()
            .map_err(|_| ObservedStochasticCoverageErrorV1::BehaviorBatchInvalid)?;
        counts.behavior_ticks = checked_increment(counts.behavior_ticks, "behavior_ticks")?;
        for event in batch.events() {
            insert_expected(
                &mut expected_scalars,
                ObservedStochasticKeyV1::agent_action_sample(batch.tick(), event.agent_id, 0),
            )?;
            counts.action_samples = checked_increment(counts.action_samples, "action_samples")?;
        }
    }

    derive_birth_scalar_coverage(
        fork,
        natural_end,
        tape,
        cfg.inheritance,
        cfg.mutation_rate,
        &mut expected_scalars,
        &mut counts,
    )?;

    compare_scalar_sets(&expected_scalars, tape)?;
    validate_scheduler_coverage(fork, natural_end, tape, &mut counts)?;

    Ok(ValidatedObservedStochasticCoverageV1 { counts })
}

fn validate_lifecycle_extension(
    fork: &ValidatedGenesisEarthExecutionCapsuleV1,
    natural_end: &ValidatedGenesisEarthExecutionCapsuleV1,
) -> Result<(), ObservedStochasticCoverageErrorV1> {
    let fork_prefix = fork
        .evidence()
        .lifecycle()
        .reconstruct_complete_prefix()
        .map_err(|_| ObservedStochasticCoverageErrorV1::ForkLifecycleReconstruction)?;
    let end_prefix = natural_end
        .evidence()
        .lifecycle()
        .reconstruct_complete_prefix()
        .map_err(|_| ObservedStochasticCoverageErrorV1::NaturalLifecycleReconstruction)?;
    if end_prefix.len() < fork_prefix.len() || end_prefix[..fork_prefix.len()] != fork_prefix[..] {
        return Err(ObservedStochasticCoverageErrorV1::LifecyclePrefixMismatch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn derive_birth_scalar_coverage(
    fork: &ValidatedGenesisEarthExecutionCapsuleV1,
    natural_end: &ValidatedGenesisEarthExecutionCapsuleV1,
    tape: &ValidatedObservedStochasticTapeV1,
    inheritance: InheritanceMode,
    mutation_rate: f64,
    expected: &mut BTreeSet<ObservedStochasticKeyV1>,
    counts: &mut ObservedStochasticCoverageCountsV1,
) -> Result<(), ObservedStochasticCoverageErrorV1> {
    let start_sequence = fork.evidence().lifecycle().next_sequence();
    let start_tick = fork.evidence().behavior_next_tick();
    let end_tick = natural_end.evidence().behavior_next_tick();
    let events = natural_end
        .evidence()
        .lifecycle()
        .reconstruct_complete_prefix()
        .map_err(|_| ObservedStochasticCoverageErrorV1::NaturalLifecycleReconstruction)?;
    let mut birth_ordinals = BTreeMap::<(u64, AgentId), u64>::new();

    for event in events.into_iter().filter(|event| event.sequence >= start_sequence) {
        let LifecycleTransitionV1::Birth {
            reproductive_parent_id,
            genome_source_id,
            genome_source_genome,
            ..
        } = event.transition
        else {
            // Death/cull evidence may legitimately be appended at the end behavior boundary
            // without consuming any stochastic input. Coverage only constrains stochastic events.
            continue;
        };
        if event.tick < start_tick || event.tick >= end_tick {
            return Err(ObservedStochasticCoverageErrorV1::BirthOutsideBehaviorInterval {
                sequence: event.sequence,
                tick: event.tick,
            });
        }

        counts.births = checked_increment(counts.births, "births")?;
        let ordinal_entry = birth_ordinals
            .entry((event.tick, reproductive_parent_id))
            .or_insert(0);
        let birth_ordinal = *ordinal_entry;
        *ordinal_entry = ordinal_entry
            .checked_add(1)
            .ok_or(ObservedStochasticCoverageErrorV1::CountOverflow {
                field: "birth_ordinal",
            })?;

        if inheritance == InheritanceMode::RandomPeer {
            insert_expected(
                expected,
                ObservedStochasticKeyV1::inheritance_source_sample(
                    event.tick,
                    reproductive_parent_id,
                    birth_ordinal,
                ),
            )?;
            counts.inheritance_source_samples = checked_increment(
                counts.inheritance_source_samples,
                "inheritance_source_samples",
            )?;
        }

        insert_expected(
            expected,
            ObservedStochasticKeyV1::offspring_construction_seed(
                event.tick,
                reproductive_parent_id,
                birth_ordinal,
            ),
        )?;
        counts.offspring_construction_seeds = checked_increment(
            counts.offspring_construction_seeds,
            "offspring_construction_seeds",
        )?;

        for trait_id in active_genome_traits(genome_source_genome) {
            let occurrence = ObservedStochasticKeyV1::mutation_occurrence(
                event.tick,
                reproductive_parent_id,
                genome_source_id,
                birth_ordinal,
                trait_id,
            );
            insert_expected(expected, occurrence)?;
            counts.mutation_occurrence_samples = checked_increment(
                counts.mutation_occurrence_samples,
                "mutation_occurrence_samples",
            )?;

            let sample = tape
                .unit_for(occurrence)
                .ok_or(ObservedStochasticCoverageErrorV1::MissingOccurrenceSample {
                    key: occurrence,
                })?;
            if sample < mutation_rate {
                let perturbation = ObservedStochasticKeyV1::mutation_perturbation(
                    event.tick,
                    reproductive_parent_id,
                    genome_source_id,
                    birth_ordinal,
                    trait_id,
                );
                insert_expected(expected, perturbation)?;
                counts.mutation_perturbation_samples = checked_increment(
                    counts.mutation_perturbation_samples,
                    "mutation_perturbation_samples",
                )?;
            }
        }
    }
    Ok(())
}

fn active_genome_traits(genome: GenomeEvidenceV1) -> Vec<GenomeTraitV1> {
    let mut traits = vec![
        GenomeTraitV1::SetPoint,
        GenomeTraitV1::ForageEfficiency,
        GenomeTraitV1::ActionTemperature,
    ];
    if genome.perceptual_grain_bits.is_some() {
        traits.push(GenomeTraitV1::PerceptualGrain);
    }
    traits
}

fn checked_increment(
    value: usize,
    field: &'static str,
) -> Result<usize, ObservedStochasticCoverageErrorV1> {
    value
        .checked_add(1)
        .ok_or(ObservedStochasticCoverageErrorV1::CountOverflow { field })
}

fn insert_expected(
    expected: &mut BTreeSet<ObservedStochasticKeyV1>,
    key: ObservedStochasticKeyV1,
) -> Result<(), ObservedStochasticCoverageErrorV1> {
    if !expected.insert(key) {
        return Err(ObservedStochasticCoverageErrorV1::DuplicateExpectedScalar { key });
    }
    Ok(())
}

fn compare_scalar_sets(
    expected: &BTreeSet<ObservedStochasticKeyV1>,
    tape: &ValidatedObservedStochasticTapeV1,
) -> Result<(), ObservedStochasticCoverageErrorV1> {
    let observed = tape
        .persisted()
        .scalar_draws
        .iter()
        .map(|draw| draw.key)
        .collect::<BTreeSet<_>>();
    if let Some(&key) = expected.difference(&observed).next() {
        return Err(ObservedStochasticCoverageErrorV1::MissingScalar { key });
    }
    if let Some(&key) = observed.difference(expected).next() {
        return Err(ObservedStochasticCoverageErrorV1::ExtraScalar { key });
    }
    Ok(())
}

fn validate_scheduler_coverage(
    fork: &ValidatedGenesisEarthExecutionCapsuleV1,
    natural_end: &ValidatedGenesisEarthExecutionCapsuleV1,
    tape: &ValidatedObservedStochasticTapeV1,
    counts: &mut ObservedStochasticCoverageCountsV1,
) -> Result<(), ObservedStochasticCoverageErrorV1> {
    let mode = fork.scheduler().mode();
    let mut fixed = fork
        .scheduler()
        .fixed_partners()
        .iter()
        .map(|entry| (entry.agent_id(), entry.partner_id()))
        .collect::<BTreeMap<_, _>>();
    let mut expected_orders = BTreeSet::<(u64, ObservedSchedulerOrderKindV1)>::new();

    for raw_batch in natural_end.evidence().behavior_batches() {
        let batch = raw_batch
            .clone()
            .validate()
            .map_err(|_| ObservedStochasticCoverageErrorV1::BehaviorBatchInvalid)?;
        let living = batch
            .events()
            .iter()
            .map(|event| event.agent_id)
            .collect::<Vec<_>>();

        match mode {
            PairingMode::Random => {
                let shuffled = if living.len() >= 2 {
                    let kind = ObservedSchedulerOrderKindV1::RandomPopulation;
                    expected_orders.insert((batch.tick(), kind));
                    counts.scheduler_orders =
                        checked_increment(counts.scheduler_orders, "scheduler_orders")?;
                    let order = tape.scheduler_order(batch.tick(), kind).ok_or(
                        ObservedStochasticCoverageErrorV1::MissingSchedulerOrder {
                            tick: batch.tick(),
                            kind,
                        },
                    )?;
                    if order.input_order != living {
                        return Err(
                            ObservedStochasticCoverageErrorV1::SchedulerInputOrderMismatch {
                                tick: batch.tick(),
                                kind,
                            },
                        );
                    }
                    order.shuffled_order.clone()
                } else {
                    living.clone()
                };
                if scheduler_pairs_from_order(&shuffled) != behavior_pairs(batch.events()) {
                    return Err(ObservedStochasticCoverageErrorV1::SchedulerPairingMismatch {
                        tick: batch.tick(),
                    });
                }
            }
            PairingMode::FixedPartners => {
                let living_set = living.iter().copied().collect::<BTreeSet<_>>();
                let eligible = living
                    .iter()
                    .copied()
                    .filter(|id| match fixed.get(id) {
                        None => true,
                        Some(partner) => !living_set.contains(partner),
                    })
                    .collect::<Vec<_>>();
                if eligible.len() >= 2 {
                    let kind = ObservedSchedulerOrderKindV1::FixedPartnerEligible;
                    expected_orders.insert((batch.tick(), kind));
                    counts.scheduler_orders =
                        checked_increment(counts.scheduler_orders, "scheduler_orders")?;
                    let order = tape.scheduler_order(batch.tick(), kind).ok_or(
                        ObservedStochasticCoverageErrorV1::MissingSchedulerOrder {
                            tick: batch.tick(),
                            kind,
                        },
                    )?;
                    if order.input_order != eligible {
                        return Err(
                            ObservedStochasticCoverageErrorV1::SchedulerInputOrderMismatch {
                                tick: batch.tick(),
                                kind,
                            },
                        );
                    }
                    for pair in order.shuffled_order.chunks_exact(2) {
                        fixed.insert(pair[0], pair[1]);
                        fixed.insert(pair[1], pair[0]);
                    }
                }
                let expected_pairs = fixed
                    .iter()
                    .filter_map(|(&id, &partner)| {
                        if living_set.contains(&id)
                            && living_set.contains(&partner)
                            && id < partner
                        {
                            Some((id, partner))
                        } else {
                            None
                        }
                    })
                    .collect::<BTreeSet<_>>();
                if expected_pairs != behavior_pairs(batch.events()) {
                    return Err(ObservedStochasticCoverageErrorV1::SchedulerPairingMismatch {
                        tick: batch.tick(),
                    });
                }
            }
        }
    }

    let observed_orders = tape
        .persisted()
        .scheduler_orders
        .iter()
        .map(|order| (order.tick, order.kind))
        .collect::<BTreeSet<_>>();
    if let Some(&(tick, kind)) = expected_orders.difference(&observed_orders).next() {
        return Err(ObservedStochasticCoverageErrorV1::MissingSchedulerOrder { tick, kind });
    }
    if let Some(&(tick, kind)) = observed_orders.difference(&expected_orders).next() {
        return Err(ObservedStochasticCoverageErrorV1::ExtraSchedulerOrder { tick, kind });
    }

    if mode == PairingMode::FixedPartners {
        let observed_final = natural_end
            .scheduler()
            .fixed_partners()
            .iter()
            .map(|entry| (entry.agent_id(), entry.partner_id()))
            .collect::<BTreeMap<_, _>>();
        if fixed != observed_final {
            return Err(ObservedStochasticCoverageErrorV1::SchedulerFinalMapMismatch);
        }
    }
    Ok(())
}

fn scheduler_pairs_from_order(order: &[AgentId]) -> BTreeSet<(AgentId, AgentId)> {
    order
        .chunks_exact(2)
        .map(|pair| {
            if pair[0] < pair[1] {
                (pair[0], pair[1])
            } else {
                (pair[1], pair[0])
            }
        })
        .collect()
}

fn behavior_pairs(events: &[GenesisEvent]) -> BTreeSet<(AgentId, AgentId)> {
    events
        .iter()
        .filter_map(|event| {
            event.partner_id.and_then(|partner| {
                if event.agent_id < partner {
                    Some((event.agent_id, partner))
                } else {
                    None
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EarthForcedEnvironment, GenesisEarthExecutionV1, ObservedSchedulerOrderV1,
        ObservedStochasticDrawV1, ObservedStochasticTapeSubjectV1, ObservedStochasticTapeV1,
        ObservedStochasticValueV1, OrganismConfig, PopulationConfig,
    };

    fn quiet_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 2.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                social_enabled: true,
                ..OrganismConfig::default()
            },
            mutation_rate: 0.0,
            mutation_std: 0.05,
            inheritance: InheritanceMode::FromParent,
        }
    }

    fn reconstruct_random_shuffle_from_behavior(events: &[GenesisEvent]) -> Vec<AgentId> {
        let mut remaining = events
            .iter()
            .map(|event| event.agent_id)
            .collect::<BTreeSet<_>>();
        let mut order = Vec::new();
        for event in events {
            if !remaining.remove(&event.agent_id) {
                continue;
            }
            if let Some(partner) = event.partner_id {
                if remaining.remove(&partner) {
                    order.push(event.agent_id);
                    order.push(partner);
                }
            }
        }
        if let Some(leftover) = remaining.into_iter().next() {
            order.push(leftover);
        }
        order
    }

    fn build_random_tape(
        fork: &ValidatedGenesisEarthExecutionCapsuleV1,
        end: &ValidatedGenesisEarthExecutionCapsuleV1,
        include_birth_keys: bool,
    ) -> ValidatedObservedStochasticTapeV1 {
        let mut scalar_draws = Vec::new();
        let mut scheduler_orders = Vec::new();
        for raw_batch in end.evidence().behavior_batches() {
            let batch = raw_batch.clone().validate().unwrap();
            for event in batch.events() {
                scalar_draws.push(ObservedStochasticDrawV1 {
                    key: ObservedStochasticKeyV1::agent_action_sample(batch.tick(), event.agent_id, 0),
                    value: ObservedStochasticValueV1::unit(0.5),
                });
            }
            if batch.events().len() >= 2 {
                scheduler_orders.push(ObservedSchedulerOrderV1 {
                    tick: batch.tick(),
                    kind: ObservedSchedulerOrderKindV1::RandomPopulation,
                    input_order: batch.events().iter().map(|event| event.agent_id).collect(),
                    shuffled_order: reconstruct_random_shuffle_from_behavior(batch.events()),
                });
            }
        }

        if include_birth_keys {
            let start_sequence = fork.evidence().lifecycle().next_sequence();
            let mut ordinals = BTreeMap::<(u64, AgentId), u64>::new();
            for lifecycle_event in end
                .evidence()
                .lifecycle()
                .reconstruct_complete_prefix()
                .unwrap()
                .into_iter()
                .filter(|event| event.sequence >= start_sequence)
            {
                let LifecycleTransitionV1::Birth {
                    reproductive_parent_id,
                    genome_source_id,
                    genome_source_genome,
                    ..
                } = lifecycle_event.transition
                else {
                    continue;
                };
                let ordinal = ordinals
                    .entry((lifecycle_event.tick, reproductive_parent_id))
                    .or_insert(0);
                let birth_ordinal = *ordinal;
                *ordinal += 1;
                scalar_draws.push(ObservedStochasticDrawV1 {
                    key: ObservedStochasticKeyV1::offspring_construction_seed(
                        lifecycle_event.tick,
                        reproductive_parent_id,
                        birth_ordinal,
                    ),
                    value: ObservedStochasticValueV1::nonzero_u64(99 + birth_ordinal),
                });
                for trait_id in active_genome_traits(genome_source_genome) {
                    scalar_draws.push(ObservedStochasticDrawV1 {
                        key: ObservedStochasticKeyV1::mutation_occurrence(
                            lifecycle_event.tick,
                            reproductive_parent_id,
                            genome_source_id,
                            birth_ordinal,
                            trait_id,
                        ),
                        value: ObservedStochasticValueV1::unit(0.5),
                    });
                }
            }
        }

        scalar_draws.sort_by_key(|draw| draw.key);
        scheduler_orders.sort_by_key(|order| (order.tick, order.kind));
        ObservedStochasticTapeV1 {
            subject: ObservedStochasticTapeSubjectV1 {
                start_behavior_next_tick: fork.evidence().behavior_next_tick(),
                start_lifecycle_next_sequence: fork.evidence().lifecycle().next_sequence(),
                end_behavior_next_tick: end.evidence().behavior_next_tick(),
            },
            scalar_draws,
            scheduler_orders,
        }
        .validate()
        .unwrap()
    }

    fn random_interval(
        cfg: PopulationConfig,
        initial_count: usize,
        steps: usize,
    ) -> (
        ValidatedGenesisEarthExecutionCapsuleV1,
        ValidatedGenesisEarthExecutionCapsuleV1,
    ) {
        let mut execution = GenesisEarthExecutionV1::new(
            cfg,
            initial_count,
            101,
            PairingMode::Random,
            202,
            EarthForcedEnvironment::earth_like(79.0),
        )
        .unwrap();
        let fork = execution.checkpoint_execution().unwrap().validated().clone();
        for _ in 0..steps {
            execution.step_social().unwrap();
        }
        let end = execution.checkpoint_execution().unwrap().validated().clone();
        (fork, end)
    }

    #[test]
    fn random_profile_action_and_scheduler_key_coverage_is_exact() {
        let (fork, end) = random_interval(quiet_cfg(), 5, 4);
        let tape = build_random_tape(&fork, &end, false);
        let covered = validate_observed_stochastic_coverage_v1(&fork, &end, &tape).unwrap();
        assert_eq!(covered.counts().behavior_ticks, 4);
        assert_eq!(covered.counts().action_samples, 20);
        assert_eq!(covered.counts().scheduler_orders, 4);
        assert_eq!(covered.counts().births, 0);
    }

    #[test]
    fn reproduction_requires_seed_and_trait_occurrence_addresses() {
        let mut cfg = quiet_cfg();
        cfg.reproduction_energy_threshold = 0.0;
        let (fork, end) = random_interval(cfg, 2, 1);
        let tape = build_random_tape(&fork, &end, true);
        let covered = validate_observed_stochastic_coverage_v1(&fork, &end, &tape).unwrap();
        assert_eq!(covered.counts().births, 2);
        assert_eq!(covered.counts().action_samples, 2);
        assert_eq!(covered.counts().offspring_construction_seeds, 2);
        assert_eq!(covered.counts().mutation_occurrence_samples, 6);
        assert_eq!(covered.counts().mutation_perturbation_samples, 0);
        assert_eq!(covered.counts().expected_scalar_total(), 10);
    }

    #[test]
    fn missing_action_sample_fails_exact_key_coverage() {
        let (fork, end) = random_interval(quiet_cfg(), 5, 2);
        let complete = build_random_tape(&fork, &end, false);
        let mut raw = complete.persisted().clone();
        raw.scalar_draws.remove(0);
        let incomplete = raw.validate().unwrap();
        assert!(matches!(
            validate_observed_stochastic_coverage_v1(&fork, &end, &incomplete),
            Err(ObservedStochasticCoverageErrorV1::MissingScalar { .. })
        ));
    }

    #[test]
    fn scheduler_order_must_reproduce_observed_behavior_pairs() {
        let (fork, end) = random_interval(quiet_cfg(), 5, 2);
        let complete = build_random_tape(&fork, &end, false);
        let mut raw = complete.persisted().clone();
        raw.scheduler_orders[0].shuffled_order.swap(1, 2);
        let wrong = raw.validate().unwrap();
        assert!(matches!(
            validate_observed_stochastic_coverage_v1(&fork, &end, &wrong),
            Err(ObservedStochasticCoverageErrorV1::SchedulerPairingMismatch { .. })
        ));
    }

    #[test]
    fn fixed_partner_shadow_state_reproduces_behavior_and_final_map() {
        let mut execution = GenesisEarthExecutionV1::new(
            quiet_cfg(),
            4,
            303,
            PairingMode::FixedPartners,
            404,
            EarthForcedEnvironment::earth_like(79.0),
        )
        .unwrap();
        let fork = execution.checkpoint_execution().unwrap().validated().clone();
        execution.step_social().unwrap();
        let end = execution.checkpoint_execution().unwrap().validated().clone();
        let batch = end.evidence().behavior_batches()[0].clone().validate().unwrap();
        let mut shuffled = Vec::new();
        for event in batch.events() {
            if let Some(partner) = event.partner_id {
                if event.agent_id < partner {
                    shuffled.push(event.agent_id);
                    shuffled.push(partner);
                }
            }
        }
        let mut scalar_draws = batch
            .events()
            .iter()
            .map(|event| ObservedStochasticDrawV1 {
                key: ObservedStochasticKeyV1::agent_action_sample(0, event.agent_id, 0),
                value: ObservedStochasticValueV1::unit(0.5),
            })
            .collect::<Vec<_>>();
        scalar_draws.sort_by_key(|draw| draw.key);
        let tape = ObservedStochasticTapeV1 {
            subject: ObservedStochasticTapeSubjectV1 {
                start_behavior_next_tick: 0,
                start_lifecycle_next_sequence: fork.evidence().lifecycle().next_sequence(),
                end_behavior_next_tick: 1,
            },
            scalar_draws,
            scheduler_orders: vec![ObservedSchedulerOrderV1 {
                tick: 0,
                kind: ObservedSchedulerOrderKindV1::FixedPartnerEligible,
                input_order: batch.events().iter().map(|event| event.agent_id).collect(),
                shuffled_order: shuffled,
            }],
        }
        .validate()
        .unwrap();
        validate_observed_stochastic_coverage_v1(&fork, &end, &tape)
            .expect("fixed-partner stochastic coverage");
    }
}
