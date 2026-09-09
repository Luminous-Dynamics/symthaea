// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed construction path for natural-trajectory stochastic evidence.
//!
//! [`crate::ObservedStochasticTapeV1`] is the persistence/validation contract. This recorder is a
//! narrower construction API for future live instrumentation: callers cannot insert an arbitrary
//! domain/value pair, canonical ordering is owned by `BTreeMap`, and duplicate semantic addresses
//! fail instead of silently overwriting earlier evidence.
//!
//! Finalization deliberately delegates semantic validation back to
//! [`crate::ObservedStochasticTapeV1::validate`]. This module therefore does not create a second
//! definition of valid stochastic evidence.
//!
//! **Claim boundary:** typed recording plus canonical finalization is not yet completeness proof.
//! A later live-owner tranche must prove that every stochastic decision made by the qualified
//! natural execution profile calls exactly one corresponding recorder method before the tape may be
//! described as complete.

use std::collections::BTreeMap;

use crate::{
    AgentId, GenomeTraitV1, ObservedSchedulerOrderKindV1, ObservedSchedulerOrderV1,
    ObservedStochasticDrawV1, ObservedStochasticKeyV1, ObservedStochasticTapeErrorV1,
    ObservedStochasticTapeSubjectV1, ObservedStochasticTapeV1, ObservedStochasticValueV1,
    ValidatedObservedStochasticTapeV1,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ObservedStochasticRecorderCountsV1 {
    pub inheritance_source_samples: usize,
    pub mutation_occurrence_samples: usize,
    pub mutation_perturbation_samples: usize,
    pub offspring_construction_seeds: usize,
    pub agent_action_samples: usize,
    pub random_population_orders: usize,
    pub fixed_partner_eligible_orders: usize,
}

impl ObservedStochasticRecorderCountsV1 {
    pub fn scalar_total(self) -> usize {
        self.inheritance_source_samples
            + self.mutation_occurrence_samples
            + self.mutation_perturbation_samples
            + self.offspring_construction_seeds
            + self.agent_action_samples
    }

    pub fn scheduler_total(self) -> usize {
        self.random_population_orders + self.fixed_partner_eligible_orders
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservedStochasticRecorderErrorV1 {
    InvalidUnitSample { bits: u64 },
    ZeroConstructionSeed,
    ReservedIdentity,
    DuplicateScalar { key: ObservedStochasticKeyV1 },
    DuplicateSchedulerOrder {
        tick: u64,
        kind: ObservedSchedulerOrderKindV1,
    },
    Tape(ObservedStochasticTapeErrorV1),
}

/// In-process builder for one observed natural stochastic interval.
///
/// The recorder is intentionally not serializable. Persist only the finalized raw tape returned by
/// [`ValidatedObservedStochasticTapeV1::persisted`], and revalidate it after loading.
pub struct ObservedStochasticRecorderV1 {
    start_behavior_next_tick: u64,
    start_lifecycle_next_sequence: u64,
    scalar_draws: BTreeMap<ObservedStochasticKeyV1, ObservedStochasticValueV1>,
    scheduler_orders: BTreeMap<(u64, ObservedSchedulerOrderKindV1), ObservedSchedulerOrderV1>,
    counts: ObservedStochasticRecorderCountsV1,
}

impl ObservedStochasticRecorderV1 {
    pub fn new(start_behavior_next_tick: u64, start_lifecycle_next_sequence: u64) -> Self {
        Self {
            start_behavior_next_tick,
            start_lifecycle_next_sequence,
            scalar_draws: BTreeMap::new(),
            scheduler_orders: BTreeMap::new(),
            counts: ObservedStochasticRecorderCountsV1::default(),
        }
    }

    pub fn start_behavior_next_tick(&self) -> u64 {
        self.start_behavior_next_tick
    }

    pub fn start_lifecycle_next_sequence(&self) -> u64 {
        self.start_lifecycle_next_sequence
    }

    pub fn counts(&self) -> ObservedStochasticRecorderCountsV1 {
        self.counts
    }

    pub fn record_inheritance_source_sample(
        &mut self,
        tick: u64,
        reproductive_parent_id: AgentId,
        birth_ordinal: u64,
        unit_sample: f64,
    ) -> Result<(), ObservedStochasticRecorderErrorV1> {
        reject_reserved(&[reproductive_parent_id])?;
        let key = ObservedStochasticKeyV1::inheritance_source_sample(
            tick,
            reproductive_parent_id,
            birth_ordinal,
        );
        self.insert_unit(key, unit_sample)?;
        self.counts.inheritance_source_samples += 1;
        Ok(())
    }

    pub fn record_mutation_occurrence(
        &mut self,
        tick: u64,
        reproductive_parent_id: AgentId,
        genome_source_id: AgentId,
        birth_ordinal: u64,
        trait_id: GenomeTraitV1,
        unit_sample: f64,
    ) -> Result<(), ObservedStochasticRecorderErrorV1> {
        reject_reserved(&[reproductive_parent_id, genome_source_id])?;
        let key = ObservedStochasticKeyV1::mutation_occurrence(
            tick,
            reproductive_parent_id,
            genome_source_id,
            birth_ordinal,
            trait_id,
        );
        self.insert_unit(key, unit_sample)?;
        self.counts.mutation_occurrence_samples += 1;
        Ok(())
    }

    pub fn record_mutation_perturbation(
        &mut self,
        tick: u64,
        reproductive_parent_id: AgentId,
        genome_source_id: AgentId,
        birth_ordinal: u64,
        trait_id: GenomeTraitV1,
        unit_sample: f64,
    ) -> Result<(), ObservedStochasticRecorderErrorV1> {
        reject_reserved(&[reproductive_parent_id, genome_source_id])?;
        let key = ObservedStochasticKeyV1::mutation_perturbation(
            tick,
            reproductive_parent_id,
            genome_source_id,
            birth_ordinal,
            trait_id,
        );
        self.insert_unit(key, unit_sample)?;
        self.counts.mutation_perturbation_samples += 1;
        Ok(())
    }

    pub fn record_offspring_construction_seed(
        &mut self,
        tick: u64,
        reproductive_parent_id: AgentId,
        birth_ordinal: u64,
        seed: u64,
    ) -> Result<(), ObservedStochasticRecorderErrorV1> {
        reject_reserved(&[reproductive_parent_id])?;
        if seed == 0 {
            return Err(ObservedStochasticRecorderErrorV1::ZeroConstructionSeed);
        }
        let key = ObservedStochasticKeyV1::offspring_construction_seed(
            tick,
            reproductive_parent_id,
            birth_ordinal,
        );
        self.insert_scalar(key, ObservedStochasticValueV1::nonzero_u64(seed))?;
        self.counts.offspring_construction_seeds += 1;
        Ok(())
    }

    pub fn record_agent_action_sample(
        &mut self,
        tick: u64,
        agent_id: AgentId,
        draw_ordinal: u64,
        unit_sample: f64,
    ) -> Result<(), ObservedStochasticRecorderErrorV1> {
        reject_reserved(&[agent_id])?;
        let key = ObservedStochasticKeyV1::agent_action_sample(tick, agent_id, draw_ordinal);
        self.insert_unit(key, unit_sample)?;
        self.counts.agent_action_samples += 1;
        Ok(())
    }

    pub fn record_scheduler_order(
        &mut self,
        tick: u64,
        kind: ObservedSchedulerOrderKindV1,
        input_order: Vec<AgentId>,
        shuffled_order: Vec<AgentId>,
    ) -> Result<(), ObservedStochasticRecorderErrorV1> {
        reject_reserved(&input_order)?;
        reject_reserved(&shuffled_order)?;
        let key = (tick, kind);
        if self.scheduler_orders.contains_key(&key) {
            return Err(ObservedStochasticRecorderErrorV1::DuplicateSchedulerOrder {
                tick,
                kind,
            });
        }
        self.scheduler_orders.insert(
            key,
            ObservedSchedulerOrderV1 {
                tick,
                kind,
                input_order,
                shuffled_order,
            },
        );
        match kind {
            ObservedSchedulerOrderKindV1::RandomPopulation => {
                self.counts.random_population_orders += 1;
            }
            ObservedSchedulerOrderKindV1::FixedPartnerEligible => {
                self.counts.fixed_partner_eligible_orders += 1;
            }
        }
        Ok(())
    }

    /// Finalize into the canonical tape contract and re-run its complete semantic validator.
    pub fn finish(
        self,
        end_behavior_next_tick: u64,
    ) -> Result<ValidatedObservedStochasticTapeV1, ObservedStochasticRecorderErrorV1> {
        let scalar_draws = self
            .scalar_draws
            .into_iter()
            .map(|(key, value)| ObservedStochasticDrawV1 { key, value })
            .collect();
        let scheduler_orders = self.scheduler_orders.into_values().collect();
        ObservedStochasticTapeV1 {
            subject: ObservedStochasticTapeSubjectV1 {
                start_behavior_next_tick: self.start_behavior_next_tick,
                start_lifecycle_next_sequence: self.start_lifecycle_next_sequence,
                end_behavior_next_tick,
            },
            scalar_draws,
            scheduler_orders,
        }
        .validate()
        .map_err(ObservedStochasticRecorderErrorV1::Tape)
    }

    fn insert_unit(
        &mut self,
        key: ObservedStochasticKeyV1,
        value: f64,
    ) -> Result<(), ObservedStochasticRecorderErrorV1> {
        if !value.is_finite() || !(0.0..1.0).contains(&value) {
            return Err(ObservedStochasticRecorderErrorV1::InvalidUnitSample {
                bits: value.to_bits(),
            });
        }
        self.insert_scalar(key, ObservedStochasticValueV1::unit(value))
    }

    fn insert_scalar(
        &mut self,
        key: ObservedStochasticKeyV1,
        value: ObservedStochasticValueV1,
    ) -> Result<(), ObservedStochasticRecorderErrorV1> {
        if self.scalar_draws.contains_key(&key) {
            return Err(ObservedStochasticRecorderErrorV1::DuplicateScalar { key });
        }
        self.scalar_draws.insert(key, value);
        Ok(())
    }
}

fn reject_reserved(ids: &[AgentId]) -> Result<(), ObservedStochasticRecorderErrorV1> {
    if ids.iter().any(|&id| id == AgentId::UNALLOCATED) {
        return Err(ObservedStochasticRecorderErrorV1::ReservedIdentity);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentIdAllocator;

    fn ids(n: usize) -> Vec<AgentId> {
        let mut alloc = AgentIdAllocator::new();
        (0..n).map(|_| alloc.allocate()).collect()
    }

    #[test]
    fn arbitrary_recording_order_finalizes_to_canonical_tape_order() {
        let ids = ids(3);
        let mut recorder = ObservedStochasticRecorderV1::new(10, 7);
        recorder
            .record_agent_action_sample(11, ids[2], 0, 0.75)
            .unwrap();
        recorder
            .record_inheritance_source_sample(10, ids[0], 0, 0.25)
            .unwrap();
        recorder
            .record_agent_action_sample(10, ids[1], 0, 0.5)
            .unwrap();

        let tape = recorder.finish(12).expect("canonical tape");
        let draws = &tape.persisted().scalar_draws;
        assert_eq!(draws.len(), 3);
        assert!(draws.windows(2).all(|pair| pair[0].key < pair[1].key));
        assert_eq!(tape.subject().start_behavior_next_tick, 10);
        assert_eq!(tape.subject().start_lifecycle_next_sequence, 7);
        assert_eq!(tape.subject().end_behavior_next_tick, 12);
    }

    #[test]
    fn duplicate_semantic_address_fails_without_overwriting_first_observation() {
        let id = ids(1)[0];
        let mut recorder = ObservedStochasticRecorderV1::new(0, 1);
        recorder
            .record_agent_action_sample(0, id, 0, 0.125)
            .unwrap();
        assert!(matches!(
            recorder.record_agent_action_sample(0, id, 0, 0.875),
            Err(ObservedStochasticRecorderErrorV1::DuplicateScalar { .. })
        ));
        assert_eq!(recorder.counts().agent_action_samples, 1);

        let tape = recorder.finish(1).unwrap();
        let key = ObservedStochasticKeyV1::agent_action_sample(0, id, 0);
        assert_eq!(tape.unit_for(key).unwrap().to_bits(), 0.125f64.to_bits());
    }

    #[test]
    fn typed_methods_emit_exact_domain_and_value_kinds() {
        let ids = ids(2);
        let mut recorder = ObservedStochasticRecorderV1::new(4, 20);
        recorder
            .record_inheritance_source_sample(4, ids[0], 0, 0.1)
            .unwrap();
        recorder
            .record_mutation_occurrence(
                4,
                ids[0],
                ids[1],
                0,
                GenomeTraitV1::SetPoint,
                0.2,
            )
            .unwrap();
        recorder
            .record_mutation_perturbation(
                4,
                ids[0],
                ids[1],
                0,
                GenomeTraitV1::SetPoint,
                0.3,
            )
            .unwrap();
        recorder
            .record_offspring_construction_seed(4, ids[0], 0, 99)
            .unwrap();
        recorder
            .record_agent_action_sample(4, ids[1], 0, 0.4)
            .unwrap();

        let counts = recorder.counts();
        assert_eq!(counts.scalar_total(), 5);
        assert_eq!(counts.inheritance_source_samples, 1);
        assert_eq!(counts.mutation_occurrence_samples, 1);
        assert_eq!(counts.mutation_perturbation_samples, 1);
        assert_eq!(counts.offspring_construction_seeds, 1);
        assert_eq!(counts.agent_action_samples, 1);

        let tape = recorder.finish(5).unwrap();
        let seed_key = ObservedStochasticKeyV1::offspring_construction_seed(4, ids[0], 0);
        assert_eq!(tape.nonzero_u64_for(seed_key), Some(99));
    }

    #[test]
    fn scheduler_observation_is_unique_per_tick_and_kind_and_validated_at_finish() {
        let ids = ids(4);
        let mut recorder = ObservedStochasticRecorderV1::new(2, 8);
        recorder
            .record_scheduler_order(
                2,
                ObservedSchedulerOrderKindV1::RandomPopulation,
                ids.clone(),
                vec![ids[2], ids[0], ids[3], ids[1]],
            )
            .unwrap();
        assert!(matches!(
            recorder.record_scheduler_order(
                2,
                ObservedSchedulerOrderKindV1::RandomPopulation,
                ids.clone(),
                ids.clone(),
            ),
            Err(ObservedStochasticRecorderErrorV1::DuplicateSchedulerOrder { .. })
        ));
        assert_eq!(recorder.counts().scheduler_total(), 1);
        let tape = recorder.finish(3).unwrap();
        assert!(
            tape.scheduler_order(2, ObservedSchedulerOrderKindV1::RandomPopulation)
                .is_some()
        );
    }

    #[test]
    fn invalid_scalar_and_seed_fail_before_mutating_recorder() {
        let id = ids(1)[0];
        let mut recorder = ObservedStochasticRecorderV1::new(0, 0);
        assert!(matches!(
            recorder.record_agent_action_sample(0, id, 0, 1.0),
            Err(ObservedStochasticRecorderErrorV1::InvalidUnitSample { .. })
        ));
        assert!(matches!(
            recorder.record_offspring_construction_seed(0, id, 0, 0),
            Err(ObservedStochasticRecorderErrorV1::ZeroConstructionSeed)
        ));
        assert_eq!(recorder.counts().scalar_total(), 0);
    }

    #[test]
    fn canonical_tape_validator_remains_final_authority() {
        let ids = ids(2);
        let mut recorder = ObservedStochasticRecorderV1::new(5, 1);
        // Input/shuffled membership differs. The typed recorder accepts the scheduler-shaped
        // observation, but final tape validation must reject the semantic inconsistency.
        recorder
            .record_scheduler_order(
                5,
                ObservedSchedulerOrderKindV1::RandomPopulation,
                ids.clone(),
                vec![ids[0], ids[0]],
            )
            .unwrap();
        assert!(matches!(
            recorder.finish(6),
            Err(ObservedStochasticRecorderErrorV1::Tape(
                ObservedStochasticTapeErrorV1::SchedulerDuplicateIdentity { .. }
            ))
        ));
    }
}
