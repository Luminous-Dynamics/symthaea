// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-evidence checkpoint contract binding Genesis behavioral batches to lifecycle history.
//!
//! Behavioral rows and lifecycle transitions answer different questions. A complete behavioral
//! tick says who acted and interacted before that tick's population bookkeeping; lifecycle evidence
//! says who existed, who was born, and who died. Exact continuation should not trust those streams
//! independently when they can cross-check each other.
//!
//! This contract is scoped to [`crate::GenesisSocialRunnerV1`]'s execution profile: social steps
//! advance lifecycle epoch exactly once, while its only exposed out-of-band mutation (`cull_weakest`)
//! records lifecycle deaths without advancing the social clock. Under that profile the lifecycle
//! continuation epoch and behavioral `next_tick` must therefore agree.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    AgentId, GenesisTickBatchErrorV1, GenesisTickBatchV1, GenesisTickChunkSummaryV1,
    LifecycleCheckpointErrorV1, LifecycleCheckpointV1, LifecycleDeathCauseV1,
    ValidatedGenesisTickBatchV1, ValidatedLifecycleCheckpointV1, validate_genesis_tick_chunk,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisEvidenceCheckpointV1 {
    behavior_start_tick: u64,
    behavior_next_tick: u64,
    behavior_batches: Vec<GenesisTickBatchV1>,
    lifecycle_checkpoint: LifecycleCheckpointV1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedGenesisEvidenceCheckpointV1 {
    behavior_start_tick: u64,
    behavior_next_tick: u64,
    behavior_batches: Vec<GenesisTickBatchV1>,
    behavior_summary: GenesisTickChunkSummaryV1,
    lifecycle: ValidatedLifecycleCheckpointV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenesisEvidenceCheckpointErrorV1 {
    Lifecycle(LifecycleCheckpointErrorV1),
    Behavior(GenesisTickBatchErrorV1),
    BehaviorStartMismatch {
        expected: u64,
        observed: u64,
    },
    BehaviorEndMismatch {
        expected: u64,
        observed: u64,
    },
    CrossClockMismatch {
        behavior_next_tick: u64,
        lifecycle_continuation_epoch: u64,
    },
    BehaviorLivingSetMismatch {
        tick: u64,
        expected: BTreeSet<AgentId>,
        observed: BTreeSet<AgentId>,
    },
    UnknownBehaviorAgent {
        tick: u64,
        agent_id: AgentId,
    },
    BehaviorIdentityMismatch {
        tick: u64,
        agent_id: AgentId,
        expected_lineage_id: AgentId,
        observed_lineage_id: AgentId,
        expected_generation: u32,
        observed_generation: u32,
    },
}

impl GenesisEvidenceCheckpointV1 {
    /// Construct a raw persistence packet from already collected evidence surfaces.
    ///
    /// This constructor is crate-visible for the qualified runner integration. The returned raw
    /// value is not authority-bearing until [`Self::validate_after`] succeeds.
    pub(crate) fn new(
        behavior_start_tick: u64,
        behavior_next_tick: u64,
        behavior_batches: Vec<GenesisTickBatchV1>,
        lifecycle_checkpoint: LifecycleCheckpointV1,
    ) -> Self {
        Self {
            behavior_start_tick,
            behavior_next_tick,
            behavior_batches,
            lifecycle_checkpoint,
        }
    }

    /// Revalidate a persisted packet against the caller's previously qualified behavior boundary.
    pub fn validate_after(
        self,
        expected_behavior_start_tick: u64,
    ) -> Result<ValidatedGenesisEvidenceCheckpointV1, GenesisEvidenceCheckpointErrorV1> {
        if self.behavior_start_tick != expected_behavior_start_tick {
            return Err(GenesisEvidenceCheckpointErrorV1::BehaviorStartMismatch {
                expected: expected_behavior_start_tick,
                observed: self.behavior_start_tick,
            });
        }

        let behavior_summary = validate_genesis_tick_chunk(
            self.behavior_start_tick,
            &self.behavior_batches,
        )
        .map_err(GenesisEvidenceCheckpointErrorV1::Behavior)?;
        if behavior_summary.next_tick != self.behavior_next_tick {
            return Err(GenesisEvidenceCheckpointErrorV1::BehaviorEndMismatch {
                expected: behavior_summary.next_tick,
                observed: self.behavior_next_tick,
            });
        }

        let lifecycle = self
            .lifecycle_checkpoint
            .validate()
            .map_err(GenesisEvidenceCheckpointErrorV1::Lifecycle)?;
        if lifecycle.continuation_epoch() != self.behavior_next_tick {
            return Err(GenesisEvidenceCheckpointErrorV1::CrossClockMismatch {
                behavior_next_tick: self.behavior_next_tick,
                lifecycle_continuation_epoch: lifecycle.continuation_epoch(),
            });
        }

        for batch in &self.behavior_batches {
            let validated = batch
                .clone()
                .validate()
                .map_err(GenesisEvidenceCheckpointErrorV1::Behavior)?;
            validate_behavior_against_lifecycle(&validated, &lifecycle)?;
        }

        Ok(ValidatedGenesisEvidenceCheckpointV1 {
            behavior_start_tick: self.behavior_start_tick,
            behavior_next_tick: self.behavior_next_tick,
            behavior_batches: self.behavior_batches,
            behavior_summary,
            lifecycle,
        })
    }
}

impl ValidatedGenesisEvidenceCheckpointV1 {
    pub fn behavior_start_tick(&self) -> u64 {
        self.behavior_start_tick
    }

    pub fn behavior_next_tick(&self) -> u64 {
        self.behavior_next_tick
    }

    pub fn behavior_batches(&self) -> &[GenesisTickBatchV1] {
        &self.behavior_batches
    }

    pub fn behavior_summary(&self) -> &GenesisTickChunkSummaryV1 {
        &self.behavior_summary
    }

    pub fn lifecycle(&self) -> &ValidatedLifecycleCheckpointV1 {
        &self.lifecycle
    }
}

fn validate_behavior_against_lifecycle(
    batch: &ValidatedGenesisTickBatchV1,
    lifecycle: &ValidatedLifecycleCheckpointV1,
) -> Result<(), GenesisEvidenceCheckpointErrorV1> {
    let tick = batch.tick();
    let ledger = lifecycle.ledger();

    let expected = ledger
        .records()
        .values()
        .filter(|record| alive_at_social_step_start(*record, tick))
        .map(|record| record.agent_id)
        .collect::<BTreeSet<_>>();
    let observed = batch
        .events()
        .iter()
        .map(|event| event.agent_id)
        .collect::<BTreeSet<_>>();

    if expected != observed {
        return Err(GenesisEvidenceCheckpointErrorV1::BehaviorLivingSetMismatch {
            tick,
            expected,
            observed,
        });
    }

    for event in batch.events() {
        let record = ledger.records().get(&event.agent_id).ok_or(
            GenesisEvidenceCheckpointErrorV1::UnknownBehaviorAgent {
                tick,
                agent_id: event.agent_id,
            },
        )?;
        if record.lineage_id != event.lineage_id || record.generation != event.generation {
            return Err(GenesisEvidenceCheckpointErrorV1::BehaviorIdentityMismatch {
                tick,
                agent_id: event.agent_id,
                expected_lineage_id: record.lineage_id,
                observed_lineage_id: event.lineage_id,
                expected_generation: record.generation,
                observed_generation: event.generation,
            });
        }
    }

    Ok(())
}

fn alive_at_social_step_start(record: &crate::AgentLifecycleRecordV1, tick: u64) -> bool {
    // Founders exist before social tick zero. All non-founder births happen during population
    // bookkeeping after that tick's behavioral rows, so a descendant born at tick T first appears
    // in behavioral evidence at a later social tick.
    let born_before_step = if record.generation == 0 {
        record.born_tick <= tick
    } else {
        record.born_tick < tick
    };
    if !born_before_step {
        return false;
    }

    match record.died_tick {
        None => true,
        Some(died_tick) if died_tick > tick => true,
        Some(died_tick) if died_tick < tick => false,
        Some(_) => {
            // A pre-step cull at epoch T removes the organism before behavioral tick T. A
            // population energy-threshold death at epoch T happens after the behavioral rows and
            // therefore remains part of that tick's pre-bookkeeping living set.
            record.death_cause == Some(LifecycleDeathCauseV1::PopulationEnergyThreshold)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentIdAllocator, EncounterScheduler, GenesisSocialRunnerV1, GenesisTickBatchV1,
        OrganismConfig, PairingMode, PopulationConfig,
    };

    fn quiet_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 2.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig::default(),
            ..Default::default()
        }
    }

    fn checkpoint_after(
        runner: &mut GenesisSocialRunnerV1,
        behavior_start_tick: u64,
    ) -> GenesisEvidenceCheckpointV1 {
        let behavior_batches = runner.drain_completed_batches();
        let lifecycle_events = runner.drain_lifecycle_events();
        let lifecycle = LifecycleCheckpointV1::from_complete_prefix(
            &lifecycle_events,
            runner.lifecycle_epoch(),
        )
        .expect("valid complete lifecycle")
        .into_checkpoint();
        GenesisEvidenceCheckpointV1::new(
            behavior_start_tick,
            runner.tick_cursor_snapshot().next_tick(),
            behavior_batches,
            lifecycle,
        )
    }

    #[test]
    fn social_behavior_and_lifecycle_validate_as_one_cross_evidence_checkpoint() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 4, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);
        runner.step_social(|_| 0.5, &mut scheduler).expect("tick 0");
        runner.step_social(|_| 0.5, &mut scheduler).expect("tick 1");

        let validated = checkpoint_after(&mut runner, 0)
            .validate_after(0)
            .expect("cross-evidence checkpoint");
        assert_eq!(validated.behavior_start_tick(), 0);
        assert_eq!(validated.behavior_next_tick(), 2);
        assert_eq!(validated.lifecycle().continuation_epoch(), 2);
        assert_eq!(validated.behavior_summary().batch_count, 2);
    }

    #[test]
    fn pre_step_cull_is_excluded_from_that_ticks_behavior_living_set() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 3, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);
        runner.cull_weakest(1).expect("authoritative cull");
        runner.step_social(|_| 0.5, &mut scheduler).expect("tick 0");

        let validated = checkpoint_after(&mut runner, 0)
            .validate_after(0)
            .expect("cull-aware cross evidence");
        assert_eq!(
            validated.behavior_batches()[0]
                .clone()
                .validate()
                .unwrap()
                .population_before(),
            2
        );
    }

    #[test]
    fn forged_behavior_set_cannot_hide_behind_a_matching_event_count() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 3, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);
        runner.step_social(|_| 0.5, &mut scheduler).expect("tick 0");

        let original = runner.completed_batches()[0].clone().validate().unwrap();
        let mut events = original.events().to_vec();
        let mut ids = AgentIdAllocator::new();
        for _ in 0..3 {
            let _ = ids.allocate();
        }
        let unknown = ids.allocate();
        let unpaired = events
            .iter_mut()
            .find(|event| event.partner_id.is_none())
            .expect("odd population must leave one event unpaired");
        unpaired.agent_id = unknown;
        unpaired.lineage_id = unknown;
        let forged = GenesisTickBatchV1::new(0, 3, events);

        let lifecycle_events = runner.drain_lifecycle_events();
        let lifecycle = LifecycleCheckpointV1::from_complete_prefix(
            &lifecycle_events,
            runner.lifecycle_epoch(),
        )
        .unwrap()
        .into_checkpoint();
        let raw = GenesisEvidenceCheckpointV1::new(0, 1, vec![forged], lifecycle);

        assert!(matches!(
            raw.validate_after(0),
            Err(GenesisEvidenceCheckpointErrorV1::BehaviorLivingSetMismatch { .. })
        ));
    }

    #[test]
    fn lifecycle_and_behavior_clocks_must_agree_in_social_runner_profile() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 0, 17);
        let lifecycle_events = runner.drain_lifecycle_events();
        let lifecycle = LifecycleCheckpointV1::from_complete_prefix(&lifecycle_events, 0)
            .unwrap()
            .into_checkpoint();
        let raw = GenesisEvidenceCheckpointV1::new(
            0,
            1,
            vec![GenesisTickBatchV1::new(0, 0, Vec::new())],
            lifecycle,
        );
        assert_eq!(
            raw.validate_after(0),
            Err(GenesisEvidenceCheckpointErrorV1::CrossClockMismatch {
                behavior_next_tick: 1,
                lifecycle_continuation_epoch: 0,
            })
        );
    }
}
