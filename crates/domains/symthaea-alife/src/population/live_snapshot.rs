// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live [`Population`](super::Population) bridge for the validated v1 population snapshot contract.
//!
//! Population persistence is allowed only at a quiescent, already-validated Genesis evidence
//! boundary. Pending flat behavioral rows or lifecycle transitions are evidence that the caller has
//! not completed the rolling checkpoint protocol and therefore must not be silently omitted from a
//! population snapshot.
//!
//! Restore never trusts serialized allocator/clock integers directly. The validated population
//! capability supplies semantically checked state, while the validated Genesis evidence checkpoint
//! independently authorizes identity allocation, construction-seed allocation, lifecycle sequence,
//! lifecycle epoch, and the social behavior clock.

use super::Population;
use crate::{
    AgentIdAllocator, AgentIdAllocatorRestoreErrorV1, EvolutionRngRestoreErrorV1,
    EvolutionRngStreamsV1, LifecycleRecorderV1, Organism, OrganismSeedAllocatorErrorV1,
    OrganismSeedAllocatorV1, PopulationConfigSnapshotV1, PopulationSnapshotErrorV1,
    PopulationSnapshotV1, ValidatedGenesisEvidenceCheckpointV1, ValidatedPopulationSnapshotV1,
};

#[derive(Debug)]
pub enum PopulationLiveSnapshotErrorV1 {
    PendingBehaviorEvidence { rows: usize },
    PendingLifecycleEvidence { events: usize },
    BehaviorClockMismatch { expected: u64, observed: u64 },
    LifecycleEpochMismatch { expected: u64, observed: u64 },
    LifecycleSequenceMismatch { expected: u64, observed: u64 },
    Snapshot(PopulationSnapshotErrorV1),
    OrganismSeedAllocator(OrganismSeedAllocatorErrorV1),
    EvolutionRng(EvolutionRngRestoreErrorV1),
    IdAllocator(AgentIdAllocatorRestoreErrorV1),
}

impl Population {
    /// Capture population-owned causal state only at one validated, fully-drained evidence boundary.
    ///
    /// The returned raw value remains serialization data, not execution authority. After loading it
    /// from storage callers must run [`PopulationSnapshotV1::validate_for_genesis_social`] again.
    pub fn snapshot_v1_at_evidence_boundary(
        &self,
        evidence: &ValidatedGenesisEvidenceCheckpointV1,
    ) -> Result<PopulationSnapshotV1, PopulationLiveSnapshotErrorV1> {
        if !self.event_log.is_empty() {
            return Err(PopulationLiveSnapshotErrorV1::PendingBehaviorEvidence {
                rows: self.event_log.len(),
            });
        }
        if !self.lifecycle_recorder.events().is_empty() {
            return Err(PopulationLiveSnapshotErrorV1::PendingLifecycleEvidence {
                events: self.lifecycle_recorder.events().len(),
            });
        }

        let expected_tick = evidence.behavior_next_tick();
        if self.current_tick != expected_tick {
            return Err(PopulationLiveSnapshotErrorV1::BehaviorClockMismatch {
                expected: expected_tick,
                observed: self.current_tick,
            });
        }

        let lifecycle = evidence.lifecycle();
        if self.lifecycle_recorder.current_epoch() != lifecycle.continuation_epoch() {
            return Err(PopulationLiveSnapshotErrorV1::LifecycleEpochMismatch {
                expected: lifecycle.continuation_epoch(),
                observed: self.lifecycle_recorder.current_epoch(),
            });
        }
        if self.lifecycle_recorder.next_sequence() != lifecycle.next_sequence() {
            return Err(PopulationLiveSnapshotErrorV1::LifecycleSequenceMismatch {
                expected: lifecycle.next_sequence(),
                observed: self.lifecycle_recorder.next_sequence(),
            });
        }

        let snapshot = PopulationSnapshotV1::new(
            PopulationConfigSnapshotV1::from_config(self.cfg),
            self.organisms.iter().map(Organism::snapshot_v1).collect(),
            self.organism_seed_allocator.snapshot(),
            self.evolution_rng.snapshot(),
            self.id_allocator.snapshot(),
            self.total_births,
            self.total_deaths,
            self.current_tick,
        );

        // A live owner should never emit persistence that its own canonical validator rejects.
        // Validate once before returning while preserving the raw object for storage.
        snapshot
            .validate_for_genesis_social(evidence)
            .map_err(PopulationLiveSnapshotErrorV1::Snapshot)?;
        Ok(snapshot)
    }

    /// Restore population-owned execution state only from a non-serializable validated population
    /// capability and the same validated cross-evidence boundary that authorized it.
    ///
    /// Scheduler and environment state are deliberately not accepted here; they remain external
    /// causal surfaces for `PopulationExecutionCapsuleV1`.
    pub fn from_validated_snapshot_v1(
        snapshot: &ValidatedPopulationSnapshotV1,
        evidence: &ValidatedGenesisEvidenceCheckpointV1,
    ) -> Result<Self, PopulationLiveSnapshotErrorV1> {
        if snapshot.current_tick() != evidence.behavior_next_tick() {
            return Err(PopulationLiveSnapshotErrorV1::BehaviorClockMismatch {
                expected: evidence.behavior_next_tick(),
                observed: snapshot.current_tick(),
            });
        }

        let lifecycle = evidence.lifecycle();
        let organism_seed_allocator = OrganismSeedAllocatorV1::from_snapshot_for_lifecycle(
            snapshot.organism_seed_allocator_snapshot(),
            lifecycle.ledger(),
        )
        .map_err(PopulationLiveSnapshotErrorV1::OrganismSeedAllocator)?;
        let evolution_rng = EvolutionRngStreamsV1::from_snapshot(snapshot.evolution_rng_snapshot())
            .map_err(PopulationLiveSnapshotErrorV1::EvolutionRng)?;
        let id_allocator = AgentIdAllocator::from_snapshot_for_lifecycle(
            snapshot.id_allocator_snapshot(),
            lifecycle.ledger(),
        )
        .map_err(PopulationLiveSnapshotErrorV1::IdAllocator)?;
        let lifecycle_recorder = LifecycleRecorderV1::from_validated_checkpoint(lifecycle);

        let restored = Self {
            organisms: snapshot
                .organisms()
                .iter()
                .map(Organism::from_validated_snapshot_v1)
                .collect(),
            cfg: snapshot.config().config(),
            organism_seed_allocator,
            evolution_rng,
            total_births: snapshot.total_births(),
            total_deaths: snapshot.total_deaths(),
            id_allocator,
            lifecycle_recorder,
            event_log: Vec::new(),
            current_tick: snapshot.current_tick(),
        };

        // Re-run the live-owner boundary after reconstruction. This proves restoration did not
        // accidentally normalize causal Vec order, regenerate nested state, or reopen evidence
        // buffers while transferring the validated capability back into production state.
        restored.snapshot_v1_at_evidence_boundary(evidence)?;
        Ok(restored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EncounterScheduler, GenesisEvidenceCheckpointV1, GenesisRollingEvidenceRunnerV1,
        OrganismConfig, PairingMode, PopulationConfig,
    };

    fn quiet_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 2.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                social_enabled: true,
                action_temperature: 0.73,
                ..OrganismConfig::default()
            },
            mutation_rate: 0.0,
            mutation_std: 0.0,
            ..Default::default()
        }
    }

    fn checkpointed_fixture() -> (
        GenesisRollingEvidenceRunnerV1,
        EncounterScheduler,
        crate::GenesisRollingEvidenceCheckpointV1,
    ) {
        let mut runner = GenesisRollingEvidenceRunnerV1::new(quiet_cfg(), 4, 0x51eed);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 0x5ced);
        for _ in 0..7 {
            runner
                .step_social(|n| 0.35 + 0.4 / n.max(1) as f64, &mut scheduler)
                .expect("qualified social prefix");
        }
        let checkpoint = runner.checkpoint_evidence().expect("quiescent evidence boundary");
        (runner, scheduler, checkpoint)
    }

    #[test]
    fn serialized_validated_population_restore_is_bit_exact_at_boundary() {
        let (runner, _scheduler, checkpoint) = checkpointed_fixture();
        let original = runner
            .population()
            .snapshot_v1_at_evidence_boundary(checkpoint.validated())
            .expect("snapshot live population");
        let encoded = serde_json::to_string(&original).expect("serialize population snapshot");
        let raw: PopulationSnapshotV1 = serde_json::from_str(&encoded).expect("deserialize");
        let validated = raw
            .validate_for_genesis_social(checkpoint.validated())
            .expect("revalidate population snapshot");
        let restored = Population::from_validated_snapshot_v1(&validated, checkpoint.validated())
            .expect("restore population");
        let restored_raw = restored
            .snapshot_v1_at_evidence_boundary(checkpoint.validated())
            .expect("snapshot restored population");
        let restored_json =
            serde_json::to_string(&restored_raw).expect("serialize restored population snapshot");
        assert_eq!(encoded, restored_json);
    }

    #[test]
    fn snapshot_refuses_uncheckpointed_social_clock_advance() {
        let (mut runner, mut scheduler, checkpoint) = checkpointed_fixture();
        runner
            .step_social(|_| 0.5, &mut scheduler)
            .expect("one pending social step");
        assert!(matches!(
            runner
                .population()
                .snapshot_v1_at_evidence_boundary(checkpoint.validated()),
            Err(PopulationLiveSnapshotErrorV1::BehaviorClockMismatch { .. })
                | Err(PopulationLiveSnapshotErrorV1::LifecycleEpochMismatch { .. })
        ));
    }

    #[test]
    fn raw_population_with_pending_flat_rows_cannot_snapshot() {
        let (runner, mut scheduler, checkpoint) = checkpointed_fixture();
        let mut population = runner.into_population();
        population.step_social(|_| 0.5, &mut scheduler);
        assert!(matches!(
            population.snapshot_v1_at_evidence_boundary(checkpoint.validated()),
            Err(PopulationLiveSnapshotErrorV1::PendingBehaviorEvidence { .. })
        ));
    }

    #[test]
    fn pending_cull_lifecycle_transition_cannot_be_omitted() {
        let (runner, _scheduler, checkpoint) = checkpointed_fixture();
        let mut population = runner.into_population();
        population.cull_weakest(1);
        assert!(matches!(
            population.snapshot_v1_at_evidence_boundary(checkpoint.validated()),
            Err(PopulationLiveSnapshotErrorV1::PendingLifecycleEvidence { .. })
        ));
    }

    #[test]
    fn persisted_population_still_requires_validation_after_deserialization() {
        let (runner, _scheduler, checkpoint) = checkpointed_fixture();
        let raw = runner
            .population()
            .snapshot_v1_at_evidence_boundary(checkpoint.validated())
            .expect("snapshot");
        let json = serde_json::to_string(&raw).expect("serialize");
        let loaded: PopulationSnapshotV1 = serde_json::from_str(&json).expect("deserialize");
        // The only route back into live Population consumes a ValidatedPopulationSnapshotV1,
        // therefore raw loaded bytes cannot call the restore API directly.
        let validated = loaded
            .validate_for_genesis_social(checkpoint.validated())
            .expect("explicit revalidation");
        Population::from_validated_snapshot_v1(&validated, checkpoint.validated())
            .expect("validated restore");
    }

    #[test]
    fn evidence_round_trip_does_not_weaken_population_restore_boundary() {
        let (runner, _scheduler, checkpoint) = checkpointed_fixture();
        let persisted_evidence =
            serde_json::to_string(checkpoint.persisted()).expect("serialize evidence");
        let raw_evidence: GenesisEvidenceCheckpointV1 =
            serde_json::from_str(&persisted_evidence).expect("deserialize evidence");
        let evidence = raw_evidence.validate_after(0).expect("revalidate evidence");
        let raw_population = runner
            .population()
            .snapshot_v1_at_evidence_boundary(&evidence)
            .expect("snapshot against revalidated evidence");
        let population_json = serde_json::to_string(&raw_population).expect("serialize population");
        let loaded: PopulationSnapshotV1 =
            serde_json::from_str(&population_json).expect("deserialize population");
        let validated = loaded
            .validate_for_genesis_social(&evidence)
            .expect("revalidate population");
        Population::from_validated_snapshot_v1(&validated, &evidence)
            .expect("restore from two independently revalidated persistence objects");
    }
}
