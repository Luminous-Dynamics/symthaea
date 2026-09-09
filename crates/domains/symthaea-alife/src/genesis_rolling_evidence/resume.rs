// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound resume constructor for the atomic Genesis rolling evidence profile.
//!
//! This bridge resumes only from a validated population capability plus the validated cross-evidence
//! checkpoint that authorized it. The retained lifecycle prefix and behavioral start cursor are
//! reconstructed from that checkpoint; no raw population, tick integer, or lifecycle cursor enters
//! the qualified wrapper.

use super::GenesisRollingEvidenceRunnerV1;
use crate::{
    GenesisSocialRunnerV1, ValidatedGenesisEvidenceCheckpointV1, ValidatedPopulationSnapshotV1,
};
use crate::population::PopulationLiveSnapshotErrorV1;

impl GenesisRollingEvidenceRunnerV1 {
    /// Resume the rolling evidence profile at one already-validated quiescent boundary.
    pub fn from_validated_resume(
        population: &ValidatedPopulationSnapshotV1,
        evidence: &ValidatedGenesisEvidenceCheckpointV1,
    ) -> Result<Self, PopulationLiveSnapshotErrorV1> {
        let runner = GenesisSocialRunnerV1::from_validated_resume(population, evidence)?;
        Ok(Self {
            runner,
            lifecycle: Some(evidence.lifecycle().clone()),
            behavior_next_tick: evidence.behavior_next_tick(),
            qualified: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EncounterScheduler, GenesisEvidenceCheckpointV1, OrganismConfig, PairingMode,
        PopulationConfig, PopulationSnapshotV1,
    };

    fn quiet_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 2.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                social_enabled: true,
                action_temperature: 0.69,
                ..OrganismConfig::default()
            },
            ..Default::default()
        }
    }

    fn persisted_resume_fixture() -> (
        PopulationSnapshotV1,
        GenesisEvidenceCheckpointV1,
        u64,
        usize,
    ) {
        let mut runner = GenesisRollingEvidenceRunnerV1::new(quiet_cfg(), 4, 0xA11F_E001);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 0x5CED_0001);
        for _ in 0..6 {
            runner
                .step_social(|n| 0.42 + 0.16 / n.max(1) as f64, &mut scheduler)
                .expect("qualified prefix");
        }
        let checkpoint = runner.checkpoint_evidence().expect("checkpoint prefix");
        let population = runner
            .population()
            .snapshot_v1_at_evidence_boundary(checkpoint.validated())
            .expect("population persistence");
        let next_tick = checkpoint.validated().behavior_next_tick();
        let live_count = runner.population().len();
        (population, checkpoint.into_persisted(), next_tick, live_count)
    }

    fn revalidate(
        population: PopulationSnapshotV1,
        evidence: GenesisEvidenceCheckpointV1,
        previous_tick: u64,
    ) -> (
        crate::ValidatedPopulationSnapshotV1,
        crate::ValidatedGenesisEvidenceCheckpointV1,
    ) {
        let evidence_json = serde_json::to_string(&evidence).expect("serialize evidence");
        let loaded_evidence: GenesisEvidenceCheckpointV1 =
            serde_json::from_str(&evidence_json).expect("deserialize evidence");
        let validated_evidence = loaded_evidence
            .validate_after(previous_tick)
            .expect("revalidate evidence");

        let population_json = serde_json::to_string(&population).expect("serialize population");
        let loaded_population: PopulationSnapshotV1 =
            serde_json::from_str(&population_json).expect("deserialize population");
        let validated_population = loaded_population
            .validate_for_genesis_social(&validated_evidence)
            .expect("revalidate population");
        (validated_population, validated_evidence)
    }

    #[test]
    fn resumed_rolling_runner_retains_exact_checkpoint_boundary() {
        let (population, evidence, next_tick, live_count) = persisted_resume_fixture();
        let (population, evidence) = revalidate(population, evidence, 0);
        let resumed = GenesisRollingEvidenceRunnerV1::from_validated_resume(&population, &evidence)
            .expect("resume rolling profile");

        assert!(resumed.is_qualified());
        assert_eq!(resumed.behavior_next_tick(), next_tick);
        assert_eq!(resumed.population().len(), live_count);
        let lifecycle = resumed.lifecycle_checkpoint().expect("retained lifecycle prefix");
        assert_eq!(lifecycle.continuation_epoch(), next_tick);
        assert_eq!(lifecycle.next_sequence(), evidence.lifecycle().next_sequence());
    }

    #[test]
    fn checkpoint_restore_checkpoint_forms_one_contiguous_evidence_lineage() {
        let (population, evidence, next_tick, _live_count) = persisted_resume_fixture();
        let (population, evidence) = revalidate(population, evidence, 0);
        let prior_lifecycle_sequence = evidence.lifecycle().next_sequence();
        let prior_births = evidence.lifecycle().birth_count();
        let prior_deaths = evidence.lifecycle().death_count();
        let mut resumed =
            GenesisRollingEvidenceRunnerV1::from_validated_resume(&population, &evidence)
                .expect("resume rolling profile");

        // Scheduler state is deliberately external in this tranche. We prove evidence continuity,
        // not trajectory equivalence: the new scheduler is an explicit fresh causal input.
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 0x5CED_C002);
        resumed
            .step_social(|n| 0.46 + 0.12 / n.max(1) as f64, &mut scheduler)
            .expect("resumed tick one");
        resumed
            .step_social(|n| 0.46 + 0.12 / n.max(1) as f64, &mut scheduler)
            .expect("resumed tick two");

        let next = resumed.checkpoint_evidence().expect("checkpoint resumed suffix");
        assert_eq!(next.validated().behavior_start_tick(), next_tick);
        assert_eq!(next.validated().behavior_next_tick(), next_tick + 2);
        assert_eq!(
            next.validated().lifecycle().continuation_epoch(),
            next_tick + 2
        );
        assert!(
            next.validated().lifecycle().next_sequence() >= prior_lifecycle_sequence,
            "continuation must never rewind lifecycle sequence authority"
        );
        assert!(next.validated().lifecycle().birth_count() >= prior_births);
        assert!(next.validated().lifecycle().death_count() >= prior_deaths);
    }

    #[test]
    fn cull_after_resume_extends_retained_lifecycle_without_advancing_behavior_clock() {
        let (population, evidence, next_tick, live_count) = persisted_resume_fixture();
        let (population, evidence) = revalidate(population, evidence, 0);
        let prior_deaths = evidence.lifecycle().death_count();
        let mut resumed =
            GenesisRollingEvidenceRunnerV1::from_validated_resume(&population, &evidence)
                .expect("resume rolling profile");

        let removed = resumed.cull_weakest(1).expect("resumed cull");
        let expected_removed = if live_count > 0 { 1 } else { 0 };
        assert_eq!(removed, expected_removed);
        let checkpoint = resumed.checkpoint_evidence().expect("checkpoint resumed cull");
        assert_eq!(checkpoint.validated().behavior_start_tick(), next_tick);
        assert_eq!(checkpoint.validated().behavior_next_tick(), next_tick);
        assert_eq!(
            checkpoint.validated().lifecycle().death_count(),
            prior_deaths + removed
        );
        assert_eq!(
            checkpoint.validated().lifecycle().continuation_epoch(),
            next_tick
        );
    }
}
