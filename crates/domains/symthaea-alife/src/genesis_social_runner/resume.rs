// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound resume constructor for the qualified Genesis social runner.
//!
//! Resume authority is deliberately composed from two non-serializable capabilities: a validated
//! population snapshot and the validated Genesis evidence checkpoint that authorized it. Raw
//! `Population`, raw population snapshots, and raw tick counters cannot enter this constructor.

use super::GenesisSocialRunnerV1;
use crate::{
    GenesisTickCursorV1, Population, ValidatedGenesisEvidenceCheckpointV1,
    ValidatedPopulationSnapshotV1,
};
use crate::population::PopulationLiveSnapshotErrorV1;

impl GenesisSocialRunnerV1 {
    /// Resume qualified social execution from one already-validated population/evidence boundary.
    ///
    /// The restored runner begins with no pending behavioral batches. Population restoration
    /// independently recreates lifecycle, identity, seed, evolutionary RNG, organism, and behavior
    /// clock state; the wrapper's own tick cursor is then derived from the same validated evidence
    /// capability rather than from a serialized integer.
    pub fn from_validated_resume(
        population: &ValidatedPopulationSnapshotV1,
        evidence: &ValidatedGenesisEvidenceCheckpointV1,
    ) -> Result<Self, PopulationLiveSnapshotErrorV1> {
        let population = Population::from_validated_snapshot_v1(population, evidence)?;
        Ok(Self {
            population,
            tick_cursor: GenesisTickCursorV1::from_validated_evidence(evidence),
            completed_batches: Vec::new(),
            qualified: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EncounterScheduler, GenesisRollingEvidenceRunnerV1, OrganismConfig, PairingMode,
        PopulationConfig, PopulationSnapshotV1,
    };

    fn quiet_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 2.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                social_enabled: true,
                action_temperature: 0.71,
                ..OrganismConfig::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn validated_resume_recreates_population_and_tick_boundary_without_pending_batches() {
        let mut rolling = GenesisRollingEvidenceRunnerV1::new(quiet_cfg(), 3, 0x51eed);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 0x5ced);
        for _ in 0..5 {
            rolling
                .step_social(|n| 0.4 + 0.2 / n.max(1) as f64, &mut scheduler)
                .expect("qualified prefix");
        }
        let checkpoint = rolling.checkpoint_evidence().expect("checkpoint");
        let raw = rolling
            .population()
            .snapshot_v1_at_evidence_boundary(checkpoint.validated())
            .expect("population snapshot");
        let encoded = serde_json::to_string(&raw).expect("serialize population");
        let loaded: PopulationSnapshotV1 = serde_json::from_str(&encoded).expect("deserialize");
        let validated = loaded
            .validate_for_genesis_social(checkpoint.validated())
            .expect("revalidate population");

        let resumed = GenesisSocialRunnerV1::from_validated_resume(
            &validated,
            checkpoint.validated(),
        )
        .expect("resume social runner");

        assert!(resumed.is_qualified());
        assert_eq!(
            resumed.tick_cursor_snapshot().next_tick(),
            checkpoint.validated().behavior_next_tick()
        );
        assert!(resumed.completed_batches().is_empty());
        assert!(resumed.lifecycle_events().is_empty());
        assert_eq!(
            resumed.lifecycle_epoch(),
            checkpoint.validated().lifecycle().continuation_epoch()
        );

        let resumed_raw = resumed
            .population()
            .snapshot_v1_at_evidence_boundary(checkpoint.validated())
            .expect("resumed population remains at exact boundary");
        assert_eq!(
            serde_json::to_string(&resumed_raw).unwrap(),
            encoded,
            "qualified resume must not normalize population-owned causal state"
        );
    }
}
