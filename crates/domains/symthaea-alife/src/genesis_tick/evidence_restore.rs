// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound restore bridge for [`super::GenesisTickCursorV1`].
//!
//! A raw `GenesisTickCursorSnapshotV1` is deliberately insufficient continuation authority. The
//! next behavioral tick is independently established by a canonically validated Genesis evidence
//! checkpoint, so resume recreates the cursor from that capability instead of trusting a persisted
//! counter supplied alongside it.

use super::GenesisTickCursorV1;
use crate::ValidatedGenesisEvidenceCheckpointV1;

impl GenesisTickCursorV1 {
    /// Recreate live behavioral tick authority only from validated cross-evidence state.
    pub fn from_validated_evidence(evidence: &ValidatedGenesisEvidenceCheckpointV1) -> Self {
        Self {
            next_tick: evidence.behavior_next_tick(),
        }
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
            organism_cfg: OrganismConfig::default(),
            ..Default::default()
        }
    }

    #[test]
    fn validated_evidence_recreates_exact_next_behavior_tick() {
        let mut runner = GenesisRollingEvidenceRunnerV1::new(quiet_cfg(), 1, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 19);
        for _ in 0..7 {
            runner
                .step_social(|_| 0.5, &mut scheduler)
                .expect("qualified social tick");
        }
        let checkpoint = runner.checkpoint_evidence().expect("checkpoint");
        assert_eq!(checkpoint.validated().lifecycle().ledger().records().len(), 1);
        let cursor = GenesisTickCursorV1::from_validated_evidence(checkpoint.validated());
        assert_eq!(cursor.snapshot().next_tick(), 7);
        assert_eq!(cursor.preflight().expect("next tick available").tick(), 7);
    }

    #[test]
    fn persistence_must_revalidate_before_tick_authority_is_recreated() {
        let mut runner = GenesisRollingEvidenceRunnerV1::new(quiet_cfg(), 1, 23);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 29);
        for _ in 0..3 {
            runner
                .step_social(|_| 0.5, &mut scheduler)
                .expect("qualified social tick");
        }
        let checkpoint = runner.checkpoint_evidence().expect("checkpoint");
        let encoded = serde_json::to_string(checkpoint.persisted()).expect("serialize evidence");
        let raw: GenesisEvidenceCheckpointV1 = serde_json::from_str(&encoded).expect("deserialize");
        let validated = raw.validate_after(0).expect("explicit evidence revalidation");
        let cursor = GenesisTickCursorV1::from_validated_evidence(&validated);
        assert_eq!(cursor.snapshot().next_tick(), 3);
    }
}
