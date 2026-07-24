// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic execution of certification scenario manifests.

use crate::embodiment::SubterraneanEmbodiment;
use crate::scenario_manifest::{ScenarioFingerprint, ScenarioManifest, ScenarioManifestError};
use crate::sensor_redundancy::{RedundantSensorFrame, SensorSourceId, SensorSourceObservation};
use crate::types::BATTERY_RATIO;
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenarioFailure {
    InvalidFinalState,
    InvariantBreach,
    ProductiveWorkAtRed,
    FinalBatteryBelowMinimum,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioRunReport {
    pub scenario_id: String,
    pub fingerprint: ScenarioFingerprint,
    pub steps_executed: u32,
    pub final_state_valid: bool,
    pub final_battery_ratio: f64,
    pub maximum_hazard_severity: f32,
    pub invariant_breach_records: u64,
    pub productive_work_at_red_records: u64,
    pub failures: Vec<ScenarioFailure>,
}

impl ScenarioRunReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ScenarioRunner;

impl ScenarioRunner {
    pub fn run(
        &self,
        manifest: &ScenarioManifest,
    ) -> Result<ScenarioRunReport, ScenarioManifestError> {
        manifest.validate()?;
        let initial_state = manifest.initial_state()?;
        let fingerprint = manifest.fingerprint()?;
        let genesis = GenesisSeed::from_phrase(&manifest.seed_phrase);
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        embodiment.ingest_redundant_sensor_frame(RedundantSensorFrame {
            observations: vec![SensorSourceObservation::from_state(
                SensorSourceId(0),
                1,
                &initial_state,
            )],
        });
        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&fingerprint.0[..8]);
        let thought = ContinuousHV::random(HDC_DIMENSION, u64::from_le_bytes(seed_bytes));
        for _ in 0..manifest.steps {
            embodiment.step(&thought, manifest.dt_seconds, manifest.phi);
        }
        let records = embodiment.evidence_records();
        let summary = embodiment.evidence_summary();
        let final_record = records.last();
        let final_state_valid = final_record
            .map(|record| record.state_channels.iter().all(|value| value.is_finite()))
            .unwrap_or(false);
        let final_battery_ratio = final_record
            .map(|record| record.state_channels[BATTERY_RATIO])
            .unwrap_or(0.0);
        let productive_work_at_red_records = records
            .iter()
            .filter(|record| {
                record.safety_level == "red"
                    && (record.command.cutter_head().abs() > 1e-5
                        || record.command.auger_feed().abs() > 1e-5)
            })
            .count() as u64;
        let mut failures = Vec::new();
        if manifest.acceptance.require_valid_final_state && !final_state_valid {
            failures.push(ScenarioFailure::InvalidFinalState);
        }
        if manifest.acceptance.require_no_invariant_breach && summary.invariant_breach_records > 0 {
            failures.push(ScenarioFailure::InvariantBreach);
        }
        if manifest.acceptance.require_no_productive_work_at_red
            && productive_work_at_red_records > 0
        {
            failures.push(ScenarioFailure::ProductiveWorkAtRed);
        }
        if let Some(minimum) = manifest.acceptance.minimum_final_battery {
            if final_battery_ratio < minimum {
                failures.push(ScenarioFailure::FinalBatteryBelowMinimum);
            }
        }
        Ok(ScenarioRunReport {
            scenario_id: manifest.scenario_id.clone(),
            fingerprint,
            steps_executed: manifest.steps,
            final_state_valid,
            final_battery_ratio,
            maximum_hazard_severity: summary.max_hazard_severity,
            invariant_breach_records: summary.invariant_breach_records,
            productive_work_at_red_records,
            failures,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requirements::RequirementId;
    use crate::scenario_manifest::StateOverride;
    use crate::types::CUTTER_TEMP_C;

    #[test]
    fn manifest_execution_is_reproducible() {
        let mut manifest = ScenarioManifest::new(
            "thermal-arrest",
            "thermal certification scenario",
            20,
            vec![RequirementId::HazardPreemption],
        );
        manifest.state_overrides.push(StateOverride {
            channel: CUTTER_TEMP_C,
            value: 150.0,
        });
        let left = ScenarioRunner.run(&manifest).expect("valid scenario");
        let right = ScenarioRunner.run(&manifest).expect("valid scenario");
        assert_eq!(left, right);
        assert!(left.passed(), "{left:?}");
    }
}
