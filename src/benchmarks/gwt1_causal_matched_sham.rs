// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Matched omission sham for the direct GWT-1 causal-lesion experiment.
//!
//! The causal intervention omits one real production specialist from the
//! Phase-B panel. This sham applies the **same omission machinery** under the
//! frozen direct-theorem baseline snapshot, where the target's authentic
//! output is required to be neutral. If omission itself creates an integrated
//! effect, the control fails.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::cognitive_loop::managers::{
    DriveManager, LearningManager, MemoryManager, PerceptionManager,
};
use crate::cognitive_loop::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, IntegratedOutput, OutputCollector, SubsystemOutput,
};

use super::gwt1_causal_lesion::{
    Gwt1CausalTargetV1, Gwt1IntegratedOutputBitsV1,
};
use super::gwt1_specialist_qualification::{
    GWT1_SPECIALIST_IDS_V1, Gwt1SpecialistOutputMapV1,
};

pub const GWT1_CAUSAL_MATCHED_SHAM_SCHEMA_V1: &str =
    "butlin-gwt1-causal-matched-omission-sham-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1MatchedShamArmRawV1 {
    pub invoked_specialists: BTreeSet<String>,
    /// Includes neutral outputs, so execution identity is not inferred from
    /// collector contribution.
    pub executed_outputs: Gwt1SpecialistOutputMapV1,
    pub recorded_contributors: BTreeSet<String>,
    pub integrated: Gwt1IntegratedOutputBitsV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1MatchedShamRowRawV1 {
    pub target: Gwt1CausalTargetV1,
    pub target_specialist: String,
    pub full_panel: Gwt1MatchedShamArmRawV1,
    pub target_omitted: Gwt1MatchedShamArmRawV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1MatchedShamObservationsV1 {
    pub schema: String,
    pub rows: Vec<Gwt1MatchedShamRowRawV1>,
}

#[derive(Default)]
struct ShamPanel {
    drive: DriveManager,
    memory: MemoryManager,
    learning: LearningManager,
    perception: PerceptionManager,
}

struct ShamExecution {
    invoked: BTreeSet<String>,
    outputs: [Option<SubsystemOutput>; 4],
}

impl ShamPanel {
    fn process(
        &mut self,
        snapshot: &CycleSnapshot,
        omit: Option<Gwt1CausalTargetV1>,
    ) -> ShamExecution {
        let mut invoked = BTreeSet::new();
        let mut outputs = [None; 4];

        if omit != Some(Gwt1CausalTargetV1::Drive) {
            invoked.insert(Gwt1CausalTargetV1::Drive.id().to_string());
            outputs[0] = Some(self.drive.process(snapshot));
        }
        if omit != Some(Gwt1CausalTargetV1::Memory) {
            invoked.insert(Gwt1CausalTargetV1::Memory.id().to_string());
            outputs[1] = Some(self.memory.process(snapshot));
        }
        if omit != Some(Gwt1CausalTargetV1::Learning) {
            invoked.insert(Gwt1CausalTargetV1::Learning.id().to_string());
            outputs[2] = Some(self.learning.process(snapshot));
        }
        if omit != Some(Gwt1CausalTargetV1::Perception) {
            invoked.insert(Gwt1CausalTargetV1::Perception.id().to_string());
            outputs[3] = Some(self.perception.process(snapshot));
        }

        ShamExecution { invoked, outputs }
    }
}

/// Frozen direct-theorem baseline. Each causal row changes exactly one target
/// field away from this state; the sham removes that change and applies the
/// same specialist omission under a no-target-effect condition.
fn neutral_control_snapshot() -> CycleSnapshot {
    CycleSnapshot {
        prediction_error: 0.30,
        prediction_confidence: 0.50,
        coherence: 0.50,
        unified_psi: 0.40,
        arousal: 0.50,
        valence: 0.0,
        dissipative_health: 0.80,
        phenomenal_binding: 0.50,
        attention_budget_exceeded: 0,
        somatic_stress: 0.0,
        ..Default::default()
    }
}

fn output_map(outputs: &[Option<SubsystemOutput>; 4]) -> Gwt1SpecialistOutputMapV1 {
    GWT1_SPECIALIST_IDS_V1
        .into_iter()
        .zip(outputs.iter())
        .filter_map(|(id, output)| output.map(|value| (id.to_string(), value.into())))
        .collect()
}

fn integrate(
    outputs: &[Option<SubsystemOutput>; 4],
) -> (BTreeSet<String>, Gwt1IntegratedOutputBitsV1) {
    let mut collector = OutputCollector::new();
    let mut contributors = BTreeSet::new();
    for (index, id) in GWT1_SPECIALIST_IDS_V1.into_iter().enumerate() {
        if let Some(output) = outputs[index] {
            if !output.is_neutral() {
                contributors.insert(id.to_string());
            }
            collector.record(id, output);
        }
    }
    let integrated: IntegratedOutput = collector.integrate();
    (contributors, integrated.into())
}

fn arm(execution: ShamExecution) -> Gwt1MatchedShamArmRawV1 {
    let (recorded_contributors, integrated) = integrate(&execution.outputs);
    Gwt1MatchedShamArmRawV1 {
        invoked_specialists: execution.invoked,
        executed_outputs: output_map(&execution.outputs),
        recorded_contributors,
        integrated,
    }
}

fn run_row(target: Gwt1CausalTargetV1) -> Gwt1MatchedShamRowRawV1 {
    let snapshot = neutral_control_snapshot();
    let full = ShamPanel::default().process(&snapshot, None);
    let omitted = ShamPanel::default().process(&snapshot, Some(target));

    Gwt1MatchedShamRowRawV1 {
        target,
        target_specialist: target.id().to_string(),
        full_panel: arm(full),
        target_omitted: arm(omitted),
    }
}

/// Execute all four matched omission controls.
pub fn run_gwt1_causal_matched_sham_v1() -> Gwt1MatchedShamObservationsV1 {
    Gwt1MatchedShamObservationsV1 {
        schema: GWT1_CAUSAL_MATCHED_SHAM_SCHEMA_V1.to_string(),
        rows: Gwt1CausalTargetV1::ALL.into_iter().map(run_row).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_specialists() -> BTreeSet<String> {
        GWT1_SPECIALIST_IDS_V1
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn target_is_neutral_before_matched_omission() {
        let observations = run_gwt1_causal_matched_sham_v1();
        for row in &observations.rows {
            let target = row
                .full_panel
                .executed_outputs
                .get(&row.target_specialist)
                .expect("full panel executes target");
            assert_eq!(target.confidence_delta, 0.0f64.to_bits());
            assert_eq!(target.lr_modulation, 1.0f64.to_bits());
            assert_eq!(target.exploration_delta, 0.0f64.to_bits());
            assert_eq!(target.arousal_delta, 0.0f32.to_bits());
            assert_eq!(target.valence_delta, 0.0f32.to_bits());
            assert_eq!(target.flags, 0);
        }
    }

    #[test]
    fn same_omission_machinery_has_no_integrated_effect_when_target_is_neutral() {
        let observations = run_gwt1_causal_matched_sham_v1();
        let all = all_specialists();

        for row in &observations.rows {
            assert_eq!(row.full_panel.invoked_specialists, all);
            assert_eq!(row.target_omitted.invoked_specialists.len(), 3);
            assert!(!row.target_omitted.invoked_specialists.contains(&row.target_specialist));
            assert_eq!(row.target_omitted.integrated, row.full_panel.integrated);
            assert_eq!(
                row.target_omitted.recorded_contributors,
                row.full_panel.recorded_contributors
            );
        }
    }

    #[test]
    fn non_target_outputs_remain_bit_identical_under_matched_omission() {
        let observations = run_gwt1_causal_matched_sham_v1();
        for row in &observations.rows {
            for id in GWT1_SPECIALIST_IDS_V1 {
                if id == row.target_specialist {
                    continue;
                }
                assert_eq!(
                    row.target_omitted.executed_outputs.get(id),
                    row.full_panel.executed_outputs.get(id),
                    "matched omission leaked into non-target {id} for target {}",
                    row.target_specialist
                );
            }
        }
    }

    #[test]
    fn schema_and_target_order_are_frozen() {
        let observations = run_gwt1_causal_matched_sham_v1();
        assert_eq!(observations.schema, GWT1_CAUSAL_MATCHED_SHAM_SCHEMA_V1);
        assert_eq!(
            observations.rows.iter().map(|row| row.target).collect::<Vec<_>>(),
            Gwt1CausalTargetV1::ALL
        );
    }
}
