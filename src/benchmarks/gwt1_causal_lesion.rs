// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Direct causal kernel for Butlin GWT-1 specialist independence.
//!
//! This module is a benchmark-only experiment. It selectively omits one real
//! production specialist from a frozen four-manager panel, separately records
//! execution identity and collector consequences, and performs an output-level
//! rescue by restoring the target's authentic baseline [`SubsystemOutput`]
//! without executing the target in the rescue arm.
//!
//! The experiment does **not** assign a Butlin support tier. Provenance,
//! qualification resolution, shams, replication, and evidence-tier promotion
//! belong to the psych-bench evidence layer.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::cognitive_loop::managers::{
    DriveManager, LearningManager, MemoryManager, PerceptionManager,
};
use crate::cognitive_loop::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, IntegratedOutput, OutputCollector, SubsystemOutput,
    output_flags,
};

use super::gwt1_specialist_qualification::{
    GWT1_SPECIALIST_IDS_V1, Gwt1SpecialistOutputMapV1, Gwt1SubsystemOutputBitsV1,
};

pub const GWT1_CAUSAL_LESION_SCHEMA_V1: &str = "butlin-gwt1-causal-lesion-v1";
pub const GWT1_CAUSAL_HELDOUT_STEPS_V1: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gwt1CausalTargetV1 {
    Drive,
    Memory,
    Learning,
    Perception,
}

impl Gwt1CausalTargetV1 {
    pub const ALL: [Self; 4] = [Self::Drive, Self::Memory, Self::Learning, Self::Perception];

    pub const fn id(self) -> &'static str {
        match self {
            Self::Drive => "drive_manager",
            Self::Memory => "memory_manager",
            Self::Learning => "learning_manager",
            Self::Perception => "perception_manager",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Drive => 0,
            Self::Memory => 1,
            Self::Learning => 2,
            Self::Perception => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gwt1CausalSignatureV1 {
    DriveValence,
    MemoryConsolidationFlag,
    LearningRate,
    PerceptionConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1IntegratedOutputBitsV1 {
    pub confidence_delta: u64,
    pub lr_modulation: u64,
    pub exploration_delta: u64,
    pub arousal_delta: u32,
    pub valence_delta: u32,
    pub flags: u32,
    pub n_contributors: usize,
}

impl From<IntegratedOutput> for Gwt1IntegratedOutputBitsV1 {
    fn from(output: IntegratedOutput) -> Self {
        Self {
            confidence_delta: output.confidence_delta.to_bits(),
            lr_modulation: output.lr_modulation.to_bits(),
            exploration_delta: output.exploration_delta.to_bits(),
            arousal_delta: output.arousal_delta.to_bits(),
            valence_delta: output.valence_delta.to_bits(),
            flags: output.flags,
            n_contributors: output.n_contributors,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1CausalArmRawV1 {
    /// Specialists whose real `process()` method executed in this arm.
    pub invoked_specialists: BTreeSet<String>,
    /// Raw outputs from executed specialists, including neutral outputs.
    pub executed_outputs: Gwt1SpecialistOutputMapV1,
    /// Non-neutral outputs that production `OutputCollector::record()` retained.
    pub recorded_contributors: BTreeSet<String>,
    /// Canonical Phase-C collector result.
    pub integrated: Gwt1IntegratedOutputBitsV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1HeldoutStepRawV1 {
    pub step: u32,
    pub baseline_non_target_outputs: Gwt1SpecialistOutputMapV1,
    pub lesion_non_target_outputs: Gwt1SpecialistOutputMapV1,
    pub rescue_non_target_outputs: Gwt1SpecialistOutputMapV1,
    pub sham_non_target_outputs: Gwt1SpecialistOutputMapV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1CausalRowRawV1 {
    pub target: Gwt1CausalTargetV1,
    pub target_specialist: String,
    pub preregistered_signature: Gwt1CausalSignatureV1,
    pub baseline: Gwt1CausalArmRawV1,
    pub lesion: Gwt1CausalArmRawV1,
    pub rescue: Gwt1CausalArmRawV1,
    pub sham: Gwt1CausalArmRawV1,
    /// Authentic target output from the baseline arm, injected into the rescue
    /// collector without executing the target in the rescue arm.
    pub rescue_injected_target_output: Gwt1SubsystemOutputBitsV1,
    pub heldout_non_target_trajectory: Vec<Gwt1HeldoutStepRawV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1CausalObservationsV1 {
    pub schema: String,
    pub rows: Vec<Gwt1CausalRowRawV1>,
}

#[derive(Default)]
struct CausalPanel {
    drive: DriveManager,
    memory: MemoryManager,
    learning: LearningManager,
    perception: PerceptionManager,
}

struct PanelExecution {
    invoked: BTreeSet<String>,
    slots: [Option<SubsystemOutput>; 4],
}

impl CausalPanel {
    fn process(&mut self, snapshot: &CycleSnapshot, omit: Option<Gwt1CausalTargetV1>) -> PanelExecution {
        let mut invoked = BTreeSet::new();
        let mut slots = [None; 4];

        if omit != Some(Gwt1CausalTargetV1::Drive) {
            invoked.insert(Gwt1CausalTargetV1::Drive.id().to_string());
            slots[0] = Some(self.drive.process(snapshot));
        }
        if omit != Some(Gwt1CausalTargetV1::Memory) {
            invoked.insert(Gwt1CausalTargetV1::Memory.id().to_string());
            slots[1] = Some(self.memory.process(snapshot));
        }
        if omit != Some(Gwt1CausalTargetV1::Learning) {
            invoked.insert(Gwt1CausalTargetV1::Learning.id().to_string());
            slots[2] = Some(self.learning.process(snapshot));
        }
        if omit != Some(Gwt1CausalTargetV1::Perception) {
            invoked.insert(Gwt1CausalTargetV1::Perception.id().to_string());
            slots[3] = Some(self.perception.process(snapshot));
        }

        PanelExecution { invoked, slots }
    }
}

fn target_snapshot(target: Gwt1CausalTargetV1) -> CycleSnapshot {
    let mut snapshot = CycleSnapshot {
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
    };

    match target {
        // Same orthogonal intervention doses as the direct specialization matrix.
        Gwt1CausalTargetV1::Drive => snapshot.valence = 0.80,
        Gwt1CausalTargetV1::Memory => snapshot.unified_psi = 0.80,
        Gwt1CausalTargetV1::Learning => snapshot.dissipative_health = 0.10,
        Gwt1CausalTargetV1::Perception => snapshot.phenomenal_binding = 0.90,
    }
    snapshot
}

fn heldout_snapshot(target: Gwt1CausalTargetV1, step: u32) -> CycleSnapshot {
    let mut snapshot = target_snapshot(target);
    snapshot.cycle_number = 10_000 + u64::from(step);
    // Deterministic held-out schedule, intentionally independent of lesion outcome.
    snapshot.prediction_error = ((step * 13 + 17) % 73) as f32 / 100.0 + 0.05;
    snapshot.prediction_confidence = f64::from(((step * 7 + 29) % 80) as f32 / 100.0 + 0.10);
    snapshot.coherence = ((step * 11 + 23) % 75) as f32 / 100.0 + 0.10;
    snapshot.arousal = ((step * 5 + 31) % 75) as f32 / 100.0 + 0.10;
    snapshot.somatic_stress = f64::from(((step * 17 + 3) % 60) as f32 / 100.0);
    snapshot
}

fn output_map(slots: &[Option<SubsystemOutput>; 4]) -> Gwt1SpecialistOutputMapV1 {
    GWT1_SPECIALIST_IDS_V1
        .into_iter()
        .zip(slots.iter())
        .filter_map(|(id, output)| output.map(|value| (id.to_string(), value.into())))
        .collect()
}

fn non_target_output_map(
    slots: &[Option<SubsystemOutput>; 4],
    target: Gwt1CausalTargetV1,
) -> Gwt1SpecialistOutputMapV1 {
    GWT1_SPECIALIST_IDS_V1
        .into_iter()
        .enumerate()
        .filter(|(index, _)| *index != target.index())
        .filter_map(|(index, id)| slots[index].map(|value| (id.to_string(), value.into())))
        .collect()
}

fn collector_result(slots: &[Option<SubsystemOutput>; 4]) -> (BTreeSet<String>, Gwt1IntegratedOutputBitsV1) {
    let mut collector = OutputCollector::new();
    let mut recorded = BTreeSet::new();

    for (index, id) in GWT1_SPECIALIST_IDS_V1.into_iter().enumerate() {
        if let Some(output) = slots[index] {
            if !output.is_neutral() {
                recorded.insert(id.to_string());
            }
            collector.record(id, output);
        }
    }

    (recorded, collector.integrate().into())
}

fn arm(execution: &PanelExecution, collector_slots: &[Option<SubsystemOutput>; 4]) -> Gwt1CausalArmRawV1 {
    let (recorded_contributors, integrated) = collector_result(collector_slots);
    Gwt1CausalArmRawV1 {
        invoked_specialists: execution.invoked.clone(),
        executed_outputs: output_map(&execution.slots),
        recorded_contributors,
        integrated,
    }
}

fn signature(target: Gwt1CausalTargetV1) -> Gwt1CausalSignatureV1 {
    match target {
        Gwt1CausalTargetV1::Drive => Gwt1CausalSignatureV1::DriveValence,
        Gwt1CausalTargetV1::Memory => Gwt1CausalSignatureV1::MemoryConsolidationFlag,
        Gwt1CausalTargetV1::Learning => Gwt1CausalSignatureV1::LearningRate,
        Gwt1CausalTargetV1::Perception => Gwt1CausalSignatureV1::PerceptionConfidence,
    }
}

fn run_row(target: Gwt1CausalTargetV1) -> Gwt1CausalRowRawV1 {
    let snapshot = target_snapshot(target);

    let mut baseline_panel = CausalPanel::default();
    let mut lesion_panel = CausalPanel::default();
    let mut rescue_panel = CausalPanel::default();
    let mut sham_panel = CausalPanel::default();

    let baseline_execution = baseline_panel.process(&snapshot, None);
    let lesion_execution = lesion_panel.process(&snapshot, Some(target));
    let rescue_execution = rescue_panel.process(&snapshot, Some(target));
    let sham_execution = sham_panel.process(&snapshot, None);

    let target_output = baseline_execution.slots[target.index()]
        .expect("baseline always executes the target specialist");

    // Canonical output order is preserved to make the production collector's
    // floating-point reduction exactly comparable to baseline.
    let mut rescue_slots = rescue_execution.slots;
    rescue_slots[target.index()] = Some(target_output);

    let baseline = arm(&baseline_execution, &baseline_execution.slots);
    let lesion = arm(&lesion_execution, &lesion_execution.slots);
    let rescue = arm(&rescue_execution, &rescue_slots);
    let sham = arm(&sham_execution, &sham_execution.slots);

    let mut heldout_non_target_trajectory = Vec::with_capacity(GWT1_CAUSAL_HELDOUT_STEPS_V1 as usize);
    for step in 0..GWT1_CAUSAL_HELDOUT_STEPS_V1 {
        let heldout = heldout_snapshot(target, step);
        let baseline_future = baseline_panel.process(&heldout, Some(target));
        let lesion_future = lesion_panel.process(&heldout, Some(target));
        let rescue_future = rescue_panel.process(&heldout, Some(target));
        let sham_future = sham_panel.process(&heldout, Some(target));

        heldout_non_target_trajectory.push(Gwt1HeldoutStepRawV1 {
            step,
            baseline_non_target_outputs: non_target_output_map(&baseline_future.slots, target),
            lesion_non_target_outputs: non_target_output_map(&lesion_future.slots, target),
            rescue_non_target_outputs: non_target_output_map(&rescue_future.slots, target),
            sham_non_target_outputs: non_target_output_map(&sham_future.slots, target),
        });
    }

    Gwt1CausalRowRawV1 {
        target,
        target_specialist: target.id().to_string(),
        preregistered_signature: signature(target),
        baseline,
        lesion,
        rescue,
        sham,
        rescue_injected_target_output: target_output.into(),
        heldout_non_target_trajectory,
    }
}

/// Execute the four-row selective-lesion experiment and return raw observations.
///
/// This is a component/causal experiment only. It does not promote evidence.
pub fn run_gwt1_causal_lesion_v1() -> Gwt1CausalObservationsV1 {
    Gwt1CausalObservationsV1 {
        schema: GWT1_CAUSAL_LESION_SCHEMA_V1.to_string(),
        rows: Gwt1CausalTargetV1::ALL.into_iter().map(run_row).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_set() -> BTreeSet<String> {
        GWT1_SPECIALIST_IDS_V1
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    fn expected_non_target_set(target: Gwt1CausalTargetV1) -> BTreeSet<String> {
        GWT1_SPECIALIST_IDS_V1
            .into_iter()
            .filter(|id| *id != target.id())
            .map(str::to_string)
            .collect()
    }

    fn target_output(row: &Gwt1CausalRowRawV1) -> &Gwt1SubsystemOutputBitsV1 {
        row.baseline
            .executed_outputs
            .get(&row.target_specialist)
            .expect("baseline target output")
    }

    #[test]
    fn manipulation_identity_is_independent_of_collector_contribution() {
        let observations = run_gwt1_causal_lesion_v1();
        let all = canonical_set();

        for row in &observations.rows {
            let controls = expected_non_target_set(row.target);
            assert_eq!(row.baseline.invoked_specialists, all);
            assert_eq!(row.sham.invoked_specialists, all);
            assert_eq!(row.lesion.invoked_specialists, controls);
            assert_eq!(row.rescue.invoked_specialists, controls);
            assert!(!row.rescue.invoked_specialists.contains(&row.target_specialist));

            // Execution identity is deliberately not inferred from collector retention.
            assert_eq!(row.baseline.executed_outputs.len(), 4);
            assert_eq!(row.lesion.executed_outputs.len(), 3);
            assert_eq!(row.rescue.executed_outputs.len(), 3);
            assert_eq!(row.sham.executed_outputs.len(), 4);
        }
    }

    #[test]
    fn target_outputs_are_non_neutral_and_match_frozen_signatures() {
        let observations = run_gwt1_causal_lesion_v1();

        for row in &observations.rows {
            let output = target_output(row);
            match row.preregistered_signature {
                Gwt1CausalSignatureV1::DriveValence => {
                    assert_ne!(output.valence_delta, 0.0f32.to_bits());
                }
                Gwt1CausalSignatureV1::MemoryConsolidationFlag => {
                    assert_ne!(output.flags & output_flags::REQUEST_CONSOLIDATION, 0);
                }
                Gwt1CausalSignatureV1::LearningRate => {
                    assert_ne!(output.lr_modulation, 1.0f64.to_bits());
                }
                Gwt1CausalSignatureV1::PerceptionConfidence => {
                    assert_ne!(output.confidence_delta, 0.0f64.to_bits());
                }
            }
        }
    }

    #[test]
    fn lesion_is_selective_for_non_target_specialist_outputs() {
        let observations = run_gwt1_causal_lesion_v1();

        for row in &observations.rows {
            for id in GWT1_SPECIALIST_IDS_V1 {
                if id == row.target_specialist {
                    assert!(!row.lesion.executed_outputs.contains_key(id));
                    assert!(!row.rescue.executed_outputs.contains_key(id));
                    continue;
                }
                let baseline = row.baseline.executed_outputs.get(id).expect("baseline control output");
                assert_eq!(row.lesion.executed_outputs.get(id), Some(baseline));
                assert_eq!(row.rescue.executed_outputs.get(id), Some(baseline));
                assert_eq!(row.sham.executed_outputs.get(id), Some(baseline));
            }
        }
    }

    #[test]
    fn lesion_changes_preregistered_integrated_channel_and_rescue_is_exact() {
        let observations = run_gwt1_causal_lesion_v1();

        for row in &observations.rows {
            assert_eq!(row.sham.integrated, row.baseline.integrated);
            assert_eq!(row.rescue.integrated, row.baseline.integrated);
            assert_ne!(row.lesion.integrated, row.baseline.integrated);

            match row.preregistered_signature {
                Gwt1CausalSignatureV1::DriveValence => assert_ne!(
                    row.lesion.integrated.valence_delta,
                    row.baseline.integrated.valence_delta
                ),
                Gwt1CausalSignatureV1::MemoryConsolidationFlag => assert_ne!(
                    row.lesion.integrated.flags & output_flags::REQUEST_CONSOLIDATION,
                    row.baseline.integrated.flags & output_flags::REQUEST_CONSOLIDATION
                ),
                Gwt1CausalSignatureV1::LearningRate => assert_ne!(
                    row.lesion.integrated.lr_modulation,
                    row.baseline.integrated.lr_modulation
                ),
                Gwt1CausalSignatureV1::PerceptionConfidence => assert_ne!(
                    row.lesion.integrated.confidence_delta,
                    row.baseline.integrated.confidence_delta
                ),
            }
        }
    }

    #[test]
    fn rescue_uses_exact_authentic_target_output_without_executing_target() {
        let observations = run_gwt1_causal_lesion_v1();

        for row in &observations.rows {
            assert_eq!(&row.rescue_injected_target_output, target_output(row));
            assert!(!row.rescue.invoked_specialists.contains(&row.target_specialist));
            assert!(!row.rescue.executed_outputs.contains_key(&row.target_specialist));
            assert!(row.rescue.recorded_contributors.contains(&row.target_specialist));
        }
    }

    #[test]
    fn heldout_non_target_trajectory_stays_bit_identical_after_lesion() {
        let observations = run_gwt1_causal_lesion_v1();

        for row in &observations.rows {
            assert_eq!(
                row.heldout_non_target_trajectory.len(),
                GWT1_CAUSAL_HELDOUT_STEPS_V1 as usize
            );
            for step in &row.heldout_non_target_trajectory {
                assert_eq!(step.lesion_non_target_outputs, step.baseline_non_target_outputs);
                assert_eq!(step.rescue_non_target_outputs, step.baseline_non_target_outputs);
                assert_eq!(step.sham_non_target_outputs, step.baseline_non_target_outputs);
                assert_eq!(step.baseline_non_target_outputs.len(), 3);
            }
        }
    }

    #[test]
    fn schema_and_row_order_are_frozen() {
        let observations = run_gwt1_causal_lesion_v1();
        assert_eq!(observations.schema, GWT1_CAUSAL_LESION_SCHEMA_V1);
        assert_eq!(observations.rows.len(), 4);
        assert_eq!(
            observations.rows.iter().map(|row| row.target).collect::<Vec<_>>(),
            Gwt1CausalTargetV1::ALL
        );
    }
}
