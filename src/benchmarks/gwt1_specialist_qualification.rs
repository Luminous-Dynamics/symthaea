// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structured raw-observation runner for the direct Butlin GWT-1 specialist theorem.
//!
//! This module is available only under the root crate's `benchmarks` feature.
//! It intentionally exposes structured experiment observations, not internal
//! manager types and not a Butlin support-tier decision.

use crate::cognitive_loop::managers::{
    DriveManager, LearningManager, MemoryManager, PerceptionManager,
};
use crate::cognitive_loop::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};
use rayon::{ThreadPoolBuilder, join};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Barrier, mpsc};
use std::time::Duration;

pub const GWT1_RAW_OBSERVATION_SCHEMA_V1: &str = "butlin-gwt1-raw-observations-v1";
pub const GWT1_SPECIALIZATION_CONTRACT_V1: &str = "gwt1-specialization-matrix-v1";
pub const GWT1_TRAJECTORY_SCHEDULE_V1: &str = "gwt1-specialist-trajectory-v1-48";
pub const GWT1_TRAJECTORY_STEPS_V1: u32 = 48;
pub const GWT1_SPECIALIST_IDS_V1: [&str; 4] = [
    "drive_manager",
    "memory_manager",
    "learning_manager",
    "perception_manager",
];
const GWT1_CONCURRENCY_RESULT_TIMEOUT_V1: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1SubsystemOutputBitsV1 {
    pub confidence_delta: u64,
    pub lr_modulation: u64,
    pub exploration_delta: u64,
    pub arousal_delta: u32,
    pub valence_delta: u32,
    pub flags: u32,
    pub reserved: u32,
}

impl From<SubsystemOutput> for Gwt1SubsystemOutputBitsV1 {
    fn from(output: SubsystemOutput) -> Self {
        Self {
            confidence_delta: output.confidence_delta.to_bits(),
            lr_modulation: output.lr_modulation.to_bits(),
            exploration_delta: output.exploration_delta.to_bits(),
            arousal_delta: output.arousal_delta.to_bits(),
            valence_delta: output.valence_delta.to_bits(),
            flags: output.flags,
            reserved: output._reserved,
        }
    }
}

pub type Gwt1SpecialistOutputMapV1 = BTreeMap<String, Gwt1SubsystemOutputBitsV1>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gwt1PerturbationRawV1 {
    pub id: String,
    pub field: String,
    pub target_specialist: String,
    pub baseline_value: f64,
    pub perturbed_value: f64,
    pub baseline_outputs: Gwt1SpecialistOutputMapV1,
    pub perturbed_outputs: Gwt1SpecialistOutputMapV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1SoloPanelRawV1 {
    pub solo_outputs: Gwt1SpecialistOutputMapV1,
    pub panel_outputs: Gwt1SpecialistOutputMapV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1TrajectoryStepRawV1 {
    pub step: u32,
    pub sequential_outputs: Gwt1SpecialistOutputMapV1,
    pub parallel_outputs: Gwt1SpecialistOutputMapV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1ConcurrencyRawV1 {
    pub requested_workers: u32,
    pub barrier_participants: u32,
    pub completed_specialists: BTreeSet<String>,
    pub worker_names: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gwt1RawObservationsV1 {
    pub schema: String,
    pub specialization_contract: String,
    pub trajectory_schedule: String,
    pub perturbations: Vec<Gwt1PerturbationRawV1>,
    pub solo_panel: Gwt1SoloPanelRawV1,
    pub trajectory: Vec<Gwt1TrajectoryStepRawV1>,
    pub concurrency: Gwt1ConcurrencyRawV1,
    /// Supplementary only until the manager checkpoint contract is replay-complete.
    pub checkpoint_equal_supplementary: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gwt1RunnerErrorV1 {
    ThreadPoolBuild(String),
    ConcurrentTaskTimeout { completed: u32 },
    ConcurrentTaskChannelClosed { completed: u32 },
}

#[derive(Default)]
struct SpecialistPanel {
    drive: DriveManager,
    memory: MemoryManager,
    learning: LearningManager,
    perception: PerceptionManager,
}

impl SpecialistPanel {
    fn process_sequential(&mut self, snapshot: &CycleSnapshot) -> [SubsystemOutput; 4] {
        [
            self.drive.process(snapshot),
            self.memory.process(snapshot),
            self.learning.process(snapshot),
            self.perception.process(snapshot),
        ]
    }

    fn process_parallel(&mut self, snapshot: &CycleSnapshot) -> [SubsystemOutput; 4] {
        let Self {
            drive,
            memory,
            learning,
            perception,
        } = self;

        let ((drive_out, memory_out), (learning_out, perception_out)) = join(
            || join(|| drive.process(snapshot), || memory.process(snapshot)),
            || join(|| learning.process(snapshot), || perception.process(snapshot)),
        );

        [drive_out, memory_out, learning_out, perception_out]
    }

    fn checkpoints(&self) -> [Vec<u8>; 4] {
        [
            self.drive.checkpoint(),
            self.memory.checkpoint(),
            self.learning.checkpoint(),
            self.perception.checkpoint(),
        ]
    }
}

fn output_map(outputs: [SubsystemOutput; 4]) -> Gwt1SpecialistOutputMapV1 {
    GWT1_SPECIALIST_IDS_V1
        .into_iter()
        .zip(outputs)
        .map(|(id, output)| (id.to_string(), output.into()))
        .collect()
}

fn baseline_snapshot() -> CycleSnapshot {
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

fn trajectory_snapshot(step: u32) -> CycleSnapshot {
    let mut snapshot = baseline_snapshot();
    snapshot.cycle_number = u64::from(step);
    snapshot.prediction_error = ((step * 17 + 13) % 91) as f32 / 100.0;
    snapshot.prediction_confidence = f64::from(((step * 11 + 37) % 90) as f32 / 100.0 + 0.05);
    snapshot.coherence = ((step * 7 + 19) % 80) as f32 / 100.0 + 0.10;
    snapshot.unified_psi = f64::from(((step * 5 + 23) % 90) as f32 / 100.0 + 0.05);
    snapshot.arousal = ((step * 13 + 29) % 90) as f32 / 100.0 + 0.05;
    snapshot.valence = (((step * 19 + 31) % 180) as f32 - 90.0) / 100.0;
    snapshot.dissipative_health =
        f64::from(((step * 3 + 41) % 90) as f32 / 100.0 + 0.05);
    snapshot.phenomenal_binding =
        f64::from(((step * 23 + 17) % 90) as f32 / 100.0 + 0.05);
    snapshot.somatic_stress = f64::from(((step * 29 + 7) % 85) as f32 / 100.0);
    snapshot.attention_budget_exceeded = u8::from(step % 5 == 0);
    snapshot
}

fn perturbation(
    id: &str,
    field: &str,
    target_specialist: &str,
    baseline_value: f64,
    perturbed_value: f64,
    baseline: CycleSnapshot,
    perturbed: CycleSnapshot,
) -> Gwt1PerturbationRawV1 {
    Gwt1PerturbationRawV1 {
        id: id.to_string(),
        field: field.to_string(),
        target_specialist: target_specialist.to_string(),
        baseline_value,
        perturbed_value,
        baseline_outputs: output_map(SpecialistPanel::default().process_sequential(&baseline)),
        perturbed_outputs: output_map(SpecialistPanel::default().process_sequential(&perturbed)),
    }
}

fn run_perturbations() -> Vec<Gwt1PerturbationRawV1> {
    let baseline = baseline_snapshot();

    let mut valence = baseline;
    valence.valence = 0.80;

    let mut psi = baseline;
    psi.unified_psi = 0.80;

    let mut health = baseline;
    health.dissipative_health = 0.10;

    let mut binding = baseline;
    binding.phenomenal_binding = 0.90;

    vec![
        perturbation(
            "drive-valence",
            "valence",
            "drive_manager",
            f64::from(baseline.valence),
            f64::from(valence.valence),
            baseline,
            valence,
        ),
        perturbation(
            "memory-unified-psi",
            "unified_psi",
            "memory_manager",
            baseline.unified_psi,
            psi.unified_psi,
            baseline,
            psi,
        ),
        perturbation(
            "learning-dissipative-health",
            "dissipative_health",
            "learning_manager",
            baseline.dissipative_health,
            health.dissipative_health,
            baseline,
            health,
        ),
        perturbation(
            "perception-phenomenal-binding",
            "phenomenal_binding",
            "perception_manager",
            baseline.phenomenal_binding,
            binding.phenomenal_binding,
            baseline,
            binding,
        ),
    ]
}

fn run_solo_panel() -> Gwt1SoloPanelRawV1 {
    let snapshot = baseline_snapshot();
    let panel_outputs = output_map(SpecialistPanel::default().process_sequential(&snapshot));
    let solo_outputs = output_map([
        DriveManager::default().process(&snapshot),
        MemoryManager::default().process(&snapshot),
        LearningManager::default().process(&snapshot),
        PerceptionManager::default().process(&snapshot),
    ]);

    Gwt1SoloPanelRawV1 {
        solo_outputs,
        panel_outputs,
    }
}

fn run_trajectory() -> (Vec<Gwt1TrajectoryStepRawV1>, bool) {
    let mut sequential = SpecialistPanel::default();
    let mut parallel = SpecialistPanel::default();
    let mut trajectory = Vec::with_capacity(GWT1_TRAJECTORY_STEPS_V1 as usize);

    for step in 0..GWT1_TRAJECTORY_STEPS_V1 {
        let snapshot = trajectory_snapshot(step);
        trajectory.push(Gwt1TrajectoryStepRawV1 {
            step,
            sequential_outputs: output_map(sequential.process_sequential(&snapshot)),
            parallel_outputs: output_map(parallel.process_parallel(&snapshot)),
        });
    }

    let checkpoint_equal = sequential.checkpoints() == parallel.checkpoints();
    (trajectory, checkpoint_equal)
}

fn run_concurrency() -> Result<Gwt1ConcurrencyRawV1, Gwt1RunnerErrorV1> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(4)
        .thread_name(|index| format!("gwt1-specialist-{index}"))
        .build()
        .map_err(|error| Gwt1RunnerErrorV1::ThreadPoolBuild(error.to_string()))?;

    let barrier = Arc::new(Barrier::new(4));
    let (tx, rx) = mpsc::channel::<(&'static str, String)>();
    let snapshot = baseline_snapshot();

    let spawn = |id: &'static str,
                 barrier: Arc<Barrier>,
                 tx: mpsc::Sender<(&'static str, String)>,
                 task: Box<dyn FnOnce() + Send>| {
        pool.spawn(move || {
            barrier.wait();
            let worker = std::thread::current()
                .name()
                .unwrap_or("unnamed-rayon-worker")
                .to_string();
            task();
            let _ = tx.send((id, worker));
        });
    };

    {
        let snapshot = snapshot;
        spawn(
            "drive_manager",
            Arc::clone(&barrier),
            tx.clone(),
            Box::new(move || {
                let _ = DriveManager::default().process(&snapshot);
            }),
        );
    }
    {
        let snapshot = snapshot;
        spawn(
            "memory_manager",
            Arc::clone(&barrier),
            tx.clone(),
            Box::new(move || {
                let _ = MemoryManager::default().process(&snapshot);
            }),
        );
    }
    {
        let snapshot = snapshot;
        spawn(
            "learning_manager",
            Arc::clone(&barrier),
            tx.clone(),
            Box::new(move || {
                let _ = LearningManager::default().process(&snapshot);
            }),
        );
    }
    {
        let snapshot = snapshot;
        spawn(
            "perception_manager",
            Arc::clone(&barrier),
            tx.clone(),
            Box::new(move || {
                let _ = PerceptionManager::default().process(&snapshot);
            }),
        );
    }
    drop(tx);

    let mut completed_specialists = BTreeSet::new();
    let mut worker_names = BTreeMap::new();
    for completed in 0..4_u32 {
        match rx.recv_timeout(GWT1_CONCURRENCY_RESULT_TIMEOUT_V1) {
            Ok((id, worker)) => {
                completed_specialists.insert(id.to_string());
                worker_names.insert(id.to_string(), worker);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Err(Gwt1RunnerErrorV1::ConcurrentTaskTimeout { completed });
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(Gwt1RunnerErrorV1::ConcurrentTaskChannelClosed { completed });
            }
        }
    }

    Ok(Gwt1ConcurrencyRawV1 {
        requested_workers: 4,
        barrier_participants: 4,
        completed_specialists,
        worker_names,
    })
}

/// Execute the direct GWT-1 specialist-independence experiment and return raw observations.
///
/// This function does not interpret the result as a Butlin support tier. The psych-bench
/// evidence layer owns qualification resolution and provenance binding.
pub fn run_gwt1_specialist_qualification_v1(
) -> Result<Gwt1RawObservationsV1, Gwt1RunnerErrorV1> {
    let perturbations = run_perturbations();
    let solo_panel = run_solo_panel();
    let (trajectory, checkpoint_equal_supplementary) = run_trajectory();
    let concurrency = run_concurrency()?;

    Ok(Gwt1RawObservationsV1 {
        schema: GWT1_RAW_OBSERVATION_SCHEMA_V1.to_string(),
        specialization_contract: GWT1_SPECIALIZATION_CONTRACT_V1.to_string(),
        trajectory_schedule: GWT1_TRAJECTORY_SCHEDULE_V1.to_string(),
        perturbations,
        solo_panel,
        trajectory,
        concurrency,
        checkpoint_equal_supplementary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_emits_frozen_complete_shape() {
        let raw = run_gwt1_specialist_qualification_v1().expect("GWT-1 runner");
        assert_eq!(raw.schema, GWT1_RAW_OBSERVATION_SCHEMA_V1);
        assert_eq!(raw.specialization_contract, GWT1_SPECIALIZATION_CONTRACT_V1);
        assert_eq!(raw.trajectory_schedule, GWT1_TRAJECTORY_SCHEDULE_V1);
        assert_eq!(raw.perturbations.len(), 4);
        assert_eq!(raw.trajectory.len(), GWT1_TRAJECTORY_STEPS_V1 as usize);
        assert!(
            raw.trajectory
                .iter()
                .enumerate()
                .all(|(index, step)| step.step == index as u32)
        );
        assert_eq!(raw.concurrency.completed_specialists.len(), 4);
        assert_eq!(raw.concurrency.worker_names.len(), 4);
    }

    #[test]
    fn raw_json_is_deterministic_for_non_concurrency_observations() {
        let first = run_gwt1_specialist_qualification_v1().expect("first run");
        let second = run_gwt1_specialist_qualification_v1().expect("second run");

        assert_eq!(first.specialization_contract, second.specialization_contract);
        assert_eq!(first.trajectory_schedule, second.trajectory_schedule);
        assert_eq!(first.perturbations, second.perturbations);
        assert_eq!(first.solo_panel, second.solo_panel);
        assert_eq!(first.trajectory, second.trajectory);
        assert_eq!(
            first.checkpoint_equal_supplementary,
            second.checkpoint_equal_supplementary
        );
    }
}
