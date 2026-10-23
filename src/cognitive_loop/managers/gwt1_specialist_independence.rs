// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification-only GWT-1 specialist independence corpus.
//!
//! This module does not add a runtime evidence API and does not promote a
//! Butlin support tier by itself. It exercises the same production manager
//! implementations called by `CognitiveLoopService::phase_dynamics` against a
//! frozen `CycleSnapshot`.

use super::{DriveManager, LearningManager, MemoryManager, PerceptionManager};
use crate::cognitive_loop::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};
use rayon::{ThreadPoolBuilder, join};
use std::collections::HashSet;
use std::sync::{Arc, Barrier, mpsc};
use std::thread::ThreadId;

const DRIVE: usize = 0;
const MEMORY: usize = 1;
const LEARNING: usize = 2;
const PERCEPTION: usize = 3;

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

fn baseline_snapshot() -> CycleSnapshot {
    CycleSnapshot {
        prediction_error: 0.30,
        prediction_confidence: 0.50,
        coherence: 0.50,
        unified_psi: 0.40,          // below MemoryManager's 0.6 Psi gate
        arousal: 0.50,
        valence: 0.0,
        dissipative_health: 0.80,   // above LearningManager's 0.5 health gate
        phenomenal_binding: 0.50,  // between PerceptionManager's low/high gates
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

fn assert_output_bits_eq(actual: &SubsystemOutput, expected: &SubsystemOutput, context: &str) {
    assert_eq!(
        actual.confidence_delta.to_bits(),
        expected.confidence_delta.to_bits(),
        "{context}: confidence_delta"
    );
    assert_eq!(
        actual.lr_modulation.to_bits(),
        expected.lr_modulation.to_bits(),
        "{context}: lr_modulation"
    );
    assert_eq!(
        actual.exploration_delta.to_bits(),
        expected.exploration_delta.to_bits(),
        "{context}: exploration_delta"
    );
    assert_eq!(
        actual.arousal_delta.to_bits(),
        expected.arousal_delta.to_bits(),
        "{context}: arousal_delta"
    );
    assert_eq!(
        actual.valence_delta.to_bits(),
        expected.valence_delta.to_bits(),
        "{context}: valence_delta"
    );
    assert_eq!(actual.flags, expected.flags, "{context}: flags");
    assert_eq!(actual._reserved, expected._reserved, "{context}: reserved");
}

fn output_differs(left: &SubsystemOutput, right: &SubsystemOutput) -> bool {
    left.confidence_delta.to_bits() != right.confidence_delta.to_bits()
        || left.lr_modulation.to_bits() != right.lr_modulation.to_bits()
        || left.exploration_delta.to_bits() != right.exploration_delta.to_bits()
        || left.arousal_delta.to_bits() != right.arousal_delta.to_bits()
        || left.valence_delta.to_bits() != right.valence_delta.to_bits()
        || left.flags != right.flags
        || left._reserved != right._reserved
}

fn assert_diagonal_perturbation(
    label: &str,
    target: usize,
    baseline: CycleSnapshot,
    perturbed: CycleSnapshot,
) {
    let base_outputs = SpecialistPanel::default().process_sequential(&baseline);
    let perturbed_outputs = SpecialistPanel::default().process_sequential(&perturbed);

    for index in 0..4 {
        if index == target {
            assert!(
                output_differs(&base_outputs[index], &perturbed_outputs[index]),
                "{label}: target specialist output did not change"
            );
        } else {
            assert_output_bits_eq(
                &perturbed_outputs[index],
                &base_outputs[index],
                &format!("{label}: non-target specialist {index}"),
            );
        }
    }
}

#[test]
fn gwt1_specialization_matrix_is_diagonal() {
    // Drive-only perturbation: with prediction_error == exploration threshold,
    // DriveManager's neutral-valence homeostasis reads valence while the other
    // three selected managers do not.
    let baseline = baseline_snapshot();
    let mut valence = baseline;
    valence.valence = 0.80;
    assert_diagonal_perturbation("drive/valence", DRIVE, baseline, valence);

    // Memory-only perturbation: cross the explicit unified-Psi consolidation gate.
    let baseline = baseline_snapshot();
    let mut psi = baseline;
    psi.unified_psi = 0.80;
    assert_diagonal_perturbation("memory/unified_psi", MEMORY, baseline, psi);

    // Learning-only perturbation: cross the explicit dissipative-health gate.
    let baseline = baseline_snapshot();
    let mut health = baseline;
    health.dissipative_health = 0.10;
    assert_diagonal_perturbation(
        "learning/dissipative_health",
        LEARNING,
        baseline,
        health,
    );

    // Perception-only perturbation: cross the high phenomenal-binding gate.
    let baseline = baseline_snapshot();
    let mut binding = baseline;
    binding.phenomenal_binding = 0.90;
    assert_diagonal_perturbation(
        "perception/phenomenal_binding",
        PERCEPTION,
        baseline,
        binding,
    );
}

#[test]
fn each_specialist_matches_its_output_when_run_alone() {
    let snapshot = baseline_snapshot();
    let panel_outputs = SpecialistPanel::default().process_sequential(&snapshot);

    let drive = DriveManager::default().process(&snapshot);
    let memory = MemoryManager::default().process(&snapshot);
    let learning = LearningManager::default().process(&snapshot);
    let perception = PerceptionManager::default().process(&snapshot);
    let solo = [drive, memory, learning, perception];

    for index in 0..4 {
        assert_output_bits_eq(
            &panel_outputs[index],
            &solo[index],
            &format!("solo equivalence specialist {index}"),
        );
    }
}

#[test]
fn sequential_and_parallel_trajectories_are_bit_identical() {
    let mut sequential = SpecialistPanel::default();
    let mut parallel = SpecialistPanel::default();

    // Long enough to exercise the managers' rolling windows, streaks, adaptive
    // thresholds, consolidation timers, and vigilance transitions.
    for step in 0..48 {
        let snapshot = trajectory_snapshot(step);
        let seq = sequential.process_sequential(&snapshot);
        let par = parallel.process_parallel(&snapshot);

        for index in 0..4 {
            assert_output_bits_eq(
                &par[index],
                &seq[index],
                &format!("step {step}, specialist {index}"),
            );
        }
    }

    // Supplementary serialized-state identity. This is not treated as a proof
    // that every private field is represented by checkpoint(); the output
    // trajectory above is the primary equivalence theorem.
    assert_eq!(parallel.checkpoints(), sequential.checkpoints());
}

#[test]
fn dedicated_rayon_pool_executes_four_specialists_concurrently() {
    let pool = ThreadPoolBuilder::new()
        .num_threads(4)
        .thread_name(|index| format!("gwt1-specialist-{index}"))
        .build()
        .expect("dedicated four-thread qualification pool");

    let barrier = Arc::new(Barrier::new(4));
    let (tx, rx) = mpsc::channel::<(&'static str, ThreadId, SubsystemOutput)>();
    let snapshot = baseline_snapshot();

    {
        let barrier = Arc::clone(&barrier);
        let tx = tx.clone();
        pool.spawn(move || {
            let mut manager = DriveManager::default();
            barrier.wait();
            tx.send(("drive", std::thread::current().id(), manager.process(&snapshot)))
                .expect("drive result receiver alive");
        });
    }
    {
        let barrier = Arc::clone(&barrier);
        let tx = tx.clone();
        pool.spawn(move || {
            let mut manager = MemoryManager::default();
            barrier.wait();
            tx.send(("memory", std::thread::current().id(), manager.process(&snapshot)))
                .expect("memory result receiver alive");
        });
    }
    {
        let barrier = Arc::clone(&barrier);
        let tx = tx.clone();
        pool.spawn(move || {
            let mut manager = LearningManager::default();
            barrier.wait();
            tx.send((
                "learning",
                std::thread::current().id(),
                manager.process(&snapshot),
            ))
            .expect("learning result receiver alive");
        });
    }
    {
        let barrier = Arc::clone(&barrier);
        let tx = tx.clone();
        pool.spawn(move || {
            let mut manager = PerceptionManager::default();
            barrier.wait();
            tx.send((
                "perception",
                std::thread::current().id(),
                manager.process(&snapshot),
            ))
            .expect("perception result receiver alive");
        });
    }
    drop(tx);

    let results: Vec<_> = rx.iter().collect();
    assert_eq!(results.len(), 4, "all four specialist tasks must complete");

    let names: HashSet<_> = results.iter().map(|(name, _, _)| *name).collect();
    assert_eq!(
        names,
        HashSet::from(["drive", "memory", "learning", "perception"]),
        "all four production specialist implementations must execute"
    );

    let worker_ids: HashSet<_> = results.iter().map(|(_, id, _)| *id).collect();
    assert_eq!(
        worker_ids.len(),
        4,
        "barrier-qualified tasks must occupy four distinct Rayon workers"
    );
}
