// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Episode-based training for the manipulator arm.

use crate::controller::{ControllerWeights, ManipulatorController};
use crate::encoder::ManipulatorHdcEncoder;
use crate::fep_agent::ActiveInferenceManipulatorAgent;
use crate::sensorimotor::{ControllerInputSchema, SensorimotorInputBinder};
use crate::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};
use crate::types::{ManipulatorConfig, NUM_JOINTS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

/// Metrics from a single training episode.
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub diverged: bool,
    /// Mean pre-update imitation MSE against the simulator's own real
    /// gravity-compensation torques -- the learning signal (real-trainer
    /// follow-up to SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md).
    pub mean_imitation_loss: f32,
}

/// Episode-based trainer for the manipulator.
pub struct ManipulatorTrainer {
    controller: ManipulatorController,
    encoder: ManipulatorHdcEncoder,
    sensorimotor_input: SensorimotorInputBinder,
    neutral_thought: ContinuousHV,
    fep: ActiveInferenceManipulatorAgent,
    simulator: SimpleManipulatorSimulator,
    config: ManipulatorConfig,
}

impl ManipulatorTrainer {
    pub fn new(config: ManipulatorConfig) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let neutral_thought = intent_hv(&genesis, "hold_stable");
        Self {
            controller: ManipulatorController::new(&genesis, &config),
            encoder: ManipulatorHdcEncoder::new(&genesis, 32),
            sensorimotor_input: SensorimotorInputBinder::new(&genesis),
            neutral_thought,
            fep: ActiveInferenceManipulatorAgent::new(),
            simulator: SimpleManipulatorSimulator::new(),
            config,
        }
    }

    pub fn run_episode(&mut self) -> EpisodeMetrics {
        self.simulator.reset();
        self.controller.reset();
        let dt = self.config.physics_dt();
        let mut total_effort = 0.0f32;
        let mut tau_factor = 1.0f32;
        let mut loss_sum = 0.0f32;

        for step in 0..self.config.steps_per_episode {
            let body_hv = self.encoder.encode(self.simulator.state());
            let hv = self
                .sensorimotor_input
                .fuse(&self.neutral_thought, &body_hv);
            if step % self.config.cognitive_interval == 0 {
                let fep = self.fep.tick(self.simulator.state());
                tau_factor = fep.tau_factor;
            }
            let cmd = self.controller.forward(&hv, dt as f32 * tau_factor);
            let target = self.simulator.gravity_compensation_torques();
            loss_sum += self.controller.train_step(&target);
            self.simulator.step(&cmd, dt);
            total_effort += cmd.control_effort();

            if !self.simulator.state().is_finite() {
                return EpisodeMetrics {
                    mean_effort: total_effort / (step + 1) as f32,
                    steps_survived: step,
                    diverged: true,
                    mean_imitation_loss: loss_sum / (step + 1) as f32,
                };
            }
        }

        EpisodeMetrics {
            mean_effort: total_effort / self.config.steps_per_episode as f32,
            steps_survived: self.config.steps_per_episode,
            diverged: false,
            mean_imitation_loss: loss_sum / self.config.steps_per_episode as f32,
        }
    }

    /// Export the trained output layer for transfer into a shipped bridge
    /// (`ManipulatorEmbodiment::install_weights`). Closes the trainer-island
    /// gap: before this, trained weights could never leave the trainer.
    pub fn export_weights(&self) -> ControllerWeights {
        self.controller
            .export_weights()
            .with_input_schema(ControllerInputSchema::SensorimotorBoundV1)
    }
}

/// Deterministic genesis-derived intent thought vector — the vocabulary the
/// intent curriculum trains against and callers steer with at runtime.
pub fn intent_hv(genesis: &GenesisSeed, intent: &str) -> ContinuousHV {
    ContinuousHV::from_genesis(
        genesis,
        &format!("manipulator::intent::{intent}"),
        HDC_DIMENSION,
    )
}

/// Intent-conditioned curriculum: teach opposing intent thoughts opposing
/// base-joint torque patterns through the same role-bound sensorimotor input
/// path used by the shipped bridge.
///
/// Both this curriculum and `ManipulatorTrainer` now bind thought and body
/// state into the `SensorimotorBoundV1` schema. The cognition-ablation
/// experiment (2026-07-08) showed that with genesis-random weights, opposite
/// intents produce zero task-axis separation through the bridge; these
/// weights make thought an actual control signal without discarding
/// proprioception. The controller is reset before each intent block so
/// training features match the bridge's from-reset serving distribution.
///
/// Returns weights mapping `intent_hv(genesis, "reach_left")` to positive
/// base-joint torque and `"reach_right"` to negative.
pub fn train_intent_weights(config: &ManipulatorConfig, epochs: usize) -> ControllerWeights {
    let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
    let mut controller = ManipulatorController::new(&genesis, config);
    let mut encoder = ManipulatorHdcEncoder::new(&genesis, 32);
    let binder = SensorimotorInputBinder::new(&genesis);
    let home_body = encoder.encode(&crate::types::ManipulatorState::home());
    let dt = config.physics_dt() as f32;

    let left = intent_hv(&genesis, "reach_left");
    let right = intent_hv(&genesis, "reach_right");
    let mut target_left = [0.0f32; NUM_JOINTS];
    target_left[0] = 0.4;
    let mut target_right = [0.0f32; NUM_JOINTS];
    target_right[0] = -0.4;

    const SETTLE_STEPS: usize = 10;
    for _ in 0..epochs {
        for (thought, target) in [(&left, &target_left), (&right, &target_right)] {
            let hv = binder.fuse(thought, &home_body);
            controller.reset();
            for _ in 0..SETTLE_STEPS {
                controller.forward(&hv, dt);
                controller.train_step(target);
            }
        }
    }
    controller
        .export_weights()
        .with_input_schema(ControllerInputSchema::SensorimotorBoundV1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episode_completes() {
        let mut config = ManipulatorConfig::default();
        config.steps_per_episode = 200;
        let mut trainer = ManipulatorTrainer::new(config);
        let metrics = trainer.run_episode();
        assert!(!metrics.diverged);
        assert_eq!(metrics.steps_survived, 200);
        assert!(metrics.mean_effort.is_finite());
        assert!(
            metrics.mean_imitation_loss > 0.0,
            "learning signal must be live"
        );
    }

    #[test]
    fn test_training_reduces_imitation_loss() {
        let mut config = ManipulatorConfig::default();
        config.steps_per_episode = 400;
        let mut trainer = ManipulatorTrainer::new(config);
        let first = trainer.run_episode().mean_imitation_loss;
        for _ in 0..3 {
            trainer.run_episode();
        }
        let last = trainer.run_episode().mean_imitation_loss;
        assert!(
            last < first,
            "imitation loss must decrease with training: first {first:.5} -> last {last:.5}"
        );
    }
    #[test]
    fn exported_training_weights_declare_sensorimotor_schema() {
        let trainer = ManipulatorTrainer::new(ManipulatorConfig::default());
        assert_eq!(
            trainer.export_weights().input_schema,
            ControllerInputSchema::SensorimotorBoundV1
        );
    }
}
