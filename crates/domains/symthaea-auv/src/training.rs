// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-rate training loop for AUV (100Hz motor / 10Hz cognitive).

use crate::controller::AuvController;
use crate::encoder::AuvHdcEncoder;
use crate::fep_agent::ActiveInferenceAuvAgent;
use crate::navigation_bridge::{AuvNavigationBridge, UnderwaterNavigationSample};
use crate::navigation_estimator::{AuvNavigationEstimate, AuvNavigationEstimator};
use crate::reflex::reflex_thrusters;
use crate::simulator::{AuvPhysicsSimulator, SimpleAuvSimulator};
use crate::store_forward::StoreForwardBuffer;
use crate::types::AuvConfig;
use positioning::Measurement;

/// Episode metrics.
#[derive(Debug, Clone, Default)]
pub struct EpisodeMetrics {
    pub mean_depth_error: f64,
    pub mean_control_effort: f32,
    pub mean_speed: f64,
    pub navigation_measurements_emitted: usize,
    pub final_navigation_sigma_m: f64,
    pub final_navigation_position_error_m: f64,
    pub samples_collected: usize,
    pub steps_survived: usize,
    pub exceeded_crush_depth: bool,
    /// Mean pre-update imitation MSE against the hand-designed depth-hold
    /// reflex -- the learning signal (real-trainer follow-up to
    /// SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md).
    pub mean_imitation_loss: f32,
}

/// AUV trainer.
pub struct AuvTrainer {
    controller: AuvController,
    encoder: AuvHdcEncoder,
    fep_agent: ActiveInferenceAuvAgent,
    simulator: SimpleAuvSimulator,
    store_forward: StoreForwardBuffer,
    navigation_bridge: AuvNavigationBridge,
    navigation_estimator: AuvNavigationEstimator,
    episode_navigation_measurements: Vec<Measurement>,
    config: AuvConfig,
    target_depth: f64,
}

impl AuvTrainer {
    pub fn new(config: AuvConfig, target_depth: f64) -> Self {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(&config.genesis_phrase);
        Self {
            controller: AuvController::new(&genesis, &config),
            encoder: AuvHdcEncoder::new(&genesis, 32),
            fep_agent: ActiveInferenceAuvAgent::new(),
            simulator: SimpleAuvSimulator::new(),
            store_forward: StoreForwardBuffer::default_capacity(),
            navigation_bridge: AuvNavigationBridge::new("auv_trainer"),
            navigation_estimator: AuvNavigationEstimator::new([0.0, 0.0, target_depth], 25.0),
            episode_navigation_measurements: Vec::new(),
            config,
            target_depth,
        }
    }

    pub fn run_episode(&mut self) -> EpisodeMetrics {
        let dt = self.config.physics_dt();
        let mut metrics = EpisodeMetrics::default();
        let mut depth_error_sum = 0.0;
        let mut effort_sum = 0.0f32;
        let mut speed_sum = 0.0;
        let mut tau_factor = 1.0f32;
        let mut loss_sum = 0.0f32;

        self.simulator.reset(self.target_depth);
        self.controller.reset();
        self.encoder.reset();
        self.fep_agent.reset();
        self.store_forward.clear();
        self.navigation_estimator
            .reset([0.0, 0.0, self.target_depth], 25.0);
        self.episode_navigation_measurements.clear();

        for step in 0..self.config.steps_per_episode {
            let hv = self.encoder.encode(self.simulator.state());

            if step % self.config.cognitive_interval == 0 {
                self.store_forward
                    .update_blackout(self.simulator.state().depth);
                self.fep_agent.set_blackout(self.store_forward.in_blackout);
                let fep = self
                    .fep_agent
                    .tick(self.simulator.state(), self.target_depth);
                tau_factor = fep.tau_factor;
            }

            let cmd = self.controller.forward(&hv, dt as f32 * tau_factor);
            let target = reflex_thrusters(self.simulator.state(), self.target_depth);
            loss_sum += self.controller.train_step(&target);
            self.simulator.step(&cmd, dt);

            let state = self.simulator.state();
            if step % self.config.cognitive_interval == 0 {
                let sample = UnderwaterNavigationSample {
                    dvl_velocity_body_mps: Some(state.linear_velocity),
                    sonar_relative_fix_m: None,
                };
                let measurements = self.navigation_bridge.measurements_from_state(
                    state,
                    step as f64 * dt,
                    &sample,
                );
                metrics.navigation_measurements_emitted += measurements.len();
                self.navigation_estimator.ingest_measurements(&measurements);
                self.episode_navigation_measurements.extend(measurements);
            }

            depth_error_sum += (state.depth - self.target_depth).abs();
            effort_sum += cmd.control_effort();
            speed_sum += state.speed();
            metrics.steps_survived = step + 1;

            // Store chemical reading
            if step % 10 == 0 {
                let reading = crate::store_forward::BufferedReading {
                    timestamp: step as f64 * dt,
                    position: state.position,
                    depth: state.depth,
                    chemical: state.chemical,
                    meets_who: state.chemical.meets_who_standards(),
                    potability_score: state.chemical.potability_score(),
                };
                self.store_forward.record(reading);
                metrics.samples_collected += 1;
            }

            if state.depth > self.config.crush_depth {
                metrics.exceeded_crush_depth = true;
                break;
            }
            if !state.is_finite() {
                break;
            }
        }

        let n = metrics.steps_survived.max(1) as f64;
        metrics.mean_depth_error = depth_error_sum / n;
        metrics.mean_control_effort = effort_sum / n as f32;
        metrics.mean_speed = speed_sum / n;
        metrics.mean_imitation_loss = loss_sum / n as f32;
        let estimate = self.navigation_estimator.estimate();
        metrics.final_navigation_sigma_m = estimate.position_sigma_m;
        metrics.final_navigation_position_error_m =
            position_error(&estimate, self.simulator.state());
        metrics
    }

    pub fn train(&mut self, num_episodes: usize) -> Vec<EpisodeMetrics> {
        (0..num_episodes).map(|_| self.run_episode()).collect()
    }

    pub fn store_forward_buffer(&self) -> &StoreForwardBuffer {
        &self.store_forward
    }

    pub fn episode_navigation_measurements(&self) -> &[Measurement] {
        &self.episode_navigation_measurements
    }

    pub fn navigation_estimate(&self) -> AuvNavigationEstimate {
        self.navigation_estimator.estimate()
    }
}

fn position_error(estimate: &AuvNavigationEstimate, state: &crate::types::AuvState) -> f64 {
    let dx = estimate.position_m[0] - state.position[0];
    let dy = estimate.position_m[1] - state.position[1];
    let dz = estimate.position_m[2] - state.position[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episode_runs() {
        let mut config = AuvConfig::default();
        config.steps_per_episode = 100;
        let mut trainer = AuvTrainer::new(config, 10.0);
        let metrics = trainer.run_episode();
        assert!(metrics.steps_survived > 0);
        assert!(metrics.mean_depth_error.is_finite());
        assert!(metrics.samples_collected > 0);
        assert!(
            metrics.mean_imitation_loss > 0.0,
            "learning signal must be live"
        );
    }

    #[test]
    fn test_training_reduces_imitation_loss() {
        let mut config = AuvConfig::default();
        config.steps_per_episode = 200;
        let mut trainer = AuvTrainer::new(config, 10.0);
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
    fn test_train_multiple() {
        let mut config = AuvConfig::default();
        config.steps_per_episode = 50;
        let mut trainer = AuvTrainer::new(config, 10.0);
        let results = trainer.train(3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_store_forward_collects_samples() {
        let mut config = AuvConfig::default();
        config.steps_per_episode = 100;
        let mut trainer = AuvTrainer::new(config, 10.0);
        trainer.run_episode();
        assert!(trainer.store_forward_buffer().total_collected() > 0);
    }

    #[test]
    fn test_episode_emits_navigation_measurements() {
        let mut config = AuvConfig::default();
        config.steps_per_episode = 100;
        let mut trainer = AuvTrainer::new(config, 10.0);
        let metrics = trainer.run_episode();
        assert!(metrics.navigation_measurements_emitted > 0);
        assert!(!trainer.episode_navigation_measurements().is_empty());
        assert!(metrics.final_navigation_sigma_m.is_finite());
        assert!(metrics.final_navigation_position_error_m.is_finite());
    }
}
