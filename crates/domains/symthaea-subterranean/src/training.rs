// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::control_context::SubterraneanControlContextEncoder;
use crate::controller::{ControllerCheckpoint, SubterraneanController};
use crate::curriculum::{ScenarioCurriculum, SubterraneanScenario, SubterraneanScenarioKind};
use crate::fep_agent::ActiveInferenceSubterraneanAgent;
use crate::reflex::reflex_command_for_mission;
use crate::safety::{HazardSupervisor, SubterraneanHazard, plan_command_with_portfolio_resources};
use crate::simulator::{SimpleSubterraneanSimulator, SubterraneanPhysicsSimulator};
use crate::types::{ConfigError, SubterraneanCommand, SubterraneanConfig};
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    pub scenario_kind: SubterraneanScenarioKind,
    pub mean_effort: f32,
    pub steps_survived: usize,
    pub diverged: bool,
    /// Mean pre-update imitation MSE against the hand-designed reflex target.
    pub mean_imitation_loss: f32,
    /// Mean FEP free energy over the episode.
    pub mean_free_energy: f64,
    /// Fraction of the executed command supplied by the teacher policy.
    pub teacher_forcing_ratio: f32,
    /// Number of timed curriculum disturbances injected this episode.
    pub events_applied: usize,
    /// Number of control frames constrained by the physical safety supervisor.
    pub safety_interventions: usize,
    pub peak_hazard_severity: f32,
    pub final_mission_progress: f64,
    /// True when the episode remained finite and ended below Orange severity.
    pub recovered: bool,
}

pub struct SubterraneanTrainer {
    controller: SubterraneanController,
    context_encoder: SubterraneanControlContextEncoder,
    fep: ActiveInferenceSubterraneanAgent,
    simulator: SimpleSubterraneanSimulator,
    hazard_supervisor: HazardSupervisor,
    curriculum: ScenarioCurriculum,
    next_scenario: usize,
    episodes_completed: usize,
    config: SubterraneanConfig,
}

impl SubterraneanTrainer {
    pub fn try_new(config: SubterraneanConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self::new(config))
    }

    pub fn new(config: SubterraneanConfig) -> Self {
        let g = GenesisSeed::from_phrase(&config.genesis_phrase);
        let curriculum = ScenarioCurriculum::standard(config.steps_per_episode);
        Self {
            controller: SubterraneanController::new(&g, &config),
            context_encoder: SubterraneanControlContextEncoder::new(&g, 32),
            fep: ActiveInferenceSubterraneanAgent::new(),
            simulator: SimpleSubterraneanSimulator::new(),
            hazard_supervisor: HazardSupervisor::new(),
            curriculum,
            next_scenario: 0,
            episodes_completed: 0,
            config,
        }
    }

    pub fn controller_checkpoint(&self) -> ControllerCheckpoint {
        self.controller.checkpoint()
    }

    pub fn curriculum(&self) -> &ScenarioCurriculum {
        &self.curriculum
    }

    pub fn episodes_completed(&self) -> usize {
        self.episodes_completed
    }

    fn teacher_forcing_ratio(&self) -> f32 {
        (1.0 - self.episodes_completed as f32 / 12.0).clamp(0.15, 1.0)
    }

    /// Run the next deterministic curriculum scenario.
    pub fn run_episode(&mut self) -> EpisodeMetrics {
        let scenario = self.curriculum.get(self.next_scenario).clone();
        self.next_scenario = (self.next_scenario + 1) % self.curriculum.len();
        self.run_scenario_episode(&scenario)
    }

    /// Run one named scenario using DAgger-style command execution.
    ///
    /// The learner is always updated toward the reflex oracle, while the plant
    /// initially executes mostly teacher commands. Teacher authority decays to
    /// a non-zero floor so the learner is gradually exposed to its own state
    /// distribution without allowing an untrained controller to create unsafe
    /// rollouts. Physical arbitration remains active throughout training.
    pub fn run_scenario_episode(&mut self, scenario: &SubterraneanScenario) -> EpisodeMetrics {
        self.simulator.reset();
        scenario.initialize(self.simulator.state_mut());
        self.controller.reset();
        self.context_encoder.reset();
        self.fep.reset();
        self.hazard_supervisor.reset();

        let dt = self.config.physics_dt();
        let teacher_forcing_ratio = self.teacher_forcing_ratio();
        let mut total_effort = 0.0f32;
        let mut loss_sum = 0.0f32;
        let mut free_energy_sum = 0.0f64;
        let mut free_energy_count = 0usize;
        let mut tau_factor = 1.0f32;
        let mut events_applied = 0usize;
        let mut safety_interventions = 0usize;
        let mut peak_hazard_severity = 0.0f32;
        let mut steps_survived = 0usize;
        let mut diverged = false;

        for step in 0..self.config.steps_per_episode {
            events_applied += scenario.apply_events(step, self.simulator.state_mut());

            let context = self.context_encoder.encode(
                self.simulator.state(),
                None,
                scenario.mission_intent(),
            );
            if step % self.config.cognitive_interval == 0 {
                let fep = self.fep.tick(self.simulator.state());
                tau_factor = fep.tau_factor;
                free_energy_sum += fep.free_energy;
                free_energy_count += 1;
            }

            let learner = self.controller.forward(&context, dt as f32 * tau_factor);
            let teacher =
                reflex_command_for_mission(self.simulator.state(), scenario.mission_intent());
            loss_sum += self.controller.train_step(&teacher);

            let blended = SubterraneanCommand::blend(learner, teacher, teacher_forcing_ratio);
            let hazard = self.hazard_supervisor.update(self.simulator.state());
            peak_hazard_severity = peak_hazard_severity.max(hazard.severity);
            if hazard.primary != SubterraneanHazard::None {
                safety_interventions += 1;
            }
            let executed = plan_command_with_portfolio_resources(
                blended,
                self.simulator.state(),
                hazard,
                self.hazard_supervisor.raw_portfolio(),
                hazard.safety_level,
                self.simulator.recovery_resources(),
            )
            .command;

            self.simulator.step(&executed, dt);
            total_effort += executed.control_effort();
            steps_survived = step + 1;
            if !self.simulator.state().is_finite() {
                diverged = true;
                break;
            }
        }

        self.episodes_completed = self.episodes_completed.saturating_add(1);
        let denominator = steps_survived.max(1) as f32;
        let final_state = self.simulator.state();
        let final_hazard = self.hazard_supervisor.update(final_state);
        let recovered = !diverged && final_hazard.severity < 0.55;

        EpisodeMetrics {
            scenario_kind: scenario.kind,
            mean_effort: total_effort / denominator,
            steps_survived,
            diverged,
            mean_imitation_loss: loss_sum / denominator,
            mean_free_energy: free_energy_sum / free_energy_count.max(1) as f64,
            teacher_forcing_ratio,
            events_applied,
            safety_interventions,
            peak_hazard_severity,
            final_mission_progress: final_state.channels[crate::types::MISSION_PROGRESS],
            recovered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episode() {
        let mut config = SubterraneanConfig::default();
        config.steps_per_episode = 200;
        let mut trainer = SubterraneanTrainer::new(config);
        let metrics = trainer.run_episode();
        assert!(!metrics.diverged);
        assert_eq!(metrics.steps_survived, 200);
        assert!(
            metrics.mean_imitation_loss > 0.0,
            "learning signal must be live"
        );
        assert!(metrics.mean_free_energy.is_finite());
        assert_eq!(
            metrics.scenario_kind,
            SubterraneanScenarioKind::NominalTransit
        );
    }

    #[test]
    fn test_training_reduces_imitation_loss_on_fixed_nominal_scenario() {
        let mut config = SubterraneanConfig::default();
        config.steps_per_episode = 400;
        let mut trainer = SubterraneanTrainer::new(config);
        let scenario = SubterraneanScenario::nominal();
        let first = trainer.run_scenario_episode(&scenario).mean_imitation_loss;
        for _ in 0..3 {
            trainer.run_scenario_episode(&scenario);
        }
        let last = trainer.run_scenario_episode(&scenario).mean_imitation_loss;
        assert!(
            last < first,
            "imitation loss must decrease with training: first {first:.5} -> last {last:.5}"
        );
    }

    #[test]
    fn emergency_scenario_activates_safety_supervision() {
        let mut config = SubterraneanConfig::default();
        config.steps_per_episode = 150;
        let mut trainer = SubterraneanTrainer::new(config);
        let scenario = trainer.curriculum().get(1).clone();
        let metrics = trainer.run_scenario_episode(&scenario);
        assert_eq!(metrics.events_applied, 1);
        assert!(metrics.safety_interventions > 0);
        assert!(metrics.peak_hazard_severity > 0.0);
    }

    #[test]
    fn sensor_fault_curriculum_episode_fails_closed_without_diverging() {
        let mut config = SubterraneanConfig::default();
        config.steps_per_episode = 100;
        let mut trainer = SubterraneanTrainer::new(config);
        let scenario = trainer
            .curriculum()
            .scenarios()
            .iter()
            .find(|scenario| scenario.kind == SubterraneanScenarioKind::SensorFault)
            .cloned()
            .expect("standard curriculum includes sensor fault");
        let metrics = trainer.run_scenario_episode(&scenario);
        assert!(!metrics.diverged);
        assert!(metrics.safety_interventions > 0);
        assert!(metrics.peak_hazard_severity >= 1.0);
    }

    #[test]
    fn teacher_forcing_decays_but_never_disappears() {
        let mut config = SubterraneanConfig::default();
        config.steps_per_episode = 10;
        let mut trainer = SubterraneanTrainer::new(config);
        let scenario = SubterraneanScenario::nominal();
        let first = trainer
            .run_scenario_episode(&scenario)
            .teacher_forcing_ratio;
        let mut last = first;
        for _ in 0..20 {
            last = trainer
                .run_scenario_episode(&scenario)
                .teacher_forcing_ratio;
        }
        assert!(last < first);
        assert!(last >= 0.15);
    }

    #[test]
    fn checked_constructor_rejects_invalid_config() {
        let mut config = SubterraneanConfig::default();
        config.cognitive_interval = 0;
        assert!(matches!(
            SubterraneanTrainer::try_new(config),
            Err(ConfigError::ZeroCognitiveInterval)
        ));
    }
}
