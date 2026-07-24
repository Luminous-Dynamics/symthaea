// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Checkpointed policy ablations on identical deterministic scenarios.
//!
//! This runner separates gains from the learned controller, the hand-designed
//! reflex oracle, and the verified safety planner. Every variant receives the
//! same initial state, event schedule, mission, timestep, and checkpoint.

use crate::control_context::SubterraneanControlContextEncoder;
use crate::controller::{CheckpointError, ControllerCheckpoint, SubterraneanController};
use crate::curriculum::{SubterraneanScenario, SubterraneanScenarioKind};
use crate::mission::MissionManager;
use crate::reflex::reflex_command_for_mission;
use crate::safety::{HazardSupervisor, SubterraneanHazard, plan_command_with_portfolio_resources};
use crate::simulator::{SimpleSubterraneanSimulator, SubterraneanPhysicsSimulator};
use crate::types::{BATTERY_RATIO, ConfigError, MISSION_PROGRESS, SubterraneanConfig};
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyVariant {
    LearnerOnly,
    ReflexOnly,
    LearnerWithSafety,
    ReflexWithSafety,
}

impl PolicyVariant {
    pub const ALL: [Self; 4] = [
        Self::LearnerOnly,
        Self::ReflexOnly,
        Self::LearnerWithSafety,
        Self::ReflexWithSafety,
    ];

    pub const fn uses_learner(self) -> bool {
        matches!(self, Self::LearnerOnly | Self::LearnerWithSafety)
    }

    pub const fn uses_safety(self) -> bool {
        matches!(self, Self::LearnerWithSafety | Self::ReflexWithSafety)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::LearnerOnly => "learner_only",
            Self::ReflexOnly => "reflex_only",
            Self::LearnerWithSafety => "learner_with_safety",
            Self::ReflexWithSafety => "reflex_with_safety",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyAblationReport {
    pub scenario_kind: SubterraneanScenarioKind,
    pub variant: PolicyVariant,
    pub steps_executed: usize,
    pub events_applied: usize,
    pub diverged: bool,
    pub intervention_frames: usize,
    pub resource_limited_frames: usize,
    /// Cutter-active frames while thermal, gas, flood, roof, or sensor risk
    /// was present. This is intentionally measured before plant sanitization.
    pub unsafe_cutter_frames: usize,
    pub peak_hazard_severity: f32,
    pub peak_simultaneous_hazards: usize,
    pub terminal_hazard_severity: f32,
    pub terminal_active_hazards: usize,
    pub mean_control_effort: f32,
    pub initial_battery_ratio: f64,
    pub terminal_battery_ratio: f64,
    pub battery_spent: f64,
    pub relay_units_spent: u8,
    pub roof_support_units_spent: u8,
    pub sealant_spent: f64,
    pub final_mission_progress: f64,
    pub recovered: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyAblationSuite {
    pub scenario_kind: SubterraneanScenarioKind,
    pub reports: Vec<PolicyAblationReport>,
}

impl PolicyAblationSuite {
    pub fn report(&self, variant: PolicyVariant) -> Option<&PolicyAblationReport> {
        self.reports.iter().find(|report| report.variant == variant)
    }

    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug)]
pub enum PolicyAblationError {
    Config(ConfigError),
    Checkpoint(CheckpointError),
}

impl std::fmt::Display for PolicyAblationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(error) => write!(f, "invalid ablation configuration: {error}"),
            Self::Checkpoint(error) => write!(f, "invalid ablation checkpoint: {error}"),
        }
    }
}

impl std::error::Error for PolicyAblationError {}

impl From<ConfigError> for PolicyAblationError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<CheckpointError> for PolicyAblationError {
    fn from(error: CheckpointError) -> Self {
        Self::Checkpoint(error)
    }
}

pub struct PolicyAblationRunner {
    config: SubterraneanConfig,
    checkpoint: ControllerCheckpoint,
}

impl PolicyAblationRunner {
    pub fn new(
        config: SubterraneanConfig,
        checkpoint: ControllerCheckpoint,
    ) -> Result<Self, PolicyAblationError> {
        config.validate()?;
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let mut validation_controller = SubterraneanController::new(&genesis, &config);
        validation_controller.load_checkpoint(&checkpoint)?;
        Ok(Self { config, checkpoint })
    }

    pub fn run_all(&self, scenario: &SubterraneanScenario) -> PolicyAblationSuite {
        let reports = PolicyVariant::ALL
            .iter()
            .copied()
            .map(|variant| self.run_variant(scenario, variant))
            .collect();
        PolicyAblationSuite {
            scenario_kind: scenario.kind,
            reports,
        }
    }

    pub fn run_variant(
        &self,
        scenario: &SubterraneanScenario,
        variant: PolicyVariant,
    ) -> PolicyAblationReport {
        let genesis = GenesisSeed::from_phrase(&self.config.genesis_phrase);
        let mut controller = SubterraneanController::new(&genesis, &self.config);
        // Constructor validation guarantees this private checkpoint loads.
        // Keep the result explicit so a future schema change cannot silently
        // turn an ablation into a genesis-weight comparison.
        let checkpoint_loaded = controller.load_checkpoint(&self.checkpoint).is_ok();
        debug_assert!(checkpoint_loaded);
        let mut context_encoder = SubterraneanControlContextEncoder::new(&genesis, 32);
        let mut simulator = SimpleSubterraneanSimulator::new();
        scenario.initialize(simulator.state_mut());
        let initial_resources = simulator.recovery_resources();
        let initial_battery_ratio = simulator.state().channels[BATTERY_RATIO];
        let mut supervisor = HazardSupervisor::new();
        let mut mission = MissionManager::new(scenario.mission_intent());
        let dt = self.config.physics_dt();
        let mut events_applied = 0usize;
        let mut intervention_frames = 0usize;
        let mut resource_limited_frames = 0usize;
        let mut unsafe_cutter_frames = 0usize;
        let mut peak_hazard_severity = 0.0f32;
        let mut peak_simultaneous_hazards = 0usize;
        let mut effort_sum = 0.0f32;
        let mut steps_executed = 0usize;
        let mut diverged = !checkpoint_loaded;

        if !checkpoint_loaded {
            return PolicyAblationReport {
                scenario_kind: scenario.kind,
                variant,
                steps_executed: 0,
                events_applied: 0,
                diverged: true,
                intervention_frames: 0,
                resource_limited_frames: 0,
                unsafe_cutter_frames: 0,
                peak_hazard_severity: 0.0,
                peak_simultaneous_hazards: 0,
                terminal_hazard_severity: 1.0,
                terminal_active_hazards: 0,
                mean_control_effort: 0.0,
                initial_battery_ratio,
                terminal_battery_ratio: initial_battery_ratio,
                battery_spent: 0.0,
                relay_units_spent: 0,
                roof_support_units_spent: 0,
                sealant_spent: 0.0,
                final_mission_progress: simulator.state().channels[MISSION_PROGRESS],
                recovered: false,
            };
        }

        for step in 0..self.config.steps_per_episode {
            events_applied =
                events_applied.saturating_add(scenario.apply_events(step, simulator.state_mut()));
            let hazard = supervisor.update(simulator.state());
            let portfolio = supervisor.raw_portfolio();
            if hazard.primary == SubterraneanHazard::SensorFault {
                simulator.state_mut().sanitize_fail_closed();
            }
            peak_hazard_severity = peak_hazard_severity.max(portfolio.max_severity());
            peak_simultaneous_hazards = peak_simultaneous_hazards.max(portfolio.active_count(0.01));
            let effective_mission = mission.update(simulator.state(), hazard);
            let context = context_encoder.encode(simulator.state(), None, effective_mission);
            let learner = controller.forward(&context, dt as f32);
            let reflex = reflex_command_for_mission(simulator.state(), effective_mission);
            let nominal = if variant.uses_learner() {
                learner
            } else {
                reflex
            };
            let mut command = nominal;

            if variant.uses_safety() {
                let plan = plan_command_with_portfolio_resources(
                    nominal,
                    simulator.state(),
                    hazard,
                    portfolio,
                    hazard.safety_level,
                    simulator.recovery_resources(),
                );
                command = plan.command;
                if hazard.primary != SubterraneanHazard::None {
                    intervention_frames = intervention_frames.saturating_add(1);
                }
                if plan.resource_limited {
                    resource_limited_frames = resource_limited_frames.saturating_add(1);
                }
            }

            let hazardous_cutting = portfolio.severity(SubterraneanHazard::Thermal) > 0.0
                || portfolio.severity(SubterraneanHazard::Flood) > 0.0
                || portfolio.severity(SubterraneanHazard::Gas) > 0.0
                || portfolio.severity(SubterraneanHazard::RoofInstability) > 0.0
                || portfolio.severity(SubterraneanHazard::SensorFault) > 0.0;
            if hazardous_cutting && command.cutter_head() > 0.05 {
                unsafe_cutter_frames = unsafe_cutter_frames.saturating_add(1);
            }

            effort_sum += command.control_effort();
            simulator.step(&command, dt);
            steps_executed = step + 1;
            if !simulator.state().is_finite() {
                diverged = true;
                break;
            }
        }

        let terminal_hazard = supervisor.update(simulator.state());
        let terminal_portfolio = supervisor.raw_portfolio();
        let terminal_resources = simulator.recovery_resources();
        let terminal_battery_ratio = simulator.state().channels[BATTERY_RATIO];
        let battery_spent = (initial_battery_ratio - terminal_battery_ratio).max(0.0);
        let recovered = !diverged && terminal_portfolio.max_severity() < 0.55;

        PolicyAblationReport {
            scenario_kind: scenario.kind,
            variant,
            steps_executed,
            events_applied,
            diverged,
            intervention_frames,
            resource_limited_frames,
            unsafe_cutter_frames,
            peak_hazard_severity,
            peak_simultaneous_hazards,
            terminal_hazard_severity: terminal_hazard.severity,
            terminal_active_hazards: terminal_portfolio.active_count(0.01),
            mean_control_effort: effort_sum / steps_executed.max(1) as f32,
            initial_battery_ratio,
            terminal_battery_ratio,
            battery_spent,
            relay_units_spent: initial_resources
                .relay_units
                .saturating_sub(terminal_resources.relay_units),
            roof_support_units_spent: initial_resources
                .roof_support_units
                .saturating_sub(terminal_resources.roof_support_units),
            sealant_spent: (initial_resources.sealant_ratio - terminal_resources.sealant_ratio)
                .max(0.0),
            final_mission_progress: simulator.state().channels[MISSION_PROGRESS],
            recovered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curriculum::ScenarioCurriculum;

    fn runner(steps: usize) -> PolicyAblationRunner {
        let mut config = SubterraneanConfig::default();
        config.steps_per_episode = steps;
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let checkpoint = SubterraneanController::new(&genesis, &config).checkpoint();
        PolicyAblationRunner::new(config, checkpoint).expect("valid baseline runner")
    }

    #[test]
    fn same_scenario_produces_all_four_ablation_variants() {
        let scenario = ScenarioCurriculum::standard(100).get(2).clone();
        let suite = runner(100).run_all(&scenario);
        assert_eq!(suite.reports.len(), PolicyVariant::ALL.len());
        assert!(suite.report(PolicyVariant::LearnerOnly).is_some());
        assert!(suite.report(PolicyVariant::ReflexWithSafety).is_some());
        assert!(suite.to_pretty_json().is_ok());
    }

    #[test]
    fn safety_variants_intervene_and_prevent_gas_cutting() {
        let scenario = ScenarioCurriculum::standard(120)
            .scenarios()
            .iter()
            .find(|scenario| scenario.kind == SubterraneanScenarioKind::GasPocket)
            .cloned()
            .expect("standard gas scenario");
        let suite = runner(120).run_all(&scenario);
        let safe = suite
            .report(PolicyVariant::ReflexWithSafety)
            .expect("safe reflex report");
        assert!(safe.intervention_frames > 0);
        assert_eq!(safe.unsafe_cutter_frames, 0);
    }

    #[test]
    fn compound_suite_uses_identical_event_schedule_for_each_variant() {
        let scenario = ScenarioCurriculum::compound_holdout(160).get(0).clone();
        let suite = runner(160).run_all(&scenario);
        let expected = scenario.events.len();
        assert!(
            suite
                .reports
                .iter()
                .all(|report| report.events_applied == expected)
        );
    }
}
